"""
Gradient-guided binder design for protein/RNA/DNA targets and designing chains.
Extends binder_design.py to support nucleic acid binders and explicit position fixing.
RNA/DNA designing chains skip ESMC regularization (protein-only LM).
"""

import os
from pathlib import Path

# Must be set before any huggingface_hub / transformers imports
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
#os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_CACHE", "/mnt/d/mol2/invESM/lib/pretrained")

import logging
import math
import random
import string
from dataclasses import dataclass
from enum import Enum
from functools import cache, partial
from typing import Any

import biotite.structure
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from transformers.models.esmc.modeling_esmc import ESMCForMaskedLM
from transformers.models.esmc.modeling_esmc import (
    UnifiedTransformerBlock as TransformerBlock,
)
from transformers.models.esmc.tokenization_esmc import ESMCTokenizer
from transformers.models.esmfold2.modeling_esmfold2_common import (
    CUE_AVAILABLE,
    PairUpdateBlock,
)
from transformers.models.esmfold2.modeling_esmfold2_common import (
    _seed_context as seed_context,
)
from transformers.models.esmfold2.modeling_esmfold2_experimental import (
    ESMFold2ExperimentalModel,
)
from transformers.models.esmfold2.modeling_esmfold2_experimental import (
    MSAEncoder as ESMFold2MSAEncoder,
)

from esm.models.esmfold2 import (
    ELEMENT_NUMBER_TO_SYMBOL,
    ProteinInput,
    StructurePredictionInput,
    load_ccd,
    prepare_esmfold2_input,
)
from esm.models.esmfold2.constants import (
    MOL_TYPE_NONPOLYMER,
    PROTEIN_1TO3,
    PROTEIN_3TO1,
    RES_TYPE_TO_CCD,
)
from esm.utils.structure.input_builder import DNAInput, RNAInput
from esm.utils.structure.protein_chain import ProteinChain
from esm.utils.structure.protein_complex import ProteinComplex

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Local weights cache — weights are stored here and never re-downloaded.
CACHE_DIR = "/mnt/d/mol2/invESM/lib/pretrained"


# ---- Molecule type ----


class MolType(str, Enum):
    PROTEIN = "protein"
    RNA = "rna"
    DNA = "dna"


# ---- Vocabulary constants ----

TOKENS = ["<pad>", "-"] + [RES_TYPE_TO_CCD[i] for i in range(2, 33)]
ELEMENTS = ["X"] * (max(ELEMENT_NUMBER_TO_SYMBOL) + 1)
ELEMENTS[0] = "<pad>"
for _n, _s in ELEMENT_NUMBER_TO_SYMBOL.items():
    ELEMENTS[_n] = _s[:1] + _s[1:].lower()
TOKEN_IDS = {token: idx for idx, token in enumerate(TOKENS)}

# 1-letter → CCD for RNA and DNA
RNA_1TO3 = {"A": "A", "G": "G", "C": "C", "U": "U"}
DNA_1TO3 = {"A": "DA", "G": "DG", "C": "DC", "T": "DT"}
RNA_3TO1 = {v: k for k, v in RNA_1TO3.items()}
DNA_3TO1 = {v: k for k, v in DNA_1TO3.items()}

# Indices into TOKENS for each mol type's designable residues (order defines small-vocab axis)
PROTEIN_TOKEN_INDICES: list[int] = list(range(2, 22))   # 20 AAs at TOKENS[2..21]
RNA_TOKEN_INDICES: list[int] = [TOKEN_IDS[RNA_1TO3[c]] for c in "AGCU"]
DNA_TOKEN_INDICES: list[int] = [TOKEN_IDS[DNA_1TO3[c]] for c in "AGCT"]

MOL_VOCAB_DIM: dict[MolType, int] = {MolType.PROTEIN: 20, MolType.RNA: 4, MolType.DNA: 4}
MOL_TOKEN_INDICES: dict[MolType, list[int]] = {
    MolType.PROTEIN: PROTEIN_TOKEN_INDICES,
    MolType.RNA: RNA_TOKEN_INDICES,
    MolType.DNA: DNA_TOKEN_INDICES,
}

# Cysteine index within the 20-AA sub-vocabulary (protein only)
_CYS_TOKEN_ID = TOKEN_IDS[PROTEIN_1TO3["C"]]
CYS_IDX = PROTEIN_TOKEN_INDICES.index(_CYS_TOKEN_ID)

MUTABLE_TOKEN = "#"


# ---- Hyperparameters ----

LOSS_WEIGHTS = {"intra_contact": 0.5, "inter_contact": 0.5, "glob": 0.2, "hotspot": 1.0}
STEPS = 150
LOG_INTERVAL = 5
LEARNING_RATE = 0.1
TEMPERATURE_MIN = 1e-2
ESMC_MASK_FRACTION = 0.15
CHECKPOINT_LM = False
CHECKPOINT_INVERSION = True   # activation checkpointing on inversion model (saves ~60% activation memory)
COMPILE = False
REUSE_ESMC = True
LM_WEIGHT_ANTIBODY = 0.05
LM_WEIGHT_PROTEIN = 0.15


# ---- DesignSpec ----


@dataclass
class DesignSpec:
    """Specification for the chain to be designed.

    sequence: residue letters at fixed positions, MUTABLE_TOKEN ('#') elsewhere.
    """

    mol_type: MolType
    sequence: str

    @property
    def length(self) -> int:
        return len(self.sequence)

    @property
    def vocab_dim(self) -> int:
        return MOL_VOCAB_DIM[self.mol_type]

    @property
    def token_indices(self) -> list[int]:
        return MOL_TOKEN_INDICES[self.mol_type]

    @property
    def is_protein(self) -> bool:
        return self.mol_type == MolType.PROTEIN


# ---- Position-fixing utilities ----


def fix_positions(sequence: str, fixes: dict[int, str]) -> str:
    """Pin residues at given 0-based positions; other positions are unchanged."""
    chars = list(sequence)
    for pos, residue in fixes.items():
        if pos < 0 or pos >= len(chars):
            raise IndexError(f"Position {pos} out of range for sequence of length {len(chars)}")
        chars[pos] = residue
    return "".join(chars)


def build_mutable_sequence(
    length: int,
    fixed_positions: dict[int, str] | None = None,
) -> str:
    """All-mutable sequence of given length with optional fixed positions."""
    seq = MUTABLE_TOKEN * length
    if fixed_positions:
        seq = fix_positions(seq, fixed_positions)
    return seq


def make_design_spec(
    mol_type: MolType | str,
    length_or_sequence: int | str,
    fixed_positions: dict[int, str] | None = None,
) -> DesignSpec:
    """
    Convenience constructor for DesignSpec.

    length_or_sequence:
        int  → fully mutable sequence of that length
        str  → sequence with '#' at mutable positions (e.g. antibody scaffold)
    fixed_positions:
        additional {0-based index: residue} pins applied on top.
    """
    mol_type = MolType(mol_type)
    if isinstance(length_or_sequence, int):
        seq = build_mutable_sequence(length_or_sequence, fixed_positions)
    else:
        seq = length_or_sequence
        if fixed_positions:
            seq = fix_positions(seq, fixed_positions)
    return DesignSpec(mol_type=mol_type, sequence=seq)


# ---- Soft-sequence helpers ----


def _residue_to_small_vocab_idx(residue: str, mol_type: MolType) -> int:
    """Convert a 1-letter residue code to its index in the small per-mol-type vocabulary."""
    if mol_type == MolType.PROTEIN:
        ccd = PROTEIN_1TO3[residue]
        return PROTEIN_TOKEN_INDICES.index(TOKEN_IDS[ccd])
    elif mol_type == MolType.RNA:
        return list("AGCU").index(residue)
    else:
        return list("AGCT").index(residue)


def build_initial_soft_sequence_logits(spec: DesignSpec, batch_size: int) -> torch.Tensor:
    """
    Initialize logits [B, L, vocab_dim].
    Fixed positions: high-confidence spike (10.0) at the correct residue.
    Mutable positions: small random noise; CYS zeroed out for proteins.
    """
    V, L = spec.vocab_dim, spec.length
    logits = torch.zeros([batch_size, L, V])

    for i, aa in enumerate(spec.sequence):
        if aa == MUTABLE_TOKEN:
            logits[:, i, :] = 0.01 * torch.randn(batch_size, V)
            if spec.is_protein:
                logits[:, i, CYS_IDX] = -1e6
        else:
            idx = _residue_to_small_vocab_idx(aa, spec.mol_type)
            logits[:, i, idx] = 10.0

    return logits.requires_grad_(True)


def build_gradient_mask(spec: DesignSpec, batch_size: int) -> torch.Tensor:
    """
    Gradient mask [B, L, vocab_dim].
    0 at fixed positions and (for proteins) at CYS; 1 elsewhere.
    """
    V, L = spec.vocab_dim, spec.length
    mask = torch.ones([batch_size, L, V])
    fixed = [i for i, aa in enumerate(spec.sequence) if aa != MUTABLE_TOKEN]
    mask[:, fixed, :] = 0.0
    if spec.is_protein:
        mask[:, :, CYS_IDX] = 0.0
    return mask


def embed_design_soft(soft_seq: torch.Tensor, token_indices: list[int]) -> torch.Tensor:
    """Lift [B, L, V_small] → [B, L, len(TOKENS)] placing values at token_indices."""
    B, L, _ = soft_seq.shape
    full = torch.zeros(B, L, len(TOKENS), device=soft_seq.device, dtype=soft_seq.dtype)
    full[:, :, token_indices] = soft_seq
    return full


# ---- One-hot for target / scoring ----


def sequence_to_one_hot_full(
    sequence: str, mol_type: MolType, device: str = "cuda"
) -> torch.Tensor:
    """Convert a sequence to [1, L, len(TOKENS)] one-hot (full token vocabulary)."""
    if mol_type == MolType.PROTEIN:
        indices = [TOKEN_IDS[PROTEIN_1TO3[c]] for c in sequence]
    elif mol_type == MolType.RNA:
        indices = [TOKEN_IDS[RNA_1TO3[c]] for c in sequence]
    else:
        indices = [TOKEN_IDS[DNA_1TO3[c]] for c in sequence]
    one_hot = F.one_hot(torch.tensor(indices), num_classes=len(TOKENS))
    return one_hot.to(device).unsqueeze(0).float()


def _decode_designed_sequences(padded_design: torch.Tensor, mol_type: MolType) -> list[str]:
    """Decode argmax of [B, L, len(TOKENS)] → list of 1-letter sequence strings."""
    if mol_type == MolType.PROTEIN:
        lookup = lambda ccd: PROTEIN_3TO1.get(ccd, "X")
    elif mol_type == MolType.RNA:
        lookup = lambda ccd: RNA_3TO1.get(ccd, "N")
    else:
        lookup = lambda ccd: DNA_3TO1.get(ccd, "N")
    token_lists = torch.argmax(padded_design, dim=-1)
    return ["".join(lookup(TOKENS[int(t)]) for t in row) for row in token_lists]


# ---- Chain input factory ----


def _make_chain_input(
    chain_id: str, sequence: str, mol_type: MolType
) -> ProteinInput | RNAInput | DNAInput:
    if mol_type == MolType.PROTEIN:
        return ProteinInput(id=chain_id, sequence=sequence, msa=None)
    elif mol_type == MolType.RNA:
        return RNAInput(id=chain_id, sequence=sequence)
    else:
        return DNAInput(id=chain_id, sequence=sequence)


# ---- Distance-bin helpers ----


def get_mid_points() -> torch.Tensor:
    boundaries = torch.linspace(2, 52.0, 127)
    lower = torch.tensor([1.0])
    upper = torch.tensor([57.0])
    exp_boundaries = torch.cat((lower, boundaries, upper))
    return (exp_boundaries[:-1] + exp_boundaries[1:]) / 2


def binned_entropy(dgram: torch.Tensor, bin_distance: torch.Tensor, cutoff: float) -> torch.Tensor:
    bin_mask = ~(bin_distance < cutoff)
    masked_dgram = dgram - (1e7 * bin_mask)
    px = torch.softmax(masked_dgram, dim=-1)
    log_px = torch.log_softmax(dgram, dim=-1)
    return -(px * log_px).sum(-1)


def masked_min_k(x: torch.Tensor, mask: torch.Tensor, k: int) -> torch.Tensor:
    mask = mask.bool()
    y = torch.sort(torch.where(mask, x, float("nan")))[0]
    k_mask = (torch.arange(y.shape[-1]).to(y.device) < k) & (~torch.isnan(y))
    return torch.where(k_mask, y, 0).sum(-1) / (k_mask.sum(-1) + 1e-8)


def masked_average(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.bool()
    return torch.where(mask, x, 0).sum(-1) / (torch.where(mask, 1, 0).sum(-1) + 1e-8)


# ---- Loss functions ----


def compute_contact_loss(
    distogram_logits: torch.Tensor,
    bin_distance: torch.Tensor,
    num_contacts: int,
    min_sep: int,
    cutoff: float,
    chain_mask: torch.Tensor,
    binder_mask: torch.Tensor,
) -> torch.Tensor:
    con_loss = binned_entropy(distogram_logits, bin_distance, cutoff)
    position = torch.arange(distogram_logits.shape[1])
    p_dist = position[:, None] - position[None, :]
    if min_sep > 0:
        separation_mask = (torch.abs(p_dist) >= min_sep).to(distogram_logits.device)
        binder_mask = torch.logical_and(separation_mask, binder_mask)
    per_residue = masked_min_k(con_loss, mask=binder_mask, k=num_contacts).to(distogram_logits.device)
    return masked_average(per_residue, mask=chain_mask).to(distogram_logits.device)


def compute_intra_contact_loss(
    distogram_logits: torch.Tensor, binder_length: int, bin_distance: torch.Tensor
) -> torch.Tensor:
    full_len = distogram_logits.shape[1]
    is_binder = torch.ones(full_len, device=distogram_logits.device)
    is_binder[:-binder_length] *= 0.0
    return compute_contact_loss(
        distogram_logits, bin_distance,
        num_contacts=2, min_sep=9, cutoff=14.0,
        chain_mask=is_binder, binder_mask=is_binder,
    )


def compute_inter_contact_loss(
    distogram_logits: torch.Tensor, binder_length: int, bin_distance: torch.Tensor
) -> torch.Tensor:
    full_len = distogram_logits.shape[1]
    is_binder = torch.ones(full_len, device=distogram_logits.device)
    is_binder[:-binder_length] *= 0.0
    return compute_contact_loss(
        distogram_logits, bin_distance,
        num_contacts=1, min_sep=0, cutoff=22.0,
        chain_mask=1 - is_binder, binder_mask=is_binder,
    )


def compute_hotspot_loss(
    distogram_logits: torch.Tensor,
    binder_length: int,
    bin_distance: torch.Tensor,
    hotspot_indices: list[int],
) -> torch.Tensor:
    """For each hotspot residue (0-based index into target), find its best binder contact."""
    full_len = distogram_logits.shape[1]
    device = distogram_logits.device
    target_length = full_len - binder_length

    is_binder = torch.zeros(full_len, device=device)
    is_binder[-binder_length:] = 1.0

    is_hotspot = torch.zeros(full_len, device=device)
    for idx in hotspot_indices:
        if 0 <= idx < target_length:
            is_hotspot[idx] = 1.0

    return compute_contact_loss(
        distogram_logits, bin_distance,
        num_contacts=1, min_sep=0, cutoff=22.0,
        chain_mask=is_hotspot, binder_mask=is_binder,
    )


def compute_globularity_loss(
    distogram_logits: torch.Tensor, binder_length: int, bin_distance: torch.Tensor
) -> torch.Tensor:
    binder_disto = distogram_logits[:, -binder_length:, -binder_length:, :]
    n = binder_disto.shape[1]
    disto_probs = torch.softmax(binder_disto, dim=-1)
    bin_distance = bin_distance.clamp(max=27)
    e_sq_dist = torch.sum(disto_probs * torch.square(bin_distance), dim=-1)
    sum_sq_dist = torch.sum(torch.tril(e_sq_dist, diagonal=-1), dim=(1, 2))
    rg_term = torch.sqrt(sum_sq_dist / (n * n))
    rg_th = 2.38 * (n ** 0.365)
    return F.elu(rg_term - rg_th)


def compute_structure_losses(
    distogram_logits: torch.Tensor,
    binder_length: int,
    hotspot_indices: list[int] | None = None,
) -> dict[str, torch.Tensor]:
    bin_distance = get_mid_points().to(distogram_logits.device)
    losses: dict[str, torch.Tensor] = {}
    losses["intra_contact_loss"] = compute_intra_contact_loss(distogram_logits, binder_length, bin_distance)
    losses["inter_contact_loss"] = compute_inter_contact_loss(distogram_logits, binder_length, bin_distance)
    losses["glob_loss"] = compute_globularity_loss(distogram_logits, binder_length, bin_distance)
    B = distogram_logits.size(0)
    total = torch.tensor([0.0] * B, device=distogram_logits.device, requires_grad=True)
    total = total + LOSS_WEIGHTS["intra_contact"] * losses["intra_contact_loss"]
    total = total + LOSS_WEIGHTS["inter_contact"] * losses["inter_contact_loss"]
    total = total + LOSS_WEIGHTS["glob"] * losses["glob_loss"]
    if hotspot_indices:
        losses["hotspot_loss"] = compute_hotspot_loss(distogram_logits, binder_length, bin_distance, hotspot_indices)
        total = total + LOSS_WEIGHTS["hotspot"] * losses["hotspot_loss"]
    losses["total_loss"] = total
    return losses


# ---- Distogram iPTM proxy ----


def _binding_confidence_entropy(
    dgram: torch.Tensor, bin_distance: torch.Tensor, cutoff: float
) -> torch.Tensor:
    probs = torch.softmax(dgram, dim=-1)
    cutoff_mask = bin_distance < cutoff
    p_cut = probs[..., cutoff_mask]
    p_cut = p_cut / (p_cut.sum(-1, keepdim=True) + 1e-8)
    return -(p_cut * torch.log(p_cut + 1e-10)).sum(-1)


def _entropy_to_confidence(mean_entropy: float) -> float:
    return float(max(0.0, min(1.0, 1.0 - mean_entropy / math.log(51))))


def _cdr_indices(binder_sequence: str) -> list[int]:
    from abnumber import Chain
    from abnumber.common import _anarci_align

    result = _anarci_align(sequences=[binder_sequence], scheme="chothia", allowed_species=None)[0]
    chains = [Chain("".join(result[i][0].values()), scheme="chothia") for i in range(len(result))]
    if len(chains) == 2 and not chains[0].is_heavy_chain():
        chains.reverse()
    indices: list[int] = []
    for chain in chains:
        for cdr in (chain.cdr1_seq, chain.cdr2_seq, chain.cdr3_seq):
            start = binder_sequence.find(cdr)
            assert start >= 0
            indices.extend(range(start, start + len(cdr)))
    return indices


def compute_distogram_iptm_proxy(
    distogram_logits: torch.Tensor,
    target_length: int,
    binder_sequence: str,
    is_antibody: bool = False,
) -> dict[str, float]:
    if distogram_logits.ndim == 4:
        distogram_logits = distogram_logits[0]
    binder_length = len(binder_sequence)
    assert distogram_logits.shape[0] == target_length + binder_length
    bin_distance = get_mid_points().to(distogram_logits.device)
    binder_start = target_length

    def _mean_lowest_k(entropies: torch.Tensor, k: int) -> float:
        sorted_entropies, _ = torch.sort(entropies.reshape(-1))
        k = min(k, sorted_entropies.numel())
        return float(sorted_entropies[:k].mean())

    binder_to_target_entropy = _binding_confidence_entropy(
        distogram_logits[binder_start:, :target_length, :], bin_distance, cutoff=22.0
    )
    distogram_iptm_proxy = _entropy_to_confidence(
        _mean_lowest_k(binder_to_target_entropy, k=binder_length)
    )

    cdr_distogram_iptm_proxy = float("nan")
    if is_antibody:
        try:
            cdr_indices = _cdr_indices(binder_sequence)
            cdr_rows = [binder_start + i for i in cdr_indices]
            cdr_to_target_entropy = _binding_confidence_entropy(
                distogram_logits[cdr_rows, :target_length, :], bin_distance, cutoff=22.0
            )
            cdr_distogram_iptm_proxy = _entropy_to_confidence(
                _mean_lowest_k(cdr_to_target_entropy, k=len(cdr_indices))
            )
        except Exception:
            pass

    return {
        "distogram_iptm_proxy": distogram_iptm_proxy,
        "cdr_distogram_iptm_proxy": cdr_distogram_iptm_proxy,
    }


# ---- Folding ----


_ATOM_FEATURE_DIMS = {
    "ref_pos": 0, "ref_element": 0, "ref_charge": 0,
    "ref_atom_name_chars": 0, "ref_space_uid": 0,
    "atom_attention_mask": 0, "atom_to_token": 0,
    "is_resolved": 0, "gt_coords": 1,
}


@cache
def _ensure_ccd_loaded() -> None:
    load_ccd()


def _resize_tensor(tensor: torch.Tensor, *, dim: int, size: int) -> torch.Tensor:
    current = tensor.shape[dim]
    if current >= size:
        return tensor.narrow(dim, 0, size)
    pad_shape = list(tensor.shape)
    pad_shape[dim] = size - current
    pad = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
    return torch.cat((tensor, pad), dim=dim)


def prepare_esmfold2_tensors(
    input: StructurePredictionInput,
    max_atoms: int | None = None,
    seed: int | None = None,
) -> dict[str, torch.Tensor]:
    _ensure_ccd_loaded()
    features, _ = prepare_esmfold2_input(input, seed=seed)
    if max_atoms is not None:
        for key, dim in _ATOM_FEATURE_DIMS.items():
            if key in features:
                features[key] = _resize_tensor(features[key], dim=dim, size=max_atoms)
    return features


def fold_and_get_distogram(
    model: ESMFold2ExperimentalModel,
    target_seq: str,
    target_mol_type: MolType,
    target_one_hot: torch.Tensor,
    design: torch.Tensor,
    design_spec: DesignSpec,
    num_loops: int = 0,
    num_sampling_steps: int = 1,
    calculate_confidence: bool = False,
    seed: int | None = None,
) -> dict:
    """Run ESMFold2 on [target | designed] complex, return distogram_logits + output."""
    padded_design = embed_design_soft(design, design_spec.token_indices)
    designed_seqs = _decode_designed_sequences(padded_design, design_spec.mol_type)

    seq_list = [target_seq + "|" + ds for ds in designed_seqs]
    max_atoms = None if len(seq_list) == 1 else ((len(seq_list[0]) - 1) * 14) // 32 * 32

    inputs_list = []
    for target_s, design_s in zip([target_seq] * len(designed_seqs), designed_seqs):
        inputs_raw = StructurePredictionInput(sequences=[
            _make_chain_input("0", target_s, target_mol_type),
            _make_chain_input("1", design_s, design_spec.mol_type),
        ])
        inputs_list.append(prepare_esmfold2_tensors(inputs_raw, max_atoms=max_atoms, seed=seed))

    # Verify chain mol_types seen by the model (0=protein,1=DNA,2=RNA,3=nonpolymer)
    if logger.isEnabledFor(logging.DEBUG):
        mt = inputs_list[0]["mol_type"][0].tolist()
        logger.debug(f"mol_type per token (first 20): {mt[:20]}")

    first_param = next(model.parameters())
    device, model_dtype = first_param.device, first_param.dtype
    inputs = {}
    for key in inputs_list[0]:
        t = torch.stack([inp[key] for inp in inputs_list], dim=0).to(device)
        inputs[key] = t.to(model_dtype) if t.is_floating_point() else t
    inputs["res_type_soft"] = torch.cat(
        (target_one_hot.repeat(design.size(0), 1, 1), padded_design), dim=1
    ).to(device=device, dtype=model_dtype)
    with seed_context(seed):
        output = model(
            **inputs,
            num_diffusion_samples=1,
            num_sampling_steps=num_sampling_steps,
            num_loops=num_loops,
            calculate_confidence=calculate_confidence,
            seed=seed,
        )

    result: dict = {
        "distogram_logits": output["distogram_logits"],
        "inputs": inputs,
        "inputs_list": inputs_list,
        "output": output,
        "seq_list": seq_list,
    }
    if calculate_confidence:
        result.update({
            "ptm": output.get("ptm"),
            "iptm": output.get("iptm"),
            "plddt": output.get("plddt"),
        })
    return result


# ---- Build ProteinComplex from output ----

_CHAIN_ID_ALPHABET = string.ascii_uppercase + string.ascii_lowercase + string.digits


def _asym_id_to_chain_label(asym_id: int) -> str:
    if asym_id < 0:
        raise ValueError(f"asym_id must be >= 0, got {asym_id}")
    label = ""
    n = len(_CHAIN_ID_ALPHABET)
    while True:
        label = _CHAIN_ID_ALPHABET[asym_id % n] + label
        asym_id = asym_id // n - 1
        if asym_id < 0:
            return label


def to_atom_array(
    coords, atom_to_token, res_type, residue_index, asym_id,
    mol_type, ref_atom_name_chars, ref_element, atom_attention_mask,
    plddt_per_atom=None,
) -> biotite.structure.AtomArray:
    atoms = []
    for atom_i, (atom_coord, token_idx, atom_name_chars, element_idx, is_not_pad) in enumerate(
        zip(coords, atom_to_token, ref_atom_name_chars, ref_element, atom_attention_mask)
    ):
        if not is_not_pad:
            continue
        atoms.append(biotite.structure.Atom(
            coord=atom_coord,
            chain_id=_asym_id_to_chain_label(int(asym_id[token_idx])),
            res_id=residue_index[token_idx] + 1,
            res_name=TOKENS[res_type[token_idx]],
            atom_name="".join(chr(c + 32) for c in atom_name_chars if c != 0),
            element=ELEMENTS[element_idx],
            ins_code=" ",
            hetero=mol_type[token_idx] == MOL_TYPE_NONPOLYMER,
            b_factor=float(plddt_per_atom[atom_i]) if plddt_per_atom is not None else 0.0,
        ))
    return biotite.structure.array(atoms)


def _save_structure(fold_result: dict, path: "Path") -> None:
    """Save a fold result to a mmCIF file without going through ProteinChain (handles RNA/DNA)."""
    from biotite.structure.io.pdbx import CIFFile  # type: ignore[import]
    from biotite.structure.io.pdbx import set_structure as _set_structure  # type: ignore[import]
    inputs, output = fold_result["inputs"], fold_result["output"]
    atom_arr = to_atom_array(
        coords=output["sample_atom_coords"][0].cpu().numpy(),
        atom_to_token=inputs["atom_to_token"][0].cpu().numpy(),
        res_type=inputs["res_type"][0].cpu().numpy(),
        residue_index=inputs["token_index"][0].cpu().numpy(),
        asym_id=inputs["asym_id"][0].cpu().numpy(),
        mol_type=inputs["mol_type"][0].cpu().numpy(),
        ref_atom_name_chars=inputs["ref_atom_name_chars"][0].cpu().numpy(),
        ref_element=inputs["ref_element"][0].cpu().numpy(),
        atom_attention_mask=inputs["atom_attention_mask"][0].cpu().numpy(),
    )
    cif = CIFFile()
    _set_structure(cif, atom_arr)
    cif.write(str(path))


def build_complex(inputs: dict[str, torch.Tensor], output: dict[str, Any]) -> ProteinComplex:
    atom_arr = to_atom_array(
        coords=output["sample_atom_coords"][0].cpu().numpy(),
        atom_to_token=inputs["atom_to_token"][0].cpu().numpy(),
        res_type=inputs["res_type"][0].cpu().numpy(),
        residue_index=inputs["token_index"][0].cpu().numpy(),
        asym_id=inputs["asym_id"][0].cpu().numpy(),
        mol_type=inputs["mol_type"][0].cpu().numpy(),
        ref_atom_name_chars=inputs["ref_atom_name_chars"][0].cpu().numpy(),
        ref_element=inputs["ref_element"][0].cpu().numpy(),
        atom_attention_mask=inputs["atom_attention_mask"][0].cpu().numpy(),
    )
    return ProteinComplex.from_chains(
        [ProteinChain.from_atomarray(a) for a in biotite.structure.chain_iter(atom_arr)]
    )


# ---- LM loss (protein designing chains only) ----


@cache
def _folding_trunk_to_lm_aa_vocab_matrix(device: torch.device) -> torch.Tensor:
    three_to_one_map = {v: k for k, v in PROTEIN_1TO3.items()}
    ft_aas = [three_to_one_map[TOKENS[i]] for i in PROTEIN_TOKEN_INDICES]
    lm_vocab = sorted(ESMCTokenizer().vocab.items(), key=lambda x: x[1])
    lm_aas = [lm_vocab[i][0] for i in range(4, 24)]
    matrix = torch.zeros(20, 20)
    for ft_idx, ft_aa in enumerate(ft_aas):
        matrix[ft_idx, lm_aas.index(ft_aa)] = 1
    return matrix.to(device=device)


def _one_hot_from_probs(probs: torch.Tensor) -> torch.Tensor:
    return F.one_hot(torch.argmax(probs, dim=-1), num_classes=probs.size(-1)).to(probs.dtype)


def _straight_through(discrete: torch.Tensor, continuous: torch.Tensor) -> torch.Tensor:
    return continuous + (discrete - continuous).detach()


def compute_esmc_pseudoperplexity_nll(
    esmc_model: ESMCForMaskedLM,
    binder_design: torch.Tensor,
    score_mask: torch.Tensor,
    batch_size: int = 4,
    n_passes: int = 4,
) -> torch.Tensor:
    device = binder_design.device
    lm_vocab_size = esmc_model.config.vocab_size
    model_dtype = esmc_model.esmc.embed.weight.dtype

    target_esm = binder_design @ _folding_trunk_to_lm_aa_vocab_matrix(device)
    input_esm = _straight_through(_one_hot_from_probs(target_esm), target_esm)
    input_ids = torch.zeros(
        (binder_design.size(0), binder_design.size(1) + 2, lm_vocab_size),
        dtype=model_dtype, device=device,
    )
    tokenizer = ESMCTokenizer()
    input_ids[:, 0, tokenizer.cls_token_id] = 1
    input_ids[:, -1, tokenizer.eos_token_id] = 1
    input_ids[:, 1:-1, 4:24] = input_esm.to(model_dtype)

    if score_mask.ndim == 1:
        score_mask = score_mask.unsqueeze(0).expand(binder_design.size(0), -1)
    score_mask = score_mask.to(device=device, dtype=torch.bool)

    mask_token = torch.zeros(lm_vocab_size, dtype=model_dtype, device=device)
    mask_token[esmc_model.config.mask_token_id] = 1
    esmc = esmc_model.esmc

    losses = []
    for batch_idx in range(binder_design.size(0)):
        position_indices = score_mask[batch_idx].nonzero(as_tuple=False).flatten()
        num_positions = int(position_indices.numel())
        if num_positions == 0:
            raise ValueError("ESMC pseudoperplexity score mask selected zero positions.")

        num_masked = max(1, math.ceil(ESMC_MASK_FRACTION * num_positions))
        random_scores = torch.rand((n_passes, num_positions), device=device)
        masked_offsets = random_scores.topk(num_masked, dim=-1, largest=False).indices
        pass_masks = torch.zeros((n_passes, binder_design.size(1)), dtype=torch.bool, device=device)
        pass_masks[torch.arange(n_passes, device=device)[:, None], position_indices[masked_offsets]] = True

        masked_sequences = input_ids[batch_idx:batch_idx + 1].repeat(n_passes, 1, 1)
        mask_rows, mask_cols = pass_masks.nonzero(as_tuple=True)
        masked_sequences[mask_rows, mask_cols + 1] = mask_token

        target_weights = target_esm[batch_idx]
        masked_nlls = []
        for start in range(0, n_passes, batch_size):
            stop = min(start + batch_size, n_passes)
            chunk = masked_sequences[start:stop]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                hidden, *_ = esmc.transformer(
                    chunk @ esmc.embed.weight.to(chunk.dtype),
                    sequence_id=None, layers_to_collect=[], output_attentions=False,
                )
                logits = esmc_model.lm_head(hidden)
            log_probs = logits.log_softmax(dim=-1)[:, 1:-1, 4:24]
            nlls = -(log_probs * target_weights.to(log_probs.dtype).unsqueeze(0)).sum(dim=-1)
            masked_nlls.append(nlls[pass_masks[start:stop]])
        losses.append(torch.cat(masked_nlls, dim=0).mean())

    return torch.stack(losses, dim=0)


# ---- Gradient normalization ----


def normalized_gradient_tensor(grad: torch.Tensor, gradient_mask: torch.Tensor) -> torch.Tensor:
    masked_grad = grad * gradient_mask
    index_has_nonzero_grad = torch.square(masked_grad).sum(-1) > 0
    eff_L = index_has_nonzero_grad.sum(-1)
    grad_norm = torch.linalg.norm(masked_grad, axis=(-1, -2))
    normalized_grad = (masked_grad / (grad_norm[:, None, None] + 1e-7)) * torch.sqrt(eff_L[:, None, None])
    return normalized_grad * gradient_mask


# ---- Main design function ----


def design_chain(
    inversion_models: dict[str, ESMFold2ExperimentalModel],
    hf_critic_models: dict[str, ESMFold2ExperimentalModel],
    esmc_model: ESMCForMaskedLM | None,
    target_sequence: str,
    target_mol_type: MolType,
    design_spec: DesignSpec,
    is_antibody: bool = False,
    seed: int = 0,
    batch_size: int = 1,
    steps: int = STEPS,
    learning_rate: float = LEARNING_RATE,
    confidence_temp_threshold: float = 0.05,
    final_num_loops: int = 3,
    final_num_sampling_steps: int = 200,
    output_dir: "Path | str | None" = None,
    hotspot_indices: list[int] | None = None,
    save_interval: int = 10,
) -> tuple[list[str], dict[int, dict[str, torch.Tensor]], list[dict]]:
    """
    Algorithm 11 (generalised): Gradient-Guided Binder Sequence Optimisation.

    Supports protein, RNA, and DNA designing chains against any target molecule.
    ESMC language-model regularisation is skipped for RNA/DNA designing chains.

    hotspot_indices: 0-based residue indices in the target that the binder must contact.
    save_interval:   save an intermediate structure every this many steps (when confidence is active).

    Returns (best_sequences, trajectory, critic_results).
    """
    assert "|" not in target_sequence, "Multi-chain targets not supported; provide a single chain."

    device = "cuda"
    target_one_hot = sequence_to_one_hot_full(target_sequence, target_mol_type, device=device)

    use_lm = design_spec.is_protein and esmc_model is not None
    lm_weight = LM_WEIGHT_ANTIBODY if is_antibody else LM_WEIGHT_PROTEIN

    with seed_context(seed), torch.device(device):
        logits = build_initial_soft_sequence_logits(design_spec, batch_size=batch_size)
        gradient_mask = build_gradient_mask(design_spec, batch_size=batch_size)

    trajectory: dict[int, dict[str, torch.Tensor]] = {}
    global_step = 0

    def run_step(
        logits: torch.Tensor,
        optimizer: optim.Optimizer,
        temperature: float,
        calculate_confidence: bool,
    ) -> tuple[torch.Tensor, list[str], list[float] | None]:
        nonlocal global_step
        optimizer.zero_grad()

        random.seed(seed + global_step)
        inversion_model = list(inversion_models.values())[random.randint(0, len(inversion_models) - 1)]
        design = F.softmax(logits / temperature, dim=-1)

        fold_result = fold_and_get_distogram(
            inversion_model, target_sequence, target_mol_type, target_one_hot,
            design, design_spec,
            num_loops=1,
            num_sampling_steps=50 if calculate_confidence else 1,
            calculate_confidence=calculate_confidence,
            seed=seed + global_step,
        )
        sequences: list[str] = fold_result["seq_list"]
        losses = compute_structure_losses(fold_result["distogram_logits"], design_spec.length, hotspot_indices)
        structure_loss = losses["total_loss"]
        structure_grad = torch.autograd.grad(structure_loss.mean(), logits)[0]

        if use_lm:
            design = F.softmax(logits / temperature, dim=-1)
            score_mask = gradient_mask.sum(dim=-1) > 0
            with seed_context(seed + global_step):
                plm_loss = compute_esmc_pseudoperplexity_nll(
                    esmc_model=esmc_model,
                    binder_design=design,
                    score_mask=score_mask,
                    batch_size=4,
                    n_passes=4,
                )
            plm_grad = torch.autograd.grad(plm_loss.mean(), logits)[0]
            logits.grad = (
                normalized_gradient_tensor(structure_grad, gradient_mask)
                + lm_weight * normalized_gradient_tensor(plm_grad, gradient_mask)
            )
            total_loss = structure_loss + plm_loss.to(structure_loss.device)
        else:
            plm_loss = torch.zeros(batch_size)
            logits.grad = normalized_gradient_tensor(structure_grad, gradient_mask)
            total_loss = structure_loss

        for g in optimizer.param_groups:
            g["lr"] = LEARNING_RATE * temperature
        optimizer.step()

        step = global_step
        step_losses = {k: v.detach().cpu() for k, v in losses.items()}
        step_losses["plm_loss"] = plm_loss.detach().cpu()
        step_losses["total_loss"] = total_loss.detach().cpu()
        trajectory[step] = step_losses

        if step % LOG_INTERVAL == 0:
            loss_str = "  ".join(f"{k}={v.mean().item():.4f}" for k, v in step_losses.items())
            iptm_val = fold_result.get("iptm")
            iptm_str = f"  iptm={iptm_val.mean().item():.4f}" if iptm_val is not None else ""
            logger.info(f"  step {step:3d}  |  {loss_str}  T={temperature:.4f}{iptm_str}")

        if calculate_confidence and fold_result.get("iptm") is not None and step % save_interval == 0:
            out_dir = _out_dir
            out_dir.mkdir(exist_ok=True)
            for b in range(batch_size):
                try:
                    path = out_dir / f"intermediate_step{step:03d}_b{b}.cif"
                    _save_structure(fold_result, path)
                    logger.info(f"  saved intermediate structure → {path}")
                except Exception as e:
                    logger.warning(f"  could not save intermediate structure: {e}")

        global_step += 1
        return logits, sequences, fold_result.get("iptm", None)

    _out_dir = Path(output_dir) if output_dir is not None else Path.cwd() / "designs"

    optimizer = optim.SGD([logits], lr=learning_rate)
    best_iptm: list[float] = [-1.0] * batch_size
    best_sequences: list[str] = [""] * batch_size
    last_sequences: list[str] = [""] * batch_size

    for step in range(steps):
        t = (step + 1) / steps
        remaining = 0.5 * (1 + math.cos(math.pi * t))
        temperature = TEMPERATURE_MIN + (1 - TEMPERATURE_MIN) * remaining
        logits, sequences, iptm = run_step(
            logits, optimizer, temperature=temperature,
            calculate_confidence=temperature < confidence_temp_threshold,
        )
        last_sequences = sequences
        if iptm is not None:
            for b in range(batch_size):
                if iptm[b] is not None and iptm[b] > best_iptm[b]:
                    best_iptm[b] = iptm[b]
                    best_sequences[b] = sequences[b]

    # Fall back to last sequences if no confident fold was encountered.
    for b in range(batch_size):
        if best_sequences[b] == "":
            best_sequences[b] = last_sequences[b]

    # Score with critic models
    target_length = len(target_sequence)
    critic_results: list[dict] = []
    for batch_idx in range(batch_size):
        best_seq = best_sequences[batch_idx]
        design_seq = best_seq.split("|")[-1]
        binder_one_hot_full = sequence_to_one_hot_full(design_seq, design_spec.mol_type, device=device)
        binder_design = binder_one_hot_full[:, :, design_spec.token_indices]

        for critic_name, critic_model in hf_critic_models.items():
            final_fold = fold_and_get_distogram(
                critic_model, target_sequence, target_mol_type, target_one_hot,
                binder_design, design_spec,
                num_loops=final_num_loops, num_sampling_steps=final_num_sampling_steps,
                calculate_confidence=True, seed=seed,
            )
            pred_complex = build_complex(final_fold["inputs"], final_fold["output"])
            out_dir = _out_dir
            out_dir.mkdir(exist_ok=True)
            try:
                final_path = out_dir / f"final_b{batch_idx}.cif"
                _save_structure(final_fold, final_path)
                logger.info(f"  saved final structure → {final_path}")
            except Exception as e:
                logger.warning(f"  could not save final structure: {e}")
            iptm_proxy_scores = compute_distogram_iptm_proxy(
                final_fold["distogram_logits"], target_length, design_seq, is_antibody
            )
            iptm = final_fold["iptm"].item() if final_fold["iptm"] is not None else None
            critic_results.append({
                "is_antibody": is_antibody,
                "critic_name": critic_name,
                "batch_idx": batch_idx,
                "designed_sequence": best_seq,
                "complex": pred_complex,
                "final_loss": trajectory[global_step - 1]["total_loss"][batch_idx].item(),
                "iptm": iptm,
                "logits": logits[batch_idx].detach().cpu(),
                **iptm_proxy_scores,
            })

    if not critic_results:
        for batch_idx in range(batch_size):
            critic_results.append({
                "batch_idx": batch_idx,
                "designed_sequence": best_sequences[batch_idx],
                "final_loss": trajectory[global_step - 1]["total_loss"][batch_idx].item(),
                "logits": logits[batch_idx].detach().cpu(),
            })

    # Persist results
    _out_dir.mkdir(exist_ok=True)
    import json

    summary = []
    for r in critic_results:
        entry = {k: v for k, v in r.items() if k not in ("complex", "logits")}
        # convert any remaining tensors to Python scalars
        entry = {k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in entry.items()}
        summary.append(entry)
    (_out_dir / "critic_results.json").write_text(json.dumps(summary, indent=2))

    logits_dict = {
        f"b{r['batch_idx']}": r["logits"]
        for r in critic_results if "logits" in r
    }
    if logits_dict:
        torch.save(logits_dict, _out_dir / "logits.pt")

    logger.info(f"  results saved to {_out_dir}")

    return best_sequences, trajectory, critic_results


# ---- Model loading ----

_ESMC = None


def _load_hf_model(
    critic_name: str, lm_dropout: float, cache_esmc: bool, device: str,
    cache_dir: str = CACHE_DIR,
) -> Any:
    global _ESMC
    repo_id = f"biohub/{critic_name}"
    model = ESMFold2ExperimentalModel.from_pretrained(
        repo_id, load_esmc=not cache_esmc, cache_dir=cache_dir
    )
    if cache_esmc:
        if _ESMC is None:
            model.load_esmc(model.config.esmc_id)
            _ESMC = model._esmc
        else:
            model._esmc = _ESMC
    model.configure_lm_dropout(lm_dropout, force_lm_dropout_during_inference=True)
    model.set_kernel_backend("cuequivariance" if CUE_AVAILABLE else None)
    model = model.to(device=device).eval().requires_grad_(False)
    # _esmc may not be a registered submodule, so .to() might not move it
    if hasattr(model, "_esmc") and model._esmc is not None:
        model._esmc = model._esmc.to(device=device)
    return model


def _apply_torch_compile(model: torch.nn.Module) -> None:
    torch._dynamo.config.cache_size_limit = 512
    torch._dynamo.config.accumulated_cache_size_limit = 512
    compile_targets = (ESMFold2MSAEncoder, PairUpdateBlock, TransformerBlock)

    def _maybe_compile_module(module: torch.nn.Module) -> None:
        if isinstance(module, compile_targets):
            module.forward = torch.compile(module.forward)  # pyright: ignore

    model.apply(_maybe_compile_module)


class UniBinderDesign:
    lm_name = "biohub/ESMC-6B"
    inversion_model_names: list[str] = [
        #"ESMFold2-Experimental-Fast",
        "ESMFold2-Experimental-Fast"
    ]
    hero_critic_hf_paths: list[str] = [
        "ESMFold2-Experimental-Fast",
    ]
    scaling_critic_hf_paths: list[str] = []

    def load(self, use_scaling_critics: bool = False, load_esmc: bool = True, cache_dir: str = CACHE_DIR) -> None:
        if use_scaling_critics:
            self.scaling_critic_hf_paths = [
                f"ESMFold2-Experimental-Fast-base{size}-step{step}k"
                for size in ("300M", "600M", "6B")
                for step in ("250", "500", "750", "1000", "1500")
            ]

        self.inversion_models = {
            name: _load_hf_model(name, lm_dropout=0.5, cache_esmc=True, device="cuda", cache_dir=cache_dir)
            for name in self.inversion_model_names
        }
        if CHECKPOINT_INVERSION:
            for model in self.inversion_models.values():
                apply_activation_checkpointing(
                    model,
                    checkpoint_wrapper_fn=partial(
                        checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
                    ),
                    check_fn=lambda m: isinstance(m, (PairUpdateBlock, TransformerBlock)),
                )
        if COMPILE:
            for model in self.inversion_models.values():
                _apply_torch_compile(model)

        # Reuse inversion models as critics — no separate CPU models.
        self.hf_critic_models: dict[str, Any] = dict(self.inversion_models)

        if load_esmc:
            self.esmc_model: ESMCForMaskedLM | None = ESMCForMaskedLM.from_pretrained(
                self.lm_name, torch_dtype=torch.float32, cache_dir=cache_dir
            )
            if REUSE_ESMC:
                del self.esmc_model.esmc
                torch.cuda.empty_cache()
                first_inv = next(iter(self.inversion_models.values()))
                self.esmc_model.esmc = first_inv._esmc
            self.esmc_model = self.esmc_model.cuda().eval().requires_grad_(False)
            if CHECKPOINT_LM:
                apply_activation_checkpointing(
                    self.esmc_model,
                    checkpoint_wrapper_fn=partial(
                        checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
                    ),
                    check_fn=lambda m: isinstance(m, TransformerBlock),
                )
        else:
            self.esmc_model = None

    def design(
        self,
        target_sequence: str,
        design_spec: DesignSpec,
        target_mol_type: MolType | str = MolType.PROTEIN,
        is_antibody: bool = False,
        seed: int = 0,
        batch_size: int = 1,
        steps: int = 200,
        learning_rate: float = LEARNING_RATE,
        confidence_temp_threshold: float = 0.5,
        final_num_loops: int = 3,
        final_num_sampling_steps: int = 200,
        output_dir: "Path | str | None" = None,
        hotspot_indices: list[int] | None = None,
        save_interval: int = 10,
    ) -> tuple[list[str], dict[int, dict[str, torch.Tensor]], list[dict]]:
        """
        Design a chain to bind target_sequence.

        target_mol_type:  molecule type of the target ("protein", "rna", "dna").
        design_spec:      use make_design_spec() to build this.
        output_dir:       where to save intermediate/final structures (default: cwd/designs).
        hotspot_indices:  0-based residue indices in the target the binder must contact.
        save_interval:    save an intermediate structure every this many low-temperature steps.
        """
        return design_chain(
            self.inversion_models,
            self.hf_critic_models,
            self.esmc_model,
            target_sequence=target_sequence,
            target_mol_type=MolType(target_mol_type),
            design_spec=design_spec,
            is_antibody=is_antibody,
            seed=seed,
            batch_size=batch_size,
            steps=steps,
            learning_rate=learning_rate,
            confidence_temp_threshold=confidence_temp_threshold,
            final_num_loops=final_num_loops,
            final_num_sampling_steps=final_num_sampling_steps,
            output_dir=output_dir,
            hotspot_indices=hotspot_indices,
            save_interval=save_interval,
        )


if __name__ == "__main__":
    # --- Example 1: protein minibinder against PD-L1 ---
    runner = UniBinderDesign()
    runner.load(use_scaling_critics=False)
    # spec = make_design_spec("protein", 80)
    # seqs, traj, results = runner.design(
    #     target_sequence="AFTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNA",
    #     design_spec=spec,
    #     target_mol_type="protein",
    #     seed=0,
    #     batch_size=1,
    # )
    # logger.info(f"Designed: {seqs}")

    # --- Example 2: RNA aptamer against a protein ---
    # spec = make_design_spec("rna", 40)
    # runner.design(target_sequence="MKTV...", design_spec=spec, target_mol_type="protein")

    #--- Example 3: fix specific positions ---
    spec = make_design_spec("protein", 60, fixed_positions={0: "A",1:"A", 58:"G",59: "G"})
    seqs, traj, results = runner.design(
        target_sequence="AFTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNA",
        design_spec=spec,
        target_mol_type="protein",
        seed=0,
        batch_size=1,
        hotspot_indices=[]
    )
    logger.info(f"Designed: {seqs}")