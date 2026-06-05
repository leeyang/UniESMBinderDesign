"""
Design a DNA aptamer that binds ATP using gradient-guided optimization.

ATP is provided as a CCD ligand.  A DNA aptamer sequence is optimized via
ESMFold2's distogram: intra-chain contacts fold the aptamer into a compact
structure; inter-chain contacts pull it onto the ATP molecule.

Usage
-----
    python design_atp.py

Outputs
-------
    designs/final_b0.cif   — final structure (ATP + aptamer)
    designs/intermediate_step*.cif — snapshots during optimization
"""

import os
import sys
import math
import random
import logging
from pathlib import Path

os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
os.environ.setdefault("HF_HUB_CACHE", "/mnt/d/mol2/invESM/lib/pretrained")

import torch
import torch.nn.functional as F
import torch.optim as optim
from transformers.models.esmfold2.modeling_esmfold2_common import _seed_context as seed_context

sys.path.insert(0, str(Path(__file__).parent))
from uni_binder_design import (
    UniBinderDesign,
    make_design_spec,
    MolType,
    build_initial_soft_sequence_logits,
    build_gradient_mask,
    embed_design_soft,
    _decode_designed_sequences,
    _save_structure,
    compute_structure_losses,
    normalized_gradient_tensor,
    prepare_esmfold2_tensors,
    TOKENS,
    LEARNING_RATE,
    STEPS,
    TEMPERATURE_MIN,
    LOG_INTERVAL,
    LOSS_WEIGHTS,
)
from esm.models.esmfold2 import StructurePredictionInput
from esm.utils.structure.input_builder import DNAInput, LigandInput

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---- Configuration ----

LIGAND_CCD       = "ATP"    # CCD code of the target small molecule
APTAMER_LENGTH   = 35       # number of nucleotides in the DNA aptamer
FIXED_POSITIONS  = {}       # e.g. {0: "G", 1: "G"} to pin specific positions
STEPS            = 150
LEARNING_RATE    = 0.1
SAVE_INTERVAL    = 10       # save a structure every N low-temp steps
OUTPUT_DIR       = Path("designs")
SEED             = 0
BATCH_SIZE       = 1
CONFIDENCE_TEMP_THRESHOLD = 0.05


# ---- Ligand-aware fold function ----

def fold_dna_vs_ligand(
    model,
    ligand_ccd: str,
    design: torch.Tensor,
    design_spec,
    num_sampling_steps: int = 1,
    calculate_confidence: bool = False,
    seed: int = 0,
) -> dict:
    """
    Run ESMFold2 on [ligand | DNA aptamer] complex.

    For the ligand tokens, res_type_soft is set to the hard one-hot of their
    discrete res_type (they are fixed — not optimized).  Only the aptamer
    tokens carry the differentiable soft assignment from `design`.
    """
    padded_design = embed_design_soft(design, design_spec.token_indices)
    designed_seqs = _decode_designed_sequences(padded_design, design_spec.mol_type)

    first_param = next(model.parameters())
    device, model_dtype = first_param.device, first_param.dtype

    inputs_list = []
    for ds in designed_seqs:
        inputs_raw = StructurePredictionInput(sequences=[
            LigandInput(id="0", ccd=ligand_ccd),
            DNAInput(id="1", sequence=ds),
        ])
        inputs_list.append(prepare_esmfold2_tensors(inputs_raw, seed=seed))

    # Find the ligand/aptamer split from asym_id
    sample_asym = inputs_list[0]["asym_id"]   # [L_total]
    ligand_asym_val = sample_asym[0].item()
    ligand_len = int((sample_asym == ligand_asym_val).sum().item())

    # Stack into a batch and move to device/dtype
    inputs: dict = {}
    for key in inputs_list[0]:
        t = torch.stack([inp[key] for inp in inputs_list], dim=0).to(device)
        inputs[key] = t.to(model_dtype) if t.is_floating_point() else t

    # res_type_soft: hard one-hot for ligand atoms, differentiable soft for aptamer
    ligand_res_type = inputs["res_type"][:, :ligand_len].long()          # [B, L_lig]
    ligand_one_hot  = F.one_hot(ligand_res_type, num_classes=len(TOKENS)) \
                        .to(dtype=model_dtype)                            # [B, L_lig, V]
    inputs["res_type_soft"] = torch.cat(
        (ligand_one_hot, padded_design.to(device=device, dtype=model_dtype)), dim=1
    )

    with seed_context(seed):
        output = model(
            **inputs,
            num_diffusion_samples=1,
            num_sampling_steps=num_sampling_steps,
            calculate_confidence=calculate_confidence,
            seed=seed,
        )

    result: dict = {
        "distogram_logits": output["distogram_logits"],
        "inputs": inputs,
        "output": output,
        "seq_list": [f"{ligand_ccd}|{ds}" for ds in designed_seqs],
        "ligand_len": ligand_len,
    }
    if calculate_confidence:
        result.update({
            "ptm":   output.get("ptm"),
            "iptm":  output.get("iptm"),
            "plddt": output.get("plddt"),
        })
    return result


# ---- Main design loop ----

def design_dna_aptamer_for_ligand(
    model,
    ligand_ccd: str = LIGAND_CCD,
    aptamer_length: int = APTAMER_LENGTH,
    fixed_positions: dict | None = None,
    steps: int = STEPS,
    learning_rate: float = LEARNING_RATE,
    confidence_temp_threshold: float = CONFIDENCE_TEMP_THRESHOLD,
    save_interval: int = SAVE_INTERVAL,
    output_dir: Path = OUTPUT_DIR,
    seed: int = SEED,
    batch_size: int = BATCH_SIZE,
) -> tuple[list[str], dict, list[dict]]:
    device = "cuda"
    design_spec = make_design_spec("dna", aptamer_length, fixed_positions)

    with seed_context(seed), torch.device(device):
        logits        = build_initial_soft_sequence_logits(design_spec, batch_size=batch_size)
        gradient_mask = build_gradient_mask(design_spec, batch_size=batch_size)

    trajectory: dict[int, dict] = {}
    _step_cell = [0]

    def run_step(logits, optimizer, temperature, calculate_confidence):
        optimizer.zero_grad()
        step = _step_cell[0]

        random.seed(seed + step)
        design = F.softmax(logits / temperature, dim=-1)

        fold_result = fold_dna_vs_ligand(
            model, ligand_ccd, design, design_spec,
            num_sampling_steps=50 if calculate_confidence else 1,
            calculate_confidence=calculate_confidence,
            seed=seed + step,
        )

        losses         = compute_structure_losses(fold_result["distogram_logits"], design_spec.length)
        structure_loss = losses["total_loss"]
        structure_grad = torch.autograd.grad(structure_loss.mean(), logits)[0]
        logits.grad    = normalized_gradient_tensor(structure_grad, gradient_mask)

        for g in optimizer.param_groups:
            g["lr"] = learning_rate * temperature
        optimizer.step()

        step_losses = {k: v.detach().cpu() for k, v in losses.items()}
        trajectory[step] = step_losses

        if step % LOG_INTERVAL == 0:
            loss_str = "  ".join(f"{k}={v.mean().item():.4f}" for k, v in step_losses.items())
            iptm_val = fold_result.get("iptm")
            iptm_str = f"  iptm={iptm_val.mean().item():.4f}" if iptm_val is not None else ""
            logger.info(f"  step {step:3d}  |  {loss_str}  T={temperature:.4f}{iptm_str}")

        if calculate_confidence and fold_result.get("iptm") is not None \
                and step % save_interval == 0:
            output_dir.mkdir(exist_ok=True)
            for b in range(batch_size):
                try:
                    path = output_dir / f"intermediate_step{step:03d}_b{b}.cif"
                    _save_structure(fold_result, path)
                    logger.info(f"  saved → {path}")
                except Exception as e:
                    logger.warning(f"  could not save intermediate structure: {e}")

        _step_cell[0] += 1
        return logits, fold_result["seq_list"], fold_result.get("iptm")

    optimizer = optim.SGD([logits], lr=learning_rate)
    best_iptm    = [-1.0] * batch_size
    best_seqs    = [""] * batch_size
    last_seqs    = [""] * batch_size

    for step in range(steps):
        t           = (step + 1) / steps
        remaining   = 0.5 * (1 + math.cos(math.pi * t))
        temperature = TEMPERATURE_MIN + (1 - TEMPERATURE_MIN) * remaining
        logits, sequences, iptm = run_step(
            logits, optimizer, temperature,
            calculate_confidence=temperature < confidence_temp_threshold,
        )
        last_seqs = sequences
        if iptm is not None:
            for b in range(batch_size):
                val = iptm[b].item() if hasattr(iptm[b], "item") else float(iptm[b])
                if val > best_iptm[b]:
                    best_iptm[b]  = val
                    best_seqs[b]  = sequences[b]

    for b in range(batch_size):
        if best_seqs[b] == "":
            best_seqs[b] = last_seqs[b]

    # Final high-quality fold
    results: list[dict] = []
    for b in range(batch_size):
        aptamer_seq = best_seqs[b].split("|")[-1]
        binder_one_hot = torch.zeros(1, design_spec.length, len(TOKENS), device=device)
        from uni_binder_design import DNA_TOKEN_INDICES, TOKEN_IDS, DNA_1TO3
        for i, nt in enumerate(aptamer_seq):
            tok_idx = TOKEN_IDS.get(DNA_1TO3.get(nt, ""), 0)
            if tok_idx < len(TOKENS):
                binder_one_hot[0, i, tok_idx] = 1.0
        binder_design = binder_one_hot[:, :, design_spec.token_indices]

        final_fold = fold_dna_vs_ligand(
            model, ligand_ccd, binder_design, design_spec,
            num_sampling_steps=200, calculate_confidence=True, seed=seed,
        )
        output_dir.mkdir(exist_ok=True)
        try:
            path = output_dir / f"final_b{b}.cif"
            _save_structure(final_fold, path)
            logger.info(f"  saved final → {path}")
        except Exception as e:
            logger.warning(f"  could not save final structure: {e}")

        iptm_val = final_fold.get("iptm")
        results.append({
            "batch_idx":        b,
            "aptamer_sequence": aptamer_seq,
            "ligand_ccd":       ligand_ccd,
            "iptm":             iptm_val.item() if iptm_val is not None else None,
            "best_iptm_during_opt": best_iptm[b],
        })
        logger.info(f"  batch {b}: {aptamer_seq}  iptm={results[-1]['iptm']}")

    return best_seqs, trajectory, results


# ---- Entry point ----

if __name__ == "__main__":
    runner = UniBinderDesign()
    runner.load(use_scaling_critics=False, load_esmc=False)   # DNA: no ESMC needed
    model  = next(iter(runner.inversion_models.values()))

    best_seqs, traj, results = design_dna_aptamer_for_ligand(
        model=model,
        ligand_ccd=LIGAND_CCD,
        aptamer_length=APTAMER_LENGTH,
        fixed_positions=FIXED_POSITIONS,
        steps=STEPS,
        learning_rate=LEARNING_RATE,
        confidence_temp_threshold=CONFIDENCE_TEMP_THRESHOLD,
        save_interval=SAVE_INTERVAL,
        output_dir=OUTPUT_DIR,
        seed=SEED,
        batch_size=BATCH_SIZE,
    )

    print("\n=== Results ===")
    for r in results:
        print(f"  b{r['batch_idx']}: {r['aptamer_sequence']}")
        print(f"         iptm={r['iptm']}  best_during_opt={r['best_iptm_during_opt']:.4f}")
