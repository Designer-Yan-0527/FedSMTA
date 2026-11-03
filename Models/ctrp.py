"""CTRP v1 -- Calibrated Temporal Residual Prompt (RUNCOMMANDS Sec.33,
pre-registered 2026-11-03; GPT-frozen 9-point verdict).

This module hosts the ONE alpha-conditioned forward kernel shared by the
production CTRP evaluator and the Probe-H replay (Test 8). Per the frozen
verdict it is:

    same evaluator kernel + different row-selection protocol

NOT two evaluators. The kernel pieces below are copied semantically
verbatim from diagnostics/probe_prompt_step_calibration.py (Probe-H,
validated on CIFAR-100 + ImageNet-R, OVERSHOOT-CONFIRMED-CROSS-DATASET):

  * ShimPrompt / forward_with_prompt: the prompt-intervention forward
    (bypasses routing; route is read from prompt_key ONLY, and CTRP never
    touches prompt_key, so route(alpha) == route(natural) for every alpha
    by construction -- asserted separately via key-hash invariance);
  * row CE = logsumexp(lg, 1) - lg.gather(1, y_local)  (fp32 logits,
    class-masked via index_select(1, mpos), y_local = searchsorted);
  * selector = np.argmin over the ASCENDING alpha grid (first extremum =
    smallest alpha on ties -- frozen tie-breaking, Probe-H semantics;
    no isclose tolerance was added there and none is added here);
  * the per-alpha token blend is realized in the Probe-H operation order
    ``(1 - a) * prev + a * cand`` -- mathematically identical to the frozen
    closed form ``prev + a * (cand - prev)`` and bitwise identical to the
    validated kernel (the alpha-sweep equivalence gate, Test 8, compares
    against Probe-H numbers at |dCE| <= 1e-6, so the op order matters).

Row-selection protocols are NOT part of the kernel:

  * production matched protocol: deterministic class-stratified
    80/20 core/val nested split of the inner 70% train, seed
    a5_stable_seed(seed, client, task) (Models.Client_DF.a5_stable_seed);
  * Test-8 probe-compat protocol (test/helper layer ONLY, no production
    CLI): probe's stratified_pick/_stratified_split with derived seeds
    51/52 and the <=512-row cap. These two helpers live here so the
    production kernel and the probe share one implementation of the
    stratification primitives.

fail-closed error codes (frozen): CTRP_V1_UNSUPPORTED_REFIT_TIMING,
CTRP_PREMISE_FAIL, CTRP_RESUME_ANCHOR_MISSING.
"""

import hashlib

import numpy as np
import torch
import torch.nn as nn

CTRP_MODES = ('off', 'endpoint', 'fixed_alpha', 'ctrp_val', 'ctrp_train')

DEFAULT_ALPHA_GRID = ",".join(
    str(round(i * 0.05, 2)) for i in range(21))   # 0.0, 0.05, ..., 1.0


# ----------------------------------------------------------------------
# hashes (house style: sha256 over sorted state, [:16] hex)
# ----------------------------------------------------------------------

def hash_prompt_module(module):
    """Hash of a full Global_Prompt state (prompt values + prompt_key +
    any other registered state). Used for the frozen assertion
    loaded_candidate_prompt_hash == server_candidate_prompt_hash."""
    h = hashlib.sha256()
    sd = module.state_dict()
    for k in sorted(sd):
        v = sd[k].detach().cpu().contiguous()
        h.update(k.encode())
        h.update(v.numpy().tobytes())
    return h.hexdigest()[:16]


def hash_tensor(t):
    """Hash of a raw tensor (prompt-value pools / anchors)."""
    arr = t.detach().cpu().contiguous().numpy()
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()) \
        .hexdigest()[:16]


# ----------------------------------------------------------------------
# config parsing / validation
# ----------------------------------------------------------------------

def parse_alpha_grid(grid_str, require_one=True):
    """Parse 'a,b,c' into an ascending unique float list in [0, 1].
    require_one: the grid must contain 1.0 (the endpoint reference arm --
    the natural forward IS the alpha=1 arm, Probe-H semantics)."""
    try:
        grid = [float(x) for x in str(grid_str).split(",") if str(x).strip()]
    except ValueError:
        raise SystemExit(
            f"CTRP: --ctrp_alpha_grid is not a comma-separated float list: "
            f"{grid_str!r}")
    if not grid:
        raise SystemExit("CTRP: --ctrp_alpha_grid is empty")
    if any(g < 0.0 or g > 1.0 for g in grid):
        raise SystemExit(
            f"CTRP: --ctrp_alpha_grid values must be within [0, 1]: {grid}")
    if any(grid[i] >= grid[i + 1] for i in range(len(grid) - 1)):
        raise SystemExit(
            f"CTRP: --ctrp_alpha_grid must be strictly ascending: {grid}")
    if require_one and 1.0 not in grid:
        raise SystemExit(
            "CTRP: --ctrp_alpha_grid must contain 1.0 (the natural / "
            "endpoint reference arm)")
    return grid


# ----------------------------------------------------------------------
# the alpha-conditioned forward kernel (Probe-H verbatim semantics)
# ----------------------------------------------------------------------

class ShimPrompt(nn.Module):
    """Replaces vit.prompt to inject a caller-supplied prompt token block.

    Replicates Global_Prompt.forward's batched_prompt branch exactly
    (official ``torch.cat([batched_prompt, x_embed], dim=1)`` +
    total_prompt_len). Must be an nn.Module: vit.prompt is a registered
    child module, so assigning a plain object would be rejected by
    Module.__setattr__.

    Probe-H _ShimPrompt, copied verbatim (the shared kernel)."""

    def __init__(self, batched_prompt):
        super().__init__()
        self.bp = batched_prompt            # (B, top_k*length, C)

    def forward(self, x_embed, prompt_mask=None, cls_features=None):
        out = dict()
        out["prompt_idx"] = None
        out["total_prompt_len"] = self.bp.shape[1]
        out["prompted_embedding"] = torch.cat([self.bp, x_embed], dim=1)
        return out


def forward_with_prompt(vit, task_id, inp, q, prompt_tokens):
    """Forward the client vit with a forced prompt token block.
    prompt_tokens: (top_k*length, C) tensor on the vit device.

    Probe-H _forward_with_prompt, same semantics (task_id passed through
    so the caller fixes the deployment eval semantics)."""
    orig = vit.prompt
    bp = prompt_tokens.unsqueeze(0).expand(inp.shape[0], -1, -1)
    vit.prompt = ShimPrompt(bp)
    try:
        out = vit(inp, task_id=task_id, cls_features=q, train=True)
    finally:
        vit.prompt = orig
    return out["feat"].float()


def row_ce(lg, y_local):
    """Per-row cross entropy from masked logits (Probe-H _record):
    CE_i = logsumexp(lg_i) - lg_i[y_local_i]. lg is fp32 (B, |mask|)."""
    return (torch.logsumexp(lg, dim=1)
            - lg.gather(1, y_local.unsqueeze(1)).squeeze(1))


def alpha_blend(prev_row, cand_row, alpha):
    """P_deploy = P_prev + alpha * (P_cand - P_prev), realized in the
    Probe-H operation order (1 - a) * prev + a * cand (bitwise the
    validated kernel; mathematically the frozen closed form)."""
    return (1.0 - float(alpha)) * prev_row + float(alpha) * cand_row


def select_alpha(pooled_ce, grid):
    """Frozen tie-breaking: argmin over the ASCENDING grid; ties -> the
    smallest alpha (np.argmin returns the FIRST extremum). Probe-H
    selector semantics, no isclose tolerance."""
    pooled_ce = np.asarray(pooled_ce, dtype=np.float64)
    return float(grid[int(np.argmin(pooled_ce))])


# ----------------------------------------------------------------------
# row-selection primitives (shared by production split and Test-8
# probe-compat replay; the PROTOCOLS differ, the primitives are common)
# ----------------------------------------------------------------------

def stratified_pick(labels, cap, rng):
    """Class-stratified subsample: per class, round(cap * n_c / N) rows
    (>=1 if the class is non-empty); returns positions into `labels`.
    Probe-H _stratified_pick, verbatim."""
    labels = np.asarray(labels)
    n = len(labels)
    if n == 0:
        return np.zeros((0,), dtype=np.int64)
    if n <= cap:
        return np.arange(n, dtype=np.int64)
    pick = []
    for c in np.unique(labels):
        idx_c = np.nonzero(labels == c)[0]
        take = max(1, int(round(cap * len(idx_c) / n)))
        take = min(take, len(idx_c))
        if take < len(idx_c):
            idx_c = idx_c[rng.permutation(len(idx_c))[:take]]
        pick.extend(int(i) for i in idx_c)
    return np.array(sorted(pick), dtype=np.int64)


def stratified_split(labels, val_frac, rng):
    """Class-stratified split flags (0=train, 1=val). A class with >=2
    rows contributes >=1 val row; a singleton class stays in train.
    Probe-H _stratified_split, verbatim."""
    labels = np.asarray(labels)
    split = np.zeros(len(labels), dtype=np.int8)
    for c in np.unique(labels):
        idx_c = np.nonzero(labels == c)[0]
        n_val = max(1, int(round(val_frac * len(idx_c)))) \
            if len(idx_c) >= 2 else 0
        n_val = min(n_val, len(idx_c) - 1) if len(idx_c) >= 2 else 0
        if n_val > 0:
            sel = rng.permutation(len(idx_c))[:n_val]
            split[idx_c[sel]] = 1
    return split
