"""Probe-H: temporal prompt step calibration (RUNCOMMANDS Sec.32, offline
only, pre-registered 2026-11-03).

Zero-training offline probe testing the ONE deployment-relevant question
raised by the Sec.31.9 post-verdict review: Probe-G's lambda scan showed
that interior blends of the SAME slot-6 prompt repeatedly outperform the
learned endpoint on BOTH old and current tasks (temporal prompt update
overshoot) -- but that grid was read with a TEST oracle (it saw historical
test rows, so it cannot be deployed). Probe-H asks (GPT frozen):

    Can the harmful endpoint be avoided using current-task information
    alone?

Frozen ladder:

  H0  premise gate (fail-closed, Sec.31 G0 lineage): deployment path
      (top_k=1, batchwise_prompt, no task mask), per-state batch-majority
      mode == monopoly slot with <=1% non-slot batches on BOTH the test
      and the selection population, monopoly slot constant across
      states, prompt VALUES identical across clients (global tensor),
      heads identical across states.
  H1  fine-grid lambda scan + deployment-legal selectors: at state s+1,
      P_lam = (1-lam) * P_{6,s} + lam * P_{6,s+1} for lam in
      {0, 0.05, ..., 1} (21 points), forwarded on every monopoly-slot
      batch over old-task AND current-task test rows. SELECTION SIGNAL =
      current task inner-train rows (data_split_indices[t]['train'],
      class-stratified seeded subsample capped at
      --select_rows_per_client per client, then a class-stratified
      20% val / 80% train split), forwarded under the same 21 arms with
      the identical deterministic eval protocol. Selectors (zero
      old-task information, ties -> smallest lam):
        lam_hat_val   = argmin pooled current-task VAL CE    (primary)
        lam_hat_train = argmin pooled current-task TRAIN CE  (secondary)
      Oracle (illegal, reported for contrast only):
        lam_star      = argmax current-task TEST accuracy.
  MAIN GATE (transition 2->3, primary selector lam_hat_val, GPT frozen):
        lam_hat < 1  AND  Acc_cur(lam_hat) >= Acc_cur(lam=1)  AND
        Acc_old(lam_hat) >  Acc_old(lam=1)
      with the old gain carried by a paired batch-bootstrap 95% CI
      whose lower bound must exceed 0.
        -> OVERSHOOT_EXPLOITABLE_CURRENT_ONLY (CTRP Sec.33 candidacy,
           still conditional on INR mechanism replication + external
           review; this probe registers NO training arm)
        -> YELLOW_REVIEW (direction holds, CI touches zero)
        -> OVERSHOOT_NOT_EXPLOITABLE otherwise (annex distinguishes
           selector-blind vs no-interior-gain via the oracle readout)

Route identity is preserved by construction: routing reads prompt_key
only, so replacing the slot-6 VALUE cannot change the route (asserted
per batch anyway; arm coverage = slot-6 majority batches).

Deviations from native evaluate() (pre-registered, Sec.31 lineage):
fixed batch order at batch_size 8, head.eval() (Chead Dropout OFF);
all gates act on cross-arm differences so the bias cancels. The
probe-vs-native gap was already LOCATED by Probe-G on this very run
(75/75 keys, max unexplained ~0.0pp) and is NOT re-measured here (GPU
budget; identical forward protocol).

fail-closed (Sec.32.5): use_prompt_mask=True / top_k!=1 /
batchwise_prompt=False -> refuse; monopoly premise violations on either
population -> refuse; prompt values differing across clients -> refuse;
canon join gaps or non-monotonic joins -> refuse; zero selection rows
at states >= 1 -> refuse; cache fingerprints (probe version, run,
checkpoint, batch, monopoly slot, lambda grid, selection cap, val
fraction) must match.

Usage (from repo root):

  python diagnostics/probe_prompt_step_calibration.py \
      --run_dir output/cifar100/fedta/cifar_a5v2_M2_feat_full/seed_42 \
      --data_path ./local_datasets --device cuda \
      --cache_dir diagnostics_output/stepcalib_cifar \
      --out_csv diagnostics_output/stepcalib_cifar.csv \
      --out_rows_csv diagnostics_output/stepcalib_cifar_rows.csv \
      --out_json diagnostics_output/stepcalib_cifar.summary.json
"""

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diag_utils import bootstrap_server, load_args_from_run_dir  # noqa: E402

PROBE_VERSION = "stepcalib-v1"
TAG = "[stepcalib-probe]"
EPS = 1e-9

DEFAULT_LAMBDAS = ",".join(
    [str(round(i * 0.05, 2)) for i in range(21)])   # 0,0.05,...,1


def parse_args():
    p = argparse.ArgumentParser("Probe-H temporal prompt step calibration")
    p.add_argument("--run_dir", required=True,
                   help="run directory (.../{data_name}/{method}/"
                        "{run_name}/seed_{seed})")
    p.add_argument("--checkpoints",
                   default="task_00_end.pth,task_01_end.pth,"
                           "task_02_end.pth,task_03_end.pth,"
                           "task_04_end.pth")
    p.add_argument("--data_path", default=None)
    p.add_argument("--device", default="cuda",
                   help="cuda required (Tail_Anchor hardcodes .to('cuda'))")
    p.add_argument("--cache_dir", required=True,
                   help="independent cache dir (Sec.29/30/31/32 "
                        "fail-closed lesson: never share caches across "
                        "probes)")
    p.add_argument("--out_csv", required=True)
    p.add_argument("--out_rows_csv", required=True)
    p.add_argument("--out_json", required=True)
    p.add_argument("--batch_size", default=8, type=int,
                   help="fixed evaluation batch (native evaluate uses 8; "
                        "fixed order is the pre-registered pairing)")
    p.add_argument("--monopoly_slot", default=6, type=int,
                   help="the monopoly slot this probe dissects (Sec.30.8/"
                        "31.8: slot 6 held >99.9% on all 5 states)")
    p.add_argument("--max_nonslot_batch_frac", default=0.01, type=float,
                   help="H0 premise: max fraction of batches whose majority "
                        "slot is not the monopoly slot (applied to BOTH "
                        "the test and the selection population)")
    p.add_argument("--lambdas", default=DEFAULT_LAMBDAS,
                   help="lambda grid (0=swap arm, 1=natural arm); must be "
                        "ascending unique")
    p.add_argument("--gate_transition", default=3, type=int,
                   help="1-based transition index of the main gate "
                        "(3 = states 2 -> 3)")
    p.add_argument("--select_rows_per_client", default=512, type=int,
                   help="per-client cap on the class-stratified "
                        "current-task train subsample used as the "
                        "selection signal (GPU budget control; Sec.32.3)")
    p.add_argument("--val_frac", default=0.2, type=float,
                   help="fraction of the selection subsample assigned to "
                        "val (class-stratified; the rest is the train "
                        "signal pool)")
    p.add_argument("--n_boot", default=1000, type=int,
                   help="paired batch bootstrap replicates (B=1000; "
                        "resampling unit = BATCH, Sec.30.7 statistics)")
    p.add_argument("--boot_seed", default=20261103, type=int)
    p.add_argument("--probe_g_csv",
                   default="diagnostics_output/contentdrift_cifar.csv",
                   help="Probe-G aggregate CSV for the pre-registered "
                        "bit-identity continuity check (Sec.32.1-7). If "
                        "the file is absent the check degrades to an "
                        "informative note; if present and mismatched the "
                        "probe REFUSES (engineering bug)")
    p.add_argument("--overwrite_cache", action="store_true")
    return p.parse_args()


# ----------------------------------------------------------------------
# shared numeric helpers (probe_prompt_content_drift.py lineage)
# ----------------------------------------------------------------------

def _hash_module(module):
    h = hashlib.sha256()
    sd = module.state_dict()
    for k in sorted(sd):
        v = sd[k].detach().cpu().contiguous()
        h.update(k.encode())
        h.update(v.numpy().tobytes())
    return h.hexdigest()[:16]


def _hash_array(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]


def _seed(*parts):
    """Stable derived seed from arbitrary ints (house stable_seed style)."""
    h = hashlib.sha256(",".join(str(int(p)) for p in parts).encode())
    return int(h.hexdigest()[:8], 16) % (2 ** 31 - 1)


class _ShimPrompt(torch.nn.Module):
    """Replaces vit.prompt to inject a caller-supplied prompt token block.

    Replicates Global_Prompt.forward's batched_prompt branch exactly
    (official ``torch.cat([batched_prompt, x_embed], dim=1)`` +
    total_prompt_len). Must be an nn.Module: vit.prompt is a registered
    child module, so assigning a plain object would be rejected by
    Module.__setattr__."""

    def __init__(self, batched_prompt):
        super().__init__()
        self.bp = batched_prompt            # (B, top_k*length, C)

    def forward(self, x_embed, prompt_mask=None, cls_features=None):
        out = dict()
        out["prompt_idx"] = None
        out["total_prompt_len"] = self.bp.shape[1]
        out["prompted_embedding"] = torch.cat([self.bp, x_embed], dim=1)
        return out


def _forward_with_prompt(client, inp, q, prompt_tokens):
    """Forward the client vit with a forced prompt token block.
    prompt_tokens: (top_k*length, C) tensor on the vit device."""
    vit = client.vit
    orig = vit.prompt
    bp = prompt_tokens.unsqueeze(0).expand(inp.shape[0], -1, -1)
    vit.prompt = _ShimPrompt(bp)
    try:
        out = vit(inp, task_id=client.task_id, cls_features=q, train=True)
    finally:
        vit.prompt = orig
    return out["feat"].float()


def _row_vecs(flat, ptr, row_idx):
    """List of per-row masked-logit vectors (ptr has n_rows+1 entries)."""
    return [flat[ptr[ri]:ptr[ri + 1]] for ri in row_idx]


def _arm_vecs(arm, key):
    """List of per-row logit vectors for an arm, from its flat+ptr arrays
    (works identically for freshly extracted and cached states)."""
    cache_key = f"__vecs_{key}"
    if cache_key not in arm:
        flat = arm.get(f"{key}_flat")
        ptr = arm.get(f"{key}_ptr")
        if flat is None:
            return []
        arm[cache_key] = [flat[ptr[i]:ptr[i + 1]]
                          for i in range(len(ptr) - 1)]
    return arm[cache_key]


def _list_correct(vecs, ys):
    out = np.empty(len(ys), dtype=bool)
    for i in range(len(ys)):
        out[i] = bool(np.argmax(vecs[i]) == int(ys[i]))
    return out


def _acc(correct):
    correct = np.asarray(correct, dtype=bool)
    return 100.0 * float(correct.mean()) if len(correct) else float("nan")


def _state_correct_ce(state):
    rows = state["rows"]
    vecs = _row_vecs(rows["logits_flat"], rows["logits_ptr"],
                     range(len(rows["client"])))
    correct = np.empty(len(vecs), dtype=bool)
    ce = np.empty(len(vecs), dtype=np.float64)
    for i, v in enumerate(vecs):
        mx = v.max()
        e = np.exp(v - mx)
        correct[i] = bool(np.argmax(v) == int(rows["y_local"][i]))
        ce[i] = math.log(e.sum()) + mx - v[int(rows["y_local"][i])]
    return correct, ce


def _batch_groups(batch_ids):
    """Row indices grouped by batch id (stable)."""
    batch_ids = np.asarray(batch_ids)
    order = np.argsort(batch_ids, kind="stable")
    sorted_ids = batch_ids[order]
    bounds = np.nonzero(np.r_[True, sorted_ids[1:] != sorted_ids[:-1],
                               True])[0]
    return [order[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]


def _paired_boot_ci(diffs, batch_ids, n_boot, seed):
    """Paired BATCH bootstrap 95% CI (in pp) for 100*mean(diffs), where
    diffs are per-ROW paired differences and the resampling unit is the
    BATCH (Sec.30.7 statistics; rows within a batch share exposure).
    Returns (lo, hi, n_valid_reps) or None."""
    diffs = np.asarray(diffs, dtype=np.float64)
    groups = _batch_groups(batch_ids)
    if not groups or len(groups) < 2:
        return None
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        rows = np.concatenate([groups[i] for i in pick])
        stats.append(100.0 * float(diffs[rows].mean()))
    if len(stats) < max(10, n_boot // 10):
        return None
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return (round(float(lo), 4), round(float(hi), 4), len(stats))


def _stratified_pick(labels, cap, rng):
    """Class-stratified subsample: per class, round(cap * n_c / N) rows
    (>=1 if the class is non-empty); returns positions into `labels`."""
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


def _stratified_split(labels, val_frac, rng):
    """Class-stratified split flags (0=train, 1=val). A class with >=2
    rows contributes >=1 val row; a singleton class stays in train."""
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


# ----------------------------------------------------------------------
# per-state extraction
# ----------------------------------------------------------------------

def _fingerprint(run_args, run_dir, ckpt_name, args_cli, lam_grid):
    return {
        "probe_version": PROBE_VERSION,
        "run_dir": str(Path(run_dir).resolve()),
        "data_name": run_args.data_name,
        "seed": run_args.seed,
        "checkpoint": ckpt_name,
        "batch_size": args_cli.batch_size,
        "top_k": run_args.top_k,
        "use_prompt_mask": bool(run_args.use_prompt_mask),
        "batchwise_prompt": bool(run_args.batchwise_prompt),
        "pool_size": run_args.size,
        "monopoly_slot": args_cli.monopoly_slot,
        "lambdas": [float(x) for x in lam_grid],
        "select_rows_per_client": args_cli.select_rows_per_client,
        "val_frac": args_cli.val_frac,
    }


def extract_state(args_cli, run_dir, ckpt_name, prev_state, lam_grid,
                  interior_lams, t_index):
    """Extract one deployment state into row/batch/arm/selection arrays.

    Test arms (state t >= 1, on every monopoly-slot majority batch, over
    ALL tasks 0..t): swap (lam=0, slot-6 value from the PREVIOUS state)
    and interior-lambda mixes (1-lam)*P_prev + lam*P_now; natural (lam=1)
    is the row pass itself. Selection signal (state t >= 1, CURRENT task
    only): class-stratified seeded subsample of inner-train rows, split
    80/20 train/val, forwarded under the same arms (Sec.32.3). Caches to
    {cache_dir}/state_{stem}.npz (+json meta) with fail-closed
    fingerprint checks."""
    cache_dir = Path(args_cli.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(ckpt_name).stem
    npz_path = cache_dir / f"state_{stem}.npz"
    meta_path = cache_dir / f"state_{stem}.json"

    run_args_probe = load_args_from_run_dir(run_dir)
    if bool(run_args_probe.use_prompt_mask):
        raise RuntimeError(
            "use_prompt_mask=True: deployment routing is the task-mask "
            "path and Probe-H semantics (monopoly-slot value "
            "interpolation) do not apply. Refusing (Sec.32 fail-closed).")
    if int(run_args_probe.top_k) != 1:
        raise RuntimeError(
            f"top_k={run_args_probe.top_k}: probe pre-registered for "
            "top_k=1 deployments only. Refusing (Sec.32 fail-closed).")
    if not bool(run_args_probe.batchwise_prompt):
        raise RuntimeError(
            "batchwise_prompt=False: routes are per-sample and the "
            "monopoly-batch coverage semantics do not apply. Refusing "
            "(Sec.32 fail-closed).")
    if args_cli.device == "cpu" or not torch.cuda.is_available():
        raise RuntimeError(
            "this probe requires --device cuda (Tail_Anchor.forward "
            "hardcodes .to('cuda'); official model semantics)")

    fp = _fingerprint(run_args_probe, run_dir, ckpt_name, args_cli,
                      lam_grid)

    if (npz_path.is_file() and meta_path.is_file()
            and not args_cli.overwrite_cache):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("fingerprint") != fp:
            raise RuntimeError(
                f"cache fingerprint mismatch for {npz_path}: refusing "
                "(delete the cache dir or pass --overwrite_cache)")
        print(f"{TAG} state {stem}: loaded from cache")
        return _load_state(npz_path, meta)

    t0 = time.time()
    run_args, server = bootstrap_server(
        run_dir, checkpoint=ckpt_name,
        data_path=args_cli.data_path, device=args_cli.device)

    # deterministic flags MUST come after bootstrap_server (same protocol
    # as extract_features.py / probe_routing_flip.py / probe-G)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device(args_cli.device)

    server.model.to(device)
    server.origin_model.to(device)
    server.origin_model.eval()

    slot_id = int(args_cli.monopoly_slot)
    n_lams = len(lam_grid)
    # grid strings -> column positions (the selection-CE matrix is indexed
    # by GRID POSITION; floats are only for interpolation math)
    lam_pos_by_float = {float(l): i for i, l in enumerate(lam_grid)}
    if len(lam_pos_by_float) != n_lams or 0.0 not in lam_pos_by_float \
            or 1.0 not in lam_pos_by_float:
        raise RuntimeError(f"lambda grid invalid: {lam_grid}")

    r = {k: [] for k in
         ("client", "task", "rel_idx", "bseq", "y", "y_local",
          "slot_major", "slot_pref", "z", "logits_flat", "logits_ptr")}
    batch = {k: [] for k in
             ("client", "task", "bseq", "slot_major", "n")}
    arm = {"rows": []}
    for lam_key in ["swap"] + [f"lam_{l}" for l in interior_lams]:
        arm[f"lg_{lam_key}"] = []
    sel = {"client": [], "rel_idx": [], "y_local": [], "split": [],
           "batch_id": [], "ce": [], "correct": []}
    class_masks = {}
    head_hashes = {}
    key_hashes = {}
    pv_cpu = {}
    n_rows = 0
    flat_len = 0
    n_nonslot_batches = 0
    sel_n_dropped = 0
    sel_n_total = 0

    for client in server.clients:
        if client.task_id < 0:
            continue
        if int(client.task_id) != t_index:
            raise RuntimeError(
                f"client {client.id}: task_id {client.task_id} != state "
                f"index {t_index} at a boundary checkpoint -- checkpoint "
                "semantics mismatch (Sec.32 fail-closed)")
        if client.prompts is None:
            raise RuntimeError(
                f"client {client.id}: prompts is None at a boundary "
                "checkpoint -- unexpected for post-broadcast states")
        cid = int(client.id)
        # the client's CURRENT prompts (official extraction semantics)
        client.vit.load_prompts(client.prompts)
        client.vit.eval()   # drop rates are 0 -> bit-identical to official
        pv_now = client.vit.prompt.prompt.detach().clone()
        pv_cpu[cid] = pv_now.cpu().numpy().astype(np.float32)
        key_hashes[cid] = _hash_array(
            client.vit.prompt.prompt_key.detach().cpu().numpy())
        client.model.to(device)

        pv_prev = None
        if prev_state is not None:
            if cid not in prev_state["pv"]:
                raise RuntimeError(
                    f"client {cid} missing from previous state's prompt "
                    "pool -- broadcast bookkeeping broken")
            pv_prev = torch.tensor(prev_state["pv"][cid],
                                   dtype=torch.float32, device=device)

        for t in range(client.task_id + 1):
            if t not in client.data_split_indices:
                continue
            indices = list(client.data_split_indices[t]["test"])
            if not indices:
                continue
            if client.heads[t] is None:
                raise RuntimeError(
                    f"client {cid} task {t}: stored head is None at a "
                    "boundary checkpoint")
            mask_j = sorted(int(x) for x in client.class_mask[t])
            class_masks[f"{cid},{t}"] = mask_j
            head_hashes[f"{cid},{t}"] = _hash_module(client.heads[t])
            mpos = torch.tensor(mask_j, dtype=torch.long, device=device)
            client.model.load_head(client.heads[t])
            client.model.eval()   # Chead Dropout OFF (pre-registered)

            subset = Subset(client.train_data[t], indices)
            loader = DataLoader(subset, batch_size=args_cli.batch_size,
                                shuffle=False,
                                num_workers=run_args.num_workers)
            bseq = -1
            n_done = 0
            for inp, tgt in loader:
                bseq += 1
                inp = inp.to(device, non_blocking=True)
                tgt = tgt.to(device, non_blocking=True)
                with torch.no_grad():
                    out_ref = server.origin_model(inp)
                    q = out_ref["pre_logits"]
                    out = client.vit(inp, task_id=client.task_id,
                                     cls_features=q, train=True)
                    z = out["feat"].float()
                    idx_nat = out["prompt_idx"]
                bsz = int(z.shape[0])
                if tuple(idx_nat.shape) != (bsz, 1):
                    raise RuntimeError(
                        f"unexpected prompt_idx shape {tuple(idx_nat.shape)}"
                        " -- top_k must be 1")
                slot_major = int(idx_nat[0, 0])
                if not bool((idx_nat == idx_nat[0]).all()):
                    raise RuntimeError(
                        "batchwise majority not constant within a batch "
                        "-- deployment semantics mismatch")

                y_local = torch.searchsorted(mpos, tgt)
                with torch.no_grad():
                    logits = client.model(z, None)[0]
                    lg = logits.index_select(1, mpos).float()

                batch["client"].append(cid)
                batch["task"].append(t)
                batch["bseq"].append(bseq)
                batch["slot_major"].append(slot_major)
                batch["n"].append(bsz)

                row_base = n_rows
                in_pool = (slot_major == slot_id)
                if not in_pool:
                    n_nonslot_batches += 1
                for bi in range(bsz):
                    r["client"].append(cid)
                    r["task"].append(t)
                    r["rel_idx"].append(int(indices[n_done + bi]))
                    r["bseq"].append(bseq)
                    r["y"].append(int(tgt[bi]))
                    r["y_local"].append(int(y_local[bi]))
                    r["slot_major"].append(slot_major)
                    r["slot_pref"].append(-1)   # unused in Probe-H gates
                    r["z"].append(z[bi].cpu().numpy().astype(np.float32))
                    r["logits_flat"].append(
                        lg[bi].cpu().numpy().astype(np.float32))
                    r["logits_ptr"].append(flat_len)
                    flat_len += int(lg.shape[1])
                n_rows += bsz
                n_done += bsz

                # ---- test arms: monopoly-slot batches only (H1) --------
                if prev_state is None or not in_pool:
                    continue
                tok_now = pv_now[slot_id]
                tok_prev = pv_prev[slot_id]
                # swap (lam=0): old slot-6 content, same route
                z_sw = _forward_with_prompt(client, inp, q, tok_prev)
                with torch.no_grad():
                    lg_sw = client.model(z_sw, None)[0].index_select(
                        1, mpos).float()
                arm["rows"].extend(range(row_base, row_base + bsz))
                for bi in range(bsz):
                    arm["lg_swap"].append(lg_sw[bi].cpu().numpy())
                # interior lambda mixes (H1 fine grid)
                for lam in interior_lams:
                    lam_f = float(lam)
                    tok_l = (1.0 - lam_f) * tok_prev + lam_f * tok_now
                    z_l = _forward_with_prompt(client, inp, q, tok_l)
                    with torch.no_grad():
                        lg_l = client.model(z_l, None)[0].index_select(
                            1, mpos).float()
                    for bi in range(bsz):
                        arm[f"lg_lam_{lam}"].append(
                            lg_l[bi].cpu().numpy())

        # ---- selection signal: CURRENT task inner-train rows -----------
        if prev_state is None:
            continue
        t_cur = int(client.task_id)
        mask_j = sorted(int(x) for x in client.class_mask[t_cur])
        mpos = torch.tensor(mask_j, dtype=torch.long, device=device)
        client.model.load_head(client.heads[t_cur])
        client.model.eval()
        train_idx = list(client.data_split_indices[t_cur]["train"])
        if not train_idx:
            raise RuntimeError(
                f"client {cid}: current task {t_cur} has no inner-train "
                "rows -- selection signal undefined (Sec.32 fail-closed)")
        # pass 1: labels only (stratification; no forward)
        labels = []
        lab_loader = DataLoader(
            Subset(client.train_data[t_cur], train_idx),
            batch_size=args_cli.batch_size, shuffle=False,
            num_workers=run_args.num_workers)
        for _, tgt_l in lab_loader:
            labels.extend(int(v) for v in tgt_l.numpy())
        if len(labels) != len(train_idx):
            raise RuntimeError(
                f"client {cid}: label pass length {len(labels)} != "
                f"{len(train_idx)} -- dataloader semantics mismatch")
        rng_pick = np.random.default_rng(
            _seed(run_args.seed, 51, t_index, cid))
        keep_pos = _stratified_pick(labels, args_cli.select_rows_per_client,
                                    rng_pick)
        kept_labels = [labels[i] for i in keep_pos]
        rng_split = np.random.default_rng(
            _seed(run_args.seed, 52, t_index, cid))
        split_flags = _stratified_split(kept_labels, args_cli.val_frac,
                                        rng_split)
        sel_idx = [train_idx[i] for i in keep_pos]   # ascending
        # pass 2: natural + all 20 arms on the selection subsample
        sel_loader = DataLoader(
            Subset(client.train_data[t_cur], sel_idx),
            batch_size=args_cli.batch_size, shuffle=False,
            num_workers=run_args.num_workers)
        tok_now = pv_now[slot_id]
        tok_prev = pv_prev[slot_id]
        s_bseq = -1
        s_done = 0
        for inp_s, tgt_s in sel_loader:
            s_bseq += 1
            bsz = int(inp_s.shape[0])
            inp_s = inp_s.to(device, non_blocking=True)
            tgt_s = tgt_s.to(device, non_blocking=True)
            y_loc_s = torch.searchsorted(mpos, tgt_s)
            if int(y_loc_s.max()) >= len(mask_j):
                raise RuntimeError(
                    f"client {cid}: selection-row label outside the "
                    "current task class mask -- split semantics mismatch")
            sel_n_total += bsz
            with torch.no_grad():
                q_s = server.origin_model(inp_s)["pre_logits"]
                out_s = client.vit(inp_s, task_id=client.task_id,
                                   cls_features=q_s, train=True)
                if tuple(out_s["prompt_idx"].shape) != (bsz, 1):
                    raise RuntimeError(
                        f"unexpected selection prompt_idx shape "
                        f"{tuple(out_s['prompt_idx'].shape)} -- top_k "
                        "must be 1")
                slot_s = int(out_s["prompt_idx"][0, 0])
                if not bool((out_s["prompt_idx"]
                             == out_s["prompt_idx"][0]).all()):
                    raise RuntimeError(
                        "batchwise majority not constant within a "
                        "selection batch -- deployment semantics mismatch")
            if slot_s != slot_id:
                sel_n_dropped += bsz
                s_done += bsz
                continue
            # column order = lam_grid order: [0, 0.05, ..., 1]
            row_ce = np.zeros((bsz, n_lams), dtype=np.float32)
            row_corr = np.zeros((bsz, n_lams), dtype=np.int8)

            def _record(lam_pos, lg_mat, yl):
                ce = (torch.logsumexp(lg_mat, dim=1)
                      - lg_mat.gather(1, yl.unsqueeze(1)).squeeze(1))
                row_ce[:, lam_pos] = ce.detach().cpu().numpy()
                row_corr[:, lam_pos] = (
                    lg_mat.argmax(dim=1) == yl).int().cpu().numpy()

            with torch.no_grad():
                # natural arm (lam=1)
                lg_nat = client.model(out_s["feat"].float(), None)[0] \
                    .index_select(1, mpos).float()
                pos_nat = lam_pos_by_float[1.0]
                _record(pos_nat, lg_nat, y_loc_s)
                # swap arm (lam=0)
                z_sw = _forward_with_prompt(client, inp_s, q_s, tok_prev)
                with torch.no_grad():
                    lg_sw = client.model(z_sw, None)[0].index_select(
                        1, mpos).float()
                pos_sw = lam_pos_by_float[0.0]
                _record(pos_sw, lg_sw, y_loc_s)
                # interior arms
                for lam in interior_lams:
                    lam_f = float(lam)
                    tok_l = (1.0 - lam_f) * tok_prev + lam_f * tok_now
                    z_l = _forward_with_prompt(client, inp_s, q_s, tok_l)
                    with torch.no_grad():
                        lg_l = client.model(z_l, None)[0].index_select(
                            1, mpos).float()
                    _record(lam_pos_by_float[lam_f], lg_l, y_loc_s)
            for bi in range(bsz):
                sel["client"].append(cid)
                sel["rel_idx"].append(int(sel_idx[s_done + bi]))
                sel["y_local"].append(int(y_loc_s[bi]))
                sel["split"].append(int(split_flags[s_done + bi]))
                sel["batch_id"].append(cid * 1_000_000 + s_bseq)
                sel["ce"].append(row_ce[bi])
                sel["correct"].append(row_corr[bi])
            s_done += bsz

    # ---- H0 premise checks (fail-closed) -------------------------------
    slots = np.array(batch["slot_major"], dtype=np.int64)
    if len(slots) == 0:
        raise RuntimeError(f"state {stem}: no evaluation batches extracted")
    mode_slot = int(np.bincount(slots).argmax())
    nonslot_frac = float(np.mean(slots != slot_id))
    if mode_slot != slot_id or nonslot_frac > args_cli.max_nonslot_batch_frac:
        raise RuntimeError(
            f"state {stem}: monopoly premise violated (batch-majority "
            f"mode slot {mode_slot} != {slot_id}, non-slot batch frac "
            f"{nonslot_frac:.4f} > {args_cli.max_nonslot_batch_frac}) -- "
            "Probe-H semantics are only defined for this deployment "
            "path. Refusing (Sec.32 fail-closed).")
    pv_hashes = {str(c): _hash_array(v) for c, v in pv_cpu.items()}
    if len(set(pv_hashes.values())) != 1:
        raise RuntimeError(
            f"state {stem}: prompt VALUES differ across clients -- "
            "P_{6,t} is not a single global tensor (broadcast premise "
            "broken). Refusing (Sec.32 fail-closed).")
    # selection-side monopoly coverage (states >= 1 only)
    sel_drop_frac = (sel_n_dropped / sel_n_total) if sel_n_total else 0.0
    if prev_state is not None:
        if sel_n_total == 0:
            raise RuntimeError(
                f"state {stem}: zero selection rows extracted -- "
                "calibration signal undefined (Sec.32 fail-closed)")
        if sel_drop_frac > args_cli.max_nonslot_batch_frac:
            raise RuntimeError(
                f"state {stem}: selection-side non-monopoly rows "
                f"{sel_n_dropped}/{sel_n_total} ({sel_drop_frac:.4f}) "
                f"> {args_cli.max_nonslot_batch_frac} -- selection "
                "population violates the monopoly premise. Refusing "
                "(Sec.32 fail-closed).")

    # ---- canonical row order (Sec.29.1c lesson) ------------------------
    client_a = np.array(r["client"], dtype=np.int64)
    task_a = np.array(r["task"], dtype=np.int64)
    rel_a = np.array(r["rel_idx"], dtype=np.int64)
    n = len(client_a)
    order = np.lexsort((rel_a, task_a, client_a))
    inv = np.empty(n, dtype=np.int64)
    inv[order] = np.arange(n)
    for k in ("client", "task", "rel_idx", "bseq", "y", "y_local",
              "slot_major", "slot_pref", "z"):
        r[k] = [r[k][i] for i in order]
    # rebuild logits flat/ptr in canonical order
    old_ptr = np.append(np.array(r["logits_ptr"], dtype=np.int64),
                        int(flat_len))
    old_flat = (np.concatenate(r["logits_flat"]) if r["logits_flat"]
                else np.zeros((0,), np.float32))
    new_flat, new_ptr = [], []
    off = 0
    for oi in order:
        v = old_flat[old_ptr[oi]:old_ptr[oi + 1]]
        new_flat.append(v)
        new_ptr.append(off)
        off += len(v)
    r["logits_flat"] = new_flat
    r["logits_ptr"] = new_ptr
    if arm["rows"]:
        mapped = inv[np.array(arm["rows"], dtype=np.int64)]
        srt = np.argsort(mapped)
        arm["rows"] = [int(v) for v in mapped[srt]]
        for k in [k for k in list(arm.keys())
                  if k == "lg_swap" or k.startswith("lg_lam")]:
            if isinstance(arm[k], list):
                arm[k] = [arm[k][i] for i in srt]

    # ---- pack ----------------------------------------------------------
    for key in [k for k in list(arm.keys())
                if k.startswith("lg_")]:
        vecs = arm.pop(key)
        if vecs:
            arm[key + "_flat"] = np.concatenate(vecs)
            widths = np.array([len(v) for v in vecs], dtype=np.int64)
            arm[key + "_ptr"] = np.concatenate(
                ([0], np.cumsum(widths))).astype(np.int64)
        else:
            arm[key + "_flat"] = np.zeros((0,), np.float32)
            arm[key + "_ptr"] = np.zeros((1,), np.int64)

    def npa(key, dtype=np.float32):
        return (np.array(r[key], dtype=dtype) if r[key]
                else np.zeros((0,), dtype=dtype))

    z_arr = (np.stack(r["z"]) if r["z"]
             else np.zeros((0, 768), np.float32))
    ptr = np.append(np.array(r["logits_ptr"], dtype=np.int64),
                    sum(len(v) for v in r["logits_flat"]))
    flat = (np.concatenate(r["logits_flat"]) if r["logits_flat"]
            else np.zeros((0,), np.float32))

    sel_packed = {
        "task": t_index,
        "client": np.array(sel["client"], dtype=np.int64),
        "rel_idx": np.array(sel["rel_idx"], dtype=np.int64),
        "y_local": np.array(sel["y_local"], dtype=np.int64),
        "split": np.array(sel["split"], dtype=np.int8),
        "batch_id": np.array(sel["batch_id"], dtype=np.int64),
        "ce": (np.stack(sel["ce"]) if sel["ce"]
               else np.zeros((0, n_lams), np.float32)),
        "correct": (np.stack(sel["correct"]) if sel["correct"]
                    else np.zeros((0, n_lams), np.int8)),
        "n_rows": len(sel["client"]),
        "n_train": int(np.sum(np.array(sel["split"],
                                       dtype=np.int8) == 0))
        if sel["split"] else 0,
        "n_val": int(np.sum(np.array(sel["split"],
                                     dtype=np.int8) == 1))
        if sel["split"] else 0,
        "n_dropped_rows": int(sel_n_dropped),
        "n_total_rows": int(sel_n_total),
    }

    state = {
        "t_index": t_index,
        "ckpt": ckpt_name,
        "rows": {
            "client": npa("client", np.int64),
            "task": npa("task", np.int64),
            "rel_idx": npa("rel_idx", np.int64),
            "bseq": npa("bseq", np.int64),
            "y": npa("y", np.int64),
            "y_local": npa("y_local", np.int64),
            "slot_major": npa("slot_major", np.int64),
            "slot_pref": npa("slot_pref", np.int64),
            "z": z_arr, "logits_flat": flat, "logits_ptr": ptr,
        },
        "batch": {k: np.array(v, dtype=(
            np.int64 if k in ("client", "task", "bseq", "slot_major", "n")
            else np.float32)) for k, v in batch.items()},
        "arms": arm,
        "sel": sel_packed,
        "masks": class_masks,
        "head_hashes": head_hashes,
        "key_hashes": key_hashes,
        "pv": pv_cpu,
        "pv_hashes": pv_hashes,
    }

    meta = {
        "fingerprint": fp,
        "checkpoint": ckpt_name,
        "t_index": t_index,
        "n_rows": int(n_rows),
        "n_batches": int(len(batch["client"])),
        "batch_mode_slot": mode_slot,
        "nonslot_batch_frac": round(nonslot_frac, 5),
        "masks": class_masks,
        "head_hashes": head_hashes,
        "key_hashes": key_hashes,
        "prompt_value_hashes": pv_hashes,
        "embedding_key": run_args_probe.embedding_key,
        "length": int(run_args_probe.length),
        "pool_size": int(run_args_probe.size),
        "sel": {"task": t_index,
                "n_rows": sel_packed["n_rows"],
                "n_train": sel_packed["n_train"],
                "n_val": sel_packed["n_val"],
                "n_dropped_rows": sel_packed["n_dropped_rows"],
                "n_total_rows": sel_packed["n_total_rows"],
                "drop_frac": round(sel_drop_frac, 5)},
        "deterministic": {
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cublas_workspace_config":
                os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
        },
    }

    _save_state(npz_path, meta_path, state, meta)
    print(f"{TAG} state {stem}: {n_rows} rows, {len(batch['client'])} "
          f"batches, nonslot-batch frac {nonslot_frac:.4f}, arm rows "
          f"{len(arm['rows'])}, sel rows {sel_packed['n_rows']} "
          f"(train {sel_packed['n_train']} / val {sel_packed['n_val']}, "
          f"dropped {sel_n_dropped}) ({time.time() - t0:.0f}s)")
    return state


def _save_state(npz_path, meta_path, state, meta):
    arrays = {}
    for k, v in state["rows"].items():
        arrays[f"row_{k}"] = v
    for k, v in state["batch"].items():
        arrays[f"bt_{k}"] = v
    arm = state["arms"]
    if arm["rows"]:
        arrays["arm_rows"] = np.array(arm["rows"], dtype=np.int64)
        for k, v in arm.items():
            if k.endswith("_flat") or k.endswith("_ptr"):
                arrays[f"arm_{k}"] = v
    # per-client prompt pools (needed by the NEXT state's arms)
    pvc = sorted(state["pv"])
    if pvc:
        arrays["pv_pool"] = np.stack([state["pv"][c] for c in pvc])
        arrays["pv_client"] = np.array(pvc, dtype=np.int64)
    sel = state["sel"]
    if sel["n_rows"]:
        arrays["sel_client"] = sel["client"]
        arrays["sel_rel_idx"] = sel["rel_idx"]
        arrays["sel_y_local"] = sel["y_local"]
        arrays["sel_split"] = sel["split"]
        arrays["sel_batch_id"] = sel["batch_id"]
        arrays["sel_ce"] = sel["ce"]
        arrays["sel_correct"] = sel["correct"]
    np.savez_compressed(npz_path, **arrays)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def _load_state(npz_path, meta):
    data = np.load(npz_path)
    rows = {k: data[f"row_{k}"] for k in
            ("client", "task", "rel_idx", "bseq", "y", "y_local",
             "slot_major", "slot_pref", "z", "logits_flat", "logits_ptr")}
    batch = {k: data[f"bt_{k}"] for k in
             ("client", "task", "bseq", "slot_major", "n")}
    arm = {"rows": []}
    if "arm_rows" in data:
        arm["rows"] = list(map(int, data["arm_rows"]))
        for k in data.files:
            if k.startswith("arm_lg") and (k.endswith("_flat")
                                           or k.endswith("_ptr")):
                arm[k[len("arm_"):]] = data[k]
    pv = {}
    if "pv_pool" in data:
        for i, c in enumerate(data["pv_client"]):
            pv[int(c)] = data["pv_pool"][i]
    sel_meta = meta.get("sel", {})
    sel = {"task": int(sel_meta.get("task", meta["t_index"])),
           "client": (data["sel_client"] if "sel_client" in data
                      else np.zeros((0,), np.int64)),
           "rel_idx": (data["sel_rel_idx"] if "sel_rel_idx" in data
                       else np.zeros((0,), np.int64)),
           "y_local": (data["sel_y_local"] if "sel_y_local" in data
                       else np.zeros((0,), np.int64)),
           "split": (data["sel_split"] if "sel_split" in data
                     else np.zeros((0,), np.int8)),
           "batch_id": (data["sel_batch_id"] if "sel_batch_id" in data
                        else np.zeros((0,), np.int64)),
           "ce": (data["sel_ce"] if "sel_ce" in data
                  else np.zeros((0, len(meta["fingerprint"]["lambdas"])),
                                np.float32)),
           "correct": (data["sel_correct"] if "sel_correct" in data
                       else np.zeros((0,
                                      len(meta["fingerprint"]["lambdas"])),
                                     np.int8)),
           "n_rows": int(sel_meta.get("n_rows", 0)),
           "n_train": int(sel_meta.get("n_train", 0)),
           "n_val": int(sel_meta.get("n_val", 0)),
           "n_dropped_rows": int(sel_meta.get("n_dropped_rows", 0)),
           "n_total_rows": int(sel_meta.get("n_total_rows", 0))}
    return {
        "t_index": meta["t_index"],
        "ckpt": meta["checkpoint"],
        "rows": rows,
        "batch": batch,
        "arms": arm,
        "sel": sel,
        "masks": meta["masks"],
        "head_hashes": meta["head_hashes"],
        "key_hashes": meta["key_hashes"],
        "pv": pv,
        "pv_hashes": meta.get("prompt_value_hashes", {}),
    }


# ----------------------------------------------------------------------
# analysis
# ----------------------------------------------------------------------

def _arm_row_lookup(arm):
    rows = (np.array(arm["rows"], dtype=np.int64) if arm["rows"]
            else np.zeros((0,), np.int64))
    return rows


def _arm_correct_with_ys(arm, key, arm_rows, arm_ys, query_rows):
    """correct flags for `query_rows` (a sorted subset of arm_rows)."""
    if len(arm_rows) == 0 or len(query_rows) == 0:
        return np.zeros((0,), dtype=bool)
    pos = np.searchsorted(arm_rows, query_rows)
    if pos.max() >= len(arm_rows) or \
            not np.array_equal(arm_rows[pos], query_rows):
        raise RuntimeError(
            f"arm rows do not cover the queried population ({key}) -- "
            "cache/alignment bug")
    corr_all = _list_correct(_arm_vecs(arm, key), arm_ys)
    return corr_all[pos]


def _probe_g_continuity(tr_recs, probe_g_csv):
    """Pre-registered cross-probe continuity check (Sec.32.1-7): Probe-H's
    natural-arm readouts (acc_nat_tm1 / acc_nat_t over the joined old rows)
    must reproduce Probe-G's transition records bit-identically (same run,
    same protocol, deterministic forward). Missing file -> informative
    note; present-and-mismatched -> RuntimeError (fail-closed)."""
    annex = {"probe_g_csv": str(probe_g_csv), "checked": 0,
             "mismatches": [], "note": None}
    path = Path(probe_g_csv)
    if not path.is_file():
        annex["note"] = ("Probe-G CSV not found -- cross-probe "
                         "continuity not verified (informative only)")
        return annex
    g_rows = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("level") == "transition":
                try:
                    g_rows[int(row["transition"])] = row
                except (KeyError, ValueError):
                    continue
    for tr, rec in sorted(tr_recs.items()):
        if tr not in g_rows:
            continue
        grow = g_rows[tr]
        for gk, hk in (("acc_nat_tm1", "acc_nat_tm1"),
                       ("acc_nat_t", "acc_nat_t_all_old_rows")):
            try:
                gv = float(grow[gk])
            except (KeyError, ValueError, TypeError):
                continue
            hv = float(rec[hk])
            annex["checked"] += 1
            if abs(gv - hv) > 1e-9:
                annex["mismatches"].append(
                    {"transition": tr, "field": gk,
                     "probe_g": gv, "probe_h": hv})
    if annex["mismatches"]:
        raise RuntimeError(
            f"Probe-G continuity check FAILED (natural readouts differ "
            f"from {probe_g_csv}: {annex['mismatches']}) -- deterministic "
            "forward must be bit-identical; treat as an engineering bug "
            "(fail-closed)")
    annex["note"] = (f"all {annex['checked']} natural readouts bit-identical"
                     if annex["checked"] else
                     "no overlapping transitions found in Probe-G CSV")
    return annex


def analyze(states, args_cli, lam_grid, interior_lams, run_dir):
    summary = {"probe_version": PROBE_VERSION, "run_dir": str(run_dir),
               "gates": {
                   "monopoly_slot": args_cli.monopoly_slot,
                   "gate_transition": args_cli.gate_transition,
                   "lambdas": [float(x) for x in lam_grid],
                   "select_rows_per_client": args_cli.select_rows_per_client,
                   "val_frac": args_cli.val_frac,
                   "n_boot": args_cli.n_boot,
               },
               "states": [], "transitions": [],
               "verdicts": {}, "annex": {}}
    slot_id = int(args_cli.monopoly_slot)
    lam_f_grid = [float(x) for x in lam_grid]

    # ---- per-state premise bookkeeping (H0 material) --------------------
    for st in states:
        rows = st["rows"]
        bt = st["batch"]
        bmaj = bt["slot_major"] if len(bt["slot_major"]) \
            else np.zeros((0,), np.int64)
        bmode = int(np.bincount(bmaj).argmax()) if len(bmaj) else -1
        sel = st.get("sel", {})
        summary["states"].append({
            "t_index": st["t_index"], "ckpt": st["ckpt"],
            "n_rows": int(len(rows["client"])),
            "n_batches": int(len(bt["client"])),
            "batch_mode_slot": bmode,
            "nonslot_batch_frac": round(
                float(np.mean(bmaj != slot_id)) if len(bmaj) else 0.0, 5),
            "prompt_value_hash": (list(st["pv_hashes"].values())[0]
                                  if st.get("pv_hashes") else None),
            "prompt_values_identical_across_clients":
                len(set(st.get("pv_hashes", {}).values())) == 1,
            "sel": {k: sel.get(k) for k in
                    ("n_rows", "n_train", "n_val",
                     "n_dropped_rows", "n_total_rows")},
            "sel_drop_frac": round(
                (sel.get("n_dropped_rows", 0)
                 / sel.get("n_total_rows", 1)), 5)
            if sel.get("n_total_rows", 0) else None,
        })

    # cross-state head-identity check (same-head premise)
    head_events = set()
    for later in states:
        for earlier in states:
            if earlier["t_index"] >= later["t_index"]:
                continue
            for key, hh in earlier["head_hashes"].items():
                if (key in later["head_hashes"]
                        and later["head_hashes"][key] != hh):
                    head_events.add(
                        f"head[{key}] changed t={earlier['t_index']}"
                        f"->t={later['t_index']}")
    head_clean = not head_events
    summary["annex"]["head_drift_events"] = sorted(head_events)

    agg_rows = []
    per_rows = []
    tr_recs = {}

    for i in range(1, len(states)):
        st_old, st_new = states[i - 1], states[i]
        tr = i
        new_rows, old_rows = st_new["rows"], st_old["rows"]
        n_old = int(len(old_rows["client"]))
        # canonical key join (Sec.29.1c / Sec.30/31 protocol, fail-closed)
        new_key = {}
        for j in range(len(new_rows["client"])):
            if int(new_rows["task"][j]) <= st_old["t_index"]:
                new_key[(int(new_rows["client"][j]),
                         int(new_rows["task"][j]),
                         int(new_rows["rel_idx"][j]))] = j
        new_idx = np.empty(n_old, dtype=np.int64)
        for j in range(n_old):
            kj = (int(old_rows["client"][j]), int(old_rows["task"][j]),
                  int(old_rows["rel_idx"][j]))
            if kj not in new_key:
                raise RuntimeError(
                    f"old row {kj} missing in state {st_new['ckpt']} -- "
                    "federated partition drifted; refusing")
            new_idx[j] = new_key[kj]
        if n_old > 1 and not bool(np.all(np.diff(new_idx) > 0)):
            raise RuntimeError(
                f"canonical join not monotonic across states "
                f"{st_old['ckpt']} -> {st_new['ckpt']} -- key space "
                "mismatch; refusing")

        correct_new_all, _ = _state_correct_ce(st_new)
        correct_old, _ = _state_correct_ce(st_old)
        c_new = correct_new_all[new_idx]

        # arm coverage on the NEW state's rows
        covered = new_rows["slot_major"][new_idx] == slot_id
        cov_new = new_idx[covered]          # new-state row ids, sorted
        cur_rows = np.nonzero(
            (new_rows["task"] == st_new["t_index"])
            & (new_rows["slot_major"] == slot_id))[0]

        arm = st_new["arms"]
        arm_rows = _arm_row_lookup(arm)
        arm_ys = [int(v) for v in
                   st_new["rows"]["y_local"][arm_rows]] \
            if len(arm_rows) else []
        if not len(cur_rows):
            raise RuntimeError(
                f"transition {tr}: no covered current-task test rows -- "
                "acc_cur undefined (fail-closed)")

        def _corr_cov(lam_s):
            if float(lam_s) == 1.0:
                return c_new[covered]
            key = "lg_swap" if float(lam_s) == 0.0 else f"lg_lam_{lam_s}"
            return _arm_correct_with_ys(arm, key, arm_rows, arm_ys,
                                        cov_new)

        def _corr_cur(lam_s):
            if float(lam_s) == 1.0:
                return correct_new_all[cur_rows]
            key = "lg_swap" if float(lam_s) == 0.0 else f"lg_lam_{lam_s}"
            return _arm_correct_with_ys(arm, key, arm_rows, arm_ys,
                                        cur_rows)

        # per-lambda accuracy curves over the full grid
        acc_old_by_lam = {str(l): round(_acc(_corr_cov(l)), 2)
                          for l in lam_grid}
        acc_cur_by_lam = {str(l): round(_acc(_corr_cur(l)), 2)
                          for l in lam_grid}
        lam_one = [l for l in lam_grid if float(l) == 1.0][0]
        a1_old = acc_old_by_lam[lam_one]
        a1_cur = acc_cur_by_lam[lam_one]

        # selectors from the selection signal (ties -> smallest lambda:
        # grid is ascending, np.argmin/argmax return the FIRST extremum)
        sel = st_new["sel"]
        rec_sel = {}
        lam_hat_train = lam_hat_val = None
        mean_ce_train = mean_ce_val = None
        if sel["n_rows"]:
            ce = np.asarray(sel["ce"], dtype=np.float64)
            split = np.asarray(sel["split"])
            tr_m = split == 0
            va_m = split == 1
            mean_ce_train = [round(float(v), 4)
                             for v in ce[tr_m].mean(axis=0)] \
                if tr_m.any() else None
            mean_ce_val = [round(float(v), 4)
                           for v in ce[va_m].mean(axis=0)] \
                if va_m.any() else None
            if tr_m.any():
                lam_hat_train = lam_grid[int(np.argmin(
                    ce[tr_m].mean(axis=0)))]
            if va_m.any():
                lam_hat_val = lam_grid[int(np.argmin(
                    ce[va_m].mean(axis=0)))]
            rec_sel = {"n_rows": int(sel["n_rows"]),
                       "n_train": int(sel["n_train"]),
                       "n_val": int(sel["n_val"]),
                       "mean_ce_train_by_lambda": mean_ce_train,
                       "mean_ce_val_by_lambda": mean_ce_val}

        # test-oracle (illegal, contrast only): argmax current-task acc
        cur_curve = [acc_cur_by_lam[str(l)]
                     for l in lam_grid if acc_cur_by_lam[str(l)]
                     is not None]
        lam_oracle = None
        if len(cur_curve) == len(lam_grid):
            lam_oracle = lam_grid[int(np.argmax(cur_curve))]

        # gains vs lam=1 at each chosen step
        def _gains(lam):
            if lam is None:
                return {"old_gain_pp": None, "cur_gain_pp": None}
            ao = acc_old_by_lam[str(lam)]
            ac = acc_cur_by_lam[str(lam)]
            return {"old_gain_pp": round(ao - a1_old, 3),
                    "cur_gain_pp": (round(ac - a1_cur, 3)
                                    if ac is not None else None)}

        # paired batch bootstrap CI of the old gain at a chosen step
        def _old_gain_ci(lam):
            if float(lam) == 1.0 or not len(cov_new):
                return None
            diffs = (_corr_cov(lam).astype(np.float64)
                     - c_new[covered].astype(np.float64))
            b_ids = (new_rows["client"][cov_new] * 10_000_000
                     + new_rows["task"][cov_new] * 10_000
                     + new_rows["bseq"][cov_new])
            return _paired_boot_ci(diffs, b_ids, args_cli.n_boot,
                                  args_cli.boot_seed + tr)

        ci_val = _old_gain_ci(lam_hat_val) if lam_hat_val else None
        ci_train = _old_gain_ci(lam_hat_train) if lam_hat_train else None
        ci_oracle = _old_gain_ci(lam_oracle) if lam_oracle else None

        drift_pool_all = 100.0 * float(correct_old.sum()
                                       - c_new.sum()) / n_old

        rec = {
            "transition": tr, "t_from": st_old["t_index"],
            "t_to": st_new["t_index"],
            "n_old_rows": n_old, "n_covered_old_rows": int(covered.sum()),
            "n_current_covered_rows": len(cur_rows),
            "acc_nat_tm1": round(_acc(correct_old), 2),
            "acc_nat_t_all_old_rows": round(_acc(c_new), 2),
            "drift_pool_pp": round(drift_pool_all, 3),
            "acc_old_by_lambda": acc_old_by_lam,
            "acc_current_by_lambda": acc_cur_by_lam,
            "acc_old_lam1": a1_old, "acc_current_lam1": a1_cur,
            "lam_hat_val": lam_hat_val,
            "lam_hat_train": lam_hat_train,
            "lam_oracle_current": lam_oracle,
            "gains_at_hat_val": _gains(lam_hat_val),
            "gains_at_hat_train": _gains(lam_hat_train),
            "gains_at_oracle": _gains(lam_oracle),
            "ci95_old_gain_hat_val": ci_val,
            "ci95_old_gain_hat_train": ci_train,
            "ci95_old_gain_oracle": ci_oracle,
            "selection": rec_sel,
        }
        tr_recs[tr] = rec
        summary["transitions"].append(rec)

        # aggregate CSV row (transition level)
        agg = {
            "level": "transition", "transition": tr, "client": -1,
            "n_rows": n_old, "n_covered_old": int(covered.sum()),
            "n_current_covered": len(cur_rows),
            "acc_nat_tm1": rec["acc_nat_tm1"],
            "acc_nat_t": rec["acc_nat_t_all_old_rows"],
            "drift_pool": rec["drift_pool_pp"],
            "lam_hat_train": lam_hat_train, "lam_hat_val": lam_hat_val,
            "lam_oracle_current": lam_oracle,
            "acc_old_lam1": a1_old, "acc_cur_lam1": a1_cur,
            "acc_old_at_hat_val": acc_old_by_lam.get(str(lam_hat_val)),
            "acc_cur_at_hat_val": acc_cur_by_lam.get(str(lam_hat_val)),
            "old_gain_val_pp": rec["gains_at_hat_val"]["old_gain_pp"],
            "cur_gain_val_pp": rec["gains_at_hat_val"]["cur_gain_pp"],
            "ci_old_gain_val_lo": ci_val[0] if ci_val else "",
            "ci_old_gain_val_hi": ci_val[1] if ci_val else "",
            "acc_old_at_hat_train": acc_old_by_lam.get(str(lam_hat_train)),
            "acc_cur_at_hat_train": acc_cur_by_lam.get(str(lam_hat_train)),
            "old_gain_train_pp": rec["gains_at_hat_train"]["old_gain_pp"],
            "cur_gain_train_pp": rec["gains_at_hat_train"]["cur_gain_pp"],
            "ci_old_gain_train_lo": ci_train[0] if ci_train else "",
            "ci_old_gain_train_hi": ci_train[1] if ci_train else "",
            "acc_old_at_oracle": acc_old_by_lam.get(str(lam_oracle)),
            "acc_cur_at_oracle": acc_cur_by_lam.get(str(lam_oracle)),
            "old_gain_oracle_pp": rec["gains_at_oracle"]["old_gain_pp"],
            "cur_gain_oracle_pp": rec["gains_at_oracle"]["cur_gain_pp"],
            "ci_old_gain_oracle_lo": ci_oracle[0] if ci_oracle else "",
            "ci_old_gain_oracle_hi": ci_oracle[1] if ci_oracle else "",
        }
        for l in lam_grid:
            agg[f"acc_old_lam{l}"] = acc_old_by_lam[str(l)]
            agg[f"acc_cur_lam{l}"] = acc_cur_by_lam[str(l)]
        agg_rows.append(agg)

        # per-client old-gain detail at the primary selector (is the
        # gain one-client noise?)
        if lam_hat_val is not None and float(lam_hat_val) != 1.0:
            corr_val_cov = _corr_cov(lam_hat_val)
            for cid in np.unique(old_rows["client"]):
                sel_cov = (old_rows["client"] == cid) & covered
                n_sc = int(sel_cov.sum())
                if n_sc:
                    gv = 100.0 * float(
                        corr_val_cov[sel_cov].sum()
                        - c_new[sel_cov].sum()) / n_sc
                else:
                    gv = ""
                crow = {k: "" for k in agg.keys()}
                crow.update({
                    "level": "client", "transition": tr,
                    "client": int(cid), "n_rows": int(
                        (old_rows["client"] == cid).sum()),
                    "n_covered_old": n_sc,
                    "old_gain_val_pp": round(gv, 3) if n_sc else "",
                })
                agg_rows.append(crow)

        # per-row analysis table (covered rows, old + current populations)
        if lam_hat_val is not None:
            for pop, rowset in (("old", cov_new), ("current", cur_rows)):
                if not len(rowset):
                    continue
                corr_by_lam = {l: (_corr_cur(l) if pop == "current"
                                   else _corr_cov(l)) for l in lam_grid}
                for k_i, nri in enumerate(rowset):
                    per_rows.append({
                        "transition": tr,
                        "population": pop,
                        "client": int(new_rows["client"][nri]),
                        "task": int(new_rows["task"][nri]),
                        "rel_idx": int(new_rows["rel_idx"][nri]),
                        "bseq": int(new_rows["bseq"][nri]),
                        "y": int(new_rows["y"][nri]),
                        **{f"correct_lam{l}":
                           int(corr_by_lam[l][k_i]) for l in lam_grid},
                    })

    # pre-registered cross-probe continuity check (Sec.32.1-7)
    summary["annex"]["probe_g_continuity"] = _probe_g_continuity(
        tr_recs, args_cli.probe_g_csv)

    # ---- gates (Sec.32.4 frozen tree) -----------------------------------
    premise_ok = all(
        s["batch_mode_slot"] == slot_id
        and s["nonslot_batch_frac"] <= args_cli.max_nonslot_batch_frac
        and s["prompt_values_identical_across_clients"]
        for s in summary["states"]) and all(
        (s["sel_drop_frac"] is None)
        or (s["sel_drop_frac"] <= args_cli.max_nonslot_batch_frac)
        for s in summary["states"])

    gate_tr = int(args_cli.gate_transition)
    gate = {"transition": gate_tr, "present": gate_tr in tr_recs,
            "primary_selector": "lam_hat_val"}
    gate_verdict = "N/A"
    if gate_tr in tr_recs:
        rec = tr_recs[gate_tr]
        lam_hat = rec["lam_hat_val"]
        if lam_hat is None:
            gate_verdict = "REFUSED"
            gate["note"] = ("no val split rows at the gate transition -- "
                            "selection signal degenerate")
        else:
            g = rec["gains_at_hat_val"]
            cond = (float(lam_hat) < 1.0
                    and g["cur_gain_pp"] is not None
                    and g["cur_gain_pp"] >= -EPS
                    and g["old_gain_pp"] > 0)
            ci = rec["ci95_old_gain_hat_val"]
            if cond and ci and ci[0] > 0:
                gate_verdict = "OVERSHOOT_EXPLOITABLE_CURRENT_ONLY"
            elif cond:
                gate_verdict = "YELLOW_REVIEW"
            else:
                gate_verdict = "OVERSHOOT_NOT_EXPLOITABLE"
            gate.update({
                "lam_hat_val": lam_hat,
                "acc_old_at_hat": rec["acc_old_by_lambda"][str(lam_hat)],
                "acc_cur_at_hat": rec["acc_current_by_lambda"][
                    str(lam_hat)],
                "acc_old_lam1": rec["acc_old_lam1"],
                "acc_cur_lam1": rec["acc_current_lam1"],
                "old_gain_pp": g["old_gain_pp"],
                "cur_gain_pp": g["cur_gain_pp"],
                "ci95_old_gain": ci,
            })
            # annex: selector-blind vs no-interior-gain (informative)
            lo = rec["lam_oracle_current"]
            oracle_interior = (lo is not None and float(lo) < 1.0
                               and rec["gains_at_oracle"]["old_gain_pp"]
                               is not None
                               and rec["gains_at_oracle"]["old_gain_pp"]
                               > 0
                               and rec["gains_at_oracle"]["cur_gain_pp"]
                               is not None
                               and rec["gains_at_oracle"]["cur_gain_pp"]
                               >= -EPS)
            gate["oracle_interior_dominates"] = bool(oracle_interior)
            gate["lam_oracle_current"] = lo
            note = ""
            if gate_verdict == "OVERSHOOT_NOT_EXPLOITABLE":
                if oracle_interior:
                    note = ("selector-blind: the test-oracle interior "
                            "step dominates both endpoints "
                            f"(lam*={lo}) but no legal selector finds it")
                else:
                    note = ("no-interior-gain: even the test oracle "
                            f"picks the endpoint (lam*={lo})")
            gate["annex_note"] = note
    gate["verdict"] = gate_verdict

    # secondary: same readout on the train selector (report only)
    sec = {"transition": gate_tr, "present": gate_tr in tr_recs}
    if gate_tr in tr_recs:
        rec = tr_recs[gate_tr]
        lam_t = rec["lam_hat_train"]
        if lam_t is None:
            sec["verdict"] = "N/A"
        else:
            g = rec["gains_at_hat_train"]
            cond = (float(lam_t) < 1.0
                    and g["cur_gain_pp"] is not None
                    and g["cur_gain_pp"] >= -EPS
                    and g["old_gain_pp"] > 0)
            ci = rec["ci95_old_gain_hat_train"]
            if cond and ci and ci[0] > 0:
                sec["verdict"] = "OVERSHOOT_EXPLOITABLE_CURRENT_ONLY"
            elif cond:
                sec["verdict"] = "YELLOW_REVIEW"
            else:
                sec["verdict"] = "OVERSHOOT_NOT_EXPLOITABLE"
            sec.update({"lam_hat_train": lam_t,
                        "old_gain_pp": g["old_gain_pp"],
                        "cur_gain_pp": g["cur_gain_pp"],
                        "ci95_old_gain": ci})
    # cross-transition state-adaptive readout at the primary selector
    lam_hats = [tr_recs[t]["lam_hat_val"] for t in sorted(tr_recs)
                if tr_recs[t]["lam_hat_val"] is not None]
    old_gains = [tr_recs[t]["gains_at_hat_val"]["old_gain_pp"]
                 for t in sorted(tr_recs)
                 if tr_recs[t]["gains_at_hat_val"]["old_gain_pp"]
                 is not None]
    cur_gains = [tr_recs[t]["gains_at_hat_val"]["cur_gain_pp"]
                 for t in sorted(tr_recs)
                 if tr_recs[t]["gains_at_hat_val"]["cur_gain_pp"]
                 is not None]
    summary["annex"]["state_adaptive"] = {
        "lam_hat_val_per_transition": {
            str(t): tr_recs[t]["lam_hat_val"] for t in sorted(tr_recs)},
        "lam_hats_distinct": len(set(map(str, lam_hats))) > 1,
        "mean_old_gain_pp_at_hat_val": round(float(np.mean(old_gains)), 3)
        if old_gains else None,
        "mean_cur_gain_pp_at_hat_val": round(float(np.mean(cur_gains)), 3)
        if cur_gains else None,
    }

    summary["verdicts"] = {
        "H0_premise": "OK" if premise_ok else "REFUSED",
        "gate": gate,
        "secondary_train_selector": sec,
        "head_identity_clean": bool(head_clean),
        "hypothesis": ("the learned prompt endpoint is temporally "
                       "miscalibrated -- a current-task-signal-calibrated "
                       "interior step avoids the endpoint's harm on BOTH "
                       "old and current tasks without accessing any "
                       "old-task data"),
    }

    # ---- FINAL (Sec.32.4 frozen tree) ----------------------------------
    if not premise_ok:
        final = ("REFUSED: H0 premise violated (monopoly / deployment "
                 "path / selection coverage / prompt-value premise) -- "
                 "no verdict (fail-closed).")
    elif gate_verdict == "N/A":
        final = ("INCOMPLETE_STATES: the pre-registered gate transition "
                 f"{gate_tr} is not present in the extracted state list "
                 "-- no formal verdict (smoke runs only).")
    elif gate_verdict == "OVERSHOOT_EXPLOITABLE_CURRENT_ONLY":
        g = gate
        final = (
            "OVERSHOOT_EXPLOITABLE_CURRENT_ONLY: the learned prompt "
            "endpoint is temporally miscalibrated, and a current-task-"
            f"signal-calibrated interior step (lam_hat_val="
            f"{g['lam_hat_val']} < 1) avoids the endpoint's harm on BOTH "
            f"old (+{g['old_gain_pp']}pp, CI95 {g['ci95_old_gain']}) and "
            f"current ({g['cur_gain_pp']:+}pp) tasks without any "
            "old-task information -- CTRP (Calibrated Temporal Residual "
            "Prompt) earns Sec.33 candidacy, conditional on INR "
            "mechanism replication + external review. No training arm is "
            "pre-registered by this probe.")
    elif gate_verdict == "YELLOW_REVIEW":
        g = gate
        final = (
            "YELLOW_REVIEW: the val-current selector picks an interior "
            f"step (lam_hat_val={g['lam_hat_val']}) with the right "
            f"direction (old_gain {g['old_gain_pp']}pp, current "
            f"{g['cur_gain_pp']:+}pp) but the paired batch-bootstrap "
            f"95% CI {g['ci95_old_gain']} does not exclude zero -- "
            "escalate to external review.")
    elif gate_verdict == "OVERSHOOT_NOT_EXPLOITABLE":
        g = gate
        final = (
            "OVERSHOOT_NOT_EXPLOITABLE: current-task signals do not "
            f"locate a dominating interior step at the {gate_tr} "
            f"boundary (lam_hat_val={g.get('lam_hat_val')}, old_gain "
            f"{g.get('old_gain_pp')}pp, cur_gain "
            f"{g.get('cur_gain_pp')}pp, CI95 "
            f"{g.get('ci95_old_gain')}) -- the overshoot is recorded "
            "but not deployable. Frozen wording: the interior gain is "
            f"not locatable by current-task signals. Annex: "
            f"{g.get('annex_note') or 'n/a'}.")
    else:
        final = f"REFUSED: {gate.get('note', 'gate degenerate')}"
    if not head_clean:
        final += (" WARNING: cross-state head drift detected (gate "
                  "CONTAMINATED) -- see annex.head_drift_events.")
    summary["verdicts"]["FINAL"] = final
    return summary, agg_rows, per_rows


# ----------------------------------------------------------------------
# output writers + main
# ----------------------------------------------------------------------

AGG_BASE_FIELDS = ["level", "transition", "client", "n_rows",
                   "n_covered_old", "n_current_covered", "acc_nat_tm1",
                   "acc_nat_t", "drift_pool", "lam_hat_train",
                   "lam_hat_val", "lam_oracle_current", "acc_old_lam1",
                   "acc_cur_lam1", "acc_old_at_hat_val",
                   "acc_cur_at_hat_val", "old_gain_val_pp",
                   "cur_gain_val_pp", "ci_old_gain_val_lo",
                   "ci_old_gain_val_hi", "acc_old_at_hat_train",
                   "acc_cur_at_hat_train", "old_gain_train_pp",
                   "cur_gain_train_pp", "ci_old_gain_train_lo",
                   "ci_old_gain_train_hi", "acc_old_at_oracle",
                   "acc_cur_at_oracle", "old_gain_oracle_pp",
                   "cur_gain_oracle_pp", "ci_old_gain_oracle_lo",
                   "ci_old_gain_oracle_hi"]

ROW_BASE_FIELDS = ["transition", "population", "client", "task",
                   "rel_idx", "bseq", "y"]


def main():
    args_cli = parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    ckpts = [c.strip() for c in args_cli.checkpoints.split(",") if c.strip()]
    lam_grid = [x.strip() for x in args_cli.lambdas.split(",") if x.strip()]
    if len(lam_grid) < 2:
        raise RuntimeError("lambda grid must have >= 2 points")
    lam_f = [float(x) for x in lam_grid]
    if lam_f != sorted(lam_f) or len(set(lam_f)) != len(lam_f):
        raise RuntimeError(
            f"lambda grid must be ascending and unique: {lam_grid}")
    if 0.0 not in lam_f or 1.0 not in lam_f:
        raise RuntimeError("lambda grid must contain both endpoints 0/1")
    interior_lams = [x for x in lam_grid if 0.0 < float(x) < 1.0]

    states = []
    for t_index, ckpt in enumerate(ckpts):
        prev = states[-1] if states else None
        st = extract_state(args_cli, args_cli.run_dir, ckpt, prev,
                           lam_grid, interior_lams, t_index)
        states.append(st)

    summary, agg_rows, per_rows = analyze(
        states, args_cli, lam_grid, interior_lams, args_cli.run_dir)

    row_fields = ROW_BASE_FIELDS + [f"correct_lam{l}" for l in lam_grid]
    with open(args_cli.out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=AGG_BASE_FIELDS
                           + [f"acc_old_lam{l}" for l in lam_grid]
                           + [f"acc_cur_lam{l}" for l in lam_grid])
        w.writeheader()
        w.writerows(agg_rows)
    with open(args_cli.out_rows_csv, "w", encoding="utf-8",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=row_fields)
        w.writeheader()
        w.writerows(per_rows)
    with open(args_cli.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    v = summary["verdicts"]
    print(f"{TAG} H0: premise {v['H0_premise']} (monopoly slot "
          f"{args_cli.monopoly_slot}; nonslot batch frac "
          f"{[s['nonslot_batch_frac'] for s in summary['states']]}; "
          f"sel drop frac "
          f"{[s['sel_drop_frac'] for s in summary['states']]}; heads "
          f"clean {v['head_identity_clean']})")
    for rec in summary["transitions"]:
        print(f"{TAG} t{rec['transition']}: lam_hat_val="
              f"{rec['lam_hat_val']} (old_gain "
              f"{rec['gains_at_hat_val']['old_gain_pp']}pp, cur_gain "
              f"{rec['gains_at_hat_val']['cur_gain_pp']}pp), "
              f"lam_hat_train={rec['lam_hat_train']}, oracle="
              f"{rec['lam_oracle_current']} (old_gain "
              f"{rec['gains_at_oracle']['old_gain_pp']}pp), CI95(val) "
              f"{rec['ci95_old_gain_hat_val']}")
    g = v["gate"]
    print(f"{TAG} MAIN GATE (transition {args_cli.gate_transition}, "
          f"primary lam_hat_val): {g.get('verdict')} "
          f"[lam_hat={g.get('lam_hat_val')}, old_gain "
          f"{g.get('old_gain_pp')}pp CI95 {g.get('ci95_old_gain')}, "
          f"cur_gain {g.get('cur_gain_pp')}pp] "
          f"{g.get('annex_note', '')}")
    sec = v["secondary_train_selector"]
    print(f"{TAG} secondary (train selector): {sec.get('verdict')} "
          f"[lam_hat_train={sec.get('lam_hat_train')}, old_gain "
          f"{sec.get('old_gain_pp')}pp]")
    print(f"{TAG} state-adaptive annex: {summary['annex']['state_adaptive']}")
    print(f"{TAG} FINAL: {v['FINAL']}")
    print(f"{TAG} rows -> {args_cli.out_rows_csv}; aggregates -> "
          f"{args_cli.out_csv}; summary -> {args_cli.out_json}")


if __name__ == "__main__":
    main()
