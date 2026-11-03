"""Probe-G: same-slot prompt-content drift (RUNCOMMANDS Sec.31, offline
only, pre-registered 2026-11-03).

Zero-training offline probe dissecting the ONE surviving representation
channel from Probe-F: the evolving VALUE of the monopoly prompt slot
(slot 6) on the preserved run's adjacent deployment-state checkpoints
(task_XX_end.pth = post-broadcast states). Frozen ladder:

  G0  premise gate (fail-closed): deployment path (top_k=1,
      batchwise_prompt, no task mask), per-state batch-majority mode ==
      monopoly slot with <=1% non-slot batches, monopoly slot constant
      across states, prompt VALUES identical across clients (global
      tensor), heads identical across states.
  G1  prompt self-drift (falsification arm, no gate): rel_norm and
      cosine drift of P_{6,s} -> P_{6,s+1} per boundary, side by side
      with the natural-arm feature drift medians. Pre-registered
      expectation: "magnitude is not the dose" (Sec.30.8 prior: the
      largest prompt drift, 0->1, is the BENEFICIAL transition).
  G2  old-sensitive direction decomposition (descriptive): decompose
      the prompt update dP = P_{6,s+1} - P_{6,s} into old-task-sensitive
      and insensitive directions. Estimator (frozen): per-row gradients
      g_i = dCE_i/dP_{6,s} of old-task masked CE at state s (analysis
      only -- torch.autograd.grad, no .grad writes, no optimizer, no
      parameter updates); SVD subspace capturing >=90% gradient energy
      (rank cap 128); report frac_old_sensitive = ||Proj dP||^2/||dP||^2,
      first-order predicted mean dCE vs actual, per-row Pearson.
  G3  prompt-swap causal oracle (MAIN GATE, GPT frozen): at state s+1,
      forward with EVERYTHING frozen except the slot-6 VALUE replaced by
      P_{6,s}. recovery = Acc(swap, old-task rows) - Acc(natural, old
      rows). Gate: at the 2->3 transition recovery >= +0.5pp AND
      paired batch-bootstrap 95% CI lower bound > 0. PASS = causal
      evidence that new prompt content -> historical performance loss.
  G4  same-slot lambda interpolation (dose-response corroboration):
      P_lambda = (1-lam) P_{6,s} + lam P_{6,s+1} (SAME-slot tensor mix,
      legal semantics unlike the cross-slot DTCPR mix), lam in
      {0, 0.25, 0.5, 0.75, 1}. Read the stability-plasticity trajectory
      on old-task AND current-task rows. PASS at 2->3 =
      Spearman(acc_old, lam) < 0 and Spearman(acc_cur, lam) > 0;
      range<0.05pp curves are FLAT. G4 alone never rescues a G3 FAIL.

Route identity is preserved by construction: routing reads prompt_key
only, so swapping/interpolating the slot-6 VALUE cannot change the
route (asserted per batch anyway; arm coverage = slot-6 majority
batches, ~99.94% of rows on this run).

Deviations from native evaluate() (pre-registered, same as Sec.30):
fixed batch order at batch_size 8 (pairing), head.eval() (Chead Dropout
OFF; absolute numbers sit ~1-2pp above the native matrix; all gates act
on cross-arm differences so the bias cancels; native-draw annex locates
the probe-vs-CSV gap, tolerance 3pp).

fail-closed (Sec.31.5): use_prompt_mask=True / top_k!=1 /
batchwise_prompt=False -> refuse; monopoly premise violations -> refuse;
prompt values differing across clients -> refuse; canon join gaps or
non-monotonic joins -> refuse; gradient batches whose majority slot is
not the monopoly slot are dropped from the G2 subsample (counted);
cache fingerprints (probe version, run, checkpoint, batch, monopoly
slot, lambdas, g2_rows) must match.

Usage (from repo root):

  python diagnostics/probe_prompt_content_drift.py \
      --run_dir output/cifar100/fedta/cifar_a5v2_M2_feat_full/seed_42 \
      --data_path ./local_datasets --device cuda \
      --cache_dir diagnostics_output/contentdrift_cifar \
      --out_csv diagnostics_output/contentdrift_cifar.csv \
      --out_rows_csv diagnostics_output/contentdrift_cifar_rows.csv \
      --out_json diagnostics_output/contentdrift_cifar.summary.json
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

PROBE_VERSION = "contentdrift-v1"
TAG = "[contentdrift-probe]"
EPS = 1e-8


def parse_args():
    p = argparse.ArgumentParser("Probe-G same-slot prompt-content drift")
    p.add_argument("--run_dir", required=True,
                   help="run directory (.../{data_name}/{method}/{run_name}/"
                        "seed_{seed})")
    p.add_argument("--checkpoints",
                   default="task_00_end.pth,task_01_end.pth,"
                           "task_02_end.pth,task_03_end.pth,"
                           "task_04_end.pth")
    p.add_argument("--data_path", default=None)
    p.add_argument("--device", default="cuda",
                   help="cuda required (Tail_Anchor hardcodes .to('cuda'))")
    p.add_argument("--cache_dir", required=True,
                   help="independent cache dir (Sec.29/30 fail-closed "
                        "lesson: never share caches across probes)")
    p.add_argument("--out_csv", required=True)
    p.add_argument("--out_rows_csv", required=True)
    p.add_argument("--out_json", required=True)
    p.add_argument("--batch_size", default=8, type=int,
                   help="fixed evaluation batch (native evaluate uses 8; "
                        "fixed order is the pre-registered pairing)")
    p.add_argument("--monopoly_slot", default=6, type=int,
                   help="the monopoly slot this probe dissects (Sec.30.8: "
                        "slot 6 held >99.9% on all 5 states)")
    p.add_argument("--max_nonslot_batch_frac", default=0.01, type=float,
                   help="G0 premise: max fraction of batches whose majority "
                        "slot is not the monopoly slot")
    p.add_argument("--lambdas", default="0,0.25,0.5,0.75,1",
                   help="lambda grid for G4 (0=swap arm, 1=natural arm)")
    p.add_argument("--g3_gate", default=0.5, type=float,
                   help="G3 main gate: swap recovery at the dissection "
                        "transition must reach this many pp")
    p.add_argument("--g3_transition", default=3, type=int,
                   help="1-based transition index of the main dissection "
                        "boundary (3 = states 2 -> 3)")
    p.add_argument("--g2_rows", default=256, type=int,
                   help="G2 gradient subsample size per boundary (stratified "
                        "by (client, task), slot-6 batches only)")
    p.add_argument("--g2_energy", default=0.9, type=float,
                   help="G2 subspace: SVD cumulative energy fraction")
    p.add_argument("--g2_max_rank", default=128, type=int)
    p.add_argument("--n_boot", default=1000, type=int,
                   help="paired batch bootstrap replicates (B=1000; "
                        "resampling unit = BATCH, Sec.30.7 statistics)")
    p.add_argument("--boot_seed", default=20261103, type=int)
    p.add_argument("--native_draws", default=2, type=int,
                   help="per (state, client, task): native-mode accuracy "
                        "draws (Chead Dropout ON + shuffle, seeded) to "
                        "LOCATE the probe-vs-client-CSV gap (informative); "
                        "formal run uses 4")
    p.add_argument("--native_tol", default=3.0, type=float)
    p.add_argument("--overwrite_cache", action="store_true")
    return p.parse_args()


# ----------------------------------------------------------------------
# shared numeric helpers (probe-routing_flip.py lineage, Sec.30)
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
    total_prompt_len). This is the swap / lambda-mix forward. Must be an
    nn.Module: vit.prompt is a registered child module, so assigning a
    plain object would be rejected by Module.__setattr__."""

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


def _median(x):
    x = np.asarray(x)
    return float(np.median(x)) if len(x) else float("nan")


def _vec_ce(vecs, ys):
    """Per-row cross-entropy from masked-logit vectors."""
    out = np.empty(len(ys), dtype=np.float64)
    for i in range(len(ys)):
        v = vecs[i]
        mx = v.max()
        out[i] = math.log(np.exp(v - mx).sum()) + mx - v[int(ys[i])]
    return out


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


def _pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 3:
        return float("nan")
    xm, ym = x - x.mean(), y - y.mean()
    denom = math.sqrt(float((xm ** 2).sum() * (ym ** 2).sum()))
    if denom <= 0:
        return float("nan")
    return float((xm * ym).sum() / denom)


def _spearman(x, y):
    """Spearman rho with average-rank tie handling."""
    def _ranks(v):
        v = np.asarray(v, dtype=np.float64)
        order = np.argsort(v, kind="stable")
        sv = v[order]
        r = np.empty(len(v), dtype=np.float64)
        i = 0
        while i < len(sv):
            j = i
            while j + 1 < len(sv) and sv[j + 1] == sv[i]:
                j += 1
            r[order[i:j + 1]] = 0.5 * (i + j) + 1.0
            i = j + 1
        return r
    return _pearson(_ranks(x), _ranks(y))


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


# ----------------------------------------------------------------------
# per-state extraction
# ----------------------------------------------------------------------

def _fingerprint(run_args, run_dir, ckpt_name, args_cli, interior_lams):
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
        "lambdas": list(interior_lams),
        "g2_rows": args_cli.g2_rows,
    }


def extract_state(args_cli, run_dir, ckpt_name, prev_state, interior_lams,
                  t_index, is_last_state):
    """Extract one deployment state into row/batch/arm arrays.

    Arms (state t >= 1, on every monopoly-slot majority batch, over ALL
    tasks 0..t): swap (lam=0, slot-6 value from the PREVIOUS state) and
    interior-lambda mixes (1-lam)*P_prev + lam*P_now; natural (lam=1) is
    the row pass itself. Caches to {cache_dir}/state_{stem}.npz (+json
    meta) with fail-closed fingerprint checks."""
    cache_dir = Path(args_cli.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(ckpt_name).stem
    npz_path = cache_dir / f"state_{stem}.npz"
    meta_path = cache_dir / f"state_{stem}.json"

    run_args_probe = load_args_from_run_dir(run_dir)
    if bool(run_args_probe.use_prompt_mask):
        raise RuntimeError(
            "use_prompt_mask=True: deployment routing is the task-mask "
            "path and Probe-G semantics (monopoly-slot value swap) do "
            "not apply. Refusing (Sec.31 fail-closed).")
    if int(run_args_probe.top_k) != 1:
        raise RuntimeError(
            f"top_k={run_args_probe.top_k}: probe pre-registered for "
            "top_k=1 deployments only. Refusing (Sec.31 fail-closed).")
    if not bool(run_args_probe.batchwise_prompt):
        raise RuntimeError(
            "batchwise_prompt=False: routes are per-sample and the "
            "monopoly-batch coverage semantics do not apply. Refusing "
            "(Sec.31 fail-closed).")
    if args_cli.device == "cpu" or not torch.cuda.is_available():
        raise RuntimeError(
            "this probe requires --device cuda (Tail_Anchor.forward "
            "hardcodes .to('cuda'); official model semantics)")

    fp = _fingerprint(run_args_probe, run_dir, ckpt_name, args_cli,
                      interior_lams)

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
    # as extract_features.py / probe_routing_flip.py)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device(args_cli.device)

    server.model.to(device)
    server.origin_model.to(device)
    server.origin_model.eval()

    slot_id = int(args_cli.monopoly_slot)

    r = {k: [] for k in
         ("client", "task", "rel_idx", "bseq", "y", "y_local",
          "slot_major", "slot_pref", "z", "logits_flat", "logits_ptr")}
    batch = {k: [] for k in
             ("client", "task", "bseq", "slot_major", "n")}
    arm = {"rows": [], "z_swap": [], "lg_swap": []}
    for lam in interior_lams:
        arm[f"lg_lam_{lam}"] = []
    g2_cand = []        # (cid, task, rel_idx, y_local) in slot-6 batches
    class_masks = {}
    head_hashes = {}
    key_hashes = {}
    pv_cpu = {}
    native_draws = {}
    n_rows = 0
    flat_len = 0
    n_nonslot_batches = 0

    for client in server.clients:
        if client.task_id < 0:
            continue
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
                    sim = out["similarity"].float()
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
                pref = sim.argmax(dim=1)

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
                    r["slot_pref"].append(int(pref[bi]))
                    r["z"].append(z[bi].cpu().numpy().astype(np.float32))
                    r["logits_flat"].append(
                        lg[bi].cpu().numpy().astype(np.float32))
                    r["logits_ptr"].append(flat_len)
                    flat_len += int(lg.shape[1])
                    if in_pool:
                        g2_cand.append((cid, t, int(indices[n_done + bi]),
                                        int(y_local[bi])))
                n_rows += bsz
                n_done += bsz

                # ---- arms: monopoly-slot batches only (G3/G4) ---------
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
                    arm["z_swap"].append(z_sw[bi].cpu().numpy())
                    arm["lg_swap"].append(lg_sw[bi].cpu().numpy())
                # interior lambda mixes (G4 dose arms)
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

            # ---- native-mode draws (annex: locate the probe-vs-CSV gap)
            if args_cli.native_draws > 0:
                draw_accs = []
                for d in range(args_cli.native_draws):
                    torch.manual_seed(_seed(run_args.seed, t_index, cid, t, d))
                    g = torch.Generator()
                    g.manual_seed(_seed(run_args.seed, 7, t_index, cid, t, d))
                    client.model.load_head(client.heads[t])  # fresh copy,
                    # training mode -> Dropout active (native semantics)
                    dl = DataLoader(subset, batch_size=args_cli.batch_size,
                                    shuffle=True, generator=g,
                                    num_workers=run_args.num_workers)
                    ncorr = 0
                    ntot = 0
                    for inp2, tgt2 in dl:
                        inp2 = inp2.to(device, non_blocking=True)
                        tgt2 = tgt2.to(device, non_blocking=True)
                        with torch.no_grad():
                            out_ref2 = server.origin_model(inp2)
                            q2 = out_ref2["pre_logits"]
                            out2 = client.vit(inp2,
                                              task_id=client.task_id,
                                              cls_features=q2, train=True)
                            lg2 = client.model(
                                out2["feat"].float(), None)[0]
                            lg2 = lg2.index_select(1, mpos).float()
                        pred2 = lg2.argmax(dim=1)
                        ncorr += int((pred2 == torch.searchsorted(
                            mpos, tgt2)).sum())
                        ntot += int(tgt2.numel())
                    draw_accs.append(100.0 * ncorr / ntot)
                native_draws[f"{cid},{t}"] = [round(a, 2) for a in draw_accs]
                # restore the deterministic eval-mode head
                client.model.load_head(client.heads[t])
                client.model.eval()

    # ---- G0 premise checks (fail-closed) -------------------------------
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
            "Probe-G semantics are only defined for this deployment "
            "path. Refusing (Sec.31 fail-closed).")
    pv_hashes = {str(c): _hash_array(v) for c, v in pv_cpu.items()}
    if len(set(pv_hashes.values())) != 1:
        raise RuntimeError(
            f"state {stem}: prompt VALUES differ across clients -- "
            "P_{6,t} is not a single global tensor (broadcast premise "
            "broken). Refusing (Sec.31 fail-closed).")

    # ---- G2 gradient subsample (analysis-only autograd) -----------------
    g2 = {"g6": np.zeros((0,), np.float32),
          "client": np.zeros((0,), np.int64),
          "task": np.zeros((0,), np.int64),
          "rel_idx": np.zeros((0,), np.int64),
          "n_dropped_batches_rows": 0, "n_rows": 0}
    if not is_last_state and len(g2_cand) > 0:
        g2 = _g2_gradient_pass(server, run_args, g2_cand, args_cli,
                               slot_id, device, t_index)

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
                  if k == "z_swap" or k.startswith("lg_")]:
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
        "masks": class_masks,
        "head_hashes": head_hashes,
        "key_hashes": key_hashes,
        "pv": pv_cpu,
        "pv_hashes": pv_hashes,
        "native_draws": native_draws,
        "g2": g2,
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
        "native_draws": native_draws,
        "prompt_value_hashes": pv_hashes,
        "embedding_key": run_args_probe.embedding_key,
        "length": int(run_args_probe.length),
        "pool_size": int(run_args_probe.size),
        "deterministic": {
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cublas_workspace_config":
                os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
        },
    }

    _save_state(npz_path, meta_path, state, meta)
    print(f"{TAG} state {stem}: {n_rows} rows, {len(batch['client'])} "
          f"batches, nonslot-batch frac {nonslot_frac:.4f}, arm rows "
          f"{len(arm['rows'])}, g2 rows {g2['n_rows']} "
          f"({time.time() - t0:.0f}s)")
    return state


def _g2_gradient_pass(server, run_args, g2_cand, args_cli, slot_id, device,
                      t_index):
    """Analysis-only per-row gradients g_i = dCE_i/dP_{6,s} (Sec.31.3 G2).

    Zero-training compliance: torch.autograd.grad only (never writes
    .grad, no optimizer, no parameter update); the network is left
    bit-identical (all forwards under eval(), no state mutation beyond
    head reloading, which the row pass does anyway). The subsample is
    stratified by (client, task) with a stable derived seed and drawn
    from monopoly-slot batches only. Gradient batches are re-grouped at
    batch_size; if any re-grouped batch's majority slot is not the
    monopoly slot its rows are DROPPED and counted (Sec.31.5-6)."""
    cand = np.array(g2_cand, dtype=np.int64)   # cols: cid, task, rel, yloc
    rng = np.random.default_rng(_seed(run_args.seed, 91, t_index))
    groups = {}
    for row in cand:
        groups.setdefault((int(row[0]), int(row[1])), []).append(row)
    quota = max(1, int(math.ceil(args_cli.g2_rows / max(1, len(groups)))))
    chosen = []
    for k in sorted(groups.keys()):
        rows = np.array(groups[k], dtype=np.int64)
        if len(rows) > quota:
            rows = rows[rng.permutation(len(rows))[:quota]]
        chosen.extend([tuple(int(v) for v in rr) for rr in rows])
    chosen = np.array(sorted(chosen), dtype=np.int64)  # (cid, task, rel, y)
    if len(chosen) > args_cli.g2_rows:
        chosen = chosen[np.sort(rng.permutation(len(chosen))
                               [:args_cli.g2_rows])]

    by_ct = {}
    yloc_of = {}
    for row in chosen:
        cid_i, task_i, rel_i, yl_i = (int(row[0]), int(row[1]),
                                      int(row[2]), int(row[3]))
        by_ct.setdefault((cid_i, task_i), []).append(rel_i)
        yloc_of[(cid_i, task_i, rel_i)] = yl_i

    g6_rows, out_keys = [], []
    n_drop = 0
    for client in server.clients:
        if client.task_id < 0:
            continue
        cid = int(client.id)
        ct_keys = sorted(k for k in by_ct if k[0] == cid)
        if not ct_keys:
            continue
        client.vit.load_prompts(client.prompts)
        client.vit.eval()
        client.model.to(device)
        pool = client.vit.prompt.prompt
        if not pool.requires_grad:
            pool.requires_grad_(True)
        for (ccid, task) in ct_keys:
            rels = sorted(int(x) for x in by_ct[(ccid, task)])
            mask_j = sorted(int(x) for x in client.class_mask[task])
            mpos = torch.tensor(mask_j, dtype=torch.long, device=device)
            client.model.load_head(client.heads[task])
            client.model.eval()
            subset = Subset(client.train_data[task], rels)
            loader = DataLoader(subset, batch_size=args_cli.batch_size,
                                shuffle=False,
                                num_workers=run_args.num_workers)
            pos = 0
            for inp, tgt in loader:
                inp = inp.to(device, non_blocking=True)
                rels_b = rels[pos:pos + int(inp.shape[0])]
                pos += int(inp.shape[0])
                with torch.no_grad():
                    out_ref = server.origin_model(inp)
                    q = out_ref["pre_logits"]
                out = client.vit(inp, task_id=client.task_id,
                                 cls_features=q, train=True)
                idx_nat = out["prompt_idx"]
                if int(idx_nat[0, 0]) != slot_id:
                    n_drop += int(inp.shape[0])
                    continue   # graph freed on scope exit
                z = out["feat"].float()
                logits = client.model(z, None)[0].index_select(1, mpos)
                for bi in range(int(inp.shape[0])):
                    yl = yloc_of[(ccid, task, rels_b[bi])]
                    ce_i = torch.nn.functional.cross_entropy(
                        logits[bi:bi + 1],
                        torch.tensor([yl], device=device))
                    # retain_graph: one shared forward graph per batch,
                    # per-row grads within it (freed when `out` is
                    # rebound on the next batch)
                    g = torch.autograd.grad(ce_i, pool,
                                            retain_graph=True)[0][slot_id]
                    g6_rows.append(g.detach().reshape(-1)
                                    .cpu().numpy().astype(np.float32))
                    out_keys.append((ccid, task, rels_b[bi]))
    g2 = {"g6": (np.stack(g6_rows) if g6_rows
                 else np.zeros((0,), np.float32)),
          "client": np.array([k[0] for k in out_keys], dtype=np.int64),
          "task": np.array([k[1] for k in out_keys], dtype=np.int64),
          "rel_idx": np.array([k[2] for k in out_keys], dtype=np.int64),
          "n_dropped_batches_rows": int(n_drop),
          "n_rows": len(out_keys)}
    return g2


def _save_state(npz_path, meta_path, state, meta):
    arrays = {}
    for k, v in state["rows"].items():
        arrays[f"row_{k}"] = v
    for k, v in state["batch"].items():
        arrays[f"bt_{k}"] = v
    arm = state["arms"]
    if arm["rows"]:
        arrays["arm_rows"] = np.array(arm["rows"], dtype=np.int64)
        arrays["arm_z_swap"] = np.stack(arm["z_swap"])
        for k, v in arm.items():
            if k.endswith("_flat") or k.endswith("_ptr"):
                arrays[f"arm_{k}"] = v
    # per-client prompt pools (needed by the NEXT state's arms)
    pvc = sorted(state["pv"])
    if pvc:
        arrays["pv_pool"] = np.stack([state["pv"][c] for c in pvc])
        arrays["pv_client"] = np.array(pvc, dtype=np.int64)
    g2 = state["g2"]
    if g2["n_rows"]:
        arrays["g2_g6"] = g2["g6"]
        arrays["g2_client"] = g2["client"]
        arrays["g2_task"] = g2["task"]
        arrays["g2_rel_idx"] = g2["rel_idx"]
    meta_out = dict(meta)
    meta_out["g2"] = {"n_rows": int(state["g2"]["n_rows"]),
                      "n_dropped_batches_rows":
                          int(state["g2"]["n_dropped_batches_rows"])}
    np.savez_compressed(npz_path, **arrays)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2)


def _load_state(npz_path, meta):
    data = np.load(npz_path)
    rows = {k: data[f"row_{k}"] for k in
            ("client", "task", "rel_idx", "bseq", "y", "y_local",
             "slot_major", "slot_pref", "z", "logits_flat", "logits_ptr")}
    batch = {k: data[f"bt_{k}"] for k in
             ("client", "task", "bseq", "slot_major", "n")}
    arm = {"rows": [], "z_swap": []}
    if "arm_rows" in data:
        arm["rows"] = list(map(int, data["arm_rows"]))
        arm["z_swap"] = data["arm_z_swap"]
        for k in data.files:
            if k.startswith("arm_lg") and (k.endswith("_flat")
                                           or k.endswith("_ptr")):
                arm[k[len("arm_"):]] = data[k]
    pv = {}
    if "pv_pool" in data:
        for i, c in enumerate(data["pv_client"]):
            pv[int(c)] = data["pv_pool"][i]
    g2 = {"g6": (data["g2_g6"] if "g2_g6" in data
                 else np.zeros((0,), np.float32)),
          "client": (data["g2_client"] if "g2_client" in data
                     else np.zeros((0,), np.int64)),
          "task": (data["g2_task"] if "g2_task" in data
                   else np.zeros((0,), np.int64)),
          "rel_idx": (data["g2_rel_idx"] if "g2_rel_idx" in data
                      else np.zeros((0,), np.int64)),
          "n_rows": int(meta.get("g2", {}).get("n_rows", 0)),
          "n_dropped_batches_rows":
              int(meta.get("g2", {}).get("n_dropped_batches_rows", 0))}
    return {
        "t_index": meta["t_index"],
        "ckpt": meta["checkpoint"],
        "rows": rows,
        "batch": batch,
        "arms": arm,
        "masks": meta["masks"],
        "head_hashes": meta["head_hashes"],
        "key_hashes": meta["key_hashes"],
        "native_draws": meta.get("native_draws", {}),
        "pv": pv,
        "pv_hashes": meta.get("prompt_value_hashes", {}),
        "g2": g2,
    }


# ----------------------------------------------------------------------
# analysis
# ----------------------------------------------------------------------

def _arm_row_lookup(arm):
    """(sorted arm rows array, positions) for canon-row -> arm indexing."""
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


def analyze(states, args_cli, lam_grid, interior_lams, run_dir):
    summary = {"probe_version": PROBE_VERSION, "run_dir": str(run_dir),
               "gates": {
                   "monopoly_slot": args_cli.monopoly_slot,
                   "max_nonslot_batch_frac": args_cli.max_nonslot_batch_frac,
                   "g3_gate": args_cli.g3_gate,
                   "g3_transition": args_cli.g3_transition,
                   "g2_rows": args_cli.g2_rows,
                   "g2_energy": args_cli.g2_energy,
                   "g2_max_rank": args_cli.g2_max_rank,
                   "native_tol": args_cli.native_tol},
               "states": [], "transitions": [],
               "verdicts": {}, "annex": {}}
    slot_id = int(args_cli.monopoly_slot)

    # ---- per-state premise bookkeeping (G0 material) --------------------
    for st in states:
        rows = st["rows"]
        bt = st["batch"]
        bmaj = bt["slot_major"] if len(bt["slot_major"]) \
            else np.zeros((0,), np.int64)
        bmode = int(np.bincount(bmaj).argmax()) if len(bmaj) else -1
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
            "nonslot_row_share": round(
                float(np.mean(rows["slot_major"] != slot_id))
                if len(rows["slot_major"]) else 0.0, 5),
        })

    # cross-state head-identity check (G3 "same head" premise)
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
    recoveries = {}          # transition -> recovery dict
    g1_records = []
    g2_records = []
    g4_records = {}

    for i in range(1, len(states)):
        st_new, st_old = states[i], states[i - 1]
        tr = i
        new_rows, old_rows = st_new["rows"], st_old["rows"]
        n_old = int(len(old_rows["client"]))
        # canonical key join (Sec.29.1c / Sec.30 protocol, fail-closed)
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

        correct_new_all, ce_new_all = _state_correct_ce(st_new)
        correct_old, ce_old = _state_correct_ce(st_old)
        c_new = correct_new_all[new_idx]
        ce_new_o = ce_new_all[new_idx]
        d_err = ce_new_o - ce_old

        # arm coverage on the NEW state's old-task rows
        covered = new_rows["slot_major"][new_idx] == slot_id
        cov_new = new_idx[covered]          # new-state row ids, sorted
        y_cov = new_rows["y_local"][cov_new]

        arm = st_new["arms"]
        arm_rows = _arm_row_lookup(arm)
        # arm rows carry their own y_local (same rows, same canon state)
        arm_ys = [int(v) for v in
                  st_new["rows"]["y_local"][arm_rows]] \
            if len(arm_rows) else []
        correct_swap_cov = _arm_correct_with_ys(
            arm, "lg_swap", arm_rows, arm_ys, cov_new)

        # natural-vs-swap paired recovery on covered old rows
        n_cov = int(covered.sum())
        recovery_pp = (100.0 * float(correct_swap_cov.sum()
                                    - c_new[covered].sum()) / n_cov
                      if n_cov else float("nan"))
        b_ids_cov = (new_rows["client"][cov_new] * 10_000_000
                     + new_rows["task"][cov_new] * 10_000
                     + new_rows["bseq"][cov_new])
        diffs_cov = (correct_swap_cov.astype(np.float64)
                     - c_new[covered].astype(np.float64))
        ci_recovery = _paired_boot_ci(diffs_cov, b_ids_cov,
                                      args_cli.n_boot,
                                      args_cli.boot_seed + tr)
        drift_pool_all = 100.0 * float(correct_old.sum()
                                       - c_new.sum()) / n_old
        drift_pool_cov = (100.0 * float(correct_old[covered].sum()
                                        - c_new[covered].sum()) / n_cov
                          if n_cov else float("nan"))
        rec_frac = (recovery_pp / drift_pool_cov
                    if (n_cov and abs(drift_pool_cov) > 1e-9)
                    else float("nan"))

        # G4 lambda curves on covered rows: old task + current task
        lam_vals = [float(x) for x in lam_grid]
        acc_old_by_lam = {str(l): None for l in lam_grid}
        acc_cur_by_lam = {str(l): None for l in lam_grid}
        cur_rows = np.nonzero(
            (new_rows["task"] == st_new["t_index"])
            & (new_rows["slot_major"] == slot_id))[0]
        for lam in lam_grid:
            lf = float(lam)
            if lf == 1.0:
                acc_old_by_lam[str(lam)] = round(
                    _acc(c_new[covered]), 2)
                acc_cur_by_lam[str(lam)] = round(
                    _acc(correct_new_all[cur_rows]), 2) \
                    if len(cur_rows) else float("nan")
            elif lf == 0.0:
                acc_old_by_lam[str(lam)] = round(
                    _acc(correct_swap_cov), 2)
                corr_cur_sw = _arm_correct_with_ys(
                    arm, "lg_swap", arm_rows, arm_ys, cur_rows)
                acc_cur_by_lam[str(lam)] = round(_acc(corr_cur_sw), 2) \
                    if len(cur_rows) else float("nan")
            else:
                corr_o = _arm_correct_with_ys(
                    arm, f"lg_lam_{lam}", arm_rows, arm_ys, cov_new)
                acc_old_by_lam[str(lam)] = round(_acc(corr_o), 2)
                corr_c = _arm_correct_with_ys(
                    arm, f"lg_lam_{lam}", arm_rows, arm_ys, cur_rows)
                acc_cur_by_lam[str(lam)] = round(_acc(corr_c), 2) \
                    if len(cur_rows) else float("nan")

        sp_old = _spearman(lam_vals,
                           [acc_old_by_lam[str(l)] for l in lam_grid])
        sp_cur = _spearman(lam_vals,
                           [acc_cur_by_lam[str(l)] for l in lam_grid])
        g4_records[tr] = {
            "acc_old_by_lambda": acc_old_by_lam,
            "acc_current_by_lambda": acc_cur_by_lam,
            "spearman_old_vs_lambda": round(sp_old, 3)
            if not math.isnan(sp_old) else None,
            "spearman_current_vs_lambda": round(sp_cur, 3)
            if not math.isnan(sp_cur) else None,
        }

        # G1: slot-6 prompt self-drift vs natural feature drift
        pv_old = st_old["pv"][min(st_old["pv"])]
        pv_new = st_new["pv"][min(st_new["pv"])]
        p_prev = pv_old[slot_id].astype(np.float64)
        p_now = pv_new[slot_id].astype(np.float64)
        dP = p_now - p_prev
        prompt_rel_norm = float(np.linalg.norm(dP)
                                / (np.linalg.norm(p_prev) + EPS))
        prompt_cos_drift = float(1.0 - (p_prev.ravel() @ p_now.ravel())
                                 / (np.linalg.norm(p_prev)
                                    * np.linalg.norm(p_now) + EPS))
        z_new_o = new_rows["z"][new_idx]
        z_old_o = old_rows["z"]
        d_rel = (np.linalg.norm(z_new_o - z_old_o, axis=1)
                 / (np.linalg.norm(z_old_o, axis=1) + EPS))
        cos_o = ((z_new_o * z_old_o).sum(1)
                 / (np.linalg.norm(z_new_o, axis=1)
                    * np.linalg.norm(z_old_o, axis=1) + EPS))
        d_cos = 1.0 - cos_o
        # annex: swap-arm feature recovery on covered old rows
        if len(arm_rows) and n_cov:
            pos = np.searchsorted(arm_rows, cov_new)
            z_swap_cov = np.asarray(arm["z_swap"])[pos] \
                if not isinstance(arm["z_swap"], np.ndarray) \
                else arm["z_swap"][pos]
            z_base = z_old_o[covered]
            d_rel_swap = (np.linalg.norm(z_swap_cov - z_base, axis=1)
                          / (np.linalg.norm(z_base, axis=1) + EPS))
        else:
            d_rel_swap = np.zeros((0,), np.float32)
        g1_records.append({
            "transition": tr,
            "prompt_rel_norm": round(prompt_rel_norm, 4),
            "prompt_cos_drift": round(prompt_cos_drift, 4),
            "med_d_rel": round(_median(d_rel), 4),
            "med_d_cos": round(_median(d_cos), 4),
            "med_d_rel_swap": round(_median(d_rel_swap), 4)
            if len(d_rel_swap) else None,
        })

        # G2: old-sensitive decomposition of dP (gradients from state s)
        g2rec = _g2_decompose(st_old, dP, d_err, args_cli)
        g2rec["transition"] = tr
        g2_records.append(g2rec)

        recoveries[tr] = {
            "recovery_pp": round(recovery_pp, 3),
            "ci95_recovery": ci_recovery,
            "recovery_frac_of_drift": round(rec_frac, 3)
            if not math.isnan(rec_frac) else None,
            "drift_pool_all_pp": round(drift_pool_all, 3),
            "drift_pool_covered_pp": round(drift_pool_cov, 3),
            "acc_nat_tm1": round(_acc(correct_old), 2),
            "acc_nat_t": round(_acc(c_new), 2),
            "acc_swap_t_cov": round(_acc(correct_swap_cov), 2),
            "n_old_rows": n_old, "n_covered_old_rows": n_cov,
        }
        summary["transitions"].append({
            "transition": tr, "t_from": st_old["t_index"],
            "t_to": st_new["t_index"],
            "n_rows": n_old, "n_covered_old_rows": n_cov,
            "drift_pool_pp": round(drift_pool_all, 3),
            **recoveries[tr],
            **{f"g1_{k}": v for k, v in g1_records[-1].items()
               if k != "transition"},
            **{f"g2_{k}": v for k, v in g2rec.items()
               if k != "transition"},
            **{f"g4_{k}": v for k, v in g4_records[tr].items()},
        })

        # aggregate rows: transition level
        agg_rows.append({
            "level": "transition", "transition": tr, "client": -1,
            "n_rows": n_old, "n_covered": n_cov,
            "acc_nat_tm1": recoveries[tr]["acc_nat_tm1"],
            "acc_nat_t": recoveries[tr]["acc_nat_t"],
            "drift_pool": round(drift_pool_all, 3),
            "acc_swap_t": recoveries[tr]["acc_swap_t_cov"],
            "recovery_pp": round(recovery_pp, 3),
            "ci_recovery_lo": ci_recovery[0] if ci_recovery else "",
            "ci_recovery_hi": ci_recovery[1] if ci_recovery else "",
            "recovery_frac_of_drift": recoveries[tr][
                "recovery_frac_of_drift"],
            "prompt_rel_norm": g1_records[-1]["prompt_rel_norm"],
            "prompt_cos_drift": g1_records[-1]["prompt_cos_drift"],
            "med_d_rel": g1_records[-1]["med_d_rel"],
            "med_d_cos": g1_records[-1]["med_d_cos"],
            "med_d_rel_swap": g1_records[-1]["med_d_rel_swap"],
            "g2_frac_old_sensitive": g2rec["frac_old_sensitive"],
            "g2_mean_pred_dce_nats": g2rec["mean_pred_dce_nats"],
            "g2_mean_actual_dce_nats": g2rec["mean_actual_dce_nats"],
            "g2_pearson_pred_vs_actual": g2rec["pearson_pred_vs_actual"],
            "spearman_old_vs_lambda":
                g4_records[tr]["spearman_old_vs_lambda"],
            "spearman_cur_vs_lambda":
                g4_records[tr]["spearman_current_vs_lambda"],
        })
        for lam in lam_grid:
            agg_rows[-1][f"acc_old_lam{lam}"] = \
                acc_old_by_lam[str(lam)]
            agg_rows[-1][f"acc_cur_lam{lam}"] = \
                acc_cur_by_lam[str(lam)]

        # per-client recovery detail (is the recovery one-client noise?)
        for cid in np.unique(old_rows["client"]):
            sel = old_rows["client"] == cid
            sel_cov = sel & covered
            n_sc = int(sel_cov.sum())
            if n_sc:
                pos = np.searchsorted(
                    arm_rows, new_idx[sel_cov])
                cs = _list_correct(_arm_vecs(arm, "lg_swap"),
                                   arm_ys)[pos]
                rp = 100.0 * float(cs.sum()
                                   - c_new[sel_cov].sum()) / n_sc
            else:
                rp = float("nan")
            agg_rows.append({
                "level": "client", "transition": tr, "client": int(cid),
                "n_rows": int(sel.sum()), "n_covered": n_sc,
                "acc_nat_tm1": round(_acc(correct_old[sel]), 2),
                "acc_nat_t": round(_acc(c_new[sel]), 2),
                "drift_pool": round(
                    100.0 * float(correct_old[sel].sum()
                                  - c_new[sel].sum()) / int(sel.sum()), 3),
                "acc_swap_t": round(_acc(cs), 2) if n_sc else "",
                "recovery_pp": round(rp, 3) if n_sc else "",
                "ci_recovery_lo": "", "ci_recovery_hi": "",
                "recovery_frac_of_drift": "",
                "prompt_rel_norm": "", "prompt_cos_drift": "",
                "med_d_rel": round(_median(d_rel[sel]), 4),
                "med_d_cos": "", "med_d_rel_swap": "",
                "g2_frac_old_sensitive": "",
                "g2_mean_pred_dce_nats": "",
                "g2_mean_actual_dce_nats": "",
                "g2_pearson_pred_vs_actual": "",
                "spearman_old_vs_lambda": "",
                "spearman_cur_vs_lambda": "",
            })
            for lam in lam_grid:
                agg_rows[-1][f"acc_old_lam{lam}"] = ""
                agg_rows[-1][f"acc_cur_lam{lam}"] = ""

        # per-row analysis CSV (old rows of this transition)
        cov_pos_map = {}
        if n_cov:
            cov_pos_map = dict(
                zip(new_idx[covered].tolist(),
                    range(int(n_cov))))
        swap_corr_all = (_list_correct(_arm_vecs(arm, "lg_swap"), arm_ys)
                         if len(arm_rows)
                         else np.zeros((0,), dtype=bool))
        lam_corr_all = {
            lam: (_list_correct(_arm_vecs(arm, f"lg_lam_{lam}"), arm_ys)
                  if len(arm_rows) else np.zeros((0,), dtype=bool))
            for lam in interior_lams}
        for ri in range(n_old):
            nri = int(new_idx[ri])
            is_cov = bool(covered[ri])
            pos = np.searchsorted(arm_rows, nri) if is_cov else 0
            in_arm = is_cov and len(arm_rows) and pos < len(arm_rows) \
                and arm_rows[pos] == nri
            cs = int(swap_corr_all[pos]) if in_arm else ""
            cl = {lam: (int(lam_corr_all[lam][pos]) if in_arm else "")
                  for lam in interior_lams}
            zsw_rel = ""
            if in_arm:
                zs = np.asarray(arm["z_swap"])[pos] \
                    if not isinstance(arm["z_swap"], np.ndarray) \
                    else arm["z_swap"][pos]
                zb = z_old_o[ri]
                zsw_rel = round(
                    float(np.linalg.norm(zs - zb)
                          / (np.linalg.norm(zb) + EPS)), 4)
            per_rows.append({
                "transition": tr,
                "client": int(old_rows["client"][ri]),
                "task": int(old_rows["task"][ri]),
                "rel_idx": int(old_rows["rel_idx"][ri]),
                "bseq": int(old_rows["bseq"][ri]),
                "slot_major": int(new_rows["slot_major"][nri]),
                "covered": int(is_cov),
                "y": int(old_rows["y"][ri]),
                "correct_tm1": int(correct_old[ri]),
                "correct_t": int(c_new[ri]),
                "correct_swap": cs,
                **{f"correct_lam{lam}": cl[lam]
                   for lam in interior_lams},
                "d_rel": round(float(d_rel[ri]), 4),
                "d_cos": round(float(d_cos[ri]), 4),
                "d_rel_swap": zsw_rel,
                "ce_tm1": round(float(ce_old[ri]), 4),
                "ce_t": round(float(ce_new_o[ri]), 4),
                "d_err": round(float(d_err[ri]), 4),
            })

    # ---- gates ----------------------------------------------------------
    premise_ok = all(
        s["batch_mode_slot"] == slot_id
        and s["nonslot_batch_frac"] <= args_cli.max_nonslot_batch_frac
        and s["prompt_values_identical_across_clients"]
        for s in summary["states"])

    g3_tr = int(args_cli.g3_transition)
    g3 = {"gate_pp": args_cli.g3_gate, "transition": g3_tr,
          "present": g3_tr in recoveries}
    g3_verdict = "N/A"
    if g3_tr in recoveries:
        rec = recoveries[g3_tr]
        ok_mag = rec["recovery_pp"] >= args_cli.g3_gate
        ok_ci = bool(rec["ci95_recovery"]
                     and rec["ci95_recovery"][0] > 0)
        if ok_mag and ok_ci:
            g3_verdict = "PASS"
        elif ok_mag:
            g3_verdict = "YELLOW_REVIEW"
        else:
            g3_verdict = "FAIL"
        g3.update({
            "recovery_pp": rec["recovery_pp"],
            "ci95_recovery": rec["ci95_recovery"],
            "recovery_frac_of_drift": rec["recovery_frac_of_drift"],
            "per_transition_recovery_pp": {
                str(k): v["recovery_pp"]
                for k, v in recoveries.items()},
        })
    g3["verdict"] = g3_verdict

    g4 = {"transition": g3_tr}
    g4_verdict = "N/A"
    if g3_tr in g4_records:
        rec = g4_records[g3_tr]
        old_curve = list(rec["acc_old_by_lambda"].values())
        cur_curve = list(rec["acc_current_by_lambda"].values())
        old_flat = (max(old_curve) - min(old_curve) < 0.05)
        cur_flat = (max(cur_curve) - min(cur_curve) < 0.05)
        sp_o = rec["spearman_old_vs_lambda"]
        sp_c = rec["spearman_current_vs_lambda"]
        if old_flat and cur_flat:
            g4_verdict = "FLAT"
        elif (sp_o is not None and sp_c is not None
              and sp_o < 0 and sp_c > 0):
            g4_verdict = "PASS"
        else:
            g4_verdict = "FAIL"
        g4.update({"verdict": g4_verdict,
                   "acc_old_by_lambda": rec["acc_old_by_lambda"],
                   "acc_current_by_lambda": rec["acc_current_by_lambda"],
                   "spearman_old_vs_lambda": sp_o,
                   "spearman_current_vs_lambda": sp_c})
    g4["verdict"] = g4_verdict

    # G1 reading: magnitude-vs-damage ordering across boundaries
    mag = [x["prompt_rel_norm"] for x in g1_records]
    dmg = [abs(x_["drift_pool_all_pp"])
           for x_ in [recoveries[k] for k in sorted(recoveries)]]
    g1_rho = _spearman(mag, dmg) if len(mag) >= 3 else float("nan")
    g1_verdict = ("magnitude-dominant (unexpected) -- escalate to "
                  "external review" if (not math.isnan(g1_rho)
                                       and g1_rho >= 0.8)
                  else "magnitude is not the dose (expected)")

    summary["verdicts"] = {
        "G0_premise": "OK" if premise_ok else "REFUSED",
        "G1": {"records": g1_records,
               "spearman_magnitude_vs_damage": round(g1_rho, 3)
               if not math.isnan(g1_rho) else None,
               "verdict": g1_verdict},
        "G2": {"records": g2_records,
               "note": "descriptive corroboration only (no hard gate); "
                       "expectation: frac_old_sensitive / predicted dCE "
                       "at 2->3 exceed 0->1"},
        "G3": g3,
        "G4": g4,
        "head_identity_clean": bool(head_clean),
        "hypothesis": ("same-slot prompt-content evolution causally owns "
                       "the boundary-dependent deployment accuracy drift "
                       "(concentrated at 2->3), carried by old-task-"
                       "sensitive directions rather than raw magnitude"),
    }

    # ---- FINAL (Sec.31.4 frozen tree) -----------------------------------
    if not premise_ok:
        final = ("REFUSED: G0 monopoly/deployment premise violated -- "
                 "no verdict (fail-closed).")
    elif g3_verdict == "N/A":
        final = ("INCOMPLETE_STATES: the pre-registered dissection "
                 f"transition {g3_tr} is not present in the extracted "
                 "state list -- no formal verdict (smoke runs only).")
    elif g3_verdict == "PASS":
        if g4_verdict in ("PASS", "FLAT"):
            final = (
                "CONTENT_MECHANISM_CONFIRMED: the same-slot prompt-swap "
                f"oracle recovers {g3['recovery_pp']}pp of the "
                f"{recoveries[g3_tr]['drift_pool_all_pp']}pp damage at "
                f"the {g3_tr}->{g3_tr + 1} transition (CI95 "
                f"{g3['ci95_recovery']}), with "
                + ("a monotone stability-plasticity lambda trajectory"
                   if g4_verdict == "PASS"
                   else "flat dose-response curves") +
                ". Same-slot prompt-content evolution is CAUSALLY "
                "responsible for the concentrated deployment damage and "
                "is promoted to a mechanism hypothesis (frozen wording: "
                "still NOT a method candidate). Stable-Plastic "
                "Factorized Prompt becomes the Sec.32 candidate ONLY "
                "after INR mechanism replication + external review; no "
                "training arm is pre-registered by this probe.")
        else:
            final = (
                "CONTENT_CONFIRMED_REVIEW: the swap oracle recovers the "
                f"damage ({g3['recovery_pp']}pp) but the lambda dose-"
                "response is not monotone -- escalate to external "
                "review before any Sec.32 candidacy.")
    elif g3_verdict == "YELLOW_REVIEW":
        final = (
            "G3_YELLOW_REVIEW: swap recovery reaches the gate "
            f"({g3['recovery_pp']}pp) but the batch-bootstrap 95% CI "
            f"({g3['ci95_recovery']}) does not exclude zero -- "
            "insufficient significance; escalate to external review.")
    else:
        final = (
            "CONTENT_NOT_INTERVENTION_SURFACE: prompt-content evolution "
            "explains representation-state change but the same-slot "
            f"swap recovers only {g3['recovery_pp']}pp at the "
            f"pre-registered dissection transition (gate "
            f"{args_cli.g3_gate}pp, CI95 {g3['ci95_recovery']}) -- it is "
            "not the recoverable-accuracy intervention surface; the "
            "structure path remains closed (no Sec.32 arm). Frozen "
            "wording: prompt-content evolution explains representation-"
            "state change but is not the intervention surface.")
    if not head_clean:
        final += (" WARNING: cross-state head drift detected (G3 "
                  "CONTAMINATED) -- see annex.head_drift_events.")
    summary["verdicts"]["FINAL"] = final

    summary["annex"]["native"] = _native_annex(states, run_dir, args_cli)
    return summary, agg_rows, per_rows


def _g2_decompose(st_old, dP, d_err, args_cli):
    """Old-sensitive decomposition of the slot-6 prompt update (Sec.31.3).

    Gradients g_i = dCE_i/dP_{6,s} were extracted at state s (st_old) on
    a seeded stratified subsample of old rows; dP = P_{6,s+1} - P_{6,s};
    d_err = actual cross-state per-row CE change over st_old's rows
    (canon-aligned). Returns frac_old_sensitive (projected energy
    fraction on the >=90% gradient-energy SVD subspace), the first-order
    predicted mean dCE, the actual mean dCE on the same subsample rows,
    and their Pearson correlation. All descriptive."""
    out = {"frac_old_sensitive": None, "subspace_rank": None,
           "mean_pred_dce_nats": None, "mean_actual_dce_nats": None,
           "pearson_pred_vs_actual": None, "n_subsample_rows": 0}
    g2 = st_old["g2"]
    if g2["n_rows"] == 0:
        out["note"] = "no G2 gradient subsample for this state"
        return out
    G = np.asarray(g2["g6"], dtype=np.float64)
    if G.ndim == 1:
        G = G.reshape(1, -1)
    dP_flat = np.asarray(dP, dtype=np.float64).reshape(-1)
    if G.shape[1] != dP_flat.shape[0]:
        raise RuntimeError(
            f"G2 gradient width {G.shape[1]} != prompt width "
            f"{dP_flat.shape[0]} -- pool shape mismatch")
    # subspace: SVD of the gradient ensemble, >= energy fraction, capped
    _, S, Vt = np.linalg.svd(G, full_matrices=False)
    energy = S ** 2
    total = float(energy.sum())
    if total <= 0:
        out["note"] = "zero gradient energy (subsample degenerate)"
        return out
    cum = np.cumsum(energy) / total
    k = int(np.searchsorted(cum, args_cli.g2_energy) + 1)
    k = min(k, args_cli.g2_max_rank, len(S))
    V = Vt[:k].T                       # (7680, k)
    proj = V @ (V.T @ dP_flat)
    frac = float((np.linalg.norm(proj) ** 2)
                 / (np.linalg.norm(dP_flat) ** 2 + EPS))
    # per-row first-order prediction vs actual cross-state dCE
    pred = G @ dP_flat
    key_to_old_row = {}
    for j in range(len(st_old["rows"]["client"])):
        key_to_old_row[(int(st_old["rows"]["client"][j]),
                        int(st_old["rows"]["task"][j]),
                        int(st_old["rows"]["rel_idx"][j]))] = j
    actual, preds = [], []
    for m in range(g2["n_rows"]):
        kj = (int(g2["client"][m]), int(g2["task"][m]),
              int(g2["rel_idx"][m]))
        if kj in key_to_old_row:
            actual.append(float(d_err[key_to_old_row[kj]]))
            preds.append(float(pred[m]))
    out.update({
        "frac_old_sensitive": round(frac, 4),
        "subspace_rank": k,
        "mean_pred_dce_nats": round(float(np.mean(preds)), 4)
        if preds else None,
        "mean_actual_dce_nats": round(float(np.mean(actual)), 4)
        if actual else None,
        "pearson_pred_vs_actual": round(_pearson(preds, actual), 4)
        if len(preds) >= 3 else None,
        "n_subsample_rows": len(preds),
        "note": "analysis-only autograd (Sec.31.5-6); descriptive",
    })
    return out


def _native_annex(states, run_dir, args_cli):
    """Informative annex (Sec.30.7-4 lineage): LOCATE the probe-vs-native
    gap via seeded native-mode draws (Chead Dropout ON + shuffle)."""
    annex = {"native_diff_max_pp": None, "native_diff_mean_pp": None,
             "n_compared": 0,
             "note": "probe = head.eval() + fixed batch order "
                     "(deterministic); native = Dropout ON + shuffle "
                     "(stochastic); draws decompose the gap"}
    run_dir = Path(run_dir)
    diffs = []
    per_key = []
    for st in states:
        rows = st["rows"]
        correct, _ = _state_correct_ce(st)
        keys = {}
        for i in range(len(rows["client"])):
            keys.setdefault((int(rows["client"][i]),
                             int(rows["task"][i])), []).append(correct[i])
        draws_map = st.get("native_draws", {}) or {}
        for (cid, task), corr in keys.items():
            csv_path = run_dir / "metrics" / f"client_{cid}_accuracy.csv"
            if not csv_path.is_file():
                continue
            native = None
            with open(csv_path, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if (int(row["after_task"]) == st["t_index"]
                            and int(row["test_task"]) == task):
                        native = float(row["accuracy"])
                        break
            if native is None:
                continue
            probe_acc = _acc(np.array(corr, dtype=bool))
            diffs.append(probe_acc - native)
            annex["n_compared"] += 1
            d = list(draws_map.get(f"{cid},{task}", []))
            rec = {"state": st["t_index"], "client": cid, "task": task,
                   "probe_acc": round(probe_acc, 2), "csv_acc": native,
                   "draws": d}
            if d:
                pool = d + [native]
                rec["draw_spread_pp"] = round(max(pool) - min(pool), 2)
                rec["eval_uplift_pp"] = round(
                    probe_acc - float(np.mean(d)), 2)
                rec["residual_pp"] = round(probe_acc - native, 2)
                rec["residual_within_draw_spread"] = bool(
                    abs(probe_acc - native) <= rec["draw_spread_pp"] + 1e-9)
            per_key.append(rec)
    if diffs:
        annex["native_diff_max_pp"] = round(
            float(np.max(np.abs(diffs))), 2)
        annex["native_diff_mean_pp"] = round(float(np.mean(diffs)), 2)
    with_draws = [r for r in per_key if "draw_spread_pp" in r]
    if with_draws:
        spreads = [r["draw_spread_pp"] for r in with_draws]
        uplifts = [r["eval_uplift_pp"] for r in with_draws]
        annex["n_with_draws"] = len(with_draws)
        annex["draw_spread_max_pp"] = round(float(np.max(spreads)), 2)
        annex["draw_spread_mean_pp"] = round(float(np.mean(spreads)), 2)
        annex["eval_uplift_mean_pp"] = round(float(np.mean(uplifts)), 2)
        annex["frac_residual_within_draw_spread"] = round(float(np.mean(
            [r["residual_within_draw_spread"] for r in with_draws])), 3)
        unexpl = [abs(r["residual_pp"]) - r["draw_spread_pp"]
                  for r in with_draws]
        annex["max_unexplained_pp"] = round(float(max(unexpl)), 2)
        worst = with_draws[int(np.argmax(unexpl))]
        annex["worst_key"] = (f"state={worst['state']} "
                              f"client={worst['client']} "
                              f"task={worst['task']}")
        annex["gap_location_conclusion"] = (
            "probe-vs-native residuals are within the native Dropout "
            "draw stochasticity (same checkpoint/head/test rows); the "
            "gap is located, not a semantics divergence"
            if annex["frac_residual_within_draw_spread"] >= 0.9
            and annex["max_unexplained_pp"] <= 0.2 else
            "residual NOT fully explained by native draw stochasticity "
            f"(max unexplained {annex['max_unexplained_pp']}pp at "
            f"{annex['worst_key']}) -- investigate before trusting gate "
            "readouts")
    else:
        annex["gap_location_conclusion"] = (
            "no native draws available (set --native_draws > 0 and "
            "re-extract) -- gap NOT located yet")
    if annex.get("native_diff_max_pp") is not None \
            and annex["native_diff_max_pp"] > args_cli.native_tol:
        annex["warning"] = (
            "probe-vs-native diff exceeds tolerance -- batch composition "
            "or head/dropout semantics divergence; investigate before "
            "trusting gate readouts")
    return annex


# ----------------------------------------------------------------------
# output writers + main
# ----------------------------------------------------------------------

AGG_FIELDS = ["level", "transition", "client", "n_rows", "n_covered",
              "acc_nat_tm1", "acc_nat_t", "drift_pool", "acc_swap_t",
              "recovery_pp", "ci_recovery_lo", "ci_recovery_hi",
              "recovery_frac_of_drift", "prompt_rel_norm",
              "prompt_cos_drift", "med_d_rel", "med_d_cos",
              "med_d_rel_swap", "g2_frac_old_sensitive",
              "g2_mean_pred_dce_nats", "g2_mean_actual_dce_nats",
              "g2_pearson_pred_vs_actual", "spearman_old_vs_lambda",
              "spearman_cur_vs_lambda"]

ROW_FIELDS = ["transition", "client", "task", "rel_idx", "bseq",
              "slot_major", "covered", "y", "correct_tm1", "correct_t",
              "correct_swap", "d_rel", "d_cos", "d_rel_swap",
              "ce_tm1", "ce_t", "d_err"]


def main():
    args_cli = parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    ckpts = [c.strip() for c in args_cli.checkpoints.split(",") if c.strip()]
    lam_grid = [x.strip() for x in args_cli.lambdas.split(",") if x.strip()]
    interior_lams = [x for x in lam_grid if 0.0 < float(x) < 1.0]

    states = []
    for t_index, ckpt in enumerate(ckpts):
        prev = states[-1] if states else None
        st = extract_state(args_cli, args_cli.run_dir, ckpt, prev,
                           interior_lams, t_index,
                           is_last_state=(t_index == len(ckpts) - 1))
        states.append(st)

    summary, agg_rows, per_rows = analyze(
        states, args_cli, lam_grid, interior_lams, args_cli.run_dir)

    for lam in interior_lams:
        fname = f"correct_lam{lam}"
        if fname not in ROW_FIELDS:
            ROW_FIELDS.append(fname)

    with open(args_cli.out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=AGG_FIELDS
                           + [f"acc_old_lam{l}" for l in lam_grid]
                           + [f"acc_cur_lam{l}" for l in lam_grid])
        w.writeheader()
        w.writerows(agg_rows)
    with open(args_cli.out_rows_csv, "w", encoding="utf-8",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=ROW_FIELDS)
        w.writeheader()
        w.writerows(per_rows)
    with open(args_cli.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    v = summary["verdicts"]
    trs = summary["transitions"]
    pv_ident = all(s["prompt_values_identical_across_clients"]
                   for s in summary["states"])
    print(f"{TAG} G0: premise {v['G0_premise']} (monopoly slot "
          f"{args_cli.monopoly_slot}; nonslot batch frac "
          f"{[s['nonslot_batch_frac'] for s in summary['states']]}; "
          f"pv identical {pv_ident}; heads clean "
          f"{v['head_identity_clean']})")
    print(f"{TAG} G1: prompt rel_norm per boundary "
          f"{[x['prompt_rel_norm'] for x in v['G1']['records']]}; "
          f"med D_rel {[x['med_d_rel'] for x in v['G1']['records']]} -> "
          f"{v['G1']['verdict']}")
    print(f"{TAG} G2: frac_old_sensitive per boundary "
          f"{[x['frac_old_sensitive'] for x in v['G2']['records']]}; "
          f"mean pred dCE "
          f"{[x['mean_pred_dce_nats'] for x in v['G2']['records']]} nats "
          f"vs actual "
          f"{[x['mean_actual_dce_nats'] for x in v['G2']['records']]}")
    per_tr_rec = [f"t{t['transition']}: {t['recovery_pp']}pp"
                  for t in trs]
    print(f"{TAG} G3 per-transition swap recovery: {per_tr_rec}")
    print(f"{TAG} G3 (main, transition {args_cli.g3_transition}): "
          f"recovery {v['G3'].get('recovery_pp')}pp (gate "
          f"{args_cli.g3_gate}), CI95 {v['G3'].get('ci95_recovery')}, "
          f"frac of drift {v['G3'].get('recovery_frac_of_drift')} -> "
          f"{v['G3']['verdict']}")
    g4 = v["G4"]
    print(f"{TAG} G4 (transition {args_cli.g3_transition}): old "
          f"acc(lam) {g4.get('acc_old_by_lambda')}, cur acc(lam) "
          f"{g4.get('acc_current_by_lambda')}, spearman old "
          f"{g4.get('spearman_old_vs_lambda')} cur "
          f"{g4.get('spearman_current_vs_lambda')} -> "
          f"{g4['verdict']}")
    print(f"{TAG} native gap location: "
          f"{summary['annex']['native'].get('gap_location_conclusion')}")
    print(f"{TAG} FINAL: {v['FINAL']}")
    print(f"{TAG} rows -> {args_cli.out_rows_csv}; aggregates -> "
          f"{args_cli.out_csv}; summary -> {args_cli.out_json}")


if __name__ == "__main__":
    main()
