"""Probe-F: routing-discontinuity mechanism identification (RUNCOMMANDS
Sec.30, offline only, pre-registered 2026-11-03).

Zero-training offline probe for the DTCPR (Dual-Timescale Consensus Prompt
Router) candidate. On the preserved run's adjacent deployment-state
checkpoints (task_XX_end.pth = post-broadcast states) it answers, in the
frozen F0-F5 ladder:

  F1  does the deployed top-k prompt route actually switch between
      adjacent states?  (FlipRate; per-state keyH is replicated for
      direct comparison with Sec.20's 0.004-bit monopoly prior)
  F2  do route flips co-occur with normalized feature jumps?
      (D_rel, D_cos on z_op under the native routing path)
  F3  do flips own a cross-state accuracy pool?
      F3a observational: E[dErr | flip] vs E[dErr | stable] (guards the
        "flip rows are just harder" confound);  F3b causal (MAIN GATE):
        Acc(forced old route at state t) - Acc(natural route at t) >= 1pp
  F4  are flips preceded by a routing-boundary margin?  (top-1 vs top-2
      similarity margin; batch vote margin) -- informative, no hard gate
  F5  does a structurally valid prompt transition repair the damage?
      Oracle-1: prediction-level lambda mix of the two deployment paths
        (diagnostic only -- dual-route complementary information);
      Oracle-2b: token-level lambda mix lam*P_t[R_t] + (1-lam)*
        P_{t-1}[R_{t-1}] (exactly what a DTCPR consensus gate computes;
        only Oracle-2b supports the DTCPR structure)

Route x contents 2x2 factorial (backbone frozen => exhaustive):
  A = (R_t,     P_t)   natural at t          (deployment truth)
  B = (R_{t-1}, P_t)   forced old route      (F3b causal arm)
  C = (R_t,     P_{t-1}) old contents        (contents channel)
  D = (R_{t-1}, P_{t-1}) = natural at t-1    (reused, zero extra forward)
  drift_pool = route_pool + content_pool + interaction_pool  (identity)

Pre-registered code facts (Sec.30.1) this probe is built on:
  top_k=1, batchwise_prompt=True, use_prompt_mask=False, pool_size=100;
  routing is a per-BATCH majority over per-sample top-1 keys; the query
  q(x) comes from the frozen original model (state-independent), so route
  flips are driven purely by prompt_key drift; z_op (head_type='prompt')
  is the mean of the routed prompt-token outputs through the frozen ViT.

Deviations from native evaluate() (pre-registered, Sec.30.1-5): fixed
batch order at batch_size 8 (pairing requirement), head.eval() (Chead
Dropout 0.5 OFF for determinism; absolute numbers sit ~1-2pp above the
native matrix -- every gate acts on cross-state / cross-arm differences,
so the bias cancels; the accuracy cross-check is an informative annex
with a 3pp tolerance).

Review round 2 hardening (2026-11-03, conditional approval):
  1. statistical unit = BATCH exposure: flip-vs-stable contrasts use
     paired BATCH bootstrap CIs (route is a batch-level quantity under
     batchwise_prompt); F4 is explicitly a batch routing-boundary
     precursor, not a per-sample one;
  2. deterministic identical batch partition across checkpoints is
     asserted via the canonical key join, plus a batch-composition
     sensitivity annex (alternative fixed partitions from cached prefs,
     zero GPU);
  3. scalar attribution is rigorous: accuracy-count pools obey the
     exact per-row additive identity; loss-level two-factor Shapley
     (phi_r + phi_p = L_D - L_A exactly) is reported alongside;
     feature-vector deltas are reported as magnitudes only, NEVER as
     percentage attribution (norms do not decompose);
  4. the probe-vs-native-CSV gap is LOCATED via seeded native-mode draws
     (Dropout ON + shuffle) per (state, client, task), decomposing it
     into dropout draw stochasticity / eval uplift / unexplained
     residual;
  5. F5 reports the deployment-wide net pool G_net = P(flip)*G_flip and
     requires G_flip>0 with G_stable=0-by-design (disagreement-gated
     gate) -- the A2-R death (-1.9pp from the always-on soft mixture)
     forbids reading a flip-subset oracle win as a net win.

fail-closed: use_prompt_mask=True or top_k!=1 or batchwise_prompt=False
-> refuse (probe semantics are only defined for this deployment path);
cache fingerprints (run/checkpoint/batch/probe version) must match;
row keys must align across states (the federated manifest guarantees it,
asserted anyway).

Usage (from repo root):

  python diagnostics/probe_routing_flip.py \
      --run_dir output/cifar100/fedta/cifar_a5v2_M2_feat_full/seed_42 \
      --data_path ./local_datasets --device cuda \
      --cache_dir diagnostics_output/routeflip_cifar \
      --out_csv diagnostics_output/routeflip_cifar.csv \
      --out_rows_csv diagnostics_output/routeflip_cifar_rows.csv \
      --out_json diagnostics_output/routeflip_cifar.summary.json
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

PROBE_VERSION = "routingflip-v3"
TAG = "[routeflip-probe]"
EPS = 1e-8


def parse_args():
    p = argparse.ArgumentParser("Probe-F routing-discontinuity probe")
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
                   help="independent cache dir (Sec.30 fail-closed lesson: "
                        "never share with other probes' caches)")
    p.add_argument("--out_csv", required=True)
    p.add_argument("--out_rows_csv", required=True)
    p.add_argument("--out_json", required=True)
    p.add_argument("--batch_size", default=8, type=int,
                   help="fixed evaluation batch (native evaluate uses 8; "
                        "fixed order is the pre-registered pairing protocol)")
    p.add_argument("--lambdas", default="0,0.25,0.5,0.75,1",
                   help="lambda grid for Oracle-1 / Oracle-2")
    p.add_argument("--flip_rate_gate", default=0.05, type=float,
                   help="F1 non-triviality floor: flip-batch fraction")
    p.add_argument("--f2_ratio_gate", default=1.2, type=float,
                   help="F2: flip median must exceed stable median by this "
                        "ratio on BOTH D_rel and D_cos")
    p.add_argument("--route_pool_gate", default=1.0, type=float,
                   help="F3b main gate: mean route_pool over transitions (pp)")
    p.add_argument("--route_pool_yellow", default=0.5, type=float,
                   help="F3b yellow floor when only a single transition "
                        "passes the gate")
    p.add_argument("--oracle_gain_gate", default=0.25, type=float,
                   help="F5: best interior lambda must beat both endpoints "
                        "by this margin (pp)")
    p.add_argument("--native_tol", default=3.0, type=float,
                   help="informative annex tolerance vs client CSVs (pp; "
                        "covers the pre-registered dropout-off uplift)")
    p.add_argument("--native_draws", default=2, type=int,
                   help="per (state, client, task): native-mode accuracy "
                        "draws (Chead Dropout ON + shuffle=True batch 8, "
                        "seeded) to LOCATE the probe-vs-client-CSV gap "
                        "(review round 2); 0 skips")
    p.add_argument("--n_boot", default=1000, type=int,
                   help="paired batch bootstrap replicates (route is a "
                        "BATCH-level quantity under batchwise_prompt; "
                        "significance of flip-vs-stable contrasts must "
                        "cluster by batch)")
    p.add_argument("--boot_seed", default=20261103, type=int)
    p.add_argument("--overwrite_cache", action="store_true")
    return p.parse_args()


# ----------------------------------------------------------------------
# helpers
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

    Replicates Global_Prompt.forward's idx branch exactly (official
    ``torch.cat([batched_prompt, x_embed], dim=1)`` + total_prompt_len),
    minus the debug print the official masked path carries. This is the
    forced-route (arm B/C) and token-mix (Oracle-2) forward. Must be an
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


def _row_correct_ce(vecs, ys):
    """Per-row (correct, cross-entropy) from a list of logit vectors."""
    n = len(ys)
    correct = np.empty(n, dtype=bool)
    ce = np.empty(n, dtype=np.float64)
    for i in range(n):
        v = vecs[i]
        mx = v.max()
        e = np.exp(v - mx)
        correct[i] = bool(np.argmax(v) == int(ys[i]))
        ce[i] = math.log(e.sum()) + mx - v[int(ys[i])]
    return correct, ce


def _acc(correct):
    correct = np.asarray(correct, dtype=bool)
    return 100.0 * float(correct.mean()) if len(correct) else float("nan")


def _median(x):
    x = np.asarray(x)
    return float(np.median(x)) if len(x) else float("nan")


def _point_biserial(x, y_bool):
    """corr between continuous x and binary y; nan-safe."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y_bool, dtype=np.float64)
    if len(x) < 2 or y.min() == y.max():
        return float("nan")
    xm = x - x.mean()
    ym = y - y.mean()
    denom = math.sqrt(float((xm ** 2).sum() * (ym ** 2).sum()))
    if denom <= 0:
        return float("nan")
    return float((xm * ym).sum() / denom)


def _acc_rows(vecs, ys):
    return _acc(_list_correct(vecs, ys))


def _vec_ce(vecs, ys):
    """Per-row cross-entropy from masked-logit vectors."""
    out = np.empty(len(ys), dtype=np.float64)
    for i in range(len(ys)):
        v = vecs[i]
        mx = v.max()
        out[i] = math.log(np.exp(v - mx).sum()) + mx - v[int(ys[i])]
    return out


def _batch_boot_ci(values, batch_ids, flip_flags, n_boot, seed, stat):
    """Paired BATCH bootstrap 95% CI for the flip-vs-stable contrast
    (review round 2, item 1: under batchwise_prompt the route is a
    batch-level quantity, so the resampling unit is the BATCH, not the
    sample; sample-level rows within a batch are perfectly correlated in
    their route exposure).

    stat: 'median' or 'mean'; returns (lo, hi, n_valid_reps) of
    stat(flip) - stat(stable), or None when one side has no batches.
    """
    values = np.asarray(values, dtype=np.float64)
    batch_ids = np.asarray(batch_ids)
    order = np.argsort(batch_ids, kind="stable")
    sorted_ids = batch_ids[order]
    bounds = np.nonzero(np.r_[True, sorted_ids[1:] != sorted_ids[:-1],
                              True])[0]
    groups = [order[bounds[i]:bounds[i + 1]]
              for i in range(len(bounds) - 1)]
    group_flip = np.array([bool(flip_flags[g[0]]) for g in groups])
    if group_flip.all() or not group_flip.any() or len(groups) < 2:
        return None
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        rows = np.concatenate([groups[i] for i in pick])
        fl = np.concatenate([np.full(len(groups[i]), group_flip[i])
                             for i in pick]).astype(bool)
        vf, vs = values[rows][fl], values[rows][~fl]
        if len(vf) == 0 or len(vs) == 0:
            continue
        if stat == "median":
            stats.append(float(np.median(vf) - np.median(vs)))
        else:
            stats.append(float(vf.mean() - vs.mean()))
    if len(stats) < max(10, n_boot // 10):
        return None
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return (round(float(lo), 4), round(float(hi), 4), len(stats))


def _alt_partition_majority_changes(pref_seq, batch_size):
    """Batch-composition sensitivity (review round 2, item 2): under the
    canon (rel_idx) order of one (client, task) population, build the base
    contiguous partition and circular-shift alternatives (shifts
    1..batch_size-1); count how many alternative partitions change any
    batch majority slot. Zero GPU: consumes only cached per-sample
    preferences."""
    n = len(pref_seq)
    if n == 0:
        return 0, 0

    def majorities(seq):
        ms = []
        for i in range(0, len(seq), batch_size):
            vals, cnts = np.unique(seq[i:i + batch_size],
                                   return_counts=True)
            ms.append(int(vals[int(np.argmax(cnts))]))
        return ms

    base = majorities(pref_seq)
    changed = 0
    for s in range(1, batch_size):
        alt = majorities(np.roll(pref_seq, s))
        changed += int(np.sum(np.array(alt) != np.array(base)))
    return changed, batch_size - 1


# ----------------------------------------------------------------------
# per-state extraction
# ----------------------------------------------------------------------

def _fingerprint(run_args, run_dir, ckpt_name, batch_size):
    return {
        "probe_version": PROBE_VERSION,
        "run_dir": str(Path(run_dir).resolve()),
        "data_name": run_args.data_name,
        "seed": run_args.seed,
        "checkpoint": ckpt_name,
        "batch_size": batch_size,
        "top_k": run_args.top_k,
        "use_prompt_mask": bool(run_args.use_prompt_mask),
        "pool_size": run_args.size,
    }


def extract_state(args_cli, run_dir, ckpt_name, prev_state, interior_lams,
                  t_index):
    """Extract one deployment state into row/batch/arm arrays.

    Caches to {cache_dir}/state_{ckpt_stem}.npz (+ .json meta) with
    fail-closed fingerprint checks; --overwrite_cache forces re-extraction."""
    cache_dir = Path(args_cli.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(ckpt_name).stem
    npz_path = cache_dir / f"state_{stem}.npz"
    meta_path = cache_dir / f"state_{stem}.json"

    run_args_probe = load_args_from_run_dir(run_dir)
    if bool(run_args_probe.use_prompt_mask):
        raise RuntimeError(
            "use_prompt_mask=True: deployment routing is the task-mask "
            "path and Probe-F semantics (top-k route flips) do not apply. "
            "Refusing (Sec.30 fail-closed).")
    if int(run_args_probe.top_k) != 1:
        raise RuntimeError(
            f"top_k={run_args_probe.top_k}: probe pre-registered for "
            "top_k=1 deployments only; generalization needs external "
            "review. Refusing (Sec.30 fail-closed).")
    if not bool(run_args_probe.batchwise_prompt):
        raise RuntimeError(
            "batchwise_prompt=False: routes are per-sample, not per-batch "
            "majority; the pre-registered pairing semantics do not apply. "
            "Refusing (Sec.30 fail-closed).")
    if args_cli.device == "cpu" or not torch.cuda.is_available():
        raise RuntimeError(
            "this probe requires --device cuda (Tail_Anchor.forward "
            "hardcodes .to('cuda'); official model semantics)")

    fp = _fingerprint(run_args_probe, run_dir, ckpt_name,
                      args_cli.batch_size)

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
    # as extract_features.py)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device(args_cli.device)

    server.model.to(device)
    server.origin_model.to(device)
    server.origin_model.eval()

    r = {k: [] for k in
         ("client", "task", "rel_idx", "bseq", "y", "y_local",
          "slot_major", "slot_pref", "s1", "s2", "margin", "vote_margin",
          "z", "logits_flat", "logits_ptr")}
    batch = {k: [] for k in
             ("client", "task", "bseq", "slot_major", "n",
              "vote_margin", "mean_margin", "med_margin")}
    arm = {"rows": [], "z_B": [], "lg_B": [], "z_C": [], "lg_C": []}
    for lam in interior_lams:
        arm[f"lg_2a_{lam}"] = []
        arm[f"lg_2b_{lam}"] = []

    class_masks = {}
    head_hashes = {}
    key_hashes = {}
    pv_cpu = {}
    native_draws = {}
    n_rows = 0
    flat_len = 0
    n_flip_batches = 0

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
                top2 = torch.topk(sim, k=2, dim=1).values
                s1 = top2[:, 0]
                s2 = top2[:, 1]
                margin = s1 - s2
                u, counts = torch.unique(pref, return_counts=True)
                cs = torch.sort(counts, descending=True).values
                vote_margin = float(cs[0] - (cs[1] if cs.numel() > 1
                                            else torch.tensor(0.0))) / bsz

                y_local = torch.searchsorted(mpos, tgt)
                with torch.no_grad():
                    logits = client.model(z, None)[0]
                    lg = logits.index_select(1, mpos).float()

                batch["client"].append(cid)
                batch["task"].append(t)
                batch["bseq"].append(bseq)
                batch["slot_major"].append(slot_major)
                batch["n"].append(bsz)
                batch["vote_margin"].append(vote_margin)
                batch["mean_margin"].append(float(margin.mean()))
                batch["med_margin"].append(float(margin.median()))

                row_base = n_rows
                for bi in range(bsz):
                    r["client"].append(cid)
                    r["task"].append(t)
                    r["rel_idx"].append(int(indices[n_done + bi]))
                    r["bseq"].append(bseq)
                    r["y"].append(int(tgt[bi]))
                    r["y_local"].append(int(y_local[bi]))
                    r["slot_major"].append(slot_major)
                    r["slot_pref"].append(int(pref[bi]))
                    r["s1"].append(float(s1[bi]))
                    r["s2"].append(float(s2[bi]))
                    r["margin"].append(float(margin[bi]))
                    r["vote_margin"].append(vote_margin)
                    r["z"].append(z[bi].cpu().numpy().astype(np.float32))
                    r["logits_flat"].append(
                        lg[bi].cpu().numpy().astype(np.float32))
                    r["logits_ptr"].append(flat_len)
                    flat_len += int(lg.shape[1])
                n_rows += bsz
                n_done += bsz

                # ---- flip arms (need the previous state's slots) -------
                flip = False
                m_prev = None
                if prev_state is not None:
                    prev_slot = prev_state["batch_slot"].get((cid, t, bseq))
                    if prev_slot is not None and prev_slot != slot_major:
                        flip = True
                        m_prev = prev_slot
                if not flip:
                    continue
                n_flip_batches += 1
                pv_prev = torch.tensor(prev_state["pv"][cid],
                                       dtype=torch.float32, device=device)

                # arm B: forced OLD route, current contents
                zB = _forward_with_prompt(client, inp, q, pv_now[m_prev])
                # arm C: current route, OLD contents
                zC = _forward_with_prompt(client, inp, q,
                                          pv_prev[slot_major])
                with torch.no_grad():
                    lgB = client.model(zB, None)[0].index_select(
                        1, mpos).float()
                    lgC = client.model(zC, None)[0].index_select(
                        1, mpos).float()
                arm["rows"].extend(range(row_base, row_base + bsz))
                for bi in range(bsz):
                    arm["z_B"].append(zB[bi].cpu().numpy())
                    arm["z_C"].append(zC[bi].cpu().numpy())
                    arm["lg_B"].append(lgB[bi].cpu().numpy())
                    arm["lg_C"].append(lgC[bi].cpu().numpy())
                for lam in interior_lams:
                    lam_f = float(lam)
                    tok2a = (lam_f * pv_now[slot_major]
                             + (1.0 - lam_f) * pv_now[m_prev])
                    tok2b = (lam_f * pv_now[slot_major]
                             + (1.0 - lam_f) * pv_prev[m_prev])
                    z2a = _forward_with_prompt(client, inp, q, tok2a)
                    z2b = _forward_with_prompt(client, inp, q, tok2b)
                    with torch.no_grad():
                        lg2a = client.model(z2a, None)[0].index_select(
                            1, mpos).float()
                        lg2b = client.model(z2b, None)[0].index_select(
                            1, mpos).float()
                    for bi in range(bsz):
                        arm[f"lg_2a_{lam}"].append(lg2a[bi].cpu().numpy())
                        arm[f"lg_2b_{lam}"].append(lg2b[bi].cpu().numpy())

            # ---- native-mode draws (annex: locate the probe-vs-CSV gap) --
            # Chead Dropout ON (fresh training-mode head copy, .eval() NOT
            # called -- the exact native evaluate() implementation-mode
            # semantics, Track E) + shuffle=True batch 8 with a seeded
            # generator: each draw is a reproducible realization of the
            # stochastic native readout. Contrasting draws with the
            # deterministic probe pass separates dropout stochasticity /
            # eval-mode uplift / batch composition (review round 2, item 4).
            if args_cli.native_draws > 0:
                draw_accs = []
                for d in range(args_cli.native_draws):
                    torch.manual_seed(
                        _seed(run_args.seed, t_index, cid, t, d))
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
                native_draws[f"{cid},{t}"] = [round(a, 2)
                                              for a in draw_accs]
                # restore the deterministic eval-mode head for later uses
                client.model.load_head(client.heads[t])
                client.model.eval()

    # ---- canonical row order (Sec.29.1c lesson) ------------------------
    # Row blocks are laid out per (client, task) in extraction order, so
    # the same logical sample lives at different absolute positions in
    # different states. Sort every state by (client, task, rel_idx) so
    # cross-state joins are clean; remap arm rows through the permutation.
    client_a = np.array(r["client"], dtype=np.int64)
    task_a = np.array(r["task"], dtype=np.int64)
    rel_a = np.array(r["rel_idx"], dtype=np.int64)
    n = len(client_a)
    order = np.lexsort((rel_a, task_a, client_a))
    inv = np.empty(n, dtype=np.int64)
    inv[order] = np.arange(n)
    for k in ("client", "task", "rel_idx", "bseq", "y", "y_local",
              "slot_major", "slot_pref", "s1", "s2", "margin",
              "vote_margin", "z"):
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
        for k in ("z_B", "z_C"):
            arm[k] = [arm[k][i] for i in srt]
        for k in [k for k in list(arm.keys()) if k.startswith("lg_")]:
            arm[k] = [arm[k][i] for i in srt]

    # ---- pack ----------------------------------------------------------
    # arm logits -> flat+ptr representation (per-row widths vary with the
    # (client, task) class mask; identical for fresh and cached states)
    for key in [k for k in list(arm.keys())
                if k not in ("rows", "z_B", "z_C")]:
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
    batch_slot = {}
    for i in range(len(batch["client"])):
        batch_slot[(int(batch["client"][i]), int(batch["task"][i]),
                    int(batch["bseq"][i]))] = int(batch["slot_major"][i])

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
            "s1": npa("s1"), "s2": npa("s2"), "margin": npa("margin"),
            "vote_margin": npa("vote_margin"),
            "z": z_arr, "logits_flat": flat, "logits_ptr": ptr,
        },
        "batch": {k: np.array(v, dtype=(
            np.int64 if k in ("client", "task", "bseq", "slot_major", "n")
            else np.float32)) for k, v in batch.items()},
        "batch_slot": batch_slot,
        "arms": arm,
        "masks": class_masks,
        "head_hashes": head_hashes,
        "key_hashes": key_hashes,
        "pv": pv_cpu,
        "native_draws": native_draws,
    }

    meta = {
        "fingerprint": fp,
        "checkpoint": ckpt_name,
        "t_index": t_index,
        "n_rows": int(n_rows),
        "n_flip_batches": int(n_flip_batches),
        "masks": class_masks,
        "head_hashes": head_hashes,
        "key_hashes": key_hashes,
        "native_draws": native_draws,
        "prompt_value_hashes": {str(c): _hash_array(v)
                                for c, v in pv_cpu.items()},
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
    print(f"{TAG} state {stem}: {n_rows} rows, "
          f"{len(batch['client'])} batches, {n_flip_batches} flip batches "
          f"({time.time() - t0:.0f}s)")
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
        arrays["arm_z_B"] = np.stack(arm["z_B"])
        arrays["arm_z_C"] = np.stack(arm["z_C"])
        for k, v in arm.items():
            if k.endswith("_flat") or k.endswith("_ptr"):
                arrays[f"arm_{k}"] = v
    # per-client prompt pools (needed by the NEXT state's arms)
    pvc = sorted(state["pv"])
    if pvc:
        arrays["pv_pool"] = np.stack([state["pv"][c] for c in pvc])
        arrays["pv_client"] = np.array(pvc, dtype=np.int64)
    np.savez_compressed(npz_path, **arrays)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def _load_state(npz_path, meta):
    data = np.load(npz_path)
    rows = {k: data[f"row_{k}"] for k in
            ("client", "task", "rel_idx", "bseq", "y", "y_local",
             "slot_major", "slot_pref", "s1", "s2", "margin",
             "vote_margin", "z", "logits_flat", "logits_ptr")}
    batch = {k: data[f"bt_{k}"] for k in
             ("client", "task", "bseq", "slot_major", "n", "vote_margin",
              "mean_margin", "med_margin")}
    batch_slot = {}
    for i in range(len(batch["client"])):
        batch_slot[(int(batch["client"][i]), int(batch["task"][i]),
                   int(batch["bseq"][i]))] = int(batch["slot_major"][i])
    arm = {"rows": [], "z_B": [], "z_C": []}
    if "arm_rows" in data:
        arm["rows"] = list(map(int, data["arm_rows"]))
        arm["z_B"] = data["arm_z_B"]
        arm["z_C"] = data["arm_z_C"]
        for k in data.files:
            if k.startswith("arm_lg") and (k.endswith("_flat")
                                           or k.endswith("_ptr")):
                arm[k[len("arm_"):]] = data[k]
    pv = {}
    if "pv_pool" in data:
        for i, c in enumerate(data["pv_client"]):
            pv[int(c)] = data["pv_pool"][i]
    return {
        "t_index": meta["t_index"],
        "ckpt": meta["checkpoint"],
        "rows": rows,
        "batch": batch,
        "batch_slot": batch_slot,
        "arms": arm,
        "masks": meta["masks"],
        "head_hashes": meta["head_hashes"],
        "key_hashes": meta["key_hashes"],
        "native_draws": meta.get("native_draws", {}),
        "pv": pv,
    }


# ----------------------------------------------------------------------
# analysis
# ----------------------------------------------------------------------

def _state_correct_ce(state):
    rows = state["rows"]
    vecs = _row_vecs(rows["logits_flat"], rows["logits_ptr"],
                     range(len(rows["client"])))
    return _row_correct_ce(vecs, rows["y_local"])


def analyze(states, args_cli, lam_grid, interior_lams, run_dir):
    summary = {"probe_version": PROBE_VERSION, "run_dir": str(run_dir),
               "gates": {
                   "flip_rate_gate": args_cli.flip_rate_gate,
                   "f2_ratio_gate": args_cli.f2_ratio_gate,
                   "route_pool_gate": args_cli.route_pool_gate,
                   "route_pool_yellow": args_cli.route_pool_yellow,
                   "oracle_gain_gate": args_cli.oracle_gain_gate,
                   "native_tol": args_cli.native_tol},
               "states": [], "transitions": [],
               "verdicts": {}, "annex": {}}

    # ---- per-state route stats (F1 material) ---------------------------
    for st in states:
        rows = st["rows"]
        vals, cnts = np.unique(rows["slot_pref"], return_counts=True)
        p = cnts.astype(np.float64) / cnts.sum()
        key_h = float(-(p * np.log2(p)).sum())
        bt = st["batch"]
        bmaj = bt["slot_major"] if len(bt["slot_major"]) \
            else np.zeros((0,), np.int64)
        bmode = int(np.bincount(bmaj).argmax()) if len(bmaj) else -1
        summary["states"].append({
            "t_index": st["t_index"], "ckpt": st["ckpt"],
            "n_rows": int(len(rows["client"])),
            "n_batches": int(len(bt["client"])),
            "keyH_bits": round(key_h, 4),
            "pref_mode_slot": int(vals[int(np.argmax(cnts))]),
            "pref_monopoly_share": round(float(cnts.max() / cnts.sum()), 4),
            "batch_mode_slot": bmode,
            "key_pool_hashes": st["key_hashes"],
            "key_pool_identical_across_clients":
                len(set(st["key_hashes"].values())) == 1,
        })

    # cross-state head-identity check (F3 "same head" premise)
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

    # ---- transitions ----------------------------------------------------
    agg_rows = []
    per_rows = []
    trans_flip_batches = []
    trans_flip_rows = []
    route_pools, drift_pools, content_pools = [], [], []
    f4_vm, f4_flip, f4_mm = [], [], []
    f5_lgA, f5_lgD, f5_y = [], [], []
    f5_correctB = []
    f5_2a, f5_2b = {}, {}
    rotation_events = []
    f1_pass = False
    # pooled accumulators for batch-clustered contrasts (review round 2)
    pool_b_ids, pool_flip = [], []
    pool_drel, pool_dcos, pool_derr, pool_n_old = [], [], [], []
    shapley_records = []
    feat_delta_records = []

    for i in range(1, len(states)):
        st_new, st_old = states[i], states[i - 1]
        tr = i
        new_rows, old_rows = st_new["rows"], st_old["rows"]
        n_old = int(len(old_rows["client"]))
        # canonical key join (both states canon-sorted by (client, task,
        # rel_idx)); every old key must exist in the new state's old-task
        # rows and the map must be order-preserving (fail-closed)
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

        correct_new, ce_new = _state_correct_ce(st_new)
        correct_old, ce_old = _state_correct_ce(st_old)
        c_new = correct_new[new_idx]
        ce_new_o = ce_new[new_idx]

        slot_new = new_rows["slot_major"][new_idx]
        slot_old = old_rows["slot_major"]
        flip = slot_new != slot_old
        flip_pref = (new_rows["slot_pref"][new_idx]
                     != old_rows["slot_pref"])
        n_flip = int(flip.sum())
        flip_idx = np.nonzero(flip)[0]
        flip_idx_new = new_idx[flip_idx]

        # F2 normalized feature jumps (natural arm, cross-state)
        z_new = new_rows["z"][new_idx]
        z_old = old_rows["z"]
        d_rel = (np.linalg.norm(z_new - z_old, axis=1)
                 / (np.linalg.norm(z_old, axis=1) + EPS))
        cos = ((z_new * z_old).sum(1)
               / (np.linalg.norm(z_new, axis=1)
                  * np.linalg.norm(z_old, axis=1) + EPS))
        d_cos = 1.0 - cos

        # F3a cross-state prediction change
        d_err = ce_new_o - ce_old
        derr_flip = float(d_err[flip].mean()) if n_flip else float("nan")
        n_stable = n_old - n_flip
        derr_stable = (float(d_err[~flip].mean()) if n_stable
                       else float("nan"))

        # paired BATCH bootstrap CIs (review round 2, item 1): route
        # exposure is batch-level under batchwise_prompt, so flip-vs-
        # stable contrasts must resample BATCHES, not samples
        b_ids = (old_rows["client"] * 10_000_000
                 + old_rows["task"] * 10_000 + old_rows["bseq"])
        bs = args_cli.boot_seed + tr
        ci_drel = _batch_boot_ci(d_rel, b_ids, flip, args_cli.n_boot,
                                 bs, "median")
        ci_dcos = _batch_boot_ci(d_cos, b_ids, flip, args_cli.n_boot,
                                 bs + 1, "median")
        ci_derr = _batch_boot_ci(d_err, b_ids, flip, args_cli.n_boot,
                                 bs + 2, "mean")
        # pooled accumulators (F2 gate uses the pooled contrast)
        p_ids = b_ids + tr * 1_000_000_000
        pool_b_ids.append(p_ids)
        pool_flip.append(flip)
        pool_drel.append(d_rel)
        pool_dcos.append(d_cos)
        pool_derr.append(d_err)
        pool_n_old.append(n_old)

        # F3b + 2x2 pools (arms live exactly on flip rows, in NEW-state
        # canonical row space)
        arm = st_new["arms"]
        arm_rows = (np.array(arm["rows"], dtype=np.int64)
                    if arm["rows"] else np.zeros((0,), np.int64))
        if n_flip and not np.array_equal(arm_rows, flip_idx_new):
            raise RuntimeError(
                f"arm rows do not match flip rows at transition "
                f"{tr - 1}->{tr} -- cache/alignment bug")
        correct_B = np.zeros((0,), dtype=bool)
        correct_C = np.zeros((0,), dtype=bool)
        if n_flip:
            y_arm = new_rows["y_local"][flip_idx_new]
            correct_B = _list_correct(_arm_vecs(arm, "lg_B"), y_arm)
            correct_C = _list_correct(_arm_vecs(arm, "lg_C"), y_arm)
            f5_correctB.append(correct_B)
            lgA = _row_vecs(new_rows["logits_flat"],
                            new_rows["logits_ptr"], flip_idx_new)
            lgD = _row_vecs(old_rows["logits_flat"],
                            old_rows["logits_ptr"], flip_idx)
            f5_lgA.extend(lgA)
            f5_lgD.extend(lgD)
            f5_y.extend(int(v) for v in y_arm)
            for lam in interior_lams:
                f5_2a.setdefault(lam, []).append(
                    _list_correct(_arm_vecs(arm, f"lg_2a_{lam}"), y_arm))
                f5_2b.setdefault(lam, []).append(
                    _list_correct(_arm_vecs(arm, f"lg_2b_{lam}"), y_arm))
            # two-factor loss Shapley + feature-vector delta magnitudes
            # (review round 2, item 3). The accuracy-count pools below
            # obey the exact per-row additive identity, so they stay the
            # primary scalar attribution; loss Shapley is the symmetric
            # check (phi_r + phi_p = L_D - L_A exactly); norms are
            # magnitudes, never percentages.
            ceA_f = _vec_ce(lgA, y_arm)
            ceB_f = _vec_ce(_arm_vecs(arm, "lg_B"), y_arm)
            ceC_f = _vec_ce(_arm_vecs(arm, "lg_C"), y_arm)
            ceD_f = _vec_ce(lgD, y_arm)
            LA, LB, LC, LD = (float(ceA_f.mean()), float(ceB_f.mean()),
                              float(ceC_f.mean()), float(ceD_f.mean()))
            phi_r = 0.5 * ((LB - LA) + (LD - LC))
            phi_p = 0.5 * ((LC - LA) + (LD - LB))
            phi_int = (LD - LA) - phi_r - phi_p
            shapley_records.append({
                "transition": tr, "n_flip_rows": n_flip,
                "L_A": round(LA, 4), "L_B": round(LB, 4),
                "L_C": round(LC, 4), "L_D": round(LD, 4),
                "phi_route_nats": round(phi_r, 4),
                "phi_content_nats": round(phi_p, 4),
                "phi_interaction_nats": round(phi_int, 4),
                "identity_check_nats": round(
                    phi_r + phi_p + phi_int - (LD - LA), 6),
            })
            zA_f = z_new[flip]
            zD_f = z_old[flip]
            zB_f = (arm["z_B"] if isinstance(arm["z_B"], np.ndarray)
                    else np.stack(arm["z_B"]))
            zC_f = (arm["z_C"] if isinstance(arm["z_C"], np.ndarray)
                    else np.stack(arm["z_C"]))
            d_r = zB_f - zA_f
            d_p = zC_f - zA_f
            d_int = zD_f - zB_f - zC_f + zA_f
            dDA = zD_f - zA_f
            feat_delta_records.append({
                "transition": tr,
                "mean_norm_delta_route": round(float(np.mean(
                    np.linalg.norm(d_r, axis=1))), 4),
                "mean_norm_delta_content": round(float(np.mean(
                    np.linalg.norm(d_p, axis=1))), 4),
                "mean_norm_delta_interaction": round(float(np.mean(
                    np.linalg.norm(d_int, axis=1))), 4),
                "mean_norm_total_drift": round(float(np.mean(
                    np.linalg.norm(dDA, axis=1))), 4),
                "note": "vector magnitudes ONLY -- the identity "
                        "D-A = dr + dp + dint holds for vectors, NOT for "
                        "norms; no percentage attribution from these "
                        "numbers",
            })

        drift_pool = 100.0 * (correct_old.sum() - c_new.sum()) / n_old
        if n_flip:
            route_pool = (100.0 * (correct_B.sum()
                                   - c_new[flip].sum()) / n_old)
            content_pool = (100.0 * (correct_C.sum()
                                     + correct_old[~flip].sum()
                                     - c_new.sum()) / n_old)
        else:
            route_pool = 0.0
            content_pool = drift_pool
        interaction_pool = drift_pool - route_pool - content_pool

        # F4: batch-level precursor (vote margin at t-1 -> flip at t)
        bt_old = st_old["batch"]
        vm = bt_old["vote_margin"]
        flip_batches = np.zeros(len(vm), dtype=bool)
        for j in range(len(vm)):
            key = (int(bt_old["client"][j]), int(bt_old["task"][j]),
                   int(bt_old["bseq"][j]))
            new_slot = st_new["batch_slot"].get(key)
            flip_batches[j] = (new_slot is not None
                               and new_slot != int(bt_old["slot_major"][j]))
        flip_rate = float(flip_batches.mean()) if len(vm) else 0.0
        f4_vm.append(vm)
        f4_flip.append(flip_batches)
        f4_mm.append(bt_old["mean_margin"])

        rotated = (summary["states"][i - 1]["batch_mode_slot"]
                   != summary["states"][i]["batch_mode_slot"])
        if rotated:
            rotation_events.append(
                f"t={i - 1}->{i}: batch majority slot "
                f"{summary['states'][i - 1]['batch_mode_slot']} -> "
                f"{summary['states'][i]['batch_mode_slot']}")
        if flip_rate >= args_cli.flip_rate_gate or rotated:
            f1_pass = True
        trans_flip_batches.append(int(flip_batches.sum()))
        trans_flip_rows.append(n_flip)
        route_pools.append(route_pool)
        drift_pools.append(drift_pool)
        content_pools.append(content_pool)

        trans_rec = {
            "transition": tr, "t_from": st_old["t_index"],
            "t_to": st_new["t_index"],
            "n_rows": n_old, "n_flip_rows": n_flip,
            "n_flip_batches": int(flip_batches.sum()),
            "n_batches": int(len(vm)),
            "flip_rate": round(flip_rate, 4),
            "majority_rotated": rotated,
            "acc_A_tm1": round(_acc(correct_old), 2),
            "acc_A_t": round(_acc(c_new), 2),
            "drift_pool_pp": round(drift_pool, 3),
            "route_pool_pp": round(route_pool, 3),
            "content_pool_pp": round(content_pool, 3),
            "interaction_pool_pp": round(interaction_pool, 3),
            "E_derr_flip": round(derr_flip, 4),
            "E_derr_stable": round(derr_stable, 4),
            "med_d_rel_flip": round(_median(d_rel[flip]), 4),
            "med_d_rel_stable": round(_median(d_rel[~flip]), 4),
            "med_d_cos_flip": round(_median(d_cos[flip]), 4),
            "med_d_cos_stable": round(_median(d_cos[~flip]), 4),
            "ci95_drel_flip_minus_stable": ci_drel,
            "ci95_dcos_flip_minus_stable": ci_dcos,
            "ci95_derr_flip_minus_stable": ci_derr,
        }
        summary["transitions"].append(trans_rec)
        agg_rows.append({
            "level": "transition", "transition": tr, "client": -1,
            "task": -1, "n_rows": n_old, "n_flip_rows": n_flip,
            "flip_frac": round(n_flip / n_old, 4) if n_old else 0.0,
            "n_flip_batches": int(flip_batches.sum()),
            "n_batches": int(len(vm)), "flip_rate": round(flip_rate, 4),
            "acc_A_tm1": trans_rec["acc_A_tm1"],
            "acc_A_t": trans_rec["acc_A_t"],
            "drift_pool": trans_rec["drift_pool_pp"],
            "route_pool": trans_rec["route_pool_pp"],
            "content_pool": trans_rec["content_pool_pp"],
            "interaction_pool": trans_rec["interaction_pool_pp"],
            "E_derr_flip": trans_rec["E_derr_flip"],
            "E_derr_stable": trans_rec["E_derr_stable"],
            "med_d_rel_flip": trans_rec["med_d_rel_flip"],
            "med_d_rel_stable": trans_rec["med_d_rel_stable"],
            "med_d_cos_flip": trans_rec["med_d_cos_flip"],
            "med_d_cos_stable": trans_rec["med_d_cos_stable"],
            "ci_drel_lo": trans_rec["ci95_drel_flip_minus_stable"][0]
            if trans_rec["ci95_drel_flip_minus_stable"] else "",
            "ci_drel_hi": trans_rec["ci95_drel_flip_minus_stable"][1]
            if trans_rec["ci95_drel_flip_minus_stable"] else "",
            "ci_dcos_lo": trans_rec["ci95_dcos_flip_minus_stable"][0]
            if trans_rec["ci95_dcos_flip_minus_stable"] else "",
            "ci_dcos_hi": trans_rec["ci95_dcos_flip_minus_stable"][1]
            if trans_rec["ci95_dcos_flip_minus_stable"] else "",
            "ci_derr_lo": trans_rec["ci95_derr_flip_minus_stable"][0]
            if trans_rec["ci95_derr_flip_minus_stable"] else "",
            "ci_derr_hi": trans_rec["ci95_derr_flip_minus_stable"][1]
            if trans_rec["ci95_derr_flip_minus_stable"] else "",
        })

        # per-client detail rows (robustness: is the pool one-client noise?)
        for cid in np.unique(old_rows["client"]):
            sel = old_rows["client"] == cid
            n_c = int(sel.sum())
            f_c = sel & flip
            n_f_c = int(f_c.sum())
            if n_f_c:
                arm_pos = np.searchsorted(flip_idx,
                                          np.nonzero(f_c)[0])
                cB = correct_B[arm_pos]
                cC = correct_C[arm_pos]
                rp = (100.0 * (cB.sum() - c_new[f_c].sum()) / n_c)
                cp = (100.0 * (cC.sum() + correct_old[sel & ~flip].sum()
                               - c_new[sel].sum()) / n_c)
            else:
                rp = 0.0
                cp = 100.0 * (correct_old[sel].sum()
                              - c_new[sel].sum()) / n_c if n_c else 0.0
            agg_rows.append({
                "level": "client", "transition": tr, "client": int(cid),
                "task": -1, "n_rows": n_c, "n_flip_rows": n_f_c,
                "flip_frac": round(n_f_c / n_c, 4) if n_c else 0.0,
                "n_flip_batches": "", "n_batches": "",
                "flip_rate": "",
                "acc_A_tm1": round(_acc(correct_old[sel]), 2),
                "acc_A_t": round(_acc(c_new[sel]), 2),
                "drift_pool": round(100.0 * (correct_old[sel].sum()
                                             - c_new[sel].sum()) / n_c, 3)
                if n_c else 0.0,
                "route_pool": round(rp, 3),
                "content_pool": round(cp, 3),
                "interaction_pool": round(
                    (100.0 * (correct_old[sel].sum() - c_new[sel].sum())
                     / n_c if n_c else 0.0) - rp - cp, 3),
                "E_derr_flip": round(float(d_err[f_c].mean()), 4)
                if n_f_c else "",
                "E_derr_stable": round(float(d_err[sel & ~flip].mean()), 4)
                if (n_c - n_f_c) else "",
                "med_d_rel_flip": round(_median(d_rel[f_c]), 4)
                if n_f_c else "",
                "med_d_rel_stable": round(_median(d_rel[sel & ~flip]), 4)
                if (n_c - n_f_c) else "",
                "med_d_cos_flip": round(_median(d_cos[f_c]), 4)
                if n_f_c else "",
                "med_d_cos_stable": round(_median(d_cos[sel & ~flip]), 4)
                if (n_c - n_f_c) else "",
                "ci_drel_lo": "", "ci_drel_hi": "",
                "ci_dcos_lo": "", "ci_dcos_hi": "",
                "ci_derr_lo": "", "ci_derr_hi": "",
            })

        # per-row analysis CSV
        arm_pos_of_row = {}
        if n_flip:
            for pos, ri in enumerate(flip_idx):
                arm_pos_of_row[int(ri)] = pos
        for ri in range(n_old):
            is_flip = bool(flip[ri])
            cb = cc = ""
            if is_flip:
                pos = arm_pos_of_row[ri]
                cb = int(correct_B[pos])
                cc = int(correct_C[pos])
            per_rows.append({
                "transition": tr,
                "client": int(new_rows["client"][ri]),
                "task": int(new_rows["task"][ri]),
                "rel_idx": int(new_rows["rel_idx"][ri]),
                "bseq": int(new_rows["bseq"][ri]),
                "slot_major_tm1": int(slot_old[ri]),
                "slot_major_t": int(slot_new[ri]),
                "flip": int(is_flip),
                "flip_pref": int(flip_pref[ri]),
                "margin_tm1": float(old_rows["margin"][ri]),
                "vote_margin_tm1": float(old_rows["vote_margin"][ri]),
                "d_rel": round(float(d_rel[ri]), 4),
                "d_cos": round(float(d_cos[ri]), 4),
                "ce_tm1": round(float(ce_old[ri]), 4),
                "ce_t": round(float(ce_new_o[ri]), 4),
                "d_err": round(float(d_err[ri]), 4),
                "correct_tm1": int(correct_old[ri]),
                "correct_t": int(c_new[ri]),
                "correct_B": cb, "correct_C": cc,
            })

    # ---- gates ----------------------------------------------------------
    f1_flip_rates = [t["flip_rate"] for t in summary["transitions"]]
    f1_verdict = "PASS" if f1_pass else "DEAD"

    any_flip = sum(trans_flip_rows) > 0
    f2_verdict = "N/A"
    f2_detail = {}
    # pooled batch-clustered contrast inputs (review round 2, item 1)
    pooled_drel = (np.concatenate(pool_drel) if pool_drel
                   else np.zeros((0,)))
    pooled_dcos = (np.concatenate(pool_dcos) if pool_dcos
                   else np.zeros((0,)))
    pooled_derr = (np.concatenate(pool_derr) if pool_derr
                   else np.zeros((0,)))
    pooled_bids = (np.concatenate(pool_b_ids) if pool_b_ids
                   else np.zeros((0,), dtype=np.int64))
    pooled_flip = (np.concatenate(pool_flip) if pool_flip
                   else np.zeros((0,), dtype=bool))
    ci_derr_pooled = None
    if any_flip:
        ok = True
        for metric, vals in (("d_rel", pooled_drel),
                             ("d_cos", pooled_dcos)):
            fv = vals[pooled_flip]
            sv = vals[~pooled_flip]
            fm = float(np.median(fv)) if len(fv) else float("nan")
            sm = float(np.median(sv)) if len(sv) else float("nan")
            ci = _batch_boot_ci(vals, pooled_bids, pooled_flip,
                                args_cli.n_boot, args_cli.boot_seed + 900,
                                "median")
            f2_detail[metric] = {"flip_median": round(fm, 4),
                                 "stable_median": round(sm, 4),
                                 "ci95_flip_minus_stable": ci}
            ratio_ok = (fm >= args_cli.f2_ratio_gate * sm)
            ci_ok = bool(ci and ci[0] > 0)
            if not (ratio_ok and ci_ok):
                ok = False
        ci_derr_pooled = _batch_boot_ci(pooled_derr, pooled_bids,
                                        pooled_flip, args_cli.n_boot,
                                        args_cli.boot_seed + 903, "mean")
        f2_detail["d_err"] = {"ci95_flip_minus_stable": ci_derr_pooled}
        f2_verdict = "PASS" if ok else "FAIL"

    f3a_detail = {
        "E_derr_flip": [t["E_derr_flip"] for t in summary["transitions"]],
        "E_derr_stable": [t["E_derr_stable"] for t in
                          summary["transitions"]],
        "ci95_derr_flip_minus_stable_pooled": ci_derr_pooled,
        "note": "corroboration only; guards the 'flip rows are harder' "
                "confound (cross-state change, not absolute difficulty); "
                "CI is batch-clustered",
    }

    mean_route_pool = float(np.mean(route_pools)) if route_pools else 0.0
    max_route_pool = float(np.max(route_pools)) if route_pools else 0.0
    if mean_route_pool >= args_cli.route_pool_gate:
        f3b_verdict = "PASS"
    elif (max_route_pool >= args_cli.route_pool_gate
          and mean_route_pool >= args_cli.route_pool_yellow):
        f3b_verdict = "YELLOW"
    else:
        f3b_verdict = "FAIL"

    vm_all = np.concatenate(f4_vm) if f4_vm else np.zeros((0,))
    fl_all = (np.concatenate(f4_flip) if f4_flip
              else np.zeros((0,), dtype=bool))
    mm_all = np.concatenate(f4_mm) if f4_mm else np.zeros((0,))
    # F4 is a BATCH routing-boundary precursor (review round 2, item 1):
    # m_{t-1,b} down => P(R_{t,b} != R_{t-1,b}) up -- batch-level margins
    # predict batch-level flips; per-sample margins are within-batch
    # context only, never the precursor unit.
    f4_detail = {"unit": "batch routing-boundary precursor",
                 "n_batches": int(len(vm_all)),
                 "point_biserial_vote_margin_vs_flip":
                     round(_point_biserial(vm_all, fl_all), 4)
                     if len(vm_all) else None,
                 "point_biserial_mean_margin_vs_flip":
                     round(_point_biserial(mm_all, fl_all), 4)
                     if len(mm_all) else None}
    if len(vm_all) >= 10 and len(np.unique(fl_all)) > 1:
        qs = np.quantile(vm_all, [0.2, 0.4, 0.6, 0.8])
        bins = np.digitize(vm_all, qs)
        f4_detail["flip_rate_by_vote_margin_quintile"] = [
            round(float(fl_all[bins == q].mean()), 4) for q in range(5)]

    # F5 two-level oracle (pooled flip rows)
    f5_detail = {"n_flip_rows_pooled": int(len(f5_y)),
                 "oracle1_curve": {}, "oracle2a_curve": {},
                 "oracle2b_curve": {}}
    o1_gain = None
    o2_gain = None
    if f5_y:
        yy = np.array(f5_y, dtype=np.int64)
        accA = _acc_rows(f5_lgA, yy)
        accD = _acc_rows(f5_lgD, yy)
        f5_detail["oracle1_endpoints"] = {"acc_A_t": round(accA, 2),
                                          "acc_D_tm1": round(accD, 2)}
        best_o1 = -1e9
        for lam in lam_grid:
            lam_f = float(lam)
            mixed = [lam_f * a + (1.0 - lam_f) * d
                     for a, d in zip(f5_lgA, f5_lgD)]
            a = _acc_rows(mixed, yy)
            f5_detail["oracle1_curve"][str(lam)] = round(a, 2)
            if 0.0 < lam_f < 1.0:
                best_o1 = max(best_o1, a - max(accA, accD))
        o1_gain = round(best_o1, 3)
        f5_detail["oracle1_best_interior_gain_pp"] = o1_gain

        accB = _acc(np.concatenate(f5_correctB)) if f5_correctB else None
        if accB is not None:
            f5_detail["oracle2a_endpoints"] = {
                "acc_A_t(lam=1)": round(accA, 2),
                "acc_B(lam=0)": round(accB, 2)}
        best_2b = -1e9
        for lam in interior_lams:
            if f5_2b.get(lam):
                a2b = _acc(np.concatenate(f5_2b[lam]))
                f5_detail["oracle2b_curve"][str(lam)] = round(a2b, 2)
                best_2b = max(best_2b, a2b - max(accA, accD))
            if f5_2a.get(lam):
                a2a = _acc(np.concatenate(f5_2a[lam]))
                f5_detail["oracle2a_curve"][str(lam)] = round(a2a, 2)
        o2_gain = round(best_2b, 3) if f5_2b else None
        f5_detail["oracle2b_endpoints"] = {
            "acc_A_t(lam=1)": round(accA, 2),
            "acc_D_tm1(lam=0)": round(accD, 2)}
        if best_2b > -1e8:
            f5_detail["oracle2b_best_interior_gain_pp"] = o2_gain
        # net-gain accounting (review round 2, A2-R precedent): a deployable
        # consensus gate is DISAGREEMENT-GATED, so it never fires on stable
        # batches => G_stable = 0 BY DESIGN (a Sec.31 design constraint,
        # not a measurement). The deployment-wide expectation is
        # G_net = P(flip) * G_flip; report it explicitly -- A2-R died of
        # the always-on soft mixture itself (-1.9pp), so an Oracle-2b pass
        # on a tiny flip population must NOT be read as a net win.
        n_old_pooled = int(np.sum(pool_n_old)) if pool_n_old else 0
        flip_frac_pooled = (int(len(f5_y)) / n_old_pooled
                            if n_old_pooled else 0.0)
        net_pool_pp = (flip_frac_pooled * (o2_gain if o2_gain is not None
                                           else 0.0))
        f5_detail["flip_fraction_pooled"] = round(flip_frac_pooled, 4)
        f5_detail["net_pool_pp"] = round(net_pool_pp, 3)
        f5_detail["g_stable_note"] = (
            "0 by design for a disagreement-gated transition gate; the "
            "A2-R -1.9pp always-on soft-mixture death means the Sec.31 "
            "arm must PROVE the gate never fires on stable batches")

    o1_pass = bool(o1_gain is not None and o1_gain > 0.0)
    o2_pass = bool(o2_gain is not None
                   and o2_gain >= args_cli.oracle_gain_gate)
    f5_verdict = "N/A" if not f5_y else ("PASS" if o2_pass else "FAIL")
    if o2_pass and f5_detail.get("net_pool_pp") is not None \
            and f5_detail["net_pool_pp"] < args_cli.oracle_gain_gate:
        f5_verdict = "PASS_NET_REVIEW"
        f5_detail["net_review_note"] = (
            "G_flip clears the gate but the deployment-wide net pool "
            f"({f5_detail['net_pool_pp']}pp) does not clear the same "
            "engineering floor -- escalate to external review before "
            "writing the Sec.31 arm")

    verdicts = {
        "F0": ("A2-R precedent stands (Sec.20.6): routing metrics were "
               "decoupled from accuracy; F1 is an explicit death gate "
               "and the F3 >=1pp gate is not relaxed"),
        "F1": f1_verdict,
        "F1_flip_rates": f1_flip_rates,
        "F1_rotation_events": rotation_events,
        "F2": f2_verdict, "F2_detail": f2_detail,
        "F3a": f3a_detail,
        "F3b": f3b_verdict,
        "F3b_mean_route_pool_pp": round(mean_route_pool, 3),
        "F3b_per_transition_pp": [round(x, 3) for x in route_pools],
        "F4": f4_detail,
        "F5_oracle1_complementary": o1_pass,
        "F5_oracle2b": f5_verdict,
        "F5_detail": f5_detail,
        "head_identity_clean": bool(head_clean),
        "content_pool_mean_pp": round(float(np.mean(content_pools)), 3)
        if content_pools else None,
        "drift_pool_mean_pp": round(float(np.mean(drift_pools)), 3)
        if drift_pools else None,
        "loss_shapley_per_transition": shapley_records,
        "feature_delta_magnitudes_per_transition": feat_delta_records,
        "priority_ladder": ("Route exists? -> Route changes temporally? "
                            "-> Route explains drift? -> Route owns "
                            "accuracy pool? (review round 2): if the "
                            "first two steps die, do NOT rescue DTCPR -- "
                            "read the 2x2 contents channel instead"),
    }
    summary["verdicts"] = verdicts

    # ---- FINAL (Sec.30.4 frozen tree, review round 2 wording) -----------
    if not f1_pass:
        final = ("HYPOTHESIS_REJECTED: the discrete temporal "
                 "route-switching hypothesis is REJECTED (flip rates "
                 f"{f1_flip_rates}; no majority-slot rotation; keyH per "
                 "state "
                 f"{[s['keyH_bits'] for s in summary['states']]} bits). "
                 "This rejects OUR OWN mechanism hypothesis, it is not a "
                 "probe failure. DTCPR is dead -- no training arm: "
                 "fast/slow dual routing would stay identical on the "
                 "deployment path and the consensus gate never fires "
                 "(structurally inert). The 2x2 factorial attributes the "
                 "cross-state drift pool to the CONTENTS channel (content "
                 f"{verdicts['content_pool_mean_pp']}pp vs route "
                 f"{round(mean_route_pool, 3)}pp) -- the representation-"
                 "side narrative moves to prompt-content evolution "
                 "(recorded; NOT auto-promoted to a candidate).")
    elif f3b_verdict == "FAIL":
        final = ("DTCPR_DEAD_F3: flips exist but own no accuracy pool "
                 f"(mean route_pool {round(mean_route_pool, 3)}pp < "
                 f"{args_cli.route_pool_gate}pp).")
    elif f3b_verdict == "YELLOW":
        final = ("DTCPR_YELLOW_REVIEW: route pool isolated in a single "
                 "transition and below the mean gate -- escalate to "
                 "external review with the per-transition numbers.")
    elif f5_verdict == "PASS_NET_REVIEW":
        final = ("DTCPR_SUPPORTED_CONDITIONAL: F3b route pool and Oracle-"
                 "2b both pass on the flip population, but the "
                 "deployment-wide net pool P(flip)*G_flip = "
                 f"{f5_detail['net_pool_pp']}pp does not clear the "
                 f"{args_cli.oracle_gain_gate}pp engineering floor -- "
                 "escalate to external review before writing the Sec.31 "
                 "arm (A2-R precedent: the always-on soft mixture itself "
                 "cost -1.9pp; the gate must be disagreement-gated).")
    elif not o2_pass:
        if o1_pass:
            final = ("DTCPR_STRUCTURE_DOUBT: Oracle-1 shows dual-path "
                     "complementary information, but the structurally "
                     "valid prompt transition (Oracle-2b) does not beat "
                     "the hard switch by the gate -- change structure, "
                     "do not ship prompt interpolation.")
        else:
            final = ("DTCPR_DEAD_F5: neither oracle level beats the hard "
                     "switch; no design basis for DTCPR.")
    else:
        final = ("DTCPR_SUPPORTED: F1 flips non-trivial, F3b route pool "
                 f">= {args_cli.route_pool_gate}pp, Oracle-2b beats the "
                 "hard switch on the flip population AND the net pool "
                 f"({f5_detail.get('net_pool_pp')}pp) clears the "
                 "engineering floor -- write the Sec.31 training-arm "
                 "pre-registration.")
    if not head_clean:
        final += (" WARNING: cross-state head drift detected (F3 "
                  "CONTAMINATED) -- see annex.head_drift_events.")
    summary["verdicts"]["FINAL"] = final

    # ---- batch-composition sensitivity annex (review round 2, item 2) --
    # Same checkpoints, ALTERNATIVE fixed batch partitions (circular
    # shifts of the canon order): if merely regrouping the same samples
    # changes batch majorities, the temporal "route" is not a stable
    # population statistic and the flip analysis itself is confounded.
    bcs = []
    for st in states:
        rows = st["rows"]
        key = rows["client"] * 100 + rows["task"]
        pref = rows["slot_pref"]
        starts = np.nonzero(np.r_[True, key[1:] != key[:-1]])[0]
        bounds = np.append(starts, len(key))
        total_changed = 0
        n_ct = len(starts)
        n_ct_changed = 0
        for bi in range(len(starts)):
            seq = pref[bounds[bi]:bounds[bi + 1]]
            changed, _ = _alt_partition_majority_changes(
                seq, args_cli.batch_size)
            total_changed += changed
            if changed:
                n_ct_changed += 1
        bcs.append({
            "t_index": st["t_index"], "ckpt": st["ckpt"],
            "n_client_task_groups": int(n_ct),
            "n_alt_partition_batches_changed_majority":
                int(total_changed),
            "n_client_task_groups_with_any_change": int(n_ct_changed),
        })
    any_change = any(x["n_alt_partition_batches_changed_majority"]
                     for x in bcs)
    summary["annex"]["batch_composition_sensitivity"] = {
        "per_state": bcs,
        "verdict": (
            "routing is BATCH-COMPOSITION SENSITIVE -- the temporal "
            "route is not a stable population statistic; DTCPR dies "
            "harder" if any_change else
            "routes composition-insensitive under all alternative fixed "
            "partitions -- the batch majority is a stable population "
            "statistic on this run"),
        "note": "alternative partitions = circular shifts 1..B-1 of the "
                "canon (rel_idx) order per (client, task); zero GPU, "
                "computed from cached per-sample preferences",
    }

    # ---- informative annex: probe acc(A) vs client CSVs ----------------
    summary["annex"].update(_native_annex(states, run_dir, args_cli))
    return summary, agg_rows, per_rows


def _native_annex(states, run_dir, args_cli):
    """Informative annex (review round 2, item 4): LOCATE the probe-vs-
    native gap instead of waving it off. probe acc(A) is deterministic
    (head.eval(), fixed batch order); the native readout is stochastic
    (Chead Dropout ON, shuffle=True). With --native_draws > 0 each
    (state, client, task) carries seeded native-mode draws, so the gap
    decomposes into (i) native draw stochasticity (dropout), (ii)
    eval-mode uplift (probe - mean draw), (iii) any unexplained residual.
    --native_tol is the informative tolerance."""
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
            f"{annex['worst_key']}) -- investigate before trusting F3 "
            "verdicts")
    else:
        annex["gap_location_conclusion"] = (
            "no native draws available (set --native_draws > 0 and "
            "re-extract) -- gap NOT located yet")
    if annex.get("native_diff_max_pp") is not None \
            and annex["native_diff_max_pp"] > args_cli.native_tol:
        annex["warning"] = (
            "probe-vs-native diff exceeds tolerance -- batch "
            "composition sensitivity or head/dropout semantics "
            "divergence; investigate before trusting gate readouts")
    return annex


# ----------------------------------------------------------------------
# output writers + main
# ----------------------------------------------------------------------

AGG_FIELDS = ["level", "transition", "client", "task", "n_rows",
              "n_flip_rows", "flip_frac", "n_flip_batches", "n_batches",
              "flip_rate", "acc_A_tm1", "acc_A_t", "drift_pool",
              "route_pool", "content_pool", "interaction_pool",
              "E_derr_flip", "E_derr_stable", "med_d_rel_flip",
              "med_d_rel_stable", "med_d_cos_flip", "med_d_cos_stable",
              "ci_drel_lo", "ci_drel_hi", "ci_dcos_lo", "ci_dcos_hi",
              "ci_derr_lo", "ci_derr_hi"]

ROW_FIELDS = ["transition", "client", "task", "rel_idx", "bseq",
              "slot_major_tm1", "slot_major_t", "flip", "flip_pref",
              "margin_tm1", "vote_margin_tm1", "d_rel", "d_cos",
              "ce_tm1", "ce_t", "d_err", "correct_tm1", "correct_t",
              "correct_B", "correct_C"]


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
                          interior_lams, t_index)
        if "batch_slot" not in st:
            bt = st["batch"]
            st["batch_slot"] = {
                (int(bt["client"][i]), int(bt["task"][i]),
                 int(bt["bseq"][i])): int(bt["slot_major"][i])
                for i in range(len(bt["client"]))}
        states.append(st)

    summary, agg_rows, per_rows = analyze(
        states, args_cli, lam_grid, interior_lams, args_cli.run_dir)

    with open(args_cli.out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=AGG_FIELDS)
        w.writeheader()
        w.writerows(agg_rows)
    with open(args_cli.out_rows_csv, "w", encoding="utf-8",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=ROW_FIELDS)
        w.writeheader()
        w.writerows(per_rows)
    with open(args_cli.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    trs = summary["transitions"]
    v = summary["verdicts"]
    print(f"{TAG} F1: flip rates per transition "
          f"{[t['flip_rate'] for t in trs]} (gate "
          f"{args_cli.flip_rate_gate}); keyH per state "
          f"{[s['keyH_bits'] for s in summary['states']]} bits; "
          f"rotations {v['F1_rotation_events']} -> {v['F1']}")
    print(f"{TAG} F2: {v['F2']} ({v['F2_detail']})")
    print(f"{TAG} F3a: E[dErr|flip] {[t['E_derr_flip'] for t in trs]} vs "
          f"stable {[t['E_derr_stable'] for t in trs]}")
    print(f"{TAG} F3b (main): route_pool per transition "
          f"{v['F3b_per_transition_pp']}pp, mean "
          f"{v['F3b_mean_route_pool_pp']}pp (gate "
          f"{args_cli.route_pool_gate}) -> {v['F3b']}")
    print(f"{TAG} 2x2 pools mean: drift "
          f"{v['drift_pool_mean_pp']} / route "
          f"{v['F3b_mean_route_pool_pp']} / content "
          f"{v['content_pool_mean_pp']}pp")
    print(f"{TAG} F4: {v['F4']}")
    print(f"{TAG} F5: Oracle-1 complementary="
          f"{v['F5_oracle1_complementary']}; Oracle-2b -> "
          f"{v['F5_oracle2b']} (curves in summary json)")
    f5d = v["F5_detail"]
    if f5d.get("net_pool_pp") is not None:
        print(f"{TAG} F5 net (A2-R constraint): P(flip)="
              f"{f5d['flip_fraction_pooled']}, G_net="
              f"{f5d['net_pool_pp']}pp -- a deployable gate must be "
              "disagreement-gated (G_stable = 0 by design)")
    print(f"{TAG} batch-composition sensitivity: "
          f"{summary['annex']['batch_composition_sensitivity']['verdict']}")
    print(f"{TAG} native gap location: "
          f"{summary['annex'].get('gap_location_conclusion')}")
    print(f"{TAG} FINAL: {v['FINAL']}")
    print(f"{TAG} rows -> {args_cli.out_rows_csv}; aggregates -> "
          f"{args_cli.out_csv}; summary -> {args_cli.out_json}")


if __name__ == "__main__":
    main()
