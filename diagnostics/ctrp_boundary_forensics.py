"""diagnostics/ctrp_boundary_forensics.py -- Archived-State Boundary
Forensics (INR ctrp_val task-1 CTRP_PREMISE_FAIL follow-up, 2026-11-04).

FORENSIC_ARCHIVED_STATES_ONLY
NO_TRAINING_REPLAY
NO_MODEL_SELECTION
NO_THRESHOLD_AMENDMENT

Context: the inr_ctrp_val_p40 run failed closed at the task-1-end hook
(1/93 = 0.010753 nonslot val batches > the frozen 0.01 gate). The
hook-time candidate state (hash 8bafe3d8...) was never checkpointed
(the hook raises BEFORE maybe_save_checkpoint), and the original run
executed in the official NON-deterministic mode
(args.deterministic=false), so bitwise reconstruction of the failed
round-9 state is NOT guaranteed and is NOT attempted here. This
script measures only the REAL archived states, with the production
routing semantics:

  M1: task_00_end.pth state x task-0 ctrp_val rows
      replicates the state0 hook measurement; anchors (from
      ctrp_transitions.jsonl): 87 batches, 0 nonslot, dominant slot 53.
  M2: task_00_end.pth state x task-1 ctrp_val rows
      the failure row set under the anchor (task-0-end) state.
  M3: round_0008.pth state x task-1 ctrp_val rows
      the failure row set one round before the failure; anchors (from
      the CTRP_PREMISE_FAIL event record): per-client batches
      21/21/18/17/16 = 93 total.

M2 -> M3 isolates the state drift on the SAME row set.

Per-row routing evidence replicates Global_Prompt.forward's math
verbatim (l2_normalize + matmul, embedding_key='cls' with the passed
cls_features=q; batchwise_prompt=True -> batch slot = per-row top-1
majority vote), and each batch's vote-majority is cross-checked
against the production forward's actual prompt_idx.

Route invariance check (alpha=0 vs alpha=1): CTRP blends prompt
VALUES only and never touches prompt_key, so deploying the
alpha=0-blended values (dominant row <- anchor row) must leave the
route identical. Verified empirically by re-running the pass with
the blended values.

Sessions run in SEPARATE processes (the two bootstrapped servers do
not fit in one GPU allocation):

    python diagnostics/ctrp_boundary_forensics.py --session A
    python diagnostics/ctrp_boundary_forensics.py --session B
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Models.ctrp import hash_prompt_module, hash_tensor
from diagnostics.diag_utils import bootstrap_server

RUN_DIR = "output/ImageNet-R/fedta/inr_ctrp_val_p40/seed_42"
OUT_DIR = Path("diagnostics_output/ctrp_forensic_inr_task1_round9")

STATE0_RECORD = {          # ctrp_transitions.jsonl, task 0 end
    "n_batches": 87, "n_nonslot": 0, "dominant_slot": 53,
    "candidate_module_hash": "6fa1b5475626454d",
    "anchor_hash": "07385204bde571d3",
}
FAIL_RECORD = {            # ctrp_events.jsonl, task 1 end round 9
    "per_client_batches": {"0": 21, "1": 21, "2": 18, "3": 17, "4": 16},
    "n_batches": 93, "n_nonslot": 1, "nonslot_frac": 0.010753,
    "dominant_slot": 53,
    "candidate_module_hash": "8bafe3d80259a3cb",  # NOT archived: gone
    "anchor_hash": "07385204bde571d3",
}
DOMINANT_SLOT = 53
BATCH_SIZE = 8
NEAR_TIE_BANDS = (1e-4, 1e-3, 1e-2, 0.05, 0.10)


def l2_normalize(x, dim=None, epsilon=1e-12):
    """Global_Prompt.l2_normalize, verbatim (bitwise-identical math)."""
    square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
    x_inv_norm = torch.rsqrt(
        torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
    return x * x_inv_norm


def _rows(client, t, rows_key="ctrp_val"):
    split = client.data_split_indices.get(t, {})
    if rows_key not in split or not split[rows_key]:
        raise RuntimeError(
            f"client {client.id}: no recorded {rows_key} rows for task {t}")
    return list(split[rows_key])


def measure(client, server_prompt, t, rows, device, alpha0_row=None):
    """One instrumented routing pass under the SERVER CANDIDATE state.

    alpha0_row: optional (length, dim) tensor; if given, a second pass
    runs with the dominant-row VALUES replaced by it (alpha=0 blend;
    route must be identical because prompt_key is untouched)."""
    snap = client._a5_rng_snapshot()
    flags = (client.vit.training, client.model.training,
             client.original_model.training)
    orig_prompt = client.vit.prompt
    orig_task_id = client.task_id
    client.task_id = t
    nw = int(getattr(client.args, "num_workers", 0) or 0)
    out = {"client": client.id, "task": t, "batches": []}
    try:
        client.original_model.eval()
        client.vit.eval()
        client.model.eval()
        client._ctrp_with_candidate(server_prompt)
        key = client.vit.prompt.prompt_key
        out["key_hash"] = hash_tensor(key)
        out["candidate_module_hash"] = hash_prompt_module(client.vit.prompt)
        prompt_norm = l2_normalize(key, dim=1)

        def _pass(slots_only=False):
            batches = []
            loader = DataLoader(Subset(client.train_data[t], list(rows)),
                                batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=nw)
            for inp, tgt in loader:
                inp = inp.to(device)
                with torch.no_grad():
                    q = client.original_model(inp)["pre_logits"]
                    fwd = client.vit(inp, task_id=t, cls_features=q,
                                     train=True)
                slot = int(fwd["prompt_idx"][0, 0].item())
                with torch.no_grad():
                    x_norm = l2_normalize(q, dim=1)
                    sim = torch.matmul(x_norm, prompt_norm.t())
                    sv, si = torch.topk(sim, 2, dim=1)
                sv = sv.float().cpu().numpy()
                si = si.cpu().numpy()
                pool = key.shape[0]
                votes = np.bincount(si[:, 0], minlength=pool)
                vote_rank = np.argsort(-votes, kind="stable")[:2]
                if slots_only:
                    batches.append(slot)
                    continue
                rec = {
                    "slot": slot,
                    "votes_top2": [[int(s), int(votes[s])]
                                    for s in vote_rank],
                    "vote_margin_rows": int(votes[vote_rank[0]]
                                           - votes[vote_rank[1]]),
                    "row_vote_matches_production": bool(
                        {int(fwd["prompt_idx"][i, 0].item())
                         for i in range(fwd["prompt_idx"].shape[0])}
                        == {slot} and int(vote_rank[0]) == slot),
                    "rows": [],
                }
                for r in range(si.shape[0]):
                    rec["rows"].append({
                        "top1": int(si[r, 0]), "top2": int(si[r, 1]),
                        "s1": float(sv[r, 0]), "s2": float(sv[r, 1]),
                        "ds": float(sv[r, 0] - sv[r, 1]),
                    })
                batches.append(rec)
            return batches

        out["batches"] = _pass(slots_only=False)
        out["key_hash_after"] = hash_tensor(client.vit.prompt.prompt_key)
        if alpha0_row is not None:
            with torch.no_grad():
                client.vit.prompt.prompt[DOMINANT_SLOT].copy_(
                    alpha0_row.to(client.vit.prompt.prompt.device))
            out["alpha0_slots"] = _pass(slots_only=True)
            out["alpha0_key_hash_after"] = hash_tensor(
                client.vit.prompt.prompt_key)
            out["alpha0_route_identical"] = bool(
                out["alpha0_slots"]
                == [b["slot"] for b in out["batches"]])
        return out
    finally:
        client.vit.prompt = orig_prompt
        client.task_id = orig_task_id
        client.model.train(flags[1])
        if flags[2] is not None:
            client.original_model.train(flags[2])
        client._a5_rng_restore(snap)


def aggregate(passes, state_ckpt, task):
    all_ds = np.array([r["ds"] for p in passes for b in p["batches"]
                       for r in b["rows"]], dtype=np.float64)
    flat = [(p["client"], b) for p in passes for b in p["batches"]]
    slots = [b["slot"] for _, b in flat]
    modal = int(np.bincount(slots).argmax())
    per_client = {}
    for p in passes:
        per_client[p["client"]] = {
            "n_batches": len(p["batches"]),
            "n_nonslot": sum(1 for b in p["batches"] if b["slot"] != modal),
            "key_hash": p["key_hash"],
        }
    dom_batches = [b for _, b in flat if b["slot"] == modal]
    closest = sorted(
        dom_batches,
        key=lambda b: (b["vote_margin_rows"],
                       min(r["ds"] for r in b["rows"])))[:10]
    a0 = [p for p in passes if "alpha0_route_identical" in p]
    return {
        "state_checkpoint": state_ckpt,
        "task": task,
        "dominant_slot": modal,
        "total_batches": len(slots),
        "total_nonslot": int(sum(1 for s in slots if s != modal)),
        "per_client": per_client,
        "slots_flat": [int(s) for s in slots],
        "clients_flat": [int(c) for c, _ in flat],
        "row_ds_stats": {
            "min": float(all_ds.min()),
            "p1": float(np.percentile(all_ds, 1)),
            "p5": float(np.percentile(all_ds, 5)),
            "median": float(np.percentile(all_ds, 50)),
            "mean": float(all_ds.mean()),
        },
        "near_tie_rows": {f"ds<{b}": int((all_ds < b).sum())
                          for b in NEAR_TIE_BANDS},
        "nonslot_batches": [{"client": c, "slot": b["slot"],
                              "votes": b["votes_top2"]}
                             for c, b in flat if b["slot"] != modal],
        "closest_to_flip_top10": [
            {"votes": b["votes_top2"],
             "vote_margin_rows": b["vote_margin_rows"],
             "row_ds_min": min(r["ds"] for r in b["rows"])}
            for b in closest],
        "vote_mismatch_batches": int(sum(
            1 for _, b in flat if not b["row_vote_matches_production"])),
        "alpha0_route_identical": (all(p["alpha0_route_identical"]
                                       for p in a0)
                                   if a0 else None),
        "per_batch_rows": [
            {"client": c, "slot": b["slot"],
             "votes": b["votes_top2"],
             "vote_margin_rows": b["vote_margin_rows"],
             "row_ds": [r["ds"] for r in b["rows"]],
             "row_top1": [r["top1"] for r in b["rows"]],
             "row_top2": [r["top2"] for r in b["rows"]]}
            for c, b in flat],
    }


def shift(m_from, m_to):
    agree = None
    if (m_from["total_batches"] == m_to["total_batches"]
            and m_from["clients_flat"] == m_to["clients_flat"]):
        agree = float(np.mean([u == v for u, v in
                               zip(m_from["slots_flat"],
                                   m_to["slots_flat"])]))
    return {
        "same_rows": True,
        "ds_min_delta": m_to["row_ds_stats"]["min"]
                        - m_from["row_ds_stats"]["min"],
        "ds_p1_delta": m_to["row_ds_stats"]["p1"]
                       - m_from["row_ds_stats"]["p1"],
        "ds_p5_delta": m_to["row_ds_stats"]["p5"]
                       - m_from["row_ds_stats"]["p5"],
        "ds_median_delta": m_to["row_ds_stats"]["median"]
                           - m_from["row_ds_stats"]["median"],
        "nonslot_from": m_from["total_nonslot"],
        "nonslot_to": m_to["total_nonslot"],
        "dominant_slot_stable": (m_from["dominant_slot"]
                                 == m_to["dominant_slot"]),
        "per_client_nonslot_from": {c: v["n_nonslot"]
                                    for c, v in m_from["per_client"].items()},
        "per_client_nonslot_to": {c: v["n_nonslot"]
                                  for c, v in m_to["per_client"].items()},
        "batch_slot_agreement_frac": agree,
    }


def _session(ckpt, device):
    args, server = bootstrap_server(RUN_DIR, checkpoint=ckpt,
                                    data_path="./local_datasets",
                                    device=device)
    server.model.to(device)
    server.origin_model.to(device)
    server.origin_model.eval()
    server.prompt = server.prompt.to(device)
    for c in server.clients:
        c.model.to(device)
    return args, server


def session_a(device):
    """round_0008 state x task-1 rows (M3) + row extraction."""
    gate_fail = []
    _, srv = _session("round_0008.pth", device)
    anchor = srv._ctrp_prev_deployed
    a_hash = hash_tensor(anchor)
    if a_hash != FAIL_RECORD["anchor_hash"]:
        gate_fail.append(f"anchor hash {a_hash} != "
                         f"{FAIL_RECORD['anchor_hash']}")
    if srv._ctrp_prev_dominant_slot != DOMINANT_SLOT:
        gate_fail.append("prev dominant slot != 53")
    rows_t1 = {}
    m3_passes = []
    for c in srv.clients:
        if c.task_id != 1:
            gate_fail.append(f"client {c.id} task_id {c.task_id} != 1")
        rows_t1[c.id] = _rows(c, 1)
        exp = FAIL_RECORD["per_client_batches"][str(c.id)]
        got = (len(rows_t1[c.id]) + BATCH_SIZE - 1) // BATCH_SIZE
        if got != exp:
            gate_fail.append(f"client {c.id}: task-1 batch count {got}"
                             f" != recorded {exp}")
        m3_passes.append(measure(c, srv.prompt, 1, rows_t1[c.id], device,
                                 alpha0_row=anchor[DOMINANT_SLOT]))
    m3 = aggregate(m3_passes, "round_0008.pth", 1)
    del m3_passes, srv
    gc.collect()
    torch.cuda.empty_cache()
    payload = {"M3_round0008_x_task1rows": m3,
               "gate_fail_A": gate_fail}
    with open(OUT_DIR / "rows_task1.json", "w") as f:
        json.dump({str(k): v for k, v in rows_t1.items()}, f)
    with open(OUT_DIR / "session_A_m3.json", "w") as f:
        json.dump(payload, f, default=str)
    print("[session A] M3 done. gate_fail:", gate_fail)
    _digest(m3)


def session_b(device):
    """task_00_end state x task-0 rows (M1) and x task-1 rows (M2),
    then merge with session A's M3 into the final report."""
    gate_fail = []
    _, srv = _session("task_00_end.pth", device)
    cand_hash = hash_prompt_module(srv.prompt)
    if cand_hash != STATE0_RECORD["candidate_module_hash"]:
        gate_fail.append(f"task_00_end candidate hash {cand_hash} != "
                         f"{STATE0_RECORD['candidate_module_hash']}")
    anchor_b = srv._ctrp_prev_deployed
    if hash_tensor(anchor_b) != STATE0_RECORD["anchor_hash"]:
        gate_fail.append("task_00_end anchor hash mismatch")
    with open(OUT_DIR / "rows_task1.json") as f:
        rows_t1 = {int(k): v for k, v in json.load(f).items()}
    m1_passes = [measure(c, srv.prompt, 0, _rows(c, 0), device,
                         alpha0_row=anchor_b[DOMINANT_SLOT])
                 for c in srv.clients]
    m1 = aggregate(m1_passes, "task_00_end.pth", 0)
    del m1_passes
    if m1["total_batches"] != STATE0_RECORD["n_batches"]:
        gate_fail.append(f"M1 batches {m1['total_batches']} != 87")
    if m1["total_nonslot"] != STATE0_RECORD["n_nonslot"]:
        gate_fail.append(f"M1 nonslot {m1['total_nonslot']} != 0")
    if m1["dominant_slot"] != STATE0_RECORD["dominant_slot"]:
        gate_fail.append(f"M1 dominant slot {m1['dominant_slot']} != 53")
    m2_passes = [measure(c, srv.prompt, 1, rows_t1[c.id], device,
                         alpha0_row=anchor_b[DOMINANT_SLOT])
                 for c in srv.clients]
    m2 = aggregate(m2_passes, "task_00_end.pth", 1)
    with open(OUT_DIR / "rows_task0.json", "w") as f:
        json.dump({str(c.id): _rows(c, 0) for c in srv.clients}, f)
    del m2_passes, srv
    gc.collect()
    torch.cuda.empty_cache()
    with open(OUT_DIR / "session_A_m3.json") as f:
        m3 = json.load(f)["M3_round0008_x_task1rows"]
    gate_fail += json.load(
        open(OUT_DIR / "session_A_m3.json"))["gate_fail_A"]
    results = {
        "header": {
            "FORENSIC_ARCHIVED_STATES_ONLY": True,
            "NO_TRAINING_REPLAY": True,
            "NO_MODEL_SELECTION": True,
            "NO_THRESHOLD_AMENDMENT": True,
            "SOURCE_CHECKPOINTS": ["task_00_end.pth", "round_0008.pth"],
            "FAILED_STATE_HASH": FAIL_RECORD["candidate_module_hash"],
            "FAILED_STATE_RECOVERABLE": False,
            "REASON": ("original run executed with deterministic=false "
                       "(official non-deterministic mode); the round-9 "
                       "hook-time state was never checkpointed"),
            "run_dir": RUN_DIR,
            "dominant_slot_pinned": DOMINANT_SLOT,
        },
        "M1_task00end_x_task0rows": m1,
        "M2_task00end_x_task1rows": m2,
        "M3_round0008_x_task1rows": m3,
        "shift_M2_to_M3_same_rows": shift(m2, m3),
        "gate_anchors": {"all_passed": not gate_fail,
                         "failures": gate_fail},
    }
    with open(OUT_DIR / "forensic_boundary.json", "w") as f:
        json.dump(results, f, default=str)
    print("=" * 70)
    for key in ("M1_task00end_x_task0rows", "M2_task00end_x_task1rows",
                "M3_round0008_x_task1rows"):
        _digest(results[key])
    print("-" * 70)
    print("[shift M2->M3 same rows]",
          json.dumps(results["shift_M2_to_M3_same_rows"], default=str))
    print("[gate]", "PASS" if not gate_fail else f"FAIL: {gate_fail}")
    print("output:", OUT_DIR / "forensic_boundary.json")
    if gate_fail:
        print("FORENSIC_STATE_ANCHOR_MISMATCH -- no interpretive "
              "diagnostics beyond the anchor table above.")


def _digest(m):
    print(f"[{m['state_checkpoint']} x task{m['task']}rows]")
    print(f"  dominant={m['dominant_slot']} batches={m['total_batches']}"
          f" nonslot={m['total_nonslot']}"
          f" per_client={ {c: v['n_nonslot'] for c, v in m['per_client'].items()} }")
    print(f"  row ds: min={m['row_ds_stats']['min']:.6f}"
          f" P1={m['row_ds_stats']['p1']:.6f}"
          f" P5={m['row_ds_stats']['p5']:.6f}"
          f" median={m['row_ds_stats']['median']:.6f}")
    print(f"  near-tie rows: {m['near_tie_rows']}")
    print(f"  vote-mismatch batches: {m['vote_mismatch_batches']}"
          f"  alpha0 route identical: {m['alpha0_route_identical']}")
    print(f"  closest-to-flip (vote margin rows, ds_min): "
          f"{[(b['vote_margin_rows'], round(b['row_ds_min'], 6)) for b in m['closest_to_flip_top10'][:5]]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", choices=["A", "B"], required=True)
    ns = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    (session_a if ns.session == "A" else session_b)(dev)
