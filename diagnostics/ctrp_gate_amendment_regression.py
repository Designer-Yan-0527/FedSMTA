"""diagnostics/ctrp_gate_amendment_regression.py -- Gate Amendment
Production-Semantic Regression over ARCHIVED INR states (READ-ONLY,
2026-11-04).

READ_ONLY_ARCHIVED_STATE_REGRESSION
NO_TRAINING
NO_CHECKPOINT_WRITES
NO_MODEL_SELECTION
ROUTING_AND_THRESHOLD_UNCHANGED

User-approved layer-1 production-semantic regression: re-run the
AMENDED production premise-gate code path (the real
Client_DF._ctrp_route_slots calls + the Server_DF sample-weighted
gate statistic transcribed verbatim) against the three REAL archived
states of inr_ctrp_val_p40:

  M1: task_00_end.pth x task-0 ctrp_val rows  (state0 normal case)
  M2: task_00_end.pth x task-1 ctrp_val rows  (anchor state x failure
      row set; task_id override, same construction as the forensic
      script)
  M3: round_0008.pth    x task-1 ctrp_val rows (one round before the
      original CTRP_PREMISE_FAIL)

Invariances asserted per measurement against the frozen forensic
record (diagnostics_output/ctrp_forensic_inr_task1_round9/
forensic_boundary.json):
  * per-client batch counts
  * per-batch DEPLOYED slot sequence (routing unchanged)
  * per-batch row counts n_rows (incl. the 2-row partial tail batch)
  * tie audit flags (vote margin == 0 batches)

The ONLY allowed change is the premise statistic itself
(batch-count -> sample-weighted): M3 must flip FAIL -> PASS while the
SAME nonslot partial batch is still present and still visible in the
diagnostic_only block (1/93 = 0.010753 batch-count, 2/731 = 0.002736
sample-weighted).

ERRATUM (2026-11-04, fixed by this very regression): the earlier
session report cited "742/744" as the archived-state sample mass.
That number was a transcription/arithmetic error: the frozen forensic
record (rows_task1.json and forensic_boundary.json per_batch_rows)
sums to 731 total rows (167/167/141/134/122), which is exactly
consistent with the archived event's per-client batch counts
21/21/18/17/16. The correct frozen reading is 729/731, violation
2/731 = 0.002736. The FAIL->PASS verdict is unchanged.

Provenance separation (frozen reporting rule): the 736/738 case in
tests/test_ctrp.py is the UNIT-TEST REPRODUCTION case (code
semantics); the 729/731 numbers measured here are the REAL
ARCHIVED-STATE FORENSIC case (mechanism). Different row sets -- they
must never be cited interchangeably.

The gate statistic below is a verbatim transcription of the amended
Server_DF hook statistic (sample-weighted mass, smaller-slot-id tie
break, INTEGER boundary test 100*n_viol <= n_total). The legacy
batch-count gate is replicated in parallel as the OLD verdict, for
the FAIL->PASS comparison only.

Sessions run in SEPARATE processes (one archived server per GPU
allocation, same split as ctrp_boundary_forensics.py):

    python diagnostics/ctrp_gate_amendment_regression.py --session A
    python diagnostics/ctrp_gate_amendment_regression.py --session B
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Models.ctrp import hash_prompt_module, hash_tensor
from diagnostics.diag_utils import bootstrap_server

RUN_DIR = "output/ImageNet-R/fedta/inr_ctrp_val_p40/seed_42"
FORENSIC_DIR = REPO_ROOT / "diagnostics_output/ctrp_forensic_inr_task1_round9"
OUT_DIR = Path("diagnostics_output/ctrp_gate_amendment_regression")

DOMINANT_SLOT = 53
ANCHOR_HASH = "07385204bde571d3"
STATE0_CANDIDATE_HASH = "6fa1b5475626454d"
TAU = 0.01

# frozen M3 anchors (real archived-state forensic case, 729/731 --
# NOT the 736/738 unit-test reproduction fixture; see ERRATUM above)
M3_FROZEN = {
    "n_val_samples": 731,
    "dominant_sample_mass": "729/731",
    "violation_rows": 2,
    "violation_rate": 0.002736,
    "old_nonslot_batches": "1/93",
    "old_nonslot_frac": 0.010753,
    "partial_nonslot_batches": 1,
    "tie_batches": 1,
    "tie_sample_mass": 2,
    "nonslot_batch": {"client": 4, "slot": 35, "n_rows": 2,
                      "tie": True, "partial": True},
}


def _gate_stats(batch_records, prev_slot):
    """Verbatim transcription of the amended Server_DF premise
    statistic (Server_DF gate block: sample-weighted mass, smaller-id
    tie break, INTEGER boundary) + the legacy batch-count verdict
    replicated for the old-vs-new comparison."""
    n_total = int(sum(b["n_rows"] for b in batch_records))
    n_batches = int(len(batch_records))
    mass = {}
    for b in batch_records:
        mass[b["slot"]] = mass.get(b["slot"], 0) + int(b["n_rows"])
    modal = min(mass, key=lambda s: (-mass[s], s))
    dominant_mass = int(mass[modal])
    violation_rows = n_total - dominant_mass
    route_ok = bool(violation_rows * 100 <= n_total)   # INTEGER gate
    rate = violation_rows / float(n_total)
    n_nonslot = sum(1 for b in batch_records if b["slot"] != modal)
    old_frac = n_nonslot / float(n_batches)           # legacy statistic
    return {
        "dominant_slot": modal,
        "previous_dominant_slot": prev_slot,
        "dominant_sample_mass": f"{dominant_mass}/{n_total}",
        "violation_rows": violation_rows,
        "deployed_route_violation_rate": round(rate, 6),
        "integer_boundary_test": (
            f"{violation_rows}*100 <= {n_total} -> {route_ok}"),
        "n_val_samples": n_total,
        "n_val_batches": n_batches,
        "premise_route_ok": route_ok,
        "premise_pass": bool(route_ok and modal == prev_slot),
        "diagnostic_only": {
            "nonslot_batch_frac": round(old_frac, 6),
            "nonslot_batches": f"{n_nonslot}/{n_batches}",
            "partial_nonslot_batches": int(sum(
                1 for b in batch_records
                if b["slot"] != modal and b.get("partial"))),
            "tie_batches": int(sum(
                1 for b in batch_records if b.get("tie"))),
            "tie_sample_mass": int(sum(
                int(b["n_rows"]) for b in batch_records
                if b.get("tie"))),
            "min_vote_margin": (min(
                (int(b["vote_margin_rows"]) for b in batch_records))
                if batch_records else None),
        },
        "legacy_batch_count_gate": {
            "nonslot_batches": f"{n_nonslot}/{n_batches}",
            "nonslot_frac": round(old_frac, 6),
            "pass_at_tau_0.01": bool(old_frac <= TAU),
        },
    }


def _nonslot_records(passes, modal):
    out = []
    for cid, recs in passes:
        for b in recs:
            if b["slot"] != modal:
                out.append({"client": cid, "slot": b["slot"],
                            "n_rows": b["n_rows"],
                            "tie": b.get("tie"),
                            "partial": b.get("partial"),
                            "vote_margin_rows": b["vote_margin_rows"]})
    return out


def _compare_forensic(name, passes, forensic_m, fails):
    """passes: [(client_id, batch_records)] in srv.clients order.
    Checks routing/rows/ties against the frozen forensic record."""
    issues = []
    fb = forensic_m["per_batch_rows"]
    exp_counts = {int(c): v["n_batches"]
                  for c, v in forensic_m["per_client"].items()}
    got_counts = {int(cid): len(recs) for cid, recs in passes}
    if got_counts != exp_counts:
        issues.append(f"per-client batch counts {got_counts} != "
                      f"forensic {exp_counts}")
    flat = [(cid, b) for cid, recs in passes for b in recs]
    n_slot = n_rows = n_tie = 0
    margin_mismatch = []
    if len(flat) != len(fb):
        issues.append(f"total batches {len(flat)} != forensic {len(fb)}")
    else:
        for (cid, b), f in zip(flat, fb):
            if int(cid) != int(f["client"]) or b["slot"] != int(f["slot"]):
                issues.append(f"batch routing mismatch: client {cid} "
                              f"slot {b['slot']} != forensic client "
                              f"{f['client']} slot {f['slot']}")
            else:
                n_slot += 1
            if int(b["n_rows"]) != len(f["row_ds"]):
                issues.append(f"client {cid}: n_rows {b['n_rows']} != "
                              f"forensic {len(f['row_ds'])}")
            else:
                n_rows += 1
            exp_tie = int(f["vote_margin_rows"]) == 0
            if bool(b.get("tie")) != exp_tie:
                issues.append(f"client {cid}: tie flag {b.get('tie')} != "
                              f"forensic margin-0={exp_tie}")
            elif int(b["vote_margin_rows"]) != int(f["vote_margin_rows"]):
                margin_mismatch.append(
                    (cid, int(b["vote_margin_rows"]),
                     int(f["vote_margin_rows"])))
            else:
                n_tie += 1
    cons = {
        "forensic_record": ("forensic_boundary.json:" + name),
        "per_client_batches_match": got_counts == exp_counts,
        "slot_sequence_match": (not any("routing mismatch" in s
                                       or "slot" in s for s in issues)),
        "n_rows_match": len(flat) == len(fb) and n_rows == len(fb),
        "tie_flags_match": (len(flat) == len(fb) and n_tie == len(fb)),
        "vote_margin_rows_identical": not margin_mismatch,
        "vote_margin_mismatches": margin_mismatch,
        "checked_batches": len(flat),
    }
    fails += [f"{name}: {s}" for s in issues]
    return cons


def _check(cond, fails, msg):
    if not cond:
        fails.append(msg)
    return bool(cond)


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


def _measure_all(srv, calls):
    """calls: [(client, rows_or_None)] -- production _ctrp_route_slots
    for each client; asserts the server prompt module is untouched
    (read-only)."""
    ph_before = hash_prompt_module(srv.prompt)
    passes = []
    for c, rows in calls:
        recs = (c._ctrp_route_slots(srv.prompt) if rows is None
                else c._ctrp_route_slots(srv.prompt, rows=list(rows)))
        passes.append((c.id, recs))
    if hash_prompt_module(srv.prompt) != ph_before:
        raise RuntimeError(
            "server prompt module mutated by the measurement "
            "(read-only violated; fail-closed)")
    return passes


def session_a(device):
    """M3: round_0008.pth state x task-1 rows (production calls)."""
    fails = []
    _, srv = _session("round_0008.pth", device)
    _check(hash_tensor(srv._ctrp_prev_deployed) == ANCHOR_HASH, fails,
           f"M3 anchor hash != {ANCHOR_HASH}")
    _check(srv._ctrp_prev_dominant_slot == DOMINANT_SLOT, fails,
           "M3 prev dominant slot != 53")
    for c in srv.clients:
        _check(c.task_id == 1, fails,
               f"M3 client {c.id} task_id {c.task_id} != 1")
    # derived task-1 val rows must equal the archived forensic rows
    with open(FORENSIC_DIR / "rows_task1.json") as f:
        archived = {int(k): v for k, v in json.load(f).items()}
    for c in srv.clients:
        split = c.data_split_indices.get(1, {})
        _check(list(split.get("ctrp_val", [])) == archived[c.id], fails,
               f"M3 client {c.id}: derived task-1 rows != archived "
               f"forensic rows")
    passes = _measure_all(srv, [(c, None) for c in srv.clients])
    forensic = json.load(open(FORENSIC_DIR / "forensic_boundary.json"))
    cons = _compare_forensic(
        "M3_round0008_x_task1rows", passes,
        forensic["M3_round0008_x_task1rows"], fails)
    prev_slot = srv._ctrp_prev_dominant_slot
    stats = _gate_stats([r for _, recs in passes for r in recs], prev_slot)
    stats["nonslot_batch_records"] = _nonslot_records(
        passes, stats["dominant_slot"])
    # frozen M3 anchors (real archived-state case: 729/731; see ERRATUM)
    _check(stats["n_val_samples"] == M3_FROZEN["n_val_samples"], fails,
           f"M3 n_val_samples {stats['n_val_samples']} != 731")
    _check(stats["dominant_sample_mass"]
           == M3_FROZEN["dominant_sample_mass"], fails,
           f"M3 dominant_sample_mass {stats['dominant_sample_mass']} "
           f"!= 729/731")
    _check(stats["violation_rows"] == M3_FROZEN["violation_rows"], fails,
           f"M3 violation_rows {stats['violation_rows']} != 2")
    _check(stats["deployed_route_violation_rate"]
           == M3_FROZEN["violation_rate"], fails,
           f"M3 violation rate {stats['deployed_route_violation_rate']} "
           f"!= 0.002736")
    _check(stats["premise_pass"] is True, fails,
           "M3 amended sample-weighted gate must PASS")
    _check(stats["legacy_batch_count_gate"]["nonslot_batches"]
           == M3_FROZEN["old_nonslot_batches"], fails,
           "M3 legacy nonslot batches != 1/93")
    _check(stats["legacy_batch_count_gate"]["nonslot_frac"]
           == M3_FROZEN["old_nonslot_frac"], fails,
           "M3 legacy nonslot frac != 0.010753")
    _check(stats["legacy_batch_count_gate"]["pass_at_tau_0.01"] is False,
           fails, "M3 legacy batch-count gate must FAIL (old verdict)")
    _check(stats["diagnostic_only"]["partial_nonslot_batches"]
           == M3_FROZEN["partial_nonslot_batches"], fails,
           "M3 partial_nonslot_batches != 1")
    _check(stats["diagnostic_only"]["tie_batches"]
           == M3_FROZEN["tie_batches"], fails, "M3 tie_batches != 1")
    _check(stats["diagnostic_only"]["tie_sample_mass"]
           == M3_FROZEN["tie_sample_mass"], fails, "M3 tie_sample_mass != 2")
    _check(stats["nonslot_batch_records"]
           == [dict(M3_FROZEN["nonslot_batch"],
                     vote_margin_rows=0)], fails,
           f"M3 nonslot batch record {stats['nonslot_batch_records']} "
           f"!= frozen client-4 2-row tie partial -> slot 35")
    payload = {"M3": {"state_checkpoint": "round_0008.pth", "task": 1,
                      "gate": stats, "forensic_consistency": cons},
               "fails": fails}
    del srv, passes
    gc.collect()
    torch.cuda.empty_cache()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "session_A_m3.json", "w") as f:
        json.dump(payload, f, default=str, indent=1)
    _digest("M3", payload["M3"])
    print("[session A] fails:", fails or "none")


def session_b(device):
    """M1 + M2 under task_00_end.pth, then merge with session A."""
    fails = []
    _, srv = _session("task_00_end.pth", device)
    _check(hash_prompt_module(srv.prompt) == STATE0_CANDIDATE_HASH, fails,
           f"M1/M2 candidate hash != {STATE0_CANDIDATE_HASH}")
    _check(hash_tensor(srv._ctrp_prev_deployed) == ANCHOR_HASH, fails,
           "M1/M2 anchor hash mismatch")
    _check(srv._ctrp_prev_dominant_slot == DOMINANT_SLOT, fails,
           "M1/M2 prev dominant slot != 53")
    for c in srv.clients:
        _check(c.task_id == 0, fails,
               f"task_00_end client {c.id} task_id {c.task_id} != 0")
    forensic = json.load(open(FORENSIC_DIR / "forensic_boundary.json"))
    prev_slot = srv._ctrp_prev_dominant_slot

    # M1: task_00_end x task-0 rows (pure production calls)
    m1_passes = _measure_all(srv, [(c, None) for c in srv.clients])
    m1_cons = _compare_forensic(
        "M1_task00end_x_task0rows", m1_passes,
        forensic["M1_task00end_x_task0rows"], fails)
    m1_stats = _gate_stats([r for _, recs in m1_passes for r in recs],
                           prev_slot)
    m1_stats["nonslot_batch_records"] = _nonslot_records(
        m1_passes, m1_stats["dominant_slot"])
    _check(m1_stats["dominant_slot"] == DOMINANT_SLOT, fails,
           f"M1 dominant slot {m1_stats['dominant_slot']} != 53")
    _check(m1_stats["violation_rows"] == 0, fails,
           f"M1 violation_rows {m1_stats['violation_rows']} != 0")
    _check(m1_stats["premise_pass"] is True, fails, "M1 gate must PASS")
    _check(m1_stats["legacy_batch_count_gate"]["pass_at_tau_0.01"] is True,
           fails, "M1 legacy batch-count gate must PASS")
    del m1_passes

    # M2: task_00_end x task-1 rows (archived forensic row set;
    # task_id override, same construction as ctrp_boundary_forensics)
    with open(FORENSIC_DIR / "rows_task1.json") as f:
        rows_t1 = {int(k): v for k, v in json.load(f).items()}
    m2_calls = []
    for c in srv.clients:
        orig_task = c.task_id
        c.task_id = 1
        m2_calls.append((c, rows_t1[c.id], orig_task))
    ph_before = hash_prompt_module(srv.prompt)
    m2_passes = []
    try:
        for c, rows, _ in m2_calls:
            m2_passes.append(
                (c.id, c._ctrp_route_slots(srv.prompt, rows=list(rows))))
    finally:
        for c, _, orig_task in m2_calls:
            c.task_id = orig_task
    if hash_prompt_module(srv.prompt) != ph_before:
        raise RuntimeError("server prompt module mutated (read-only "
                           "violated; fail-closed)")
    m2_cons = _compare_forensic(
        "M2_task00end_x_task1rows", m2_passes,
        forensic["M2_task00end_x_task1rows"], fails)
    m2_stats = _gate_stats([r for _, recs in m2_passes for r in recs],
                           prev_slot)
    m2_stats["nonslot_batch_records"] = _nonslot_records(
        m2_passes, m2_stats["dominant_slot"])
    _check(m2_stats["dominant_slot"] == DOMINANT_SLOT, fails,
           f"M2 dominant slot {m2_stats['dominant_slot']} != 53")
    _check(m2_stats["violation_rows"] == 0, fails,
           f"M2 violation_rows {m2_stats['violation_rows']} != 0")
    _check(m2_stats["n_val_samples"] == 731, fails,
           f"M2 n_val_samples {m2_stats['n_val_samples']} != 731")
    _check(m2_stats["premise_pass"] is True, fails, "M2 gate must PASS")
    _check(m2_stats["legacy_batch_count_gate"]["pass_at_tau_0.01"] is True,
           fails, "M2 legacy batch-count gate must PASS")
    _check(m2_stats["diagnostic_only"]["tie_batches"] == 1, fails,
           "M2 tie_batches != 1 (the 1:1 53-vs-96 tie batch)")
    del m2_passes, srv
    gc.collect()
    torch.cuda.empty_cache()

    with open(OUT_DIR / "session_A_m3.json") as f:
        m3_payload = json.load(f)
    fails += m3_payload["fails"]
    results = {
        "header": {
            "REGRESSION": "CTRP gate amendment production-semantic "
                          "regression (layer 1, user-approved)",
            "READ_ONLY": True,
            "NO_TRAINING": True,
            "NO_CHECKPOINT_WRITES": True,
            "ROUTING_AND_THRESHOLD_UNCHANGED": True,
            "AMENDMENT": ("sample-weighted deployed-route dominance; "
                          "integer boundary 100*n_viol <= n_total; "
                          "tau=0.01 unchanged"),
            "unit_test_reproduction_case": (
                "736/738 rows fixture in tests/test_ctrp.py (code "
                "semantics ONLY)"),
            "real_archived_forensic_case": (
                "729/731 rows, inr_ctrp_val_p40 archived states "
                "(mechanism; ERRATUM: earlier reports citing 742/744 "
                "were a transcription error)"),
            "source_checkpoints": ["task_00_end.pth", "round_0008.pth"],
            "run_dir": RUN_DIR,
        },
        "M1_task00end_x_task0rows": {
            "state_checkpoint": "task_00_end.pth", "task": 0,
            "gate": m1_stats, "forensic_consistency": m1_cons},
        "M2_task00end_x_task1rows": {
            "state_checkpoint": "task_00_end.pth", "task": 1,
            "gate": m2_stats, "forensic_consistency": m2_cons},
        "M3_round0008_x_task1rows": m3_payload["M3"],
        "failures": fails,
        "verdict": ("PASS" if not fails else "FAIL"),
    }
    with open(OUT_DIR / "gate_amendment_regression.json", "w") as f:
        json.dump(results, f, default=str, indent=1)
    for key in ("M1_task00end_x_task0rows", "M2_task00end_x_task1rows",
                "M3_round0008_x_task1rows"):
        _digest(key.split("_")[0], results[key])
    print("-" * 70)
    print("REGRESSION VERDICT:", results["verdict"])
    if fails:
        print("failures:")
        for s in fails:
            print("  -", s)
    else:
        print("M3 headline: same nonslot partial batch still present "
              "(client 4, 2-row tie tail -> slot 35); old batch-count "
              "gate 1/93 = 0.010753 FAIL -> amended sample-weighted "
              "gate 2/731 = 0.002736 PASS; routing/dominant slot/batch "
              "counts/n_rows/tie flags all identical to the frozen "
              "forensic record.")
    print("output:", OUT_DIR / "gate_amendment_regression.json")


def _digest(name, m):
    g = m["gate"]
    print(f"[{name} {m['state_checkpoint']} x task{m['task']} rows]")
    print(f"  dominant={g['dominant_slot']} batches={g['n_val_batches']}"
          f" rows={g['n_val_samples']}"
          f" mass={g['dominant_sample_mass']}"
          f" viol={g['violation_rows']}"
          f" rate={g['deployed_route_violation_rate']}")
    print(f"  new gate: {g['integer_boundary_test']} "
          f"premise_pass={g['premise_pass']}")
    print(f"  old gate: {g['legacy_batch_count_gate']['nonslot_batches']}"
          f" = {g['legacy_batch_count_gate']['nonslot_frac']}"
          f" pass={g['legacy_batch_count_gate']['pass_at_tau_0.01']}")
    print(f"  diagnostic_only: {g['diagnostic_only']}")
    if "nonslot_batch_records" in g and g["nonslot_batch_records"]:
        print(f"  nonslot batches: {g['nonslot_batch_records']}")
    c = m["forensic_consistency"]
    print(f"  forensic consistency: batches={c['checked_batches']}"
          f" slots_match={c['slot_sequence_match']}"
          f" rows_match={c['n_rows_match']}"
          f" ties_match={c['tie_flags_match']}"
          f" margins_identical={c['vote_margin_rows_identical']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", choices=["A", "B"], required=True)
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"],
                    default="auto")
    ns = ap.parse_args()
    if ns.device == "cpu":
        dev = "cpu"
    elif ns.device == "cuda":
        dev = "cuda"
    else:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (session_a if ns.session == "A" else session_b)(dev)
