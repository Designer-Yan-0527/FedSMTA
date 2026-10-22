"""A5 PR-3 four-arm experiment-switch tests (GPU short runs + CPU units).

Final review 2026-10-22 (PASS, migration guard closed) approved the PR-3
four-arm development with three implementation requirements:

  (a) Shared branch supports CE-only / Proto-CE via lambda = 0 / 0.5;
  (b) FedAvg / Local-only toggles ONLY the U federated aggregation;
  (c) arm identity must be verifiable from actual checkpoints and logs,
      not just from CLI names; migrated (a5_proto_migration) runs must be
      excluded from formal metric aggregation.

Cases:
  N0  (CPU) validate_method_args: --a5_u_agg local is ONLY valid for
      --method fedta_a5 (rejected for fedta / fedta_a2).
  N1  (GPU) default arm (u_agg=fedavg) behavior UNCHANGED: after a round
      every client U is bit-identical (the PR-2C broadcast ran); the
      checkpoint carries a5_arm{u_agg=fedavg} and the [A5-ARM] banner is
      in the log.
  N2  (GPU) Local Adapter arm: --a5_u_agg local -> client U's DIVERGE
      (no broadcast), a5_arm{u_agg=local}, per-client a5_state{u_agg=
      local}, banner shows u_agg=local.
  N3  (GPU) Local-arm resume: same mode resumes fine; switching to fedavg
      mid-run is REFUSED (RESUME_MUST_MATCH + a5_state fail-closed).
  N4  (GPU) pre-PR-3 compatibility: a checkpoint with the u_agg key
      stripped (simulating a PR-2-era checkpoint) resumes under fedavg
      (implicit) but is REFUSED under --a5_u_agg local.
  N5  (GPU) Shared CE-only arm: --a5_proto_lambda 0 -> a5_arm{proto_ce=
      off, proto_lambda=0.0} while the proto table STILL syncs (budget-
      fair: a5_proto_state populated at the task-0 boundary).
  N6  (CPU) diagnostics/aggregate_results.py formal-pipeline guards:
      migration-marker run REJECTED; a5_arm/args mismatch REJECTED;
      pre-PR-3 checkpoint (no a5_arm) REJECTED; clean A5 run and non-A5
      run ACCEPTED.

Usage (from repo root, on the server; Tail_Anchor hardcodes cuda):
  python tests/test_a5_pr3_arms.py --data_path <DATA_PATH>
"""

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from test_resume_regression import torch_load_compat  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser("A5 PR-3 four-arm switch tests")
    p.add_argument("--data_name", default="cifar100")
    p.add_argument("--data_path", default="./local_datasets")
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--device", default="cuda")
    p.add_argument("--global_epoch", default=2, type=int)
    p.add_argument("--task_num", default=2, type=int)
    p.add_argument("--local_epoch", default=1, type=int)
    p.add_argument("--work_dir", default="output/regression_a5_pr3_arms",
                   type=Path)
    return p.parse_args()


def run_main(extra, expect_fail=False, expect_msg=None, label=""):
    cmd = [sys.executable, "main.py", "datasets_delay"] + extra
    print("+", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT),
                          capture_output=True, text=True)
    if expect_fail:
        assert proc.returncode != 0, (
            f"{label}: expected failure but command succeeded")
        combined = (proc.stdout or "") + (proc.stderr or "")
        if expect_msg is not None:
            assert expect_msg in combined, (
                f"{label}: expected error message containing "
                f"{expect_msg!r}, got tail:\n"
                + "\n".join(combined.splitlines()[-30:]))
        print(f"[PASS] {label}: rejected as expected")
    else:
        if proc.returncode != 0:
            print((proc.stdout or "")[-2000:])
            print((proc.stderr or "")[-2000:])
            raise RuntimeError(f"{label}: command failed")
        print(f"[PASS] {label}")
    return proc


def base_flags(a, run_name):
    return [
        "--data_name", a.data_name,
        "--data-path", a.data_path,
        "--seed", str(a.seed),
        "--device", a.device,
        "--global_epoch", str(a.global_epoch),
        "--task_num", str(a.task_num),
        "--local_epoch", str(a.local_epoch),
        "--deterministic",
        "--save_every", "1",
        "--output_dir", str(a.work_dir),
        "--method", "fedta_a5",
        "--run_name", run_name,
    ]


def latest_ckpt(a, run_name):
    return (a.work_dir / a.data_name / "fedta_a5" / run_name
            / f"seed_{a.seed}" / "checkpoints" / "latest.pth")


def strip_keys(src, dst, strip_server=(), strip_client_a5=()):
    """Copy a checkpoint, removing keys to synthesize older schemas."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    ckpt = torch_load_compat(src)
    for k in strip_server:
        ckpt.pop(k, None)
    if "args" in ckpt:
        for k in strip_server:
            ckpt["args"].pop(k, None)
    for c in ckpt.get("clients", []):
        if "a5_state" in c:
            for k in strip_client_a5:
                c["a5_state"].pop(k, None)
    import torch
    torch.save(ckpt, dst)
    return dst


def n0_validation():
    print("=== N0: --a5_u_agg local is A5-only (validate_method_args) ===")
    from argparse import Namespace
    from main import validate_method_args
    for method in ("fedta", "fedta_a2", "fedta_a3"):
        try:
            validate_method_args(Namespace(method=method, a5_u_agg="local"))
            raise AssertionError(
                f"N0: --a5_u_agg local accepted for method {method!r}")
        except ValueError as e:
            assert "ONLY valid" in str(e), f"N0: unexpected error: {e}"
    # valid: fedta_a5 accepts both modes (and the other PR-3 switches)
    validate_method_args(Namespace(
        method="fedta_a5", a5_u_agg="local", a2_arm="none",
        a5_proto_lambda=0.0, a5_proto_tau=0.07, a5_rank=8))
    validate_method_args(Namespace(
        method="fedta_a5", a5_u_agg="fedavg", a2_arm="anchor",
        a5_proto_lambda=0.5, a5_proto_tau=0.07, a5_rank=8))
    print("[PASS] N0: local rejected for non-A5 methods; accepted for A5")


def n1_default_fedavg(a):
    print("=== N1: default arm (fedavg) unchanged ===")
    run_name = "n1_fed"
    proc = run_main(base_flags(a, run_name) + ["--max_rounds", "2"],
                    label="N1: default fedta_a5 short run")
    ckpt = torch_load_compat(latest_ckpt(a, run_name))
    Us = [c["a5_state"]["U"] for c in ckpt["clients"]]
    for i, U in enumerate(Us[1:], start=1):
        assert torch_equal(Us[0], U), (
            f"N1: client {i} U differs after aggregation (broadcast broken)")
    arm = ckpt.get("a5_arm")
    assert isinstance(arm, dict) and arm["u_agg"] == "fedavg", (
        f"N1: a5_arm descriptor wrong: {arm}")
    assert arm["a2_arm"] == "anchor" and arm["proto_ce"] == "on", (
        f"N1: a5_arm descriptor wrong: {arm}")
    assert all(c["a5_state"]["u_agg"] == "fedavg"
               for c in ckpt["clients"]), "N1: a5_state u_agg wrong"
    assert "[A5-ARM]" in proc.stdout and "u_agg=fedavg" in proc.stdout, (
        "N1: arm banner missing from the log")
    print("[PASS] N1: U's bit-identical (broadcast ran); a5_arm + banner ok")
    return latest_ckpt(a, run_name)


def torch_equal(x, y):
    import torch
    return torch.equal(x, y)


def n2_local_arm(a):
    print("=== N2: Local Adapter arm (--a5_u_agg local) ===")
    run_name = "n2_loc"
    proc = run_main(base_flags(a, run_name)
                    + ["--a5_u_agg", "local", "--max_rounds", "2"],
                    label="N2: local-adapter short run")
    ckpt = torch_load_compat(latest_ckpt(a, run_name))
    Us = [c["a5_state"]["U"] for c in ckpt["clients"]]
    assert len(Us) >= 2, "N2: need >= 2 clients"
    diverged = any(not torch_equal(Us[0], U) for U in Us[1:])
    assert diverged, (
        "N2: all client U's identical under --a5_u_agg local — the "
        "broadcast was NOT disabled")
    arm = ckpt.get("a5_arm")
    assert arm and arm["u_agg"] == "local", f"N2: a5_arm wrong: {arm}"
    assert all(c["a5_state"]["u_agg"] == "local"
               for c in ckpt["clients"]), "N2: a5_state u_agg wrong"
    assert "u_agg=local" in proc.stdout, "N2: banner missing u_agg=local"
    print("[PASS] N2: U's diverged (no broadcast); a5_arm + banner ok")
    return latest_ckpt(a, run_name)


def n3_local_resume(a, loc_ckpt):
    print("=== N3: local-arm resume (same mode ok, switch refused) ===")
    # same mode: the a5_state loader + RESUME_MUST_MATCH both pass
    run_main(base_flags(a, "n2_loc")
             + ["--a5_u_agg", "local",
                "--resume", str(loc_ckpt), "--max_rounds", "3"],
             label="N3: resume local arm with local mode")
    # switch to fedavg mid-run: refused by RESUME_MUST_MATCH (server-side
    # config-consistency check fires before the per-client a5_state loader)
    run_main(base_flags(a, "n2_loc")
             + ["--a5_u_agg", "fedavg",
                "--resume", str(loc_ckpt), "--max_rounds", "3"],
             expect_fail=True,
             expect_msg="a5_u_agg checkpoint=local, current=fedavg",
             label="N3: switch to fedavg")
    print("[PASS] N3: same-mode resume ok; mid-run switch refused")


def n4_pre_pr3_compat(a, fed_ckpt):
    print("=== N4: pre-PR-3 checkpoint (u_agg key absent) ===")
    synth_dir = a.work_dir / a.data_name / "fedta_a5" / "synth_prepr3" \
        / f"seed_{a.seed}" / "checkpoints"
    synth = strip_keys(fed_ckpt, synth_dir / "prepr3.pth",
                       strip_server=("a5_arm", "a5_u_agg"),
                       strip_client_a5=("u_agg",))
    flags = base_flags(a, "synth_prepr3")
    run_main(flags + ["--resume", str(synth), "--max_rounds", "3"],
             label="N4: pre-PR-3 ckpt resumes under fedavg (implicit)")
    run_main(flags + ["--a5_u_agg", "local",
                      "--resume", str(synth), "--max_rounds", "3"],
             expect_fail=True,
             expect_msg="u_agg mismatch", label="N4: pre-PR-3 + local")
    print("[PASS] N4: implicit fedavg backward-compat; local refused")


def n5_ce_only(a):
    print("=== N5: Shared CE-only arm (lambda = 0) ===")
    run_name = "n5_ce0"
    proc = run_main(base_flags(a, run_name)
                    + ["--a5_proto_lambda", "0.0", "--max_rounds", "2"],
                    label="N5: CE-only short run")
    ckpt = torch_load_compat(latest_ckpt(a, run_name))
    arm = ckpt.get("a5_arm")
    assert arm and arm["proto_ce"] == "off" and arm["proto_lambda"] == 0.0, (
        f"N5: a5_arm wrong: {arm}")
    # budget-fair: the reference-prototype table STILL syncs at the task
    # boundary even with proto-CE off (P2 v1.2 §4: identical budget)
    proto = ckpt.get("a5_proto_state")
    assert proto and proto.get("version", -1) >= 0 and proto.get("counts"), (
        "N5: proto table did not sync under lambda=0")
    assert "proto_ce=off" in proc.stdout, "N5: banner missing proto_ce=off"
    print("[PASS] N5: proto_ce=off in a5_arm/banner; table still synced")


def n6_aggregator(a):
    print("=== N6: aggregate_results.py formal-pipeline guards (CPU) ===")
    import torch
    root = a.work_dir / "agg_root"

    def make_run(method, run_name, arm=None, migration=False,
                 mismatch_uagg=False, no_arm=False, make_ckpt=True):
        run_dir = root / "cifar100" / method / run_name / "seed_42"
        (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
        cfg = {"method": method, "a2_arm": "anchor", "a5_proto_lambda": 0.5,
               "a5_proto_tau": 0.07, "a5_u_agg": "fedavg", "a5_rank": 8}
        with open(run_dir / "args.json", "w") as f:
            json.dump(cfg, f)
        with open(run_dir / "metrics" / "summary_metrics.csv", "w",
                  newline="") as f:
            w = csv.writer(f)
            w.writerow(["after_task", "avg_accuracy", "forgetting", "bwt"])
            w.writerow([0, "0.5000", "", ""])
        if make_ckpt:
            ckpt = {}
            if not no_arm:
                ck = dict(arm or {})
                if mismatch_uagg:
                    ck["u_agg"] = "local"
                ckpt["a5_arm"] = ck
            if migration:
                ckpt["a5_proto_migration"] = {"from": "legacy"}
            (run_dir / "checkpoints").mkdir(exist_ok=True)
            torch.save(ckpt, run_dir / "checkpoints" / "latest.pth")

    good_arm = {"a2_arm": "anchor", "proto_ce": "on", "proto_lambda": 0.5,
                "proto_tau": 0.07, "u_agg": "fedavg", "rank": 8}
    make_run("fedta_a5", "clean", arm=good_arm)               # accepted
    make_run("fedta_a5", "migrated", arm=good_arm, migration=True)
    make_run("fedta_a5", "arm_mismatch", arm=good_arm, mismatch_uagg=True)
    make_run("fedta_a5", "pre_pr3", no_arm=True)
    make_run("fedta", "plain", make_ckpt=False)               # accepted

    out_csv = root / "out.csv"
    report_json = root / "report.json"
    proc = subprocess.run(
        [sys.executable, "diagnostics/aggregate_results.py",
         "--roots", str(root), "--out", str(out_csv),
         "--report", str(report_json)],
        cwd=str(REPO_ROOT), capture_output=True, text=True)
    assert proc.returncode == 0, (
        f"N6: aggregator failed:\n{proc.stdout}\n{proc.stderr}")
    with open(report_json) as f:
        report = json.load(f)
    status = {(r["method"], r["run_name"]): r["status"]
              for r in report["runs"]}
    assert status[("fedta_a5", "clean")] == "accepted"
    assert status[("fedta", "plain")] == "accepted"
    rejected = [k for k, v in status.items() if v == "rejected"]
    assert sorted(k[1] for k in rejected) == [
        "arm_mismatch", "migrated", "pre_pr3"], f"N6: wrong set: {status}"
    reasons = {r["run_name"]: r.get("reason", "")
               for r in report["runs"] if r["status"] == "rejected"}
    assert "a5_proto_migration" in reasons["migrated"]
    assert "a5_arm descriptor disagrees" in reasons["arm_mismatch"]
    assert "no a5_arm descriptor" in reasons["pre_pr3"]
    with open(out_csv) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2 and all(r["method"] in ("fedta_a5", "fedta")
                                  for r in rows)
    print("[PASS] N6: migration/arm-mismatch/pre-PR-3 runs rejected; "
          "clean runs aggregated")


def main():
    a = parse_args()
    if a.work_dir.exists():
        shutil.rmtree(a.work_dir)
    a.work_dir.mkdir(parents=True)

    n0_validation()
    fed_ckpt = n1_default_fedavg(a)
    loc_ckpt = n2_local_arm(a)
    n3_local_resume(a, loc_ckpt)
    n4_pre_pr3_compat(a, fed_ckpt)
    n5_ce_only(a)
    n6_aggregator(a)

    print("\nRESULT: PASS (PR-3 four-arm switches: u_agg local/fedavg "
          "fail-closed + arm descriptor/banner + CE-only + aggregator "
          "migration exclusion)")


if __name__ == "__main__":
    main()
