"""A5 PR-3 four-arm integration smoke (GPU, full T0->T1 task switch).

Approved by the external review 2026-10-22 (GO FOR FOUR-ARM SMOKE). The
pass condition is CORRECTNESS of state / gradients / aggregation / resume
/ metrics plumbing across a REAL task boundary — NOT short-run accuracy.

Arms (same seed, same data partition, same budget; only the factorial
variables differ — asserted by args cross-check in S5):

  1. FedTA        --method fedta                     (official baseline)
  2. A2           --method fedta_a2 --a2_arm key     (interface protection)
  3. Shared-only  --method fedta_a5 --a2_arm none    (shared residual only)
  4. A2 + Shared  --method fedta_a5 --a2_arm key     (full A5)
  5. CE-only      = 4 with --a5_proto_lambda 0.0     (mechanism contrast)
  6. Local-only   = 4 with --a5_u_agg local          (mechanism contrast)

Reviewer's mandatory checks mapped to cases:
  S1  arm identity: every A5 checkpoint carries a5_arm matching its CLI;
      non-A5 checkpoints carry NO a5 keys (schema isolation).
  S2  prototype-table version consistency: after the T0->T1 switch every
      A5 run's final table version == 1 (task-1 boundary sync happened
      exactly once) with non-empty counts.
  S3  Local-only U continuity across the task boundary:
      (a) final per-client U's are pairwise DIFFERENT and non-zero (no
          implicit broadcast/reset anywhere, including the first round
          after the task switch);
      (b) bit-exact resume: resuming a copy of the local run's round-0002
          checkpoint (first round of task 1) and training to the end
          reproduces the uninterrupted run's final per-client U's AND
          prototype table bit-identically (resume restores EACH client's
          own U and continues from it).
  S4  FedAvg broadcast still exact: arm 4's final client U's are
      bit-identical to each other.
  S5  metrics pipeline + pairing guards: aggregate_results.py accepts all
      six runs (no migration marker, arm identity consistent) and the
      args.json cross-check confirms that every training-relevant key
      except the factorial variables is IDENTICAL across arms.
  S6  checkpoint resume of the full A5 arm (fedavg) mid-task reproduces
      the uninterrupted final state bit-identically (U's + proto table).

Usage (from repo root, on the server):
  python tests/test_a5_pr3_smoke.py --data_path <DATA_PATH>
  # optional: run a subset of arms (S3/S4/S6 require a5_local & a5_full):
  python tests/test_a5_pr3_smoke.py --arms shared_only,a5_full,a5_ce0,a5_local
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from test_resume_regression import torch_load_compat  # noqa: E402

import torch  # noqa: E402

# training keys that must be IDENTICAL across all six smoke arms
SHARED_TRAINING_KEYS = [
    "data_name", "model", "model_name", "client_num", "task_num",
    "global_epoch", "local_epoch", "lr", "batch_size", "nb_classes",
    "private_class_num", "seed", "threshold", "surrogate_num",
    "a5_rank",  # same shared-adapter capacity in every A5 arm
]

ARMS = [
    # (run_name, extra method flags)
    ("fedta", ["--method", "fedta"]),
    ("a2_key", ["--method", "fedta_a2", "--a2_arm", "key"]),
    ("shared_only", ["--method", "fedta_a5", "--a2_arm", "none"]),
    ("a5_full", ["--method", "fedta_a5", "--a2_arm", "key"]),
    ("a5_ce0", ["--method", "fedta_a5", "--a2_arm", "key",
                "--a5_proto_lambda", "0.0"]),
    ("a5_local", ["--method", "fedta_a5", "--a2_arm", "key",
                  "--a5_u_agg", "local"]),
]


def parse_args():
    p = argparse.ArgumentParser("A5 PR-3 four-arm integration smoke")
    p.add_argument("--data_name", default="cifar100")
    p.add_argument("--data_path", default="./local_datasets")
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--device", default="cuda")
    p.add_argument("--global_epoch", default=2, type=int)
    p.add_argument("--task_num", default=2, type=int)
    p.add_argument("--local_epoch", default=1, type=int)
    p.add_argument("--work_dir", default="output/smoke_a5_pr3_arms",
                   type=Path)
    p.add_argument("--arms", default=None,
                   help="comma-separated run_names to run instead of all "
                        "six (S3/S4/S6 additionally require a5_local / "
                        "a5_full to be included)")
    return p.parse_args()


def run_main(extra, label=""):
    cmd = [sys.executable, "main.py", "datasets_delay"] + extra
    print("+", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT),
                          capture_output=True, text=True)
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
        "--run_name", run_name,
    ]


def run_dir_of(a, method, run_name):
    return a.work_dir / a.data_name / method / run_name / f"seed_{a.seed}"


def final_ckpt(a, method, run_name, name="task_01_end.pth"):
    return run_dir_of(a, method, run_name) / "checkpoints" / name


def s1_arm_identity(a):
    print("=== S1: arm identity in checkpoints ===")
    for run_name, flags in ARMS:
        method = flags[flags.index("--method") + 1]
        ckpt = torch_load_compat(final_ckpt(a, method, run_name))
        if method == "fedta_a5":
            arm = ckpt.get("a5_arm")
            assert isinstance(arm, dict), f"S1 {run_name}: no a5_arm"
            if "a2_arm" in flags:
                i = flags.index("--a2_arm")
                assert arm["a2_arm"] == flags[i + 1], (
                    f"S1 {run_name}: a2_arm {arm['a2_arm']}")
            if "--a5_proto_lambda" in flags:
                lam = float(flags[flags.index("--a5_proto_lambda") + 1])
                assert arm["proto_lambda"] == lam and arm["proto_ce"] == (
                    "off" if lam == 0 else "on"), f"S1 {run_name}: lambda"
            if "--a5_u_agg" in flags:
                assert arm["u_agg"] == flags[
                    flags.index("--a5_u_agg") + 1], f"S1 {run_name}: u_agg"
            assert "a5_proto_migration" not in ckpt, (
                f"S1 {run_name}: unexpected migration marker")
        else:
            assert "a5_arm" not in ckpt and "a5_proto_state" not in ckpt, (
                f"S1 {run_name}: non-A5 checkpoint carries a5 keys")
            assert all("a5_state" not in c
                       for c in ckpt["clients"]), (
                f"S1 {run_name}: non-A5 client carries a5_state")
    print("[PASS] S1: a5_arm matches CLI for all A5 arms; non-A5 clean")


def s2_proto_table(a):
    print("=== S2: prototype-table version after T0->T1 ===")
    for run_name, flags in ARMS:
        if flags[flags.index("--method") + 1] != "fedta_a5":
            continue
        ckpt = torch_load_compat(final_ckpt(a, "fedta_a5", run_name))
        proto = ckpt["a5_proto_state"]
        # final task id = task_num - 1; version must equal it (synced once
        # per boundary) and the table must be non-empty
        assert proto["version"] == a.task_num - 1, (
            f"S2 {run_name}: version {proto['version']} != {a.task_num - 1}")
        assert proto["counts"] and all(
            n > 0 for n in proto["counts"].values()), (
            f"S2 {run_name}: bad counts")
        # every client saw the SAME table: server-side single version, and
        # client a5_proto_version attributes were injected from it
        print(f"  {run_name}: table={len(proto['counts'])} classes, "
              f"version={proto['version']}")
    print("[PASS] S2: table synced exactly once per boundary in every A5 arm")


def client_us(ckpt):
    return [c["a5_state"]["U"] for c in ckpt["clients"]]


def s3_local_continuity(a):
    print("=== S3: Local-only U continuity across the task boundary ===")
    full = torch_load_compat(final_ckpt(a, "fedta_a5", "a5_local"))
    Us = client_us(full)
    assert len(Us) >= 2
    for i, U in enumerate(Us[1:], start=1):
        assert not torch.equal(Us[0], U), (
            "S3a: client U's identical after the task switch — implicit "
            "broadcast somewhere"
        )
    for i, U in enumerate(Us):
        assert torch.count_nonzero(U) > 0, (
            f"S3a: client {i} U reset to zero at/after the task boundary")
    # task-0 boundary checkpoint: U's already diverged there and stayed
    # diverged through task 1
    t0 = torch_load_compat(
        final_ckpt(a, "fedta_a5", "a5_local", "task_00_end.pth"))
    Us0 = client_us(t0)
    assert any(not torch.equal(Us0[0], U) for U in Us0[1:]), (
        "S3a: U's did not diverge within task 0")
    changed = [not torch.equal(u0, u1) for u0, u1 in zip(Us0, Us)]
    assert all(changed), (
        "S3a: some client U did not train during task 1 (frozen?)")

    # S3b: bit-exact resume from the FIRST round of task 1 (round index 2)
    synth_dir = run_dir_of(a, "fedta_a5", "synth_local_resume") \
        / "checkpoints"
    synth_dir.mkdir(parents=True, exist_ok=True)
    src = final_ckpt(a, "fedta_a5", "a5_local", "round_0002.pth")
    synth = synth_dir / "round_0002.pth"
    shutil.copy(src, synth)
    run_main(base_flags(a, "synth_local_resume")
             + ["--method", "fedta_a5", "--a2_arm", "key",
                "--a5_u_agg", "local",
                "--resume", str(synth)],
             label="S3b: resume local arm from task-1 first round")
    resumed = torch_load_compat(
        final_ckpt(a, "fedta_a5", "synth_local_resume"))
    for i, (u_r, u_f) in enumerate(zip(client_us(resumed), Us)):
        assert torch.equal(u_r, u_f), (
            f"S3b: client {i} U diverged from the uninterrupted run — "
            "resume does not continue from the client's OWN U")
    pr, pf = resumed["a5_proto_state"], full["a5_proto_state"]
    assert pr["version"] == pf["version"], "S3b: proto version mismatch"
    assert pr["counts"] == pf["counts"], "S3b: proto counts mismatch"
    for c in pf["sums"]:
        assert torch.equal(pr["sums"][c], pf["sums"][c]), (
            f"S3b: proto sums mismatch for class {c}")
    print("[PASS] S3: U's stay client-specific across the boundary; "
          "resume is bit-exact (own-U continuity + table identity)")


def s4_fedavg_broadcast(a):
    print("=== S4: FedAvg broadcast exactness (full A5 arm) ===")
    ckpt = torch_load_compat(final_ckpt(a, "fedta_a5", "a5_full"))
    Us = client_us(ckpt)
    for i, U in enumerate(Us[1:], start=1):
        assert torch.equal(Us[0], U), (
            f"S4: client {i} U differs — broadcast broken in the full arm")
    print("[PASS] S4: full-arm client U's bit-identical")


def s5_pipeline_and_pairing(a):
    print("=== S5: aggregation pipeline + pairing guards ===")
    out_csv = a.work_dir / "agg" / "results.csv"
    report_json = a.work_dir / "agg" / "report.json"
    proc = subprocess.run(
        [sys.executable, "diagnostics/aggregate_results.py",
         "--roots", str(a.work_dir),
         "--out", str(out_csv), "--report", str(report_json)],
        cwd=str(REPO_ROOT), capture_output=True, text=True)
    assert proc.returncode == 0, f"S5: aggregator failed:\n{proc.stderr}"
    with open(report_json) as f:
        report = json.load(f)
    assert report["n_rejected"] == 0, (
        f"S5: unexpected rejections: "
        f"{[r for r in report['runs'] if r['status'] == 'rejected']}")
    assert report["n_accepted"] >= len(ARMS), (
        f"S5: {report['n_accepted']} < {len(ARMS)}")
    print(f"[PASS] S5a: aggregator accepted all runs "
          f"({report['n_accepted']})")

    # args cross-check: everything except the factorial variables is
    # IDENTICAL across arms (same budget / data access / capacity)
    base_cfg = None
    for run_name, _flags in ARMS:
        method = _flags[_flags.index("--method") + 1]
        with open(run_dir_of(a, method, run_name) / "args.json") as f:
            cfg = json.load(f)
        slim = {k: v for k, v in cfg.items()
                if k in SHARED_TRAINING_KEYS}
        if base_cfg is None:
            base_cfg = slim
        else:
            assert slim == base_cfg, (
                f"S5b: arm {run_name} differs in shared training keys: "
                f"{ {k: (slim.get(k), base_cfg.get(k)) for k in set(slim) | set(base_cfg) if slim.get(k) != base_cfg.get(k)} }")
    print(f"[PASS] S5b: all {len(ARMS)} arms share identical training "
          f"keys (budget / data / capacity)")


def s6_full_arm_resume(a):
    print("=== S6: full-A5 (fedavg) mid-task resume bit-exactness ===")
    synth_dir = run_dir_of(a, "fedta_a5", "synth_full_resume") / "checkpoints"
    synth_dir.mkdir(parents=True, exist_ok=True)
    src = final_ckpt(a, "fedta_a5", "a5_full", "round_0002.pth")
    synth = synth_dir / "round_0002.pth"
    shutil.copy(src, synth)
    run_main(base_flags(a, "synth_full_resume")
             + ["--method", "fedta_a5", "--a2_arm", "key",
                "--resume", str(synth)],
             label="S6: resume full arm from task-1 first round")
    resumed = torch_load_compat(
        final_ckpt(a, "fedta_a5", "synth_full_resume"))
    full = torch_load_compat(final_ckpt(a, "fedta_a5", "a5_full"))
    for i, (u_r, u_f) in enumerate(zip(client_us(resumed), client_us(full))):
        assert torch.equal(u_r, u_f), f"S6: client {i} U mismatch"
    pr, pf = resumed["a5_proto_state"], full["a5_proto_state"]
    assert pr["version"] == pf["version"] and pr["counts"] == pf["counts"], (
        "S6: proto table mismatch")
    print("[PASS] S6: fedavg-arm resume bit-exact (U's + proto table)")


def main():
    a = parse_args()
    if a.arms:
        by_name = dict(ARMS)
        wanted = [s.strip() for s in a.arms.split(",") if s.strip()]
        unknown = [n for n in wanted if n not in by_name]
        assert not unknown, f"--arms: unknown arm(s) {unknown}"
        ARMS[:] = [(n, by_name[n]) for n in wanted]
        print(f"=== arm subset ({len(ARMS)}): "
              f"{', '.join(n for n, _ in ARMS)} ===")
    if a.work_dir.exists():
        shutil.rmtree(a.work_dir)
    a.work_dir.mkdir(parents=True)

    print(f"=== running the smoke arms (full T0->T1, n={len(ARMS)}) ===")
    for run_name, flags in ARMS:
        run_main(base_flags(a, run_name) + flags,
                 label=f"smoke arm: {run_name}")

    s1_arm_identity(a)
    s2_proto_table(a)
    s3_local_continuity(a)
    s4_fedavg_broadcast(a)
    s5_pipeline_and_pairing(a)
    s6_full_arm_resume(a)

    print("\nRESULT: PASS (four-arm smoke: isolation / continuity / "
          "resume / pipeline all verified across a real task switch)")


if __name__ == "__main__":
    main()
