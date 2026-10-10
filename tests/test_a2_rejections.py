"""A2 fail-closed rejection tests (GPU, short runs).

Verifies the two rejection paths required by the A2 resume discipline:

  R1. arm mismatch: resuming an `anchor`-arm checkpoint with `--a2_arm key`
      must FAIL (RuntimeError) instead of silently changing arm semantics.
  R2. missing a2_state: resuming a fedta_a2 run from a checkpoint WITHOUT
      a2_state (e.g. a plain fedta checkpoint) must FAIL (RuntimeError)
      instead of silently continuing as unprotected A2.

Also sanity-checks that the short fedta_a2 run actually PRODUCED a
non-empty a2_state once it entered task 1 (protection set computed).

Usage (from repo root, on the server):
  python tests/test_a2_rejections.py --data-path <DATA_PATH>
  python tests/test_a2_rejections.py --data-path <DATA_PATH> --device cpu
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tests"))

from test_resume_regression import torch_load_compat  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser("A2 fail-closed rejection tests")
    p.add_argument("--data_name", default="cifar100")
    p.add_argument("--data_path", default="./local_datasets")
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--device", default="cuda")
    p.add_argument("--global_epoch", default=2, type=int)
    p.add_argument("--task_num", default=2, type=int)
    p.add_argument("--local_epoch", default=1, type=int)
    p.add_argument("--rounds", default=3, type=int,
                   help="must reach >= 1 round of task 1 so a protection "
                        "set exists (task 1 starts at round global_epoch)")
    p.add_argument("--work_dir", default="output/regression_a2", type=Path)
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
    return proc


def main():
    args = parse_args()
    assert args.rounds > args.global_epoch, (
        "--rounds must reach task 1 (i.e. be > global_epoch) so that a "
        "protection set exists in the checkpoint")

    base = ["--data_name", args.data_name, "--data-path", args.data_path,
            "--seed", str(args.seed), "--device", args.device,
            "--global_epoch", str(args.global_epoch),
            "--task_num", str(args.task_num),
            "--local_epoch", str(args.local_epoch),
            "--deterministic", "--save_every", "1",
            "--output_dir", str(args.work_dir)]

    run_a2 = args.work_dir / args.data_name / "fedta_a2" / "rej_a2" / f"seed_{args.seed}"
    run_ft = args.work_dir / args.data_name / "fedta" / "rej_fedta" / f"seed_{args.seed}"
    for run in (run_a2, run_ft):
        if run.exists():
            print(f"[cleanup] removing previous test run dir {run}")
            shutil.rmtree(run)

    # 1) short fedta_a2 anchor run crossing into task 1
    run_main(base + ["--method", "fedta_a2", "--a2_arm", "anchor",
                     "--run_name", "rej_a2", "--max_rounds", str(args.rounds)],
             label="fedta_a2 anchor run")

    # sanity: a2_state exists and carries a non-empty protection set
    ckpt = torch_load_compat(run_a2 / "checkpoints" / "latest.pth")
    for c in ckpt["clients"]:
        a2 = c.get("a2_state")
        assert isinstance(a2, dict) and a2, (
            f"client {c['id']}: a2_state missing after task-1 entry "
            "(false-pass guard)")
        assert a2["a2_arm"] == "anchor"
        assert a2["task_id"] == c["task_id"], (
            f"client {c['id']}: a2 task_id {a2['task_id']} != "
            f"checkpoint task_id {c['task_id']}")
        assert a2["protected_slots"], (
            f"client {c['id']}: expected non-empty protected_slots after "
            "task-1 entry (D0 support should be non-empty)")
        assert a2["anchor_snapshot"] is not None and \
            a2["anchor_snapshot"].shape[0] == len(a2["protected_slots"])
        assert a2["key_snapshot"] is None, (
            f"client {c['id']}: anchor arm must not snapshot key rows")
    print(f"[PASS] a2_state present, non-empty, arm-consistent "
          f"({len(ckpt['clients'])} clients)")

    # 2) short fedta run (no a2_state in checkpoints)
    run_main(base + ["--method", "fedta", "--run_name", "rej_fedta",
                     "--max_rounds", str(args.rounds)],
             label="fedta run")
    ckpt_ft = torch_load_compat(run_ft / "checkpoints" / "latest.pth")
    for c in ckpt_ft["clients"]:
        assert "a2_state" not in c, (
            f"client {c['id']}: fedta baseline checkpoint must NOT carry "
            "the a2_state key (schema invariance)")

    # R1: arm mismatch on resume -> must fail
    run_main(base + ["--method", "fedta_a2", "--a2_arm", "key",
                     "--run_name", "rej_a2", "--resume", "auto",
                     "--max_rounds", str(args.rounds)],
             expect_fail=True, expect_msg="arm mismatch",
             label="R1 arm-mismatch resume")

    # R2: fedta_a2 resume from a fedta checkpoint (no a2_state) -> must fail
    run_main(base + ["--method", "fedta_a2", "--a2_arm", "anchor",
                     "--run_name", "rej_fedta",
                     "--resume", str(run_ft / "checkpoints" / "latest.pth"),
                     "--max_rounds", str(args.rounds)],
             expect_fail=True, expect_msg="a2_state",
             label="R2 missing-a2_state resume")

    print("\nRESULT: PASS (all rejection paths enforced)")


if __name__ == "__main__":
    main()
