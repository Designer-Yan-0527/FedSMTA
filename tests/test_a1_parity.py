"""A1 <-> 2x2 parity test (GPU, short runs).

Verifies that a1_snapshot_eval.py reproduces the 2x2 grid of
compatibility_drift.py bit-for-bit on the same run:

  A1 (client, old task t, gen t)  ==  2x2 (client, t, prompt=current, ka=old)
  A1 (client, old task t, cur)    ==  2x2 (client, t, prompt=current, ka=current)

Both scripts use the fixed seed 9700 + 100*client + task and the identical
evaluation path (explicit prompt load -> KA load -> seed ->
evaluate_with_margin), so accuracies must match EXACTLY (string compare on
the %.2f CSV values).

Usage (from repo root, on the server):
  python tests/test_a1_parity.py --data-path <DATA_PATH>
"""

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser("A1 vs 2x2 parity test")
    p.add_argument("--data_name", default="cifar100")
    p.add_argument("--data_path", default="./local_datasets")
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--device", default="cuda")
    p.add_argument("--global_epoch", default=2, type=int)
    p.add_argument("--task_num", default=2, type=int)
    p.add_argument("--local_epoch", default=1, type=int)
    p.add_argument("--work_dir", default="output/regression_a1", type=Path)
    return p.parse_args()


def run_cmd(cmd, label):
    print("+", " ".join(map(str, cmd)), flush=True)
    proc = subprocess.run(list(map(str, cmd)), cwd=str(REPO_ROOT))
    if proc.returncode != 0:
        raise RuntimeError(f"{label}: command failed")


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    args = parse_args()
    run_dir = (args.work_dir / args.data_name / "fedta" / "a1_parity"
               / f"seed_{args.seed}")
    if run_dir.exists():
        print(f"[cleanup] removing previous test run dir {run_dir}")
        shutil.rmtree(run_dir)

    # 1) short fedta run producing task_00_end.pth + latest.pth
    run_cmd([sys.executable, "main.py", "datasets_delay",
             "--data_name", args.data_name, "--data-path", args.data_path,
             "--seed", str(args.seed), "--device", args.device,
             "--global_epoch", str(args.global_epoch),
             "--task_num", str(args.task_num),
             "--local_epoch", str(args.local_epoch),
             "--method", "fedta", "--deterministic", "--save_every", "1",
             "--output_dir", str(args.work_dir),
             "--run_name", "a1_parity"], "fedta run")

    out_dir = args.work_dir / "a1_parity_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_2x2 = out_dir / "a1_parity_2x2.csv"
    csv_a1 = out_dir / "a1_parity_a1.csv"

    # 2) 2x2 diagnostic: old = task 0 end, new = latest
    run_cmd([sys.executable, "diagnostics/compatibility_drift.py",
             "--run_dir", run_dir,
             "--old_ckpt", "checkpoints/task_00_end.pth",
             "--new_ckpt", "checkpoints/latest.pth",
             "--out_csv", csv_2x2,
             "--data_path", args.data_path,
             "--device", args.device], "compatibility_drift")

    # 3) A1 snapshot evaluation on the same run
    run_cmd([sys.executable, "diagnostics/a1_snapshot_eval.py",
             "--run_dir", run_dir,
             "--cur_ckpt", "checkpoints/latest.pth",
             "--out_csv", csv_a1,
             "--data_path", args.data_path,
             "--device", args.device], "a1_snapshot_eval")

    # 4) compare
    rows_2x2 = read_rows(csv_2x2)
    rows_a1 = read_rows(csv_a1)

    grid = {(int(r["client_id"]), int(r["old_task"]), r["prompt"], r["key_anchor"]):
            r["accuracy"] for r in rows_2x2}
    a1 = {(int(r["client_id"]), int(r["task"]), r["ka_generation"]):
          r["accuracy"] for r in rows_a1}

    checked = 0
    for (cid, task, gen), acc in sorted(a1.items()):
        if gen == "cur":
            continue  # current task deployment row has no 2x2 counterpart
        ref = grid.get((cid, task, "current", "old"))
        assert ref is not None, (
            f"client {cid} task {task}: 2x2 (current, old) row missing")
        assert acc == ref, (
            f"client {cid} task {task}: A1 gen{task} acc {acc} != "
            f"2x2 (current, old) {ref}")
        checked += 1
    for (cid, task, gen), acc in sorted(a1.items()):
        # A1 baseline rows (ka_generation == 'cur' on OLD tasks) map to
        # 2x2 (current, current); 'cur' on the CURRENT task has no mapping
        if gen != "cur":
            continue
        ref = grid.get((cid, task, "current", "current"))
        if ref is None:
            continue
        assert acc == ref, (
            f"client {cid} task {task}: A1 baseline acc {acc} != "
            f"2x2 (current, current) {ref}")
        checked += 1

    # false-pass guard: both old tasks and current task must be present
    tasks_in_a1 = {t for (_, t, _) in a1}
    assert tasks_in_a1 == set(range(args.task_num)), (
        f"A1 rows must cover all tasks 0..{args.task_num - 1}, "
        f"got {sorted(tasks_in_a1)}")
    assert checked > 0, "no comparable rows found (false-pass guard)"
    print(f"\nRESULT: PASS ({checked} A1/2x2 rows bit-identical, "
          f"tasks covered: {sorted(tasks_in_a1)})")


if __name__ == "__main__":
    main()
