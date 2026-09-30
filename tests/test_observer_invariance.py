"""Phase 0 observer-invariance test.

Proves that the Phase-0 instrumentation (train_log.csv local-phase
evaluation, full seen-task evaluation, margin recording, metrics CSVs)
acts as a PURE OBSERVER: with instrumentation ON vs OFF
(--no_instrumentation), the training trajectory must be bit-identical.

Runs two short training runs:
  A) --no_instrumentation (official FedTA loop, no extra evaluations)
  B) default (Phase-0 instrumentation ON)

and compares the final checkpoints:
  server model / global head / global protos / server prompt / fix_keys,
  per-client heads / tail-anchor key / anchor_pool / prompts /
  local+global protos, RNG states, inner 70/30 split indices (including
  the NEXT-TASK split: --rounds defaults to global_epoch + 1 so the run
  enters task 1 and its data split is compared too), and the federated
  outer split manifest.

Note: this complements test_resume_regression.py, which checks
continuous-vs-resume equivalence WITHIN the current code. This test
checks instrumentation-OFF vs instrumentation-ON, i.e. that the
observers themselves do not perturb the training random stream.

Writes metrics/observer_invariance.json under --work_dir.

Usage (ON THE SERVER, from the repo root):

  python tests/test_observer_invariance.py --data_name cifar100 \
      --data-path ./local_datasets --rounds 3 \
      --global_epoch 2 --task_num 2 --local_epoch 2
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

from test_resume_regression import (
    REPO_ROOT, max_diff, state_dict_diff, protos_diff, deep_equal, hash_obj,
    require_outer_manifest,
)


def parse_args():
    p = argparse.ArgumentParser("observer-invariance test")
    p.add_argument("--data_name", default="cifar100",
                   choices=["cifar100", "ImageNet-R"])
    p.add_argument("--data_path", default="./local_datasets")
    p.add_argument("--rounds", default=0, type=int,
                   help="total rounds (0 = global_epoch + 1, so the run "
                        "enters task 1 and the next-task split is compared)")
    p.add_argument("--global_epoch", default=2, type=int)
    p.add_argument("--task_num", default=2, type=int)
    p.add_argument("--local_epoch", default=2, type=int,
                   help="keep small: this only tests observer invariance")
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--device", default="cuda")
    p.add_argument("--work_dir", default="output/regression", type=Path)
    p.add_argument("--method", default="fedta")
    return p.parse_args()


def run_main(extra):
    cmd = [sys.executable, "main.py", "datasets_delay"] + extra
    print("+", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if proc.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}")


def client_prompt_diff(clients_a, clients_b):
    diffs = []
    for ca, cb in zip(clients_a, clients_b):
        pa, pb = ca.get("prompts"), cb.get("prompts")
        if (pa is None) != (pb is None):
            return float("inf")
        if pa is not None:
            diffs.append(state_dict_diff(pa, pb))
    return max(diffs) if diffs else 0.0


def main():
    args = parse_args()
    rounds = args.rounds if args.rounds > 0 else args.global_epoch + 1
    if rounds > args.task_num * args.global_epoch:
        raise SystemExit(
            f"--rounds {rounds} exceeds task_num*global_epoch "
            f"({args.task_num * args.global_epoch})")

    base = ["--data_name", args.data_name, "--data-path", args.data_path,
            "--seed", str(args.seed), "--device", args.device,
            "--global_epoch", str(args.global_epoch),
            "--task_num", str(args.task_num),
            "--local_epoch", str(args.local_epoch),
            "--method", args.method, "--deterministic",
            "--save_every", "1", "--output_dir", str(args.work_dir),
            "--max_rounds", str(rounds)]

    run_off = args.work_dir / args.data_name / args.method / "obs_off" / f"seed_{args.seed}"
    run_on = args.work_dir / args.data_name / args.method / "obs_on" / f"seed_{args.seed}"

    # clean previous test runs: stale checkpoints would trigger auto-resume
    # and stale metrics would corrupt the artifact checks below
    for run in (run_off, run_on):
        if run.exists():
            print(f"[cleanup] removing previous test run dir {run}")
            shutil.rmtree(run)

    # A: official loop, Phase-0 observers disabled
    run_main(base + ["--run_name", "obs_off", "--no_instrumentation"])
    # B: Phase-0 instrumentation ON (default)
    run_main(base + ["--run_name", "obs_on"])

    ckpt_off = torch.load(run_off / "checkpoints" / "latest.pth",
                          map_location="cpu", weights_only=False)
    ckpt_on = torch.load(run_on / "checkpoints" / "latest.pth",
                         map_location="cpu", weights_only=False)

    # Gate 0C seal: prove run A really ran with instrumentation OFF and
    # run B with ON — otherwise the comparison below is vacuous.
    assert ckpt_off["args"].get("no_instrumentation") is True, (
        "obs_off checkpoint must record no_instrumentation=True "
        "(was the run actually started with --no_instrumentation?)")
    assert ckpt_on["args"].get("no_instrumentation") is False, (
        "obs_on checkpoint must record no_instrumentation=False")

    # ---------- training-trajectory comparison ----------
    head_diffs = [state_dict_diff(ckpt_off["global_head"],
                                  ckpt_on["global_head"])]
    key_diffs, anchor_diffs = [], []
    for c_off, c_on in zip(ckpt_off["clients"], ckpt_on["clients"]):
        for h_off, h_on in zip(c_off["heads"], c_on["heads"]):
            head_diffs.append(state_dict_diff(h_off, h_on))
        ta_off, ta_on = c_off["tail_anchor_model"], c_on["tail_anchor_model"]
        key_diffs.append(max_diff(ta_off.get("key"), ta_on.get("key")))
        anchor_diffs.append(max_diff(ta_off.get("anchor_pool"),
                                     ta_on.get("anchor_pool")))

    inner_off, inner_on = (ckpt_off.get("data_split_state") or {},
                           ckpt_on.get("data_split_state") or {})
    # Gate 0C seal: fail-fast on a missing/empty manifest (no {} == {} pass)
    outer_off = require_outer_manifest(ckpt_off, str(run_off))
    outer_on = require_outer_manifest(ckpt_on, str(run_on))

    # Gate 0C seal: instrumentation artifacts must actually differ — the
    # OFF run must produce NONE of the Phase-0 observer files, the ON run
    # must produce them (proves the flag really disables the observers).
    # accuracy_matrix.csv / margins.csv are only expected once the run has
    # reached a task boundary (run_full_evaluation is task-end gated).
    reached_task_end = rounds >= args.global_epoch
    observer_artifacts = ["train_log.csv", "round_metrics.jsonl"]
    if reached_task_end:
        observer_artifacts += ["accuracy_matrix.csv", "margins.csv"]
    artifacts_off = {n: (run_off / "metrics" / n).exists()
                     for n in observer_artifacts}
    artifacts_on = {n: (run_on / "metrics" / n).exists()
                    for n in observer_artifacts}
    for n in observer_artifacts:
        assert not artifacts_off[n], (
            f"OFF run must not produce metrics/{n} "
            "(instrumentation is not fully disabled)")
        assert artifacts_on[n], (
            f"ON run must produce metrics/{n} "
            "(instrumentation artifacts missing)")

    comparison = {
        "completed_round_diff": abs(int(ckpt_off["completed_round"])
                                    - int(ckpt_on["completed_round"])),
        "server_model_max_diff": state_dict_diff(ckpt_off["server_model"],
                                                 ckpt_on["server_model"]),
        "global_proto_max_diff": protos_diff(ckpt_off["global_protos"],
                                             ckpt_on["global_protos"]),
        "prompt_max_diff": state_dict_diff(ckpt_off["server_prompt"],
                                           ckpt_on["server_prompt"]),
        "client_prompt_max_diff": client_prompt_diff(ckpt_off["clients"],
                                                     ckpt_on["clients"]),
        "head_max_diff": max(head_diffs),
        "key_max_diff": max(key_diffs),
        "anchor_max_diff": max(anchor_diffs),
        # inner 70/30 splits; includes the NEXT-task split because rounds
        # runs into task 1
        "split_equal": bool(deep_equal(inner_off, inner_on)),
        "outer_split_equal": bool(deep_equal(outer_off, outer_on)),
        "fix_keys_equal": sorted(map(str, ckpt_off["fix_keys"]))
                          == sorted(map(str, ckpt_on["fix_keys"])),
    }

    rng_off, rng_on = ckpt_off["rng_state"], ckpt_on["rng_state"]
    rng_diffs = {
        "torch": max_diff(rng_off["torch"], rng_on["torch"]),
        "numpy": 0.0 if rng_off["numpy"] == rng_on["numpy"] else float("inf"),
        "python": 0.0 if rng_off["python"] == rng_on["python"] else float("inf"),
    }
    if rng_off.get("cuda") is not None:
        rng_diffs["cuda"] = max(max_diff(x, y)
                                for x, y in zip(rng_off["cuda"], rng_on["cuda"]))

    # ---------- pass/fail report ----------
    report = []
    for name, diff in comparison.items():
        if name in ("split_equal", "outer_split_equal", "fix_keys_equal"):
            report.append((name, diff is True, 0.0 if diff else float("inf"), ""))
        else:
            report.append((name, diff == 0.0, diff, ""))
    for name, d in rng_diffs.items():
        report.append((f"rng.{name}", d == 0.0, d, ""))
    for n in observer_artifacts:
        report.append((f"artifact_off_absent[{n}]", not artifacts_off[n],
                       0.0 if not artifacts_off[n] else float("inf"), ""))
        report.append((f"artifact_on_present[{n}]", artifacts_on[n],
                       0.0 if artifacts_on[n] else float("inf"), ""))

    print("\n===== observer-invariance report =====")
    all_ok = True
    for name, ok, diff, note in report:
        status = "PASS" if ok else "FAIL"
        all_ok &= ok
        print(f"[{status}] {name:26s} max_diff={diff}"
              f"{('  (' + note + ')') if note else ''}")

    result = "PASS" if all_ok else "FAIL"

    def records(ckpt, run_dir):
        return {
            "run_dir": str(run_dir),
            "completed_round": int(ckpt["completed_round"]),
            "outer_split_hash": hash_obj(ckpt.get("outer_split_manifest")),
            "inner_split_hash": hash_obj(ckpt.get("data_split_state")),
            "global_protos_checksum": hash_obj(ckpt.get("global_protos")),
            "server_prompt_checksum": hash_obj(ckpt.get("server_prompt")),
            "rng_checksum": hash_obj(ckpt.get("rng_state")),
        }

    verification = {
        "test": "instrumentation_off_vs_on_invariance",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "meaning": "Phase-0 observers (train_log local-phase eval, full "
                   "seen-task eval, margin recording) must not perturb the "
                   "training trajectory",
        "result": result,
        "config": {
            "data_name": args.data_name, "method": args.method,
            "seed": args.seed, "rounds": rounds,
            "global_epoch": args.global_epoch,
            "task_num": args.task_num, "local_epoch": args.local_epoch,
            "deterministic": True,
        },
        "run_off": records(ckpt_off, run_off),
        "run_on": records(ckpt_on, run_on),
        "observer_artifacts": {"off": artifacts_off, "on": artifacts_on},
        "comparison": {k: (None if v == float("inf") else v)
                       for k, v in comparison.items()},
        "rng_state_diffs": {k: (None if v == float("inf") else v)
                            for k, v in rng_diffs.items()},
    }
    out_path = Path(args.work_dir) / "metrics" / "observer_invariance.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(verification, f, indent=2, default=str)

    print(f"\nsaved {out_path}")
    print("\nRESULT:", result if all_ok else
          "FAIL (the Phase-0 instrumentation perturbs training; check the "
          "RNG guards in _log_local_phase / run_full_evaluation)")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
