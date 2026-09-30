"""Phase 0 baseline regression test: resume equivalence.

Compares:
  A) one continuous run of N rounds
  B) a run stopped after K rounds, then resumed for the remaining rounds

Final checkpoints (model states / protos / fix_keys / heads / prompts /
RNG states / data-split indices) and the accuracy matrices must match.
Exact bitwise equality needs --deterministic (passed automatically);
without it, small CUDA nondeterminism may appear and the test falls back
to reporting max diffs.

Writes metrics/baseline_verification.json under --work_dir containing the
Phase-0 baseline records for both runs (split hashes, checksums of protos /
head / prompt / keys / anchors / RNG) and the comparison fields:
  accuracy_max_diff, global_proto_max_diff, prompt_max_diff,
  head_max_diff, key_max_diff, anchor_max_diff, split_equal.

Gate 0: Phase 1 is only allowed when result == PASS.

Run ON THE SERVER from the repo root, e.g.:

  python tests/test_resume_regression.py --data_name cifar100 \
      --data_path ./local_datasets --rounds 4 --split_at 2 \
      --global_epoch 2 --task_num 2 --local_epoch 2
"""

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser("resume regression test")
    p.add_argument("--data_name", default="cifar100", choices=["cifar100", "ImageNet-R"])
    p.add_argument("--data_path", default="./local_datasets")
    p.add_argument("--rounds", default=4, type=int, help="total rounds (N)")
    p.add_argument("--split_at", default=2, type=int, help="stop run B after K rounds")
    p.add_argument("--global_epoch", default=2, type=int)
    p.add_argument("--task_num", default=2, type=int)
    p.add_argument("--local_epoch", default=2, type=int,
                   help="keep small: this only tests resume mechanics")
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--device", default="cuda")
    p.add_argument("--work_dir", default="output/regression")
    p.add_argument("--method", default="fedta")
    return p.parse_args()


def run_main(extra):
    cmd = [sys.executable, "main.py", "datasets_delay"] + extra
    print("+", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if proc.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}")


def max_diff(a, b):
    if isinstance(a, torch.Tensor):
        if not isinstance(b, torch.Tensor) or a.shape != b.shape:
            return float("inf")
        if a.dtype in (torch.int64, torch.int32, torch.bool):
            return 0.0 if torch.equal(a, b) else float("inf")
        return float((a.float() - b.float()).abs().max().item())
    return 0.0 if a == b else float("inf")


def state_dict_diff(da, db):
    if da is None and db is None:
        return 0.0
    if da is None or db is None or set(da.keys()) != set(db.keys()):
        return float("inf")
    return max(max_diff(da[k], db[k]) for k in da)


def protos_diff(pa, pb):
    if pa is None and pb is None:
        return 0.0
    if pa is None or pb is None or set(pa.keys()) != set(pb.keys()):
        return float("inf")
    return max(max_diff(torch.as_tensor(pa[k]), torch.as_tensor(pb[k]))
               for k in pa)


def deep_equal(a, b):
    """Recursive equality that handles tensors inside nested dicts/lists."""
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
                and a.dtype == b.dtype and a.shape == b.shape
                and torch.equal(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(deep_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(deep_equal(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def hash_obj(obj):
    """Stable sha256 over tensors / nested dict-list structures."""
    h = hashlib.sha256()

    def feed(x):
        if isinstance(x, torch.Tensor):
            h.update(x.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(x, np.ndarray):
            h.update(np.ascontiguousarray(x).tobytes())
        elif isinstance(x, dict):
            for k in sorted(x.keys(), key=repr):
                feed(x[k])
        elif isinstance(x, (list, tuple)):
            for v in x:
                feed(v)
        else:
            h.update(repr(x).encode("utf-8"))

    feed(obj)
    return h.hexdigest()


def load_matrix(run_dir):
    path = Path(run_dir) / "metrics" / "accuracy_matrix.csv"
    if not path.is_file():
        return None
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


def matrix_diff(ma, mb):
    """Max numeric difference between two accuracy matrices (inf on mismatch)."""
    if ma is None and mb is None:
        return 0.0
    if ma is None or mb is None or len(ma) != len(mb):
        return float("inf")
    worst = 0.0
    for ra, rb in zip(ma, mb):
        if len(ra) != len(rb):
            return float("inf")
        for va, vb in zip(ra, rb):
            try:
                fa = float(va) if va != "" else None
                fb = float(vb) if vb != "" else None
            except ValueError:
                if va != vb:
                    return float("inf")
                continue
            if (fa is None) != (fb is None):
                return float("inf")
            if fa is not None:
                worst = max(worst, abs(fa - fb))
    return worst


def baseline_records(run_dir, ckpt):
    """The Phase-0 baseline verification record of one run."""
    split_state = ckpt.get("data_split_state") or {}
    split_index_hashes = {}
    for cid, tasks in split_state.items():
        per_task = {}
        for t, splits in tasks.items():
            per_task[str(t)] = {k: hash_obj(v) for k, v in splits.items()}
        split_index_hashes[str(cid)] = per_task

    key_cs, anchor_cs, head_cs = {}, {}, {}
    for c in ckpt["clients"]:
        ta = c["tail_anchor_model"]
        key_cs[str(c["id"])] = hash_obj(ta.get("key"))
        anchor_cs[str(c["id"])] = hash_obj(ta.get("anchor_pool"))
        head_cs[str(c["id"])] = hash_obj(c.get("heads"))

    mat_path = Path(run_dir) / "metrics" / "accuracy_matrix.csv"
    mat_hash = (hashlib.sha256(mat_path.read_bytes()).hexdigest()
                if mat_path.is_file() else None)

    return {
        "run_dir": str(run_dir),
        "completed_round": int(ckpt["completed_round"]),
        "outer_split_hash": hash_obj(split_state),
        "split_index_hashes": split_index_hashes,
        "accuracy_matrix_hash": mat_hash,
        "global_protos_checksum": hash_obj(ckpt.get("global_protos")),
        "global_head_checksum": hash_obj(ckpt.get("global_head")),
        "server_prompt_checksum": hash_obj(ckpt.get("server_prompt")),
        "client_task_head_checksums": head_cs,
        "key_checksums": key_cs,
        "anchor_pool_checksums": anchor_cs,
        "rng_checksum": hash_obj(ckpt.get("rng_state")),
    }


def main():
    args = parse_args()
    if args.split_at >= args.rounds:
        raise SystemExit("--split_at must be < --rounds")

    base = ["--data_name", args.data_name, "--data_path", args.data_path,
            "--seed", str(args.seed), "--device", args.device,
            "--global_epoch", str(args.global_epoch),
            "--task_num", str(args.task_num),
            "--local_epoch", str(args.local_epoch),
            "--method", args.method, "--deterministic",
            "--save_every", "1", "--output_dir", args.work_dir]

    run_a = args.work_dir / args.data_name / args.method / "reg_full" / f"seed_{args.seed}"
    run_b = args.work_dir / args.data_name / args.method / "reg_split" / f"seed_{args.seed}"

    # A: continuous N rounds
    run_main(base + ["--run_name", "reg_full", "--max_rounds", str(args.rounds)])
    # B1: first K rounds only
    run_main(base + ["--run_name", "reg_split", "--max_rounds", str(args.split_at)])
    # B2: resume auto -> runs the remaining rounds
    run_main(base + ["--run_name", "reg_split", "--resume", "auto"])

    ckpt_a = torch.load(run_a / "checkpoints" / "latest.pth",
                        map_location="cpu", weights_only=False)
    ckpt_b = torch.load(run_b / "checkpoints" / "latest.pth",
                        map_location="cpu", weights_only=False)

    # ---------- comparison (doc-named fields) ----------
    head_diffs = [state_dict_diff(ckpt_a["global_head"], ckpt_b["global_head"])]
    key_diffs, anchor_diffs = [], []
    for ca, cb in zip(ckpt_a["clients"], ckpt_b["clients"]):
        for ha, hb in zip(ca["heads"], cb["heads"]):
            head_diffs.append(state_dict_diff(ha, hb))
        ta, tb = ca["tail_anchor_model"], cb["tail_anchor_model"]
        key_diffs.append(max_diff(ta.get("key"), tb.get("key")))
        anchor_diffs.append(max_diff(ta.get("anchor_pool"), tb.get("anchor_pool")))

    split_state_a, split_state_b = (ckpt_a.get("data_split_state") or {},
                                    ckpt_b.get("data_split_state") or {})
    split_equal = deep_equal(split_state_a, split_state_b)

    ma, mb = load_matrix(run_a), load_matrix(run_b)

    comparison = {
        "completed_round_diff": abs(int(ckpt_a["completed_round"])
                                    - int(ckpt_b["completed_round"])),
        "server_model_max_diff": state_dict_diff(ckpt_a["server_model"],
                                                 ckpt_b["server_model"]),
        "global_proto_max_diff": protos_diff(ckpt_a["global_protos"],
                                             ckpt_b["global_protos"]),
        "prompt_max_diff": state_dict_diff(ckpt_a["server_prompt"],
                                           ckpt_b["server_prompt"]),
        "head_max_diff": max(head_diffs),
        "key_max_diff": max(key_diffs),
        "anchor_max_diff": max(anchor_diffs),
        "accuracy_max_diff": matrix_diff(ma, mb),
        "split_equal": bool(split_equal),
        "fix_keys_equal": sorted(map(str, ckpt_a["fix_keys"]))
                          == sorted(map(str, ckpt_b["fix_keys"])),
    }

    rng_a, rng_b = ckpt_a["rng_state"], ckpt_b["rng_state"]
    rng_diffs = {
        "torch": max_diff(rng_a["torch"], rng_b["torch"]),
        "numpy": 0.0 if rng_a["numpy"] == rng_b["numpy"] else float("inf"),
        "python": 0.0 if rng_a["python"] == rng_b["python"] else float("inf"),
    }
    if rng_a.get("cuda") is not None:
        rng_diffs["cuda"] = max(max_diff(x, y)
                                for x, y in zip(rng_a["cuda"], rng_b["cuda"]))

    # ---------- pass/fail report ----------
    report = []
    for name, diff in comparison.items():
        if name in ("split_equal", "fix_keys_equal"):
            report.append((name, diff is True, 0.0 if diff else float("inf"), ""))
        else:
            report.append((name, diff == 0.0, diff, ""))
    for name, d in rng_diffs.items():
        report.append((f"rng.{name}", d == 0.0, d, ""))

    print("\n===== resume regression report =====")
    all_ok = True
    for name, ok, diff, note in report:
        status = "PASS" if ok else "FAIL"
        all_ok &= ok
        print(f"[{status}] {name:26s} max_diff={diff}"
              f"{('  (' + note + ')') if note else ''}")

    result = "PASS" if all_ok else "FAIL"

    # ---------- metrics/baseline_verification.json ----------
    verification = {
        "test": "continuous_vs_resume_equivalence",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gate": "Phase 1 is only allowed when result == PASS",
        "result": result,
        "config": {
            "data_name": args.data_name, "method": args.method,
            "seed": args.seed, "rounds": args.rounds,
            "split_at": args.split_at, "global_epoch": args.global_epoch,
            "task_num": args.task_num, "local_epoch": args.local_epoch,
            "deterministic": True,
        },
        "run_a": baseline_records(run_a, ckpt_a),
        "run_b": baseline_records(run_b, ckpt_b),
        "comparison": {k: (None if v == float("inf") else v)
                       for k, v in comparison.items()},
        "rng_state_diffs": {k: (None if v == float("inf") else v)
                            for k, v in rng_diffs.items()},
    }
    out_path = Path(args.work_dir) / "metrics" / "baseline_verification.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(verification, f, indent=2, default=str)

    print(f"\nsaved {out_path}")
    print("\nRESULT:", result if all_ok else
          "FAIL (if only rng.* rows fail with tiny diffs, rerun with "
          "--deterministic on a quieter GPU; model-state rows must pass)")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
