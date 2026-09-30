"""Phase 0 runtime utilities.

Only engineering infrastructure lives here:
run directory management, tee logging, command recording, checkpoint
path helpers, RNG state, json/csv writing.

No FedTA algorithm logic (BGPS / SIKF / Tail Anchor / loss) is touched.
"""

import csv
import hashlib
import io
import json
import os
import random
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------------------
# run name / canonical command
# ---------------------------------------------------------------------------

# args that describe "how to run" instead of "what experiment this is"
RUNTIME_ONLY_ARGS = {
    "run_name", "resume", "output_dir", "save_every", "keep_last",
    "eval_all_every", "num_workers", "pin_mem", "device", "eval",
}


def canonical_command(args):
    """Deterministic string identifying the experiment configuration."""
    payload = {k: repr(v) for k, v in sorted(vars(args).items())
               if k not in RUNTIME_ONLY_ARGS}
    return json.dumps(payload, sort_keys=True)


def auto_run_name(args):
    """{dataset}_{method}_c.._t.._ge.._le.._lr.._seed.._{8-bit hash}"""
    digest = hashlib.sha1(canonical_command(args).encode("utf-8")).hexdigest()[:8]
    return (f"{args.data_name}_{args.method}"
            f"_c{args.client_num}_t{args.task_num}"
            f"_ge{args.global_epoch}_le{args.local_epoch}"
            f"_lr{args.lr}_seed{args.seed}_{digest}")


# ---------------------------------------------------------------------------
# tee logging: terminal + logs/train.log at the same time
# ---------------------------------------------------------------------------

class Tee(object):
    def __init__(self, stream, path):
        self.stream = stream
        self.file = open(path, "a", buffering=1, encoding="utf-8", errors="replace")

    def write(self, data):
        try:
            self.stream.write(data)
        except Exception:
            pass
        try:
            self.file.write(data)
        except Exception:
            pass

    def flush(self):
        try:
            self.stream.flush()
        except Exception:
            pass
        try:
            self.file.flush()
        except Exception:
            pass

    def close(self):
        self.flush()

    def isatty(self):
        try:
            return self.stream.isatty()
        except Exception:
            return False


# ---------------------------------------------------------------------------
# RNG state
# ---------------------------------------------------------------------------

def rng_state_dict():
    cuda_state = None
    if torch.cuda.is_available():
        try:
            cuda_state = torch.cuda.get_rng_state_all()
        except Exception:
            cuda_state = None
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": cuda_state,
    }


def restore_rng_state(state):
    """Must be called AFTER model creation / client creation / checkpoint
    loading / dataset split rebuilding, otherwise model init itself consumes
    RNG and the restored stream gets misaligned."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    cuda_state = state.get("cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


# ---------------------------------------------------------------------------
# module states (never pickle whole nn.Module)
# ---------------------------------------------------------------------------

def module_state_or_none(module):
    if module is None:
        return None
    return {k: v.detach().cpu() for k, v in module.state_dict().items()}


def clone_prompt_template(vit):
    """Deep-copy the vit's prompt module to be used as a load template."""
    prompt = getattr(vit, "prompt", None)
    if prompt is None:
        vit.init_prompts()
        prompt = vit.prompt
    return deepcopy(prompt)


# ---------------------------------------------------------------------------
# checkpoint io
# ---------------------------------------------------------------------------

def load_checkpoint_file(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # older torch without weights_only kwarg
        return torch.load(path, map_location="cpu")


def atomic_save_checkpoint(checkpoint, *paths):
    """Serialize once, write every target atomically (tmp + os.replace)."""
    buf = io.BytesIO()
    torch.save(checkpoint, buf)
    data = buf.getvalue()
    for path in paths:
        path = str(path)
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)


# ---------------------------------------------------------------------------
# misc
# ---------------------------------------------------------------------------

def git_commit_hash():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


# ---------------------------------------------------------------------------
# run manager
# ---------------------------------------------------------------------------

class RunManager(object):
    """Owns the run directory:
        {output_dir}/{data_name}/{method}/{run_name}/seed_{seed}/
            command.txt / args.json / git_commit.txt
            logs/train.log
            checkpoints/latest.pth, round_XXXX.pth, task_XX_end.pth
            metrics/accuracy_matrix.csv, client_X_accuracy.csv, round_metrics.jsonl
    """

    def __init__(self, args):
        self.args = args
        self.start_time = time.strftime("%Y-%m-%d %H:%M:%S")

        resume = (args.resume or "").strip()

        if resume and resume.lower() != "auto":
            # explicit checkpoint path: run_dir is inferred from it
            ckpt = Path(resume).expanduser().resolve()
            if not ckpt.is_file():
                raise FileNotFoundError(
                    f"--resume checkpoint not found: {ckpt}")
            self.checkpoint_path = str(ckpt)
            self.run_dir = ckpt.parent.parent
            self.run_name = self.run_dir.name
        else:
            self.checkpoint_path = None
            self.run_name = args.run_name if args.run_name else auto_run_name(args)
            self.run_dir = (Path(args.output_dir) / args.data_name / args.method
                            / self.run_name / f"seed_{args.seed}")

        if resume.lower() == "auto":
            latest = self.run_dir / "checkpoints" / "latest.pth"
            if not latest.is_file():
                raise FileNotFoundError(
                    "--resume auto: checkpoint not found: "
                    f"{latest}\n"
                    "Make sure --output_dir / --run_name / --seed / --data_name / "
                    "--method match the original run.")
            self.checkpoint_path = str(latest)

        self.run_dir = Path(self.run_dir)
        self.log_dir = self.run_dir / "logs"
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.metrics_dir = self.run_dir / "metrics"

        # never silently overwrite an old experiment
        if self.checkpoint_path is None and resume == "":
            if self.run_dir.exists() and self._has_checkpoints():
                raise RuntimeError(
                    "Run directory already exists and is not empty.\n"
                    f"RUN_DIR = {self.run_dir}\n"
                    "Use --resume auto / checkpoint path,\n"
                    "or choose another --run_name.")

        for d in (self.run_dir, self.log_dir, self.ckpt_dir, self.metrics_dir):
            d.mkdir(parents=True, exist_ok=True)

        # record the real command (append: resume invocations are kept too)
        with open(self.run_dir / "command.txt", "a", encoding="utf-8") as f:
            f.write("python " + " ".join(sys.argv) + "\n")

        # args.json: only written for a fresh run, keep the original on resume
        args_json = self.run_dir / "args.json"
        if not args_json.exists():
            with open(args_json, "w", encoding="utf-8") as f:
                json.dump(vars(args), f, indent=2, default=str)

        with open(self.run_dir / "git_commit.txt", "w", encoding="utf-8") as f:
            f.write(git_commit_hash() + "\n")

        # tee: console + file, append mode so resume never overwrites old logs
        self.log_path = self.log_dir / "train.log"
        sys.stdout = Tee(sys.stdout, self.log_path)
        sys.stderr = Tee(sys.stderr, self.log_path)

        self.print_header()

    # -- setup helpers ------------------------------------------------------

    def _has_checkpoints(self):
        if not (self.run_dir / "checkpoints").exists():
            return any(self.run_dir.iterdir())
        return any((self.run_dir / "checkpoints").glob("*.pth"))

    def print_header(self):
        print("=" * 72)
        print(f"RUN_DIR: {self.run_dir}")
        print(f"COMMAND: python {' '.join(sys.argv)}")
        print(f"ARGS: {vars(self.args)}")
        print(f"SEED: {self.args.seed}")
        print(f"DEVICE: {self.args.device}")
        print(f"START_TIME: {self.start_time}")
        if self.checkpoint_path:
            print(f"RESUME_CHECKPOINT: {self.checkpoint_path}")
        print("=" * 72)

    # -- checkpoint paths ---------------------------------------------------

    def round_ckpt_path(self, round_id):
        return self.ckpt_dir / f"round_{round_id:04d}.pth"

    def task_ckpt_path(self, task_id):
        return self.ckpt_dir / f"task_{task_id:02d}_end.pth"

    def latest_ckpt_path(self):
        return self.ckpt_dir / "latest.pth"

    def cleanup_round_checkpoints(self, keep_last):
        """Keep only the last N round_XXXX.pth; latest.pth and task_XX_end.pth
        are never removed."""
        rounds = sorted(self.ckpt_dir.glob("round_[0-9][0-9][0-9][0-9].pth"))
        if keep_last and keep_last > 0:
            for p in rounds[:-keep_last]:
                try:
                    p.unlink()
                except OSError:
                    pass

    # -- metrics io ---------------------------------------------------------

    def append_jsonl(self, path, obj):
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, default=str) + "\n")

    def load_accuracy_records(self):
        """Reload previously saved accuracy records (used when resuming, so the
        accuracy matrix keeps full history)."""
        records = []
        if not self.metrics_dir.exists():
            return records
        for path in sorted(self.metrics_dir.glob("client_*_accuracy.csv")):
            try:
                with open(path, newline="", encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        records.append({
                            "client_id": int(row["client_id"]),
                            "after_task": int(row["after_task"]),
                            "test_task": int(row["test_task"]),
                            "accuracy": float(row["accuracy"]),
                        })
            except (OSError, ValueError, KeyError):
                continue
        return records

    def write_accuracy_csvs(self, records, task_num):
        """Rewrite client_X_accuracy.csv and the client-mean accuracy_matrix.csv
        from the full record list."""
        by_client = {}
        for r in records:
            by_client.setdefault(r["client_id"], []).append(r)

        for cid, rows in by_client.items():
            rows = sorted(rows, key=lambda r: (r["after_task"], r["test_task"]))
            path = self.metrics_dir / f"client_{cid}_accuracy.csv"
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["client_id", "after_task", "test_task", "accuracy"])
                for r in rows:
                    w.writerow([r["client_id"], r["after_task"],
                                r["test_task"], f"{r['accuracy']:.2f}"])

        # latest value per (after_task, test_task, client), then client mean
        latest = {}
        for r in records:
            latest[(r["after_task"], r["test_task"], r["client_id"])] = r["accuracy"]
        after_tasks = sorted({key[0] for key in latest})

        with open(self.metrics_dir / "accuracy_matrix.csv", "w",
                  newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["after_task"] + [f"T{t}" for t in range(task_num)])
            for a in after_tasks:
                row = [a]
                for t in range(task_num):
                    vals = [v for (aa, tt, _c), v in latest.items()
                            if aa == a and tt == t]
                    row.append(f"{sum(vals) / len(vals):.2f}" if vals else "")
                w.writerow(row)

    # -- logging ------------------------------------------------------------

    def flush_log(self):
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
