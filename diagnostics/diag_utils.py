"""Shared bootstrap for Phase-1 diagnostic scripts.

Rebuilds the exact training environment (data partition -> models -> clients)
from a run directory's args.json by calling main.build_server(), so the RNG
stream (and therefore the client data partition) is bit-identical to the
original run. Then a checkpoint can be loaded to restore model/训练状态.

These utilities are diagnostics-only: they never touch the FedTA training
logic.
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_args_from_run_dir(run_dir, data_path=None, device=None):
    """Rebuild an args Namespace from {run_dir}/args.json.

    Runtime knobs (resume / run_name / max_rounds / deterministic) are reset
    so the bootstrap behaves like a fresh, full-length run."""
    run_dir = Path(run_dir)
    args_json = run_dir / "args.json"
    if not args_json.is_file():
        raise FileNotFoundError(
            f"{args_json} not found. Pass the run directory "
            f"(.../output/{{data_name}}/{{method}}/{{run_name}}/seed_{{seed}}).")
    with open(args_json, "r", encoding="utf-8") as f:
        raw = json.load(f)
    args = argparse.Namespace(**raw)

    # defaults for args introduced after the run was recorded
    for key, default in (("deterministic", False), ("max_rounds", 0),
                         ("eval_all_every", 0)):
        if not hasattr(args, key):
            setattr(args, key, default)

    # runtime knobs: always reset for diagnostics
    args.resume = ""
    args.run_name = ""
    args.max_rounds = 0
    args.deterministic = False

    # optional overrides (e.g. dataset lives elsewhere on this machine)
    if data_path is not None:
        args.data_path = data_path
    if device is not None:
        args.device = device
    return args


def resolve_checkpoint(run_dir, checkpoint=None):
    """Return the checkpoint path: explicit arg, or run_dir/checkpoints/latest.pth.

    A relative checkpoint argument is accepted in two forms: relative to the
    current working directory (full-path form used in RUNCOMMANDS.md) or
    relative to run_dir. The returned path is absolute so repeated resolution
    (scripts may pre-resolve and pass the result to bootstrap_server, which
    resolves again) stays idempotent.
    """
    run_dir = Path(run_dir)
    if checkpoint is None:
        candidates = [run_dir / "checkpoints" / "latest.pth"]
    else:
        ckpt = Path(checkpoint)
        if ckpt.is_absolute():
            candidates = [ckpt]
        else:
            # cwd-relative first, then run_dir-relative
            candidates = [ckpt, run_dir / ckpt]
    for cand in candidates:
        if cand.is_file():
            return str(cand.resolve())
    raise FileNotFoundError(
        f"checkpoint not found (tried: "
        f"{', '.join(str(c) for c in candidates)})")


def bootstrap_server(run_dir, checkpoint=None, data_path=None, device=None):
    """(args, server) with the environment rebuilt and checkpoint loaded.

    server.init_client() is called; if checkpoint is given the full
    round-boundary state (models / heads / prompts / protos / test loaders /
    RNG) is restored via Server_DF.load_checkpoint."""
    from main import build_server

    args = load_args_from_run_dir(run_dir, data_path=data_path, device=device)
    server = build_server(args, run_manager=None)
    server.init_client()
    if checkpoint is not None:
        ckpt_path = resolve_checkpoint(run_dir, checkpoint)
        server.load_checkpoint(ckpt_path)
    return args, server
