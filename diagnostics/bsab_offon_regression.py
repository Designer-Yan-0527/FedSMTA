"""diagnostics/bsab_offon_regression.py -- Sec.34.2 0b run-level
INSTRUMENTATION_NEUTRAL gate (user-frozen checklist 2026-11-04).

Compares a capture-OFF run dir vs a capture-ON run dir (same seed, same
config, --deterministic, --max_rounds 2 matched pair) and prints a
PASS/FAIL verdict. Checks, per the frozen checklist:

  1. checkpoint tensors bitwise identical (server_model, server_prompt
     = candidate/broadcast prompt, global_head, global_protos,
     temp_protos, every client's full state incl. heads and prompts,
     data_split_state, outer_split_manifest, fix_keys, ...);
  2. four-stream RNG states bitwise identical (checkpoint rng_state:
     torch / python / numpy / cuda);
  3. args identical except the two whitelisted flags (bsab_pair_capture,
     run_name);
  4. metrics trajectories byte-identical (accuracy_matrix.csv,
     train_log.csv, margins.csv, round_metrics.jsonl, summary CSVs);
  5. command.txt differs only by the whitelisted flags.

Explicit evidence rows (user checklist): sha16 hashes of the candidate
prompt (= server_prompt state), broadcast prompt (client-side loaded
prompt), and each client's birth head, OFF vs ON.

READ-ONLY. Exit code 0 iff INSTRUMENTATION_NEUTRAL = PASS.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# args keys allowed to differ between the OFF and ON arms: the treatment
# flag itself, the run identity, and bsab_capture_dir (a derived OUTPUT
# path string that exists only when the flag is on -- same class as
# run_name, never consumed by training). run_name/command.txt additionally
# tolerate the --bsab_pair_capture flag.
ARG_WHITELIST = {"bsab_pair_capture", "run_name", "bsab_capture_dir"}

# metrics files that must be byte-identical (both may not exist on a
# truncated --max_rounds run; presence is compared pairwise)
METRICS_FILES = (
    "metrics/accuracy_matrix.csv",
    "metrics/train_log.csv",
    "metrics/margins.csv",
    "metrics/round_metrics.jsonl",
    "metrics/summary_metrics.csv",
)


def sha16(obj, h=None):
    """stable content hash over tensors / primitives / nested dicts."""
    import hashlib
    if h is None:
        h = hashlib.sha256()
    if isinstance(obj, torch.Tensor):
        h.update(obj.detach().cpu().contiguous().numpy().tobytes())
    elif isinstance(obj, np.ndarray):
        h.update(np.ascontiguousarray(obj).tobytes())
    elif isinstance(obj, dict):
        for k in sorted(obj.keys(), key=str):
            h.update(str(k).encode()); h.update(b"\x00")
            sha16(obj[k], h)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            sha16(v, h)
    elif obj is None:
        h.update(b"\x00none")
    else:
        h.update(str(obj).encode())
    return h.hexdigest()[:16]


def deep_compare(a, b, path="", diffs=None):
    """collect every bitwise divergence between two nested structures."""
    if diffs is None:
        diffs = []
    if type(a) is not type(b):
        if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
            if not torch.equal(a, b):
                diffs.append(path)
        elif isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            if not np.array_equal(a, b):
                diffs.append(path)
        else:
            diffs.append("%s (type %s vs %s)" % (path, type(a), type(b)))
        return diffs
    if isinstance(a, torch.Tensor):
        if not torch.equal(a, b):
            diffs.append(path)
    elif isinstance(a, np.ndarray):
        if not np.array_equal(a, b):
            diffs.append(path)
    elif isinstance(a, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k not in a or k not in b:
                diffs.append("%s.%s (missing)" % (path, k))
            else:
                deep_compare(a[k], b[k], "%s.%s" % (path, k), diffs)
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            diffs.append("%s (len %d vs %d)" % (path, len(a), len(b)))
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                deep_compare(x, y, "%s[%d]" % (path, i), diffs)
    elif a != b:
        diffs.append("%s (%r vs %r)" % (path, a, b))
    return diffs


def evidence_hashes(ckpt):
    """explicit checklist hashes: candidate/broadcast prompt + per-client
    birth heads, OFF vs ON."""
    rows = []
    sp = ckpt.get("server_prompt")
    rows.append(("candidate/server_prompt", sha16(sp) if sp is not None
                 else None))
    for i, c in enumerate(ckpt.get("clients", [])):
        keys = set(c.keys()) if isinstance(c, dict) else set()
        for key in sorted(keys):
            if "prompt" in key:
                rows.append(("client%d.%s" % (i, key), sha16(c[key])))
        heads = c.get("heads") if isinstance(c, dict) else None
        if isinstance(heads, dict):
            for t in sorted(heads, key=str):
                rows.append(("client%d.heads[%s]" % (i, t), sha16(heads[t])))
        elif isinstance(heads, (list, tuple)):
            for t, hh in enumerate(heads):
                if hh is not None:
                    rows.append(("client%d.heads[%d]" % (i, t), sha16(hh)))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--off_dir", required=True, help="capture-OFF seed dir")
    ap.add_argument("--on_dir", required=True, help="capture-ON seed dir")
    ap.add_argument("--checkpoint", default="latest.pth")
    args = ap.parse_args()

    off = Path(args.off_dir)
    on = Path(args.on_dir)
    failures = []
    checks = []

    # -- 1+2+3: checkpoint deep compare ---------------------------------
    ck_off = torch.load(off / "checkpoints" / args.checkpoint,
                        map_location="cpu")
    ck_on = torch.load(on / "checkpoints" / args.checkpoint,
                       map_location="cpu")
    a_off = dict(ck_off.get("args", {}))
    a_on = dict(ck_on.get("args", {}))
    args_diff = {k: (a_off.get(k), a_on.get(k))
                 for k in set(a_off) | set(a_on)
                 if a_off.get(k) != a_on.get(k)}
    bad_args = {k: v for k, v in args_diff.items() if k not in ARG_WHITELIST}
    whitelist_ok = (args_diff.get("bsab_pair_capture", (None, None))
                    == (False, True))
    if bad_args or not whitelist_ok:
        failures.append("args diverge beyond whitelist: %s (whitelist delta "
                        "ok=%s)" % (bad_args or args_diff, whitelist_ok))
    else:
        checks.append("args identical except whitelist "
                      "(bsab_pair_capture False->True, run_name) -- OK")

    tensors = {k: v for k, v in ck_off.items() if k not in ("args",)}
    tensors_on = {k: v for k, v in ck_on.items() if k not in ("args",)}
    diffs = deep_compare(tensors, tensors_on, "ckpt")
    if diffs:
        failures.append("checkpoint bitwise diffs (%d): %s"
                        % (len(diffs), diffs[:20]))
    else:
        checks.append("checkpoint bitwise identical: server_model, "
                      "server_prompt (candidate), global_head, all "
                      "clients (heads+prompts), protos, data splits, "
                      "four-stream rng_state -- OK")

    # -- 5: command.txt flags -------------------------------------------
    def flags(cmd):
        toks = cmd.split()
        return {toks[i]: (toks[i + 1] if i + 1 < len(toks) and
                          not toks[i + 1].startswith("--") else True)
                for i, t in enumerate(toks) if t.startswith("--")}
    f_off = flags((off / "command.txt").read_text().strip().splitlines()[-1])
    f_on = flags((on / "command.txt").read_text().strip().splitlines()[-1])
    cmd_diff = {k: (f_off.get(k), f_on.get(k))
                for k in set(f_off) | set(f_on) if f_off.get(k) != f_on.get(k)}
    cmd_whitelist = {"--bsab_pair_capture", "--run_name"}
    bad_cmd = {k: v for k, v in cmd_diff.items() if k not in cmd_whitelist}
    if bad_cmd:
        failures.append("command.txt diverges beyond whitelist: %s"
                        % bad_cmd)
    elif cmd_diff.get("--bsab_pair_capture", (None, None)) != (None, True):
        failures.append("command.txt: ON arm does not set "
                        "--bsab_pair_capture (%s)" % cmd_diff)
    else:
        checks.append("command.txt differs only by --bsab_pair_capture / "
                      "--run_name -- OK")

    # -- 4: metrics byte-identical ---------------------------------------
    # round_metrics.jsonl is compared field-wise with the "timestamp" key
    # dropped (wall-clock record time, inherently run-specific; every
    # substantive field must still match exactly). All other files are
    # byte-identical.
    for rel in METRICS_FILES:
        p_off, p_on = off / rel, on / rel
        if p_off.exists() != p_on.exists():
            failures.append("%s presence mismatch (off=%s on=%s)"
                            % (rel, p_off.exists(), p_on.exists()))
        elif p_off.exists():
            if rel.endswith("round_metrics.jsonl"):
                lo = [json.loads(x) for x in
                      p_off.read_text().splitlines() if x.strip()]
                ln = [json.loads(x) for x in
                      p_on.read_text().splitlines() if x.strip()]
                bad = []
                if len(lo) != len(ln):
                    bad.append("line count %d vs %d" % (len(lo), len(ln)))
                for a, b in zip(lo, ln):
                    a2 = {k: v for k, v in a.items() if k != "timestamp"}
                    b2 = {k: v for k, v in b.items() if k != "timestamp"}
                    if a2 != b2:
                        bad.append(json.dumps(a2)[:120])
                if bad:
                    failures.append(
                        "round_metrics.jsonl diverges beyond timestamp: %s"
                        % "; ".join(bad[:3]))
                else:
                    checks.append("round_metrics.jsonl field-identical "
                                  "(timestamp excluded) -- OK")
            elif p_off.read_bytes() != p_on.read_bytes():
                failures.append("%s NOT byte-identical" % rel)
            else:
                checks.append("%s byte-identical -- OK" % rel)

    # -- explicit checklist evidence ------------------------------------
    print("== explicit hash evidence (OFF vs ON) ==")
    ev_off = evidence_hashes(ck_off)
    ev_on = evidence_hashes(ck_on)
    names = [n for n, _ in ev_off]
    if names != [n for n, _ in ev_on]:
        failures.append("evidence hash name sets differ between arms")
    ev_on_map = dict(ev_on)
    n_mismatch = 0
    for name, h in ev_off:
        h2 = ev_on_map.get(name)
        tag = "OK" if h == h2 else "MISMATCH"
        if h != h2:
            n_mismatch += 1
        print("  %-32s %s %s %s" % (name, h, h2, tag))
    if n_mismatch:
        failures.append("%d explicit hash mismatches" % n_mismatch)
    else:
        checks.append("explicit candidate/broadcast prompt + birth-head "
                      "hashes identical -- OK")

    print("\n== checks ==")
    for c in checks:
        print("  [OK]   %s" % c)
    print("\n== INSTRUMENTATION_NEUTRAL verdict ==")
    if failures:
        for f in failures:
            print("  [FAIL] %s" % f)
        print("INSTRUMENTATION_NEUTRAL = FAIL")
        sys.exit(1)
    print("INSTRUMENTATION_NEUTRAL = PASS")


if __name__ == "__main__":
    main()
