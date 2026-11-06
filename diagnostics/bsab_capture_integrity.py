"""diagnostics/bsab_capture_integrity.py -- Sec.34.2 0b capture artifact
integrity summary (user-requested 2026-11-04, pre-formal-rerun table).

Read-only scanner over a <run_dir>/bsab_capture/ directory: NO training
path is touched, nothing is re-fitted, no labels are consumed beyond
row counting. One row per (client, task, round, entry) boundary:

  n_train / n_test / n_rows        from the sidecar .json
  row_index_hash / head_hash      from the sidecar .json
  pre_prompt_hash / post_prompt_hash
  z_ref_max_abs_diff               max |z_ref_post - z_ref_pre|
  z_op_mean/max_disp               mean / max ||z_op_post - z_op_pre||_2
  capture_file_hash                sha256 of the .npz
  low_power                        n_fit < BSAB_CAPTURE_MIN_FIT

Plus cross-boundary consistency checks (fail-closed):
  * every .npz has its .json twin and vice versa;
  * every .npz carries all frozen fields (Sec.34.2.2);
  * z_ref invariance tolerance (EPS_ZREF, relative to per-row norm) --
    violation only FLAGS the boundary as ZREF_SUSPECT for B0 triage;
    the official verdict gate lives in the B0 probe, not here;
  * labels/split/sample_index arrays share the declared n_rows.

Output: summary CSV + JSON next to the capture dir (default) or --out.
Exit code 0 iff no structural failure (z_ref violations are reported,
not fatal here).
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Models.Client_DF import BSAB_CAPTURE_MIN_FIT  # noqa: E402

REQUIRED_NPZ_FIELDS = (
    "client_ids", "task_ids", "round_ids", "sample_index", "split",
    "label", "z_ref_pre", "z_op_pre", "z_ref_post", "z_op_post",
)
REQUIRED_JSON_FIELDS = (
    "client_id", "task_id", "round_id", "entry",
    "pre_prompt_hash", "post_prompt_hash", "head_hash",
    "row_index_hash", "n_rows", "n_train", "n_test", "n_fit",
    "low_power", "feature_shape",
)
EPS_ZREF = 1e-4  # relative z_ref stability tolerance (B_pre frozen value)


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def scan_boundary(npz_path, json_path):
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    missing = [k for k in REQUIRED_JSON_FIELDS if k not in meta]
    if missing:
        raise RuntimeError(
            "%s: missing json fields %s (fail-closed)"
            % (json_path.name, missing))
    z = np.load(npz_path)
    missing = [k for k in REQUIRED_NPZ_FIELDS if k not in z.files]
    if missing:
        raise RuntimeError(
            "%s: missing npz fields %s (fail-closed)"
            % (npz_path.name, missing))
    n = int(meta["n_rows"])
    shape_ok = all(int(z[k].shape[0]) == n for k in REQUIRED_NPZ_FIELDS)
    if not shape_ok or int(meta["n_train"]) + int(meta["n_test"]) != n:
        raise RuntimeError(
            "%s: declared n_rows=%d inconsistent with arrays (fail-closed)"
            % (npz_path.name, n))
    if (int(meta["n_fit"]) != int(meta["n_train"])
            or bool(meta["low_power"]) != (int(meta["n_fit"])
                                          < BSAB_CAPTURE_MIN_FIT)):
        raise RuntimeError(
            "%s: LOW_POWER / n_fit inconsistent with n_train "
            "(fail-closed)" % json_path.name)
    z_ref_pre, z_ref_post = z["z_ref_pre"], z["z_ref_post"]
    z_op_pre, z_op_post = z["z_op_pre"], z["z_op_post"]
    d_ref = np.abs(z_ref_post - z_ref_pre)
    denom = np.maximum(np.abs(z_ref_pre).max(axis=1), 1e-12)
    disp = np.linalg.norm(z_op_post - z_op_pre, axis=1)
    return {
        "file": npz_path.name,
        "client_id": int(meta["client_id"]),
        "task_id": int(meta["task_id"]),
        "round_id": int(meta["round_id"]),
        "entry": meta["entry"],
        "n_train": int(meta["n_train"]),
        "n_test": int(meta["n_test"]),
        "n_rows": n,
        "low_power": bool(meta["low_power"]),
        "row_index_hash": meta["row_index_hash"],
        "head_hash": meta["head_hash"],
        "pre_prompt_hash": meta["pre_prompt_hash"],
        "post_prompt_hash": meta["post_prompt_hash"],
        "z_ref_max_abs_diff": float(d_ref.max()),
        "z_ref_max_rel_diff": float((d_ref.max(axis=1) / denom).max()),
        "z_op_mean_disp": float(disp.mean()),
        "z_op_max_disp": float(disp.max()),
        "z_op_static": bool(float(disp.max()) == 0.0),
        "capture_file_hash": _sha256_file(npz_path),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--capture_dir", required=True,
                    help="the run's <run_dir>/bsab_capture/ directory")
    ap.add_argument("--out_prefix", default="",
                    help="output prefix (default: <capture_dir>/integrity)")
    args = ap.parse_args()

    cap = Path(args.capture_dir)
    if not cap.is_dir():
        raise SystemExit("not a directory: %s" % cap)
    npz_files = sorted(cap.glob("bsab_*_*.npz"))
    json_files = sorted(cap.glob("bsab_*_*.json"))
    npz_set = {p.stem for p in npz_files}
    json_set = {p.stem for p in json_files}
    if npz_set != json_set:
        raise SystemExit(
            "STRUCTURAL_FAILURE: %d npz vs %d json twins mismatch "
            "(orphans: %s)"
            % (len(npz_files), len(json_files),
               sorted(npz_set ^ json_set)[:10]))

    rows = []
    zref_violations = []
    for npz_path in npz_files:
        row = scan_boundary(npz_path, cap / (npz_path.stem + ".json"))
        rows.append(row)
        if row["z_ref_max_rel_diff"] >= EPS_ZREF:
            zref_violations.append(
                {"file": row["file"],
                 "z_ref_max_rel_diff": row["z_ref_max_rel_diff"]})

    rows.sort(key=lambda r: (r["round_id"], r["client_id"], r["task_id"],
                             r["entry"]))
    out_prefix = Path(args.out_prefix) if args.out_prefix else cap / "integrity"
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = Path(str(out_prefix) + ".csv")
    json_path = Path(str(out_prefix) + ".json")
    cols = list(rows[0].keys()) if rows else []
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
    summary = {
        "capture_dir": str(cap),
        "n_boundaries": len(rows),
        "n_low_power": sum(1 for r in rows if r["low_power"]),
        "n_z_op_static": sum(1 for r in rows if r["z_op_static"]),
        "z_ref_tolerance_rel": EPS_ZREF,
        "zref_suspect_boundaries": zref_violations,
        "rows": rows,
        "note": ("read-only integrity table (Sec.34.2, user-requested "
                 "2026-11-04); labels are counted, never consumed; "
                 "ZREF_SUSPECT rows are triage flags only -- the "
                 "authoritative ZREF_INVARIANCE_VIOLATION gate lives in "
                 "the offline B0 probe; z_op_static boundaries (zero "
                 "broadcast displacement) fail the two-state evidence "
                 "gate and must be investigated before any B0 use"),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print("boundaries scanned: %d (low_power=%d, zref_suspect=%d, "
          "z_op_static=%d)"
          % (len(rows), summary["n_low_power"], len(zref_violations),
             summary["n_z_op_static"]))
    for r in rows:
        print("  c%d t%d r%04d %-19s n=%4d z_ref_max=%.3e z_op_max=%.4f "
              "low_power=%d" % (r["client_id"], r["task_id"],
                                r["round_id"], r["entry"], r["n_rows"],
                                r["z_ref_max_rel_diff"], r["z_op_max_disp"],
                                int(r["low_power"])))
    if zref_violations:
        print("ZREF_SUSPECT: %d boundary(ies) exceed rel %.0e -- see %s"
              % (len(zref_violations), EPS_ZREF, json_path))
    print("wrote: %s , %s" % (csv_path, json_path))


if __name__ == "__main__":
    main()
