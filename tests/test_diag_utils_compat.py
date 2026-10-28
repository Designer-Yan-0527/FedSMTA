"""Regression test for diag_utils old-run compatibility.

cifar_hsn (a pre-A2-R run) crashed feature extraction with
AttributeError: 'Namespace' object has no attribute 'route_fix':
load_args_from_run_dir used a hand-maintained backfill list of 3 keys,
which went stale every time build_server grew a new parameter. The fix
backfills EVERY current-parser key missing from the old args.json with
the parser's parsed default; this test pins that semantics so the next
new switch cannot reintroduce the crash.
"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diagnostics.diag_utils import (  # noqa: E402
    _current_parser_defaults, load_args_from_run_dir)


class TestDiagUtilsOldRunCompat(unittest.TestCase):

    def test_current_parser_defaults_cover_new_switches(self):
        d = _current_parser_defaults()
        for key in ("route_fix", "route_key_no_wd", "route_selected_ce_grad",
                    "skip_fed_avg_head", "method"):
            self.assertIn(key, d)
        # every post-hoc switch defaults to OFF = official behavior, so
        # replaying an old run stays faithful
        self.assertFalse(d["route_fix"])
        self.assertFalse(d["route_key_no_wd"])
        self.assertFalse(d["route_selected_ce_grad"])

    def test_old_args_json_gets_backfilled(self):
        # an args.json from BEFORE --route_fix existed
        old = {"data_name": "cifar100", "seed": 42, "client_num": 5,
               "resume": "auto", "max_rounds": 1, "data_path": "x"}
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "args.json"), "w",
                      encoding="utf-8") as f:
                json.dump(old, f)
            args = load_args_from_run_dir(td)
        # missing keys: current parser defaults
        self.assertEqual(args.route_fix, False)
        self.assertFalse(hasattr(args, "nope_not_a_real_key"))
        # present keys: NOT overwritten
        self.assertEqual(args.seed, 42)
        # runtime knobs: always reset for diagnostics
        self.assertEqual(args.resume, "")
        self.assertEqual(args.run_name, "")
        self.assertEqual(args.max_rounds, 0)
        self.assertFalse(args.deterministic)

    def test_missing_args_json_fails_loud(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError):
                load_args_from_run_dir(td)


if __name__ == "__main__":
    unittest.main()
