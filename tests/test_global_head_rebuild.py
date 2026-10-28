"""Unit tests for Server_DF.rebuild_global_head (checkpoint replay fix).

Bug (hit by the cifar_hsn oracle-headroom probe): load_checkpoint
unconditionally rebuilt global_head as a deepcopy of clients[0].vit.head
(a plain Linear), assuming fed_avg_head() always ran. skip_fed_avg_head
runs (HS-N arm) never replace global_head, so their checkpoints store
the original __init__ Chead structure (head.0.* / head.3.*) -> resume
and diagnostics replay crashed with a state_dict key mismatch. This is
a TRAINING-path bug too: resuming any skip_fed_avg_head run hits it.
"""
import os
import sys
import unittest

import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Models.Server_DF import rebuild_global_head  # noqa: E402
from Models.classification_head import Chead  # noqa: E402


class TestRebuildGlobalHead(unittest.TestCase):

    def test_linear_state_rebuilds_vit_head_deepcopy(self):
        vit_head = nn.Linear(16, 10)
        gh = rebuild_global_head(vit_head.state_dict(), vit_head, 10)
        gh.load_state_dict(vit_head.state_dict())
        self.assertIsInstance(gh, nn.Linear)
        # must be a COPY, not the live vit.head itself
        self.assertIsNot(gh, vit_head)

    def test_chead_state_rebuilds_chead(self):
        # skip_fed_avg_head run: checkpoint stores the __init__ Chead
        head = Chead(10)
        gh = rebuild_global_head(head.state_dict(), nn.Linear(16, 10), 10)
        gh.load_state_dict(head.state_dict())  # must not raise
        self.assertIsInstance(gh, Chead)

    def test_unknown_structure_fails_loud(self):
        bad = {"weird.0.weight": torch.zeros(1), "weird.0.bias": torch.zeros(1)}
        with self.assertRaises(RuntimeError):
            rebuild_global_head(bad, nn.Linear(16, 10), 10)

    def test_partial_head_keys_fail_loud(self):
        # e.g. a truncated/corrupt state: head.* prefix but not the full
        # Chead key set -> the Chead branch would misload; the deepcopy
        # branch is keyed on the exact {weight, bias} set so this must
        # NOT silently fall through to Linear
        bad = {"head.0.weight": torch.zeros(4, 8)}
        with self.assertRaises(RuntimeError):
            rebuild_global_head(bad, nn.Linear(16, 10), 10)


if __name__ == "__main__":
    unittest.main()
