"""Track E + B2 unit tests (CPU-only; 2026-10-28).

Covers:
  1. ResidualAdapter B2 axes: zero-init identity (fixed & learnable),
     learnable alpha init=1.0, dropout module present iff dropout>0,
     invalid scale_mode rejected.
  2. build_arm cfg wiring (bottleneck / P0 no-feat / default 128).
  3. eval_in_mode toggle semantics: flags per mode, exact restore,
     native stochastic vs eval-mode deterministic.
  4. train_one(evaluate_test=False): TEST never touched (no test_acc /
     generalization_gap fields); frozen best_state replayable via
     test_from_state deterministically.
"""
import sys
import unittest
from copy import deepcopy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from diagnostics.probe_ssra_headroom import (ResidualAdapter, build_arm,
                                             train_one, test_from_state,
                                             _acc_eval_mode)
from diagnostics.probe_eval_dropout_causal import (eval_in_mode,
                                                   snapshot_flags,
                                                   head_flags_during_forward)


class _ModelStub(torch.nn.Module):
    """Tail_Anchor-like stand-in: load_head RE-INSTALLS a deepcopy of the
    source head (which carries the SOURCE's training flag) -- this is the
    official bypass path that the Track E fix must survive."""

    def __init__(self):
        super().__init__()
        self.head = torch.nn.Identity()

    def load_head(self, head):
        self.head = deepcopy(head)

    def forward(self, x):
        return self.head(x)


class _FakeClient(object):
    """Minimal client stand-in: model/vit/original_model with Dropout."""

    def __init__(self):
        torch.manual_seed(0)
        self.id = 0
        self.model = _ModelStub()
        # stored heads live in TRAIN mode, exactly like the real system
        self.heads = [torch.nn.Sequential(
            torch.nn.Linear(4, 4), torch.nn.ReLU(),
            torch.nn.Dropout(0.5), torch.nn.Linear(4, 3))]
        self.vit = torch.nn.Sequential(torch.nn.Linear(4, 4),
                                        torch.nn.Dropout(0.5))
        self.original_model = torch.nn.Sequential(torch.nn.Linear(4, 4),
                                                  torch.nn.Dropout(0.5))
        self.observed_flags = []

    def evaluate(self, task, nb_classes):
        # mimic official call order; record flags DURING the call
        # (task / nb_classes unused: signature matches the official path)
        self.observed_flags.append(snapshot_flags(self))
        self.model.load_head(self.heads[task])  # official bypass path
        x = torch.ones(16, 4)
        self.original_model(x)
        self.vit(x)
        return float(self.model(x).sum().detach())


class TestResidualAdapterB2(unittest.TestCase):

    def test_identity_at_init_fixed_and_learnable(self):
        x = torch.randn(5, 8)
        for sm in ("fixed", "learnable"):
            ad = ResidualAdapter(8, bottleneck=4, scale_mode=sm)
            self.assertTrue(torch.equal(ad(x), x),
                            "zero-init W2 must give identity (%s)" % sm)

    def test_learnable_alpha_and_dropout_wiring(self):
        ad = ResidualAdapter(8, bottleneck=4, dropout=0.2,
                             scale_mode="learnable")
        self.assertIsInstance(ad.alpha, torch.nn.Parameter)
        self.assertEqual(float(ad.alpha), 1.0)
        self.assertTrue(ad.alpha.requires_grad)
        self.assertTrue(any(isinstance(m, torch.nn.Dropout)
                            for m in ad.modules()))
        ad0 = ResidualAdapter(8, bottleneck=4, dropout=0.0)
        self.assertFalse(any(isinstance(m, torch.nn.Dropout)
                             for m in ad0.modules()))
        with self.assertRaises(ValueError):
            ResidualAdapter(8, scale_mode="bogus")

    def test_build_arm_cfg(self):
        feat0, _head0, _p0 = build_arm("P0", 8, 3)
        self.assertIsNone(feat0)
        f4, _, p4 = build_arm("P1", 8, 3, {"bottleneck": 4})
        _f16, _, p16 = build_arm("P1", 8, 3, {"bottleneck": 16})
        self.assertIsInstance(f4, ResidualAdapter)
        self.assertLess(p4, p16)
        fdef, _, _ = build_arm("P1", 8, 3)  # gate default = v2 config
        self.assertEqual(fdef.down.out_features, 128)
        self.assertIsNone(fdef.alpha)


class TestEvalModeToggle(unittest.TestCase):

    def test_flags_per_mode_and_restore(self):
        c = _FakeClient()
        eval_in_mode(c, 0, 3, "native")
        self.assertEqual(c.observed_flags[-1],
                         {"model": True, "vit": True,
                          "original_model": True})
        eval_in_mode(c, 0, 3, "head_eval")
        self.assertEqual(c.observed_flags[-1],
                         {"model": False, "vit": True,
                          "original_model": True})
        eval_in_mode(c, 0, 3, "full_eval")
        self.assertEqual(c.observed_flags[-1],
                         {"model": False, "vit": False,
                          "original_model": False})
        # flags fully restored after every call
        self.assertEqual(snapshot_flags(c),
                         {"model": True, "vit": True,
                          "original_model": True})

    def test_native_stochastic_eval_deterministic(self):
        c = _FakeClient()
        native = [eval_in_mode(c, 0, 3, "native") for _ in range(8)]
        self.assertGreater(max(native) - min(native), 1e-6,
                           "train-mode Dropout must make native stochastic")
        head = [eval_in_mode(c, 0, 3, "head_eval") for _ in range(3)]
        self.assertEqual(len(set(head)), 1,
                         "head-only eval must be deterministic here")
        full = [eval_in_mode(c, 0, 3, "full_eval") for _ in range(3)]
        self.assertEqual(len(set(full)), 1,
                         "full eval must be deterministic here")
        with self.assertRaises(ValueError):
            eval_in_mode(c, 0, 3, "bogus_mode")

    def test_head_eval_survives_load_head_bypass(self):
        """Regression for the reviewer-found bug: evaluate() re-installs a
        deepcopy of heads[task] whose OWN training flag is train-mode, so
        toggling client.model alone leaves Dropout ACTIVE. eval_in_mode
        must also switch the SOURCE heads[task]. The installed head's
        flag is verified AT FORWARD TIME (a post-call read is misleading:
        the finally-restore flips the installed head back to train)."""
        c = _FakeClient()
        # native: the head actually used at forward time is train-mode
        _acc, flags = head_flags_during_forward(c, 0, 3, "native")
        self.assertEqual(flags, [True])
        # corrected modes: installed head must be eval-mode AND
        # deterministic despite the load_head re-install
        for mode in ("head_eval", "full_eval"):
            vals = []
            for _ in range(4):
                a, flags = head_flags_during_forward(c, 0, 3, mode)
                vals.append(a)
                self.assertEqual(flags, [False],
                                 "%s: head must be eval-mode at forward "
                                 "time" % mode)
            self.assertEqual(len(set(vals)), 1,
                             "%s: Dropout must be OFF (deterministic)"
                             % mode)
        # source head flag restored after every call
        self.assertTrue(c.heads[0].training)
        self.assertEqual(snapshot_flags(c),
                         {"model": True, "vit": True,
                          "original_model": True})


class TestB2NoTestLeakage(unittest.TestCase):

    @staticmethod
    def _data():
        rng = np.random.default_rng(7)
        X = rng.normal(size=(90, 8)).astype(np.float32)
        y = (np.arange(90) % 3).astype(np.int64)
        return X[:60], y[:60], X[60:75], y[60:75], X[75:], y[75:]

    def test_search_run_never_touches_test(self):
        """Fail-closed: search runs pass X_te=y_te=None entirely -- test is
        not read for n_classes nor materialized as a tensor."""
        Xtr, ytr, Xva, yva, _Xte, _yte = self._data()
        cfg = {"bottleneck": 4, "scale_mode": "learnable", "dropout": 0.0,
               "weight_decay": 1e-3}
        r, s = train_one("P1", Xtr, ytr, Xva, yva, None, None, seed=42,
                         epochs=6, batch_size=16, lr=1e-2,
                         weight_decay=cfg["weight_decay"], device="cpu",
                         fisher_in=1.0, cfg=cfg, evaluate_test=False)
        self.assertNotIn("test_acc", r)
        self.assertNotIn("generalization_gap", r)
        self.assertIn("val_acc_at_best", r)
        for key in ("feat", "head", "n_classes", "dim"):
            self.assertIn(key, s)

    def test_evaluate_test_true_with_none_raises(self):
        Xtr, ytr, Xva, yva, _Xte, _yte = self._data()
        with self.assertRaises(ValueError):
            train_one("P0", Xtr, ytr, Xva, yva, None, None, seed=42,
                      epochs=6, batch_size=16, lr=1e-2, weight_decay=1e-3,
                      device="cpu", fisher_in=1.0, evaluate_test=True)

    def test_acc_eval_mode_deterministic_and_restores(self):
        """Val/test protocol: _acc_eval_mode must switch dropout modules
        to eval for the measurement and restore train mode afterwards."""
        torch.manual_seed(1)
        feat = ResidualAdapter(8, bottleneck=4, dropout=0.5,
                               scale_mode="fixed")
        head = torch.nn.Linear(8, 3)
        feat.train()
        X = torch.randn(32, 8)
        y = torch.randint(0, 3, (32,))
        a1 = _acc_eval_mode(feat, head, X, y)
        a2 = _acc_eval_mode(feat, head, X, y)
        self.assertEqual(a1, a2, "eval-mode accuracy must be deterministic")
        self.assertTrue(feat.training, "train flag must be restored")
        self.assertTrue(head.training)
        # must equal a manual eval-mode measurement
        feat.eval(), head.eval()
        manual = float((head(feat(X)).argmax(dim=1) == y).float().mean())
        self.assertEqual(a1, round(manual * 100.0, 6))

    def test_frozen_state_replay_is_deterministic(self):
        Xtr, ytr, Xva, yva, Xte, yte = self._data()
        cfg = {"bottleneck": 4, "scale_mode": "fixed", "dropout": 0.0,
               "weight_decay": 1e-3}
        _, s = train_one("P1", Xtr, ytr, Xva, yva, Xte, yte, seed=42,
                         epochs=6, batch_size=16, lr=1e-2,
                         weight_decay=cfg["weight_decay"], device="cpu",
                         fisher_in=1.0, cfg=cfg, evaluate_test=False)
        a1 = test_from_state("P1", cfg, s, Xte, yte, "cpu")
        a2 = test_from_state("P1", cfg, s, Xte, yte, "cpu")
        self.assertEqual(a1, a2)
        self.assertTrue(0.0 <= a1 <= 100.0)

    def test_evaluate_test_true_returns_test_fields(self):
        Xtr, ytr, Xva, yva, Xte, yte = self._data()
        r, _ = train_one("P0", Xtr, ytr, Xva, yva, Xte, yte, seed=42,
                         epochs=6, batch_size=16, lr=1e-2,
                         weight_decay=1e-3, device="cpu", fisher_in=1.0,
                         evaluate_test=True)
        self.assertIn("test_acc", r)
        self.assertIn("generalization_gap", r)


if __name__ == "__main__":
    unittest.main()
