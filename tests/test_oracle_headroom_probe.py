"""Unit tests for diagnostics/probe_oracle_headroom.py (A5 Gate A-2).

Pinned semantics (mirroring the ncm_probe test conventions):

  * standardization stats are fit on TRAIN rows only (an eval-row mean
    must not shift them) -- the probe must never see test statistics;
  * head_predict fails closed when an allowed class has no OUTPUT UNIT
    (a join error); a mask class with zero train rows keeps its
    suppressed unit in the choice set -- native Chead semantics under
    the CIFAR public-class protocol (a (client, task) share of a public
    class can be 0-1 samples), never shrinks the candidate set;
  * masked-argmax restriction: predictions are always inside the
    allowed class set, even for a head trained on many more classes;
  * prediction must not mutate the fitted head (no-grad eval, no state
    drift) and must be bitwise deterministic under the same seed;
  * a linear head actually fits linearly separable data (probe sanity).
"""
import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diagnostics.probe_oracle_headroom import (  # noqa: E402
    fit_head, head_predict, stable_seed)

_HP = dict(hidden=16, epochs=30, lr=1e-2, wd=1e-4, dropout=0.0,
           batch_size=8, device="cpu")


def _two_class_data(n_per=20, d=8, seed=0):
    rng = np.random.RandomState(seed)
    classes = [3, 7]
    z0 = rng.randn(n_per, d) + 2.0
    z1 = rng.randn(n_per, d) - 2.0
    z = np.concatenate([z0, z1]).astype(np.float32)
    y = np.array([classes[0]] * n_per + [classes[1]] * n_per)
    return z, y, classes


class TestOracleHeadroomProbe(unittest.TestCase):

    def test_standardize_fit_on_train_rows_only(self):
        z_tr, y, classes = _two_class_data()
        handle = fit_head(z_tr, y, classes, seed=stable_seed("a"), **_HP)
        self.assertTrue(np.allclose(handle.mu, z_tr.mean(axis=0)))
        # test rows with wildly different scale must not have shifted the
        # fitted stats
        z_evil = z_tr + 100.0
        handle2 = fit_head(z_tr, y, classes, seed=stable_seed("a"), **_HP)
        self.assertTrue(np.allclose(handle2.mu, z_tr.mean(axis=0)))
        _ = head_predict(handle, z_evil, classes)  # must not mutate stats
        self.assertTrue(np.allclose(handle.mu, z_tr.mean(axis=0)))
        del z_evil

    def test_fail_closed_allowed_class_without_output_unit(self):
        z_tr, y, classes = _two_class_data()
        handle = fit_head(z_tr, y, classes, seed=stable_seed("b"), **_HP)
        with self.assertRaises(RuntimeError):
            head_predict(handle, z_tr[:4], allowed=[3, 7, 11])

    def test_zero_train_row_mask_class_keeps_suppressed_unit(self):
        # CIFAR public-class reality: a (client, task) mask class can
        # have ZERO train rows (client 0/class 10 in cifar_hsn). The
        # head still carries an output unit for it (trained only by the
        # softmax suppression gradient), so it competes in the argmax
        # exactly like the native Chead -- the choice set is never
        # shrunk and prediction stays inside the allowed set.
        z_tr, y, _ = _two_class_data(n_per=40)
        mask = [3, 7, 11]  # class 11 has no train rows
        handle = fit_head(z_tr, y, mask, seed=stable_seed("b2"), **_HP)
        self.assertIn(11, handle.idx_of)  # unit exists
        pred = head_predict(handle, z_tr, mask)
        self.assertTrue(set(int(p) for p in pred) <= set(mask))
        # the suppressed unit must lose the argmax against classes it
        # was actually trained on (softmax suppression pushed it down)
        self.assertFalse(any(int(p) == 11 for p in pred))

    def test_unknown_train_label_fails_loud(self):
        z_tr, y, classes = _two_class_data()
        y = y.copy()
        y[0] = 42  # label outside `classes`
        with self.assertRaises(KeyError):
            fit_head(z_tr, y, classes, seed=stable_seed("c"), **_HP)

    def test_masked_argmax_restriction(self):
        # head trained on 5 classes, eval restricted to a 2-class subset:
        # every prediction must be inside the subset
        rng = np.random.RandomState(1)
        d = 8
        classes = [0, 1, 2, 3, 4]
        zs, ys = [], []
        for ci, c in enumerate(classes):
            zs.append(rng.randn(16, d) + 3.0 * ci)
            ys.append(np.full(16, c))
        z = np.concatenate(zs).astype(np.float32)
        y = np.concatenate(ys)
        handle = fit_head(z, y, classes, seed=stable_seed("d"), **_HP)
        allowed = [1, 3]
        pred = head_predict(handle, z[:32], allowed)
        self.assertTrue(set(np.unique(pred)).issubset(set(allowed)))

    def test_predict_is_deterministic_and_does_not_mutate_head(self):
        z_tr, y, classes = _two_class_data()
        handle = fit_head(z_tr, y, classes, seed=stable_seed("e"), **_HP)
        w_before = handle.model[0].weight.detach().clone()
        p1 = head_predict(handle, z_tr, classes)
        p2 = head_predict(handle, z_tr, classes)
        self.assertTrue(np.array_equal(p1, p2))
        self.assertTrue(torch.equal(w_before,
                                    handle.model[0].weight.detach()))

    def test_same_seed_bitwise_reproducible(self):
        z_tr, y, classes = _two_class_data(seed=7)
        h1 = fit_head(z_tr, y, classes, seed=stable_seed("f"), **_HP)
        h2 = fit_head(z_tr, y, classes, seed=stable_seed("f"), **_HP)
        p1 = head_predict(h1, z_tr, classes)
        p2 = head_predict(h2, z_tr, classes)
        self.assertTrue(np.array_equal(p1, p2))

    def test_linear_head_fits_separable_data(self):
        # sanity: the probe machinery can actually fit its training data
        z_tr, y, classes = _two_class_data(seed=3)
        handle = fit_head(z_tr, y, classes, seed=stable_seed("g"),
                          hidden=0, epochs=100, lr=1e-2, wd=0.0,
                          dropout=0.0, batch_size=8, device="cpu")
        acc = float(np.mean([p == int(g)
                             for p, g in zip(head_predict(handle, z_tr,
                                                          classes), y)]))
        self.assertGreaterEqual(acc, 0.99)

    def test_stable_seed_is_stable(self):
        self.assertEqual(stable_seed("x", 1, 2), stable_seed("x", 1, 2))
        self.assertNotEqual(stable_seed("x", 1), stable_seed("x", 2))


if __name__ == "__main__":
    unittest.main()
