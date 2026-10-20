#!/usr/bin/env python
"""PR-2A tests: A5 task-boundary reference-prototype statistics & sync
(P2 v1.2 §4 steps 1-3, NO training loss attached yet).

  P1. exact merge: per-client (sum, count) merge is EXACT (sums add, counts
      add; mu = S/n recomputed from the merged totals, never
      average-of-averages), including a class shared by several clients.
  P2. no future class: a stats key outside every client's class_mask for
      tasks <= task_id makes the sync FAIL-CLOSED.
  P3. publication identity: every client receives the SAME dict object and
      the SAME version; version == task_id (monotonic).
  P4. double accumulation refused: re-syncing an already-synced task raises.
  P5. cumulative table: syncing task 0 then task 1 keeps task-0 classes and
      adds task-1 classes (exact counts across the two boundaries).
  P6. client stats pass (synthetic z_ref): per-class sum/count exact over
      the inner-train split only (inner-test samples excluded), the pass is
      RNG-NEUTRAL (torch/python/numpy streams restored), non-A5 methods
      are refused, and a missing inner split is refused.

Run:  python tests/test_a5_pr2a.py            (or pytest tests/test_a5_pr2a.py)
CPU only.
"""

import random
import sys
import types
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from Models import Client_DF as client_df_mod   # noqa: E402
from Models import Server_DF as server_df_mod  # noqa: E402


# ------------------------------------------------------------------ helpers

def _stub_client_a5(cid, stats, task_id=0):
    """Client stub carrying exactly what a5_sync_ref_protos touches."""
    obj = types.SimpleNamespace(
        id=cid, method='fedta_a5', task_id=task_id,
        a5_ref_protos=None, a5_proto_version=-1)
    obj.a5_compute_proto_stats = (lambda _s=stats: dict(_s))
    return obj


def _stub_server(clients, class_mask):
    """Server stub carrying exactly what a5_sync_ref_protos touches."""
    obj = types.SimpleNamespace(
        method='fedta_a5', clients=clients, class_mask=class_mask,
        client_num=len(clients),
        a5_proto_version=-1, a5_ref_protos=None,
        a5_proto_sums=None, a5_proto_counts=None)
    obj.a5_sync_ref_protos = types.MethodType(
        server_df_mod.Server_DF.a5_sync_ref_protos, obj)
    return obj


def _stats(entries):
    """{class: (sum_vector, count)} with deterministic sum vectors."""
    return {c: (torch.arange(768, dtype=torch.float32) + c + k, n)
            for k, (c, n) in enumerate(entries)}


def _masks(task_classes):
    """class_mask[k][t] = list of classes; None = not-yet-seen task."""
    n_tasks = max(len(tc) for tc in task_classes)
    return [[(list(task_classes[k][t]) if t < len(task_classes[k]) and
              task_classes[k][t] is not None else [])
             for t in range(n_tasks)] for k in range(len(task_classes))]


# ---------------------------------------------------------------- P1 + P3

def test_p1_exact_merge_and_publication():
    """Two clients share class 5 -> sums/counts ADD exactly; every client
    receives the same published object and version."""
    s0 = _stats([(0, 3), (5, 4)])
    s1 = _stats([(5, 6), (9, 2)])
    clients = [_stub_client_a5(0, s0), _stub_client_a5(1, s1)]
    masks = _masks([[[0, 5]], [[5, 9]]])
    srv = _stub_server(clients, masks)
    srv.a5_sync_ref_protos(0)

    assert srv.a5_proto_version == 0
    assert srv.a5_proto_counts == {0: 3, 5: 10, 9: 2}
    assert torch.equal(srv.a5_proto_sums[5], s0[5][0] + s1[5][0])
    assert torch.equal(srv.a5_proto_sums[0], s0[0][0])
    # mu = merged sum / merged count (NOT average of client means)
    assert torch.equal(srv.a5_ref_protos[5],
                       (s0[5][0] + s1[5][0]) / 10)
    # publication identity: SAME object, SAME version, everywhere
    assert all(c.a5_ref_protos is srv.a5_ref_protos for c in clients)
    assert all(c.a5_proto_version == 0 for c in clients)


# ---------------------------------------------------------------- P2

def test_p2_future_class_fail_closed():
    """A stats key owned by no client at task <= 0 (class 99 belongs to a
    FUTURE task) must abort the sync."""
    clients = [_stub_client_a5(0, _stats([(0, 3), (99, 1)]))]
    masks = _masks([[[0], [99]]])  # class 99 only appears in task 1
    srv = _stub_server(clients, masks)
    try:
        srv.a5_sync_ref_protos(0)
        raise AssertionError("future-class leak must raise")
    except RuntimeError as e:
        assert "future" in str(e)
    # nothing was published
    assert srv.a5_ref_protos is None
    assert clients[0].a5_ref_protos is None


# ---------------------------------------------------------------- P4 + P7

def test_p4_and_p7_double_sync_refused():
    clients = [_stub_client_a5(0, _stats([(0, 3)]))]
    srv = _stub_server(clients, _masks([[[0], [1]]]))
    srv.a5_sync_ref_protos(0)
    try:                            # same task re-sync
        srv.a5_sync_ref_protos(0)
        raise AssertionError("re-sync must raise")
    except RuntimeError as e:
        assert "double accumulation" in str(e)
    # task 1 syncs fine, then stale task-0 re-sync is refused
    clients[0].a5_compute_proto_stats = lambda: _stats([(1, 2)])
    srv.a5_sync_ref_protos(1)
    try:
        srv.a5_sync_ref_protos(0)
        raise AssertionError("stale re-sync must raise")
    except RuntimeError as e:
        assert "double accumulation" in str(e)


# ---------------------------------------------------------------- P5

def test_p5_cumulative_across_tasks():
    clients = [_stub_client_a5(0, _stats([(0, 3)]))]
    srv = _stub_server(clients, _masks([[[0], [1]]]))
    srv.a5_sync_ref_protos(0)
    clients[0].a5_compute_proto_stats = lambda: _stats([(1, 2), (0, 5)])
    srv.a5_sync_ref_protos(1)
    # task-0 class kept with ACCUMULATED count (3 + 5), task-1 class added
    assert srv.a5_proto_counts == {0: 8, 1: 2}
    assert srv.a5_proto_version == 1
    assert torch.equal(srv.a5_ref_protos[0],
                       (srv.a5_proto_sums[0]) / 8)


# ---------------------------------------------------------------- P6

class _FakeOriginal(torch.nn.Module):
    """Deterministic stand-in for the frozen backbone: pre_logits is an
    injective function of the raw input, so exact sums are checkable."""

    def __init__(self, in_dim, feat_dim):
        super().__init__()
        self.lin = torch.nn.Linear(in_dim, feat_dim, bias=False)
        with torch.no_grad():
            self.lin.weight.copy_(torch.arange(feat_dim * in_dim).float()
                                  .reshape(feat_dim, in_dim) * 0.01)

    def forward(self, x):
        return {'pre_logits': self.lin(x)}


def _stub_real_client(nb_classes=6, in_dim=4, feat_dim=768):
    """Client with REAL a5_compute_proto_stats bound, synthetic data."""
    torch.manual_seed(7)
    n = 24
    xs = torch.randn(n, in_dim)
    ys = torch.tensor([c % nb_classes for c in range(n)])
    obj = types.SimpleNamespace(
        id=0, method='fedta_a5', task_id=0, device='cpu',
        nb_classes=nb_classes,
        original_model=_FakeOriginal(in_dim, feat_dim),
        train_data=[TensorDataset(xs, ys)],
        # inner-train = first 20 samples ONLY (last 4 are inner-test and
        # must be excluded from the statistics)
        data_split_indices={0: {"train": list(range(20)),
                                "test": list(range(20, 24))}},
    )
    obj.a5_compute_proto_stats = types.MethodType(
        client_df_mod.Client_DF.a5_compute_proto_stats, obj)
    return obj, xs, ys


def test_p6_client_stats_exact_and_rng_neutral():
    c, xs, ys = _stub_real_client()
    W = c.original_model.lin.weight
    z_all = xs @ W.t()

    # advance ALL RNG streams, then verify the pass restores them
    torch.manual_seed(101)
    _ = torch.randn(3)
    random.seed(101)
    _ = random.random()
    np.random.seed(101)
    _ = np.random.rand(3)
    snap = {"torch": torch.get_rng_state(),
            "python": random.getstate(),
            "numpy": np.random.get_state()}

    stats = c.a5_compute_proto_stats()

    assert torch.equal(torch.get_rng_state(), snap["torch"])
    assert random.getstate() == snap["python"]
    cur = np.random.get_state()
    old = snap["numpy"]
    assert cur[0] == old[0] and np.array_equal(cur[1], old[1]) \
        and cur[2:] == old[2:]

    # exactness: per-class sum/count over the inner-TRAIN split only
    train_idx = c.data_split_indices[0]["train"]
    for cls in range(6):
        rows = [i for i in train_idx if int(ys[i]) == cls]
        if not rows:
            assert cls not in stats
            continue
        s, n = stats[cls]
        assert n == len(rows), f"class {cls}: count {n} != {len(rows)}"
        # float32 batched index_add vs whole-block sum: addition order
        # differs, so compare with a small tolerance (same lesson as S3)
        assert torch.allclose(s, z_all[rows].sum(0), atol=1e-4), \
            f"class {cls}: sum mismatch"

    # non-A5 method refused; missing inner split refused
    c.method = 'fedta_a2'
    try:
        c.a5_compute_proto_stats()
        raise AssertionError("non-A5 method must raise")
    except RuntimeError as e:
        assert "only fedta_a5" in str(e)
    c.method = 'fedta_a5'
    c.data_split_indices = {}
    try:
        c.a5_compute_proto_stats()
        raise AssertionError("missing split must raise")
    except RuntimeError as e:
        assert "inner split missing" in str(e)

    # frozen 768-dim z_ref space: a non-768 backbone is refused, not adapted
    c2, _, _ = _stub_real_client(feat_dim=8)
    try:
        c2.a5_compute_proto_stats()
        raise AssertionError("non-768 z_ref must raise")
    except RuntimeError as e:
        assert "768" in str(e)


def main():
    tests = [
        test_p1_exact_merge_and_publication,
        test_p2_future_class_fail_closed,
        test_p4_and_p7_double_sync_refused,
        test_p5_cumulative_across_tasks,
        test_p6_client_stats_exact_and_rng_neutral,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  [PASS] {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  [FAIL] {t.__name__}: {e}")
    if failed:
        sys.exit(1)
    print("all PR-2A structural tests passed")


if __name__ == "__main__":
    main()
