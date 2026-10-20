#!/usr/bin/env python
"""PR-2C tests: A5 U-only sample-weighted FedAvg (P2 v1.2 §3).

  C1. exact sample-weighted average: U^{r+1} = sum_k (n_k / N) U_k with
      n_k = len(traindata); after aggregation EVERY client holds the same
      averaged U (bit-identical across clients).
  C2. V is NEVER touched (bit-identical before/after).
  C3. aggregation identity: (avg U) V z == sum_k p_k (U_k V z) on a probe
      (exact-average-of-effective-transforms property, §1 rationale).
  C4. independence: each client's new U is a separate leaf — a gradient
      step on one client's copy leaves the others bit-identical.
  C5. fresh-leaf discipline: requires_grad=True and .grad is None.
  C6. fail-closed: non-A5 method raises; empty traindata raises.

Run:  python tests/test_a5_pr2c.py      (CPU only)
"""

import sys
import types
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from Models import Server_DF as server_df_mod  # noqa: E402


def _client(k, n, r=4, seed=None):
    torch.manual_seed(seed if seed is not None else 100 + k)
    obj = types.SimpleNamespace(
        id=k, method='fedta_a5',
        traindata=[0] * n,                      # only len() is used
        a5_V=torch.randn(r, 768),
        a5_U=(torch.randn(768, r) * 0.1).requires_grad_(True),
    )
    return obj


def _server(clients):
    obj = types.SimpleNamespace(
        method='fedta_a5', clients=clients, client_num=len(clients))
    obj.a5_fedavg_U = types.MethodType(
        server_df_mod.Server_DF.a5_fedavg_U, obj)
    return obj


def test_c1_exact_sample_weighted_average():
    cs = [_client(0, 100), _client(1, 300), _client(2, 600)]
    U_before = [c.a5_U.detach().clone() for c in cs]
    srv = _server(cs)
    srv.a5_fedavg_U([0, 1, 2])

    n = [100, 300, 600]
    N = sum(n)
    U_ref = sum(U_ * (nk / N) for U_, nk in zip(U_before, n))
    for c in cs:
        assert torch.equal(c.a5_U.detach(), U_ref), \
            "aggregated U != sample-weighted average"
    # all clients bit-identical
    for c in cs[1:]:
        assert torch.equal(cs[0].a5_U.detach(), c.a5_U.detach())


def test_c2_V_untouched():
    cs = [_client(0, 50), _client(1, 50)]
    V_before = [c.a5_V.clone() for c in cs]
    _server(cs).a5_fedavg_U([0, 1])
    for c, Vb in zip(cs, V_before):
        assert torch.equal(c.a5_V, Vb), "V was modified"


def test_c3_aggregation_identity():
    """(avg U) V z == sum_k p_k (U_k V z) on a deterministic probe."""
    cs = [_client(0, 200), _client(1, 100), _client(2, 700)]
    U_before = [c.a5_U.detach().clone() for c in cs]
    n = [200, 100, 700]
    N = sum(n)
    _server(cs).a5_fedavg_U([0, 1, 2])
    V = cs[0].a5_V
    z = torch.randn(4, 768)
    lhs = (z @ V.t()) @ cs[0].a5_U.detach().t()
    rhs = sum((z @ V.t()) @ U_.t() * (nk / N)
              for U_, nk in zip(U_before, n))
    assert torch.allclose(lhs, rhs, atol=1e-5), \
        "effective-residual identity violated"


def test_c4_c5_independent_fresh_leaves():
    cs = [_client(0, 10), _client(1, 10)]
    _server(cs).a5_fedavg_U([0, 1])
    # separate leaf objects
    assert cs[0].a5_U is not cs[1].a5_U
    # gradient on client 0's copy does not leak to client 1
    snap1 = cs[1].a5_U.detach().clone()
    (cs[0].a5_U.sum()).backward()
    assert cs[0].a5_U.grad is not None
    assert cs[1].a5_U.grad is None
    assert torch.equal(cs[1].a5_U.detach(), snap1)
    # fresh-leaf discipline
    for c in cs:
        assert c.a5_U.requires_grad is True
    assert cs[1].a5_U.grad is None


def test_c6_fail_closed():
    cs = [_client(0, 10), _client(1, 10)]
    srv = _server(cs)
    srv.method = 'fedta_a2'
    try:
        srv.a5_fedavg_U([0, 1])
        raise AssertionError("non-A5 must raise")
    except RuntimeError as e:
        assert "fedta_a5" in str(e) or "a5_fedavg_U" in str(e)
    srv.method = 'fedta_a5'

    cs_empty = [_client(0, 0), _client(1, 0)]
    try:
        _server(cs_empty).a5_fedavg_U([0, 1])
        raise AssertionError("empty traindata must raise")
    except RuntimeError as e:
        assert "empty traindata" in str(e)


def main():
    tests = [
        test_c1_exact_sample_weighted_average,
        test_c2_V_untouched,
        test_c3_aggregation_identity,
        test_c4_c5_independent_fresh_leaves,
        test_c6_fail_closed,
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
    print("all PR-2C structural tests passed")


if __name__ == "__main__":
    main()
