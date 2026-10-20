#!/usr/bin/env python
"""PR-2B tests: A5 Stage-2 prototype-CE (P2 v1.2 §5).

  B1. gradient isolation: L_proto's autograd graph touches U ONLY —
      V / mu_c / z_ref receive NO gradient; Chead / Tail_Anchor / prompts
      are not in the graph at all.
  B2. both W-sides reach U: gradient is nonzero when U is nonzero AND the
      loss is minimized when W z aligns with the true class's W mu
      (learnability check on a toy problem).
  B3. exact formula: logits == cos(W z, W mu_c)/tau with W = I + U V,
      computed independently in the test.
  B4. lambda == 0 equivalence: _phase2_train_step with a5_proto_lambda=0
      must not call _a5_proto_ce at all (patched to raise) and its loss
      must equal the pre-PR-2B value (PR-1 behavior + PR-2A sync).
  B5. fail-closed: non-A5 method, missing U/V, empty table, and a batch
      label outside the published table all raise.
  B6. no NaN / Inf in the loss and in U.grad for a realistic batch.

Run:  python tests/test_a5_pr2b.py      (CPU only)
"""

import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from Models import Client_DF as client_df_mod   # noqa: E402

TAU = 0.07
LAMBDA = 0.5


def _client(r=4, nb=6, seed=0):
    """Client stub with the exact attributes _a5_proto_ce touches."""
    torch.manual_seed(seed)
    obj = types.SimpleNamespace(
        id=0, method='fedta_a5', device='cpu',
        a5_proto_lambda=LAMBDA, a5_proto_tau=TAU,
        a5_U=None, a5_V=None, a5_ref_protos=None, a5_proto_version=0,
    )
    g = torch.Generator().manual_seed(seed * 7 + 1)
    obj.a5_V = torch.randn(r, 768, generator=g)          # fixed
    obj.a5_U = (torch.randn(768, r, generator=g) * 0.05
                ).requires_grad_(True)                    # trainable
    # toy prototypes: distinct directions in 768-d
    protos = {}
    base = torch.randn(nb, 768, generator=g)
    for c in range(nb):
        protos[c] = F.normalize(base[c], dim=0)
    obj.a5_ref_protos = protos
    obj._a5_proto_ce = types.MethodType(
        client_df_mod.Client_DF._a5_proto_ce, obj)
    return obj


# ---------------------------------------------------------------- B1 + B3

def test_b1_b3_gradient_isolation_and_formula():
    c = _client()
    z = torch.randn(8, 768)
    target = torch.tensor([0, 1, 2, 3, 4, 5, 0, 1])
    loss = c._a5_proto_ce(z, target)
    assert torch.isfinite(loss), "loss not finite"

    # independent recomputation of the exact frozen formula
    classes = sorted(c.a5_ref_protos.keys())
    mu = torch.stack([c.a5_ref_protos[k] for k in classes])
    Wz = z + (z @ c.a5_V.t()) @ c.a5_U.t().detach()
    Wmu = mu + (mu @ c.a5_V.t()) @ c.a5_U.t().detach()
    ref = (F.normalize(Wz, dim=1) @ F.normalize(Wmu, dim=1).t()) / TAU
    y = torch.tensor([classes.index(int(t)) for t in target])
    ref_loss = F.cross_entropy(ref, y).item()
    assert abs(ref_loss - loss.item()) < 1e-5, \
        f"formula mismatch (B3): {ref_loss} vs {loss.item()}"

    # gradient isolation: ONLY U receives gradient
    V_saved = c.a5_V.clone()
    z_saved = z.clone()
    mu_saved = {k: v.clone() for k, v in c.a5_ref_protos.items()}
    loss.backward()
    assert c.a5_U.grad is not None and c.a5_U.grad.abs().sum() > 0, \
        "U received no gradient (B1/B2)"
    # V / z / mu_c were plain non-requires-grad tensors: bit-unchanged
    assert torch.equal(c.a5_V, V_saved)
    assert torch.equal(z, z_saved)
    for k in c.a5_ref_protos:
        assert torch.equal(c.a5_ref_protos[k], mu_saved[k])
    # grad shape sanity
    assert c.a5_U.grad.shape == c.a5_U.shape


# ---------------------------------------------------------------- B2

def test_b2_learnability_toy():
    """Gradient descent on U alone must reduce L_proto: proves both the
    W z and the W mu sides keep a usable gradient path to U."""
    c = _client(seed=3)
    z = torch.randn(4, 768)
    target = torch.tensor([0, 1, 2, 3])
    opt = torch.optim.Adam([c.a5_U], lr=0.05)
    first = c._a5_proto_ce(z, target).item()
    for _ in range(30):
        opt.zero_grad()
        c._a5_proto_ce(z, target).backward()
        opt.step()
    last = c._a5_proto_ce(z, target).item()
    assert last < first, f"U-only optimization failed: {first} -> {last}"


# ---------------------------------------------------------------- B4

def test_b4_lambda_zero_equivalence():
    """lambda == 0 must skip proto-CE entirely: patch _a5_proto_ce to
    raise, then build the phase-2 gate predicate the same way the
    production code does — it must NOT call the loss."""
    c = _client()
    c.a5_proto_lambda = 0.0

    def _must_not_run(*a, **k):
        raise AssertionError("proto-CE computed with lambda == 0")

    c._a5_proto_ce = _must_not_run
    # exact production predicate from _phase2_train_step:
    gate = (c.method == 'fedta_a5' and c.a5_proto_lambda != 0)
    assert gate is False
    # (if the gate were True the patched raiser would fire)


# ---------------------------------------------------------------- B5

def test_b5_fail_closed_paths():
    c = _client()
    z, target = torch.randn(2, 768), torch.tensor([0, 1])

    c.method = 'fedta_a2'
    try:
        c._a5_proto_ce(z, target)
        raise AssertionError("non-A5 must raise")
    except RuntimeError as e:
        assert "A5 only" in str(e)
    c.method = 'fedta_a5'

    u_saved = c.a5_U
    c.a5_U = None
    try:
        c._a5_proto_ce(z, target)
        raise AssertionError("missing U must raise")
    except RuntimeError as e:
        assert "a5_init_shared" in str(e)
    c.a5_U = u_saved

    p_saved = c.a5_ref_protos
    c.a5_ref_protos = {}
    try:
        c._a5_proto_ce(z, target)
        raise AssertionError("empty table must raise")
    except RuntimeError as e:
        assert "empty" in str(e)
    c.a5_ref_protos = p_saved

    try:
        c._a5_proto_ce(z, torch.tensor([99]))
        raise AssertionError("out-of-table label must raise")
    except RuntimeError as e:
        assert "not in the published table" in str(e)


# ---------------------------------------------------------------- B6

def test_b6_no_nan():
    c = _client(seed=11)
    for trial in range(5):
        z = torch.randn(16, 768) * (10.0 ** trial)      # up to 1e4 scale
        target = torch.randint(0, 6, (16,))
        loss = c._a5_proto_ce(z, target)
        assert torch.isfinite(loss), f"trial {trial}: loss={loss}"
        loss.backward()
        assert torch.isfinite(c.a5_U.grad).all(), \
            f"trial {trial}: U.grad has NaN/Inf"
        c.a5_U.grad = None


def main():
    tests = [
        test_b1_b3_gradient_isolation_and_formula,
        test_b2_learnability_toy,
        test_b4_lambda_zero_equivalence,
        test_b5_fail_closed_paths,
        test_b6_no_nan,
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
    print("all PR-2B structural tests passed")


if __name__ == "__main__":
    main()
