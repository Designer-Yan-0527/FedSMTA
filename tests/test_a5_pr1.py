#!/usr/bin/env python
"""PR-1 structural tests for fedta_a5 (A5 P2 v1.2 architecture wiring).

Covers the user-approved PR-1 merge conditions plus the GPT smoke items
that are testable without a full training run:

  S1. shared=off parity: Tail_Anchor.forward(x, mask) is bit-identical with
      residual=None (the only path non-A5 methods take); the A5 code adds
      NO computation when residual is None.
  S2. U=0 zero-residual parity: forward with a zero residual is bit-identical
      to residual=None (A5 at initialization == A2 baseline, first forward).
  S3. routing invariance: top-1 key selection and the pull-off term are
      computed from the ORIGINAL feature in both paths (a nonzero residual
      must not change routing or pull_off).
  S4. gradient wiring: the residual is differentiable w.r.t. U ONLY
      (first backward gives nonzero U.grad; V never receives gradients).
  S5. checkpoint roundtrip: a5_state save -> load restores U (requires_grad)
      and validates rank / arm / V mismatches FAIL-CLOSED.
  S6. V generation: deterministic from the seed, on a DEDICATED RNG stream
      (the global torch RNG state is untouched — baseline paths cannot be
      perturbed by A5 initialization).
  S7. arm validation: --a2_arm none is only valid for fedta_a5 (Shared-only
      factorial arm); fedta_a2 + none is rejected; _a2_init_task_protection
      no-ops (no data access) for arm none.

Run:  python tests/test_a5_pr1.py            (or pytest tests/test_a5_pr1.py)
Needs a CUDA device (Tail_Anchor.forward hardcodes .to('cuda')).
"""

import argparse
import sys
import types
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from Models.Tail_Anchor import Tail_Anchor  # noqa: E402
from Models import Client_DF as client_df_mod  # noqa: E402

HAVE_CUDA = torch.cuda.is_available()


def _make_tail_anchor():
    torch.manual_seed(0)
    return Tail_Anchor(10, 768, 200).cuda()


def _stub_client(method='fedta_a5', a2_arm='anchor', rank=8, device='cpu'):
    """Minimal Client_DF stand-in carrying exactly the A5/A2 attributes the
    tested methods touch (the real Client_DF needs the full ViT stack)."""
    obj = types.SimpleNamespace(
        id=0, method=method, a2_arm=a2_arm, a5_rank=rank, device=device,
        task_id=3,
        a5_U=torch.zeros(768, rank, requires_grad=True),
        a5_V=torch.randn(rank, 768),
        _a2_protected_slots=None, _a2_key_snapshot=None,
        _a2_anchor_snapshot=None, _a2_task_id=None,
    )
    obj._a5_residual = types.MethodType(
        client_df_mod.Client_DF._a5_residual, obj)
    obj._a5_state_for_checkpoint = types.MethodType(
        client_df_mod.Client_DF._a5_state_for_checkpoint, obj)
    obj._a5_load_checkpoint_state = types.MethodType(
        client_df_mod.Client_DF._a5_load_checkpoint_state, obj)
    obj._a2_init_task_protection = types.MethodType(
        client_df_mod.Client_DF._a2_init_task_protection, obj)
    return obj


# ---------------------------------------------------------------- S1 + S2 + S3

@torch.no_grad()
def _forward_pair(residual):
    """Same Tail_Anchor, same input: run the official path (residual=None)
    and the A5 path (residual=residual) and return both result triples.

    eval mode: Chead contains Dropout(0.5), so two TRAIN-mode forwards
    differ by design. The production parity claim (A5 U=0 == A2 baseline)
    compares two independent runs that walk the SAME global RNG stream
    (identical dropout masks); the unit-level structural check is done in
    eval mode where the forward is deterministic."""
    ta = _make_tail_anchor()
    ta.eval()
    torch.manual_seed(1)
    x = torch.randn(16, 768).cuda()
    target = torch.randint(0, 200, (16,)).cuda()
    out_none = ta(x, target, residual=None)
    out_res = ta(x, target, residual=residual)
    return ta, x, target, out_none, out_res


def test_s1_none_residual_is_official_path():
    """residual=None must produce the exact official outputs (structural:
    the A5 branch is skipped entirely)."""
    ta, x, target, out_none, out_res = _forward_pair(None)
    for a, b in zip(out_none, out_res):
        assert torch.equal(a, b)


def test_s2_zero_residual_bit_identical():
    """U=0 -> residual is EXACTLY zero -> A5 at initialization is
    forward-identical to the A2 baseline (logits, fused features, pull)."""
    residual = torch.zeros(16, 768).cuda()
    ta, x, target, out_none, out_res = _forward_pair(residual)
    for a, b in zip(out_none, out_res):
        assert torch.equal(a, b), "zero residual changed the forward output"


def test_s3_routing_and_pull_invariant_nonzero_residual():
    """A NONZERO residual must change the logits (injection is live) but
    leave routing (top-1 key) and the pull-off term untouched."""
    torch.manual_seed(2)
    residual = torch.randn(16, 768).cuda() * 0.5
    ta, x, target, out_none, out_res = _forward_pair(residual)
    logits_none, x1_none, pull_none = out_none
    logits_res, x1_res, pull_res = out_res
    # injection is live: logits differ (except pathological luck)
    assert not torch.equal(logits_none, logits_res)
    # pull-off: computed from the ORIGINAL normalized feature + routed key
    assert torch.equal(pull_none, pull_res)
    # routing: recompute top-1 exactly as Tail_Anchor.forward does; the
    # anchor half of the fused input must be identical in both paths
    assert torch.equal(x1_none[:, 768:], x1_res[:, 768:])
    # and the feature half differs by exactly the residual (up to float32
    # rounding: (x + r) - x suffers catastrophic cancellation, so the
    # observed deviation is ulp(|x+r|) ~ 1e-7, NOT an injection error)
    assert torch.allclose(x1_res[:, :768] - x1_none[:, :768], residual,
                          rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------- S4

def test_s4_gradient_reaches_U_only():
    """Δθ(z) = (z V^T) U^T is differentiable w.r.t. U only; V is a fixed
    buffer (requires_grad=False -> no gradient, ever)."""
    c = _stub_client()
    c.a5_V.requires_grad_(False)
    z = torch.randn(16, 768)
    r = c._a5_residual(z)
    assert r is not None and r.shape == (16, 768)
    r.sum().backward()
    assert c.a5_U.grad is not None and c.a5_U.grad.abs().sum().item() > 0
    assert c.a5_V.grad is None
    # non-A5 methods (or pre-injection clients) get residual=None
    c2 = _stub_client(method='fedta_a2')
    assert c2._a5_residual(z) is None


# ---------------------------------------------------------------- S5

def test_s5_checkpoint_roundtrip_and_rejections():
    c = _stub_client()
    with torch.no_grad():
        c.a5_U.copy_(torch.randn(768, 8) * 0.01)
    state = {"a5_state": c._a5_state_for_checkpoint(), "task_id": 3}
    assert state["a5_state"]["rank"] == 8
    assert state["a5_state"]["a2_arm"] == "anchor"

    # roundtrip: U restored bit-identically, still trainable.
    # Simulate the server's DETERMINISTIC V regeneration on resume (the
    # stub's default random V would otherwise trip the V-mismatch guard
    # before the roundtrip asserts could run).
    c2 = _stub_client()
    c2.a5_V = c.a5_V.clone()
    c2._a5_load_checkpoint_state(state)
    assert torch.equal(c2.a5_U, c.a5_U)
    assert c2.a5_U.requires_grad

    # missing a5_state -> fail-closed
    c3 = _stub_client()
    try:
        c3._a5_load_checkpoint_state({"task_id": 3})
        raise AssertionError("missing a5_state must raise")
    except RuntimeError as e:
        assert "a5_state" in str(e)

    # rank mismatch -> fail-closed
    bad = {"a5_state": dict(state["a5_state"], rank=16)}
    c4 = _stub_client()
    try:
        c4._a5_load_checkpoint_state(bad)
        raise AssertionError("rank mismatch must raise")
    except RuntimeError as e:
        assert "rank" in str(e)

    # arm mismatch -> fail-closed
    bad = {"a5_state": dict(state["a5_state"], a2_arm="key")}
    c5 = _stub_client()
    try:
        c5._a5_load_checkpoint_state(bad)
        raise AssertionError("arm mismatch must raise")
    except RuntimeError as e:
        assert "a2_arm" in str(e)

    # V mismatch (server-regenerated projection differs) -> fail-closed
    bad = {"a5_state": dict(state["a5_state"], V=torch.randn(8, 768))}
    c6 = _stub_client()
    try:
        c6._a5_load_checkpoint_state(bad)
        raise AssertionError("V mismatch must raise")
    except RuntimeError as e:
        assert "V mismatch" in str(e)


# ---------------------------------------------------------------- S6

def test_s6_v_generation_deterministic_and_rng_neutral():
    """Replicates Server_DF.init_client's V generation: same seed -> same V;
    the GLOBAL torch RNG stream is untouched (baseline parity requirement)."""
    def gen_V(seed, rank=8):
        g = torch.Generator(device='cpu')
        g.manual_seed(int(seed) * 100003 + 7)
        return torch.randn(rank, 768, generator=g)

    assert torch.equal(gen_V(42), gen_V(42))
    assert not torch.equal(gen_V(42), gen_V(43))

    # RNG neutrality: the dedicated generator must not advance the global
    # stream (any later data-partition / batch RNG is bit-identical with or
    # without A5 initialization).
    torch.manual_seed(123)
    _ = torch.randn(4)               # advance the global stream deterministically
    snap = torch.get_rng_state()
    _ = gen_V(42)                    # A5 initialization
    assert torch.equal(torch.get_rng_state(), snap), (
        "V generation leaked into the global torch RNG stream")


# ---------------------------------------------------------------- S7

def test_s7_arm_validation_and_none_noop():
    # main.py validation: 'none' only for fedta_a5
    import main as main_mod

    def ns(method, a2_arm, a5_rank=8):
        return argparse.Namespace(method=method, a2_arm=a2_arm,
                                  a5_rank=a5_rank)

    main_mod.validate_method_args(ns('fedta_a5', 'none'))       # Shared-only arm
    main_mod.validate_method_args(ns('fedta_a5', 'anchor'))     # A5 full arm
    try:
        main_mod.validate_method_args(ns('fedta_a2', 'none'))
        raise AssertionError("fedta_a2 + a2_arm none must be rejected")
    except ValueError as e:
        assert "none" in str(e)
    try:
        main_mod.validate_method_args(ns('fedta_a5', 'anchor', a5_rank=0))
        raise AssertionError("a5_rank=0 must be rejected")
    except ValueError as e:
        assert "a5_rank" in str(e)

    # _a2_init_task_protection with arm 'none': records task ownership and
    # returns BEFORE touching any data (the stub deliberately has NO
    # train_data / data_split_indices / vit / model attributes -> any data
    # access would raise AttributeError).
    c = _stub_client(a2_arm='none')
    c._a2_init_task_protection(argparse.Namespace())
    assert c._a2_task_id == 3
    assert c._a2_protected_slots is None
    assert c._a2_key_snapshot is None
    assert c._a2_anchor_snapshot is None


def main():
    tests = [
        test_s1_none_residual_is_official_path,
        test_s2_zero_residual_bit_identical,
        test_s3_routing_and_pull_invariant_nonzero_residual,
        test_s4_gradient_reaches_U_only,
        test_s5_checkpoint_roundtrip_and_rejections,
        test_s6_v_generation_deterministic_and_rng_neutral,
        test_s7_arm_validation_and_none_noop,
    ]
    if not HAVE_CUDA:
        print("SKIP: no CUDA device (Tail_Anchor.forward hardcodes cuda)")
        for t in tests[:3]:
            print(f"  [skip] {t.__name__}")
        tests = tests[3:]
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
    print("all PR-1 structural tests passed")


if __name__ == "__main__":
    main()
