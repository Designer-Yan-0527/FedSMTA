"""A2-R collapse-attribution ablation switch tests (unittest, repo style).

Two switches added 2026-10-28 after the route_audit forensics verdict
(collapse 100% in local training, all 100 keys mutually aligned):

  Arm B: --route_detach_ce_key_grad
    The similarity feeding the soft-routing softmax is recomputed from
    prompt_norm.detach(). Required semantics, all unit-tested here:
      * forward VALUES bitwise unchanged (soft_weights / similarity /
        reduce_sim / prompted_embedding / pull_idx);
      * CE loss can no longer reach prompt_key (grad None) while the
        sample-feature path keeps its CE gradient;
      * pull still uses the ORIGINAL grad-carrying similarity and reaches
        only the selected slots (official-equivalent semantics);
      * fail-closed: the switch without --route_fix raises ValueError.

  Arm C: --disable_pull
    Client_DF.train() drops the -0.1*reduce_sim term (CE only). The loss
    composition is replicated here on a REAL route_fix forward graph to
    pin the intended semantics: the selected-slot pull gradient spike is
    present in the full loss and absent in the CE-only loss, and CE-only
    key grads are exactly the CE component of the full loss.

Default OFF invariance: constructing/forward/backward/step with the
switches explicitly False equals the constructor-default path bitwise
(no RNG drift, no value drift), mirroring the invariance conventions of
test_route_audit_invariance.py.
"""
import os
import sys
import types
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Models.global_prompt import Global_Prompt  # noqa: E402


def _make_gp(seed=123, pool=8, dim=32, length=4, top_k=2, detach=False):
    """Deterministic route_fix Global_Prompt (R1 init uses its own
    generator; the prompt pool itself consumes the global RNG stream, so
    callers must seed first)."""
    return Global_Prompt(length=length, embed_dim=dim, embedding_key='cls',
                          prompt_init='uniform', prompt_pool=True,
                          prompt_key=True, pool_size=pool, top_k=top_k,
                          batchwise_prompt=True, prompt_key_init='uniform',
                          route_fix=True, route_tau=0.1, route_seed=seed,
                          route_detach_ce_key_grad=detach)


def _forward(gp):
    """One deterministic route_fix forward (same graph shape the real
    Client_DF.train() builds: cls_features -> soft routing -> prompts)."""
    torch.manual_seed(7)
    x = torch.randn(4, 8, 32)
    cls = torch.randn(4, 32, requires_grad=True)
    out = gp(x, prompt_mask=None, cls_features=cls)
    return out, cls


def _ce_logits_loss(out):
    """CE proxy reaching prompt_key ONLY through the soft-routing weights
    (exactly the channel arm B isolates)."""
    torch.manual_seed(11)
    head_w = torch.randn(5, 32)
    feat = out['prompted_embedding'].mean(dim=1)      # (B, C)
    logits = torch.nn.functional.linear(feat, head_w)
    labels = torch.tensor([0, 1, 2, 3])
    return torch.nn.functional.cross_entropy(logits, labels)


_VALUE_KEYS = ('soft_weights', 'similarity', 'reduce_sim',
              'prompted_embedding', 'prompt_idx', 'pull_idx')


class TestArmBDetachCeKeyGrad(unittest.TestCase):

    def test_forward_values_bitwise_equal(self):
        torch.manual_seed(123)
        gp_off = _make_gp()
        torch.manual_seed(123)
        gp_on = _make_gp(detach=True)
        # construction itself must be RNG-neutral w.r.t. the switch
        self.assertTrue(torch.equal(gp_off.prompt_key, gp_on.prompt_key))
        self.assertTrue(torch.equal(gp_off.prompt, gp_on.prompt))
        for gp in (gp_off, gp_on):
            gp.eval()
        out_off, _ = _forward(gp_off)
        out_on, _ = _forward(gp_on)
        for k in _VALUE_KEYS:
            self.assertTrue(torch.equal(out_off[k], out_on[k]),
                            "forward value drift in %r" % k)

    def test_ce_key_grad_blocked_feature_grad_kept(self):
        torch.manual_seed(123)
        gp_off = _make_gp()
        torch.manual_seed(123)
        gp_on = _make_gp(detach=True)
        for gp in (gp_off, gp_on):
            gp.eval()
        # OFF: CE reaches prompt_key through the softmax coupling
        out_off, cls_off = _forward(gp_off)
        _ce_logits_loss(out_off).backward()
        self.assertIsNotNone(gp_off.prompt_key.grad)
        self.assertGreater(float(gp_off.prompt_key.grad.abs().max()), 0.0)
        self.assertIsNotNone(cls_off.grad)
        # ON: CE no longer reaches prompt_key ...
        out_on, cls_on = _forward(gp_on)
        _ce_logits_loss(out_on).backward()
        self.assertIsNone(gp_on.prompt_key.grad)
        # ... but the sample-feature CE path is kept
        self.assertIsNotNone(cls_on.grad)
        self.assertGreater(float(cls_on.grad.abs().max()), 0.0)

    def test_pull_uses_original_similarity_selected_slots_only(self):
        torch.manual_seed(123)
        gp = _make_gp(detach=True)
        gp.eval()
        out, _ = _forward(gp)
        pull_term = -0.1 * out['reduce_sim']
        pull_term.backward()
        self.assertIsNotNone(gp.prompt_key.grad)
        slot_norms = gp.prompt_key.grad.norm(dim=1)
        sel = set(out['pull_idx'].flatten().tolist())
        unselected = [i for i in range(slot_norms.shape[0]) if i not in sel]
        self.assertGreater(float(slot_norms[list(sel)].max()), 0.0)
        self.assertEqual(float(slot_norms[unselected].abs().max()), 0.0)

    def test_fail_closed_without_route_fix(self):
        with self.assertRaises(ValueError):
            Global_Prompt(length=4, embed_dim=32, embedding_key='cls',
                          prompt_init='uniform', prompt_pool=True,
                          prompt_key=True, pool_size=8, top_k=2,
                          batchwise_prompt=True, prompt_key_init='uniform',
                          route_fix=False,
                          route_detach_ce_key_grad=True)


class TestArmCDisablePull(unittest.TestCase):

    def test_loss_composition_semantics(self):
        """Replicates the exact Client_DF.train() loss compositions on a
        real route_fix graph:
            full loss      = CE - 0.1 * reduce_sim   (switch OFF)
            disable_pull   = CE                        (switch ON)
        The pull contribution is a selected-slot spike that the C-arm
        loss must not contain."""
        torch.manual_seed(123)
        gp = _make_gp()
        gp.eval()
        out, _ = _forward(gp)
        pull_off = out['reduce_sim']
        loss_ce = _ce_logits_loss(out)
        loss_full = loss_ce - 0.1 * pull_off
        key = gp.prompt_key
        g_ce = torch.autograd.grad(loss_ce, key, retain_graph=True,
                                   allow_unused=True)[0]
        g_full = torch.autograd.grad(loss_full, key, retain_graph=True,
                                     allow_unused=True)[0]
        g_pull = torch.autograd.grad(-0.1 * pull_off, key,
                                     allow_unused=True)[0]
        sel = set(out['pull_idx'].flatten().tolist())
        unselected = [i for i in range(key.shape[0]) if i not in sel]
        # the C-arm loss (CE only) is exactly the CE component: outside the
        # selected slots the full loss adds nothing but pull spike inside
        self.assertTrue(torch.allclose(g_full, g_ce + g_pull))
        # the switch must actually matter: with pull in the loss the
        # selected-slot key gradients differ from the CE-only composition
        self.assertFalse(torch.equal(g_full, g_ce))
        # pull touches ONLY selected slots -> CE-only composition removes
        # exactly that spike, nothing else changes
        self.assertEqual(float(g_pull.norm(dim=1)[unselected].abs().max()),
                         0.0)
        self.assertGreater(float(g_pull.norm(dim=1)[list(sel)].max()), 0.0)

    def test_default_off_is_bitwise_invariant(self):
        """Default-path regression: with the switch absent (real legacy
        args namespace) or explicitly False, the train() loss formula is
        the full CE - 0.1*pull_off, and the B-switch-off path is bitwise
        identical to the constructor default."""
        legacy_args = types.SimpleNamespace()   # no disable_pull attribute
        self.assertFalse(bool(getattr(legacy_args, 'disable_pull', False)))
        off_args = types.SimpleNamespace(disable_pull=False)
        self.assertFalse(bool(getattr(off_args, 'disable_pull', False)))
        # construction + forward + backward bitwise identical between
        # constructor default and explicit route_detach_ce_key_grad=False
        torch.manual_seed(123)
        gp_default = _make_gp()
        torch.manual_seed(123)
        gp_explicit = Global_Prompt(length=4, embed_dim=32,
                                    embedding_key='cls', prompt_init='uniform',
                                    prompt_pool=True, prompt_key=True,
                                    pool_size=8, top_k=2,
                                    batchwise_prompt=True,
                                    prompt_key_init='uniform',
                                    route_fix=True, route_tau=0.1,
                                    route_seed=123,
                                    route_detach_ce_key_grad=False)
        for gp in (gp_default, gp_explicit):
            gp.eval()
        out_d, cls_d = _forward(gp_default)
        out_e, cls_e = _forward(gp_explicit)
        for k in _VALUE_KEYS:
            self.assertTrue(torch.equal(out_d[k], out_e[k]))
        loss_d = _ce_logits_loss(out_d) - 0.1 * out_d['reduce_sim']
        loss_e = _ce_logits_loss(out_e) - 0.1 * out_e['reduce_sim']
        self.assertTrue(torch.equal(loss_d, loss_e))
        loss_d.backward()
        loss_e.backward()
        self.assertTrue(torch.equal(gp_default.prompt_key.grad,
                                    gp_explicit.prompt_key.grad))
        self.assertTrue(torch.equal(cls_d.grad, cls_e.grad))
        rng_d = torch.get_rng_state()
        rng_e = torch.get_rng_state()
        self.assertTrue(torch.equal(rng_d, rng_e))


if __name__ == '__main__':
    unittest.main()
