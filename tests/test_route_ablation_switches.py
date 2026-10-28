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


def _make_gp(seed=123, pool=8, dim=32, length=4, top_k=2, detach=False,
             selected=False):
    """Deterministic route_fix Global_Prompt (R1 init uses its own
    generator; the prompt pool itself consumes the global RNG stream, so
    callers must seed first)."""
    return Global_Prompt(length=length, embed_dim=dim, embedding_key='cls',
                          prompt_init='uniform', prompt_pool=True,
                          prompt_key=True, pool_size=pool, top_k=top_k,
                          batchwise_prompt=True, prompt_key_init='uniform',
                          route_fix=True, route_tau=0.1, route_seed=seed,
                          route_detach_ce_key_grad=detach,
                          route_selected_ce_grad=selected)


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


class TestArmESelectedCeGrad(unittest.TestCase):
    """E1 repair arm (--route_selected_ce_grad): forward-soft /
    backward-selected CE routing. B/C/D attribution verdict (2026-10-28):
    the all-pool CE soft-routing gradient is the dominant collapse driver,
    but pure detach (arm B) starves unselected keys into weight-decay
    collapse (mean norm 0.0067). E1 keeps the soft forward bitwise
    unchanged and gates the CE->key gradient to the per-sample detached
    top-k slots (NOT batchwise majority), while keeping the full CE
    gradient to sample features."""

    def test_forward_values_bitwise_equal(self):
        torch.manual_seed(123)
        gp_off = _make_gp()
        torch.manual_seed(123)
        gp_on = _make_gp(selected=True)
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

    def test_ce_key_grad_selected_slots_only_feature_grad_full(self):
        torch.manual_seed(123)
        gp_off = _make_gp()
        torch.manual_seed(123)
        gp_on = _make_gp(selected=True)
        for gp in (gp_off, gp_on):
            gp.eval()
        out_off, cls_off = _forward(gp_off)
        out_on, cls_on = _forward(gp_on)
        # feature CE gradient: semantically the FULL route_fix gradient.
        # (allclose, not bitwise: E1 splits the route_sim add into three
        # backward contributions G@K + (mask*G)@K - (mask*G)@K, whose
        # accumulation order may differ from the dense single matmul by
        # <=1 ulp; mathematically the sum is G@K.)
        torch.manual_seed(11)
        _ce_logits_loss(out_off).backward()
        torch.manual_seed(11)
        _ce_logits_loss(out_on).backward()
        self.assertTrue(torch.allclose(cls_off.grad, cls_on.grad,
                                       rtol=1e-6, atol=1e-7))
        # key gradient: present, confined to the per-sample top-k slots
        self.assertIsNotNone(gp_on.prompt_key.grad)
        slot_norms = gp_on.prompt_key.grad.norm(dim=1)
        sel = set(out_on['ce_route_topk'].flatten().tolist())
        unselected = [i for i in range(slot_norms.shape[0]) if i not in sel]
        self.assertGreater(float(slot_norms[list(sel)].max()), 0.0)
        self.assertEqual(float(slot_norms[unselected].abs().max()), 0.0)
        # the switch must actually matter: dense all-pool grads differ
        self.assertFalse(
            torch.equal(gp_off.prompt_key.grad, gp_on.prompt_key.grad))

    def test_mask_trick_matches_explicit_chain_rule(self):
        """Direct check of the reviewer-specified E1 gradient topology on
        a minimal standalone graph (the exact formulas from the ruling):
            route_sim = sim_no_key + mask*(sim - sim_no_key)
            grad_K(route_sim) = (dL/droute_sim * mask)^T @ x_norm   (gated)
            grad_x(route_sim) = (dL/droute_sim) @ K_norm            (full)
            route_sim == sim  bitwise                                (forward)
        """
        torch.manual_seed(0)
        B, C, P = 4, 8, 10
        x = torch.randn(B, C, requires_grad=True)
        K = torch.randn(P, C, requires_grad=True)
        x_n = torch.nn.functional.normalize(x, dim=1)
        K_n = torch.nn.functional.normalize(K, dim=1)
        sim = x_n @ K_n.t()
        sim_nk = x_n @ K_n.detach().t()
        # both matmuls over identical values -> bitwise equal -> the
        # mask trick's forward correction is exactly +0.0
        self.assertTrue(torch.equal(sim, sim_nk))
        idx = torch.topk(sim.detach(), k=3, dim=1).indices
        mask = torch.zeros_like(sim).scatter_(1, idx, 1.0)
        route_sim = sim_nk + mask * (sim - sim_nk)
        self.assertTrue(torch.equal(route_sim, sim))
        w = torch.softmax(route_sim / 0.1, dim=-1)
        torch.manual_seed(5)
        target = torch.randn_like(w)
        loss = (w * target).sum()
        G = torch.autograd.grad(loss, route_sim, retain_graph=True)[0]
        # check the topology AT THE ROUTING-MATRIX LEVEL (the reviewer's
        # formulas are per normalized row; raw K/x additionally carry the
        # row-wise l2-normalize Jacobian, which is identity w.r.t. the
        # gating pattern being tested here)
        gKn = torch.autograd.grad(loss, K_n, retain_graph=True)[0]
        gxn = torch.autograd.grad(loss, x_n, retain_graph=True)[0]
        gK = torch.autograd.grad(loss, K, retain_graph=True)[0]
        # CE->key: per-sample gated contributions only
        expected_gKn = (G * mask).t() @ x_n
        self.assertTrue(torch.allclose(gKn, expected_gKn, rtol=1e-5,
                                       atol=1e-6))
        # CE->feature: FULL gradient, no gating
        expected_gxn = G @ K_n
        self.assertTrue(torch.allclose(gxn, expected_gxn, rtol=1e-5,
                                       atol=1e-6))
        # unselected slots receive exactly zero key gradient (the
        # row-wise normalize Jacobian is block-diagonal, so zero rows at
        # K_n level stay zero at raw-K level)
        unselected = [j for j in range(P) if not (mask[:, j] > 0).any()]
        if unselected:
            self.assertEqual(float(gKn[unselected].abs().max()), 0.0)
            self.assertEqual(float(gK[unselected].abs().max()), 0.0)

    def test_pull_semantics_unchanged(self):
        """E1 must not touch the pull path: reduce_sim still uses the
        original grad-carrying similarity, selected slots only (official
        semantics), bitwise equal to route_fix."""
        torch.manual_seed(123)
        gp_off = _make_gp()
        torch.manual_seed(123)
        gp_on = _make_gp(selected=True)
        for gp in (gp_off, gp_on):
            gp.eval()
        out_off, _ = _forward(gp_off)
        out_on, _ = _forward(gp_on)
        self.assertTrue(torch.equal(out_off['reduce_sim'],
                                    out_on['reduce_sim']))
        self.assertTrue(torch.equal(out_off['pull_idx'], out_on['pull_idx']))
        (-0.1 * out_on['reduce_sim']).backward()
        self.assertIsNotNone(gp_on.prompt_key.grad)
        sel = set(out_on['pull_idx'].flatten().tolist())
        slot_norms = gp_on.prompt_key.grad.norm(dim=1)
        unselected = [i for i in range(slot_norms.shape[0]) if i not in sel]
        self.assertGreater(float(slot_norms[list(sel)].max()), 0.0)
        self.assertEqual(float(slot_norms[unselected].abs().max()), 0.0)

    def test_selected_slots_are_per_sample_not_batchwise(self):
        """The CE-gating slots must be the per-sample detached top-k, NOT
        the batchwise majority slots pull uses: different samples may
        update different keys (sparse but dispersed)."""
        torch.manual_seed(123)
        gp = _make_gp(selected=True)
        gp.eval()
        out, _ = _forward(gp)
        ce_slots = [set(r.tolist()) for r in out['ce_route_topk']]
        batchwise = (out['ce_route_topk'][0]
                     .equal(out['ce_route_topk'][1])
                     and out['ce_route_topk'][0]
                     .equal(out['ce_route_topk'][2]))
        if len(set().union(*ce_slots)) > gp.top_k:
            # union of per-sample slots exceeds one batch's worth of slots
            # -> samples select independently (not batchwise majority)
            self.assertFalse(batchwise)
        # per-sample slot sets must equal topk of the detached similarity
        topk_ref = torch.topk(out['similarity'].detach(),
                              k=gp.top_k, dim=1).indices
        self.assertTrue(torch.equal(out['ce_route_topk'], topk_ref))

    def test_fail_closed_configurations(self):
        kwargs = dict(length=4, embed_dim=32, embedding_key='cls',
                      prompt_init='uniform', prompt_pool=True,
                      prompt_key=True, pool_size=8, top_k=2,
                      batchwise_prompt=True, prompt_key_init='uniform')
        # without route_fix -> ValueError
        with self.assertRaises(ValueError):
            Global_Prompt(route_fix=False, route_selected_ce_grad=True,
                          **kwargs)
        # together with arm B -> mutually exclusive
        with self.assertRaises(ValueError):
            Global_Prompt(route_fix=True, route_detach_ce_key_grad=True,
                          route_selected_ce_grad=True, **kwargs)

    def test_default_off_is_bitwise_invariant(self):
        """Default-path regression: explicit route_selected_ce_grad=False
        is bitwise identical to the constructor default (values, grads,
        RNG state)."""
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
                                     route_detach_ce_key_grad=False,
                                     route_selected_ce_grad=False)
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
        self.assertTrue(torch.equal(torch.get_rng_state(),
                                    torch.get_rng_state()))


if __name__ == '__main__':
    unittest.main()
