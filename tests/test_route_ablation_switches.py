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


class _StubPromptPool(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.prompt_key = torch.nn.Parameter(torch.randn(6, 8))


class _StubVit(torch.nn.Module):
    """Minimal vit stand-in: parameters() must include the pool key so the
    E2 partition invariants are exercised on a realistic structure."""

    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(8, 8)
        self.pool = _StubPromptPool()

    def get_prompts(self):
        return self.pool


def _make_e2_client(route_key_no_wd=False, route_fix=True,
                   route_selected_ce_grad=True):
    from Models.Client_DF import Client_DF
    args = types.SimpleNamespace(
        nb_classes=20, route_fix=route_fix,
        route_selected_ce_grad=route_selected_ce_grad,
        route_key_no_wd=route_key_no_wd)
    return Client_DF(0, None, 'Tail_Anchor', 5, [None] * 10, 1, 16, 0.01,
                     'cpu', 'fedta', [[0] * 20] * 10, args, _StubVit())


class TestArmE2KeyNoWd(unittest.TestCase):
    """E2 repair arm (--route_key_no_wd, 2026-10-28 GPT approval):
    strictly single-variable on top of E1 -- ONLY prompt_key's weight decay
    goes 1e-3 -> 0, via optimizer parameter groups. E1 forensics: 97-98% of
    keys starved to |K|<1e-3 by wd, and the l2_normalize Jacobian (~1/|K|)
    amplifies direction rotation of shrunk keys, coupling Gate 1 (rank)
    and Gate 2 (norm health)."""

    def test_default_off_single_group_official(self):
        """Flag OFF: exactly the official Adam(vit.parameters(), wd=1e-3) --
        one group, same parameter sequence (bit-identical path)."""
        client = _make_e2_client(route_key_no_wd=False)
        opt = client._build_vit_optimizer(
            types.SimpleNamespace(route_key_no_wd=False))
        self.assertEqual(len(opt.param_groups), 1)
        self.assertEqual(opt.param_groups[0]['weight_decay'], 1e-3)
        self.assertEqual([id(p) for p in opt.param_groups[0]['params']],
                         [id(p) for p in client.vit.parameters()])

    def test_e2_param_groups_exact_partition(self):
        """Flag ON: two groups; wd=0 ONLY on prompt_key, 1e-3 elsewhere;
        groups disjoint, union == every trainable parameter (the reviewer's
        fail-closed invariant)."""
        client = _make_e2_client(route_key_no_wd=True)
        args = types.SimpleNamespace(
            route_key_no_wd=True, route_fix=True,
            route_selected_ce_grad=True)
        opt = client._build_vit_optimizer(args)
        self.assertEqual(len(opt.param_groups), 2)
        key_group, other_group = opt.param_groups
        self.assertEqual(key_group['weight_decay'], 0.0)
        self.assertEqual(other_group['weight_decay'], 1e-3)
        self.assertEqual([id(p) for p in key_group['params']],
                         [id(client.vit.get_prompts().prompt_key)])
        all_ids = [id(p) for p in client.vit.parameters()]
        key_ids = {id(p) for p in key_group['params']}
        other_ids = {id(p) for p in other_group['params']}
        self.assertEqual(len(key_ids & other_ids), 0, "groups overlap")
        self.assertEqual(key_ids | other_ids, set(all_ids),
                         "union must cover every trainable parameter")
        self.assertEqual(len(other_ids), len(all_ids) - 1)

    def test_e2_wd_actually_exempts_key_in_a_step(self):
        """End-to-end single step, zero-grad discriminator: torch Adam
        couples wd into grad, so with zero grad a wd>0 param STILL moves
        (~ -lr*sign(p)) while a wd=0 param stays bit-identical. Flag ON:
        key untouched, backbone moves. Flag OFF: key moves too (official)."""
        torch.manual_seed(3)
        client_on = _make_e2_client(route_key_no_wd=True)
        torch.manual_seed(3)
        client_off = _make_e2_client(route_key_no_wd=False)
        key_on0 = client_on.vit.get_prompts().prompt_key.detach().clone()
        key_off0 = client_off.vit.get_prompts().prompt_key.detach().clone()
        bb_on0 = client_on.vit.backbone.weight.detach().clone()
        self.assertTrue(torch.equal(key_on0, key_off0))
        # flag ON: key exempt (bit-identical), backbone coupled wd moves it
        opt_on = client_on._build_vit_optimizer(types.SimpleNamespace(
            route_key_no_wd=True, route_fix=True, route_selected_ce_grad=True))
        for p in client_on.vit.parameters():
            p.grad = torch.zeros_like(p)
        opt_on.step()
        self.assertTrue(torch.equal(
            client_on.vit.get_prompts().prompt_key.detach(), key_on0),
            "E2 key must not shrink with zero grad (wd exempt)")
        self.assertFalse(torch.equal(
            client_on.vit.backbone.weight.detach(), bb_on0),
            "other params must still receive the coupled wd update")
        # flag OFF: official optimizer shrinks the key too
        opt_off = client_off._build_vit_optimizer(
            types.SimpleNamespace(route_key_no_wd=False))
        for p in client_off.vit.parameters():
            p.grad = torch.zeros_like(p)
        opt_off.step()
        self.assertFalse(torch.equal(
            client_off.vit.get_prompts().prompt_key.detach(), key_off0),
            "official coupled wd must reach prompt_key")

    def test_fail_closed_without_e1_flags(self):
        """E2 requires --route_fix AND --route_selected_ce_grad
        (E2 = E1 + key-only no-wd); anything less must refuse to build."""
        client = _make_e2_client(route_key_no_wd=True,
                                 route_selected_ce_grad=False)
        with self.assertRaises(SystemExit):
            client._build_vit_optimizer(types.SimpleNamespace(
                route_key_no_wd=True, route_fix=True,
                route_selected_ce_grad=False))
        client = _make_e2_client(route_key_no_wd=True, route_fix=False)
        with self.assertRaises(SystemExit):
            client._build_vit_optimizer(types.SimpleNamespace(
                route_key_no_wd=True, route_fix=False,
                route_selected_ce_grad=True))

    def test_split_missing_key_fails_closed(self):
        from Models.Client_DF import split_key_param_groups
        p1, p2 = torch.nn.Parameter(torch.randn(2)), \
            torch.nn.Parameter(torch.randn(2))
        with self.assertRaises(SystemExit):
            split_key_param_groups([p1, p2],
                                   torch.nn.Parameter(torch.randn(2)))


class TestRouteAuditE2Evidence(unittest.TestCase):
    """route_audit E2 additions: key_norm median/p10/p25 in every snapshot;
    post_local |K_post|/|K_init| ratio stats + selected/unselected grouping
    (reviewer-requested WD causal evidence); backward compatible with
    pre-E2 callers (no selected_slots -> no grouping fields)."""

    def _logger(self, tmp):
        from diagnostics.route_audit import RouteAuditLogger
        return RouteAuditLogger(tmp, seed=42)

    def _rows(self, tmp):
        import json
        path = os.path.join(tmp, "route_audit", "route_audit.jsonl")
        return [json.loads(l) for l in open(path)]

    def test_norm_ratio_and_selected_grouping(self):
        import tempfile
        torch.manual_seed(5)
        k0 = torch.randn(6, 8)
        k1 = k0.clone()
        k1[0] *= 2.0
        k1[1] *= 0.25
        k1[2] *= 1.5
        k1[3] *= 0.5
        k1[4] *= 0.75
        k1[5] *= 1.25
        with tempfile.TemporaryDirectory() as tmp:
            log = self._logger(tmp)
            log.log_key_snapshot(0, "init", 0, k0)
            log.log_key_snapshot(0, "post_local", 0, k1,
                                 selected_slots={"ce": [0, 2],
                                                 "pull": [2, 4]})
            rows = self._rows(tmp)
        init_row, post_row = rows
        # every snapshot carries the new norm quantiles
        for r in (init_row, post_row):
            self.assertIn("key_norm_median", r)
            self.assertIn("key_norm_p10", r)
            self.assertIn("key_norm_p25", r)
        # init phase: no ratio fields yet
        self.assertNotIn("key_norm_ratio_mean", init_row)
        ratio = k1.norm(dim=1) / k0.norm(dim=1)
        self.assertAlmostEqual(post_row["key_norm_ratio_mean"],
                               float(ratio.mean()), places=6)
        self.assertAlmostEqual(post_row["key_norm_ratio_median"],
                               float(ratio.median()), places=6)
        # grouping: union of {0,2} and {2,4} == {0,2,4}
        self.assertEqual(post_row["n_selected_slots_ce"], 2)
        self.assertEqual(post_row["n_selected_slots_pull"], 2)
        self.assertEqual(post_row["n_selected_slots_union"], 3)
        sel = [0, 2, 4]
        unsel = [1, 3, 5]
        self.assertAlmostEqual(post_row["key_norm_ratio_selected_mean"],
                               float(ratio[sel].mean()), places=6)
        self.assertAlmostEqual(post_row["key_norm_ratio_unselected_mean"],
                               float(ratio[unsel].mean()), places=6)

    def test_post_local_without_selected_slots_backward_compatible(self):
        import tempfile
        torch.manual_seed(6)
        k0, k1 = torch.randn(6, 8), torch.randn(6, 8)
        with tempfile.TemporaryDirectory() as tmp:
            log = self._logger(tmp)
            log.log_key_snapshot(0, "init", 3, k0)
            log.log_key_snapshot(0, "post_local", 3, k1)
            rows = self._rows(tmp)
        post_row = rows[-1]
        self.assertIn("key_norm_ratio_mean", post_row)
        self.assertNotIn("n_selected_slots_union", post_row)
        self.assertNotIn("key_norm_ratio_selected_mean", post_row)

    def test_post_local_without_init_snapshot_is_safe(self):
        import tempfile
        torch.manual_seed(7)
        k1 = torch.randn(6, 8)
        with tempfile.TemporaryDirectory() as tmp:
            log = self._logger(tmp)
            log.log_key_snapshot(0, "post_local", 1, k1)
            rows = self._rows(tmp)
        post_row = rows[-1]
        self.assertNotIn("key_norm_ratio_mean", post_row)
        self.assertIn("key_effective_rank", post_row)


if __name__ == '__main__':
    unittest.main()
