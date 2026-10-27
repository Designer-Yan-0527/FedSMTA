"""--route_audit instrumentation invariance tests (unittest, repo style).

Per external review (2026-10-27), the audit must be PROVEN non-invasive:
  1. key_stats calibre: orthonormal keys -> effective rank == pool-1
     (centered), off-diagonal cosine ~ 0, sha256 stable.
  2. The production grad-audit path (RouteAuditLogger.log_client_grad_audit
     on a REAL route_fix forward graph) must not write .grad, must not
     consume RNG, and the subsequent loss.backward() + optimizer step must
     be BITWISE identical to a no-audit run from the same seed.
  3. The fusion audit must distinguish the model-side student key (finite
     gradient, the throwaway deepcopy the forward runs through) from the
     optimizer-target key (my_result.prompt_key) -- the suspected
     structural no-op of kd_fusion_prompt is reproduced synthetically
     here and must be reported as optimizer_target_grad_was_none=True.
  4. JSONL/NPZ outputs carry the required fields.
"""
import hashlib
import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Models.global_prompt import Global_Prompt  # noqa: E402
from diagnostics.route_audit import (  # noqa: E402
    RouteAuditLogger, key_stats, sha256_tensor)


def _make_gp(seed=123, pool=8, dim=32, length=4, top_k=2):
    """Deterministic route_fix Global_Prompt (R1 init uses its own
    generator; the prompt pool itself consumes the global RNG stream, so
    callers must seed first)."""
    return Global_Prompt(length=length, embed_dim=dim, embedding_key='cls',
                          prompt_init='uniform', prompt_pool=True,
                          prompt_key=True, pool_size=pool, top_k=top_k,
                          batchwise_prompt=True, prompt_key_init='uniform',
                          route_fix=True, route_tau=0.1, route_seed=seed)


def _forward_loss(gp, head_w):
    """One route_fix forward + CE (through soft routing) + official pull.

    CE reaches prompt_key only through the soft-routing weights
    w = softmax(sim/tau); the pull term is the official -0.1 * reduce_sim."""
    torch.manual_seed(7)
    x = torch.randn(4, 8, 32)
    cls = torch.randn(4, 32)
    labels = torch.tensor([0, 1, 2, 3])
    out = gp(x, prompt_mask=None, cls_features=cls)
    emb = out['prompted_embedding']
    feat = emb.mean(dim=1)               # (B, C): mean over prompt tokens
    logits = torch.nn.functional.linear(feat, head_w)
    loss_ce = torch.nn.functional.cross_entropy(logits, labels)
    pull_term = -0.1 * out['reduce_sim']
    return loss_ce + pull_term, loss_ce, pull_term, out


def _run_one_step(audit_logger):
    """Full determinism-controlled run of one local-training-like step,
    with or without the production grad audit. Returns post-step params
    and the RNG state afterwards."""
    torch.manual_seed(0)
    gp = _make_gp()
    head_w = torch.randn(5, 32)  # requires_grad=False by default

    loss, loss_ce, pull_term, out = _forward_loss(gp, head_w)
    if audit_logger is not None:
        audit_logger.log_client_grad_audit(
            0, 0, 0, loss_ce, pull_term, gp.prompt_key,
            out.get('pull_idx'))
    opt = torch.optim.SGD([gp.prompt_key, gp.prompt], lr=0.1)
    opt.zero_grad()
    loss.backward()
    opt.step()
    return (gp.prompt_key.detach().clone(),
            gp.prompt.detach().clone(),
            torch.get_rng_state())


class RouteAuditInvarianceTest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="route_audit_test_")
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def _read_rows(self, logger):
        with open(logger.jsonl_path) as f:
            return [json.loads(line) for line in f]

    def test_key_stats_orthonormal(self):
        torch.manual_seed(3)
        raw = torch.randn(32, 8)          # dim x pool
        q, _ = torch.linalg.qr(raw)       # columns orthonormal
        keys = q.t().contiguous()         # pool x dim, rows orthonormal
        st = key_stats(keys)
        self.assertEqual(st["pool_size"], 8)
        # centered rank ceiling is pool-1
        self.assertTrue(abs(st["key_effective_rank"] - 7.0) < 0.2)
        self.assertTrue(st["mean_abs_offdiag_cosine"] < 1e-5)
        self.assertEqual(st["key_sha256"], sha256_tensor(keys))
        self.assertEqual(
            st["key_sha256"],
            hashlib.sha256(
                keys.detach().cpu().float().contiguous().numpy().tobytes()
            ).hexdigest())

    def test_grad_audit_is_bitwise_invariant(self):
        # baseline run without audit
        k0, p0, rng0 = _run_one_step(None)
        # audited run through the PRODUCTION logger path
        logger = RouteAuditLogger(self.tmp, seed=42)
        k1, p1, rng1 = _run_one_step(logger)

        self.assertTrue(torch.equal(k0, k1),
                        "prompt_key update changed by audit")
        self.assertTrue(torch.equal(p0, p1),
                        "prompt update changed by audit")
        self.assertTrue(torch.equal(rng0, rng1),
                        "global RNG stream shifted by audit")

        # audit semantics: pull grads confined to selected slots; CE
        # reaches the whole pool through soft routing (A-plan property)
        row = self._read_rows(logger)[-1]
        self.assertEqual(row["phase"], "grad_audit")
        self.assertTrue(row["g_ce_exists"])
        self.assertTrue(row["g_pull_exists"])
        self.assertTrue(row["g_pull_unselected_is_zero"])
        self.assertTrue(row["key_grad_none_after_audit"])
        npz = np.load(os.path.join(logger.dir, row["npz"]))
        sel = npz["pull_selected_idx"]
        slot_pull = npz["g_pull_slot_norms"]
        mask = np.zeros(slot_pull.shape[0], dtype=bool)
        mask[np.unique(sel)] = True
        self.assertEqual(float(np.abs(slot_pull[~mask]).max()), 0.0)

    def test_grad_audit_does_not_write_grad(self):
        torch.manual_seed(0)
        gp = _make_gp()
        head_w = torch.randn(5, 32)
        loss, loss_ce, pull_term, out = _forward_loss(gp, head_w)
        self.assertIsNone(gp.prompt_key.grad)
        logger = RouteAuditLogger(self.tmp, seed=1)
        logger.log_client_grad_audit(0, 0, 0, loss_ce, pull_term,
                                     gp.prompt_key, out.get('pull_idx'))
        self.assertIsNone(
            gp.prompt_key.grad, "autograd.grad must not write .grad")
        # graph still usable for the real training backward
        loss.backward()
        self.assertIsNotNone(gp.prompt_key.grad)

    def test_fusion_audit_detects_optimizer_target_mismatch(self):
        """Synthetic replica of the kd_fusion_prompt structure: the forward
        runs through a deepcopy installed into the 'model' while the
        optimizer targets the ORIGINAL object. The audit must report finite
        gradients on the model-side key and None on the optimizer-side key
        (structural no-op detector)."""
        from copy import deepcopy
        torch.manual_seed(0)
        src = _make_gp()                    # my_result analog
        model_prompt = deepcopy(src)        # self.model.prompt analog
        x = torch.randn(4, 8, 32)
        cls = torch.randn(4, 32)
        labels = torch.tensor([0, 1, 2, 3])
        head_w = torch.randn(5, 32)
        out = model_prompt(x, prompt_mask=None, cls_features=cls)
        emb = out['prompted_embedding']
        feat = emb.mean(dim=1)
        loss = torch.nn.functional.cross_entropy(
            torch.nn.functional.linear(feat, head_w), labels) \
            - 0.1 * out['reduce_sim']

        logger = RouteAuditLogger(self.tmp, seed=42)
        logger.log_fusion_audit(
            0, 0, loss,
            model_prompt.prompt_key,   # what the forward runs through
            src.prompt_key,            # what the optimizer steps
            out['prompted_embedding'], [out['prompted_embedding']],
            [1], [float(loss.detach())])

        row = [r for r in self._read_rows(logger)
               if r["phase"] == "fusion_audit"][-1]
        self.assertTrue(row["g_model_key_exists"])
        self.assertFalse(row["g_optimizer_key_exists"])
        self.assertTrue(row["optimizer_target_grad_was_none"])
        self.assertGreater(row["g_model_key_total_norm"], 0.0)

    def test_logger_snapshot_and_cross_branch(self):
        torch.manual_seed(2)
        logger = RouteAuditLogger(self.tmp, seed=42)
        ka = torch.randn(8, 32)
        kb = torch.randn(8, 32)
        logger.log_key_snapshot(0, "init", 0, ka)
        logger.log_key_snapshot(0, "post_local", 1, kb)
        logger.log_cross_branch(0, {0: ka, 1: kb})

        rows = self._read_rows(logger)
        for row in rows:
            for field in ("ts", "seed", "round", "phase"):
                self.assertIn(field, row)
        snaps = [r for r in rows if r["phase"] in ("init", "post_local")]
        self.assertEqual([r["client_id"] for r in snaps], [0, 1])
        for row in snaps:
            self.assertTrue(
                os.path.exists(os.path.join(logger.dir, row["npz"])))
        cb = [r for r in rows if r["phase"] == "cross_branch"][0]
        m = np.array(cb["same_slot_cosine_matrix"])
        self.assertEqual(m.shape, (2, 2))
        self.assertTrue(np.allclose(m, m.T))
        sub = np.array(cb["subspace_top10_cosine_matrix"])
        self.assertEqual(sub.shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
