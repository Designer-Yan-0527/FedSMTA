"""A5-v2 readout-refit / consolidation switch tests (unittest, repo style).

Switches added 2026-10-28 (pre-registered RUNCOMMANDS Sec.23):

  F1 --head_dropout           train-side Dropout p of the phase-2 jointly
                             trained Chead (default 0.5 = official)
  F2 --head_refit             post-phase-2 readout-only refit of heads[task_id]
  F3 --head_standardize       standardize refit input (StandardizedChead,
                             mu/sd as persistent buffers)
     --interface_consolidate  task-boundary refit of heads[t'], t' < task_id

Required semantics, all unit-tested here:

  * construction RNG-neutrality: Chead/Tail_Anchor construction consumes the
    same RNG stream for ANY dropout p (F1 default 0.5 = official bitwise);
  * legacy-args tolerance: missing flags -> official defaults via getattr;
  * default OFF = complete no-op: no joint-head snapshot, heads/model
    untouched, no checkpoint schema key (frozen baseline stays bit-identical);
  * refit is readout-only AND RNG-neutral: key/anchor/joint-head parameters
    and all four global RNG streams are bit-identical after the hook (the
    corrected-metric delta is 100% attributable to the interface swap);
  * heads[] replacement + joint-head trajectory snapshot/restore;
  * refit is reproducible (deterministic stable seed -> identical head);
  * consolidation fires only at task-boundary rounds and only for t' <
    task_id; the current-task refit belongs to --head_refit alone;
  * rebuild_client_head: template / StandardizedChead branches, truncated
    or foreign key sets fail closed;
  * fail-closed: --head_standardize alone refuses to start; an A5 resume
    without a5_joint_head refuses to continue (joint trajectory protection).

Note: Tail_Anchor.forward hardcodes .to('cuda') on the key matrix, so the
hook-level tests require CUDA and are skipped otherwise (repo runs on GPU).
"""
import os
import random
import sys
import types
import unittest
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Models.classification_head import Chead, StandardizedChead  # noqa: E402
from Models.Tail_Anchor import Tail_Anchor  # noqa: E402

_CUDA = torch.cuda.is_available()
_DEVICE = 'cuda' if _CUDA else 'cpu'


class _A5Vit(torch.nn.Module):
    """Minimal vit stand-in: passes the 768-dim input through as 'feat'
    (the only interface _a5_collect_head_inputs needs from the real ViT)."""

    def forward(self, input, task_id=None, cls_features=None, train=False):
        return {'feat': input}


def _task_dataset(task, n, seed):
    """Deterministic per-task dataset; labels are INSIDE the task's class
    mask (masked CE masks everything outside to -inf)."""
    g = torch.Generator().manual_seed(seed + 17 * task)
    feats = torch.randn(n, 768, generator=g)
    labels = torch.randint(2 * task, 2 * task + 2, (n,), generator=g)
    return TensorDataset(feats, labels)


def _make_a5_client(head_refit=False, head_standardize=False,
                   interface_consolidate=False, head_dropout=0.5,
                   task_id=0, legacy_args=False, seed=123, n_rows=24,
                   refit_input=None):
    from Models.Client_DF import Client_DF
    kwargs = dict(nb_classes=20, seed=seed)
    if not legacy_args:
        kwargs.update(head_dropout=head_dropout, head_refit=head_refit,
                      head_standardize=head_standardize,
                      interface_consolidate=interface_consolidate)
        if refit_input is not None:
            kwargs['refit_input'] = refit_input
    args = types.SimpleNamespace(**kwargs)
    class_mask = [[2 * t, 2 * t + 1] for t in range(10)]
    train_data = [_task_dataset(t, n_rows, seed) for t in range(10)]
    torch.manual_seed(seed)
    client = Client_DF(0, None, 'Tail_Anchor', 5, train_data, 1, 16, 0.01,
                       _DEVICE, 'fedta', class_mask, args, _A5Vit())
    client.model.to(_DEVICE)
    client.task_id = task_id
    client.data_split_indices = {
        t: {'train': list(range(n_rows)), 'test': []}
        for t in range(task_id + 1)}
    return client


def _state_equal(s1, s2):
    if set(s1.keys()) != set(s2.keys()):
        return False
    return all(torch.equal(s1[k], s2[k]) for k in s1)


def _np_state_equal(a, b):
    return (a[0] == b[0] and np.array_equal(a[1], b[1]) and a[2] == b[2]
            and np.array_equal(a[3], b[3]) and a[4] == b[4])


def _cuda_rng_equal(a, b):
    if len(a) != len(b):
        return False
    return all(torch.equal(x, y) for x, y in zip(a, b))


class TestConstructionRngNeutrality(unittest.TestCase):
    """F1: dropout p must not change the construction RNG stream (default
    0.5 = official bitwise; p is a pure train-side hyperparameter)."""

    def test_chead_dropout_p_construction_rng_neutral(self):
        torch.manual_seed(123)
        a = Chead(20)
        torch.manual_seed(123)
        b = Chead(20, dropout=0.5)
        torch.manual_seed(123)
        c = Chead(20, dropout=0.1)
        sa, sb, sc = a.state_dict(), b.state_dict(), c.state_dict()
        self.assertEqual(set(sa), set(sb))
        self.assertEqual(set(sa), set(sc))
        for k in sa:
            self.assertTrue(torch.equal(sa[k], sb[k]), k)
            self.assertTrue(torch.equal(sa[k], sc[k]), k)
        # identical RNG budget: the next global draw after construction is
        # the same for any p
        torch.manual_seed(123)
        Chead(20)
        d1 = torch.rand(4)
        torch.manual_seed(123)
        Chead(20, dropout=0.1)
        d2 = torch.rand(4)
        self.assertTrue(torch.equal(d1, d2))

    def test_tail_anchor_dropout_transparent(self):
        torch.manual_seed(123)
        a = Tail_Anchor(10, 768, 200)
        torch.manual_seed(123)
        b = Tail_Anchor(10, 768, 200, dropout=0.1)
        self.assertTrue(torch.equal(a.key, b.key))
        self.assertTrue(torch.equal(a.anchor_pool, b.anchor_pool))
        self.assertTrue(_state_equal(a.head.state_dict(),
                                     b.head.state_dict()))


class TestInitLocalModelDefaults(unittest.TestCase):
    """args wiring: legacy args (no A5 flags) and args=None keep the
    official dropout 0.5; --head_dropout reaches the Tail_Anchor Chead."""

    def test_legacy_args_keep_official_dropout(self):
        client = _make_a5_client(legacy_args=True)
        self.assertEqual(client.model.head.head[2].p, 0.5)
        torch.manual_seed(123)
        legacy = Tail_Anchor(10, 768, 200)
        torch.manual_seed(123)
        explicit = client.init_local_model(
            'Tail_Anchor', types.SimpleNamespace(head_dropout=0.5))
        self.assertTrue(torch.equal(legacy.key, explicit.key))
        self.assertTrue(torch.equal(legacy.anchor_pool, explicit.anchor_pool))
        self.assertTrue(_state_equal(legacy.head.state_dict(),
                                     explicit.head.state_dict()))

    def test_head_dropout_injected(self):
        client = _make_a5_client(head_dropout=0.1)
        self.assertEqual(client.model.head.head[2].p, 0.1)

    def test_args_none_keeps_official_dropout(self):
        client = _make_a5_client()
        m = client.init_local_model('Tail_Anchor', None)
        self.assertEqual(m.head.head[2].p, 0.5)


class TestDefaultOffNoOp(unittest.TestCase):
    """All flags OFF (or absent): the hook is a complete no-op and the
    checkpoint schema keeps the frozen-baseline key set."""

    def test_hook_is_noop_when_off(self):
        client = _make_a5_client(legacy_args=True)
        client.heads[0] = deepcopy(client.model.get_head())
        head_state = deepcopy(client.model.get_head().state_dict())
        key0 = client.model.key.detach().clone()
        rng_t = torch.get_rng_state()
        rng_p = random.getstate()
        rng_n = np.random.get_state()
        client._a5_post_phase2_interface(0, types.SimpleNamespace())
        self.assertIsNone(client._a5_joint_head)
        self.assertTrue(torch.equal(client.model.key.detach(), key0))
        self.assertTrue(_state_equal(client.heads[0].state_dict(), head_state))
        self.assertTrue(
            _state_equal(client.model.get_head().state_dict(), head_state))
        self.assertTrue(torch.equal(torch.get_rng_state(), rng_t))
        self.assertEqual(random.getstate(), rng_p)
        self.assertTrue(_np_state_equal(np.random.get_state(), rng_n))

    def test_checkpoint_schema_off_vs_on(self):
        client_off = _make_a5_client(legacy_args=True)
        self.assertNotIn('a5_joint_head',
                        client_off.state_dict_for_checkpoint())
        client_on = _make_a5_client(head_refit=True)
        self.assertIn('a5_joint_head', client_on.state_dict_for_checkpoint())


@unittest.skipUnless(_CUDA, "Tail_Anchor.forward hardcodes .to('cuda')")
class TestRefitIsolationRngNeutral(unittest.TestCase):
    """Core attribution property (RUNCOMMANDS Sec.23.1): with the switches
    ON, the representation-side trajectory (key / anchor / joint head) and
    all four RNG streams are bit-identical to the official run; only
    heads[] (the interface readout) changes."""

    def _prefill_heads(self, client):
        for t in range(client.task_id + 1):
            torch.manual_seed(1000 + t)
            client.heads[t] = Chead(200, dropout=0.1)
            client.heads[t].to(_DEVICE)

    def _hook_args(self, **flags):
        base = dict(nb_classes=20, seed=7, head_refit=False,
                    head_standardize=False, interface_consolidate=False)
        base.update(flags)
        return types.SimpleNamespace(**base)

    def test_refit_rng_neutral_and_readout_only(self):
        client = _make_a5_client(head_refit=True, task_id=0)
        self._prefill_heads(client)
        key0 = client.model.key.detach().clone()
        anchor0 = client.model.anchor_pool.detach().clone()
        joint_state = deepcopy(client.model.get_head().state_dict())
        rng_t = torch.get_rng_state()
        rng_c = torch.cuda.get_rng_state_all()
        rng_p = random.getstate()
        rng_n = np.random.get_state()

        client._a5_post_phase2_interface(3, self._hook_args(head_refit=True))

        # readout-only: no gradient ever reached the representation side
        self.assertTrue(torch.equal(client.model.key.detach(), key0))
        self.assertTrue(torch.equal(client.model.anchor_pool.detach(),
                                    anchor0))
        # joint-head trajectory: snapshotted AND restored into model.head
        self.assertTrue(_state_equal(client._a5_joint_head.state_dict(),
                                     joint_state))
        self.assertTrue(_state_equal(client.model.get_head().state_dict(),
                                     joint_state))
        # heads[] replaced by a different, freshly-trained readout head
        self.assertFalse(_state_equal(client.heads[0].state_dict(),
                                     joint_state))
        self.assertIsInstance(client.heads[0], Chead)
        self.assertEqual(client.heads[0].head[2].p, 0.1)
        # four RNG streams restored bit-identically (training-stream neutral)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng_t))
        self.assertTrue(_cuda_rng_equal(torch.cuda.get_rng_state_all(),
                                       rng_c))
        self.assertEqual(random.getstate(), rng_p)
        self.assertTrue(_np_state_equal(np.random.get_state(), rng_n))

    def test_refit_deterministic(self):
        """Stable-seed refit: two identically-constructed clients produce
        bit-identical refit heads (reproducibility of the arm itself)."""
        c1 = _make_a5_client(head_refit=True, task_id=0, seed=123)
        c2 = _make_a5_client(head_refit=True, task_id=0, seed=123)
        args = self._hook_args(head_refit=True)
        c1._a5_post_phase2_interface(2, args)
        c2._a5_post_phase2_interface(2, args)
        self.assertTrue(_state_equal(c1.heads[0].state_dict(),
                                     c2.heads[0].state_dict()))

    def test_standardize_refit_head_structure(self):
        """F3: the refit head is a StandardizedChead whose buffers equal the
        train-split statistics of the actually-collected head inputs."""
        client = _make_a5_client(head_refit=True, head_standardize=True,
                                 task_id=0)
        x1, _ = client._a5_collect_head_inputs(0)
        client._a5_post_phase2_interface(
            0, self._hook_args(head_refit=True, head_standardize=True))
        h = client.heads[0]
        self.assertIsInstance(h, StandardizedChead)
        self.assertTrue(torch.allclose(h.mean, x1.mean(dim=0)))
        self.assertTrue(torch.allclose(h.std, x1.std(dim=0) + 1e-8))
        h.eval()
        with torch.no_grad():
            logits = h(x1)
        self.assertEqual(tuple(logits.shape), (x1.shape[0], 200))

    def test_refit_only_touches_current_task(self):
        client = _make_a5_client(head_refit=True, task_id=1)
        self._prefill_heads(client)
        pre = [deepcopy(client.heads[t].state_dict())
               for t in range(client.task_id + 1)]
        client._a5_post_phase2_interface(2, self._hook_args(head_refit=True))
        # old task untouched
        self.assertTrue(_state_equal(client.heads[0].state_dict(), pre[0]))
        # current task replaced
        self.assertFalse(_state_equal(client.heads[1].state_dict(), pre[1]))

    def test_consolidate_boundary_semantics(self):
        """--interface_consolidate: refits ONLY old tasks and ONLY at
        task-boundary rounds ((round+1) % task_per_global_epoch == 0)."""
        client = _make_a5_client(interface_consolidate=True, task_id=2)
        self._prefill_heads(client)
        pre = [deepcopy(client.heads[t].state_dict())
               for t in range(client.task_id + 1)]
        args = self._hook_args(interface_consolidate=True)
        # non-boundary round: nothing replaced anywhere
        client._a5_post_phase2_interface(3, args)
        for t in range(client.task_id + 1):
            self.assertTrue(_state_equal(client.heads[t].state_dict(),
                                        pre[t]), t)
        # boundary round: old tasks t' < task_id replaced ...
        client._a5_post_phase2_interface(4, args)
        for t in range(client.task_id):
            self.assertFalse(_state_equal(client.heads[t].state_dict(),
                                          pre[t]), t)
        # ... current task NOT (that belongs to --head_refit)
        self.assertTrue(_state_equal(client.heads[2].state_dict(), pre[2]))


class TestRebuildClientHead(unittest.TestCase):
    """resume head rebuild: template / StandardizedChead branches + foreign
    or truncated key sets fail closed (rebuild_global_head precedent)."""

    def test_template_branch(self):
        from Models.Client_DF import rebuild_client_head
        torch.manual_seed(5)
        template = Chead(200)
        torch.manual_seed(6)
        source = Chead(200, dropout=0.1)
        template_before = deepcopy(template.state_dict())
        rebuilt = rebuild_client_head(source.state_dict(), template)
        self.assertIsInstance(rebuilt, Chead)
        self.assertTrue(_state_equal(rebuilt.state_dict(),
                                     source.state_dict()))
        # template itself must not be mutated
        self.assertTrue(_state_equal(template.state_dict(), template_before))

    def test_standardized_branch(self):
        from Models.Client_DF import rebuild_client_head
        torch.manual_seed(7)
        mean = torch.randn(1536)
        std = torch.rand(1536) + 0.5
        src = StandardizedChead(200, mean, std)
        rebuilt = rebuild_client_head(src.state_dict(), Chead(200))
        self.assertIsInstance(rebuilt, StandardizedChead)
        self.assertTrue(_state_equal(rebuilt.state_dict(),
                                     src.state_dict()))
        src.eval()
        rebuilt.eval()
        x = torch.randn(3, 1536)
        with torch.no_grad():
            self.assertTrue(torch.equal(src(x), rebuilt(x)))

    def test_foreign_or_truncated_keys_fail_closed(self):
        from Models.Client_DF import rebuild_client_head
        with self.assertRaises(RuntimeError):
            rebuild_client_head({'bogus': torch.zeros(1)}, Chead(200))
        s = Chead(200).state_dict()
        s.pop('head.0.bias')
        with self.assertRaises(RuntimeError):
            rebuild_client_head(s, Chead(200))


class TestFailClosed(unittest.TestCase):
    """fail-closed semantics: refuse to start / resume on configurations
    whose science would silently differ from the intended arm."""

    def _make_server(self, **flags):
        from Models.Server_DF import Server_DF
        args = types.SimpleNamespace(nb_classes=20, **flags)
        return Server_DF('Server', None, 'Tail_Anchor', 1, 1, [None],
                         [[0, 1]], 0.01, 5, 1, 16, 'cpu', 'fedta', 0.25,
                         None, None, args, None)

    def test_standardize_alone_refuses_to_start(self):
        with self.assertRaises(SystemExit):
            self._make_server(head_standardize=True)

    def test_standardize_with_refit_starts(self):
        self.assertIsNotNone(
            self._make_server(head_standardize=True, head_refit=True))

    def test_standardize_with_consolidate_starts(self):
        self.assertIsNotNone(
            self._make_server(head_standardize=True,
                              interface_consolidate=True))

    def test_legacy_server_args_accepted(self):
        """old checkpoints / legacy args carry no A5 flags at all: the
        getattr defaults keep the official behavior."""
        self.assertIsNotNone(self._make_server())

    def test_resume_missing_a5_joint_head_refused(self):
        client = _make_a5_client(head_refit=True)
        state = {'id': 0, 'task_id': 0,
                 'tail_anchor_model': client.model.state_dict(),
                 'prompts': None, 'heads': []}
        with self.assertRaises(RuntimeError):
            client.load_checkpoint_state(state)

    def test_resume_with_a5_joint_head_roundtrip(self):
        client = _make_a5_client(head_refit=True)
        client._a5_joint_head = deepcopy(client.model.get_head())
        state = client.state_dict_for_checkpoint()
        self.assertIn('a5_joint_head', state)
        client2 = _make_a5_client(head_refit=True)
        client2.load_checkpoint_state(state)
        self.assertTrue(_state_equal(client2._a5_joint_head.state_dict(),
                                     client._a5_joint_head.state_dict()))

    def test_tail_anchor_state_uses_joint_head_when_on(self):
        """修复回归：end-of-round evaluate 会把 model.head 装成 refit 头
        （standardize 臂 = StandardizedChead，带 mean/std buffer）。ON 臂
        checkpoint 的 tail_anchor_model 必须用联训头覆写 head.*，schema
        保持 Chead 键集（否则后续 resume 严格 load_state_dict 拒载）。"""
        client = _make_a5_client(head_refit=True)
        client._a5_joint_head = deepcopy(client.model.get_head())
        torch.manual_seed(11)
        polluted = StandardizedChead(
            200, torch.randn(1536), torch.rand(1536) + 0.5)
        client.model.load_head(polluted)  # 模拟 evaluate 后的污染状态
        state = client.state_dict_for_checkpoint()
        tam = state['tail_anchor_model']
        self.assertNotIn('head.mean', tam)
        self.assertNotIn('head.std', tam)
        jh = client._a5_joint_head.state_dict()
        for k in jh:
            self.assertTrue(torch.equal(tam['head.' + k], jh[k].cpu()), k)

    def test_legacy_standardized_checkpoint_resume(self):
        """修复回归：老版本（2026-10-29 修复前）standardize 臂 checkpoint 的
        tail_anchor_model 混有 head.mean/head.std 污染键，ON 臂 resume 剔除
        后可正常装载（联训头由 a5_joint_head 恢复）；缺 a5_joint_head 仍
        fail-closed。"""
        from runtime_utils import module_state_or_none
        client = _make_a5_client(head_refit=True, head_standardize=True)
        ms = module_state_or_none(client.model)
        ms['head.mean'] = torch.zeros(1536)
        ms['head.std'] = torch.ones(1536)
        state = {'id': 0, 'task_id': 0, 'tail_anchor_model': ms,
                 'prompts': None, 'heads': []}
        # 缺 a5_joint_head -> fail-closed（在污染键剔除之后仍能命中）
        with self.assertRaises(RuntimeError):
            client.load_checkpoint_state(state)
        # 补上 a5_joint_head -> 剔除污染键后成功装载
        client._a5_joint_head = deepcopy(client.model.get_head())
        state['a5_joint_head'] = module_state_or_none(client._a5_joint_head)
        client2 = _make_a5_client(head_refit=True, head_standardize=True)
        client2.load_checkpoint_state(state)
        self.assertTrue(_state_equal(client2._a5_joint_head.state_dict(),
                                     client._a5_joint_head.state_dict()))


class TestRefitInputFeat(unittest.TestCase):
    """A5-v2 §23.6-1 修复臂（--refit_input feat，INR 双探针诊断
    2026-10-29 预注册判读的落地）：refit 输入 = 768 维 pre-anchor z_op，
    部署为 1536 输入、锚半区权重硬置零的头——evaluate/checkpoint/resume
    路径与 x1 臂完全同构（state keys 不变）。"""

    def _hook_args(self, **flags):
        base = dict(nb_classes=20, seed=7, head_refit=False,
                    head_standardize=False, interface_consolidate=False,
                    refit_input='x1')
        base.update(flags)
        return types.SimpleNamespace(**base)

    def test_feat_head_zeroes_anchor_half(self):
        """置零结构：feat 臂 first Linear 锚半区恒 0、z_op 半区非 0；
        x1 臂（默认）锚半区正常训练不置零——两臂部署形式可区分。"""
        client = _make_a5_client(head_refit=True, task_id=0)
        client._a5_post_phase2_interface(
            0, self._hook_args(head_refit=True, refit_input='feat'))
        w = client.heads[0].head[0].weight
        self.assertEqual(w.shape[1], 1536)
        self.assertTrue(torch.equal(w[:, 768:], torch.zeros_like(w[:, 768:])))
        self.assertFalse(torch.equal(w[:, :768], torch.zeros_like(w[:, :768])))
        client2 = _make_a5_client(head_refit=True, task_id=0)
        client2._a5_post_phase2_interface(
            0, self._hook_args(head_refit=True))
        w2 = client2.heads[0].head[0].weight
        self.assertFalse(torch.equal(w2[:, 768:], torch.zeros_like(w2[:, 768:])))

    def test_feat_head_deploy_equivalence(self):
        """部署语义（evaluate 路径零改动的等价性）：head(x1) 只依赖
        前 768 维（z_op），锚半区替换成任意噪声不改变 logits。对
        Chead 与 StandardizedChead 两种部署形态都成立。"""
        for standardize in (False, True):
            client = _make_a5_client(head_refit=True, task_id=0)
            client._a5_post_phase2_interface(
                0, self._hook_args(head_refit=True, refit_input='feat',
                                   head_standardize=standardize))
            h = client.heads[0]
            h.eval()
            g = torch.Generator().manual_seed(5)
            z = torch.randn(6, 768, generator=g).to(h.head[0].weight.device)
            anchor = torch.randn(6, 768, generator=g).to(z.device)
            with torch.no_grad():
                out_anchor = h(torch.cat([z, anchor], dim=1))
                out_zero = h(torch.cat([z, torch.zeros_like(anchor)], dim=1))
            self.assertTrue(torch.equal(out_anchor, out_zero),
                            'standardize=%s' % standardize)

    def test_feat_head_state_keys_unchanged(self):
        """schema 兼容：feat 臂头的 state keys 与 x1 臂同形——resume 的
        rebuild_client_head 无需新分支。"""
        cx = _make_a5_client(head_refit=True, task_id=0)
        cx._a5_post_phase2_interface(0, self._hook_args(head_refit=True))
        cf = _make_a5_client(head_refit=True, task_id=0)
        cf._a5_post_phase2_interface(
            0, self._hook_args(head_refit=True, refit_input='feat'))
        self.assertEqual(set(cx.heads[0].state_dict().keys()),
                         set(cf.heads[0].state_dict().keys()))

    def test_refit_input_default_is_x1(self):
        """legacy args（无 refit_input 键）与显式 'x1' 的 refit 头逐位
        一致（getattr 默认路径）。"""
        c1 = _make_a5_client(head_refit=True, task_id=0)
        a1 = self._hook_args(head_refit=True)
        del a1.refit_input  # legacy checkpoint / 旧 args
        c2 = _make_a5_client(head_refit=True, task_id=0)
        c1._a5_post_phase2_interface(0, a1)
        c2._a5_post_phase2_interface(
            0, self._hook_args(head_refit=True, refit_input='x1'))
        self.assertTrue(_state_equal(c1.heads[0].state_dict(),
                                     c2.heads[0].state_dict()))

    def test_refit_input_feat_alone_refuses_to_start(self):
        """fail-closed：--refit_input feat 无 refit/consolidation 时拒绝
        启动（无作用对象）。"""
        from Models.Server_DF import Server_DF
        args = types.SimpleNamespace(nb_classes=20, refit_input='feat')
        with self.assertRaises(SystemExit):
            Server_DF('Server', None, 'Tail_Anchor', 1, 1, [None],
                      [[0, 1]], 0.01, 5, 1, 16, 'cpu', 'fedta', 0.25,
                      None, None, args, None)

    def test_refit_input_feat_with_refit_starts(self):
        from Models.Server_DF import Server_DF
        args = types.SimpleNamespace(nb_classes=20, refit_input='feat',
                                     head_refit=True)
        server = Server_DF('Server', None, 'Tail_Anchor', 1, 1, [None],
                            [[0, 1]], 0.01, 5, 1, 16, 'cpu', 'fedta', 0.25,
                            None, None, args, None)
        self.assertIsNotNone(server)


if __name__ == '__main__':
    unittest.main()
