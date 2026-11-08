import json
import time
from copy import deepcopy
from pathlib import Path
import random

import numpy as np
import torch
from timm.optim import create_optimizer
from torch import nn
from torch.autograd import Variable
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from tqdm import tqdm

from runtime_utils import module_state_or_none, clone_prompt_template

# from Models.ResNet import resnet18_cbam
from Models.Tail_Anchor import Tail_Anchor

from Models.classification_head import Chead

from data.cifar100_subset_spliter import CustomedSubset
from data.iCIFAR100c import iCIFAR100c
from utils import accuracy, CosineSimilarityClassifier
import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# from sklearn.manifold import MDS
# import pandas as pd
#
# import seaborn as sns


# ----------------------------------------------------------------------
# A5-v2 解耦接口 refit（RUNCOMMANDS §23，预注册 2026-10-28；
# F1 head_dropout / F4 巩固臂已随判死方向清理删除，存档见 git 历史）
# ----------------------------------------------------------------------
import hashlib as _a5_hashlib

from Models.classification_head import StandardizedChead as _A5StandardizedChead

from Models.ctrp import (forward_with_prompt, row_ce, hash_prompt_module,
                        hash_tensor, alpha_blend, stratified_split)

# §34.4/§34.4a B1 DSAR sidecar（冻结部署：当前任务 z⁺→B→L→TA_t、
# 历史任务 z⁺→B→TA_k；插入点 = evaluate 的 self.model() 之前）
from Models.DSAR import (DSARSidecar, DSAR_VERSION, dsar_identity_sidecar,
                         dsar_fit_bridge, dsar_fit_l, dsar_stable_seed,
                         dsar_recipe)

# refit 固定配方（预注册；探针 probe_oracle_headroom.py 的训练日程，
# 仅 hidden 取 512 以保持 heads 存储 = Chead schema，见 §23.6-2）
A5_REFIT_HIDDEN = 512
A5_REFIT_DROPOUT = 0.1
A5_REFIT_EPOCHS = 100
A5_REFIT_LR = 1e-3
A5_REFIT_WD = 1e-4
A5_REFIT_BATCH = 64

# §34.2 0b BSAB_PAIR_CAPTURE：全行缓存的最低 power 参考线（train rows 少于
# 此值时用全部可用行并记 LOW_POWER；N_fit≳2d≈1536 为 full-affine 欠定线）
# 与固定 diagnostic loader batch（B=16 = 正式 refit cache / feature
# extraction 口径；z_op 依赖 batch 组成，pre/post 必须同 batch 切分）
BSAB_CAPTURE_MIN_FIT = 2048
BSAB_CAPTURE_BATCH = 16


def a5_stable_seed(*parts):
    """确定性 31 位本地种子（探针 stable_seed 同构）。refit 的
    manual_seed 全程被四流 RNG 快照包裹，训练流不受扰动。"""
    key = "|".join(str(p) for p in parts)
    return int.from_bytes(_a5_hashlib.sha256(key.encode()).digest()[:4],
                          "little")


def rebuild_client_head(h_state, template):
    """resume 时按 state keys 识别客户端头存储的结构并重建
    （rebuild_global_head 的 heads 版先例，75ead14）：
      * keys == template（官方 Chead）的 keys -> deepcopy(template) 装载；
      * 含 mean/std buffer -> StandardizedChead（维度从 state 自证）；
      * 其他 -> fail-closed 拒绝（截断/外来 keys 不允许静默误装载）。
    """
    keys = set(h_state.keys())
    if keys == set(deepcopy(template).state_dict().keys()):
        head = deepcopy(template)
        head.load_state_dict(h_state)
        return head
    if "mean" in keys and "std" in keys:
        in_features = h_state["mean"].shape[0]
        out_features = h_state["head.3.weight"].shape[0]
        dummy = _A5StandardizedChead(
            out_features,
            torch.zeros(in_features), torch.ones(in_features))
        if keys == set(dummy.state_dict().keys()):
            head = _A5StandardizedChead(
                out_features,
                torch.zeros(in_features), torch.ones(in_features))
            head.load_state_dict(h_state)
            return head
    raise RuntimeError(
        "unrecognized client head checkpoint structure: "
        + str(sorted(keys)))


class Client_DF(object):

    def __init__(self,id,original_model,model_name,task_per_global_epoch,subset,local_epoch,batch_size,lr,device,method,class_mask,args,vit):
        self.id = id
        self.original_model = original_model
        self.vit = vit

        self.task_id = -1
        self.task_per_global_epoch = task_per_global_epoch
        self.test_loader=[]
        # subset应该是一个【】，其中包含了num_task个数据以及类别，以[[(类别)：[数据]]，{}]的形式保存
        self.train_data =subset
        self.local_epoch = local_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.device=device
        self.method =method
        self.nb_classes = args.nb_classes
        self.model = self.init_local_model(model_name, args)
        self.class_mask = class_mask

        self.local_protos = None
        self.global_protos = None
        self.prompts = None

        self.heads = [None,None,None,None,None,None,None,None,None,None]
        self.vit_heads = [None,None,None,None,None,None,None,None,None,None]

        # Phase 0: per-task train/test split indices, recorded so a resumed run
        # can rebuild exactly the same test_loaders (fresh runs still use the
        # original random_split).
        self.data_split_indices = {}

        # Phase-1 margin stats; ONLY written by evaluate_with_margin()
        # (observer-only), never by the official evaluate()
        self.last_margin_stats = None
        self.head = Chead(args.nb_classes)

        # A5-v2（RUNCOMMANDS §23）：joint-head 轨迹 + 开关旗标（args 派
        # 生；默认 OFF = 官方路径逐位一致）
        self._a5_joint_head = None
        self._a5_refit_on = bool(getattr(args, 'head_refit', False))
        # §28 判据臂：refit 时点（默认 local = M2' 现行为；post_broadcast
        # = 广播路径内 refit，特征缓存用广播后接口状态）
        self._a5_refit_post_broadcast = (
            str(getattr(args, 'refit_timing', 'local')) == 'post_broadcast')
        # 广播路径钩子需要 args（get_global_proto_and_head 不透传）
        self.args = args

        # §33 CTRP v1：matched 协议开关（off = 精确 legacy 路径，无
        # nested split、无 calibration）
        self._ctrp_on = (
            str(getattr(args, 'ctrp_mode', 'off') or 'off') != 'off')
        self._ctrp_val_frac = float(getattr(args, 'ctrp_val_frac', 0.20))

        # §34.2 0b：BSAB_PAIR_CAPTURE instrumentation（默认 OFF = 完全
        # no-op，逐位一致）。授权边界 instrumentation-only：不做任何 BSAB
        # 结构实现，不改 head/loss/prompt/optimizer/训练顺序/广播行为。
        # 输出目录由 Server_DF 从 RunManager 接线（args.bsab_capture_dir），
        # client 不自行推导路径（诊断构建 fail-closed，见 Server_DF）。
        self._bsab_capture_on = bool(
            getattr(args, 'bsab_pair_capture', False))
        self._bsab_pending = None

        # §34.4/§34.4a B1 DSAR sidecar：client-local per-broadcast r8
        # readout adapters（默认 OFF = 完全 no-op，逐位一致）。ON 时
        # evaluate/evaluate_with_margin 在 self.model() 之前按冻结
        # dispatch 应用（历史任务 B-only；B identity ≠ DSAR identity，
        # 两个 adapter 各自独立短路）。--dsar_identity = G1 逐位中立性
        # 测试模式（跳过全部拟合，仅提交 exact-identity sidecar）。
        self._dsar_on = bool(getattr(args, 'dsar', False))
        self._dsar_identity = bool(getattr(args, 'dsar_identity', False))
        if self._dsar_identity and not self._dsar_on:
            raise RuntimeError(
                "DSAR: --dsar_identity requires --dsar (fail-closed)")
        if self._dsar_on and bool(getattr(args, 'head_refit', False)):
            raise RuntimeError(
                "DSAR_ON_HEAD_REFIT_UNSUPPORTED: B1 evidence is defined on "
                "the plain fedta baseline; refit-arm readout semantics are "
                "not part of the frozen DSAR deployment (fail-closed)")
        if self._dsar_on and str(
                getattr(args, 'ctrp_mode', 'off') or 'off') != 'off':
            raise RuntimeError(
                "DSAR_ON_CTRP_UNSUPPORTED: CTRP matched split changes the "
                "train-row manifest; B1 is pre-registered on the plain "
                "fedta baseline (fail-closed)")
        # §34.7 数据集条件配方：CIFAR = §34.4 冻结值（γ* 0.396419 / L r8，
        # 逐位不变）；INR = γ*_INR 2.0579 / L r16（结构其余部分与 CIFAR
        # 完全同构：B r8 + mse guard、L 无 guard、zm 基底、B→L dispatch）。
        self._dsar_recipe = dsar_recipe(getattr(args, 'data_name',
                                                'cifar100'))
        self._dsar_sidecar = None
        self._dsar_pending = None

        # A2 routing-interface protection（2026-11-08 恢复，最终方案
        # A2+L+B 的训练侧模块；原始实现 db01cf4/8fb9b87，f37590d 删除，
        # 按开关语义重接入 = 与 --dsar 可组合）。保护集 = 已完成任务
        # t' < task_id 的 train-split 特征经【当前 key】top-1 路由命中槽位
        # 并集（D0 emergent support）；任务开始时快照被保护行，phase-2
        # 每次 optimizer.step() 之后回写（== freeze + wd 豁免，优化器
        # 零改动）。默认 off = 完全 no-op，逐位一致。
        self._a2_arm = str(getattr(args, 'a2_arm', 'off') or 'off')
        self._a2_protected_slots = None   # LongTensor: protected slot ids
        self._a2_key_snapshot = None      # frozen key rows (arm key/both)
        self._a2_anchor_snapshot = None   # frozen anchor rows (arm anchor/both)
        self._a2_task_id = None           # task the current protection belongs to


    def init_local_model(self,model_name,args=None):
        if model_name=='Tail_Anchor':
            return Tail_Anchor(10,768,200, dropout=0.5)
        else:
            return resnet18_cbam(pretrained=False)

    def get_data(self,task_id):
        self.train_dataset = self.train_data[task_id]
        self.current_class = self.class_mask[task_id]
        print(f'{self.id} client，{task_id} task has {len(self.current_class)} classes:{self.current_class}')
        trainset = self.train_dataset
        traindata, testdata = random_split(trainset,
                                           [int(len(trainset) * 0.7), len(trainset) - int(len(trainset) * 0.7)])

        # Phase 0: record split indices (the split itself is unchanged) so
        # resume can rebuild identical train/test subsets.
        self.data_split_indices[task_id] = {
            "train": list(traindata.indices),
            "test": list(testdata.indices),
        }

        self.test_loader.append(testdata)

        self.traindata = traindata
        print(len(traindata))

        # ------------------------------------------------------------------
        # §33 CTRP v1 matched protocol（GPT 冻结 §8）：四个 matched 臂
        # (endpoint / fixed_alpha / ctrp_val / ctrp_train) 逐位共享同一
        # split —— 从 inner 70% train 内做 deterministic class-stratified
        # 80% core / 20% val 嵌套切分（≈ 56% core / 14% val / 30% test；
        # test 与旧任务数据严禁参与 alpha selection）。seed =
        # a5_stable_seed(seed, client, task)（批准的正式训练协议升级，
        # 与 Probe-H 的 51/52 派生 seed 不同，后者仅用于 Test 8 replay）。
        # off 臂不执行本块 = legacy 70/30 路径逐位一致。
        # ------------------------------------------------------------------
        if self._ctrp_on:
            inner_idx = list(traindata.indices)
            labels = self._ctrp_read_labels(trainset, inner_idx)
            rng = np.random.default_rng(
                a5_stable_seed(getattr(self.args, 'seed', 0),
                               self.id, task_id))
            flags = stratified_split(labels, self._ctrp_val_frac, rng)
            core_idx = [inner_idx[i] for i in range(len(inner_idx))
                        if not flags[i]]
            val_idx = [inner_idx[i] for i in range(len(inner_idx))
                       if flags[i]]
            if not core_idx or not val_idx:
                raise RuntimeError(
                    f"CTRP matched split produced an empty side on client "
                    f"{self.id} task {task_id} (core={len(core_idx)}, "
                    f"val={len(val_idx)}); fail-closed")
            # 索引 = trainset 绝对索引（与 "train"/"test" 同一坐标系）
            self.data_split_indices[task_id]["ctrp_core"] = core_idx
            self.data_split_indices[task_id]["ctrp_val"] = val_idx
            self.traindata = Subset(trainset, core_idx)


    def update_data(self,round, args):
        task = round // self.task_per_global_epoch
        flag = True
        if self.task_id != task:
            flag = False
            # only cifar100 / ImageNet-R are supported in this project
            self.get_data(task)
            self.task_id = task
            # A2: entering a new task -> recompute the protected slot set
            # and snapshot the protected rows (no-op when --a2_arm off)
            if self._a2_arm != 'off':
                self._a2_init_task_protection()

    # ------------------------------------------------------------------
    # A2 routing-interface protection（2026-11-08 恢复，见 __init__ 注释）
    # ------------------------------------------------------------------

    def _a2_init_task_protection(self):
        """A2: compute the protected slot set and row snapshots at task start.

        Protection set = union over all completed tasks t' < task_id of the
        top-1 routing destinations of their train-split features under the
        CURRENT key (= D0 emergent support, B_t > 0). The snapshot is taken
        once at task start and the protected rows are restored after every
        Tail_Anchor optimizer step for the whole task (post-step restore ==
        freeze + weight-decay exemption; the official optimizer is never
        modified).

        The feature pass is RNG-neutral: sequential loader (shuffle=False,
        num_workers=0) + four-stream RNG snapshot/restore.
        """
        self._a2_task_id = self.task_id
        completed = [t for t in range(self.task_id)
                     if t in self.data_split_indices]
        if not completed:
            self._a2_protected_slots = None
            self._a2_key_snapshot = None
            self._a2_anchor_snapshot = None
            return

        rng_state = self._a5_rng_snapshot()
        vit_was_training = self.vit.training
        orig_was_training = self.original_model.training
        try:
            self.original_model.eval()
            self.vit.eval()
            self.model.to(self.device)
            self.vit.to(self.device)
            # well-defined prompt state (same convention as train())
            if self.prompts is not None:
                self.vit.load_prompts(self.prompts)

            hit_slots = []
            for t in completed:
                dataset = Subset(self.train_data[t],
                                 self.data_split_indices[t]["train"])
                loader = DataLoader(dataset, batch_size=16, num_workers=0,
                                    shuffle=False)
                feats = []
                for input, _target in loader:
                    input = input.to(self.device, non_blocking=True)
                    with torch.no_grad():
                        output = self.original_model(input)
                        cls_features = output['pre_logits']
                        output = self.vit(input, task_id=self.task_id,
                                         cls_features=cls_features, train=True)
                    feats.append(output['feat'].to(self.device))
                if not feats:
                    continue
                feat = torch.cat(feats, dim=0)
                # routing exactly as Tail_Anchor.forward: l2-normalized
                # feature vs l2-normalized key rows, top-1
                x_embed_norm = self.model.l2_normalize(feat, dim=1)
                key = self.model.key.reshape(-1, self.model.key_size)
                key_norm = self.model.l2_normalize(key, dim=1)
                similarity = torch.matmul(x_embed_norm, key_norm.t())
                _, index = torch.topk(similarity, k=1)
                hit_slots.append(index.reshape(-1))

            if hit_slots:
                slots = torch.unique(torch.cat(hit_slots)).to(self.device)
            else:
                slots = None
            self._a2_protected_slots = slots
            self._a2_key_snapshot = (self.model.key.data[slots].clone()
                                     if (slots is not None and
                                         self._a2_arm in ('key', 'both'))
                                     else None)
            self._a2_anchor_snapshot = (self.model.anchor_pool.data[slots].clone()
                                        if (slots is not None and
                                            self._a2_arm in ('anchor', 'both'))
                                        else None)
            n = 0 if slots is None else int(slots.numel())
            print(f'[A2] client {self.id} task {self.task_id} arm={self._a2_arm}: '
                  f'protected {n}/{self.model.key.shape[0]} slots')
        finally:
            # restore module modes (train() never sets vit.train(); leaving
            # eval() on would silently change training; original_model is
            # shared across clients, so its mode must be restored exactly
            # as found) + ALL RNG streams
            self.vit.train(vit_was_training)
            self.original_model.train(orig_was_training)
            self._a5_rng_restore(rng_state)

    def _a2_post_step_restore(self):
        """A2: restore protected key/anchor rows after a Tail_Anchor
        optimizer step. Only called when --a2_arm != off."""
        if self._a2_protected_slots is None or self._a2_task_id != self.task_id:
            return
        if self._a2_arm in ('key', 'both') and self._a2_key_snapshot is not None:
            self.model.key.data[self._a2_protected_slots] = self._a2_key_snapshot
        if self._a2_arm in ('anchor', 'both') and self._a2_anchor_snapshot is not None:
            self.model.anchor_pool.data[self._a2_protected_slots] = self._a2_anchor_snapshot

    def _phase2_train_step(self, input, target, round, args, criterion,
                           optimizer):
        """OFFICIAL phase-2 per-batch update, extracted verbatim from
        train(): forward (no_grad original_model + vit) -> base loss ->
        zero_grad/backward/step."""
        with torch.no_grad():
            if self.original_model is not None:
                output = self.original_model(input)
                cls_features = output['pre_logits']
                output = self.vit(input, task_id=self.task_id, cls_features=cls_features, train=True)
        pre, output_mixed, pull_off2 = self.model(output['feat'].to(self.device), target.to(self.device))
        logits = pre
        # NCE loss
        if self.global_protos is None:
            loss_InfoNCE = 0
        else:
            count = 0
            loss_InfoNCE = None
            for i, label in enumerate(target):
                if label.item() in self.global_protos.keys():
                    count += 1
                    feature = output_mixed[i].unsqueeze(0)
                    loss_instance = self.calculate_infonce(feature,label.item(),(round+1)%self.task_per_global_epoch==0)
                    if loss_InfoNCE is None:
                        loss_InfoNCE = loss_instance
                    else:
                        loss_InfoNCE += loss_instance
            if count != 0:
                loss_InfoNCE = loss_InfoNCE / count
            else:
                loss_InfoNCE = 0
        loss_InfoNCE = loss_InfoNCE

        mask = self.current_class
        not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
        not_mask = torch.tensor(not_mask, dtype=torch.int64).to(self.device)
        logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

        loss = criterion(logits, target) + 0.2*self.task_per_global_epoch *loss_InfoNCE - 0.1*pull_off2

        if round==16 and self.id==0:
            print(loss_InfoNCE)
            print(criterion(logits, target))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # A2: restore protected key/anchor rows after the Tail_Anchor
        # step. No-op when --a2_arm off (official path bit-identical).
        if self._a2_arm != 'off':
            self._a2_post_step_restore()

    def train(self, round, args):
        self.original_model.eval()

        if self.prompts is not None:
            self.vit.load_prompts(self.prompts)
        else:
            self.vit.init_prompts()

        ###train###
        # update train data

        self.model.to(self.device)
        self.vit.to(self.device)
        # STRICT BASELINE (Phase 0): official FedTA hardcodes the local
        # training batch size to 16 (see official Client_DF.train()).
        # Do NOT replace with --batch-size; that changes the protocol.
        train_loader = DataLoader(self.traindata, batch_size=16, num_workers=args.num_workers, shuffle=True)
        print(f'Client {self.id} on Task {self.task_id} is training prompts')

        # Input-Enhancement
        optimizer = torch.optim.Adam(self.vit.parameters(), lr=self.lr,
                                     weight_decay=1e-03)
        criterion = torch.nn.CrossEntropyLoss().to(self.device)
        cos = nn.CosineEmbeddingLoss()
        for epoch in tqdm(range(self.local_epoch)):
            for iteration, (input,target) in enumerate(train_loader):
                input, target = Variable(input, requires_grad=False).to(self.device, non_blocking=True), target.long().to(self.device,non_blocking=True)

                with torch.no_grad():
                    if self.original_model is not None:
                        output = self.original_model(input)
                        cls_features = output['pre_logits']
                output = self.vit(input, task_id=self.task_id, cls_features=cls_features, train=True)
                # pre, output_mixed, pull_off2 = self.model(output['feat'].to(self.device), target.to(self.device))
                # logits = pre
                logits = output['logits']
                # output_mixed = output['pre_logits']
                pull_off = output['reduce_sim']
                # class_mask
                mask = self.current_class
                not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                not_mask = torch.tensor(not_mask, dtype=torch.int64).to(self.device)
                logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

                loss = criterion(logits,target)  - 0.1 * pull_off
                optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip_grad)
                optimizer.step()

        ###evaluate###

        # self.evaluate_only_prompts(0,args.nb_classes)
        # self.evaluate_only_prompts(self.task_id,args.nb_classes)

        # A5-v2：开关 ON 时 phase-2 从官方 joint-head 轨迹继续——上一轮
        # evaluate 的 load_head 会把（可能的）refit 头装回 model.head，这
        # 里恢复联训轨迹（OFF 臂 _a5_joint_head 恒为 None = 官方路径）
        if self._a5_joint_head is not None:
            self.model.load_head(deepcopy(self._a5_joint_head))

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-03)
        for epoch in tqdm(range(self.local_epoch)):
            for iteration, (input, target) in enumerate(train_loader):
                input, target = Variable(input, requires_grad=False).to(self.device, non_blocking=True), target.long().to(
                    self.device, non_blocking=True)

                # OFFICIAL phase-2 single-batch update (forward -> base
                # loss -> optimizer step)
                self._phase2_train_step(input, target, round, args,
                                        criterion, optimizer)

        target_list = []
        feature_list = []
        for idx,(input,target) in enumerate(train_loader):
            input, target = Variable(input, requires_grad=False).to(self.device, non_blocking=True), target.to(
                self.device, non_blocking=True)
            with torch.no_grad():
                if self.original_model is not None:
                    output = self.original_model(input.to(self.device))
                    output = output['pre_logits'].requires_grad_(False)
                output = self.vit(input, task_id=self.task_id, cls_features=output, train=True)
                _, output_mixed, _ = self.model(output['feat'].to(self.device),target.to(self.device))

            # print(np.isnan(output_mixed.cpu().detach().numpy()).any())
            if not np.isnan(output_mixed.cpu().detach().numpy()).any():
                target_list.append(target)
                feature_list.append(output_mixed)
        local_protos = {}
        if target_list !=[]:
            target_list = torch.cat(target_list, dim=0)
            feature_list = torch.cat(feature_list, dim=0)

            local_protos = {}
            for class_index in self.current_class:
                data_index = (target_list == class_index).nonzero().squeeze(-1)
                if data_index.shape[0] != 0:
                    all_features = feature_list[data_index]
                    proto = all_features.mean(0).cpu().detach().numpy()
                    local_protos[class_index] = proto

        self.local_protos =local_protos

        # save local classifier
        self.heads[self.task_id] = deepcopy(self.model.get_head())

        # A5-v2：readout refit 钩子（默认 OFF = no-op）
        self._a5_post_phase2_interface(round, args)

        # Official FedTA (GitHub main) calls evaluate unconditionally twice
        # here (evaluate(0) + evaluate(task_id)); on task 0 this evaluates
        # task 0 twice. This is intentional baseline behavior: evaluate()'s
        # test loader uses shuffle=True and consumes the global torch RNG,
        # so the call count MUST stay identical to the official code.
        self.evaluate( 0,args.nb_classes)
        self.evaluate(self.task_id, args.nb_classes)

        # save local input enhancement
        self.prompts = deepcopy(self.vit.get_prompts())

    # ------------------------------------------------------------------
    # A5-v2: post-phase-2 readout refit (RUNCOMMANDS §23)
    # ------------------------------------------------------------------

    def _a5_arms_on(self):
        """refit ON（joint-head 轨迹管理生效条件）。
        head_standardize 单独 ON 在 Server_DF 侧已 fail-closed 拒绝。"""
        return bool(self._a5_refit_on)

    def _a5_rng_snapshot(self):
        """四流 RNG 快照（torch cpu/cuda / python / numpy）——refit 对
        训练流完全中性。"""
        return {
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda_all": (torch.cuda.get_rng_state_all()
                               if torch.cuda.is_available() else None),
            "python": random.getstate(),
            "numpy": np.random.get_state(),
        }

    def _a5_rng_restore(self, snap):
        torch.set_rng_state(snap["torch_cpu"])
        if snap["torch_cuda_all"] is not None:
            torch.cuda.set_rng_state_all(snap["torch_cuda_all"])
        random.setstate(snap["python"])
        np.random.set_state(snap["numpy"])

    # ------------------------------------------------------------------
    # §33 CTRP v1: client-side alpha evaluator (RUNCOMMANDS Sec.33,
    # pre-registered 2026-11-03, GPT-frozen verdict Sec.4/5/6)
    #
    # The forward/CE/selector kernel is SHARED with Probe-H
    # (Models/ctrp.py, same-evaluator-kernel + different
    # row-selection-protocol, NOT a second evaluator). Per the frozen
    # verdict Sec.4 the sweep's true baseline is the SERVER CANDIDATE
    # prompt state: each client's full original state is saved, the
    # server candidate prompt module is loaded (full state: prompt
    # values + prompt_key + any other Global_Prompt state), the loaded
    # state hash is asserted == the server's hash BEFORE any sweep step,
    # only the dominant slot row is temporally calibrated per alpha,
    # every other slot keeps the candidate value, and the client's
    # original complete state is restored afterwards.
    # ------------------------------------------------------------------

    def _ctrp_read_labels(self, dataset, indices):
        """Label-only pass over the inner-train rows (Probe-H pass-1
        semantics; no forward). RNG-guarded: loader construction / worker
        spawning must not perturb the training random stream."""
        snap = self._a5_rng_snapshot()
        try:
            labels = []
            loader = DataLoader(Subset(dataset, list(indices)),
                                batch_size=64, shuffle=False,
                                num_workers=getattr(self.args,
                                                    'num_workers', 0))
            for _, tgt in loader:
                labels.extend(int(v) for v in tgt.numpy())
        finally:
            self._a5_rng_restore(snap)
        if len(labels) != len(indices):
            raise RuntimeError(
                f"client {self.id}: label pass returned {len(labels)} "
                f"labels for {len(indices)} rows (fail-closed)")
        return labels

    def _ctrp_with_candidate(self, server_prompt_module):
        """Save the client's complete prompt-module state and load the
        server candidate (full module state, load_prompts deepcopies).
        Returns (orig_prompt_module, server_state_hash); the caller
        restores orig_prompt_module in its finally block and asserts
        hash_prompt_module(vit.prompt) == server_state_hash before any
        sweep step (frozen Sec.4 assertion)."""
        server_hash = hash_prompt_module(server_prompt_module)
        orig_prompt = self.vit.prompt
        self.vit.load_prompts(server_prompt_module)
        loaded_hash = hash_prompt_module(self.vit.prompt)
        if loaded_hash != server_hash:
            raise RuntimeError(
                "CTRP: loaded candidate prompt hash "
                f"{loaded_hash} != server candidate prompt hash "
                f"{server_hash} on client {self.id} (frozen Sec.4 "
                "assertion; fail-closed)")
        return orig_prompt, server_hash

    def _ctrp_route_slots(self, server_prompt_module, rows_key='ctrp_val',
                          rows=None, batch_size=8):
        """Dominant-slot discovery pass: natural forward over the
        current-task validation rows under the SERVER CANDIDATE prompt
        state; collects the FORWARD-ACTUAL Global_Prompt route slot per
        batch (prompt_idx -- deployment route, NOT preference
        statistics). Asserts the route is constant within each batch
        (batchwise top-1 routing, top_k=1). Never hardcoded to a dataset
        slot. Deterministic: fixed row order, shuffle=False, no_grad,
        full state save/restore + RNG guard.

        Gate Amendment (frozen 2026-11-04): returns one record per
        batch -- {slot, n_rows, vote_margin_rows, tie, partial} --
        where slot is the DEPLOYED route (unchanged) and n_rows feeds
        the sample-weighted premise statistic
            M_s = sum_b n_b * 1[p_b = s],  p* = argmax_s M_s,
            r = 1 - M_{p*} / sum_b n_b,      PASS iff r <= 0.01.
        vote_margin_rows / tie / partial are AUDIT-ONLY diagnostics
        (per-row top-1 preference vote margin, recomputed with the
        Global_Prompt routing math; they never enter PASS/FAIL and
        never alter the deployed route)."""
        if rows is None:
            split = self.data_split_indices.get(self.task_id, {})
            if rows_key not in split or not split[rows_key]:
                raise RuntimeError(
                    f"client {self.id}: task {self.task_id} has no "
                    f"recorded {rows_key} rows (matched split missing; "
                    "fail-closed)")
            rows = list(split[rows_key])
        t = self.task_id
        snap = self._a5_rng_snapshot()
        flags = (self.vit.training, self.model.training,
                 self.original_model.training
                 if self.original_model is not None else None)
        orig_prompt = self.vit.prompt
        try:
            self.original_model.eval()
            self.vit.eval()
            self.model.eval()
            self._ctrp_with_candidate(server_prompt_module)
            batches = []
            loader = DataLoader(Subset(self.train_data[t], list(rows)),
                               batch_size=batch_size, shuffle=False,
                               num_workers=getattr(self.args,
                                                   'num_workers', 0))
            for inp, tgt in loader:
                inp = inp.to(self.device, non_blocking=True)
                with torch.no_grad():
                    q = self.original_model(inp)["pre_logits"]
                    out = self.vit(inp, task_id=t, cls_features=q,
                                   train=True)
                idx = out["prompt_idx"]
                if idx is None or idx.dim() != 2 or idx.shape[1] != 1:
                    raise RuntimeError(
                        "CTRP: Global_Prompt returned no batchwise "
                        f"top-1 prompt_idx on client {self.id} (got "
                        f"{None if idx is None else tuple(idx.shape)}); "
                        "route discovery needs the deployment route "
                        "(fail-closed)")
                if not bool(torch.all(idx == idx[0, 0])):
                    raise RuntimeError(
                        "CTRP: prompt_idx is not constant within a batch "
                        f"on client {self.id} (batchwise top-1 top_k=1 "
                        "premise violated; fail-closed)")
                slot = int(idx[0, 0].item())
                n_rows = int(idx.shape[0])
                # audit-only per-row preference votes (Global_Prompt
                # routing math, embedding_key='cls' via cls_features;
                # same l2_normalize + matmul as the deployed forward)
                pm = self.vit.prompt
                with torch.no_grad():
                    key = pm.prompt_key
                    prompt_norm = pm.l2_normalize(key, dim=1)
                    x_norm = pm.l2_normalize(q, dim=1)
                    sim = torch.matmul(x_norm, prompt_norm.t())
                    _, row_pref = torch.topk(sim, k=1, dim=1)
                    votes = torch.bincount(
                        row_pref.reshape(-1).to(torch.int64),
                        minlength=int(key.shape[0]))
                    if votes.numel() >= 2:
                        top2 = torch.topk(votes, k=2).values
                        margin = int(top2[0] - top2[1])
                    else:
                        margin = n_rows
                batches.append({
                    "slot": slot,
                    "n_rows": n_rows,
                    "vote_margin_rows": margin,
                    "tie": bool(margin == 0),
                    "partial": bool(n_rows < batch_size),
                })
            return batches
        finally:
            self.vit.prompt = orig_prompt
            if flags[0] is not None:
                self.vit.train(flags[0])
            self.model.train(flags[1])
            if flags[2] is not None:
                self.original_model.train(flags[2])
            self._a5_rng_restore(snap)

    def _ctrp_alpha_sweep(self, server_prompt_module, anchor_values,
                          dominant_slot, alpha_grid, rows=None,
                          rows_key='ctrp_val', batch_size=8,
                          record_row_ce=False):
        """The alpha-conditioned sweep on ONE client (shared kernel with
        Probe-H, Models/ctrp.py). Per the frozen Sec.4:

          * save the client's complete original state first, then load
            the full SERVER CANDIDATE prompt module state (values +
            prompt_key + all Global_Prompt state); the loaded hash is
            asserted == server candidate hash BEFORE any sweep step;
          * every alpha evaluates the server candidate state with ONLY
            the dominant prompt-value row temporally calibrated,
            P_s*(a) = (1-a) P^deploy_{t-1,s*} + a P^cand_{t,s*} (Probe-H
            op order, bitwise the validated kernel); every other slot
            keeps the candidate value P^cand_{t,j};
          * alpha == 1 IS the natural forward of the loaded candidate
            (Probe-H semantics); alpha != 1 arms go through the shim
            intervention kernel, which bypasses routing -- CTRP never
            touches prompt_key, so route(a=0) == route(a=1) by
            construction and the key-hash invariance is asserted below;
          * head = the client's CURRENT-TASK head snapshot
            (client.heads[task]), eval mode (Chead dropout OFF);
            CE = logsumexp(lg,1) - lg.gather(1,y_local) on class-masked
            fp32 logits, y_local = searchsorted(mask_pos, y) --
            all Probe-H evaluator semantics;
          * rows in batches routed to a non-dominant slot (<= 1% premise
            tail) keep the candidate value and are NOT scored
            (Probe-H row protocol); the caller aggregates with
            sample-weighted pooled CE.

        All temporary state (prompt module, train/eval flags, RNG) is
        restored in finally. record_row_ce=True keeps per-row CE + row
        alignment meta (Test-8 replay helper layer only)."""
        t = self.task_id
        mask = sorted(int(x) for x in self.class_mask[t])
        mpos = torch.tensor(mask, dtype=torch.long, device=self.device)
        if rows is None:
            split = self.data_split_indices.get(t, {})
            if rows_key not in split or not split[rows_key]:
                raise RuntimeError(
                    f"client {self.id}: task {t} has no recorded "
                    f"{rows_key} rows (matched split missing; "
                    "fail-closed)")
            rows = list(split[rows_key])
        if 1.0 not in [float(a) for a in alpha_grid]:
            raise RuntimeError(
                "CTRP: alpha grid must contain the 1.0 natural-arm "
                "position (Probe-H semantics; fail-closed)")
        pos_one = [float(a) for a in alpha_grid].index(1.0)

        snap = self._a5_rng_snapshot()
        flags = (self.vit.training, self.model.training,
                 self.original_model.training
                 if self.original_model is not None else None)
        orig_prompt = self.vit.prompt
        orig_head = self.model.head
        try:
            self.original_model.eval()
            self.vit.eval()
            # Probe-H mode chain: eval() must come AFTER load_head.
            # load_head deepcopies the STORED head, which carries the
            # training-time training=True flags (heads are snapshotted
            # mid-train via deepcopy); eval()-then-load would silently
            # run Chead Dropout(p=0.5) in train mode and make the sweep
            # stochastic (caught by the Test-8 replay gate: row-CE
            # max|diff| 4.46 with dropout vs 0.0 when aligned).
            self.model.load_head(self.heads[t])
            self.model.eval()   # Chead Dropout OFF (Probe-H, pre-registered)
            self._ctrp_with_candidate(server_prompt_module)
            # key hash of the LOADED candidate; the whole sweep must not
            # move it (route(alpha=0) == route(alpha=1) by construction)
            key_hash_before = hash_tensor(self.vit.prompt.prompt_key)

            cand_values = self.vit.prompt.prompt.detach()
            cand_row = cand_values[dominant_slot].detach()
            prev_row = anchor_values.to(cand_row.device)[dominant_slot]

            ce_sum = np.zeros(len(alpha_grid), dtype=np.float64)
            n_rows = 0
            n_dropped = 0
            n_batches = 0
            batch_slots = []
            row_ce_rows = [] if record_row_ce else None
            row_meta = [] if record_row_ce else None

            loader = DataLoader(Subset(self.train_data[t], list(rows)),
                               batch_size=batch_size, shuffle=False,
                               num_workers=getattr(self.args,
                                                   'num_workers', 0))
            for inp, tgt in loader:
                inp = inp.to(self.device, non_blocking=True)
                tgt = tgt.long()
                with torch.no_grad():
                    q = self.original_model(inp)["pre_logits"]
                    out = self.vit(inp, task_id=t, cls_features=q,
                                   train=True)
                idx = out["prompt_idx"]
                if idx is None or idx.dim() != 2 or idx.shape[1] != 1:
                    raise RuntimeError(
                        "CTRP: Global_Prompt returned no batchwise top-1 "
                        f"prompt_idx on client {self.id} (fail-closed)")
                if not bool(torch.all(idx == idx[0, 0])):
                    raise RuntimeError(
                        "CTRP: prompt_idx not constant within a batch "
                        f"on client {self.id} (fail-closed)")
                slot = int(idx[0, 0].item())
                n_batches += 1
                batch_slots.append(slot)
                bsz = int(tgt.shape[0])
                if slot != dominant_slot:
                    # <= 1% premise tail: keeps the candidate value,
                    # not calibrated, not scored (Probe-H row protocol)
                    n_dropped += bsz
                    continue
                y_local = torch.searchsorted(mpos, tgt.to(self.device))
                if int(y_local.max().item()) >= len(mask) or \
                        bool((y_local < 0).any()):
                    raise RuntimeError(
                        "CTRP: label outside the current-task class mask "
                        f"on client {self.id} (fail-closed)")
                with torch.no_grad():
                    lg_nat = (self.model(out["feat"].float(), None)[0]
                              .index_select(1, mpos).float())
                ce_rows = np.zeros((bsz, len(alpha_grid)),
                                   dtype=np.float32)
                ce_rows[:, pos_one] = row_ce(lg_nat, y_local) \
                    .detach().cpu().numpy()
                for a_pos, a in enumerate(alpha_grid):
                    if a_pos == pos_one:
                        continue
                    tok = alpha_blend(prev_row, cand_row, float(a))
                    with torch.no_grad():
                        z = forward_with_prompt(
                            self.vit, t, inp, q, tok)
                        lg = (self.model(z, None)[0]
                              .index_select(1, mpos).float())
                        ce_rows[:, a_pos] = row_ce(lg, y_local) \
                            .detach().cpu().numpy()
                ce_sum += ce_rows.sum(axis=0, dtype=np.float64)
                n_rows += bsz
                if record_row_ce:
                    row_ce_rows.append(ce_rows)
                    row_meta.extend(
                        {"slot": slot,
                         "y_local": int(v)}
                        for v in y_local.detach().cpu().numpy())

            if hash_tensor(self.vit.prompt.prompt_key) != key_hash_before:
                raise RuntimeError(
                    "CTRP: prompt_key moved during the alpha sweep "
                    "(route(alpha=0)==route(alpha=1) violated; "
                    "fail-closed)")
            return {
                "ce_sum_by_alpha": ce_sum,
                "n_rows": int(n_rows),
                "n_dropped": int(n_dropped),
                "n_batches": int(n_batches),
                "batch_slots": batch_slots,
                "row_ce": (np.concatenate(row_ce_rows, axis=0)
                           if record_row_ce else None),
                "row_meta": row_meta,
            }
        finally:
            self.vit.prompt = orig_prompt
            self.model.head = orig_head
            if flags[0] is not None:
                self.vit.train(flags[0])
            self.model.train(flags[1])
            if flags[2] is not None:
                self.original_model.train(flags[2])
            self._a5_rng_restore(snap)

    def _a5_collect_head_inputs(self, task_id, feat_only=False):
        """refit 特征缓存的只读收集：task 的 train split、当前接口状态
        （vit 用当前任务 prompt = native evaluate 的 task_id=self.task_id
        语义），no_grad + 顺序 loader，test split 永不触碰。与官方 proto
        收集同数据流（train split 带增强）。
          feat_only=False（默认，x1 臂）：再过 model 前向取 head 真实
        输入 x1（1536 维 anchor-concat）；
          feat_only=True（--refit_input feat，§23.6-1 修复臂）：直接取
        vit 输出 z_op（768 维 pre-anchor），不过 model 前向——无 key
        路由、无 anchor 混入（INR 双探针诊断 2026-10-29 判读的病灶）。"""
        if task_id not in self.data_split_indices:
            raise RuntimeError(
                "client %d: task %d has no recorded train split (seen "
                "tasks must have data_split_indices) -- refusing to "
                "refit an unseen task (fail-closed)"
                % (self.id, task_id))
        dataset = Subset(self.train_data[task_id], self._ctrp_train_indices(task_id))
        loader = DataLoader(dataset, batch_size=16, num_workers=0,
                            shuffle=False)
        xs, ys = [], []
        for input, target in loader:
            input = input.to(self.device, non_blocking=True)
            target = target.long().to(self.device, non_blocking=True)
            with torch.no_grad():
                cls_features = None
                if self.original_model is not None:
                    output = self.original_model(input)
                    cls_features = output['pre_logits']
                output = self.vit(input, task_id=self.task_id,
                                  cls_features=cls_features, train=True)
                if feat_only:
                    x1 = output['feat'].to(self.device)
                else:
                    _, x1, _ = self.model(output['feat'].to(self.device),
                                          target)
            xs.append(x1)
            ys.append(target)
        if not xs:
            return None, None
        return torch.cat(xs, dim=0), torch.cat(ys, dim=0)

    def _a5_refit_one_head(self, task_id, round_id, args):
        """readout-only refit：在缓存特征上重训一个 Chead 形状读出头
        （masked CE 同 phase-2 口径，mask 用该任务 class_mask 对
        nb_classes 取补；标准化按 --head_standardize；输入路径按
        --refit_input）。返回新头模块；train split 为空（协议现实）时
        返回 None = 保留现有头。梯度只到读出头——特征已缓存、no_grad
        计算，key/anchor/prompt 不在图中。

        §23.6-1 feat 臂（--refit_input feat）：收集 768 维 z_op，训练
        输入 = z_op ⊕ 0（垫零成 1536 形状，锚半区恒 0 → 其一阶权重梯度
        恒 0）；训练后把 first Linear 的锚半区权重硬置零——部署语义：
        head(x1) 功能上 = z_op 头（x1[:, :768] 即 z_op），evaluate/
        checkpoint/resume 路径与 x1 臂完全同构（state keys 不变）。"""
        refit_feat = (str(getattr(args, 'refit_input', 'x1')) == 'feat')
        x1, y = self._a5_collect_head_inputs(task_id, feat_only=refit_feat)
        if x1 is None:
            return None
        label_num = self.model.nb_class
        classes = self.class_mask[task_id]
        not_mask = np.setdiff1d(np.arange(self.nb_classes), classes)
        not_mask = torch.tensor(not_mask, dtype=torch.int64).to(self.device)
        torch.manual_seed(a5_stable_seed(
            args.seed, 'a5refit', self.id, task_id, round_id))
        half = x1.shape[1] if refit_feat else None
        if bool(getattr(args, 'head_standardize', False)):
            if refit_feat:
                # 标准化只定义在 z_op 半区；锚半区统计量置恒等
                # （mean=0/std=1），部署时其贡献被置零权重消掉
                mean = torch.cat([x1.mean(dim=0),
                                  torch.zeros(half, device=x1.device,
                                               dtype=x1.dtype)])
                std = torch.cat([x1.std(dim=0) + 1e-8,
                                 torch.ones(half, device=x1.device,
                                            dtype=x1.dtype)])
            else:
                mean = x1.mean(dim=0)
                std = x1.std(dim=0) + 1e-8
            head = _A5StandardizedChead(label_num, mean, std,
                                       hidden=A5_REFIT_HIDDEN,
                                       dropout=A5_REFIT_DROPOUT)
        else:
            head = Chead(label_num, dropout=A5_REFIT_DROPOUT)
        if refit_feat:
            x1 = torch.nn.functional.pad(x1, (0, half))
        head.to(self.device)
        head.train()
        opt = torch.optim.Adam(head.parameters(), lr=A5_REFIT_LR,
                               weight_decay=A5_REFIT_WD)
        criterion = torch.nn.CrossEntropyLoss().to(self.device)
        n = x1.shape[0]
        bs = min(A5_REFIT_BATCH, n)
        for _ in range(A5_REFIT_EPOCHS):
            perm = torch.randperm(n, device=x1.device)
            for s in range(0, n, bs):
                idx = perm[s:s + bs]
                logits = head(x1[idx])
                logits = logits.index_fill(dim=1, index=not_mask,
                                           value=float('-inf'))
                loss = criterion(logits, y[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
        if refit_feat:
            with torch.no_grad():
                head.head[0].weight[:, half:] = 0.0
        print('[A5-v2] client %d task %d round %d: refit head on %d train '
              'rows (standardize=%s, input=%s, timing=%s)'
              % (self.id, task_id, round_id, n,
                 bool(getattr(args, 'head_standardize', False)),
                 'feat' if refit_feat else 'x1',
                 'post_broadcast' if self._a5_refit_post_broadcast
                 else 'local'))
        return head

    def _a5_post_phase2_interface(self, round, args):
        """train() 末尾的 A5-v2 钩子（官方两 evaluate 之前、heads 保存
        之后；默认 OFF = 完全 no-op）：
          1. joint-head 轨迹快照（任何存储替换之前，_a5_joint_head）；
          2. --head_refit（且 --refit_timing local，§28 默认）：当前任务
             heads[task_id] 换 refit 头（每轮）。post_broadcast 时点下
             本钩子只做快照——refit 移到广播路径
             （_a5_post_broadcast_interface），特征缓存用广播后接口状态。
        全程四流 RNG 快照恢复 + 模块模式恢复——表征侧轨迹（prompt/key/
        anchor/joint head）与官方 run 逐位一致，corrected 指标变化 100%
        归因于接口替换（单测覆盖，RUNCOMMANDS §23.1 RNG 中立）。"""
        if not self._a5_arms_on():
            return
        self._a5_joint_head = deepcopy(self.model.get_head())
        rng = self._a5_rng_snapshot()
        vit_was_training = self.vit.training
        orig_was_training = (self.original_model.training
                             if self.original_model is not None else None)
        try:
            if self.original_model is not None:
                self.original_model.eval()
            if (bool(getattr(args, 'head_refit', False))
                    and not self._a5_refit_post_broadcast):
                head = self._a5_refit_one_head(self.task_id, round, args)
                if head is not None:
                    self.heads[self.task_id] = head
        finally:
            if orig_was_training is not None:
                self.original_model.train(orig_was_training)
            self.vit.train(vit_was_training)
            self._a5_rng_restore(rng)
        # model.head 保持在 joint 轨迹上（后续 evaluate 的 load_head 会按
        # 官方顺序装回 heads[task]；phase-2 前另有恢复，双保险）
        self.model.load_head(deepcopy(self._a5_joint_head))

    def _a5_post_broadcast_interface(self, round, args):
        """§28 判据臂（--refit_timing post_broadcast）的广播路径钩子：
        get_global_proto_and_head 内、vit.load_prompts(广播 prompt) 之后、
        两 evaluate 之前调用。特征缓存在广播后接口状态（融合 prompt +
        本地 phase-2 后 Tail_Anchor 状态）上采集 = 评估特征流同源，闭合
        §27.7-1 归因的训练-评估特征时点错位。refit 配方与 local 时点完全
        一致（_a5_refit_one_head）；四流 RNG 快照恢复保证插入点前后的
        RNG 消费流与官方逐位一致（表征侧轨迹不变式，单测覆盖）。默认
        （local 时点）= 完全 no-op。"""
        if not (self._a5_refit_on and self._a5_refit_post_broadcast):
            return
        rng = self._a5_rng_snapshot()
        vit_was_training = self.vit.training
        orig_was_training = (self.original_model.training
                             if self.original_model is not None else None)
        try:
            if self.original_model is not None:
                self.original_model.eval()
            head = self._a5_refit_one_head(self.task_id, round, args)
            if head is not None:
                self.heads[self.task_id] = head
        finally:
            if orig_was_training is not None:
                self.original_model.train(orig_was_training)
            self.vit.train(vit_was_training)
            self._a5_rng_restore(rng)

    # ------------------------------------------------------------------
    # §34.2 0b: BSAB_PAIR_CAPTURE -- instrumentation-only pre/post-
    # broadcast pair cache（user 2026-11-04 APPROVED FOR IMPLEMENTATION；
    # capture run = M2'-matched，见 Server_DF fail-closed 与 RUNCOMMANDS
    # §34.2）。授权范围 = capture ONLY：不加 BSAB module、不改 head/loss/
    # prompt/optimizer/训练顺序/dataloader split/广播行为，不接入任何
    # CTRP/endpoint 正式臂。六条 hard gate 落点：
    #   (1) 严格同一批 sample 配对：pre/post 两半各自独立重建同一 row
    #       manifest，row_index_hash 不一致即 fail-closed（不依赖
    #       dataloader 顺序，直接保存 sample_index）；
    #   (2) birth head 在 load_prompts 前锚定（heads[task_id] = local
    #       timing 下的 M2' refit birth head）：保存完整 state + head_hash，
    #       post 半段重验 hash 不变；
    #   (3) capture 不影响正式轨迹：四流 RNG snapshot/restore + 全模块
    #       mode 恢复 + 独立 deterministic loader，OFF/ON 逐位一致由
    #       tests/test_bsab_pair_capture.py + 正式 rerun 前的 OFF/ON
    #       regression 双层把关；
    #   (4) bridge fit 严格 label-free：label 字段只供离线 accuracy
    #       evaluation / M2'' positive control（离线探针侧契约）；
    #   (5)(6) 为离线 B0 侧纪律（reproduction gate / confirmatory），
    #       不在本 hook 内实现。
    # ------------------------------------------------------------------

    def _bsab_capture_dir(self):
        """capture 输出目录（Server_DF 从 RunManager 接线；缺失即
        fail-closed，绝不静默改写别处）。"""
        path = str(getattr(self.args, 'bsab_capture_dir', '') or '')
        if not path:
            raise RuntimeError(
                "client %d: --bsab_pair_capture is ON but no capture "
                "directory was wired (Server_DF sets args.bsab_capture_dir "
                "from the run's RunManager; diagnostic builds are "
                "fail-closed)" % self.id)
        return Path(path)

    def _bsab_capture_rows(self):
        """gate-1 row manifest：当前任务的全部 train rows（refit cache 同
        口径 = _ctrp_train_indices；capture run 为 CTRP-off 的 M2'-matched
        rerun，即完整 inner-train）+ 全部 test rows，记录顺序固定为
        train-block 在前、test-block 在后。两半独立重建并以
        row_index_hash 相互校验。"""
        split = self.data_split_indices.get(self.task_id, {})
        train_idx = list(self._ctrp_train_indices(self.task_id))
        test_idx = list(split.get("test", []))
        rows = ([("train", i) for i in train_idx]
                + [("test", i) for i in test_idx])
        if not rows:
            raise RuntimeError(
                "client %d task %d: BSAB capture found no train/test rows "
                "(fail-closed)" % (self.id, self.task_id))
        h = _a5_hashlib.sha256()
        h.update("|".join("%s:%d" % r for r in rows).encode())
        return rows, h.hexdigest()[:16]

    def _bsab_forward_rows(self, rows):
        """§34.2.3 deterministic diagnostic forward：eval + no_grad、固定
        行序、shuffle=False、num_workers=0 的独立顺序 loader（绝不复用
        正式训练 loader 的 shuffle/iterator 状态）；调用方包四流 RNG
        snapshot/restore。特征流语义 = refit cache / extract_features：
        z_ref = frozen original model 的 feat；z_op = 当前任务 prompt 增强
        的 pre-anchor feat（task_id=self.task_id，train=True flag 在
        eval+no_grad 下与正式缓存采集一致），batch=BSAB_CAPTURE_BATCH。
        返回 (z_ref, z_op, labels, split_codes, sample_idx)，float32。"""
        if self.original_model is None:
            raise RuntimeError(
                "client %d: BSAB capture requires the frozen original "
                "model (z_ref stream; fail-closed)" % self.id)
        by_split = {"train": [], "test": []}
        for name, idx in rows:
            by_split[name].append(idx)
        dataset = self.train_data[self.task_id]
        z_refs, z_ops, labels, split_codes, sample_idx = [], [], [], [], []
        for split_name, code in (("train", 0), ("test", 1)):
            if not by_split[split_name]:
                continue
            loader = DataLoader(
                Subset(dataset, list(by_split[split_name])),
                batch_size=BSAB_CAPTURE_BATCH, shuffle=False, num_workers=0)
            for input, target in loader:
                input = input.to(self.device, non_blocking=True)
                with torch.no_grad():
                    out_ref = self.original_model(input)
                    out_op = self.vit(input, task_id=self.task_id,
                                      cls_features=out_ref["pre_logits"],
                                      train=True)
                z_refs.append(out_ref["feat"].float().cpu().numpy())
                z_ops.append(out_op["feat"].float().cpu().numpy())
                labels.append(np.asarray(target, dtype=np.int64))
                split_codes.append(np.full(len(target), code, np.int64))
            sample_idx.extend(by_split[split_name])
        z_ref = np.concatenate(z_refs).astype(np.float32)
        z_op = np.concatenate(z_ops).astype(np.float32)
        return (z_ref, z_op, np.concatenate(labels),
                np.concatenate(split_codes),
                np.asarray(sample_idx, dtype=np.int64))

    def _bsab_capture_pre(self, round_id, entry):
        """pre（z⁻）半段：get_global_proto_and_head* 入口处、
        `self.prompts = prompt` / load_prompts 之前调用。z⁻ 必须在
        client 自己的本地训练后 prompt 状态下前向 —— 该 birth state
        在 train() 末尾保存进 self.prompts；而共享 vit 此刻持的是
        别的 client 的残留状态（服务器先全体训练、再依次广播），
        直接用 vit 现状会把 c1..c4 的 z⁻ 采成已广播态（两臂 regression
        的 z_op_static=8/10 即此泄漏，bugfix 2026-11-04）。因此本函数
        临时装载 self.prompts → z⁻ 前向 → 恢复 vit 原状（观察者中立：
        装载/恢复均为 deepcopy，不消耗 RNG；且紧随其后的
        load_prompts(广播 prompt) 无论如何都会覆盖）。
        gate-2：此时锚定 birth head（heads[task_id] 的完整 state +
        head_hash）。全程四流 RNG snapshot/restore + 模块 mode 恢复。"""
        if not self._bsab_capture_on:
            return
        self._bsab_pending = None
        if (self.task_id < 0
                or self.task_id not in self.data_split_indices):
            return  # 该 client 尚无任何 birth state（从未见过任务数据）
        if self.original_model is None:
            raise RuntimeError(
                "client %d: BSAB capture requires the frozen original "
                "model (z_ref stream; fail-closed)" % self.id)
        rng = self._a5_rng_snapshot()
        flags = (self.vit.training, self.model.training,
                 self.original_model.training)
        birth_prompts = self.prompts
        if birth_prompts is None:
            raise RuntimeError(
                "client %d: BSAB capture found no local birth prompt "
                "state in self.prompts at the broadcast boundary "
                "(fail-closed)" % self.id)
        vit_prompt_snapshot = deepcopy(self.vit.get_prompts())
        try:
            self.original_model.eval()
            self.vit.eval()
            self.model.eval()
            rows, row_index_hash = self._bsab_capture_rows()
            head = self.heads[self.task_id]
            if head is None:
                raise RuntimeError(
                    "client %d task %d: no birth head in heads[task] at "
                    "the broadcast boundary (fail-closed)"
                    % (self.id, self.task_id))
            head_state = {k: v.detach().cpu().contiguous().clone()
                          for k, v in head.state_dict().items()}
            # z⁻ 在 client 自己的 birth prompt 状态下前向（见 docstring：
            # 共享 vit 持的是其他 client 的残留态/上一轮广播态）
            self.vit.load_prompts(deepcopy(birth_prompts))
            z_ref, z_op, labels, split_codes, sample_idx = \
                self._bsab_forward_rows(rows)
            self._bsab_pending = {
                "round_id": int(round_id), "entry": entry,
                "row_index_hash": row_index_hash,
                "head_hash": hash_prompt_module(head),
                "head_state": head_state,
                "pre_prompt_hash": hash_prompt_module(birth_prompts),
                "z_ref_pre": z_ref, "z_op_pre": z_op,
                "labels": labels, "split_codes": split_codes,
                "sample_idx": sample_idx,
                "n_train": sum(1 for r in rows if r[0] == "train"),
                "n_test": sum(1 for r in rows if r[0] == "test"),
                "feature_shape": list(z_op.shape[1:]),
            }
        finally:
            self.vit.load_prompts(vit_prompt_snapshot)
            self.original_model.train(flags[2])
            self.model.train(flags[1])
            self.vit.train(flags[0])
            self._a5_rng_restore(rng)

    def _bsab_capture_post(self, round_id, entry):
        """post（z⁺）半段：vit.load_prompts(广播 prompt) 之后、
        _a5_post_broadcast_interface / 任何 evaluate 之前调用。两半配对
        校验（gate-1 row manifest + gate-2 birth head hash 均重验）后原子
        写出 npz + json 到独立 capture 目录。已存在同名文件 = fail-closed
        （绝不静默覆盖旧 capture）。"""
        if not self._bsab_capture_on:
            return
        pend = self._bsab_pending
        self._bsab_pending = None
        if pend is None:
            return
        if int(round_id) != pend["round_id"] or entry != pend["entry"]:
            raise RuntimeError(
                "client %d: BSAB capture halves mismatched (pre r%d %s vs "
                "post r%d %s; fail-closed)"
                % (self.id, pend["round_id"], pend["entry"],
                   int(round_id), entry))
        rng = self._a5_rng_snapshot()
        flags = (self.vit.training, self.model.training,
                 self.original_model.training)
        try:
            self.original_model.eval()
            self.vit.eval()
            self.model.eval()
            rows, row_index_hash = self._bsab_capture_rows()
            if row_index_hash != pend["row_index_hash"]:
                raise RuntimeError(
                    "client %d round %d: GATE-1 VIOLATION: the row "
                    "manifest changed between the pre and post capture "
                    "halves (%s != %s; strict same-sample pairing is "
                    "broken; fail-closed)"
                    % (self.id, round_id, pend["row_index_hash"],
                       row_index_hash))
            head = self.heads[self.task_id]
            head_hash = (hash_prompt_module(head)
                         if head is not None else None)
            if head_hash != pend["head_hash"]:
                raise RuntimeError(
                    "client %d round %d: GATE-2 VIOLATION: the birth head "
                    "changed between the pre and post capture halves "
                    "(%s != %s; fail-closed)"
                    % (self.id, round_id, pend["head_hash"], head_hash))
            post_prompt_hash = hash_prompt_module(self.vit.prompt)
            z_ref_post, z_op_post, labels, split_codes, sample_idx = \
                self._bsab_forward_rows(rows)
            if (not np.array_equal(labels, pend["labels"])
                    or not np.array_equal(split_codes,
                                          pend["split_codes"])
                    or not np.array_equal(sample_idx,
                                          pend["sample_idx"])):
                raise RuntimeError(
                    "client %d round %d: GATE-1 VIOLATION: paired row "
                    "metadata diverged between capture halves "
                    "(fail-closed)" % (self.id, round_id))
            out_dir = self._bsab_capture_dir()
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = "bsab_c%d_t%d_r%04d_%s" % (self.id, self.task_id,
                                              int(round_id), entry)
            npz_path = out_dir / (stem + ".npz")
            json_path = out_dir / (stem + ".json")
            if npz_path.exists() or json_path.exists():
                raise RuntimeError(
                    "client %d round %d: BSAB capture output already "
                    "exists (%s); refusing to overwrite an earlier "
                    "capture (fail-closed)"
                    % (self.id, round_id, npz_path))
            n = len(pend["labels"])
            arrays = dict(
                client_ids=np.full(n, self.id, dtype=np.int64),
                task_ids=np.full(n, self.task_id, dtype=np.int64),
                round_ids=np.full(n, int(round_id), dtype=np.int64),
                sample_index=pend["sample_idx"],
                split=pend["split_codes"],
                label=pend["labels"],
                z_ref_pre=pend["z_ref_pre"],
                z_op_pre=pend["z_op_pre"],
                z_ref_post=z_ref_post,
                z_op_post=z_op_post,
            )
            for k, v in pend["head_state"].items():
                arrays["head__" + k] = v.numpy()
            np.savez(npz_path, **arrays)
            meta = {
                "client_id": self.id,
                "task_id": self.task_id,
                "round_id": int(round_id),
                "entry": entry,
                "pre_prompt_hash": pend["pre_prompt_hash"],
                "post_prompt_hash": post_prompt_hash,
                "head_hash": pend["head_hash"],
                "row_index_hash": row_index_hash,
                "n_rows": int(n),
                "n_train": int(pend["n_train"]),
                "n_test": int(pend["n_test"]),
                "n_fit": int(pend["n_train"]),
                "low_power": bool(pend["n_train"] < BSAB_CAPTURE_MIN_FIT),
                "feature_shape": pend["feature_shape"],
                "seed": int(getattr(self.args, "seed", -1)),
                "head_refit": bool(getattr(self.args, "head_refit",
                                           False)),
                "refit_input": str(getattr(self.args, "refit_input",
                                           "x1")),
                "refit_timing": str(getattr(self.args, "refit_timing",
                                            "local")),
                "gates": {"row_pairing_verified": True,
                          "birth_head_hash_verified": True},
                "note": ("instrumentation-only BSAB_PAIR_CAPTURE "
                         "(RUNCOMMANDS Sec.34.2); M2'-matched; BSAB "
                         "architecture implementation FROZEN; label is "
                         "cached for offline accuracy evaluation only "
                         "and must NEVER enter bridge fitting/selection"),
            }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, sort_keys=True)
        finally:
            self.original_model.train(flags[2])
            self.model.train(flags[1])
            self.vit.train(flags[0])
            self._a5_rng_restore(rng)

    # ------------------------------------------------------------------
    # §34.4/§34.4a B1 DSAR：per-broadcast client-local readout sidecar
    # 拟合与提交。默认 OFF = 完全 no-op（不进本节任何分支）。
    # --dsar_identity 测试模式跳过全部拟合、仅提交 exact-identity
    # sidecar（G1 OFF/ON(identity) 逐位中立性验证用，非方法臂）。
    # 拟合时序（§34.4 item 3 冻结）：broadcast 入口（pre 半段，z⁻ 时代）
    # 拟合 L -> load_prompts(广播) -> post 半段采 z⁺ 拟合 B -> 提交
    # sidecar -> evaluate 读 B/L。每轮 fresh-fit from identity，不
    # warm-start、不跨轮 compose。全程四流 RNG snapshot/restore + 模块
    # mode/head/prompt 状态恢复；拟合数值走 Models/DSAR.py 独立
    # Generator 流。
    # ------------------------------------------------------------------

    def _dsar_train_rows(self):
        """B1 fit row manifest：当前任务 train split 全行（=
        _ctrp_train_indices，与 BSAB capture / refit cache 同口径），
        固定行序。pre/post 两个半段各自重建并以 hash 互验。"""
        train_idx = list(self._ctrp_train_indices(self.task_id))
        if not train_idx:
            raise RuntimeError(
                "client %d task %d: DSAR fit found no train rows "
                "(fail-closed)" % (self.id, self.task_id))
        h = _a5_hashlib.sha256()
        h.update("|".join(str(int(i)) for i in train_idx).encode())
        return train_idx, h.hexdigest()[:16]

    def _dsar_forward_train_rows(self, train_idx):
        """确定性诊断前向（_bsab_forward_rows 同纪律：eval + no_grad、
        固定行序、shuffle=False、num_workers=0、bs=BSAB_CAPTURE_BATCH
        ——z_op 依赖 batch 组成，pre/post 两半必须同 batch 切分）。
        返回 (z_op, y)，torch 张量留在 self.device 供 L 拟合直接使用。"""
        if self.original_model is None:
            raise RuntimeError(
                "client %d: DSAR requires the frozen original model "
                "(fail-closed)" % self.id)
        loader = DataLoader(Subset(self.train_data[self.task_id],
                                   list(train_idx)),
                            batch_size=BSAB_CAPTURE_BATCH, shuffle=False,
                            num_workers=0)
        z_ops, ys = [], []
        for input, target in loader:
            input = input.to(self.device, non_blocking=True)
            with torch.no_grad():
                out_ref = self.original_model(input)
                out_op = self.vit(input, task_id=self.task_id,
                                  cls_features=out_ref["pre_logits"],
                                  train=True)
            z_ops.append(out_op["feat"].float())
            ys.append(target.long().to(self.device))
        return torch.cat(z_ops, dim=0), torch.cat(ys, dim=0)

    def _dsar_fit_pre(self, round_id, entry):
        """pre 半段（get_global_proto_and_head* 入口、load_prompts 之前）：
        z⁻ 采集（client 自己的 birth prompt 状态，与 _bsab_capture_pre
        同一防残留态泄漏纪律）+ L_{c,r} 拟合（冻结 H_end = heads[task]，
        eval 语义 + 官方 masked-CE 口径）。
        identity 测试模式 / 无 birth state：仅提交 exact-identity
        sidecar（保证"最新已提交 sidecar 匹配当前广播状态"的 G4
        不变量），不拟合。"""
        if not self._dsar_on:
            return
        self._dsar_pending = None
        if self._dsar_identity:
            self._dsar_sidecar = dsar_identity_sidecar(
                round_id, entry, self.task_id, status="IDENTITY_TEST")
            return
        if (self.task_id < 0
                or self.task_id not in self.data_split_indices):
            self._dsar_sidecar = dsar_identity_sidecar(
                round_id, entry, self.task_id, status="NO_BIRTH_STATE")
            return
        rng = self._a5_rng_snapshot()
        flags = (self.vit.training, self.model.training,
                 self.original_model.training)
        head_snapshot = deepcopy(self.model.get_head())
        vit_prompt_snapshot = deepcopy(self.vit.get_prompts())
        try:
            self.original_model.eval()
            self.vit.eval()
            self.model.eval()
            birth_prompts = self.prompts
            if birth_prompts is None:
                raise RuntimeError(
                    "client %d: DSAR found no local birth prompt state at "
                    "the broadcast boundary (fail-closed)" % self.id)
            birth_hash = hash_prompt_module(birth_prompts)
            # z⁻ 在 client 自己的 birth prompt 状态下前向（共享 vit 此刻
            # 持别的 client 残留态；_bsab_capture_pre 同一 bugfix 纪律）
            self.vit.load_prompts(deepcopy(birth_prompts))
            train_idx, row_hash = self._dsar_train_rows()
            z_minus, y = self._dsar_forward_train_rows(train_idx)
            # L 拟合：冻结 H_end 读出（heads[task]，load_head 内部
            # deepcopy）+ 官方 class-mask CE（phase-2 同口径）
            if self.heads[self.task_id] is None:
                raise RuntimeError(
                    "client %d task %d: no H_end head at the broadcast "
                    "boundary (fail-closed)" % (self.id, self.task_id))
            self.model.load_head(self.heads[self.task_id])
            not_mask = np.setdiff1d(np.arange(self.nb_classes),
                                    self.class_mask[self.task_id])
            not_mask = torch.tensor(not_mask,
                                    dtype=torch.int64).to(self.device)

            def readout_logits(zb, yb):
                pre, _, _ = self.model(zb, yb)
                return pre.index_fill(dim=1, index=not_mask,
                                      value=float('-inf'))

            seed_l = dsar_stable_seed(getattr(self.args, 'seed', 0),
                                      'dsarL', self.id, self.task_id,
                                      int(round_id))
            l_factors, l_status, l_ce = dsar_fit_l(
                z_minus, y, readout_logits, seed_l,
                rank=self._dsar_recipe["l_rank"])
            # L 拟合的 backward 会在冻结 Tail_Anchor 参数上留下 grad
            # 缓冲；清掉（训练流 optimizer 每轮新建，不受影响）
            self.model.zero_grad(set_to_none=True)
            self._dsar_pending = {
                "round_id": int(round_id), "entry": str(entry),
                "row_hash": row_hash, "train_idx": train_idx,
                "birth_prompt_hash": birth_hash,
                "z_minus": z_minus, "y": y,
                "l_factors": l_factors, "l_status": l_status,
                "l_final_ce": l_ce, "seed_l": seed_l,
            }
            print('[DSAR] client %d task %d round %d: L %s (n=%d%s)'
                  % (self.id, self.task_id, int(round_id), l_status,
                     int(z_minus.shape[0]),
                     '' if l_ce is None else ', final_ce=%.4f' % l_ce))
        finally:
            self.model.load_head(head_snapshot)
            self.vit.load_prompts(vit_prompt_snapshot)
            self.original_model.train(flags[2])
            self.model.train(flags[1])
            self.vit.train(flags[0])
            self._a5_rng_restore(rng)

    def _dsar_fit_post(self, round_id, entry):
        """post 半段（load_prompts(广播) 之后、任何 evaluate 之前）：
        同 manifest 采 z⁺ -> 无标签 B_{c,r} 拟合（固定 λ=N·s·γ* +
        80/20 sample-index-hash split + paired-MSE identity guard +
        全行重拟合 + SVD r8 截断；B 不接触任何 label）-> 与 pending 的
        L 组装 sidecar 一次提交（fresh-fit from identity，每轮覆盖）。"""
        if not self._dsar_on:
            return
        if self._dsar_sidecar is not None and (
                self._dsar_identity or self._dsar_pending is None):
            # identity 测试模式 / 无 birth state：pre 已提交 identity，
            # 这里只补记当前广播 prompt hash（G4 状态匹配标记）
            self._dsar_sidecar.meta["prompt_hash"] = \
                hash_prompt_module(self.vit.prompt)
            self._dsar_pending = None
            return
        pend = self._dsar_pending
        self._dsar_pending = None
        if pend is None:
            return
        rng = self._a5_rng_snapshot()
        flags = (self.vit.training, self.model.training,
                 self.original_model.training)
        try:
            self.original_model.eval()
            self.vit.eval()
            self.model.eval()
            if (int(round_id) != pend["round_id"]
                    or str(entry) != pend["entry"]):
                raise RuntimeError(
                    "client %d: DSAR halves mismatched (pre r%d %s vs "
                    "post r%d %s; fail-closed)"
                    % (self.id, pend["round_id"], pend["entry"],
                       int(round_id), entry))
            train_idx, row_hash = self._dsar_train_rows()
            if row_hash != pend["row_hash"]:
                raise RuntimeError(
                    "client %d round %d: DSAR GATE VIOLATION: the row "
                    "manifest changed between the capture halves "
                    "(fail-closed)" % (self.id, round_id))
            z_plus, y2 = self._dsar_forward_train_rows(train_idx)
            if not torch.equal(y2, pend["y"]):
                raise RuntimeError(
                    "client %d round %d: DSAR GATE VIOLATION: paired row "
                    "labels diverged between capture halves (fail-closed)"
                    % (self.id, round_id))
            seed_b = dsar_stable_seed(getattr(self.args, 'seed', 0),
                                      'dsarB', self.id, self.task_id,
                                      int(round_id))
            b_factors, b_status, b_aux = dsar_fit_bridge(
                z_plus, pend["z_minus"], train_idx, seed_b,
                gamma_star=self._dsar_recipe["gamma_star"])
            meta = {
                "version": DSAR_VERSION,
                "recipe": self._dsar_recipe["recipe"],
                "round_id": int(round_id), "entry": str(entry),
                "task_id": int(self.task_id),
                "b_status": b_status, "l_status": pend["l_status"],
                "l_final_ce": pend["l_final_ce"],
                "birth_prompt_hash": pend["birth_prompt_hash"],
                "prompt_hash": hash_prompt_module(self.vit.prompt),
                "n_train": int(len(train_idx)),
                "b_aux": b_aux,
            }
            sidecar = DSARSidecar(
                b_factors is not None,
                (b_factors or {}).get("M1"),
                (b_factors or {}).get("M2"),
                (b_factors or {}).get("bias"),
                pend["l_factors"] is not None,
                (pend["l_factors"] or {}).get("M1"),
                (pend["l_factors"] or {}).get("M2"),
                (pend["l_factors"] or {}).get("bias"),
                meta)
            if not sidecar.rank_shapes_ok(
                    l_rank=self._dsar_recipe["l_rank"]) \
                    or not sidecar.factors_finite():
                raise RuntimeError(
                    "client %d round %d: DSAR committed malformed r8 "
                    "factors (fail-closed)" % (self.id, round_id))
            self._dsar_sidecar = sidecar
            print('[DSAR] client %d task %d round %d: B %s (n_fit=%d '
                  'n_cal=%d) + L %s -> sidecar committed'
                  % (self.id, self.task_id, int(round_id), b_status,
                     int(b_aux.get("n_fit", -1)),
                     int(b_aux.get("n_cal", -1)), pend["l_status"]))
        finally:
            self.original_model.train(flags[2])
            self.model.train(flags[1])
            self.vit.train(flags[0])
            self._a5_rng_restore(rng)

    def load_global_weights(self,weights):
        self.model.load_state_dict(weights)
        self.evaluate(self.task_id,self.nb_classes)

    def get_global_proto_and_head(self,proto,head,prompt,round):
        # §34.2 0b pre（z⁻）捕获：load_prompts 之前 = 本地训练后 birth
        # state；默认（capture OFF）= 完全 no-op
        self._bsab_capture_pre(round, entry='broadcast')

        # §34.4/§34.4a B1 DSAR pre 半段（z⁻ 采集 + L 拟合；
        # 默认 OFF = 完全 no-op）
        self._dsar_fit_pre(round, entry='broadcast')

        self.global_protos = deepcopy(proto)

        self.global_head = head

        self.prompts = prompt
        self.vit.load_prompts(self.prompts)

        # §34.2 0b post（z⁺）捕获：广播 prompt 已装载、任何 post-broadcast
        # refit / evaluate 之前；默认（capture OFF）= 完全 no-op
        self._bsab_capture_post(round, entry='broadcast')

        # §34.4/§34.4a B1 DSAR post 半段（z⁺ 采集 + B 拟合 + sidecar
        # 提交；随后的 evaluate 读到最新已提交 sidecar；默认 OFF =
        # 完全 no-op）
        self._dsar_fit_post(round, entry='broadcast')

        # A5-v2 §28 判据臂（--refit_timing post_broadcast）：广播后接口
        # 状态（融合 prompt 已装载）上的 refit，特征流与随后的 evaluate
        # 同源；默认（local 时点 / OFF）= 完全 no-op
        self._a5_post_broadcast_interface(round, self.args)

        acc_task0 = self.evaluate(0, self.nb_classes)
        acc_current = self.evaluate(self.task_id,self.nb_classes)
        # Phase 0: return the (unchanged) evaluation results so the server
        # can log them to metrics/train_log.csv; behavior is identical.
        return {0: acc_task0, self.task_id: acc_current}

    def get_global_proto_and_head_no_test(self, proto, head, prompt, round):
        # §34.2 0b pre（z⁻）捕获（passive-client 广播路径；默认 OFF =
        # 完全 no-op）
        self._bsab_capture_pre(round, entry='broadcast_no_test')

        # §34.4/§34.4a B1 DSAR pre 半段（passive client 同样每轮拟合/
        # 提交 sidecar——task-end 全任务评估走所有客户端的已提交
        # sidecar；默认 OFF = 完全 no-op）
        self._dsar_fit_pre(round, entry='broadcast_no_test')

        self.global_protos = deepcopy(proto)

        self.global_head = head

        self.prompts = prompt
        self.vit.load_prompts(self.prompts)

        # §34.2 0b post（z⁺）捕获（默认 OFF = 完全 no-op）
        self._bsab_capture_post(round, entry='broadcast_no_test')

        # §34.4/§34.4a B1 DSAR post 半段（默认 OFF = 完全 no-op）
        self._dsar_fit_post(round, entry='broadcast_no_test')


    def get_head(self,head):
        self.heads[self.task_id] =deepcopy(head)
        self.evaluate_only_heads(0, self.nb_classes)
        self.evaluate_only_heads(self.task_id, self.nb_classes)

    def train_only_heads(self, round, args):

        ###train###
        # update train data
        task = round // self.task_per_global_epoch
        flag = True
        if self.task_id != task:
            flag = False
            self.get_data(task)
            self.task_id = task

        if self.heads[self.task_id] is not None:
            self.head.load_head(self.heads[self.task_id])

        self.head.to(self.device)
        train_loader = DataLoader(self.traindata, batch_size=16, num_workers=args.num_workers,
            pin_memory=args.pin_mem, shuffle=True)
        print(f'Client {self.id} on Task {self.task_id} is training prompts')

        # Input-Enhancement
        optimizer = torch.optim.Adam(self.head.parameters(), lr=self.lr,weight_decay=1e-03)
        criterion = torch.nn.CrossEntropyLoss().to(self.device)

        for epoch in range(self.local_epoch):
            for iteration, (input,target) in enumerate(train_loader):
                input, target = Variable(input, requires_grad=False).to(self.device, non_blocking=True), target.to(self.device,non_blocking=True)
                # input走一遍获得cls——token
                with torch.no_grad():
                    if self.original_model is not None:
                        output = self.original_model(input)
                output = self.head(output['feat'])
                logits = output
                # 加了class_mask
                mask = self.current_class
                not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                not_mask = torch.tensor(not_mask, dtype=torch.int64).to(self.device)
                logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))
                loss = criterion(logits,target)
                optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip_grad)
                optimizer.step()

        ###evaluation###
        self.heads[self.task_id] = deepcopy(self.head.get_head())
        self.evaluate_only_heads(0,args.nb_classes)
        self.evaluate_only_heads(self.task_id,args.nb_classes)


    def evaluate(self,task=0,nb_classes=None):
        """Official FedTA evaluate(), restored verbatim: accuracy only.

        Phase-0/1 margin recording has been isolated into
        evaluate_with_margin() so that --no_instrumentation gives the exact
        official evaluation path (the only addition is `return float(acc)`,
        which does not affect any computation)."""
        test_data = self.test_loader[task]
        test_loader = DataLoader(test_data,batch_size=8,shuffle=True)
        correct =0
        total = 0

        self.model.load_head(self.heads[task])
        self.model.to(self.device)

        for iteration, (input,target) in enumerate(test_loader):
            input = input.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            # if iteration==0:
            #     input_ = input[0]
            #     self.model.forward_visual(input_)
            with torch.no_grad():
                if self.original_model is not None:
                    output = self.original_model(input)
                    output = output['pre_logits'].requires_grad_(False)
                    output = self.vit(input, task_id=self.task_id, cls_features=output, train=True)
                    z_feat = output['feat'].to(self.device)
                    if self._dsar_sidecar is not None:
                        # §34.4a frozen dispatch: historical task = B-only,
                        # current task = B -> L. Exact identity adapters
                        # short-circuit independently (B identity != DSAR
                        # identity); with both inactive the SAME tensor
                        # object is returned (bitwise-neutral).
                        z_feat = self._dsar_sidecar.apply(
                            z_feat, task == self.task_id)
                    pre, output_mixed, _ = self.model(z_feat, target.to(self.device))
            logits = pre

            # logits = output['logits']

            # class_mask
            mask = self.class_mask[task]
            not_mask = np.setdiff1d(np.arange(nb_classes), mask)
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(self.device)
            logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

            predicts = torch.max(logits, dim=1)[1].cpu()
            # print(predicts)
            # print(target)
            correct += (predicts == target.cpu()).sum()
            total += len(target)

        acc = 100 * correct / total

        print(f'{acc}')
        return float(acc)

    def evaluate_with_margin(self,task=0,nb_classes=None):
        """Observer-only variant of evaluate(): identical accuracy
        computation, additionally records margin stats into
        self.last_margin_stats.

        margin = logit_y - max_{j != y} logit_j on the task-masked logits
        the prediction actually uses. Pure recording: no RNG consumption,
        no parameter change. Called ONLY from instrumentation paths
        (evaluate_all_seen_tasks -> run_full_evaluation); the official
        training loop must keep calling evaluate()."""
        test_data = self.test_loader[task]
        test_loader = DataLoader(test_data,batch_size=8,shuffle=True)
        correct =0
        total = 0

        self.model.load_head(self.heads[task])
        self.model.to(self.device)

        margin_list = []

        for iteration, (input,target) in enumerate(test_loader):
            input = input.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            # if iteration==0:
            #     input_ = input[0]
            #     self.model.forward_visual(input_)
            with torch.no_grad():
                if self.original_model is not None:
                    output = self.original_model(input)
                    output = output['pre_logits'].requires_grad_(False)
                    output = self.vit(input, task_id=self.task_id, cls_features=output, train=True)
                    z_feat = output['feat'].to(self.device)
                    if self._dsar_sidecar is not None:
                        # §34.4a frozen dispatch (same insertion as
                        # evaluate(): task-end run_full_evaluation reads
                        # through this path and MUST see the latest
                        # committed sidecar)
                        z_feat = self._dsar_sidecar.apply(
                            z_feat, task == self.task_id)
                    pre, output_mixed, _ = self.model(z_feat, target.to(self.device))
            logits = pre

            # logits = output['logits']

            # class_mask
            mask = self.class_mask[task]
            not_mask = np.setdiff1d(np.arange(nb_classes), mask)
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(self.device)
            logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

            predicts = torch.max(logits, dim=1)[1].cpu()
            # print(predicts)
            # print(target)
            correct += (predicts == target.cpu()).sum()
            total += len(target)

            # margin recording (masked logits, y = ground truth)
            with torch.no_grad():
                logit_y = logits.gather(1, target.unsqueeze(1)).squeeze(1)
                other = logits.scatter(1, target.unsqueeze(1), float('-inf'))
                margin_list.append((logit_y - other.max(dim=1)[0]).cpu())

        acc = 100 * correct / total

        if margin_list:
            m = torch.cat(margin_list)
            self.last_margin_stats = {
                "mean": float(m.mean()),
                "median": float(m.median()),
                "p10": float(torch.quantile(m, 0.10)),
                "n_samples": int(m.numel()),
            }
        else:
            self.last_margin_stats = None

        print(f'{acc}')
        return float(acc)

    def evaluate_all_seen_tasks(self, nb_classes=None):
        """Evaluate task0 -> current task in ascending order, so the model is
        left with the current task head loaded. Returns {task: accuracy}.

        Phase 1: also collects the per-task margin stats recorded by
        evaluate_with_margin() into self.last_margin_stats_by_task."""
        results = {}
        margin_by_task = {}
        for task in range(self.task_id + 1):
            if task >= len(self.heads) or self.heads[task] is None:
                continue
            results[task] = self.evaluate_with_margin(task, nb_classes)
            margin_by_task[task] = self.last_margin_stats
        self.last_margin_stats_by_task = margin_by_task
        return results

    # ------------------------------------------------------------------
    # Phase 0: checkpoint state (round-boundary resume)
    # ------------------------------------------------------------------

    def _a5_tail_anchor_state(self):
        """tail_anchor_model checkpoint 状态。OFF 臂 = 官方原样；ON 臂把
        head.* 键用联训头覆写（联训轨迹语义 + schema 兼容，见
        state_dict_for_checkpoint 注释）。"""
        ms = module_state_or_none(self.model)
        if not self._a5_refit_on or self._a5_joint_head is None:
            return ms
        joint_state = self._a5_joint_head.state_dict()
        merged = {k: v for k, v in ms.items()
                  if not k.startswith("head.")}
        for k, v in joint_state.items():
            merged["head." + k] = v.detach().cpu()
        return merged

    def state_dict_for_checkpoint(self):
        heads_state = [module_state_or_none(h) for h in self.heads]
        state = {
            "id": self.id,
            "task_id": self.task_id,

            # Tail_Anchor: key / anchor_pool / current Chead
            # A5-v2 ON 臂保存"联训轨迹"状态：训练结束时刻 model.head 可能
            # 已被 end-of-round evaluate 装成 refit 头（standardize 臂为
            # StandardizedChead，含 mean/std buffer，schema 与 Chead 不同，
            # 直接存会破坏后续 load_state_dict 兼容性）。联训头独立存于
            # _a5_joint_head，phase-2 前由 P4 恢复——这里的 head.* 键用联训
            # 头覆写，保证 tail_anchor_model 始终 = 联训轨迹状态。
            "tail_anchor_model": self._a5_tail_anchor_state(),

            "prompts": module_state_or_none(self.prompts),

            # task-specific heads, stored as state_dicts (None stays None)
            "heads": heads_state,

            "global_protos": self.global_protos,
            "local_protos": self.local_protos,

            "current_class": getattr(self, "current_class", None),
            "class_mask": self.class_mask,

            "data_split_indices": self.data_split_indices,
        }

        # A5-v2（--head_refit）：joint-head 轨迹状态。schema 仅 ON 臂新增
        # key（frozen 基线 checkpoint 逐位不变）
        if self._a5_refit_on:
            state["a5_joint_head"] = module_state_or_none(self._a5_joint_head)

        # §34.4/§34.4a B1 DSAR：最新已提交 sidecar（task-end 全任务评估
        # / resume 都必须读到与该时刻 broadcast 状态匹配的 sidecar）。
        # schema 仅 ON 臂新增 key（OFF 基线 checkpoint 逐位不变）。
        if self._dsar_on:
            state["dsar_sidecar"] = (
                self._dsar_sidecar.state_dict()
                if self._dsar_sidecar is not None else None)

        # A2：保护状态封章。schema 仅 ON 臂新增 key（OFF 基线 checkpoint
        # 逐位不变）；与 dsar_sidecar 独立共存
        if self._a2_arm != 'off':
            state["a2_state"] = self._a2_state_for_checkpoint()

        return state

    def _a2_state_for_checkpoint(self):
        """Serialize A2 protection state; always a dict when --a2_arm != off,
        never called for the off baseline (schema untouched)."""
        return {
            "a2_arm": self._a2_arm,
            "task_id": self._a2_task_id,
            "protected_slots": (None if self._a2_protected_slots is None
                                else self._a2_protected_slots.cpu().tolist()),
            "key_snapshot": (None if self._a2_key_snapshot is None
                             else self._a2_key_snapshot.detach().cpu()),
            "anchor_snapshot": (None if self._a2_anchor_snapshot is None
                                else self._a2_anchor_snapshot.detach().cpu()),
        }

    def _a2_load_checkpoint_state(self, state):
        """Restore A2 protection state, FAIL-CLOSED.

        An --a2_arm != off resume from a checkpoint without a2_state would
        silently continue as unprotected A2 (wrong arm semantics, wrong
        science). The state MUST exist and be internally consistent, or we
        refuse.
        """
        a2 = state.get("a2_state")
        if not a2:
            raise RuntimeError(
                f"client {self.id}: A2 resume requires a2_state in the "
                f"checkpoint, got "
                f"{'missing key' if 'a2_state' not in state else 'None/empty'}"
                f" -- refusing to silently continue as unprotected A2 "
                f"(fail-closed; re-run from scratch or use an --a2_arm "
                f"checkpoint)")
        if a2.get("a2_arm") != self._a2_arm:
            raise RuntimeError(
                f"client {self.id}: A2 arm mismatch on resume "
                f"(checkpoint={a2.get('a2_arm')}, current={self._a2_arm})")
        if a2.get("task_id") != state.get("task_id"):
            raise RuntimeError(
                f"client {self.id}: A2 protection task_id "
                f"({a2.get('task_id')}) != checkpoint task_id "
                f"({state.get('task_id')}); corrupted a2_state")
        self._a2_task_id = a2.get("task_id")
        slots = a2.get("protected_slots")
        self._a2_protected_slots = (None if slots is None
                                    else torch.tensor(slots, dtype=torch.long,
                                                      device=self.device))
        key_snap = a2.get("key_snapshot")
        anchor_snap = a2.get("anchor_snapshot")
        # consistency: protected slots must have exactly-matching snapshots
        # for the rows the arm protects
        if slots is not None:
            n = len(slots)
            if self._a2_arm in ('key', 'both') and (
                    key_snap is None or key_snap.shape[0] != n):
                raise RuntimeError(
                    f"client {self.id}: arm {self._a2_arm} requires a "
                    f"key_snapshot with {n} rows, got "
                    f"{'None' if key_snap is None else key_snap.shape[0]}")
            if self._a2_arm in ('anchor', 'both') and (
                    anchor_snap is None or anchor_snap.shape[0] != n):
                raise RuntimeError(
                    f"client {self.id}: arm {self._a2_arm} requires an "
                    f"anchor_snapshot with {n} rows, got "
                    f"{'None' if anchor_snap is None else anchor_snap.shape[0]}")
        self._a2_key_snapshot = (None if key_snap is None
                                 else key_snap.to(self.device))
        self._a2_anchor_snapshot = (None if anchor_snap is None
                                    else anchor_snap.to(self.device))

    def load_checkpoint_state(self, state):
        if state["id"] != self.id:
            raise ValueError(
                f"Client id mismatch: checkpoint={state['id']}, current={self.id}")

        self.task_id = state["task_id"]

        # Tail_Anchor (key / anchor_pool / head)
        # A5-v2 老版本 checkpoint（2026-10-29 修复前的 standardize 臂）在
        # 保存时 model.head 仍是 refit 的 StandardizedChead，带
        # head.mean/head.std buffer 键，严格 load_state_dict 会拒载。联训
        # 轨迹头由 a5_joint_head 恢复、phase-2 前由 P4 装回，这里剔除
        # 多余 head.* 键即可恢复兼容（仅 ON 臂；OFF 臂遇外来键保持报错）。
        tam = state["tail_anchor_model"]
        if self._a5_refit_on:
            expected = {k for k in self.model.state_dict()
                         if k.startswith("head.")}
            foreign = sorted(k for k in tam
                            if k.startswith("head.") and k not in expected)
            if foreign:
                tam = {k: v for k, v in tam.items() if k not in set(foreign)}
                state = dict(state)
                state["tail_anchor_model"] = tam
                print("[A5-v2] client %d resume: dropping foreign head keys "
                      "from legacy checkpoint: %s (joint head restored "
                      "from a5_joint_head)" % (self.id, foreign))
        self.model.load_state_dict(tam)

        # prompts: deepcopy current vit prompt as template, then load state
        prompt_state = state.get("prompts")
        if prompt_state is not None:
            prompt = clone_prompt_template(self.vit)
            prompt.load_state_dict(prompt_state)
            self.prompts = prompt
        else:
            self.prompts = None

        # task-specific heads: template = deepcopy of current Tail_Anchor head
        # resume 时 self.model 尚在 CPU（train()/evaluate() 才 .to(device)），
        # deepcopy 出的模板也在 CPU -> 重建的 heads 落在 CPU；而它们是普通
        # Python 属性，train() 的 model.to(device) 搬不动 -> 首个 phase-2 前
        # 向 / evaluate 的 load_head 即崩（Tail_Anchor.head cpu vs cuda:0）。
        # 在重建源头统一搬到 self.device（首跑时 heads 训练中在 CUDA 创建，
        # 此 .to() 无操作，逐位不变）。
        heads = []
        for h_state in state.get("heads", []):
            if h_state is None:
                heads.append(None)
            else:
                heads.append(rebuild_client_head(
                    h_state, deepcopy(self.model.get_head())).to(self.device))
        self.heads = heads

        # A5-v2：joint-head 轨迹恢复，fail-closed（ON 臂 resume 从缺 key
        # 的 checkpoint 继续 = 联训轨迹静默漂移，拒绝）
        if self._a5_refit_on:
            jh = state.get("a5_joint_head")
            if jh is None:
                raise RuntimeError(
                    "client %d: A5-v2 resume requires a5_joint_head in "
                    "the checkpoint -- refusing to silently continue "
                    "with a drifted joint-head trajectory (fail-closed; "
                    "re-run from scratch)" % self.id)
            self._a5_joint_head = rebuild_client_head(
                jh, deepcopy(self.model.get_head())).to(self.device)

        # §34.4/§34.4a B1 DSAR：sidecar 恢复，fail-closed（ON 臂 resume
        # 从缺 key 的 checkpoint 继续 = 评估读出的 adapter 状态静默漂移，
        # 拒绝；结构与 r8 factor 形状再验）
        if self._dsar_on:
            if "dsar_sidecar" not in state:
                raise RuntimeError(
                    "client %d: DSAR resume requires dsar_sidecar in the "
                    "checkpoint -- refusing to silently continue with a "
                    "drifted adapter state (fail-closed; re-run from "
                    "scratch)" % self.id)
            sc_state = state["dsar_sidecar"]
            self._dsar_sidecar = (
                DSARSidecar.from_state(sc_state, self.device)
                if sc_state is not None else None)
            if (self._dsar_sidecar is not None
                    and not (self._dsar_sidecar.rank_shapes_ok(
                        l_rank=self._dsar_recipe["l_rank"])
                        and self._dsar_sidecar.factors_finite())):
                raise RuntimeError(
                    "client %d: DSAR resume found malformed r8 factors in "
                    "the checkpoint (fail-closed; a CIFAR-recipe r8 L "
                    "checkpoint cannot resume under the INR r16 recipe and "
                    "vice versa)" % self.id)

        # A2：保护状态恢复，fail-closed（同 DSAR 纪律：缺 key / arm 不匹
        # 配 / 内部不一致都拒绝，不允许静默变成无保护 A2）
        if self._a2_arm != 'off':
            self._a2_load_checkpoint_state(state)

        self.global_protos = state.get("global_protos")
        self.local_protos = state.get("local_protos")

        if state.get("current_class") is not None:
            self.current_class = state["current_class"]
        if state.get("class_mask") is not None:
            self.class_mask = state["class_mask"]

        self.data_split_indices = state.get("data_split_indices", {})

        # rebuild all seen-task test loaders + current traindata
        self.rebuild_data_from_indices()

    def rebuild_data_from_indices(self):
        """Rebuild test_loader[0..task_id] and the current traindata from the
        recorded split indices, so evaluate(old_task) works after resume."""
        self.test_loader = []
        for task_id in sorted(self.data_split_indices.keys()):
            base_dataset = self.train_data[task_id]
            split = self.data_split_indices[task_id]
            train_subset = Subset(base_dataset, self._ctrp_train_indices(task_id))
            test_subset = Subset(base_dataset, list(split["test"]))
            self.test_loader.append(test_subset)
            if task_id == self.task_id:
                self.traindata = train_subset
                self.train_dataset = base_dataset
        print(f'Client {self.id}: restored {len(self.test_loader)} test loader(s) from checkpoint')

    def _ctrp_train_indices(self, task_id):
        """§33 CTRP v1: under the matched protocol the training budget is
        the 80% core split of the inner train (the 20% val side NEVER
        feeds the refit cache -- otherwise the alpha-selection rows leak
        into the head); off = the full inner-train (legacy) indices."""
        split = self.data_split_indices[task_id]
        if self._ctrp_on:
            if "ctrp_core" not in split or not split["ctrp_core"]:
                raise RuntimeError(
                    "client %d: task %d is missing the matched ctrp_core "
                    "split (CTRP resume requires a CTRP checkpoint; "
                    "fail-closed)" % (self.id, task_id))
            return list(split["ctrp_core"])
        return list(split["train"])

    def evaluate_cosin_similarity(self,task=0,nb_classes=None):
        test_data = self.test_loader[task]
        test_loader = DataLoader(test_data, batch_size=4, shuffle=True, num_workers=2)
        correct = 0
        total = 0

        for iteration, (input, target) in enumerate(test_loader):
            input = input.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            with torch.no_grad():
                if self.original_model is not None:
                    output = self.original_model(input)

                _, output_mix, _ = self.model(output, None)
            for i, label in enumerate(target):
                predicts = CosineSimilarityClassifier(output_mix[i].squeeze(0),self.global_protos,self.current_class)
                if predicts ==label:
                    correct +=1
            total += len(target)
        acc = 100 * correct / total
        print(f'Client {self.id} on Task {task} acc is {acc}')

    def evaluate_only_prompts(self,task=0,nb_classes=None):
        test_data = self.test_loader[task]
        test_loader = DataLoader(test_data,batch_size=4,shuffle=True,num_workers=2)

        correct =0
        total = 0
        for iteration, (input,target) in enumerate(test_loader):
            input = input.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            # if iteration==0:
            #     input_ = input[0]
            #     self.model.forward_visual(input_)
            with torch.no_grad():
                if self.original_model is not None:
                    output = self.original_model(input)
                    output = output['pre_logits'].requires_grad_(False)
                output = self.vit(input, task_id=self.task_id, cls_features=output, train=True)
            #     pre, output_mixed, _ = self.model(output['feat'].to(self.device), target.to(self.device))
            # logits = pre

            logits = output['logits']
            # class_mask
            mask = self.class_mask[task]
            not_mask = np.setdiff1d(np.arange(nb_classes), mask)
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(self.device)
            logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

            predicts = torch.max(logits, dim=1)[1].cpu()
            # print(predicts)
            # print(target)
            correct += (predicts == target.cpu()).sum()
            total += len(target)

        acc = 100 * correct / total

        print(f'{acc}')


    def evaluate_only_heads(self,task=0,nb_classes=None):
        test_data = self.test_loader[task]
        test_loader = DataLoader(test_data, batch_size=4, shuffle=True, num_workers=2)
        self.vit.load_head(self.heads[task])
        self.vit.to(self.device)
        correct = 0
        total = 0
        for iteration, (input, target) in enumerate(test_loader):
            input = input.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            with torch.no_grad():
                if self.original_model is not None:
                    output = self.original_model(input)

                output = self.head(output['feat'])
            logits = output
            # 加了class_mask
            mask = self.class_mask[task]
            not_mask = np.setdiff1d(np.arange(nb_classes), mask)
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(self.device)
            logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))
            predicts = torch.max(logits, dim=1)[1].cpu()
            # print(predicts)
            # print(target)
            correct += (predicts == target.cpu()).sum()
            total += len(target)

        acc = 100 * correct / total

        print(f'{acc}')

    # Contrastive Learning
    def calculate_infonce(self, feature, label,is_last):
        # print(self.global_protos.keys())
        # print(label)

        all_global_protos_keys = np.array(list(self.global_protos.keys()))
        all_protos = []
        for protos_key in all_global_protos_keys:
            all_protos.append(self.global_protos[protos_key])
        all_protos = np.vstack(all_protos)


        pos_index = np.where(all_global_protos_keys == label)[0]
        neg_index = np.where(
            (all_global_protos_keys != label)
        )[0]
        f_pos = torch.from_numpy(all_protos[pos_index]).to(self.device)
        f_neg = torch.from_numpy(all_protos[neg_index]).to(self.device)
        f_proto = torch.cat((f_pos, f_neg), dim=0)

        l = torch.cosine_similarity(feature, f_proto, dim=1)


        l = l / 0.2

        exp_l = torch.exp(l)
        exp_l = exp_l.view(1, -1)
        pos_mask = [1 for _ in range(f_pos.shape[0])] + [
            0 for _ in range(f_neg.shape[0])]
        pos_mask = torch.tensor(pos_mask, dtype=torch.float).to(self.device)
        pos_mask = pos_mask.view(1, -1)
        pos_l = exp_l * pos_mask
        sum_pos_l = pos_l.sum(1)
        sum_exp_l = exp_l.sum(1)
        if is_last:
            infonce_loss = 1-torch.log(sum_pos_l)
        else:
            infonce_loss = -torch.log(sum_pos_l / sum_exp_l)
        return infonce_loss


    def show_embeddings(self,train_loader,task):
        target_list = []
        feature_list = []
        keep = [48,71,84,89,93]

        self.model.to(self.device)
        for idx,(input,target) in enumerate(train_loader):
            input, target = Variable(input, requires_grad=False).to(self.device, non_blocking=True), target.to(
                self.device, non_blocking=True)
            with torch.no_grad():
                if self.original_model is not None:
                    output = self.original_model(input)
                    output = output['pre_logits'].requires_grad_(False)
                    output = self.vit(input, task_id=self.task_id, cls_features=output, train=True)
                    pre, output_mixed, _ = self.model(output['feat'].to(self.device), target.to(self.device))
                    for i in range(len(target)):
                        if target[i] in keep:
                            target_list.append(target[i])
                            feature_list.append(output_mixed[i])

        target_list = torch.Tensor(target_list)
        feature_list = torch.cat(feature_list,dim=0)

        target_list=target_list.cpu()
        feature_list = feature_list.reshape(-1,1536).cpu()

        # print(target_list.shape)
        # print(feature_list.shape)
        tsne = TSNE(n_components=2, early_exaggeration=2.0,metric='cosine', random_state=42)
        x_tsne = tsne.fit_transform(feature_list)
        self.plot_xy(x_tsne, target_list,task)

    def plot_xy(self,x_values, label,task):

        df = pd.DataFrame(x_values, columns=['x', 'y'])
        df['label'] = np.array(label)
        df['label'].astype(str)
        colors = ['darkorange','darkgreen','darkblue','darkred','darkorchid']
        sns.scatterplot(data=df,x="x", y="y", hue='label',palette=colors)

        plt.axis('off')
        # plt.legend(False)
        plt.savefig(f'{task} scattor_plot client {self.id}', bbox_inches='tight', pad_inches=0.0)
        plt.show()


    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm


    def train_only_prompts(self, round, args):
        self.original_model.eval()

        if self.prompts is not None:
            self.vit.load_prompts(self.prompts)
        else:
            self.vit.init_prompts()

        if self.heads[self.task_id] is not None:
            self.vit.load_head(self.heads[self.task_id])

        ###train###
        # update train data
        task = round // self.task_per_global_epoch
        flag = True
        if self.task_id != task:
            flag = False
            self.get_data(task)
            self.task_id = task

        self.model.to(self.device)
        self.vit.to(self.device)
        train_loader = DataLoader(self.traindata, batch_size=16, num_workers=args.num_workers,
            pin_memory=args.pin_mem, shuffle=True)
        print(f'Client {self.id} on Task {self.task_id} is training prompts')

        # Input-Enhancement
        optimizer = self._build_vit_optimizer(args)
        criterion = torch.nn.CrossEntropyLoss().to(self.device)
        cos = nn.CosineEmbeddingLoss()
        for epoch in tqdm(range(self.local_epoch)):
            for iteration, (input,target) in enumerate(train_loader):
                input, target = Variable(input, requires_grad=False).to(self.device, non_blocking=True), target.to(self.device,non_blocking=True)

                with torch.no_grad():
                    if self.original_model is not None:
                        output = self.original_model(input)
                        cls_features = output['pre_logits']
                output = self.vit(input, task_id=self.task_id, cls_features=cls_features, train=True)
                # pre, output_mixed, pull_off2 = self.model(output['feat'].to(self.device), target.to(self.device))
                # logits = pre
                logits = output['logits']
                # output_mixed = output['pre_logits']
                pull_off = output['reduce_sim']
                # class_mask
                mask = self.current_class
                not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                not_mask = torch.tensor(not_mask, dtype=torch.int64).to(self.device)
                logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

                loss = criterion(logits,target)  - 0.1 * pull_off
                optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip_grad)
                optimizer.step()

        self.heads[self.task_id] = deepcopy(self.vit.head)
        self.prompts = deepcopy(self.vit.get_prompts())
        self.evaluate_only_prompts(0,args.nb_classes)
        self.evaluate_only_prompts(self.task_id,args.nb_classes)



    def get_global_prompt_head(self,head,prompt):
        # self.global_protos = deepcopy(proto)
        self.heads[self.task_id] = deepcopy(head)
        self.model.head.to(self.device)
        self.prompts = prompt
        self.vit.load_prompts(self.prompts)
        self.evaluate_only_prompts(0, self.nb_classes)
        self.evaluate_only_prompts(self.task_id,self.nb_classes)




    def evaluate_on_global_testset(self,testdata):

        test_loader = DataLoader(testdata, batch_size=16, shuffle=True, num_workers=2)

        self.vit.load_head(self.global_head)
        self.vit.to(self.device)

        correct = 0
        total = 0
        for iteration, (input, target) in enumerate(test_loader):
            input = input.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            with torch.no_grad():
                if self.original_model is not None:
                    output = self.original_model(input)
                    output = output['pre_logits'].requires_grad_(False)
                    output = self.vit(input, task_id=self.task_id, cls_features=output, train=True)
                    pre, output_mixed, _ = self.model(output['feat'].to(self.device), target.to(self.device))

                output = self.head(output['feat'])
            logits = output

            # 加了class_mask

            predicts = torch.max(logits, dim=1)[1].cpu()
            # print(predicts)
            # print(target)
            correct += (predicts == target.cpu()).sum()
            total += len(target)

        acc = 100 * correct / total

        print(f'{acc}')
