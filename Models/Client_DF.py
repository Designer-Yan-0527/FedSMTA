import time
from copy import deepcopy
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


# 每个客户端拥有的
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
        self.model = self.init_local_model(model_name)
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

        # Phase 4 A2 (--method fedta_a2): routing-interface protection state.
        # Attributes are ONLY created for the A2 method; the frozen fedta
        # baseline keeps its __init__ / checkpoint schema untouched.
        if method == 'fedta_a2':
            self.a2_arm = getattr(args, 'a2_arm', 'anchor')
            self._a2_protected_slots = None   # LongTensor: protected slot ids
            self._a2_key_snapshot = None      # frozen key rows (arm key/both)
            self._a2_anchor_snapshot = None   # frozen anchor rows (arm anchor/both)
            self._a2_task_id = None           # task the current protection belongs to

        self.head = Chead(args.nb_classes)


    def init_local_model(self,model_name):
        if model_name=='Tail_Anchor':
            return Tail_Anchor(10,768,200)
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


    def update_data(self,round, args):
        task = round // self.task_per_global_epoch
        flag = True
        if self.task_id != task:
            flag = False
            # only cifar100 / ImageNet-R are supported in this project
            self.get_data(task)
            self.task_id = task
            # Phase 4 A2: entering a new task -> recompute the protected slot
            # set + row snapshots from the CURRENT (post-previous-task) key /
            # anchor state. No-op for task 0 (empty protection set).
            if self.method == 'fedta_a2':
                self._a2_init_task_protection(args)

    def _a2_init_task_protection(self, args):
        """A2: compute the protected slot set and row snapshots at task start.

        Protection set (pre-registered, roadmap S6.2) = union over all
        completed tasks t' < task_id of the top-1 routing destinations of
        their train-split features under the CURRENT key (= D0 emergent
        support, B_t > 0). The snapshot is taken once at task start and the
        protected rows are restored after every Tail_Anchor optimizer step
        for the whole task (post-step restore == freeze + weight-decay
        exemption; the official optimizer is never modified).

        The feature pass is RNG-neutral: sequential loader (shuffle=False,
        num_workers=0) + torch RNG state snapshot/restore.
        """
        self._a2_task_id = self.task_id
        completed = [t for t in range(self.task_id)
                     if t in self.data_split_indices]
        if not completed:
            self._a2_protected_slots = None
            self._a2_key_snapshot = None
            self._a2_anchor_snapshot = None
            return

        # RNG-neutral: snapshot ALL RNG streams the feature pass could touch
        # (python `random` and numpy feed dataset transforms/augmentation;
        # a leaked advance here would perturb subsequent training batches and
        # contaminate the causal comparison between arms)
        rng_state = {
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda_all": (torch.cuda.get_rng_state_all()
                               if torch.cuda.is_available() else None),
            "python": random.getstate(),
            "numpy": np.random.get_state(),
        }
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
                                         self.a2_arm in ('key', 'both'))
                                     else None)
            self._a2_anchor_snapshot = (self.model.anchor_pool.data[slots].clone()
                                        if (slots is not None and
                                            self.a2_arm in ('anchor', 'both'))
                                        else None)
            n = 0 if slots is None else int(slots.numel())
            print(f'[A2] client {self.id} task {self.task_id} arm={self.a2_arm}: '
                  f'protected {n}/{self.model.key.shape[0]} slots')
        finally:
            # restore ALL RNG streams and module modes (train() never sets
            # vit.train(); leaving eval() on would silently change training;
            # original_model is shared across clients, so its mode must be
            # restored exactly as found)
            self.vit.train(vit_was_training)
            self.original_model.train(orig_was_training)
            torch.set_rng_state(rng_state["torch_cpu"])
            if rng_state["torch_cuda_all"] is not None:
                torch.cuda.set_rng_state_all(rng_state["torch_cuda_all"])
            random.setstate(rng_state["python"])
            np.random.set_state(rng_state["numpy"])

    def _a2_post_step_restore(self):
        """A2: restore protected key/anchor rows after a Tail_Anchor
        optimizer step. Only called when method == 'fedta_a2'."""
        if self._a2_protected_slots is None or self._a2_task_id != self.task_id:
            return
        if self.a2_arm in ('key', 'both') and self._a2_key_snapshot is not None:
            self.model.key.data[self._a2_protected_slots] = self._a2_key_snapshot
        if self.a2_arm in ('anchor', 'both') and self._a2_anchor_snapshot is not None:
            self.model.anchor_pool.data[self._a2_protected_slots] = self._a2_anchor_snapshot


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

        # training input enhancement
        optimizer = torch.optim.Adam(self.vit.parameters(), lr=self.lr,weight_decay=1e-03)
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

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-03)
        for epoch in tqdm(range(self.local_epoch)):
            for iteration, (input, target) in enumerate(train_loader):
                input, target = Variable(input, requires_grad=False).to(self.device, non_blocking=True), target.long().to(
                    self.device, non_blocking=True)

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
                # Phase 4 A2: restore protected key/anchor rows after the
                # Tail_Anchor step. No-op for the frozen fedta baseline.
                if self.method == 'fedta_a2':
                    self._a2_post_step_restore()


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

        # Official FedTA (GitHub main) calls evaluate unconditionally twice
        # here (evaluate(0) + evaluate(task_id)); on task 0 this evaluates
        # task 0 twice. This is intentional baseline behavior: evaluate()'s
        # test loader uses shuffle=True and consumes the global torch RNG,
        # so the call count MUST stay identical to the official code.
        self.evaluate( 0,args.nb_classes)
        self.evaluate(self.task_id, args.nb_classes)

        # save local input enhancement
        self.prompts = deepcopy(self.vit.get_prompts())

    def load_global_weights(self,weights):
        self.model.load_state_dict(weights)
        self.evaluate(self.task_id,self.nb_classes)

    def get_global_proto_and_head(self,proto,head,prompt,round):
        self.global_protos = deepcopy(proto)

        self.global_head = head

        self.prompts = prompt
        self.vit.load_prompts(self.prompts)

        acc_task0 = self.evaluate(0, self.nb_classes)
        acc_current = self.evaluate(self.task_id,self.nb_classes)
        # Phase 0: return the (unchanged) evaluation results so the server
        # can log them to metrics/train_log.csv; behavior is identical.
        return {0: acc_task0, self.task_id: acc_current}

    def get_global_proto_and_head_no_test(self, proto, head, prompt, round):
        self.global_protos = deepcopy(proto)

        self.global_head = head

        self.prompts = prompt
        self.vit.load_prompts(self.prompts)


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
                    pre, output_mixed, _ = self.model(output['feat'].to(self.device), target.to(self.device))
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
                    pre, output_mixed, _ = self.model(output['feat'].to(self.device), target.to(self.device))
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

    def state_dict_for_checkpoint(self):
        heads_state = [module_state_or_none(h) for h in self.heads]
        state = {
            "id": self.id,
            "task_id": self.task_id,

            # Tail_Anchor: key / anchor_pool / current Chead
            "tail_anchor_model": module_state_or_none(self.model),

            "prompts": module_state_or_none(self.prompts),

            # task-specific heads, stored as state_dicts (None stays None)
            "heads": heads_state,

            "global_protos": self.global_protos,
            "local_protos": self.local_protos,

            "current_class": getattr(self, "current_class", None),
            "class_mask": self.class_mask,

            "data_split_indices": self.data_split_indices,
        }

        # Phase 4 A2 protection state. Key is ONLY added for fedta_a2 so the
        # frozen fedta baseline keeps a bit-identical checkpoint schema.
        # (Required so a mid-task resume restores the task-start snapshot
        # instead of recomputing it from the already-drifted state.)
        if self.method == 'fedta_a2':
            state["a2_state"] = self._a2_state_for_checkpoint()

        return state

    def _a2_state_for_checkpoint(self):
        """Serialize A2 protection state; always a dict for fedta_a2,
        never called for the fedta baseline (schema untouched)."""
        if self.method != 'fedta_a2':
            return None
        return {
            "a2_arm": self.a2_arm,
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

        A fedta_a2 resume from a checkpoint without a2_state would silently
        continue as an unprotected A2 (wrong arm semantics, wrong science).
        Same failure class as the Phase-0 missing-manifest false-PASS: the
        state MUST exist and be internally consistent, or we refuse.
        """
        if self.method != 'fedta_a2':
            return
        a2 = state.get("a2_state")
        if not a2:
            raise RuntimeError(
                f"client {self.id}: fedta_a2 resume requires a2_state in the "
                f"checkpoint, got "
                f"{'missing key' if 'a2_state' not in state else 'None/empty'}"
                f" — refusing to silently continue as unprotected A2 "
                f"(fail-closed; re-run from scratch or use a fedta_a2 "
                f"checkpoint)")
        if a2.get("a2_arm") != self.a2_arm:
            raise RuntimeError(
                f"client {self.id}: A2 arm mismatch on resume "
                f"(checkpoint={a2.get('a2_arm')}, current={self.a2_arm})")
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
            if self.a2_arm in ('key', 'both') and (
                    key_snap is None or key_snap.shape[0] != n):
                raise RuntimeError(
                    f"client {self.id}: arm {self.a2_arm} requires a "
                    f"key_snapshot with {n} rows, got "
                    f"{'None' if key_snap is None else key_snap.shape[0]}")
            if self.a2_arm in ('anchor', 'both') and (
                    anchor_snap is None or anchor_snap.shape[0] != n):
                raise RuntimeError(
                    f"client {self.id}: arm {self.a2_arm} requires an "
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
        self.model.load_state_dict(state["tail_anchor_model"])

        # prompts: deepcopy current vit prompt as template, then load state
        prompt_state = state.get("prompts")
        if prompt_state is not None:
            prompt = clone_prompt_template(self.vit)
            prompt.load_state_dict(prompt_state)
            self.prompts = prompt
        else:
            self.prompts = None

        # task-specific heads: template = deepcopy of current Tail_Anchor head
        heads = []
        for h_state in state.get("heads", []):
            if h_state is None:
                heads.append(None)
            else:
                template = deepcopy(self.model.get_head())
                template.load_state_dict(h_state)
                heads.append(template)
        self.heads = heads

        self.global_protos = state.get("global_protos")
        self.local_protos = state.get("local_protos")

        if state.get("current_class") is not None:
            self.current_class = state["current_class"]
        if state.get("class_mask") is not None:
            self.class_mask = state["class_mask"]

        self.data_split_indices = state.get("data_split_indices", {})

        # rebuild all seen-task test loaders + current traindata
        self.rebuild_data_from_indices()

        # Phase 4 A2: restore protection snapshot AFTER task_id is restored,
        # so update_data() does not recompute it mid-task.
        self._a2_load_checkpoint_state(state)

    def rebuild_data_from_indices(self):
        """Rebuild test_loader[0..task_id] and the current traindata from the
        recorded split indices, so evaluate(old_task) works after resume."""
        self.test_loader = []
        for task_id in sorted(self.data_split_indices.keys()):
            base_dataset = self.train_data[task_id]
            split = self.data_split_indices[task_id]
            train_subset = Subset(base_dataset, list(split["train"]))
            test_subset = Subset(base_dataset, list(split["test"]))
            self.test_loader.append(test_subset)
            if task_id == self.task_id:
                self.traindata = train_subset
                self.train_dataset = base_dataset
        print(f'Client {self.id}: restored {len(self.test_loader)} test loader(s) from checkpoint')

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
        optimizer = torch.optim.Adam(self.vit.parameters(), lr=self.lr,weight_decay=1e-03)
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
