import random
import time
from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from Models.Client_DF import Client_DF
from Models.classification_head import Chead

from utils import accuracy, global_distillation_loss
from torch.nn import functional as F
from sklearn.metrics.pairwise import cosine_similarity

from runtime_utils import (module_state_or_none, clone_prompt_template,
                           rng_state_dict, restore_rng_state,
                           atomic_save_checkpoint, load_checkpoint_file)


def rebuild_global_head(global_head_state, vit_head, nb_classes):
    """Rebuild the training-time global_head structure matching the
    checkpoint's saved state (the saved structure depends on whether
    fed_avg_head() ran during training):

      * official behavior: fed_avg_head() replaced global_head with a
        deepcopy of clients[0].vit.head (a plain Linear) every round ->
        checkpoint stores Linear keys {weight, bias};
      * skip_fed_avg_head runs (HS-N arm): global_head keeps the __init__
        Chead structure -> checkpoint stores head.0.* / head.3.* keys.

    __init__ always creates a Chead, so the structure must be rebuilt
    from the checkpoint before load_state_dict. Anything unrecognized
    fails loud rather than silently misloading."""
    gh_keys = set(global_head_state.keys())
    if gh_keys == {"weight", "bias"}:
        return deepcopy(vit_head)
    if gh_keys == set(Chead(nb_classes).state_dict().keys()):
        # exact key-set match against a template Chead: a truncated or
        # foreign head.* state must not slip through this branch and
        # misload later
        return Chead(nb_classes)
    raise RuntimeError(
        "unrecognized global_head checkpoint structure: "
        f"{sorted(gh_keys)}")


class Server_DF(object):

    def __init__(self,id,origin_model,model_name,client_num,task_num,subset,class_mask,lr,global_epoch,local_epoch,batch_size,device,method,threshold,surrogate_data,test_data,args,model,run_manager=None):
        self.id = id
        self.model_name = model_name
        self.origin_model = origin_model
        self.model = model

        self.client_num=client_num
        self.task_num = task_num
        self.clients =[]

        # Every client has a subset and mask
        self.client_data = subset
        self.class_mask = class_mask


        self.surrogate_data = surrogate_data
        self.lr = lr
        self.batch_size = batch_size
        self.global_epoch = global_epoch
        self.local_epoch = local_epoch
        self.device = device
        self.method = method
        self.threshold = threshold
        self.test_data = test_data
        self.args = args

        self.task_id = -1

        self.existing_class =set()
        self.global_head = Chead(args.nb_classes)

        self.global_protos = None
        self.temp_protos =None

        self.fix_keys = []

        # Phase 0: round-boundary checkpoint / resume state
        self.run_manager = run_manager
        # Phase 0: observer instrumentation switch (tests/test_observer_
        # invariance.py runs with it OFF to prove the extra evaluations do
        # not perturb the training trajectory). Official FedTA behavior =
        # no instrumentation; ON adds RNG-guarded observers only.
        self.instrumentation = not bool(
            getattr(args, 'no_instrumentation', False))
        self.start_round = 0
        self.prompt = None
        self.accuracy_records = (
            run_manager.load_accuracy_records()
            if run_manager is not None else [])

        # A5-v2 F3 fail-closed（RUNCOMMANDS §23.1）：standardize 只有在
        # refit/consolidation 存在时才有作用对象；单独开启 = 无效配置，
        # 拒绝启动而非静默忽略
        if bool(getattr(args, 'head_standardize', False)) and not (
                bool(getattr(args, 'head_refit', False))
                or bool(getattr(args, 'interface_consolidate', False))):
            raise SystemExit(
                "--head_standardize requires --head_refit or "
                "--interface_consolidate (nothing to standardize; "
                "fail-closed, RUNCOMMANDS Sec.23)")
        # A5-v2 §23.6-1 fail-closed：refit_input 只作用于 refit/consolidation
        # 的读出头；单独开启 = 无效配置，拒绝启动而非静默忽略
        if str(getattr(args, 'refit_input', 'x1')) != 'x1' and not (
                bool(getattr(args, 'head_refit', False))
                or bool(getattr(args, 'interface_consolidate', False))):
            raise SystemExit(
                "--refit_input feat requires --head_refit or "
                "--interface_consolidate (nothing to refit; "
                "fail-closed, RUNCOMMANDS Sec.23.6-1)")

        # PSC fail-closed（RUNCOMMANDS Sec.25）：未验证组合拒绝启动，
        # 不允许静默跑出不可判读的臂
        if bool(getattr(args, 'psc', False)):
            if self.method == 'fedta_a2':
                raise SystemExit(
                    "--psc is mutually exclusive with --method fedta_a2 "
                    "(A2 is a judged-dead arm; the A2+PSC combination was "
                    "never validated; fail-closed, RUNCOMMANDS Sec.25)")
            if bool(getattr(args, 'route_fix', False)):
                raise SystemExit(
                    "--psc is mutually exclusive with --route_fix "
                    "(soft-routing weights change the merge/neighborhood "
                    "semantics; the combination was never validated; "
                    "fail-closed, RUNCOMMANDS Sec.25)")
            if int(getattr(args, 'psc_max_merge', 2)) < 0:
                raise SystemExit(
                    "--psc_max_merge must be >= 0 (got "
                    f"{args.psc_max_merge}; fail-closed, RUNCOMMANDS Sec.25)")

        # PSC server-side consolidation state（--psc OFF 时全部闲置，
        # 官方轨迹逐位不变）
        self.psc_on = bool(getattr(args, 'psc', False))
        self._psc_core_slots = []     # sorted row ids designated CORE
        self._psc_pair_streak = {}    # "i|j" -> consecutive similar fusion rounds
        self._psc_merged_pairs = []   # "i|j" already merged (never re-merge)
    def init_client(self):
        print('Initialize clients')
        for i in range(self.client_num):    #
            # id,original_model,vit,task_per_global_epoch,subset,local_epoch,batch_size,lr,device,method
            self.clients.append(Client_DF(i,self.origin_model,self.model_name,self.global_epoch,self.client_data[i],
                                          self.local_epoch,self.batch_size,self.lr,self.device,self.method,self.class_mask[i],self.args,self.model
                                       ))
        print("Initialization completes")

    def fedavg(self, sample_nums):
        training_num = 0
        for idx in range(len(sample_nums)):
            samplenum = sample_nums[idx]
            training_num += samplenum
        sample_num = sample_nums[0]
        averaged_params = self.clients[0].model.cpu().state_dict()
        for k in averaged_params.keys():
            for i in range(0, self.client_num):
                local_sample_number = sample_nums[i]
                local_model_params = self.clients[i].model.cpu().state_dict()
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
        return averaged_params




    def train_clients(self):
        used = []
        start_round = self.start_round

        # Phase 0: --max_rounds lets the resume regression test stop a run
        # early; it never changes what happens inside a round.
        total_rounds = self.task_num * self.global_epoch
        max_rounds = int(getattr(self.args, 'max_rounds', 0) or 0)
        end_round = total_rounds if max_rounds <= 0 else min(total_rounds, max_rounds)

        # --route_audit: read-only forensics logger (default OFF). Attached
        # here so every client can log A/B timepoints; the server logs the
        # cross-branch alignment and the fusion (C) timepoints.
        if getattr(self.args, 'route_audit', False) and self.run_manager is not None:
            from diagnostics.route_audit import RouteAuditLogger
            self.route_audit_logger = RouteAuditLogger(
                self.run_manager.run_dir, self.args.seed)
            for c in self.clients:
                c.route_audit_logger = self.route_audit_logger
        else:
            self.route_audit_logger = None

        for i in range(start_round, end_round):
            self.thisclients = [i for i in range(self.client_num)]

            # if len(used) ==0:
            #     used = self.thisclients
            # else:
            #     used.extend(self.thisclients)
            #     used = list(set(used))
            # if len(used) !=20 and i >0:
            #     not_used = [j for j in range(self.client_num) if j not in used]
            #     chosen = random.choice(not_used)
            #     for j in range(len(self.thisclients)):
            #         if j in used:
            #             self.thisclients[j] = chosen
            #             used.append(chosen)
            #             break
            # print(used)


            for j in range(self.client_num):
                self.clients[j].update_data(round=i, args=self.args)


            if self.task_id != i//self.global_epoch:
                self.task_id = i//self.global_epoch


            print(f"--------round {i},task number {i//self.global_epoch}-----------")
            for j in self.thisclients:
                self.clients[j].train(round=i,args=self.args)

            # --route_audit: cross-branch key alignment after all local
            # trainings (same-slot cosine / cross-Gram / principal angles),
            # read from each client's OWN snapshot. Separates "each branch
            # collapsed internally" from "branches healthy but drifted into
            # different subspaces" (distillation targets incoherent).
            if self.route_audit_logger is not None:
                _keys = {j: self.clients[j].prompts.prompt_key
                         for j in self.thisclients
                         if self.clients[j].prompts is not None}
                if _keys:
                    self.route_audit_logger.log_cross_branch(i, _keys)

            # Phase 0: local-phase accuracies (after local training, before
            # aggregation) -> metrics/train_log.csv
            # (skipped entirely under --no_instrumentation)
            if self.run_manager is not None and self.instrumentation:
                self._log_local_phase(i)

            # FedTA
            # if i ==15 or i==16:
            #     for j in self.thisclients:
            #         print(f'{j} client')
            #         for q in self.clients[j].local_protos.keys():
            #             print(f'{q} is {self.clients[j].local_protos[q]}')

            # BUGFIX: threshold was hardcoded to 0.25 and --threshold was
            # silently ignored; now uses args (config default keeps 0.25 so
            # behavior is unchanged)
            self.choose_best_proto_greedy_similarity_fixed_key((i+2)%self.global_epoch==0,threshold=self.threshold,round=i)

            # BUGFIX: hardcoded `!= 4` assumed global_epoch == 5; with other
            # values the last round of a task wrongly skipped SIKF fusion
            # --route_audit timepoint C: the server key is the deepcopy of
            # client[0]'s branch fused by kd_fusion_prompt below; pre/post
            # hashes settle empirically whether the fusion changed it at all.
            _fusion_ran = i % self.global_epoch != self.global_epoch - 1
            if self.route_audit_logger is not None:
                _src = self.clients[self.thisclients[0]].prompts
                if _src is not None:
                    self.route_audit_logger.log_key_snapshot(
                        i, "pre_fusion", self.thisclients[0],
                        _src.prompt_key)
            if _fusion_ran:
                self.kd_fusion_prompt(self.thisclients, round_i=i)
                # PSC: consolidation only runs on fusion rounds (self.prompt
                # is fresh); merges commit at task-boundary fusion rounds
                # (rounds i%ge==ge-2, the LAST fusion of a task), so the
                # merged broadcast is exactly what the next task starts
                # from and the task-start freeze snapshots are coherent.
                if self.psc_on:
                    self.consolidate_prompts(i)
            # PSC: broadcast the core-row set every round (idempotent; the
            # clients consume it at task start via _psc_init_task_protection)
            if self.psc_on:
                _psc_core = set(self._psc_core_slots)
                for j in range(self.client_num):
                    self.clients[j]._psc_core_slots = _psc_core
            if self.route_audit_logger is not None and self.prompt is not None:
                self.route_audit_logger.log_key_snapshot(
                    i, "post_fusion", None, self.prompt.prompt_key,
                    note="fusion_ran" if _fusion_ran
                    else "fusion_skipped_this_round_prompt_stale")

            # HS-N fix switch (default OFF = official behavior, bit-identical):
            # skip fed_avg_head entirely. The paper (arXiv 2412.18355 Fig. 3 /
            # Sec. 4.1) defines no head aggregation, the averaged global_head
            # is only consumed by dead code (evaluate_on_global_testset, zero
            # call sites), and because all clients share one ViT object the
            # "average" is a mathematical identity — its only live effect is
            # the x2^(N-1)/round in-place aliasing corruption of vit.head
            # (INR_PHASE1_HEAD_DIVERGENCE.md Sec. 4.5).
            if not getattr(self.args, 'skip_fed_avg_head', False):
                self.fed_avg_head(self.thisclients)

            print('Server aggregation Compelte, results are ()：')
            # Phase 0: collect the (already existing) post-broadcast
            # evaluation results for metrics/train_log.csv
            server_rows = []
            for j in range(self.client_num):
                # FedTA
                if j in self.thisclients:
                    result = self.clients[j].get_global_proto_and_head(self.global_protos,self.global_head,self.prompt,i)
                    if result:
                        for t, acc in result.items():
                            server_rows.append({
                                "round": i, "task": i // self.global_epoch,
                                "client": j, "phase": "server",
                                "test_task": t, "accuracy": acc})
                    print('-------')

                else:
                    self.clients[j].get_global_proto_and_head_no_test(self.global_protos, self.global_head, self.prompt, i)
            # writing the structured CSV is Phase-0 instrumentation; the
            # broadcast/evaluation loop above is original FedTA and always
            # runs (--no_instrumentation must equal FedTA + checkpoint only)
            if self.run_manager is not None and self.instrumentation:
                self.run_manager.append_train_log_rows(server_rows)


            # ---------------- Phase 0: evaluation / metrics / checkpoint ----------------
            # strict per-round order:
            # clients train -> BGPS -> SIKF -> fed_avg_head -> broadcast
            # -> (optional) full-task evaluation -> metrics -> checkpoint -> flush
            is_task_end = (i + 1) % self.global_epoch == 0
            eval_all_every = getattr(self.args, 'eval_all_every', 0)
            do_eval = is_task_end or (
                eval_all_every and eval_all_every > 0
                and (i + 1) % eval_all_every == 0)
            # full seen-task evaluation is Phase-0 instrumentation; it is
            # RNG-guarded and skipped entirely under --no_instrumentation
            if do_eval and self.instrumentation:
                self.run_full_evaluation(i)

            # round_metrics.jsonl is Phase-0 instrumentation; skipped under
            # --no_instrumentation (checkpoint saving below always runs)
            if self.instrumentation:
                self.append_round_metrics(i)
            self.maybe_save_checkpoint(i)

            if self.run_manager is not None:
                self.run_manager.flush_log()


        print("All Process completes")

    def fuse_protos(self):
        global_protos = dict()
        temp = dict()
        for i in self.clients:
            for label in i.local_protos.keys():
                if label in temp:
                    temp[label].append(i.local_protos[label])
                else:
                    temp[label] = [i.local_protos[label]]
        for label in temp.keys():
            global_protos[label] = np.mean(temp[label], axis=0)
            temp[label] = np.vstack(temp[label])
        self.global_protos = global_protos

        self.temp_protos = temp





    def choose_best_proto_greedy_similarity_fixed_key(self,is_fix=False,threshold=0.2,round=15):

        if self.global_protos != None:
            global_protos = self.global_protos  # global protos from last round
        else:
            global_protos = dict()


        temp = dict()
        this_round = dict()
        # {1:[x,x,x], 2:[q,q,q]}


        for i in self.thisclients:
            for label in self.clients[i].local_protos.keys():
                if label not in self.fix_keys:
                    if label in temp:
                        temp[label].append(self.clients[i].local_protos[label])
                    else:
                        temp[label] = [self.clients[i].local_protos[label]]

                    if label in this_round:
                        if not np.isnan(self.clients[i].local_protos[label]).any():
                            this_round[label].append(self.clients[i].local_protos[label])
                    else:
                        if not np.isnan(self.clients[i].local_protos[label]).any():
                            this_round[label] = [self.clients[i].local_protos[label]]



        if len(this_round.keys()) !=0 and self.fix_keys !=[]:
            keys = list(this_round.keys())
            keys.extend(self.fix_keys)
        elif len(this_round.keys())==0  and self.fix_keys != []:

            keys = self.fix_keys
        else:
            keys = list(this_round.keys())


        matrix = None
        first_shape = None
        for key, value in global_protos.items():
            if first_shape is None:
                first_shape = value.shape
                break
        different_shapes = {}
        print(first_shape)
        for key, value in global_protos.items():
            if value.shape != first_shape:
                different_shapes[key] = value.shape
            if len(value.shape) == 3:
                global_protos[key] = global_protos[key].squeeze()
        if different_shapes:
            print("Some values have different shapes:")
            for key, shape in different_shapes.items():
                print(f"Key: {key}, Shape: {shape}")
        else:
            print("All values have the same shape.")


        num = []
        for i in range(len(keys)):
            if keys[i] not in self.fix_keys:
                if matrix is None:
                    matrix = np.array(this_round[keys[i]])
                else:
                    matrix = np.concatenate([matrix, this_round[keys[i]]])

                if keys[i] in global_protos.keys():

                    matrix = np.concatenate([matrix, global_protos[keys[i]].unsqueeze(0)])
                    num.append(len(this_round[keys[i]]) + 1)
                else:
                    num.append(len(this_round[keys[i]]))
            else:
                if matrix is None:
                    matrix = np.array(self.global_protos[keys[i]].unsqueeze(0))
                else:
                    matrix = np.concatenate([matrix, self.global_protos[keys[i]].unsqueeze(0)])
                num.append(1)

        # if round ==15:
        #     contains_nan_rows = np.isnan(matrix).any(axis=1)
        #     print(f"Rows containing NaN: {contains_nan_rows}")
        #     print(matrix[contains_nan_rows])
        #     if self.global_protos:
        #         for q in self.global_protos.keys():
        #             print(f'{q} global proto is {self.global_protos[q]}')
        #     for q in this_round.keys():
        #         print(f'{q} proto is {this_round[q]}')


        # matrix = self.l2_normalize(torch.Tensor(matrix).to(self.device),dim=1)


        # adjecent matrix
        adj_matrix = torch.tensor(cosine_similarity(matrix))
        print(adj_matrix.shape)
        matrix = torch.Tensor(matrix).to(self.device)

        # set the distance of the same class' protos to 1
        low_bound = 0
        for j in num:
            high_bound = low_bound + j
            for h in range(low_bound, high_bound):
                for k in range(low_bound, high_bound):
                    adj_matrix[h][k] = 1.0
            low_bound = high_bound


        # print(adj_matrix)
        remain_keys = set(temp.keys())


        low_bound = 0
        map = {}
        for j in range(len(num)):
            if keys[j] not in self.fix_keys:
                high_bound = low_bound + num[j]
                min = 999
                chose = None
                for h in range(low_bound, high_bound):
                    aver_simi = torch.mean(adj_matrix[h])
                    if aver_simi < min:
                        min = aver_simi
                        chose = h
                if min <=threshold:
                    self.fix_keys.append(keys[j])
                map[keys[j]] = min
                global_protos[keys[j]] = matrix[chose].cpu()
                low_bound = high_bound

            else:
                high_bound = low_bound + num[j]
                aver_simi = torch.mean(adj_matrix[low_bound])
                map[keys[j]] = aver_simi
                global_protos[keys[j]] = self.global_protos[keys[j]]
                low_bound = high_bound

        # print(map)

        # print(global_protos.keys())
        self.global_protos = global_protos
        if is_fix:
            if self.fix_keys !=[]:
                self.fix_keys.extend(list(global_protos.keys()))
                self.fix_keys = list(set(self.fix_keys))
            else:
                self.fix_keys = list(global_protos.keys())


        print(len(self.fix_keys))

        first_shape = None
        for key, value in self.global_protos.items():
            if first_shape is None:
                first_shape = value.shape
                break
        different_shapes = {}
        for key, value in self.global_protos.items():
            if value.shape != first_shape:
                different_shapes[key] = value.shape
            if len(value.shape) == 3:
                self.global_protos[key] = self.global_protos[key].squeeze()

        if different_shapes:
            print("Some values have different shapes:")
            for key, shape in different_shapes.items():
                print(f"Key: {key}, Shape: {shape}")
        else:
            print("All values have the same shape.")

    def choose_best_proto_greedy_similarity_fixed_key_UPDATE(self,is_fix=False,threshold=0.25):

        if self.global_protos != None:
            global_protos = self.global_protos  # global protos from last round
        else:
            global_protos = dict()


        this_round = dict()
        # {1:[x,x,x], 2:[q,q,q]}


        temp = dict()


        for i in self.thisclients:
            for label in self.clients[i].local_protos.keys():
                if label not in self.fix_keys:
                    if label in temp:
                        temp[label].append(self.clients[i].local_protos[label])
                    else:
                        temp[label] = [self.clients[i].local_protos[label]]


                    if label in this_round:
                        this_round[label].append(self.clients[i].local_protos[label])
                    else:
                        this_round[label] = [self.clients[i].local_protos[label]]


        # for label in self.global_protos.keys():
        #     if label in self.fix_keys:
        #         temp[label] = self.global_protos[label]
        #         this_round[label] = self.global_protos[label]



        if len(this_round.keys()) !=0 and self.fix_keys !=[]:
            keys = list(this_round.keys())
            keys.extend(self.fix_keys)

        elif len(this_round.keys())==0  and self.fix_keys != []:

            keys = self.fix_keys
        else:
            keys = list(this_round.keys())



        first_shape = None
        for key, value in global_protos.items():
            if first_shape is None:
                first_shape = value.shape
                break
        different_shapes = {}
        print(first_shape)
        for key, value in global_protos.items():
            if value.shape != first_shape:
                different_shapes[key] = value.shape
            if len(value.shape) == 3:
                global_protos[key] = global_protos[key].squeeze()
        if different_shapes:
            print("Some values have different shapes:")
            for key, shape in different_shapes.items():
                print(f"Key: {key}, Shape: {shape}")
        else:
            print("All values have the same shape.")



        num = []
        for i in range(len(keys)):

            if keys[i] not in self.fix_keys:

                if matrix is None:
                    matrix = np.array(this_round[keys[i]])
                else:
                    matrix = np.concatenate([matrix, this_round[keys[i]]])

                if keys[i] in global_protos.keys():
                    matrix = np.concatenate([matrix, global_protos[keys[i]].unsqueeze(0)])
                    num.append(len(this_round[keys[i]]) + 1)
                else:
                    num.append(len(this_round[keys[i]]))



            else:
                if matrix is None:
                    matrix = np.array(self.global_protos[keys[i]].unsqueeze(0))
                else:
                    matrix = np.concatenate([matrix, self.global_protos[keys[i]].unsqueeze(0)])
                num.append(1)

        matrix =self.l2_normalize(torch.Tensor(matrix).to(self.device),dim=1)

        # adjecent matrix
        adj_matrix = torch.matmul(matrix,matrix.t())

        # set the distance of the same class' protos to 1
        low_bound = 0
        for j in num:
            high_bound = low_bound + j
            for h in range(low_bound, high_bound):
                for k in range(low_bound, high_bound):
                    adj_matrix[h][k] = 1.0
            low_bound = high_bound
        # print(adj_matrix)

        remain_keys = set(temp.keys())

        print(adj_matrix)

        low_bound = 0
        map = {}
        for j in range(len(num)):
            if keys[j] not in self.fix_keys:
                high_bound = low_bound + num[j]
                min = 999
                chose = None
                for h in range(low_bound, high_bound):
                    aver_simi = torch.mean(adj_matrix[h])
                    if aver_simi < min:
                        min = aver_simi
                        chose = h
                if min <=threshold:
                    self.fix_keys.append(keys[j])
                map[keys[j]] = min
                global_protos[keys[j]] = matrix[chose].cpu()
                low_bound = high_bound
            else:
                high_bound = low_bound + num[j]
                aver_simi = torch.mean(adj_matrix[low_bound])
                map[keys[j]] = aver_simi
                global_protos[keys[j]] = self.global_protos[keys[j]]
                low_bound = high_bound

        # print(map)

        # print(global_protos.keys())
        self.global_protos = global_protos
        if is_fix:
            if self.fix_keys !=[]:
                self.fix_keys.extend(list(global_protos.keys()))
                self.fix_keys = list(set(self.fix_keys))
            else:
                self.fix_keys = list(global_protos.keys())

        print(self.fix_keys)
        print(len(self.fix_keys))


        first_shape = None
        for key, value in self.global_protos.items():
            if first_shape is None:
                first_shape = value.shape
                break
        different_shapes = {}
        for key, value in self.global_protos.items():
            if value.shape != first_shape:
                different_shapes[key] = value.shape
            if len(value.shape) == 3:
                self.global_protos[key] = self.global_protos[key].squeeze()

        if different_shapes:
            print("Some values have different shapes:")
            for key, shape in different_shapes.items():
                print(f"Key: {key}, Shape: {shape}")
        else:
            print("All values have the same shape.")


    def kd_fusion_prompt(self,chosen_clients,round_i=None):
        my_result = None
        if len(chosen_clients)==1:
            # BUGFIX: clients have no .global_prompt attribute -> AttributeError
            my_result= deepcopy(self.clients[chosen_clients[0]].prompts)
            # BUGFIX: this branch never assigned self.prompt, leaving it None
            # for the broadcast step
            self.prompt = my_result
        else:
            classes = set()
            for i in chosen_clients:
                temp = set(self.clients[i].current_class)
                classes = set.union(classes,temp)

            self.surrogate_data.getTrainData(list(classes))


            my_result = deepcopy(self.clients[chosen_clients[0]].prompts)
            test_loader = DataLoader(self.surrogate_data, batch_size=self.batch_size, shuffle=True, num_workers=2)
            self.model.to(self.device)

            optimizer = torch.optim.Adam(my_result.parameters(), lr=self.lr, weight_decay=1e-03)
            for h in tqdm(range(10)):
                for iteration, (index,x,y) in enumerate(test_loader):
                    x = Variable(x, requires_grad=True).to(self.device, non_blocking=True)
                    y = y.long().to(self.device)
                    my_result.to(self.device)
                    with torch.no_grad():
                        if self.origin_model is not None:
                            output = self.origin_model(x)
                            cls_features = output['pre_logits']
                        else:
                            cls_features = None

                    self.model.load_prompts(my_result)
                    # --route_audit: capture the model-side STUDENT key NOW:
                    # load_prompts() deepcopies my_result into self.model, so
                    # the forward runs through this throwaway parameter while
                    # Adam steps my_result (whose .grad stays None -- the
                    # suspected structural no-op). The teacher loop below
                    # replaces self.model.prompt, so the reference must be
                    # taken before it.
                    _audit_model_key = (
                        self.model.get_prompts().prompt_key
                        if getattr(self, 'route_audit_logger', None) is not None
                        else None)
                    output = self.model.forward_features(x, cls_features=cls_features, train=False)['x']
                    outputs = []
                    with torch.no_grad():
                        for localmodel in chosen_clients[1:]:
                            prompt = self.clients[localmodel].prompts
                            self.model.load_prompts(prompt)
                            my = self.model.forward_features(x, cls_features=cls_features, train=False)['x']
                            outputs.append(my)

                    loss = global_distillation_loss(output,outputs)
                    # --route_audit: first fusion minibatch of the round.
                    # Captures per-teacher distill losses, teacher-output
                    # pairwise consistency, and the distillation gradient
                    # w.r.t. BOTH key objects:
                    #   * model_key   = the throwaway deepcopy the student
                    #     forward actually ran through (captured above), and
                    #   * optimizer_key = my_result.prompt_key (what the
                    #     Adam above actually steps).
                    # If optimizer_key grads are None while model_key grads
                    # are finite, the SIKF fusion is EMPIRICALLY a no-op.
                    if (getattr(self, 'route_audit_logger', None) is not None
                            and h == 0 and iteration == 0):
                        with torch.no_grad():
                            _per_teacher = [
                                float(global_distillation_loss(output, [o]))
                                for o in outputs]
                        self.route_audit_logger.log_fusion_audit(
                            round_i, iteration, loss,
                            _audit_model_key,
                            my_result.prompt_key,
                            output, outputs,
                            list(chosen_clients[1:]), _per_teacher)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            self.prompt = my_result
            print('prompt fusion complete')
        return my_result

    # ------------------------------------------------------------------
    # PSC (--psc, RUNCOMMANDS Sec.25): server-side prompt consolidation.
    # The fused broadcast prompt's rows are merged into shared CORE
    # directions when they are functionally redundant, so later tasks
    # reuse the common directions (core rows keep training) while the
    # old-task nuance stays locked in the residual rows (frozen at the
    # clients via the A2-style post-step restore).
    # ------------------------------------------------------------------

    @staticmethod
    def _psc_l2_rows(key):
        """Global_Prompt.l2_normalize semantics, row-wise (the routing
        normalizes both the query and the key rows)."""
        sq = torch.sum(key ** 2, dim=1, keepdim=True)
        return key * torch.rsqrt(
            torch.maximum(sq, torch.tensor(1e-12, device=key.device)))

    def _psc_seen_classes(self):
        """Union of class masks of all clients over all SEEN tasks
        (t <= current task_id) — the surrogate universe for the PSC
        neighborhood measurement. Old-task rows must be measured on
        old-task queries, so this is NOT restricted to the current task
        (unlike the kd_fusion loader, which only loads current-task
        classes)."""
        classes = set()
        for mask_per_client in self.class_mask:
            for t in range(self.task_id + 1):
                classes |= set(mask_per_client[t])
        return sorted(classes)

    def _psc_surrogate_routing(self):
        """One RNG-neutral surrogate pass under the CURRENT fused prompt:
        per-sample top-1 prompt routing (l2-normalized cls features vs
        l2-normalized prompt_key rows, per-sample top-1 = the D0-style
        support convention) + per-row neighborhood centroids of the
        routing queries. Read-only w.r.t. the prompt.

        RNG-neutral: the official kd_fusion loader legitimately consumes
        the RNG stream (shuffle=True); this measurement pass must NOT
        shift it (four-stream snapshot/restore, sequential loader,
        no workers)."""
        if self.prompt is None:
            return None
        prompt = self.prompt
        P = prompt.prompt_key.shape[0]
        rng_state = rng_state_dict()
        orig_was_training = self.origin_model.training
        try:
            self.origin_model.eval()
            self.origin_model.to(self.device)
            self.surrogate_data.getTrainData(self._psc_seen_classes())
            loader = DataLoader(self.surrogate_data,
                                batch_size=self.batch_size,
                                shuffle=False, num_workers=0)
            feats = []
            for _index, x, _y in loader:
                x = x.to(self.device, non_blocking=True)
                with torch.no_grad():
                    cf = self.origin_model(x)['pre_logits']
                feats.append(cf.to(self.device))
            if not feats:
                return None
            cf = torch.cat(feats, dim=0)
        finally:
            self.origin_model.train(orig_was_training)
            restore_rng_state(rng_state)

        key_norm = self._psc_l2_rows(prompt.prompt_key.detach())
        x_norm = self._psc_l2_rows(cf)
        similarity = torch.matmul(x_norm, key_norm.t())
        top1 = torch.argmax(similarity, dim=1)
        counts = torch.zeros(P, dtype=torch.long, device=cf.device)
        counts.scatter_add_(0, top1, torch.ones_like(top1))
        centroids = torch.zeros(P, cf.shape[1], device=cf.device)
        centroids.index_add_(0, top1, cf)
        nonempty = counts > 0
        centroids[nonempty] = (
            centroids[nonempty]
            / counts[nonempty].unsqueeze(1).float())
        return {
            "x_norm": x_norm,
            "top1": top1,
            "counts": counts,
            "centroids": centroids,
            "n_samples": int(top1.numel()),
        }

    def consolidate_prompts(self, round_i):
        """PSC consolidation step (called right after kd_fusion on fusion
        rounds, RUNCOMMANDS Sec.25).

        Every fusion round: measure the row-pair functional similarity —
        cosine between routing-neighborhood centroids on the surrogate
        seen-class queries — and maintain the consecutive-round streaks.

        Task-boundary fusion rounds (round_i % global_epoch == ge-2, the
        LAST fusion of a task, so the merged broadcast is exactly what the
        next task starts from): pairs whose streak >= --psc_streak AND
        whose current similarity >= --psc_tau are merge candidates.
        Merging a pair (i, j), i < j: row i becomes a CORE row carrying
        the common direction (prompt_key[i] and prompt[i] := average of
        the two rows) while the residual row j keeps its EXACT value
        (merge-only, no deletion: old-task traffic routed to j is
        untouched; the difference direction stays in the residual row).
        Each merge is gated by a top-1-routing drift check on the same
        surrogate queries: the fraction of samples whose top-1 row changed
        in a way that is NOT a pure i<->j swap must stay <= --psc_drift,
        otherwise the merge is rolled back from the pre-merge snapshot.
        At most --psc_max_merge pairs commit per round; pairs sharing a
        row with an already-committed merge are skipped (a later rollback
        must never undo an earlier committed merge)."""
        is_boundary = (round_i % self.global_epoch
                       == self.global_epoch - 2)
        meas = self._psc_surrogate_routing()
        if meas is None:
            print(f'[PSC] round {round_i}: no surrogate queries, '
                  f'consolidation skipped')
            return
        prompt = self.prompt
        P = prompt.prompt_key.shape[0]
        tau = float(getattr(self.args, 'psc_tau', 0.95))
        streak_k = int(getattr(self.args, 'psc_streak', 3))
        max_merge = int(getattr(self.args, 'psc_max_merge', 2))
        drift_gate = float(getattr(self.args, 'psc_drift', 0.05))
        # singleton neighborhoods carry no usable moment information
        MIN_NEIGHBORHOOD = 2

        core = set(self._psc_core_slots)
        merged = set(self._psc_merged_pairs)
        counts = meas["counts"]
        # rows with measurable neighborhoods; core rows are never merge
        # candidates (the shared basis must not be re-averaged)
        measurable = [r for r in range(P)
                      if r not in core and int(counts[r]) >= MIN_NEIGHBORHOOD]

        # ---- streak update (every fusion round) ----
        cent = meas["centroids"]
        cent_norm = cent / (cent.norm(dim=1, keepdim=True) + 1e-12)
        gram = cent_norm @ cent_norm.t()
        n_similar = 0
        measured_keys = set()
        for a in range(len(measurable)):
            i = measurable[a]
            for b in range(a + 1, len(measurable)):
                j = measurable[b]
                key = f"{i}|{j}"
                if key in merged:
                    continue
                measured_keys.add(key)
                if float(gram[i, j]) >= tau:
                    self._psc_pair_streak[key] = \
                        self._psc_pair_streak.get(key, 0) + 1
                    n_similar += 1
                else:
                    self._psc_pair_streak.pop(key, None)
        # "consecutive" semantics: a pair that could not be measured this
        # round (row became core / neighborhood shrank below the minimum)
        # breaks the streak — drop stale entries so an old count can never
        # resurrect a candidate after a gap
        for stale in [k for k in self._psc_pair_streak
                      if k not in measured_keys]:
            self._psc_pair_streak.pop(stale, None)

        if not is_boundary:
            print(f'[PSC] round {round_i}: {len(measurable)} measurable '
                  f'rows, {n_similar} pairs over tau, '
                  f'{len(self._psc_pair_streak)} active streaks '
                  f'(merge decision only at task-boundary fusion rounds)')
            return

        # ---- merge decision (task-boundary fusion round only) ----
        candidates = []
        for key, streak in self._psc_pair_streak.items():
            if streak < streak_k:
                continue
            i_s, j_s = key.split('|')
            sim = float(gram[int(i_s), int(j_s)])
            if sim >= tau:
                candidates.append((sim, key))
        candidates.sort(reverse=True)
        print(f'[PSC] round {round_i} (task-boundary fusion): '
              f'{len(candidates)} merge candidates '
              f'(streak>={streak_k}, sim>={tau}), cap {max_merge}')
        if not candidates or max_merge <= 0:
            return

        # pre-merge snapshot (rollback target)
        snap_key = prompt.prompt_key.detach().clone()
        snap_prompt = prompt.prompt.detach().clone()

        x_norm = meas["x_norm"]
        committed = 0
        used_rows = set()
        for sim, key in candidates:
            if committed >= max_merge:
                break
            i, j = (int(v) for v in key.split('|'))
            if i in used_rows or j in used_rows:
                continue
            # pre-merge top-1 for THIS merge (earlier committed merges
            # this round are part of the accepted state)
            key_norm = self._psc_l2_rows(prompt.prompt_key.detach())
            top1_pre = torch.argmax(x_norm @ key_norm.t(), dim=1)
            # apply merge: common direction -> core row i; residual row j
            # keeps its value (merge-only, no deletion)
            prompt.prompt_key.data[i] = 0.5 * (
                prompt.prompt_key.data[i] + prompt.prompt_key.data[j])
            prompt.prompt.data[i] = 0.5 * (
                prompt.prompt.data[i] + prompt.prompt.data[j])
            key_norm = self._psc_l2_rows(prompt.prompt_key.detach())
            top1_post = torch.argmax(x_norm @ key_norm.t(), dim=1)
            # drift = top-1 changes that are NOT pure i<->j swaps (leaving
            # the pair breaks old-task routing; entering the pair steals
            # traffic from other rows)
            changed = top1_pre != top1_post
            in_pair_pre = (top1_pre == i) | (top1_pre == j)
            in_pair_post = (top1_post == i) | (top1_post == j)
            drift = float((changed & ~(in_pair_pre & in_pair_post))
                          .float().mean())
            if drift > drift_gate:
                prompt.prompt_key.data[i] = snap_key[i]
                prompt.prompt.data[i] = snap_prompt[i]
                self._psc_pair_streak.pop(key, None)
                print(f'[PSC] round {round_i}: merge {key} ROLLED BACK '
                      f'(drift {drift:.4f} > {drift_gate})')
                continue
            self._psc_core_slots = sorted(set(self._psc_core_slots) | {i})
            self._psc_merged_pairs.append(key)
            self._psc_pair_streak.pop(key, None)
            used_rows.update((i, j))
            committed += 1
            print(f'[PSC] round {round_i}: merged pair {key} '
                  f'(sim {sim:.4f}, drift {drift:.4f}) -> core row {i}; '
                  f'core rows now {len(self._psc_core_slots)}')

    def start(self):
        self.init_client()

        # checkpoint must be loaded AFTER init_client: client models have to
        # exist before load_state_dict can be called on them.
        if (self.args.resume and self.run_manager is not None
                and self.run_manager.checkpoint_path):
            self.load_checkpoint(self.run_manager.checkpoint_path)

        self.train_clients()
        # self.test_surro()


    # ------------------------------------------------------------------
    # Phase 0: round-boundary checkpoint / resume
    # ------------------------------------------------------------------

    # configs that must match the checkpoint, otherwise training silently
    # diverges from the original run
    RESUME_MUST_MATCH = [
        "data_name", "method", "model", "model_name",
        "client_num", "task_num", "global_epoch", "local_epoch",
        "lr", "batch_size", "nb_classes", "private_class_num",
        "seed", "threshold", "surrogate_num",
        # HS-N: resuming across ON/OFF would silently diverge (the
        # x2^(N-1)/round corruption changes the phase-1 trajectory);
        # old checkpoints without this key are unaffected (two-sided
        # membership check below)
        "skip_fed_avg_head",
        # route_fix (A2-R): resuming an ON checkpoint without
        # --route_fix would silently fall back to hard routing and
        # contaminate the ablation arm; two-sided check keeps old
        # checkpoints without the key compatible.
        "route_fix", "route_tau",
        # A5-v2（RUNCOMMANDS §23）：跨 ON/OFF resume 会静默改变接口语
        # 义与 joint-head 轨迹管理；two-sided 检查保持老 checkpoint 兼容
        "head_dropout", "head_refit", "head_standardize",
        "interface_consolidate", "refit_input",
        # PSC（RUNCOMMANDS §25）：跨 ON/OFF 或跨参数 resume 会静默改变
        # 巩固/冻结语义；two-sided 检查保持老 checkpoint 兼容
        "psc", "psc_tau", "psc_streak", "psc_max_merge", "psc_drift",
    ]

    def validate_checkpoint_args(self, ckpt_args):
        if not ckpt_args:
            return
        current = vars(self.args)
        mismatches = []
        for key in self.RESUME_MUST_MATCH:
            if key in ckpt_args and key in current:
                if str(ckpt_args[key]) != str(current[key]):
                    mismatches.append(
                        f"{key} checkpoint={ckpt_args[key]}, current={current[key]}")
        if mismatches:
            raise ValueError(
                "Checkpoint mismatch:\n" + "\n".join(mismatches))

    def build_outer_split_manifest(self):
        """The true federated outer partition manifest: for every client and
        task, the raw sample indices (into the ORIGINAL train set, BEFORE the
        per-task 70/30 random_split) plus the class mask. data_split_state
        only records the inner 70/30 indices relative to these subsets, so
        without this manifest a checkpoint is not self-contained w.r.t. the
        data protocol."""
        manifest = {}
        for i in range(self.client_num):
            per_task = {}
            for t in range(self.task_num):
                subset = self.client_data[i][t]
                per_task[str(t)] = {
                    "raw_indices": [int(x) for x in subset.indices],
                    "class_mask": [int(c) for c in self.class_mask[i][t]],
                }
            manifest[str(i)] = per_task
        return manifest

    def build_checkpoint(self, round_id):
        return {
            "version": 1,

            "completed_round": round_id,
            "server_task_id": self.task_id,

            "server_model": module_state_or_none(self.model),

            "global_head": module_state_or_none(self.global_head),

            "global_protos": self.global_protos,
            "temp_protos": self.temp_protos,
            "fix_keys": list(self.fix_keys),
            "existing_class": sorted(self.existing_class),

            "server_prompt": module_state_or_none(self.prompt),

            "clients": [c.state_dict_for_checkpoint() for c in self.clients],

            "rng_state": rng_state_dict(),

            "data_split_state": {c.id: c.data_split_indices
                                 for c in self.clients},

            # true outer federated partition (raw sample indices + class
            # masks), verified against the regenerated partition on resume
            "outer_split_manifest": self.build_outer_split_manifest(),

            "args": vars(self.args),
        }

        # PSC（--psc，RUNCOMMANDS §25）：server 端巩固状态（core 行集合 /
        # 连续轮 streak / 已合并对）。schema 仅 ON 臂新增 key（frozen 基线
        # checkpoint 逐位不变）。streak 必须存：mid-task resume 后跨边界轮
        # 的合并决策依赖 task 内已积累的连续计数。
        if self.psc_on:
            checkpoint["psc_state"] = {
                "core_slots": list(self._psc_core_slots),
                "pair_streak": dict(self._psc_pair_streak),
                "merged_pairs": list(self._psc_merged_pairs),
            }
        return checkpoint

    def maybe_save_checkpoint(self, round_id):
        if self.run_manager is None:
            return
        save_every = getattr(self.args, 'save_every', 1)
        keep_last = getattr(self.args, 'keep_last', 0)

        is_task_end = (round_id + 1) % self.global_epoch == 0
        should_save = save_every > 0 and (round_id + 1) % save_every == 0
        if not (should_save or is_task_end):
            return

        checkpoint = self.build_checkpoint(round_id)

        paths = []
        if should_save:
            paths.append(self.run_manager.round_ckpt_path(round_id))
        if is_task_end:
            # task boundary checkpoint is permanent (never removed by keep_last)
            paths.append(self.run_manager.task_ckpt_path(round_id // self.global_epoch))
        paths.append(self.run_manager.latest_ckpt_path())

        atomic_save_checkpoint(checkpoint, *[str(p) for p in paths])
        print(f"[checkpoint] completed_round={round_id} saved -> "
              f"{[p.name for p in paths]}")

        if keep_last and keep_last > 0:
            self.run_manager.cleanup_round_checkpoints(keep_last)

    def load_checkpoint(self, ckpt_path):
        print(f"Loading checkpoint: {ckpt_path}")
        checkpoint = load_checkpoint_file(ckpt_path)

        self.validate_checkpoint_args(checkpoint.get("args", {}))

        # Phase 0: the federated outer partition is regenerated from the
        # seed; verify it is bit-identical to the manifest stored in the
        # checkpoint (raw sample indices + class masks per client-task).
        # Gate 0B seal requirement: a checkpoint WITHOUT a manifest cannot
        # be verified and is rejected — silently accepting old checkpoints
        # would let the resume-regression test pass vacuously.
        if "outer_split_manifest" not in checkpoint:
            raise RuntimeError(
                "Checkpoint missing outer_split_manifest: the federated "
                "outer partition cannot be verified. Phase-0 freeze "
                "requires every checkpoint to carry the manifest (raw "
                "sample indices + class masks per client-task).")
        ckpt_manifest = checkpoint["outer_split_manifest"]
        if not isinstance(ckpt_manifest, dict) or not ckpt_manifest:
            raise RuntimeError(
                "Invalid or empty outer_split_manifest in checkpoint.")
        current_manifest = self.build_outer_split_manifest()
        if ckpt_manifest != current_manifest:
            raise RuntimeError(
                "Federated outer split manifest mismatch: the partition "
                "regenerated from this seed does NOT match the one "
                "recorded in the checkpoint. Refusing to resume with a "
                "different data partition (check seed / client_num / "
                "task_num / private_class_num / data_name / data_path).")
        print("[checkpoint] outer split manifest verified "
              "(bit-identical raw indices + class masks)")

        self.task_id = checkpoint["server_task_id"]
        self.existing_class = set(checkpoint.get("existing_class", []))

        if checkpoint.get("global_head") is not None:
            # rebuild the training-time global_head structure matching the
            # saved state (see rebuild_global_head), then load
            self.global_head = rebuild_global_head(
                checkpoint["global_head"],
                self.clients[0].vit.head,
                self.args.nb_classes)
            self.global_head.load_state_dict(checkpoint["global_head"])
        if checkpoint.get("server_model") is not None:
            self.model.load_state_dict(checkpoint["server_model"])

        self.global_protos = checkpoint.get("global_protos")
        self.temp_protos = checkpoint.get("temp_protos")
        self.fix_keys = checkpoint.get("fix_keys", [])

        # server prompt: template = deepcopy of vit prompt, then load state
        prompt_state = checkpoint.get("server_prompt")
        if prompt_state is not None:
            prompt = clone_prompt_template(self.model)
            prompt.load_state_dict(prompt_state)
            self.prompt = prompt
        else:
            self.prompt = None

        # client states (models / heads / prompts / protos / data splits)
        for client_state in checkpoint.get("clients", []):
            cid = client_state["id"]
            self.clients[cid].load_checkpoint_state(client_state)

        # PSC server-side consolidation state, FAIL-CLOSED (Sec.25): a --psc
        # resume from a checkpoint without psc_state would silently restart
        # consolidation from an empty core/streak state — refuse instead.
        if self.psc_on:
            psc = checkpoint.get("psc_state")
            if not psc:
                raise RuntimeError(
                    "--psc resume requires the server psc_state in the "
                    "checkpoint, got "
                    f"{'missing key' if 'psc_state' not in checkpoint else 'None/empty'}"
                    " — refusing to silently restart consolidation from an "
                    "empty state (fail-closed; re-run from scratch or use "
                    "a --psc checkpoint)")
            self._psc_core_slots = list(psc.get("core_slots", []))
            self._psc_pair_streak = dict(psc.get("pair_streak", {}))
            self._psc_merged_pairs = list(psc.get("merged_pairs", []))
            print(f"[PSC] resume: core rows {len(self._psc_core_slots)}, "
                  f"active streaks {len(self._psc_pair_streak)}, merged "
                  f"pairs {len(self._psc_merged_pairs)}")

        # RNG state must be restored LAST: model creation and split rebuild
        # above must not consume the restored random stream.
        restore_rng_state(checkpoint["rng_state"])

        self.start_round = checkpoint["completed_round"] + 1
        print(f"RESUME_FROM = {ckpt_path}")
        print(f"COMPLETED_ROUND = {checkpoint['completed_round']}")
        print(f"NEXT_ROUND = {self.start_round}")
        print(f"Current task: {self.task_id}")
        print("Resume successful.")

    # ------------------------------------------------------------------
    # Phase 0: full seen-task evaluation + metrics
    # ------------------------------------------------------------------

    def _log_local_phase(self, round_id):
        """Phase 0: local-phase accuracy (after local training, before
        aggregation) for metrics/train_log.csv.

        RNG-snapshot guarded: this extra evaluation is NOT part of the
        original FedTA loop, so it must not perturb the training random
        stream (unlike the post-broadcast evaluation, which is original
        behavior)."""
        task = round_id // self.global_epoch
        rows = []
        rng_snapshot = rng_state_dict()
        try:
            for j in self.thisclients:
                client = self.clients[j]
                for t in sorted({0, client.task_id}):
                    if (t < 0 or t > client.task_id
                            or t >= len(client.heads)
                            or client.heads[t] is None
                            or t >= len(client.test_loader)):
                        continue
                    acc = client.evaluate(t, self.args.nb_classes)
                    rows.append({"round": round_id, "task": task, "client": j,
                                 "phase": "local", "test_task": t,
                                 "accuracy": acc})
        finally:
            restore_rng_state(rng_snapshot)
        self.run_manager.append_train_log_rows(rows)

    def run_full_evaluation(self, round_id):
        """Evaluate every client on every seen task (task0 -> current) and
        persist the full accuracy matrix.

        Phase 0: the evaluation must not perturb the training RNG stream,
        so the RNG state is snapshotted before evaluating and restored
        afterwards (also on exception). The next round then sees exactly
        the random state it would have seen without the evaluation, and
        checkpointed RNG states stay evaluation-free."""
        after_task = round_id // self.global_epoch
        print(f"===== Full seen-task evaluation after task {after_task} "
              f"(round {round_id}) =====")
        rng_snapshot = rng_state_dict()
        try:
            all_client_results = {}
            margin_rows = []
            for client in self.clients:
                results = client.evaluate_all_seen_tasks(self.args.nb_classes)
                all_client_results[client.id] = results
                for test_task, acc in results.items():
                    self.accuracy_records.append({
                        "client_id": client.id,
                        "after_task": after_task,
                        "test_task": test_task,
                        "accuracy": acc,
                    })
                # Phase 1: per-task margin stats recorded by evaluate()
                for test_task, stats in getattr(
                        client, "last_margin_stats_by_task", {}).items():
                    if not stats:
                        continue
                    margin_rows.append({
                        "client_id": client.id,
                        "after_task": after_task,
                        "test_task": test_task,
                        "accuracy": results.get(test_task),
                        **stats,
                    })
        finally:
            restore_rng_state(rng_snapshot)
        if self.run_manager is not None:
            self.run_manager.write_accuracy_csvs(
                self.accuracy_records, self.task_num)
            self.run_manager.append_margin_rows(margin_rows)
        return all_client_results

    def append_round_metrics(self, round_id):
        if self.run_manager is None:
            return
        entry = {
            "round": round_id,
            "task": round_id // self.global_epoch,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "global_proto_count": len(self.global_protos) if self.global_protos else 0,
            "fixed_proto_count": len(self.fix_keys) if self.fix_keys else 0,
        }
        self.run_manager.append_jsonl(
            self.run_manager.metrics_dir / "round_metrics.jsonl", entry)



    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm


    def fed_avg_prompt(self, chosen_clients):
        prompts = []
        for i in chosen_clients:
            prompts.append(self.clients[i].prompts)

        # key 和 prompt都要平均

        result_prompt = deepcopy(prompts[0].state_dict())

        for k in result_prompt.keys():
            for i in range(len(prompts)):
                local_model_params = prompts[i].state_dict()
                if i == 0:
                    result_prompt[k] = local_model_params[k]
                else:
                    result_prompt[k] += local_model_params[k]

            result_prompt[k] = result_prompt[k] / len(chosen_clients)

        self.prompt = deepcopy(self.clients[0].prompts)
        self.prompt.load_state_dict(result_prompt)
        return self.prompt



    def fed_avg_head(self, chosen_clients):
        heads = []
        for i in chosen_clients:
            heads.append(self.clients[i].vit.head)

        # key 和 prompt都要平均

        result_head = deepcopy(heads[0].state_dict())

        for k in result_head.keys():
            for i in range(len(heads)):
                local_model_params = heads[i].state_dict()
                if i == 0:
                    result_head[k] = local_model_params[k]
                else:
                    result_head[k] += local_model_params[k]

            result_head[k] = result_head[k] / len(chosen_clients)

        self.global_head = deepcopy(self.clients[0].vit.head)
        self.global_head.load_state_dict(result_head)





