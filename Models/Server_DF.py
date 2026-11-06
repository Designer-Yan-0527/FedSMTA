import random
import json
import time
from copy import deepcopy
from pathlib import Path

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
from Models.ctrp import (CTRP_MODES, parse_alpha_grid, hash_prompt_module,
                         hash_tensor, select_alpha)


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
                bool(getattr(args, 'head_refit', False))):
            raise SystemExit(
                "--head_standardize requires --head_refit "
                "(nothing to standardize; "
                "fail-closed, RUNCOMMANDS Sec.23)")
        # A5-v2 §23.6-1 fail-closed：refit_input 只作用于 refit 的
        # 读出头；单独开启 = 无效配置，拒绝启动而非静默忽略
        if str(getattr(args, 'refit_input', 'x1')) != 'x1' and not (
                bool(getattr(args, 'head_refit', False))):
            raise SystemExit(
                "--refit_input feat requires --head_refit "
                "(nothing to refit; "
                "fail-closed, RUNCOMMANDS Sec.23.6-1)")
        # A5-v2 §28 fail-closed：refit_timing 只改 refit 的时点；单独
        # 开启 = 无效配置，拒绝启动而非静默忽略
        if str(getattr(args, 'refit_timing', 'local')) == 'post_broadcast' and not (
                bool(getattr(args, 'head_refit', False))):
            raise SystemExit(
                "--refit_timing post_broadcast requires --head_refit "
                "(nothing to refit; "
                "fail-closed, RUNCOMMANDS Sec.28)")

        # ------------------------------------------------------------------
        # §33 CTRP v1（pre-registered 2026-11-03；GPT 冻结 9 节裁决）。
        # off = 完整 legacy 路径，逐位一致（本块之外不执行任何 CTRP 代码）。
        # ------------------------------------------------------------------
        self._ctrp_mode = str(getattr(args, 'ctrp_mode', 'off') or 'off')
        if self._ctrp_mode not in CTRP_MODES:
            raise SystemExit(
                f"CTRP: unknown ctrp_mode {self._ctrp_mode!r} "
                f"(valid: {CTRP_MODES})")
        if self._ctrp_mode != 'off':
            # 裁决 §1.1：post-broadcast refit 会在 CTRP calibration 之后
            # 改写部署 head，selector 与 deployment semantics 不一致 ->
            # CTRP v1 不支持，fail-closed（off 不受影响）
            if str(getattr(args, 'refit_timing', 'local')) == 'post_broadcast':
                raise SystemExit(
                    "CTRP_V1_UNSUPPORTED_REFIT_TIMING: ctrp_mode != off is "
                    "incompatible with --refit_timing post_broadcast "
                    "(post-broadcast refit rewrites the deployed head AFTER "
                    "CTRP calibration, so the selector would disagree with "
                    "the final deployment semantics). Use "
                    "--refit_timing local or --ctrp_mode off.")
            self._ctrp_alpha_grid = parse_alpha_grid(
                getattr(args, 'ctrp_alpha_grid', None))
            self._ctrp_alpha = getattr(args, 'ctrp_alpha', None)
            self._ctrp_val_frac = float(getattr(args, 'ctrp_val_frac', 0.20))
            if self._ctrp_mode == 'fixed_alpha':
                if self._ctrp_alpha is None:
                    raise SystemExit(
                        "--ctrp_mode fixed_alpha requires --ctrp_alpha "
                        "(the fixed step size, in [0, 1]; fail-closed)")
                if not (0.0 <= float(self._ctrp_alpha) <= 1.0):
                    raise SystemExit(
                        f"--ctrp_alpha must lie in [0, 1], got "
                        f"{self._ctrp_alpha}")
            elif self._ctrp_alpha is not None:
                raise SystemExit(
                    "--ctrp_alpha is only valid with --ctrp_mode "
                    "fixed_alpha (fail-closed: no dual source of truth)")
            if not (0.0 < self._ctrp_val_frac < 1.0):
                raise SystemExit(
                    f"--ctrp_val_frac must lie strictly within (0, 1), "
                    f"got {self._ctrp_val_frac}")
        else:
            self._ctrp_alpha_grid = None
            self._ctrp_alpha = None
            self._ctrp_val_frac = None

        # ------------------------------------------------------------------
        # §34.2 0b：BSAB_PAIR_CAPTURE（user 2026-11-04 APPROVED FOR
        # IMPLEMENTATION；instrumentation-only，BSAB 结构实现仍 FROZEN）。
        # capture run = M2'-matched 冻结：--refit_timing post_broadcast 会让
        # in-run head 变成 M2'' 状态、birth head 反而成为需要额外重构与解释
        # 的中间状态 —— fail-closed 拒绝。诊断构建（run_manager=None）没有
        # run 目录，同样 fail-closed。capture 输出目录 = <run_dir>/bsab_capture
        # （经 args 传给 client，client 不自行推导路径）。
        # ------------------------------------------------------------------
        if bool(getattr(args, 'bsab_pair_capture', False)):
            if str(getattr(args, 'refit_timing', 'local')) == 'post_broadcast':
                raise SystemExit(
                    "BSAB_CAPTURE_UNSUPPORTED_REFIT_TIMING: "
                    "--bsab_pair_capture is only defined for the M2'-matched "
                    "capture run (--refit_timing local; RUNCOMMANDS Sec.34.2). "
                    "--refit_timing post_broadcast makes the in-run deployed "
                    "head the post-broadcast M2'' state, so the frozen birth "
                    "head would need offline reconstruction instead of direct "
                    "pre-broadcast anchoring.")
            if run_manager is None:
                raise SystemExit(
                    "BSAB_PAIR_CAPTURE requires the main-run RunManager "
                    "(a diagnostic build without a run directory has nowhere "
                    "to anchor the capture output; fail-closed)")
            self.args.bsab_capture_dir = str(
                self.run_manager.run_dir / "bsab_capture")

        # anchor 状态机（裁决 §D/§2）：state 0 不校准，
        # P_0_deploy = P_0_cand；t>=1 时 DeltaP 相对 deployed anchor。
        # _ctrp_prev_deployed = 上一 task 的 deployed prompt VALUES（完整
        # pool 张量；prompt_key 不参与——CTRP 不改 key）。
        self._ctrp_prev_deployed = None
        self._ctrp_prev_dominant_slot = None
        self._ctrp_dominant_slot = None
        self._ctrp_selected_alpha = None
        self._ctrp_val_ce_by_alpha = None
        self._ctrp_candidate_prompt_hash = None
        self._ctrp_prev_deployed_prompt_hash = None
        self._ctrp_deployed_prompt_hash = None
        self._ctrp_history = []

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
            _fusion_ran = i % self.global_epoch != self.global_epoch - 1
            if _fusion_ran:
                self.kd_fusion_prompt(self.thisclients, round_i=i)

            self.fed_avg_head(self.thisclients)

            # §33 CTRP v1：任务末 temporal prompt calibration。冻结插入点 =
            # fed_avg_head 之后、broadcast（get_global_proto_and_head）之前，
            # 使任务末 native evaluate / full evaluation / task_XX_end.pth
            # 全部看到 P_t^deploy 而非 P_t^cand。candidate = 原算法正常产生的
            # task-end candidate（不修改任务末轮跳过 fusion 的语义）。
            # ctrp_mode=off = 精确 no-op（legacy 路径逐位一致）。
            if (self._ctrp_mode != 'off'
                    and (i + 1) % self.global_epoch == 0):
                self._ctrp_calibrate_task_end(i)

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
                    output = self.model.forward_features(x, cls_features=cls_features, train=False)['x']
                    outputs = []
                    with torch.no_grad():
                        for localmodel in chosen_clients[1:]:
                            prompt = self.clients[localmodel].prompts
                            self.model.load_prompts(prompt)
                            my = self.model.forward_features(x, cls_features=cls_features, train=False)['x']
                            outputs.append(my)

                    loss = global_distillation_loss(output,outputs)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            self.prompt = my_result
            print('prompt fusion complete')
        return my_result

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
        # A5-v2（RUNCOMMANDS §23）：跨 ON/OFF resume 会静默改变接口语
        # 义与 joint-head 轨迹管理；two-sided 检查保持老 checkpoint 兼容
        "head_refit", "head_standardize", "refit_input", "refit_timing",
        # §33 CTRP v1：任何会改变 CTRP 语义的参数都必须匹配（two-sided：
        # pre-CTRTP 老 checkpoint 不含这些 key，自动跳过保持兼容）
        "ctrp_mode", "ctrp_alpha", "ctrp_alpha_grid", "ctrp_val_frac",
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
        checkpoint = {
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
        # §33 CTRP v1：anchor 状态机等 CTRP 状态收进单一 ctrp_state dict
        #（off 臂不加 key = pre-CTRTP checkpoint schema 逐位不变）。
        # calibration_done：当前 task 的 hook 是否已跑（task 末 checkpoint
        # = True；round 中途 = False）。
        if self._ctrp_mode != 'off':
            checkpoint["ctrp_state"] = {
                "mode": self._ctrp_mode,
                "current_task": self.task_id,
                "calibration_done": bool(
                    (round_id + 1) % self.global_epoch == 0),
                "prev_deployed_prompt": (
                    self._ctrp_prev_deployed.detach().cpu()
                    if self._ctrp_prev_deployed is not None else None),
                "prev_dominant_slot": self._ctrp_prev_dominant_slot,
                "dominant_slot": self._ctrp_dominant_slot,
                "selected_alpha": self._ctrp_selected_alpha,
                "alpha_grid": ([float(a) for a in self._ctrp_alpha_grid]
                               if self._ctrp_alpha_grid else None),
                "val_ce_by_alpha": self._ctrp_val_ce_by_alpha,
                "candidate_prompt_hash": self._ctrp_candidate_prompt_hash,
                "prev_deployed_prompt_hash":
                    self._ctrp_prev_deployed_prompt_hash,
                "deployed_prompt_hash": self._ctrp_deployed_prompt_hash,
                "history": list(self._ctrp_history),
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

        # ---- §33 CTRP v1 resume (frozen cases A/B/C, Sec.2) -------------
        self._ctrp_resume(checkpoint)

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

    # ------------------------------------------------------------------
    # §33 CTRP v1: task-end temporal prompt calibration
    # (RUNCOMMANDS Sec.33, pre-registered 2026-11-03, GPT-frozen verdict)
    # ------------------------------------------------------------------

    # premise gate threshold (Gate Amendment, frozen 2026-11-04):
    # sample-weighted deployed-route violation rate
    #   M_s = sum_b n_b * 1[p_b = s],  p* = argmax_s M_s,
    #   r = 1 - M_{p*} / sum_b n_b   (p_b = forward-returned deployed
    #   prompt_idx, n_b = actual batch rows) <= 1%. Replaces the
    # batch-count statistic (granularity mismatch: a 2-row partial batch
    # weighed as much as a full 8-row batch); routing itself, the 1%
    # tolerance and every other frozen ingredient are UNCHANGED.
    CTRP_PREMISE_VIOLATION_RATE_MAX = 0.01

    # Gate Amendment identifier (frozen 2026-11-04, user-approved): every
    # CTRP record of the formal run carries the amendment identity so any
    # CIFAR/INR result can be attributed to these frozen semantics
    # (threshold=0.01, routing_unchanged, gate_unit=sample_mass).
    CTRP_GATE_AMENDMENT = "CTRP_PREMISE_GATE_V2_SAMPLE_WEIGHTED"
    CTRP_GATE_UNIT = "sample_mass"

    def _ctrp_persist_event(self, record):
        """Append a CTRP event record to the run dir (immediate flush:
        append_jsonl opens/writes/closes). run_manager=None (diagnostics
        / unit tests) -> print only, so the reason is never lost."""
        record = dict(record)
        record["gate_amendment"] = self.CTRP_GATE_AMENDMENT
        record["gate_unit"] = self.CTRP_GATE_UNIT
        record["routing_unchanged"] = True
        if self.run_manager is not None:
            self.run_manager.append_jsonl(
                self.run_manager.metrics_dir / "ctrp_events.jsonl", record)
        print("[CTRP] " + json.dumps(record, default=str))

    def _ctrp_log_transition(self, record):
        record = dict(record)
        record["gate_amendment"] = self.CTRP_GATE_AMENDMENT
        record["gate_unit"] = self.CTRP_GATE_UNIT
        record["routing_unchanged"] = True
        if self.run_manager is not None:
            self.run_manager.append_jsonl(
                self.run_manager.metrics_dir / "ctrp_transitions.jsonl",
                record)
        self._ctrp_history.append(record)
        print("[CTRP] transition " + json.dumps(record, default=str))

    def _ctrp_fail_premise(self, record):
        """Frozen Sec.3: log the premise failure reason to disk FIRST,
        then raise. Never silently fall back to alpha=1."""
        record = dict(record)
        record["event"] = "CTRP_PREMISE_FAIL"
        self._ctrp_persist_event(record)
        raise RuntimeError(
            "CTRP_PREMISE_FAIL: " + str(record.get("reason")))

    def _ctrp_calibrate_task_end(self, round_id):
        """CTRP hook body. Runs at the task-end round only, between
        fed_avg_head and the broadcast loop. Frozen semantics:

          state 0 : no temporal calibration (P_0_deploy = P_0_cand);
                    record the discovered dominant slot s_0* and the
                    anchor = deepcopy of the candidate VALUES.
          t >= 1  : premise gate on the FORWARD-actual route (prompt_idx
                    from Global_Prompt.forward; Gate Amendment frozen
                    2026-11-04: sample-weighted deployed-route violation
                    rate r = 1 - M_p*/n_total <= 1%, with the dominant
                    slot p* itself chosen by sample mass -- the
                    batch-count statistic is kept only as a diagnostic;
                    s_t* == s_{t-1}*). endpoint = matched
                    baseline, premise failures only logged; every other
                    matched arm fails closed. alpha from --ctrp_alpha
                    (fixed_alpha), pooled val/train CE argmin with the
                    Probe-H tie-break (ctrp_val / ctrp_train), or 1
                    (endpoint). Deploy = dominant-row-only calibration
                    P_deploy[s*] = (1-a) P_prev[s*] + a P_cand[s*] with
                    every other slot keeping the candidate value; the
                    prompt_key is NEVER modified (route(alpha=0) ==
                    route(alpha=1) by construction; asserted via key-hash
                    invariance around the whole hook).
        """
        t = round_id // self.global_epoch
        if self.prompt is None:
            raise RuntimeError(
                "CTRP: no server candidate prompt at the task end round "
                f"(round {round_id}) -- cannot calibrate (fail-closed)")
        # resume 边角：restored server prompt 可能在 CPU（load 时 vit 尚
        # 在 CPU；正常首跑时它来自 kd_fusion 的 vit clone，已在设备上，
        # 此处为 no-op）。校准/broadcast 必须与 client vit 同设备。
        vit_device = next(self.clients[0].vit.parameters()).device
        if next(self.prompt.parameters()).device != vit_device:
            self.prompt = self.prompt.to(vit_device)
        key_hash_before = hash_tensor(self.prompt.prompt_key)
        cand_state_hash = hash_prompt_module(self.prompt)
        cand_values = self.prompt.prompt.detach().clone()

        # pass 1: route discovery under the SERVER CANDIDATE prompt state
        # (frozen Sec.4: the sweep's baseline is the server candidate, not
        # the client's stale local prompt; clients load it and the loaded
        # state hash is asserted against the server's). Gate Amendment
        # (frozen 2026-11-04): each batch record carries its deployed slot
        # AND its actual row count n_b; the premise statistic and the
        # dominant-slot choice are both SAMPLE-WEIGHTED:
        #   M_s = sum_b n_b 1[p_b = s];  p* = argmax_s M_s (ties -> the
        #   SMALLER slot id, deterministic);  r = 1 - M_{p*} / sum_b n_b.
        batch_records = []
        per_client_batches = {}
        for c in self.clients:
            recs = c._ctrp_route_slots(self.prompt)
            per_client_batches[c.id] = len(recs)
            batch_records.extend(recs)
        if not batch_records:
            self._ctrp_fail_premise({
                "round": round_id, "task": t, "mode": self._ctrp_mode,
                "reason": "zero current-task validation batches under the "
                          "server candidate state (CTRP_VAL_ROWS_EMPTY)",
            })
        n_total = int(sum(b["n_rows"] for b in batch_records))
        n_batches = int(len(batch_records))
        mass = {}
        for b in batch_records:
            mass[b["slot"]] = mass.get(b["slot"], 0) + int(b["n_rows"])
        # p* by sample mass; an argmax tie resolves to the smaller slot id
        modal = min(mass, key=lambda s: (-mass[s], s))
        dominant_mass = int(mass[modal])
        # INTEGER gate (frozen tau = 0.01): 100 * n_viol <= n_total is
        # mathematically exact at the boundary (8/800 == 1% PASSES),
        # with no dependence on float rounding; the violation rate below
        # is display/diagnostic only
        violation_rows = n_total - dominant_mass
        premise_route_ok = violation_rows * 100 <= n_total
        violation_rate = violation_rows / float(n_total)
        n_nonslot = sum(1 for b in batch_records if b["slot"] != modal)
        # legacy batch-count statistic + tie/partial audit: diagnostic_only,
        # never enters PASS/FAIL (the INR forensic: 1/93 = 0.010753 as a
        # batch fraction vs 2/731 = 0.002736 sample-weighted)
        diagnostics = {
            "nonslot_batch_frac": round(n_nonslot / float(n_batches), 6),
            "nonslot_batches": f"{n_nonslot}/{n_batches}",
            "partial_nonslot_batches": int(sum(
                1 for b in batch_records
                if b["slot"] != modal and b.get("partial"))),
            "tie_batches": int(sum(
                1 for b in batch_records if b.get("tie"))),
            "tie_sample_mass": int(sum(
                int(b["n_rows"]) for b in batch_records if b.get("tie"))),
            "min_vote_margin": (min(
                (int(b["vote_margin_rows"]) for b in batch_records))
                if batch_records else None),
        }

        # ---- state 0: no temporal calibration (frozen) ------------------
        if t == 0:
            self._ctrp_prev_deployed = cand_values.clone()
            self._ctrp_prev_dominant_slot = modal
            self._ctrp_dominant_slot = modal
            self._ctrp_prev_deployed_prompt_hash = hash_tensor(cand_values)
            self._ctrp_deployed_prompt_hash = hash_tensor(cand_values)
            self._ctrp_candidate_prompt_hash = cand_state_hash
            self._ctrp_log_transition({
                "round": round_id, "task": t, "mode": self._ctrp_mode,
                "transition": "state0_init",
                "dominant_slot": modal,
                "dominant_sample_mass": f"{dominant_mass}/{n_total}",
                "deployed_route_violation_rate": round(violation_rate, 6),
                "premise_threshold": self.CTRP_PREMISE_VIOLATION_RATE_MAX,
                "n_val_samples": n_total,
                "n_val_batches": n_batches,
                "diagnostic_only": diagnostics,
                "premise_status": "state0_no_calibration",
                "selector": self._ctrp_mode,
                "split_seed_schema": "a5_stable_seed(seed,client,task)",
                "candidate_prompt_hash": cand_state_hash,
                "prev_deployed_prompt_hash": hash_tensor(cand_values),
                "deployed_prompt_hash": hash_tensor(cand_values),
                "selected_alpha": None,
                "val_ce_by_alpha": None,
            })
            if hash_tensor(self.prompt.prompt_key) != key_hash_before:
                raise RuntimeError(
                    "CTRP: prompt_key changed during the state-0 hook "
                    "(route invariance violated; fail-closed)")
            return

        if self._ctrp_prev_deployed is None:
            # load-time validation makes this unreachable for resumed
            # runs (CTRP_RESUME_ANCHOR_MISSING fires at load); a fresh
            # run always passes through state 0 first
            raise RuntimeError(
                "CTRP: anchor state machine broken (task >= 1 but no "
                "prev_deployed anchor); refusing to guess the anchor "
                "(fail-closed)")

        prev_slot = self._ctrp_prev_dominant_slot
        anchor_values = self._ctrp_prev_deployed.to(cand_values.device)
        anchor_hash = hash_tensor(anchor_values)
        premise_pass = (premise_route_ok and modal == prev_slot)

        base_record = {
            "round": round_id, "task": t, "mode": self._ctrp_mode,
            "transition": f"{t - 1}->{t}",
            "dominant_slot": modal,
            "previous_dominant_slot": prev_slot,
            "dominant_sample_mass": f"{dominant_mass}/{n_total}",
            "deployed_route_violation_rate": round(violation_rate, 6),
            "premise_threshold": self.CTRP_PREMISE_VIOLATION_RATE_MAX,
            "n_val_samples": n_total,
            "n_val_batches": n_batches,
            "dominant_slot_share": round(dominant_mass / float(n_total), 6),
            "diagnostic_only": diagnostics,
            "per_client_val_batches": per_client_batches,
            "selector": self._ctrp_mode,
            "split_seed_schema": "a5_stable_seed(seed,client,task)",
            "candidate_prompt_hash": cand_state_hash,
            "prev_deployed_prompt_hash": anchor_hash,
        }

        # ---- premise gate (Gate Amendment frozen 2026-11-04;
        # fail-closed Sec.3 / Sec.7 semantics unchanged) ----------------
        if not premise_pass:
            reason = (
                f"sample-weighted deployed-route violation rate "
                f"{violation_rate:.6f} > "
                f"{self.CTRP_PREMISE_VIOLATION_RATE_MAX} "
                f"({violation_rows}/{n_total} rows off the dominant slot)"
                if not premise_route_ok
                else f"dominant slot changed across states "
                     f"({prev_slot} -> {modal})")
            record = dict(base_record)
            record["premise_status"] = "fail"
            record["reason"] = reason
            if self._ctrp_mode == 'endpoint':
                # matched-split endpoint control: log route premise
                # diagnostics but NEVER terminate the baseline run
                self._ctrp_persist_event(dict(
                    record, event="CTRP_PREMISE_DIAGNOSTIC"))
            else:
                self._ctrp_fail_premise(record)

        alpha = None
        val_ce_by_alpha = None
        if self._ctrp_mode == 'endpoint':
            alpha = 1.0
        elif self._ctrp_mode == 'fixed_alpha':
            alpha = float(self._ctrp_alpha)
        else:
            # ctrp_val / ctrp_train: pooled sample-weighted current-task CE
            # over the matched split rows, Probe-H selector + tie-break
            rows_key = ('ctrp_val' if self._ctrp_mode == 'ctrp_val'
                        else 'ctrp_core')
            grid = self._ctrp_alpha_grid
            ce_sum = np.zeros(len(grid), dtype=np.float64)
            n_rows_total = 0
            per_client_rows = {}
            for c in self.clients:
                res = c._ctrp_alpha_sweep(
                    self.prompt, anchor_values, modal, grid,
                    rows_key=rows_key)
                ce_sum += res["ce_sum_by_alpha"]
                n_rows_total += res["n_rows"]
                per_client_rows[c.id] = {
                    "n_rows": res["n_rows"],
                    "n_dropped": res["n_dropped"],
                    "n_batches": res["n_batches"],
                }
            if n_rows_total == 0:
                self._ctrp_fail_premise({
                    "round": round_id, "task": t, "mode": self._ctrp_mode,
                    "reason": "zero scored selection rows at a transition "
                              "(CTRP_VAL_ROWS_EMPTY)",
                })
            ce_pool = ce_sum / n_rows_total
            alpha = select_alpha(ce_pool, grid)
            val_ce_by_alpha = [float(v) for v in ce_pool]
            base_record["per_client_sweep"] = per_client_rows
            base_record["alpha_grid"] = [float(a) for a in grid]

        # ---- deploy: dominant-slot-row-only temporal calibration -------
        # P_deploy[s*] = P_prev_deploy[s*] + a (P_cand[s*] - P_prev[s*]),
        # realized in the Probe-H op order; every other slot keeps the
        # candidate value. alpha == 1 -> deployed IS the candidate, the
        # write is skipped (avoids float no-op drift).
        if float(alpha) != 1.0:
            with torch.no_grad():
                row_prev = anchor_values[modal]
                row_cand = cand_values[modal]
                self.prompt.prompt.data[modal] = (
                    (1.0 - float(alpha)) * row_prev
                    + float(alpha) * row_cand)
        deployed_values = self.prompt.prompt.detach().clone()
        deployed_hash = hash_tensor(deployed_values)

        # route invariance: CTRP never touches prompt_key, so
        # route(alpha=0) == route(alpha=1) by construction; assert the
        # key survived the hook (frozen Sec.7 assertion)
        if hash_tensor(self.prompt.prompt_key) != key_hash_before:
            raise RuntimeError(
                "CTRP: prompt_key changed during calibration "
                "(route(alpha=0)==route(alpha=1) violated; fail-closed)")

        # ---- anchor state machine update -------------------------------
        self._ctrp_prev_deployed = deployed_values.clone()
        self._ctrp_prev_dominant_slot = modal
        self._ctrp_dominant_slot = modal
        self._ctrp_selected_alpha = float(alpha)
        self._ctrp_val_ce_by_alpha = val_ce_by_alpha
        self._ctrp_candidate_prompt_hash = cand_state_hash
        self._ctrp_prev_deployed_prompt_hash = anchor_hash
        self._ctrp_deployed_prompt_hash = deployed_hash

        record = dict(base_record)
        record.update({
            # endpoint arm continuing past a logged premise failure
            # keeps running (diagnostic_only); all other arms raised above
            "premise_status": ("pass" if premise_pass
                               else "diagnostic_only"),
            "selected_alpha": float(alpha),
            "val_ce_by_alpha": val_ce_by_alpha,
            "selected_ce": (val_ce_by_alpha[
                self._ctrp_alpha_grid.index(float(alpha))]
                if val_ce_by_alpha is not None else None),
            "ce_at_alpha1": (val_ce_by_alpha[
                self._ctrp_alpha_grid.index(1.0)]
                if val_ce_by_alpha is not None else None),
            "interior_alpha": bool(float(alpha) < 1.0),
            "deployed_prompt_hash": deployed_hash,
        })
        self._ctrp_log_transition(record)

    def _ctrp_resume(self, checkpoint):
        """§33 CTRP v1 resume (frozen cases A/B/C, Sec.2):

          A) ctrp_mode=off：无 ctrp_state = 正常 legacy 加载（本方法
             无事可做；RESUME_MUST_MATCH 已拦截 off <-> matched 切换）。
          B) ctrp_mode!=off 且 checkpoint 早于 state-0 hook 点
             (completed_round < global_epoch - 1)：resume 后仍会执行
             task 0 末轮的 hook，anchor 由该次 hook 正确建立 -> 全新
             初始化状态机，不需要任何猜测。
          C) ctrp_mode!=off、temporal transitions 已开始（state-0 hook
             点已过）但缺可靠 prev_deployed_prompt：fail-closed
             CTRP_RESUME_ANCHOR_MISSING。禁止从当前 server_prompt 猜
             anchor（mid-task 的 server_prompt 可能已是当前 task 的
             fusion/candidate 中间状态）、禁止从 candidate 反推、禁止
             自动退回 endpoint、禁止静默重初始化。
        """
        if self._ctrp_mode == 'off':
            return
        ctrp_state = checkpoint.get("ctrp_state")
        completed_round = int(checkpoint["completed_round"])
        state0_hook_passed = completed_round >= self.global_epoch - 1
        if ctrp_state is None:
            if state0_hook_passed:
                raise RuntimeError(
                    "CTRP_RESUME_ANCHOR_MISSING: resuming a "
                    f"ctrp_mode={self._ctrp_mode!r} run at round "
                    f"{completed_round + 1} (the state-0 calibration "
                    "point has passed, temporal transitions already "
                    "started) but the checkpoint carries no "
                    "ctrp_state.prev_deployed_prompt. Refusing to "
                    "guess the anchor from server_prompt (mid-task "
                    "server_prompt may already be this task's fusion/"
                    "candidate intermediate state), to back-derive it "
                    "from the candidate, to fall back to endpoint, or "
                    "to silently re-initialize (fail-closed; resume "
                    "from a CTRP checkpoint or re-run from task 0).")
            # case B: the state-0 hook still lies ahead of the
            # resumed run; it will set the anchor itself
            print("[CTRP] resume: fresh state machine (the state-0 "
                  "calibration point is still ahead; anchor will be "
                  "set by the task-0 end hook)")
            return
        ckpt_mode = str(ctrp_state.get("mode", self._ctrp_mode))
        if ckpt_mode != self._ctrp_mode:
            raise RuntimeError(
                f"CTRP resume mode mismatch: checkpoint ctrp_state "
                f"mode {ckpt_mode!r} != current "
                f"{self._ctrp_mode!r} (RESUME_MUST_MATCH)")
        prev_anchor = ctrp_state.get("prev_deployed_prompt")
        if prev_anchor is None and state0_hook_passed:
            raise RuntimeError(
                "CTRP_RESUME_ANCHOR_MISSING: the checkpoint's "
                "ctrp_state lacks prev_deployed_prompt although "
                "temporal transitions already started (task "
                f"{checkpoint.get('server_task_id')}); the anchor "
                "cannot be reconstructed (fail-closed).")
        self._ctrp_prev_deployed = (
            prev_anchor.clone() if prev_anchor is not None else None)
        self._ctrp_prev_dominant_slot = ctrp_state.get("prev_dominant_slot")
        self._ctrp_dominant_slot = ctrp_state.get("dominant_slot")
        self._ctrp_selected_alpha = ctrp_state.get("selected_alpha")
        self._ctrp_val_ce_by_alpha = ctrp_state.get("val_ce_by_alpha")
        self._ctrp_candidate_prompt_hash = ctrp_state.get(
            "candidate_prompt_hash")
        self._ctrp_prev_deployed_prompt_hash = ctrp_state.get(
            "prev_deployed_prompt_hash")
        self._ctrp_deployed_prompt_hash = ctrp_state.get(
            "deployed_prompt_hash")
        self._ctrp_history = list(ctrp_state.get("history", []))
        if (self._ctrp_alpha_grid is not None
                and ctrp_state.get("alpha_grid") is not None
                and [float(a) for a in self._ctrp_alpha_grid]
                    != [float(a) for a in ctrp_state["alpha_grid"]]):
            raise RuntimeError(
                "CTRP resume alpha grid mismatch (RESUME_MUST_MATCH "
                "should have caught this; refusing to continue)")
        print(f"[CTRP] resume: ctrp_state restored "
              f"(prev_dominant_slot="
              f"{self._ctrp_prev_dominant_slot}, "
              f"anchor={'present' if prev_anchor is not None else 'none'}, "
              f"history={len(self._ctrp_history)})")





