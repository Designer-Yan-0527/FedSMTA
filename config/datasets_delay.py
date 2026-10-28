import argparse

def get_args_parser(subparsers):
    # both dash and underscore spellings are accepted (the regression test
    # scripts historically used the underscore form)
    subparsers.add_argument('--batch-size', '--batch_size', dest='batch_size',
                            default=16, type=int,
                            help='Batch size per device (SIKF distillation loader; '
                                 'local training stays at the official hardcoded 16)')
    subparsers.add_argument('--epochs', default=5, type=int)

    # Model parameters
    subparsers.add_argument('--model', default='vit_base_patch16_224', type=str, metavar='MODEL', help='Name of model to train')
    subparsers.add_argument('--input-size', default=224, type=int, help='images input size')
    subparsers.add_argument('--pretrained', default=True, help='Load pretrained model or not')
    subparsers.add_argument('--drop', type=float, default=0.0, metavar='PCT', help='Dropout rate (default: 0.)')
    subparsers.add_argument('--drop-path', type=float, default=0.0, metavar='PCT', help='Drop path rate (default: 0.)')

    # Optimizer parameters
    subparsers.add_argument('--opt', default='adam', type=str, metavar='OPTIMIZER', help='Optimizer (default: "adam"')
    subparsers.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON', help='Optimizer Epsilon (default: 1e-8)')
    subparsers.add_argument('--opt-betas', default=(0.9, 0.999), type=float, nargs='+', metavar='BETA', help='Optimizer Betas (default: (0.9, 0.999), use opt default)')
    subparsers.add_argument('--clip-grad', type=float, default=1.0, metavar='NORM',  help='Clip gradient norm (default: None, no clipping)')
    subparsers.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    subparsers.add_argument('--weight-decay', type=float, default=0.0, help='weight decay (default: 0.0)')
    subparsers.add_argument('--reinit_optimizer', type=bool, default=True, help='reinit optimizer (default: True)')

    # Learning rate schedule parameters
    subparsers.add_argument('--sched', default='constant', type=str, metavar='SCHEDULER', help='LR scheduler (default: "constant"')
    subparsers.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.03)')
    subparsers.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct', help='learning rate noise on/off epoch percentages')
    subparsers.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT', help='learning rate noise limit percent (default: 0.67)')
    subparsers.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV', help='learning rate noise std-dev (default: 1.0)')
    subparsers.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR', help='warmup learning rate (default: 1e-6)')
    subparsers.add_argument('--min-lr', type=float, default=1e-5, metavar='LR', help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    subparsers.add_argument('--decay-epochs', type=float, default=30, metavar='N', help='epoch interval to decay LR')
    subparsers.add_argument('--warmup-epochs', type=int, default=5, metavar='N', help='epochs to warmup LR, if scheduler supports')
    subparsers.add_argument('--cooldown-epochs', type=int, default=10, metavar='N', help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    subparsers.add_argument('--patience-epochs', type=int, default=10, metavar='N', help='patience epochs for Plateau LR scheduler (default: 10')
    subparsers.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE', help='LR decay rate (default: 0.1)')
    subparsers.add_argument('--unscale_lr', type=bool, default=True, help='scaling lr by batch size (default: True)')

    # Augmentation parameters
    subparsers.add_argument('--color-jitter', type=float, default=None, metavar='PCT', help='Color jitter factor (default: 0.3)')
    subparsers.add_argument('--aa', type=str, default=None, metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    subparsers.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    subparsers.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # * Random Erase params
    subparsers.add_argument('--reprob', type=float, default=0.0, metavar='PCT', help='Random erase prob (default: 0.25)')
    subparsers.add_argument('--remode', type=str, default='pixel', help='Random erase mode (default: "pixel")')
    subparsers.add_argument('--recount', type=int, default=1, help='Random erase count (default: 1)')

    # Data parameters
    # both dash and underscore spellings are accepted (the regression test
    # scripts historically used the underscore form)
    subparsers.add_argument('--data-path', '--data_path', dest='data_path',
                            default='./local_datasets', type=str,
                            help='dataset root path')
    subparsers.add_argument('--shuffle', default=False, help='shuffle the data order')
    subparsers.add_argument('--output_dir', default='output/', help='path where to save, empty for no saving')
    subparsers.add_argument('--device', default='cuda', help='device to use for training / testing')
    subparsers.add_argument('--seed', default=42, type=int)
    subparsers.add_argument('--eval', action='store_true', help='Perform evaluation only')
    subparsers.add_argument('--num_workers', default=2, type=int)
    subparsers.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    subparsers.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    subparsers.set_defaults(pin_mem=True)



    # Continual learning parameters

    subparsers.add_argument('--train_mask', default=True, type=bool, help='if using the class mask at training')
    subparsers.add_argument('--task_inc', default=False, type=bool, help='if doing task incremental')


    # pool_size
    subparsers.add_argument('--size', default=100, type=int,)
    subparsers.add_argument('--length', default=10,type=int, )
    subparsers.add_argument('--top_k', default=1, type=int, )
    subparsers.add_argument('--initializer', default='uniform', type=str,)
    subparsers.add_argument('--prompt_key', default=True, type=bool,)
    subparsers.add_argument('--prompt_key_init', default='uniform', type=str)
    subparsers.add_argument('--use_prompt_mask', default=False, type=bool)
    subparsers.add_argument('--shared_prompt_pool', default=False, type=bool)
    subparsers.add_argument('--shared_prompt_key', default=False, type=bool)
    subparsers.add_argument('--batchwise_prompt', default=True, type=bool)
    subparsers.add_argument('--embedding_key', default='cls', type=str)
    subparsers.add_argument('--predefined_key', default='', type=str)
    subparsers.add_argument('--pull_constraint', default=True)
    subparsers.add_argument('--pull_constraint_coeff', default=0.1, type=float)

    # ViT parameters
    subparsers.add_argument('--global_pool', default='token', choices=['token', 'avg'], type=str, help='type of global pooling for final sequence')
    subparsers.add_argument('--head_type', default='prompt', choices=['token', 'gap', 'prompt', 'token+prompt'], type=str, help='input type of classification head')
    subparsers.add_argument('--freeze', default=['blocks', 'patch_embed', 'cls_token', 'norm', 'pos_embed'], nargs='*', type=list, help='freeze part in backbone model')

    # Misc parameters
    subparsers.add_argument('--print_freq', type=int, default=10, help = 'The frequency of printing')

    # Ours
    subparsers.add_argument('--method', type=str, default='fedta', choices=['fedta', 'fedta_a2'],
                            help='fedta = frozen Phase-0 baseline (read-only); '
                                 'fedta_a2 = Phase 4 A2 routing-interface protection arms')
    subparsers.add_argument('--a2_arm', type=str, default='anchor', choices=['key', 'anchor', 'both'],
                            help='A2 protection arm (only used with --method fedta_a2): '
                                 'key = freeze protected key rows; anchor = freeze protected '
                                 'anchor rows; both = freeze both. Implemented as post-step '
                                 'restore (= freeze + weight-decay exemption), leaving the '
                                 'official FedTA optimizer untouched.')
    subparsers.add_argument('--skip_fed_avg_head', action='store_true',
                            help='HS-N fix (orthogonal to --method, default OFF = official '
                                 'behavior): skip the fed_avg_head call entirely. The FedTA '
                                 'paper (arXiv 2412.18355, Fig. 3 / Sec. 4.1) defines no head '
                                 'aggregation; fed_avg_head\'s only live effect is the '
                                 'x2^(N-1)/round in-place aliasing corruption of the shared '
                                 'vit.head (INR_PHASE1_HEAD_DIVERGENCE.md Sec. 4.5).')
    subparsers.add_argument('--route_fix', action='store_true',
                            help='A2-R routing fix (default OFF = official behavior, '
                                 'bit-identical): R1 = orthogonal-spherical prompt_key init '
                                 '(independent RNG stream, seeded by --seed) + R2 = '
                                 'differentiable soft routing w = softmax(sim/tau) over the '
                                 'whole pool with token count length*top_k preserved; train '
                                 'and infer use the same soft weighting. Probe verdict '
                                 '(diagnostics/probe_route_fix.py): keyH revived 5/5 '
                                 'clients (+3~4 bits) with accuracy held. Orthogonal to '
                                 '--method / --a2_arm / --skip_fed_avg_head.')
    subparsers.add_argument('--route_tau', default=0.1, type=float,
                            help='R2 soft routing temperature: w = softmax(similarity/tau). '
                                 'Only used with --route_fix (probe-validated default 0.1).')
    subparsers.add_argument('--route_detach_ce_key_grad', action='store_true',
                            help='A2-R ablation arm B (2026-10-28, default OFF = '
                                 'bit-identical). Only takes effect with --route_fix: '
                                 'the similarity used to build the soft routing '
                                 'weights w is recomputed from prompt_norm.detach(), '
                                 'so the CE loss can no longer reach prompt_key '
                                 'through the softmax coupling (grad to the sample '
                                 'features is kept). Forward VALUES are bitwise '
                                 'unchanged; pull keeps using the original '
                                 'grad-carrying similarity (official-equivalent, '
                                 'selected slots only). Isolates whether the CE '
                                 'all-pool soft-routing gradient is the round-0 '
                                 'key-rank collapse driver. See RUNCOMMANDS.md Sec.20.4.')
    subparsers.add_argument('--route_selected_ce_grad', action='store_true',
                            help='A2-R repair arm E1 (2026-10-28, default OFF '
                                 '= bit-identical). Only takes effect with '
                                 '--route_fix and mutually exclusive with '
                                 '--route_detach_ce_key_grad. Forward is '
                                 'bitwise unchanged: route_sim = sim_no_key '
                                 '+ mask*(similarity - sim_no_key) where '
                                 'sim_no_key uses prompt_norm.detach() and '
                                 'mask is the per-sample detached top-k '
                                 '(deliberately NOT batchwise majority). '
                                 'Backward: CE gradient to prompt_key only '
                                 'through the selected slots (sparse but '
                                 'per-sample dispersed); CE gradient to '
                                 'sample features fully kept; pull keeps '
                                 'official semantics. Fixes the dominant '
                                 'collapse driver (B/C/D verdict) without '
                                 'the B-arm key-norm starvation side effect. '
                                 'See RUNCOMMANDS.md Sec.20.5.')
    subparsers.add_argument('--route_key_no_wd', action='store_true',
                            help='A2-R repair arm E2 (2026-10-28 GPT ruling, '
                                 'default OFF = bit-identical). Strictly '
                                 'single-variable on top of E1: ONLY the '
                                 'prompt_key weight decay 1e-3 -> 0 via '
                                 'optimizer parameter groups (tau, R1 init, '
                                 'selected-slot CE backward, pull, '
                                 'prompt/value wd, optimizer/LR/epochs/data/'
                                 'seed all unchanged). Requires --route_fix '
                                 'AND --route_selected_ce_grad (E2 = E1 + '
                                 'key-only no-wd; fail-closed otherwise). '
                                 'Rationale: E1 showed 97-98 percent of keys '
                                 'starved to |K|<1e-3 by wd, and the '
                                 'l2_normalize '
                                 'Jacobian (~1/|K|) amplifies direction '
                                 'rotation of shrunk keys, coupling Gate 1 '
                                 '(rank) and Gate 2 (norm health). See '
                                 'RUNCOMMANDS.md Sec.20.5b.')
    subparsers.add_argument('--disable_pull', action='store_true',
                            help='A2-R ablation arm C (2026-10-28, default OFF = '
                                 'bit-identical). Removes the -0.1*reduce_sim pull '
                                 'term from the active fedta Client_DF.train() loss '
                                 '(CE only); forward computation of reduce_sim is '
                                 'untouched (route_audit still records what pull '
                                 'WOULD contribute, read-only). Other loss terms, '
                                 'optimizer, data, RNG and seeds unchanged. Tests '
                                 'whether the pull term alone (official-equivalent '
                                 'semantics) drives the key-rank collapse. '
                                 'See RUNCOMMANDS.md Sec.20.4.')
    subparsers.add_argument('--route_audit', action='store_true',
                            help='A2-R collapse forensics instrumentation '
                                 '(default OFF = official path bit-identical). '
                                 'ON adds read-only diagnostics only (never '
                                 'changes loss/optimizer/data/RNG): key '
                                 'snapshots at three timepoints (branch '
                                 'init/start, per-client post-local, '
                                 'pre/post SIKF fusion), first-minibatch '
                                 'CE/pull gradient audit per client, and '
                                 'fusion-batch distillation gradient audit '
                                 'vs BOTH the model key and the optimizer '
                                 'key (empirically settles whether '
                                 'kd_fusion is a structural no-op). '
                                 'Output: <run_dir>/route_audit/*.jsonl+npz. '
                                 'See RUNCOMMANDS.md Sec.20.3.')
    subparsers.add_argument('--e_prompt_layer_idx', default=[2, 3, 4], type=int, nargs="+",
                            help='the layer index of the E-Prompt')
    # A5-v2 三因素归因消融 + 巩固重对齐（2026-10-28，预注册 RUNCOMMANDS §23）
    subparsers.add_argument('--head_dropout', default=0.5, type=float,
                            help='A5-v2 F1: train-side Dropout p of the phase-2 '
                                 'jointly-trained Chead (default 0.5 = official, '
                                 'bit-identical). Distinct from the CLOSED Track-E '
                                 'eval-side dropout bug (corrected protocol). '
                                 'p=0.0 is the attribution-ablation arm; it '
                                 'changes phase-2 forward RNG consumption '
                                 '(expected, arm-internal trajectory).')
    subparsers.add_argument('--head_refit', action='store_true',
                            help='A5-v2 F2 (default OFF = official, bit-identical): '
                                 'post-phase-2 readout-only refit of the current '
                                 'task head every round. Feature cache = the '
                                 'task train split under the CURRENT interface '
                                 'state (no_grad, shuffle=False); retrain a '
                                 'Chead-shaped readout (masked CE, Adam 1e-3/wd '
                                 '1e-4/100 epochs/batch 64, deterministic local '
                                 'seed); replace heads[task_id] storage. RNG-neutral '
                                 '(four-stream snapshot/restore): representation-side '
                                 'trajectory (prompt/key/anchor/joint head) stays '
                                 'bit-identical to the official run; the joint-head '
                                 'trajectory is preserved separately (_a5_joint_head) '
                                 'and restored before the next phase-2. See '
                                 'RUNCOMMANDS.md Sec.23.')
    subparsers.add_argument('--head_standardize', action='store_true',
                            help='A5-v2 F3 (default OFF): standardize the refit '
                                 'input with train-split mean/std (StandardizedChead, '
                                 'mu/sd stored as buffers). Fail-closed: requires '
                                 '--head_refit or --interface_consolidate.')
    subparsers.add_argument('--interface_consolidate', action='store_true',
                            help='A5-v2 component 2 (default OFF = official): at '
                                 'every task-boundary round, refit heads[t] for all '
                                 'OLD tasks t < task_id under the CURRENT interface '
                                 'state (readout-only, same refit recipe, data = the '
                                 'old-task train split via data_split_indices; the '
                                 'test split is never touched). RNG-neutral like '
                                 '--head_refit; the current-task refit belongs to '
                                 '--head_refit (orthogonal switches). Absorbs the '
                                 'interface-drift headroom (RUNCOMMANDS Sec.22.2-4).')
    subparsers.add_argument('--client_num', default=5, type=int,
                            help='the num of client')
    subparsers.add_argument('--task_num', default=5, type=int,
                            help='the num of tasks')
    subparsers.add_argument('--private_class_num', default=15, type=int,
                            help='the num of private class per client')
    subparsers.add_argument('--surrogate_num', default=20, type=int,
                            help='the num of surrogate samples per class')
    subparsers.add_argument('--global_epoch', default=5, type=int,
                            help='global rounds per task')
    subparsers.add_argument('--local_epoch', default=30, type=int,
                            help='local epochs per round')
    subparsers.add_argument('--threshold', default=0.25, type=float,
                            help='BGPS fix threshold (was hardcoded to 0.25 before)')
    subparsers.add_argument('--data_name', default='cifar100', choices=['cifar100', 'ImageNet-R'], type=str,
                            help="dataset to train on (exact match, case sensitive)")
    subparsers.add_argument('--model_name', default='Tail_Anchor', choices=['AlexNet', 'VGG16','ResNet18','SimpleCNN','Tail_Anchor'], type=str)

    # Phase 0 runtime arguments (engineering only, no algorithm change)
    subparsers.add_argument('--run_name', default='', type=str,
                            help='Experiment name. Empty = auto-generate '
                                 '{dataset}_{method}_c.._t.._ge.._le.._lr.._seed.._{hash}')
    subparsers.add_argument('--resume', default='', type=str,
                            help='"" = fresh run; "auto" = resume from '
                                 'RUN_DIR/checkpoints/latest.pth; or an explicit checkpoint path')
    subparsers.add_argument('--save_every', default=1, type=int,
                            help='Save round_XXXX.pth every N global rounds')
    subparsers.add_argument('--keep_last', default=0, type=int,
                            help='Keep only the last N round_XXXX.pth (0 = keep all). '
                                 'latest.pth and task_XX_end.pth are always kept')
    subparsers.add_argument('--eval_all_every', default=0, type=int,
                            help='Evaluate all seen tasks every N global rounds (0 = only at task end)')
    subparsers.add_argument('--deterministic', action='store_true',
                            help='Optional deterministic mode (cudnn.deterministic=True, '
                                 'benchmark=False). Default OFF = official FedTA behavior. '
                                 'Only needed for the resume regression test.')
    subparsers.add_argument('--max_rounds', default=0, type=int,
                            help='Stop after N global rounds (0 = full task_num*global_epoch). '
                                 'Engineering-only switch for the resume regression test; '
                                 'does not change per-round training logic.')
    subparsers.add_argument('--no_instrumentation', action='store_true',
                            help='Disable the Phase-0 observer instrumentation '
                                 '(train_log.csv local-phase evaluation, full '
                                 'seen-task evaluation, margin CSV output). '
                                 'Engineering-only switch used by '
                                 'tests/test_observer_invariance.py to prove the '
                                 'observers do not perturb the training trajectory.')





