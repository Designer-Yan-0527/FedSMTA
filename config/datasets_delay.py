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
    subparsers.add_argument('--method', type=str, default='fedta', choices=['fedta'],
                            help='fedta = frozen Phase-0 baseline (read-only). '
                                 '(fedta_a2 及其 route_fix 家族已判死并删除，'
                                 '存档见 git 历史)')
    subparsers.add_argument('--e_prompt_layer_idx', default=[2, 3, 4], type=int, nargs="+",
                            help='the layer index of the E-Prompt')
    # A5-v2 refit 机制（2026-10-28，预注册 RUNCOMMANDS §23；
    # F1 head_dropout 消融与 F4 巩固臂已随判死方向清理删除）
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
                                 '--head_refit.')
    subparsers.add_argument('--refit_input', default='x1', choices=['x1', 'feat'],
                            help='A5-v2 Sec.23.6-1 verdict switch (default x1 = current '
                                 'behavior, bit-identical): the input fed to the '
                                 'refit/consolidation readout. x1 = the native 1536-d '
                                 'anchor-concat interface; feat = the 768-d pre-anchor '
                                 'z_op (vit output). Probe-verified on ImageNet-R '
                                 '(2026-10-29): the anchor half of x1 costs ~1.2pp '
                                 '(top-1 hard routing picks wrong anchors) and the '
                                 'cifar-tuned recipe overfits it on ~1k-row clients. '
                                 'feat heads are DEPLOYED as 1536-input modules with '
                                 'the anchor-half weights hard-zeroed (x1[:, :768] IS '
                                 'z_op), so evaluate/checkpoint/resume paths are '
                                 'unchanged. Fail-closed: requires --head_refit.')
    subparsers.add_argument('--refit_timing', default='local',
                            choices=['local', 'post_broadcast'],
                            help='A5-v2 Sec.28 criterion arm (default local = '
                                 'current M2\' behavior, bit-identical): WHEN the '
                                 'readout refit runs. local = end of train(), on '
                                 'the local post-phase-2 (pre-broadcast) interface '
                                 'state; post_broadcast = inside the broadcast '
                                 'path (get_global_proto_and_head), on the '
                                 'post-broadcast state (fused prompt already '
                                 'loaded into vit) -- the same feature stream '
                                 'evaluation/deployment uses, closing the ~2pp '
                                 'train-eval feature-timing gap attributed by the '
                                 'Sec.27.7-1 ladder probe. Orthogonal to '
                                 '--refit_input/--head_standardize; checkpoint '
                                 'schema unchanged. Fail-closed: requires '
                                 '--head_refit. See RUNCOMMANDS.md Sec.28.')
    # §33 CTRP v1 -- Calibrated Temporal Residual Prompt (pre-registered
    # RUNCOMMANDS Sec.33, 2026-11-03; GPT-frozen 9-point verdict).
    # off = frozen legacy path, bit-identical to the pre-CTRPTA HEAD.
    # The four matched arms (endpoint / fixed_alpha / ctrp_val /
    # ctrp_train) ALL run on the identical deterministic nested split
    # 56% core-train / 14% current-task val / 30% untouched test
    # (seed a5_stable_seed(seed, client, task)); `off` keeps the legacy
    # 70/30 path and is NOT a matched CTRP baseline.
    subparsers.add_argument('--ctrp_mode', default='off', type=str,
                            choices=['off', 'endpoint', 'fixed_alpha',
                                     'ctrp_val', 'ctrp_train'],
                            help='CTRP temporal prompt calibration mode. '
                                 'off = exact legacy path (default; no '
                                 'nested split, no calibration, checkpoint '
                                 'schema unchanged). endpoint = matched '
                                 'protocol with alpha=1 (matched-split '
                                 'endpoint control; premise failures are '
                                 'logged but do not stop the run). '
                                 'fixed_alpha = matched protocol with the '
                                 'fixed alpha from --ctrp_alpha. ctrp_val = '
                                 'primary selector (alpha chosen by pooled '
                                 'current-task VAL CE, Probe-H tie-breaks). '
                                 'ctrp_train = secondary selector contrast '
                                 '(alpha chosen by pooled core-train CE). '
                                 'Fail-closed: incompatible with '
                                 '--refit_timing post_broadcast '
                                 '(CTRP_V1_UNSUPPORTED_REFIT_TIMING).')
    subparsers.add_argument('--ctrp_alpha', default=None, type=float,
                            help='the fixed alpha for --ctrp_mode '
                                 'fixed_alpha (required and only valid in '
                                 'that mode; must lie in [0, 1])')
    subparsers.add_argument('--ctrp_alpha_grid', default='0.0,0.05,0.1,'
                            '0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,'
                            '0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0',
                            type=str,
                            help='ascending unique alpha grid in [0,1]; '
                                 'must contain 1.0 (the natural endpoint '
                                 'reference arm). Default = the Probe-H '
                                 'lambda grid (21 points, 0.05 step)')
    subparsers.add_argument('--ctrp_val_frac', default=0.20, type=float,
                            help='fraction of the inner 70%% train assigned '
                                 'to the current-task-only validation split '
                                 '(deterministic class-stratified, seed '
                                 'a5_stable_seed(seed, client, task); must '
                                 'lie strictly within (0, 1))')
    # §34.2 0b -- BSAB_PAIR_CAPTURE (pre-registered RUNCOMMANDS Sec.34.2;
    # user APPROVED FOR IMPLEMENTATION 2026-11-04, instrumentation-only):
    # pre/post-broadcast feature-pair cache for the B0 birth test.
    # Default OFF = completely bit-identical no-op. The capture run is
    # FROZEN as M2'-matched (--refit_timing local; fail-closed otherwise):
    # the in-run deployed head IS the M2' birth head, so B0 cleanly asks
    # whether a coordinate bridge can substitute for the M2'' post-
    # broadcast refit. Scope: capture ONLY (per-row z_ref/z_op pre+post,
    # prompt/head/row-index hashes, full-row train+test cache, RNG-neutral
    # diagnostic forward, independent bsab_capture/ output dir). No BSAB
    # module, no head/loss/prompt/optimizer/broadcast changes -- BSAB
    # architecture implementation stays FROZEN until the true birth-pair
    # B0 verdict. Labels are cached for offline accuracy evaluation only
    # and never enter bridge fitting. See RUNCOMMANDS.md Sec.34.2.
    subparsers.add_argument('--bsab_pair_capture', action='store_true',
                            help='Sec.34.2 0b instrumentation-only pre/post-'
                                 'broadcast pair capture (default OFF = '
                                 'bit-identical). Per broadcast boundary, '
                                 'caches the current task FULL train+test '
                                 'rows with z_ref/z_op on both sides of '
                                 'load_prompts, plus pre/post prompt '
                                 'hashes, the anchored birth head (state '
                                 '+ hash) and the row_index_hash (same '
                                 'sample-index sequence for both halves). '
                                 'Four-stream RNG snapshot/restore + mode '
                                 'restore: capture OFF vs ON trajectories '
                                 'must be bitwise identical. Fail-closed '
                                 'with --refit_timing post_broadcast '
                                 '(capture run = M2\'-matched, '
                                 'BSAB_CAPTURE_UNSUPPORTED_REFIT_TIMING). '
                                 'Outputs to <run_dir>/bsab_capture/.')
    # §34.4/§34.4a B1 DSAR sidecar (frozen 2026-11-06, SHARED_PATH_
    # CONFIRMED): client-local per-broadcast r8 readout adapters.
    # Deployment: current task z+ -> B -> L -> Tail_Anchor_t; historical
    # task z+ -> B -> Tail_Anchor_k (B-only). B = label-free fixed-
    # lambda ridge bridge (lambda = N*s*gamma_star, gamma_star =
    # 0.390324) with an unlabeled 80/20 paired-MSE identity guard; L =
    # rank-8 residual adapter trained by masked CE through the frozen
    # H_end (100ep / bs64 / Adam / lr1e-3 / wd1e-4). Default OFF =
    # completely bit-identical no-op (official path untouched). Zero
    # added FL payload: sidecar state never enters fed_avg / server
    # state transfer; checkpoints gain a dsar_sidecar key on ON runs
    # only. Fail-closed with --head_refit / --ctrp_mode != off.
    subparsers.add_argument('--dsar', action='store_true',
                            help='Sec.34.4/34.4a B1 DSAR readout sidecar '
                                 '(default OFF = bit-identical). Per prompt '
                                 'broadcast, each client locally fits '
                                 'B (label-free ridge bridge z+ -> z-, '
                                 'fixed lambda, identity-guarded) and L '
                                 '(rank-8 residual CE adapter on the '
                                 'current task); evaluate applies the '
                                 'frozen dispatch (current: B->L, '
                                 'historical: B-only) before Tail_Anchor. '
                                 'RNG/mode-neutral fits; zero FL payload.')
    subparsers.add_argument('--dsar_identity', action='store_true',
                            help='G1 verification mode (requires --dsar): '
                                 'skip all fits, commit exact-identity '
                                 'sidecars every broadcast. OFF vs '
                                 'ON(identity) must be bit-identical '
                                 '(logits/checkpoints); NOT a method arm.')
    # A2 routing-interface protection (restored 2026-11-08 for the final
    # A2+L+B plan; original validated implementation db01cf4/8fb9b87,
    # deleted in f37590d, re-introduced as a switch so it composes with
    # --dsar). Protection set = union over completed tasks t' < task_id of
    # the top-1 routing destinations of their train-split features under
    # the CURRENT key (D0 emergent support). At task start the protected
    # key/anchor rows are snapshotted; after EVERY Tail_Anchor optimizer
    # step (phase-2) they are restored (== freeze + weight-decay
    # exemption; optimizer untouched). Task 0 = empty set = no-op.
    # Default off = completely bit-identical no-op. RNG-neutral feature
    # pass (four-stream snapshot/restore); checkpoints gain an a2_state
    # key on ON runs only; fail-closed resume. Orthogonal to --dsar
    # (training-side vs readout-side).
    subparsers.add_argument('--a2_arm', default='off', type=str,
                            choices=['off', 'key', 'anchor', 'both'],
                            help='A2 routing-interface protection arm '
                                 '(default off = bit-identical). key = '
                                 'freeze protected key rows only '
                                 '(validated arm, cifar_a2_key_hsn); '
                                 'anchor = anchor rows only; both = both.')
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





