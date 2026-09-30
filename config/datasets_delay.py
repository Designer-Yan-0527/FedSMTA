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
    subparsers.add_argument('--method', type=str, default='fedta', help='The method of Prompt')
    subparsers.add_argument('--e_prompt_layer_idx', default=[2, 3, 4], type=int, nargs="+",
                            help='the layer index of the E-Prompt')
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





