import argparse
import math
import os

from pathlib import Path
import random

import numpy as np
import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader

from Models.Server_DF import Server_DF
from config.datasets_delay import get_args_parser
from runtime_utils import RunManager

from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
import Models.vision_transformer__l2p
from data.iCIFAR100c import iCIFAR100c
from data.imagenet_r_subset_spliter import ImagenetR_spliter

torch.set_printoptions(threshold=float('inf'))

from data.cifar100_subset_spliter import cifar100_Data_Spliter
import warnings
warnings.filterwarnings("ignore")


def setup_determinism(args):
    """Phase 0: optional deterministic mode (default OFF, official behavior).

    The official FedTA runs with cudnn.benchmark=True and non-deterministic
    kernels. Determinism is only needed by the resume regression test; it is
    never enabled silently."""
    if getattr(args, 'deterministic', False):
        cudnn.deterministic = True
        cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        print("[deterministic] ON (cudnn.deterministic=True, benchmark=False)")
    else:
        cudnn.benchmark = True


def build_server(args, run_manager=None):
    """Shared Phase-0 bootstrap: seed -> data partition -> models -> Server_DF.

    The RNG-consumption order here is bit-identical to the original main()
    (seed -> create_model default_cfg -> partition -> models -> clients), so
    Phase-1 diagnostics calling this function reproduce exactly the same
    client data partition as the training run.

    main() additionally creates a RunManager (RNG-neutral) before calling
    this; diagnostics pass run_manager=None."""
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    pretrained_cfg = create_model(args.model).default_cfg
    pretrained_cfg['file'] = 'pretrain_model/ViT-B_16.npz'

    setup_determinism(args)

    print(args.data_name)
    if args.data_name == 'cifar100':
        client_data, client_mask = cifar100_Data_Spliter(client_num=args.client_num, task_num=args.task_num, private_class_num=args.private_class_num, input_size=args.input_size, data_path=args.data_path).random_split()
        surro_data, test_data = cifar100_Data_Spliter(client_num=args.client_num, task_num=args.task_num,
                                                      private_class_num=args.private_class_num, input_size=args.input_size, data_path=args.data_path).process_testdata(args.surrogate_num)
        surro_data = iCIFAR100c(subset=surro_data)
        args.nb_classes = 100

    elif args.data_name == 'ImageNet-R':
        data_spliter = ImagenetR_spliter(client_num=args.client_num, task_num=args.task_num,
                                         private_class_num=args.private_class_num,
                                         input_size=args.input_size,
                                         data_path=args.data_path)

        client_data, client_mask = data_spliter.random_split()
        args.nb_classes = 200

        # STRICT BASELINE (Phase 0): official FedTA fixes the ImageNet-R
        # surrogate count to 5 per class (official main.py: process_testdata(5)).
        # --surrogate_num is only honored for cifar100 (official default 20).
        surro_data, test_data = ImagenetR_spliter(client_num=args.client_num, task_num=args.task_num,
                                                  private_class_num=args.private_class_num,
                                                  input_size=args.input_size,
                                                  data_path=args.data_path).process_testdata(5)
        surro_data = iCIFAR100c(subset=surro_data)

    else:
        raise ValueError(
            f"Unsupported data_name: {args.data_name!r}. "
            "This project only supports 'cifar100' and 'ImageNet-R'.")

    print(f"Creating original model: {args.model}")
    original_model = create_model(
        args.model,
        pretrained=True,
        pretrained_cfg=pretrained_cfg,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )

    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=True,
        pretrained_cfg=pretrained_cfg,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        prompt_length=args.length,
        embedding_key=args.embedding_key,
        prompt_init=args.prompt_key_init,
        prompt_pool=True,
        prompt_key=args.prompt_key,
        pool_size=args.size,
        top_k=args.top_k,
        batchwise_prompt=args.batchwise_prompt,
        prompt_key_init=args.prompt_key_init,
        head_type=args.head_type,
        use_prompt_mask=args.use_prompt_mask,
    )

    original_model.to(device)
    model.to(device)

    if args.freeze:
        # all parameters are frozen for original vit model
        # freeze args.freeze[blocks, patch_embed, cls_token] parameters
        for n, p in model.named_parameters():
            if n.startswith(tuple(args.freeze)):
                p.requires_grad = False

    print(args)
    for n, p in model.named_parameters():
        if p.requires_grad == True:
            print(n)

    myServer = Server_DF(id='Server', origin_model=original_model, model_name=args.model_name, client_num=args.client_num, task_num=args.task_num,
                         subset=client_data, class_mask=client_mask, lr=args.lr, global_epoch=args.global_epoch, local_epoch=args.local_epoch,
                         batch_size=args.batch_size, device=args.device, method=args.method, threshold=args.threshold,
                         surrogate_data=surro_data, test_data=None, args=args, model=model, run_manager=run_manager)
    return myServer


def validate_method_args(args):
    """Phase 5 A3 / Phase 6 A4: fail-fast config validation (method
    isolation).

    A3 (fedta_a3) and A4 (fedta_a4) are mutually exclusive with A2
    (fedta_a2) by construction: a single --method choice. A3 never freezes
    key/anchor rows and does not use the A2 protection-set mask; A4 reuses
    the A2 protection-set mask and snapshot code but replaces the hard
    post-step restore with a soft direction loss (key rows only, anchors
    fully FedTA-trained). The frozen fedta baseline accepts no A3/A4
    arguments.
    """
    # Phase 7 A5 PR-3: the legacy-checkpoint migration declaration is an
    # A5-only policy flag; every other method has no prototype table to
    # migrate, so accepting it there would be cargo-cult usage.
    if (getattr(args, 'allow_proto_migration', False)
            and args.method != 'fedta_a5'):
        raise ValueError(
            "invalid configuration: --allow_proto_migration is ONLY valid "
            "with --method fedta_a5 (it declares an explicit legacy-A5 "
            "checkpoint migration; no other method carries the prototype "
            "table)")
    # Phase 7 A5 PR-3: the Local Adapter arm switch is an A5-only policy
    # flag; every other method has no shared U to (not) aggregate, so
    # accepting it there would be cargo-cult usage.
    if (getattr(args, 'a5_u_agg', 'fedavg') == 'local'
            and args.method != 'fedta_a5'):
        raise ValueError(
            "invalid configuration: --a5_u_agg local is ONLY valid with "
            "--method fedta_a5 (the Local Adapter mechanism contrast; no "
            "other method carries the shared U adapter)")
    if args.method == 'fedta_a4':
        if getattr(args, 'a4_lambda', None) is None:
            raise ValueError(
                "invalid fedta_a4 configuration: --a4_lambda is REQUIRED "
                "(no working default; set it from "
                "diagnostics/a4_lambda_probe.py, not by guess)")
        if not (float(args.a4_lambda) > 0) or not math.isfinite(float(args.a4_lambda)):
            raise ValueError(
                "invalid fedta_a4 configuration: --a4_lambda must be a "
                "finite value > 0")
        return
    if args.method == 'fedta_a5':
        if getattr(args, 'a2_arm', 'anchor') == 'none':
            # valid: Shared-only factorial arm (P2 v1.2 §6)
            pass
        if int(getattr(args, 'a5_rank', 8)) <= 0:
            raise ValueError(
                "invalid fedta_a5 configuration: --a5_rank must be > 0")
        if float(getattr(args, 'a5_proto_lambda', 0.5)) < 0:
            raise ValueError(
                "invalid fedta_a5 configuration: --a5_proto_lambda must be "
                ">= 0 (0 = proto-CE OFF, the Shared CE-only arm)")
        if float(getattr(args, 'a5_proto_tau', 0.07)) <= 0:
            raise ValueError(
                "invalid fedta_a5 configuration: --a5_proto_tau must be > 0")
        return
    if args.method == 'fedta_a2' and getattr(args, 'a2_arm', 'anchor') == 'none':
        raise ValueError(
            "invalid fedta_a2 configuration: --a2_arm none is ONLY valid "
            "for --method fedta_a5 (the Shared-only factorial arm). A "
            "fedta_a2 run with no protection arm is just the fedta "
            "baseline under a wrong name (fail-closed).")
    if args.method != 'fedta_a3':
        return
    problems = []
    if not (float(args.a3_lambda) > 0):
        problems.append("--a3_lambda must be > 0")
    if not (float(args.a3_tau) > 0):
        problems.append("--a3_tau must be > 0")
    if int(args.a3_n_ref) <= 0:
        problems.append("--a3_n_ref must be > 0")
    if int(args.a3_ref_batch_size) <= 0:
        problems.append("--a3_ref_batch_size must be > 0")
    if int(args.a3_compat_every) < 1:
        problems.append("--a3_compat_every must be >= 1")
    if problems:
        raise ValueError(
            f"invalid fedta_a3 configuration: " + "; ".join(problems))


def main(args):
    # Phase 0: run directory / tee logging / resume target resolution.
    # Must happen before any training output so everything is logged.
    # (RunManager is RNG-neutral, so its position does not affect the
    # partition RNG stream.)
    validate_method_args(args)
    run_manager = RunManager(args)

    myServer = build_server(args, run_manager=run_manager)

    myServer.start()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('FedTA training and evaluation configs')

    # single neutral entry point for both datasets; the dataset is
    # selected with --data_name (cifar100 / ImageNet-R)
    config = 'datasets_delay'

    subparser = parser.add_subparsers(dest='subparser_name')
    config_parser = subparser.add_parser(config, help='FedTA configs (cifar100 / ImageNet-R)')

    get_args_parser(config_parser)

    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
