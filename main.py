import argparse
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


def main(args):
    # Phase 0: run directory / tee logging / resume target resolution.
    # Must happen before any training output so everything is logged.
    # (RunManager is RNG-neutral, so its position does not affect the
    # partition RNG stream.)
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
