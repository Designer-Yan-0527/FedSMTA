"""Phase-0 Gate 0A diagnostic: initialization-state parity probe.

Temporary diagnostic for locating the official-vs-fork divergence found by
tests/test_official_fedta_parity.py. This script does NOT touch training
code; it only rebuilds the two model instances exactly the way each repo's
main.py does (official: pretrained=False + explicit load_pretrained;
fork: pretrained=True + timm custom load), then dumps the initial weights
and the post-construction torch RNG state for a bitwise comparison.

Usage (run from the repo root of each side):
    cd /path/to/official_FedTA  && python diagnostics/parity_init_probe.py official /tmp/init_official.pth
    cd /path/to/FedSMTA         && python diagnostics/parity_init_probe.py fork     /tmp/init_fork.pth
    python diagnostics/parity_init_probe.py compare /tmp/init_official.pth /tmp/init_fork.pth

Interpretation:
    * torch RNG state differs  -> construction paths consume RNG differently
      (e.g. the timm custom-load path), so training batch streams diverge
      from round 0 even if all weights match.
    * weight tensors differ    -> the two npz loading paths land different
      initial weights (initial models are not the same).
    * everything equal         -> initialization is identical; the divergence
      happens inside the training loop (Server_DF / Client_DF logic).
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
from torch.backends import cudnn


def build_args():
    sys.path.insert(0, os.getcwd())
    from timm.models import create_model  # noqa: F401  (import check)
    import Models.vision_transformer__l2p  # noqa: F401  (registers repo-local ViT)

    try:
        from config.cifar100 import get_args_parser
    except ImportError:
        try:
            from config.cifar100_delay import get_args_parser
        except ImportError:
            from config.datasets_delay import get_args_parser

    parser = argparse.ArgumentParser()
    get_args_parser(parser)
    return parser.parse_args([])


def run_side(side, out_path):
    from timm.models import create_model

    args = build_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    # replicate main.py bootstrap order (RNG-consumption order matters):
    # seed -> default_cfg probe -> model construction (+ npz load)
    pretrained_cfg = create_model(args.model).default_cfg
    pretrained_cfg['file'] = 'pretrain_model/ViT-B_16.npz'

    common = dict(num_classes=100, drop_rate=args.drop,
                  drop_path_rate=args.drop_path, drop_block_rate=None)
    prompt = dict(prompt_length=args.length, embedding_key=args.embedding_key,
                  prompt_init=args.prompt_key_init, prompt_pool=True,
                  prompt_key=args.prompt_key, pool_size=args.size,
                  top_k=args.top_k, batchwise_prompt=args.batchwise_prompt,
                  prompt_key_init=args.prompt_key_init, head_type=args.head_type,
                  use_prompt_mask=args.use_prompt_mask)

    if side == 'official':
        original_model = create_model(args.model, pretrained=False, **common)
        model = create_model(args.model, pretrained=False, **common, **prompt)
        original_model.load_pretrained('pretrain_model/ViT-B_16.npz')
        model.load_pretrained('pretrain_model/ViT-B_16.npz')
    else:
        original_model = create_model(args.model, pretrained=True,
                                      pretrained_cfg=pretrained_cfg, **common)
        model = create_model(args.model, pretrained=True,
                             pretrained_cfg=pretrained_cfg, **common, **prompt)

    dump = {
        'side': side,
        'original_model': {k: v.cpu().clone()
                           for k, v in original_model.state_dict().items()},
        'model': {k: v.cpu().clone() for k, v in model.state_dict().items()},
        'torch_rng_state': torch.get_rng_state(),
    }
    torch.save(dump, out_path)
    print(f"[{side}] saved -> {out_path} "
          f"(model: {len(dump['model'])} tensors, "
          f"original_model: {len(dump['original_model'])} tensors)")


def run_compare(path_a, path_b):
    def load_compat(p):
        try:
            return torch.load(p, map_location='cpu', weights_only=False)
        except TypeError:  # older torch without the weights_only kwarg
            return torch.load(p, map_location='cpu')

    a = load_compat(path_a)
    b = load_compat(path_b)
    print(f"side A: {a.get('side')}  side B: {b.get('side')}")

    for key in ('original_model', 'model'):
        sa, sb = a[key], b[key]
        only_a = sorted(set(sa) - set(sb))
        only_b = sorted(set(sb) - set(sa))
        print(f"== {key}: {len(sa)} vs {len(sb)} tensors")
        if only_a:
            print(f"   only-A keys: {only_a[:10]}")
        if only_b:
            print(f"   only-B keys: {only_b[:10]}")
        diffs = []
        for k in sorted(set(sa) & set(sb)):
            if sa[k].shape != sb[k].shape:
                diffs.append((k, f"shape {tuple(sa[k].shape)} vs {tuple(sb[k].shape)}"))
            else:
                d = (sa[k].float() - sb[k].float()).abs().max().item()
                if d != 0:
                    diffs.append((k, f"max_diff={d}"))
        print(f"   differing tensors: {len(diffs)}")
        for k, msg in diffs[:20]:
            print(f"     {k}: {msg}")
        if len(diffs) > 20:
            print(f"     ... and {len(diffs) - 20} more")

    rng_equal = torch.equal(a['torch_rng_state'], b['torch_rng_state'])
    print("torch RNG state equal:", rng_equal)
    if not rng_equal:
        print("=> construction paths consumed RNG differently; training "
              "batch streams diverge from round 0.")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        raise SystemExit(1)
    mode = sys.argv[1]
    if mode == 'compare':
        if len(sys.argv) != 4:
            print(__doc__)
            raise SystemExit(1)
        run_compare(sys.argv[2], sys.argv[3])
    elif mode in ('official', 'fork'):
        if len(sys.argv) != 3:
            print(__doc__)
            raise SystemExit(1)
        run_side(mode, sys.argv[2])
    else:
        print(__doc__)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
