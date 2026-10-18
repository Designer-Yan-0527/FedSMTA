"""FedMGP training entry with the frozen protocol manifest injected.

Runs the OFFICIAL FedMGP (external_baselines/FedMGP, never edited) with
only its dataset-partition interface replaced by the FedSMTA split
manifest (see protocol_adapters/manifest_splitter.py). Everything else —
training loop, losses, prompts, aggregation, evaluation — is FedMGP's own
code, executed from its own repo root (cwd is chdir'ed there so its
'./local_datasets' and 'pretrain_model/ViT-B_16.npz' relative paths
resolve; both are symlinks into the FedSMTA repo).

Protocol policy baked into this wrapper:

  frozen (must match the manifest, enforced):
    data_name=cifar100, client_num=5, task_num=5, private_class_num=15,
    surrogate_num=20, global_epoch=5, input_size=224,
    model=vit_base_patch16_224, seed (manifest's seed)
  protocol-aligned (default, overridable):
    local_epoch=30 (FedMGP config default is 5; FedMGP is FedTA-lineage so
    the local-epoch semantics are directly comparable and the frozen
    comparison protocol uses 30)
  method-native (FedMGP official defaults, overridable):
    lr=0.001875, threshold=0.1, batch_size=16, prompt params
    (size=1, length=15, top_k=1, ...)

Usage (from the FedSMTA repo root, FedSMTA conda env or a clone):

  python protocol_adapters/run_fedmgp.py \
      --manifest protocol_manifests/cifar100_c5t5p15s20_seed42.json \
      [--local_epoch 30] [--lr 0.001875] [--seed 42] [--device cuda] \
      [--output_dir output/]

Unknown CLI flags are FedMGP's own config flags (config/cifar100_delay.py)
and are forwarded as-is.
"""

import argparse
import sys
from pathlib import Path

ADAPTER_DIR = Path(__file__).resolve().parent
REPO_ROOT = ADAPTER_DIR.parent
if str(ADAPTER_DIR) not in sys.path:
    sys.path.insert(0, str(ADAPTER_DIR))

import manifest_splitter  # noqa: E402

FEDMGP_ROOT = REPO_ROOT / "external_baselines" / "FedMGP"

# keys whose values must equal the manifest protocol (fail-loud)
FROZEN_KEYS = ("client_num", "task_num", "private_class_num",
               "surrogate_num", "global_epoch")


def build_args(argv):
    """Build the FedMGP args namespace from its own cifar100_delay config,
    with the frozen/aligned defaults applied before parsing."""
    # put FedMGP root on sys.path for config import (torch-level patch is
    # already applied by manifest_splitter before Models.Client_DF loads)
    root = str(FEDMGP_ROOT.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)

    from config.cifar100_delay import get_args_parser

    parser = argparse.ArgumentParser(
        "FedMGP + protocol adapter (forwarded flags: FedMGP "
        "config/cifar100_delay.py)")
    sub = parser.add_subparsers(dest="subparser_name")
    config_parser = sub.add_parser("cifar100_delay")
    get_args_parser(config_parser)

    m = manifest_splitter.manifest()
    p = m["protocol"]
    # protocol-aligned defaults (official defaults: local_epoch=5; the
    # frozen comparison protocol uses 30; global_epoch official default
    # is already 5 = the frozen protocol)
    config_parser.set_defaults(local_epoch=30, seed=m["seed"],
                               global_epoch=5)

    args = parser.parse_args(["cifar100_delay"] + list(argv))

    # FedMGP's config declares the protocol params with nargs="+" (so a
    # CLI value becomes a 1-element list) while its own code consumes
    # plain ints and its official defaults ARE ints. Normalize: exactly
    # one value per param, as an int/float.
    for key in ("client_num", "task_num", "private_class_num",
                "surrogate_num", "global_epoch", "local_epoch"):
        v = getattr(args, key)
        if isinstance(v, list):
            if len(v) != 1:
                raise SystemExit(
                    f"--{key} expects exactly one value, got {v}")
            setattr(args, key, int(v[0]))
    if isinstance(args.threshold, list):
        args.threshold = args.threshold[0]

    # hard-freeze the data protocol (the manifest's domain)
    args.data_name = "cifar100"
    args.client_num = p["client_num"]
    args.task_num = p["task_num"]
    args.private_class_num = p["private_class_num"]
    args.surrogate_num = p["surrogate_num"]
    args.input_size = p["input_size"]
    args.model = "vit_base_patch16_224"
    if int(args.seed) != int(m["seed"]):
        raise SystemExit(
            f"--seed {args.seed} does not match the manifest seed "
            f"{m['seed']}: export a matching manifest instead of running "
            f"off-protocol")

    # FedMGP main.py expects data_name to be a LIST (official
    # five_datasets config uses nargs='+'); with the str default from
    # config/cifar100_delay.py, main's `args.data_name[0]` would slice the
    # string itself
    args.data_name = [args.data_name]

    return args


def main():
    cli = argparse.ArgumentParser(add_help=False)
    cli.add_argument("--manifest", required=True,
                     help="split manifest JSON "
                          "(diagnostics/export_split_manifest.py output)")
    known, rest = cli.parse_known_args()

    manifest_splitter.load_manifest(known.manifest)
    manifest_splitter.patch_torch_random_split()

    import os
    root = str(FEDMGP_ROOT.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    os.chdir(root)

    # FedMGP main.py imports the office-home spliter at module top, which
    # imports `deeplake` (only used in a comment; the cifar100 path never
    # touches it). Stub it instead of polluting the env with a heavy
    # office-home-only dependency.
    import types
    if "deeplake" not in sys.modules:
        sys.modules["deeplake"] = types.ModuleType("deeplake")

    import main as fedmgp_main

    manifest_splitter.patch_spliter()
    manifest_splitter.patch_client_get_data()
    manifest_splitter.patch_client_train_nb_classes_bug()
    manifest_splitter.patch_client_train_update_data()
    manifest_splitter.patch_customed_subset_pil()
    print("[protocol-adapter] applied: manifest spliter injection, inner "
          "70/30 injection, bug fix 1 (Server never calls "
          "Client_DF.update_data for cifar100; FedTA-lineage semantics "
          "restored), bug fix 2 (CustomedSubset PIL conversion line is "
          "commented out in the official repo; restored), bug fix 3 "
          "(local-prompt phase hardcodes np.arange(200) -> CUDA assert on "
          "100-class logits; replaced with args.nb_classes, matching the "
          "global phase of the same method)")

    args = build_args(rest)

    print("=" * 70)
    print("[protocol-adapter] FedMGP run under the frozen split manifest")
    print(f"[protocol-adapter] manifest : {known.manifest}")
    print(f"[protocol-adapter] fedmgp   : {root}")
    for k in FROZEN_KEYS + ("seed", "local_epoch", "lr", "batch_size",
                            "threshold"):
        print(f"[protocol-adapter] {k:14s}= {getattr(args, k, None)}")
    print("=" * 70)

    fedmgp_main.main(args)


if __name__ == "__main__":
    main()
