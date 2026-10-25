"""FedMGP training entry with the frozen protocol manifest injected.

Runs the OFFICIAL FedMGP (external_baselines/FedMGP, never edited) with
only its dataset-partition interface replaced by the FedSMTA split
manifest (see protocol_adapters/manifest_splitter.py). Everything else —
training loop, losses, prompts, aggregation, evaluation — is FedMGP's own
code, executed from its own repo root (cwd is chdir'ed there so its
'./local_datasets' and 'pretrain_model/ViT-B_16.npz' relative paths
resolve; both are symlinks into the FedSMTA repo).

The dataset is selected by the MANIFEST's dataset field (same dispatch
pattern as run_fl_baseline.py):

  cifar100    -> the original adapter path, unchanged; official FedMGP
                 main.py is executed with its spliter manifest-injected.
  ImageNet-R  -> FedMGP's official repo never shipped an INR route (no
                 config/, no spliter, main.py has no branch; only the
                 internal name 'imagenetr' survives in Client_DF). The
                 adapter supplies ONLY the missing route pieces at
                 runtime: a manifest-driven ImagenetR_spliter (main-repo
                 lineage class, the exact enumeration the manifest
                 indices refer to), an INR-capable surrogate wrapper
                 (main-repo iCIFAR100c; FedMGP's own is CIFAR-only), and
                 the one-line init_client route fix. Models, Server_DF,
                 Client_DF training/eval bodies remain FedMGP's own.

Additionally (metrics parity, 2026-10): a RunManager-format metrics
recorder (protocol_adapters/metrics_recorder.py) records the full
R[t][k] accuracy matrix at every task boundary via an RNG-neutral
read-only evaluation added by install_server_recorder(). The official
training loop body is preserved verbatim; see that function's docstring.

Protocol policy baked into this wrapper (frozen fields are enforced
against the manifest; both datasets share the same policy shape):

  frozen cifar100 : data_name=cifar100, 5c x 5t, private_class_num=15,
                    surrogate_num=20, global_epoch=5, input_size=224,
                    nb_classes=100, model=vit_base_patch16_224, seed
  frozen ImageNet-R: data_name='imagenetr' (FedMGP's own internal INR
                    identifier — the name Client_DF/update_data already
                    recognize; manifest dataset 'ImageNet-R'), 5c x 5t,
                    private_class_num=40, surrogate_num=5 (official FedTA
                    semantics: process_testdata(5)), global_epoch=5,
                    input_size=224, nb_classes=200,
                    model=vit_base_patch16_224, seed
  protocol-aligned (default, overridable):
    local_epoch=30 (FedMGP config default is 5; FedMGP is FedTA-lineage so
    the local-epoch semantics are directly comparable and the frozen
    comparison protocol uses 30)
  method-native (FedMGP official defaults, overridable):
    lr=0.001875, threshold=0.1, batch_size=16, prompt params
    (size=1, length=15, top_k=1, ...). The official repo has NO INR
    config, so the ImageNet-R run deliberately keeps the cifar100_delay
    config's method-native values (lr 0.001875 / threshold 0.1 /
    batch 16 / prompt pool size 1, length 15, top_k 1) — documented as
    "official has no INR config; CIFAR native values retained".

Usage (from the FedSMTA repo root, FedSMTA conda env or a clone):

  python protocol_adapters/run_fedmgp.py \
      --manifest protocol_manifests/cifar100_c5t5p15s20_seed42.json \
      [--local_epoch 30] [--lr 0.001875] [--seed 42] [--device cuda] \
      [--output_dir output/]

  python protocol_adapters/run_fedmgp.py \
      --manifest protocol_manifests/inr_c5t5p40s5_seed42.json \
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
import metrics_recorder  # noqa: E402

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

    if m["dataset"] == "ImageNet-R":
        # FedMGP's own internal INR identifier: Client_DF.__init__ /
        # update_data recognize 'imagenetr' (the route the official repo
        # intended but never finished); the manifest/paper name is
        # 'ImageNet-R' and that is what the metrics recorder uses.
        args.data_name = "imagenetr"
        if int(p["nb_classes"]) != 200:
            raise SystemExit(
                f"INR manifest protocol nb_classes={p['nb_classes']} != 200")
    elif m["dataset"] != "cifar100":
        raise SystemExit(
            f"manifest dataset {m['dataset']!r} is not supported by this "
            f"adapter (expected 'cifar100' or 'ImageNet-R')")

    # FedMGP main.py expects data_name to be a LIST (official
    # five_datasets config uses nargs='+'); with the str default from
    # config/cifar100_delay.py, main's `args.data_name[0]` would slice the
    # string itself
    args.data_name = [args.data_name]

    return args


def _fedmgp_eval_task(client, task, args):
    """RNG-neutral read-only evaluation of one client on one seen task,
    mirroring Client_DF.evaluate's inference path (local-prompt semantics,
    train_global_prompt=False, same as the official post-train evals) with
    shuffle=False so no RNG is consumed. The 'head_only' branch is omitted
    (both routed datasets, cifar100 and ImageNet-R, run method='delay')."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    client.vit.set_local_head(client.local_head[task])
    test_data = client.test_loader[task]
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)
    correct = 0
    total = 0
    for input_, target in test_loader:
        input_ = input_.to(client.device, non_blocking=True)
        target = target.to(client.device, non_blocking=True)
        with torch.no_grad():
            if client.original_model is not None:
                cls_features = client.original_model(input_)['pre_logits']
            else:
                cls_features = None
            if client.method == 'delay':
                output = client.vit(input_, task_id=task,
                                    cls_features=cls_features, train=False,
                                    train_global_prompt=False)
            else:
                output = client.vit(input_, task_id=client.task_id,
                                    cls_features=cls_features, train=False)
            logits = output['logits']
        mask = client.class_mask[task]
        not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
        not_mask = torch.tensor(not_mask, dtype=torch.int64).to(client.device)
        logits = logits.index_fill(dim=1, index=not_mask,
                                   value=float('-inf'))
        predicts = torch.max(logits, dim=1)[1].cpu()
        correct += (predicts == target.cpu()).sum()
        total += len(target)
    return float(100.0 * correct / total)


def install_server_recorder(fedmgp_main, recorder, args):
    """Replace Server_DF.train_clients with the official loop body
    (Models/Server_DF.py, verbatim) plus ONE adapter hook: at every task
    boundary, after aggregation + global-prompt sync, run an RNG-neutral
    full-matrix evaluation (every client x every seen task) and record
    R[t][k] into the RunManager-format CSV tree. The official per-round
    behavior is unchanged; the hook's evaluation is isolated with
    torch.random.fork_rng + numpy/random state snapshots and a
    shuffle=False loader, so the downstream training RNG stream is
    untouched (FedMGP natively evaluates only task 0 + the current task,
    so without this hook there is no R[t][k] matrix)."""
    Server_DF = fedmgp_main.Server_DF

    def _boundary_eval(server):
        import random as _pyrandom
        import numpy as _np
        import torch
        py_state = _pyrandom.getstate()
        np_state = _np.random.get_state()
        try:
            with torch.random.fork_rng():
                for j, client in enumerate(server.clients):
                    for k in range(int(server.task_id) + 1):
                        acc = _fedmgp_eval_task(client, k, args)
                        recorder.record(int(getattr(client, "id", j)),
                                        int(server.task_id), k, acc,
                                        phase="boundary")
                        print(f"[metrics-recorder] client {j} after_task "
                              f"{int(server.task_id)} test_task {k} "
                              f"acc {acc:.2f}")
        finally:
            _pyrandom.setstate(py_state)
            _np.random.set_state(np_state)
        recorder.flush(int(server.task_num))

    def train_clients_recorded(self):
        # official body (Models/Server_DF.py train_clients), verbatim,
        # with the boundary hook appended after the global-prompt sync
        for i in range(self.task_num * self.global_epoch):
            if self.task_id != i//self.global_epoch:
                self.task_id = i//self.global_epoch
                print(self.args.data_name)
                if self.args.data_name=='office_home':
                    datas,mask = self.client_data.random_split(domain=self.task_id)
                    for j in range(self.client_num):
                        self.clients[j].get_data_office_home(self.task_id,datas[j],mask[j])
            print(f"--------round {i},task number {i//self.global_epoch}-----------")
            for j in range(self.client_num):
                self.clients[j].train(round=i,args=self.args)

            if self.args.data_name == 'office_home':
                self.fed_avg([i for i in range(self.client_num)])
                self.fed_avg_head([i for i in range(self.client_num)])
            else:
                self.filter(self.threshold)
            # self.evaluate_global()
            print('Server aggregation Compelte, results are ()：')
            for j in range(self.client_num):
                self.clients[j].set_global_prompt(self.global_prompt)
                print('-------')
            # [protocol-adapter] task-boundary full-matrix eval (RNG-neutral)
            if (i + 1) % self.global_epoch == 0:
                _boundary_eval(self)
        print("训练完成！")

    Server_DF.train_clients = train_clients_recorded
    return Server_DF


def _main_inr(args, ImagenetR_spliter):
    """ImageNet-R route: FedMGP main.py's main() body with the INR branch
    the official repo never shipped (its cifar100 branch pattern, verbatim
    seeding / model construction / Server wiring; only the data lines are
    INR and they run through the manifest-patched ImagenetR_spliter).
    Everything FedMGP-side below (Models/, Server_DF, Client_DF training
    and evaluation bodies) is the official code under the documented
    runtime fixes. Not executed for cifar100 runs."""
    import random

    import numpy as np
    import torch
    from timm.models import create_model
    from torch.backends import cudnn

    from Models.Server_DF import Server_DF

    m = manifest_splitter.manifest()
    p = m["protocol"]

    device = torch.device(args.device)
    # fix the seed for reproducibility (official main.py lines)
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    pretrained_cfg = create_model(args.model).default_cfg
    pretrained_cfg['file'] = 'pretrain_model/ViT-B_16.npz'
    print(pretrained_cfg)

    cudnn.benchmark = True
    args.data_name = args.data_name[0]
    if args.data_name != 'imagenetr':
        raise SystemExit(
            f"internal data_name {args.data_name!r} != 'imagenetr': the "
            "INR route must be reached with FedMGP's own INR identifier")

    # INR data (official cifar100-branch pattern; surrogate count 5 =
    # official FedTA INR semantics -- process_testdata(5) -- frozen by the
    # manifest, which patch_imagenet_r_spliter enforces)
    data_spliter = ImagenetR_spliter(client_num=args.client_num,
                                     task_num=args.task_num,
                                     private_class_num=args.private_class_num,
                                     input_size=args.input_size,
                                     data_path='./local_datasets')
    client_data, client_mask = data_spliter.random_split()
    surro_data, test_data = ImagenetR_spliter(
        client_num=args.client_num, task_num=args.task_num,
        private_class_num=args.private_class_num,
        input_size=args.input_size,
        data_path='./local_datasets').process_testdata(5)
    # surrogate wrapper: main-repo iCIFAR100c (INR-capable, FedTA-lineage
    # semantics identical to FedMGP's own wrapper; FedMGP's own copy is
    # CIFAR-only and crashes on INR -- see manifest_splitter)
    surro_data = manifest_splitter.load_mainrepo_icifar100c()(
        subset=surro_data)
    args.nb_classes = int(p["nb_classes"])

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
        prompt_pool=args.prompt_pool,
        prompt_key=args.prompt_key,
        pool_size=args.size,
        top_k=args.top_k,
        batchwise_prompt=args.batchwise_prompt,
        prompt_key_init=args.prompt_key_init,
        head_type=args.head_type,
        use_prompt_mask=args.use_prompt_mask,
        e_prompt_layer_idx=args.e_prompt_layer_idx,
        method=args.method,
    )

    model.init_local_prompt()
    original_model.to(device)
    model.to(device)

    if args.freeze:
        # all parameters are frozen for original vit model
        for prm in original_model.parameters():
            prm.requires_grad = False
        # freeze args.freeze[blocks, patch_embed, cls_token] parameters
        for n, prm in model.named_parameters():
            if n.startswith(tuple(args.freeze)):
                prm.requires_grad = False

    print(args)
    for n, prm in model.named_parameters():
        if prm.requires_grad == True:
            print(n)

    myServer = Server_DF(id='Server', origin_model=original_model,
                         model=model, client_num=args.client_num,
                         task_num=args.task_num, subset=client_data,
                         class_mask=client_mask, lr=args.lr,
                         global_epoch=args.global_epoch,
                         local_epoch=args.local_epoch,
                         batch_size=args.batch_size, device=args.device,
                         method=args.method, threshold=args.threshold,
                         surrogate_data=surro_data, test_data=None,
                         args=args)
    myServer.start()


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

    dataset = manifest_splitter.manifest()["dataset"]
    if dataset == "cifar100":
        # (unchanged cifar100 patch set, exact order)
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
        inr_spliter = None
    elif dataset == "ImageNet-R":
        # INR route: the official repo has no INR spliter/config/main
        # branch, so the adapter supplies the missing route pieces
        # (manifest-driven spliter + init_client route fix); the dataset-
        # agnostic fixes below are the SAME ones the cifar100 path uses.
        # The np.arange(200) fix is a semantic no-op at nb_classes=200
        # (the official constant is CORRECT for INR; the patch still
        # keeps the method nb_classes-driven, identical behavior).
        inr_spliter = manifest_splitter.patch_imagenet_r_spliter()
        manifest_splitter.patch_client_get_data()
        manifest_splitter.patch_client_train_nb_classes_bug()
        manifest_splitter.patch_client_train_update_data()
        manifest_splitter.patch_server_init_client_imagenetr()
        print("[protocol-adapter] applied (ImageNet-R): manifest "
              "ImagenetR_spliter injection, inner 70/30 injection, bug "
              "fix 1 (Server never calls Client_DF.update_data; "
              "FedTA-lineage semantics restored), bug fix 3 "
              "(np.arange(200) -> args.nb_classes; a no-op at the INR "
              "nb_classes=200, kept for one shared code path), INR route "
              "fix (Server_DF.init_client misses the 'imagenetr' branch "
              "that Client_DF/update_data already recognize; restored); "
              "surrogate wrapper = main-repo iCIFAR100c (FedMGP's own is "
              "CIFAR-only); CustomedSubset PIL fix skipped (INR subsets "
              "already convert via the main-repo lineage class)")
    else:
        raise SystemExit(
            f"manifest dataset {dataset!r} is not supported by this "
            f"adapter (expected 'cifar100' or 'ImageNet-R')")

    args = build_args(rest)

    meta = {"manifest": str(Path(known.manifest).resolve()),
            "adapter": "protocol_adapters/run_fedmgp.py",
            "eval_pool": "per-(client, task) manifest test split (the "
                         "pool Client_DF.evaluate uses, gate_fedmgp-"
                         "sealed); boundary full-matrix eval added by "
                         "the adapter, RNG-neutral"}
    if dataset == "ImageNet-R":
        meta["eval_pool"] = ("per-(client, task) manifest test split (the "
                             "pool Client_DF.evaluate uses, gate_fedmgp_inr-"
                             "sealed); boundary full-matrix eval added by "
                             "the adapter, RNG-neutral")
        meta["inr_route"] = ("official FedMGP has no ImageNet-R config/"
                             "spliter/main branch; adapter supplies the "
                             "manifest-driven ImagenetR_spliter + the "
                             "init_client 'imagenetr' route fix + the "
                             "INR-capable main-repo iCIFAR100c surrogate "
                             "wrapper (method-native FedMGP transforms "
                             "and training/eval code unchanged)")
        meta["method_native_hp"] = ("official has no INR config; CIFAR "
                                    "native values retained (lr 0.001875 "
                                    "/ threshold 0.1 / batch 16 / prompt "
                                    "size 1, length 15, top_k 1)")
    recorder = metrics_recorder.MetricsRecorder(
        out_root=REPO_ROOT / "output", dataset=dataset,
        method="fedmgp", seed=int(args.seed),
        meta=meta)
    install_server_recorder(fedmgp_main, recorder, args)

    print("=" * 70)
    print("[protocol-adapter] FedMGP run under the frozen split manifest")
    print(f"[protocol-adapter] manifest : {known.manifest}")
    print(f"[protocol-adapter] fedmgp   : {root}")
    for k in FROZEN_KEYS + ("seed", "local_epoch", "lr", "batch_size",
                            "threshold"):
        print(f"[protocol-adapter] {k:14s}= {getattr(args, k, None)}")
    print(f"[protocol-adapter] metrics     = {recorder.run_dir}")
    print("=" * 70)

    if dataset == "cifar100":
        fedmgp_main.main(args)
    else:
        _main_inr(args, inr_spliter)


if __name__ == "__main__":
    main()
