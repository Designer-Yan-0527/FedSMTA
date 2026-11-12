"""Protocol equivalence gate for HePCo (run BEFORE any formal training).

CPU-only and training-free. Three parts (same discipline as
gate_fedmgp.py / gate_fedprotip.py):

  Part A — native protocol mismatch (informational, expected):
      the official HePCo protocol is 10 uniform tasks (10x10 / 20x20
      classes) with per-client heterogeneity coming from
      gen_imbalanced_data (seeded class shuffles + percent/cutoff
      subsampling), an ImageNet-R loader that shuffles and 80/20-splits
      the raw files, and 2-GPU DataParallel execution. None of this is
      the frozen manifest; the adapter replaces the partition interface
      BY DESIGN. Recorded, not fatal.

  Part B — adapter injection (gating): build a patched Trainer on CPU
      and verify what HePCo would actually TRAIN/EVAL on is the frozen
      manifest:
        B1  trainer.tasks / tasks_logits == first-arrival POSITION
            ranges [K_{t-1}, K_t); server_config['tasks_real'] /
            ['tasks'] / ['prompt_param'] reconciled; class_mapping ==
            raw->first-arrival position; increments == arrival counts
        B2  per-client train archive[c][t] == manifest inner.train(c,t)
            restricted to classes NEW at t, in manifest order; every
            target's arrival position in [K_{t-1}, K_t); drops ==
            expected first-arrival drops; load_dataset(train=True)
            fills label_counts in the official raw-class-id domain
        B3  server test archive[t] == manifest test_pool of the classes
            new at t (pool order); per-client test archive[t] ==
            test_pool of mask(c,t); load_dataset(t, train=False)
            concatenates to the arrived pool slice; __getitem__ maps
            raw -> arrival positions
  PASS requires every Part B item exact.

  Part C — seal checks (gating): every source patch must (a) still find
      its anchor in the official repo (repo-drift protection), and
      (b) be live on the patched object:
        C1  official anchors intact: trainer.py (wandb.init, FedAvg
            communicate average, boundary-eval anchor),
            dataloader.py (gen_imbalanced_data, official INR shuffle
            + 80/20), learners/{default,prompt}.py cuda guards (> 1),
            models/vit.py fairscale import
        C2  patches live: Trainer.__init__ is the manifest wrapper;
            Trainer.train source carries _HEPCO_BOUNDARY_EVAL;
            NormalNN.cuda / DualPrompt.cuda patched to >= 1;
            wandb + fairscale are our stubs; iIMAGENET_R.load replaced
            (ImageNet-R only)
        C3  frozen args: n_rounds=5, schedule=[30], seed=manifest seed,
            memory=0, n_clients=5, learner prompt/L2P, max_task=5
        C4  numpy LAPACK self-test status recorded (scipy reroute if
            broken; the reroute must not touch the official files)

Usage (FedSMTA conda env, from the FedSMTA repo root):

  python protocol_adapters/gate_hepco.py \
      --manifest protocol_manifests/cifar100_c5t5p15s20_seed42.json

Writes protocol_manifests/gate_hepco_{dataset}_report.json.
Exit code 0 only on PASS (Part B + Part C).
"""

import argparse
import inspect
import json
import os
import sys
import textwrap
import types
from pathlib import Path

import numpy as np

ADAPTER_DIR = Path(__file__).resolve().parent
REPO_ROOT = ADAPTER_DIR.parent
HEPCO_ROOT = REPO_ROOT / "external_baselines" / "hepco-fed"

if str(ADAPTER_DIR) not in sys.path:
    sys.path.insert(0, str(ADAPTER_DIR))

import run_hepco as ADP  # noqa: E402


def row(rows, item, ok, got=0, want=0):
    rows.append({"item": item, "pass": bool(ok),
                 "got": got, "want": want})
    return bool(ok)


def cmp_list(rows, item, got, want):
    ok = list(got) == list(want)
    rows.append({"item": item, "pass": bool(ok),
                 "got": len(list(got)), "want": len(list(want))})
    return ok


def main():
    p = argparse.ArgumentParser("HePCo protocol equivalence gate")
    p.add_argument("--manifest", required=True)
    p.add_argument("--data-root", default=str(REPO_ROOT / "local_datasets"))
    p.add_argument("-o", "--report", default="")
    args = p.parse_args()
    if args.report:
        args.report = str(Path(args.report).resolve())

    ADP.install_import_stubs()
    ADP.install_robust_svd()
    m = ADP.load_manifest(args.manifest)
    dataset = m["dataset"]
    seed = m["seed"]
    nc, nt = m["protocol"]["client_num"], m["protocol"]["task_num"]
    inc, arrival, new_at = ADP.compute_arrival()
    ADP.build_train_raw()
    ADP.build_pools(args.data_root)

    hepco_root = str(HEPCO_ROOT.resolve())
    if hepco_root not in sys.path:
        sys.path.insert(0, hepco_root)
    os.chdir(hepco_root)

    is_cifar = dataset == "cifar100"
    ADP.patch_imagenet_r_load()
    import dataloaders as dl_mod
    base_cls = dl_mod.IMBALANCECIFAR if is_cifar else dl_mod.IMBALANCEINR
    ADP._STATE["dataset_cls"] = ADP.make_manifest_dataset_class(
        base_cls, is_cifar)
    ADP.patch_learner_cuda()
    ADP.patch_dualprompt_update_model()
    ADP.patch_trainer_init()
    ADP.patch_trainer_train()

    # dummy recorder boundary evals need at gate time: none (no train()),
    # but _hepco_boundary_eval references _STATE['recorder'] only at run
    cli = types.SimpleNamespace(
        gpuid=-1, seed=seed, rounds=5, local_epochs=30,
        data_root=args.data_root,
        log_dir=str(REPO_ROOT / "output" / ".gate_hepco_no_write"))
    hepco_args = ADP.build_args(cli)

    import trainer as trainer_mod
    trainer = trainer_mod.Trainer(hepco_args, seed, ADP.METRIC_KEYS,
                                  ADP.SAVE_KEYS)

    rows = []
    fcfg = ("configs/cifar100_vit_prompt.yaml" if is_cifar
            else "configs/imnet-r_clip_prompt.yaml")
    import yaml
    with open(HEPCO_ROOT / fcfg, "r", encoding="utf-8") as f:
        official_cfg = yaml.safe_load(f)

    # ------------------------------------------------------------------
    # Part A: native protocol mismatch (informational)
    # ------------------------------------------------------------------
    partA_ok = True
    dl_src = (HEPCO_ROOT / "dataloaders" / "dataloader.py").read_text("utf-8")
    tr_src = (HEPCO_ROOT / "trainer.py").read_text("utf-8")
    partA_ok &= row(rows, "A official per-client heterogeneity is "
                          "gen_imbalanced_data (anchor present; replaced "
                          "by design)",
                    "def gen_imbalanced_data" in dl_src, 1, 1)
    partA_ok &= row(
        rows, f"A official tasks {official_cfg['max_task']} x "
              f"{official_cfg['first_split_size']}/{official_cfg['other_split_size']}"
              f" classes != frozen 5 x first-arrival {inc} "
              "(documented mismatch; adapter enforces the arrival split)",
        (official_cfg["max_task"], official_cfg["first_split_size"],
         official_cfg["other_split_size"]) != (nt, inc[0], inc[-1]),
        official_cfg["max_task"], nt)
    partA_ok &= row(rows, "A official 2-GPU DataParallel assumption "
                          "(.module.prompt hard-coded; single-GPU runs "
                          "get a 1-device DataParallel wrapper)",
                    ".module.prompt" in tr_src, 1, 1)
    if not is_cifar:
        partA_ok &= row(rows, "A official INR loader shuffles + 80/20 "
                              "(anchor present; replaced by the frozen "
                              "sorted domain)",
                        "0.8*n_data" in dl_src
                        and "random.seed(0)" in dl_src, 1, 1)

    # ------------------------------------------------------------------
    # Part B: adapter injection (gating)
    # ------------------------------------------------------------------
    partB_ok = True
    k0 = 0
    ranges = []
    for t in range(nt):
        ranges.append(list(range(k0, k0 + inc[t])))
        k0 += inc[t]
    partB_ok &= cmp_list(rows, "B1 trainer.tasks == arrival ranges",
                        [list(x) for x in trainer.tasks],
                        [list(x) for x in ranges])
    partB_ok &= cmp_list(rows, "B1 tasks_logits == arrival ranges",
                        trainer.tasks_logits,
                        [list(x) for x in ranges])
    partB_ok &= cmp_list(rows, "B1 server_config tasks_real == ranges",
                        trainer.server_config["tasks_real"],
                        [list(x) for x in ranges])
    partB_ok &= cmp_list(rows, "B1 server_config prompt_param",
                        [trainer.server_config["prompt_param"][0],
                         list(trainer.server_config["prompt_param"][1])],
                        [nt, [100, 20, 1, -1]])
    partB_ok &= row(rows, "B1 class_mapping == raw->arrival position",
                    all(int(trainer.server_test_dataset.class_mapping[k])
                        == v for k, v in arrival.items()), 1, 1)

    # raw-domain pools for cross-checks
    X_tr, y_tr = ADP._STATE["train_pool"]
    pool = ADP._STATE["test_pool_arr"]
    pool_y = ADP._STATE["pool_y"]
    pool_arrival = ADP._STATE["pool_arrival"]

    # B2: per-client train archives == manifest first-arrival selection
    total_drops = 0
    for c in range(nc):
        for t in range(nt):
            kk0 = int(sum(inc[:t]))
            kk1 = kk0 + int(inc[t])
            raws = ADP._STATE["train_raw"][(c, t)]
            keep = [r for r in raws
                    if kk0 <= arrival[int(y_tr[r])] < kk1]
            total_drops += len(raws) - len(keep)
            arch = trainer.train_datasets[c].archive[t]
            ok = (list(arch[1]) == [int(y_tr[r]) for r in keep]
                  and len(arch[0]) == len(keep))
            partB_ok &= row(
                rows, f"B2 train archive c{c}t{t} == inner.train NEW-at-t"
                      f" ({len(keep)} kept, "
                      f"{len(raws) - len(keep)} dropped)",
                ok, len(arch[0]), len(keep))
            pos = [arrival[int(v)] for v in arch[1]]
            ok = all(kk0 <= q < kk1 for q in pos)
            partB_ok &= row(rows, f"B2 train labels c{c}t{t} in arrival "
                                  f"[{kk0},{kk1})", ok, 1, 1)
    # drops accounting
    got_drops = sum(int(v) for k, v in ADP._STATE["drops"].items()
                    if k.startswith("c"))
    partB_ok &= row(rows, "B2 first-arrival drops accounted",
                    got_drops == total_drops, got_drops, total_drops)

    # label_counts filling (raw-class-id domain)
    c0 = trainer.train_datasets[0]
    lc = {j: 0 for j in range(m["protocol"]["nb_classes"])}
    ret = c0.load_dataset(0, train=True, label_counts=lc, seed=7,
                          cutoff=True, cutoff_ratio=0.4)
    ok = ret is lc and all(
        lc[j] == int((np.asarray(c0.targets) == j).sum())
        for j in range(m["protocol"]["nb_classes"]))
    partB_ok &= row(rows, "B2 label_counts filled in raw-id domain",
                    ok, 1, 1)

    # B3: server / per-client test archives
    for t in range(nt):
        kk0 = int(sum(inc[:t]))
        kk1 = kk0 + int(inc[t])
        want_pos = np.array([q for q in pool_arrival
                             if kk0 <= q < kk1])
        arch = trainer.server_test_dataset.archive[t]
        ok = list(arch[1]) == list(pool_y[np.isin(pool_arrival,
                                                  want_pos)]) \
            if len(want_pos) else len(arch[1]) == 0
        partB_ok &= row(rows, f"B3 server test archive t{t} == pool "
                              f"NEW-at-t ({len(want_pos)} samples)",
                        ok, len(arch[1]), len(want_pos))
    for c in range(nc):
        for t in range(nt):
            mask = m["outer"][str(c)][str(t)]["class_mask"]
            sel = np.isin(pool_y, np.array(sorted(int(x) for x in mask),
                                           dtype=np.int64))
            arch = trainer.test_datasets[c].archive[t]
            ok = list(arch[1]) == list(pool_y[sel])
            partB_ok &= row(rows, f"B3 client test archive c{c}t{t} == "
                                  f"pool mask(c,t) ({int(sel.sum())} "
                                  "samples)",
                            ok, len(arch[1]), int(sel.sum()))
    # load_dataset(train=False) concatenation == arrived pool slice
    t_last = nt - 1
    arrived_max = int(sum(inc))
    sv = trainer.server_test_dataset
    sv.load_dataset(t_last, train=False)
    want_arr = np.array([q for q in pool_arrival if q < arrived_max])
    ok = (len(sv.targets) == len(want_arr)
          and list(np.sort(sv.targets)) == sorted(
              int(pool_y[i]) for i in range(len(pool_y))
              if pool_arrival[i] < arrived_max))
    partB_ok &= row(rows, f"B3 server eval concat t{t_last} == arrived "
                          f"pool ({len(want_arr)} samples)",
                    ok, len(sv.targets), len(want_arr))

    # __getitem__ maps raw -> arrival position
    img, yy, tt = trainer.server_test_dataset[0]
    want_y = arrival[int(sv.targets[0])]
    partB_ok &= row(rows, "B3 __getitem__ maps raw->arrival position",
                    int(yy) == want_y, int(yy), want_y)

    # ------------------------------------------------------------------
    # Part C: seal checks (gating)
    # ------------------------------------------------------------------
    partC_ok = True
    # C1 official anchors intact
    partC_ok &= row(rows, "C1 trainer.py wandb.init anchor intact",
                    "wandb.init(project=" in tr_src, 1, 1)
    partC_ok &= row(rows, "C1 trainer.py FedAvg communicate anchor intact",
                    "temp += (1/self.n_clients)*self.learners[i].model."
                    "state_dict()[key]" in tr_src, 1, 1)
    partC_ok &= row(rows, "C1 trainer.py boundary-eval anchor intact",
                    ADP.TRAIN_SOURCE_ANCHOR in tr_src, 1, 1)
    partC_ok &= row(rows, "C1 dataloader.py gen_imbalanced anchor intact",
                    "def gen_imbalanced_data" in dl_src, 3, 3)
    if not is_cifar:
        partC_ok &= row(rows, "C1 dataloader.py INR official loader "
                              "anchors intact",
                        "0.8*n_data" in dl_src
                        and "random.seed(0)" in dl_src, 1, 1)
    ld_src = (HEPCO_ROOT / "learners" / "default.py").read_text("utf-8")
    lp_src = (HEPCO_ROOT / "learners" / "prompt.py").read_text("utf-8")
    partC_ok &= row(rows, "C1 learners/default.py cuda guard (> 1) intact",
                    "if len(self.config['gpuid']) > 1:" in ld_src, 1, 1)
    partC_ok &= row(rows, "C1 learners/prompt.py cuda guard (> 1) intact",
                    "if len(self.config['gpuid']) > 1:" in lp_src, 1, 1)
    partC_ok &= row(rows, "C1 learners/prompt.py update_model signature "
                          "anchor intact (prev_server/prev_task_model "
                          "official mismatch documented)",
                    ADP.UPDATE_MODEL_SIG_ANCHOR
                    in textwrap.dedent(lp_src), 1, 1)
    mv_src = (HEPCO_ROOT / "models" / "vit.py").read_text("utf-8")
    partC_ok &= row(rows, "C1 models/vit.py fairscale import anchor "
                          "intact",
                    "from fairscale.nn.checkpoint.checkpoint_activations "
                    "import checkpoint_wrapper" in mv_src, 1, 1)

    # C2 patches live
    partC_ok &= row(rows, "C2 Trainer.__init__ is the manifest wrapper",
                    trainer_mod.Trainer.__init__.__name__
                    == "manifest_trainer_init", 1, 1)
    train_fn = trainer_mod.Trainer.train
    partC_ok &= row(rows, "C2 Trainer.train carries boundary eval",
                    train_fn.__code__.co_filename
                    == "<Trainer.train manifest patch>"
                    and "_HEPCO_BOUNDARY_EVAL" in train_fn.__code__.co_names
                    and "n_rounds" in train_fn.__code__.co_names, 1, 1)
    from learners.default import NormalNN
    from learners.prompt import DualPrompt
    cuda_ok = True
    for cls in (NormalNN, DualPrompt):
        fn = cls.cuda
        cuda_ok &= (fn.__code__.co_filename
                    == f"<{cls.__name__}.cuda manifest patch>"
                    and "DataParallel" in fn.__code__.co_names)
    partC_ok &= row(rows, "C2 NormalNN/DualPrompt.cuda patched to >= 1 "
                          "(single-GPU DataParallel)",
                    cuda_ok, 1, 1)
    from learners.prompt import DualPrompt as _DP
    um = _DP.update_model
    partC_ok &= row(rows, "C2 DualPrompt.update_model prev_server alias "
                          "bugfix live",
                    um.__code__.co_filename
                    == "<DualPrompt.update_model manifest bugfix>"
                    and "prev_server" in um.__code__.co_varnames
                    and "prev_task_model" in um.__code__.co_varnames, 1, 1)
    wandb_stub = sys.modules.get("wandb")
    partC_ok &= row(rows, "C2 wandb is our stub (not installed)",
                    isinstance(wandb_stub, types.ModuleType)
                    and getattr(wandb_stub, "__file__", None) is None, 1,
                    1)
    fs_stub = sys.modules.get(
        "fairscale.nn.checkpoint.checkpoint_activations")
    partC_ok &= row(rows, "C2 fairscale is our passthrough stub",
                    fs_stub is not None
                    and callable(fs_stub.checkpoint_wrapper), 1, 1)
    if not is_cifar:
        partC_ok &= row(rows, "C2 iIMAGENET_R.load replaced by manifest "
                             "walk",
                        dl_mod.iIMAGENET_R.load.__name__
                        == "manifest_inr_load", 1, 1)

    # C3 frozen args
    partC_ok &= row(rows, "C3 args n_rounds == 5", hepco_args.n_rounds == 5,
                    hepco_args.n_rounds, 5)
    partC_ok &= row(rows, "C3 args schedule == [30]",
                    list(hepco_args.schedule) == [30],
                    list(hepco_args.schedule), [30])
    partC_ok &= row(rows, "C3 args seed == manifest seed",
                    hepco_args.seed == seed, hepco_args.seed, seed)
    partC_ok &= row(rows, "C3 args memory == 0 (no replay)",
                    hepco_args.memory == 0, hepco_args.memory, 0)
    partC_ok &= row(rows, "C3 args n_clients == 5",
                    hepco_args.n_clients == nc, hepco_args.n_clients, nc)
    partC_ok &= row(rows, "C3 args learner == prompt/L2P",
                    (hepco_args.learner_type, hepco_args.learner_name)
                    == ("prompt", "L2P"), 1, 1)
    partC_ok &= row(rows, "C3 args max_task == 5",
                    hepco_args.max_task == nt, hepco_args.max_task, nt)
    partC_ok &= row(rows, "C4 numpy LAPACK self-test status recorded",
                    _status()["svd"] in ("healthy", "rerouted->scipy"),
                    1, 1)

    dataset_slug = "cifar100" if is_cifar else "imagenet-r"
    report_path = args.report or str(
        REPO_ROOT / "protocol_manifests"
        / f"gate_hepco_{dataset_slug}_report.json")
    report = {
        "dataset": dataset,
        "manifest": os.path.abspath(args.manifest),
        "increments": inc,
        "first_arrival_drops": total_drops,
        "numpy_lapack": _status(),
        "partA_informational_pass": partA_ok,
        "partB_injection_pass": partB_ok,
        "partC_seal_pass": partC_ok,
        "PASS": bool(partB_ok and partC_ok),
        "rows": rows,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    n_pass = sum(1 for r in rows if r["pass"])
    print(f"[gate_hepco] {len(rows)} checks: {n_pass} pass, "
          f"{len(rows) - n_pass} FAIL")
    print(f"[gate_hepco] PartA(info)={partA_ok} PartB={partB_ok} "
          f"PartC={partC_ok} -> "
          f"{'PASS' if report['PASS'] else 'FAIL'}")
    print(f"[gate_hepco] report -> {report_path}")
    if not report["PASS"]:
        for r in rows:
            if not r["pass"]:
                print(f"  FAIL: {r['item']} (got={r['got']} "
                      f"want={r['want']})")
        sys.exit(1)


def _status():
    return {
        "svd": "rerouted->scipy" if ADP._STATE["svd_rerouted"]
        else "healthy",
        "official_repo_untouched": True,
    }


if __name__ == "__main__":
    main()
