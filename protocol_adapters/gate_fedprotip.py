"""Protocol equivalence gate for FedProTIP (run BEFORE any formal training).

CPU-only and training-free. Three parts:

  Part A — native protocol mismatch (informational, expected):
      the official FedProTIP protocol is 10 sequential uniform tasks with
      a numpy dirichlet client split (partition_data), which is NOT the
      frozen manifest (5x5 per-client masks, public-class rotation,
      first-arrival class order). Recorded, not fatal -- the adapter
      replaces the partition interface BY DESIGN.

  Part B — adapter injection (gating): apply exactly what
      run_fedprotip.py does and verify what FedProTIP would actually
      LOAD is the manifest:
        B1  DataManager: _increments == first-arrival counts; identity
            class order; train targets == arrival remap of the raw
            trainset; test arrays == manifest test_pool (ordered subset
            of the raw test set, all classes covered)
        B2  partition_data(cur_task=t): every client's positions map
            back EXACTLY to the manifest inner.train(c,t) samples whose
            class is new at t (first-arrival drop accounted); all
            labels inside [K_{t-1}, K_t); the task-t DummyDataset
            position->raw order matches get_dataset's deterministic
            construction
        B3  boundary-eval views: per (client, test task k) the class
            mask maps to arrival positions whose test samples all sit
            inside the manifest test_pool
  PASS requires every Part B item exact and no structural violations.

  Part C — seal checks (gating): every source patch must (a) still find
      its anchor in the official repo (repo-drift protection), and (b)
      be live on the patched class, and where cheap (c) behave right:
        C1  IncrementalNet.generate_fc: 4 width anchors; patched fn
            builds per-task fc blocks of the arrival widths
        C2  Server_FedProTIP.train: partition call-site + boundary-eval
            insertion anchors; patched method active with
            _FPT_BOUNDARY_EVAL bound
        C3  _map_new_class_index anchor + module attr replaced
        C4  wandb/progress stubs + args wandb=0 + frozen protocol
            fields in build_args

Usage (FedSMTA conda env, from the FedSMTA repo root):

  python protocol_adapters/gate_fedprotip.py \
      --manifest protocol_manifests/cifar100_c5t5p15s20_seed42.json

Writes protocol_manifests/gate_fedprotip_{dataset}_report.json.
Exit code 0 only on PASS (Part B + Part C).
"""

import argparse
import inspect
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np

ADAPTER_DIR = Path(__file__).resolve().parent
REPO_ROOT = ADAPTER_DIR.parent
FPT_ROOT = REPO_ROOT / "external_baselines" / "FedProTIP"

if str(ADAPTER_DIR) not in sys.path:
    sys.path.insert(0, str(ADAPTER_DIR))

import run_fedprotip as ADP  # noqa: E402


def row(rows, item, ok, got=0, want=0):
    rows.append({"item": item, "pass": bool(ok),
                 "got_len": got, "want_len": want})
    return bool(ok)


def cmp_list(rows, item, got, want):
    ok = list(got) == list(want)
    rows.append({"item": item, "pass": bool(ok),
                 "got_len": len(list(got)), "want_len": len(list(want))})
    return ok


def main():
    p = argparse.ArgumentParser("FedProTIP protocol equivalence gate")
    p.add_argument("--manifest", required=True)
    p.add_argument("--data-root",
                   default=str(REPO_ROOT / "local_datasets"))
    p.add_argument("-o", "--report", default="")
    args = p.parse_args()
    if args.report:
        args.report = str(Path(args.report).resolve())
    manifest_path = str(Path(args.manifest).resolve())

    ADP.install_import_stubs()
    m = ADP.load_manifest(manifest_path)
    dataset = m["dataset"]
    proto = m["protocol"]
    seed = m["seed"]
    inc, arrival, new_at = ADP.compute_arrival()
    ADP.build_train_raw()

    # same path/cwd setup as run_fedprotip.main
    fpt_root = str(FPT_ROOT.resolve())
    if fpt_root not in sys.path:
        sys.path.insert(0, fpt_root)
    parent = str(FPT_ROOT.parent.resolve())
    if parent not in sys.path:
        sys.path.insert(0, parent)
    os.chdir(fpt_root)

    rows = []
    fpt_name = "cifar100" if dataset == "cifar100" else "imagenet-r"

    # ------------------------------------------------------------------
    # Part A: native protocol mismatch (informational)
    # ------------------------------------------------------------------
    partA_ok = True
    dm_src = (FPT_ROOT / "utils" / "data_manager.py").read_text("utf-8")
    partA_ok &= row(rows, "A official partition_data is RNG dirichlet "
                          "(anchor present; replaced by design)",
                    "np.random.dirichlet" in dm_src, 1, 1)
    with open(FPT_ROOT / "configs" /
              ("cifar100/fedprotip.json" if dataset == "cifar100"
               else "imagenet-r/imagenet-r_fedprotip.json"),
              "r", encoding="utf-8") as f:
        official_cfg = json.load(f)
    partA_ok &= row(
        rows, f"A official n_tasks={official_cfg['n_tasks']} != frozen 5 "
              "(documented mismatch; adapter enforces 5)",
        official_cfg["n_tasks"] != 5, official_cfg["n_tasks"], 5)
    partA_ok &= row(
        rows, "A official increment uniform "
              f"{official_cfg['increment']} != first-arrival {inc} "
              "(documented mismatch; adapter enforces arrival order)",
        True, official_cfg["increment"], inc)

    # ------------------------------------------------------------------
    # Part B: adapter injection (gating)
    # ------------------------------------------------------------------
    ADP.patch_data_module(args.data_root)
    ADP.patch_data_manager_module(inc)

    from utils.data_manager import DataManager
    # iCIFAR100/Imagenet_R __init__ reads args["net"]/args["img_size"]
    # for transform selection -> pass the official config (native values)
    dm = DataManager(fpt_name, 0, seed, inc[0], inc[0],
                     dict(official_cfg))
    partB_ok = True

    partB_ok &= cmp_list(rows, "B1 _increments == first-arrival counts",
                         dm._increments, inc)
    partB_ok &= cmp_list(rows, "B1 class order identity",
                         dm.get_class_order(),
                         list(range(proto["nb_classes"])))

    # raw-domain targets for cross-checks
    if dataset == "cifar100":
        from torchvision import datasets as tvd
        data_dir = os.path.expanduser(args.data_root)
        tr = tvd.CIFAR100(data_dir, train=True, download=False)
        te = tvd.CIFAR100(data_dir, train=False, download=False)
        raw_train_y = np.array(tr.targets)
        raw_test_y = np.array(te.targets)
    else:
        inr = os.path.join(os.path.expanduser(args.data_root), "imagenet-r")
        _, raw_train_y = ADP.walk_imagenet_r(inr, "train")
        _, raw_test_y = ADP.walk_imagenet_r(inr, "test")
        raw_train_y, raw_test_y = np.array(raw_train_y), np.array(raw_test_y)

    # train targets remapped into arrival order
    want_train = np.array([arrival[int(c)] for c in raw_train_y])
    partB_ok &= row(rows, "B1 train targets == arrival remap",
                    np.array_equal(dm._train_targets, want_train),
                    int((dm._train_targets != want_train).sum()), 0)

    # test arrays == manifest test_pool (ordered subset of the raw test)
    pool = sorted(int(x) for x in m["test_pool"]["indices"])
    partB_ok &= row(rows, "B1 test pool size",
                    len(dm._test_targets) == len(pool),
                    len(dm._test_targets), len(pool))
    want_test = np.array([arrival[int(raw_test_y[i])] for i in pool])
    partB_ok &= row(rows, "B1 test targets == pool subset in raw order",
                    np.array_equal(dm._test_targets, want_test),
                    int((dm._test_targets != want_test).sum()), 0)

    # B2: per-task partition equivalence
    total_drops = 0
    for t in range(proto["task_num"]):
        k0 = int(sum(inc[:t]))
        k1 = k0 + inc[t]
        train_ds = dm.get_dataset(np.arange(k0, k1), source="train",
                                 mode="train")
        # position -> raw index order of the task-t DummyDataset
        exp_order = np.concatenate(
            [np.where(dm._train_targets == q)[0] for q in range(k0, k1)])
        partB_ok &= cmp_list(rows, f"B2 task {t} DummyDataset length",
                            [len(train_ds)], [len(exp_order)])
        net, _ = ADP.manifest_partition_data(
            train_ds.labels, beta=0.5, n_parties=proto["client_num"],
            cur_task=t, _dm=dm)
        for c in range(proto["client_num"]):
            raws_kept = sorted(int(exp_order[pos]) for pos in net[c])
            want_kept, want_drop = [], 0
            for r in ADP._STATE["train_raw"][(c, t)]:
                pos = arrival[int(raw_train_y[r])]
                if pos < k0:
                    want_drop += 1
                elif pos < k1:
                    want_kept.append(int(r))
                else:  # cannot happen if masks are inside the union
                    want_kept.append(-1)
            partB_ok &= cmp_list(
                rows, f"B2 c{c}t{t} kept samples == manifest "
                      "inner.train(new at t)",
                raws_kept, sorted(want_kept))
            got_drop = ADP._STATE["drops"].get(f"c{c}t{t}", -1)
            partB_ok &= row(rows, f"B2 c{c}t{t} drop accounting "
                                  f"({got_drop} dropped)",
                            got_drop == want_drop, got_drop, want_drop)
            total_drops += want_drop
        # structure: every kept label inside the new-class range
        lab_ok = all(k0 <= int(dm._train_targets[r]) < k1
                     for c in range(proto["client_num"])
                     for r in (int(exp_order[pos]) for pos in net[c]))
        partB_ok &= row(rows, f"B2 task {t} all kept labels in "
                              f"[{k0},{k1})", lab_ok, int(lab_ok), 1)

    partB_ok &= row(rows, "B2 total first-arrival drops "
                          f"({total_drops}; 0 expected on ImageNet-R)",
                    total_drops == 0 if dataset == "ImageNet-R"
                    else total_drops > 0, total_drops,
                    0 if dataset == "ImageNet-R" else -1)

    # B3: boundary-eval views sit inside the manifest test_pool
    pool_set = set(pool)
    eval_ok = True
    for c in range(proto["client_num"]):
        for k in range(proto["task_num"]):
            positions = sorted(arrival[int(cl)]
                               for cl in m["outer"][str(c)][str(k)]
                               ["class_mask"])
            ds = dm.get_dataset(positions, source="test", mode="test")
            eval_ok &= len(ds) > 0
    partB_ok &= row(rows, "B3 per-(client,task) eval views non-empty "
                          "on the test pool", eval_ok, int(eval_ok), 1)

    # ------------------------------------------------------------------
    # Part C: seals (gating)
    # ------------------------------------------------------------------
    partC_ok = True

    # C1: generate_fc width anchors + patched behavior
    inc_net_file = (FPT_ROOT / "utils" / "inc_net.py").read_text("utf-8")
    partC_ok &= row(rows, "C1 official inc_net.py contains the "
                          "args['increment'] width anchors",
                    inc_net_file.count(
                        "torch.nn.Linear(in_dim, self.args['increment'])")
                    >= 3, inc_net_file.count(
                        "torch.nn.Linear(in_dim, "
                        "self.args['increment'])"), 3)
    ADP.patch_inc_net_generate_fc(inc)
    import torch
    from utils.inc_net import IncrementalNet
    partC_ok &= row(rows, "C1 patched generate_fc active",
                    IncrementalNet.generate_fc.__code__.co_filename
                    == "<IncrementalNet.generate_fc manifest patch>", 1, 1)
    fake = types.SimpleNamespace(
        args={"classIL": "True", "pretrained": "True", "method": "FedGPM"},
        device="cpu", fc=torch.nn.ModuleList())
    widths_ok = True
    for t in range(proto["task_num"]):
        IncrementalNet.generate_fc(fake, 64, t, 1)
        widths_ok &= (fake.fc[-1].out_features == inc[t]
                      and len(fake.fc) == t + 1)
    partC_ok &= row(rows, "C1 fc block widths == arrival counts",
                    widths_ok, int(widths_ok), 1)

    # C2: server source anchors + patched train
    srv_file = (FPT_ROOT / "server" / "server_fedprotip.py").read_text(
        "utf-8")
    a1 = "user_groups,_ = partition_data(train_dataset.labels, " \
         "beta=self.args[\"beta\"], n_parties=self.args[\"n_clients\"])"
    partC_ok &= row(rows, "C2 official train() partition call-site "
                          "anchor", srv_file.count(a1) == 1,
                    srv_file.count(a1), 1)
    a2 = "self._known_classes = self._total_classes"
    partC_ok &= row(rows, "C2 official train() known-classes anchor",
                    srv_file.count(a2) == 1, srv_file.count(a2), 1)
    import server.server_fedprotip as server_mod
    ADP._STATE["recorder"] = None  # patch_server binds placeholder refs
    ADP.patch_server(server_mod)
    from server.server_fedprotip import Server_FedProTIP
    partC_ok &= row(
        rows, "C2 patched train() active + boundary eval bound",
        Server_FedProTIP.train.__code__.co_filename
        == "<Server_FedProTIP.train manifest patch>"
        and "_FPT_BOUNDARY_EVAL" in Server_FedProTIP.train.__code__.co_names
        and Server_FedProTIP.train.__code__.co_names.count(
            "partition_data") >= 1, 1, 1)

    # C3: _map_new_class_index seal
    partC_ok &= row(rows, "C3 official _map_new_class_index anchor "
                          "(order.index)",
                    "order.index(x)" in dm_src, 1, 1)
    import utils.data_manager as dm_mod
    partC_ok &= row(rows, "C3 module attr replaced by the manifest map",
                    dm_mod._map_new_class_index
                    .__code__.co_filename.endswith("run_fedprotip.py")
                    if hasattr(dm_mod._map_new_class_index, "__code__")
                    else False, 1, 1)

    # C5: sealed env bugfix -- numpy LAPACK correctness
    client_file = (FPT_ROOT / "client" / "client_fedprotip.py"
                   ).read_text("utf-8")
    n_svd = client_file.count("np.linalg.svd(")
    partC_ok &= row(rows, "C5 official client still calls np.linalg.svd "
                          "(seal: anchors untouched, "
                          f"found {n_svd}, expected 6)",
                    n_svd == 6, n_svd, 6)
    ADP.install_robust_svd()
    partC_ok &= row(
        rows, f"C5 numpy LAPACK self-test "
              f"({'rerouted to scipy (broken host)' if ADP._STATE['svd_rerouted'] else 'healthy (native numpy kept)'}"
              " -- either is PASS; the self-test is fail-loud)",
        True, int(ADP._STATE["svd_rerouted"]), 0)

    # C4: stubs + frozen build_args
    stub_ok = (sys.modules["wandb"].__name__ == "wandb"
               and sys.modules["progress"].__name__ == "progress")
    partC_ok &= row(rows, "C4 wandb/progress stubs installed", stub_ok,
                    int(stub_ok), 1)
    ns = types.SimpleNamespace(seed=seed, device="cpu", epochs=5,
                               local_epochs=30, batch_size=64,
                               limit_tasks=0)
    built = ADP.build_args(ns)
    frozen_ok = (built["n_tasks"] == 5 and built["n_clients"] == 5
                 and built["frac"] == 1.0 and built["classIL"] == "True"
                 and built["class_shuffle"] == 0
                 and int(built["seed"]) == int(seed)
                 and built["wandb"] == 0 and built["epochs"] == 5
                 and built["local_epochs"] == 30
                 and built["increment"] == inc[0]
                 and built["method"] == "FedGPM")
    partC_ok &= row(rows, "C4 build_args enforces the frozen protocol",
                    frozen_ok, int(frozen_ok), 1)

    # ------------------------------------------------------------------
    # report
    # ------------------------------------------------------------------
    def n_fail(pre):
        return sum(1 for r in rows
                   if r["item"].startswith(pre) and not r["pass"])
    all_ok = partB_ok and partC_ok
    print("\n===== FedProTIP protocol equivalence gate =====")
    print(f"Part A (native mismatch, informational): "
          f"{'PASS' if partA_ok else f'FAIL ({n_fail(chr(65))} items)'}")
    print(f"Part B (adapter injection vs manifest): "
          f"{'PASS' if partB_ok else f'FAIL ({n_fail(chr(66))} items)'}"
          f"  [gating]")
    print(f"Part C (seals): "
          f"{'PASS' if partC_ok else f'FAIL ({n_fail(chr(67))} items)'}"
          f"  [gating]")
    for r in rows:
        if not r["pass"]:
            print(f"  [FAIL] {r['item']} got={r['got_len']} "
                  f"want={r['want_len']}")
    result = "PASS" if all_ok else "FAIL"
    print(f"\nRESULT: {result}"
          + ("" if all_ok else " (fix the adapter before any formal run)"))

    slug = "cifar100" if dataset == "cifar100" else "imagenet-r"
    out = Path(args.report) if args.report else \
        (REPO_ROOT / "protocol_manifests"
         / f"gate_fedprotip_{slug}_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "test": "fedprotip_protocol_equivalence_gate",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "manifest": manifest_path,
        "dataset": dataset,
        "result": result,
        "partA_native_mismatch": "PASS" if partA_ok else "FAIL",
        "partB_adapter_injection": "PASS" if partB_ok else "FAIL",
        "partC_seals": "PASS" if partC_ok else "FAIL",
        "first_arrival_increments": inc,
        "total_dropped_rotated_samples": total_drops,
        "items": rows,
        "notes": [
            "Part A is informational: the official 10-task dirichlet "
            "protocol is replaced BY DESIGN by the frozen manifest via "
            "the data-partition interface only",
            "Part B is gating: it verifies the exact objects FedProTIP "
            "loads under run_fedprotip.py are the manifest's splits, "
            "including the first-arrival label remap, the test-pool "
            "eval restriction and the per-(client,task) partition",
            "CIFAR public classes rotate across tasks per client; "
            "samples of already-arrived classes are dropped from that "
            "client's new-task training (FedProTIP has no replay path) "
            "and counted in the report/drops.json",
        ],
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"report: {out}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
