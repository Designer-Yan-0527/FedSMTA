"""Protocol equivalence gate for the GLFC manifest adapter
(run_glfc.py). Three parts, same policy as gate_fedmgp/gate_hepco:

  Part A (informational): the OFFICIAL GLFC protocol is documented as-is
    (125 shared-dataset clients, client churn, partial participation,
    uniform task_size growth, 6-class random.sample partition, raw-id
    interval eval) and the official interfaces the adapter duck-types
    are exercised on CPU (iCIFAR100.getTrainData/get_image_class,
    network.Incremental_learning, LeNet).

  Part B (adapter injection): with run_glfc's patches applied, every
    data structure entering GLFC must equal the frozen manifest ground
    truth -- per-client train arrays/labels (arrival space), exemplar
    concatenation, per-class ownership, the test-pool eval arrays, the
    boundary subsets, the patched beforeTrain bookkeeping and the
    patched fl_main source.

  Part C (bugfix seals): every anchor the adapter patches is still
    present verbatim in the OFFICIAL files (official repo untouched;
    patches fail loud if the repo drifts).

Report: protocol_manifests/gate_glfc_{cifar100,imagenet-r}_report.json
PASS = Part B AND Part C fully pass (Part A is informational).
"""

import argparse
import inspect
import json
import os
import sys
import types
from pathlib import Path

import numpy as np

ADAPTER_DIR = Path(__file__).resolve().parent
REPO_ROOT = ADAPTER_DIR.parent
if str(ADAPTER_DIR) not in sys.path:
    sys.path.insert(0, str(ADAPTER_DIR))

import run_glfc as ADP  # noqa: E402

GLFC_SRC = ADP.GLFC_ROOT


def row(rows, item, ok, got=0, want=0):
    rows.append({"item": item, "pass": bool(ok),
                 "got": got, "want": want})
    return bool(ok)


def main():
    p = argparse.ArgumentParser("GLFC protocol equivalence gate")
    p.add_argument("--manifest", required=True)
    p.add_argument("--data-root", default=str(REPO_ROOT / "local_datasets"))
    p.add_argument("-o", "--report", default="")
    args = p.parse_args()

    ADP.install_import_stubs()
    m = ADP.load_manifest(args.manifest)
    dataset = m["dataset"]
    seed = m["seed"]
    nc, nt = m["protocol"]["client_num"], m["protocol"]["task_num"]
    nb = m["protocol"]["nb_classes"]
    inc, arrival, new_at = ADP.compute_arrival()
    ADP.build_train_raw()
    ADP.load_base(args.data_root)
    ADP.build_datasets()

    is_cifar = dataset == "cifar100"
    inc_expected = ([32, 26, 14, 16, 12] if is_cifar else [40] * 5)
    rows = []

    # ------------------------------------------------------------------
    # Part A: native protocol (informational) + official-interface sanity
    # ------------------------------------------------------------------
    partA_ok = True
    fl_src = (GLFC_SRC / "fl_main.py").read_text("utf-8")
    glfc_src = (GLFC_SRC / "GLFC.py").read_text("utf-8")
    fed_src = (GLFC_SRC / "Fed_utils.py").read_text("utf-8")
    icifar_src = (GLFC_SRC / "iCIFAR100.py").read_text("utf-8")
    proxy_src = (GLFC_SRC / "ProxyServer.py").read_text("utf-8")

    partA_ok &= row(rows, "A1 official: 125 shared-dataset clients",
                    "for i in range(125):" in fl_src, 1, 1)
    partA_ok &= row(rows, "A2 official: churn adds task_size clients/task",
                    "overall_client + args.task_size" in fl_src, 1, 1)
    partA_ok &= row(
        rows, "A3 official: partial participation (local_clients sample)",
        "random.sample(range(num_clients), args.local_clients)" in fl_src,
        1, 1)
    partA_ok &= row(rows, "A4 official: uniform task_size head growth",
                    "classes_learned += args.task_size" in fl_src, 1, 1)
    partA_ok &= row(
        rows, "A5 official: 6-class random.sample partition in beforeTrain",
        "random.sample([x for x in range(self.numclass - self.task_size, "
        "self.numclass)], 6)" in glfc_src, 1, 1)
    partA_ok &= row(
        rows, "A6 official: raw-id interval eval [0, task_size*(t+1))",
        "test_dataset.getTestData([0, task_size * (task_id + 1)])"
        in fed_src, 1, 1)
    partA_ok &= row(rows, "A7 official: single shared global trainset",
                    "iCIFAR100('dataset', transform=train_transform, "
                    "download=True)" in fl_src, 1, 1)

    # official interface semantics on real local data (CPU)
    sys.path.insert(0, str(GLFC_SRC))
    os.environ.setdefault("MPLBACKEND", "Agg")
    from iCIFAR100 import iCIFAR100
    from torchvision import transforms
    root = os.path.expanduser(args.data_root)
    tt = transforms.ToTensor()
    ods = iCIFAR100(root, transform=tt, test_transform=tt,
                    download=False) if is_cifar else None
    if is_cifar:
        ods.getTrainData([0, 2], [], [])
        n02 = int(np.isin(np.asarray(ods.targets), [0, 2]).sum())
        ok = (len(ods) == n02
              and set(np.unique(ods.TrainLabels).tolist()) == {0, 2})
        partA_ok &= row(rows, "A8 native iCIFAR100.getTrainData semantics",
                        ok, len(ods), n02)
        ods.get_image_class(0)
        n0 = int((np.asarray(ods.targets) == 0).sum())
        partA_ok &= row(rows, "A8b native get_image_class(0) size",
                        len(ods.get_image_class(0)) == n0,
                        len(ods.get_image_class(0)), n0)
    else:
        partA_ok &= row(rows, "A8 native iCIFAR100 semantics (INR: n/a)",
                        True, 1, 1)

    import torch
    from ResNet import resnet18_cbam
    from myNetwork import network, LeNet
    net = network(10, resnet18_cbam())
    w0 = net.fc.weight.data.clone()
    net.Incremental_learning(20)
    ok = (net.fc.out_features == 20
          and torch.equal(net.fc.weight.data[:10], w0))
    partA_ok &= row(rows, "A9 native network.Incremental_learning",
                    ok, net.fc.out_features, 20)
    lenet = LeNet(num_classes=100)
    out = lenet(torch.zeros(2, 3, 32, 32))
    partA_ok &= row(rows, "A10 native LeNet forward shape",
                    tuple(out.shape) == (2, 100), tuple(out.shape),
                    (2, 100))

    # ------------------------------------------------------------------
    # Part B: adapter injection == manifest ground truth
    # ------------------------------------------------------------------
    partB_ok = True
    partB_ok &= row(rows, "B1 arrival increments",
                    list(inc) == inc_expected, inc, inc_expected)
    cum = ADP._STATE["cumulative"]
    partB_ok &= row(rows, "B2 cumulative head widths",
                    cum == [sum(inc[:t + 1]) for t in range(nt)],
                    cum, [sum(inc[:t + 1]) for t in range(nt)])

    base = ADP._STATE["base"]
    pool = sorted(int(x) for x in m["test_pool"]["indices"])
    pool_t = base["test_targets"][np.array(pool, dtype=np.int64)]

    starved_expected = {(0, 4): [ADP._STATE["arrival"][10]],
                        (2, 3): [ADP._STATE["arrival"][56]]} \
        if is_cifar else {}
    om_seen = {}
    for c in range(nc):
        for t in range(nt):
            ds = ADP._STATE["datasets"][c]
            mask = m["outer"][str(c)][str(t)]["class_mask"]
            kept = [int(cl) for cl in mask
                    if len(ds._own_by_class.get(int(cl), [])) > 0]
            om = [int(cl) for cl in mask if int(cl) not in kept]
            if om:
                om_seen[(c, t)] = sorted(
                    ADP._STATE["arrival"][cl] for cl in om)
            positions = sorted(arrival[cl] for cl in kept)
            ds.getTrainData(positions, [], [])
            raw_own = ADP._STATE["train_raw"][(c, t)]
            ok = (len(ds) == len(raw_own)
                  and sorted(np.unique(ds.TrainLabels).tolist())
                  == positions)
            ok &= all(int(x) in positions
                      for x in np.unique(ds.TrainLabels).tolist())
            partB_ok &= row(
                rows, f"B3 c{c}t{t} train labels/size (manifest space)",
                ok, (len(ds), sorted(set(ds.TrainLabels.tolist()))[:3]),
                (len(raw_own), positions[:3]))
            # spot equality: every label's block count matches the
            # client's own inner.train samples of that class
            raw = ADP._STATE["pos_to_raw"]
            own_targets = base["train_targets"][np.array(raw_own,
                                                          dtype=np.int64)]
            for pos in positions:
                n_own = int((own_targets == raw[pos]).sum())
                n_blk = int((ds.TrainLabels == pos).sum())
                if n_own != n_blk:
                    partB_ok &= row(
                        rows, f"B3b c{c}t{t} pos{pos} block count",
                        False, n_blk, n_own)
                    break

    partB_ok &= row(
        rows, "B3c zero-sample omissions == expected starved set",
        om_seen == starved_expected,
        {f"c{c}t{t}": v for (c, t), v in om_seen.items()},
        {f"c{c}t{t}": v for (c, t), v in starved_expected.items()})

    # exemplar concatenation (official order: exemplars first)
    ds0 = ADP._STATE["datasets"][0]
    mask0 = m["outer"]["0"]["0"]["class_mask"]
    pos0 = sorted(arrival[int(cl)] for cl in mask0)
    fake = np.zeros((3,) + base["train_data"].shape[1:],
                    dtype=base["train_data"].dtype)
    ds0.getTrainData(pos0, [fake], [pos0[0]])
    ok = (ds0.TrainLabels[0] == pos0[0]
          and len(ds0) == 3 + len(ADP._STATE["train_raw"][(0, 0)]))
    partB_ok &= row(rows, "B4 exemplar replay concatenation order",
                    ok, int(ds0.TrainLabels[0]), pos0[0])

    # per-(client, class) ownership follows the manifest exactly
    ok = True
    for c in range(nc):
        own = ADP._STATE["datasets"][c]._own_by_class
        for t in range(nt):
            for cl in m["outer"][str(c)][str(t)]["class_mask"]:
                n_own = int((base["train_targets"][
                    np.array(ADP._STATE["train_raw"][(c, t)],
                             dtype=np.int64)] == cl).sum())
                if int(len(own.get(int(cl), []))) != n_own:
                    ok = False
    partB_ok &= row(rows, "B5 per-(client,class) ownership == manifest",
                    ok, 1, 1)

    # server eval arrays: getTestData([0, K_t]) == pool samples of the
    # classes arrived by t, arrival-space labels, pool order
    tds = ADP._STATE["test_dataset"]
    for t in range(nt):
        arrived_raw = {raw for pos in range(ADP._STATE["cumulative"][t])
                       for raw in [ADP._STATE["pos_to_raw"][pos]]}
        want_n = int(np.isin(pool_t, sorted(arrived_raw)).sum())
        tds.getTestData([0, ADP._STATE["cumulative"][t]])
        ok = (len(tds) == want_n
              and sorted(np.unique(tds.TestLabels).tolist())
              == list(range(ADP._STATE["cumulative"][t])))
        partB_ok &= row(rows, f"B6 eval pool after task {t}",
                        ok, len(tds), want_n)

    # boundary subsets: subset(mask(c,k)) == pool samples of mask(c,k)
    for c in range(nc):
        for k in range(nt):
            mask = m["outer"][str(c)][str(k)]["class_mask"]
            positions = sorted(arrival[int(cl)] for cl in mask)
            tds.subset(positions)
            want_n = int(np.isin(pool_t, mask).sum())
            ok = (len(tds) == want_n
                  and sorted(np.unique(tds.TestLabels).tolist())
                  == positions)
            partB_ok &= row(rows, f"B7 boundary c{c}k{k}", ok, len(tds),
                            want_n)

    # patched GLFC_model.bookkeeping (CPU)
    glfc_mod = ADP.patch_glfc_module()
    ADP.patch_proxy_server(nb)
    from myNetwork import LeNet as _LeNet
    cli = glfc_mod.GLFC_model(
        ADP._STATE["cumulative"][0], resnet18_cbam(), 128,
        ADP._STATE["increments"][0], 2000, 1, 2.0,
        ADP._STATE["datasets"][0], -1, _LeNet(num_classes=nb))
    cli.beforeTrain(0, 1)
    mask00 = sorted(arrival[int(cl)] for cl in m["outer"]["0"]["0"]
                    ["class_mask"])
    ok = (cli.numclass == ADP._STATE["cumulative"][0]
          and cli.current_class == mask00
          and len(cli.train_loader.dataset)
          == len(ADP._STATE["train_raw"][(0, 0)]))
    partB_ok &= row(rows, "B8 patched beforeTrain task 0", ok,
                    cli.numclass, ADP._STATE["cumulative"][0])
    cli.beforeTrain(1, 1)
    mask01 = sorted(arrival[int(cl)] for cl in m["outer"]["0"]["1"]
                    ["class_mask"])
    ok = (cli.numclass == ADP._STATE["cumulative"][1]
          and cli.current_class == mask01
          and cli.last_class == mask00)
    partB_ok &= row(rows, "B9 patched beforeTrain task 1 advance", ok,
                    cli.numclass, ADP._STATE["cumulative"][1])

    train_src = ADP._STATE.get("train_patch_src", "")
    partB_ok &= row(
        rows, "B10 train LR hardcode parameterized",
        ("ep_g * self.epochs" in train_src
         and "ep_g * 20" not in train_src
         and glfc_mod.GLFC_model.train.__code__.co_filename
         == "<GLFC_model.train manifest patch>"),
        glfc_mod.GLFC_model.train.__code__.co_filename,
        "<GLFC_model.train manifest patch>")

    patched_fl = ADP.patch_fl_main_source(nb)
    ok_markers = [
        "_GLFC_ATTACH_TRANSFORMS(train_transform, test_transform)",
        "_GLFC_STATE['datasets'][i]",
        "pass  # [protocol-adapter] frozen: fixed 5-client cohort",
        "classes_learned += _GLFC_STATE['increments'][task_id]",
        "clients_index = list(range(num_clients))",
        "_GLFC_GLOBAL_EVAL(model_g, test_dataset, task_id, ep_g, "
        "args.tasks_global, args.device)",
        "output_dir = str(_GLFC_STATE['run_dir'])",
    ]
    if nb != 100:
        ok_markers.append("LeNet(num_classes=_GLFC_STATE['nb_classes'])")
    ok = all(x in patched_fl for x in ok_markers) and all(
        x not in patched_fl for x in [
            "for i in range(125):",
            "random.sample(range(num_clients), args.local_clients)",
            "overall_client + args.task_size",
            "classes_learned += args.task_size",
        ])
    partB_ok &= row(rows, "B11 patched fl_main source markers", ok, 1, 1)

    import ProxyServer as proxy_mod
    recon_src = ADP._STATE.get("proxy_patch_src", "")
    partB_ok &= row(
        rows, "B12 proxyServer.reconstruction nb_classes patched",
        "_GLFC_STATE['nb_classes']" in recon_src
        and "np.zeros((1, 100))" not in recon_src
        and proxy_mod.proxyServer.reconstruction.__code__.co_filename
        == "<proxyServer.reconstruction manifest patch>", 1, 1)
    partB_ok &= row(
        rows, "B12b proxyServer.reconstruction C' cpu reroute",
        "recon_model = recon_model.cpu()" in recon_src
        and "g.detach().cpu()" in recon_src
        and ".to(self.device)" not in recon_src
        and "model_to_device(recon_model, False, self.device)"
        not in recon_src, 1, 1)

    # ------------------------------------------------------------------
    # Part C: bugfix seals (official repo untouched, anchors verbatim)
    # ------------------------------------------------------------------
    partC_ok = True
    for name, anchor, want in [
        ("C1 fl_main 125-client anchor", "for i in range(125):", 1),
        ("C1b fl_main churn anchor",
         "overall_client + args.task_size", 1),
        ("C1c fl_main participation anchor",
         "random.sample(range(num_clients), args.local_clients)", 1),
        ("C1d fl_main task_size growth anchor",
         "classes_learned += args.task_size", 1),
        ("C1e fl_main training_log anchor",
         "osp.join('./training_log', args.method", 1),
        ("C1f fl_main shared-dataset anchor",
         "iCIFAR100('dataset', transform=train_transform, download=True)",
         1),
    ]:
        partC_ok &= row(rows, name, fl_src.count(anchor) == want,
                        fl_src.count(anchor), want)
    partC_ok &= row(rows, "C2 GLFC.py ep_g*20 hardcode x3",
                    glfc_src.count("ep_g * 20") == 3,
                    glfc_src.count("ep_g * 20"), 3)
    partC_ok &= row(rows, "C2b GLFC.py beforeTrain random.sample anchor",
                    glfc_src.count(
                        "random.sample([x for x in range(self.numclass "
                        "- self.task_size, self.numclass)], 6)") == 1, 1, 1)
    partC_ok &= row(rows, "C3 ProxyServer (1,100) anchor",
                    proxy_src.count("np.zeros((1, 100))") == 1, 1, 1)
    partC_ok &= row(rows, "C3b ProxyServer range(100) anchor",
                    proxy_src.count("for label_i in range(100):") == 1,
                    1, 1)
    partC_ok &= row(rows, "C3c ProxyServer device anchors (C' seal)",
                    proxy_src.count(
                        "torch.randn((1, 3, 32, 32)).to(self.device)")
                    == 1
                    and proxy_src.count(
                        "model_to_device(recon_model, False, self.device)")
                    == 1
                    and proxy_src.count(
                        "nn.CrossEntropyLoss().to(self.device)") == 1
                    and proxy_src.count(
                        "grad_truth_temp = self.pool_grad[grad_index[0][j]]")
                    == 1, 1, 1)
    partC_ok &= row(rows, "C4 iCIFAR100.py untouched (getTrainData)",
                    "def getTrainData(self, classes, exemplar_set, "
                    "exemplar_label_set):" in icifar_src, 1, 1)
    partC_ok &= row(rows, "C4b iCIFAR100.py untouched (getTestData)",
                    "def getTestData(self, classes):" in icifar_src,
                    1, 1)
    partC_ok &= row(rows, "C4c iCIFAR100.py untouched (get_image_class)",
                    "def get_image_class(self,label):" in icifar_src,
                    1, 1)
    partC_ok &= row(rows, "C5 Fed_utils.py model_global_eval untouched",
                    "test_dataset.getTestData([0, task_size * (task_id "
                    "+ 1)])" in fed_src, 1, 1)
    partC_ok &= row(
        rows, "C6 numpy LAPACK self-test status recorded",
        ADP._STATE["svd_rerouted"] in (True, False), 1, 1)
    try:
        import cv2
        cv2_fail = False
        try:
            cv2.imread("x")
        except RuntimeError:
            cv2_fail = True
        partC_ok &= row(rows, "C7 cv2 stub importable + fail-loud",
                        cv2_fail, 1, 1)
    except Exception:
        partC_ok &= row(rows, "C7 cv2 stub importable + fail-loud",
                        False, 0, 1)

    dataset_slug = "cifar100" if is_cifar else "imagenet-r"
    report_path = args.report or str(
        REPO_ROOT / "protocol_manifests"
        / f"gate_glfc_{dataset_slug}_report.json")
    report = {
        "dataset": dataset,
        "manifest": os.path.abspath(args.manifest),
        "increments": inc,
        "drops": 0,
        "replay": "exemplar memory + proxy gradient-inversion rehearsal",
        "numpy_lapack": ("rerouted->scipy" if ADP._STATE["svd_rerouted"]
                         else "healthy"),
        "partA_informational_pass": partA_ok,
        "partB_injection_pass": partB_ok,
        "partC_seal_pass": partC_ok,
        "PASS": bool(partB_ok and partC_ok),
        "rows": rows,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    n_pass = sum(1 for r in rows if r["pass"])
    print(f"[gate_glfc] {len(rows)} checks: {n_pass} pass, "
          f"{len(rows) - n_pass} FAIL")
    print(f"[gate_glfc] PartA(info)={partA_ok} PartB={partB_ok} "
          f"PartC={partC_ok} -> "
          f"{'PASS' if report['PASS'] else 'FAIL'}")
    print(f"[gate_glfc] report -> {report_path}")
    if not report["PASS"]:
        for r in rows:
            if not r["pass"]:
                print(f"  FAIL: {r['item']} (got={r['got']} "
                      f"want={r['want']})")
        sys.exit(1)


if __name__ == "__main__":
    main()
