"""Protocol equivalence gate for FedMGP on ImageNet-R (run BEFORE any
formal INR training). CPU-only and training-free.

FedMGP's official repo has NO ImageNet-R route (main.py branches only
cifar100 / 5datasets / office_home; config/ has no INR entry; data/ has
no INR spliter — the only trace is the internal name 'imagenetr' that
Models/Client_DF.py recognizes), so unlike the cifar100 gate there is no
FedMGP-native INR splitter to compare against. The manifest's native
counterpart is the MAIN repo's ImagenetR_spliter — the exact class and
RNG order diagnostics/export_split_manifest.py used to regenerate the
frozen split (random.seed -> np.random.seed -> random_split ->
process_testdata(5), itself sealed S1 against the reference run's
checkpoint). This gate mirrors gate_fedmgp.py's three parts:

  Part A — native regeneration (informational, expected PASS):
      run the MAIN repo's own ImagenetR_spliter with the frozen protocol
      (seed 42, 5c x 5t, p40, surrogate 5/class) in a fresh RNG order
      identical to main.build_server, and compare against the manifest:
      per (client, task): class_mask, ordered raw train indices;
      surrogate indices, test-pool indices (ordered). A FAIL here means
      RNG-version drift or data-layout drift — not fatal (the adapter
      replaces the splitter anyway) but it must be recorded.

  Part B — adapter injection (gating):
      apply the INR patches (exactly what run_fedmgp.py's ImageNet-R
      path does) and verify that what FedMGP would actually LOAD is the
      manifest:
        patched ImagenetR_spliter.random_split()     -> outer subsets+masks
        patched ImagenetR_spliter.process_testdata(5) -> surrogate / pool
        patched torch random_split under a (client, task) ctx
                                                    -> inner 70/30 (evals)
      plus INR structural seals: unique raw indices; surrogate+pool
      partition {0..5999}; 24000 train / 6000 test enumeration with the
      sorted-block class mapping; inner disjoint covers; every outer
      subset's targets inside its class_mask (index-semantics seal).

  Part C — INR route + bug-fix seals (gating):
      apply run_fedmgp.py's exact patch order and verify each:
        C1  Client_DF.train np.arange(200) -> args.nb_classes
            (anchor must still exist in the official source; at the INR
            nb_classes=200 the official constant is already CORRECT, so
            this is a sealed semantic no-op — behavior identical either
            way; also asserts manifest nb_classes == 200)
        C2  Client_DF.train wrapped with update_data (official
            Server_DF.train_clients never calls it for any dataset)
        C4  Server_DF.init_client 'imagenetr' route fix (official source
            must still lack it; patched code object active; behavioral:
            a bare Server_DF with data_name='imagenetr' builds 5 real
            clients wired to the manifest subsets/masks — the exact crash
            every official INR attempt would hit)
        C3  INR CustomedSubset behavioral: a real batch through the
            manifest outer subset (PIL conversion + FedMGP transform)
        C5  INR surrogate wrapper behavioral: main-repo iCIFAR100c
            (FedMGP's own is CIFAR-only and crashes on INR) —
            getTrainData(class_mask) selects exactly the test images of
            those classes and yields (index, x, y) triples

PASS requires every Part B and Part C item to hold. Exit code 0 only on
PASS. Data cost: one 24k-image load + one 6k-image load (~2-4 min total;
all later parts share the cached instances).

Usage (FedSMTA conda env, from the FedSMTA repo root):

  python protocol_adapters/gate_fedmgp_inr.py \
      --manifest protocol_manifests/inr_c5t5p40s5_seed42.json

Writes protocol_manifests/gate_fedmgp_inr_report.json.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

ADAPTER_DIR = Path(__file__).resolve().parent
REPO_ROOT = ADAPTER_DIR.parent
FEDMGP_ROOT = REPO_ROOT / "external_baselines" / "FedMGP"

if str(ADAPTER_DIR) not in sys.path:
    sys.path.insert(0, str(ADAPTER_DIR))

import manifest_splitter  # noqa: E402

# the frozen INR comparison protocol (manifest must match all of these)
EXPECTED = {"client_num": 5, "task_num": 5, "private_class_num": 40,
            "surrogate_num": 5, "input_size": 224, "nb_classes": 200}


def cmp_list(name, got, want, rows):
    ok = (got == want)
    rows.append({"item": name, "pass": bool(ok),
                 "got_len": len(got), "want_len": len(want)})
    return ok


def main():
    p = argparse.ArgumentParser(
        "FedMGP protocol equivalence gate (ImageNet-R)")
    p.add_argument("--manifest", required=True)
    p.add_argument("--fedmgp_root", default=str(FEDMGP_ROOT))
    p.add_argument("--data_path", default="./local_datasets",
                   help="Imagenet_R root (the class expects "
                        "<root>/imagenet-r/{train,test})")
    p.add_argument("-o", "--report", default="")
    args = p.parse_args()
    # resolve the -o path against the invocation cwd BEFORE the chdir
    # into the FedMGP root below: a relative report path would
    # otherwise land inside the external repo, polluting the zero-edit
    # checkout
    if args.report:
        args.report = str(Path(args.report).resolve())

    import numpy as np
    import os

    manifest_path = str(Path(args.manifest).resolve())
    fedmgp_root = str(Path(args.fedmgp_root).resolve())
    if fedmgp_root not in sys.path:
        sys.path.insert(0, fedmgp_root)
    os.chdir(fedmgp_root)

    m = manifest_splitter.load_manifest(manifest_path)
    proto = m["protocol"]
    seed = m["seed"]
    rows = []

    # ------------------------------------------------------------------
    # manifest identity seals (the frozen INR comparison protocol)
    # ------------------------------------------------------------------
    ident_ok = (m["dataset"] == "ImageNet-R"
                and all(int(proto[k]) == v for k, v in EXPECTED.items()))
    rows.append({"item": "0 manifest identity (ImageNet-R, 5c x 5t, p40, "
                         "s5, nb_classes 200, input 224)",
                 "pass": bool(ident_ok), "got_len": 0, "want_len": 1})
    if not ident_ok:
        print(f"[FAIL] manifest identity: dataset={m['dataset']!r} "
              f"protocol={proto}")
        _finish(rows, args, manifest_path, fedmgp_root, False,
                False, False, False)
        return

    t_start = time.time()

    # ------------------------------------------------------------------
    # Part A: native regeneration (main-repo lineage ImagenetR_spliter,
    # fresh RNG order = main.build_server / export_split_manifest:
    # seed -> random_split -> process_testdata(5))
    # ------------------------------------------------------------------
    random.seed(seed)
    np.random.seed(seed)
    import torch  # noqa: F401  (build_server seeds torch first too)
    torch.manual_seed(seed)

    spl_mod = manifest_splitter.load_mainrepo_imagenet_r_spliter_module()
    ImagenetR_spliter = spl_mod.ImagenetR_spliter

    def native_spliter():
        return ImagenetR_spliter(
            client_num=proto["client_num"], task_num=proto["task_num"],
            private_class_num=proto["private_class_num"],
            input_size=proto["input_size"], data_path=args.data_path)

    spl = native_spliter()
    native_data, native_mask = spl.random_split()
    spl2 = native_spliter()
    native_surro, native_pool = spl2.process_testdata(5)

    partA_ok = True
    for i in range(proto["client_num"]):
        for t in range(proto["task_num"]):
            want = m["outer"][str(i)][str(t)]
            partA_ok &= cmp_list(
                f"A outer class_mask c{i}t{t}",
                [int(c) for c in native_mask[i][t]], want["class_mask"],
                rows)
            partA_ok &= cmp_list(
                f"A outer raw_indices c{i}t{t}",
                [int(x) for x in native_data[i][t].indices],
                want["raw_indices"], rows)
    partA_ok &= cmp_list("A surrogate indices",
                         [int(x) for x in native_surro.indices],
                         m["surrogate"]["indices"], rows)
    partA_ok &= cmp_list("A test_pool indices",
                         [int(x) for x in native_pool.indices],
                         m["test_pool"]["indices"], rows)

    # ------------------------------------------------------------------
    # Part B: adapter injection vs manifest (the gating part) — exactly
    # run_fedmgp.py's ImageNet-R patch set, same order
    # ------------------------------------------------------------------
    manifest_splitter.patch_torch_random_split()
    ImagenetR_spliter = manifest_splitter.patch_imagenet_r_spliter()
    manifest_splitter.patch_client_get_data()

    import torch.utils.data as tud

    def adapter_spliter():
        return ImagenetR_spliter(
            client_num=proto["client_num"], task_num=proto["task_num"],
            private_class_num=proto["private_class_num"],
            input_size=proto["input_size"], data_path=args.data_path)

    spl3 = adapter_spliter()
    ad_data, ad_mask = spl3.random_split()
    spl4 = adapter_spliter()
    ad_surro, ad_pool = spl4.process_testdata(5)

    partB_ok = True
    for i in range(proto["client_num"]):
        for t in range(proto["task_num"]):
            want = m["outer"][str(i)][str(t)]
            partB_ok &= cmp_list(
                f"B outer class_mask c{i}t{t}",
                [int(c) for c in ad_mask[i][t]], want["class_mask"], rows)
            partB_ok &= cmp_list(
                f"B outer raw_indices c{i}t{t}",
                [int(x) for x in ad_data[i][t].indices],
                want["raw_indices"], rows)

            # inner 70/30 through the REAL patched code path
            n = len(ad_data[i][t])
            lengths = [int(n * 0.7), n - int(n * 0.7)]
            manifest_splitter._CTX["ctx"] = (i, t)
            try:
                tr, te = tud.random_split(ad_data[i][t], lengths)
            finally:
                manifest_splitter._CTX["ctx"] = None
            want_inner = m["inner"][str(i)][str(t)]
            partB_ok &= cmp_list(f"B inner train c{i}t{t}",
                                 [int(x) for x in tr.indices],
                                 want_inner["train"], rows)
            partB_ok &= cmp_list(f"B inner test (eval set) c{i}t{t}",
                                 [int(x) for x in te.indices],
                                 want_inner["test"], rows)
            inner_cover = (sorted(want_inner["train"] + want_inner["test"])
                           == list(range(n)))
            rows.append({"item": f"B inner disjoint cover c{i}t{t}",
                         "pass": bool(inner_cover), "got_len": n,
                         "want_len": n})
            partB_ok &= inner_cover

    partB_ok &= cmp_list("B surrogate indices",
                         [int(x) for x in ad_surro.indices],
                         m["surrogate"]["indices"], rows)
    partB_ok &= cmp_list("B test_pool indices",
                         [int(x) for x in ad_pool.indices],
                         m["test_pool"]["indices"], rows)

    # INR structural seals on the adapter output:
    #  - no raw train index assigned twice across the whole outer split;
    #    total equals the manifest's own total (official dirichlet int()
    #    truncation drops samples in the lineage codebase too — 24000 is
    #    NOT the right expectation)
    #  - surrogate + test_pool partition the 6000-image test set exactly
    #  - the enumeration the indices refer to is the frozen one:
    #    24000 train / 6000 test, sorted-block class mapping over 200
    #    classes, every outer subset's targets inside its class_mask
    all_raw = [x for i in range(proto["client_num"])
               for t in range(proto["task_num"])
               for x in ad_data[i][t].indices]
    want_total = sum(len(m["outer"][str(i)][str(t)]["raw_indices"])
                     for i in range(proto["client_num"])
                     for t in range(proto["task_num"]))
    trainset = manifest_splitter.imagenet_r_instance(
        True, root=args.data_path)
    testset = manifest_splitter.imagenet_r_instance(
        False, root=args.data_path)
    train_targets = [int(x) for x in trainset.targets]
    test_targets = [int(x) for x in testset.targets]
    masks_ok = True
    for i in range(proto["client_num"]):
        for t in range(proto["task_num"]):
            entry = m["outer"][str(i)][str(t)]
            got = {train_targets[idx] for idx in entry["raw_indices"]}
            if not got.issubset(set(entry["class_mask"])):
                masks_ok = False
    enum_ok = (len(train_targets) == 24000 and len(test_targets) == 6000
               and train_targets == sorted(train_targets)
               and test_targets == sorted(test_targets)
               and len(set(train_targets)) == 200
               and len(set(test_targets)) == 200)
    struct_ok = (len(all_raw) == len(set(all_raw)) == want_total
                 and sorted(ad_surro.indices + ad_pool.indices)
                 == list(range(6000))
                 and enum_ok and masks_ok)
    rows.append({"item": "B structure (unique raw indices, "
                         "surrogate+pool={0..5999}, total==manifest, "
                         "24000/6000 sorted-block enumeration, "
                         "outer targets within class_mask)",
                 "pass": bool(struct_ok),
                 "got_len": len(all_raw), "want_len": want_total})
    all_ok = partB_ok and struct_ok

    # ------------------------------------------------------------------
    # Part C: INR route + bug-fix seals (gating) — in run_fedmgp.py's
    # exact order
    # ------------------------------------------------------------------
    import inspect
    import textwrap

    partC_ok = True

    # C1 (applied FIRST: it rewrites Client_DF.train's source; the C2
    # wrapper must wrap the FIXED train). At the INR nb_classes=200 the
    # official constant np.arange(200) is already correct, so the patch
    # is a sealed semantic no-op (kept for one shared code path).
    from Models.Client_DF import Client_DF
    official_train = textwrap.dedent(inspect.getsource(Client_DF.train))
    c1_anchor = "np.setdiff1d(np.arange(200), mask)" in official_train
    manifest_splitter.patch_client_train_nb_classes_bug()
    c1_ok = (c1_anchor
             and Client_DF.train.__code__.co_filename
             == "<Client_DF.train nb_classes bugfix>"
             and int(proto["nb_classes"]) == 200)
    partC_ok &= c1_ok
    rows.append({"item": "C1 Client_DF.train np.arange(200) -> "
                         "args.nb_classes (official anchor + patched code "
                         "object active + no-op at nb_classes=200)",
                 "pass": bool(c1_ok), "got_len": 0, "want_len": 1})

    # C2: official Server_DF.train_clients must NOT call update_data
    # (that IS the bug, for every dataset); the patch wraps train with
    # FedMGP's own update_data
    from Models.Server_DF import Server_DF
    server_src = textwrap.dedent(
        inspect.getsource(Server_DF.train_clients))
    c2_anchor = "update_data" not in server_src
    manifest_splitter.patch_client_train_update_data()
    c2_ok = (c2_anchor
             and "update_data" in Client_DF.train.__code__.co_names
             and Client_DF.train.__code__.co_filename.endswith(
                 "manifest_splitter.py"))
    partC_ok &= c2_ok
    rows.append({"item": "C2 Client_DF.train wrapped with update_data "
                         "(official Server never calls it + wrapper "
                         "active)",
                 "pass": bool(c2_ok), "got_len": 0, "want_len": 1})

    # C4: Server_DF.init_client official source lacks the 'imagenetr'
    # branch (the INR blocker); patched code object must be active and
    # behavioral: a bare server with data_name='imagenetr' builds 5 real
    # clients wired to the manifest subsets/masks
    init_src = textwrap.dedent(
        inspect.getsource(Server_DF.init_client))
    c4_anchor = ("imagenetr" not in init_src
                 and "self.args.data_name == '5datasets'" in init_src)
    manifest_splitter.patch_server_init_client_imagenetr()
    c4_patched = (Server_DF.init_client.__code__.co_filename
                  == "<Server_DF.init_client imagenetr route fix>")

    class _NS:
        pass
    ns = _NS()
    ns.data_name = "imagenetr"
    ns.task_num = proto["task_num"]
    ns.nb_classes = proto["nb_classes"]
    ns.top_k = 1
    bare = object.__new__(Server_DF)
    bare.client_num = proto["client_num"]
    bare.client_data = ad_data
    bare.class_mask = ad_mask
    bare.clients = []
    bare.origin_model = None
    bare.vit = None
    bare.global_epoch = 5
    bare.local_epoch = 30
    bare.batch_size = 16
    bare.lr = 0.001875
    bare.device = "cpu"
    bare.method = "delay"
    bare.args = ns
    try:
        bare.init_client()
        c4_behavior = len(bare.clients) == proto["client_num"] and all(
            bare.clients[j].id == j
            and bare.clients[j].train_data is ad_data[j]
            and bare.clients[j].class_mask == ad_mask[j]
            and bare.clients[j].local_head is not None
            for j in range(proto["client_num"]))
    except Exception as exc:  # noqa: BLE001  (seal reports the crash)
        print(f"[C4] init_client behavioral seal raised: {exc!r}")
        c4_behavior = False
    c4_ok = c4_anchor and c4_patched and c4_behavior
    partC_ok &= c4_ok
    rows.append({"item": "C4 Server_DF.init_client 'imagenetr' route fix "
                         "(official anchor + patched code object + "
                         "behavioral: 5 real clients on the manifest "
                         "subsets/masks)",
                 "pass": bool(c4_ok), "got_len": 0, "want_len": 1})

    # C3: INR CustomedSubset behavioral — a real batch through the
    # manifest outer subset (PIL conversion + FedMGP transform; the
    # native office lineage crashes on raw numpy pre-conversion)
    import torch as _torch
    from torch.utils.data import DataLoader
    sub = ad_data[0][0]
    img0, tgt0 = sub[0]
    loader = DataLoader(sub, batch_size=4, shuffle=False)
    imgs, tgts = next(iter(loader))
    mask_set = set(m["outer"]["0"]["0"]["class_mask"])
    c3_ok = (_torch.is_tensor(img0) and tuple(img0.shape) == (3, 224, 224)
             and int(tgt0) in mask_set
             and _torch.is_tensor(imgs) and tuple(imgs.shape) == (4, 3, 224,
                                                                 224)
             and set(int(v) for v in tgts).issubset(mask_set))
    partC_ok &= c3_ok
    rows.append({"item": "C3 INR CustomedSubset behavioral (PIL + "
                         "transform; real batch, targets within the "
                         "(client 0, task 0) class_mask)",
                 "pass": bool(c3_ok), "got_len": 0, "want_len": 1})

    # C5: INR surrogate wrapper behavioral — the main-repo iCIFAR100c
    # (FedMGP's own data/iCIFAR100c.py is CIFAR-only: boolean indexing a
    # plain list of ragged numpy images raises TypeError). The wrapper
    # unwraps the manifest surrogate CustomedSubset to the full 6000-image
    # test set (official FedTA/FedMGP semantics), selects by class, and
    # yields the (index, x, y) triples Server_DF.filter/kd_fusion expect.
    wrapper_cls = manifest_splitter.load_mainrepo_icifar100c()
    wrapper = wrapper_cls(subset=ad_surro)
    classes = list(m["outer"]["0"]["0"]["class_mask"])
    wrapper.getTrainData(classes)
    class_set = set(classes)
    expected_n = sum(1 for v in test_targets if v in class_set)
    got_n = len(wrapper.TrainData)
    wloader = DataLoader(wrapper, batch_size=4, shuffle=False)
    widx, wx, wy = next(iter(wloader))
    c5_ok = (got_n == expected_n
             and _torch.is_tensor(wx) and tuple(wx.shape) == (4, 3, 224,
                                                             224)
             and set(int(v) for v in wy).issubset(class_set)
             and wrapper_cls.__module__ == "fedsmta_data_icifar100c"
             and str(Path(inspect.getfile(wrapper_cls)).resolve())
             == str((REPO_ROOT / "data" / "iCIFAR100c.py").resolve()))
    partC_ok &= c5_ok
    rows.append({"item": f"C5 INR surrogate wrapper (main-repo iCIFAR100c; "
                         f"getTrainData selects {got_n}/{expected_n} "
                         f"test images of the c0t0 classes; (index, x, y) "
                         f"triples OK)",
                 "pass": bool(c5_ok), "got_len": got_n,
                 "want_len": expected_n})

    all_ok = all_ok and partC_ok
    elapsed = time.time() - t_start

    n_fail_A = sum(1 for r in rows if r["item"].startswith("A ")
                   and not r["pass"])
    n_fail_B = sum(1 for r in rows if r["item"].startswith("B ")
                   and not r["pass"])
    n_fail_C = sum(1 for r in rows if r["item"].startswith("C ")
                   and not r["pass"])
    print("\n===== FedMGP protocol equivalence gate (ImageNet-R) =====")
    print(f"Part A (native main-repo regeneration vs manifest): "
          f"{'PASS' if partA_ok else f'FAIL ({n_fail_A} items)'}"
          f"  [informational: same-lineage splitter, same env]")
    print(f"Part B (adapter injection vs manifest): "
          f"{'PASS' if partB_ok else f'FAIL ({n_fail_B} items)'}"
          f"  [gating]")
    print(f"structure seals: {'PASS' if struct_ok else 'FAIL'}")
    print(f"Part C (INR route + bug-fix seals): "
          f"{'PASS' if partC_ok else f'FAIL ({n_fail_C} items)'}"
          f"  [gating]")
    for r in rows:
        if not r["pass"]:
            print(f"  [FAIL] {r['item']} got_len={r['got_len']} "
                  f"want_len={r['want_len']}")
    result = "PASS" if all_ok else "FAIL"
    print(f"\nRESULT: {result} ({elapsed:.0f}s)"
          + ("" if all_ok else " (fix the adapter before any formal run)"))

    _finish(rows, args, manifest_path, fedmgp_root, partA_ok, partB_ok,
            struct_ok, partC_ok, all_ok=all_ok, elapsed=elapsed)


def _finish(rows, args, manifest_path, fedmgp_root, partA_ok, partB_ok,
            struct_ok, partC_ok, all_ok=False, elapsed=0.0):
    report = {
        "test": "fedmgp_protocol_equivalence_gate_imagenet_r",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "manifest": manifest_path,
        "fedmgp_root": fedmgp_root,
        "result": "PASS" if all_ok else "FAIL",
        "partA_native_regeneration": "PASS" if partA_ok else "FAIL",
        "partB_adapter_injection": "PASS" if partB_ok else "FAIL",
        "structure_seals": "PASS" if struct_ok else "FAIL",
        "partC_inr_route_bugfix_seals": "PASS" if partC_ok else "FAIL",
        "elapsed_seconds": round(elapsed, 1),
        "items": rows,
        "notes": [
            "Part A compares the MAIN repo's ImagenetR_spliter (the "
            "manifest's own generator class, fresh build_server RNG "
            "order) against the manifest; a FAIL means RNG/data-layout "
            "drift and is recorded, not fatal (the adapter replaces the "
            "splitter regardless). FedMGP itself has NO native INR "
            "splitter to compare against.",
            "Part B is gating: it verifies the exact objects FedMGP "
            "loads under run_fedmgp.py's ImageNet-R path are the "
            "manifest's splits, including the inner 30% evaluation "
            "sets, over the frozen 24000/6000 sorted-block enumeration.",
            "run_fedmgp.py's ImageNet-R path additionally supplies the "
            "missing INR route pieces (manifest-driven "
            "ImagenetR_spliter; init_client 'imagenetr' branch that "
            "Client_DF/update_data already recognize; INR-capable "
            "main-repo iCIFAR100c surrogate wrapper — FedMGP's own is "
            "CIFAR-only) and applies two of the three documented "
            "cifar100 bug fixes (update_data wrapper; np.arange(200) -> "
            "args.nb_classes, a sealed no-op at the INR nb_classes=200). "
            "The cifar100-only CustomedSubset PIL fix is skipped (INR "
            "subsets already convert via the main-repo lineage class).",
            "Method-native hyperparameters on ImageNet-R are the "
            "cifar100_delay config values (lr 0.001875 / threshold 0.1 "
            "/ batch 16 / prompt size 1, length 15, top_k 1): the "
            "official repo has no INR config.",
        ],
    }
    out = Path(args.report) if args.report else \
        (REPO_ROOT / "protocol_manifests" / "gate_fedmgp_inr_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"report: {out}")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
