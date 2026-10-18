"""Protocol equivalence gate for FedMGP (run BEFORE any formal training).

Two parts, both CPU-only and training-free:

  Part A — native equivalence (informational, expected PASS):
      run FedMGP's OWN cifar100_Data_Spliter with the frozen protocol
      (seed 42, 5/5/15, surrogate 20) in a fresh RNG order identical to
      its main.py, and compare against the manifest:
        per (client, task): class_mask, ordered raw train indices
        surrogate indices, test-pool indices (ordered)
      FedMGP's splitter is the same FedTA-lineage code, so same-seed
      equality is expected in the same Python/numpy environment. A FAIL
      here means RNG-version drift — not fatal (the adapter replaces the
      splitter anyway) but it must be recorded.

  Part B — adapter injection (gating):
      apply the manifest patches (exactly what run_fedmgp.py does) and
      verify that what FedMGP would actually LOAD is the manifest:
        patched random_split()      -> outer subsets + masks
        patched process_testdata()  -> surrogate / test pool
        patched torch random_split under a (client, task) ctx
                                    -> inner 70/30 (the evaluation sets)

PASS requires every Part B item bit-identical (ordered lists) and no
structural violations. Exit code 0 only on PASS.

  Part C — bug-fix seals (gating):
      apply the three documented official-repo bug fixes in
      run_fedmgp.py's exact order and verify each: (a) the official
      source still contains the bug anchor (repo-drift protection:
      if upstream fixes the bug the seal fails loudly -> re-audit),
      (b) the patch is in effect on the live class, (c) where cheap,
      the fixed behavior is exercised for real (CustomedSubset gets a
      PIL-only transform that crashes on raw numpy pre-fix).
        C1  Client_DF.train np.arange(200) -> args.nb_classes
        C2  Client_DF.train wrapped with update_data (official
            Server_DF.train_clients never calls it for cifar100)
        C3  CustomedSubset.__getitem__ PIL conversion restored

Usage (FedSMTA conda env, from the FedSMTA repo root):

  python protocol_adapters/gate_fedmgp.py \
      --manifest protocol_manifests/cifar100_c5t5p15s20_seed42.json

Writes protocol_manifests/gate_fedmgp_report.json.
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


def cmp_list(name, got, want, rows):
    ok = (got == want)
    rows.append({"item": name, "pass": bool(ok),
                 "got_len": len(got), "want_len": len(want)})
    return ok


def main():
    p = argparse.ArgumentParser("FedMGP protocol equivalence gate")
    p.add_argument("--manifest", required=True)
    p.add_argument("--fedmgp_root", default=str(FEDMGP_ROOT))
    p.add_argument("-o", "--report", default="")
    args = p.parse_args()

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
    all_ok = True

    # ------------------------------------------------------------------
    # Part A: native FedMGP splitter vs manifest (fresh RNG order,
    # exactly FedMGP main.py: seed -> random_split -> process_testdata)
    # ------------------------------------------------------------------
    random.seed(seed)
    np.random.seed(seed)
    from data.cifar100_subset_spliter import cifar100_Data_Spliter

    spl = cifar100_Data_Spliter(client_num=proto["client_num"],
                                task_num=proto["task_num"],
                                private_class_num=proto["private_class_num"],
                                input_size=proto["input_size"])
    native_data, native_mask = spl.random_split()
    spl2 = cifar100_Data_Spliter(client_num=proto["client_num"],
                                 task_num=proto["task_num"],
                                 private_class_num=proto["private_class_num"],
                                 input_size=proto["input_size"])
    native_surro, native_pool = spl2.process_testdata(
        proto["surrogate_num"])

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
    # Part B: adapter injection vs manifest (the gating part)
    # ------------------------------------------------------------------
    manifest_splitter.patch_torch_random_split()
    manifest_splitter.patch_spliter()
    manifest_splitter.patch_client_get_data()

    import torch.utils.data as tud

    spl3 = cifar100_Data_Spliter(client_num=proto["client_num"],
                                 task_num=proto["task_num"],
                                 private_class_num=proto["private_class_num"],
                                 input_size=proto["input_size"])
    ad_data, ad_mask = spl3.random_split()
    spl4 = cifar100_Data_Spliter(client_num=proto["client_num"],
                                 task_num=proto["task_num"],
                                 private_class_num=proto["private_class_num"],
                                 input_size=proto["input_size"])
    ad_surro, ad_pool = spl4.process_testdata(proto["surrogate_num"])

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

            # inner 70/30 through the REAL patched code path: ctx-tagged
            # torch.utils.data.random_split on the outer CustomedSubset,
            # with the exact get_data lengths convention
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

    partB_ok &= cmp_list("B surrogate indices",
                         [int(x) for x in ad_surro.indices],
                         m["surrogate"]["indices"], rows)
    partB_ok &= cmp_list("B test_pool indices",
                         [int(x) for x in ad_pool.indices],
                         m["test_pool"]["indices"], rows)

    # structural seals on the adapter output:
    #  - no raw train index is assigned twice across the whole outer split
    #  - surrogate + test_pool partition the 10k test set exactly
    #  - total train count equals the manifest's own total (the official
    #    dirichlet int() truncation drops ~64 samples in BOTH codebases;
    #    50000 is NOT the right expectation — S1 already proved the
    #    reference run has the same total)
    all_raw = [x for i in range(proto["client_num"])
               for t in range(proto["task_num"])
               for x in ad_data[i][t].indices]
    want_total = sum(len(m["outer"][str(i)][str(t)]["raw_indices"])
                     for i in range(proto["client_num"])
                     for t in range(proto["task_num"]))
    struct_ok = (len(all_raw) == len(set(all_raw)) == want_total
                 and sorted(ad_surro.indices + ad_pool.indices)
                 == list(range(10000)))
    rows.append({"item": "B structure (unique raw indices, "
                         "surrogate+pool=10k, total==manifest)",
                 "pass": bool(struct_ok),
                 "got_len": len(all_raw), "want_len": want_total})

    all_ok = partB_ok and struct_ok

    # ------------------------------------------------------------------
    # Part C: bug-fix seals (gating) — in run_fedmgp.py's exact order
    # ------------------------------------------------------------------
    import inspect
    import textwrap

    partC_ok = True

    # C1 (applied FIRST: it rewrites Client_DF.train's source; the C2
    # wrapper must wrap the FIXED train). The patch itself fail-louds
    # if the official source no longer contains the anchor line.
    from Models.Client_DF import Client_DF
    official_train = textwrap.dedent(inspect.getsource(Client_DF.train))
    c1_anchor = "np.setdiff1d(np.arange(200), mask)" in official_train
    manifest_splitter.patch_client_train_nb_classes_bug()
    c1_ok = (c1_anchor
             and Client_DF.train.__code__.co_filename
             == "<Client_DF.train nb_classes bugfix>")
    partC_ok &= c1_ok
    rows.append({"item": "C1 Client_DF.train np.arange(200) -> "
                         "args.nb_classes (official anchor + patched "
                         "code object active)",
                 "pass": bool(c1_ok), "got_len": 0, "want_len": 1})

    # C2: official Server_DF.train_clients must NOT call update_data
    # (that IS the bug); the patch wraps train with FedMGP's own
    # update_data
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

    # C3: official CustomedSubset.__getitem__ has the PIL conversion
    # commented out; apply the fix and exercise it with a PIL-only
    # transform (RandomResizedCrop raises TypeError on raw numpy
    # pre-fix)
    from data.cifar100_subset_spliter import CustomedSubset
    official_getitem = textwrap.dedent(
        inspect.getsource(CustomedSubset.__getitem__))
    c3_anchor = "# img = Image.fromarray(img)" in official_getitem
    manifest_splitter.patch_customed_subset_pil()
    import torch as _torch
    import torchvision.transforms as _T

    class _FakeDS:
        data = np.zeros((2, 32, 32, 3), dtype=np.uint8)
        targets = [3, 7]
    sub = CustomedSubset(_FakeDS(), [0, 1],
                         _T.Compose([_T.RandomResizedCrop(224),
                                     _T.ToTensor()]))
    img, tgt = sub[0]
    c3_ok = (c3_anchor and _torch.is_tensor(img)
             and tuple(img.shape) == (3, 224, 224) and int(tgt) == 3)
    partC_ok &= c3_ok
    rows.append({"item": "C3 CustomedSubset PIL restore (official "
                         "anchor + behavioral: PIL-only transform "
                         "returns tensor)",
                 "pass": bool(c3_ok), "got_len": 0, "want_len": 1})

    all_ok = all_ok and partC_ok

    # ------------------------------------------------------------------
    # report
    # ------------------------------------------------------------------
    n_fail_A = sum(1 for r in rows if r["item"].startswith("A ")
                   and not r["pass"])
    n_fail_B = sum(1 for r in rows if r["item"].startswith("B ")
                   and not r["pass"])
    n_fail_C = sum(1 for r in rows if r["item"].startswith("C ")
                   and not r["pass"])
    print("\n===== FedMGP protocol equivalence gate =====")
    print(f"Part A (native splitter vs manifest): "
          f"{'PASS' if partA_ok else f'FAIL ({n_fail_A} items)'}"
          f"  [informational: same-lineage splitter, same env]")
    print(f"Part B (adapter injection vs manifest): "
          f"{'PASS' if partB_ok else f'FAIL ({n_fail_B} items)'}"
          f"  [gating]")
    print(f"structure seals: {'PASS' if struct_ok else 'FAIL'}")
    print(f"Part C (bug-fix seals): "
          f"{'PASS' if partC_ok else f'FAIL ({n_fail_C} items)'}"
          f"  [gating]")
    for r in rows:
        if not r["pass"]:
            print(f"  [FAIL] {r['item']} got_len={r['got_len']} "
                  f"want_len={r['want_len']}")
    result = "PASS" if all_ok else "FAIL"
    print(f"\nRESULT: {result}"
          + ("" if all_ok else " (fix the adapter before any formal run)"))

    report = {
        "test": "fedmgp_protocol_equivalence_gate",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "manifest": manifest_path,
        "fedmgp_root": fedmgp_root,
        "result": result,
        "partA_native_equivalence": "PASS" if partA_ok else "FAIL",
        "partB_adapter_injection": "PASS" if partB_ok else "FAIL",
        "structure_seals": "PASS" if struct_ok else "FAIL",
        "partC_bugfix_seals": "PASS" if partC_ok else "FAIL",
        "items": rows,
        "notes": [
            "Part A compares FedMGP's own same-seed split against the "
            "manifest; a FAIL means RNG-version drift and is recorded, "
            "not fatal (the adapter replaces the splitter regardless)",
            "Part B is gating: it verifies the exact objects FedMGP "
            "loads under run_fedmgp.py are the manifest's splits, "
            "including the inner 30% evaluation sets",
            "run_fedmgp.py additionally applies three documented "
            "official-repo bug fixes (Server never calls "
            "Client_DF.update_data for cifar100; CustomedSubset PIL "
            "conversion commented out; local-prompt phase hardcodes "
            "np.arange(200) causing a CUDA assert on 100-class "
            "logits) — none of them touch the data protocol; see "
            "protocol_adapters/manifest_splitter.py docstrings",
        ],
    }
    out = Path(args.report) if args.report else \
        (REPO_ROOT / "protocol_manifests" / "gate_fedmgp_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"report: {out}")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
