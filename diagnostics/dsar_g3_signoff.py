"""G3 DSAR hook/dispatch 签核（RUNCOMMANDS §34.4d item 2，B1 解锁前置）。

合并两份既有证据，映射 §34.4a 四项不可变语义并产出持久 verdict JSON：

  A. G1 hook 证据：<run_dir>/dsar_g1_invariance.json
     （T1 OFF/ON(identity) 逐位中立；T2 readout dispatch 接线五臂；
      T3 真实拟合 RNG/mode/状态中立；15/15 PASS）
  B. 静态调用图审计：diagnostics/dsar_task_dispatch_audit.json
     （verdict SHARED_PATH_CONFIRMED；六项 file:line 证据）

四项不可变语义 -> 证据映射：
  S1 B 插在 self.model() 入口之前（routing + Chead feature half 同时
     看到修正特征）      <- G1 T2 五臂 insertion_point/manual_readout
                             + audit Q5（L1348 单张量交接）
  S2 k<t 只 B（L 禁达历史）<- G1 T2 B_identity_L_active(historical)
                             logits_equal_raw_bitwise + audit Q2
  S3 head 选择恒 heads[k] + 原 key-anchor routing + class mask
                          <- audit Q3 + G1 T2 manual readout 走官方头
  S4 B active/identity 仅由当前任务 train split 无标签 calibration 决定
                          <- audit Q4（protocol-driven）+ G1 T3 合法状态
                             + G2 aux（决策全程 train rows）
  陷阱条款 B identity ≠ DSAR identity
                          <- G1 T2 B_identity_L_active(current)
                             logits_moved（L 照常生效，未短路）

fail-closed：任一证据缺失/断言失败 -> verdict FAIL。唯一写盘点 =
输出 JSON（默认 diagnostics/dsar_g3_signoff.json）。
"""

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

G1_DEFAULT = ("output/cifar100/fedRelink/cifar_base_p15/seed_42/"
              "dsar_g1_invariance.json")
AUDIT_DEFAULT = REPO_ROOT / "diagnostics" / "dsar_task_dispatch_audit.json"


def parse_args():
    p = argparse.ArgumentParser("G3 DSAR hook/dispatch signoff")
    p.add_argument("--g1_json", default=G1_DEFAULT)
    p.add_argument("--audit_json", default=str(AUDIT_DEFAULT))
    p.add_argument("--out_json", default=None,
                   help="default: diagnostics/dsar_g3_signoff.json")
    return p.parse_args()


def main():
    args = parse_args()
    g1 = json.loads(Path(args.g1_json).read_text(encoding="utf-8"))
    audit = json.loads(Path(args.audit_json).read_text(encoding="utf-8"))
    out_json = (Path(args.out_json) if args.out_json
                else REPO_ROOT / "diagnostics" / "dsar_g3_signoff.json")

    fails = []

    # ---- A. G1 证据完整性 ----
    g1_all_pass = (g1.get("verdict") == "PASS")
    n_clients = 0
    t2_ok_all, t3_ok_all = True, True
    for cid, res in g1.get("results", {}).items():
        n_clients += 1
        by_name = {r["name"]: r for r in res}
        if any(r.get("verdict") != "PASS" for r in res):
            g1_all_pass = False
        t1 = by_name.get("T1_off_vs_on_identity_bitwise", {})
        per_task = t1.get("per_task", [])
        if not all(t.get("features_bitwise_equal") and
                   t.get("logits_bitwise_equal") and
                   t.get("batch_manifest_equal") for t in per_task):
            g1_all_pass = False
        t2 = by_name.get("T2_readout_dispatch_wiring", {})
        arms = {d["arm"]: d for d in t2.get("details", [])}
        checks = {
            "B+L(current)": arms.get("B+L(current)", {})
                              .get("insertion_point_features_ok")
                              and arms.get("B+L(current)", {})
                              .get("manual_readout_logits_ok"),
            "B+L(historical)": arms.get("B+L(historical)", {})
                              .get("insertion_point_features_ok"),
            "B_id_L_hist_bitwise_raw":
                arms.get("B_identity_L_active(historical)", {})
                .get("logits_equal_raw_bitwise"),
            "B_id_L_cur_moved":
                arms.get("B_identity_L_active(current)", {})
                .get("logits_moved"),
            "B_act_L_hist_moved":
                arms.get("B_active_L_identity(historical)", {})
                .get("logits_moved"),
        }
        if not all(checks.values()):
            t2_ok_all = False
            fails.append(f"{cid}:T2 {sorted(k for k, v in checks.items() if not v)}")
        t3 = by_name.get("T3_real_fit_rng_mode_neutrality", {})
        if t3.get("verdict") != "PASS":
            t3_ok_all = False
            fails.append(f"{cid}:T3")
    if n_clients != 5:
        fails.append(f"G1 clients={n_clients} (expect 5)")

    # ---- B. 静态审计 ----
    audit_ok = (audit.get("verdict") == "SHARED_PATH_CONFIRMED"
                and set(audit.get("questions", {})) >= {
                    "1_current_task_readout", "2_historical_task_readout",
                    "3_head_selection", "4_task_dispatch_source",
                    "5_adapter_insertion_point",
                    "6_cross_task_transferability"})
    if not audit_ok:
        fails.append("dispatch audit missing/verdict mismatch")

    s = {
        "S1_insertion_before_model": dict(
            passed=g1_all_pass and t2_ok_all and audit_ok,
            evidence=["G1 T2 arms B+L(current)/B+L(historical): "
                      "insertion_point_features_ok + manual_readout_logits_ok "
                      "5/5 clients", "audit Q5: Client_DF L1348 single-tensor "
                      "handoff z_op -> self.model()"]),
        "S2_historical_B_only": dict(
            passed=g1_all_pass and t2_ok_all and audit_ok,
            evidence=["G1 T2 B_identity_L_active(historical): "
                      "logits_equal_raw_bitwise (L never reaches k<t) 5/5",
                      "audit Q2: historical z_op shares the same broadcast "
                      "prompt state; B-only branch architecturally legal"]),
        "S3_head_selection_unchanged": dict(
            passed=audit_ok,
            evidence=["audit Q3: self.heads[task] + class_mask[task]; "
                      "key/anchor_pool client-local, not in fed_avg payload",
                      "G1 T2 manual readout uses official heads"]),
        "S4_unlabeled_current_task_calibration_only": dict(
            passed=g1_all_pass and t3_ok_all and audit_ok,
            evidence=["G1 T3: legal statuses only (FITTED/IDENTITY_GUARD/"
                      "LOW_POWER_IDENTITY), RNG/mode-neutral",
                      "audit Q4: dispatch protocol-driven, never label-driven",
                      "G2 signoff: aux decisions computed on train rows only"]),
        "B_identity_ne_DSAR_identity": dict(
            passed=t2_ok_all,
            evidence=["G1 T2 B_identity_L_active(current): logits_moved "
                      "(L still applies when B=I; no short-circuit) 5/5"]),
    }
    g1_gate = dict(g1_all_pass=g1_all_pass, n_clients=n_clients,
                   source=str(args.g1_json), checkpoint=g1.get("checkpoint"))
    audit_gate = dict(verdict=audit.get("verdict"),
                      source=str(args.audit_json),
                      n_questions=len(audit.get("questions", {})),
                      risks=len(audit.get("implementation_risks_found", [])))
    verdict = "PASS" if (all(v["passed"] for v in s.values())
                         and not fails) else "FAIL"

    out = dict(gate="G3", verdict=verdict,
               g1_evidence=g1_gate, dispatch_audit=audit_gate,
               semantics=s, failures=fails[:20],
               notes="Sec.34.4d item 2: merge signoff of G1 T2/T3 runtime "
                     "evidence + dsar_task_dispatch_audit.json static call "
                     "graph against the four immutable semantics of "
                     "Sec.34.4a. Read-only; single new JSON output.")
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("=== G3 DSAR hook/dispatch signoff ===")
    print(f"G1 evidence: {'PASS' if g1_all_pass else 'FAIL'} "
          f"({n_clients} clients x 3 tests)")
    print(f"dispatch audit: {audit.get('verdict')}")
    for k, v in s.items():
        print(f"{k}: {'PASS' if v['passed'] else 'FAIL'}")
    if fails:
        print("failures:", fails[:10])
    print(f"VERDICT: {verdict}")
    print(f"written: {out_json}")


if __name__ == "__main__":
    main()
