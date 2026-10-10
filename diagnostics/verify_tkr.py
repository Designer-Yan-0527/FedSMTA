"""TKR exact-caliber recomputation from FedTA author's definition (2026-10-10).

Author's reply (GitHub issue) defines Temporal Knowledge Retention as:

  denominator = accuracy of the AGGREGATED GLOBAL model on the local test
                set of task 0 upon completion of training on task 0
                (round = global_epoch - 1, i.e. round 4)
  numerator   = LOCAL accuracy when the LOCAL model after each task t (t>0)
                is evaluated back on task 0

Both quantities already exist in metrics/train_log.csv — no GPU needed:

  phase=server, test_task=0, round=r  -> post-broadcast evaluation inside
      get_global_proto_and_head() (official FedTA path): server prompt +
      local key/anchor + local heads[0]  == the "aggregated global model"
  phase=local,   test_task=0, round=r  -> _log_local_phase() evaluates right
      after local training and BEFORE aggregation: local prompt + local
      key/anchor + local heads[0]  == the "local model"

NOTE: task-end checkpoints CANNOT reconstruct the numerator state (they
store the post-broadcast prompt), so train_log.csv is the only source.

Calibers reported (both aggregations: ratio-of-means and mean-of-ratios):
  author   local(t) / server(t=0)   <- the author's literal definition
  local    local(t) / local(t=0)    <- same-side control
  server   server(t) / server(t=0)  <- same-side control (our previous
                                       "broadcast-state KRt" caliber)
Plus the t=0 local/server gap = pure client-drift premium without forgetting.

Usage (from repo root):
  python diagnostics/verify_tkr.py --run_dir output/cifar100/fedta/<run>/seed_42
  python diagnostics/verify_tkr.py --train_log <path>/train_log.csv
"""

import argparse
import csv
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser("TKR exact-caliber recomputation")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--run_dir", help="run directory containing metrics/train_log.csv")
    src.add_argument("--train_log", help="direct path to a train_log.csv")
    p.add_argument("--global_epoch", type=int, default=0,
                   help="rounds per task (default: read from run_dir/args.json, else 5)")
    p.add_argument("--out_prefix", default=None,
                   help="output prefix (default: diagnostics_output/tkr)")
    return p.parse_args()


def load_rows(path):
    rows = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            key = (int(r["round"]), int(r["client"]), r["phase"], int(r["test_task"]))
            if key in rows:
                raise ValueError(f"duplicate row in train_log: {key}")
            rows[key] = float(r["accuracy"])
    return rows


def get_acc(rows, rnd, client, phase, test_task, what):
    key = (rnd, client, phase, test_task)
    if key not in rows:
        raise KeyError(
            f"missing {what}: round={rnd} client={client} phase={phase} "
            f"test_task={test_task}. Was the run started with instrumentation "
            f"ON (--no_instrumentation off)? train_log.csv must contain "
            f"phase=local rows for the numerator.")
    return rows[key]


def main():
    args = parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
        log_path = run_dir / "metrics" / "train_log.csv"
        ge = args.global_epoch
        if ge <= 0:
            args_json = run_dir / "args.json"
            if args_json.is_file():
                with open(args_json, "r", encoding="utf-8") as f:
                    ge = int(json.load(f).get("global_epoch", 5))
            else:
                ge = 5
    else:
        log_path = Path(args.train_log)
        ge = args.global_epoch or 5
    if not log_path.is_file():
        raise FileNotFoundError(f"train_log.csv not found: {log_path}")

    rows = load_rows(log_path)
    n_rounds = max(k[0] for k in rows) + 1
    task_num = n_rounds // ge
    clients = sorted({k[1] for k in rows})
    print(f"train_log: {log_path}")
    print(f"rounds={n_rounds} global_epoch={ge} tasks={task_num} "
          f"clients={clients}")

    r0 = ge - 1                       # last round of task 0 (author: round = 4)
    # denominator candidates at task-0 end
    denom_server = {j: get_acc(rows, r0, j, "server", 0, "denominator (server/global)")
                    for j in clients}
    denom_local = {j: get_acc(rows, r0, j, "local", 0, "denominator (local control)")
                   for j in clients}

    # t=0 reference: local vs server gap at task-0 end (client-drift premium,
    # no forgetting yet)
    t0_gap = {j: denom_local[j] - denom_server[j] for j in clients}
    print(f"\nt=0 local-vs-server gap (pure client-drift premium, pp): "
          f"mean={sum(t0_gap.values())/len(clients):+.2f}")

    per_client = []
    for t in range(1, task_num):
        rt = (t + 1) * ge - 1         # last round of task t
        for j in clients:
            num_local = get_acc(rows, rt, j, "local", 0,
                                f"numerator local@task{t}")
            num_server = get_acc(rows, rt, j, "server", 0,
                                 f"numerator server@task{t}")
            per_client.append({
                "client_id": j,
                "task": t,
                "round": rt,
                "num_local": num_local,
                "num_server": num_server,
                "denom_server": denom_server[j],
                "denom_local": denom_local[j],
                "tkr_author": num_local / denom_server[j],
                "tkr_local": num_local / denom_local[j],
                "tkr_server": num_server / denom_server[j],
                "local_minus_server_pp": num_local - num_server,
            })

    def agg(vals):
        return sum(vals) / len(vals)

    print(f"\n{'t':>2} | {'author L/G':>10} {'local L/L':>10} {'server S/S':>10}"
          f" | {'num_local':>9} {'num_server':>10} | {'L-S gap':>8}")
    print("-" * 78)
    summary = {"global_epoch": ge, "task_num": task_num, "denominator_server": agg(list(denom_server.values())),
               "denominator_local": agg(list(denom_local.values())),
               "t0_client_drift_gap_pp": agg(list(t0_gap.values())), "calibers": {}}
    for t in range(1, task_num):
        sub = [r for r in per_client if r["task"] == t]
        # ratio-of-means (aggregate accuracies first, then divide)
        mean_num_local = agg([r["num_local"] for r in sub])
        mean_num_server = agg([r["num_server"] for r in sub])
        mean_denom_server = agg(list(denom_server.values()))
        mean_denom_local = agg(list(denom_local.values()))
        tkr_author_rom = mean_num_local / mean_denom_server
        tkr_local_rom = mean_num_local / mean_denom_local
        tkr_server_rom = mean_num_server / mean_denom_server
        tkr_author_mor = agg([r["tkr_author"] for r in sub])
        tkr_local_mor = agg([r["tkr_local"] for r in sub])
        tkr_server_mor = agg([r["tkr_server"] for r in sub])
        gap = agg([r["local_minus_server_pp"] for r in sub])
        print(f"{t:>2} | {tkr_author_rom:>9.1%} {tkr_local_rom:>10.1%}"
              f" {tkr_server_rom:>10.1%}"
              f" | {mean_num_local:>8.2f} {mean_num_server:>9.2f} | {gap:>+7.2f}")
        summary["calibers"][str(t)] = {
            "author": {"ratio_of_means": tkr_author_rom,
                       "mean_of_ratios": tkr_author_mor,
                       "per_client": {r["client_id"]: r["tkr_author"] for r in sub}},
            "local": {"ratio_of_means": tkr_local_rom,
                      "mean_of_ratios": tkr_local_mor},
            "server": {"ratio_of_means": tkr_server_rom,
                       "mean_of_ratios": tkr_server_mor},
            "mean_num_local": mean_num_local,
            "mean_num_server": mean_num_server,
            "local_minus_server_pp": gap,
        }

    print("\n(mean-of-ratios aggregation)")
    for t in range(1, task_num):
        sub = [r for r in per_client if r["task"] == t]
        print(f"t={t}: author {agg([r['tkr_author'] for r in sub]):.1%} | "
              f"local {agg([r['tkr_local'] for r in sub]):.1%} | "
              f"server {agg([r['tkr_server'] for r in sub]):.1%}")

    out_prefix = args.out_prefix or "diagnostics_output/tkr"
    out_csv = Path(f"{out_prefix}.csv")
    out_json = Path(f"{out_prefix}.json")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_client[0].keys()))
        w.writeheader()
        w.writerows(per_client)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsaved {out_csv} ({len(per_client)} rows) and {out_json}")

    print("\nVerdict guide: paper Fig.4(b) FedTA KRt ~105-115% on CIFAR-100.")
    print("author caliber (local/global) reaching that range closes the "
          "residual 7-15pp gap of Roadmap 4.10 (model-asymmetry explanation).")


if __name__ == "__main__":
    main()
