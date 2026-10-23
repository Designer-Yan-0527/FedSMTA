"""Head row-norm audit (zero-training, read-only).

P0 follow-up to INR_PHASE1_HEAD_DIVERGENCE.md: direct weight-level
evidence for the "last-trained client's head rows survive, earlier
clients' rows decay" hypothesis, on BOTH datasets (CIFAR-100 is the
healthy-head control where the ln(k) uniform degeneracy also appears).

For every (run, task_end checkpoint, client), using the client's ACTUAL
class mask (client.current_class):

  row groups (L2 norms of vit.head.weight rows, log10 scale for
  cross-dataset comparability with INR's exploded magnitudes):
    own      = client.current_class rows
    peer     = union of the OTHER clients' current-class rows
    inactive = all remaining rows
  per group: log10 row-norm max / median / min, |bias| max / median

  forward statistics (same fixed batches and phase-1 forward convention
  as je_gradient_probe.py: original.eval() no_grad -> vit train=True,
    seed 9700 + 100*cid + task, 8 batches x 16):
    ce_mean          masked phase-1 CE (uniform baseline = ln(k))
    logits_std       mean per-sample std across the k masked logits
    entropy_norm     mean softmax entropy / ln(k)   (1.0 == uniform)
    top1_prob        mean max softmax prob

Adjudication hints (flag columns, NOT verdicts):
    own_decayed  = own_median_lognorm < peer_median_lognorm - 1.0
                   (>10x smaller than peers: row-decay signature)
    uniform_out  = entropy_norm > 0.999  (ln(k)-pinned output)

Read-only guarantees: pure no_grad forward (no autograd at all), no
optimizer, no .grad; RNG streams snapshotted/restored around loaders;
checkpoints opened read-only via bootstrap_server.

Usage:
  python diagnostics/head_row_norm_audit.py \
      --run_dir output/ImageNet-R/fedta_a2/inr_a2_key_p40/seed_42 \
      --run_dir output/cifar100/fedta_a2/a2_key/seed_42 [--tasks 0,1,2,3,4]
"""

import argparse
import csv
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from diag_utils import bootstrap_server  # noqa: E402

N_BATCHES = 8
BATCH_SIZE = 16  # official hardcoded value (Client_DF)


def snapshot_rng():
    return (torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            random.getstate(),
            np.random.get_state())


def restore_rng(state):
    torch_cpu, torch_cuda, py, npy = state
    torch.set_rng_state(torch_cpu)
    if torch_cuda is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(torch_cuda)
    random.setstate(py)
    np.random.set_state(npy)


def group_stats(W, b, rows):
    """log10 row-norm stats + bias stats for a set of class rows."""
    if len(rows) == 0:
        return dict(n=0, lognorm_max=float('nan'), lognorm_med=float('nan'),
                    lognorm_min=float('nan'), bias_absmax=float('nan'),
                    bias_absmed=float('nan'))
    rn = W[rows].double().norm(dim=1)
    ab = b[rows].double().abs()
    return dict(n=len(rows),
                lognorm_max=float(torch.log10(rn.max())),
                lognorm_med=float(torch.log10(rn.median())),
                lognorm_min=float(torch.log10(rn.min())),
                bias_absmax=float(ab.max()),
                bias_absmed=float(ab.median()))


def audit_run(run_dir, tasks, device):
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    metrics_dir = os.path.join(run_dir, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    summary_path = os.path.join(metrics_dir, 'head_row_norm_audit.csv')
    detail_path = os.path.join(metrics_dir, 'head_row_norm_audit_rows.csv')
    summary_rows, detail_rows = [], []

    for task in tasks:
        ckpt = os.path.join(ckpt_dir, f'task_{task:02d}_end.pth')
        if not os.path.isfile(ckpt):
            print(f'[audit] MISSING {ckpt}, skipped')
            continue
        args, server = bootstrap_server(run_dir, checkpoint=ckpt, device=device)
        nb = args.nb_classes
        criterion = torch.nn.CrossEntropyLoss().to(device)

        W = server.model.head.weight.detach()
        b = server.model.head.bias.detach()
        all_rows = np.arange(nb)
        masks = {c.id: list(c.current_class) for c in server.clients}

        for client in server.clients:
            own = np.array(masks[client.id], dtype=np.int64)
            peer = np.array(sorted({r for cid, m in masks.items()
                                    if cid != client.id for r in m}),
                            dtype=np.int64)
            inactive = np.array(sorted(set(all_rows.tolist())
                                       - set(own.tolist())
                                       - set(peer.tolist())), dtype=np.int64)

            # ---- forward statistics (pure no_grad, probe convention) ----
            client.vit.load_prompts(client.prompts)
            client.vit.to(device)
            client.model.to(device)
            client.original_model.eval()
            client.vit.train()
            not_mask = torch.tensor(np.setdiff1d(all_rows, own),
                                    dtype=torch.int64).to(device)
            rng = snapshot_rng()
            torch.manual_seed(9700 + 100 * client.id + task)
            g = torch.Generator()
            g.manual_seed(9700 + 100 * client.id + task)
            loader = torch.utils.data.DataLoader(
                client.traindata, batch_size=BATCH_SIZE, num_workers=0,
                shuffle=True, generator=g)
            ces, stds, ents, tops = [], [], [], []
            k = len(own)
            with torch.no_grad():
                for bi, (x, y) in enumerate(loader):
                    if bi >= N_BATCHES:
                        break
                    x, y = x.to(device), y.long().to(device)
                    cls_features = client.original_model(x)['pre_logits']
                    res = client.vit(x, task_id=client.task_id,
                                     cls_features=cls_features, train=True)
                    lm = res['logits'].index_fill(
                        dim=1, index=not_mask, value=float('-inf'))
                    ces.append(float(criterion(lm, y)))
                    stds.append(float(lm.std(dim=1).mean()))
                    p = torch.softmax(lm, dim=1)
                    ent = -(p * torch.log(p.clamp_min(1e-12))).sum(dim=1)
                    ents.append(float((ent / np.log(k)).mean()))
                    tops.append(float(p.max(dim=1).values.mean()))
            restore_rng(rng)

            so, sp, si = (group_stats(W, b, own), group_stats(W, b, peer),
                          group_stats(W, b, inactive))
            logdiff = so['lognorm_med'] - sp['lognorm_med']
            row = dict(
                run=run_dir, task=task, client=client.id, n_mask=k,
                own_n=so['n'], own_lognorm_max=so['lognorm_max'],
                own_lognorm_med=so['lognorm_med'], own_lognorm_min=so['lognorm_min'],
                own_bias_absmax=so['bias_absmax'], own_bias_absmed=so['bias_absmed'],
                peer_lognorm_med=sp['lognorm_med'],
                peer_lognorm_max=sp['lognorm_max'],
                inactive_lognorm_med=si['lognorm_med'],
                own_minus_peer_logmed=logdiff,
                ce_mean=float(np.mean(ces)), ln_k=float(np.log(k)),
                logits_std=float(np.mean(stds)),
                entropy_norm=float(np.mean(ents)),
                top1_prob=float(np.mean(tops)),
                own_decayed=int(logdiff < -1.0),
                uniform_out=int(np.mean(ents) > 0.999),
            )
            summary_rows.append(row)
            for cidx in own:
                rn = float(W[cidx].double().norm())
                detail_rows.append(dict(
                    run=run_dir, task=task, client=client.id,
                    cls=int(cidx), lognorm=float(np.log10(max(rn, 1e-30))),
                    bias=float(b[cidx])))
            flags = []
            if row['own_decayed']:
                flags.append('OWN-DECAYED')
            if row['uniform_out']:
                flags.append('UNIFORM')
            print(f"[audit] {os.path.basename(os.path.dirname(run_dir))} "
                  f"T{task} c{client.id}: own_med={so['lognorm_med']:+.2f} "
                  f"peer_med={sp['lognorm_med']:+.2f} "
                  f"inactive_med={si['lognorm_med']:+.2f} "
                  f"CE={row['ce_mean']:.4f} (ln{k}={np.log(k):.4f}) "
                  f"H_norm={row['entropy_norm']:.4f} "
                  f"top1={row['top1_prob']:.3f} {' '.join(flags)}")
        del server

    if summary_rows:
        with open(summary_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
        with open(detail_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
            w.writeheader()
            w.writerows(detail_rows)
        print(f'saved {summary_path} ({len(summary_rows)} rows)')
        print(f'saved {detail_path} ({len(detail_rows)} rows)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', action='append', required=True,
                    help='run directory (repeatable)')
    ap.add_argument('--tasks', default='0,1,2,3,4')
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    tasks = [int(t) for t in a.tasks.split(',')]
    print('===== head row-norm audit (zero-training) =====')
    for run_dir in a.run_dir:
        audit_run(run_dir, tasks, a.device)


if __name__ == '__main__':
    main()
