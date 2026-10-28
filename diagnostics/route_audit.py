"""--route_audit instrumentation library: A/B/C timepoint key forensics.

Context (2026-10-27): the A-plan (pull decoupled from soft routing, commit
b4311c8 + be2d491) FAILED the round-0 routing health gate
(key effective rank 87.8 -> 6.59, between-sample w cosine 0.87, usage
entropy 0.85 bits), i.e. the full-pool soft pull was NOT the (sole)
collapse driver. Per external review, before any further algorithmic
change we instrument the three timepoints that partition the round:

  A  branch start     Client_DF.train() head, right after load_prompts /
                      init_prompts (round 0: every client re-initializes
                      from the SAME deterministic R1 orthogonal basis --
                      verified empirically via sha256, not assumed).
  B  post-local x5    Client_DF.train() end, from the client's OWN
                      self.prompts snapshot (NOT the shared ViT, which by
                      then holds another client's state).
  C  fusion           Server_DF.kd_fusion_prompt: pre-fusion (client-0
                      branch, the deepcopy source), first-batch
                      distillation gradient audit, post-fusion server key.

Engineering constraints (per external review, hard requirements):
  * --route_audit default OFF; when OFF, the official training path is
    untouched (bit-identical; proven by tests/test_route_audit_
    invariance.py + the official parity suite).
  * When ON, instrumentation only READS state: torch.autograd.grad with
    retain_graph=True (never writes .grad, never consumes RNG),
    detach()/deepcopy-free summaries, no loss/optimizer/seed/data change.
  * JSONL rows carry round / phase / client_id / batch_idx / seed / ts;
    raw tensors go to separate NPZ files under <run_dir>/route_audit/.

Known structural fact recorded for the C audit: kd_fusion_prompt's
optimizer targets my_result.parameters(), but the forward runs through
self.model.prompt, which load_prompts() DEEPCOPIES from my_result every
batch -- gradients land on the throwaway copy inside the model while
my_result keeps .grad=None (Adam skips grad-None params). The fusion-batch
audit therefore captures gradients w.r.t. BOTH key objects; if
optimizer-key grads come back None/zero while model-key grads are nonzero,
the SIKF fusion is empirically confirmed to be a structural no-op
(server prompt == client-0 branch verbatim), and the collapse must have
happened entirely during local training.
"""
import hashlib
import json
import math
import os
import time

import numpy as np
import torch


# ---------------------------------------------------------------------------
# key statistics (caliber MUST match diagnostics/route_weight_stats.py:
# participation ratio of the centered l2-normalized key matrix; the code is
# duplicated here so Models/ code never imports the CLI diagnostics module
# and stays free of import cycles)
# ---------------------------------------------------------------------------

def _l2_normalize(x, dim=None, epsilon=1e-12):
    """Same formula as Models/global_prompt.py l2_normalize."""
    denominator = x.norm(2, dim, keepdim=True)
    return x / (denominator + epsilon)


def key_effective_rank(key):
    """Participation ratio of centered normalized keys; fail-closed 0.0."""
    k = _l2_normalize(key.float(), dim=1)
    centered = k - k.mean(0, keepdim=True)
    s = torch.linalg.svdvals(centered)
    lam = s ** 2
    num = lam.sum() ** 2
    den = (lam ** 2).sum()
    if den == 0 or not torch.isfinite(num) or not torch.isfinite(den):
        return 0.0
    ratio = float(num / den)
    return ratio if math.isfinite(ratio) else 0.0


def sha256_tensor(t):
    """SHA-256 of the raw float32 CPU bytes -- identical keys, identical
    hash; used to empirically verify that the five round-0 re-initializations
    are byte-identical (never assumed from the shared route_seed)."""
    arr = t.detach().cpu().to(torch.float32).contiguous().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


def key_stats(key):
    """Per-pool key statistics shared by all audit timepoints."""
    k = key.detach().cpu().float()
    pool = int(k.shape[0])
    if pool == 0:
        return {"pool_size": 0, "key_effective_rank": 0.0}
    kn = _l2_normalize(k, dim=1)
    gram = kn @ kn.t()
    off = gram - torch.eye(pool, dtype=gram.dtype)
    denom = max(pool * (pool - 1), 1)
    mean_abs_off = float(off.abs().sum().item()) / denom
    norms = k.norm(dim=1)
    return {
        "pool_size": pool,
        "key_effective_rank": round(key_effective_rank(k), 4),
        "mean_abs_offdiag_cosine": round(mean_abs_off, 6),
        "key_norm_mean": round(float(norms.mean().item()), 6),
        "key_norm_std": round(float(norms.std().item()), 6),
        "key_norm_min": round(float(norms.min().item()), 6),
        "key_norm_max": round(float(norms.max().item()), 6),
        # E2 Gate 2（2026-10-28 终审 §5）：mean 容易被少数大 key 掩盖，
        # 补 median 与 p10/p25 分位。
        "key_norm_median": round(float(norms.median().item()), 6),
        "key_norm_p10": round(float(
            torch.quantile(norms, 0.10).item()), 6),
        "key_norm_p25": round(float(
            torch.quantile(norms, 0.25).item()), 6),
        # E1 臂监控（2026-10-28 终审 Gate 2）：B 臂暴露了 rank healthy ≠
        # key pool healthy——detach 后无梯度 key 被 weight decay 磨到近零
        # （mean 0.0067）。near-zero key fraction 按审查者建议取
        # ε=1e-3 / 1e-2 两档，判读时健康状态应要求两档都 ≈0。
        "key_frac_norm_lt_1e-3": round(
            float((norms < 1e-3).float().mean().item()), 6),
        "key_frac_norm_lt_1e-2": round(
            float((norms < 1e-2).float().mean().item()), 6),
        "key_sha256": sha256_tensor(k),
    }


# ---------------------------------------------------------------------------
# logger
# ---------------------------------------------------------------------------

class RouteAuditLogger(object):
    """Append-only JSONL + NPZ writer under <run_dir>/route_audit/."""

    def __init__(self, run_dir, seed):
        base = str(run_dir) if run_dir else "."
        self.dir = os.path.join(base, "route_audit")
        os.makedirs(self.dir, exist_ok=True)
        self.jsonl_path = os.path.join(self.dir, "route_audit.jsonl")
        self.seed = seed
        self.t0 = time.time()
        self._f = open(self.jsonl_path, "a", encoding="utf-8", buffering=1)
        # E2 mechanism evidence: per-(round, client) init/branch_start key
        # snapshot cache so post_local can report |K_post|/|K_init| ratio
        # stats, grouped by slots the CE selected-slot backward and/or pull
        # actually touched this round (reviewer request, non-blocking).
        self._init_key_cache = {}

    # -- internals ---------------------------------------------------------

    def _base_row(self, round_id, phase, client_id=None, batch_idx=None):
        return {
            "ts": round(time.time(), 3),
            "elapsed_s": round(time.time() - self.t0, 3),
            "seed": self.seed,
            "round": round_id,
            "phase": phase,
            "client_id": client_id,
            "batch_idx": batch_idx,
        }

    def _write(self, row, tensors=None, tag=None):
        if tensors:
            tag = tag or "r%s_%s" % (row["round"], row["phase"])
            if row["client_id"] is not None:
                tag += "_c%s" % row["client_id"]
            if row["batch_idx"] is not None:
                tag += "_b%s" % row["batch_idx"]
            npz_path = os.path.join(self.dir, tag + ".npz")
            arrays = {}
            for k, v in tensors.items():
                if torch.is_tensor(v):
                    t = v.detach().cpu()
                    arrays[k] = (t.float().numpy() if t.is_floating_point()
                                  else t.long().numpy())
                else:
                    arrays[k] = np.asarray(v)
            np.savez(npz_path, **arrays)
            row["npz"] = os.path.basename(npz_path)
        self._f.write(json.dumps(row) + "\n")
        self._f.flush()

    # -- timepoint A / B / C key snapshots ----------------------------------

    def log_key_snapshot(self, round_id, phase, client_id, key, note=None,
                         selected_slots=None):
        """Key snapshot at a timepoint. On post_local, additionally reports
        per-key norm ratio |K_post|/|K_init| (vs the cached init snapshot
        for the same (round, client)) and, when selected_slots is given
        ({"ce": [...], "pull": [...]}), the ratio grouped by touched vs
        untouched slots. Read-only; no effect on training."""
        row = self._base_row(round_id, phase, client_id=client_id)
        row.update(key_stats(key))
        if note:
            row["note"] = note
        if phase in ("init", "branch_start"):
            self._init_key_cache[(round_id, client_id)] = (
                key.detach().cpu().float().clone())
        elif phase == "post_local":
            init_key = self._init_key_cache.pop((round_id, client_id), None)
            if init_key is not None and init_key.shape == key.shape:
                init_norms = init_key.norm(dim=1)
                norms = key.detach().cpu().float().norm(dim=1)
                ratio = norms / (init_norms + 1e-12)
                row["key_norm_ratio_mean"] = round(
                    float(ratio.mean().item()), 6)
                row["key_norm_ratio_median"] = round(
                    float(ratio.median().item()), 6)
                row["key_norm_ratio_p10"] = round(float(
                    torch.quantile(ratio, 0.10).item()), 6)
                row["key_norm_ratio_p25"] = round(float(
                    torch.quantile(ratio, 0.25).item()), 6)
                row["key_norm_ratio_min"] = round(
                    float(ratio.min().item()), 6)
                row["key_norm_ratio_max"] = round(
                    float(ratio.max().item()), 6)
            if selected_slots:
                ce = sorted(set(int(s) for s in
                                (selected_slots.get("ce") or [])))
                pull = sorted(set(int(s) for s in
                                  (selected_slots.get("pull") or [])))
                row["n_selected_slots_ce"] = len(ce)
                row["n_selected_slots_pull"] = len(pull)
                union = sorted(set(ce) | set(pull))
                row["n_selected_slots_union"] = len(union)
                if init_key is not None and init_key.shape == key.shape \
                        and union:
                    mask = torch.zeros(ratio.shape[0], dtype=torch.bool)
                    mask[torch.tensor(union, dtype=torch.long)] = True
                    row["key_norm_ratio_selected_mean"] = round(float(
                        ratio[mask].mean().item()), 6)
                    row["key_norm_ratio_unselected_mean"] = round(float(
                        ratio[~mask].mean().item()), 6) \
                        if bool((~mask).any()) else None
        self._write(row, tensors={"key": key})
        return row

    # -- local-training first-batch gradient audit (timepoint B support) ----

    def log_client_grad_audit(self, round_id, client_id, batch_idx,
                              loss_ce, pull_term, key_param, selected_idx):
        """torch.autograd.grad on the REAL forward graph of the first local
        minibatch. Never writes .grad / consumes RNG; the training
        loss.backward() afterwards is untouched."""
        g_ce = torch.autograd.grad(loss_ce, key_param,
                                   retain_graph=True, allow_unused=True)[0]
        g_pull = torch.autograd.grad(pull_term, key_param,
                                     retain_graph=True, allow_unused=True)[0]
        row = self._base_row(round_id, "grad_audit", client_id=client_id,
                             batch_idx=batch_idx)
        row["key_grad_none_after_audit"] = key_param.grad is None
        if selected_idx is not None:
            sel = sorted(set(int(s) for s in
                             selected_idx.detach().cpu().flatten().tolist()))
            row["pull_selected_slots"] = sel
        else:
            sel = None
            row["pull_selected_slots"] = None
            row["note"] = "no pull_idx in output (official hard path)"
        if g_ce is None and g_pull is None:
            row["note"] = (row.get("note", "") +
                           " no gradient path from loss to prompt_key")
            self._write(row)
            return row
        pool = key_param.shape[0]
        row["g_ce_exists"] = g_ce is not None
        row["g_pull_exists"] = g_pull is not None
        if g_ce is not None:
            slot_ce = g_ce.norm(dim=1)
            row["g_ce_total_norm"] = round(float(slot_ce.norm().item()), 6)
        if g_pull is not None:
            slot_pull = g_pull.norm(dim=1)
            row["g_pull_total_norm"] = round(float(slot_pull.norm().item()), 6)
        if g_ce is not None and g_pull is not None:
            row["cos_g_ce_g_pull"] = round(float(torch.nn.functional.cosine_similarity(
                g_ce.reshape(-1), g_pull.reshape(-1), dim=0).item()), 6)
            mask = torch.zeros(pool, dtype=torch.bool)
            if sel:
                mask[torch.tensor(sel, dtype=torch.long)] = True
            row["g_ce_norm_selected_mean"] = round(float(
                slot_ce[mask].mean().item()), 6) if mask.any() else None
            row["g_ce_norm_unselected_mean"] = round(float(
                slot_ce[~mask].mean().item()), 6) if (~mask).any() else None
            row["g_pull_norm_selected_mean"] = round(float(
                slot_pull[mask].mean().item()), 6) if mask.any() else None
            row["g_pull_norm_unselected_mean"] = round(float(
                slot_pull[~mask].mean().item()), 6) if (~mask).any() else None
            row["g_pull_unselected_is_zero"] = bool(
                float(slot_pull[~mask].abs().max().item()) == 0.0) \
                if (~mask).any() else True
        tensors = {}
        if g_ce is not None:
            tensors["g_ce"] = g_ce
            tensors["g_ce_slot_norms"] = g_ce.norm(dim=1)
        if g_pull is not None:
            tensors["g_pull"] = g_pull
            tensors["g_pull_slot_norms"] = g_pull.norm(dim=1)
        if selected_idx is not None:
            tensors["pull_selected_idx"] = selected_idx
        self._write(row, tensors=tensors)
        return row

    # -- cross-branch alignment (after all local trainings) ------------------

    def log_cross_branch(self, round_id, keys_by_client):
        """Between-branch key alignment: same-slot cosine, full cross-Gram
        (NPZ), and top-10 principal-angle cosine summary of the row
        subspaces. Separates 'each branch collapsed internally' from
        'branches healthy but drifted into different subspaces'."""
        ids = sorted(keys_by_client)
        n = len(ids)
        kn = {cid: _l2_normalize(keys_by_client[cid].detach().cpu().float(),
                                dim=1) for cid in ids}
        same_slot = np.zeros((n, n), dtype=np.float64)
        subspace = np.zeros((n, n), dtype=np.float64)
        grams = {}
        diag_rank = []
        for a, i in enumerate(ids):
            diag_rank.append(round(key_effective_rank(kn[i]), 4))
            for b, range_j in enumerate(ids):
                if range_j <= i:
                    continue
                Ki, Kj = kn[i], kn[range_j]
                same_slot[a, b] = same_slot[b, a] = float(
                    (Ki * Kj).sum(dim=1).mean().item())
                grams["gram_c%s_c%s" % (i, range_j)] = (Ki @ Kj.t()).numpy()
                Qi, _ = torch.linalg.qr(Ki.t().float())   # dim x pool
                Qj, _ = torch.linalg.qr(Kj.t().float())
                sv = torch.linalg.svdvals(Qi.t() @ Qj)
                subspace[a, b] = subspace[b, a] = float(sv[:10].mean().item())
        row = self._base_row(round_id, "cross_branch")
        row["clients"] = ids
        row["per_branch_effective_rank"] = diag_rank
        row["same_slot_cosine_matrix"] = np.round(same_slot, 6).tolist()
        row["subspace_top10_cosine_matrix"] = np.round(subspace, 6).tolist()
        if n > 1:
            off = same_slot[np.triu_indices(n, k=1)]
            row["same_slot_cosine_mean"] = round(float(off.mean()), 6)
            off_sub = subspace[np.triu_indices(n, k=1)]
            row["subspace_top10_cosine_mean"] = round(float(off_sub.mean()), 6)
        self._write(row, tensors=grams)
        return row

    # -- fusion (SIKF) first-batch audit (timepoint C) -----------------------

    def log_fusion_audit(self, round_id, batch_idx, loss, model_key_param,
                         optimizer_key_param, student_out, teacher_outs,
                         teacher_ids, per_teacher_losses):
        """Distillation audit on the first fusion minibatch.

        Gradients are captured w.r.t. BOTH key objects:
          * model_key  = self.model.prompt.prompt_key  (the throwaway
            deepcopy the forward actually runs through), and
          * optimizer_key = my_result.prompt_key (the deepcopy client-0
            branch the Adam optimizer targets).
        If optimizer_key grads are None while model_key grads are finite,
        the SIKF fusion is empirically confirmed to be a structural no-op
        (load_prompts deepcopy severs the optimizer->forward link).
        """
        g_model = torch.autograd.grad(loss, model_key_param,
                                      retain_graph=True,
                                      allow_unused=True)[0]
        g_opt = torch.autograd.grad(loss, optimizer_key_param,
                                    retain_graph=True,
                                    allow_unused=True)[0]
        with torch.no_grad():
            flats = [t.reshape(-1).float() for t in teacher_outs]
            student = student_out.reshape(-1).float()
            n = len(flats)
            pair = np.zeros((n, n), dtype=np.float64)
            for a in range(n):
                for b in range(a + 1, n):
                    c = float(torch.nn.functional.cosine_similarity(
                        flats[a], flats[b], dim=0).item())
                    pair[a, b] = pair[b, a] = c
            s_vs_t = [round(float(torch.nn.functional.cosine_similarity(
                student, f, dim=0).item()), 6) for f in flats]
        row = self._base_row(round_id, "fusion_audit", batch_idx=batch_idx)
        row["teacher_ids"] = teacher_ids
        row["per_teacher_distill_loss"] = [
            round(float(v), 6) for v in per_teacher_losses]
        row["teacher_pairwise_output_cosine"] = np.round(pair, 6).tolist()
        row["student_teacher_output_cosine"] = s_vs_t
        row["g_model_key_exists"] = g_model is not None
        row["g_optimizer_key_exists"] = g_opt is not None
        row["optimizer_target_grad_was_none"] = g_opt is None
        if g_model is not None:
            row["g_model_key_total_norm"] = round(
                float(g_model.norm().item()), 6)
        if g_opt is not None:
            row["g_optimizer_key_total_norm"] = round(
                float(g_opt.norm().item()), 6)
        tensors = {}
        if g_model is not None:
            tensors["g_model_key"] = g_model
        if g_opt is not None:
            tensors["g_optimizer_key"] = g_opt
        self._write(row, tensors=tensors)
        return row
