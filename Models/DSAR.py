"""DSAR (Decoupled State-Aligned Readout) B1 sidecar -- numerical core.

RUNCOMMANDS §34.4 (B1 prereg v1, frozen 2026-11-06) + §34.4a (Dispatch
Audit Addendum, APPROVED 2026-11-06, verdict SHARED_PATH_CONFIRMED).

Deployment graph (§34.4a erratum; z_ref is a frozen-reference stream and
NEVER enters the head -- the true Chead input is
[corrected z_op || routed anchor(corrected z_op)]):

    current task  k = t :  y_t = TA_t(L_{c,r}(B_{c,r}(z_op+)))
    historical    k < t :  y_k = TA_k(B_{c,r}(z_op+))     (B-only, L forbidden)

B/L are inserted in Client_DF.evaluate()/evaluate_with_margin() BEFORE
the self.model() call, so the corrected feature feeds BOTH the
key-anchor routing and the Chead feature half. The routed anchor index
is allowed to change naturally (B0-preview deployment semantics); fixing
baseline anchor indices is forbidden.

Four immutable semantics (§34.4a):
  1. B before self.model() -> routing + readout both see corrected z.
  2. k<t gets B only.
  3. head selection stays self.heads[k] + original key routing + class
     mask.
  4. B active/identity is decided ONLY by the current-task train-split
     unlabeled calibration; historical accuracy / test labels must
     never enter the decision. B identity != DSAR identity: each
     adapter short-circuits independently (a B=identity sidecar still
     applies L to the current task).

Frozen engineering constants:
  gamma_star = 0.396419        (bsab_b0/gamma_star_corrected.json export;
                                ERRATUM-gamma-split §34.4c 2026-11-06:
                                superseded 0.390324 was computed on test
                                rows; corrected value uses train rows)
  lambda_{c,r} = N * s * gamma_star,  s = tr(X^T X)/(N*d), X = centered
  z+ train rows of the current cell (fit half). No GCV, no sweep.
  rank_B = rank_L = 8 (deployed factors: 2 x 768x8 + 768 bias)
  LOW_POWER_IDENTITY: fit rows < 32 OR calibration rows < 8 -> exact
  identity (integrity failures -- hash mismatch / NaN / Inf / solver
  error -- must NEVER be downgraded via this clause; they raise).
  L recipe: 100 epochs, bs=64, torch.optim.Adam (NOT AdamW), lr=1e-3,
  weight_decay=1e-4, explicit lambda_L = 0 (no extra residual term),
  U nonzero / V zero / bias zero init, no early stopping, no val/test
  parameter selection.

Identity guard (unlabeled, §34.4 item 2, 2026-11-06 review ruling):
train rows are split 80/20 by a deterministic sample-index hash (no RNG
consumed); the fixed-lambda fit-half solution is TRUNCATED to the
deployed rank-8 candidate (with its bias recomputed under the truncated
operator, b_8 = mu_y - mu_x (I + dA_8)); the guard compares THIS r8
candidate against identity by paired MSE on the 20% calibration rows --
never the untruncated full-ridge (which can win MSE while r8 loses).
If the r8 candidate wins it is refitted once on ALL train rows and
truncated again. Test/val rows never enter any decision.

RNG discipline: every routine uses a dedicated torch.Generator seeded
from a sha256-stable derivation; the global torch / python / numpy
streams are never consumed. Client_DF hooks additionally wrap the fits
in the four-stream RNG snapshot (belt and braces).

Numerics: the ridge solve runs in torch float64 on CPU (host
numpy/scipy float64 BLAS is broken; torch is not -- ENVIRONMENT_AUDIT).
Factors are deployed as float32 on the capture device.
"""

import hashlib

import torch
import torch.nn.functional as F

FEAT_DIM = 768

# ---- frozen constants (§34.4 / §34.4a) --------------------------------
# ERRATUM-gamma-split (§34.4c, 2026-11-06): superseded GAMMA_STAR=0.390324
# was exported with s computed on test rows (split==1 misread); corrected
# export uses train rows (split==0). Approved pre-formal-run correction.
GAMMA_STAR = 0.396419
DSAR_RANK = 8
MIN_FIT_ROWS = 32      # LOW_POWER_IDENTITY threshold for fit rows
MIN_CAL_ROWS = 8       # LOW_POWER_IDENTITY threshold for calibration rows

# ---- INR dataset-conditional recipe (§34.7, frozen 2026-11-08) ---------
# CIFAR keeps the §34.4 frozen recipe bitwise (gamma_star 0.396419 / L r8);
# ImageNet-R swaps two dataset-conditional constants and NOTHING else:
#   gamma_star_inr = 2.0579118185313345  (provenance §34.6a, commit 6cd83d9)
#   L rank = 16                          (te_pooled +2.287 vs r8 +2.361,
#                                         but worst cell -1.20 vs -2.36 and
#                                         p5 -0.56 vs -0.84; near-identical
#                                         mean with a materially safer tail)
# Evidence (baseline joint-head probe, paper readout scope):
#   L16(zm, no guard) chained after B8(gamma_inr) = +2.21 te-pooled;
#   L zp basis is NOT adopted -- production fits L on z- rows and deploys
#   on B(z+), the probe zm-chain semantics; switching to zp buys +0.08pp
#   and breaks CIFAR structural parity. B rank 8 + mse guard and the
#   no-guard L are shared with CIFAR (production L never had a CE guard;
#   probe L*_ce/cer arms measured the hypothetical guard and it is
#   ANTI-selective on INR: the rejected 16% of cells average +11.5pp true
#   gain). Historical-row evidence (dsar_inr_hist_probe): B8/gamma_inr
#   +0.36 hist_mean, the only all-distance-positive config -> B stays r8
#   on INR (r24 was current-row-optimal but historical-row-negative).
GAMMA_STAR_INR = 2.0579118185313345
DSAR_RANK_L_INR = 16


def dsar_recipe(data_name):
    """Dataset-conditional DSAR recipe. CIFAR path returns the §34.4
    frozen constants (bitwise-unchanged defaults for every fit function);
    ImageNet-R returns the §34.7 INR recipe. Callers must pass these into
    dsar_fit_bridge / dsar_fit_l / rank_shapes_ok."""
    if str(data_name) == "ImageNet-R":
        return {"gamma_star": GAMMA_STAR_INR,
                "l_rank": DSAR_RANK_L_INR,
                "recipe": "dsar-inr-v1"}
    return {"gamma_star": GAMMA_STAR, "l_rank": DSAR_RANK,
            "recipe": "dsar-b1-cifar"}

L_EPOCHS = 100
L_BATCH = 64
L_LR = 1e-3
L_WD = 1e-4
LAMBDA_L = 0.0         # explicit residual regularizer (frozen at zero)

DSAR_VERSION = "dsar-b1-shell-v2-gamma-erratum"


def dsar_stable_seed(*parts):
    """Deterministic 31-bit seed (same construction as
    Client_DF.a5_stable_seed / probe stable_seed)."""
    key = "|".join(str(p) for p in parts)
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4],
                          "little")


# ------------------------------------------------------------------ split

def dsar_split_fit_cal(sample_idx):
    """80/20 fit/calibration split by deterministic sample-index hash
    (consumes no RNG). Returns boolean masks (fit, cal)."""
    n = len(sample_idx)
    fit = [True] * n
    cal = [False] * n
    for i, idx in enumerate(sample_idx):
        h = int.from_bytes(
            hashlib.sha256(("dsar_split:%d" % int(idx)).encode())
            .digest()[:4], "little")
        if h % 5 == 0:
            fit[i] = False
            cal[i] = True
    return fit, cal


def _apply_factors(z, M1, M2, bias):
    """Deployment readout convention: z + (z @ M1) @ M2.T + bias."""
    return z + (z @ M1) @ M2.t() + bias


# ------------------------------------------------------------------ sidecar

class DSARSidecar(object):
    """Client-local committed sidecar state for ONE broadcast boundary.

    Holds ONLY r8 factors (M1, M2 in R^{768x8}, bias in R^768) per
    adapter -- the full 768x768 matrices exist only inside the solver
    and are never stored (FL communication = 0; ~13k floats/adapter).

    apply(z, is_current_task) implements the frozen task dispatch:
    B applies to every task, L only to the current task. Inactive
    adapters short-circuit independently; with both inactive the SAME
    tensor object is returned (bitwise-neutral path).
    """

    __slots__ = ("b_active", "b_M1", "b_M2", "b_bias",
                 "l_active", "l_M1", "l_M2", "l_bias", "meta")

    def __init__(self, b_active, b_M1, b_M2, b_bias,
                 l_active, l_M1, l_M2, l_bias, meta):
        self.b_active = bool(b_active)
        self.b_M1 = b_M1
        self.b_M2 = b_M2
        self.b_bias = b_bias
        self.l_active = bool(l_active)
        self.l_M1 = l_M1
        self.l_M2 = l_M2
        self.l_bias = l_bias
        self.meta = dict(meta) if meta is not None else {}

    # -- frozen dispatch ------------------------------------------------
    def apply(self, z, is_current_task):
        if self.b_active:
            z = _apply_factors(z, self.b_M1, self.b_M2, self.b_bias)
        if is_current_task and self.l_active:
            z = _apply_factors(z, self.l_M1, self.l_M2, self.l_bias)
        return z

    def is_identity(self):
        return not (self.b_active or self.l_active)

    # -- checkpoint -----------------------------------------------------
    def state_dict(self):
        def _f(M1, M2, bias):
            if M1 is None:
                return None
            return {"M1": M1.detach().cpu().clone(),
                    "M2": M2.detach().cpu().clone(),
                    "bias": bias.detach().cpu().clone()}
        return {
            "version": DSAR_VERSION,
            "b_active": self.b_active, "b": _f(self.b_M1, self.b_M2,
                                              self.b_bias),
            "l_active": self.l_active, "l": _f(self.l_M1, self.l_M2,
                                              self.l_bias),
            "meta": self.meta,
        }

    @classmethod
    def from_state(cls, state, device):
        def _f(d):
            if d is None:
                return None, None, None
            return (d["M1"].to(device), d["M2"].to(device),
                    d["bias"].to(device))
        bM1, bM2, bb = _f(state.get("b"))
        lM1, lM2, lb = _f(state.get("l"))
        return cls(state["b_active"], bM1, bM2, bb,
                   state["l_active"], lM1, lM2, lb,
                   state.get("meta", {}))

    def factors_finite(self):
        ok = True
        for M1, M2, bias in ((self.b_M1, self.b_M2, self.b_bias),
                             (self.l_M1, self.l_M2, self.l_bias)):
            if M1 is None:
                continue
            ok = ok and bool(torch.isfinite(M1).all()
                            and torch.isfinite(M2).all()
                            and torch.isfinite(bias).all())
        return ok

    def rank_shapes_ok(self, b_rank=DSAR_RANK, l_rank=DSAR_RANK):
        """Per-adapter rank validation. Defaults keep the §34.4 CIFAR
        semantics (r8 everywhere); INR callers pass l_rank=16 (§34.7)."""
        for M1, M2, bias, r in ((self.b_M1, self.b_M2, self.b_bias, b_rank),
                                 (self.l_M1, self.l_M2, self.l_bias, l_rank)):
            if M1 is None:
                continue
            if (tuple(M1.shape) != (FEAT_DIM, r)
                    or tuple(M2.shape) != (FEAT_DIM, r)
                    or tuple(bias.shape) != (FEAT_DIM,)):
                return False
        return True


def dsar_identity_sidecar(round_id, entry, task_id, status,
                           prompt_hash=None):
    """Exact-identity sidecar (no adapters). status documents WHY
    (IDENTITY_TEST / NO_BIRTH_STATE / LOW_POWER_IDENTITY /
    IDENTITY_GUARD at commit time)."""
    return DSARSidecar(
        False, None, None, None, False, None, None, None,
        {"version": DSAR_VERSION, "round_id": int(round_id),
         "entry": str(entry), "task_id": int(task_id),
         "status": str(status), "prompt_hash": prompt_hash})


# ------------------------------------------------------------------ B fit

def dsar_fit_bridge(zp, zm, sample_idx, seed, gamma_star=GAMMA_STAR):
    """B_{c,r}: label-free fixed-lambda residual ridge z+ -> z-.

    Frozen recipe (§34.4 item 2, no GCV; 2026-11-06 review rulings
    applied -- truncated-bias recompute + r8-candidate guard).
    gamma_star defaults to the CIFAR frozen constant (bitwise-unchanged
    CIFAR path); INR callers pass GAMMA_STAR_INR (§34.7).
      1. 80/20 fit/cal split by sample-index hash (deterministic).
      2. LOW_POWER_IDENTITY: fit rows < 32 or cal rows < 8 -> identity.
      3. Fixed lambda = N * s * GAMMA_STAR on the fit half
         (s = tr(X^T X)/(N*d) of the centered z+ fit rows); residual
         form: solve (XtX + lam*I) dA = Xc^T (Yc - Xc), A = I + dA.
      4. Identity guard (unlabeled, calibration rows, paired MSE):
         the fit-half solution is TRUNCATED to the deployed rank-8
         candidate with its bias RECOMPUTED under the truncated
         operator, b_8 = mu_y - mu_x (I + dA_8); the guard compares
         THIS r8 candidate (the exact deployment object) against
         identity. Identity wins ties (mse_id <= mse_r8 -> identity).
      5. If the r8 candidate wins: refit once on ALL train rows,
         truncate again to rank 8, and RECOMPUTE the deployment bias
         under the truncated operator:
             b_8 = mu_y - mu_x - mu_x @ (M1 @ M2^T)
         (NOT the full-refit intercept; ruling 2026-11-06, supersedes
         the B0 bt_rank keep-full-intercept precedent).

    zp/zm: (n, 768) paired float32 tensors (same device); zm is the
    birth-side (z-) target. Labels are NOT accepted by this function.
    Returns (factors or None, status, aux); aux records
    mse_bridge_r8_cal / mse_identity_cal / lam_fit / lam_refit for
    G2 numerical assertions (IDENTITY_GUARD <=> mse_id <= mse_r8_cal).
    """
    n = zp.shape[0]
    if zp.shape != zm.shape or zp.shape[1] != FEAT_DIM:
        raise RuntimeError(
            "dsar_fit_bridge: paired shapes mismatch %s vs %s (fail-closed)"
            % (tuple(zp.shape), tuple(zm.shape)))
    fit, cal = dsar_split_fit_cal(sample_idx)
    n_fit, n_cal = int(sum(fit)), int(sum(cal))
    aux = {"n_rows": int(n), "n_fit": n_fit, "n_cal": n_cal, "seed": seed}
    if n_fit < MIN_FIT_ROWS or n_cal < MIN_CAL_ROWS:
        return None, "LOW_POWER_IDENTITY", aux

    def _solve(zp_rows, zm_rows):
        """One fixed-lambda residual-ridge solve on CPU float64.
        Returns (dA, xbar, ybar, lam); dA is the residual (A = I + dA)."""
        X64 = zp_rows.detach().double().cpu()
        Y64 = zm_rows.detach().double().cpu()
        n_, d_ = X64.shape
        xbar, ybar = X64.mean(dim=0), Y64.mean(dim=0)
        xc, yc = X64 - xbar, Y64 - ybar
        s = float((xc * xc).sum()) / float(n_ * d_)
        lam = float(n_) * s * gamma_star
        XtX = xc.t() @ xc
        reg = torch.eye(d_, dtype=torch.float64) * lam
        D = yc - xc
        dA = torch.linalg.solve(XtX + reg, xc.t() @ D)
        if not bool(torch.isfinite(dA).all()):
            raise RuntimeError(
                "dsar_fit_bridge: non-finite ridge solution (integrity "
                "failure; NOT downgradable to LOW_POWER; fail-closed)")
        return dA, xbar, ybar, lam

    def _trunc8(dA):
        """Deterministic SVD truncation of the residual map.
        Returns (M1, M2) float64 with dA_8 = M1 @ M2^T
        (M1 = U_r * S_r, M2 = V_r)."""
        U_, S_, Vt_ = torch.linalg.svd(dA, full_matrices=False)
        if not (bool(torch.isfinite(U_).all()) and bool(torch.isfinite(S_).all())
                and bool(torch.isfinite(Vt_).all())):
            raise RuntimeError(
                "dsar_fit_bridge: non-finite SVD factors (integrity "
                "failure; NOT downgradable to LOW_POWER; fail-closed)")
        r = DSAR_RANK
        return U_[:, :r] * S_[:r], Vt_[:r].t().contiguous()

    def _bias8(xbar, ybar, M1, M2):
        """Bias recomputed under the TRUNCATED operator (ruling
        2026-11-06): b_8 = mu_y - mu_x (I + dA_8)
                         = mu_y - mu_x - mu_x @ (M1 @ M2^T)."""
        b8 = ybar - xbar - xbar @ (M1 @ M2.t())
        if not bool(torch.isfinite(b8).all()):
            raise RuntimeError(
                "dsar_fit_bridge: non-finite truncated bias (integrity "
                "failure; NOT downgradable to LOW_POWER; fail-closed)")
        return b8

    def _mse_r8(zp_rows, zm_rows, M1, M2, b8):
        """Paired MSE of the exact r8 deployment candidate
        z -> z + (z @ M1) @ M2^T + b8 on the given rows (float64)."""
        Z = zp_rows.detach().double().cpu()
        T = zm_rows.detach().double().cpu()
        out = Z + (Z @ M1) @ M2.t() + b8
        resid = out - T
        return float((resid * resid).mean())

    # -- guard: fit-half solve -> TRUNCATE to r8 -> recompute b_8 ------
    zp_fit = zp[torch.tensor(fit, device=zp.device)]
    zm_fit = zm[torch.tensor(fit, device=zp.device)]
    zp_cal = zp[torch.tensor(cal, device=zp.device)]
    zm_cal = zm[torch.tensor(cal, device=zp.device)]
    dA_g, xbar_g, ybar_g, lam_g = _solve(zp_fit, zm_fit)
    M1_g, M2_g = _trunc8(dA_g)
    b_g = _bias8(xbar_g, ybar_g, M1_g, M2_g)
    mse_bridge_r8_cal = _mse_r8(zp_cal, zm_cal, M1_g, M2_g, b_g)
    diff = zp_cal.detach().double().cpu() - zm_cal.detach().double().cpu()
    mse_id = float((diff * diff).mean())
    aux.update({"lam_fit": lam_g, "mse_bridge_r8_cal": mse_bridge_r8_cal,
                "mse_identity_cal": mse_id})
    if mse_id <= mse_bridge_r8_cal:
        return None, "IDENTITY_GUARD", aux

    # -- active: refit once on ALL train rows, truncate again to r8,
    #    and recompute the deployment bias under the truncated operator.
    dA_f, xbar_f, ybar_f, lam_f = _solve(zp, zm)
    M1_f64, M2_f64 = _trunc8(dA_f)
    bias_f64 = _bias8(xbar_f, ybar_f, M1_f64, M2_f64)
    M1 = M1_f64.float().to(zp.device)
    M2 = M2_f64.float().to(zp.device)
    bias = bias_f64.float().to(zp.device)
    if not (bool(torch.isfinite(M1).all()) and bool(torch.isfinite(M2).all())
            and bool(torch.isfinite(bias).all())):
        raise RuntimeError(
            "dsar_fit_bridge: non-finite r8 factors (integrity failure; "
            "NOT downgradable to LOW_POWER; fail-closed)")
    aux.update({"lam_refit": lam_f, "rank": DSAR_RANK})
    factors = {"M1": M1, "M2": M2, "bias": bias}
    return factors, "FITTED", aux


# ------------------------------------------------------------------ L fit

def dsar_fit_l(zm, y, readout_logits, seed, epochs=L_EPOCHS,
               batch=L_BATCH, lr=L_LR, wd=L_WD, rank=DSAR_RANK):
    """L_{c,r}: residual adapter trained by masked CE through the
    FROZEN current-task readout (eval semantics, dropout OFF; the caller
    loads heads[task_id] into self.model before fitting).

    rank defaults to the CIFAR frozen value 8 (bitwise-unchanged CIFAR
    path); INR callers pass 16 (§34.7).

    Parameterization (fit_lt convention): z + (z@V)@U.T + b with the
    FROZEN init U nonzero (randn*0.01) / V zero / bias zero -> exact
    identity at epoch 0. torch.optim.Adam (NOT AdamW), lr 1e-3,
    weight_decay 1e-4, explicit lambda_L = 0, 100 epochs, bs 64, no
    early stopping, no val/test selection.

    All randomness comes from a dedicated torch.Generator (CPU); the
    global RNG streams are never touched. MIN_FIT_ROWS < 32 -> LOW_POWER
    identity.

    readout_logits(z_batch, y_batch) must return the CLASS-MASKED logits
    (the official phase-2 masked-CE semantics).
    Returns (factors or None, status, final_ce).
    """
    n, d = zm.shape
    if d != FEAT_DIM:
        raise RuntimeError("dsar_fit_l: expected %d-dim features, got %d"
                           % (FEAT_DIM, d))
    if n < MIN_FIT_ROWS:
        return None, "LOW_POWER_IDENTITY", None
    g = torch.Generator()          # dedicated stream, CPU-side only
    g.manual_seed(int(seed))

    dev = zm.device
    U = (torch.randn(FEAT_DIM, rank, generator=g) * 0.01).to(dev)
    V = torch.zeros(FEAT_DIM, rank, device=dev)
    b = torch.zeros(FEAT_DIM, device=dev)
    U.requires_grad_(True)
    V.requires_grad_(True)
    b.requires_grad_(True)
    opt = torch.optim.Adam([U, V, b], lr=lr, weight_decay=wd)
    bs = min(batch, n)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g).to(dev)
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            zb, yb = zm[idx], y[idx]
            z_adj = zb + (zb @ V) @ U.t() + b
            loss = F.cross_entropy(readout_logits(z_adj, yb), yb)
            # lambda_L = 0 (frozen): no explicit residual regularizer is
            # added; the ONLY regularization is Adam's weight_decay=wd.
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        z_adj = zm + (zm @ V) @ U.t() + b
        final = float(F.cross_entropy(readout_logits(z_adj, y), y).item())
    if not (bool(torch.isfinite(U).all()) and bool(torch.isfinite(V).all())
            and bool(torch.isfinite(b).all()) and final == final):
        raise RuntimeError(
            "dsar_fit_l: non-finite L factors or CE (integrity failure; "
            "NOT downgradable to LOW_POWER; fail-closed)")
    factors = {"M1": V.detach(), "M2": U.detach(), "bias": b.detach()}
    return factors, "FITTED", final
