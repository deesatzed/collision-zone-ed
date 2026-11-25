"""Core statistical engine for Three-Predicates Collision-Zone Discovery & Specialist Stage-2 Learning.

Implements:
  1. Predicate 1 (Schervish Beta-Kernel Collision-Zone Proper Scoring Rule S_{tau*, kappa}):
     Eliminates the 1/(tau*(1-tau)) True-Negative singularity of ordinary log-loss while
     preserving strict properness, closed-form regularized incomplete beta evaluation, and
     exact two-phase Horvitz-Thompson variance + Holm step-down LCB null FWER control.
  2. Predicate 2 (High-Signal False Positive Constrained Neyman Allocation):
     Concentrates Phase-2 pilot controls on high-baseline-risk non-events (Y=0, p_1 ~ tau*)
     alongside boundary cases rather than easy True Negatives.
  3. Predicate 3 (Posterior-Invariant Collision-Zone Specialist Stage-2 GBDT):
     Anchors Stage-2 trees on calibrated Stage-1 baseline logits and weights training split-gain
     by X_base-measurable collision relevance omega_CZ(X_base), leaving P(Y=1 | X_base, Z_S)
     invariant while reallocating tree capacity from easy True Negatives to the collision zone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from scipy.special import betainc, betaln, expit, logit
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

_EPS = 1e-12


@dataclass(frozen=True, slots=True)
class ClinicalContract:
    """Clinical decision contract inducing action threshold tau* = C_FP / (C_FP + C_FN)."""

    cost_fp: float = 1.0
    cost_fn: float = 10.0
    target_precision: float = 0.40
    per_instance_query_cost: float = 0.04

    @property
    def tau_star(self) -> float:
        return float(self.cost_fp / (self.cost_fp + self.cost_fn))


@dataclass(frozen=True, slots=True)
class ChannelDef:
    """Candidate Stage-2 clinical diagnostic or HIE channel."""

    channel_id: str
    columns: tuple[str, ...]
    cost: float = 1.0
    is_null_stress: bool = False
    is_twin_stress: bool = False


@dataclass(frozen=True, slots=True)
class DiscoveryStepLog:
    step: int
    selected_id: str | None
    gain: float
    se_total: float
    z_stat: float
    p_holm: float
    cost: float


@dataclass
class DiscoveryOutcome:
    method: str
    pilot_fraction: float
    selected_ids: list[str]
    total_cost: float
    pilot_high_signal_fp_share: float
    steps: list[DiscoveryStepLog] = field(default_factory=list)
    step1_scores: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Predicate 1: Schervish Beta-Kernel Collision-Zone Proper Scoring Rule
# ---------------------------------------------------------------------------


def collision_beta_params(tau_star: float, kappa: float = 24.0) -> tuple[float, float]:
    """Return (alpha, beta) with exact mode at tau* and w(0) = w(1) = 0."""
    t = float(np.clip(tau_star, 1e-4, 1.0 - 1e-4))
    k = max(0.0, float(kappa))
    return 1.0 + k * t, 1.0 + k * (1.0 - t)


def schervish_collision_weight(
    p: np.ndarray,
    tau_star: float,
    kappa: float = 24.0,
) -> np.ndarray:
    """Evaluate normalized Beta(1 + kappa*tau*, 1 + kappa*(1-tau*)) density at p."""
    pv = np.clip(np.asarray(p, dtype=float), _EPS, 1.0 - _EPS)
    a, b = collision_beta_params(tau_star, kappa)
    log_w = (a - 1.0) * np.log(pv) + (b - 1.0) * np.log(1.0 - pv) - betaln(a, b)
    return np.exp(log_w)


def schervish_collision_proper_score(
    y: np.ndarray,
    p: np.ndarray,
    tau_star: float,
    kappa: float = 24.0,
) -> np.ndarray:
    """Closed-form strictly proper Collision-Zone score S_{tau*, kappa}(y, p) <= 0 (higher is better).

    Derivative w.r.t. p of E_{Y ~ Bern(eta)}[S(Y, p)] is identically:
        w_{alpha, beta}(p) * (eta - p),
    proving strict properness on (0, 1) while eliminating the 1/(tau*(1-tau)) singularity at 0.
    """
    yv = np.asarray(y, dtype=float)
    pv = np.clip(np.asarray(p, dtype=float), _EPS, 1.0 - _EPS)
    a, b = collision_beta_params(tau_star, kappa)
    ab = a + b
    # Schervish integral:
    # y=1 branch: -int_p^1 (1 - tau) w(tau) dtau = -(b / (a+b)) * (1 - I_p(a, b+1))
    s1 = -(b / ab) * (1.0 - betainc(a, b + 1.0, pv))
    # y=0 branch: -int_0^p tau w(tau) dtau       = -(a / (a+b)) * I_p(a+1, b)
    s0 = -(a / ab) * betainc(a + 1.0, b, pv)
    return np.where(yv >= 0.5, s1, s0)


def log_likelihood_score(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Standard Bernoulli log-likelihood (Paper 1 global CMI score)."""
    yv = np.asarray(y, dtype=float)
    pv = np.clip(np.asarray(p, dtype=float), _EPS, 1.0 - _EPS)
    return yv * np.log(pv) + (1.0 - yv) * np.log(1.0 - pv)


def collision_zone_relevance_weights(
    p_base: np.ndarray,
    tau_star: float,
    kappa: float = 24.0,
    floor: float = 0.05,
) -> np.ndarray:
    """Compute X_base-measurable collision weight omega_CZ(p_base) in [floor, 1 + floor].

    Because omega_CZ depends strictly on X_base (never Y), P_Q(Y=1 | X_base, Z_S) = P(Y=1 | X_base, Z_S)
    identically (Proposition 3: zero prior-shift calibration distortion).
    """
    pv = np.clip(np.asarray(p_base, dtype=float), _EPS, 1.0 - _EPS)
    raw = schervish_collision_weight(pv, tau_star, kappa)
    peak = float(schervish_collision_weight(np.array([tau_star]), tau_star, kappa)[0])
    normed = raw / max(_EPS, peak)
    return float(floor) + normed


# ---------------------------------------------------------------------------
# Predicate 2: Two-Phase Water-Filling Pilot Allocation & Exact Variance
# ---------------------------------------------------------------------------


def waterfill_neyman(
    pop_counts: np.ndarray,
    dispersions: np.ndarray,
    n_pilot: int,
    min_per_stratum: int = 2,
) -> np.ndarray:
    """Exact constrained Neyman allocation min sum_h N_h^2 (1 - n_h/N_h) s_h^2 / n_h."""
    h_count = int(pop_counts.size)
    total = int(np.sum(pop_counts))
    if n_pilot >= total:
        return pop_counts.astype(int).copy()

    lo = np.minimum(pop_counts, min_per_stratum).astype(int)
    if int(np.sum(lo)) >= n_pilot:
        lo = np.minimum(pop_counts, 1).astype(int)

    weights = pop_counts.astype(float) * np.maximum(1e-8, dispersions.astype(float))
    fixed = np.zeros(h_count, dtype=bool)
    alloc_f = np.zeros(h_count, dtype=float)

    for _ in range(h_count * 2 + 4):
        free = ~fixed
        if not np.any(free):
            break
        rem = float(n_pilot) - float(np.sum(alloc_f[fixed]))
        wsum = float(np.sum(weights[free]))
        if wsum <= 0.0 or rem <= 0.0:
            alloc_f[free] = lo[free].astype(float)
            break
        prop = rem * (weights[free] / wsum)
        over = prop > pop_counts[free]
        under = prop < lo[free]
        if not np.any(over) and not np.any(under):
            alloc_f[free] = prop
            break
        fidx = np.flatnonzero(free)
        if np.any(over):
            alloc_f[fidx[over]] = pop_counts[fidx[over]].astype(float)
            fixed[fidx[over]] = True
        if np.any(under):
            alloc_f[fidx[under]] = lo[fidx[under]].astype(float)
            fixed[fidx[under]] = True

    alloc = np.clip(np.floor(alloc_f).astype(int), lo, pop_counts)
    diff = int(n_pilot - int(np.sum(alloc)))
    if diff > 0:
        frac = alloc_f - np.floor(alloc_f)
        for idx in np.argsort(-frac):
            if diff <= 0:
                break
            if alloc[idx] < pop_counts[idx]:
                alloc[idx] += 1
                diff -= 1
    elif diff < 0:
        for idx in np.argsort(weights / np.maximum(1, alloc)):
            if diff >= 0:
                break
            while alloc[idx] > lo[idx] and diff < 0:
                alloc[idx] -= 1
                diff += 1
    return alloc


@dataclass(frozen=True, slots=True)
class PilotSample:
    indices: np.ndarray
    strata: np.ndarray
    ht_weights: np.ndarray
    pop_counts: dict[int, int]
    sample_counts: dict[int, int]
    n_total: int
    high_signal_fp_share_of_controls: float


def sample_pilot(
    y: np.ndarray,
    p_base: np.ndarray,
    *,
    pilot_fraction: float = 0.15,
    mode: str = "collision_neyman",
    tau_star: float = 0.10,
    kappa: float = 24.0,
    n_bins: int = 8,
    seed: int = 42,
) -> PilotSample:
    """Sample Phase-2 pilot under 'collision_neyman', 'global_neyman', or 'srs'."""
    yv = np.asarray(y, dtype=int).ravel()
    pv = np.clip(np.asarray(p_base, dtype=float).ravel(), _EPS, 1.0 - _EPS)
    n_total = int(yv.size)
    rng = np.random.default_rng(seed)

    if pilot_fraction >= 0.999 or mode == "full":
        hs_fp = float(np.mean(pv[yv == 0] >= 0.5 * tau_star)) if np.any(yv == 0) else 0.0
        return PilotSample(
            indices=np.arange(n_total, dtype=int),
            strata=np.zeros(n_total, dtype=int),
            ht_weights=np.ones(n_total, dtype=float),
            pop_counts={0: n_total},
            sample_counts={0: n_total},
            n_total=n_total,
            high_signal_fp_share_of_controls=hs_fp,
        )

    n_pilot = int(np.clip(round(pilot_fraction * n_total), min(40, n_total), n_total))
    if mode == "srs":
        idx = np.sort(rng.choice(n_total, size=n_pilot, replace=False))
        w = np.full(n_pilot, n_total / n_pilot, dtype=float)
        ctrl = idx[yv[idx] == 0]
        hs_fp = float(np.mean(pv[ctrl] >= 0.5 * tau_star)) if ctrl.size > 0 else 0.0
        return PilotSample(
            indices=idx,
            strata=np.zeros(n_pilot, dtype=int),
            ht_weights=w,
            pop_counts={0: n_total},
            sample_counts={0: n_pilot},
            n_total=n_total,
            high_signal_fp_share_of_controls=hs_fp,
        )

    quantiles = np.quantile(logit(pv), np.linspace(0.0, 1.0, n_bins + 1))
    quantiles[0] = -np.inf
    quantiles[-1] = np.inf
    bins = np.clip(np.digitize(logit(pv), quantiles[1:-1], right=False), 0, n_bins - 1)
    raw_stratum = yv * n_bins + bins
    uniq, inv = np.unique(raw_stratum, return_inverse=True)

    pop_counts = np.bincount(inv, minlength=uniq.size)
    dispersions = np.zeros(uniq.size, dtype=float)
    for h_i in range(uniq.size):
        mask = inv == h_i
        yh = float(np.mean(yv[mask]))
        ph = float(np.mean(pv[mask]))
        if mode == "global_neyman":
            dispersions[h_i] = abs(yh - ph) + 0.25 * ph * (1.0 - ph)
        elif mode == "collision_neyman":
            w_cz = float(schervish_collision_weight(np.array([ph]), tau_star, kappa)[0])
            dispersions[h_i] = w_cz * ph * (1.0 - ph) * (abs(yh - ph) + 0.15) + 1e-3
        else:
            raise ValueError(f"Unknown pilot mode: {mode}")

    alloc = waterfill_neyman(pop_counts, dispersions, n_pilot)
    chosen_parts = []
    strata_parts = []
    w_parts = []
    for h_i in range(uniq.size):
        pool = np.flatnonzero(inv == h_i)
        nh = int(alloc[h_i])
        if nh <= 0:
            continue
        sel = np.sort(rng.choice(pool, size=nh, replace=False))
        chosen_parts.append(sel)
        strata_parts.append(np.full(nh, h_i, dtype=int))
        w_parts.append(np.full(nh, float(pop_counts[h_i]) / float(nh), dtype=float))

    idx = np.concatenate(chosen_parts)
    st = np.concatenate(strata_parts)
    wt = np.concatenate(w_parts)
    order = np.argsort(idx)
    idx, st, wt = idx[order], st[order], wt[order]

    ctrl = idx[yv[idx] == 0]
    hs_fp = float(np.mean(pv[ctrl] >= 0.5 * tau_star)) if ctrl.size > 0 else 0.0
    return PilotSample(
        indices=idx,
        strata=st,
        ht_weights=wt,
        pop_counts={int(i): int(pop_counts[i]) for i in range(uniq.size)},
        sample_counts={int(i): int(alloc[i]) for i in range(uniq.size)},
        n_total=n_total,
        high_signal_fp_share_of_controls=hs_fp,
    )


def two_phase_ht_mean_and_se(
    psi: np.ndarray,
    pilot: PilotSample,
) -> tuple[float, float]:
    """Exact two-phase Horvitz-Thompson estimate and sqrt(V_I + V_II) standard error."""
    n_tot = float(pilot.n_total)
    w = pilot.ht_weights
    n_p = int(psi.size)
    g_hat = float(np.sum(w * psi) / n_tot)
    resid = psi - g_hat
    v1 = float((n_p / max(1, n_p - 1)) * np.sum(w * (resid**2)) / (n_tot**2))
    v2 = 0.0
    for h_i, n_h_pop in pilot.pop_counts.items():
        n_h = pilot.sample_counts.get(h_i, 0)
        if n_h < 2 or n_h >= n_h_pop:
            continue
        mask = pilot.strata == h_i
        s_h2 = float(np.var(psi[mask], ddof=1))
        f_h = float(n_h) / float(n_h_pop)
        v2 += (float(n_h_pop) ** 2) * (1.0 - f_h) * s_h2 / float(n_h)
    v2 /= n_tot**2
    se = float(np.sqrt(max(1e-16, v1 + v2)))
    return g_hat, se


def holm_adjust(pvals: list[float]) -> list[float]:
    m = len(pvals)
    if m == 0:
        return []
    order = np.argsort(pvals)
    adj = np.zeros(m, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, float(pvals[idx]) * (m - rank))
        running_max = max(running_max, val)
        adj[idx] = running_max
    return adj.tolist()


# ---------------------------------------------------------------------------
# Fast Leak-Free Matrix Preprocessor + Offset/Collision-Specialist LightGBM
# ---------------------------------------------------------------------------


class FastFrameEncoder:
    """Leak-free numeric/categorical matrix encoder fit strictly on training rows."""

    def __init__(self, *, include_missing_indicators: bool = True) -> None:
        self.include_missing_indicators = bool(include_missing_indicators)
        self.num_cols_: list[str] = []
        self.cat_cols_: list[str] = []
        self.medians_: dict[str, float] = {}
        self.cat_maps_: dict[str, dict[str, int]] = {}

    def fit(self, df: pd.DataFrame) -> FastFrameEncoder:
        self.num_cols_ = []
        self.cat_cols_ = []
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]):
                self.num_cols_.append(c)
                arr = df[c].to_numpy(dtype=float)
                med = 0.0 if np.all(~np.isfinite(arr)) else float(np.nanmedian(arr))
                self.medians_[c] = med
            else:
                self.cat_cols_.append(c)
                vals = df[c].astype(str).fillna("Missing")
                vc = vals.value_counts()
                keep = vc[vc >= 10].index.tolist()[:64]
                self.cat_maps_[c] = {k: i + 1 for i, k in enumerate(keep)}
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        blocks: list[np.ndarray] = []
        n = len(df)
        if self.num_cols_:
            stride = 2 if self.include_missing_indicators else 1
            num_mat = np.empty((n, len(self.num_cols_) * stride), dtype=np.float32)
            for j, c in enumerate(self.num_cols_):
                arr = df[c].to_numpy(dtype=np.float32, copy=True)
                miss = ~np.isfinite(arr)
                arr[miss] = self.medians_[c]
                if self.include_missing_indicators:
                    num_mat[:, 2 * j] = arr
                    num_mat[:, 2 * j + 1] = miss.astype(np.float32)
                else:
                    num_mat[:, j] = arr
            blocks.append(num_mat)
        if self.cat_cols_:
            cat_mat = np.empty((n, len(self.cat_cols_)), dtype=np.float32)
            for j, c in enumerate(self.cat_cols_):
                mp = self.cat_maps_[c]
                vals = df[c].astype(str).fillna("Missing")
                cat_mat[:, j] = vals.map(mp).fillna(0).to_numpy(dtype=np.float32)
            blocks.append(cat_mat)
        if not blocks:
            return np.zeros((n, 1), dtype=np.float32)
        return np.hstack(blocks)


class AgnielDeconfounder:
    """Eliminates observational clinician-ordering ('informative presence') confounding.

    For each continuous Stage-2 assay column Z_c, fits E[Z_c | X_base, D_c=1] + empirical residual
    distribution strictly on measured training rows (D_c = 1). Unmeasured entries (D_c = 0) are
    imputed via mu_hat(X_base) + eps_draw, enforcing Z_tilde _|_ D_c | X_base so no downstream tree
    can exploit whether the clinician ordered the assay!
    """

    def __init__(self, alpha: float = 10.0) -> None:
        self.alpha = float(alpha)
        self.enc_base_: FastFrameEncoder | None = None
        self.col_means_: np.ndarray | None = None
        self.col_stds_: np.ndarray | None = None
        self.models_: dict[str, Ridge | float] = {}
        self.residuals_: dict[str, np.ndarray] = {}

    def fit(self, df_base: pd.DataFrame, df_s2: pd.DataFrame) -> AgnielDeconfounder:
        self.enc_base_ = FastFrameEncoder(include_missing_indicators=True).fit(df_base)
        xb = self.enc_base_.transform(df_base).astype(np.float64)
        self.col_means_ = np.mean(xb, axis=0)
        self.col_stds_ = np.maximum(np.std(xb, axis=0), 1e-6)
        xb_std = (xb - self.col_means_) / self.col_stds_

        self.models_ = {}
        self.residuals_ = {}
        for col in df_s2.columns:
            if not pd.api.types.is_numeric_dtype(df_s2[col]):
                continue
            vals = df_s2[col].to_numpy(dtype=np.float64)
            obs = np.isfinite(vals)
            n_obs = int(np.sum(obs))
            if n_obs < 25:
                fallback = float(np.nanmedian(vals)) if n_obs > 0 else 0.0
                self.models_[col] = fallback
                self.residuals_[col] = np.zeros(1, dtype=np.float64)
                continue
            reg = Ridge(alpha=self.alpha)
            reg.fit(xb_std[obs], vals[obs])
            preds = reg.predict(xb_std[obs])
            res = vals[obs] - preds
            self.models_[col] = reg
            self.residuals_[col] = res[:4096].copy()
        return self

    def transform(self, df_base: pd.DataFrame, df_s2: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
        assert self.enc_base_ is not None and self.col_means_ is not None and self.col_stds_ is not None
        xb = self.enc_base_.transform(df_base).astype(np.float64)
        xb_std = (xb - self.col_means_) / self.col_stds_
        out = df_s2.copy()
        rng = np.random.default_rng(seed)
        for col in out.columns:
            if col not in self.models_:
                continue
            vals = out[col].to_numpy(dtype=np.float64, copy=True)
            miss = ~np.isfinite(vals)
            n_miss = int(np.sum(miss))
            if n_miss == 0:
                continue
            mdl = self.models_[col]
            res_pool = self.residuals_[col]
            noise = rng.choice(res_pool, size=n_miss, replace=True)
            if isinstance(mdl, Ridge):
                vals[miss] = mdl.predict(xb_std[miss]) + noise
            else:
                vals[miss] = float(mdl) + noise
            out[col] = vals.astype(np.float32)
        return out


def fit_platt_calibrator(raw_logits: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    lr = LogisticRegression(C=1e4, solver="lbfgs", max_iter=500)
    lr.fit(raw_logits.reshape(-1, 1), y.astype(int))
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def apply_platt(raw_logits: np.ndarray, slope_intercept: tuple[float, float]) -> np.ndarray:
    a, b = slope_intercept
    return expit(np.clip(a * raw_logits + b, -25.0, 25.0))


def compute_oof_stage1(
    df_base: pd.DataFrame,
    y: np.ndarray,
    n_folds: int = 4,
    seed: int = 42,
) -> np.ndarray:
    """Compute leak-free out-of-fold calibrated baseline probabilities p_base(X_base)."""
    yv = np.asarray(y, dtype=int)
    n = yv.size
    rng = np.random.default_rng(seed)
    pos = rng.permutation(np.flatnonzero(yv == 1))
    neg = rng.permutation(np.flatnonzero(yv == 0))
    folds = [np.concatenate([pos[k::n_folds], neg[k::n_folds]]) for k in range(n_folds)]

    oof_p = np.zeros(n, dtype=float)
    for k in range(n_folds):
        val_idx = folds[k]
        tr_all = np.concatenate([folds[j] for j in range(n_folds) if j != k])
        # Disjoint stratified 85/15 split inside tr_all so Platt calibration is strictly out-of-sample
        tr_pos = tr_all[yv[tr_all] == 1]
        tr_neg = tr_all[yv[tr_all] == 0]
        np_fit = max(1, round(0.85 * tr_pos.size))
        nn_fit = max(1, round(0.85 * tr_neg.size))
        fit_sub = np.concatenate([tr_pos[:np_fit], tr_neg[:nn_fit]])
        cal_sub = np.concatenate([tr_pos[np_fit:], tr_neg[nn_fit:]])
        if cal_sub.size < 20 or np.unique(yv[cal_sub]).size < 2:
            fit_sub = tr_all
            cal_sub = tr_all

        enc = FastFrameEncoder().fit(df_base.iloc[fit_sub])
        x_tr = enc.transform(df_base.iloc[fit_sub])
        x_ca = enc.transform(df_base.iloc[cal_sub])
        x_va = enc.transform(df_base.iloc[val_idx])
        clf = lgb.LGBMClassifier(
            n_estimators=120,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=30,
            max_delta_step=1.0,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=seed + k,
            n_jobs=4,
            verbose=-1,
        )
        clf.fit(x_tr, yv[fit_sub])
        raw_ca = clf.predict(x_ca, raw_score=True)
        raw_va = clf.predict(x_va, raw_score=True)
        platt = fit_platt_calibrator(raw_ca, yv[cal_sub])
        oof_p[val_idx] = apply_platt(raw_va, platt)
    return np.clip(oof_p, 1e-5, 1.0 - 1e-5)


# ---------------------------------------------------------------------------
# Sequential Discovery: Global CMI (Paper 1) vs. Collision-Zone (Paper 2)
# ---------------------------------------------------------------------------


def run_sequential_discovery(
    df_train: pd.DataFrame,
    y_train: np.ndarray,
    base_cols: list[str],
    channels: list[ChannelDef],
    oof_p_base: np.ndarray,
    contract: ClinicalContract,
    *,
    method: str = "collision_zone",
    pilot_fraction: float = 0.15,
    budget: float = 4.0,
    kappa: float = 24.0,
    alpha: float = 0.05,
    use_holm_lcb: bool = True,
    deconfound_stage2: bool = False,
    adaptive_kappa: bool = True,
    seed: int = 42,
) -> DiscoveryOutcome:
    """Compare Global CMI ('global_cmi') vs. Collision-Zone ('collision_zone') discovery."""
    yv = np.asarray(y_train, dtype=int)
    tau = contract.tau_star
    pilot_mode = (
        "collision_neyman"
        if method == "collision_zone"
        else ("srs" if method == "srs_global_cmi" else "global_neyman")
    )
    pilot = sample_pilot(
        yv,
        oof_p_base,
        pilot_fraction=pilot_fraction,
        mode=pilot_mode,
        tau_star=tau,
        kappa=kappa,
        seed=seed,
    )

    p_idx = pilot.indices
    yp = yv[p_idx]
    p0_p = np.clip(oof_p_base[p_idx], 1e-5, 1.0 - 1e-5)
    init_logit_p = logit(p0_p)
    w_ht = pilot.ht_weights

    kappa_eff = (
        solve_ess_kappa(
            bernoulli_kl_from_tau(p0_p, tau),
            kappa_max=kappa,
            min_effective_cz_samples=max(160.0, 0.22 * float(yp.size)),
        )
        if (method == "collision_zone" and adaptive_kappa)
        else float(kappa)
    )

    if method == "collision_zone":
        fit_weights = w_ht * collision_zone_relevance_weights(
            p0_p, tau, kappa=kappa_eff, floor=0.05
        )
    else:
        fit_weights = w_ht.copy()

    # 4-fold stratified pilot splits + pre-encoded channel matrices
    rng = np.random.default_rng(seed + 17)
    pos = rng.permutation(np.flatnonzero(yp == 1))
    neg = rng.permutation(np.flatnonzero(yp == 0))
    n_folds = 4
    folds = [np.concatenate([pos[k::n_folds], neg[k::n_folds]]) for k in range(n_folds)]

    df_p = df_train.iloc[p_idx].reset_index(drop=True)
    all_ch_cols = list(dict.fromkeys(c for ch in channels for c in ch.columns))
    fold_base: list[tuple[np.ndarray, np.ndarray]] = []
    fold_ch: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {ch.channel_id: [] for ch in channels}

    for k in range(n_folds):
        va = folds[k]
        tr = np.concatenate([folds[j] for j in range(n_folds) if j != k])
        enc_b = FastFrameEncoder().fit(df_p[base_cols].iloc[tr])
        fold_base.append(
            (enc_b.transform(df_p[base_cols].iloc[tr]), enc_b.transform(df_p[base_cols].iloc[va]))
        )
        if deconfound_stage2 and all_ch_cols:
            deconf = AgnielDeconfounder().fit(df_p[base_cols].iloc[tr], df_p[all_ch_cols].iloc[tr])
            s2_tr_df = deconf.transform(
                df_p[base_cols].iloc[tr], df_p[all_ch_cols].iloc[tr], seed=seed + 50 + k
            )
            s2_va_df = deconf.transform(
                df_p[base_cols].iloc[va], df_p[all_ch_cols].iloc[va], seed=seed + 90 + k
            )
        else:
            s2_tr_df = df_p[all_ch_cols].iloc[tr]
            s2_va_df = df_p[all_ch_cols].iloc[va]

        for ch in channels:
            cols_c = list(ch.columns)
            enc_c = FastFrameEncoder(include_missing_indicators=not deconfound_stage2).fit(
                s2_tr_df[cols_c]
            )
            fold_ch[ch.channel_id].append(
                (enc_c.transform(s2_tr_df[cols_c]), enc_c.transform(s2_va_df[cols_c]))
            )

    def _oof_probs_for_set(sel_ids: list[str]) -> np.ndarray:
        oof = np.zeros(yp.size, dtype=float)
        for k in range(n_folds):
            va = folds[k]
            tr = np.concatenate([folds[j] for j in range(n_folds) if j != k])
            tr_mats = [fold_base[k][0]] + [fold_ch[cid][k][0] for cid in sel_ids]
            va_mats = [fold_base[k][1]] + [fold_ch[cid][k][1] for cid in sel_ids]
            x_tr = np.hstack(tr_mats)
            x_va = np.hstack(va_mats)
            clf = lgb.LGBMClassifier(
                n_estimators=70,
                learning_rate=0.05,
                num_leaves=15,
                min_child_samples=25,
                max_delta_step=1.0,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=seed + 100 + k,
                n_jobs=4,
                verbose=-1,
            )
            clf.fit(
                x_tr,
                yp[tr],
                sample_weight=fit_weights[tr],
                init_score=init_logit_p[tr],
            )
            raw_va = init_logit_p[va] + clf.predict(x_va, raw_score=True)
            oof[va] = expit(np.clip(raw_va, -20.0, 20.0))
        return np.clip(oof, 1e-6, 1.0 - 1e-6)

    def _score_vec(p_vec: np.ndarray) -> np.ndarray:
        if method == "collision_zone":
            return schervish_collision_proper_score(yp, p_vec, tau, kappa=kappa_eff)
        return log_likelihood_score(yp, p_vec)

    selected: list[str] = []
    spent = 0.0
    steps: list[DiscoveryStepLog] = []
    step1_scores: dict[str, float] = {}

    cur_p = _oof_probs_for_set([])
    cur_s = _score_vec(cur_p)

    step_num = 0
    while True:
        step_num += 1
        affordable = [
            ch for ch in channels if ch.channel_id not in selected and spent + ch.cost <= budget + 1e-9
        ]
        if not affordable:
            break

        cand_gains: list[float] = []
        cand_ses: list[float] = []
        cand_pvals: list[float] = []
        cand_probs: list[np.ndarray] = []

        for ch in affordable:
            p_aug = _oof_probs_for_set(selected + [ch.channel_id])
            psi = _score_vec(p_aug) - cur_s
            g_hat, se_tot = two_phase_ht_mean_and_se(psi, pilot)
            z = g_hat / max(1e-12, se_tot)
            pval = float(1.0 - norm.cdf(z))
            cand_gains.append(g_hat)
            cand_ses.append(se_tot)
            cand_pvals.append(pval)
            cand_probs.append(p_aug)
            if step_num == 1:
                step1_scores[ch.channel_id] = g_hat

        p_holm = holm_adjust(cand_pvals)
        best_idx = -1
        best_ratio = -np.inf
        for idx_c, ch in enumerate(affordable):
            passes = (p_holm[idx_c] < alpha) if use_holm_lcb else (cand_gains[idx_c] > 0.0)
            if passes and cand_gains[idx_c] > 0.0:
                ratio = cand_gains[idx_c] / ch.cost
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_idx = idx_c

        if best_idx < 0:
            break

        winner = affordable[best_idx]
        selected.append(winner.channel_id)
        spent += winner.cost
        cur_p = cand_probs[best_idx]
        cur_s = _score_vec(cur_p)
        steps.append(
            DiscoveryStepLog(
                step=step_num,
                selected_id=winner.channel_id,
                gain=cand_gains[best_idx],
                se_total=cand_ses[best_idx],
                z_stat=cand_gains[best_idx] / max(1e-12, cand_ses[best_idx]),
                p_holm=p_holm[best_idx],
                cost=winner.cost,
            )
        )

    return DiscoveryOutcome(
        method=method,
        pilot_fraction=pilot_fraction,
        selected_ids=selected,
        total_cost=spent,
        pilot_high_signal_fp_share=pilot.high_signal_fp_share_of_controls,
        steps=steps,
        step1_scores=step1_scores,
    )


# ---------------------------------------------------------------------------
# Predicate 3: Global vs. Collision-Specialist Stage-2 Fit & Martingale VOI
# ---------------------------------------------------------------------------


_GH_X, _GH_W_RAW = np.polynomial.hermite.hermgauss(9)
_GH_NODES = _GH_X * np.sqrt(2.0)
_GH_WEIGHTS = _GH_W_RAW / np.sqrt(np.pi)


def solve_martingale_offset(p1: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Solve sum_r w_r expit(logit(p1) + mu* + sigma * xi_r) = p1 (Proposition 2)."""
    pv = np.clip(np.asarray(p1, dtype=float), 1e-6, 1.0 - 1e-6)
    sv = np.maximum(0.0, np.asarray(sigma, dtype=float))
    u1 = logit(pv)
    mu = -0.5 * (sv**2) * (1.0 - 2.0 * pv) * pv * (1.0 - pv)
    mu = np.where(sv <= 1e-12, 0.0, mu)
    for _ in range(18):
        z = np.clip(u1[:, None] + mu[:, None] + sv[:, None] * _GH_NODES[None, :], -30.0, 30.0)
        q = expit(z)
        err = q @ _GH_WEIGHTS - pv
        if float(np.max(np.abs(err))) < 1e-13:
            break
        deriv = (q * (1.0 - q)) @ _GH_WEIGHTS
        mu -= np.clip(err / np.maximum(1e-12, deriv), -2.0, 2.0)
    return np.where(sv <= 1e-12, 0.0, mu)


def mean_centered_voi(
    p1: np.ndarray,
    sigma: np.ndarray,
    contract: ClinicalContract,
) -> np.ndarray:
    pv = np.clip(np.asarray(p1, dtype=float), 1e-6, 1.0 - 1e-6)
    sv = np.maximum(0.0, np.asarray(sigma, dtype=float))
    mu = solve_martingale_offset(pv, sv)
    u1 = logit(pv)
    p2_nodes = expit(np.clip(u1[:, None] + mu[:, None] + sv[:, None] * _GH_NODES[None, :], -30.0, 30.0))
    loss_p1 = np.minimum(pv * contract.cost_fn, (1.0 - pv) * contract.cost_fp)
    loss_nodes = np.minimum(p2_nodes * contract.cost_fn, (1.0 - p2_nodes) * contract.cost_fp)
    return np.maximum(0.0, loss_p1 - (loss_nodes @ _GH_WEIGHTS))


def empirical_martingale_voi(
    p1: np.ndarray,
    bin_indices: np.ndarray,
    bin_delta_quantiles: np.ndarray,
    contract: ClinicalContract,
) -> np.ndarray:
    """Distribution-free empirical conditional VOI preserving skewed/bimodal Stage-2 updates.

    For each instance i in baseline bin b = bin_indices[i], takes K empirical quantiles
    delta_{b, 1..K} observed on D_cal, solves for the unique martingale shift mu*_i such that
    (1/K) sum_r expit(logit(p1_i) + mu*_i + delta_{b, r}) == p1_i, and evaluates exact Bayes gain.
    """
    pv = np.clip(np.asarray(p1, dtype=float), 1e-6, 1.0 - 1e-6)
    b_idx = np.asarray(bin_indices, dtype=int)
    n_tot = pv.size
    out = np.zeros(n_tot, dtype=float)
    k_q = int(bin_delta_quantiles.shape[1])
    w_unif = 1.0 / float(k_q)

    # Process in 40,000-row chunks so peak RAM stays < 15 MB even on 560,000-row cohorts
    chunk = 40000
    for start in range(0, n_tot, chunk):
        end = min(n_tot, start + chunk)
        p_c = pv[start:end]
        u1 = logit(p_c)
        d_mat = bin_delta_quantiles[b_idx[start:end], :]  # (m, K)
        spread = np.std(d_mat, axis=1)
        mu = np.zeros(end - start, dtype=float)
        for _ in range(16):
            z = np.clip(u1[:, None] + mu[:, None] + d_mat, -30.0, 30.0)
            q = expit(z)
            err = np.mean(q, axis=1) - p_c
            if float(np.max(np.abs(err))) < 1e-12:
                break
            deriv = np.mean(q * (1.0 - q), axis=1)
            mu -= np.clip(err / np.maximum(1e-12, deriv), -2.5, 2.5)
        mu = np.where(spread <= 1e-12, 0.0, mu)
        p2_nodes = expit(np.clip(u1[:, None] + mu[:, None] + d_mat, -30.0, 30.0))
        loss_p1 = np.minimum(p_c * contract.cost_fn, (1.0 - p_c) * contract.cost_fp)
        loss_nodes = np.minimum(p2_nodes * contract.cost_fn, (1.0 - p2_nodes) * contract.cost_fp)
        out[start:end] = np.maximum(0.0, loss_p1 - np.sum(loss_nodes, axis=1) * w_unif)
    return out


def net_benefit_per_1k(
    y: np.ndarray,
    p_routed: np.ndarray,
    queried_mask: np.ndarray,
    contract: ClinicalContract,
) -> float:
    yv = np.asarray(y, dtype=int)
    act = np.asarray(p_routed, dtype=float) >= contract.tau_star
    tp = float(np.sum((act == 1) & (yv == 1)))
    fp = float(np.sum((act == 1) & (yv == 0)))
    n_q = float(np.sum(np.asarray(queried_mask, dtype=float)))
    n = float(yv.size)
    return (
        1000.0
        * (
            tp
            - (contract.cost_fp / contract.cost_fn) * fp
            - (contract.per_instance_query_cost / contract.cost_fn) * n_q
        )
        / n
    )


def paired_bootstrap_nb_diff(
    y: np.ndarray,
    p_arm: np.ndarray,
    q_arm: np.ndarray,
    p_ref: np.ndarray,
    q_ref: np.ndarray,
    contract: ClinicalContract,
    *,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """Exact patient-level paired bootstrap CI & two-sided p-value for NB(arm) - NB(ref)."""
    yv = np.asarray(y, dtype=int)
    tau = contract.tau_star
    w_fp = contract.cost_fp / contract.cost_fn
    w_q = contract.per_instance_query_cost / contract.cost_fn

    def _unit_nb(pv: np.ndarray, qv: np.ndarray) -> np.ndarray:
        act = pv >= tau
        return 1000.0 * (
            ((act == 1) & (yv == 1)).astype(float)
            - w_fp * ((act == 1) & (yv == 0)).astype(float)
            - w_q * qv.astype(float)
        )

    d_i = _unit_nb(np.asarray(p_arm, dtype=float), np.asarray(q_arm, dtype=float)) - _unit_nb(
        np.asarray(p_ref, dtype=float), np.asarray(q_ref, dtype=float)
    )
    n = d_i.size
    obs_diff = float(np.mean(d_i))
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[b] = float(np.mean(d_i[idx]))
    se = float(np.std(boot_means, ddof=1))
    ci_low, ci_high = np.quantile(boot_means, [0.025, 0.975])
    # Studentized / percentile two-sided p-value
    p_pos = float(np.mean(boot_means <= 0.0))
    p_neg = float(np.mean(boot_means >= 0.0))
    p_two = min(1.0, 2.0 * min(p_pos, p_neg))
    return {
        "diff_mean": obs_diff,
        "boot_se": se,
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "p_two_sided": p_two,
    }


def compute_decision_curve(
    y: np.ndarray,
    p_routed: np.ndarray,
    queried_mask: np.ndarray,
    taus: list[float] | np.ndarray,
    per_instance_query_cost: float = 0.0,
) -> dict[str, list[float]]:
    """Vickers & Elkin (2006) Net Benefit curve across a clinical threshold grid tau in (0, 1)."""
    yv = np.asarray(y, dtype=int)
    pv = np.asarray(p_routed, dtype=float)
    qv = np.asarray(queried_mask, dtype=float)
    n = float(yv.size)
    n_q = float(np.sum(qv))
    t_list = [float(t) for t in taus]
    nb_model: list[float] = []
    nb_treat_all: list[float] = []
    pos_tot = float(np.sum(yv == 1))
    neg_tot = float(np.sum(yv == 0))
    for tau in t_list:
        odds = tau / (1.0 - tau)
        # Under cost_fp = 1, cost_fn = (1 - tau)/tau, so c_query / cost_fn = c_query * odds
        act = pv >= tau
        tp = float(np.sum((act == 1) & (yv == 1)))
        fp = float(np.sum((act == 1) & (yv == 0)))
        nb_m = 1000.0 * (tp - odds * fp - (per_instance_query_cost * odds) * n_q) / n
        nb_all = 1000.0 * (pos_tot - odds * neg_tot) / n
        nb_model.append(nb_m)
        nb_treat_all.append(nb_all)
    return {"taus": t_list, "nb_model": nb_model, "nb_treat_all": nb_treat_all}


def cfbh_selection(
    y_cal: np.ndarray,
    s_cal: np.ndarray,
    y_test: np.ndarray,
    s_test: np.ndarray,
    fdr_target: float,
    seed: int = 42,
) -> dict[str, float]:
    """Jin & Candes (2023) smoothed conformal BH selection."""
    yc = np.asarray(y_cal, dtype=int)
    sc = np.asarray(s_cal, dtype=float)
    yt = np.asarray(y_test, dtype=int)
    st = np.asarray(s_test, dtype=float)
    n_cal = sc.size
    neg_sc = np.sort(sc[yc == 0])
    # Strictly greater count + equal count via searchsorted
    right_idx = np.searchsorted(neg_sc, st, side="right")
    left_idx = np.searchsorted(neg_sc, st, side="left")
    gt_cnt = neg_sc.size - right_idx
    eq_cnt = right_idx - left_idx
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 1.0, size=st.size)
    p_jc = (1.0 + gt_cnt + u * eq_cnt) / float(1 + n_cal)

    order = np.argsort(p_jc)
    p_sorted = p_jc[order]
    m = st.size
    thresh = fdr_target * (np.arange(1, m + 1) / float(m))
    passing = np.flatnonzero(p_sorted <= thresh)
    sel = np.zeros(m, dtype=bool)
    if passing.size > 0:
        sel[order[: int(passing[-1]) + 1]] = True

    n_sel = int(np.sum(sel))
    tp = int(np.sum(sel & (yt == 1)))
    fp = int(np.sum(sel & (yt == 0)))
    fdp = float(fp / n_sel) if n_sel > 0 else 0.0
    rec = float(tp / max(1, int(np.sum(yt == 1))))
    return {"fdp": fdp, "recall": rec, "n_selected": float(n_sel)}



# ---------------------------------------------------------------------------
# Fluid KL-Gated Residual Backbone (Eliminating GBDT Leaf Starvation)
# ---------------------------------------------------------------------------


def bernoulli_kl_from_tau(p: np.ndarray, tau_star: float) -> np.ndarray:
    """Exact Bernoulli KL divergence D_KL(Bern(tau*) || Bern(p)) >= 0."""
    pv = np.clip(np.asarray(p, dtype=float), _EPS, 1.0 - _EPS)
    t = float(np.clip(tau_star, 1e-6, 1.0 - 1e-6))
    return t * np.log(t / pv) + (1.0 - t) * np.log((1.0 - t) / (1.0 - pv))


def solve_ess_kappa(
    d_kl: np.ndarray,
    kappa_max: float = 18.0,
    min_effective_cz_samples: float = 240.0,
) -> float:
    """Solve via monotone bisection for sharpest kappa* keeping active Hessian mass sum(pi^2) >= floor."""
    d_arr = np.asarray(d_kl, dtype=float)
    n = float(d_arr.size)
    if n <= 10.0 or kappa_max <= 1e-9:
        return 0.0
    floor = min(0.85 * n, float(min_effective_cz_samples))

    def _active_mass(k: float) -> float:
        return float(np.sum(np.exp(-2.0 * k * d_arr)))

    if _active_mass(kappa_max) >= floor:
        return float(kappa_max)
    if _active_mass(0.0) <= floor:
        return 0.0
    return float(brentq(lambda k: _active_mass(k) - floor, 0.0, kappa_max, xtol=1e-3))


def union_collision_gate(
    f_base: np.ndarray,
    f_glob: np.ndarray,
    tau_star: float,
    kappa: float,
) -> np.ndarray:
    """Smooth probabilistic-OR gate: 1 - (1 - pi(p_base)) * (1 - pi(p_glob))."""
    if kappa <= 1e-9:
        return np.ones_like(f_base, dtype=float)
    pi_b = np.exp(-kappa * bernoulli_kl_from_tau(expit(np.clip(f_base, -25.0, 25.0)), tau_star))
    pi_g = np.exp(-kappa * bernoulli_kl_from_tau(expit(np.clip(f_glob, -25.0, 25.0)), tau_star))
    return 1.0 - (1.0 - pi_b) * (1.0 - pi_g)


def _fit_gated_residual_booster(
    z_aug: np.ndarray,
    y_tr: np.ndarray,
    f_glob_tr: np.ndarray,
    pi_tr: np.ndarray,
    *,
    n_estimators: int = 50,
    learning_rate: float = 0.05,
    num_leaves: int = 8,
    reg_lambda_eff: float = 3.5,
    seed: int = 42,
) -> lgb.Booster:
    """Exact unweighted Bernoulli MLE on f = f_glob + pi * Delta_CZ(ell_1, f_glob, Z_S)."""
    dtrain = lgb.Dataset(z_aug, label=y_tr, free_raw_data=False)
    p0 = expit(np.clip(f_glob_tr, -25.0, 25.0))
    h0 = np.clip(p0 * (1.0 - p0), 1e-9, 0.25)
    fisher_ratio = max(0.04, float(np.sum((pi_tr**2) * h0) / max(float(np.sum(h0)), 1e-9)))
    params = {
        "learning_rate": learning_rate,
        "num_leaves": num_leaves,
        "min_child_samples": 12,
        "min_sum_hessian_in_leaf": max(0.15, 1.5 * fisher_ratio),
        "reg_lambda": reg_lambda_eff * fisher_ratio,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.90,
        "verbose": -1,
        "n_jobs": 4,
        "seed": seed,
    }
    f_curr = np.asarray(f_glob_tr, dtype=float).copy()
    booster = lgb.Booster(params, dtrain)
    for t in range(n_estimators):
        p = expit(np.clip(f_curr, -25.0, 25.0))
        grad = pi_tr * (p - y_tr)
        hess = (pi_tr**2) * np.clip(p * (1.0 - p), 1e-9, 0.25)

        def _fobj(
            _preds: np.ndarray,
            _data: lgb.Dataset,
            _g: np.ndarray = grad,
            _h: np.ndarray = hess,
        ) -> tuple[np.ndarray, np.ndarray]:
            return _g, _h

        booster.update(fobj=_fobj)
        step_t = booster.predict(z_aug, raw_score=True, start_iteration=t, num_iteration=1)
        f_curr += pi_tr * step_t
    return booster


class FluidCollisionBackboneClassifier:
    """Two-scale fluid Stage-2 estimator avoiding GBDT leaf starvation:
      f(x, z) = alpha_0 + beta_glob * f_glob(x, z)
                + beta_cz * pi_union(ell_1(x), f_glob(x, z)) * Delta_CZ(ell_1(x), f_glob(x, z), z_S)
    Fit strictly on D_train via 3-fold OOF stacking so D_cal remains 100% exchangeable for VOI & CfBH.
    """

    def __init__(
        self,
        tau_star: float,
        *,
        kappa_max: float = 18.0,
        min_cz_ess: float = 240.0,
        seed: int = 42,
    ) -> None:
        self.tau_star = float(tau_star)
        self.kappa_max = float(kappa_max)
        self.min_cz_ess = float(min_cz_ess)
        self.seed = int(seed)
        self.kappa_: float = 0.0
        self.alpha_0_: float = 0.0
        self.beta_glob_: float = 1.0
        self.beta_cz_: float = 0.0
        self.platt_glob_: tuple[float, float] = (1.0, 0.0)
        self.m_glob_: lgb.LGBMClassifier | None = None
        self.m_cz_: lgb.Booster | None = None

    def _make_glob_clf(self, seed_offset: int = 0) -> lgb.LGBMClassifier:
        return lgb.LGBMClassifier(
            n_estimators=140,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=30,
            max_delta_step=1.0,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=self.seed + seed_offset,
            n_jobs=4,
            verbose=-1,
        )

    def fit(
        self,
        x_all: np.ndarray,
        z_s2: np.ndarray,
        y: np.ndarray,
        ell1_base: np.ndarray,
    ) -> FluidCollisionBackboneClassifier:
        yv = np.asarray(y, dtype=int)
        ell1 = np.clip(np.asarray(ell1_base, dtype=float), -20.0, 20.0)
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.seed)

        # Pass 1: Out-of-fold Global Stage-2 Backbone on 100% of training samples
        oof_raw_glob = np.zeros(yv.size, dtype=float)
        splits = list(skf.split(x_all, yv))
        for k, (tr_i, va_i) in enumerate(splits):
            mg = self._make_glob_clf(seed_offset=10 + k)
            mg.fit(x_all[tr_i], yv[tr_i])
            oof_raw_glob[va_i] = mg.predict(x_all[va_i], raw_score=True)

        self.platt_glob_ = fit_platt_calibrator(oof_raw_glob, yv)
        ag, bg = self.platt_glob_
        oof_f_glob = np.clip(ag * oof_raw_glob + bg, -20.0, 20.0)

        # Dynamic Kish-ESS bisection for optimal collision focus kappa* + adaptive leaf capacity
        dkl_base = bernoulli_kl_from_tau(expit(ell1), self.tau_star)
        self.kappa_ = solve_ess_kappa(dkl_base, self.kappa_max, self.min_cz_ess)
        pi_oof = union_collision_gate(ell1, oof_f_glob, self.tau_star, self.kappa_)
        active_mass = float(np.sum(pi_oof**2))
        if active_mass >= 2500.0:
            cz_leaves, cz_trees = 23, 80
        elif active_mass >= 800.0:
            cz_leaves, cz_trees = 15, 60
        else:
            cz_leaves, cz_trees = 8, 50

        # Pass 2: Out-of-fold Bottlenecked Collision Specialist on [ell_1, f_glob, Z_S]
        z_aug = np.column_stack([ell1, oof_f_glob, z_s2]).astype(np.float32)
        oof_d_cz = np.zeros(yv.size, dtype=float)
        for k, (tr_i, va_i) in enumerate(splits):
            mcz = _fit_gated_residual_booster(
                z_aug[tr_i],
                yv[tr_i],
                oof_f_glob[tr_i],
                pi_oof[tr_i],
                n_estimators=cz_trees,
                num_leaves=cz_leaves,
                seed=self.seed + 30 + k,
            )
            oof_d_cz[va_i] = mcz.predict(z_aug[va_i], raw_score=True)

        # Pass 3: Strictly Proper Schervish Collision-Zone OOF Gate (with tiny ridge anchor)
        def _obj(params: np.ndarray) -> float:
            a0, b_g, b_cz = float(params[0]), float(params[1]), float(params[2])
            f = np.clip(a0 + b_g * oof_f_glob + b_cz * pi_oof * oof_d_cz, -25.0, 25.0)
            p = expit(f)
            s_cz = -float(
                np.mean(schervish_collision_proper_score(yv, p, self.tau_star, kappa=20.0))
            )
            ll = -float(np.mean(log_likelihood_score(yv, p)))
            reg = 2e-4 * ((b_g - 1.0) ** 2 + b_cz**2 + a0**2)
            return s_cz + 0.05 * ll + reg

        opt = minimize(
            _obj,
            x0=np.array([0.0, 1.0, 0.25]),
            bounds=[(-1.5, 1.5), (0.5, 1.3), (0.0, 1.5)],
            method="L-BFGS-B",
        )
        self.alpha_0_ = float(opt.x[0])
        self.beta_glob_ = float(opt.x[1])
        self.beta_cz_ = float(opt.x[2])

        # Final refit on 100% of D_train (training m_cz against oof_f_glob avoids residual shrinkage)
        self.m_glob_ = self._make_glob_clf(seed_offset=7)
        self.m_glob_.fit(x_all, yv)
        if self.beta_cz_ > 1e-4:
            self.m_cz_ = _fit_gated_residual_booster(
                z_aug,
                yv,
                oof_f_glob,
                pi_oof,
                n_estimators=cz_trees,
                num_leaves=cz_leaves,
                seed=self.seed + 77,
            )
        else:
            self.m_cz_ = None
        return self

    def decision_function(
        self,
        x_all: np.ndarray,
        z_s2: np.ndarray,
        ell1_base: np.ndarray,
    ) -> np.ndarray:
        assert self.m_glob_ is not None
        ell1 = np.clip(np.asarray(ell1_base, dtype=float), -20.0, 20.0)
        raw_g = self.m_glob_.predict(x_all, raw_score=True)
        ag, bg = self.platt_glob_
        f_glob = np.clip(ag * raw_g + bg, -20.0, 20.0)
        f_out = self.alpha_0_ + self.beta_glob_ * f_glob
        if self.m_cz_ is None or self.beta_cz_ <= 1e-4:
            return f_out
        pi = union_collision_gate(ell1, f_glob, self.tau_star, self.kappa_)
        z_aug = np.column_stack([ell1, f_glob, z_s2]).astype(np.float32)
        d_cz = self.m_cz_.predict(z_aug, raw_score=True)
        return f_out + self.beta_cz_ * pi * d_cz

    def predict_proba(
        self,
        x_all: np.ndarray,
        z_s2: np.ndarray,
        ell1_base: np.ndarray,
    ) -> np.ndarray:
        raw = np.clip(self.decision_function(x_all, z_s2, ell1_base), -25.0, 25.0)
        return np.clip(expit(raw), 1e-6, 1.0 - 1e-6)


def fit_stage1_predictions(
    df_train: pd.DataFrame,
    y_train: np.ndarray,
    df_cal: pd.DataFrame,
    y_cal: np.ndarray,
    df_tests: list[pd.DataFrame],
    base_cols: list[str],
    seed: int = 42,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Fit Platt-calibrated Stage-1 intake model once and evaluate on D_cal + multiple test sets."""
    y_tr = np.asarray(y_train, dtype=int)
    y_ca = np.asarray(y_cal, dtype=int)
    enc1 = FastFrameEncoder().fit(df_train[base_cols])
    clf1 = lgb.LGBMClassifier(
        n_estimators=140,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=30,
        max_delta_step=1.0,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=seed,
        n_jobs=4,
        verbose=-1,
    )
    clf1.fit(enc1.transform(df_train[base_cols]), y_tr)
    raw1_cal = clf1.predict(enc1.transform(df_cal[base_cols]), raw_score=True)
    platt1 = fit_platt_calibrator(raw1_cal, y_ca)
    p1_cal = np.clip(apply_platt(raw1_cal, platt1), 1e-6, 1.0 - 1e-6)
    p1_tests: list[np.ndarray] = []
    for dft in df_tests:
        raw_t = clf1.predict(enc1.transform(dft[base_cols]), raw_score=True)
        p1_tests.append(np.clip(apply_platt(raw_t, platt1), 1e-6, 1.0 - 1e-6))
    return p1_cal, p1_tests


def evaluate_stage2_multi_test(
    df_train: pd.DataFrame,
    y_train: np.ndarray,
    df_cal: pd.DataFrame,
    y_cal: np.ndarray,
    test_sets: list[tuple[pd.DataFrame, np.ndarray]],
    base_cols: list[str],
    selected_cols: list[str],
    oof_p_base_train: np.ndarray,
    contract: ClinicalContract,
    *,
    stage2_mode: str = "fluid_backbone",
    voi_mode: str = "empirical_martingale",
    deconfound_stage2: bool = False,
    precomputed_stage1: tuple[np.ndarray, list[np.ndarray]] | None = None,
    out_routed_traces: list[dict[str, np.ndarray]] | None = None,
    kappa: float = 24.0,
    seed: int = 42,
) -> list[dict[str, float]]:
    """Fit Stage-2 once on (D_train, D_cal) and evaluate streaming VOI across 1+ test cohorts."""
    y_tr = np.asarray(y_train, dtype=int)
    y_ca = np.asarray(y_cal, dtype=int)
    df_tests = [t[0] for t in test_sets]

    if precomputed_stage1 is not None:
        p1_cal, p1_tests = precomputed_stage1
    else:
        p1_cal, p1_tests = fit_stage1_predictions(
            df_train, y_tr, df_cal, y_ca, df_tests, base_cols, seed=seed
        )

    if not selected_cols:
        out_empty: list[dict[str, float]] = []
        for idx_t, (_, yt_raw) in enumerate(test_sets):
            y_te = np.asarray(yt_raw, dtype=int)
            p1_te = p1_tests[idx_t]
            q_zero = np.zeros(y_te.size, dtype=bool)
            if out_routed_traces is not None:
                out_routed_traces.append(
                    {"p1_te": p1_te, "p_auto_te": p1_te, "auto_te": q_zero, "p2_te": p1_te}
                )
            nb0 = net_benefit_per_1k(y_te, p1_te, q_zero, contract)
            cf0 = cfbh_selection(
                y_ca, p1_cal, y_te, p1_te, 1.0 - contract.target_precision, seed=seed
            )
            out_empty.append(
                {
                    "auprc_base": float(average_precision_score(y_te, p1_te)),
                    "auprc_full_s2": float(average_precision_score(y_te, p1_te)),
                    "auroc_full_s2": float(roc_auc_score(y_te, p1_te)),
                    "nb_base_0pct": nb0,
                    "nb_voi_20pct": nb0,
                    "realized_q_20pct": 0.0,
                    "nb_voi_auto": nb0,
                    "realized_q_auto": 0.0,
                    "nb_full_100pct": net_benefit_per_1k(
                        y_te, p1_te, np.ones(y_te.size, dtype=bool), contract
                    ),
                    "cfbh_voi20_fdp": cf0["fdp"],
                    "cfbh_voi20_recall": cf0["recall"],
                    "cfbh_auto_fdp": cf0["fdp"],
                    "cfbh_auto_recall": cf0["recall"],
                    "learned_beta_cz": 0.0,
                    "learned_kappa": 0.0,
                }
            )
        return out_empty

    enc_b2 = FastFrameEncoder().fit(df_train[base_cols])
    xb_tr = enc_b2.transform(df_train[base_cols])
    xb_cal = enc_b2.transform(df_cal[base_cols])
    xb_tests = [enc_b2.transform(dft[base_cols]) for dft in df_tests]

    if deconfound_stage2:
        deconf = AgnielDeconfounder().fit(df_train[base_cols], df_train[selected_cols])
        s2_tr_df = deconf.transform(df_train[base_cols], df_train[selected_cols], seed=seed + 31)
        s2_cal_df = deconf.transform(df_cal[base_cols], df_cal[selected_cols], seed=seed + 32)
        s2_test_dfs = [
            deconf.transform(dft[base_cols], dft[selected_cols], seed=seed + 40 + i)
            for i, dft in enumerate(df_tests)
        ]
        enc_z = FastFrameEncoder(include_missing_indicators=False).fit(s2_tr_df)
    else:
        s2_tr_df = df_train[selected_cols]
        s2_cal_df = df_cal[selected_cols]
        s2_test_dfs = [dft[selected_cols] for dft in df_tests]
        enc_z = FastFrameEncoder(include_missing_indicators=True).fit(s2_tr_df)

    z_tr = enc_z.transform(s2_tr_df)
    z_cal = enc_z.transform(s2_cal_df)
    z_tests = [enc_z.transform(df_s) for df_s in s2_test_dfs]

    x2_tr = np.hstack([xb_tr, z_tr])
    x2_cal = np.hstack([xb_cal, z_cal])
    x2_tests = [np.hstack([xb_tests[i], z_tests[i]]) for i in range(len(df_tests))]

    learned_beta_cz = 0.0
    learned_kappa = 0.0

    if stage2_mode == "fluid_backbone":
        ell1_tr = logit(np.clip(oof_p_base_train, 1e-5, 1.0 - 1e-5))
        fluid_clf = FluidCollisionBackboneClassifier(
            tau_star=contract.tau_star,
            kappa_max=min(20.0, kappa),
            min_cz_ess=240.0,
            seed=seed + 7,
        ).fit(x2_tr, z_tr, y_tr, ell1_tr)
        raw2_cal = fluid_clf.decision_function(x2_cal, z_cal, logit(p1_cal))
        raw2_tests = [
            fluid_clf.decision_function(x2_tests[i], z_tests[i], logit(p1_tests[i]))
            for i in range(len(df_tests))
        ]
        platt2 = fit_platt_calibrator(raw2_cal, y_ca)
        learned_beta_cz = fluid_clf.beta_cz_
        learned_kappa = fluid_clf.kappa_
    elif stage2_mode == "naive_downweighted":
        clf2 = lgb.LGBMClassifier(
            n_estimators=140,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=30,
            max_delta_step=1.0,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=seed + 7,
            n_jobs=4,
            verbose=-1,
        )
        p0_tr = np.clip(oof_p_base_train, 1e-5, 1.0 - 1e-5)
        w_cz = collision_zone_relevance_weights(p0_tr, contract.tau_star, kappa=kappa, floor=0.05)
        clf2.fit(x2_tr, y_tr, sample_weight=w_cz, init_score=logit(p0_tr))
        raw2_cal = logit(p1_cal) + clf2.predict(x2_cal, raw_score=True)
        raw2_tests = [
            logit(p1_tests[i]) + clf2.predict(x2_tests[i], raw_score=True)
            for i in range(len(df_tests))
        ]
        lr = LogisticRegression(C=1e4, solver="lbfgs", max_iter=500)
        w_cal = collision_zone_relevance_weights(p1_cal, contract.tau_star, kappa=kappa, floor=0.10)
        lr.fit(raw2_cal.reshape(-1, 1), y_ca, sample_weight=w_cal)
        platt2 = (float(lr.coef_[0, 0]), float(lr.intercept_[0]))
        learned_beta_cz = 1.0
        learned_kappa = float(kappa)
    elif stage2_mode == "global":
        clf2 = lgb.LGBMClassifier(
            n_estimators=140,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=30,
            max_delta_step=1.0,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=seed + 7,
            n_jobs=4,
            verbose=-1,
        )
        clf2.fit(x2_tr, y_tr)
        raw2_cal = clf2.predict(x2_cal, raw_score=True)
        raw2_tests = [clf2.predict(x2_tests[i], raw_score=True) for i in range(len(df_tests))]
        platt2 = fit_platt_calibrator(raw2_cal, y_ca)
    else:
        raise ValueError(f"Unknown stage2_mode: {stage2_mode}")

    p2_cal = np.clip(apply_platt(raw2_cal, platt2), 1e-6, 1.0 - 1e-6)

    # Fit 12-bin empirical or parametric logit-update distribution on D_cal
    delta_cal = logit(p2_cal) - logit(p1_cal)
    edges = np.quantile(p1_cal, np.linspace(0.0, 1.0, 13))
    edges[0], edges[-1] = 0.0, 1.0
    b_cal = np.clip(np.digitize(p1_cal, edges[1:-1], right=False), 0, 11)

    q_levels = np.linspace(0.02, 0.98, 33)
    global_q = (
        np.quantile(delta_cal, q_levels) if delta_cal.size >= 5 else np.zeros(33, dtype=float)
    )
    bin_quantiles = np.zeros((12, 33), dtype=float)
    bin_sds = np.zeros(12, dtype=float)
    global_sd = float(np.std(delta_cal, ddof=1)) if delta_cal.size > 1 else 0.5
    for b_i in range(12):
        m = b_cal == b_i
        if int(np.sum(m)) >= 15:
            bin_sds[b_i] = float(np.std(delta_cal[m], ddof=1))
            bin_quantiles[b_i, :] = np.quantile(delta_cal[m], q_levels)
        else:
            bin_sds[b_i] = global_sd
            bin_quantiles[b_i, :] = global_q

    if voi_mode == "empirical_martingale":
        voi_cal = empirical_martingale_voi(p1_cal, b_cal, bin_quantiles, contract)
    else:
        voi_cal = mean_centered_voi(p1_cal, bin_sds[b_cal], contract)

    q_target = 0.20
    gamma = float(np.quantile(voi_cal, 1.0 - q_target))
    gt_cal = int(np.sum(voi_cal > gamma))
    eq_cal = int(np.sum(voi_cal == gamma))
    rho = float(np.clip((q_target * voi_cal.size - gt_cal) / max(1, eq_cal), 0.0, 1.0))

    rng_cal = np.random.default_rng(seed + 999)
    q20_cal = (voi_cal > gamma) | ((voi_cal == gamma) & (rng_cal.uniform(size=voi_cal.size) < rho))
    p_route_20_cal = np.where(q20_cal, p2_cal, p1_cal)
    auto_cal = voi_cal > contract.per_instance_query_cost
    p_auto_cal = np.where(auto_cal, p2_cal, p1_cal)
    fdr_alpha = 1.0 - contract.target_precision

    results: list[dict[str, float]] = []
    for idx_t, (_, yt_raw) in enumerate(test_sets):
        y_te = np.asarray(yt_raw, dtype=int)
        p1_te = p1_tests[idx_t]
        p2_te = np.clip(apply_platt(raw2_tests[idx_t], platt2), 1e-6, 1.0 - 1e-6)
        b_te = np.clip(np.digitize(p1_te, edges[1:-1], right=False), 0, 11)
        if voi_mode == "empirical_martingale":
            voi_te = empirical_martingale_voi(p1_te, b_te, bin_quantiles, contract)
        else:
            voi_te = mean_centered_voi(p1_te, bin_sds[b_te], contract)

        rng_te = np.random.default_rng(seed + 1100 + idx_t)
        q20_te = (voi_te > gamma) | ((voi_te == gamma) & (rng_te.uniform(size=voi_te.size) < rho))
        p_route_20_te = np.where(q20_te, p2_te, p1_te)

        auto_te = voi_te > contract.per_instance_query_cost
        p_auto_te = np.where(auto_te, p2_te, p1_te)
        if out_routed_traces is not None:
            out_routed_traces.append(
                {"p1_te": p1_te, "p_auto_te": p_auto_te, "auto_te": auto_te, "p2_te": p2_te}
            )

        cf20 = cfbh_selection(
            y_ca, p_route_20_cal, y_te, p_route_20_te, fdr_alpha, seed=seed + 1 + 10 * idx_t
        )
        cfa = cfbh_selection(
            y_ca, p_auto_cal, y_te, p_auto_te, fdr_alpha, seed=seed + 2 + 10 * idx_t
        )
        results.append(
            {
                "auprc_base": float(average_precision_score(y_te, p1_te)),
                "auprc_full_s2": float(average_precision_score(y_te, p2_te)),
                "auroc_full_s2": float(roc_auc_score(y_te, p2_te)),
                "nb_base_0pct": net_benefit_per_1k(
                    y_te, p1_te, np.zeros(y_te.size, dtype=bool), contract
                ),
                "nb_voi_20pct": net_benefit_per_1k(y_te, p_route_20_te, q20_te, contract),
                "realized_q_20pct": float(np.mean(q20_te)),
                "nb_voi_auto": net_benefit_per_1k(y_te, p_auto_te, auto_te, contract),
                "realized_q_auto": float(np.mean(auto_te)),
                "nb_full_100pct": net_benefit_per_1k(
                    y_te, p2_te, np.ones(y_te.size, dtype=bool), contract
                ),
                "cfbh_voi20_fdp": cf20["fdp"],
                "cfbh_voi20_recall": cf20["recall"],
                "cfbh_auto_fdp": cfa["fdp"],
                "cfbh_auto_recall": cfa["recall"],
                "learned_beta_cz": float(learned_beta_cz),
                "learned_kappa": float(learned_kappa),
            }
        )
    return results


def evaluate_stage2_and_routing(
    df_train: pd.DataFrame,
    y_train: np.ndarray,
    df_cal: pd.DataFrame,
    y_cal: np.ndarray,
    df_test: pd.DataFrame,
    y_test: np.ndarray,
    base_cols: list[str],
    selected_cols: list[str],
    oof_p_base_train: np.ndarray,
    contract: ClinicalContract,
    *,
    stage2_mode: str | None = None,
    specialist_stage2: bool = True,
    kappa: float = 24.0,
    seed: int = 42,
) -> dict[str, float]:
    """Single-test wrapper around evaluate_stage2_multi_test."""
    mode = (
        stage2_mode
        if stage2_mode is not None
        else ("fluid_backbone" if specialist_stage2 else "global")
    )
    return evaluate_stage2_multi_test(
        df_train,
        y_train,
        df_cal,
        y_cal,
        [(df_test, y_test)],
        base_cols,
        selected_cols,
        oof_p_base_train,
        contract,
        stage2_mode=mode,
        kappa=kappa,
        seed=seed,
    )[0]
