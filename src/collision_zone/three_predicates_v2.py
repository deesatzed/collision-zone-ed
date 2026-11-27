"""Paper 2 v2 / JAMIA Strengthening Extensions to `collision_zone.three_predicates`.

Provides read-only-compatible extensions (zero edits to upstream `collision_zone_ed`):
  1. Configurable `n_bins`, `min_cz_ess`, `kappa_max_s2`, and `n_jobs` wrappers preserving
     100% bit-exactness with archived Paper 2 runs at default hyperparameters.
  2. Empirical Step-1 & Post-Signal All-Null Schervish Gain extraction for P2-5 verification.
  3. Published Sequential Instancewise Active Feature Acquisition comparator (`CovertGreedyCMIAFA`,
     faithful GBDT instantiation of Covert et al., ICML 2023).
  4. Calibration-in-the-Large (`CITL`), Logistic Calibration Slope, Tricube LOESS Integrated
     Calibration Index (`ICI`), Isotonic `ICI`, and Pure Prior-Odds / Marginal-Prevalence
     Intercept Recalibration (`P2-8`).
  5. Davison-Hinkley `(k + 1) / (B + 1)` 10,000-replicate paired & cluster-paired bootstraps,
     formal 2x2 superadditivity interaction bootstrap (`P2-1`, `P2-11`), Standardized Net Benefit
     (`sNB`), and concrete clinical event/draw/shift counts (`P2-10`).
  6. Analytical & empirical `(tau*, kappa)` subthreshold-suppression vs. active-mass frontier (`P2-13`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from scipy.special import expit, logit
from scipy.stats import binomtest, norm
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from collision_zone.three_predicates import (
    AgnielDeconfounder,
    ChannelDef,
    ClinicalContract,
    DiscoveryOutcome,
    DiscoveryStepLog,
    FastFrameEncoder,
    apply_platt,
    bernoulli_kl_from_tau,
    cfbh_selection,
    collision_zone_relevance_weights,
    empirical_martingale_voi,
    fit_platt_calibrator,
    holm_adjust,
    log_likelihood_score,
    mean_centered_voi,
    net_benefit_per_1k,
    sample_pilot,
    schervish_collision_proper_score,
    schervish_collision_weight,
    solve_ess_kappa,
    two_phase_ht_mean_and_se,
    union_collision_gate,
)

_EPS = 1e-12


def clopper_pearson_ci(k: int, n: int, confidence_level: float = 0.95) -> dict[str, float]:
    """Exact Clopper-Pearson binomial proportion 95% confidence interval."""
    if n <= 0:
        return {"k": 0.0, "n": 0.0, "rate": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    kk = int(np.clip(k, 0, n))
    ci = binomtest(kk, int(n)).proportion_ci(confidence_level=confidence_level, method="exact")
    return {
        "k": float(kk),
        "n": float(n),
        "rate": float(kk) / float(n),
        "ci95_low": float(ci.low),
        "ci95_high": float(ci.high),
    }


def standardized_net_benefit(nb_per_1k: float, prevalence: float) -> float:
    """Standardized Net Benefit sNB = NB_per_1k / (1000 * prevalence)."""
    denom = 1000.0 * float(prevalence)
    if denom <= 1e-12:
        return 0.0
    return float(nb_per_1k) / denom


def clinical_impact_counts_per_1k(
    y: np.ndarray,
    p_routed: np.ndarray,
    queried_mask: np.ndarray,
    contract: ClinicalContract,
    *,
    y_cal: np.ndarray | None = None,
    p_cal: np.ndarray | None = None,
    seed: int = 42,
) -> dict[str, float]:
    """Concrete clinical counts per 1,000 arrivals and per 100-visit shift."""
    yv = np.asarray(y, dtype=int)
    pv = np.asarray(p_routed, dtype=float)
    qv = np.asarray(queried_mask, dtype=float)
    n = float(max(1, yv.size))
    prev = float(np.mean(yv))
    act = pv >= contract.tau_star

    tp = float(np.sum(act & (yv == 1)))
    fp = float(np.sum(act & (yv == 0)))
    fn = float(np.sum((~act) & (yv == 1)))
    tn = float(np.sum((~act) & (yv == 0)))
    nq = float(np.sum(qv))

    nb = net_benefit_per_1k(yv, pv, qv, contract)
    snb = standardized_net_benefit(nb, prev)

    out: dict[str, float] = {
        "n_eval": n,
        "prevalence": prev,
        "events_per_1k": 1000.0 * prev,
        "tp_caught_per_1k": 1000.0 * tp / n,
        "fp_triggered_per_1k": 1000.0 * fp / n,
        "fn_missed_per_1k": 1000.0 * fn / n,
        "tn_spared_per_1k": 1000.0 * tn / n,
        "sensitivity": float(tp / max(1.0, tp + fn)),
        "specificity": float(tn / max(1.0, tn + fp)),
        "ppv_at_tau": float(tp / max(1.0, tp + fp)) if (tp + fp) > 0 else 0.0,
        "stage2_queries_per_1k": 1000.0 * nq / n,
        "alerts_at_tau_per_100_shift": 100.0 * (tp + fp) / n,
        "nb_per_1k": nb,
        "snb": snb,
    }
    if y_cal is not None and p_cal is not None:
        cf = cfbh_selection(
            y_cal,
            p_cal,
            yv,
            pv,
            fdr_target=1.0 - contract.target_precision,
            seed=seed,
        )
        n_sel = cf["n_selected"]
        out["cfbh_alerts_per_100_shift"] = 100.0 * n_sel / n
        out["cfbh_alerts_per_1k"] = 1000.0 * n_sel / n
        out["cfbh_realized_fdp"] = cf["fdp"]
        out["cfbh_realized_precision"] = 1.0 - cf["fdp"] if n_sel > 0 else 0.0
        out["cfbh_recall"] = cf["recall"]
        out["cfbh_tp_caught_per_1k"] = 1000.0 * prev * cf["recall"]
    return out


# ---------------------------------------------------------------------------
# Davison-Hinkley Paired Bootstrap & Formal 2x2 Superadditivity Interaction
# ---------------------------------------------------------------------------


def _unit_nb_vector(
    y: np.ndarray,
    p_routed: np.ndarray,
    queried_mask: np.ndarray,
    contract: ClinicalContract,
) -> np.ndarray:
    yv = np.asarray(y, dtype=int)
    pv = np.asarray(p_routed, dtype=float)
    qv = np.asarray(queried_mask, dtype=float)
    act = pv >= contract.tau_star
    w_fp = contract.cost_fp / contract.cost_fn
    w_q = contract.per_instance_query_cost / contract.cost_fn
    return 1000.0 * (
        (act & (yv == 1)).astype(float)
        - w_fp * (act & (yv == 0)).astype(float)
        - w_q * qv.astype(float)
    )


def bootstrap_mean_1d(
    values: np.ndarray,
    *,
    cluster_ids: np.ndarray | None = None,
    n_boot: int = 10000,
    seed: int = 42,
) -> dict[str, float]:
    """Vectorized paired or cluster-paired bootstrap CI and Davison-Hinkley (k+1)/(B+1) p-value."""
    v = np.asarray(values, dtype=float).ravel()
    n_obs = v.size
    if n_obs == 0:
        return {
            "mean": 0.0,
            "boot_se": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
            "p_two_sided_dh": 1.0,
            "p_two_sided_raw": 1.0,
        }

    obs_mean = float(np.mean(v))
    if float(np.max(np.abs(v - obs_mean))) < 1e-15:
        p_const = 1.0 if abs(obs_mean) < 1e-15 else 2.0 / float(n_boot + 1)
        return {
            "mean": obs_mean,
            "boot_se": 0.0,
            "ci95_low": obs_mean,
            "ci95_high": obs_mean,
            "p_two_sided_dh": p_const,
            "p_two_sided_raw": 1.0 if abs(obs_mean) < 1e-15 else 0.0,
        }

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot, dtype=float)

    if cluster_ids is not None:
        _, inv = np.unique(cluster_ids, return_inverse=True)
        n_cl = int(np.max(inv)) + 1
        if n_cl < n_obs:
            cl_sums = np.bincount(inv, weights=v, minlength=n_cl)
            cl_cnts = np.bincount(inv, minlength=n_cl).astype(float)
            chunk = max(1, min(1000, int(20_000_000 // max(1, n_cl))))
            for start in range(0, n_boot, chunk):
                end = min(n_boot, start + chunk)
                idx = rng.integers(0, n_cl, size=(end - start, n_cl))
                num = np.sum(cl_sums[idx], axis=1)
                den = np.sum(cl_cnts[idx], axis=1)
                boot_means[start:end] = num / np.maximum(1.0, den)
        else:
            cluster_ids = None

    if cluster_ids is None:
        chunk = max(1, min(1000, int(20_000_000 // max(1, n_obs))))
        for start in range(0, n_boot, chunk):
            end = min(n_boot, start + chunk)
            idx = rng.integers(0, n_obs, size=(end - start, n_obs))
            boot_means[start:end] = np.mean(v[idx], axis=1)

    se = float(np.std(boot_means, ddof=1))
    ci_low, ci_high = np.quantile(boot_means, [0.025, 0.975])

    k_le = int(np.sum(boot_means <= 0.0))
    k_ge = int(np.sum(boot_means >= 0.0))
    p_dh = min(1.0, 2.0 * min((k_le + 1.0) / (n_boot + 1.0), (k_ge + 1.0) / (n_boot + 1.0)))
    p_raw = min(1.0, 2.0 * min(k_le / float(n_boot), k_ge / float(n_boot)))

    return {
        "mean": obs_mean,
        "boot_se": se,
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "p_two_sided_dh": float(p_dh),
        "p_two_sided_raw": float(p_raw),
    }


def paired_bootstrap_nb_diff_v2(
    y: np.ndarray,
    p_arm: np.ndarray,
    q_arm: np.ndarray,
    p_ref: np.ndarray,
    q_ref: np.ndarray,
    contract: ClinicalContract,
    *,
    cluster_ids: np.ndarray | None = None,
    n_boot: int = 10000,
    seed: int = 42,
) -> dict[str, float]:
    """Patient-level paired bootstrap for NB(arm) - NB(ref) with Davison-Hinkley p-values."""
    d_i = _unit_nb_vector(y, p_arm, q_arm, contract) - _unit_nb_vector(y, p_ref, q_ref, contract)
    res = bootstrap_mean_1d(d_i, cluster_ids=cluster_ids, n_boot=n_boot, seed=seed)
    return {
        "diff_mean": res["mean"],
        "boot_se": res["boot_se"],
        "ci95_low": res["ci95_low"],
        "ci95_high": res["ci95_high"],
        "p_two_sided_dh": res["p_two_sided_dh"],
        "p_two_sided_raw": res["p_two_sided_raw"],
    }


def paired_bootstrap_interaction(
    u_base: np.ndarray,
    u_glob_cmi: np.ndarray,
    u_cz_glob: np.ndarray,
    u_glob_fluid: np.ndarray,
    u_cz_fluid: np.ndarray,
    *,
    cluster_ids: np.ndarray | None = None,
    n_boot: int = 10000,
    seed: int = 42,
) -> dict[str, float]:
    """Formal 2x2 interaction test:
      1. Relative to Global-CMI (2x2 factorial over {Global-Disc, CZ-Disc} x {Global-S2, Fluid-S2}):
         I_fac = (CZ_Fluid - Global_CMI) - [(CZ_Global - Global_CMI) + (Global_Fluid - Global_CMI)]
               = CZ_Fluid - CZ_Global - Global_Fluid + Global_CMI.
      2. Relative to Stage-1 Base (prompt contrast):
         I_base = (CZ_Fluid - Base) - [(CZ_Global - Base) + (Global_Fluid - Base)]
                = CZ_Fluid - CZ_Global - Global_Fluid + Base.
    """
    contrast_base = u_cz_fluid - u_cz_glob - u_glob_fluid + u_base
    contrast_fac = u_cz_fluid - u_cz_glob - u_glob_fluid + u_glob_cmi
    r_base = bootstrap_mean_1d(contrast_base, cluster_ids=cluster_ids, n_boot=n_boot, seed=seed)
    r_fac = bootstrap_mean_1d(contrast_fac, cluster_ids=cluster_ids, n_boot=n_boot, seed=seed + 1)
    return {
        "interaction_vs_base_mean": r_base["mean"],
        "interaction_vs_base_se": r_base["boot_se"],
        "interaction_vs_base_ci95_low": r_base["ci95_low"],
        "interaction_vs_base_ci95_high": r_base["ci95_high"],
        "interaction_vs_base_p_dh": r_base["p_two_sided_dh"],
        "interaction_2x2_factorial_mean": r_fac["mean"],
        "interaction_2x2_factorial_se": r_fac["boot_se"],
        "interaction_2x2_factorial_ci95_low": r_fac["ci95_low"],
        "interaction_2x2_factorial_ci95_high": r_fac["ci95_high"],
        "interaction_2x2_factorial_p_dh": r_fac["p_two_sided_dh"],
    }


# ---------------------------------------------------------------------------
# Calibration Intercept (CITL), Slope, Flexible LOESS ICI & Site Recalibration
# ---------------------------------------------------------------------------


def solve_citl_intercept(y_or_prev: np.ndarray | float, p: np.ndarray) -> float:
    """Solve mean(expit(alpha + logit(p))) == target_prev for Calibration-in-the-Large alpha."""
    pv = np.clip(np.asarray(p, dtype=float).ravel(), 1e-6, 1.0 - 1e-6)
    ell = logit(pv)
    target = float(y_or_prev) if np.ndim(y_or_prev) == 0 else float(np.mean(y_or_prev))
    target = float(np.clip(target, 1e-6, 1.0 - 1e-6))

    def _f(a: float) -> float:
        return float(np.mean(expit(np.clip(a + ell, -35.0, 35.0)))) - target

    if _f(-30.0) >= 0.0:
        return -30.0
    if _f(30.0) <= 0.0:
        return 30.0
    return float(brentq(_f, -30.0, 30.0, xtol=1e-10))


def fit_logistic_calibration_slope_intercept(
    y: np.ndarray,
    p: np.ndarray,
) -> tuple[float, float]:
    """Unpenalized Bernoulli MLE for logit P(Y=1) = b0 + b1 * logit(p)."""
    yv = np.asarray(y, dtype=float).ravel()
    pv = np.clip(np.asarray(p, dtype=float).ravel(), 1e-6, 1.0 - 1e-6)
    ell = logit(pv)
    if yv.size < 5 or float(np.std(ell)) < 1e-8 or float(np.std(yv)) < 1e-8:
        return solve_citl_intercept(yv, pv), 1.0

    b0 = solve_citl_intercept(yv, pv)
    b1 = 1.0
    for _ in range(30):
        eta = np.clip(b0 + b1 * ell, -30.0, 30.0)
        mu = expit(eta)
        w = np.maximum(1e-9, mu * (1.0 - mu))
        r = yv - mu
        g0 = float(np.sum(r))
        g1 = float(np.sum(r * ell))
        if max(abs(g0), abs(g1)) < 1e-7:
            break
        h00 = float(np.sum(w))
        h01 = float(np.sum(w * ell))
        h11 = float(np.sum(w * (ell**2))) + 1e-6
        det = h00 * h11 - h01 * h01
        if det <= 1e-12:
            break
        db0 = (h11 * g0 - h01 * g1) / det
        db1 = (-h01 * g0 + h00 * g1) / det
        b0 += float(np.clip(db0, -2.0, 2.0))
        b1 += float(np.clip(db1, -2.0, 2.0))
    return float(b0), float(b1)


def tricube_loess_calibration_curve(
    y: np.ndarray,
    p: np.ndarray,
    *,
    span: float = 0.35,
    n_anchors: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fast 1D tricube local-linear LOESS calibration smoother (Austin & Steyerberg 2019).

    Returns (p_anchors, y_smooth_anchors, y_smooth_at_each_p).
    """
    yv = np.asarray(y, dtype=float).ravel()
    pv = np.clip(np.asarray(p, dtype=float).ravel(), 1e-6, 1.0 - 1e-6)
    n = pv.size
    order = np.argsort(pv, kind="mergesort")
    ps = pv[order]
    ys = yv[order]

    k_nn = max(min(n, 60), min(n, round(span * n)))
    anchor_idx = np.unique(np.linspace(0, n - 1, min(n, n_anchors), dtype=int))
    p_anc = ps[anchor_idx]
    y_anc = np.empty(anchor_idx.size, dtype=float)

    left = 0
    for j, a_i in enumerate(anchor_idx):
        p0 = ps[a_i]
        # Advance contiguous window [left, left + k_nn) to minimize max(|ps[left]-p0|, |ps[right]-p0|)
        while left + k_nn < n and abs(ps[left + k_nn] - p0) < abs(ps[left] - p0):
            left += 1
        right = min(n, left + k_nn)
        px = ps[left:right]
        yx = ys[left:right]
        dx = np.abs(px - p0)
        dmax = float(np.max(dx))
        if dmax <= 1e-12:
            y_anc[j] = float(np.mean(yx))
            continue
        u = dx / (dmax * 1.0001)
        w = (1.0 - u**3) ** 3
        sw = float(np.sum(w))
        if sw <= 1e-12:
            y_anc[j] = float(np.mean(yx))
            continue
        xc = px - p0
        mx = float(np.sum(w * xc) / sw)
        my = float(np.sum(w * yx) / sw)
        vx = float(np.sum(w * ((xc - mx) ** 2)))
        if vx <= 1e-14:
            pred = my
        else:
            slope_loc = float(np.sum(w * (xc - mx) * (yx - my)) / vx)
            pred = my - slope_loc * mx
        y_anc[j] = float(np.clip(pred, 0.0, 1.0))

    # Deduplicate identical p_anc values before interpolation
    uniq_p, inv_p = np.unique(p_anc, return_inverse=True)
    if uniq_p.size < p_anc.size:
        uniq_y = np.bincount(inv_p, weights=y_anc) / np.maximum(1, np.bincount(inv_p))
    else:
        uniq_y = y_anc
    y_at_p = np.interp(pv, uniq_p, uniq_y)
    return uniq_p, uniq_y, y_at_p


def compute_calibration_metrics(
    y: np.ndarray,
    p: np.ndarray,
    *,
    n_boot: int = 500,
    seed: int = 42,
    return_curve: bool = False,
) -> dict[str, object]:
    """Compute CITL, Calibration Slope, LOESS ICI, Isotonic ICI, E90, O/E, Brier + Bootstrap CIs."""
    yv = np.asarray(y, dtype=int).ravel()
    pv = np.clip(np.asarray(p, dtype=float).ravel(), 1e-6, 1.0 - 1e-6)
    n = yv.size

    citl = solve_citl_intercept(yv, pv)
    cal_int, cal_slope = fit_logistic_calibration_slope_intercept(yv, pv)
    p_anc, y_anc, y_loess = tricube_loess_calibration_curve(yv, pv)
    abs_err = np.abs(pv - y_loess)
    ici_loess = float(np.mean(abs_err))
    e50 = float(np.median(abs_err))
    e90 = float(np.quantile(abs_err, 0.90))

    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    y_iso = iso.fit_transform(pv, yv)
    ici_iso = float(np.mean(np.abs(pv - y_iso)))

    obs_prev = float(np.mean(yv))
    mean_pred = float(np.mean(pv))
    oe_ratio = obs_prev / max(1e-9, mean_pred)
    brier = float(brier_score_loss(yv, pv))

    out: dict[str, object] = {
        "n": int(n),
        "obs_prev": obs_prev,
        "mean_pred": mean_pred,
        "oe_ratio": oe_ratio,
        "citl": citl,
        "cal_intercept": cal_int,
        "cal_slope": cal_slope,
        "ici_loess": ici_loess,
        "ici_isotonic": ici_iso,
        "e50_loess": e50,
        "e90_loess": e90,
        "brier": brier,
    }

    if n_boot > 0:
        rng = np.random.default_rng(seed)
        b_citl = np.empty(n_boot, dtype=float)
        b_slope = np.empty(n_boot, dtype=float)
        b_ici = np.empty(n_boot, dtype=float)
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            yb, pb = yv[idx], pv[idx]
            if int(np.sum(yb)) < 3 or int(np.sum(yb)) > n - 3:
                b_citl[b] = citl
                b_slope[b] = cal_slope
                b_ici[b] = ici_loess
                continue
            b_citl[b] = solve_citl_intercept(yb, pb)
            _, b_slope[b] = fit_logistic_calibration_slope_intercept(yb, pb)
            _, _, yl_b = tricube_loess_calibration_curve(yb, pb, n_anchors=120)
            b_ici[b] = float(np.mean(np.abs(pb - yl_b)))
        out["citl_ci95"] = [float(x) for x in np.quantile(b_citl, [0.025, 0.975])]
        out["cal_slope_ci95"] = [float(x) for x in np.quantile(b_slope, [0.025, 0.975])]
        out["ici_loess_ci95"] = [float(x) for x in np.quantile(b_ici, [0.025, 0.975])]

    if return_curve:
        # Subsample up to 80 curve points for clean JSON storage & plotting
        step = max(1, p_anc.size // 80)
        out["curve_p"] = [float(x) for x in p_anc[::step]]
        out["curve_y_loess"] = [float(x) for x in y_anc[::step]]
    return out


def prior_odds_recalibrate_probabilities(
    p: np.ndarray,
    source_prev: float,
    target_prev: float,
) -> tuple[np.ndarray, float]:
    """Closed-form Bayes prior-odds shift: logit(p_new) = logit(p) + logit(pi_tgt) - logit(pi_src).

    Uses ZERO individual target outcome labels — only scalar marginal prevalence pi_tgt.
    """
    pv = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    offset = float(
        logit(np.clip(target_prev, 1e-6, 1.0 - 1e-6))
        - logit(np.clip(source_prev, 1e-6, 1.0 - 1e-6))
    )
    p_recal = np.clip(expit(np.clip(logit(pv) + offset, -25.0, 25.0)), 1e-6, 1.0 - 1e-6)
    return p_recal, offset


# ---------------------------------------------------------------------------
# Configurable Discovery & Stage-2 Helpers (Bit-Exact with v1 at Defaults)
# ---------------------------------------------------------------------------


@dataclass
class DetailedDiscoveryOutcome:
    outcome: DiscoveryOutcome
    step1_pvals: dict[str, float] = field(default_factory=dict)
    step1_ses: dict[str, float] = field(default_factory=dict)
    pure_null_pool_fwer_reject: bool = False
    pure_null_step1_gains: list[float] = field(default_factory=list)
    post_signal_null_gains: list[float] = field(default_factory=list)


def compute_oof_stage1_v2(
    df_base: pd.DataFrame,
    y: np.ndarray,
    n_folds: int = 4,
    seed: int = 42,
    *,
    n_jobs: int = 4,
) -> np.ndarray:
    """Bit-exact configurable-thread version of `compute_oof_stage1`."""
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
            n_jobs=n_jobs,
            verbose=-1,
        )
        clf.fit(x_tr, yv[fit_sub])
        raw_ca = clf.predict(x_ca, raw_score=True)
        raw_va = clf.predict(x_va, raw_score=True)
        platt = fit_platt_calibrator(raw_ca, yv[cal_sub])
        oof_p[val_idx] = apply_platt(raw_va, platt)
    return np.clip(oof_p, 1e-5, 1.0 - 1e-5)


def run_sequential_discovery_v2(
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
    n_bins: int = 8,
    alpha: float = 0.05,
    use_holm_lcb: bool = True,
    deconfound_stage2: bool = False,
    adaptive_kappa: bool = True,
    n_jobs: int = 4,
    seed: int = 42,
) -> DetailedDiscoveryOutcome:
    """Bit-exact extension of `run_sequential_discovery` recording null-gain diagnostics & sensitivity."""
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
        n_bins=n_bins,
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
                n_jobs=n_jobs,
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
    step1_pvals: dict[str, float] = {}
    step1_ses: dict[str, float] = {}
    pure_null_step1_gains: list[float] = []
    post_signal_null_gains: list[float] = []
    pure_null_pool_fwer_reject = False

    cur_p = _oof_probs_for_set([])
    cur_s = _score_vec(cur_p)

    step_num = 0
    while True:
        step_num += 1
        affordable = [
            ch
            for ch in channels
            if ch.channel_id not in selected and spent + ch.cost <= budget + 1e-9
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
                step1_pvals[ch.channel_id] = pval
                step1_ses[ch.channel_id] = se_tot
                if ch.is_null_stress:
                    pure_null_step1_gains.append(g_hat)

        if step_num == 1:
            null_pvals = [
                cand_pvals[i] for i, ch in enumerate(affordable) if ch.is_null_stress
            ]
            null_gains = [
                cand_gains[i] for i, ch in enumerate(affordable) if ch.is_null_stress
            ]
            if null_pvals:
                null_holm = holm_adjust(null_pvals)
                pure_null_pool_fwer_reject = any(
                    (null_holm[j] < alpha) and (null_gains[j] > 0.0)
                    for j in range(len(null_pvals))
                )

        if all(ch.is_null_stress for ch in affordable) and step_num > 1:
            post_signal_null_gains.extend(cand_gains)

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

    outcome = DiscoveryOutcome(
        method=method,
        pilot_fraction=pilot_fraction,
        selected_ids=selected,
        total_cost=spent,
        pilot_high_signal_fp_share=pilot.high_signal_fp_share_of_controls,
        steps=steps,
        step1_scores=step1_scores,
    )
    return DetailedDiscoveryOutcome(
        outcome=outcome,
        step1_pvals=step1_pvals,
        step1_ses=step1_ses,
        pure_null_pool_fwer_reject=pure_null_pool_fwer_reject,
        pure_null_step1_gains=pure_null_step1_gains,
        post_signal_null_gains=post_signal_null_gains,
    )


def _fit_gated_residual_booster_v2(
    z_aug: np.ndarray,
    y_tr: np.ndarray,
    f_glob_tr: np.ndarray,
    pi_tr: np.ndarray,
    *,
    n_estimators: int = 50,
    learning_rate: float = 0.05,
    num_leaves: int = 8,
    reg_lambda_eff: float = 3.5,
    n_jobs: int = 4,
    seed: int = 42,
) -> lgb.Booster:
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
        "n_jobs": n_jobs,
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


class FluidCollisionBackboneClassifierV2:
    """Configurable-thread/hyperparameter equivalent of `FluidCollisionBackboneClassifier`."""

    def __init__(
        self,
        tau_star: float,
        *,
        kappa_max: float = 20.0,
        min_cz_ess: float = 240.0,
        n_jobs: int = 4,
        seed: int = 42,
    ) -> None:
        self.tau_star = float(tau_star)
        self.kappa_max = float(kappa_max)
        self.min_cz_ess = float(min_cz_ess)
        self.n_jobs = int(n_jobs)
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
            n_jobs=self.n_jobs,
            verbose=-1,
        )

    def fit(
        self,
        x_all: np.ndarray,
        z_s2: np.ndarray,
        y: np.ndarray,
        ell1_base: np.ndarray,
    ) -> FluidCollisionBackboneClassifierV2:
        yv = np.asarray(y, dtype=int)
        ell1 = np.clip(np.asarray(ell1_base, dtype=float), -20.0, 20.0)
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.seed)

        oof_raw_glob = np.zeros(yv.size, dtype=float)
        splits = list(skf.split(x_all, yv))
        for k, (tr_i, va_i) in enumerate(splits):
            mg = self._make_glob_clf(seed_offset=10 + k)
            mg.fit(x_all[tr_i], yv[tr_i])
            oof_raw_glob[va_i] = mg.predict(x_all[va_i], raw_score=True)

        self.platt_glob_ = fit_platt_calibrator(oof_raw_glob, yv)
        ag, bg = self.platt_glob_
        oof_f_glob = np.clip(ag * oof_raw_glob + bg, -20.0, 20.0)

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

        z_aug = np.column_stack([ell1, oof_f_glob, z_s2]).astype(np.float32)
        oof_d_cz = np.zeros(yv.size, dtype=float)
        for k, (tr_i, va_i) in enumerate(splits):
            mcz = _fit_gated_residual_booster_v2(
                z_aug[tr_i],
                yv[tr_i],
                oof_f_glob[tr_i],
                pi_oof[tr_i],
                n_estimators=cz_trees,
                num_leaves=cz_leaves,
                n_jobs=self.n_jobs,
                seed=self.seed + 30 + k,
            )
            oof_d_cz[va_i] = mcz.predict(z_aug[va_i], raw_score=True)

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

        self.m_glob_ = self._make_glob_clf(seed_offset=7)
        self.m_glob_.fit(x_all, yv)
        if self.beta_cz_ > 1e-4:
            self.m_cz_ = _fit_gated_residual_booster_v2(
                z_aug,
                yv,
                oof_f_glob,
                pi_oof,
                n_estimators=cz_trees,
                num_leaves=cz_leaves,
                n_jobs=self.n_jobs,
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


def evaluate_stage2_multi_test_v2(
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
    out_cal_trace: dict[str, np.ndarray] | None = None,
    kappa: float = 24.0,
    kappa_max_s2: float | None = None,
    min_cz_ess: float = 240.0,
    n_jobs: int = 4,
    seed: int = 42,
) -> list[dict[str, float]]:
    """Bit-exact extension of `evaluate_stage2_multi_test` with configurable sensitivity params & cal traces."""
    y_tr = np.asarray(y_train, dtype=int)
    y_ca = np.asarray(y_cal, dtype=int)
    df_tests = [t[0] for t in test_sets]

    if precomputed_stage1 is not None:
        p1_cal, p1_tests = precomputed_stage1
    else:
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
            n_jobs=n_jobs,
            verbose=-1,
        )
        clf1.fit(enc1.transform(df_train[base_cols]), y_tr)
        raw1_cal = clf1.predict(enc1.transform(df_cal[base_cols]), raw_score=True)
        platt1 = fit_platt_calibrator(raw1_cal, y_ca)
        p1_cal = np.clip(apply_platt(raw1_cal, platt1), 1e-6, 1.0 - 1e-6)
        p1_tests = [
            np.clip(apply_platt(clf1.predict(enc1.transform(dft[base_cols]), raw_score=True), platt1), 1e-6, 1.0 - 1e-6)
            for dft in df_tests
        ]

    if not selected_cols:
        if out_cal_trace is not None:
            out_cal_trace["p1_cal"] = p1_cal
            out_cal_trace["p2_cal"] = p1_cal
            out_cal_trace["p_auto_cal"] = p1_cal
        out_empty: list[dict[str, float]] = []
        for idx_t, (_, yt_raw) in enumerate(test_sets):
            y_te = np.asarray(yt_raw, dtype=int)
            p1_te = p1_tests[idx_t]
            q_zero = np.zeros(y_te.size, dtype=bool)
            if out_routed_traces is not None:
                out_routed_traces.append(
                    {
                        "p1_te": p1_te,
                        "p_auto_te": p1_te,
                        "auto_te": q_zero,
                        "p_20_te": p1_te,
                        "q20_te": q_zero,
                        "p2_te": p1_te,
                    }
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
        eff_kmax = float(kappa_max_s2) if kappa_max_s2 is not None else min(20.0, kappa)
        fluid_clf = FluidCollisionBackboneClassifierV2(
            tau_star=contract.tau_star,
            kappa_max=eff_kmax,
            min_cz_ess=min_cz_ess,
            n_jobs=n_jobs,
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
            n_jobs=n_jobs,
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
            n_jobs=n_jobs,
            verbose=-1,
        )
        clf2.fit(x2_tr, y_tr)
        raw2_cal = clf2.predict(x2_cal, raw_score=True)
        raw2_tests = [clf2.predict(x2_tests[i], raw_score=True) for i in range(len(df_tests))]
        platt2 = fit_platt_calibrator(raw2_cal, y_ca)
    else:
        raise ValueError(f"Unknown stage2_mode: {stage2_mode}")

    p2_cal = np.clip(apply_platt(raw2_cal, platt2), 1e-6, 1.0 - 1e-6)

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

    if out_cal_trace is not None:
        out_cal_trace["p1_cal"] = p1_cal
        out_cal_trace["p2_cal"] = p2_cal
        out_cal_trace["p_auto_cal"] = p_auto_cal
        out_cal_trace["edges"] = edges
        out_cal_trace["bin_quantiles"] = bin_quantiles

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
                {
                    "p1_te": p1_te,
                    "p_auto_te": p_auto_te,
                    "auto_te": auto_te,
                    "p_20_te": p_route_20_te,
                    "q20_te": q20_te,
                    "p2_te": p2_te,
                }
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


# ---------------------------------------------------------------------------
# P2-7: Published Sequential Instancewise Greedy-CMI AFA Comparator
#       (Covert et al., ICML 2023 — Dynamic Feature Selection via CMI)
# ---------------------------------------------------------------------------


def evaluate_covert_greedy_cmi_afa(
    df_train: pd.DataFrame,
    y_train: np.ndarray,
    df_cal: pd.DataFrame,
    y_cal: np.ndarray,
    test_sets: list[tuple[pd.DataFrame, np.ndarray]],
    base_cols: list[str],
    channels: list[ChannelDef],
    oof_p_base_train: np.ndarray,
    precomputed_stage1: tuple[np.ndarray, list[np.ndarray]],
    contract: ClinicalContract,
    *,
    budget: float = 3.2,
    deconfound_stage2: bool = False,
    out_routed_traces: list[dict[str, np.ndarray]] | None = None,
    n_masks_per_row: int = 5,
    n_jobs: int = 4,
    seed: int = 42,
) -> list[dict[str, float]]:
    """Faithful GBDT instantiation of Covert et al. (2023) sequential instancewise greedy CMI AFA.

    1. Trains a masked residual GBDT predictor p_theta(Y | X_base, Z_S, M_S) anchored at Stage-1
       logit(p1) across random within-budget subsets S subseteq {1..M} (returning exact Stage-1
       p1 when S = empty).
    2. Trains per-channel GBDT value regressors v_phi(m | logit(p_S), X_base, Z_S, M_S) predicting
       non-negative conditional mutual information gain D_KL(Bern(p_{S u {m}}) || Bern(p_S)) >= 0
       (Covert et al. 2023, Sec. 3.2).
    3. Tunes the cost-penalized CMI stopping threshold lambda_CMI* strictly on D_cal to maximize
       deployment Net Benefit NB_K, then executes greedy instancewise rollouts zero-shot on test sets.
    """
    y_tr = np.asarray(y_train, dtype=int)
    y_ca = np.asarray(y_cal, dtype=int)
    p1_cal, p1_tests = precomputed_stage1
    df_tests = [t[0] for t in test_sets]

    m_ch = len(channels)
    costs = np.array([float(ch.cost) for ch in channels], dtype=float)
    all_ch_cols = list(dict.fromkeys(c for ch in channels for c in ch.columns))

    enc_b = FastFrameEncoder().fit(df_train[base_cols])
    xb_tr = enc_b.transform(df_train[base_cols]).astype(np.float32)
    xb_ca = enc_b.transform(df_cal[base_cols]).astype(np.float32)
    xb_tests = [enc_b.transform(dft[base_cols]).astype(np.float32) for dft in df_tests]

    if deconfound_stage2 and all_ch_cols:
        deconf = AgnielDeconfounder().fit(df_train[base_cols], df_train[all_ch_cols])
        s2_tr_df = deconf.transform(df_train[base_cols], df_train[all_ch_cols], seed=seed + 11)
        s2_ca_df = deconf.transform(df_cal[base_cols], df_cal[all_ch_cols], seed=seed + 12)
        s2_te_dfs = [
            deconf.transform(dft[base_cols], dft[all_ch_cols], seed=seed + 20 + i)
            for i, dft in enumerate(df_tests)
        ]
        inc_miss = False
    else:
        s2_tr_df = df_train[all_ch_cols]
        s2_ca_df = df_cal[all_ch_cols]
        s2_te_dfs = [dft[all_ch_cols] for dft in df_tests]
        inc_miss = True

    # Encode each channel separately with sentinel -999.0 when unobserved by the AFA policy
    ch_slices: list[tuple[int, int]] = []
    z_blocks_tr: list[np.ndarray] = []
    z_blocks_ca: list[np.ndarray] = []
    z_blocks_te: list[list[np.ndarray]] = [[] for _ in df_tests]
    col_cursor = 0
    for ch in channels:
        cols_c = list(ch.columns)
        enc_c = FastFrameEncoder(include_missing_indicators=inc_miss).fit(s2_tr_df[cols_c])
        b_tr = enc_c.transform(s2_tr_df[cols_c]).astype(np.float32)
        b_ca = enc_c.transform(s2_ca_df[cols_c]).astype(np.float32)
        w_c = b_tr.shape[1]
        ch_slices.append((col_cursor, col_cursor + w_c))
        col_cursor += w_c
        z_blocks_tr.append(b_tr)
        z_blocks_ca.append(b_ca)
        for i_t, df_s in enumerate(s2_te_dfs):
            z_blocks_te[i_t].append(enc_c.transform(df_s[cols_c]).astype(np.float32))

    z_full_tr = np.hstack(z_blocks_tr)
    z_full_ca = np.hstack(z_blocks_ca)
    z_full_tes = [np.hstack(blks) for blks in z_blocks_te]
    sentinel = np.float32(-999.0)

    def _build_masked_features(
        xb: np.ndarray,
        z_full: np.ndarray,
        mask_mat: np.ndarray,
    ) -> np.ndarray:
        z_m = np.full_like(z_full, sentinel)
        for j, (c0, c1) in enumerate(ch_slices):
            obs_j = mask_mat[:, j] > 0.5
            if np.any(obs_j):
                z_m[obs_j, c0:c1] = z_full[obs_j, c0:c1]
        return np.hstack([xb, z_m, mask_mat.astype(np.float32)])

    rng = np.random.default_rng(seed + 303)

    def _sample_affordable_masks(n_rows: int, rng_local: np.random.Generator) -> np.ndarray:
        masks = np.zeros((n_rows, m_ch), dtype=np.float32)
        for i in range(n_rows):
            perm = rng_local.permutation(m_ch)
            target_k = int(rng_local.integers(1, min(4, m_ch) + 1))
            rem_b = budget
            picked = 0
            for j in perm:
                if picked >= target_k:
                    break
                if costs[j] <= rem_b + 1e-9:
                    masks[i, j] = 1.0
                    rem_b -= costs[j]
                    picked += 1
        return masks

    # 1. Fit masked predictor p_theta anchored on Stage-1 logit(p1)
    n_tr = y_tr.size
    ell1_tr = logit(np.clip(oof_p_base_train, 1e-5, 1.0 - 1e-5))
    x_rep_list: list[np.ndarray] = []
    y_rep_list: list[np.ndarray] = []
    init_rep_list: list[np.ndarray] = []
    for _ in range(n_masks_per_row):
        m_mat = _sample_affordable_masks(n_tr, rng)
        x_rep_list.append(_build_masked_features(xb_tr, z_full_tr, m_mat))
        y_rep_list.append(y_tr)
        init_rep_list.append(ell1_tr)

    x_masked_tr = np.vstack(x_rep_list)
    y_masked_tr = np.concatenate(y_rep_list)
    init_masked_tr = np.concatenate(init_rep_list)

    clf_masked = lgb.LGBMClassifier(
        n_estimators=140,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=30,
        max_delta_step=1.0,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=seed + 41,
        n_jobs=n_jobs,
        verbose=-1,
    )
    clf_masked.fit(x_masked_tr, y_masked_tr, init_score=init_masked_tr)

    # Calibrate masked predictor on random within-budget subsets of D_cal
    m_cal_rand = _sample_affordable_masks(y_ca.size, rng)
    raw_ca_masked = logit(p1_cal) + clf_masked.predict(
        _build_masked_features(xb_ca, z_full_ca, m_cal_rand), raw_score=True
    )
    platt_m = fit_platt_calibrator(raw_ca_masked, y_ca)

    def _predict_masked_p(
        xb: np.ndarray,
        z_full: np.ndarray,
        p1_vec: np.ndarray,
        mask_mat: np.ndarray,
    ) -> np.ndarray:
        has_any = np.any(mask_mat > 0.5, axis=1)
        out = p1_vec.copy()
        if np.any(has_any):
            xm = _build_masked_features(xb[has_any], z_full[has_any], mask_mat[has_any])
            raw = logit(p1_vec[has_any]) + clf_masked.predict(xm, raw_score=True)
            out[has_any] = np.clip(apply_platt(raw, platt_m), 1e-6, 1.0 - 1e-6)
        return out

    # 2. Fit per-channel CMI regressors v_phi(m | state) on training states (empty + 1-step partial masks)
    sub_n = min(n_tr, 6000)
    sub_idx = rng.choice(n_tr, size=sub_n, replace=False) if sub_n < n_tr else np.arange(n_tr)
    xb_s = xb_tr[sub_idx]
    zf_s = z_full_tr[sub_idx]
    p1_s = np.clip(oof_p_base_train[sub_idx], 1e-5, 1.0 - 1e-5)

    m_zero = np.zeros((sub_n, m_ch), dtype=np.float32)
    m_part = np.zeros((sub_n, m_ch), dtype=np.float32)
    for i in range(sub_n):
        j0 = int(rng.integers(0, m_ch))
        if costs[j0] < budget - 0.5:
            m_part[i, j0] = 1.0

    states_m = np.vstack([m_zero, m_part])
    states_xb = np.vstack([xb_s, xb_s])
    states_zf = np.vstack([zf_s, zf_s])
    states_p1 = np.concatenate([p1_s, p1_s])
    p_cur_state = _predict_masked_p(states_xb, states_zf, states_p1, states_m)
    x_state_feat = np.column_stack(
        [
            logit(p_cur_state).astype(np.float32),
            _build_masked_features(states_xb, states_zf, states_m),
        ]
    )

    cmi_regs: list[lgb.LGBMRegressor | None] = []
    for j in range(m_ch):
        valid = (states_m[:, j] < 0.5) & ((states_m @ costs) + costs[j] <= budget + 1e-9)
        if int(np.sum(valid)) < 40:
            cmi_regs.append(None)
            continue
        m_plus = states_m[valid].copy()
        m_plus[:, j] = 1.0
        p_plus = _predict_masked_p(states_xb[valid], states_zf[valid], states_p1[valid], m_plus)
        p_base_v = p_cur_state[valid]
        # Exact conditional Bernoulli KL divergence D_KL(Bern(p_plus) || Bern(p_cur)) >= 0
        cmi_target = (
            p_plus * np.log(p_plus / p_base_v)
            + (1.0 - p_plus) * np.log((1.0 - p_plus) / (1.0 - p_base_v))
        )
        cmi_target = np.maximum(0.0, cmi_target)
        reg_j = lgb.LGBMRegressor(
            n_estimators=55,
            learning_rate=0.06,
            num_leaves=15,
            min_child_samples=25,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=seed + 200 + j,
            n_jobs=n_jobs,
            verbose=-1,
        )
        reg_j.fit(x_state_feat[valid], cmi_target)
        cmi_regs.append(reg_j)

    def _rollout_greedy_cmi(
        xb: np.ndarray,
        z_full: np.ndarray,
        p1_vec: np.ndarray,
        lambda_stop: float,
        max_steps: int = 3,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Execute sequential instancewise greedy CMI acquisition up to budget."""
        n = p1_vec.size
        mask_cur = np.zeros((n, m_ch), dtype=np.float32)
        spent_cur = np.zeros(n, dtype=float)
        p_cur = p1_vec.copy()
        active = np.ones(n, dtype=bool)
        step1_best_ratio = np.zeros(n, dtype=float)

        for step in range(max_steps):
            act_idx = np.flatnonzero(active)
            if act_idx.size == 0:
                break
            x_st = np.column_stack(
                [
                    logit(p_cur[act_idx]).astype(np.float32),
                    _build_masked_features(xb[act_idx], z_full[act_idx], mask_cur[act_idx]),
                ]
            )
            ratios = np.full((act_idx.size, m_ch), -np.inf, dtype=float)
            for j, reg_j in enumerate(cmi_regs):
                if reg_j is None:
                    continue
                can_buy = (mask_cur[act_idx, j] < 0.5) & (
                    spent_cur[act_idx] + costs[j] <= budget + 1e-9
                )
                if not np.any(can_buy):
                    continue
                pred_cmi = np.maximum(0.0, reg_j.predict(x_st[can_buy]))
                ratios[can_buy, j] = pred_cmi / costs[j]

            best_j = np.argmax(ratios, axis=1)
            best_r = ratios[np.arange(act_idx.size), best_j]
            if step == 0:
                step1_best_ratio[act_idx] = np.where(np.isfinite(best_r), best_r, 0.0)

            go = np.isfinite(best_r) & (best_r > lambda_stop)
            if not np.any(go):
                break
            go_rows = act_idx[go]
            go_ch = best_j[go]
            mask_cur[go_rows, go_ch] = 1.0
            spent_cur[go_rows] += costs[go_ch]
            p_cur[go_rows] = _predict_masked_p(
                xb[go_rows], z_full[go_rows], p1_vec[go_rows], mask_cur[go_rows]
            )
            active[~np.isin(np.arange(n), go_rows)] = False

        queried = np.any(mask_cur > 0.5, axis=1)
        return p_cur, queried, step1_best_ratio

    # 3. Tune stopping threshold lambda_CMI* strictly on D_cal
    _, _, cal_step1_r = _rollout_greedy_cmi(xb_ca, z_full_ca, p1_cal, lambda_stop=1e9, max_steps=1)
    pos_r = cal_step1_r[cal_step1_r > 0.0]
    if pos_r.size >= 10:
        cand_lambdas = np.unique(
            np.concatenate(
                [
                    np.quantile(pos_r, [0.10, 0.25, 0.40, 0.55, 0.70, 0.80, 0.90, 0.96]),
                    [1e9],
                ]
            )
        )
    else:
        cand_lambdas = np.array([1e-4, 1e-3, 1e-2, 1e9])

    best_lam = float(cand_lambdas[-1])
    best_cal_nb = -np.inf
    best_p_auto_cal = p1_cal
    for lam in cand_lambdas:
        p_ca_try, q_ca_try, _ = _rollout_greedy_cmi(xb_ca, z_full_ca, p1_cal, float(lam))
        nb_try = net_benefit_per_1k(y_ca, p_ca_try, q_ca_try, contract)
        if nb_try > best_cal_nb:
            best_cal_nb = nb_try
            best_lam = float(lam)
            best_p_auto_cal = p_ca_try

    lam_20 = float(np.quantile(cal_step1_r, 0.80)) if cal_step1_r.size > 0 else 1e9
    p_20_cal, _, _ = _rollout_greedy_cmi(xb_ca, z_full_ca, p1_cal, lam_20)
    fdr_alpha = 1.0 - contract.target_precision

    # Full-budget rollout for AUROC / AUPRC / 100%-query comparison
    results: list[dict[str, float]] = []
    for i_t, (_, yt_raw) in enumerate(test_sets):
        y_te = np.asarray(yt_raw, dtype=int)
        p1_te = p1_tests[i_t]
        xb_te = xb_tests[i_t]
        zf_te = z_full_tes[i_t]

        p_auto_te, auto_te, _ = _rollout_greedy_cmi(xb_te, zf_te, p1_te, best_lam)
        p_20_te, q20_te, _ = _rollout_greedy_cmi(xb_te, zf_te, p1_te, lam_20)
        p_full_te, _, _ = _rollout_greedy_cmi(xb_te, zf_te, p1_te, 0.0)

        if out_routed_traces is not None:
            out_routed_traces.append(
                {
                    "p1_te": p1_te,
                    "p_auto_te": p_auto_te,
                    "auto_te": auto_te,
                    "p_20_te": p_20_te,
                    "q20_te": q20_te,
                    "p2_te": p_full_te,
                }
            )

        cf20 = cfbh_selection(y_ca, p_20_cal, y_te, p_20_te, fdr_alpha, seed=seed + 51 + i_t)
        cfa = cfbh_selection(y_ca, best_p_auto_cal, y_te, p_auto_te, fdr_alpha, seed=seed + 61 + i_t)
        results.append(
            {
                "auprc_base": float(average_precision_score(y_te, p1_te)),
                "auprc_full_s2": float(average_precision_score(y_te, p_full_te)),
                "auroc_full_s2": float(roc_auc_score(y_te, p_full_te)),
                "nb_base_0pct": net_benefit_per_1k(
                    y_te, p1_te, np.zeros(y_te.size, dtype=bool), contract
                ),
                "nb_voi_20pct": net_benefit_per_1k(y_te, p_20_te, q20_te, contract),
                "realized_q_20pct": float(np.mean(q20_te)),
                "nb_voi_auto": net_benefit_per_1k(y_te, p_auto_te, auto_te, contract),
                "realized_q_auto": float(np.mean(auto_te)),
                "nb_full_100pct": net_benefit_per_1k(
                    y_te, p_full_te, np.ones(y_te.size, dtype=bool), contract
                ),
                "cfbh_voi20_fdp": cf20["fdp"],
                "cfbh_voi20_recall": cf20["recall"],
                "cfbh_auto_fdp": cfa["fdp"],
                "cfbh_auto_recall": cfa["recall"],
                "tuned_lambda_cmi": best_lam,
            }
        )
    return results


# ---------------------------------------------------------------------------
# P2-13: Analytical & Empirical (tau*, kappa) Suppression-vs-Active-Mass Frontier
# ---------------------------------------------------------------------------


def compute_tau_kappa_frontier(
    p_base_samples: np.ndarray,
    tau_star: float,
    kappas: list[float] | np.ndarray,
    *,
    min_cz_ess: float = 240.0,
) -> dict[str, object]:
    """Compute subthreshold suppression exponent kappa*tau* vs. active collision mass M(kappa)."""
    pv = np.clip(np.asarray(p_base_samples, dtype=float).ravel(), 1e-6, 1.0 - 1e-6)
    dkl = bernoulli_kl_from_tau(pv, tau_star)
    k_grid = [float(k) for k in kappas]
    n = float(pv.size)
    p_sub = max(1e-4, 0.10 * float(tau_star))

    suppression_exponents: list[float] = []
    decoy_suppression_ratios: list[float] = []
    active_masses: list[float] = []
    active_fracs: list[float] = []

    for k in k_grid:
        suppression_exponents.append(float(k * tau_star))
        if k <= 1e-9:
            decoy_suppression_ratios.append(1.0)
            active_masses.append(n)
            active_fracs.append(1.0)
        else:
            w_peak = float(schervish_collision_weight(np.array([tau_star]), tau_star, k)[0])
            w_sub = float(schervish_collision_weight(np.array([p_sub]), tau_star, k)[0])
            decoy_suppression_ratios.append(w_peak / max(1e-30, w_sub))
            m_k = float(np.sum(np.exp(-2.0 * k * dkl)))
            active_masses.append(m_k)
            active_fracs.append(m_k / max(1.0, n))

    kappa_star_s2 = solve_ess_kappa(dkl, kappa_max=20.0, min_effective_cz_samples=min_cz_ess)
    m_at_star = float(np.sum(np.exp(-2.0 * kappa_star_s2 * dkl)))
    return {
        "tau_star": float(tau_star),
        "n_samples": int(pv.size),
        "kappas": k_grid,
        "suppression_exponent_kappa_tau": suppression_exponents,
        "decoy_suppression_ratio_at_0p1_tau": decoy_suppression_ratios,
        "active_hessian_mass": active_masses,
        "active_hessian_fraction": active_fracs,
        "solved_kappa_star_s2": kappa_star_s2,
        "solved_exponent_kappa_tau": float(kappa_star_s2 * tau_star),
        "solved_active_mass": m_at_star,
    }
