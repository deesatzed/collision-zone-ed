"""Paper 2 v2.3 Complete Experimental Remediation Suite (`F1`–`F15`, Sections 8, 12, 13).

Modular block runner writing deterministic JSON checkpoints to `results/v2_3_blocks/`:
  - Block 1 (`--block 1`): `F3/F4/F6` identities, `F5` 6-scenario stress suite (cross-fitted vs.
    50/50 split-pilot confirmation), `F14` 6-arm discovery factorial, `F8` Oracle vs. estimated
    VOI calibration/routing, and `Section 8` 36-cell fractional-factorial regime map.
  - Block 2 (`--block 2`): `F1/F2/F7/F8/F12/F14` Sepsis BIDMC -> Emory primary evaluation with
    explicit `PanelCostSchedule` ledger, `point_in_time_last` vs. `window_6h_summary` channels,
    Common-Panel Bagged VOI (primary), Union-Panel Bagged VOI, Mean Single-Split Policy,
    Fractionally Billed Ensemble, Channel-Billed Covert-CMI & Direct-VOI AFA, clinical 5x3
    discovery-by-backbone factorial, directional/sensitivity-constrained VOI policies, cost
    sensitivity sweep, `R_CZ` diagnostics, and Track A / B1a deconfounded cohorts.
  - Block 3 (`--block 3`): `F9` 2D Observation-Window (4/6/8/12h) x Fixed Prediction-Horizon
    (next 2h, 6h, 12h, remaining stay) matrix + pre-onset lead-time stratification + SOFA
    incorporation-bias removal; `F10` Zero-Shot vs. Unlabeled Saerens EM vs. Historical Split vs.
    Oracle Target Prevalence + error sensitivity; `F11` per-channel availability & complete-case
    observability audit.
  - Block 4 (`--block 4`): `F13` 9-configuration hyperparameter grid on primary Common-Panel
    Bagged VOI; Yale Cross-Campus `Campus A -> Campus B` and untested `Campus A -> Campus C`
    (`N = 71,706` all-comers; `N = 31,430` ESI 4-5); `F15` 11-subgroup utility/calibration audit
    vs. both Stage-1 and Global-CMI; and `R = 200` disjoint repeated-subsample Campus B `CfBH`
    burn-in governance curve across `n_burn in {0, 250, 500, 1000, 2000, 4000}`.

Strict Governance:
  - Enforces `OMP_NUM_THREADS=4` and `n_jobs <= 4` everywhere.
  - Read-only on `/Volumes/WS4TB/imboreceipt_data`.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")
N_JOBS = int(os.environ.get("CZ_N_JOBS", "4"))

import json
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for p in (str(REPO_ROOT), str(SRC_ROOT), str(SCRIPTS_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import lightgbm as lgb
import numpy as np
import pandas as pd
from run_paper2_v2_suite import (
    CMS_CPT_VENIPUNCTURE_CHANNELS,
    FAIRNESS_EXCLUDED_BASE_COLS,
    PROC_DIR,
    RAW_DIR,
    _agg,
    make_stratified_3way_split,
)
from scipy.special import expit, logit
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from collision_zone.three_predicates import (
    AgnielDeconfounder,
    ChannelDef,
    ClinicalContract,
    FastFrameEncoder,
    PilotSample,
    apply_platt,
    cfbh_selection,
    collision_zone_relevance_weights,
    empirical_martingale_voi,
    fit_platt_calibrator,
    fit_stage1_predictions,
    net_benefit_per_1k,
    two_phase_ht_mean_and_se,
)
from collision_zone.three_predicates_v2 import (
    FluidCollisionBackboneClassifierV2,
    bootstrap_mean_1d,
    clopper_pearson_ci,
    compute_calibration_metrics,
    compute_oof_stage1_v2,
    evaluate_stage2_multi_test_v2,
    paired_bootstrap_interaction,
    prior_odds_recalibrate_probabilities,
    run_sequential_discovery_v2,
    solve_citl_intercept,
    standardized_net_benefit,
)
from collision_zone.three_predicates_v3 import (
    PanelCostSchedule,
    apply_directional_voi_policy,
    compute_specialist_diagnostics,
    covariate_enriched_martingale_voi,
    direct_conditional_bayes_voi,
    evaluate_channel_billed_afa_v3,
    fit_matched_ungated_residual_stage2,
    kl_boltzmann_threshold_gate,
    net_benefit_with_cost_ledger_per_1k,
    normalized_beta_threshold_kernel,
    routing_priority_scores,
    run_discovery_v3,
    saerens_em_target_prevalence,
    unit_nb_ledger_vector,
)

BLOCKS_DIR = REPO_ROOT / "results" / "v2_3_blocks"
OUT_MERGED = REPO_ROOT / "results" / "paper2_v2_3_remediation_results.json"
CACHE_DIR = REPO_ROOT / "results" / "data_cache"


def _init_1thread_worker() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"


def paired_bootstrap_ledger_diff(
    u_a: np.ndarray,
    u_b: np.ndarray,
    *,
    n_boot: int = 10000,
    seed: int = 42,
) -> dict[str, float]:
    """Paired patient-level multinomial bootstrap on unit Net Benefit vectors (per 1,000)."""
    diff = np.asarray(u_a, dtype=float) - np.asarray(u_b, dtype=float)
    n = diff.size
    rng = np.random.default_rng(seed)
    probs = np.full(n, 1.0 / float(n))
    boots = np.empty(n_boot, dtype=float)
    for st in range(0, n_boot, 2500):
        en = min(n_boot, st + 2500)
        cnts = rng.multinomial(n, probs, size=en - st).astype(float) / float(n)
        boots[st:en] = cnts @ diff
    lo, hi = np.quantile(boots, [0.025, 0.975])
    k_le = int(np.sum(boots <= 0.0))
    k_ge = int(np.sum(boots >= 0.0))
    p_dh = min(1.0, 2.0 * float(min(k_le, k_ge) + 1) / float(n_boot + 1))
    return {
        "diff_mean": round(float(np.mean(diff)), 4),
        "se_boot": round(float(np.std(boots, ddof=1)), 4),
        "ci95_low": round(float(lo), 4),
        "ci95_high": round(float(hi), 4),
        "p_two_sided_dh": round(float(p_dh), 5),
        "prob_positive": round(float(np.mean(boots > 0.0)), 4),
        "prob_above_mcid_1_0": round(float(np.mean(boots > 1.0)), 4),
    }


def build_sepsis_channels_by_mode(
    df_sample: pd.DataFrame,
    *,
    summary_mode: str = "point_in_time_last",
    exclude_ids: set[str] | None = None,
) -> list[ChannelDef]:
    """Build 14 CPT laboratory channels using either single-draw `_last` or 6h `{last,max,min}`."""
    excl = exclude_ids or set()
    suffixes = ("last",) if summary_mode == "point_in_time_last" else ("last", "max", "min")
    channels: list[ChannelDef] = []
    for cid, (labs, cost) in CMS_CPT_VENIPUNCTURE_CHANNELS.items():
        if cid in excl:
            continue
        cols = [
            f"s2_{lab}_{st}"
            for lab in labs
            for st in suffixes
            if f"s2_{lab}_{st}" in df_sample.columns
        ]
        if cols:
            channels.append(ChannelDef(cid, tuple(cols), cost=float(cost)))
    return channels


# ===========================================================================
# BLOCK 1: Synthetic Mathematical & Methodological Suite (F3-F8, F14, Sec. 8)
# ===========================================================================


def _run_stress_rep(args: tuple[str, int]) -> dict[str, object]:
    scenario, r = args
    rng = np.random.default_rng(10000 + 500 * hash(scenario) % 10000 + r)
    n = 6000
    contract = ClinicalContract(cost_fp=1.0, cost_fn=9.0, target_precision=0.45, per_instance_query_cost=0.02)
    in_cz = rng.uniform(size=n) < 0.08
    b1 = in_cz.astype(float) + rng.normal(0.0, 0.08, size=n)
    b2 = rng.normal(0.0, 1.0, size=n)
    z_cz = rng.normal(0.0, 1.0, size=n)
    z_tn = rng.normal(0.0, 1.0, size=n)

    budget = 3.0
    if scenario == "correlated_nulls_r07":
        cov = np.full((4, 4), 0.70)
        np.fill_diagonal(cov, 1.0)
        null_mat = rng.multivariate_normal(np.zeros(4), cov, size=n)
        true_logit = np.where(in_cz, -2.8 + 1.80 * z_cz, -4.1 + 1.05 * z_tn)
        cols_extra = {f"null_{j+1}": null_mat[:, j] for j in range(4)}
        costs = {"ch_cz": 1.0, "ch_tn": 1.0, **{f"null_{j+1}": 1.0 for j in range(4)}}
        target_set = {"ch_cz"}
    elif scenario == "nonlinear_echo_nulls":
        true_logit = np.where(in_cz, -2.8 + 1.80 * z_cz, -4.1 + 1.05 * z_tn)
        cols_extra = {
            "null_1": np.sin(2.5 * b1) + 0.35 * rng.normal(size=n),
            "null_2": (b1 * b2) + 0.35 * rng.normal(size=n),
            "null_3": (b2**2 - 1.0) + 0.35 * rng.normal(size=n),
            "null_4": rng.normal(size=n),
        }
        costs = {"ch_cz": 1.0, "ch_tn": 1.0, **{f"null_{j+1}": 1.0 for j in range(4)}}
        target_set = {"ch_cz"}
    elif scenario == "weak_collision_signal":
        true_logit = np.where(in_cz, -2.8 + 0.95 * z_cz, -4.1 + 1.05 * z_tn)
        cols_extra = {f"null_{j+1}": rng.normal(size=n) for j in range(4)}
        costs = {"ch_cz": 1.0, "ch_tn": 1.0, **{f"null_{j+1}": 1.0 for j in range(4)}}
        target_set = {"ch_cz"}
    elif scenario == "suppressor_channel":
        u_sig = rng.normal(size=n)
        u_sup = rng.normal(size=n)
        z_cz = u_sig + u_sup  # noisy composite
        true_logit = np.where(in_cz, -2.8 + 1.85 * u_sig, -4.1 + 1.00 * z_tn)
        cols_extra = {
            "ch_supp": u_sup,
            "null_1": rng.normal(size=n),
            "null_2": rng.normal(size=n),
            "null_3": rng.normal(size=n),
        }
        costs = {"ch_cz": 1.0, "ch_tn": 1.0, "ch_supp": 1.0, "null_1": 1.0, "null_2": 1.0, "null_3": 1.0}
        target_set = {"ch_cz", "ch_supp"}
    elif scenario == "unequal_channel_costs":
        true_logit = np.where(in_cz, -2.8 + 1.85 * z_cz, -4.1 + 1.05 * z_tn)
        cols_extra = {f"null_{j+1}": rng.normal(size=n) for j in range(4)}
        costs = {"ch_cz": 1.4, "ch_tn": 0.8, "null_1": 0.6, "null_2": 0.9, "null_3": 1.2, "null_4": 1.5}
        budget = 2.6
        target_set = {"ch_cz"}
    elif scenario == "pairwise_synergy_cz":
        z_cz2 = rng.normal(size=n)
        # Mild main effect + strong multiplicative collision interaction
        true_logit = np.where(
            in_cz,
            -2.8 + 0.85 * z_cz + 0.85 * z_cz2 + 1.35 * (z_cz * z_cz2),
            -4.1 + 1.05 * z_tn,
        )
        cols_extra = {
            "ch_cz2": np.where(in_cz, z_cz2, rng.normal(size=n)),
            "null_1": rng.normal(size=n),
            "null_2": rng.normal(size=n),
            "null_3": rng.normal(size=n),
        }
        costs = {"ch_cz": 1.0, "ch_cz2": 1.0, "ch_tn": 1.0, "null_1": 1.0, "null_2": 1.0, "null_3": 1.0}
        target_set = {"ch_cz", "ch_cz2"}
    else:
        raise ValueError(scenario)

    y = rng.binomial(1, expit(true_logit))
    df = pd.DataFrame(
        {
            "b1": b1,
            "b2": b2,
            "ch_cz": np.where(in_cz, z_cz, rng.normal(size=n)),
            "ch_tn": np.where(~in_cz, z_tn, rng.normal(size=n)),
            **cols_extra,
        }
    )
    channels = [
        ChannelDef(cid, (cid,), cost=cval, is_null_stress=cid.startswith("null_"))
        for cid, cval in costs.items()
    ]
    tr, _, _ = make_stratified_3way_split(y, seed=21000 + r)
    oof_p = compute_oof_stage1_v2(df.iloc[tr][["b1", "b2"]], y[tr], n_folds=4, seed=22000 + r, n_jobs=1)

    d_glob = run_discovery_v3(
        df.iloc[tr], y[tr], ["b1", "b2"], channels, oof_p, contract,
        objective="global_cmi", evaluation_protocol="cross_fitted",
        pilot_fraction=0.25, budget=budget, n_jobs=1, seed=23000 + r,
    )
    d_cz_cf = run_discovery_v3(
        df.iloc[tr], y[tr], ["b1", "b2"], channels, oof_p, contract,
        objective="collision_zone", evaluation_protocol="cross_fitted",
        pilot_fraction=0.25, budget=budget, n_jobs=1, seed=23000 + r,
    )
    d_cz_sp = run_discovery_v3(
        df.iloc[tr], y[tr], ["b1", "b2"], channels, oof_p, contract,
        objective="collision_zone", evaluation_protocol="split_confirmation",
        pilot_fraction=0.25, budget=budget, n_jobs=1, seed=23000 + r,
    )

    # Also test step-1 B=cost(ch_cz) recovery vs decoy when budget allows only 1 primary signal
    b1_glob = d_glob.selected_ids[:1]
    b1_cz_cf = d_cz_cf.selected_ids[:1]
    b1_cz_sp = d_cz_sp.selected_ids[:1]

    return {
        "glob_hit_step1": int(bool(set(b1_glob) & target_set)),
        "cz_cf_hit_step1": int(bool(set(b1_cz_cf) & target_set)),
        "cz_sp_hit_step1": int(bool(set(b1_cz_sp) & target_set)),
        "glob_full_target": int(target_set.issubset(set(d_glob.selected_ids))),
        "cz_cf_full_target": int(target_set.issubset(set(d_cz_cf.selected_ids))),
        "cz_sp_full_target": int(target_set.issubset(set(d_cz_sp.selected_ids))),
        "glob_any_null": int(any(c.startswith("null_") for c in d_glob.selected_ids)),
        "cz_cf_any_null": int(any(c.startswith("null_") for c in d_cz_cf.selected_ids)),
        "cz_sp_any_null": int(any(c.startswith("null_") for c in d_cz_sp.selected_ids)),
    }


def _run_f14_synth_rep(r: int) -> dict[str, object]:
    contract = ClinicalContract(cost_fp=1.0, cost_fn=9.0, target_precision=0.45, per_instance_query_cost=0.02)
    rng = np.random.default_rng(31000 + r)
    n = 8000
    in_cz = rng.uniform(size=n) < 0.06
    z_cz = rng.normal(0.0, 1.0, size=n)
    z_tn = rng.normal(0.0, 1.0, size=n)
    true_logit = np.where(in_cz, -2.9 + 1.85 * z_cz, -4.2 + 1.08 * z_tn)
    y = rng.binomial(1, expit(true_logit))
    df = pd.DataFrame(
        {
            "b1": in_cz.astype(float) + rng.normal(0, 0.06, size=n),
            "b2": rng.normal(0, 1.0, size=n),
            "ch_cz": np.where(in_cz, z_cz, rng.normal(0, 1.0, size=n)),
            "ch_tn": np.where(~in_cz, z_tn, rng.normal(0, 1.0, size=n)),
            "null_1": rng.normal(0, 1.0, size=n),
            "null_2": rng.normal(0, 1.0, size=n),
            "null_3": rng.normal(0, 1.0, size=n),
            "null_4": rng.normal(0, 1.0, size=n),
        }
    )
    base_cols = ["b1", "b2"]
    channels = [
        ChannelDef("ch_cz", ("ch_cz",), cost=1.0),
        ChannelDef("ch_tn", ("ch_tn",), cost=1.0),
        ChannelDef("null_1", ("null_1",), cost=1.0, is_null_stress=True),
        ChannelDef("null_2", ("null_2",), cost=1.0, is_null_stress=True),
        ChannelDef("null_3", ("null_3",), cost=1.0, is_null_stress=True),
        ChannelDef("null_4", ("null_4",), cost=1.0, is_null_stress=True),
    ]
    tr, ca, te = make_stratified_3way_split(y, seed=32000 + r)
    oof_p = compute_oof_stage1_v2(df.iloc[tr][base_cols], y[tr], n_folds=4, seed=33000 + r, n_jobs=1)
    s1_cache = fit_stage1_predictions(
        df.iloc[tr], y[tr], df.iloc[ca], y[ca], [df.iloc[te]], base_cols, seed=34000 + r
    )

    # Precompute Stage-2 performance for each unique single-channel selection to avoid redundant fits
    s2_cache: dict[tuple[tuple[str, ...], str], float] = {}

    def _get_nb(sel: list[str], s2_mode: str) -> float:
        key = (tuple(sel), s2_mode)
        if key not in s2_cache:
            ev = evaluate_stage2_multi_test_v2(
                df.iloc[tr], y[tr], df.iloc[ca], y[ca], [(df.iloc[te], y[te])],
                base_cols, sel, oof_p, contract,
                stage2_mode=s2_mode, precomputed_stage1=s1_cache, n_jobs=1, seed=35000 + r,
            )[0]
            s2_cache[key] = float(ev["nb_voi_auto"])
        return s2_cache[key]

    out_r: dict[str, object] = {}
    for obj in ("global_cmi", "brier", "collision_zone", "direct_utility", "local_utility"):
        d_res = run_discovery_v3(
            df.iloc[tr], y[tr], base_cols, channels, oof_p, contract,
            objective=obj, evaluation_protocol="cross_fitted",
            pilot_fraction=0.20, budget=5.0, n_jobs=1, seed=36000 + r,
        )
        sel1 = d_res.selected_ids[:1]
        out_r[obj] = {
            "picked_cz": int(sel1 == ["ch_cz"]),
            "picked_tn": int(sel1 == ["ch_tn"]),
            "any_null_b5": int(any(c.startswith("null_") for c in d_res.selected_ids)),
            "nb_glob_s2": _get_nb(sel1, "global"),
            "nb_fluid_s2": _get_nb(sel1, "fluid_backbone"),
        }

    out_r["oracle_utility"] = {
        "picked_cz": 1,
        "picked_tn": 0,
        "any_null_b5": 0,
        "nb_glob_s2": _get_nb(["ch_cz"], "global"),
        "nb_fluid_s2": _get_nb(["ch_cz"], "fluid_backbone"),
    }
    return out_r


def _run_f8_voi_rep(r: int) -> dict[str, float]:
    """Compare Oracle Bayes VOI vs. Empirical Martingale VOI and 5 routing baselines."""
    rng = np.random.default_rng(41000 + r)
    n = 5000
    contract = ClinicalContract(cost_fp=1.0, cost_fn=9.0, target_precision=0.45, per_instance_query_cost=0.02)
    tau = contract.tau_star  # 0.10
    # Heterogeneous volatility: x2 controls whether z_s2 has high or near-zero conditional effect!
    x1 = rng.normal(0.0, 1.0, size=n)
    x2 = rng.binomial(1, 0.40, size=n).astype(float)
    z_s2 = rng.normal(0.0, 1.0, size=n)
    ell1_true = -2.6 + 0.85 * x1
    sigma_z = np.where(x2 > 0.5, 1.55, 0.10)
    # Solve exact true Stage-1 marginal posterior P1*(x1, x2) via 25-node Gauss-Hermite quadrature
    gh_x, gh_w = np.polynomial.hermite.hermgauss(25)
    z_nodes = np.sqrt(2.0) * gh_x[None, :]  # (1, 25)
    w_nodes = gh_w / np.sqrt(np.pi)  # (25,)
    p2_true_mat = expit(ell1_true[:, None] + sigma_z[:, None] * z_nodes)  # (N, 25)
    p1_true = p2_true_mat @ w_nodes  # exact E[P2* | x1, x2]
    oracle_voi = (contract.cost_fn + contract.cost_fp) * (
        (np.maximum(0.0, p2_true_mat - tau) @ w_nodes) - np.maximum(0.0, p1_true - tau)
    )

    p2_realized_true = expit(ell1_true + sigma_z * z_s2)
    y = rng.binomial(1, p2_realized_true)
    df = pd.DataFrame({"x1": x1, "x2": x2, "z_s2": z_s2})
    tr, ca, te = make_stratified_3way_split(y, seed=42000 + r)
    oof_p = compute_oof_stage1_v2(df.iloc[tr][["x1", "x2"]], y[tr], n_folds=4, seed=43000 + r, n_jobs=1)
    s1_cache = fit_stage1_predictions(
        df.iloc[tr], y[tr], df.iloc[ca], y[ca], [df.iloc[te]], ["x1", "x2"], seed=44000 + r
    )
    traces: list[dict[str, np.ndarray]] = []
    cal_tr: dict[str, np.ndarray] = {}
    evaluate_stage2_multi_test_v2(
        df.iloc[tr], y[tr], df.iloc[ca], y[ca], [(df.iloc[te], y[te])],
        ["x1", "x2"], ["z_s2"], oof_p, contract,
        stage2_mode="fluid_backbone", precomputed_stage1=s1_cache,
        out_routed_traces=traces, out_cal_trace=cal_tr, n_jobs=1, seed=45000 + r,
    )
    yt = y[te]
    p1_te = traces[0]["p1_te"]
    p2_te = traces[0]["p2_te"]
    edges = cal_tr["edges"]
    bq = cal_tr["bin_quantiles"]
    b_te = np.clip(np.digitize(p1_te, edges[1:-1], right=False), 0, 11)
    xb_ca = df.iloc[ca][["x1", "x2"]].to_numpy(dtype=float)
    xb_te = df.iloc[te][["x1", "x2"]].to_numpy(dtype=float)
    orc_te = oracle_voi[te]

    v_mart = routing_priority_scores(p1_te, b_te, bq, contract, mode="empirical_martingale_voi")
    v_unc = routing_priority_scores(p1_te, b_te, bq, contract, mode="uncentered_empirical_voi")
    v_cov = covariate_enriched_martingale_voi(
        cal_tr["p1_cal"], cal_tr["p2_cal"], xb_ca, p1_te, xb_te, contract, n_jobs=1, seed=46000 + r
    )
    v_dir = direct_conditional_bayes_voi(
        cal_tr["p1_cal"], cal_tr["p2_cal"], xb_ca, p1_te, xb_te, contract, n_jobs=1, seed=47000 + r
    )
    v_dist = routing_priority_scores(p1_te, b_te, bq, contract, mode="distance_to_threshold")
    v_ent = routing_priority_scores(p1_te, b_te, bq, contract, mode="shannon_entropy")

    q_orc = orc_te > contract.per_instance_query_cost
    q_mart = v_mart > contract.per_instance_query_cost
    q_unc = v_unc > contract.per_instance_query_cost
    q_cov = v_cov > contract.per_instance_query_cost
    q_dir = v_dir > contract.per_instance_query_cost
    # Match query budget of martingale VOI for rank-only heuristics (distance & entropy)
    k_q = max(1, int(np.sum(q_mart)))
    q_dist = np.zeros(yt.size, dtype=bool)
    q_dist[np.argsort(v_dist)[-k_q:]] = True
    q_ent = np.zeros(yt.size, dtype=bool)
    q_ent[np.argsort(v_ent)[-k_q:]] = True

    return {
        "spearman_martingale_vs_oracle": float(spearmanr(v_mart, orc_te).statistic),
        "spearman_covariate_vs_oracle": float(spearmanr(v_cov, orc_te).statistic),
        "spearman_uncentered_vs_oracle": float(spearmanr(v_unc, orc_te).statistic),
        "spearman_direct_vs_oracle": float(spearmanr(v_dir, orc_te).statistic),
        "mae_martingale_vs_oracle": float(np.mean(np.abs(v_mart - orc_te))),
        "mae_covariate_vs_oracle": float(np.mean(np.abs(v_cov - orc_te))),
        "mae_uncentered_vs_oracle": float(np.mean(np.abs(v_unc - orc_te))),
        "nb_oracle_voi": net_benefit_per_1k(yt, np.where(q_orc, p2_te, p1_te), q_orc, contract),
        "nb_martingale_voi": net_benefit_per_1k(yt, np.where(q_mart, p2_te, p1_te), q_mart, contract),
        "nb_covariate_voi": net_benefit_per_1k(yt, np.where(q_cov, p2_te, p1_te), q_cov, contract),
        "nb_uncentered_voi": net_benefit_per_1k(yt, np.where(q_unc, p2_te, p1_te), q_unc, contract),
        "nb_direct_reg_voi": net_benefit_per_1k(yt, np.where(q_dir, p2_te, p1_te), q_dir, contract),
        "nb_dist_matched_q": net_benefit_per_1k(yt, np.where(q_dist, p2_te, p1_te), q_dist, contract),
        "nb_entropy_matched_q": net_benefit_per_1k(yt, np.where(q_ent, p2_te, p1_te), q_ent, contract),
    }


def _run_regime_cell_rep(args: tuple[float, float, str, str, int]) -> dict[str, float | str]:
    cz_mass, decoy_str, het_mode, tau_regime, r = args
    rng = np.random.default_rng(
        50000 + int(cz_mass * 1000) + int(decoy_str * 100) + (0 if het_mode == "homogeneous" else 500) + r
    )
    n = 4000
    # Set contract threshold: above_prevalence (tau* = 0.10, C_FN=9) vs near_prevalence (tau* = 0.04, C_FN=24)
    c_fn = 9.0 if tau_regime == "above_prevalence" else 24.0
    contract = ClinicalContract(cost_fp=1.0, cost_fn=c_fn, target_precision=0.40, per_instance_query_cost=0.02)
    in_cz = rng.uniform(size=n) < cz_mass
    z_sig = rng.normal(0.0, 1.0, size=n)
    z_decoy = rng.normal(0.0, 1.0, size=n)

    if het_mode == "localized":
        true_logit = np.where(in_cz, -2.7 + 1.80 * z_sig, -4.2 + decoy_str * z_decoy)
        ch_sig_obs = np.where(in_cz, z_sig, rng.normal(0.0, 1.0, size=n))
    else:
        # Homogeneous global signal: ch_sig shifts log-odds uniformly everywhere, competing with subthreshold decoy
        true_logit = np.where(in_cz, -2.7 + 1.15 * z_sig, -4.2 + 1.15 * z_sig + decoy_str * z_decoy)
        ch_sig_obs = z_sig

    y = rng.binomial(1, expit(true_logit))
    df = pd.DataFrame(
        {
            "b1": in_cz.astype(float) + rng.normal(0, 0.08, size=n),
            "b2": rng.normal(0, 1.0, size=n),
            "ch_sig": ch_sig_obs,
            "ch_decoy": np.where(~in_cz, z_decoy, rng.normal(0, 1.0, size=n)),
            "null_1": rng.normal(0, 1.0, size=n),
            "null_2": rng.normal(0, 1.0, size=n),
        }
    )
    channels = [
        ChannelDef("ch_sig", ("ch_sig",), cost=1.0),
        ChannelDef("ch_decoy", ("ch_decoy",), cost=1.0),
        ChannelDef("null_1", ("null_1",), cost=1.0, is_null_stress=True),
        ChannelDef("null_2", ("null_2",), cost=1.0, is_null_stress=True),
    ]
    tr, ca, te = make_stratified_3way_split(y, seed=55000 + r)
    oof_p = compute_oof_stage1_v2(df.iloc[tr][["b1", "b2"]], y[tr], n_folds=4, seed=56000 + r, n_jobs=1)
    d_glob = run_discovery_v3(
        df.iloc[tr], y[tr], ["b1", "b2"], channels, oof_p, contract,
        objective="global_cmi", pilot_fraction=0.25, budget=1.0, n_jobs=1, seed=57000 + r,
    )
    d_cz = run_discovery_v3(
        df.iloc[tr], y[tr], ["b1", "b2"], channels, oof_p, contract,
        objective="collision_zone", pilot_fraction=0.25, budget=1.0, n_jobs=1, seed=57000 + r,
    )
    s1_cache = fit_stage1_predictions(
        df.iloc[tr], y[tr], df.iloc[ca], y[ca], [df.iloc[te]], ["b1", "b2"], seed=58000 + r
    )
    ev_g = evaluate_stage2_multi_test_v2(
        df.iloc[tr], y[tr], df.iloc[ca], y[ca], [(df.iloc[te], y[te])],
        ["b1", "b2"], d_glob.selected_ids[:1], oof_p, contract,
        stage2_mode="global", precomputed_stage1=s1_cache, n_jobs=1, seed=59000 + r,
    )[0]
    ev_c = evaluate_stage2_multi_test_v2(
        df.iloc[tr], y[tr], df.iloc[ca], y[ca], [(df.iloc[te], y[te])],
        ["b1", "b2"], d_cz.selected_ids[:1], oof_p, contract,
        stage2_mode="fluid_backbone", precomputed_stage1=s1_cache, n_jobs=1, seed=59000 + r,
    )[0]
    return {
        "prev": float(np.mean(y)),
        "glob_hit": float(d_glob.selected_ids[:1] == ["ch_sig"]),
        "cz_hit": float(d_cz.selected_ids[:1] == ["ch_sig"]),
        "nb_glob": float(ev_g["nb_voi_auto"]),
        "nb_cz": float(ev_c["nb_voi_auto"]),
        "beta_cz": float(ev_c["learned_beta_cz"]),
    }


def run_block1_synthetic() -> dict[str, object]:
    t0 = time.time()
    print("[Block 1] Starting Synthetic Mathematical & Methodological Suite...")

    # 1. F3, F4, F6 exact numerical identities
    p_grid = np.linspace(0.001, 0.999, 1000)
    max_f3_diff = max(
        float(np.max(np.abs(normalized_beta_threshold_kernel(p_grid, tau, k) - kl_boltzmann_threshold_gate(p_grid, tau, k))))
        for tau in (0.04, 0.0625, 0.10, 0.25)
        for k in (4.0, 12.0, 20.0, 32.0)
    )
    rng_v = np.random.default_rng(42)
    psi_srs = rng_v.normal(0.05, 0.20, size=200)
    pil_srs = PilotSample(
        indices=np.arange(200),
        strata=np.zeros(200, dtype=int),
        ht_weights=np.full(200, 1000.0 / 200.0),
        pop_counts={0: 1000},
        sample_counts={0: 200},
        n_total=1000,
        high_signal_fp_share_of_controls=0.1,
    )
    _, se_srs = two_phase_ht_mean_and_se(psi_srs, pil_srs)
    s2_srs = float(np.var(psi_srs, ddof=1))
    f6_table = []
    for p0 in (0.01, 0.001, 0.0001):
        p_up = 1.5 * p0
        dkl = float(p_up * np.log(p_up / p0) + (1.0 - p_up) * np.log((1.0 - p_up) / (1.0 - p0)))
        f6_table.append(
            {
                "p_baseline": p0,
                "p_updated_1_5x": p_up,
                "logloss_curvature_w": round(float(1.0 / (p0 * (1.0 - p0))), 2),
                "exact_bernoulli_kl_gain": round(dkl, 7),
            }
        )

    identities = {
        "f3_max_abs_diff_beta_kernel_vs_kl_gate": max_f3_diff,
        "f4_srs_implemented_se_minus_s_over_sqrt_n": abs(se_srs - float(np.sqrt(s2_srs / 200.0))),
        "f4_printed_eq7_w_squared_variance_inflation_at_20pct_pilot": 2.0 - (200.0 / 1000.0),
        "f6_curvature_vs_proportional_gain_examples": f6_table,
    }

    # 2. F5 6-scenario stress suite (200 reps/scenario)
    scenarios = [
        "correlated_nulls_r07",
        "nonlinear_echo_nulls",
        "weak_collision_signal",
        "suppressor_channel",
        "unequal_channel_costs",
        "pairwise_synergy_cz",
    ]
    n_stress = 200
    stress_out: dict[str, object] = {}
    with ProcessPoolExecutor(max_workers=N_JOBS, initializer=_init_1thread_worker) as pool:
        for sc in scenarios:
            ts = time.time()
            rows = list(pool.map(_run_stress_rep, [(sc, r) for r in range(n_stress)], chunksize=5))
            stress_out[sc] = {
                "n_reps": n_stress,
                "glob_cf_step1_recovery": clopper_pearson_ci(sum(int(d["glob_hit_step1"]) for d in rows), n_stress),
                "cz_cf_step1_recovery": clopper_pearson_ci(sum(int(d["cz_cf_hit_step1"]) for d in rows), n_stress),
                "cz_split_conf_step1_recovery": clopper_pearson_ci(sum(int(d["cz_sp_hit_step1"]) for d in rows), n_stress),
                "glob_cf_full_target": clopper_pearson_ci(sum(int(d["glob_full_target"]) for d in rows), n_stress),
                "cz_cf_full_target": clopper_pearson_ci(sum(int(d["cz_cf_full_target"]) for d in rows), n_stress),
                "cz_split_conf_full_target": clopper_pearson_ci(sum(int(d["cz_sp_full_target"]) for d in rows), n_stress),
                "glob_cf_seq_null_fwer": clopper_pearson_ci(sum(int(d["glob_any_null"]) for d in rows), n_stress),
                "cz_cf_seq_null_fwer": clopper_pearson_ci(sum(int(d["cz_cf_any_null"]) for d in rows), n_stress),
                "cz_split_conf_seq_null_fwer": clopper_pearson_ci(sum(int(d["cz_sp_any_null"]) for d in rows), n_stress),
            }
            print(
                f"  [F5 Stress: {sc}] ({time.time()-ts:.1f}s) "
                f"CZ-CF={stress_out[sc]['cz_cf_step1_recovery']['rate']*100:.1f}% (FWER {stress_out[sc]['cz_cf_seq_null_fwer']['rate']*100:.1f}%) | "
                f"CZ-Split={stress_out[sc]['cz_split_conf_step1_recovery']['rate']*100:.1f}% (FWER {stress_out[sc]['cz_split_conf_seq_null_fwer']['rate']*100:.1f}%)"
            )

    # 3. F14 Decisive Synthetic 6-Arm Discovery x 2-Backbone Factorial (300 reps)
    n_f14 = 300
    t_f14 = time.time()
    with ProcessPoolExecutor(max_workers=N_JOBS, initializer=_init_1thread_worker) as pool:
        f14_rows = list(pool.map(_run_f14_synth_rep, range(n_f14), chunksize=5))
    f14_summary: dict[str, object] = {}
    for arm in ("global_cmi", "brier", "collision_zone", "direct_utility", "local_utility", "oracle_utility"):
        k_cz = sum(int(r[arm]["picked_cz"]) for r in f14_rows)  # type: ignore[index]
        k_tn = sum(int(r[arm]["picked_tn"]) for r in f14_rows)  # type: ignore[index]
        k_nl = sum(int(r[arm]["any_null_b5"]) for r in f14_rows)  # type: ignore[index]
        nb_g = np.array([float(r[arm]["nb_glob_s2"]) for r in f14_rows])  # type: ignore[index]
        nb_f = np.array([float(r[arm]["nb_fluid_s2"]) for r in f14_rows])  # type: ignore[index]
        f14_summary[arm] = {
            "tiebreaker_recovery": clopper_pearson_ci(k_cz, n_f14),
            "decoy_hijack": clopper_pearson_ci(k_tn, n_f14),
            "seq_null_fwer": clopper_pearson_ci(k_nl, n_f14),
            "nb_global_s2": _agg(nb_g),
            "nb_fluid_s2": _agg(nb_f),
        }
    nb_cz_fl = np.array([float(r["collision_zone"]["nb_fluid_s2"]) for r in f14_rows])  # type: ignore[index]
    nb_du_fl = np.array([float(r["direct_utility"]["nb_fluid_s2"]) for r in f14_rows])  # type: ignore[index]
    nb_lu_fl = np.array([float(r["local_utility"]["nb_fluid_s2"]) for r in f14_rows])  # type: ignore[index]
    nb_gl_gl = np.array([float(r["global_cmi"]["nb_glob_s2"]) for r in f14_rows])  # type: ignore[index]
    f14_summary["paired_contrasts"] = {
        "cz_beta_fluid_vs_direct_utility_fluid": bootstrap_mean_1d(nb_cz_fl - nb_du_fl, n_boot=10000, seed=11),
        "cz_beta_fluid_vs_local_utility_fluid": bootstrap_mean_1d(nb_cz_fl - nb_lu_fl, n_boot=10000, seed=12),
        "cz_beta_fluid_vs_global_cmi_global": bootstrap_mean_1d(nb_cz_fl - nb_gl_gl, n_boot=10000, seed=13),
    }
    print(
        f"  [F14 Synthetic Factorial] ({time.time()-t_f14:.1f}s) "
        f"Glob={f14_summary['global_cmi']['tiebreaker_recovery']['rate']*100:.1f}%, "
        f"Brier={f14_summary['brier']['tiebreaker_recovery']['rate']*100:.1f}%, "
        f"DirectUtil={f14_summary['direct_utility']['tiebreaker_recovery']['rate']*100:.1f}%, "
        f"LocalUtil={f14_summary['local_utility']['tiebreaker_recovery']['rate']*100:.1f}%, "
        f"BetaCZ={f14_summary['collision_zone']['tiebreaker_recovery']['rate']*100:.1f}%"
    )

    # 4. F8 Oracle vs Estimated VOI Calibration & Routing (200 reps)
    n_f8 = 200
    t_f8 = time.time()
    with ProcessPoolExecutor(max_workers=N_JOBS, initializer=_init_1thread_worker) as pool:
        f8_rows = list(pool.map(_run_f8_voi_rep, range(n_f8), chunksize=5))
    f8_summary = {k: _agg([r[k] for r in f8_rows]) for k in f8_rows[0]}
    print(
        f"  [F8 Oracle VOI Audit] ({time.time()-t_f8:.1f}s) "
        f"Spearman(Mart,Orc)={f8_summary['spearman_martingale_vs_oracle']['mean']:.3f}, "
        f"Spearman(Cov,Orc)={f8_summary['spearman_covariate_vs_oracle']['mean']:.3f}, "
        f"NB Mart={f8_summary['nb_martingale_voi']['mean']:.2f} vs Dist={f8_summary['nb_dist_matched_q']['mean']:.2f}"
    )

    # 5. Section 8 36-cell Regime Map (40 reps/cell)
    t_reg = time.time()
    n_reg_reps = 40
    regime_cells: list[dict[str, object]] = []
    grid_specs = [
        (cm, ds, hm, tr_reg)
        for cm in (0.04, 0.12, 0.25)
        for ds in (0.0, 0.95, 1.45)
        for hm in ("homogeneous", "localized")
        for tr_reg in ("near_prevalence", "above_prevalence")
    ]
    with ProcessPoolExecutor(max_workers=N_JOBS, initializer=_init_1thread_worker) as pool:
        all_tasks = [
            (cm, ds, hm, tr_reg, r)
            for (cm, ds, hm, tr_reg) in grid_specs
            for r in range(n_reg_reps)
        ]
        all_res = list(pool.map(_run_regime_cell_rep, all_tasks, chunksize=8))

    cursor = 0
    for cm, ds, hm, tr_reg in grid_specs:
        sub = all_res[cursor : cursor + n_reg_reps]
        cursor += n_reg_reps
        nb_g = np.array([float(x["nb_glob"]) for x in sub])
        nb_c = np.array([float(x["nb_cz"]) for x in sub])
        regime_cells.append(
            {
                "cz_mass": cm,
                "decoy_strength": ds,
                "heterogeneity": hm,
                "threshold_regime": tr_reg,
                "mean_prevalence": round(float(np.mean([float(x["prev"]) for x in sub])), 4),
                "glob_recovery_rate": round(float(np.mean([float(x["glob_hit"]) for x in sub])), 3),
                "cz_recovery_rate": round(float(np.mean([float(x["cz_hit"]) for x in sub])), 3),
                "nb_global_cmi": round(float(np.mean(nb_g)), 3),
                "nb_cz_fluid": round(float(np.mean(nb_c)), 3),
                "delta_nb_cz_minus_glob": round(float(np.mean(nb_c - nb_g)), 3),
                "delta_nb_se": round(float(np.std(nb_c - nb_g, ddof=1) / np.sqrt(n_reg_reps)), 3),
                "mean_beta_cz": round(float(np.mean([float(x["beta_cz"]) for x in sub])), 3),
            }
        )
    print(f"  [Sec. 8 Regime Map] Completed 36 cells x {n_reg_reps} reps in {time.time()-t_reg:.1f}s.")

    out = {
        "elapsed_sec": round(time.time() - t0, 2),
        "f3_f4_f6_identities": identities,
        "f5_stress_suite": stress_out,
        "f14_synthetic_discovery_factorial": f14_summary,
        "f8_oracle_voi_calibration_and_routing": f8_summary,
        "sec8_regime_map_36_cells": regime_cells,
    }
    BLOCKS_DIR.mkdir(parents=True, exist_ok=True)
    (BLOCKS_DIR / "block1_synthetic.json").write_text(json.dumps(out, indent=2))
    print(f"[Block 1] Saved checkpoint in {time.time()-t0:.1f}s.")
    return out


# ===========================================================================
# Core Repaired Sepsis Cohort Evaluator (Explicit PanelCostSchedule Ledger)
# ===========================================================================


def _fit_single_panel_stage2_split(
    df_tr: pd.DataFrame,
    y_tr: np.ndarray,
    df_ca: pd.DataFrame,
    y_ca: np.ndarray,
    df_te: pd.DataFrame,
    base_cols: list[str],
    selected_cols: list[str],
    oof_p_tr: np.ndarray,
    s1_cache: tuple[np.ndarray, list[np.ndarray]],
    contract: ClinicalContract,
    *,
    stage2_mode: str = "fluid_backbone",
    deconfound_stage2: bool = False,
    kappa_max_s2: float = 20.0,
    min_cz_ess: float = 240.0,
    n_jobs: int = N_JOBS,
    seed: int = 42,
) -> dict[str, object]:
    """Fit Stage-2 model on a specified column panel and return cal/test posteriors + F7 diagnostics."""
    p1_cal = s1_cache[0]
    p1_te = s1_cache[1][0]
    if not selected_cols:
        return {
            "p1_cal": p1_cal,
            "p2_cal": p1_cal.copy(),
            "p1_te": p1_te,
            "p2_te": p1_te.copy(),
            "diag": {
                "beta_cz": 0.0,
                "rms_gated_log_odds_r_cz": 0.0,
                "rms_cz_zone_r_cz": 0.0,
                "decision_flip_rate_all": 0.0,
                "decision_flip_rate_cz": 0.0,
            },
        }

    enc_b = FastFrameEncoder().fit(df_tr[base_cols])
    xb_tr = enc_b.transform(df_tr[base_cols])
    xb_ca = enc_b.transform(df_ca[base_cols])
    xb_te = enc_b.transform(df_te[base_cols])

    if deconfound_stage2:
        deconf = AgnielDeconfounder().fit(df_tr[base_cols], df_tr[selected_cols])
        s2_tr_df = deconf.transform(df_tr[base_cols], df_tr[selected_cols], seed=seed + 31)
        s2_ca_df = deconf.transform(df_ca[base_cols], df_ca[selected_cols], seed=seed + 32)
        s2_te_df = deconf.transform(df_te[base_cols], df_te[selected_cols], seed=seed + 41)
        enc_z = FastFrameEncoder(include_missing_indicators=False).fit(s2_tr_df)
    else:
        s2_tr_df = df_tr[selected_cols]
        s2_ca_df = df_ca[selected_cols]
        s2_te_df = df_te[selected_cols]
        enc_z = FastFrameEncoder(include_missing_indicators=True).fit(s2_tr_df)

    z_tr = enc_z.transform(s2_tr_df)
    z_ca = enc_z.transform(s2_ca_df)
    z_te = enc_z.transform(s2_te_df)
    x2_tr = np.hstack([xb_tr, z_tr])
    x2_ca = np.hstack([xb_ca, z_ca])
    x2_te = np.hstack([xb_te, z_te])

    diag: dict[str, float] = {
        "beta_cz": 0.0,
        "rms_gated_log_odds_r_cz": 0.0,
        "rms_cz_zone_r_cz": 0.0,
        "decision_flip_rate_all": 0.0,
        "decision_flip_rate_cz": 0.0,
    }

    if stage2_mode == "fluid_backbone":
        ell1_tr = logit(np.clip(oof_p_tr, 1e-5, 1.0 - 1e-5))
        clf_f = FluidCollisionBackboneClassifierV2(
            tau_star=contract.tau_star,
            kappa_max=kappa_max_s2,
            min_cz_ess=min_cz_ess,
            n_jobs=n_jobs,
            seed=seed + 7,
        ).fit(x2_tr, z_tr, y_tr, ell1_tr)
        raw_ca = clf_f.decision_function(x2_ca, z_ca, logit(p1_cal))
        raw_te = clf_f.decision_function(x2_te, z_te, logit(p1_te))
        platt2 = fit_platt_calibrator(raw_ca, y_ca)
        diag = compute_specialist_diagnostics(clf_f, x2_te, z_te, logit(p1_te))
    elif stage2_mode == "ungated_residual":
        ell1_tr = logit(np.clip(oof_p_tr, 1e-5, 1.0 - 1e-5))
        (raw_ca, raw_te), dg_u = fit_matched_ungated_residual_stage2(
            x2_tr, z_tr, y_tr, ell1_tr,
            [x2_ca, x2_te], [z_ca, z_te], [logit(p1_cal), logit(p1_te)],
            res_leaves=8, res_trees=50, n_jobs=n_jobs, seed=seed + 7,
        )
        platt2 = fit_platt_calibrator(raw_ca, y_ca)
        diag["beta_cz"] = dg_u["beta_ungated"]
        diag["rms_gated_log_odds_r_cz"] = dg_u["rms_ungated_log_odds"]
    elif stage2_mode == "naive_downweighted":
        p0_tr = np.clip(oof_p_tr, 1e-5, 1.0 - 1e-5)
        w_cz = collision_zone_relevance_weights(p0_tr, contract.tau_star, kappa=24.0, floor=0.05)
        clf_n = lgb.LGBMClassifier(
            n_estimators=140, learning_rate=0.05, num_leaves=31, min_child_samples=30,
            max_delta_step=1.0, subsample=0.85, colsample_bytree=0.85,
            random_state=seed + 7, n_jobs=n_jobs, verbose=-1,
        )
        clf_n.fit(x2_tr, y_tr, sample_weight=w_cz, init_score=logit(p0_tr))
        raw_ca = logit(p1_cal) + clf_n.predict(x2_ca, raw_score=True)
        raw_te = logit(p1_te) + clf_n.predict(x2_te, raw_score=True)
        platt2 = fit_platt_calibrator(raw_ca, y_ca)
    elif stage2_mode == "global":
        clf_g = lgb.LGBMClassifier(
            n_estimators=140, learning_rate=0.05, num_leaves=31, min_child_samples=30,
            max_delta_step=1.0, subsample=0.85, colsample_bytree=0.85,
            random_state=seed + 7, n_jobs=n_jobs, verbose=-1,
        )
        clf_g.fit(x2_tr, y_tr)
        raw_ca = clf_g.predict(x2_ca, raw_score=True)
        raw_te = clf_g.predict(x2_te, raw_score=True)
        platt2 = fit_platt_calibrator(raw_ca, y_ca)
    else:
        raise ValueError(stage2_mode)

    p2_cal = np.clip(apply_platt(raw_ca, platt2), 1e-6, 1.0 - 1e-6)
    p2_te = np.clip(apply_platt(raw_te, platt2), 1e-6, 1.0 - 1e-6)
    return {
        "p1_cal": p1_cal,
        "p2_cal": p2_cal,
        "p1_te": p1_te,
        "p2_te": p2_te,
        "xb_cal": xb_ca,
        "xb_te": xb_te,
        "diag": diag,
    }


def compute_bagged_voi_from_split_posteriors(
    y_te: np.ndarray,
    p1_ens_te: np.ndarray,
    p2_ens_te: np.ndarray,
    p1_cal_cat: np.ndarray,
    p2_cal_cat: np.ndarray,
    panel_cost: float,
    contract: ClinicalContract,
) -> dict[str, object]:
    """Compute binary bagged VOI policy using exact panel_cost (in C_FP units)."""
    prev = float(np.mean(y_te))
    if panel_cost <= 0.0:
        q = np.zeros(y_te.size, dtype=bool)
        c_vec = np.zeros(y_te.size, dtype=float)
        nb = net_benefit_with_cost_ledger_per_1k(y_te, p1_ens_te, c_vec, contract)
        return {
            "p_routed": p1_ens_te,
            "q_mask": q,
            "voi_te": np.zeros(y_te.size, dtype=float),
            "cost_vec": c_vec,
            "unit_nb": unit_nb_ledger_vector(y_te, p1_ens_te, c_vec, contract),
            "nb_per_1k": round(nb, 4),
            "snb": round(standardized_net_benefit(nb, prev), 4),
            "query_rate": 0.0,
            "mean_cost_cfp": 0.0,
            "panel_cost_cfp": 0.0,
            "auroc_routed": round(float(roc_auc_score(y_te, p1_ens_te)), 4),
            "auprc_routed": round(float(average_precision_score(y_te, p1_ens_te)), 4),
            "edges": np.linspace(0.0, 1.0, 13),
            "bin_quantiles": np.zeros((12, 33), dtype=float),
        }

    d_cal = logit(np.clip(p2_cal_cat, 1e-6, 1.0 - 1e-6)) - logit(np.clip(p1_cal_cat, 1e-6, 1.0 - 1e-6))
    edges = np.quantile(p1_cal_cat, np.linspace(0.0, 1.0, 13))
    edges[0], edges[-1] = 0.0, 1.0
    b_cal = np.clip(np.digitize(p1_cal_cat, edges[1:-1], right=False), 0, 11)
    qs = np.linspace(0.02, 0.98, 33)
    gq = np.quantile(d_cal, qs)
    bq = np.array(
        [np.quantile(d_cal[b_cal == b], qs) if int(np.sum(b_cal == b)) >= 15 else gq for b in range(12)]
    )
    b_te = np.clip(np.digitize(p1_ens_te, edges[1:-1], right=False), 0, 11)
    voi_te = empirical_martingale_voi(p1_ens_te, b_te, bq, contract)
    q = voi_te > float(panel_cost)
    p_routed = np.where(q, p2_ens_te, p1_ens_te)
    c_vec = float(panel_cost) * q.astype(float)
    nb = net_benefit_with_cost_ledger_per_1k(y_te, p_routed, c_vec, contract)
    return {
        "p_routed": p_routed,
        "q_mask": q,
        "voi_te": voi_te,
        "cost_vec": c_vec,
        "unit_nb": unit_nb_ledger_vector(y_te, p_routed, c_vec, contract),
        "nb_per_1k": round(nb, 4),
        "snb": round(standardized_net_benefit(nb, prev), 4),
        "query_rate": round(float(np.mean(q)), 4),
        "mean_cost_cfp": round(float(np.mean(c_vec)), 5),
        "panel_cost_cfp": round(float(panel_cost), 5),
        "auroc_routed": round(float(roc_auc_score(y_te, p_routed)), 4),
        "auprc_routed": round(float(average_precision_score(y_te, p_routed)), 4),
        "edges": edges,
        "bin_quantiles": bq,
    }


def _clinical_counts_row(
    y: np.ndarray,
    p_routed: np.ndarray,
    q_mask: np.ndarray,
    cost_vec: np.ndarray,
    tau: float,
) -> dict[str, float]:
    yv = np.asarray(y, dtype=int)
    act = p_routed >= tau
    n = float(yv.size)
    pos = max(1.0, float(np.sum(yv == 1)))
    tp = float(np.sum(act & (yv == 1)))
    fp = float(np.sum(act & (yv == 0)))
    q_cnt = float(np.sum(q_mask))
    odds = tau / (1.0 - tau)
    c_fn_tau = (1.0 - tau) / tau
    nb_tau = 1000.0 * ((tp / n) - odds * (fp / n) - float(np.mean(cost_vec)) / c_fn_tau)
    return {
        "tau": round(tau, 4),
        "tp_count": int(tp),
        "fp_count": int(fp),
        "queried_count": int(q_cnt),
        "tp_per_1k": round(1000.0 * tp / n, 2),
        "fp_per_1k": round(1000.0 * fp / n, 2),
        "queries_per_1k": round(1000.0 * q_cnt / n, 2),
        "sensitivity_pct": round(100.0 * tp / pos, 2),
        "ppv_pct": round(100.0 * tp / max(1.0, tp + fp), 2),
        "alert_rate_pct": round(100.0 * (tp + fp) / n, 2),
        "mean_cost_cfp": round(float(np.mean(cost_vec)), 5),
        "nb_ledger_per_1k": round(nb_tau, 3),
    }


# ===========================================================================
# BLOCK 2: Sepsis Primary Evaluation, Estimands, Factorial & Policy Repairs
# ===========================================================================


def _evaluate_sepsis_mode_full(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    base_cols: list[str],
    *,
    summary_mode: str,
    n_splits: int = 10,
    run_full_factorial_and_afa: bool = True,
    deconfound_stage2: bool = False,
) -> dict[str, object]:
    t0 = time.time()
    channels = build_sepsis_channels_by_mode(df_a, summary_mode=summary_mode)
    cost_map = {c.channel_id: float(c.cost) for c in channels}
    ch_cols_map = {c.channel_id: list(c.columns) for c in channels}
    sched = PanelCostSchedule(venipuncture_cost=0.010, abg_extra_procedure_cost=0.010, unit_assay_scale=0.020)
    contract = ClinicalContract(cost_fp=1.0, cost_fn=15.0, target_precision=0.30, per_instance_query_cost=0.03)

    y_a = df_a["y_sepsis_future"].to_numpy(dtype=int)
    y_b = df_b["y_sepsis_future"].to_numpy(dtype=int)
    n_b = y_b.size

    # Step 1: Full-source BIDMC locked discovery panels (F1 Common-Panel Bagged VOI + F14 5 objectives + F5 split_confirmation)
    oof_full_a = compute_oof_stage1_v2(df_a[base_cols], y_a, n_folds=4, seed=600, n_jobs=N_JOBS)
    objectives = (
        ["global_cmi", "brier", "collision_zone", "direct_utility", "local_utility"]
        if run_full_factorial_and_afa
        else ["global_cmi", "collision_zone"]
    )
    common_panels: dict[str, list[str]] = {}
    common_panel_costs: dict[str, float] = {}
    for obj in objectives:
        disc = run_discovery_v3(
            df_a, y_a, base_cols, channels, oof_full_a, contract,
            objective=obj, evaluation_protocol="cross_fitted",
            pilot_fraction=0.20, budget=3.2, deconfound_stage2=deconfound_stage2,
            n_jobs=N_JOBS, seed=900,
        )
        common_panels[obj] = list(disc.selected_ids)
        common_panel_costs[obj] = round(sched.panel_cost(disc.selected_ids, cost_map), 5)

    disc_cz_split_conf = run_discovery_v3(
        df_a, y_a, base_cols, channels, oof_full_a, contract,
        objective="collision_zone", evaluation_protocol="split_confirmation",
        pilot_fraction=0.20, budget=3.2, deconfound_stage2=deconfound_stage2,
        n_jobs=N_JOBS, seed=900,
    )
    common_panels["collision_zone_split_conf"] = list(disc_cz_split_conf.selected_ids)
    common_panel_costs["collision_zone_split_conf"] = round(
        sched.panel_cost(disc_cz_split_conf.selected_ids, cost_map), 5
    )
    print(f"  [{summary_mode}] Locked Common Panels:", common_panels, "Costs:", common_panel_costs)

    # Precompute 10 source splits (seeds 700..709)
    splits_data: list[dict[str, object]] = []
    for s in range(n_splits):
        tr, ca, te = make_stratified_3way_split(y_a, seed=700 + s)
        oof_p = compute_oof_stage1_v2(df_a.iloc[tr][base_cols], y_a[tr], n_folds=4, seed=800 + s, n_jobs=N_JOBS)
        s1_b = fit_stage1_predictions(
            df_a.iloc[tr], y_a[tr], df_a.iloc[ca], y_a[ca], [df_b], base_cols, seed=1500 + s
        )
        splits_data.append({"s": s, "tr": tr, "ca": ca, "te": te, "oof_p": oof_p, "s1_b": s1_b})

    p1_ens_b = np.mean([sp["s1_b"][1][0] for sp in splits_data], axis=0)  # type: ignore[index]
    p1_cal_cat = np.concatenate([sp["s1_b"][0] for sp in splits_data])  # type: ignore[index]

    # Step 2: Evaluate Common-Panel Bagged VOI across all (objective x backbone) combinations!
    backbones = (
        ["global", "ungated_residual", "fluid_backbone"]
        if run_full_factorial_and_afa
        else ["global", "fluid_backbone"]
    )
    factorial_bagged: dict[str, dict[str, object]] = {}
    stored_bagged: dict[str, dict[str, object]] = {}
    specialist_diags_by_arm: dict[str, list[dict[str, float]]] = {}

    panel_keys_to_eval = list(common_panels.keys())
    for pkey in panel_keys_to_eval:
        sel_ids = common_panels[pkey]
        scols = [col for cid in sel_ids for col in ch_cols_map[cid]]
        pcost = sched.panel_cost(sel_ids, cost_map)
        s2_modes = backbones if pkey in objectives else ["fluid_backbone"]
        for s2m in s2_modes:
            arm_key = f"{pkey}__{s2m}"
            p2_cal_list: list[np.ndarray] = []
            p2_te_list: list[np.ndarray] = []
            diags: list[dict[str, float]] = []
            last_xb_ca: np.ndarray | None = None
            last_xb_te: np.ndarray | None = None
            for sp in splits_data:
                s = int(sp["s"])  # type: ignore[arg-type]
                tr, ca = sp["tr"], sp["ca"]  # type: ignore[assignment]
                fit_out = _fit_single_panel_stage2_split(
                    df_a.iloc[tr], y_a[tr], df_a.iloc[ca], y_a[ca], df_b,
                    base_cols, scols, sp["oof_p"], sp["s1_b"], contract,  # type: ignore[arg-type]
                    stage2_mode=s2m, deconfound_stage2=deconfound_stage2,
                    n_jobs=N_JOBS, seed=1500 + s,
                )
                p2_cal_list.append(fit_out["p2_cal"])  # type: ignore[arg-type]
                p2_te_list.append(fit_out["p2_te"])  # type: ignore[arg-type]
                diags.append(fit_out["diag"])  # type: ignore[arg-type]
                if "xb_cal" in fit_out:
                    last_xb_ca = fit_out["xb_cal"]  # type: ignore[assignment]
                    last_xb_te = fit_out["xb_te"]  # type: ignore[assignment]

            p2_ens_b = np.mean(p2_te_list, axis=0)
            p2_cal_all = np.concatenate(p2_cal_list)
            bag_res = compute_bagged_voi_from_split_posteriors(
                y_b, p1_ens_b, p2_ens_b, p1_cal_cat, p2_cal_all, pcost, contract
            )
            bag_res["p2_ens_b"] = p2_ens_b
            bag_res["p2_cal_all"] = p2_cal_all
            bag_res["last_xb_ca"] = last_xb_ca
            bag_res["last_xb_te"] = last_xb_te
            bag_res["last_p1_ca"] = splits_data[-1]["s1_b"][0]  # type: ignore[index]
            bag_res["last_p2_ca"] = p2_cal_list[-1]
            stored_bagged[arm_key] = bag_res
            specialist_diags_by_arm[arm_key] = diags
            factorial_bagged[arm_key] = {
                "selected_channels": sel_ids,
                "panel_cost_cfp": round(pcost, 5),
                "nb_per_1k": bag_res["nb_per_1k"],
                "snb": bag_res["snb"],
                "query_rate": bag_res["query_rate"],
                "mean_cost_cfp": bag_res["mean_cost_cfp"],
                "auroc_routed": bag_res["auroc_routed"],
                "auprc_routed": bag_res["auprc_routed"],
                "auroc_full_s2": round(float(roc_auc_score(y_b, p2_ens_b)), 4),
                "mean_beta_cz": round(float(np.mean([d["beta_cz"] for d in diags])), 4),
                "mean_rms_r_cz": round(float(np.mean([d["rms_gated_log_odds_r_cz"] for d in diags])), 4),
                "mean_rms_cz_zone_r_cz": round(float(np.mean([d["rms_cz_zone_r_cz"] for d in diags])), 4),
                "mean_flip_rate_cz": round(float(np.mean([d["decision_flip_rate_cz"] for d in diags])), 4),
            }

    # Step 3: Per-Split Discovery Runs -> Union-Panel Bagged VOI, Per-Split Policy Mean, Fractionally Billed Ensemble, Split-Consensus Panel, & Legacy Flat-0.03 Reference
    split_sel_glob: list[list[str]] = []
    split_sel_cz: list[list[str]] = []
    split_p2_glob: list[np.ndarray] = []
    split_p2_cz: list[np.ndarray] = []
    split_p2_cal_glob: list[np.ndarray] = []
    split_p2_cal_cz: list[np.ndarray] = []
    split_unit_nb_glob_ledger = np.zeros(n_b, dtype=float)
    split_unit_nb_cz_ledger = np.zeros(n_b, dtype=float)
    split_nb_list_glob: list[float] = []
    split_nb_list_cz: list[float] = []
    frac_p_glob = np.zeros(n_b, dtype=float)
    frac_p_cz = np.zeros(n_b, dtype=float)
    frac_c_glob = np.zeros(n_b, dtype=float)
    frac_c_cz = np.zeros(n_b, dtype=float)

    for sp in splits_data:
        s = int(sp["s"])  # type: ignore[arg-type]
        tr, ca = sp["tr"], sp["ca"]  # type: ignore[assignment]
        oof_p = sp["oof_p"]  # type: ignore[assignment]
        s1_b = sp["s1_b"]  # type: ignore[assignment]
        dg = run_sequential_discovery_v2(
            df_a.iloc[tr], y_a[tr], base_cols, channels, oof_p, contract,
            method="global_cmi", pilot_fraction=0.20, budget=3.2,
            deconfound_stage2=deconfound_stage2, n_jobs=N_JOBS, seed=900 + s,
        ).outcome
        dc = run_sequential_discovery_v2(
            df_a.iloc[tr], y_a[tr], base_cols, channels, oof_p, contract,
            method="collision_zone", pilot_fraction=0.20, budget=3.2,
            deconfound_stage2=deconfound_stage2, n_jobs=N_JOBS, seed=900 + s,
        ).outcome
        split_sel_glob.append(list(dg.selected_ids))
        split_sel_cz.append(list(dc.selected_ids))

        for tag_s, sids, s2m in [("glob", dg.selected_ids, "global"), ("cz", dc.selected_ids, "fluid_backbone")]:
            scols_s = [col for cid in sids for col in ch_cols_map[cid]]
            pc_s = sched.panel_cost(sids, cost_map)
            fout = _fit_single_panel_stage2_split(
                df_a.iloc[tr], y_a[tr], df_a.iloc[ca], y_a[ca], df_b,
                base_cols, scols_s, oof_p, s1_b, contract,
                stage2_mode=s2m, deconfound_stage2=deconfound_stage2, n_jobs=N_JOBS, seed=1500 + s,
            )
            # Per-split VOI routing under exact split panel cost pc_s
            r_sp = compute_bagged_voi_from_split_posteriors(
                y_b, fout["p1_te"], fout["p2_te"], fout["p1_cal"], fout["p2_cal"], pc_s, contract  # type: ignore[arg-type]
            )
            if tag_s == "glob":
                split_p2_glob.append(fout["p2_te"])  # type: ignore[arg-type]
                split_p2_cal_glob.append(fout["p2_cal"])  # type: ignore[arg-type]
                split_unit_nb_glob_ledger += r_sp["unit_nb"] / float(n_splits)  # type: ignore[operator]
                split_nb_list_glob.append(float(r_sp["nb_per_1k"]))  # type: ignore[arg-type]
                frac_p_glob += r_sp["p_routed"] / float(n_splits)  # type: ignore[operator]
                frac_c_glob += r_sp["cost_vec"] / float(n_splits)  # type: ignore[operator]
            else:
                split_p2_cz.append(fout["p2_te"])  # type: ignore[arg-type]
                split_p2_cal_cz.append(fout["p2_cal"])  # type: ignore[arg-type]
                split_unit_nb_cz_ledger += r_sp["unit_nb"] / float(n_splits)  # type: ignore[operator]
                split_nb_list_cz.append(float(r_sp["nb_per_1k"]))  # type: ignore[arg-type]
                frac_p_cz += r_sp["p_routed"] / float(n_splits)  # type: ignore[operator]
                frac_c_cz += r_sp["cost_vec"] / float(n_splits)  # type: ignore[operator]

    union_glob = sorted({cid for sids in split_sel_glob for cid in sids})
    union_cz = sorted({cid for sids in split_sel_cz for cid in sids})
    cost_union_glob = sched.panel_cost(union_glob, cost_map)
    cost_union_cz = sched.panel_cost(union_cz, cost_map)

    p2_het_ens_glob = np.mean(split_p2_glob, axis=0)
    p2_het_ens_cz = np.mean(split_p2_cz, axis=0)
    p2_het_cal_glob = np.concatenate(split_p2_cal_glob)
    p2_het_cal_cz = np.concatenate(split_p2_cal_cz)

    # Legacy flat-0.03 binary bagged VOI (exact v2.2 anchor) vs Union-Panel Bagged VOI (fixed Q and rerouted Q)
    leg_glob = compute_bagged_voi_from_split_posteriors(
        y_b, p1_ens_b, p2_het_ens_glob, p1_cal_cat, p2_het_cal_glob, 0.030, contract
    )
    leg_cz = compute_bagged_voi_from_split_posteriors(
        y_b, p1_ens_b, p2_het_ens_cz, p1_cal_cat, p2_het_cal_cz, 0.030, contract
    )
    # Union-panel billed on fixed legacy Q vs rerouted at union cost
    u_fixed_glob_c = cost_union_glob * leg_glob["q_mask"].astype(float)  # type: ignore[attr-defined]
    u_fixed_cz_c = cost_union_cz * leg_cz["q_mask"].astype(float)  # type: ignore[attr-defined]
    nb_union_fixed_glob = net_benefit_with_cost_ledger_per_1k(y_b, leg_glob["p_routed"], u_fixed_glob_c, contract)  # type: ignore[arg-type]
    nb_union_fixed_cz = net_benefit_with_cost_ledger_per_1k(y_b, leg_cz["p_routed"], u_fixed_cz_c, contract)  # type: ignore[arg-type]
    union_rerouted_glob = compute_bagged_voi_from_split_posteriors(
        y_b, p1_ens_b, p2_het_ens_glob, p1_cal_cat, p2_het_cal_glob, cost_union_glob, contract
    )
    union_rerouted_cz = compute_bagged_voi_from_split_posteriors(
        y_b, p1_ens_b, p2_het_ens_cz, p1_cal_cat, p2_het_cal_cz, cost_union_cz, contract
    )

    # Split-consensus panel (>= 5/10 splits) check
    freq_g = {c.channel_id: sum(c.channel_id in s for s in split_sel_glob) for c in channels}
    freq_c = {c.channel_id: sum(c.channel_id in s for s in split_sel_cz) for c in channels}
    cons_glob = [cid for cid, cnt in freq_g.items() if cnt >= 5]
    cons_cz = [cid for cid, cnt in freq_c.items() if cnt >= 5]

    # Fractionally billed ensemble NB (F12: explicitly named Fractionally Billed Probability-Averaged Ensemble)
    nb_frac_glob = net_benefit_with_cost_ledger_per_1k(y_b, frac_p_glob, frac_c_glob, contract)
    nb_frac_cz = net_benefit_with_cost_ledger_per_1k(y_b, frac_p_cz, frac_c_cz, contract)
    u_frac_glob = unit_nb_ledger_vector(y_b, frac_p_glob, frac_c_glob, contract)
    u_frac_cz = unit_nb_ledger_vector(y_b, frac_p_cz, frac_c_cz, contract)

    u_base = unit_nb_ledger_vector(y_b, p1_ens_b, np.zeros(n_b), contract)
    nb_base = float(np.mean(u_base))

    prim_cz = stored_bagged["collision_zone__fluid_backbone"]
    prim_glob = stored_bagged["global_cmi__global"]
    abl_cz_glob = stored_bagged["collision_zone__global"]
    abl_glob_fluid = stored_bagged["global_cmi__fluid_backbone"]
    prim_cz_split_conf = stored_bagged["collision_zone_split_conf__fluid_backbone"]

    estimands_summary = {
        "stage1_base_zero_cost": {
            "nb_per_1k": round(nb_base, 4),
            "auroc": round(float(roc_auc_score(y_b, p1_ens_b)), 4),
        },
        "primary_common_panel_bagged_voi": {
            "cz_fluid": factorial_bagged["collision_zone__fluid_backbone"],
            "global_cmi": factorial_bagged["global_cmi__global"],
            "cz_split_conf_fluid": factorial_bagged["collision_zone_split_conf__fluid_backbone"],
            "boot_cz_vs_global_cmi": paired_bootstrap_ledger_diff(prim_cz["unit_nb"], prim_glob["unit_nb"], seed=101),  # type: ignore[arg-type]
            "boot_cz_vs_stage1_base": paired_bootstrap_ledger_diff(prim_cz["unit_nb"], u_base, seed=102),  # type: ignore[arg-type]
            "boot_cz_split_conf_vs_global_cmi": paired_bootstrap_ledger_diff(
                prim_cz_split_conf["unit_nb"], prim_glob["unit_nb"], seed=103  # type: ignore[arg-type]
            ),
            "interaction_2x2": paired_bootstrap_interaction(
                u_base,
                prim_glob["unit_nb"],  # type: ignore[arg-type]
                abl_cz_glob["unit_nb"],  # type: ignore[arg-type]
                abl_glob_fluid["unit_nb"],  # type: ignore[arg-type]
                prim_cz["unit_nb"],  # type: ignore[arg-type]
                n_boot=10000,
                seed=104,
            ),
        },
        "union_panel_bagged_voi_diagnostic": {
            "union_channels_cz": union_cz,
            "union_cost_cz_cfp": round(cost_union_cz, 5),
            "union_channels_glob": union_glob,
            "union_cost_glob_cfp": round(cost_union_glob, 5),
            "fixed_q_nb_cz": round(nb_union_fixed_cz, 4),
            "fixed_q_nb_glob": round(nb_union_fixed_glob, 4),
            "rerouted_nb_cz": union_rerouted_cz["nb_per_1k"],
            "rerouted_query_rate_cz": union_rerouted_cz["query_rate"],
            "rerouted_nb_glob": union_rerouted_glob["nb_per_1k"],
            "rerouted_query_rate_glob": union_rerouted_glob["query_rate"],
            "boot_rerouted_cz_vs_glob": paired_bootstrap_ledger_diff(
                union_rerouted_cz["unit_nb"], union_rerouted_glob["unit_nb"], seed=105  # type: ignore[arg-type]
            ),
        },
        "mean_single_split_policy_utility": {
            "cz_fluid_mean_nb": round(float(np.mean(split_nb_list_cz)), 4),
            "cz_fluid_sd_across_splits": round(float(np.std(split_nb_list_cz, ddof=1)), 4),
            "global_cmi_mean_nb": round(float(np.mean(split_nb_list_glob)), 4),
            "global_cmi_sd_across_splits": round(float(np.std(split_nb_list_glob, ddof=1)), 4),
            "boot_cz_vs_glob_conditional_on_cohort": paired_bootstrap_ledger_diff(
                split_unit_nb_cz_ledger, split_unit_nb_glob_ledger, seed=106
            ),
            "split_selection_counts_cz": {k: v for k, v in freq_c.items() if v > 0},
            "split_selection_counts_glob": {k: v for k, v in freq_g.items() if v > 0},
            "consensus_ge5_cz": cons_cz,
            "consensus_ge5_glob": cons_glob,
        },
        "fractionally_billed_prob_averaged_ensemble": {
            "cz_fluid_nb": round(nb_frac_cz, 4),
            "global_cmi_nb": round(nb_frac_glob, 4),
            "boot_cz_vs_glob": paired_bootstrap_ledger_diff(u_frac_cz, u_frac_glob, seed=107),
        },
        "legacy_v2_2_flat_003_billing_anchor": {
            "cz_fluid_nb": leg_cz["nb_per_1k"],
            "global_cmi_nb": leg_glob["nb_per_1k"],
            "boot_cz_vs_glob": paired_bootstrap_ledger_diff(leg_cz["unit_nb"], leg_glob["unit_nb"], seed=108),  # type: ignore[arg-type]
        },
    }

    if not run_full_factorial_and_afa:
        return {
            "summary_mode": summary_mode,
            "elapsed_sec": round(time.time() - t0, 2),
            "common_panels": common_panels,
            "common_panel_costs": common_panel_costs,
            "estimands": estimands_summary,
            "factorial_bagged": factorial_bagged,
        }

    # Step 4: Patient-Level Channel-Billed Sequential AFA (Covert-CMI vs Direct-VOI, F1 & F14)
    afa_summary: dict[str, object] = {}
    for crit in ("cmi", "direct_voi"):
        afa_p_sum = np.zeros(n_b, dtype=float)
        afa_m_votes = np.zeros((n_b, len(channels)), dtype=float)
        afa_u_split_mean = np.zeros(n_b, dtype=float)
        afa_split_nbs: list[float] = []
        for sp in splits_data:
            s = int(sp["s"])  # type: ignore[arg-type]
            tr, ca = sp["tr"], sp["ca"]  # type: ignore[assignment]
            res_afa = evaluate_channel_billed_afa_v3(
                df_a.iloc[tr], y_a[tr], df_a.iloc[ca], y_a[ca], [(df_b, y_b)],
                base_cols, channels, sp["oof_p"], sp["s1_b"], contract, sched,  # type: ignore[arg-type]
                criterion=crit, budget=3.2, n_masks_per_row=4, n_jobs=N_JOBS, seed=1700 + s,
            )[0]
            p_s = res_afa["p_auto_te"]  # type: ignore[assignment]
            c_s = res_afa["cost_vec_te"]  # type: ignore[assignment]
            m_s = res_afa["mask_te"]  # type: ignore[assignment]
            afa_p_sum += p_s / float(n_splits)
            afa_m_votes += m_s / float(n_splits)
            afa_u_split_mean += unit_nb_ledger_vector(y_b, p_s, c_s, contract) / float(n_splits)
            afa_split_nbs.append(float(res_afa["nb_ledger_per_1k"]))  # type: ignore[arg-type]

        # Deployable majority-vote channel mask M_bag = (afa_m_votes >= 0.5) billed via exact PanelCostSchedule!
        m_maj = (afa_m_votes >= 0.5).astype(float)
        c_maj = sched.instance_cost_vector(m_maj, channels)
        p_maj = np.where(np.any(m_maj > 0.5, axis=1), afa_p_sum, p1_ens_b)
        u_maj = unit_nb_ledger_vector(y_b, p_maj, c_maj, contract)
        afa_summary[crit] = {
            "per_split_mean_nb": round(float(np.mean(afa_split_nbs)), 4),
            "per_split_sd_nb": round(float(np.std(afa_split_nbs, ddof=1)), 4),
            "majority_mask_deployable_nb": round(float(np.mean(u_maj)), 4),
            "majority_mask_query_rate": round(float(np.mean(np.any(m_maj > 0.5, axis=1))), 4),
            "majority_mask_mean_cost_cfp": round(float(np.mean(c_maj)), 5),
            "boot_primary_cz_vs_afa_majority": paired_bootstrap_ledger_diff(prim_cz["unit_nb"], u_maj, seed=115),  # type: ignore[arg-type]
            "boot_primary_cz_vs_afa_persplit": paired_bootstrap_ledger_diff(
                prim_cz["unit_nb"], afa_u_split_mean, seed=116  # type: ignore[arg-type]
            ),
        }

    # Step 5: F14 Factorial Bootstrap Contrasts on Common-Panel Bagged VOI
    factorial_contrasts = {
        "cz_beta_fluid_vs_direct_utility_fluid": paired_bootstrap_ledger_diff(
            stored_bagged["collision_zone__fluid_backbone"]["unit_nb"],  # type: ignore[arg-type]
            stored_bagged["direct_utility__fluid_backbone"]["unit_nb"],  # type: ignore[arg-type]
            seed=121,
        ),
        "cz_beta_fluid_vs_local_utility_fluid": paired_bootstrap_ledger_diff(
            stored_bagged["collision_zone__fluid_backbone"]["unit_nb"],  # type: ignore[arg-type]
            stored_bagged["local_utility__fluid_backbone"]["unit_nb"],  # type: ignore[arg-type]
            seed=122,
        ),
        "cz_beta_fluid_vs_brier_fluid": paired_bootstrap_ledger_diff(
            stored_bagged["collision_zone__fluid_backbone"]["unit_nb"],  # type: ignore[arg-type]
            stored_bagged["brier__fluid_backbone"]["unit_nb"],  # type: ignore[arg-type]
            seed=123,
        ),
        "cz_beta_fluid_vs_cz_beta_ungated_residual": paired_bootstrap_ledger_diff(
            stored_bagged["collision_zone__fluid_backbone"]["unit_nb"],  # type: ignore[arg-type]
            stored_bagged["collision_zone__ungated_residual"]["unit_nb"],  # type: ignore[arg-type]
            seed=124,
        ),
    }

    # Step 6: F2 Complete 3-Way Exchange Table, Directional/Sensitivity-Constrained Policies, & Cost Sweep
    exchange_table = []
    for t_val in (0.04, 0.0625, 0.10, 0.125):
        r_cz = _clinical_counts_row(y_b, prim_cz["p_routed"], prim_cz["q_mask"], prim_cz["cost_vec"], t_val)  # type: ignore[arg-type]
        r_gl = _clinical_counts_row(y_b, prim_glob["p_routed"], prim_glob["q_mask"], prim_glob["cost_vec"], t_val)  # type: ignore[arg-type]
        r_bs = _clinical_counts_row(y_b, p1_ens_b, np.zeros(n_b, dtype=bool), np.zeros(n_b), t_val)
        exchange_table.append(
            {
                "tau": t_val,
                "cz_fluid": r_cz,
                "global_cmi": r_gl,
                "stage1_base": r_bs,
                "diff_cz_minus_glob": {
                    "delta_tp_count": r_cz["tp_count"] - r_gl["tp_count"],
                    "delta_fp_count": r_cz["fp_count"] - r_gl["fp_count"],
                    "delta_queried_count": r_cz["queried_count"] - r_gl["queried_count"],
                    "delta_tp_per_1k": round(r_cz["tp_per_1k"] - r_gl["tp_per_1k"], 2),
                    "false_alerts_spared_per_1k": round(r_gl["fp_per_1k"] - r_cz["fp_per_1k"], 2),
                    "additional_queries_per_1k": round(r_cz["queries_per_1k"] - r_gl["queries_per_1k"], 2),
                    "delta_nb_per_1k": round(r_cz["nb_ledger_per_1k"] - r_gl["nb_ledger_per_1k"], 3),
                },
            }
        )

    # Directional / Sensitivity-Constrained VOI policies on Primary Common-Panel CZ-Fluid
    pc_cz = float(prim_cz["panel_cost_cfp"])  # type: ignore[arg-type]
    edges_cz = prim_cz["edges"]  # type: ignore[assignment]
    bq_cz = prim_cz["bin_quantiles"]  # type: ignore[assignment]
    b_cal_all = np.clip(np.digitize(p1_cal_cat, edges_cz[1:-1], right=False), 0, 11)
    voi_cal_all = empirical_martingale_voi(p1_cal_cat, b_cal_all, bq_cz, contract)
    y_cal_all = np.concatenate([y_a[sp["ca"]] for sp in splits_data])  # type: ignore[index]

    directional_policies: dict[str, object] = {}
    for pmode in ("two_sided", "escalation_only", "deescalation_only", "sensitivity_constrained"):
        p_pol, q_pol = apply_directional_voi_policy(
            p1_ens_b,
            prim_cz["p2_ens_b"],  # type: ignore[arg-type]
            prim_cz["voi_te"],  # type: ignore[arg-type]
            pc_cz,
            contract,
            policy_mode=pmode,
            p1_cal=p1_cal_cat,
            p2_cal=prim_cz["p2_cal_all"],  # type: ignore[arg-type]
            y_cal=y_cal_all,
            voi_cal=voi_cal_all,
            max_rel_sens_drop=0.01,
        )
        c_pol = pc_cz * q_pol.astype(float)
        u_pol = unit_nb_ledger_vector(y_b, p_pol, c_pol, contract)
        cnt_pol = _clinical_counts_row(y_b, p_pol, q_pol, c_pol, contract.tau_star)
        directional_policies[pmode] = {
            **cnt_pol,
            "boot_vs_global_cmi": paired_bootstrap_ledger_diff(u_pol, prim_glob["unit_nb"], seed=130),  # type: ignore[arg-type]
            "boot_vs_stage1_base": paired_bootstrap_ledger_diff(u_pol, u_base, seed=131),
        }

    # Acquisition-cost sensitivity sweep (multiplier m in {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0})
    pc_gl = float(prim_glob["panel_cost_cfp"])  # type: ignore[arg-type]
    cost_sweep = []
    for mult in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0):
        # Fixed-policy NB under rescaled cost vs VOI-rerouted NB under rescaled cost
        c_cz_fix = (mult * pc_cz) * prim_cz["q_mask"].astype(float)  # type: ignore[attr-defined]
        c_gl_fix = (mult * pc_gl) * prim_glob["q_mask"].astype(float)  # type: ignore[attr-defined]
        nb_cz_fix = net_benefit_with_cost_ledger_per_1k(y_b, prim_cz["p_routed"], c_cz_fix, contract)  # type: ignore[arg-type]
        nb_gl_fix = net_benefit_with_cost_ledger_per_1k(y_b, prim_glob["p_routed"], c_gl_fix, contract)  # type: ignore[arg-type]
        rr_cz = compute_bagged_voi_from_split_posteriors(
            y_b, p1_ens_b, prim_cz["p2_ens_b"], p1_cal_cat, prim_cz["p2_cal_all"], mult * pc_cz, contract  # type: ignore[arg-type]
        )
        rr_gl = compute_bagged_voi_from_split_posteriors(
            y_b, p1_ens_b, prim_glob["p2_ens_b"], p1_cal_cat, prim_glob["p2_cal_all"], mult * pc_gl, contract  # type: ignore[arg-type]
        )
        cost_sweep.append(
            {
                "cost_multiplier": mult,
                "cz_panel_cost_cfp": round(mult * pc_cz, 5),
                "glob_panel_cost_cfp": round(mult * pc_gl, 5),
                "fixed_policy_delta_nb": round(nb_cz_fix - nb_gl_fix, 4),
                "rerouted_cz_nb": rr_cz["nb_per_1k"],
                "rerouted_glob_nb": rr_gl["nb_per_1k"],
                "rerouted_delta_nb": round(float(rr_cz["nb_per_1k"]) - float(rr_gl["nb_per_1k"]), 4),  # type: ignore[arg-type]
                "rerouted_cz_q_rate": rr_cz["query_rate"],
            }
        )

    # Exact fixed-policy break-even cost multiplier where NB(CZ-Fluid) == NB(Global-CMI)
    nb0_cz = net_benefit_with_cost_ledger_per_1k(y_b, prim_cz["p_routed"], np.zeros(n_b), contract)  # type: ignore[arg-type]
    nb0_gl = net_benefit_with_cost_ledger_per_1k(y_b, prim_glob["p_routed"], np.zeros(n_b), contract)  # type: ignore[arg-type]
    q_diff_cost = float(np.mean(pc_cz * prim_cz["q_mask"].astype(float) - pc_gl * prim_glob["q_mask"].astype(float)))  # type: ignore[attr-defined]
    breakeven_mult = (
        float((nb0_cz - nb0_gl) * contract.cost_fn / (1000.0 * q_diff_cost))
        if q_diff_cost > 1e-9
        else float("inf")
    )

    # Step 7: F8 Stage-B Routing Comparators holding Common-Panel CZ-Fluid posteriors fixed
    b_te_cz = np.clip(np.digitize(p1_ens_b, edges_cz[1:-1], right=False), 0, 11)
    p2_cz_ens = prim_cz["p2_ens_b"]  # type: ignore[assignment]
    k_match = max(1, int(np.sum(prim_cz["q_mask"])))  # type: ignore[arg-type]

    v_unc_b = routing_priority_scores(p1_ens_b, b_te_cz, bq_cz, contract, mode="uncentered_empirical_voi")
    v_dist_b = routing_priority_scores(p1_ens_b, b_te_cz, bq_cz, contract, mode="distance_to_threshold")
    v_ent_b = routing_priority_scores(p1_ens_b, b_te_cz, bq_cz, contract, mode="shannon_entropy")
    v_cov_b = covariate_enriched_martingale_voi(
        prim_cz["last_p1_ca"], prim_cz["last_p2_ca"], prim_cz["last_xb_ca"],  # type: ignore[arg-type]
        p1_ens_b, prim_cz["last_xb_te"], contract, n_jobs=N_JOBS, seed=141,  # type: ignore[arg-type]
    )
    v_dir_b = direct_conditional_bayes_voi(
        prim_cz["last_p1_ca"], prim_cz["last_p2_ca"], prim_cz["last_xb_ca"],  # type: ignore[arg-type]
        p1_ens_b, prim_cz["last_xb_te"], contract, n_jobs=N_JOBS, seed=142,  # type: ignore[arg-type]
    )

    def _eval_q_mask(qm: np.ndarray) -> dict[str, float]:
        pr = np.where(qm, p2_cz_ens, p1_ens_b)
        cv = pc_cz * qm.astype(float)
        return {
            "nb_per_1k": round(net_benefit_with_cost_ledger_per_1k(y_b, pr, cv, contract), 4),
            "query_rate": round(float(np.mean(qm)), 4),
            "auroc_routed": round(float(roc_auc_score(y_b, pr)), 4),
        }

    q_dist_m = np.zeros(n_b, dtype=bool)
    q_dist_m[np.argsort(v_dist_b)[-k_match:]] = True
    q_ent_m = np.zeros(n_b, dtype=bool)
    q_ent_m[np.argsort(v_ent_b)[-k_match:]] = True

    routing_comparators = {
        "empirical_martingale_voi_cost_threshold": _eval_q_mask(prim_cz["q_mask"]),  # type: ignore[arg-type]
        "uncentered_empirical_voi_cost_threshold": _eval_q_mask(v_unc_b > pc_cz),
        "covariate_enriched_martingale_voi_cost_threshold": _eval_q_mask(v_cov_b > pc_cz),
        "direct_conditional_bayes_voi_cost_threshold": _eval_q_mask(v_dir_b > pc_cz),
        "distance_to_threshold_matched_q": _eval_q_mask(q_dist_m),
        "shannon_entropy_matched_q": _eval_q_mask(q_ent_m),
        "query_all_100pct": _eval_q_mask(np.ones(n_b, dtype=bool)),
        "query_none_0pct": _eval_q_mask(np.zeros(n_b, dtype=bool)),
    }

    print(
        f"  [{summary_mode}] Done in {time.time()-t0:.1f}s | "
        f"Common-Panel Bagged NB: CZ={prim_cz['nb_per_1k']:.3f} vs Glob={prim_glob['nb_per_1k']:.3f} "
        f"(Delta={estimands_summary['primary_common_panel_bagged_voi']['boot_cz_vs_global_cmi']['diff_mean']:+.3f}, "
        f"p={estimands_summary['primary_common_panel_bagged_voi']['boot_cz_vs_global_cmi']['p_two_sided_dh']})"
    )

    return {
        "summary_mode": summary_mode,
        "elapsed_sec": round(time.time() - t0, 2),
        "common_panels": common_panels,
        "common_panel_costs": common_panel_costs,
        "estimands": estimands_summary,
        "factorial_bagged": factorial_bagged,
        "factorial_contrasts_10k": factorial_contrasts,
        "channel_billed_afa": afa_summary,
        "exchange_rate_table": exchange_table,
        "directional_voi_policies": directional_policies,
        "acquisition_cost_sweep": cost_sweep,
        "fixed_policy_breakeven_cost_multiplier": round(breakeven_mult, 3),
        "fixed_policy_breakeven_panel_cost_cfp": round(breakeven_mult * pc_cz, 4) if np.isfinite(breakeven_mult) else None,
        "stage_b_routing_comparators": routing_comparators,
    }


def run_block2_sepsis_primary() -> dict[str, object]:
    t0 = time.time()
    print("[Block 2] Starting Sepsis BIDMC -> Emory Primary Ledger & Factorial Repairs...")
    manifest = json.loads((PROC_DIR / "ws4tb_cohorts_manifest.json").read_text())
    base_cols: list[str] = list(manifest["sepsis_6h"]["base_columns"])

    df_a6 = pd.read_parquet(PROC_DIR / "sepsis_setA_bidmc_6h.parquet")
    df_b6 = pd.read_parquet(PROC_DIR / "sepsis_setB_emory_6h.parquet")
    df_a_meas = df_a6[(df_a6["s2_Creatinine_drawn"] == 1) & (df_a6["s2_WBC_drawn"] == 1)].reset_index(drop=True)
    df_b_meas = df_b6[(df_b6["s2_Creatinine_drawn"] == 1) & (df_b6["s2_WBC_drawn"] == 1)].reset_index(drop=True)

    # 1. Primary B1b under BOTH point-in-time single-draw (`_last` only) and 6h window summary (`_last, _max, _min`)
    b1b_last_only = _evaluate_sepsis_mode_full(
        df_a_meas, df_b_meas, base_cols, summary_mode="point_in_time_last", n_splits=10, run_full_factorial_and_afa=True
    )
    b1b_window_6h = _evaluate_sepsis_mode_full(
        df_a_meas, df_b_meas, base_cols, summary_mode="window_6h_summary", n_splits=10, run_full_factorial_and_afa=True
    )

    # 2. All-Comers Track A (N_B = 18,359) & Deconfounded Track B1a under Common-Panel Bagged VOI (`point_in_time_last`)
    b1a_deconf = _evaluate_sepsis_mode_full(
        df_a6, df_b6, base_cols, summary_mode="point_in_time_last", n_splits=10,
        run_full_factorial_and_afa=False, deconfound_stage2=True,
    )
    track_a_allcomers = _evaluate_sepsis_mode_full(
        df_a6, df_b6, base_cols, summary_mode="point_in_time_last", n_splits=10,
        run_full_factorial_and_afa=False, deconfound_stage2=False,
    )

    out = {
        "elapsed_sec": round(time.time() - t0, 2),
        "b1b_measured_point_in_time_last": b1b_last_only,
        "b1b_measured_window_6h_summary": b1b_window_6h,
        "b1a_allcomers_agniel_deconfounded": b1a_deconf,
        "track_a_allcomers_observational": track_a_allcomers,
    }
    BLOCKS_DIR.mkdir(parents=True, exist_ok=True)
    (BLOCKS_DIR / "block2_sepsis_primary_ledger.json").write_text(json.dumps(out, indent=2))
    print(f"[Block 2] Saved checkpoint in {time.time()-t0:.1f}s.")
    return out


# ===========================================================================
# BLOCK 3: 2D Event-Time Horizons (F9), Label-Free EM Shift (F10), & Missingness (F11)
# ===========================================================================


def _ensure_onset_metadata_from_archive() -> pd.DataFrame:
    """Extract first positive SepsisLabel hour (0-based) for every patient in archive.zip."""
    cache_p = CACHE_DIR / "sepsis_onset_hours.parquet"
    if cache_p.exists():
        return pd.read_parquet(cache_p)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print("[Block 3] Extracting exact SepsisLabel onset hour per patient from archive.zip...")
    rows: list[dict[str, object]] = []
    with zipfile.ZipFile(RAW_DIR / "archive.zip", "r") as zf:
        names = sorted(n for n in zf.namelist() if n.endswith(".psv"))
        hdr = zf.read(names[0]).decode("utf-8").splitlines()[0].split("|")
        idx_sep = hdr.index("SepsisLabel")
        for nm in names:
            lines = zf.read(nm).decode("utf-8").splitlines()[1:]
            sep_vals = [float(ln.split("|")[idx_sep]) for ln in lines if ln]
            pos_hrs = [h for h, val in enumerate(sep_vals) if val == 1.0]
            rows.append(
                {
                    "patient_id": Path(nm).stem,
                    "site": "BIDMC_SetA" if "training_setA" in nm else "Emory_SetB",
                    "n_stay_hours": len(sep_vals),
                    "first_sepsis_label_hour": int(pos_hrs[0]) if pos_hrs else -1,
                }
            )
    df_on = pd.DataFrame(rows)
    df_on.to_parquet(cache_p, index=False)
    return df_on


def _eval_compact_common_bagged(
    df_a: pd.DataFrame,
    y_a: np.ndarray,
    df_b: pd.DataFrame,
    y_b: np.ndarray,
    base_cols: list[str],
    channels: list[ChannelDef],
    *,
    summary_mode: str = "point_in_time_last",
    n_splits: int = 8,
) -> dict[str, object]:
    """Fast deterministic Common-Panel Bagged VOI comparison for 2D horizon & sensitivity cells."""
    sched = PanelCostSchedule(venipuncture_cost=0.010, abg_extra_procedure_cost=0.010, unit_assay_scale=0.020)
    cost_map = {c.channel_id: float(c.cost) for c in channels}
    ch_cols_map = {c.channel_id: list(c.columns) for c in channels}
    contract = ClinicalContract(cost_fp=1.0, cost_fn=15.0, target_precision=0.30, per_instance_query_cost=0.03)

    oof_a = compute_oof_stage1_v2(df_a[base_cols], y_a, n_folds=4, seed=600, n_jobs=N_JOBS)
    d_gl = run_discovery_v3(
        df_a, y_a, base_cols, channels, oof_a, contract,
        objective="global_cmi", pilot_fraction=0.20, budget=3.2, n_jobs=N_JOBS, seed=900,
    )
    d_cz = run_discovery_v3(
        df_a, y_a, base_cols, channels, oof_a, contract,
        objective="collision_zone", pilot_fraction=0.20, budget=3.2, n_jobs=N_JOBS, seed=900,
    )
    cols_gl = [col for cid in d_gl.selected_ids for col in ch_cols_map[cid]]
    cols_cz = [col for cid in d_cz.selected_ids for col in ch_cols_map[cid]]
    pc_gl = sched.panel_cost(d_gl.selected_ids, cost_map)
    pc_cz = sched.panel_cost(d_cz.selected_ids, cost_map)

    p1_b_list, p1_ca_list = [], []
    p2_gl_b, p2_gl_ca = [], []
    p2_cz_b, p2_cz_ca = [], []
    for s in range(n_splits):
        tr, ca, _ = make_stratified_3way_split(y_a, seed=700 + s)
        oof_s = compute_oof_stage1_v2(df_a.iloc[tr][base_cols], y_a[tr], n_folds=4, seed=800 + s, n_jobs=N_JOBS)
        s1_b = fit_stage1_predictions(df_a.iloc[tr], y_a[tr], df_a.iloc[ca], y_a[ca], [df_b], base_cols, seed=1500 + s)
        p1_ca_list.append(s1_b[0])
        p1_b_list.append(s1_b[1][0])
        fg = _fit_single_panel_stage2_split(
            df_a.iloc[tr], y_a[tr], df_a.iloc[ca], y_a[ca], df_b,
            base_cols, cols_gl, oof_s, s1_b, contract, stage2_mode="global", n_jobs=N_JOBS, seed=1500 + s,
        )
        fc = _fit_single_panel_stage2_split(
            df_a.iloc[tr], y_a[tr], df_a.iloc[ca], y_a[ca], df_b,
            base_cols, cols_cz, oof_s, s1_b, contract, stage2_mode="fluid_backbone", n_jobs=N_JOBS, seed=1500 + s,
        )
        p2_gl_ca.append(fg["p2_cal"])  # type: ignore[arg-type]
        p2_gl_b.append(fg["p2_te"])  # type: ignore[arg-type]
        p2_cz_ca.append(fc["p2_cal"])  # type: ignore[arg-type]
        p2_cz_b.append(fc["p2_te"])  # type: ignore[arg-type]

    p1_ens = np.mean(p1_b_list, axis=0)
    p1_ca_all = np.concatenate(p1_ca_list)
    bg_gl = compute_bagged_voi_from_split_posteriors(
        y_b, p1_ens, np.mean(p2_gl_b, axis=0), p1_ca_all, np.concatenate(p2_gl_ca), pc_gl, contract
    )
    bg_cz = compute_bagged_voi_from_split_posteriors(
        y_b, p1_ens, np.mean(p2_cz_b, axis=0), p1_ca_all, np.concatenate(p2_cz_ca), pc_cz, contract
    )
    bt = paired_bootstrap_ledger_diff(bg_cz["unit_nb"], bg_gl["unit_nb"], n_boot=5000, seed=201)  # type: ignore[arg-type]
    return {
        "n_bidmc": int(y_a.size),
        "events_bidmc": int(np.sum(y_a)),
        "n_emory": int(y_b.size),
        "events_emory": int(np.sum(y_b)),
        "prev_emory": round(float(np.mean(y_b)), 4),
        "selected_glob": list(d_gl.selected_ids),
        "selected_cz": list(d_cz.selected_ids),
        "nb_glob": bg_gl["nb_per_1k"],
        "nb_cz": bg_cz["nb_per_1k"],
        "delta_nb": bt["diff_mean"],
        "ci95_low": bt["ci95_low"],
        "ci95_high": bt["ci95_high"],
        "p_two_sided_dh": bt["p_two_sided_dh"],
        "p_routed_cz": bg_cz["p_routed"],
        "p_routed_glob": bg_gl["p_routed"],
        "p1_ens": p1_ens,
        "p2_ens_cz": np.mean(p2_cz_b, axis=0),
        "p1_ca_all": p1_ca_all,
        "p2_ca_cz": np.concatenate(p2_cz_ca),
        "pc_cz": pc_cz,
    }


def run_block3_horizons_shift_missingness() -> dict[str, object]:
    t0 = time.time()
    print("[Block 3] Starting 2D Horizon Matrix (F9), Label-Free EM Shift (F10), & Missingness Audit (F11)...")
    from run_paper2_v2_suite import ensure_staged_sepsis_landmarks

    lm_paths = ensure_staged_sepsis_landmarks([4, 6, 8, 12])
    df_onset = _ensure_onset_metadata_from_archive()
    onset_map = dict(zip(df_onset["patient_id"], df_onset["first_sepsis_label_hour"]))
    manifest = json.loads((PROC_DIR / "ws4tb_cohorts_manifest.json").read_text())
    base_cols: list[str] = list(manifest["sepsis_6h"]["base_columns"])

    # 1. F9: 2D Observation-Window (4, 6, 8, 12h) x Prediction-Horizon (next 2h, 6h, 12h, remaining stay)
    horizon_specs = [("next_2h", 2), ("next_6h", 6), ("next_12h", 12), ("remaining_stay", 99999)]
    matrix_2d: list[dict[str, object]] = []
    lead_time_strat_6h: dict[str, object] = {}

    for w in (4, 6, 8, 12):
        pa, pb = lm_paths[w]
        dfa_raw = pd.read_parquet(pa)
        dfb_raw = pd.read_parquet(pb)
        dfa = dfa_raw[(dfa_raw["s2_Creatinine_drawn"] == 1) & (dfa_raw["s2_WBC_drawn"] == 1)].reset_index(drop=True)
        dfb = dfb_raw[(dfb_raw["s2_Creatinine_drawn"] == 1) & (dfb_raw["s2_WBC_drawn"] == 1)].reset_index(drop=True)
        on_a = np.array([int(onset_map.get(str(pid), -1)) for pid in dfa["patient_id"]])
        on_b = np.array([int(onset_map.get(str(pid), -1)) for pid in dfb["patient_id"]])
        chs = build_sepsis_channels_by_mode(dfa, summary_mode="point_in_time_last")

        for h_name, h_hrs in horizon_specs:
            # Landmark w uses rows 0..w-1; future label in next H hours checks onset in [w-1, w - 1 + h_hrs]
            ya_h = ((on_a >= (w - 1)) & (on_a <= (w - 1 + h_hrs))).astype(int)
            yb_h = ((on_b >= (w - 1)) & (on_b <= (w - 1 + h_hrs))).astype(int)
            if int(np.sum(ya_h)) < 25 or int(np.sum(yb_h)) < 15:
                continue
            res_cell = _eval_compact_common_bagged(dfa, ya_h, dfb, yb_h, base_cols, chs, n_splits=6)
            cell_clean = {
                k: v
                for k, v in res_cell.items()
                if k not in {"p_routed_cz", "p_routed_glob", "p1_ens", "p2_ens_cz", "p1_ca_all", "p2_ca_cz", "pc_cz"}
            }
            matrix_2d.append({"landmark_hours": w, "horizon": h_name, **cell_clean})
            print(
                f"  [F9 2D W={w}h x {h_name}] Emory events={res_cell['events_emory']} | "
                f"Delta NB={res_cell['delta_nb']:+.2f} [{res_cell['ci95_low']:+.2f}, {res_cell['ci95_high']:+.2f}]"
            )

            # At W=6h remaining_stay, stratify Emory true cases by lead time (on_b - 5) to SepsisLabel=1!
            if w == 6 and h_name == "remaining_stay":
                lead_hrs = on_b - 5  # hours after W=6h landmark until SepsisLabel turns 1
                p_cz_6 = res_cell["p_routed_cz"]  # type: ignore[assignment]
                p_gl_6 = res_cell["p_routed_glob"]  # type: ignore[assignment]
                p_bs_6 = res_cell["p1_ens"]  # type: ignore[assignment]
                for b_lbl, lo_h, hi_h in [("0_to_2h", 0, 2), ("3_to_6h", 3, 6), ("7_to_12h", 7, 12), ("gt_12h", 13, 9999)]:
                    m_pos = (yb_h == 1) & (lead_hrs >= lo_h) & (lead_hrs <= hi_h)
                    n_p = int(np.sum(m_pos))
                    if n_p > 0:
                        lead_time_strat_6h[b_lbl] = {
                            "n_events": n_p,
                            "sens_stage1_pct": round(100.0 * float(np.mean(p_bs_6[m_pos] >= 0.0625)), 1),
                            "sens_glob_cmi_pct": round(100.0 * float(np.mean(p_gl_6[m_pos] >= 0.0625)), 1),
                            "sens_cz_fluid_pct": round(100.0 * float(np.mean(p_cz_6[m_pos] >= 0.0625)), 1),
                        }

    # F9 Incorporation-Bias Audit at W=6h: exclude SOFA organ-score channels (Creatinine/BUN, Bilirubin, Platelets, Blood Gas)
    pa6, pb6 = lm_paths[6]
    dfa6 = pd.read_parquet(pa6)
    dfb6 = pd.read_parquet(pb6)
    dfa6_m = dfa6[(dfa6["s2_Creatinine_drawn"] == 1) & (dfa6["s2_WBC_drawn"] == 1)].reset_index(drop=True)
    dfb6_m = dfb6[(dfb6["s2_Creatinine_drawn"] == 1) & (dfb6["s2_WBC_drawn"] == 1)].reset_index(drop=True)
    ya6 = dfa6_m["y_sepsis_future"].to_numpy(dtype=int)
    yb6 = dfb6_m["y_sepsis_future"].to_numpy(dtype=int)

    sofa_excl = {"lab_renal_azotemia", "lab_bilirubin", "lab_platelets", "lab_blood_gas_acidbase"}
    chs_no_sofa = build_sepsis_channels_by_mode(dfa6_m, summary_mode="point_in_time_last", exclude_ids=sofa_excl)
    res_no_sofa = _eval_compact_common_bagged(dfa6_m, ya6, dfb6_m, yb6, base_cols, chs_no_sofa, n_splits=10)
    chs_no_sofa_no_lac = build_sepsis_channels_by_mode(
        dfa6_m, summary_mode="point_in_time_last", exclude_ids=sofa_excl | {"lab_lactate"}
    )
    res_no_sofa_no_lac = _eval_compact_common_bagged(
        dfa6_m, ya6, dfb6_m, yb6, base_cols, chs_no_sofa_no_lac, n_splits=10
    )
    incorp_audit = {
        "no_sofa_score_labs_lactate_allowed": {
            k: v
            for k, v in res_no_sofa.items()
            if k not in {"p_routed_cz", "p_routed_glob", "p1_ens", "p2_ens_cz", "p1_ca_all", "p2_ca_cz", "pc_cz"}
        },
        "no_sofa_and_no_lactate": {
            k: v
            for k, v in res_no_sofa_no_lac.items()
            if k not in {"p_routed_cz", "p_routed_glob", "p1_ens", "p2_ens_cz", "p1_ca_all", "p2_ca_cz", "pc_cz"}
        },
    }

    # 2. F10: Label-Free Saerens EM vs Historical Split vs Oracle Target Prevalence on Track B2 (4h + 2h gap)
    dfa_b2 = pd.read_parquet(PROC_DIR / "sepsis_setA_bidmc_4h_gap2h.parquet")
    dfb_b2 = pd.read_parquet(PROC_DIR / "sepsis_setB_emory_4h_gap2h.parquet")
    ya_b2 = dfa_b2["y_sepsis_future"].to_numpy(dtype=int)
    yb_b2 = dfb_b2["y_sepsis_future"].to_numpy(dtype=int)
    base_b2 = [c for c in base_cols if c in dfa_b2.columns]
    chs_b2 = build_sepsis_channels_by_mode(dfa_b2, summary_mode="point_in_time_last")
    b2_fit = _eval_compact_common_bagged(dfa_b2, ya_b2, dfb_b2, yb_b2, base_b2, chs_b2, n_splits=8)

    pi_src_b2 = float(np.mean(ya_b2))
    pi_orc_b2 = float(np.mean(yb_b2))
    p1_b2 = b2_fit["p1_ens"]  # type: ignore[assignment]
    p2_b2 = b2_fit["p2_ens_cz"]  # type: ignore[assignment]
    p1_ca_b2 = b2_fit["p1_ca_all"]  # type: ignore[assignment]
    p2_ca_b2 = b2_fit["p2_ca_cz"]  # type: ignore[assignment]
    pc_b2 = float(b2_fit["pc_cz"])  # type: ignore[arg-type]
    contract = ClinicalContract(cost_fp=1.0, cost_fn=15.0, target_precision=0.30, per_instance_query_cost=0.03)

    def _eval_shifted_b2(pi_est: float, eval_mask: np.ndarray | None = None) -> dict[str, float]:
        idx_e = np.arange(yb_b2.size) if eval_mask is None else np.flatnonzero(eval_mask)
        p1_s, d0 = prior_odds_recalibrate_probabilities(p1_b2[idx_e], pi_src_b2, pi_est)
        p2_s, _ = prior_odds_recalibrate_probabilities(p2_b2[idx_e], pi_src_b2, pi_est)
        p1_cs, _ = prior_odds_recalibrate_probabilities(p1_ca_b2, pi_src_b2, pi_est)
        p2_cs, _ = prior_odds_recalibrate_probabilities(p2_ca_b2, pi_src_b2, pi_est)
        bg = compute_bagged_voi_from_split_posteriors(
            yb_b2[idx_e], p1_s, p2_s, p1_cs, p2_cs, pc_b2, contract
        )
        cal_m = compute_calibration_metrics(yb_b2[idx_e], bg["p_routed"], n_boot=0, return_curve=False)  # type: ignore[arg-type]
        return {
            "pi_target_est": round(float(pi_est), 5),
            "delta_0_logit": round(float(d0), 4),
            "nb_per_1k": bg["nb_per_1k"],
            "query_rate": bg["query_rate"],
            "citl": round(float(cal_m["citl"]), 4),
            "ici": round(float(cal_m["ici_loess"]), 4),
        }

    # Unlabeled Saerens-Latinne-Decaestecker EM (strictly zero target labels!)
    pi_em_b2, _ = saerens_em_target_prevalence(p1_b2, pi_src_b2)
    # Disjoint first-25% Emory cohort slice for historical prevalence -> evaluate strictly on remaining 75%!
    n_hist = yb_b2.size // 4
    hist_eval_mask = np.arange(yb_b2.size) >= n_hist
    pi_hist_25 = float(np.mean(yb_b2[:n_hist]))

    f10_b2_shift = {
        "pi_source_bidmc": round(pi_src_b2, 5),
        "pi_target_true_emory": round(pi_orc_b2, 5),
        "pi_saerens_em_unlabeled": round(pi_em_b2, 5),
        "pi_historical_first25pct": round(pi_hist_25, 5),
        "full_cohort_comparison": {
            "zero_shot_unadjusted": _eval_shifted_b2(pi_src_b2),
            "unlabeled_saerens_em": _eval_shifted_b2(pi_em_b2),
            "oracle_target_prevalence_upper_bound": _eval_shifted_b2(pi_orc_b2),
        },
        "disjoint_75pct_holdout_comparison": {
            "zero_shot_unadjusted": _eval_shifted_b2(pi_src_b2, hist_eval_mask),
            "unlabeled_saerens_em_on_holdout": _eval_shifted_b2(
                saerens_em_target_prevalence(p1_b2[hist_eval_mask], pi_src_b2)[0], hist_eval_mask
            ),
            "historical_first25pct_prior": _eval_shifted_b2(pi_hist_25, hist_eval_mask),
            "oracle_holdout_prevalence": _eval_shifted_b2(float(np.mean(yb_b2[hist_eval_mask])), hist_eval_mask),
        },
        "prevalence_misspecification_sweep": [
            {"error_pp": round(100.0 * err, 2), **_eval_shifted_b2(np.clip(pi_orc_b2 + err, 0.005, 0.50))}
            for err in (-0.015, -0.010, -0.005, 0.0, 0.005, 0.010, 0.015, 0.020)
        ],
    }
    print("  [F10 Track B2 Shift Taxonomy]:", json.dumps(f10_b2_shift["full_cohort_comparison"], indent=2))

    # 3. F11: Per-Channel Availability by Site & Outcome + Measurement-Count Calibration + Complete-Panel Subset
    avail_rows: list[dict[str, object]] = []
    drawn_cnt_b = np.zeros(len(dfb6_m), dtype=int)
    for cid, (labs, _) in CMS_CPT_VENIPUNCTURE_CHANNELS.items():
        d_cols_a = [f"s2_{lb}_drawn" for lb in labs if f"s2_{lb}_drawn" in dfa6_m.columns]
        d_cols_b = [f"s2_{lb}_drawn" for lb in labs if f"s2_{lb}_drawn" in dfb6_m.columns]
        ma = np.any(dfa6_m[d_cols_a].to_numpy() == 1, axis=1)
        mb = np.any(dfb6_m[d_cols_b].to_numpy() == 1, axis=1)
        drawn_cnt_b += mb.astype(int)
        avail_rows.append(
            {
                "channel_id": cid,
                "bidmc_meas_all_pct": round(100.0 * float(np.mean(ma)), 1),
                "bidmc_meas_y0_pct": round(100.0 * float(np.mean(ma[ya6 == 0])), 1),
                "bidmc_meas_y1_pct": round(100.0 * float(np.mean(ma[ya6 == 1])), 1),
                "emory_meas_all_pct": round(100.0 * float(np.mean(mb)), 1),
                "emory_meas_y0_pct": round(100.0 * float(np.mean(mb[yb6 == 0])), 1),
                "emory_meas_y1_pct": round(100.0 * float(np.mean(mb[yb6 == 1])), 1),
            }
        )

    # Evaluate primary W=6h model across measurement-count strata & 100%-observed selected-panel subset
    chs_6 = build_sepsis_channels_by_mode(dfa6_m, summary_mode="point_in_time_last")
    fit_6 = _eval_compact_common_bagged(dfa6_m, ya6, dfb6_m, yb6, base_cols, chs_6, n_splits=10)
    p_cz_m = fit_6["p_routed_cz"]  # type: ignore[assignment]
    p_gl_m = fit_6["p_routed_glob"]  # type: ignore[assignment]

    meas_strata: dict[str, object] = {}
    for s_lbl, c_lo, c_hi in [("2_to_5_channels", 2, 5), ("6_to_8_channels", 6, 8), ("9_to_14_channels", 9, 14)]:
        ms = (drawn_cnt_b >= c_lo) & (drawn_cnt_b <= c_hi)
        if int(np.sum(ms)) >= 50 and int(np.sum(yb6[ms])) >= 5:
            cm_s = compute_calibration_metrics(yb6[ms], p_cz_m[ms], n_boot=0, return_curve=False)
            meas_strata[s_lbl] = {
                "n": int(np.sum(ms)),
                "events": int(np.sum(yb6[ms])),
                "prev_pct": round(100.0 * float(np.mean(yb6[ms])), 2),
                "citl_cz": round(float(cm_s["citl"]), 3),
                "ici_cz": round(float(cm_s["ici_loess"]), 4),
            }

    # Physical 100%-observed subset for the selected CZ panel (zero missing-value imputation on selected channels!)
    sel_cz_6 = fit_6["selected_cz"]  # type: ignore[assignment]
    obs_mask_b = np.ones(len(dfb6_m), dtype=bool)
    for cid in sel_cz_6:
        labs_c = CMS_CPT_VENIPUNCTURE_CHANNELS[cid][0]
        for lb in labs_c:
            if f"s2_{lb}_last" in dfb6_m.columns:
                obs_mask_b &= dfb6_m[f"s2_{lb}_last"].notna().to_numpy()
    complete_panel_subset = {
        "selected_cz_channels": sel_cz_6,
        "n_physically_observed_emory": int(np.sum(obs_mask_b)),
        "pct_of_emory_cohort": round(100.0 * float(np.mean(obs_mask_b)), 1),
        "events_observed": int(np.sum(yb6[obs_mask_b])),
        "auroc_cz_routed": round(float(roc_auc_score(yb6[obs_mask_b], p_cz_m[obs_mask_b])), 4)
        if int(np.sum(yb6[obs_mask_b])) >= 5
        else None,
        "auroc_glob_routed": round(float(roc_auc_score(yb6[obs_mask_b], p_gl_m[obs_mask_b])), 4)
        if int(np.sum(yb6[obs_mask_b])) >= 5
        else None,
    }

    out = {
        "elapsed_sec": round(time.time() - t0, 2),
        "f9_landmark_by_horizon_matrix": matrix_2d,
        "f9_w6h_pre_onset_lead_stratification": lead_time_strat_6h,
        "f9_sofa_incorporation_bias_audit": incorp_audit,
        "f10_prevalence_shift_taxonomy_b2": f10_b2_shift,
        "f11_channel_availability_by_site_and_outcome": avail_rows,
        "f11_measurement_count_strata": meas_strata,
        "f11_physically_observed_selected_panel_subset": complete_panel_subset,
    }
    BLOCKS_DIR.mkdir(parents=True, exist_ok=True)
    (BLOCKS_DIR / "block3_horizons_shift_missingness.json").write_text(json.dumps(out, indent=2))
    print(f"[Block 3] Saved checkpoint in {time.time()-t0:.1f}s.")
    return out


# ===========================================================================
# BLOCK 4: 9-Config Sensitivity Grid, Yale Campus C Transfer, Subgroups & CfBH (F13, F15)
# ===========================================================================


def _run_yale_cross_campus_pair(
    df_src: pd.DataFrame,
    y_src: np.ndarray,
    df_targets: dict[str, tuple[pd.DataFrame, np.ndarray, pd.DataFrame]],
    fair_base_cols: list[str],
    channels: list[ChannelDef],
    contract: ClinicalContract,
    hie_sched: PanelCostSchedule,
    *,
    budget: float = 3.0,
    n_splits: int = 6,
) -> dict[str, object]:
    """Train locked Common-Panel Bagged VOI on Campus A and evaluate zero-shot on Campus B & Campus C."""
    cost_map = {c.channel_id: float(c.cost) for c in channels}
    ch_cols_map = {c.channel_id: list(c.columns) for c in channels}

    # Subsample up to 70k source visits for Stage-A full-source discovery OOF
    rng = np.random.default_rng(5001)
    disc_idx = rng.choice(y_src.size, size=min(70000, y_src.size), replace=False)
    oof_disc = compute_oof_stage1_v2(
        df_src.iloc[disc_idx][fair_base_cols], y_src[disc_idx], n_folds=4, seed=5002, n_jobs=N_JOBS
    )
    d_gl = run_discovery_v3(
        df_src.iloc[disc_idx], y_src[disc_idx], fair_base_cols, channels, oof_disc, contract,
        objective="global_cmi", pilot_fraction=0.15, budget=budget, n_jobs=N_JOBS, seed=5003,
    )
    d_cz = run_discovery_v3(
        df_src.iloc[disc_idx], y_src[disc_idx], fair_base_cols, channels, oof_disc, contract,
        objective="collision_zone", pilot_fraction=0.15, budget=budget, n_jobs=N_JOBS, seed=5003,
    )
    cols_gl = [col for cid in d_gl.selected_ids for col in ch_cols_map[cid]]
    cols_cz = [col for cid in d_cz.selected_ids for col in ch_cols_map[cid]]
    pc_gl = hie_sched.panel_cost(d_gl.selected_ids, cost_map)
    pc_cz = hie_sched.panel_cost(d_cz.selected_ids, cost_map)

    tgt_names = list(df_targets.keys())
    tgt_dfs = [df_targets[k][0] for k in tgt_names]
    p1_ca_list: list[np.ndarray] = []
    p2_gl_ca_list: list[np.ndarray] = []
    p2_cz_ca_list: list[np.ndarray] = []
    y_ca_list: list[np.ndarray] = []
    p1_tgt_sums = {k: np.zeros(len(df_targets[k][0]), dtype=float) for k in tgt_names}
    p2_gl_tgt_sums = {k: np.zeros(len(df_targets[k][0]), dtype=float) for k in tgt_names}
    p2_cz_tgt_sums = {k: np.zeros(len(df_targets[k][0]), dtype=float) for k in tgt_names}
    diags_cz: list[dict[str, float]] = []

    for s in range(n_splits):
        rng_s = np.random.default_rng(5100 + s)
        perm = rng_s.permutation(y_src.size)
        n_tr = min(50000, int(0.65 * y_src.size))
        n_ca = min(15000, int(0.20 * y_src.size))
        tr, ca = np.sort(perm[:n_tr]), np.sort(perm[n_tr : n_tr + n_ca])
        oof_s = compute_oof_stage1_v2(
            df_src.iloc[tr][fair_base_cols], y_src[tr], n_folds=4, seed=5200 + s, n_jobs=N_JOBS
        )
        s1_cache = fit_stage1_predictions(
            df_src.iloc[tr], y_src[tr], df_src.iloc[ca], y_src[ca], tgt_dfs, fair_base_cols, seed=5400 + s
        )
        p1_ca_list.append(s1_cache[0])
        y_ca_list.append(y_src[ca])
        for ik, kname in enumerate(tgt_names):
            p1_tgt_sums[kname] += s1_cache[1][ik] / float(n_splits)

        # Fit Global S2 & Fluid S2 once per split and predict on all target campuses
        for tag_m, scols_m, s2m in [("gl", cols_gl, "global"), ("cz", cols_cz, "fluid_backbone")]:
            traces_m: list[dict[str, np.ndarray]] = []
            cal_m: dict[str, np.ndarray] = {}
            ev_list = evaluate_stage2_multi_test_v2(
                df_src.iloc[tr], y_src[tr], df_src.iloc[ca], y_src[ca],
                [(df_targets[k][0], df_targets[k][1]) for k in tgt_names],
                fair_base_cols, scols_m, oof_s, contract,
                stage2_mode=s2m, precomputed_stage1=s1_cache,
                out_routed_traces=traces_m, out_cal_trace=cal_m,
                n_jobs=N_JOBS, seed=5400 + s,
            )
            if tag_m == "gl":
                p2_gl_ca_list.append(cal_m["p2_cal"])
                for ik, kname in enumerate(tgt_names):
                    p2_gl_tgt_sums[kname] += traces_m[ik]["p2_te"] / float(n_splits)
            else:
                p2_cz_ca_list.append(cal_m["p2_cal"])
                diags_cz.append({"beta_cz": float(ev_list[0]["learned_beta_cz"])})
                for ik, kname in enumerate(tgt_names):
                    p2_cz_tgt_sums[kname] += traces_m[ik]["p2_te"] / float(n_splits)

    p1_ca_all = np.concatenate(p1_ca_list)
    p2_gl_ca_all = np.concatenate(p2_gl_ca_list)
    p2_cz_ca_all = np.concatenate(p2_cz_ca_list)
    y_ca_all = np.concatenate(y_ca_list)

    out_targets: dict[str, object] = {}
    for kname in tgt_names:
        _, y_t, meta_t = df_targets[kname]
        p1_t = p1_tgt_sums[kname]
        bg_gl = compute_bagged_voi_from_split_posteriors(
            y_t, p1_t, p2_gl_tgt_sums[kname], p1_ca_all, p2_gl_ca_all, pc_gl, contract
        )
        bg_cz = compute_bagged_voi_from_split_posteriors(
            y_t, p1_t, p2_cz_tgt_sums[kname], p1_ca_all, p2_cz_ca_all, pc_cz, contract
        )
        u_bs = unit_nb_ledger_vector(y_t, p1_t, np.zeros(y_t.size), contract)
        bt_gl = paired_bootstrap_ledger_diff(bg_cz["unit_nb"], bg_gl["unit_nb"], n_boot=10000, seed=301)  # type: ignore[arg-type]
        bt_bs = paired_bootstrap_ledger_diff(bg_cz["unit_nb"], u_bs, n_boot=10000, seed=302)  # type: ignore[arg-type]
        cal_cz = compute_calibration_metrics(y_t, bg_cz["p_routed"], n_boot=0, return_curve=False)  # type: ignore[arg-type]

        # Also test Saerens unlabeled EM recalibration when prevalence shifts substantially (e.g. Campus C all-comers 31.9% -> 15.6%!)
        pi_src = float(np.mean(y_src))
        pi_em, _ = saerens_em_target_prevalence(p1_t, pi_src)
        p1_em, _ = prior_odds_recalibrate_probabilities(p1_t, pi_src, pi_em)
        p2_cz_em, _ = prior_odds_recalibrate_probabilities(p2_cz_tgt_sums[kname], pi_src, pi_em)
        p2_gl_em, _ = prior_odds_recalibrate_probabilities(p2_gl_tgt_sums[kname], pi_src, pi_em)
        p1_ca_em, _ = prior_odds_recalibrate_probabilities(p1_ca_all, pi_src, pi_em)
        p2_cz_ca_em, _ = prior_odds_recalibrate_probabilities(p2_cz_ca_all, pi_src, pi_em)
        p2_gl_ca_em, _ = prior_odds_recalibrate_probabilities(p2_gl_ca_all, pi_src, pi_em)
        bg_cz_em = compute_bagged_voi_from_split_posteriors(
            y_t, p1_em, p2_cz_em, p1_ca_em, p2_cz_ca_em, pc_cz, contract
        )
        bg_gl_em = compute_bagged_voi_from_split_posteriors(
            y_t, p1_em, p2_gl_em, p1_ca_em, p2_gl_ca_em, pc_gl, contract
        )

        out_targets[kname] = {
            "n_visits": int(y_t.size),
            "events": int(np.sum(y_t)),
            "prevalence": round(float(np.mean(y_t)), 5),
            "selected_glob": list(d_gl.selected_ids),
            "panel_cost_glob_cfp": round(pc_gl, 5),
            "selected_cz": list(d_cz.selected_ids),
            "panel_cost_cz_cfp": round(pc_cz, 5),
            "mean_beta_cz": round(float(np.mean([d["beta_cz"] for d in diags_cz])), 4),
            "nb_stage1_base": round(float(np.mean(u_bs)), 4),
            "nb_global_cmi": bg_gl["nb_per_1k"],
            "nb_cz_fluid": bg_cz["nb_per_1k"],
            "q_rate_global_cmi": bg_gl["query_rate"],
            "q_rate_cz_fluid": bg_cz["query_rate"],
            "auroc_cz_fluid": bg_cz["auroc_routed"],
            "citl_cz_fluid": round(float(cal_cz["citl"]), 4),
            "ici_cz_fluid": round(float(cal_cz["ici_loess"]), 4),
            "boot_cz_vs_global_cmi": bt_gl,
            "boot_cz_vs_stage1_base": bt_bs,
            "unlabeled_saerens_em_recalibrated": {
                "pi_em_est": round(float(pi_em), 5),
                "nb_cz_fluid_em": bg_cz_em["nb_per_1k"],
                "nb_global_cmi_em": bg_gl_em["nb_per_1k"],
                "boot_cz_vs_glob_em": paired_bootstrap_ledger_diff(
                    bg_cz_em["unit_nb"], bg_gl_em["unit_nb"], n_boot=5000, seed=303  # type: ignore[arg-type]
                ),
            },
            "_traces": {
                "y_t": y_t,
                "meta_t": meta_t,
                "p1_t": p1_t,
                "bg_gl": bg_gl,
                "bg_cz": bg_cz,
                "y_ca_all": y_ca_all,
                "p_ca_cz": np.where(
                    empirical_martingale_voi(
                        p1_ca_all,
                        np.clip(np.digitize(p1_ca_all, bg_cz["edges"][1:-1], right=False), 0, 11),  # type: ignore[index]
                        bg_cz["bin_quantiles"],  # type: ignore[arg-type]
                        contract,
                    )
                    > pc_cz,
                    p2_cz_ca_all,
                    p1_ca_all,
                ),
            },
        }
    return out_targets


def run_block4_yale_and_sensitivity() -> dict[str, object]:
    t0 = time.time()
    print("[Block 4] Starting 9-Config Sensitivity Grid, Yale Campus C Transfer, Subgroups & CfBH Curve...")
    manifest = json.loads((PROC_DIR / "ws4tb_cohorts_manifest.json").read_text())
    base_sepsis: list[str] = list(manifest["sepsis_6h"]["base_columns"])
    dfa6 = pd.read_parquet(PROC_DIR / "sepsis_setA_bidmc_6h.parquet")
    dfb6 = pd.read_parquet(PROC_DIR / "sepsis_setB_emory_6h.parquet")
    dfa_m = dfa6[(dfa6["s2_Creatinine_drawn"] == 1) & (dfa6["s2_WBC_drawn"] == 1)].reset_index(drop=True)
    dfb_m = dfb6[(dfb6["s2_Creatinine_drawn"] == 1) & (dfb6["s2_WBC_drawn"] == 1)].reset_index(drop=True)
    ya_m = dfa_m["y_sepsis_future"].to_numpy(dtype=int)
    yb_m = dfb_m["y_sepsis_future"].to_numpy(dtype=int)

    # 1. F13: 9 Distinct (kappa_max in {12,20,32} x B_score in {4,8,12}) Configurations on Repaired Common-Panel Bagged VOI
    sched_s = PanelCostSchedule(venipuncture_cost=0.010, abg_extra_procedure_cost=0.010, unit_assay_scale=0.020)
    contract_s = ClinicalContract(cost_fp=1.0, cost_fn=15.0, target_precision=0.30, per_instance_query_cost=0.03)
    oof_a_s = compute_oof_stage1_v2(dfa_m[base_sepsis], ya_m, n_folds=4, seed=600, n_jobs=N_JOBS)
    splits_s = []
    for s in range(10):
        tr, ca, _ = make_stratified_3way_split(ya_m, seed=700 + s)
        oof_sp = compute_oof_stage1_v2(dfa_m.iloc[tr][base_sepsis], ya_m[tr], n_folds=4, seed=800 + s, n_jobs=N_JOBS)
        s1_sp = fit_stage1_predictions(dfa_m.iloc[tr], ya_m[tr], dfa_m.iloc[ca], ya_m[ca], [dfb_m], base_sepsis, seed=1500 + s)
        splits_s.append((s, tr, ca, oof_sp, s1_sp))
    p1_ens_s = np.mean([x[4][1][0] for x in splits_s], axis=0)
    p1_ca_s = np.concatenate([x[4][0] for x in splits_s])

    sens_9_grid: dict[str, list[dict[str, object]]] = {}
    for smode in ("point_in_time_last", "window_6h_summary"):
        chs_s = build_sepsis_channels_by_mode(dfa_m, summary_mode=smode)
        cost_map_s = {c.channel_id: float(c.cost) for c in chs_s}
        ch_cols_s = {c.channel_id: list(c.columns) for c in chs_s}
        rows_sm: list[dict[str, object]] = []
        for b_bins in (4, 8, 12):
            dg_b = run_discovery_v3(
                dfa_m, ya_m, base_sepsis, chs_s, oof_a_s, contract_s,
                objective="global_cmi", pilot_fraction=0.20, budget=3.2, n_bins=b_bins, n_jobs=N_JOBS, seed=900,
            )
            c_gl_b = [col for cid in dg_b.selected_ids for col in ch_cols_s[cid]]
            pc_gl_b = sched_s.panel_cost(dg_b.selected_ids, cost_map_s)
            p2_gl_te, p2_gl_ca = [], []
            for s, tr, ca, oof_sp, s1_sp in splits_s:
                fg = _fit_single_panel_stage2_split(
                    dfa_m.iloc[tr], ya_m[tr], dfa_m.iloc[ca], ya_m[ca], dfb_m,
                    base_sepsis, c_gl_b, oof_sp, s1_sp, contract_s, stage2_mode="global", n_jobs=N_JOBS, seed=1500 + s,
                )
                p2_gl_ca.append(fg["p2_cal"])  # type: ignore[arg-type]
                p2_gl_te.append(fg["p2_te"])  # type: ignore[arg-type]
            bg_gl_b = compute_bagged_voi_from_split_posteriors(
                yb_m, p1_ens_s, np.mean(p2_gl_te, axis=0), p1_ca_s, np.concatenate(p2_gl_ca), pc_gl_b, contract_s
            )

            for k_max in (12.0, 20.0, 32.0):
                dc_bk = run_discovery_v3(
                    dfa_m, ya_m, base_sepsis, chs_s, oof_a_s, contract_s,
                    objective="collision_zone", pilot_fraction=0.20, budget=3.2, kappa=k_max, n_bins=b_bins, n_jobs=N_JOBS, seed=900,
                )
                c_cz_bk = [col for cid in dc_bk.selected_ids for col in ch_cols_s[cid]]
                pc_cz_bk = sched_s.panel_cost(dc_bk.selected_ids, cost_map_s)
                p2_cz_te, p2_cz_ca = [], []
                for s, tr, ca, oof_sp, s1_sp in splits_s:
                    fc = _fit_single_panel_stage2_split(
                        dfa_m.iloc[tr], ya_m[tr], dfa_m.iloc[ca], ya_m[ca], dfb_m,
                        base_sepsis, c_cz_bk, oof_sp, s1_sp, contract_s,
                        stage2_mode="fluid_backbone", kappa_max_s2=k_max, min_cz_ess=240.0, n_jobs=N_JOBS, seed=1500 + s,
                    )
                    p2_cz_ca.append(fc["p2_cal"])  # type: ignore[arg-type]
                    p2_cz_te.append(fc["p2_te"])  # type: ignore[arg-type]
                bg_cz_bk = compute_bagged_voi_from_split_posteriors(
                    yb_m, p1_ens_s, np.mean(p2_cz_te, axis=0), p1_ca_s, np.concatenate(p2_cz_ca), pc_cz_bk, contract_s
                )
                bt_bk = paired_bootstrap_ledger_diff(bg_cz_bk["unit_nb"], bg_gl_b["unit_nb"], n_boot=10000, seed=401)  # type: ignore[arg-type]
                _s0, tr0, ca0, oof0, s10 = splits_s[0]
                fc_160 = _fit_single_panel_stage2_split(
                    dfa_m.iloc[tr0], ya_m[tr0], dfa_m.iloc[ca0], ya_m[ca0], dfb_m,
                    base_sepsis, c_cz_bk, oof0, s10, contract_s,
                    stage2_mode="fluid_backbone", kappa_max_s2=k_max, min_cz_ess=160.0, n_jobs=N_JOBS, seed=1500,
                )
                fc_320 = _fit_single_panel_stage2_split(
                    dfa_m.iloc[tr0], ya_m[tr0], dfa_m.iloc[ca0], ya_m[ca0], dfb_m,
                    base_sepsis, c_cz_bk, oof0, s10, contract_s,
                    stage2_mode="fluid_backbone", kappa_max_s2=k_max, min_cz_ess=320.0, n_jobs=N_JOBS, seed=1500,
                )
                m_min_diff = float(np.max(np.abs(fc_160["p2_te"] - fc_320["p2_te"])))  # type: ignore[operator]

                rows_sm.append(
                    {
                        "b_score_bins": b_bins,
                        "kappa_max": k_max,
                        "selected_cz": list(dc_bk.selected_ids),
                        "selected_glob": list(dg_b.selected_ids),
                        "nb_cz_bagged": bg_cz_bk["nb_per_1k"],
                        "nb_glob_bagged": bg_gl_b["nb_per_1k"],
                        "delta_nb": bt_bk["diff_mean"],
                        "ci95_low": bt_bk["ci95_low"],
                        "ci95_high": bt_bk["ci95_high"],
                        "p_two_sided_dh": bt_bk["p_two_sided_dh"],
                        "m_min_160_vs_320_max_diff": round(m_min_diff, 8),
                    }
                )
                print(
                    f"  [F13 Sens {smode} B_bins={b_bins}, k={k_max}] Delta={bt_bk['diff_mean']:+.3f} "
                    f"[{bt_bk['ci95_low']:+.3f}, {bt_bk['ci95_high']:+.3f}] (p={bt_bk['p_two_sided_dh']})"
                )
        sens_9_grid[smode] = rows_sm

    # 2. F13 & F15: Yale Cross-Campus (Campus A -> Campus B AND Untested Campus A -> Campus C)
    yale_meta_cfg = manifest["yale_ed"]
    meta_full = pd.read_parquet(PROC_DIR / "yale_trackb_meta.parquet")
    df_full = pd.read_parquet(PROC_DIR / "yale_ed_full.parquet")
    fair_base = [
        c for c in yale_meta_cfg["base_columns"] if c in df_full.columns and c not in FAIRNESS_EXCLUDED_BASE_COLS
    ]
    yale_channels = [
        ChannelDef(cid, tuple(cinfo["columns"]), cost=float(cinfo["cost"]))
        for cid, cinfo in yale_meta_cfg["channels"].items()
    ]
    # Explicit HIE / EHR retrieval ledger: connection overhead 0.005 + 0.010 * sum(cost_m)
    hie_sched_esi = PanelCostSchedule(venipuncture_cost=0.005, abg_extra_procedure_cost=0.0, unit_assay_scale=0.010, abg_channel_ids=())
    hie_sched_all = PanelCostSchedule(venipuncture_cost=0.004, abg_extra_procedure_cost=0.0, unit_assay_scale=0.008, abg_channel_ids=())

    m_a = (meta_full["dep_name"] == "A").to_numpy()
    m_b = (meta_full["dep_name"] == "B").to_numpy()
    m_c = (meta_full["dep_name"] == "C").to_numpy()
    esi_m = meta_full["esi"].isin(["4", "5", "4.0", "5.0"]).to_numpy()

    # Low-Acuity ESI 4-5 Cross-Campus: Campus A (70,360) -> Campus B (51,205) & Campus C (31,430)
    contract_esi = ClinicalContract(cost_fp=1.0, cost_fn=24.0, target_precision=0.25, per_instance_query_cost=0.025)
    esi_targets = {
        "campus_b_esi45": (
            df_full.iloc[np.flatnonzero(m_b & esi_m)].reset_index(drop=True),
            df_full.iloc[np.flatnonzero(m_b & esi_m)]["y_admit"].to_numpy(dtype=int),
            meta_full.iloc[np.flatnonzero(m_b & esi_m)].reset_index(drop=True),
        ),
        "campus_c_esi45_untested": (
            df_full.iloc[np.flatnonzero(m_c & esi_m)].reset_index(drop=True),
            df_full.iloc[np.flatnonzero(m_c & esi_m)]["y_admit"].to_numpy(dtype=int),
            meta_full.iloc[np.flatnonzero(m_c & esi_m)].reset_index(drop=True),
        ),
    }
    res_esi_xc = _run_yale_cross_campus_pair(
        df_full.iloc[np.flatnonzero(m_a & esi_m)].reset_index(drop=True),
        df_full.iloc[np.flatnonzero(m_a & esi_m)]["y_admit"].to_numpy(dtype=int),
        esi_targets,
        fair_base,
        yale_channels,
        contract_esi,
        hie_sched_esi,
        budget=3.0,
        n_splits=6,
    )

    # All-Comers Cross-Campus: Campus A (322,283) -> Campus B (166,497) & Campus C (71,706)
    contract_all = ClinicalContract(cost_fp=1.0, cost_fn=9.0, target_precision=0.45, per_instance_query_cost=0.020)
    all_targets = {
        "campus_b_allcomers": (
            df_full.iloc[np.flatnonzero(m_b)].reset_index(drop=True),
            df_full.iloc[np.flatnonzero(m_b)]["y_admit"].to_numpy(dtype=int),
            meta_full.iloc[np.flatnonzero(m_b)].reset_index(drop=True),
        ),
        "campus_c_allcomers_untested": (
            df_full.iloc[np.flatnonzero(m_c)].reset_index(drop=True),
            df_full.iloc[np.flatnonzero(m_c)]["y_admit"].to_numpy(dtype=int),
            meta_full.iloc[np.flatnonzero(m_c)].reset_index(drop=True),
        ),
    }
    res_all_xc = _run_yale_cross_campus_pair(
        df_full.iloc[np.flatnonzero(m_a)].reset_index(drop=True),
        df_full.iloc[np.flatnonzero(m_a)]["y_admit"].to_numpy(dtype=int),
        all_targets,
        fair_base,
        yale_channels,
        contract_all,
        hie_sched_all,
        budget=3.0,
        n_splits=6,
    )

    # 3. F15: Subgroup Decision Utility & Calibration Audit on Pooled External Campuses B + C ESI 4-5 (vs Stage-1 AND vs Global-CMI)
    tr_b = res_esi_xc["campus_b_esi45"].pop("_traces")  # type: ignore[union-attr]
    tr_c = res_esi_xc["campus_c_esi45_untested"].pop("_traces")  # type: ignore[union-attr]
    res_all_xc["campus_b_allcomers"].pop("_traces")  # type: ignore[union-attr]
    res_all_xc["campus_c_allcomers_untested"].pop("_traces")  # type: ignore[union-attr]

    y_bc = np.concatenate([tr_b["y_t"], tr_c["y_t"]])
    meta_bc = pd.concat([tr_b["meta_t"], tr_c["meta_t"]], ignore_index=True)
    p1_bc = np.concatenate([tr_b["p1_t"], tr_c["p1_t"]])
    p_gl_bc = np.concatenate([tr_b["bg_gl"]["p_routed"], tr_c["bg_gl"]["p_routed"]])
    q_gl_bc = np.concatenate([tr_b["bg_gl"]["q_mask"], tr_c["bg_gl"]["q_mask"]])
    u_gl_bc = np.concatenate([tr_b["bg_gl"]["unit_nb"], tr_c["bg_gl"]["unit_nb"]])
    p_cz_bc = np.concatenate([tr_b["bg_cz"]["p_routed"], tr_c["bg_cz"]["p_routed"]])
    q_cz_bc = np.concatenate([tr_b["bg_cz"]["q_mask"], tr_c["bg_cz"]["q_mask"]])
    u_cz_bc = np.concatenate([tr_b["bg_cz"]["unit_nb"], tr_c["bg_cz"]["unit_nb"]])
    u_bs_bc = unit_nb_ledger_vector(y_bc, p1_bc, np.zeros(y_bc.size), contract_esi)

    subgroup_rows: list[dict[str, object]] = []
    tau_esi = contract_esi.tau_star
    for attr in ("race", "lang", "insurance_status"):
        vals_a = meta_bc[attr].fillna("Missing").astype(str).to_numpy()
        for gval in sorted(np.unique(vals_a)):
            if gval in {"Missing", "nan", "None", "Unknown", "<NA>", ""}:
                continue
            gm = vals_a == gval
            n_g = int(np.sum(gm))
            ev_g = int(np.sum(y_bc[gm]))
            if n_g < 400 or ev_g < 8:
                continue
            yg = y_bc[gm]
            bt_vs_bs = paired_bootstrap_ledger_diff(u_cz_bc[gm], u_bs_bc[gm], n_boot=5000, seed=601)
            bt_vs_gl = paired_bootstrap_ledger_diff(u_cz_bc[gm], u_gl_bc[gm], n_boot=5000, seed=602)
            act_cz = p_cz_bc[gm] >= tau_esi
            act_gl = p_gl_bc[gm] >= tau_esi
            subgroup_rows.append(
                {
                    "attribute": attr,
                    "group": gval,
                    "n": n_g,
                    "events": ev_g,
                    "prevalence_pct": round(100.0 * float(np.mean(yg)), 2),
                    "q_rate_cz_pct": round(100.0 * float(np.mean(q_cz_bc[gm])), 1),
                    "q_rate_glob_pct": round(100.0 * float(np.mean(q_gl_bc[gm])), 1),
                    "sens_cz_pct": round(100.0 * float(np.sum(act_cz & (yg == 1))) / float(ev_g), 1),
                    "sens_glob_pct": round(100.0 * float(np.sum(act_gl & (yg == 1))) / float(ev_g), 1),
                    "fp_per_1k_cz": round(1000.0 * float(np.sum(act_cz & (yg == 0))) / float(n_g), 1),
                    "fp_per_1k_glob": round(1000.0 * float(np.sum(act_gl & (yg == 0))) / float(n_g), 1),
                    "citl_cz": round(solve_citl_intercept(yg, p_cz_bc[gm]), 3),
                    "nb_stage1": round(float(np.mean(u_bs_bc[gm])), 3),
                    "nb_glob_cmi": round(float(np.mean(u_gl_bc[gm])), 3),
                    "nb_cz_fluid": round(float(np.mean(u_cz_bc[gm])), 3),
                    "delta_cz_vs_stage1": bt_vs_bs["diff_mean"],
                    "ci95_cz_vs_stage1": [bt_vs_bs["ci95_low"], bt_vs_bs["ci95_high"]],
                    "delta_cz_vs_glob_cmi": bt_vs_gl["diff_mean"],
                    "ci95_cz_vs_glob_cmi": [bt_vs_gl["ci95_low"], bt_vs_gl["ci95_high"]],
                    "p_cz_vs_glob_cmi": bt_vs_gl["p_two_sided_dh"],
                }
            )

    # 4. F15: Repeated-Subsample Campus B Conformal (CfBH) Burn-In Governance Curve (R=200 disjoint splits)
    yb_b = tr_b["y_t"]
    p_cz_b = tr_b["bg_cz"]["p_routed"]
    y_ca_a = tr_b["y_ca_all"]
    p_ca_a = tr_b["p_ca_cz"]
    fdr_alpha = 1.0 - contract_esi.target_precision  # 0.75

    cf_zero = cfbh_selection(y_ca_a, p_ca_a, yb_b, p_cz_b, fdr_alpha, seed=701)
    burn_curve: list[dict[str, object]] = [
        {
            "n_burn": 0,
            "mode": "zero_shot_source_campus_a_cal",
            "n_reps": 1,
            "mean_fdp_empirical_fdr": round(float(cf_zero["fdp"]), 4),
            "sd_fdp": 0.0,
            "q95_fdp": round(float(cf_zero["fdp"]), 4),
            "prob_fdp_gt_alpha": float(cf_zero["fdp"] > fdr_alpha),
            "mean_recall_pct": round(100.0 * float(cf_zero["recall"]), 2),
            "mean_tp_per_1k": round(
                1000.0 * float(cf_zero["recall"]) * float(np.mean(yb_b)), 2
            ),
            "mean_alerts_per_100": round(100.0 * float(cf_zero["n_selected"]) / float(yb_b.size), 2),
        }
    ]
    n_tot_b = yb_b.size
    for nb_sz in (250, 500, 1000, 2000, 4000):
        fdps, recs, alts, tps = [], [], [], []
        rng_cf = np.random.default_rng(7000 + nb_sz)
        for rep in range(200):
            perm_b = rng_cf.permutation(n_tot_b)
            b_idx, e_idx = perm_b[:nb_sz], perm_b[nb_sz:]
            cfr = cfbh_selection(yb_b[b_idx], p_cz_b[b_idx], yb_b[e_idx], p_cz_b[e_idx], fdr_alpha, seed=8000 + rep)
            fdps.append(float(cfr["fdp"]))
            recs.append(100.0 * float(cfr["recall"]))
            alts.append(100.0 * float(cfr["n_selected"]) / float(e_idx.size))
            tps.append(1000.0 * float(cfr["recall"]) * float(np.mean(yb_b[e_idx])))
        fdp_arr = np.asarray(fdps)
        burn_curve.append(
            {
                "n_burn": nb_sz,
                "mode": "disjoint_local_campus_b_burnin",
                "n_eval": int(n_tot_b - nb_sz),
                "n_reps": 200,
                "mean_fdp_empirical_fdr": round(float(np.mean(fdp_arr)), 4),
                "sd_fdp": round(float(np.std(fdp_arr, ddof=1)), 4),
                "q025_fdp": round(float(np.quantile(fdp_arr, 0.025)), 4),
                "q975_fdp": round(float(np.quantile(fdp_arr, 0.975)), 4),
                "prob_fdp_gt_alpha": round(float(np.mean(fdp_arr > fdr_alpha)), 4),
                "mean_recall_pct": round(float(np.mean(recs)), 2),
                "sd_recall_pct": round(float(np.std(recs, ddof=1)), 2),
                "mean_tp_per_1k": round(float(np.mean(tps)), 2),
                "mean_alerts_per_100": round(float(np.mean(alts)), 3),
            }
        )
    print("  [F15 Campus B CfBH Burn-In Curve]:", json.dumps(burn_curve, indent=2))

    out = {
        "elapsed_sec": round(time.time() - t0, 2),
        "f13_yale_identifier_disclosure": (
            "In the public Hong et al. (2018) Yale ED repository, pat_mrn_id has 560,486 unique values across "
            "560,486 rows (encounter-level index; longitudinal repeat-visit patient identifiers are unavailable)."
        ),
        "f13_sepsis_9_distinct_hyperparameter_grid": sens_9_grid,
        "f13_yale_cross_campus_esi45": res_esi_xc,
        "f13_yale_cross_campus_allcomers": res_all_xc,
        "f15_yale_external_subgroup_utility_and_calibration": subgroup_rows,
        "f15_campus_b_cfbh_repeated_burnin_curve": burn_curve,
    }
    BLOCKS_DIR.mkdir(parents=True, exist_ok=True)
    (BLOCKS_DIR / "block4_yale_and_sensitivity.json").write_text(json.dumps(out, indent=2))
    print(f"[Block 4] Saved checkpoint in {time.time()-t0:.1f}s.")
    return out


def merge_blocks() -> None:
    merged: dict[str, object] = {}
    for k, fname in [
        ("block1_synthetic", "block1_synthetic.json"),
        ("block2_sepsis_primary_ledger", "block2_sepsis_primary_ledger.json"),
        ("block3_horizons_shift_missingness", "block3_horizons_shift_missingness.json"),
        ("block4_yale_and_sensitivity", "block4_yale_and_sensitivity.json"),
    ]:
        p = BLOCKS_DIR / fname
        if p.exists():
            merged[k] = json.loads(p.read_text())
    OUT_MERGED.write_text(json.dumps(merged, indent=2))
    print(f"[Merge] Wrote {OUT_MERGED} ({len(merged)}/4 blocks present).")


def main() -> None:
    blk = "all"
    if "--block" in sys.argv:
        blk = sys.argv[sys.argv.index("--block") + 1]
    if blk in ("1", "all"):
        run_block1_synthetic()
    if blk in ("2", "all"):
        run_block2_sepsis_primary()
    if blk in ("3", "all"):
        run_block3_horizons_shift_missingness()
    if blk in ("4", "all"):
        run_block4_yale_and_sensitivity()
    merge_blocks()


if __name__ == "__main__":
    main()
