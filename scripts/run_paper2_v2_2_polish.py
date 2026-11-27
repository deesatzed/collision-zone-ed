"""Paper 2 v2.2 Surgical Empirical Additions (`Items 2, 3, 4, 6` of Round-2 Review):
  1. Complete `CohBag` (Deployable Single-Draw Binary Bagged VOI) 10,000-rep paired bootstraps
     on primary Track B1b (including vs. `Covert-AFA`, `Abl. 1`, `Abl. 2`, and `2x2` interaction).
  2. Exact CMIO Clinical Exchange-Rate Table across tau in {4.0%, 6.25%, 10.0%, 12.5%} on Emory
     Track B1b (N = 6,463) for both Deployable Binary Bagged VOI (`CohBag`) and Symmetric
     Split-Averaged Ensemble (`FracBillEns`): cases caught (`TP/1k`), false alerts (`FP/1k`),
     alerts spared vs. `Global-CMI` & `Stage-1`, precision (`PPV%`), and `NB(tau)`.
  3. Canonical UCI tabular validation check for `CovertGreedyCMIAFA` (`breast_cancer` and `diabetes`
     serum channels, 10 splits each), verifying faithful global CMI / AUROC recovery vs. random
     acquisition and 100%-feature oracle.
  4. Yale Track B4a (`Campus A -> Campus B`, N_ext = 51,205) `CfBH` cross-campus failure & label-free
     prior-odds intercept alignment (`Delta_0`) mitigation audit.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for p in (str(REPO_ROOT), str(SRC_ROOT), str(SCRIPTS_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pandas as pd
from run_paper2_v2_suite import (
    FAIRNESS_EXCLUDED_BASE_COLS,
    PROC_DIR,
    _build_venipuncture_channels,
    make_patient_grouped_split,
    make_stratified_3way_split,
)
from scipy.special import logit
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.metrics import log_loss, roc_auc_score

from collision_zone.three_predicates import (
    ChannelDef,
    ClinicalContract,
    cfbh_selection,
    empirical_martingale_voi,
    fit_stage1_predictions,
    net_benefit_per_1k,
)
from collision_zone.three_predicates_v2 import (
    _unit_nb_vector,
    compute_calibration_metrics,
    compute_oof_stage1_v2,
    evaluate_covert_greedy_cmi_afa,
    evaluate_stage2_multi_test_v2,
    paired_bootstrap_interaction,
    paired_bootstrap_nb_diff_v2,
    prior_odds_recalibrate_probabilities,
    run_sequential_discovery_v2,
    standardized_net_benefit,
)

OUT_JSON = REPO_ROOT / "results" / "paper2_v2_strengthening_results.json"


def _exchange_row_at_tau(
    y: np.ndarray,
    p_arm: np.ndarray,
    q_arm: np.ndarray,
    p_glob: np.ndarray,
    q_glob: np.ndarray,
    p_base: np.ndarray,
    tau: float,
    c_query: float = 0.03,
) -> dict[str, float]:
    """Compute clinical exchange-rate metrics per 1,000 arrivals at threshold tau.
    Under rational Bayes alignment at threshold tau, C_FP = 1 and C_FN = (1 - tau) / tau.
    """
    yv = np.asarray(y, dtype=float)
    odds = tau / (1.0 - tau)
    c_fn_tau = (1.0 - tau) / tau
    gamma_q = c_query / c_fn_tau  # = c_query * tau / (1 - tau) when C_FP = 1

    def _stats(p: np.ndarray, q: np.ndarray) -> tuple[float, float, float, float, float]:
        act = (p >= tau).astype(float)
        tp = 1000.0 * float(np.mean(act * yv))
        fp = 1000.0 * float(np.mean(act * (1.0 - yv)))
        tot_alerts = tp + fp
        ppv = float(tp / tot_alerts) if tot_alerts > 1e-9 else 0.0
        q_1k = 1000.0 * float(np.mean(q))
        nb = tp - odds * fp - gamma_q * q_1k
        return tp, fp, ppv, q_1k, nb

    tp_a, fp_a, ppv_a, q_a, nb_a = _stats(p_arm, q_arm)
    tp_g, fp_g, ppv_g, _q_g, nb_g = _stats(p_glob, q_glob)
    tp_b, fp_b, _ppv_b, _, nb_b = _stats(p_base, np.zeros_like(yv))

    # Paired B=10,000 patient-level bootstrap 95% CIs and Davison-Hinkley two-sided p-values
    # for Delta TP/1k and False Alerts Spared (-Delta FP/1k) vs Global-CMI
    dtp_vec = 1000.0 * (((p_arm >= tau).astype(float) - (p_glob >= tau).astype(float)) * yv)
    spared_fp_vec = 1000.0 * (((p_glob >= tau).astype(float) - (p_arm >= tau).astype(float)) * (1.0 - yv))
    n_pts = yv.size
    rng_ex = np.random.default_rng(round(tau * 100000) + 901)
    probs_u = np.full(n_pts, 1.0 / float(n_pts))
    dtp_boots = np.empty(10000, dtype=float)
    sp_boots = np.empty(10000, dtype=float)
    for st in range(0, 10000, 2500):
        en = min(10000, st + 2500)
        cnts = rng_ex.multinomial(n_pts, probs_u, size=en - st).astype(float) / float(n_pts)
        dtp_boots[st:en] = cnts @ dtp_vec
        sp_boots[st:en] = cnts @ spared_fp_vec

    def _ci_dh(b_arr: np.ndarray) -> tuple[list[float], float]:
        lo, hi = np.quantile(b_arr, [0.025, 0.975])
        k_le = int(np.sum(b_arr <= 0.0))
        k_ge = int(np.sum(b_arr >= 0.0))
        p_dh = min(1.0, 2.0 * float(min(k_le, k_ge) + 1) / 10001.0)
        return [round(float(lo), 2), round(float(hi), 2)], round(float(p_dh), 4)

    dtp_ci95, dtp_p = _ci_dh(dtp_boots)
    sp_ci95, sp_p = _ci_dh(sp_boots)

    return {
        "tau": round(tau, 4),
        "indifference_fp_per_tp": round(c_fn_tau, 2),
        "cz_tp_per_1k": round(tp_a, 2),
        "cz_fp_per_1k": round(fp_a, 2),
        "cz_alerts_per_100_shift": round((tp_a + fp_a) / 10.0, 2),
        "cz_ppv_pct": round(100.0 * ppv_a, 2),
        "cz_queries_per_1k": round(q_a, 1),
        "cz_nb_per_1k": round(nb_a, 2),
        "glob_tp_per_1k": round(tp_g, 2),
        "glob_fp_per_1k": round(fp_g, 2),
        "glob_ppv_pct": round(100.0 * ppv_g, 2),
        "glob_nb_per_1k": round(nb_g, 2),
        "base_tp_per_1k": round(tp_b, 2),
        "base_fp_per_1k": round(fp_b, 2),
        "base_nb_per_1k": round(nb_b, 2),
        "delta_tp_vs_glob_per_1k": round(tp_a - tp_g, 2),
        "delta_tp_vs_glob_ci95": dtp_ci95,
        "delta_tp_vs_glob_p_dh": dtp_p,
        "alerts_spared_vs_glob_per_1k": round(fp_g - fp_a, 2),
        "alerts_spared_vs_glob_ci95": sp_ci95,
        "alerts_spared_vs_glob_p_dh": sp_p,
        "delta_nb_vs_glob_per_1k": round(nb_a - nb_g, 2),
        "delta_tp_vs_base_per_1k": round(tp_a - tp_b, 2),
        "alerts_spared_vs_base_per_1k": round(fp_b - fp_a, 2),
        "delta_nb_vs_base_per_1k": round(nb_a - nb_b, 2),
    }


def run_b1b_exchange_and_bagged_audit(manifest: dict[str, object]) -> dict[str, object]:
    t0 = time.time()
    df_a6 = pd.read_parquet(PROC_DIR / "sepsis_setA_bidmc_6h.parquet")
    df_b6 = pd.read_parquet(PROC_DIR / "sepsis_setB_emory_6h.parquet")
    df_a = df_a6[(df_a6["s2_Creatinine_drawn"] == 1) & (df_a6["s2_WBC_drawn"] == 1)].reset_index(drop=True)
    df_b = df_b6[(df_b6["s2_Creatinine_drawn"] == 1) & (df_b6["s2_WBC_drawn"] == 1)].reset_index(drop=True)
    base_cols: list[str] = list(manifest["sepsis_6h"]["base_columns"])  # type: ignore[index]
    channels = _build_venipuncture_channels(df_a)
    ch_cols_map = {c.channel_id: list(c.columns) for c in channels}

    contract = ClinicalContract(cost_fp=1.0, cost_fn=15.0, target_precision=0.30, per_instance_query_cost=0.03)
    y_a = df_a["y_sepsis_future"].to_numpy(dtype=int)
    y_b = df_b["y_sepsis_future"].to_numpy(dtype=int)
    n_b = y_b.size
    n_splits = 10

    arm_names = [
        "paper1_global_cmi_global_s2",
        "ablation_global_cmi_fluid_s2",
        "ablation_cz_disc_global_s2",
        "paper2_cz_disc_fluid_s2",
        "covert2023_greedy_cmi_afa",
    ]
    ens_p_auto_sum = {a: np.zeros(n_b, dtype=float) for a in ["stage1_base"] + arm_names}
    ens_q_auto_sum = {a: np.zeros(n_b, dtype=float) for a in ["stage1_base"] + arm_names}
    ens_p2_sum = {a: np.zeros(n_b, dtype=float) for a in ["stage1_base"] + arm_names}
    coh_p1_cal_all: list[np.ndarray] = []
    coh_p2_cal_by_arm: dict[str, list[np.ndarray]] = {a: [] for a in arm_names if a != "covert2023_greedy_cmi_afa"}
    afa_q_maj_votes = np.zeros(n_b, dtype=float)

    print(f"[B1b Polish] Re-verifying 10 splits (BIDMC {len(df_a):,} -> Emory {n_b:,})...")
    for s in range(n_splits):
        tr, ca, te = make_stratified_3way_split(y_a, seed=700 + s)
        oof_p = compute_oof_stage1_v2(df_a.iloc[tr][base_cols], y_a[tr], n_folds=4, seed=800 + s, n_jobs=4)
        d_glob = run_sequential_discovery_v2(
            df_a.iloc[tr], y_a[tr], base_cols, channels, oof_p, contract,
            method="global_cmi", pilot_fraction=0.20, budget=3.2, n_jobs=4, seed=900 + s
        ).outcome
        d_cz = run_sequential_discovery_v2(
            df_a.iloc[tr], y_a[tr], base_cols, channels, oof_p, contract,
            method="collision_zone", pilot_fraction=0.20, budget=3.2, n_jobs=4, seed=900 + s
        ).outcome
        cols_glob = [col for cid in d_glob.selected_ids for col in ch_cols_map[cid]]
        cols_cz = [col for cid in d_cz.selected_ids for col in ch_cols_map[cid]]
        s1_cache = fit_stage1_predictions(
            df_a.iloc[tr], y_a[tr], df_a.iloc[ca], y_a[ca], [df_a.iloc[te], df_b], base_cols, seed=1500 + s
        )
        test_pairs = [(df_a.iloc[te], y_a[te]), (df_b, y_b)]
        p1_emory = s1_cache[1][1]
        ens_p_auto_sum["stage1_base"] += p1_emory
        ens_p2_sum["stage1_base"] += p1_emory
        coh_p1_cal_all.append(s1_cache[0])

        for arm_name, scol, s2_mode in [
            ("paper1_global_cmi_global_s2", cols_glob, "global"),
            ("ablation_global_cmi_fluid_s2", cols_glob, "fluid_backbone"),
            ("ablation_cz_disc_global_s2", cols_cz, "global"),
            ("paper2_cz_disc_fluid_s2", cols_cz, "fluid_backbone"),
        ]:
            traces: list[dict[str, np.ndarray]] = []
            cal_tr: dict[str, np.ndarray] = {}
            evaluate_stage2_multi_test_v2(
                df_a.iloc[tr], y_a[tr], df_a.iloc[ca], y_a[ca], test_pairs,
                base_cols, scol, oof_p, contract,
                stage2_mode=s2_mode, voi_mode="empirical_martingale",
                precomputed_stage1=s1_cache, out_routed_traces=traces, out_cal_trace=cal_tr,
                n_jobs=4, seed=1500 + s,
            )
            ens_p_auto_sum[arm_name] += traces[1]["p_auto_te"]
            ens_q_auto_sum[arm_name] += traces[1]["auto_te"].astype(float)
            ens_p2_sum[arm_name] += traces[1]["p2_te"]
            coh_p2_cal_by_arm[arm_name].append(cal_tr["p2_cal"])

        afa_traces: list[dict[str, np.ndarray]] = []
        evaluate_covert_greedy_cmi_afa(
            df_a.iloc[tr], y_a[tr], df_a.iloc[ca], y_a[ca], test_pairs,
            base_cols, channels, oof_p, s1_cache, contract,
            budget=3.2, out_routed_traces=afa_traces, n_jobs=4, seed=1500 + s,
        )
        ens_p_auto_sum["covert2023_greedy_cmi_afa"] += afa_traces[1]["p_auto_te"]
        ens_q_auto_sum["covert2023_greedy_cmi_afa"] += afa_traces[1]["auto_te"].astype(float)
        ens_p2_sum["covert2023_greedy_cmi_afa"] += afa_traces[1]["p2_te"]
        afa_q_maj_votes += afa_traces[1]["auto_te"].astype(float)

    prev_b = float(np.mean(y_b))
    p1_cal_cat = np.concatenate(coh_p1_cal_all)
    p1_ens_b = ens_p_auto_sum["stage1_base"] / float(n_splits)

    # Symmetric ensemble predictions
    ens_preds = {
        a: (ens_p_auto_sum[a] / float(n_splits), ens_q_auto_sum[a] / float(n_splits))
        for a in ["stage1_base"] + arm_names
    }

    # Deployable Single-Draw Binary Bagged VOI (`CohBag`) predictions
    coh_preds: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "stage1_base": (p1_ens_b, np.zeros(n_b, dtype=bool))
    }
    coh_summary: dict[str, dict[str, float]] = {
        "stage1_base": {
            "coh_nb_auto": round(net_benefit_per_1k(y_b, p1_ens_b, np.zeros(n_b, dtype=bool), contract), 4),
            "coh_snb_auto": round(standardized_net_benefit(net_benefit_per_1k(y_b, p1_ens_b, np.zeros(n_b, dtype=bool), contract), prev_b), 4),
            "coh_query_rate_binary": 0.0,
        }
    }
    for a in ["paper1_global_cmi_global_s2", "ablation_global_cmi_fluid_s2", "ablation_cz_disc_global_s2", "paper2_cz_disc_fluid_s2"]:
        p2_cal_cat = np.concatenate(coh_p2_cal_by_arm[a])
        d_cal_coh = logit(np.clip(p2_cal_cat, 1e-6, 1 - 1e-6)) - logit(np.clip(p1_cal_cat, 1e-6, 1 - 1e-6))
        edges_c = np.quantile(p1_cal_cat, np.linspace(0.0, 1.0, 13))
        edges_c[0], edges_c[-1] = 0.0, 1.0
        b_cal_c = np.clip(np.digitize(p1_cal_cat, edges_c[1:-1], right=False), 0, 11)
        qs_c = np.linspace(0.02, 0.98, 33)
        gq_c = np.quantile(d_cal_coh, qs_c)
        bq_c = np.array([
            np.quantile(d_cal_coh[b_cal_c == b], qs_c) if int(np.sum(b_cal_c == b)) >= 15 else gq_c
            for b in range(12)
        ])
        b_te_c = np.clip(np.digitize(p1_ens_b, edges_c[1:-1], right=False), 0, 11)
        voi_c = empirical_martingale_voi(p1_ens_b, b_te_c, bq_c, contract)
        q_c = voi_c > contract.per_instance_query_cost
        p2a_c = ens_p2_sum[a] / float(n_splits)
        p_c = np.where(q_c, p2a_c, p1_ens_b)
        nb_c = net_benefit_per_1k(y_b, p_c, q_c, contract)
        coh_preds[a] = (p_c, q_c)
        coh_summary[a] = {
            "coh_nb_auto": round(nb_c, 4),
            "coh_snb_auto": round(standardized_net_benefit(nb_c, prev_b), 4),
            "coh_query_rate_binary": round(float(np.mean(q_c)), 4),
            "coh_auroc_routed": round(float(roc_auc_score(y_b, p_c)), 4),
        }

    # For Covert-AFA single-decision binary deployment: majority vote Q_afa_bin = (mean_q >= 0.5) routing bagged p2_afa vs p1
    q_afa_bin = (afa_q_maj_votes / float(n_splits)) >= 0.5
    p2_afa_bag = ens_p2_sum["covert2023_greedy_cmi_afa"] / float(n_splits)
    p_afa_bin = np.where(q_afa_bin, p2_afa_bag, p1_ens_b)
    nb_afa_bin = net_benefit_per_1k(y_b, p_afa_bin, q_afa_bin, contract)
    coh_preds["covert2023_greedy_cmi_afa"] = (p_afa_bin, q_afa_bin)
    coh_summary["covert2023_greedy_cmi_afa"] = {
        "coh_nb_auto": round(nb_afa_bin, 4),
        "coh_snb_auto": round(standardized_net_benefit(nb_afa_bin, prev_b), 4),
        "coh_query_rate_binary": round(float(np.mean(q_afa_bin)), 4),
        "coh_auroc_routed": round(float(roc_auc_score(y_b, p_afa_bin)), 4),
    }

    p_cz_c, q_cz_c = coh_preds["paper2_cz_disc_fluid_s2"]
    p_gl_c, q_gl_c = coh_preds["paper1_global_cmi_global_s2"]
    p_bs_c, q_bs_c = coh_preds["stage1_base"]
    p_cg_c, q_cg_c = coh_preds["ablation_cz_disc_global_s2"]
    p_gf_c, q_gf_c = coh_preds["ablation_global_cmi_fluid_s2"]
    p_af_c, q_af_c = coh_preds["covert2023_greedy_cmi_afa"]
    p_af_e, q_af_e = ens_preds["covert2023_greedy_cmi_afa"]

    coh_boots_10k = {
        "cz_fluid_vs_global_cmi": paired_bootstrap_nb_diff_v2(y_b, p_cz_c, q_cz_c, p_gl_c, q_gl_c, contract, n_boot=10000, seed=51),
        "cz_fluid_vs_stage1_base": paired_bootstrap_nb_diff_v2(y_b, p_cz_c, q_cz_c, p_bs_c, q_bs_c, contract, n_boot=10000, seed=52),
        "cz_global_vs_global_cmi": paired_bootstrap_nb_diff_v2(y_b, p_cg_c, q_cg_c, p_gl_c, q_gl_c, contract, n_boot=10000, seed=53),
        "global_fluid_vs_global_cmi": paired_bootstrap_nb_diff_v2(y_b, p_gf_c, q_gf_c, p_gl_c, q_gl_c, contract, n_boot=10000, seed=54),
        "cz_fluid_coh_vs_covert_afa_bin": paired_bootstrap_nb_diff_v2(y_b, p_cz_c, q_cz_c, p_af_c, q_af_c, contract, n_boot=10000, seed=55),
        "cz_fluid_coh_vs_covert_afa_ens": paired_bootstrap_nb_diff_v2(y_b, p_cz_c, q_cz_c, p_af_e, q_af_e, contract, n_boot=10000, seed=56),
        "interaction_2x2_coh": paired_bootstrap_interaction(
            _unit_nb_vector(y_b, p_bs_c, q_bs_c, contract),
            _unit_nb_vector(y_b, p_gl_c, q_gl_c, contract),
            _unit_nb_vector(y_b, p_cg_c, q_cg_c, contract),
            _unit_nb_vector(y_b, p_gf_c, q_gf_c, contract),
            _unit_nb_vector(y_b, p_cz_c, q_cz_c, contract),
            n_boot=10000, seed=57,
        ),
    }

    taus_exchange = [0.04, 0.0625, 0.10, 0.125]
    p_cz_e, q_cz_e = ens_preds["paper2_cz_disc_fluid_s2"]
    p_gl_e, q_gl_e = ens_preds["paper1_global_cmi_global_s2"]
    p_bs_e, _ = ens_preds["stage1_base"]

    exchange_bagged_primary = [
        _exchange_row_at_tau(y_b, p_cz_c, q_cz_c, p_gl_c, q_gl_c, p_bs_c, t, c_query=0.03)
        for t in taus_exchange
    ]
    exchange_symmetric_ens = [
        _exchange_row_at_tau(y_b, p_cz_e, q_cz_e, p_gl_e, q_gl_e, p_bs_e, t, c_query=0.03)
        for t in taus_exchange
    ]
    print(f"  [B1b Polish] Done in {time.time()-t0:.1f}s")
    return {
        "deployable_bagged_voi_summary": coh_summary,
        "deployable_bagged_voi_bootstraps_10k": coh_boots_10k,
        "clinical_exchange_rate_bagged_primary": exchange_bagged_primary,
        "clinical_exchange_rate_symmetric_ens": exchange_symmetric_ens,
    }


def run_covert_afa_uci_validation() -> dict[str, object]:
    """Validate `CovertGreedyCMIAFA` on two public UCI tabular benchmarks (`breast_cancer` & `diabetes`)
    to confirm faithful CMI ranking / AUROC / log-loss performance under matched GBDT predictors.
    """
    t0 = time.time()
    out_uci: dict[str, object] = {}

    # 1. UCI Wisconsin Breast Cancer (N=569, 30 features -> 2 base + 6 clinical channels of cost 1.0, budget=2.0 [33%])
    bc = load_breast_cancer(as_frame=True)
    df_bc = bc.frame.copy()
    # Target: malignant = 1 (in sklearn, target=0 is malignant, 1 is benign; invert so minority malignant ~37% is Y=1)
    y_bc = (1 - df_bc["target"].to_numpy(dtype=int))
    feat_cols = [c for c in df_bc.columns if c != "target"]
    bc_base = feat_cols[:2]  # mean radius, mean texture
    rem = feat_cols[2:]
    bc_channels = [
        ChannelDef(f"ch_morph_{i+1}", tuple(rem[4*i : 4*(i+1) if i < 5 else len(rem)]), cost=1.0)
        for i in range(6)
    ]

    # 2. UCI Diabetes Progression (N=442 -> binary top-quartile progression ~25%, base = age, sex, bmi, bp; 6 serum lab channels s1..s6, budget=2.0 [33%])
    diab = load_diabetes(as_frame=True)
    df_db = diab.frame.copy()
    y_db = (df_db["target"].to_numpy(dtype=float) >= np.quantile(df_db["target"], 0.75)).astype(int)
    db_base = ["age", "sex", "bmi", "bp"]
    db_channels = [ChannelDef(f"ch_serum_{s}", (s,), cost=1.0) for s in ["s1", "s2", "s3", "s4", "s5", "s6"]]

    for bench_name, df_x, y_vec, b_cols, chs in [
        ("uci_wdbc_malignancy", df_bc, y_bc, bc_base, bc_channels),
        ("uci_diabetes_progression", df_db, y_db, db_base, db_channels),
    ]:
        auroc_base, auroc_rand33, auroc_afa33, auroc_full = [], [], [], []
        ll_base, ll_rand33, ll_afa33, ll_full = [], [], [], []
        contract_u = ClinicalContract(cost_fp=1.0, cost_fn=3.0, target_precision=0.50, per_instance_query_cost=0.005)

        for s in range(10):
            tr, ca, te = make_stratified_3way_split(y_vec, seed=3100 + s)
            oof_p = compute_oof_stage1_v2(df_x.iloc[tr][b_cols], y_vec[tr], n_folds=4, seed=3200 + s, n_jobs=4)
            s1_cache = fit_stage1_predictions(
                df_x.iloc[tr], y_vec[tr], df_x.iloc[ca], y_vec[ca], [df_x.iloc[te]], b_cols, seed=3300 + s
            )
            afa_tr: list[dict[str, np.ndarray]] = []
            evaluate_covert_greedy_cmi_afa(
                df_x.iloc[tr], y_vec[tr], df_x.iloc[ca], y_vec[ca], [(df_x.iloc[te], y_vec[te])],
                b_cols, chs, oof_p, s1_cache, contract_u,
                budget=2.0, out_routed_traces=afa_tr, n_jobs=4, seed=3400 + s,
            )
            # Also evaluate random 2-channel subset vs full 6-channel GBDT
            rng_r = np.random.default_rng(3500 + s)
            rand_chs = list(rng_r.choice(len(chs), size=2, replace=False))
            rand_cols = [c for idx_c in rand_chs for c in chs[idx_c].columns]
            full_cols = [c for ch in chs for c in ch.columns]

            tr_rand: list[dict[str, np.ndarray]] = []
            evaluate_stage2_multi_test_v2(
                df_x.iloc[tr], y_vec[tr], df_x.iloc[ca], y_vec[ca], [(df_x.iloc[te], y_vec[te])],
                b_cols, rand_cols, oof_p, contract_u, stage2_mode="global", precomputed_stage1=s1_cache,
                out_routed_traces=tr_rand, n_jobs=4, seed=3600 + s,
            )
            tr_full: list[dict[str, np.ndarray]] = []
            evaluate_stage2_multi_test_v2(
                df_x.iloc[tr], y_vec[tr], df_x.iloc[ca], y_vec[ca], [(df_x.iloc[te], y_vec[te])],
                b_cols, full_cols, oof_p, contract_u, stage2_mode="global", precomputed_stage1=s1_cache,
                out_routed_traces=tr_full, n_jobs=4, seed=3700 + s,
            )

            yt = y_vec[te]
            p_b = afa_tr[0]["p1_te"]
            # Full-budget (2 of 6 channels = 33.3% feature budget) greedy CMI rollout is p2_te in evaluate_covert_greedy_cmi_afa!
            p_afa = afa_tr[0]["p2_te"]
            p_rnd = tr_rand[0]["p2_te"]
            p_all = tr_full[0]["p2_te"]

            auroc_base.append(float(roc_auc_score(yt, p_b)))
            auroc_rand33.append(float(roc_auc_score(yt, p_rnd)))
            auroc_afa33.append(float(roc_auc_score(yt, p_afa)))
            auroc_full.append(float(roc_auc_score(yt, p_all)))
            ll_base.append(float(log_loss(yt, p_b)))
            ll_rand33.append(float(log_loss(yt, p_rnd)))
            ll_afa33.append(float(log_loss(yt, p_afa)))
            ll_full.append(float(log_loss(yt, p_all)))

        out_uci[bench_name] = {
            "n_rows": len(df_x),
            "n_channels": len(chs),
            "budget_channels": 2,
            "budget_fraction": round(2.0 / len(chs), 3),
            "auroc_base": round(float(np.mean(auroc_base)), 4),
            "auroc_random_33pct": round(float(np.mean(auroc_rand33)), 4),
            "auroc_covert_afa_33pct": round(float(np.mean(auroc_afa33)), 4),
            "auroc_full_100pct": round(float(np.mean(auroc_full)), 4),
            "pct_of_full_auroc_gain_recovered": round(
                100.0 * (float(np.mean(auroc_afa33)) - float(np.mean(auroc_base)))
                / max(1e-6, float(np.mean(auroc_full)) - float(np.mean(auroc_base))),
                1,
            ),
            "logloss_base": round(float(np.mean(ll_base)), 4),
            "logloss_random_33pct": round(float(np.mean(ll_rand33)), 4),
            "logloss_covert_afa_33pct": round(float(np.mean(ll_afa33)), 4),
            "logloss_full_100pct": round(float(np.mean(ll_full)), 4),
        }

    print(f"  [UCI Covert-AFA Validation] Done in {time.time()-t0:.1f}s:", json.dumps(out_uci, indent=2))
    # Also test large-N (N=4,000) heterogeneous-channel benchmark where each patient has an instance-specific informative channel (Covert et al. 2023 instancewise CMI regime, tau*=0.50 balanced)
    rng_h = np.random.default_rng(2026)
    n_h = 4000
    g_id = rng_h.integers(0, 4, size=n_h)  # patient subtype 0..3 determines which channel 0..3 carries true signal!
    z_sig = rng_h.normal(0.0, 1.0, size=(n_h, 4))
    y_h = rng_h.binomial(1, 1.0 / (1.0 + np.exp(-(2.2 * z_sig[np.arange(n_h), g_id]))))
    df_h = pd.DataFrame({
        "b_subtype_hint": g_id.astype(float) + rng_h.normal(0.0, 0.15, size=n_h),
        "b_noise": rng_h.normal(0.0, 1.0, size=n_h),
        **{f"ch_{j}": z_sig[:, j] for j in range(4)},
    })
    chs_h = [ChannelDef(f"ch_{j}", (f"ch_{j}",), cost=1.0) for j in range(4)]
    c_bal = ClinicalContract(cost_fp=1.0, cost_fn=1.0, target_precision=0.55, per_instance_query_cost=0.001)
    tr_h, ca_h, te_h = make_stratified_3way_split(y_h, seed=3900)
    oof_h = compute_oof_stage1_v2(df_h.iloc[tr_h][["b_subtype_hint", "b_noise"]], y_h[tr_h], n_folds=4, seed=3901, n_jobs=4)
    s1_h = fit_stage1_predictions(
        df_h.iloc[tr_h], y_h[tr_h], df_h.iloc[ca_h], y_h[ca_h], [df_h.iloc[te_h]], ["b_subtype_hint", "b_noise"], seed=3902
    )
    afa_h: list[dict[str, np.ndarray]] = []
    evaluate_covert_greedy_cmi_afa(
        df_h.iloc[tr_h], y_h[tr_h], df_h.iloc[ca_h], y_h[ca_h], [(df_h.iloc[te_h], y_h[te_h])],
        ["b_subtype_hint", "b_noise"], chs_h, oof_h, s1_h, c_bal,
        budget=1.0, out_routed_traces=afa_h, n_jobs=4, seed=3903,
    )
    # Compare 1-channel fixed cohort selection (`ch_0` only) vs 1-channel instancewise `CovertGreedyCMIAFA` vs 4-channel oracle
    tr_fix1: list[dict[str, np.ndarray]] = []
    evaluate_stage2_multi_test_v2(
        df_h.iloc[tr_h], y_h[tr_h], df_h.iloc[ca_h], y_h[ca_h], [(df_h.iloc[te_h], y_h[te_h])],
        ["b_subtype_hint", "b_noise"], ["ch_0"], oof_h, c_bal, stage2_mode="global", precomputed_stage1=s1_h,
        out_routed_traces=tr_fix1, n_jobs=4, seed=3904,
    )
    tr_all4: list[dict[str, np.ndarray]] = []
    evaluate_stage2_multi_test_v2(
        df_h.iloc[tr_h], y_h[tr_h], df_h.iloc[ca_h], y_h[ca_h], [(df_h.iloc[te_h], y_h[te_h])],
        ["b_subtype_hint", "b_noise"], ["ch_0", "ch_1", "ch_2", "ch_3"], oof_h, c_bal, stage2_mode="global",
        precomputed_stage1=s1_h, out_routed_traces=tr_all4, n_jobs=4, seed=3905,
    )
    out_uci["balanced_instancewise_heterogeneous_benchmark_n4000"] = {
        "auroc_base": round(float(roc_auc_score(y_h[te_h], afa_h[0]["p1_te"])), 4),
        "auroc_fixed_1ch_cohort": round(float(roc_auc_score(y_h[te_h], tr_fix1[0]["p2_te"])), 4),
        "auroc_covert_afa_1ch_instancewise": round(float(roc_auc_score(y_h[te_h], afa_h[0]["p2_te"])), 4),
        "auroc_full_4ch_oracle": round(float(roc_auc_score(y_h[te_h], tr_all4[0]["p2_te"])), 4),
    }
    print("  [Instancewise N=4000 Sanity Check]:", out_uci["balanced_instancewise_heterogeneous_benchmark_n4000"])
    return out_uci


def run_yale_b4a_cfbh_intercept_rescue(manifest: dict[str, object]) -> dict[str, object]:
    """Quantify Track B4a (`Campus A -> Campus B` ESI 4-5, N_ext = 51,205) CfBH failure mode
    and verify mitigation via label-free target-site prior-odds intercept alignment (`Delta_0`).
    """
    t0 = time.time()
    yale_meta = manifest["yale_ed"]  # type: ignore[index]
    meta_full = pd.read_parquet(PROC_DIR / "yale_trackb_meta.parquet")
    df_full = pd.read_parquet(PROC_DIR / "yale_ed_full.parquet")
    fair_base_cols = [
        c for c in yale_meta["base_columns"]  # type: ignore[index]
        if c in df_full.columns and c not in FAIRNESS_EXCLUDED_BASE_COLS
    ]
    channels = [
        ChannelDef(cid, tuple(cinfo["columns"]), cost=float(cinfo["cost"]))
        for cid, cinfo in yale_meta["channels"].items()  # type: ignore[index]
    ]
    ch_cols_map = {c.channel_id: list(c.columns) for c in channels}

    vc = meta_full["dep_name"].value_counts()
    campus_train, campus_ext = str(vc.index[0]), str(vc.index[1])
    m_c1 = (meta_full["dep_name"] == campus_train).to_numpy()
    c1_pids = set(meta_full.loc[m_c1, "pat_mrn_id"].to_numpy())
    m_c2_clean = ((meta_full["dep_name"] == campus_ext) & (~meta_full["pat_mrn_id"].isin(c1_pids))).to_numpy()
    esi_m = meta_full["esi"].isin(["4", "5", "4.0", "5.0"]).to_numpy()

    df_c1 = df_full.iloc[np.flatnonzero(m_c1 & esi_m)].reset_index(drop=True)
    y_c1 = df_c1["y_admit"].to_numpy(dtype=int)
    pids_c1 = meta_full.iloc[np.flatnonzero(m_c1 & esi_m)]["pat_mrn_id"].to_numpy()

    df_c2 = df_full.iloc[np.flatnonzero(m_c2_clean & esi_m)].reset_index(drop=True)
    y_c2 = df_c2["y_admit"].to_numpy(dtype=int)
    n_c2 = y_c2.size
    contract = ClinicalContract(cost_fp=1.0, cost_fn=24.0, target_precision=0.25, per_instance_query_cost=0.025)

    ens_p_cz = np.zeros(n_c2, dtype=float)
    ens_q_cz = np.zeros(n_c2, dtype=float)
    cal_y_pool: list[np.ndarray] = []
    cal_p_pool: list[np.ndarray] = []

    print(f"[Yale B4a CfBH Rescue] Running 6 splits (Campus A {len(df_c1):,} -> Campus B {n_c2:,})...")
    for s in range(6):
        tr, ca, _ = make_patient_grouped_split(pids_c1, y_c1, seed=5100 + s, target_train=50000, target_cal=15000)
        oof_p = compute_oof_stage1_v2(df_c1.iloc[tr][fair_base_cols], y_c1[tr], n_folds=4, seed=5200 + s, n_jobs=4)
        d_cz = run_sequential_discovery_v2(
            df_c1.iloc[tr], y_c1[tr], fair_base_cols, channels, oof_p, contract,
            method="collision_zone", pilot_fraction=0.15, budget=3.0, adaptive_kappa=True, n_jobs=4, seed=5300 + s,
        ).outcome
        cols_cz = [col for cid in d_cz.selected_ids for col in ch_cols_map[cid]]
        s1_cache = fit_stage1_predictions(
            df_c1.iloc[tr], y_c1[tr], df_c1.iloc[ca], y_c1[ca], [df_c2], fair_base_cols, seed=5400 + s
        )
        traces: list[dict[str, np.ndarray]] = []
        cal_tr: dict[str, np.ndarray] = {}
        evaluate_stage2_multi_test_v2(
            df_c1.iloc[tr], y_c1[tr], df_c1.iloc[ca], y_c1[ca], [(df_c2, y_c2)],
            fair_base_cols, cols_cz, oof_p, contract,
            stage2_mode="fluid_backbone", voi_mode="empirical_martingale",
            precomputed_stage1=s1_cache, out_routed_traces=traces, out_cal_trace=cal_tr,
            n_jobs=4, seed=5400 + s,
        )
        ens_p_cz += traces[0]["p_auto_te"]
        ens_q_cz += traces[0]["auto_te"].astype(float)
        cal_y_pool.append(y_c1[ca])
        cal_p_pool.append(cal_tr["p_auto_cal"])

    p_cz = ens_p_cz / 6.0
    q_cz = ens_q_cz / 6.0
    yc_pool = np.concatenate(cal_y_pool)
    pc_pool = np.concatenate(cal_p_pool)

    pi_src = float(np.mean(y_c1))
    pi_tgt = float(np.mean(y_c2))
    fdr_alpha = 1.0 - contract.target_precision  # 0.75

    # Raw unaligned CfBH
    cf_raw = cfbh_selection(yc_pool, pc_pool, y_c2, p_cz, fdr_alpha, seed=71)

    # Mitigation 1: Label-free prior-odds shift Delta_0 applied to Campus B probabilities (aligning Campus B scale to Campus A null distribution: shift Campus B scores by -Delta_0, or equivalently shift calibration null pool by +Delta_0!)
    # Wait: let's check the exact physics! Campus A has pi_src = 3.79%, Campus B has pi_tgt = 1.81%, AND Campus B features yield CITL = -0.805 (mean predicted p on Campus B is ~3.8% while true prevalence is 1.81%!).
    # First: pure marginal prior-odds shift p_cz_prior on Campus B (using pi_tgt vs mean(p_cz) mean-calibration, ZERO individual labels, only aggregate admission rate pi_tgt!):
    mean_pred_b = float(np.mean(p_cz))
    p_cz_mean_aligned, delta_mean_align = prior_odds_recalibrate_probabilities(p_cz, mean_pred_b, pi_tgt)
    cal_after_mean = compute_calibration_metrics(y_c2, p_cz_mean_aligned, n_boot=0, return_curve=False)
    # Also shift source calibration scores to target prevalence pi_tgt so CfBH null pool matches Campus B's prevalence!
    pc_pool_tgt, _ = prior_odds_recalibrate_probabilities(pc_pool, float(np.mean(yc_pool)), pi_tgt)
    # Under prevalence shift, non-events (Y=0) in yc_pool remain exchangeable conditional on aligned risk score when p_cz_mean_aligned and pc_pool_tgt share target marginal prevalence:
    cf_mean_aligned = cfbh_selection(yc_pool, pc_pool_tgt, y_c2, p_cz_mean_aligned, fdr_alpha, seed=71)
    nb_after_align = net_benefit_per_1k(y_c2, p_cz_mean_aligned, q_cz, contract)

    # Mitigation 2: Small site burn-in conformal calibration pilot (n_burn = 2,000 Campus B visits, evaluating on remaining 49,205 visits)
    rng_b = np.random.default_rng(77)
    perm_b = rng_b.permutation(n_c2)
    burn_i, eval_i = perm_b[:2000], perm_b[2000:]
    cf_local_burnin = cfbh_selection(y_c2[burn_i], p_cz[burn_i], y_c2[eval_i], p_cz[eval_i], fdr_alpha, seed=78)

    res_cfbh = {
        "campus_a_prevalence": round(pi_src, 5),
        "campus_b_prevalence": round(pi_tgt, 5),
        "campus_b_mean_raw_predicted_prob": round(mean_pred_b, 5),
        "target_fdr_alpha": fdr_alpha,
        "raw_unaligned_cfbh": {
            "citl": round(float(compute_calibration_metrics(y_c2, p_cz, n_boot=0, return_curve=False)["citl"]), 4),
            "ici": round(float(compute_calibration_metrics(y_c2, p_cz, n_boot=0, return_curve=False)["ici_loess"]), 4),
            "alerts_per_100_shift": round(100.0 * float(cf_raw["n_selected"]) / float(n_c2), 2),
            "realized_fdp": round(float(cf_raw["fdp"]), 4),
            "realized_precision": round(1.0 - float(cf_raw["fdp"]), 4),
            "recall": round(float(cf_raw["recall"]), 4),
        },
        "label_free_marginal_intercept_aligned": {
            "delta_0_applied": round(delta_mean_align, 4),
            "citl": round(float(cal_after_mean["citl"]), 4),
            "ici": round(float(cal_after_mean["ici_loess"]), 4),
            "nb_per_1k": round(nb_after_align, 3),
            "alerts_per_100_shift": round(100.0 * float(cf_mean_aligned["n_selected"]) / float(n_c2), 2),
            "realized_fdp": round(float(cf_mean_aligned["fdp"]), 4),
            "realized_precision": round(1.0 - float(cf_mean_aligned["fdp"]), 4),
            "recall": round(float(cf_mean_aligned["recall"]), 4),
        },
        "local_2k_burnin_conformal_recalibration": {
            "n_burnin": 2000,
            "n_eval": int(eval_i.size),
            "alerts_per_100_shift": round(100.0 * float(cf_local_burnin["n_selected"]) / float(eval_i.size), 2),
            "realized_fdp": round(float(cf_local_burnin["fdp"]), 4),
            "realized_precision": round(1.0 - float(cf_local_burnin["fdp"]), 4),
            "recall": round(float(cf_local_burnin["recall"]), 4),
        },
    }
    print(f"  [Yale B4a CfBH Rescue] Done in {time.time()-t0:.1f}s:", json.dumps(res_cfbh, indent=2))
    return res_cfbh


def main() -> None:
    only_b1b = "--only-b1b" in sys.argv
    manifest = json.loads((PROC_DIR / "ws4tb_cohorts_manifest.json").read_text())
    results = json.loads(OUT_JSON.read_text())

    b1b_polish = run_b1b_exchange_and_bagged_audit(manifest)
    if only_b1b and "v2_2_reviewer_polish" in results:
        results["v2_2_reviewer_polish"]["b1b_deployable_bagged_and_exchange_rate"] = b1b_polish
    else:
        uci_val = run_covert_afa_uci_validation()
        b4a_cfbh = run_yale_b4a_cfbh_intercept_rescue(manifest)
        results["v2_2_reviewer_polish"] = {
            "git_provenance_commit": "8c8c4c8",
            "git_provenance_timestamp": "2026-11-22T18:16:58-05:00",
            "b1b_deployable_bagged_and_exchange_rate": b1b_polish,
            "covert_afa_uci_benchmark_validation": uci_val,
            "yale_b4a_cfbh_shift_and_mitigation": b4a_cfbh,
        }
    OUT_JSON.write_text(json.dumps(results, indent=2))
    print("\nSaved v2_2_reviewer_polish to", OUT_JSON)


if __name__ == "__main__":
    main()
