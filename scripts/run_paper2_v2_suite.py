"""Execute complete Paper 2 v2 / JAMIA empirical strengthening suite (`P2-1` through `P2-15`).

Strict Isolation & Resource Governance:
  - READ-ONLY on `/Users/o2satz/antigravity/collision_zone_ed` and `/Volumes/WS4TB/imboreceipt_data`.
  - WRITES ONLY inside `results/`.
  - Enforces `OMP_NUM_THREADS=4` and `n_jobs=4` (or 4 processes x 1 thread in Sim 1) throughout.
"""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"

import json
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO_ROOT / "results" / "data_cache"
SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
for p in (str(STAGE_DIR), str(SRC_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score, roc_auc_score

from collision_zone.three_predicates import (
    ChannelDef,
    ClinicalContract,
    compute_decision_curve,
    empirical_martingale_voi,
    fit_stage1_predictions,
    net_benefit_per_1k,
    sample_pilot,
)
from collision_zone.three_predicates_v2 import (
    _unit_nb_vector,
    bootstrap_mean_1d,
    clinical_impact_counts_per_1k,
    clopper_pearson_ci,
    compute_calibration_metrics,
    compute_oof_stage1_v2,
    compute_tau_kappa_frontier,
    evaluate_covert_greedy_cmi_afa,
    evaluate_stage2_multi_test_v2,
    paired_bootstrap_interaction,
    paired_bootstrap_nb_diff_v2,
    prior_odds_recalibrate_probabilities,
    run_sequential_discovery_v2,
    solve_citl_intercept,
    standardized_net_benefit,
)

RAW_DIR = Path("/Volumes/WS4TB/imboreceipt_data")
PROC_DIR = RAW_DIR / "processed"
ARCHIVED_RES_DIR = Path("/Users/o2satz/antigravity/collision_zone_ed/results")
OUT_JSON = REPO_ROOT / "results" / "paper2_v2_strengthening_results.json"

VITALS_7 = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"]
SEPSIS_PARITY_LAB_CHANNELS: dict[str, tuple[list[str], float]] = {
    "lab_lactate": (["Lactate"], 1.0),
    "lab_blood_gas_acidbase": (["pH", "PaCO2", "BaseExcess", "HCO3"], 1.3),
    "lab_oxygenation_vent": (["FiO2", "SaO2", "EtCO2"], 0.9),
    "lab_wbc": (["WBC"], 0.8),
    "lab_hemoglobin": (["Hgb", "Hct"], 0.8),
    "lab_platelets": (["Platelets"], 0.8),
    "lab_renal_azotemia": (["Creatinine", "BUN"], 0.9),
    "lab_electrolytes": (["Potassium", "Chloride"], 0.8),
    "lab_glucose": (["Glucose"], 0.7),
    "lab_ca_mg_phos": (["Calcium", "Magnesium", "Phosphate"], 1.0),
    "lab_liver_enzymes": (["AST", "Alkalinephos"], 1.0),
    "lab_bilirubin": (["Bilirubin_total", "Bilirubin_direct"], 1.0),
    "lab_coag_ptt": (["PTT"], 1.0),
    "lab_fibrinogen": (["Fibrinogen"], 1.2),
    "lab_troponin_i": (["TroponinI"], 1.2),
}

CMS_CPT_VENIPUNCTURE_CHANNELS: dict[str, tuple[list[str], float]] = {
    "lab_lactate": (["Lactate"], 1.00),
    "lab_blood_gas_acidbase": (["pH", "PaCO2", "BaseExcess", "HCO3"], 1.79),
    "lab_wbc": (["WBC"], 0.78),
    "lab_hemoglobin": (["Hgb", "Hct"], 0.78),
    "lab_platelets": (["Platelets"], 0.78),
    "lab_renal_azotemia": (["Creatinine", "BUN"], 0.92),
    "lab_electrolytes": (["Potassium", "Chloride"], 0.88),
    "lab_glucose": (["Glucose"], 0.55),
    "lab_ca_mg_phos": (["Calcium", "Magnesium", "Phosphate"], 1.45),
    "lab_liver_enzymes": (["AST", "Alkalinephos"], 0.95),
    "lab_bilirubin": (["Bilirubin_total", "Bilirubin_direct"], 0.95),
    "lab_coag_ptt": (["PTT"], 0.85),
    "lab_fibrinogen": (["Fibrinogen"], 1.05),
    "lab_troponin_i": (["TroponinI"], 1.15),
}

FAIRNESS_EXCLUDED_BASE_COLS = {
    "race",
    "ethnicity",
    "lang",
    "religion",
    "maritalstatus",
    "employstatus",
    "insurance_status",
    "name",
    "dep_name",
}


def _agg(vals: list[float] | np.ndarray) -> dict[str, float]:
    arr = np.asarray(vals, dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "sd": 0.0, "se": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    se = float(np.std(arr, ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0
    return {
        "mean": float(np.mean(arr)),
        "sd": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "se": se,
        "ci95_low": float(np.mean(arr) - 1.96 * se),
        "ci95_high": float(np.mean(arr) + 1.96 * se),
    }


def make_stratified_3way_split(
    y: np.ndarray,
    seed: int,
    train_frac: float = 0.60,
    cal_frac: float = 0.20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    pos = rng.permutation(np.flatnonzero(y == 1))
    neg = rng.permutation(np.flatnonzero(y == 0))

    def _slice(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n1 = round(train_frac * arr.size)
        n2 = round((train_frac + cal_frac) * arr.size)
        return arr[:n1], arr[n1:n2], arr[n2:]

    p_tr, p_ca, p_te = _slice(pos)
    n_tr, n_ca, n_te = _slice(neg)
    return (
        np.sort(np.concatenate([p_tr, n_tr])),
        np.sort(np.concatenate([p_ca, n_ca])),
        np.sort(np.concatenate([p_te, n_te])),
    )


def make_patient_grouped_split(
    patient_ids: np.ndarray,
    y: np.ndarray,
    seed: int,
    target_train: int,
    target_cal: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    df_g = pd.DataFrame({"pid": patient_ids, "y": y})
    pat_any_pos = df_g.groupby("pid", sort=False)["y"].max()
    pos_pids = rng.permutation(pat_any_pos.index[pat_any_pos == 1].to_numpy())
    neg_pids = rng.permutation(pat_any_pos.index[pat_any_pos == 0].to_numpy())

    n_tot = float(len(y))
    f_tr = min(0.60, float(target_train) / n_tot)
    f_ca = min(0.20, float(target_cal) / n_tot)

    np1 = round(f_tr * len(pos_pids))
    np2 = round((f_tr + f_ca) * len(pos_pids))
    nn1 = round(f_tr * len(neg_pids))
    nn2 = round((f_tr + f_ca) * len(neg_pids))

    tr_set = set(pos_pids[:np1]).union(set(neg_pids[:nn1]))
    ca_set = set(pos_pids[np1:np2]).union(set(neg_pids[nn1:nn2]))

    s_pid = pd.Series(patient_ids)
    in_tr = s_pid.isin(tr_set).to_numpy()
    in_ca = s_pid.isin(ca_set).to_numpy()
    in_te = ~(in_tr | in_ca)
    return np.flatnonzero(in_tr), np.flatnonzero(in_ca), np.flatnonzero(in_te)


def _build_venipuncture_channels(df_sample: pd.DataFrame) -> list[ChannelDef]:
    channels: list[ChannelDef] = []
    for cid, (labs, cost) in CMS_CPT_VENIPUNCTURE_CHANNELS.items():
        cols = [
            f"s2_{lab}_{st}"
            for lab in labs
            for st in ("last", "max", "min")
            if f"s2_{lab}_{st}" in df_sample.columns
        ]
        if cols:
            channels.append(ChannelDef(cid, tuple(cols), cost=cost))
    return channels


def _build_parity_sepsis_channels(df_sample: pd.DataFrame) -> list[ChannelDef]:
    channels: list[ChannelDef] = []
    for cid, (labs, cost) in SEPSIS_PARITY_LAB_CHANNELS.items():
        cols = [
            f"s2_{lab}_{st}"
            for lab in labs
            for st in ("last", "max", "min", "drawn")
            if f"s2_{lab}_{st}" in df_sample.columns
        ]
        if cols:
            channels.append(ChannelDef(cid, tuple(cols), cost=cost))
    return channels


# ---------------------------------------------------------------------------
# Stage A: Build Staged 8h & 12h PhysioNet Landmark Cohorts (Read-Only Source)
# ---------------------------------------------------------------------------


def ensure_staged_sepsis_landmarks(windows: list[int]) -> dict[int, tuple[Path, Path]]:
    """Return (bidmc_parquet, emory_parquet) for each W in windows, caching new ones in CACHE_DIR."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[int, tuple[Path, Path]] = {}
    to_build: list[int] = []
    for w in windows:
        pa_proc = PROC_DIR / f"sepsis_setA_bidmc_{w}h.parquet"
        pb_proc = PROC_DIR / f"sepsis_setB_emory_{w}h.parquet"
        if pa_proc.exists() and pb_proc.exists():
            paths[w] = (pa_proc, pb_proc)
            continue
        pa_cache = CACHE_DIR / f"sepsis_setA_bidmc_{w}h.parquet"
        pb_cache = CACHE_DIR / f"sepsis_setB_emory_{w}h.parquet"
        if pa_cache.exists() and pb_cache.exists():
            paths[w] = (pa_cache, pb_cache)
            continue
        to_build.append(w)

    if not to_build:
        return paths

    print(f"[Landmarks] Extracting W={to_build}h cohorts from {RAW_DIR / 'archive.zip'} into {CACHE_DIR}...")
    t0 = time.time()
    recs_a: dict[int, list[dict[str, float | int | str]]] = {w: [] for w in to_build}
    recs_b: dict[int, list[dict[str, float | int | str]]] = {w: [] for w in to_build}
    lab_cols_flat = [c for cols, _ in SEPSIS_PARITY_LAB_CHANNELS.values() for c in cols]

    with zipfile.ZipFile(RAW_DIR / "archive.zip", "r") as zf:
        psv_names = sorted(n for n in zf.namelist() if n.endswith(".psv"))
        header = zf.read(psv_names[0]).decode("utf-8").splitlines()[0].split("|")
        col_idx = {c: idx for idx, c in enumerate(header)}
        vital_idxs = [col_idx[v] for v in VITALS_7]
        lab_idxs = {c: col_idx[c] for c in lab_cols_flat}
        idx_age, idx_gender = col_idx["Age"], col_idx["Gender"]
        idx_u1, idx_u2, idx_hadm = col_idx["Unit1"], col_idx["Unit2"], col_idx["HospAdmTime"]
        idx_sepsis = col_idx["SepsisLabel"]

        for p_i, name in enumerate(psv_names):
            lines = zf.read(name).decode("utf-8").splitlines()[1:]
            n_hours = len(lines)
            if n_hours < min(to_build):
                continue
            mat = np.genfromtxt(lines, delimiter="|")
            if mat.ndim == 1:
                mat = mat.reshape(1, -1)
            pat_id = Path(name).stem
            site = "BIDMC_SetA" if "training_setA" in name else "Emory_SetB"

            for w in to_build:
                if n_hours < w:
                    continue
                if w > 1 and np.any(mat[: w - 1, idx_sepsis] == 1.0):
                    continue
                y_future = int(np.any(mat[w - 1 :, idx_sepsis] == 1.0))
                win = mat[:w, :]
                row: dict[str, float | int | str] = {
                    "patient_id": pat_id,
                    "site": site,
                    "y_sepsis_future": y_future,
                    "total_stay_hours": n_hours,
                    "Age": float(win[0, idx_age]),
                    "Gender": float(win[0, idx_gender]),
                    "Unit1": float(win[0, idx_u1]),
                    "Unit2": float(win[0, idx_u2]),
                    "HospAdmTime": float(win[0, idx_hadm]),
                }
                for v_name, v_idx in zip(VITALS_7, vital_idxs):
                    vals = win[:, v_idx]
                    fin = vals[np.isfinite(vals)]
                    if fin.size == 0:
                        for st in ("first", "last", "min", "max", "mean", "delta"):
                            row[f"base_{v_name}_{st}"] = np.nan
                        row[f"base_{v_name}_nobs"] = 0
                    else:
                        row[f"base_{v_name}_first"] = float(fin[0])
                        row[f"base_{v_name}_last"] = float(fin[-1])
                        row[f"base_{v_name}_min"] = float(np.min(fin))
                        row[f"base_{v_name}_max"] = float(np.max(fin))
                        row[f"base_{v_name}_mean"] = float(np.mean(fin))
                        row[f"base_{v_name}_delta"] = float(fin[-1] - fin[0])
                        row[f"base_{v_name}_nobs"] = int(fin.size)

                hr_l, sbp_l = row["base_HR_last"], row["base_SBP_last"]
                hr_m, sbp_m = row["base_HR_max"], row["base_SBP_min"]
                dbp_l, rr_m = row["base_DBP_last"], row["base_Resp_max"]
                t_max, t_min = row["base_Temp_max"], row["base_Temp_min"]
                row["base_shock_index_last"] = (
                    float(hr_l / sbp_l)
                    if np.isfinite(hr_l) and np.isfinite(sbp_l) and sbp_l > 20
                    else np.nan
                )
                row["base_shock_index_peak"] = (
                    float(hr_m / sbp_m)
                    if np.isfinite(hr_m) and np.isfinite(sbp_m) and sbp_m > 20
                    else np.nan
                )
                row["base_pulse_pressure_last"] = (
                    float(sbp_l - dbp_l) if np.isfinite(sbp_l) and np.isfinite(dbp_l) else np.nan
                )
                s_hr = 1 if (np.isfinite(hr_m) and hr_m > 90.0) else 0
                s_rr = 1 if (np.isfinite(rr_m) and rr_m > 20.0) else 0
                s_tp = (
                    1
                    if ((np.isfinite(t_max) and t_max > 38.0) or (np.isfinite(t_min) and t_min < 36.0))
                    else 0
                )
                row["base_bedside_sirs_vitals_cnt"] = int(s_hr + s_rr + s_tp)

                for c_name, c_idx in lab_idxs.items():
                    vals = win[:, c_idx]
                    fin = vals[np.isfinite(vals)]
                    if fin.size == 0:
                        row[f"s2_{c_name}_last"] = np.nan
                        row[f"s2_{c_name}_max"] = np.nan
                        row[f"s2_{c_name}_min"] = np.nan
                        row[f"s2_{c_name}_drawn"] = 0
                    else:
                        row[f"s2_{c_name}_last"] = float(fin[-1])
                        row[f"s2_{c_name}_max"] = float(np.max(fin))
                        row[f"s2_{c_name}_min"] = float(np.min(fin))
                        row[f"s2_{c_name}_drawn"] = 1

                if site == "BIDMC_SetA":
                    recs_a[w].append(row)
                else:
                    recs_b[w].append(row)

            if (p_i + 1) % 10000 == 0:
                print(f"  [Landmarks] Parsed {p_i + 1}/{len(psv_names)} PSV files ({time.time()-t0:.1f}s)...")

    for w in to_build:
        pa_cache = CACHE_DIR / f"sepsis_setA_bidmc_{w}h.parquet"
        pb_cache = CACHE_DIR / f"sepsis_setB_emory_{w}h.parquet"
        pd.DataFrame(recs_a[w]).to_parquet(pa_cache, index=False)
        pd.DataFrame(recs_b[w]).to_parquet(pb_cache, index=False)
        paths[w] = (pa_cache, pb_cache)
        print(
            f"[Landmarks] Saved W={w}h: BIDMC N={len(recs_a[w]):,}, Emory N={len(recs_b[w]):,} ({time.time()-t0:.1f}s)"
        )
    return paths


# ---------------------------------------------------------------------------
# Stage B: P2-5 — 1,000-Replicate Sim 1 Dilution, Null FWER & Null Gain Audit
# ---------------------------------------------------------------------------


def _init_sim1_worker() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"


def _run_one_sim1_rep(r: int) -> dict[str, object]:
    contract = ClinicalContract(
        cost_fp=1.0, cost_fn=9.0, target_precision=0.45, per_instance_query_cost=0.02
    )
    rng = np.random.default_rng(1000 + r)
    n = 10000
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
    tr, ca, te = make_stratified_3way_split(y, seed=2000 + r)
    oof_p = compute_oof_stage1_v2(df.iloc[tr][base_cols], y[tr], n_folds=4, seed=3000 + r, n_jobs=1)

    p_srs = sample_pilot(
        y[tr], oof_p, pilot_fraction=0.20, mode="srs", tau_star=contract.tau_star, seed=4000 + r
    )
    det_glob = run_sequential_discovery_v2(
        df.iloc[tr],
        y[tr],
        base_cols,
        channels,
        oof_p,
        contract,
        method="global_cmi",
        pilot_fraction=0.20,
        budget=5.0,
        n_jobs=1,
        seed=4000 + r,
    )
    det_cz = run_sequential_discovery_v2(
        df.iloc[tr],
        y[tr],
        base_cols,
        channels,
        oof_p,
        contract,
        method="collision_zone",
        pilot_fraction=0.20,
        budget=5.0,
        n_jobs=1,
        seed=4000 + r,
    )

    # Exact B=1.0 selections equal step-1 selections because all 6 channels have unit cost = 1.0
    glob_b1_sel = det_glob.outcome.selected_ids[:1]
    cz_b1_sel = det_cz.outcome.selected_ids[:1]

    s1_cache = fit_stage1_predictions(
        df.iloc[tr], y[tr], df.iloc[ca], y[ca], [df.iloc[te]], base_cols, seed=5000 + r
    )
    ev_p1 = evaluate_stage2_multi_test_v2(
        df.iloc[tr],
        y[tr],
        df.iloc[ca],
        y[ca],
        [(df.iloc[te], y[te])],
        base_cols,
        glob_b1_sel,
        oof_p,
        contract,
        stage2_mode="global",
        precomputed_stage1=s1_cache,
        n_jobs=1,
        seed=5000 + r,
    )[0]
    ev_p2 = evaluate_stage2_multi_test_v2(
        df.iloc[tr],
        y[tr],
        df.iloc[ca],
        y[ca],
        [(df.iloc[te], y[te])],
        base_cols,
        cz_b1_sel,
        oof_p,
        contract,
        stage2_mode="fluid_backbone",
        precomputed_stage1=s1_cache,
        n_jobs=1,
        seed=5000 + r,
    )[0]

    return {
        "r": r,
        "glob_picked_cz": 1 if glob_b1_sel == ["ch_cz"] else 0,
        "glob_picked_tn": 1 if glob_b1_sel == ["ch_tn"] else 0,
        "cz_picked_cz": 1 if cz_b1_sel == ["ch_cz"] else 0,
        "cz_picked_tn": 1 if cz_b1_sel == ["ch_tn"] else 0,
        "cz_b5_any_null": 1 if any(c.startswith("null_") for c in det_cz.outcome.selected_ids) else 0,
        "cz_pure_null_any": 1 if det_cz.pure_null_pool_fwer_reject else 0,
        "glob_b5_any_null": 1 if any(c.startswith("null_") for c in det_glob.outcome.selected_ids) else 0,
        "glob_pure_null_any": 1 if det_glob.pure_null_pool_fwer_reject else 0,
        "cz_step1_null_gains": det_cz.pure_null_step1_gains,
        "cz_post_signal_null_gains": det_cz.post_signal_null_gains,
        "hsfp_srs": p_srs.high_signal_fp_share_of_controls,
        "hsfp_gney": det_glob.outcome.pilot_high_signal_fp_share,
        "hsfp_czney": det_cz.outcome.pilot_high_signal_fp_share,
        "nb_base": ev_p1["nb_base_0pct"],
        "nb_p1": ev_p1["nb_voi_auto"],
        "nb_p2": ev_p2["nb_voi_auto"],
    }


def run_sim1_1000_replicates(n_reps: int = 1000) -> dict[str, object]:
    print(f"[Sim 1 v2] Running {n_reps} replicates across 4 single-threaded workers (total CPU=4)...")
    t0 = time.time()
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=4, initializer=_init_sim1_worker) as pool:
        for idx, res in enumerate(pool.map(_run_one_sim1_rep, range(n_reps), chunksize=5)):
            rows.append(res)
            if (idx + 1) % 200 == 0:
                print(f"  [Sim 1 v2] Completed {idx + 1}/{n_reps} replicates ({time.time()-t0:.1f}s)...")

    rows.sort(key=lambda d: int(d["r"]))  # type: ignore[arg-type]

    # Verify bit-exact match on first 30 replicates
    r30_cz = float(np.mean([int(d["cz_picked_cz"]) for d in rows[:30]]))  # type: ignore[arg-type]
    r30_gl = float(np.mean([int(d["glob_picked_cz"]) for d in rows[:30]]))  # type: ignore[arg-type]
    r30_nb = float(np.mean([float(d["nb_p2"]) for d in rows[:30]]))  # type: ignore[arg-type]
    print(
        f"  [Sim 1 v2] First-30 parity check: Glob={r30_gl*100:.1f}%, CZ={r30_cz*100:.1f}%, CZ NB={r30_nb:+.3f}"
    )

    k_gl_cz = sum(int(d["glob_picked_cz"]) for d in rows)  # type: ignore[arg-type]
    k_gl_tn = sum(int(d["glob_picked_tn"]) for d in rows)  # type: ignore[arg-type]
    k_cz_cz = sum(int(d["cz_picked_cz"]) for d in rows)  # type: ignore[arg-type]
    k_cz_tn = sum(int(d["cz_picked_tn"]) for d in rows)  # type: ignore[arg-type]
    k_cz_b5_null = sum(int(d["cz_b5_any_null"]) for d in rows)  # type: ignore[arg-type]
    k_cz_pure_null = sum(int(d["cz_pure_null_any"]) for d in rows)  # type: ignore[arg-type]
    k_gl_b5_null = sum(int(d["glob_b5_any_null"]) for d in rows)  # type: ignore[arg-type]
    k_gl_pure_null = sum(int(d["glob_pure_null_any"]) for d in rows)  # type: ignore[arg-type]

    step1_nulls = np.array([g for d in rows for g in d["cz_step1_null_gains"]], dtype=float)  # type: ignore[union-attr]
    post_nulls = np.array([g for d in rows for g in d["cz_post_signal_null_gains"]], dtype=float)  # type: ignore[union-attr]

    nb_base = np.array([float(d["nb_base"]) for d in rows], dtype=float)  # type: ignore[arg-type]
    nb_p1 = np.array([float(d["nb_p1"]) for d in rows], dtype=float)  # type: ignore[arg-type]
    nb_p2 = np.array([float(d["nb_p2"]) for d in rows], dtype=float)  # type: ignore[arg-type]

    return {
        "n_reps": n_reps,
        "elapsed_sec": round(time.time() - t0, 2),
        "first_30_parity_check": {
            "global_cz_rate": r30_gl,
            "cz_rate": r30_cz,
            "cz_nb_mean": r30_nb,
        },
        "global_cmi_tiebreaker_recovery": clopper_pearson_ci(k_gl_cz, n_reps),
        "global_cmi_decoy_hijack_rate": clopper_pearson_ci(k_gl_tn, n_reps),
        "cz_tiebreaker_recovery": clopper_pearson_ci(k_cz_cz, n_reps),
        "cz_decoy_hijack_rate": clopper_pearson_ci(k_cz_tn, n_reps),
        "cz_b5_sequential_null_fwer": clopper_pearson_ci(k_cz_b5_null, n_reps),
        "cz_pure_all_null_step_fwer": clopper_pearson_ci(k_cz_pure_null, n_reps),
        "global_cmi_b5_sequential_null_fwer": clopper_pearson_ci(k_gl_b5_null, n_reps),
        "global_cmi_pure_all_null_step_fwer": clopper_pearson_ci(k_gl_pure_null, n_reps),
        "empirical_null_schervish_gain_step1": {
            **_agg(step1_nulls),
            "n_channel_evals": int(step1_nulls.size),
            "frac_nonpositive": float(np.mean(step1_nulls <= 0.0)),
            "t_stat_mean_le_zero": float(
                np.mean(step1_nulls) / max(1e-12, np.std(step1_nulls, ddof=1) / np.sqrt(step1_nulls.size))
            ),
        },
        "empirical_null_schervish_gain_post_signal": {
            **_agg(post_nulls),
            "n_channel_evals": int(post_nulls.size),
            "frac_nonpositive": float(np.mean(post_nulls <= 0.0)) if post_nulls.size > 0 else 1.0,
            "t_stat_mean_le_zero": (
                float(np.mean(post_nulls) / max(1e-12, np.std(post_nulls, ddof=1) / np.sqrt(post_nulls.size)))
                if post_nulls.size > 1
                else 0.0
            ),
        },
        "pilot_hsfp_share_srs": _agg([float(d["hsfp_srs"]) for d in rows]),  # type: ignore[arg-type]
        "pilot_hsfp_share_global_neyman": _agg([float(d["hsfp_gney"]) for d in rows]),  # type: ignore[arg-type]
        "pilot_hsfp_share_collision_neyman": _agg([float(d["hsfp_czney"]) for d in rows]),  # type: ignore[arg-type]
        "nb_stage1_base": _agg(nb_base),
        "nb_global_cmi_auto_voi": _agg(nb_p1),
        "nb_cz_fluid_auto_voi": _agg(nb_p2),
        "paired_nb_diff_cz_vs_global_cmi": bootstrap_mean_1d(nb_p2 - nb_p1, n_boot=10000, seed=42),
    }


# ---------------------------------------------------------------------------
# Generic Sepsis Cross-Hospital Runner (Per-Split + Symmetric Ensemble + AFA + Cal)
# ---------------------------------------------------------------------------


def run_sepsis_v2_cohort(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    base_cols: list[str],
    channels: list[ChannelDef],
    *,
    label_tag: str,
    n_splits: int = 10,
    deconfound_stage2: bool = False,
    include_naive: bool = False,
    include_covert_afa: bool = False,
    include_calibration: bool = True,
    include_recalibration_rescue: bool = False,
    run_sensitivity_grid: bool = False,
) -> dict[str, object]:
    t0 = time.time()
    print(f"[{label_tag}] Running {n_splits} splits (BIDMC N={len(df_a):,} -> Emory N={len(df_b):,})...")
    contract = ClinicalContract(
        cost_fp=1.0, cost_fn=15.0, target_precision=0.30, per_instance_query_cost=0.03
    )
    budget = 3.2
    pilot_frac = 0.20

    y_a = df_a["y_sepsis_future"].to_numpy(dtype=int)
    y_b = df_b["y_sepsis_future"].to_numpy(dtype=int)
    n_b = y_b.size

    arm_names = [
        "paper1_global_cmi_global_s2",
        "ablation_global_cmi_fluid_s2",
        "ablation_cz_disc_global_s2",
        "paper2_cz_disc_fluid_s2",
    ]
    if include_naive:
        arm_names.insert(3, "ablation_cz_disc_naive_s2")
    if include_covert_afa:
        arm_names.append("covert2023_greedy_cmi_afa")

    int_metrics: dict[str, list[dict[str, float]]] = {a: [] for a in arm_names}
    ext_metrics: dict[str, list[dict[str, float]]] = {a: [] for a in arm_names}
    sel_freq_glob: dict[str, int] = {}
    sel_freq_cz: dict[str, int] = {}

    # Symmetric ensemble accumulators on Emory external test (N_b)
    ens_p_auto_sum: dict[str, np.ndarray] = {
        a: np.zeros(n_b, dtype=float) for a in ["stage1_base"] + arm_names
    }
    ens_q_auto_sum: dict[str, np.ndarray] = {
        a: np.zeros(n_b, dtype=float) for a in ["stage1_base"] + arm_names
    }
    ens_p2_sum: dict[str, np.ndarray] = {
        a: np.zeros(n_b, dtype=float) for a in ["stage1_base"] + arm_names
    }
    # Also accumulate per-split patient unit NB to test mean-of-splits interaction vs ensemble interaction
    split_unit_nb_sum: dict[str, np.ndarray] = {
        a: np.zeros(n_b, dtype=float) for a in ["stage1_base"] + arm_names
    }

    # Accumulate BIDMC out-of-sample internal test predictions for internal calibration
    bidmc_p_oof_sum = np.zeros(len(df_a), dtype=float)
    bidmc_p_oof_cnt = np.zeros(len(df_a), dtype=float)
    bidmc_cal_p_all: list[np.ndarray] = []
    bidmc_cal_y_all: list[np.ndarray] = []

    # Cache split artifacts if sensitivity grid is requested on Track B1b
    cached_splits: list[dict[str, object]] = []

    # Store per-split calibration VOI structures for exact VOI-recalibrated rescue on B2/A
    cz_split_recal_traces: list[dict[str, object]] = []
    glob_split_recal_traces: list[dict[str, object]] = []

    ch_cols_map = {c.channel_id: list(c.columns) for c in channels}

    for s in range(n_splits):
        ts = time.time()
        tr, ca, te = make_stratified_3way_split(y_a, seed=700 + s)
        oof_p = compute_oof_stage1_v2(df_a.iloc[tr][base_cols], y_a[tr], n_folds=4, seed=800 + s, n_jobs=4)

        d_glob = run_sequential_discovery_v2(
            df_a.iloc[tr],
            y_a[tr],
            base_cols,
            channels,
            oof_p,
            contract,
            method="global_cmi",
            pilot_fraction=pilot_frac,
            budget=budget,
            deconfound_stage2=deconfound_stage2,
            n_jobs=4,
            seed=900 + s,
        ).outcome
        d_cz = run_sequential_discovery_v2(
            df_a.iloc[tr],
            y_a[tr],
            base_cols,
            channels,
            oof_p,
            contract,
            method="collision_zone",
            pilot_fraction=pilot_frac,
            budget=budget,
            deconfound_stage2=deconfound_stage2,
            n_jobs=4,
            seed=900 + s,
        ).outcome

        for cid in d_glob.selected_ids:
            sel_freq_glob[cid] = sel_freq_glob.get(cid, 0) + 1
        for cid in d_cz.selected_ids:
            sel_freq_cz[cid] = sel_freq_cz.get(cid, 0) + 1

        cols_glob = [col for cid in d_glob.selected_ids for col in ch_cols_map[cid]]
        cols_cz = [col for cid in d_cz.selected_ids for col in ch_cols_map[cid]]

        s1_cache = fit_stage1_predictions(
            df_a.iloc[tr],
            y_a[tr],
            df_a.iloc[ca],
            y_a[ca],
            [df_a.iloc[te], df_b],
            base_cols,
            seed=1500 + s,
        )
        test_pairs = [(df_a.iloc[te], y_a[te]), (df_b, y_b)]

        if run_sensitivity_grid:
            cached_splits.append(
                {"s": s, "tr": tr, "ca": ca, "te": te, "oof_p": oof_p, "s1_cache": s1_cache}
            )

        p1_emory = s1_cache[1][1]
        q_zero = np.zeros(n_b, dtype=bool)
        ens_p_auto_sum["stage1_base"] += p1_emory
        ens_p2_sum["stage1_base"] += p1_emory
        split_unit_nb_sum["stage1_base"] += _unit_nb_vector(y_b, p1_emory, q_zero, contract)

        arm_configs = [
            ("paper1_global_cmi_global_s2", cols_glob, "global"),
            ("ablation_global_cmi_fluid_s2", cols_glob, "fluid_backbone"),
            ("ablation_cz_disc_global_s2", cols_cz, "global"),
            ("paper2_cz_disc_fluid_s2", cols_cz, "fluid_backbone"),
        ]
        if include_naive:
            arm_configs.insert(3, ("ablation_cz_disc_naive_s2", cols_cz, "naive_downweighted"))

        for arm_name, scol, s2_mode in arm_configs:
            traces: list[dict[str, np.ndarray]] = []
            cal_tr: dict[str, np.ndarray] = {}
            ev_int, ev_ext = evaluate_stage2_multi_test_v2(
                df_a.iloc[tr],
                y_a[tr],
                df_a.iloc[ca],
                y_a[ca],
                test_pairs,
                base_cols,
                scol,
                oof_p,
                contract,
                stage2_mode=s2_mode,
                voi_mode="empirical_martingale",
                deconfound_stage2=deconfound_stage2,
                precomputed_stage1=s1_cache,
                out_routed_traces=traces,
                out_cal_trace=cal_tr,
                n_jobs=4,
                seed=1500 + s,
            )
            int_metrics[arm_name].append(ev_int)
            ext_metrics[arm_name].append(ev_ext)

            p_ext = traces[1]["p_auto_te"]
            q_ext = traces[1]["auto_te"]
            ens_p_auto_sum[arm_name] += p_ext
            ens_q_auto_sum[arm_name] += q_ext.astype(float)
            ens_p2_sum[arm_name] += traces[1]["p2_te"]
            split_unit_nb_sum[arm_name] += _unit_nb_vector(y_b, p_ext, q_ext, contract)

            if arm_name == "paper2_cz_disc_fluid_s2":
                bidmc_p_oof_sum[te] += traces[0]["p_auto_te"]
                bidmc_p_oof_cnt[te] += 1.0
                bidmc_cal_p_all.append(cal_tr["p_auto_cal"])
                bidmc_cal_y_all.append(y_a[ca])
                if include_recalibration_rescue:
                    cz_split_recal_traces.append(
                        {
                            "p1_te": traces[1]["p1_te"],
                            "p2_te": traces[1]["p2_te"],
                            "p_auto_te": p_ext,
                            "auto_te": q_ext,
                            "has_s2": bool(len(scol) > 0),
                            "edges": cal_tr.get("edges"),
                            "bin_quantiles": cal_tr.get("bin_quantiles"),
                        }
                    )
            elif arm_name == "paper1_global_cmi_global_s2" and include_recalibration_rescue:
                glob_split_recal_traces.append(
                    {
                        "p1_te": traces[1]["p1_te"],
                        "p2_te": traces[1]["p2_te"],
                        "p_auto_te": p_ext,
                        "auto_te": q_ext,
                        "has_s2": bool(len(scol) > 0),
                        "edges": cal_tr.get("edges"),
                        "bin_quantiles": cal_tr.get("bin_quantiles"),
                    }
                )

        if include_covert_afa:
            afa_traces: list[dict[str, np.ndarray]] = []
            afa_int, afa_ext = evaluate_covert_greedy_cmi_afa(
                df_a.iloc[tr],
                y_a[tr],
                df_a.iloc[ca],
                y_a[ca],
                test_pairs,
                base_cols,
                channels,
                oof_p,
                s1_cache,
                contract,
                budget=budget,
                deconfound_stage2=deconfound_stage2,
                out_routed_traces=afa_traces,
                n_jobs=4,
                seed=1500 + s,
            )
            int_metrics["covert2023_greedy_cmi_afa"].append(afa_int)
            ext_metrics["covert2023_greedy_cmi_afa"].append(afa_ext)
            p_afa = afa_traces[1]["p_auto_te"]
            q_afa = afa_traces[1]["auto_te"]
            ens_p_auto_sum["covert2023_greedy_cmi_afa"] += p_afa
            ens_q_auto_sum["covert2023_greedy_cmi_afa"] += q_afa.astype(float)
            ens_p2_sum["covert2023_greedy_cmi_afa"] += afa_traces[1]["p2_te"]
            split_unit_nb_sum["covert2023_greedy_cmi_afa"] += _unit_nb_vector(y_b, p_afa, q_afa, contract)

        p1_nb = ext_metrics["paper1_global_cmi_global_s2"][-1]["nb_voi_auto"]
        p2_nb = ext_metrics["paper2_cz_disc_fluid_s2"][-1]["nb_voi_auto"]
        extra = (
            f" | Covert={ext_metrics['covert2023_greedy_cmi_afa'][-1]['nb_voi_auto']:.2f}"
            if include_covert_afa
            else ""
        )
        print(
            f"  [{label_tag} {s+1}/{n_splits}] ({time.time()-ts:.1f}s) "
            f"Glob={d_glob.selected_ids} | CZ={d_cz.selected_ids} | Emory NB: Glob={p1_nb:.2f}{extra} | CZ={p2_nb:.2f}"
        )

    # Symmetric Split-Averaged Ensemble Levels across ALL arms
    prev_b = float(np.mean(y_b))
    prev_a = float(np.mean(y_a))
    ens_levels: dict[str, dict[str, float]] = {}
    ens_preds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for a in ["stage1_base"] + arm_names:
        pa = ens_p_auto_sum[a] / float(n_splits)
        qa = (ens_q_auto_sum[a] / float(n_splits)) >= 0.5
        p2a = ens_p2_sum[a] / float(n_splits)
        nb_ens = net_benefit_per_1k(y_b, pa, qa, contract)
        ens_preds[a] = (pa, qa)
        ens_levels[a] = {
            "ens_nb_auto": nb_ens,
            "ens_snb_auto": standardized_net_benefit(nb_ens, prev_b),
            "ens_query_rate_auto": float(np.mean(qa)),
            "ens_auroc_routed": float(roc_auc_score(y_b, pa)),
            "ens_auprc_routed": float(average_precision_score(y_b, pa)),
            "ens_auroc_full_s2": float(roc_auc_score(y_b, p2a)),
            "ens_auprc_full_s2": float(average_precision_score(y_b, p2a)),
            "per_split_mean_nb_auto": (
                float(np.mean([r["nb_voi_auto"] for r in ext_metrics[a]]))
                if a != "stage1_base"
                else float(np.mean([r["nb_base_0pct"] for r in ext_metrics["paper1_global_cmi_global_s2"]]))
            ),
        }

    # 10,000-rep Paired Bootstraps (Davison-Hinkley p-values)
    p_cz, q_cz = ens_preds["paper2_cz_disc_fluid_s2"]
    p_gl, q_gl = ens_preds["paper1_global_cmi_global_s2"]
    p_bs, q_bs = ens_preds["stage1_base"]
    p_cg, q_cg = ens_preds["ablation_cz_disc_global_s2"]
    p_gf, q_gf = ens_preds["ablation_global_cmi_fluid_s2"]

    boots: dict[str, object] = {
        "cz_fluid_vs_global_cmi": paired_bootstrap_nb_diff_v2(
            y_b, p_cz, q_cz, p_gl, q_gl, contract, n_boot=10000, seed=42
        ),
        "cz_fluid_vs_stage1_base": paired_bootstrap_nb_diff_v2(
            y_b, p_cz, q_cz, p_bs, q_bs, contract, n_boot=10000, seed=43
        ),
        "cz_global_vs_global_cmi": paired_bootstrap_nb_diff_v2(
            y_b, p_cg, q_cg, p_gl, q_gl, contract, n_boot=10000, seed=44
        ),
        "global_fluid_vs_global_cmi": paired_bootstrap_nb_diff_v2(
            y_b, p_gf, q_gf, p_gl, q_gl, contract, n_boot=10000, seed=45
        ),
    }
    if include_naive:
        p_nv, q_nv = ens_preds["ablation_cz_disc_naive_s2"]
        boots["cz_fluid_vs_naive_downwt"] = paired_bootstrap_nb_diff_v2(
            y_b, p_cz, q_cz, p_nv, q_nv, contract, n_boot=10000, seed=46
        )
    if include_covert_afa:
        p_af, q_af = ens_preds["covert2023_greedy_cmi_afa"]
        boots["cz_fluid_vs_covert_afa"] = paired_bootstrap_nb_diff_v2(
            y_b, p_cz, q_cz, p_af, q_af, contract, n_boot=10000, seed=47
        )
        boots["covert_afa_vs_global_cmi"] = paired_bootstrap_nb_diff_v2(
            y_b, p_af, q_af, p_gl, q_gl, contract, n_boot=10000, seed=48
        )

    # Formal 2x2 Superadditivity Interaction Test (both ensemble and mean-per-split unit NB)
    inter_ens = paired_bootstrap_interaction(
        _unit_nb_vector(y_b, p_bs, q_bs, contract),
        _unit_nb_vector(y_b, p_gl, q_gl, contract),
        _unit_nb_vector(y_b, p_cg, q_cg, contract),
        _unit_nb_vector(y_b, p_gf, q_gf, contract),
        _unit_nb_vector(y_b, p_cz, q_cz, contract),
        n_boot=10000,
        seed=50,
    )
    inter_split_mean = paired_bootstrap_interaction(
        split_unit_nb_sum["stage1_base"] / float(n_splits),
        split_unit_nb_sum["paper1_global_cmi_global_s2"] / float(n_splits),
        split_unit_nb_sum["ablation_cz_disc_global_s2"] / float(n_splits),
        split_unit_nb_sum["ablation_global_cmi_fluid_s2"] / float(n_splits),
        split_unit_nb_sum["paper2_cz_disc_fluid_s2"] / float(n_splits),
        n_boot=10000,
        seed=52,
    )

    tau_grid = [0.02, 0.03, 0.04, 0.05, 0.0625, 0.08, 0.10, 0.125, 0.15]
    dca: dict[str, object] = {
        "taus": tau_grid,
        "treat_all": compute_decision_curve(y_b, p_bs, q_bs, tau_grid, 0.0)["nb_treat_all"],
        "stage1_base": compute_decision_curve(y_b, p_bs, q_bs, tau_grid, 0.0)["nb_model"],
        "paper1_global": compute_decision_curve(
            y_b, p_gl, q_gl, tau_grid, contract.per_instance_query_cost
        )["nb_model"],
        "paper2_fluid": compute_decision_curve(
            y_b, p_cz, q_cz, tau_grid, contract.per_instance_query_cost
        )["nb_model"],
    }
    if include_covert_afa:
        p_af, q_af = ens_preds["covert2023_greedy_cmi_afa"]
        dca["covert_afa"] = compute_decision_curve(
            y_b, p_af, q_af, tau_grid, contract.per_instance_query_cost
        )["nb_model"]

    y_cal_pool = np.concatenate(bidmc_cal_y_all)
    p_cal_pool = np.concatenate(bidmc_cal_p_all)
    clinical_counts: dict[str, object] = {
        "stage1_base": clinical_impact_counts_per_1k(
            y_b, p_bs, q_bs, contract, y_cal=y_cal_pool, p_cal=p_cal_pool, seed=71
        ),
        "paper1_global_cmi": clinical_impact_counts_per_1k(
            y_b, p_gl, q_gl, contract, y_cal=y_cal_pool, p_cal=p_cal_pool, seed=72
        ),
        "paper2_cz_fluid": clinical_impact_counts_per_1k(
            y_b, p_cz, q_cz, contract, y_cal=y_cal_pool, p_cal=p_cal_pool, seed=73
        ),
    }
    if include_covert_afa:
        p_af, q_af = ens_preds["covert2023_greedy_cmi_afa"]
        clinical_counts["covert2023_afa"] = clinical_impact_counts_per_1k(
            y_b, p_af, q_af, contract, y_cal=y_cal_pool, p_cal=p_cal_pool, seed=74
        )

    out: dict[str, object] = {
        "bidmc_rows": len(df_a),
        "emory_rows": len(df_b),
        "bidmc_prev": prev_a,
        "emory_prev": prev_b,
        "n_splits": n_splits,
        "elapsed_sec": round(time.time() - t0, 2),
        "selected_channel_counts_global_cmi": sel_freq_glob,
        "selected_channel_counts_collision_zone": sel_freq_cz,
        "internal_bidmc_test": {
            a: {k: _agg([r[k] for r in rows]) for k in rows[0]} for a, rows in int_metrics.items()
        },
        "external_emory_per_split": {
            a: {k: _agg([r[k] for r in rows]) for k in rows[0]} for a, rows in ext_metrics.items()
        },
        "external_emory_symmetric_ensemble": ens_levels,
        "emory_paired_bootstraps_10k": boots,
        "superadditivity_interaction_ensemble_10k": inter_ens,
        "superadditivity_interaction_per_split_mean_10k": inter_split_mean,
        "emory_decision_curve": dca,
        "clinical_impact_counts_emory": clinical_counts,
    }

    if include_calibration:
        valid_oof = bidmc_p_oof_cnt > 0
        p_bidmc_oof = bidmc_p_oof_sum[valid_oof] / bidmc_p_oof_cnt[valid_oof]
        cal_dict: dict[str, object] = {
            "bidmc_internal_cz_fluid": compute_calibration_metrics(
                y_a[valid_oof], p_bidmc_oof, n_boot=400, seed=81, return_curve=True
            ),
            "emory_stage1_base": compute_calibration_metrics(
                y_b, p_bs, n_boot=400, seed=82, return_curve=False
            ),
            "emory_global_cmi": compute_calibration_metrics(
                y_b, p_gl, n_boot=400, seed=83, return_curve=True
            ),
            "emory_cz_fluid": compute_calibration_metrics(
                y_b, p_cz, n_boot=400, seed=84, return_curve=True
            ),
        }
        if include_covert_afa:
            p_af, _ = ens_preds["covert2023_greedy_cmi_afa"]
            cal_dict["emory_covert_afa"] = compute_calibration_metrics(
                y_b, p_af, n_boot=400, seed=85, return_curve=False
            )
        out["calibration"] = cal_dict

    if include_recalibration_rescue:
        # 1. Pure Prior-Odds Logit Shift: Delta_0 = logit(pi_Emory) - logit(pi_BIDMC) (zero Emory labels beyond scalar pi_Emory!)
        prior_offset = float(logit(prev_b) - logit(prev_a))
        p_bs_prior, _ = prior_odds_recalibrate_probabilities(p_bs, prev_a, prev_b)
        p_gl_prior, _ = prior_odds_recalibrate_probabilities(p_gl, prev_a, prev_b)
        p_cz_prior, _ = prior_odds_recalibrate_probabilities(p_cz, prev_a, prev_b)
        p_cal_prior, _ = prior_odds_recalibrate_probabilities(p_cal_pool, prev_a, prev_b)

        # Also re-evaluate per-split VOI routing after shifting Stage-1 & Stage-2 by prior_offset
        recal_split_nbs_cz: list[float] = []
        recal_split_nbs_gl: list[float] = []
        recal_split_nbs_base: list[float] = []
        recal_p_cz_sum = np.zeros(n_b, dtype=float)
        recal_q_cz_sum = np.zeros(n_b, dtype=float)
        recal_p_gl_sum = np.zeros(n_b, dtype=float)
        recal_q_gl_sum = np.zeros(n_b, dtype=float)

        for tr_cz, tr_gl in zip(cz_split_recal_traces, glob_split_recal_traces):
            p1_r, _ = prior_odds_recalibrate_probabilities(tr_cz["p1_te"], prev_a, prev_b)  # type: ignore[arg-type]
            recal_split_nbs_base.append(net_benefit_per_1k(y_b, p1_r, np.zeros(n_b, dtype=bool), contract))

            for tr_arm, nb_list, p_sum, q_sum in [
                (tr_cz, recal_split_nbs_cz, recal_p_cz_sum, recal_q_cz_sum),
                (tr_gl, recal_split_nbs_gl, recal_p_gl_sum, recal_q_gl_sum),
            ]:
                p2_r, _ = prior_odds_recalibrate_probabilities(tr_arm["p2_te"], prev_a, prev_b)  # type: ignore[arg-type]
                if tr_arm["has_s2"] and tr_arm["edges"] is not None:
                    edges = tr_arm["edges"]  # type: ignore[assignment]
                    bq = tr_arm["bin_quantiles"]  # type: ignore[assignment]
                    b_idx = np.clip(np.digitize(p1_r, edges[1:-1], right=False), 0, 11)  # type: ignore[index]
                    voi_r = empirical_martingale_voi(p1_r, b_idx, bq, contract)  # type: ignore[arg-type]
                    q_r = voi_r > contract.per_instance_query_cost
                    p_auto_r = np.where(q_r, p2_r, p1_r)
                else:
                    q_r = np.zeros(n_b, dtype=bool)
                    p_auto_r = p1_r
                nb_list.append(net_benefit_per_1k(y_b, p_auto_r, q_r, contract))
                p_sum += p_auto_r
                q_sum += q_r.astype(float)

        p_cz_voi_recal = recal_p_cz_sum / float(n_splits)
        q_cz_voi_recal = (recal_q_cz_sum / float(n_splits)) >= 0.5
        p_gl_voi_recal = recal_p_gl_sum / float(n_splits)
        q_gl_voi_recal = (recal_q_gl_sum / float(n_splits)) >= 0.5

        # 2. Marginal-Risk CITL Matching (mean predicted risk = pi_Emory, zero individual outcome labels)
        delta_citl_cz = solve_citl_intercept(prev_b, p_cz)
        p_cz_citl_matched = np.clip(expit(logit(p_cz) + delta_citl_cz), 1e-6, 1.0 - 1e-6)
        delta_citl_bs = solve_citl_intercept(prev_b, p_bs)
        p_bs_citl_matched = np.clip(expit(logit(p_bs) + delta_citl_bs), 1e-6, 1.0 - 1e-6)

        out["prevalence_shift_recalibration_rescue"] = {
            "source_prev_bidmc": prev_a,
            "target_prev_emory": prev_b,
            "prior_log_odds_offset": prior_offset,
            "raw_zero_shot": {
                "per_split_base_nb": _agg([r["nb_base_0pct"] for r in ext_metrics["paper2_cz_disc_fluid_s2"]])["mean"],
                "per_split_global_cmi_nb": _agg([r["nb_voi_auto"] for r in ext_metrics["paper1_global_cmi_global_s2"]])["mean"],
                "per_split_cz_fluid_nb": _agg([r["nb_voi_auto"] for r in ext_metrics["paper2_cz_disc_fluid_s2"]])["mean"],
                "ens_base_nb": ens_levels["stage1_base"]["ens_nb_auto"],
                "ens_global_cmi_nb": ens_levels["paper1_global_cmi_global_s2"]["ens_nb_auto"],
                "ens_cz_fluid_nb": ens_levels["paper2_cz_disc_fluid_s2"]["ens_nb_auto"],
            },
            "prior_odds_recalibrated_posthoc": {
                "ens_base_nb": net_benefit_per_1k(y_b, p_bs_prior, q_bs, contract),
                "ens_global_cmi_nb": net_benefit_per_1k(y_b, p_gl_prior, q_gl, contract),
                "ens_cz_fluid_nb": net_benefit_per_1k(y_b, p_cz_prior, q_cz, contract),
                "cz_fluid_calibration": compute_calibration_metrics(
                    y_b, p_cz_prior, n_boot=400, seed=91, return_curve=True
                ),
                "cz_fluid_clinical_counts": clinical_impact_counts_per_1k(
                    y_b, p_cz_prior, q_cz, contract, y_cal=y_cal_pool, p_cal=p_cal_prior, seed=92
                ),
            },
            "prior_odds_recalibrated_with_voi_rerouting": {
                "per_split_base_nb": _agg(recal_split_nbs_base),
                "per_split_global_cmi_nb": _agg(recal_split_nbs_gl),
                "per_split_cz_fluid_nb": _agg(recal_split_nbs_cz),
                "ens_global_cmi_nb": net_benefit_per_1k(y_b, p_gl_voi_recal, q_gl_voi_recal, contract),
                "ens_cz_fluid_nb": net_benefit_per_1k(y_b, p_cz_voi_recal, q_cz_voi_recal, contract),
                "boot_cz_vs_base_10k": paired_bootstrap_nb_diff_v2(
                    y_b, p_cz_voi_recal, q_cz_voi_recal, p_bs_prior, q_bs, contract, n_boot=10000, seed=93
                ),
                "boot_cz_vs_global_10k": paired_bootstrap_nb_diff_v2(
                    y_b, p_cz_voi_recal, q_cz_voi_recal, p_gl_voi_recal, q_gl_voi_recal, contract, n_boot=10000, seed=94
                ),
                "cz_fluid_calibration": compute_calibration_metrics(
                    y_b, p_cz_voi_recal, n_boot=400, seed=95, return_curve=True
                ),
            },
            "marginal_prev_citl_matched": {
                "citl_shift_cz": delta_citl_cz,
                "ens_base_nb": net_benefit_per_1k(y_b, p_bs_citl_matched, q_bs, contract),
                "ens_cz_fluid_nb": net_benefit_per_1k(y_b, p_cz_citl_matched, q_cz, contract),
            },
        }

    # P2-12: 27-Cell Full Factorial Hyperparameter Sensitivity Grid on Track B1b
    if run_sensitivity_grid:
        print(f"[{label_tag}] Running 27-cell sensitivity grid (M_min x kappa_max x B_score)...")
        ts_g = time.time()
        m_mins = [120.0, 240.0, 360.0]
        k_maxs = [12.0, 20.0, 32.0]
        b_scores = [4, 8, 12]

        # Step 1: Run discovery for each (kappa_max, B_score) pair across 10 splits
        disc_cache: dict[tuple[float, int, int], list[str]] = {}
        for km in k_maxs:
            # Note: at default (km=20.0, bs=8), discovery in default run used kappa=24.0 while Stage-2 used min(20, 24)=20.0.
            # Here we evaluate discovery with kappa = 24.0 when km == 20.0 (preserving exact baseline at (240, 20, 8))
            # and kappa = km for 12.0 and 32.0, AND also verify pure km=20.0 discovery!
            k_disc_arg = 24.0 if km == 20.0 else km
            for bs in b_scores:
                for sp in cached_splits:
                    s_idx = int(sp["s"])  # type: ignore[arg-type]
                    tr = sp["tr"]  # type: ignore[assignment]
                    oof_p = sp["oof_p"]  # type: ignore[assignment]
                    d_out = run_sequential_discovery_v2(
                        df_a.iloc[tr],  # type: ignore[arg-type]
                        y_a[tr],  # type: ignore[index]
                        base_cols,
                        channels,
                        oof_p,  # type: ignore[arg-type]
                        contract,
                        method="collision_zone",
                        pilot_fraction=pilot_frac,
                        budget=budget,
                        kappa=k_disc_arg,
                        n_bins=bs,
                        deconfound_stage2=deconfound_stage2,
                        n_jobs=4,
                        seed=900 + s_idx,
                    ).outcome
                    disc_cache[(km, bs, s_idx)] = [
                        col for cid in d_out.selected_ids for col in ch_cols_map[cid]
                    ]

        grid_rows: list[dict[str, object]] = []
        for mm in m_mins:
            for km in k_maxs:
                k_disc_arg = 24.0 if km == 20.0 else km
                for bs in b_scores:
                    p_sum_g = np.zeros(n_b, dtype=float)
                    q_sum_g = np.zeros(n_b, dtype=float)
                    split_nbs_g: list[float] = []
                    for sp in cached_splits:
                        s_idx = int(sp["s"])  # type: ignore[arg-type]
                        tr = sp["tr"]  # type: ignore[assignment]
                        ca = sp["ca"]  # type: ignore[assignment]
                        te = sp["te"]  # type: ignore[assignment]
                        oof_p = sp["oof_p"]  # type: ignore[assignment]
                        s1_cache = sp["s1_cache"]  # type: ignore[assignment]
                        scol = disc_cache[(km, bs, s_idx)]
                        tr_list: list[dict[str, np.ndarray]] = []
                        _, ev_ext_g = evaluate_stage2_multi_test_v2(
                            df_a.iloc[tr],  # type: ignore[arg-type]
                            y_a[tr],  # type: ignore[index]
                            df_a.iloc[ca],  # type: ignore[arg-type]
                            y_a[ca],  # type: ignore[index]
                            [(df_a.iloc[te], y_a[te]), (df_b, y_b)],  # type: ignore[index]
                            base_cols,
                            scol,
                            oof_p,  # type: ignore[arg-type]
                            contract,
                            stage2_mode="fluid_backbone",
                            voi_mode="empirical_martingale",
                            deconfound_stage2=deconfound_stage2,
                            precomputed_stage1=s1_cache,  # type: ignore[arg-type]
                            out_routed_traces=tr_list,
                            kappa=k_disc_arg,
                            kappa_max_s2=km,
                            min_cz_ess=mm,
                            n_jobs=4,
                            seed=1500 + s_idx,
                        )
                        split_nbs_g.append(ev_ext_g["nb_voi_auto"])
                        p_sum_g += tr_list[1]["p_auto_te"]
                        q_sum_g += tr_list[1]["auto_te"].astype(float)

                    pg = p_sum_g / float(n_splits)
                    qg = (q_sum_g / float(n_splits)) >= 0.5
                    ens_nb_g = net_benefit_per_1k(y_b, pg, qg, contract)
                    boot_g = paired_bootstrap_nb_diff_v2(
                        y_b, pg, qg, p_gl, q_gl, contract, n_boot=10000, seed=600 + int(mm) + int(km) + bs
                    )
                    grid_rows.append(
                        {
                            "M_min": mm,
                            "kappa_max": km,
                            "B_score": bs,
                            "is_prespecified_default": bool(mm == 240.0 and km == 20.0 and bs == 8),
                            "per_split_nb_mean": float(np.mean(split_nbs_g)),
                            "per_split_nb_se": float(np.std(split_nbs_g, ddof=1) / np.sqrt(len(split_nbs_g))),
                            "ens_nb_auto": ens_nb_g,
                            "diff_vs_global_cmi": boot_g["diff_mean"],
                            "ci95_low": boot_g["ci95_low"],
                            "ci95_high": boot_g["ci95_high"],
                            "p_two_sided_dh": boot_g["p_two_sided_dh"],
                        }
                    )
        print(f"  [{label_tag}] Completed 27-cell sensitivity grid in {time.time()-ts_g:.1f}s")
        out["hyperparameter_sensitivity_grid_27"] = grid_rows

    return out


# ---------------------------------------------------------------------------
# Stage G & H: Yale Cross-Campus Track B4 (Primary) & Track B3 Fairness Audit
# ---------------------------------------------------------------------------


def run_yale_v2_cross_campus_and_fairness(manifest: dict[str, object]) -> dict[str, object]:
    yale_meta = manifest["yale_ed"]  # type: ignore[index]
    meta_full = pd.read_parquet(PROC_DIR / "yale_trackb_meta.parquet")
    df_full = pd.read_parquet(PROC_DIR / "yale_ed_full.parquet")
    df_low = pd.read_parquet(PROC_DIR / "yale_ed_esi45_lowacuity.parquet")

    fair_base_cols = [
        c
        for c in yale_meta["base_columns"]  # type: ignore[index]
        if c in df_full.columns and c not in FAIRNESS_EXCLUDED_BASE_COLS
    ]
    channels = [
        ChannelDef(cid, tuple(cinfo["columns"]), cost=float(cinfo["cost"]))
        for cid, cinfo in yale_meta["channels"].items()  # type: ignore[index]
    ]
    ch_cols_map = {c.channel_id: list(c.columns) for c in channels}

    vc = meta_full["dep_name"].value_counts()
    campus_train = str(vc.index[0])
    campus_ext = str(vc.index[1])
    m_c1 = (meta_full["dep_name"] == campus_train).to_numpy()
    c1_pids = set(meta_full.loc[m_c1, "pat_mrn_id"].to_numpy())
    m_c2_clean = (
        (meta_full["dep_name"] == campus_ext) & (~meta_full["pat_mrn_id"].isin(c1_pids))
    ).to_numpy()

    out_b4: dict[str, object] = {}

    for tag, esi_only, contract, n_splits, n_tr_cap, n_ca_cap in [
        (
            "b4a_esi45_cross_campus",
            True,
            ClinicalContract(cost_fp=1.0, cost_fn=24.0, target_precision=0.25, per_instance_query_cost=0.025),
            6,
            50000,
            15000,
        ),
        (
            "b4b_allcomers_cross_campus",
            False,
            ClinicalContract(cost_fp=1.0, cost_fn=5.6667, target_precision=0.65, per_instance_query_cost=0.03),
            5,
            65000,
            20000,
        ),
    ]:
        t0 = time.time()
        if esi_only:
            esi_m = meta_full["esi"].isin(["4", "5", "4.0", "5.0"]).to_numpy()
            idx_c1 = np.flatnonzero(m_c1 & esi_m)
            idx_c2 = np.flatnonzero(m_c2_clean & esi_m)
        else:
            idx_c1 = np.flatnonzero(m_c1)
            idx_c2 = np.flatnonzero(m_c2_clean)

        df_c1 = df_full.iloc[idx_c1].reset_index(drop=True)
        y_c1 = df_c1["y_admit"].to_numpy(dtype=int)
        pids_c1 = meta_full.iloc[idx_c1]["pat_mrn_id"].to_numpy()

        df_c2 = df_full.iloc[idx_c2].reset_index(drop=True)
        y_c2 = df_c2["y_admit"].to_numpy(dtype=int)
        meta_c2 = meta_full.iloc[idx_c2].reset_index(drop=True)
        n_c2 = y_c2.size

        print(
            f"[Yale {tag}] Running {n_splits} splits (Campus {campus_train} N={len(df_c1):,} -> Campus {campus_ext} N={n_c2:,})..."
        )
        arm_names = [
            "paper1_global_cmi_global_s2",
            "ablation_global_cmi_fluid_s2",
            "ablation_cz_disc_global_s2",
            "paper2_cz_disc_fluid_s2",
        ]
        ext_metrics: dict[str, list[dict[str, float]]] = {a: [] for a in arm_names}
        sel_glob: dict[str, int] = {}
        sel_cz: dict[str, int] = {}
        ens_p: dict[str, np.ndarray] = {a: np.zeros(n_c2, dtype=float) for a in ["stage1_base"] + arm_names}
        ens_q: dict[str, np.ndarray] = {a: np.zeros(n_c2, dtype=float) for a in ["stage1_base"] + arm_names}
        ens_p2: dict[str, np.ndarray] = {a: np.zeros(n_c2, dtype=float) for a in ["stage1_base"] + arm_names}
        cal_y_pool: list[np.ndarray] = []
        cal_p_pool: list[np.ndarray] = []

        for s in range(n_splits):
            ts = time.time()
            tr, ca, _ = make_patient_grouped_split(
                pids_c1, y_c1, seed=5100 + s, target_train=n_tr_cap, target_cal=n_ca_cap
            )
            oof_p = compute_oof_stage1_v2(
                df_c1.iloc[tr][fair_base_cols], y_c1[tr], n_folds=4, seed=5200 + s, n_jobs=4
            )
            d_glob = run_sequential_discovery_v2(
                df_c1.iloc[tr],
                y_c1[tr],
                fair_base_cols,
                channels,
                oof_p,
                contract,
                method="global_cmi",
                pilot_fraction=0.15,
                budget=3.0,
                n_jobs=4,
                seed=5300 + s,
            ).outcome
            d_cz = run_sequential_discovery_v2(
                df_c1.iloc[tr],
                y_c1[tr],
                fair_base_cols,
                channels,
                oof_p,
                contract,
                method="collision_zone",
                pilot_fraction=0.15,
                budget=3.0,
                adaptive_kappa=True,
                n_jobs=4,
                seed=5300 + s,
            ).outcome
            for cid in d_glob.selected_ids:
                sel_glob[cid] = sel_glob.get(cid, 0) + 1
            for cid in d_cz.selected_ids:
                sel_cz[cid] = sel_cz.get(cid, 0) + 1

            cols_glob = [col for cid in d_glob.selected_ids for col in ch_cols_map[cid]]
            cols_cz = [col for cid in d_cz.selected_ids for col in ch_cols_map[cid]]

            s1_cache = fit_stage1_predictions(
                df_c1.iloc[tr],
                y_c1[tr],
                df_c1.iloc[ca],
                y_c1[ca],
                [df_c2],
                fair_base_cols,
                seed=5400 + s,
            )
            ens_p["stage1_base"] += s1_cache[1][0]
            ens_p2["stage1_base"] += s1_cache[1][0]

            for arm_name, scol, s2_mode in [
                ("paper1_global_cmi_global_s2", cols_glob, "global"),
                ("ablation_global_cmi_fluid_s2", cols_glob, "fluid_backbone"),
                ("ablation_cz_disc_global_s2", cols_cz, "global"),
                ("paper2_cz_disc_fluid_s2", cols_cz, "fluid_backbone"),
            ]:
                traces: list[dict[str, np.ndarray]] = []
                cal_tr: dict[str, np.ndarray] = {}
                ev_ext = evaluate_stage2_multi_test_v2(
                    df_c1.iloc[tr],
                    y_c1[tr],
                    df_c1.iloc[ca],
                    y_c1[ca],
                    [(df_c2, y_c2)],
                    fair_base_cols,
                    scol,
                    oof_p,
                    contract,
                    stage2_mode=s2_mode,
                    voi_mode="empirical_martingale",
                    precomputed_stage1=s1_cache,
                    out_routed_traces=traces,
                    out_cal_trace=cal_tr,
                    n_jobs=4,
                    seed=5400 + s,
                )[0]
                ext_metrics[arm_name].append(ev_ext)
                ens_p[arm_name] += traces[0]["p_auto_te"]
                ens_q[arm_name] += traces[0]["auto_te"].astype(float)
                ens_p2[arm_name] += traces[0]["p2_te"]
                if arm_name == "paper2_cz_disc_fluid_s2":
                    cal_y_pool.append(y_c1[ca])
                    cal_p_pool.append(cal_tr["p_auto_cal"])

            print(
                f"  [{tag} {s+1}/{n_splits}] ({time.time()-ts:.1f}s) "
                f"Glob={ext_metrics['paper1_global_cmi_global_s2'][-1]['nb_voi_auto']:.2f} | "
                f"CZ={ext_metrics['paper2_cz_disc_fluid_s2'][-1]['nb_voi_auto']:.2f}"
            )

        prev_c2 = float(np.mean(y_c2))
        ens_levels: dict[str, dict[str, float]] = {}
        ens_preds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for a in ["stage1_base"] + arm_names:
            pa = ens_p[a] / float(n_splits)
            qa = (ens_q[a] / float(n_splits)) >= 0.5
            p2a = ens_p2[a] / float(n_splits)
            nb_e = net_benefit_per_1k(y_c2, pa, qa, contract)
            ens_preds[a] = (pa, qa)
            ens_levels[a] = {
                "ens_nb_auto": nb_e,
                "ens_snb_auto": standardized_net_benefit(nb_e, prev_c2),
                "ens_query_rate_auto": float(np.mean(qa)),
                "ens_auroc_routed": float(roc_auc_score(y_c2, pa)),
                "ens_auprc_routed": float(average_precision_score(y_c2, pa)),
                "ens_auroc_full_s2": float(roc_auc_score(y_c2, p2a)),
                "ens_auprc_full_s2": float(average_precision_score(y_c2, p2a)),
                "per_split_mean_nb_auto": (
                    float(np.mean([r["nb_voi_auto"] for r in ext_metrics[a]]))
                    if a != "stage1_base"
                    else float(np.mean([r["nb_base_0pct"] for r in ext_metrics["paper1_global_cmi_global_s2"]]))
                ),
            }

        p_bs, q_bs = ens_preds["stage1_base"]
        p_gl, q_gl = ens_preds["paper1_global_cmi_global_s2"]
        p_gf, q_gf = ens_preds["ablation_global_cmi_fluid_s2"]
        p_cg, q_cg = ens_preds["ablation_cz_disc_global_s2"]
        p_cz, q_cz = ens_preds["paper2_cz_disc_fluid_s2"]

        yc_pool = np.concatenate(cal_y_pool)
        pc_pool = np.concatenate(cal_p_pool)

        sub_c2_audit: dict[str, object] = {}
        if esi_only:
            for attr in ("race", "lang", "insurance_status"):
                vals_attr = meta_c2[attr].fillna("Missing").astype(str).to_numpy()
                for gval in np.unique(vals_attr):
                    if gval in {"Missing", "nan", "None", "Unknown", "<NA>", ""}:
                        continue
                    gm = vals_attr == gval
                    if int(np.sum(gm)) >= 250 and int(np.sum(y_c2[gm])) >= 5:
                        bt_sub = paired_bootstrap_nb_diff_v2(
                            y_c2[gm], p_cz[gm], q_cz[gm], p_bs[gm], q_bs[gm], contract, n_boot=5000, seed=101
                        )
                        sub_c2_audit[f"{attr}:{gval}"] = {
                            "n": int(np.sum(gm)),
                            "prev": float(np.mean(y_c2[gm])),
                            "ens_nb_stage1": net_benefit_per_1k(y_c2[gm], p_bs[gm], q_bs[gm], contract),
                            "ens_nb_cz_fluid": net_benefit_per_1k(y_c2[gm], p_cz[gm], q_cz[gm], contract),
                            "delta_nb": bt_sub["diff_mean"],
                            "ci95_low": bt_sub["ci95_low"],
                            "ci95_high": bt_sub["ci95_high"],
                            "p_two_sided_dh": bt_sub["p_two_sided_dh"],
                            "citl_cz_fluid": solve_citl_intercept(y_c2[gm], p_cz[gm]),
                        }

        out_b4[tag] = {
            "train_campus": campus_train,
            "external_campus": campus_ext,
            "train_campus_visits": len(df_c1),
            "external_campus_visits": n_c2,
            "train_prev": float(np.mean(y_c1)),
            "external_prev": prev_c2,
            "n_splits": n_splits,
            "elapsed_sec": round(time.time() - t0, 2),
            "selected_channel_counts_global_cmi": sel_glob,
            "selected_channel_counts_collision_zone": sel_cz,
            "external_per_split": {
                a: {k: _agg([r[k] for r in rows]) for k in rows[0]} for a, rows in ext_metrics.items()
            },
            "external_symmetric_ensemble": ens_levels,
            "paired_bootstraps_10k": {
                "cz_fluid_vs_stage1_base": paired_bootstrap_nb_diff_v2(
                    y_c2, p_cz, q_cz, p_bs, q_bs, contract, n_boot=10000, seed=61
                ),
                "cz_fluid_vs_global_cmi": paired_bootstrap_nb_diff_v2(
                    y_c2, p_cz, q_cz, p_gl, q_gl, contract, n_boot=10000, seed=62
                ),
                "global_fluid_vs_global_cmi": paired_bootstrap_nb_diff_v2(
                    y_c2, p_gf, q_gf, p_gl, q_gl, contract, n_boot=10000, seed=63
                ),
                "cz_global_vs_global_cmi": paired_bootstrap_nb_diff_v2(
                    y_c2, p_cg, q_cg, p_gl, q_gl, contract, n_boot=10000, seed=64
                ),
            },
            "superadditivity_interaction_10k": paired_bootstrap_interaction(
                _unit_nb_vector(y_c2, p_bs, q_bs, contract),
                _unit_nb_vector(y_c2, p_gl, q_gl, contract),
                _unit_nb_vector(y_c2, p_cg, q_cg, contract),
                _unit_nb_vector(y_c2, p_gf, q_gf, contract),
                _unit_nb_vector(y_c2, p_cz, q_cz, contract),
                n_boot=10000,
                seed=65,
            ),
            "calibration": {
                "stage1_base": compute_calibration_metrics(y_c2, p_bs, n_boot=300, seed=66, return_curve=False),
                "global_cmi": compute_calibration_metrics(y_c2, p_gl, n_boot=300, seed=67, return_curve=True),
                "cz_fluid": compute_calibration_metrics(y_c2, p_cz, n_boot=300, seed=68, return_curve=True),
            },
            "clinical_impact_counts": {
                "stage1_base": clinical_impact_counts_per_1k(
                    y_c2, p_bs, q_bs, contract, y_cal=yc_pool, p_cal=pc_pool, seed=69
                ),
                "global_cmi": clinical_impact_counts_per_1k(
                    y_c2, p_gl, q_gl, contract, y_cal=yc_pool, p_cal=pc_pool, seed=70
                ),
                "cz_fluid": clinical_impact_counts_per_1k(
                    y_c2, p_cz, q_cz, contract, y_cal=yc_pool, p_cal=pc_pool, seed=71
                ),
            },
            "external_campus_b_subgroup_audit": sub_c2_audit,
        }

    # Track B3: Fairness-Blinded ESI 4-5 (N=152,995, R=6) with OOF Ensemble + pat_mrn_id CIs & CITL across 11 Subgroups
    t_b3 = time.time()
    esi_mask = meta_full["esi"].isin(["4", "5", "4.0", "5.0"]).to_numpy()
    meta_low = meta_full.iloc[np.flatnonzero(esi_mask)].reset_index(drop=True)
    y_low = df_low["y_admit"].to_numpy(dtype=int)
    pids_low = meta_low["pat_mrn_id"].to_numpy()
    n_low = y_low.size
    contract_low = ClinicalContract(
        cost_fp=1.0, cost_fn=24.0, target_precision=0.25, per_instance_query_cost=0.025
    )
    n_splits_b3 = 6
    print(f"[Yale B3 v2] Running {n_splits_b3} splits across {n_low:,} ESI 4-5 visits for OOF subgroup CIs & CITL...")

    b3_arms = [
        "paper1_global_cmi_global_s2",
        "ablation_global_cmi_fluid_s2",
        "ablation_cz_disc_global_s2",
        "paper2_cz_disc_fluid_s2",
    ]
    b3_metrics: dict[str, list[dict[str, float]]] = {a: [] for a in b3_arms}
    oof_cnt = np.zeros(n_low, dtype=float)
    oof_p_sum: dict[str, np.ndarray] = {a: np.zeros(n_low, dtype=float) for a in ["stage1_base"] + b3_arms}
    oof_q_sum: dict[str, np.ndarray] = {a: np.zeros(n_low, dtype=float) for a in ["stage1_base"] + b3_arms}
    per_split_sub: dict[str, list[dict[str, float]]] = {}
    b3_cal_y: list[np.ndarray] = []
    b3_cal_p: list[np.ndarray] = []

    for s in range(n_splits_b3):
        ts = time.time()
        tr, ca, te = make_patient_grouped_split(
            pids_low, y_low, seed=4100 + s, target_train=60000, target_cal=20000
        )
        oof_p = compute_oof_stage1_v2(
            df_low.iloc[tr][fair_base_cols], y_low[tr], n_folds=4, seed=4200 + s, n_jobs=4
        )
        d_glob = run_sequential_discovery_v2(
            df_low.iloc[tr],
            y_low[tr],
            fair_base_cols,
            channels,
            oof_p,
            contract_low,
            method="global_cmi",
            pilot_fraction=0.15,
            budget=3.0,
            n_jobs=4,
            seed=4300 + s,
        ).outcome
        d_cz = run_sequential_discovery_v2(
            df_low.iloc[tr],
            y_low[tr],
            fair_base_cols,
            channels,
            oof_p,
            contract_low,
            method="collision_zone",
            pilot_fraction=0.15,
            budget=3.0,
            adaptive_kappa=True,
            n_jobs=4,
            seed=4300 + s,
        ).outcome
        cols_glob = [col for cid in d_glob.selected_ids for col in ch_cols_map[cid]]
        cols_cz = [col for cid in d_cz.selected_ids for col in ch_cols_map[cid]]

        s1_cache = fit_stage1_predictions(
            df_low.iloc[tr],
            y_low[tr],
            df_low.iloc[ca],
            y_low[ca],
            [df_low.iloc[te]],
            fair_base_cols,
            seed=4400 + s,
        )
        oof_cnt[te] += 1.0
        oof_p_sum["stage1_base"][te] += s1_cache[1][0]

        p2_te_split: dict[str, np.ndarray] = {}
        for arm_name, scol, s2_mode in [
            ("paper1_global_cmi_global_s2", cols_glob, "global"),
            ("ablation_global_cmi_fluid_s2", cols_glob, "fluid_backbone"),
            ("ablation_cz_disc_global_s2", cols_cz, "global"),
            ("paper2_cz_disc_fluid_s2", cols_cz, "fluid_backbone"),
        ]:
            tr_out: list[dict[str, np.ndarray]] = []
            cal_tr: dict[str, np.ndarray] = {}
            ev = evaluate_stage2_multi_test_v2(
                df_low.iloc[tr],
                y_low[tr],
                df_low.iloc[ca],
                y_low[ca],
                [(df_low.iloc[te], y_low[te])],
                fair_base_cols,
                scol,
                oof_p,
                contract_low,
                stage2_mode=s2_mode,
                voi_mode="empirical_martingale",
                precomputed_stage1=s1_cache,
                out_routed_traces=tr_out,
                out_cal_trace=cal_tr,
                n_jobs=4,
                seed=4400 + s,
            )[0]
            b3_metrics[arm_name].append(ev)
            oof_p_sum[arm_name][te] += tr_out[0]["p_auto_te"]
            oof_q_sum[arm_name][te] += tr_out[0]["auto_te"].astype(float)
            if arm_name == "paper2_cz_disc_fluid_s2":
                p2_te_split = tr_out[0]
                b3_cal_y.append(y_low[ca])
                b3_cal_p.append(cal_tr["p_auto_cal"])

        # Exact per-split subgroup audit matching v1
        y_te = y_low[te]
        p1_te = p2_te_split["p1_te"]
        pa_te = p2_te_split["p_auto_te"]
        qa_te = p2_te_split["auto_te"]
        meta_te = meta_low.iloc[te]
        for attr in ("race", "lang", "insurance_status"):
            vals_attr = meta_te[attr].fillna("Missing").astype(str).to_numpy()
            for gval in np.unique(vals_attr):
                if gval in {"Missing", "nan", "None", "Unknown", "<NA>", ""}:
                    continue
                gm = vals_attr == gval
                if int(np.sum(gm)) >= 400 and int(np.sum(y_te[gm])) >= 10:
                    nb0_g = net_benefit_per_1k(
                        y_te[gm], p1_te[gm], np.zeros(int(np.sum(gm)), dtype=bool), contract_low
                    )
                    nb2_g = net_benefit_per_1k(y_te[gm], pa_te[gm], qa_te[gm], contract_low)
                    per_split_sub.setdefault(f"{attr}:{gval}", []).append(
                        {
                            "n": float(np.sum(gm)),
                            "prev": float(np.mean(y_te[gm])),
                            "nb_stage1": nb0_g,
                            "nb_cz_fluid": nb2_g,
                            "delta_nb": nb2_g - nb0_g,
                            "query_rate": float(np.mean(qa_te[gm])),
                        }
                    )
        print(
            f"  [Yale B3 {s+1}/{n_splits_b3}] ({time.time()-ts:.1f}s) "
            f"Glob={b3_metrics['paper1_global_cmi_global_s2'][-1]['nb_voi_auto']:.2f} | "
            f"CZ={b3_metrics['paper2_cz_disc_fluid_s2'][-1]['nb_voi_auto']:.2f}"
        )

    valid_m = oof_cnt > 0
    yv_oof = y_low[valid_m]
    pids_oof = pids_low[valid_m]
    meta_oof = meta_low.iloc[np.flatnonzero(valid_m)].reset_index(drop=True)

    b3_oof_ens: dict[str, dict[str, float]] = {}
    b3_oof_preds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for a in ["stage1_base"] + b3_arms:
        pa = oof_p_sum[a][valid_m] / oof_cnt[valid_m]
        qa = (oof_q_sum[a][valid_m] / oof_cnt[valid_m]) >= 0.5
        nb_e = net_benefit_per_1k(yv_oof, pa, qa, contract_low)
        b3_oof_preds[a] = (pa, qa)
        b3_oof_ens[a] = {
            "oof_ens_nb_auto": nb_e,
            "oof_ens_snb_auto": standardized_net_benefit(nb_e, float(np.mean(yv_oof))),
            "oof_ens_query_rate": float(np.mean(qa)),
            "per_split_mean_nb_auto": (
                float(np.mean([r["nb_voi_auto"] for r in b3_metrics[a]]))
                if a != "stage1_base"
                else float(np.mean([r["nb_base_0pct"] for r in b3_metrics["paper1_global_cmi_global_s2"]]))
            ),
        }

    p_bs_3, q_bs_3 = b3_oof_preds["stage1_base"]
    p_gl_3, q_gl_3 = b3_oof_preds["paper1_global_cmi_global_s2"]
    p_cz_3, q_cz_3 = b3_oof_preds["paper2_cz_disc_fluid_s2"]

    subgroup_11_with_cis: dict[str, object] = {}
    for idx_g, (k_sub, split_rows) in enumerate(sorted(per_split_sub.items())):
        attr, gval = k_sub.split(":", 1)
        gm = meta_oof[attr].fillna("Missing").astype(str).to_numpy() == gval
        bt_cz_vs_base = paired_bootstrap_nb_diff_v2(
            yv_oof[gm],
            p_cz_3[gm],
            q_cz_3[gm],
            p_bs_3[gm],
            q_bs_3[gm],
            contract_low,
            cluster_ids=pids_oof[gm],
            n_boot=5000,
            seed=300 + idx_g,
        )
        bt_cz_vs_glob = paired_bootstrap_nb_diff_v2(
            yv_oof[gm],
            p_cz_3[gm],
            q_cz_3[gm],
            p_gl_3[gm],
            q_gl_3[gm],
            contract_low,
            cluster_ids=pids_oof[gm],
            n_boot=5000,
            seed=400 + idx_g,
        )
        cal_sub = compute_calibration_metrics(yv_oof[gm], p_cz_3[gm], n_boot=200, seed=500 + idx_g)
        subgroup_11_with_cis[k_sub] = {
            "mean_test_n_per_split": float(np.mean([x["n"] for x in split_rows])),
            "oof_unique_n": int(np.sum(gm)),
            "prevalence": float(np.mean(yv_oof[gm])),
            "per_split_nb_stage1": float(np.mean([x["nb_stage1"] for x in split_rows])),
            "per_split_nb_cz_fluid": float(np.mean([x["nb_cz_fluid"] for x in split_rows])),
            "per_split_delta_nb": float(np.mean([x["delta_nb"] for x in split_rows])),
            "per_split_query_rate": float(np.mean([x["query_rate"] for x in split_rows])),
            "oof_ens_nb_stage1": net_benefit_per_1k(yv_oof[gm], p_bs_3[gm], q_bs_3[gm], contract_low),
            "oof_ens_nb_cz_fluid": net_benefit_per_1k(yv_oof[gm], p_cz_3[gm], q_cz_3[gm], contract_low),
            "oof_ens_delta_nb": bt_cz_vs_base["diff_mean"],
            "oof_ens_delta_ci95_low": bt_cz_vs_base["ci95_low"],
            "oof_ens_delta_ci95_high": bt_cz_vs_base["ci95_high"],
            "oof_ens_delta_p_dh": bt_cz_vs_base["p_two_sided_dh"],
            "oof_ens_cz_vs_glob_diff": bt_cz_vs_glob["diff_mean"],
            "oof_ens_cz_vs_glob_ci95": [bt_cz_vs_glob["ci95_low"], bt_cz_vs_glob["ci95_high"]],
            "citl_cz_fluid": cal_sub["citl"],
            "citl_ci95": cal_sub["citl_ci95"],
            "ici_loess": cal_sub["ici_loess"],
            "oe_ratio": cal_sub["oe_ratio"],
        }

    out_b3 = {
        "total_visits": n_low,
        "oof_covered_visits": int(np.sum(valid_m)),
        "public_archive_mrn_note": (
            "Hong et al. (2018) CC BY 4.0 release strips longitudinal MRNs prior to public archiving; "
            "pat_mrn_id resolves to 152,995 unique encounter IDs, motivating Cross-Campus Track B4 as primary."
        ),
        "n_splits": n_splits_b3,
        "elapsed_sec": round(time.time() - t_b3, 2),
        "per_split_arms": {
            a: {k: _agg([r[k] for r in rows]) for k in rows[0]} for a, rows in b3_metrics.items()
        },
        "oof_symmetric_ensemble": b3_oof_ens,
        "oof_paired_bootstraps_10k": {
            "cz_fluid_vs_stage1_base": paired_bootstrap_nb_diff_v2(
                yv_oof, p_cz_3, q_cz_3, p_bs_3, q_bs_3, contract_low, cluster_ids=pids_oof, n_boot=10000, seed=310
            ),
            "cz_fluid_vs_global_cmi": paired_bootstrap_nb_diff_v2(
                yv_oof, p_cz_3, q_cz_3, p_gl_3, q_gl_3, contract_low, cluster_ids=pids_oof, n_boot=10000, seed=311
            ),
        },
        "calibration_overall": {
            "stage1_base": compute_calibration_metrics(yv_oof, p_bs_3, n_boot=300, seed=312),
            "global_cmi": compute_calibration_metrics(yv_oof, p_gl_3, n_boot=300, seed=313),
            "cz_fluid": compute_calibration_metrics(yv_oof, p_cz_3, n_boot=300, seed=314, return_curve=True),
        },
        "clinical_impact_counts": clinical_impact_counts_per_1k(
            yv_oof,
            p_cz_3,
            q_cz_3,
            contract_low,
            y_cal=np.concatenate(b3_cal_y),
            p_cal=np.concatenate(b3_cal_p),
            seed=315,
        ),
        "fairness_subgroups_11_with_cis_and_citl": subgroup_11_with_cis,
    }

    return {"yale_b4_cross_campus_primary": out_b4, "yale_b3_fairness_blinded": out_b3}


# ---------------------------------------------------------------------------
# Stage I: P2-13 — Analytical & Empirical (tau*, kappa) Frontiers
# ---------------------------------------------------------------------------


def run_tau_kappa_frontiers(manifest: dict[str, object]) -> dict[str, object]:
    print("[Frontier P2-13] Computing analytical & empirical (tau*, kappa) frontiers...")
    k_grid = np.linspace(0.0, 36.0, 73).tolist()

    # 1. BIDMC Sepsis 6h (tau* = 0.0625)
    df_sep = pd.read_parquet(PROC_DIR / "sepsis_setA_bidmc_6h.parquet")
    sep_cols: list[str] = list(manifest["sepsis_6h"]["base_columns"])  # type: ignore[index]
    y_sep = df_sep["y_sepsis_future"].to_numpy(dtype=int)
    tr_s, _, _ = make_stratified_3way_split(y_sep, seed=700)
    oof_sep = compute_oof_stage1_v2(df_sep.iloc[tr_s][sep_cols], y_sep[tr_s], n_folds=4, seed=800, n_jobs=4)
    fr_sepsis = compute_tau_kappa_frontier(oof_sep, 0.0625, k_grid, min_cz_ess=240.0)

    # 2. Yale ESI 4-5 (tau* = 0.04)
    df_low = pd.read_parquet(PROC_DIR / "yale_ed_esi45_lowacuity.parquet")
    y_low = df_low["y_admit"].to_numpy(dtype=int)
    yale_cols = [
        c
        for c in manifest["yale_ed"]["base_columns"]  # type: ignore[index]
        if c in df_low.columns and c not in FAIRNESS_EXCLUDED_BASE_COLS
    ]
    tr_l, _, _ = make_stratified_3way_split(y_low, seed=2100, train_frac=60000.0 / len(y_low), cal_frac=20000.0 / len(y_low))
    oof_low = compute_oof_stage1_v2(df_low.iloc[tr_l][yale_cols], y_low[tr_l], n_folds=4, seed=2200, n_jobs=4)
    fr_yale_low = compute_tau_kappa_frontier(oof_low, 0.04, k_grid, min_cz_ess=240.0)

    # 3. Yale All-Comers (tau* = 0.15)
    df_full = pd.read_parquet(PROC_DIR / "yale_ed_full.parquet")
    y_full = df_full["y_admit"].to_numpy(dtype=int)
    tr_f, _, _ = make_stratified_3way_split(y_full, seed=3100, train_frac=75000.0 / len(y_full), cal_frac=25000.0 / len(y_full))
    oof_full = compute_oof_stage1_v2(df_full.iloc[tr_f][yale_cols], y_full[tr_f], n_folds=4, seed=3200, n_jobs=4)
    fr_yale_full = compute_tau_kappa_frontier(oof_full, 0.15, k_grid, min_cz_ess=240.0)

    return {
        "sepsis_bidmc_tau_0p0625": fr_sepsis,
        "yale_esi45_tau_0p04": fr_yale_low,
        "yale_allcomers_tau_0p15": fr_yale_full,
    }


def main() -> None:
    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((PROC_DIR / "ws4tb_cohorts_manifest.json").read_text())
    sepsis_base_cols: list[str] = list(manifest["sepsis_6h"]["base_columns"])

    results: dict[str, object] = {}
    if OUT_JSON.exists():
        try:
            results = json.loads(OUT_JSON.read_text())
            print(f"[Resume] Loaded existing keys from {OUT_JSON.name}: {list(results.keys())}")
        except Exception:
            results = {}

    def _save() -> None:
        OUT_JSON.write_text(json.dumps(results, indent=2))
        print(f"  [Checkpoint] Saved {OUT_JSON}")

    # 1. Ensure 4h, 6h, 8h, 12h landmark Parquets exist
    landmark_paths = ensure_staged_sepsis_landmarks([4, 6, 8, 12])

    # 2. P2-5: 1,000-Replicate Sim 1 Dilution & Null FWER + Empirical Null Gain
    if "sim1_1000_replicates" not in results:
        results["sim1_1000_replicates"] = run_sim1_1000_replicates(n_reps=1000)
        _save()

    # 3. Primary Endpoint Track B1b (6h Measured CPT Bloodwork + Covert-AFA + Calibration + 27-Cell Sensitivity Grid)
    df_a6 = pd.read_parquet(landmark_paths[6][0])
    df_b6 = pd.read_parquet(landmark_paths[6][1])
    df_a6_meas = df_a6[(df_a6["s2_Creatinine_drawn"] == 1) & (df_a6["s2_WBC_drawn"] == 1)].reset_index(drop=True)
    df_b6_meas = df_b6[(df_b6["s2_Creatinine_drawn"] == 1) & (df_b6["s2_WBC_drawn"] == 1)].reset_index(drop=True)

    if "sepsis_track_b1b_primary" not in results:
        results["sepsis_track_b1b_primary"] = run_sepsis_v2_cohort(
            df_a6_meas,
            df_b6_meas,
            sepsis_base_cols,
            _build_venipuncture_channels(df_a6_meas),
            label_tag="Track B1b Primary (6h Measured CPT)",
            n_splits=10,
            deconfound_stage2=False,
            include_naive=False,
            include_covert_afa=True,
            include_calibration=True,
            include_recalibration_rescue=True,
            run_sensitivity_grid=True,
        )
        _save()

    # 4. Track A Parity (6h Full Cohort + Naive + Covert-AFA + Superadditivity Interaction + Calibration)
    if "sepsis_track_a_parity" not in results:
        results["sepsis_track_a_parity"] = run_sepsis_v2_cohort(
            df_a6,
            df_b6,
            sepsis_base_cols,
            _build_parity_sepsis_channels(df_a6),
            label_tag="Track A Parity (6h Full)",
            n_splits=10,
            deconfound_stage2=False,
            include_naive=True,
            include_covert_afa=True,
            include_calibration=True,
            include_recalibration_rescue=True,
            run_sensitivity_grid=False,
        )
        _save()

    # 5. Track B1a (Agniel Deconfounded Sparse Stress, R=8)
    if "sepsis_track_b1a_agniel_sparse" not in results:
        results["sepsis_track_b1a_agniel_sparse"] = run_sepsis_v2_cohort(
            df_a6,
            df_b6,
            sepsis_base_cols,
            _build_venipuncture_channels(df_a6),
            label_tag="Track B1a Agniel Sparse (6h)",
            n_splits=8,
            deconfound_stage2=True,
            include_naive=False,
            include_covert_afa=False,
            include_calibration=True,
            include_recalibration_rescue=True,
            run_sensitivity_grid=False,
        )
        _save()

    # 6. Track B2 (Strict 4h+2h Lead-Time Gap + Prevalence-Shift Intercept Recalibration Rescue, R=8)
    if "sepsis_track_b2_gap_and_recal_rescue" not in results:
        df_agap = pd.read_parquet(PROC_DIR / "sepsis_setA_bidmc_4h_gap2h.parquet")
        df_bgap = pd.read_parquet(PROC_DIR / "sepsis_setB_emory_4h_gap2h.parquet")
        results["sepsis_track_b2_gap_and_recal_rescue"] = run_sepsis_v2_cohort(
            df_agap,
            df_bgap,
            sepsis_base_cols,
            _build_venipuncture_channels(df_agap),
            label_tag="Track B2 Strict 4h+2h Gap",
            n_splits=8,
            deconfound_stage2=False,
            include_naive=False,
            include_covert_afa=False,
            include_calibration=True,
            include_recalibration_rescue=True,
            run_sensitivity_grid=False,
        )
        _save()

    # 7. P2-14: PhysioNet Landmark Sweep (4h, 6h, 8h, 12h) for BOTH Measured CPT (B1b) and Full Parity (Track A)
    if "sepsis_landmark_sweep" not in results:
        sweep_b1b: dict[str, object] = {}
        sweep_track_a: dict[str, object] = {}
        for w in [4, 6, 8, 12]:
            if w == 6:
                b1b_6 = results["sepsis_track_b1b_primary"]  # type: ignore[assignment]
                ta_6 = results["sepsis_track_a_parity"]  # type: ignore[assignment]
                sweep_b1b["6h"] = {
                    "bidmc_rows": b1b_6["bidmc_rows"],  # type: ignore[index]
                    "emory_rows": b1b_6["emory_rows"],  # type: ignore[index]
                    "bidmc_prev": b1b_6["bidmc_prev"],  # type: ignore[index]
                    "emory_prev": b1b_6["emory_prev"],  # type: ignore[index]
                    "external_emory_per_split": b1b_6["external_emory_per_split"],  # type: ignore[index]
                    "external_emory_symmetric_ensemble": b1b_6["external_emory_symmetric_ensemble"],  # type: ignore[index]
                    "emory_paired_bootstraps_10k": b1b_6["emory_paired_bootstraps_10k"],  # type: ignore[index]
                }
                sweep_track_a["6h"] = {
                    "bidmc_rows": ta_6["bidmc_rows"],  # type: ignore[index]
                    "emory_rows": ta_6["emory_rows"],  # type: ignore[index]
                    "bidmc_prev": ta_6["bidmc_prev"],  # type: ignore[index]
                    "emory_prev": ta_6["emory_prev"],  # type: ignore[index]
                    "external_emory_per_split": ta_6["external_emory_per_split"],  # type: ignore[index]
                    "external_emory_symmetric_ensemble": ta_6["external_emory_symmetric_ensemble"],  # type: ignore[index]
                    "emory_paired_bootstraps_10k": ta_6["emory_paired_bootstraps_10k"],  # type: ignore[index]
                }
                continue

            dfa_w = pd.read_parquet(landmark_paths[w][0])
            dfb_w = pd.read_parquet(landmark_paths[w][1])
            dfa_wm = dfa_w[(dfa_w["s2_Creatinine_drawn"] == 1) & (dfa_w["s2_WBC_drawn"] == 1)].reset_index(drop=True)
            dfb_wm = dfb_w[(dfb_w["s2_Creatinine_drawn"] == 1) & (dfb_w["s2_WBC_drawn"] == 1)].reset_index(drop=True)

            res_wm = run_sepsis_v2_cohort(
                dfa_wm,
                dfb_wm,
                sepsis_base_cols,
                _build_venipuncture_channels(dfa_wm),
                label_tag=f"Landmark {w}h Measured CPT",
                n_splits=10,
                deconfound_stage2=False,
                include_calibration=False,
            )
            res_wa = run_sepsis_v2_cohort(
                dfa_w,
                dfb_w,
                sepsis_base_cols,
                _build_parity_sepsis_channels(dfa_w),
                label_tag=f"Landmark {w}h Full Parity",
                n_splits=10,
                deconfound_stage2=False,
                include_calibration=False,
            )
            sweep_b1b[f"{w}h"] = res_wm
            sweep_track_a[f"{w}h"] = res_wa

        results["sepsis_landmark_sweep"] = {
            "measured_bloodwork_cpt": sweep_b1b,
            "full_riskset_parity": sweep_track_a,
        }
        _save()

    # 8. Yale Cross-Campus Track B4 (Primary) & Track B3 Fairness Audit
    if "yale_strengthened" not in results:
        results["yale_strengthened"] = run_yale_v2_cross_campus_and_fairness(manifest)
        _save()

    # 9. P2-13: (tau*, kappa) Frontiers + Archived Reference Cross-Links
    if "tau_kappa_frontiers" not in results:
        results["tau_kappa_frontiers"] = run_tau_kappa_frontiers(manifest)
        arch_init = json.loads((ARCHIVED_RES_DIR / "paper2_initial_campaign_summary.json").read_text())
        arch_sim2 = json.loads((ARCHIVED_RES_DIR / "sim2_leaf_starvation_summary.json").read_text())
        results["archived_reference_summaries"] = {
            "sim2_leaf_starvation": arch_sim2,
            "yale_track_a_esi45_parity": arch_init["yale_ed_esi45_lowacuity"],
            "yale_track_a_full_560k_parity": arch_init["yale_ed_full_560k"],
        }
        _save()

    print("\n=== ALL PAPER 2 v2 STRENGTHENING SUITES COMPLETED SUCCESSFULLY ===")


if __name__ == "__main__":
    main()
