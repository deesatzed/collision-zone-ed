"""Run Paper 2 Track B (Hardened Clinical, Causal & Cross-Campus Audit) alongside Track A Parity.

Executes four hardened stress tests addressing every biostatistical & clinical-informatics critique:
  1. B1 — Sepsis Hardened Venipuncture Assay-Only Transfer (Beth Israel -> Emory, N_ext=19,696, R=10):
     Removes ventilator settings (FiO2/SaO2/EtCO2) & ordering indicators (_drawn), applies
     AgnielDeconfounder (enforcing Z_tilde _|_ drawn | X_base), uses CMS Medicare CPT dollar costs,
     Nonparametric Empirical Martingale VOI, 1,000-rep patient-level paired bootstrap CIs/p-values,
     and full Vickers Decision Curves across tau in [2%, 16%].
  2. B2 — Sepsis Strict Gap-Window Transfer (1-4h Obs -> Sepsis-Free Through 6h -> Onset >= 7h, R=8):
     Eliminates any proximity to PhysioNet's 6h pre-shifted SOFA onset window.
  3. B3 — Yale ED ESI 4-5 Fairness-Blinded + Patient-Grouped (pat_mrn_id) Triage (152,995 visits, R=6):
     Strips race/ethnicity/lang/religion/marital/employment/insurance/name from X_base, enforces zero
     patient overlap across train/cal/test, and audits subgroup Net Benefit across race, language,
     and insurance status.
  4. B4 — Second Multi-Hospital Zero-Shot External Transfer (Yale Academic Trauma Campus -> Community
     Urban Campus, zero retraining & 100% patient-disjoint across hospitals):
     Evaluates both External Low-Acuity ESI 4-5 (R=6) and External All-Comers (R=5) with patient-level
     paired bootstrap 95% CIs and Vickers Decision Curves.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from collision_zone.three_predicates import (
    ChannelDef,
    ClinicalContract,
    compute_decision_curve,
    compute_oof_stage1,
    evaluate_stage2_multi_test,
    fit_stage1_predictions,
    net_benefit_per_1k,
    paired_bootstrap_nb_diff,
    run_sequential_discovery,
)

RAW_DIR = Path("/Volumes/WS4TB/imboreceipt_data")
PROC_DIR = RAW_DIR / "processed"
REPO_ROOT = Path(__file__).resolve().parents[1]
RES_DIR = REPO_ROOT / "results"

# Published CMS Medicare Clinical Laboratory Fee Schedule (CPT) relative blood-draw costs
# Normalized to CPT 83605 (Lactate = 1.00); excludes bedside ventilator settings (FiO2/SaO2/EtCO2)
CMS_CPT_VENIPUNCTURE_CHANNELS: dict[str, tuple[list[str], float]] = {
    "lab_lactate": (["Lactate"], 1.00),  # CPT 83605
    "lab_blood_gas_acidbase": (["pH", "PaCO2", "BaseExcess", "HCO3"], 1.79),  # CPT 82803
    "lab_wbc": (["WBC"], 0.78),  # CPT 85025/85048
    "lab_hemoglobin": (["Hgb", "Hct"], 0.78),  # CPT 85018/85014
    "lab_platelets": (["Platelets"], 0.78),  # CPT 85049
    "lab_renal_azotemia": (["Creatinine", "BUN"], 0.92),  # CPT 82565 + 84520
    "lab_electrolytes": (["Potassium", "Chloride"], 0.88),  # CPT 84132 + 82435
    "lab_glucose": (["Glucose"], 0.55),  # CPT 82947
    "lab_ca_mg_phos": (["Calcium", "Magnesium", "Phosphate"], 1.45),  # CPT 82310+83735+84100
    "lab_liver_enzymes": (["AST", "Alkalinephos"], 0.95),  # CPT 84450 + 84075
    "lab_bilirubin": (["Bilirubin_total", "Bilirubin_direct"], 0.95),  # CPT 82247 + 82248
    "lab_coag_ptt": (["PTT"], 0.85),  # CPT 85730
    "lab_fibrinogen": (["Fibrinogen"], 1.05),  # CPT 85384
    "lab_troponin_i": (["TroponinI"], 1.15),  # CPT 84484
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

VITALS_7 = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"]


def _agg(vals: list[float]) -> dict[str, float]:
    arr = np.asarray(vals, dtype=float)
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
    """Strict patient-grouped split ensuring zero unique patient_ids overlap train, cal, and test."""
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


def ensure_sepsis_gap_cohorts() -> None:
    """Create strict 4h-obs + 2h-clean-buffer (sepsis-free through hour 6) cohorts if missing."""
    out_a = PROC_DIR / "sepsis_setA_bidmc_4h_gap2h.parquet"
    out_b = PROC_DIR / "sepsis_setB_emory_4h_gap2h.parquet"
    if out_a.exists() and out_b.exists():
        return

    print("[Prep] Building strict gap-window sepsis cohorts (1-4h obs, sepsis-free through 6h)...")
    zpath = RAW_DIR / "archive.zip"
    obs_h = 4
    clean_through_h = 6
    records_a: list[dict[str, float | int | str]] = []
    records_b: list[dict[str, float | int | str]] = []

    lab_cols_flat = [c for cols, _ in CMS_CPT_VENIPUNCTURE_CHANNELS.values() for c in cols]

    with zipfile.ZipFile(zpath, "r") as zf:
        psv_names = sorted(n for n in zf.namelist() if n.endswith(".psv"))
        header = zf.read(psv_names[0]).decode("utf-8").splitlines()[0].split("|")
        col_idx = {c: idx for idx, c in enumerate(header)}
        vital_idxs = [col_idx[v] for v in VITALS_7]
        lab_idxs = {c: col_idx[c] for c in lab_cols_flat}
        idx_age, idx_gender = col_idx["Age"], col_idx["Gender"]
        idx_u1, idx_u2, idx_hadm = col_idx["Unit1"], col_idx["Unit2"], col_idx["HospAdmTime"]
        idx_sepsis = col_idx["SepsisLabel"]

        for name in psv_names:
            lines = zf.read(name).decode("utf-8").splitlines()[1:]
            if len(lines) <= clean_through_h:
                continue
            mat = np.genfromtxt(lines, delimiter="|")
            if mat.ndim == 1:
                mat = mat.reshape(1, -1)
            # Must remain SepsisLabel == 0 strictly through hour 6 (rows 0..5)
            if np.any(mat[:clean_through_h, idx_sepsis] == 1.0):
                continue

            y_future = int(np.any(mat[clean_through_h:, idx_sepsis] == 1.0))
            win = mat[:obs_h, :]
            site = "BIDMC_SetA" if "training_setA" in name else "Emory_SetB"
            row: dict[str, float | int | str] = {
                "patient_id": Path(name).stem,
                "site": site,
                "y_sepsis_future": y_future,
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
                float(hr_l / sbp_l) if np.isfinite(hr_l) and np.isfinite(sbp_l) and sbp_l > 20 else np.nan
            )
            row["base_shock_index_peak"] = (
                float(hr_m / sbp_m) if np.isfinite(hr_m) and np.isfinite(sbp_m) and sbp_m > 20 else np.nan
            )
            row["base_pulse_pressure_last"] = (
                float(sbp_l - dbp_l) if np.isfinite(sbp_l) and np.isfinite(dbp_l) else np.nan
            )
            s_hr = 1 if (np.isfinite(hr_m) and hr_m > 90.0) else 0
            s_rr = 1 if (np.isfinite(rr_m) and rr_m > 20.0) else 0
            s_tp = 1 if ((np.isfinite(t_max) and t_max > 38.0) or (np.isfinite(t_min) and t_min < 36.0)) else 0
            row["base_bedside_sirs_vitals_cnt"] = int(s_hr + s_rr + s_tp)

            for c_name, c_idx in lab_idxs.items():
                vals = win[:, c_idx]
                fin = vals[np.isfinite(vals)]
                if fin.size == 0:
                    row[f"s2_{c_name}_last"] = np.nan
                    row[f"s2_{c_name}_max"] = np.nan
                    row[f"s2_{c_name}_min"] = np.nan
                else:
                    row[f"s2_{c_name}_last"] = float(fin[-1])
                    row[f"s2_{c_name}_max"] = float(np.max(fin))
                    row[f"s2_{c_name}_min"] = float(np.min(fin))

            if site == "BIDMC_SetA":
                records_a.append(row)
            else:
                records_b.append(row)

    pd.DataFrame(records_a).to_parquet(out_a, index=False)
    pd.DataFrame(records_b).to_parquet(out_b, index=False)
    print(f"[Prep] Saved {out_a.name} (N={len(records_a)}) & {out_b.name} (N={len(records_b)})")


def ensure_yale_trackb_meta() -> pd.DataFrame:
    """Extract row-aligned pat_mrn_id & campus metadata from 5v_cleandf.RData without touching Track A Parquets."""
    meta_path = PROC_DIR / "yale_trackb_meta.parquet"
    if meta_path.exists():
        return pd.read_parquet(meta_path)

    import pyreadr

    print("[Prep] Extracting pat_mrn_id & campus metadata sidecar from admissionprediction-v1.0.zip...")
    zpath = RAW_DIR / "admissionprediction-v1.0.zip"
    tmp_rdata = PROC_DIR / "_tmp_meta_5v.RData"
    member = "yaleemmlc-admissionprediction-5cf50bc/Results/5v_cleandf.RData"
    with zipfile.ZipFile(zpath, "r") as zf, zf.open(member) as src_f, open(tmp_rdata, "wb") as out_f:
        while True:
            chunk = src_f.read(16 * 1024 * 1024)
            if not chunk:
                break
            out_f.write(chunk)

    res = pyreadr.read_r(str(tmp_rdata))
    df_raw: pd.DataFrame = res[next(iter(res.keys()))]
    tmp_rdata.unlink(missing_ok=True)

    if "disposition" in df_raw.columns:
        df_raw = df_raw[df_raw["disposition"].isin(["Admit", "Discharge"])].copy()
    if "age" in df_raw.columns:
        df_raw = df_raw[(df_raw["age"] >= 18) | (df_raw["age"].isna())].copy()

    y_admit = (df_raw["disposition"] == "Admit").astype(np.int8).to_numpy()
    if "pat_mrn_id" in df_raw.columns:
        pid = df_raw["pat_mrn_id"].astype(str).to_numpy()
    elif "pat_id" in df_raw.columns:
        pid = df_raw["pat_id"].astype(str).to_numpy()
    else:
        pid = np.arange(len(df_raw)).astype(str)

    meta_df = pd.DataFrame(
        {
            "pat_mrn_id": pid,
            "dep_name": df_raw["dep_name"].astype(str).to_numpy() if "dep_name" in df_raw.columns else "Campus_1",
            "esi": df_raw["esi"].astype(str).str.strip().to_numpy() if "esi" in df_raw.columns else "Missing",
            "race": df_raw["race"].astype(str).to_numpy() if "race" in df_raw.columns else "Missing",
            "lang": df_raw["lang"].astype(str).to_numpy() if "lang" in df_raw.columns else "Missing",
            "insurance_status": (
                df_raw["insurance_status"].astype(str).to_numpy()
                if "insurance_status" in df_raw.columns
                else "Missing"
            ),
            "y_admit": y_admit,
        }
    )
    meta_df.to_parquet(meta_path, index=False)
    print(
        f"[Prep] Saved {meta_path.name}: {len(meta_df):,} visits across "
        f"{meta_df['pat_mrn_id'].nunique():,} unique patients; campuses={meta_df['dep_name'].value_counts().to_dict()}"
    )
    return meta_df


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


def run_sepsis_hardened_transfer(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    base_cols: list[str],
    *,
    label_tag: str,
    deconfound_stage2: bool = True,
    n_splits: int = 10,
) -> dict[str, object]:
    """Run Agniel-deconfounded, CPT-costed pure venipuncture sepsis transfer (BIDMC -> Emory)."""
    print(f"[{label_tag}] Running {n_splits} splits (BIDMC N={len(df_a):,} -> Emory N={len(df_b):,})...")
    channels = _build_venipuncture_channels(df_a)
    contract = ClinicalContract(
        cost_fp=1.0, cost_fn=15.0, target_precision=0.30, per_instance_query_cost=0.03
    )
    budget = 3.2
    pilot_frac = 0.20

    arms = [
        "paper1_global_cmi_global_s2",
        "ablation_cz_disc_global_s2",
        "paper2_cz_disc_fluid_s2",
    ]
    internal_metrics: dict[str, list[dict[str, float]]] = {a: [] for a in arms}
    external_metrics: dict[str, list[dict[str, float]]] = {a: [] for a in arms}
    selected_freq_global: dict[str, int] = {}
    selected_freq_cz: dict[str, int] = {}

    y_a = df_a["y_sepsis_future"].to_numpy(dtype=int)
    y_b = df_b["y_sepsis_future"].to_numpy(dtype=int)

    # Accumulate Emory patient-level routed probabilities across splits for exact patient-level paired bootstrap & DCA
    emory_p_base_sum = np.zeros(len(df_b), dtype=float)
    emory_p_p1_sum = np.zeros(len(df_b), dtype=float)
    emory_q_p1_sum = np.zeros(len(df_b), dtype=float)
    emory_p_p2_sum = np.zeros(len(df_b), dtype=float)
    emory_q_p2_sum = np.zeros(len(df_b), dtype=float)

    for s in range(n_splits):
        t_s = time.time()
        tr, ca, te = make_stratified_3way_split(y_a, seed=700 + s)
        oof_p = compute_oof_stage1(df_a.iloc[tr][base_cols], y_a[tr], n_folds=4, seed=800 + s)

        d_glob = run_sequential_discovery(
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
            seed=900 + s,
        )
        d_cz = run_sequential_discovery(
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
            seed=900 + s,
        )
        for cid in d_glob.selected_ids:
            selected_freq_global[cid] = selected_freq_global.get(cid, 0) + 1
        for cid in d_cz.selected_ids:
            selected_freq_cz[cid] = selected_freq_cz.get(cid, 0) + 1

        ch_cols_map = {c.channel_id: list(c.columns) for c in channels}
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

        for arm_name, scol, s2_mode in [
            ("paper1_global_cmi_global_s2", cols_glob, "global"),
            ("ablation_cz_disc_global_s2", cols_cz, "global"),
            ("paper2_cz_disc_fluid_s2", cols_cz, "fluid_backbone"),
        ]:
            traces: list[dict[str, np.ndarray]] = []
            ev_int, ev_ext = evaluate_stage2_multi_test(
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
                seed=1500 + s,
            )
            internal_metrics[arm_name].append(ev_int)
            external_metrics[arm_name].append(ev_ext)
            if arm_name == "paper1_global_cmi_global_s2":
                emory_p_base_sum += traces[1]["p1_te"]
                emory_p_p1_sum += traces[1]["p_auto_te"]
                emory_q_p1_sum += traces[1]["auto_te"].astype(float)
            elif arm_name == "paper2_cz_disc_fluid_s2":
                emory_p_p2_sum += traces[1]["p_auto_te"]
                emory_q_p2_sum += traces[1]["auto_te"].astype(float)

        p1_nb = external_metrics["paper1_global_cmi_global_s2"][-1]["nb_voi_auto"]
        p2_nb = external_metrics["paper2_cz_disc_fluid_s2"][-1]["nb_voi_auto"]
        print(
            f"  [{label_tag} {s+1}/{n_splits}] ({time.time()-t_s:.1f}s) "
            f"Glob={d_glob.selected_ids} | CZ={d_cz.selected_ids} | "
            f"Emory NB: P1={p1_nb:.2f} -> P2_Fluid={p2_nb:.2f}"
        )

    p_base_ens = emory_p_base_sum / float(n_splits)
    p_p1_ens = emory_p_p1_sum / float(n_splits)
    q_p1_ens = (emory_q_p1_sum / float(n_splits)) >= 0.5
    p_p2_ens = emory_p_p2_sum / float(n_splits)
    q_p2_ens = (emory_q_p2_sum / float(n_splits)) >= 0.5

    boot_p2_vs_p1 = paired_bootstrap_nb_diff(
        y_b, p_p2_ens, q_p2_ens, p_p1_ens, q_p1_ens, contract, n_boot=1000, seed=42
    )
    boot_p2_vs_base = paired_bootstrap_nb_diff(
        y_b,
        p_p2_ens,
        q_p2_ens,
        p_base_ens,
        np.zeros_like(q_p2_ens),
        contract,
        n_boot=1000,
        seed=43,
    )
    tau_grid = [0.02, 0.03, 0.04, 0.05, 0.0625, 0.08, 0.10, 0.125, 0.15]
    dca = {
        "taus": tau_grid,
        "stage1_base": compute_decision_curve(
            y_b, p_base_ens, np.zeros_like(q_p2_ens), tau_grid, 0.0
        )["nb_model"],
        "paper1_global": compute_decision_curve(
            y_b, p_p1_ens, q_p1_ens, tau_grid, contract.per_instance_query_cost
        )["nb_model"],
        "paper2_fluid": compute_decision_curve(
            y_b, p_p2_ens, q_p2_ens, tau_grid, contract.per_instance_query_cost
        )["nb_model"],
        "treat_all": compute_decision_curve(
            y_b, p_base_ens, np.zeros_like(q_p2_ens), tau_grid, 0.0
        )["nb_treat_all"],
    }

    return {
        "bidmc_rows": len(df_a),
        "emory_rows": len(df_b),
        "bidmc_prev": float(np.mean(y_a)),
        "emory_prev": float(np.mean(y_b)),
        "n_splits": n_splits,
        "selected_channel_counts_global_cmi": selected_freq_global,
        "selected_channel_counts_collision_zone": selected_freq_cz,
        "internal_bidmc_test": {
            a: {k: _agg([r[k] for r in rows]) for k in rows[0]} for a, rows in internal_metrics.items()
        },
        "external_emory_transfer": {
            a: {k: _agg([r[k] for r in rows]) for k in rows[0]} for a, rows in external_metrics.items()
        },
        "emory_paired_bootstrap_p2_vs_p1": boot_p2_vs_p1,
        "emory_paired_bootstrap_p2_vs_base": boot_p2_vs_base,
        "emory_decision_curve": dca,
    }


def run_yale_patient_grouped_fairness_esi45(
    manifest: dict[str, object],
    meta_full: pd.DataFrame,
    n_splits: int = 6,
) -> dict[str, object]:
    """Track B3: Fairness-blinded X_base + strict pat_mrn_id patient-grouped splits on Yale ESI 4-5."""
    yale_meta = manifest["yale_ed"]  # type: ignore[index]
    df_low = pd.read_parquet(PROC_DIR / "yale_ed_esi45_lowacuity.parquet")
    esi_mask = meta_full["esi"].isin(["4", "5", "4.0", "5.0"]).to_numpy()
    meta_low = meta_full.iloc[np.flatnonzero(esi_mask)].reset_index(drop=True)
    assert len(meta_low) == len(df_low), f"Row mismatch: {len(meta_low)} vs {len(df_low)}"

    fair_base_cols = [
        c
        for c in yale_meta["base_columns"]  # type: ignore[index]
        if c in df_low.columns and c not in FAIRNESS_EXCLUDED_BASE_COLS
    ]
    channels = [
        ChannelDef(cid, tuple(cinfo["columns"]), cost=float(cinfo["cost"]))
        for cid, cinfo in yale_meta["channels"].items()  # type: ignore[index]
    ]
    y = df_low["y_admit"].to_numpy(dtype=int)
    pids = meta_low["pat_mrn_id"].to_numpy()
    contract = ClinicalContract(
        cost_fp=1.0, cost_fn=24.0, target_precision=0.25, per_instance_query_cost=0.025
    )
    budget = 3.0
    pilot_frac = 0.15

    print(
        f"[Yale B3: Fairness-Blinded Patient-Grouped ESI 4-5] N={len(df_low):,} visits across "
        f"{len(np.unique(pids)):,} unique patients, {len(fair_base_cols)} blinded base cols..."
    )

    arms = [
        "paper1_global_cmi_global_s2",
        "ablation_cz_disc_global_s2",
        "paper2_cz_disc_fluid_s2",
    ]
    metrics: dict[str, list[dict[str, float]]] = {a: [] for a in arms}
    selected_freq_global: dict[str, int] = {}
    selected_freq_cz: dict[str, int] = {}
    subgroup_nb: dict[str, list[dict[str, float]]] = {}

    for s in range(n_splits):
        t_s = time.time()
        tr, ca, te = make_patient_grouped_split(pids, y, seed=4100 + s, target_train=60000, target_cal=20000)
        # Verify zero patient leakage
        assert len(set(pids[tr]).intersection(set(pids[te]))) == 0
        assert len(set(pids[ca]).intersection(set(pids[te]))) == 0

        oof_p = compute_oof_stage1(df_low.iloc[tr][fair_base_cols], y[tr], n_folds=4, seed=4200 + s)
        d_glob = run_sequential_discovery(
            df_low.iloc[tr],
            y[tr],
            fair_base_cols,
            channels,
            oof_p,
            contract,
            method="global_cmi",
            pilot_fraction=pilot_frac,
            budget=budget,
            seed=4300 + s,
        )
        d_cz = run_sequential_discovery(
            df_low.iloc[tr],
            y[tr],
            fair_base_cols,
            channels,
            oof_p,
            contract,
            method="collision_zone",
            pilot_fraction=pilot_frac,
            budget=budget,
            adaptive_kappa=True,
            seed=4300 + s,
        )
        for cid in d_glob.selected_ids:
            selected_freq_global[cid] = selected_freq_global.get(cid, 0) + 1
        for cid in d_cz.selected_ids:
            selected_freq_cz[cid] = selected_freq_cz.get(cid, 0) + 1

        ch_cols_map = {c.channel_id: list(c.columns) for c in channels}
        cols_glob = [col for cid in d_glob.selected_ids for col in ch_cols_map[cid]]
        cols_cz = [col for cid in d_cz.selected_ids for col in ch_cols_map[cid]]

        s1_cache = fit_stage1_predictions(
            df_low.iloc[tr],
            y[tr],
            df_low.iloc[ca],
            y[ca],
            [df_low.iloc[te]],
            fair_base_cols,
            seed=4400 + s,
        )
        p2_trace: list[dict[str, np.ndarray]] = []
        for arm_name, scol, s2_mode in [
            ("paper1_global_cmi_global_s2", cols_glob, "global"),
            ("ablation_cz_disc_global_s2", cols_cz, "global"),
            ("paper2_cz_disc_fluid_s2", cols_cz, "fluid_backbone"),
        ]:
            tr_out = p2_trace if arm_name == "paper2_cz_disc_fluid_s2" else None
            ev = evaluate_stage2_multi_test(
                df_low.iloc[tr],
                y[tr],
                df_low.iloc[ca],
                y[ca],
                [(df_low.iloc[te], y[te])],
                fair_base_cols,
                scol,
                oof_p,
                contract,
                stage2_mode=s2_mode,
                voi_mode="empirical_martingale",
                precomputed_stage1=s1_cache,
                out_routed_traces=tr_out,
                seed=4400 + s,
            )[0]
            metrics[arm_name].append(ev)

        # Post-hoc demographic & insurance fairness audit on Paper 2 routed predictions
        y_te = y[te]
        p1_te = p2_trace[0]["p1_te"]
        p_auto_te = p2_trace[0]["p_auto_te"]
        q_auto_te = p2_trace[0]["auto_te"]
        meta_te = meta_low.iloc[te]

        for attr in ("race", "lang", "insurance_status"):
            vals_attr = meta_te[attr].fillna("Missing").astype(str).to_numpy()
            for gval in np.unique(vals_attr):
                if gval in {"Missing", "nan", "None", "Unknown", "<NA>", ""}:
                    continue
                gm = vals_attr == gval
                if int(np.sum(gm)) >= 400 and int(np.sum(y_te[gm])) >= 10:
                    nb0_g = net_benefit_per_1k(
                        y_te[gm], p1_te[gm], np.zeros(int(np.sum(gm)), dtype=bool), contract
                    )
                    nb2_g = net_benefit_per_1k(y_te[gm], p_auto_te[gm], q_auto_te[gm], contract)
                    subgroup_nb.setdefault(f"{attr}:{gval}", []).append(
                        {
                            "n": float(np.sum(gm)),
                            "prev": float(np.mean(y_te[gm])),
                            "nb_stage1": nb0_g,
                            "nb_paper2_auto": nb2_g,
                            "delta_nb": nb2_g - nb0_g,
                            "query_rate": float(np.mean(q_auto_te[gm])),
                        }
                    )

        p1_nb = metrics["paper1_global_cmi_global_s2"][-1]["nb_voi_auto"]
        p2_nb = metrics["paper2_cz_disc_fluid_s2"][-1]["nb_voi_auto"]
        print(
            f"  [Yale B3 {s+1}/{n_splits}] ({time.time()-t_s:.1f}s, test={len(te):,}) "
            f"Glob={d_glob.selected_ids} | CZ={d_cz.selected_ids} | "
            f"Auto NB: P1={p1_nb:.2f} -> P2_Fluid={p2_nb:.2f}"
        )

    return {
        "total_visits": len(df_low),
        "unique_patients": len(np.unique(pids)),
        "blinded_base_col_count": len(fair_base_cols),
        "n_splits": n_splits,
        "selected_channel_counts_global_cmi": selected_freq_global,
        "selected_channel_counts_collision_zone": selected_freq_cz,
        "arms": {
            a: {k: _agg([r[k] for r in rows]) for k in rows[0]} for a, rows in metrics.items()
        },
        "fairness_subgroup_audit": {
            k: {
                "mean_test_n": float(np.mean([x["n"] for x in v])),
                "prevalence": float(np.mean([x["prev"] for x in v])),
                "nb_stage1": float(np.mean([x["nb_stage1"] for x in v])),
                "nb_paper2_auto": float(np.mean([x["nb_paper2_auto"] for x in v])),
                "delta_nb_gain": float(np.mean([x["delta_nb"] for x in v])),
                "query_rate": float(np.mean([x["query_rate"] for x in v])),
            }
            for k, v in sorted(subgroup_nb.items())
        },
    }


def run_yale_cross_campus_external_transfer(
    manifest: dict[str, object],
    meta_full: pd.DataFrame,
) -> dict[str, object]:
    """Track B4: Zero-shot hospital transfer across Yale ED campuses (Campus 1 -> Campus 2, zero shared patients)."""
    yale_meta = manifest["yale_ed"]  # type: ignore[index]
    df_full = pd.read_parquet(PROC_DIR / "yale_ed_full.parquet")
    vc = meta_full["dep_name"].value_counts()
    campus_train = str(vc.index[0])
    campus_ext = str(vc.index[1])

    fair_base_cols = [
        c
        for c in yale_meta["base_columns"]  # type: ignore[index]
        if c in df_full.columns and c not in FAIRNESS_EXCLUDED_BASE_COLS
    ]
    channels = [
        ChannelDef(cid, tuple(cinfo["columns"]), cost=float(cinfo["cost"]))
        for cid, cinfo in yale_meta["channels"].items()  # type: ignore[index]
    ]

    # Strict hospital AND patient disjointness: remove any patient from Campus 2 who ever visited Campus 1
    m_c1 = (meta_full["dep_name"] == campus_train).to_numpy()
    c1_pids = set(meta_full.loc[m_c1, "pat_mrn_id"].to_numpy())
    m_c2_clean = (
        (meta_full["dep_name"] == campus_ext) & (~meta_full["pat_mrn_id"].isin(c1_pids))
    ).to_numpy()

    print(
        f"[Yale B4 Cross-Campus] Train Campus='{campus_train}' (N={int(np.sum(m_c1)):,}) -> "
        f"Zero-Shot External Campus='{campus_ext}' patient-disjoint (N={int(np.sum(m_c2_clean)):,})"
    )

    out_subcohorts: dict[str, object] = {}

    for tag, esi_only, contract, n_splits, n_tr_cap, n_ca_cap in [
        (
            "esi45_lowacuity_cross_campus",
            True,
            ClinicalContract(cost_fp=1.0, cost_fn=24.0, target_precision=0.25, per_instance_query_cost=0.025),
            6,
            50000,
            15000,
        ),
        (
            "allcomers_cross_campus",
            False,
            ClinicalContract(cost_fp=1.0, cost_fn=5.6667, target_precision=0.65, per_instance_query_cost=0.03),
            5,
            65000,
            20000,
        ),
    ]:
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

        arms = [
            "paper1_global_cmi_global_s2",
            "ablation_cz_disc_global_s2",
            "paper2_cz_disc_fluid_s2",
        ]
        ext_metrics: dict[str, list[dict[str, float]]] = {a: [] for a in arms}
        p_base_sum = np.zeros(len(df_c2), dtype=float)
        p_p1_sum = np.zeros(len(df_c2), dtype=float)
        q_p1_sum = np.zeros(len(df_c2), dtype=float)
        p_p2_sum = np.zeros(len(df_c2), dtype=float)
        q_p2_sum = np.zeros(len(df_c2), dtype=float)

        for s in range(n_splits):
            t_s = time.time()
            tr, ca, _ = make_patient_grouped_split(
                pids_c1, y_c1, seed=5100 + s, target_train=n_tr_cap, target_cal=n_ca_cap
            )
            oof_p = compute_oof_stage1(df_c1.iloc[tr][fair_base_cols], y_c1[tr], n_folds=4, seed=5200 + s)
            d_glob = run_sequential_discovery(
                df_c1.iloc[tr],
                y_c1[tr],
                fair_base_cols,
                channels,
                oof_p,
                contract,
                method="global_cmi",
                pilot_fraction=0.15,
                budget=3.0,
                seed=5300 + s,
            )
            d_cz = run_sequential_discovery(
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
                seed=5300 + s,
            )
            ch_cols_map = {c.channel_id: list(c.columns) for c in channels}
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
            for arm_name, scol, s2_mode in [
                ("paper1_global_cmi_global_s2", cols_glob, "global"),
                ("ablation_cz_disc_global_s2", cols_cz, "global"),
                ("paper2_cz_disc_fluid_s2", cols_cz, "fluid_backbone"),
            ]:
                traces: list[dict[str, np.ndarray]] = []
                ev_ext = evaluate_stage2_multi_test(
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
                    seed=5400 + s,
                )[0]
                ext_metrics[arm_name].append(ev_ext)
                if arm_name == "paper1_global_cmi_global_s2":
                    p_base_sum += traces[0]["p1_te"]
                    p_p1_sum += traces[0]["p_auto_te"]
                    q_p1_sum += traces[0]["auto_te"].astype(float)
                elif arm_name == "paper2_cz_disc_fluid_s2":
                    p_p2_sum += traces[0]["p_auto_te"]
                    q_p2_sum += traces[0]["auto_te"].astype(float)

            p1_nb = ext_metrics["paper1_global_cmi_global_s2"][-1]["nb_voi_auto"]
            p2_nb = ext_metrics["paper2_cz_disc_fluid_s2"][-1]["nb_voi_auto"]
            print(
                f"  [{tag} {s+1}/{n_splits}] ({time.time()-t_s:.1f}s, ext_N={len(df_c2):,}) "
                f"Glob={d_glob.selected_ids} | CZ={d_cz.selected_ids} | Ext NB: P1={p1_nb:.2f} -> P2={p2_nb:.2f}"
            )

        p_base_ens = p_base_sum / float(n_splits)
        p_p1_ens = p_p1_sum / float(n_splits)
        q_p1_ens = (q_p1_sum / float(n_splits)) >= 0.5
        p_p2_ens = p_p2_sum / float(n_splits)
        q_p2_ens = (q_p2_sum / float(n_splits)) >= 0.5
        boot_vs_base = paired_bootstrap_nb_diff(
            y_c2, p_p2_ens, q_p2_ens, p_base_ens, np.zeros_like(q_p2_ens), contract, n_boot=1000, seed=61
        )
        boot_vs_p1 = paired_bootstrap_nb_diff(
            y_c2, p_p2_ens, q_p2_ens, p_p1_ens, q_p1_ens, contract, n_boot=1000, seed=62
        )
        out_subcohorts[tag] = {
            "train_campus": campus_train,
            "external_campus": campus_ext,
            "train_campus_visits": len(df_c1),
            "external_campus_visits": len(df_c2),
            "train_prev": float(np.mean(y_c1)),
            "external_prev": float(np.mean(y_c2)),
            "n_splits": n_splits,
            "external_arms": {
                a: {k: _agg([r[k] for r in rows]) for k in rows[0]} for a, rows in ext_metrics.items()
            },
            "paired_bootstrap_p2_vs_base": boot_vs_base,
            "paired_bootstrap_p2_vs_p1": boot_vs_p1,
        }

    return out_subcohorts


def main() -> None:
    RES_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((PROC_DIR / "ws4tb_cohorts_manifest.json").read_text())

    ensure_sepsis_gap_cohorts()
    meta_yale = ensure_yale_trackb_meta()

    out_file = RES_DIR / "paper2_track_b_hardened_summary.json"
    track_b_summary: dict[str, object] = {}

    def _ckpt() -> None:
        out_file.write_text(json.dumps(track_b_summary, indent=2))

    # B1a: Sepsis 6h Full-Cohort Agniel Noise-Deconfounded Stress Test (92%-unmeasured assays)
    df_a6 = pd.read_parquet(PROC_DIR / "sepsis_setA_bidmc_6h.parquet")
    df_b6 = pd.read_parquet(PROC_DIR / "sepsis_setB_emory_6h.parquet")
    sepsis_base_cols: list[str] = list(manifest["sepsis_6h"]["base_columns"])
    track_b_summary["b1a_sepsis_6h_agniel_sparse_stress"] = run_sepsis_hardened_transfer(
        df_a6,
        df_b6,
        sepsis_base_cols,
        label_tag="Sepsis B1a: 6h Agniel-Blocked Sparse",
        deconfound_stage2=True,
        n_splits=8,
    )
    _ckpt()

    # B1b: Sepsis 6h Measured-Bloodwork Subcohort (patients with admission CBC+BMP drawn, zero _drawn flags, CMS CPT costs)
    df_a6_meas = df_a6[(df_a6["s2_Creatinine_drawn"] == 1) & (df_a6["s2_WBC_drawn"] == 1)].reset_index(drop=True)
    df_b6_meas = df_b6[(df_b6["s2_Creatinine_drawn"] == 1) & (df_b6["s2_WBC_drawn"] == 1)].reset_index(drop=True)
    track_b_summary["b1b_sepsis_6h_measured_bloodwork_cpt"] = run_sepsis_hardened_transfer(
        df_a6_meas,
        df_b6_meas,
        sepsis_base_cols,
        label_tag="Sepsis B1b: 6h Measured-Assay CPT",
        deconfound_stage2=False,
        n_splits=10,
    )
    _ckpt()

    # B2: Sepsis Strict Lead-Time Gap (1-4h obs, sepsis-free through 6h -> onset >= 7h, value-only CPT panels)
    df_agap = pd.read_parquet(PROC_DIR / "sepsis_setA_bidmc_4h_gap2h.parquet")
    df_bgap = pd.read_parquet(PROC_DIR / "sepsis_setB_emory_4h_gap2h.parquet")
    track_b_summary["b2_sepsis_4h_gap2h_strict_lead"] = run_sepsis_hardened_transfer(
        df_agap,
        df_bgap,
        sepsis_base_cols,
        label_tag="Sepsis B2: 4h+2h Gap Value-Only CPT",
        deconfound_stage2=False,
        n_splits=8,
    )
    _ckpt()

    # B3: Yale ED ESI 4-5 Fairness-Blinded + Subgroup Fairness Audit
    track_b_summary["b3_yale_esi45_fairness_patient_grouped"] = run_yale_patient_grouped_fairness_esi45(
        manifest, meta_yale, n_splits=6
    )
    _ckpt()

    # B4: Yale Cross-Campus Zero-Shot External ED Transfer (Campus A -> Campus B)
    track_b_summary["b4_yale_cross_campus_external_transfer"] = run_yale_cross_campus_external_transfer(
        manifest, meta_yale
    )
    _ckpt()
    print("Saved complete Track B Hardened summary:", out_file)

    subprocess.run([sys.executable, str(Path(__file__).with_name("plot_paper2_dual_track.py"))], check=True)


if __name__ == "__main__":
    main()
