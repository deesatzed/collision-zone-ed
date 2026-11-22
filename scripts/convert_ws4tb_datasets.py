"""Unpack and convert WS4TB acute-care archives into leak-free Parquet cohorts.

Reads:
  1. /Volumes/WS4TB/imboreceipt_data/archive.zip (PhysioNet/CinC 2019: 40,336 patients,
     Beth Israel Set A [20,336] + Emory Set B [20,000]).
  2. /Volumes/WS4TB/imboreceipt_data/admissionprediction-v1.0.zip (Yale New Haven ED:
     560,486 adult ED visits x 972 features, Hong, Haimovich & Taylor, PLOS ONE 2018).

Writes clean Parquet files + JSON schema manifests into:
  /Volumes/WS4TB/imboreceipt_data/processed/
"""

from __future__ import annotations

import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("/Volumes/WS4TB/imboreceipt_data")
OUT_DIR = RAW_DIR / "processed"

VITALS_7 = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"]
SEPSIS_LAB_CHANNELS: dict[str, tuple[list[str], float]] = {
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


def convert_sepsis_archive(window_hours: int = 6) -> dict[str, object]:
    """Build strictly causal W-hour risk-set landmark cohorts for Set A (BIDMC) and Set B (Emory)."""
    t0 = time.time()
    zpath = RAW_DIR / "archive.zip"
    records_a: list[dict[str, float | int | str]] = []
    records_b: list[dict[str, float | int | str]] = []

    excluded_short = 0
    excluded_already_septic = 0

    with zipfile.ZipFile(zpath, "r") as zf:
        psv_names = sorted(n for n in zf.namelist() if n.endswith(".psv"))
        header = zf.read(psv_names[0]).decode("utf-8").splitlines()[0].split("|")
        col_idx = {c: idx for idx, c in enumerate(header)}

        vital_idxs = [col_idx[v] for v in VITALS_7]
        lab_cols_flat = [c for cols, _ in SEPSIS_LAB_CHANNELS.values() for c in cols]
        lab_idxs = {c: col_idx[c] for c in lab_cols_flat}

        idx_age = col_idx["Age"]
        idx_gender = col_idx["Gender"]
        idx_u1 = col_idx["Unit1"]
        idx_u2 = col_idx["Unit2"]
        idx_hadm = col_idx["HospAdmTime"]
        idx_sepsis = col_idx["SepsisLabel"]

        for p_i, name in enumerate(psv_names):
            lines = zf.read(name).decode("utf-8").splitlines()[1:]
            n_hours = len(lines)
            if n_hours < window_hours:
                excluded_short += 1
                continue

            mat = np.genfromtxt(lines, delimiter="|")
            if mat.ndim == 1:
                mat = mat.reshape(1, -1)

            # Outcome-independent risk-set check prior to landmark W:
            # Patient must still be sepsis-free strictly prior to hour W (rows 0..W-2)
            if window_hours > 1 and np.any(mat[: window_hours - 1, idx_sepsis] == 1.0):
                excluded_already_septic += 1
                continue

            y_future = int(np.any(mat[window_hours - 1 :, idx_sepsis] == 1.0))
            win = mat[:window_hours, :]

            pat_id = Path(name).stem
            site = "BIDMC_SetA" if "training_setA" in name else "Emory_SetB"
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

            # Zero-cost bedside monitor features across [1..W]
            for v_name, v_idx in zip(VITALS_7, vital_idxs):
                vals = win[:, v_idx]
                finite = vals[np.isfinite(vals)]
                if finite.size == 0:
                    row[f"base_{v_name}_first"] = np.nan
                    row[f"base_{v_name}_last"] = np.nan
                    row[f"base_{v_name}_min"] = np.nan
                    row[f"base_{v_name}_max"] = np.nan
                    row[f"base_{v_name}_mean"] = np.nan
                    row[f"base_{v_name}_delta"] = np.nan
                    row[f"base_{v_name}_nobs"] = 0
                else:
                    row[f"base_{v_name}_first"] = float(finite[0])
                    row[f"base_{v_name}_last"] = float(finite[-1])
                    row[f"base_{v_name}_min"] = float(np.min(finite))
                    row[f"base_{v_name}_max"] = float(np.max(finite))
                    row[f"base_{v_name}_mean"] = float(np.mean(finite))
                    row[f"base_{v_name}_delta"] = float(finite[-1] - finite[0])
                    row[f"base_{v_name}_nobs"] = int(finite.size)

            # Derived bedside physiology (Shock Index & bedside SIRS vitals count)
            hr_last, sbp_last = row["base_HR_last"], row["base_SBP_last"]
            hr_max, sbp_min = row["base_HR_max"], row["base_SBP_min"]
            dbp_last = row["base_DBP_last"]
            rr_max = row["base_Resp_max"]
            t_max, t_min = row["base_Temp_max"], row["base_Temp_min"]

            row["base_shock_index_last"] = (
                float(hr_last / sbp_last)
                if np.isfinite(hr_last) and np.isfinite(sbp_last) and sbp_last > 20
                else np.nan
            )
            row["base_shock_index_peak"] = (
                float(hr_max / sbp_min)
                if np.isfinite(hr_max) and np.isfinite(sbp_min) and sbp_min > 20
                else np.nan
            )
            row["base_pulse_pressure_last"] = (
                float(sbp_last - dbp_last)
                if np.isfinite(sbp_last) and np.isfinite(dbp_last)
                else np.nan
            )
            sirs_hr = 1 if (np.isfinite(hr_max) and hr_max > 90.0) else 0
            sirs_rr = 1 if (np.isfinite(rr_max) and rr_max > 20.0) else 0
            sirs_temp = (
                1
                if ((np.isfinite(t_max) and t_max > 38.0) or (np.isfinite(t_min) and t_min < 36.0))
                else 0
            )
            row["base_bedside_sirs_vitals_cnt"] = int(sirs_hr + sirs_rr + sirs_temp)

            # Stage-2 quantitative laboratory assays strictly within [1..W]
            for c_name, c_idx in lab_idxs.items():
                vals = win[:, c_idx]
                finite = vals[np.isfinite(vals)]
                if finite.size == 0:
                    row[f"s2_{c_name}_last"] = np.nan
                    row[f"s2_{c_name}_max"] = np.nan
                    row[f"s2_{c_name}_min"] = np.nan
                    row[f"s2_{c_name}_drawn"] = 0
                else:
                    row[f"s2_{c_name}_last"] = float(finite[-1])
                    row[f"s2_{c_name}_max"] = float(np.max(finite))
                    row[f"s2_{c_name}_min"] = float(np.min(finite))
                    row[f"s2_{c_name}_drawn"] = 1

            if site == "BIDMC_SetA":
                records_a.append(row)
            else:
                records_b.append(row)

            if (p_i + 1) % 10000 == 0:
                print(f"  [Sepsis W={window_hours}h] Parsed {p_i + 1}/{len(psv_names)} patients...")

    df_a = pd.DataFrame(records_a)
    df_b = pd.DataFrame(records_b)
    out_a = OUT_DIR / f"sepsis_setA_bidmc_{window_hours}h.parquet"
    out_b = OUT_DIR / f"sepsis_setB_emory_{window_hours}h.parquet"
    df_a.to_parquet(out_a, index=False)
    df_b.to_parquet(out_b, index=False)

    base_cols = [
        c for c in df_a.columns if c.startswith("base_") or c in {"Age", "Gender", "Unit1", "Unit2", "HospAdmTime"}
    ]
    channel_map = {
        ch_id: {
            "cost": cost,
            "columns": [
                f"s2_{lab}_{stat}"
                for lab in labs
                for stat in ("last", "max", "min", "drawn")
            ],
            "value_only_columns": [
                f"s2_{lab}_{stat}"
                for lab in labs
                for stat in ("last", "max", "min")
            ],
        }
        for ch_id, (labs, cost) in SEPSIS_LAB_CHANNELS.items()
    }
    meta = {
        "window_hours": window_hours,
        "excluded_short_stay": excluded_short,
        "excluded_already_septic_before_W": excluded_already_septic,
        "setA_bidmc_rows": len(df_a),
        "setA_bidmc_pos": int(df_a["y_sepsis_future"].sum()),
        "setA_bidmc_prev": float(df_a["y_sepsis_future"].mean()),
        "setB_emory_rows": len(df_b),
        "setB_emory_pos": int(df_b["y_sepsis_future"].sum()),
        "setB_emory_prev": float(df_b["y_sepsis_future"].mean()),
        "base_columns": base_cols,
        "channels": channel_map,
        "elapsed_sec": round(time.time() - t0, 2),
    }
    print(
        f"[Sepsis W={window_hours}h] Saved {out_a.name} (N={len(df_a)}, prev={meta['setA_bidmc_prev']:.4f}) "
        f"& {out_b.name} (N={len(df_b)}, prev={meta['setB_emory_prev']:.4f}) in {meta['elapsed_sec']}s"
    )
    return meta


def _classify_yale_stage2_channel(col: str) -> str:
    """Partition Yale ED's ~750 non-triage EHR/HIE columns into 18 clinical domain channels."""
    cl = col.lower()
    if col in {"n_edvisits", "n_admissions", "previousdispo"}:
        return "hie_prior_ed_admissions"
    if col == "n_surgeries":
        return "ehr_past_surgical_hx"
    if cl.startswith("vitals_") or ("vital" in cl and not cl.startswith("triage_vital")):
        return "hie_prior_vitals_trends"
    if any(k in cl for k in ("ekg", "echo")):
        return "hie_prior_ekg_echo"
    if any(k in cl for k in ("cxr", "headct", "otherct", "mri", "otherus", "otherxr", "otherimg")):
        return "hie_prior_radiology_ct_mri"
    if cl.startswith("meds_"):
        if any(k in cl for k in ("cardiac", "cardio", "antihyperten", "diuretic", "betablock", "statin", "antiangin", "antiarrhyth")):
            return "meds_cardiovascular"
        if any(k in cl for k in ("anticoag", "antiplatelet", "blood")):
            return "meds_anticoag_heme"
        if any(k in cl for k in ("diabet", "insulin", "hypoglyc", "thyroid", "hormon", "steroid", "endo")):
            return "meds_endocrine_diabetes"
        if any(k in cl for k in ("asthma", "broncho", "resp", "pulm", "cough", "allergy")):
            return "meds_respiratory"
        if any(k in cl for k in ("analges", "opioid", "nsaid", "anesthe", "psych", "antidep", "antipsych", "anticonv", "sedat", "cns", "neuro")):
            return "meds_neuro_psych_pain"
        return "meds_gi_infectious_other"
    if any(k in cl for k in ("ua_last", "ua_count", "ua_pos", "urine", "bloodculture", "culture")):
        return "labs_hx_ua_cultures"
    if any(sc in cl for sc in ("_min", "_max", "_median", "_last")):
        if any(k in cl for k in ("trop", "bnp", "pro-bnp", "ck", "cpk", "d-dimer", "ddimer")):
            return "labs_hx_cardiac_trop_bnp"
        if any(k in cl for k in ("lactate", "ph", "pco2", "po2", "hco3", "base", "anion")):
            return "labs_hx_lactate_acidbase"
        if any(k in cl for k in ("wbc", "hgb", "hct", "plt", "platelet", "mcv", "rdw", "neut", "lymph", "inr", "ptt", "pt_")):
            return "labs_hx_cbc_coags"
        if any(k in cl for k in ("creat", "bun", "egfr", "sodium", "potassium", "chloride", "calcium", "magnesium", "phosph", "glucose", "a1c")):
            return "labs_hx_bmp_renal_glucose"
        return "labs_hx_hepatic_other"
    # Remaining columns are AHRQ CCS Past Medical History binary flags
    if any(k in cl for k in ("heart", "card", "chf", "coron", "ami", "dysrhyth", "valve", "htn", "hypertens", "aneurysm", "periphasc", "pulm", "copd", "asthma", "resp", "pneum", "renal", "kidney", "ckd", "liver", "hepat")):
        return "pmh_cardiopulm_renal_liver"
    return "pmh_onc_neuro_endo_other"


def convert_yale_archive() -> dict[str, object]:
    """Extract and convert Yale 560,486 x 972 RData to fast Parquet + clinical subcohorts."""
    import pyreadr

    t0 = time.time()
    zpath = RAW_DIR / "admissionprediction-v1.0.zip"
    tmp_rdata = OUT_DIR / "_tmp_5v_cleandf.RData"
    member = "yaleemmlc-admissionprediction-5cf50bc/Results/5v_cleandf.RData"

    print("[Yale ED] Extracting 5v_cleandf.RData to temporary file on WS4TB...")
    with zipfile.ZipFile(zpath, "r") as zf, zf.open(member) as src_f, open(tmp_rdata, "wb") as out_f:
        while True:
            chunk = src_f.read(16 * 1024 * 1024)
            if not chunk:
                break
            out_f.write(chunk)

    print("[Yale ED] Reading RData via pyreadr...")
    res = pyreadr.read_r(str(tmp_rdata))
    obj_name = next(iter(res.keys()))
    df: pd.DataFrame = res[obj_name]
    print(f"[Yale ED] Loaded '{obj_name}': shape={df.shape}")
    tmp_rdata.unlink(missing_ok=True)

    # Apply exact deterministic eligibility from 20_cleanmerged.R (without inspecting features)
    if "disposition" in df.columns:
        df = df[df["disposition"].isin(["Admit", "Discharge"])].copy()
    if "age" in df.columns:
        df = df[(df["age"] >= 18) | (df["age"].isna())].copy()

    df["y_admit"] = (df["disposition"] == "Admit").astype(np.int8)

    # Identify exact column ranges matching 23_useonlytriage.R
    cols = list(df.columns)
    for drop_id in ("pat_mrn_id", "pat_enc_csn_id", "transfer_dep_name", "arrivaltime", "roomtime", "arrivalhour"):
        if drop_id in df.columns:
            df.drop(columns=[drop_id], inplace=True)

    cols = list(df.columns)
    triage_vital_start = cols.index("triage_vital_hr")
    triage_vital_end = cols.index("triage_vital_temp")
    cc_start = cols.index("cc_abdominalcramping")
    cc_end = cols.index("cc_wristpain")

    triage_demo_candidates = [
        "dep_name",
        "esi",
        "age",
        "gender",
        "ethnicity",
        "race",
        "lang",
        "religion",
        "maritalstatus",
        "employstatus",
        "insurance_status",
        "name",
        "arrivalmode",
        "arrivalmonth",
        "arrivalday",
        "arrivalhour_bin",
    ]
    base_demo = [c for c in triage_demo_candidates if c in df.columns]
    base_vitals = cols[triage_vital_start : triage_vital_end + 1]
    base_ccs = cols[cc_start : cc_end + 1]
    base_cols = list(dict.fromkeys(base_demo + base_vitals + base_ccs))

    excluded_meta = {"disposition", "y_admit"}
    stage2_cols = [c for c in df.columns if c not in set(base_cols) and c not in excluded_meta]

    # Downcast float64 sparse 0/1/count columns to float32 to cut Parquet size & speed I/O
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]) or isinstance(df[c].dtype, pd.CategoricalDtype):
            df[c] = df[c].astype(str).replace({"nan": "Missing", "None": "Missing", "<NA>": "Missing", "": "Missing"})
        elif pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].astype(np.float32)

    # Group Stage-2 columns into clinical domain channels
    ch_groups: dict[str, list[str]] = {}
    for c in stage2_cols:
        ch = _classify_yale_stage2_channel(c)
        ch_groups.setdefault(ch, []).append(c)

    ch_costs = {
        "hie_prior_ed_admissions": 0.9,
        "ehr_past_surgical_hx": 0.8,
        "hie_prior_vitals_trends": 1.0,
        "hie_prior_ekg_echo": 1.2,
        "hie_prior_radiology_ct_mri": 1.2,
        "meds_cardiovascular": 0.9,
        "meds_anticoag_heme": 0.9,
        "meds_endocrine_diabetes": 0.8,
        "meds_respiratory": 0.8,
        "meds_neuro_psych_pain": 0.8,
        "meds_gi_infectious_other": 0.8,
        "labs_hx_ua_cultures": 1.0,
        "labs_hx_cardiac_trop_bnp": 1.2,
        "labs_hx_lactate_acidbase": 1.1,
        "labs_hx_cbc_coags": 1.0,
        "labs_hx_bmp_renal_glucose": 1.0,
        "labs_hx_hepatic_other": 1.0,
        "pmh_cardiopulm_renal_liver": 1.0,
        "pmh_onc_neuro_endo_other": 1.0,
    }

    out_full = OUT_DIR / "yale_ed_full.parquet"
    df.to_parquet(out_full, index=False)

    # Also save the Low-Acuity Triage Cohort (ESI 4 & 5: classic rare occult-illness admission ~4-6%)
    esi_str = df["esi"].astype(str).str.strip()
    df_low = df[esi_str.isin(["4", "5", "4.0", "5.0"])].copy()
    out_low = OUT_DIR / "yale_ed_esi45_lowacuity.parquet"
    df_low.to_parquet(out_low, index=False)

    meta = {
        "full_rows": len(df),
        "full_admit_pos": int(df["y_admit"].sum()),
        "full_admit_prev": float(df["y_admit"].mean()),
        "esi_counts": esi_str.value_counts().to_dict(),
        "esi45_rows": len(df_low),
        "esi45_admit_pos": int(df_low["y_admit"].sum()),
        "esi45_admit_prev": float(df_low["y_admit"].mean()) if len(df_low) > 0 else 0.0,
        "n_base_cols": len(base_cols),
        "n_stage2_cols": len(stage2_cols),
        "n_channels": len(ch_groups),
        "base_columns": base_cols,
        "channels": {
            ch: {"cost": ch_costs.get(ch, 1.0), "n_cols": len(c_list), "columns": c_list}
            for ch, c_list in sorted(ch_groups.items())
        },
        "elapsed_sec": round(time.time() - t0, 2),
    }
    print(
        f"[Yale ED] Saved {out_full.name} (N={len(df)}, prev={meta['full_admit_prev']:.4f}) & "
        f"{out_low.name} (N={len(df_low)}, prev={meta['esi45_admit_prev']:.4f}, M={len(ch_groups)} ch) "
        f"in {meta['elapsed_sec']}s"
    )
    return meta


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sepsis_6h = convert_sepsis_archive(window_hours=6)
    sepsis_4h = convert_sepsis_archive(window_hours=4)
    yale_meta = convert_yale_archive()
    manifest = {
        "sepsis_6h": sepsis_6h,
        "sepsis_4h": sepsis_4h,
        "yale_ed": yale_meta,
    }
    manifest_path = OUT_DIR / "ws4tb_cohorts_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print("Wrote complete manifest:", manifest_path)


if __name__ == "__main__":
    main()
