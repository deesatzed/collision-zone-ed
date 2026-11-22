"""Run Paper 2 empirical campaign with KL-Gated Fluid Collision Backbone on WS4TB acute-care cohorts.

Evaluates:
  1. Synthetic Mechanism Audit (30 Monte Carlo replicates):
     True-Negative Abundance Dilution under Global CMI vs. Collision-Zone Discovery + Fluid Backbone,
     plus family-wise pure-noise null error rate (FWER).
  2. Cohort 1 & 2 — Beth Israel Acute Sepsis (N=19,972, 6h Landmark) -> Emory External Transfer (N=19,696, R=10 splits):
     Compares Paper 1 (Global CMI + Global S2), Global CMI + Fluid S2, Collision Discovery + Global S2,
     Collision Discovery + Naive Downweighted S2 (leaf-starvation ablation), and
     Paper 2 Flagship (Collision-Zone Discovery + Fluid KL-Gated Backbone S2), plus Null/Twin Stress Pool.
  3. Cohort 3 — Yale New Haven ED Low-Acuity Occult Illness Triage (ESI 4-5, all 152,995 visits:
     60k train / 20k cal / 72,995 held-out test per split, R=8 splits).
  4. Cohort 4 — Yale New Haven ED Full All-Comers (all 560,486 adult ED visits:
     75k train / 25k cal / 460,486 held-out test per split, R=6 splits).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from collision_zone.three_predicates import (
    ChannelDef,
    ClinicalContract,
    compute_oof_stage1,
    evaluate_stage2_and_routing,
    evaluate_stage2_multi_test,
    fit_stage1_predictions,
    run_sequential_discovery,
)

PROC_DIR = Path("/Volumes/WS4TB/imboreceipt_data/processed")
RES_DIR = Path("/Users/o2satz/antigravity/collision_zone_ed/results")


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


def make_capped_dev_full_test_split(
    y: np.ndarray,
    seed: int,
    n_train: int,
    n_cal: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified split drawing n_train + n_cal development visits and testing on ALL remaining visits."""
    n_tot = int(y.size)
    if n_train + n_cal >= int(0.85 * n_tot):
        return make_stratified_3way_split(y, seed=seed, train_frac=0.60, cal_frac=0.20)
    f_tr = float(n_train) / float(n_tot)
    f_ca = float(n_cal) / float(n_tot)
    return make_stratified_3way_split(y, seed=seed, train_frac=f_tr, cal_frac=f_ca)


def run_synthetic_dilution_simulation(n_reps: int = 30) -> dict[str, object]:
    """Monte Carlo verification of Predicates 1-3 under 94% True-Negative Abundance."""
    print(f"[Sim 1] Running {n_reps} replicates of True-Negative Dilution & High-Signal FP benchmark...")
    contract = ClinicalContract(
        cost_fp=1.0, cost_fn=9.0, target_precision=0.45, per_instance_query_cost=0.02
    )

    global_picked_cz: list[float] = []
    cz_picked_cz: list[float] = []
    cz_null_fwer: list[float] = []
    nb_base_list: list[float] = []
    nb_paper1_list: list[float] = []
    nb_paper2_fluid_list: list[float] = []
    hsfp_srs_list: list[float] = []
    hsfp_gney_list: list[float] = []
    hsfp_czney_list: list[float] = []

    for r in range(n_reps):
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
        oof_p = compute_oof_stage1(df.iloc[tr][base_cols], y[tr], n_folds=4, seed=3000 + r)

        d_srs = run_sequential_discovery(
            df.iloc[tr],
            y[tr],
            base_cols,
            channels,
            oof_p,
            contract,
            method="srs_global_cmi",
            pilot_fraction=0.20,
            budget=1.0,
            seed=4000 + r,
        )
        d_glob = run_sequential_discovery(
            df.iloc[tr],
            y[tr],
            base_cols,
            channels,
            oof_p,
            contract,
            method="global_cmi",
            pilot_fraction=0.20,
            budget=1.0,
            seed=4000 + r,
        )
        d_cz = run_sequential_discovery(
            df.iloc[tr],
            y[tr],
            base_cols,
            channels,
            oof_p,
            contract,
            method="collision_zone",
            pilot_fraction=0.20,
            budget=1.0,
            seed=4000 + r,
        )
        d_cz_b5 = run_sequential_discovery(
            df.iloc[tr],
            y[tr],
            base_cols,
            channels,
            oof_p,
            contract,
            method="collision_zone",
            pilot_fraction=0.20,
            budget=5.0,
            seed=4000 + r,
        )
        any_null = any(cid.startswith("null_") for cid in d_cz_b5.selected_ids)
        cz_null_fwer.append(1.0 if any_null else 0.0)
        global_picked_cz.append(1.0 if "ch_cz" in d_glob.selected_ids else 0.0)
        cz_picked_cz.append(1.0 if "ch_cz" in d_cz.selected_ids else 0.0)

        hsfp_srs_list.append(d_srs.pilot_high_signal_fp_share)
        hsfp_gney_list.append(d_glob.pilot_high_signal_fp_share)
        hsfp_czney_list.append(d_cz.pilot_high_signal_fp_share)

        ev_p1 = evaluate_stage2_and_routing(
            df.iloc[tr],
            y[tr],
            df.iloc[ca],
            y[ca],
            df.iloc[te],
            y[te],
            base_cols,
            d_glob.selected_ids,
            oof_p,
            contract,
            stage2_mode="global",
            seed=5000 + r,
        )
        ev_p2 = evaluate_stage2_and_routing(
            df.iloc[tr],
            y[tr],
            df.iloc[ca],
            y[ca],
            df.iloc[te],
            y[te],
            base_cols,
            d_cz.selected_ids,
            oof_p,
            contract,
            stage2_mode="fluid_backbone",
            seed=5000 + r,
        )
        nb_base_list.append(ev_p1["nb_base_0pct"])
        nb_paper1_list.append(ev_p1["nb_voi_auto"])
        nb_paper2_fluid_list.append(ev_p2["nb_voi_auto"])

    return {
        "n_reps": n_reps,
        "global_cmi_true_tiebreaker_rate": _agg(global_picked_cz),
        "collision_zone_true_tiebreaker_rate": _agg(cz_picked_cz),
        "collision_zone_null_fwer": _agg(cz_null_fwer),
        "pilot_hsfp_share_srs": _agg(hsfp_srs_list),
        "pilot_hsfp_share_global_neyman": _agg(hsfp_gney_list),
        "pilot_hsfp_share_collision_neyman": _agg(hsfp_czney_list),
        "nb_stage1_base": _agg(nb_base_list),
        "nb_paper1_global_cmi_auto_voi": _agg(nb_paper1_list),
        "nb_paper2_collision_zone_auto_voi": _agg(nb_paper2_fluid_list),
    }


def _add_stress_channels(
    df: pd.DataFrame,
    native_channels: list[ChannelDef],
    seed: int,
) -> tuple[pd.DataFrame, list[ChannelDef]]:
    rng = np.random.default_rng(seed)
    df_aug = df.copy()
    aug_channels = list(native_channels)
    for j in range(3):
        cname = f"stress_pure_null_{j}"
        df_aug[cname] = rng.normal(0.0, 1.0, size=len(df)).astype(np.float32)
        aug_channels.append(ChannelDef(cname, (cname,), cost=0.8, is_null_stress=True))
    for parent in native_channels[:2]:
        pcol = parent.columns[0]
        arr = df[pcol].to_numpy(dtype=float, copy=True)
        sd = float(np.nanstd(arr))
        sd = 1.0 if (not np.isfinite(sd) or sd <= 1e-8) else sd
        tname = f"stress_twin_of_{parent.channel_id}"
        df_aug[tname] = (arr + rng.normal(0.0, 0.35 * sd, size=len(df))).astype(np.float32)
        aug_channels.append(ChannelDef(tname, (tname,), cost=parent.cost, is_twin_stress=True))
    return df_aug, aug_channels


def run_sepsis_campaign(manifest: dict[str, object], n_splits: int = 10) -> dict[str, object]:
    """Evaluate Beth Israel 6h sepsis discovery + external Emory hospital validation."""
    print(f"[Sepsis] Running {n_splits} splits on Beth Israel (N=19,972) -> Emory (N=19,696)...")
    meta = manifest["sepsis_6h"]  # type: ignore[index]
    df_a = pd.read_parquet(PROC_DIR / "sepsis_setA_bidmc_6h.parquet")
    df_b = pd.read_parquet(PROC_DIR / "sepsis_setB_emory_6h.parquet")
    base_cols: list[str] = list(meta["base_columns"])  # type: ignore[index]
    channels = [
        ChannelDef(cid, tuple(cinfo["columns"]), cost=float(cinfo["cost"]))
        for cid, cinfo in meta["channels"].items()  # type: ignore[index]
    ]
    contract = ClinicalContract(
        cost_fp=1.0, cost_fn=15.0, target_precision=0.30, per_instance_query_cost=0.03
    )
    budget = 3.2
    pilot_frac = 0.20

    arms = [
        "paper1_global_cmi_global_s2",
        "ablation_global_cmi_fluid_s2",
        "ablation_cz_disc_global_s2",
        "ablation_cz_disc_naive_s2",
        "paper2_cz_disc_fluid_s2",
    ]
    internal_metrics: dict[str, list[dict[str, float]]] = {a: [] for a in arms}
    external_emory_metrics: dict[str, list[dict[str, float]]] = {a: [] for a in arms}
    selected_freq_global: dict[str, int] = {}
    selected_freq_cz: dict[str, int] = {}
    stress_null_twin_cz = 0
    hsfp_shares: dict[str, list[float]] = {"global_neyman": [], "collision_neyman": []}

    y_a = df_a["y_sepsis_future"].to_numpy(dtype=int)
    y_b = df_b["y_sepsis_future"].to_numpy(dtype=int)

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
            seed=900 + s,
        )
        hsfp_shares["global_neyman"].append(d_glob.pilot_high_signal_fp_share)
        hsfp_shares["collision_neyman"].append(d_cz.pilot_high_signal_fp_share)

        for cid in d_glob.selected_ids:
            selected_freq_global[cid] = selected_freq_global.get(cid, 0) + 1
        for cid in d_cz.selected_ids:
            selected_freq_cz[cid] = selected_freq_cz.get(cid, 0) + 1

        df_tr_aug, aug_ch = _add_stress_channels(
            df_a.iloc[tr].reset_index(drop=True), channels, seed=1200 + s
        )
        d_stress = run_sequential_discovery(
            df_tr_aug,
            y_a[tr],
            base_cols,
            aug_ch,
            oof_p,
            contract,
            method="collision_zone",
            pilot_fraction=pilot_frac,
            budget=budget,
            seed=1300 + s,
        )
        ch_map_aug = {c.channel_id: c for c in aug_ch}
        parent_selected = set(d_stress.selected_ids)
        for cid in d_stress.selected_ids:
            cobj = ch_map_aug[cid]
            if cobj.is_null_stress:
                stress_null_twin_cz += 1
            elif cobj.is_twin_stress:
                orig_id = cid.replace("stress_twin_of_", "")
                if orig_id in parent_selected:
                    stress_null_twin_cz += 1

        ch_cols_map = {c.channel_id: list(c.columns) for c in channels}
        cols_glob = [col for cid in d_glob.selected_ids for col in ch_cols_map[cid]]
        cols_cz = [col for cid in d_cz.selected_ids for col in ch_cols_map[cid]]

        # Precompute Stage-1 once for both BIDMC test and Emory external transfer
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

        configs = [
            ("paper1_global_cmi_global_s2", cols_glob, "global"),
            ("ablation_global_cmi_fluid_s2", cols_glob, "fluid_backbone"),
            ("ablation_cz_disc_global_s2", cols_cz, "global"),
            ("ablation_cz_disc_naive_s2", cols_cz, "naive_downweighted"),
            ("paper2_cz_disc_fluid_s2", cols_cz, "fluid_backbone"),
        ]
        for arm_name, scol, s2_mode in configs:
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
                precomputed_stage1=s1_cache,
                seed=1500 + s,
            )
            internal_metrics[arm_name].append(ev_int)
            external_emory_metrics[arm_name].append(ev_ext)

        p1_nb = external_emory_metrics["paper1_global_cmi_global_s2"][-1]["nb_voi_auto"]
        naive_nb = external_emory_metrics["ablation_cz_disc_naive_s2"][-1]["nb_voi_auto"]
        p2_nb = external_emory_metrics["paper2_cz_disc_fluid_s2"][-1]["nb_voi_auto"]
        bcz = external_emory_metrics["paper2_cz_disc_fluid_s2"][-1]["learned_beta_cz"]
        print(
            f"  [Sepsis {s+1}/{n_splits}] ({time.time()-t_s:.1f}s) "
            f"Glob={d_glob.selected_ids} | CZ={d_cz.selected_ids} | "
            f"Emory NB: P1={p1_nb:.2f} | Naive={naive_nb:.2f} | P2_Fluid={p2_nb:.2f} (beta_cz={bcz:.2f})"
        )

    def _summarize_arm_dict(mdict: dict[str, list[dict[str, float]]]) -> dict[str, object]:
        return {
            arm_name: {k: _agg([r[k] for r in rows]) for k in rows[0]}
            for arm_name, rows in mdict.items()
        }

    return {
        "n_splits": n_splits,
        "contract_tau_star": contract.tau_star,
        "budget": budget,
        "pilot_fraction": pilot_frac,
        "pilot_hsfp_shares": {k: _agg(v) for k, v in hsfp_shares.items()},
        "selected_channel_counts_global_cmi": selected_freq_global,
        "selected_channel_counts_collision_zone": selected_freq_cz,
        "stress_spurious_null_or_dup_twin_count_cz": stress_null_twin_cz,
        "internal_bidmc_test": _summarize_arm_dict(internal_metrics),
        "external_emory_transfer": _summarize_arm_dict(external_emory_metrics),
    }


def run_yale_lowacuity_campaign(manifest: dict[str, object], n_splits: int = 8) -> dict[str, object]:
    """Evaluate ALL 152,995 Yale ED Low-Acuity (ESI 4-5) visits (60k tr / 20k cal / 72,995 test)."""
    meta = manifest["yale_ed"]  # type: ignore[index]
    df_low = pd.read_parquet(PROC_DIR / "yale_ed_esi45_lowacuity.parquet")
    print(f"[Yale ESI 4-5] Running {n_splits} splits across all {len(df_low):,} low-acuity visits...")

    base_cols: list[str] = [c for c in meta["base_columns"] if c in df_low.columns]  # type: ignore[index]
    channels = [
        ChannelDef(cid, tuple(cinfo["columns"]), cost=float(cinfo["cost"]))
        for cid, cinfo in meta["channels"].items()  # type: ignore[index]
    ]
    y = df_low["y_admit"].to_numpy(dtype=int)
    contract = ClinicalContract(
        cost_fp=1.0, cost_fn=24.0, target_precision=0.25, per_instance_query_cost=0.025
    )
    budget = 3.0
    pilot_frac = 0.15

    arms = [
        "paper1_global_cmi_global_s2",
        "ablation_global_cmi_fluid_s2",
        "ablation_cz_disc_global_s2",
        "ablation_cz_disc_naive_s2",
        "paper2_cz_disc_fluid_s2",
    ]
    metrics: dict[str, list[dict[str, float]]] = {a: [] for a in arms}
    selected_freq_global: dict[str, int] = {}
    selected_freq_cz: dict[str, int] = {}
    subgroup_records: dict[str, list[dict[str, float]]] = {}

    for s in range(n_splits):
        t_s = time.time()
        tr, ca, te = make_capped_dev_full_test_split(y, seed=2100 + s, n_train=60000, n_cal=20000)
        oof_p = compute_oof_stage1(df_low.iloc[tr][base_cols], y[tr], n_folds=4, seed=2200 + s)

        d_glob = run_sequential_discovery(
            df_low.iloc[tr],
            y[tr],
            base_cols,
            channels,
            oof_p,
            contract,
            method="global_cmi",
            pilot_fraction=pilot_frac,
            budget=budget,
            seed=2300 + s,
        )
        d_cz = run_sequential_discovery(
            df_low.iloc[tr],
            y[tr],
            base_cols,
            channels,
            oof_p,
            contract,
            method="collision_zone",
            pilot_fraction=pilot_frac,
            budget=budget,
            seed=2300 + s,
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
            base_cols,
            seed=2400 + s,
        )

        for arm_name, scol, s2_mode in [
            ("paper1_global_cmi_global_s2", cols_glob, "global"),
            ("ablation_global_cmi_fluid_s2", cols_glob, "fluid_backbone"),
            ("ablation_cz_disc_global_s2", cols_cz, "global"),
            ("ablation_cz_disc_naive_s2", cols_cz, "naive_downweighted"),
            ("paper2_cz_disc_fluid_s2", cols_cz, "fluid_backbone"),
        ]:
            ev = evaluate_stage2_multi_test(
                df_low.iloc[tr],
                y[tr],
                df_low.iloc[ca],
                y[ca],
                [(df_low.iloc[te], y[te])],
                base_cols,
                scol,
                oof_p,
                contract,
                stage2_mode=s2_mode,
                precomputed_stage1=s1_cache,
                seed=2400 + s,
            )[0]
            metrics[arm_name].append(ev)

        df_te = df_low.iloc[te]
        y_te = y[te]
        for grp_col in ("race", "lang"):
            if grp_col in df_te.columns:
                for gval, gdf in df_te.groupby(grp_col, observed=False):
                    if len(gdf) >= 250 and int(y_te[df_te[grp_col] == gval].sum()) >= 10:
                        key = f"{grp_col}:{gval}"
                        subgroup_records.setdefault(key, []).append(
                            {
                                "n": float(len(gdf)),
                                "prev": float(np.mean(y_te[df_te[grp_col] == gval])),
                            }
                        )

        p1_nb = metrics["paper1_global_cmi_global_s2"][-1]["nb_voi_auto"]
        naive_nb = metrics["ablation_cz_disc_naive_s2"][-1]["nb_voi_auto"]
        p2_nb = metrics["paper2_cz_disc_fluid_s2"][-1]["nb_voi_auto"]
        bcz = metrics["paper2_cz_disc_fluid_s2"][-1]["learned_beta_cz"]
        print(
            f"  [Yale ESI 4-5 {s+1}/{n_splits}] ({time.time()-t_s:.1f}s, test={len(te):,}) "
            f"Glob={d_glob.selected_ids} | CZ={d_cz.selected_ids} | "
            f"Auto NB: P1={p1_nb:.2f} | Naive={naive_nb:.2f} | P2_Fluid={p2_nb:.2f} (beta_cz={bcz:.2f})"
        )

    return {
        "total_cohort_rows": len(df_low),
        "heldout_test_rows_per_split": int(len(df_low) - 80000),
        "prevalence": float(np.mean(y)),
        "n_splits": n_splits,
        "contract_tau_star": contract.tau_star,
        "budget": budget,
        "pilot_fraction": pilot_frac,
        "selected_channel_counts_global_cmi": selected_freq_global,
        "selected_channel_counts_collision_zone": selected_freq_cz,
        "arms": {
            arm_name: {k: _agg([r[k] for r in rows]) for k in rows[0]}
            for arm_name, rows in metrics.items()
        },
        "subgroups_summary": {
            k: {
                "mean_test_n": float(np.mean([x["n"] for x in v])),
                "prev": float(np.mean([x["prev"] for x in v])),
            }
            for k, v in subgroup_records.items()
        },
    }


def run_yale_full_allcomers_campaign(manifest: dict[str, object], n_splits: int = 6) -> dict[str, object]:
    """Evaluate ALL 560,486 Yale ED adult visits (75k tr / 25k cal / 460,486 held-out test per split)."""
    meta = manifest["yale_ed"]  # type: ignore[index]
    df_full = pd.read_parquet(PROC_DIR / "yale_ed_full.parquet")
    print(f"[Yale Full 560k] Running {n_splits} splits across all {len(df_full):,} adult ED visits...")

    base_cols: list[str] = [c for c in meta["base_columns"] if c in df_full.columns]  # type: ignore[index]
    channels = [
        ChannelDef(cid, tuple(cinfo["columns"]), cost=float(cinfo["cost"]))
        for cid, cinfo in meta["channels"].items()  # type: ignore[index]
    ]
    y = df_full["y_admit"].to_numpy(dtype=int)
    # Inpatient disposition contract across all ESI 1-5 arrivals: tau* = 1 / (1 + 5.67) = 0.15
    contract = ClinicalContract(
        cost_fp=1.0, cost_fn=5.6667, target_precision=0.65, per_instance_query_cost=0.03
    )
    budget = 3.0
    pilot_frac = 0.12

    arms = [
        "paper1_global_cmi_global_s2",
        "ablation_cz_disc_global_s2",
        "ablation_cz_disc_naive_s2",
        "paper2_cz_disc_fluid_s2",
    ]
    metrics: dict[str, list[dict[str, float]]] = {a: [] for a in arms}
    selected_freq_global: dict[str, int] = {}
    selected_freq_cz: dict[str, int] = {}

    for s in range(n_splits):
        t_s = time.time()
        tr, ca, te = make_capped_dev_full_test_split(y, seed=3100 + s, n_train=75000, n_cal=25000)
        oof_p = compute_oof_stage1(df_full.iloc[tr][base_cols], y[tr], n_folds=4, seed=3200 + s)

        d_glob = run_sequential_discovery(
            df_full.iloc[tr],
            y[tr],
            base_cols,
            channels,
            oof_p,
            contract,
            method="global_cmi",
            pilot_fraction=pilot_frac,
            budget=budget,
            seed=3300 + s,
        )
        d_cz = run_sequential_discovery(
            df_full.iloc[tr],
            y[tr],
            base_cols,
            channels,
            oof_p,
            contract,
            method="collision_zone",
            pilot_fraction=pilot_frac,
            budget=budget,
            seed=3300 + s,
        )
        for cid in d_glob.selected_ids:
            selected_freq_global[cid] = selected_freq_global.get(cid, 0) + 1
        for cid in d_cz.selected_ids:
            selected_freq_cz[cid] = selected_freq_cz.get(cid, 0) + 1

        ch_cols_map = {c.channel_id: list(c.columns) for c in channels}
        cols_glob = [col for cid in d_glob.selected_ids for col in ch_cols_map[cid]]
        cols_cz = [col for cid in d_cz.selected_ids for col in ch_cols_map[cid]]

        s1_cache = fit_stage1_predictions(
            df_full.iloc[tr],
            y[tr],
            df_full.iloc[ca],
            y[ca],
            [df_full.iloc[te]],
            base_cols,
            seed=3400 + s,
        )

        for arm_name, scol, s2_mode in [
            ("paper1_global_cmi_global_s2", cols_glob, "global"),
            ("ablation_cz_disc_global_s2", cols_cz, "global"),
            ("ablation_cz_disc_naive_s2", cols_cz, "naive_downweighted"),
            ("paper2_cz_disc_fluid_s2", cols_cz, "fluid_backbone"),
        ]:
            ev = evaluate_stage2_multi_test(
                df_full.iloc[tr],
                y[tr],
                df_full.iloc[ca],
                y[ca],
                [(df_full.iloc[te], y[te])],
                base_cols,
                scol,
                oof_p,
                contract,
                stage2_mode=s2_mode,
                precomputed_stage1=s1_cache,
                seed=3400 + s,
            )[0]
            metrics[arm_name].append(ev)

        p1_nb = metrics["paper1_global_cmi_global_s2"][-1]["nb_voi_auto"]
        naive_nb = metrics["ablation_cz_disc_naive_s2"][-1]["nb_voi_auto"]
        p2_nb = metrics["paper2_cz_disc_fluid_s2"][-1]["nb_voi_auto"]
        bcz = metrics["paper2_cz_disc_fluid_s2"][-1]["learned_beta_cz"]
        print(
            f"  [Yale Full {s+1}/{n_splits}] ({time.time()-t_s:.1f}s, test={len(te):,}) "
            f"Glob={d_glob.selected_ids} | CZ={d_cz.selected_ids} | "
            f"Auto NB: P1={p1_nb:.2f} | Naive={naive_nb:.2f} | P2_Fluid={p2_nb:.2f} (beta_cz={bcz:.2f})"
        )

    return {
        "total_cohort_rows": len(df_full),
        "heldout_test_rows_per_split": int(len(df_full) - 100000),
        "prevalence": float(np.mean(y)),
        "n_splits": n_splits,
        "contract_tau_star": contract.tau_star,
        "budget": budget,
        "pilot_fraction": pilot_frac,
        "selected_channel_counts_global_cmi": selected_freq_global,
        "selected_channel_counts_collision_zone": selected_freq_cz,
        "arms": {
            arm_name: {k: _agg([r[k] for r in rows]) for k in rows[0]}
            for arm_name, rows in metrics.items()
        },
    }


def main() -> None:
    RES_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((PROC_DIR / "ws4tb_cohorts_manifest.json").read_text())

    sim_res = run_synthetic_dilution_simulation(n_reps=30)
    (RES_DIR / "sim1_true_negative_dilution.json").write_text(json.dumps(sim_res, indent=2))

    sepsis_res = run_sepsis_campaign(manifest, n_splits=10)
    (RES_DIR / "sepsis_bidmc_emory_results.json").write_text(json.dumps(sepsis_res, indent=2))

    yale_low_res = run_yale_lowacuity_campaign(manifest, n_splits=8)
    (RES_DIR / "yale_esi45_results.json").write_text(json.dumps(yale_low_res, indent=2))

    yale_full_res = run_yale_full_allcomers_campaign(manifest, n_splits=6)
    (RES_DIR / "yale_full_560k_results.json").write_text(json.dumps(yale_full_res, indent=2))

    summary = {
        "synthetic_dilution_sim": sim_res,
        "sepsis_bidmc_to_emory": sepsis_res,
        "yale_ed_esi45_lowacuity": yale_low_res,
        "yale_ed_full_560k": yale_full_res,
    }
    out_file = RES_DIR / "paper2_initial_campaign_summary.json"
    out_file.write_text(json.dumps(summary, indent=2))
    art_dir = Path("/Users/o2satz/.gemini/antigravity/brain/5be51496-dafb-4e38-a543-75823c062544/scratch")
    if art_dir.exists():
        shutil.copy2(out_file, art_dir / "paper2_initial_campaign_summary.json")
    print("Saved complete Paper 2 campaign summary:", out_file)
    subprocess.run([sys.executable, str(Path(__file__).with_name("plot_paper2_figures.py"))], check=True)


if __name__ == "__main__":
    main()
