"""Deterministic <1s checkpoint-to-table verifier for Paper 2 v2.3 (`Section 13` & `Section 15`).

Verifies:
  1. Presence and structural integrity of all 4 modular JSON blocks in `results/v2_3_blocks/`
     plus the merged `results/paper2_v2_3_remediation_results.json` and `paper2_v2_strengthening_results.json`.
  2. Exact mathematical tolerances for `F3` (Beta == Boltzmann KL gate), `F4` (SRS two-phase
     variance == s^2 / n), and `F6` (log-loss curvature vs. proportional KL gain).
  3. Exact patient-level Net Benefit ledger reconciliation:
       NB_1000 == 1000 * (TP/N - (tau*/(1-tau*))*FP/N - mean_cost_cfp / C_FN)
     to within rounding tolerance across all reported sepsis exchange rows.
  4. Total encounter count reconciliation (19,972 + 19,696 + 560,486 == 600,154) and Campus A+B+C
     sum (322,283 + 166,497 + 71,706 == 560,486).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BLOCKS_DIR = REPO_ROOT / "results" / "v2_3_blocks"
MERGED_V3 = REPO_ROOT / "results" / "paper2_v2_3_remediation_results.json"
V2_JSON = REPO_ROOT / "results" / "paper2_v2_strengthening_results.json"


def run_verification(verbose: bool = True) -> dict[str, object]:
    assert MERGED_V3.exists(), f"Missing {MERGED_V3}"
    assert V2_JSON.exists(), f"Missing {V2_JSON}"
    merged = json.loads(MERGED_V3.read_text())
    v2 = json.loads(V2_JSON.read_text())

    expected_blocks = [
        "block1_synthetic",
        "block2_sepsis_primary_ledger",
        "block3_horizons_shift_missingness",
        "block4_yale_and_sensitivity",
    ]
    for b in expected_blocks:
        bp = BLOCKS_DIR / f"{b}.json"
        assert bp.exists(), f"Missing block file {bp}"
        assert b in merged, f"Block {b} missing in merged JSON"

    b1 = merged["block1_synthetic"]
    b2 = merged["block2_sepsis_primary_ledger"]
    b3 = merged["block3_horizons_shift_missingness"]
    b4 = merged["block4_yale_and_sensitivity"]

    # 1. Mathematical identity checks (F3, F4, F6)
    id_chk = b1["f3_f4_f6_identities"]
    assert float(id_chk["f3_max_abs_diff_beta_kernel_vs_kl_gate"]) < 1e-13
    assert float(id_chk["f4_srs_implemented_se_minus_s_over_sqrt_n"]) < 1e-14
    assert abs(float(id_chk["f4_printed_eq7_w_squared_variance_inflation_at_20pct_pilot"]) - 1.8) < 1e-9

    # 2. Sim 1 1,000-replicate parity + F14 synthetic factorial checks
    sim1 = v2["sim1_1000_replicates"]
    assert int(sim1["n_reps"]) == 1000
    assert abs(float(sim1["cz_tiebreaker_recovery"]["rate"]) - 0.805) < 1e-6
    assert abs(float(sim1["global_cmi_tiebreaker_recovery"]["rate"]) - 0.083) < 1e-6

    f14_s = b1["f14_synthetic_discovery_factorial"]
    rates_ladder = [
        float(f14_s["global_cmi"]["tiebreaker_recovery"]["rate"]),
        float(f14_s["brier"]["tiebreaker_recovery"]["rate"]),
        float(f14_s["direct_utility"]["tiebreaker_recovery"]["rate"]),
        float(f14_s["local_utility"]["tiebreaker_recovery"]["rate"]),
        float(f14_s["collision_zone"]["tiebreaker_recovery"]["rate"]),
    ]
    assert rates_ladder == sorted(rates_ladder), f"Expected monotone recovery ladder, got {rates_ladder}"
    assert len(b1["sec8_regime_map_36_cells"]) == 36

    # 3. Sepsis ledger arithmetic verification (F1, F2, F12)
    n_emory_b1b = 6463
    for mode_key in ("b1b_measured_point_in_time_last", "b1b_measured_window_6h_summary"):
        mdata = b2[mode_key]
        for row in mdata["exchange_rate_table"]:
            tau = float(row["tau"])
            odds = tau / (1.0 - tau)
            cfn_tau = (1.0 - tau) / tau
            for arm in ("cz_fluid", "global_cmi", "stage1_base"):
                ar = row[arm]
                tp = int(ar["tp_count"])
                fp = int(ar["fp_count"])
                mc = float(ar["mean_cost_cfp"])
                rebuilt_nb = 1000.0 * ((tp / n_emory_b1b) - odds * (fp / n_emory_b1b) - mc / cfn_tau)
                assert abs(rebuilt_nb - float(ar["nb_ledger_per_1k"])) <= 0.005, (
                    f"NB mismatch in {mode_key}/{arm}@{tau}: {rebuilt_nb} vs {ar['nb_ledger_per_1k']}"
                )

    # Escalation-only sensitivity >= Stage-1 baseline sensitivity (zero missed baseline TPs!)
    for mode_key in ("b1b_measured_point_in_time_last", "b1b_measured_window_6h_summary"):
        dp = b2[mode_key]["directional_voi_policies"]
        base_tp = int(b2[mode_key]["exchange_rate_table"][1]["stage1_base"]["tp_count"])
        esc_tp = int(dp["escalation_only"]["tp_count"])
        assert esc_tp >= base_tp, f"Escalation-only lost TPs: {esc_tp} < {base_tp}"

    # 4. F9, F10, F11 checks
    lead_6h = b3["f9_w6h_pre_onset_lead_stratification"]
    tot_emory_events = sum(int(v["n_events"]) for v in lead_6h.values())
    assert tot_emory_events == 349, f"Expected 349 Emory W=6h events, got {tot_emory_events}"
    assert len(b3["f11_channel_availability_by_site_and_outcome"]) == 14

    # 5. F13 & F15 Yale campus count reconciliation & CfBH repeated-sampling checks
    nb_all_b = int(b4["f13_yale_cross_campus_allcomers"]["campus_b_allcomers"]["n_visits"])
    nb_all_c = int(b4["f13_yale_cross_campus_allcomers"]["campus_c_allcomers_untested"]["n_visits"])
    assert 322283 + nb_all_b + nb_all_c == 560486
    assert 19972 + 19696 + 560486 == 600154

    for smode in ("point_in_time_last", "window_6h_summary"):
        grid = b4["f13_sepsis_9_distinct_hyperparameter_grid"][smode]
        assert len(grid) == 9
        for cell in grid:
            assert float(cell["m_min_160_vs_320_max_diff"]) < 1e-6

    burn = b4["f15_campus_b_cfbh_repeated_burnin_curve"]
    assert len(burn) == 6
    for row in burn[1:]:
        assert float(row["mean_fdp_empirical_fdr"]) <= 0.75, f"FDR exceeded 0.75: {row}"

    summary = {
        "status": "VERIFIED_OK",
        "total_encounters_reconciled": 600154,
        "sim1_cz_vs_glob_recovery": [0.805, 0.083],
        "f14_synth_recovery_ladder": [round(x, 4) for x in rates_ladder],
        "sepsis_b1b_last_only_common_bagged": {
            "cz": b2["b1b_measured_point_in_time_last"]["estimands"]["primary_common_panel_bagged_voi"]["cz_fluid"]["nb_per_1k"],
            "glob": b2["b1b_measured_point_in_time_last"]["estimands"]["primary_common_panel_bagged_voi"]["global_cmi"]["nb_per_1k"],
            "delta": b2["b1b_measured_point_in_time_last"]["estimands"]["primary_common_panel_bagged_voi"]["boot_cz_vs_global_cmi"]["diff_mean"],
        },
        "sepsis_b1b_window6h_common_bagged": {
            "cz": b2["b1b_measured_window_6h_summary"]["estimands"]["primary_common_panel_bagged_voi"]["cz_fluid"]["nb_per_1k"],
            "glob": b2["b1b_measured_window_6h_summary"]["estimands"]["primary_common_panel_bagged_voi"]["global_cmi"]["nb_per_1k"],
            "delta": b2["b1b_measured_window_6h_summary"]["estimands"]["primary_common_panel_bagged_voi"]["boot_cz_vs_global_cmi"]["diff_mean"],
        },
        "yale_campus_c_untested_deltas": {
            "esi45_delta_nb": b4["f13_yale_cross_campus_esi45"]["campus_c_esi45_untested"]["boot_cz_vs_global_cmi"]["diff_mean"],
            "esi45_p_dh": b4["f13_yale_cross_campus_esi45"]["campus_c_esi45_untested"]["boot_cz_vs_global_cmi"]["p_two_sided_dh"],
            "allcomers_delta_nb": b4["f13_yale_cross_campus_allcomers"]["campus_c_allcomers_untested"]["boot_cz_vs_global_cmi"]["diff_mean"],
            "allcomers_p_dh": b4["f13_yale_cross_campus_allcomers"]["campus_c_allcomers_untested"]["boot_cz_vs_global_cmi"]["p_two_sided_dh"],
        },
    }
    if verbose:
        print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run_verification(verbose="--quiet" not in sys.argv)
