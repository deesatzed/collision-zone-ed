"""Generate 5 publication-grade 300-DPI figures for Paper 2 v2.3 directly from checkpoints."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from collision_zone.three_predicates_v3 import (
    kl_boltzmann_threshold_gate,
    normalized_beta_threshold_kernel,
)

MERGED_V3 = REPO_ROOT / "results" / "paper2_v2_3_remediation_results.json"
V2_JSON = REPO_ROOT / "results" / "paper2_v2_strengthening_results.json"
FIG_DIR = REPO_ROOT / "manuscript"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "legend.fontsize": 8.2,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

C_BLUE = "#1b4f72"
C_TEAL = "#117a65"
C_RED = "#b03a2e"
C_ORANGE = "#d35400"
C_GRAY = "#5d6d7e"
C_PURPLE = "#6c3483"


def plot_fig1_geometry_and_frontier(v3: dict, v2: dict) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.3), dpi=300)

    # Panel A: Exact Beta-KL threshold geometry vs Log-Loss & Brier
    p = np.linspace(0.002, 0.35, 600)
    tau = 0.0625
    kappa = 24.0
    w_beta_norm = normalized_beta_threshold_kernel(p, tau, kappa)
    w_kl_gate = kl_boltzmann_threshold_gate(p, tau, kappa)
    w_log_scaled = (tau * (1.0 - tau)) / (p * (1.0 - p))  # normalized to 1.0 at p = tau*
    w_log_clipped = np.minimum(w_log_scaled, 3.2)

    ax1.plot(p * 100, w_log_clipped, color=C_RED, lw=2.0, ls="--", label=r"Normalized Log-Loss $[\tau^*(1-\tau^*)]/[p(1-p)]$")
    ax1.axhline(1.0, color=C_GRAY, lw=1.3, ls=":", label=r"Normalized Brier Weight $\equiv 1$")
    ax1.plot(p * 100, w_beta_norm, color=C_BLUE, lw=2.8, label=r"Normalized Beta Kernel $w_{\tau^*,\kappa}(p)/w_{\tau^*,\kappa}(\tau^*)$")
    ax1.plot(
        p[::18] * 100,
        w_kl_gate[::18],
        "o",
        color=C_ORANGE,
        ms=4.5,
        label=r"Boltzmann KL Gate $\exp(-\kappa D_{\mathrm{KL}}(\tau^* \| p))$",
    )
    ax1.axvline(tau * 100, color="#2c3e50", lw=1.1, ls="-.", alpha=0.8)
    ax1.text(tau * 100 + 0.6, 2.85, r"$\tau^* = 6.25\%$", fontsize=8.5, color="#2c3e50")
    ax1.set_xlim(0, 35)
    ax1.set_ylim(0, 3.15)
    ax1.set_xlabel("Predicted Probability $p$ (%)")
    ax1.set_ylabel("Threshold-Normalized Weight (at $\\tau^* = 1.0$)")
    ax1.set_title("(A) Unified Beta–KL Threshold Geometry (Exact Identity)")
    ax1.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.92)

    # Panel B: Empirical (tau*, kappa) Active-Mass Frontier from v2 frontier data
    fr = v2["tau_kappa_frontiers"]
    specs = [
        ("sepsis_bidmc_tau_0p0625", "BIDMC Sepsis ($\\tau^*=6.25\\%$)", C_BLUE, "o"),
        ("yale_esi45_tau_0p04", "Yale ED ESI 4–5 ($\\tau^*=4.0\\%$)", C_TEAL, "s"),
        ("yale_allcomers_tau_0p15", "Yale ED All-Comers ($\\tau^*=15.0\\%$)", C_PURPLE, "^"),
    ]
    for key, lbl, col, mk in specs:
        kt = [float(x) for x in fr[key]["suppression_exponent_kappa_tau"]]
        m_act = [float(x) for x in fr[key]["active_hessian_mass"]]
        ax2.plot(kt, m_act, color=col, lw=2.1, marker=mk, ms=3.8, markevery=6, label=lbl)
        kt_star = float(fr[key]["solved_exponent_kappa_tau"])
        m_star = float(fr[key]["solved_active_mass"])
        ax2.scatter([kt_star], [m_star], color=col, s=65, zorder=5, edgecolor="black", linewidth=0.9)

    ax2.set_yscale("log")
    ax2.set_xlabel(r"Subthreshold Suppression Exponent $\kappa \tau^*$")
    ax2.set_ylabel(r"Active Collision Sample Mass $M(\kappa)$ (log scale)")
    ax2.set_title(r"(B) $(\tau^*, \kappa)$ Suppression vs. Active-Mass Frontier")
    ax2.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.92)
    ax2.grid(True, which="both", alpha=0.22)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_unified_geometry_and_frontier.png", dpi=300)
    plt.close(fig)


def plot_fig2_synthetic_factorial_and_regime_map(v3: dict) -> None:
    b1 = v3["block1_synthetic"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.4), dpi=300)

    # Panel A: F14 5-Objective Discovery Ladder (300 reps)
    f14 = b1["f14_synthetic_discovery_factorial"]
    arms = [
        ("global_cmi", "Global\nLog-Loss", C_RED),
        ("brier", "Brier\nScore", C_GRAY),
        ("direct_utility", "Direct\nThresh. Util.", C_ORANGE),
        ("local_utility", "Local Band\nUtility", C_TEAL),
        ("collision_zone", "Beta Kernel\n(Proposed)", C_BLUE),
    ]
    xs = np.arange(len(arms))
    rates = [100.0 * float(f14[k]["tiebreaker_recovery"]["rate"]) for k, _, _ in arms]
    lo = [rates[i] - 100.0 * float(f14[arms[i][0]]["tiebreaker_recovery"]["ci95_low"]) for i in range(len(arms))]
    hi = [100.0 * float(f14[arms[i][0]]["tiebreaker_recovery"]["ci95_high"]) - rates[i] for i in range(len(arms))]
    nbs_fluid = [float(f14[k]["nb_fluid_s2"]["mean"]) for k, _, _ in arms]

    bars = ax1.bar(xs, rates, width=0.56, color=[c for _, _, c in arms], alpha=0.88, edgecolor="black", linewidth=0.7)
    ax1.errorbar(xs, rates, yerr=[lo, hi], fmt="none", ecolor="black", capsize=4, lw=1.2)
    for idx, b in enumerate(bars):
        ax1.text(
            b.get_x() + b.get_width() / 2.0,
            rates[idx] + hi[idx] + 2.2,
            f"{rates[idx]:.1f}%\n(NB {nbs_fluid[idx]:+.2f})",
            ha="center",
            va="bottom",
            fontsize=7.8,
            fontweight="bold" if idx == 4 else "normal",
        )
    ax1.set_xticks(xs)
    ax1.set_xticklabels([lbl for _, lbl, _ in arms])
    ax1.set_ylim(0, 84)
    ax1.set_ylabel("True Tie-Breaker Recovery % (95% CI)")
    ax1.set_title("(A) Synthetic Discovery-Objective Factorial (300 Reps)")
    ax1.grid(axis="y", alpha=0.22)

    # Panel B: Section 8 36-cell Regime Map (Above-Prevalence Threshold tau* > pi: No Decoy vs Moderate vs Dominant)
    cells = [c for c in b1["sec8_regime_map_36_cells"] if c["threshold_regime"] == "above_prevalence"]
    decoy_vals = [0.0, 0.95, 1.45]
    decoy_lbls = ["No Decoy (Global Suffices)", "Moderate Decoy (0.95)", "Dominant Decoy (1.45)"]
    cz_masses = [0.04, 0.12, 0.25]
    width = 0.24
    x_m = np.arange(len(cz_masses))
    cols_d = [C_GRAY, C_TEAL, C_BLUE]

    for i_d, (dv, dlbl, col) in enumerate(zip(decoy_vals, decoy_lbls, cols_d)):
        means = []
        ses = []
        for cm in cz_masses:
            sub = [c for c in cells if abs(float(c["cz_mass"]) - cm) < 1e-6 and abs(float(c["decoy_strength"]) - dv) < 1e-6]
            vals = [float(c["delta_nb_cz_minus_glob"]) for c in sub]
            se_pool = float(np.sqrt(np.mean([float(c["delta_nb_se"]) ** 2 for c in sub])) / np.sqrt(len(sub)))
            means.append(float(np.mean(vals)))
            ses.append(1.96 * se_pool)
        pos = x_m + (i_d - 1) * width
        ax2.bar(pos, means, width=width, color=col, alpha=0.88, edgecolor="black", linewidth=0.7, label=dlbl)
        ax2.errorbar(pos, means, yerr=ses, fmt="none", ecolor="black", capsize=3.5, lw=1.1)

    ax2.axhline(0.0, color="black", lw=1.0, ls="--")
    ax2.set_xticks(x_m)
    ax2.set_xticklabels([f"Collision Mass {int(m*100)}%" for m in cz_masses])
    ax2.set_ylabel(r"$\Delta\mathrm{NB}_{1000}$ (CZ-Fluid $-$ Global-CMI)")
    ax2.set_title(r"(B) Regime Map ($\tau^* > \pi$): Decoy Boundary")
    ax2.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.92)
    ax2.grid(axis="y", alpha=0.22)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_synthetic_factorial_and_regime_map.png", dpi=300)
    plt.close(fig)


def plot_fig3_sepsis_ledger_and_factorial(v3: dict) -> None:
    b2 = v3["block2_sepsis_primary_ledger"]
    m_last = b2["b1b_measured_point_in_time_last"]
    m_win = b2["b1b_measured_window_6h_summary"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5), dpi=300)

    # Panel A: Estimand & Temporal Channel Reconciliation on Emory B1b (N=6,463)
    labels = [
        "Stage-1 Vitals\n(0% Queries)",
        "Common-Panel Bagged\n(Single-Draw _last)",
        "Common-Panel Bagged\n(6h Window Summary)",
        "Union-Panel Bagged\n(Rerouted Union Cost)",
        "Mean Single-Split\nPolicy (6h Window)",
        "Channel-Billed AFA\n(Covert / Direct-VOI)",
    ]
    cz_vals = [
        3.435,
        float(m_last["estimands"]["primary_common_panel_bagged_voi"]["cz_fluid"]["nb_per_1k"]),
        float(m_win["estimands"]["primary_common_panel_bagged_voi"]["cz_fluid"]["nb_per_1k"]),
        float(m_win["estimands"]["union_panel_bagged_voi_diagnostic"]["rerouted_nb_cz"]),
        float(m_win["estimands"]["mean_single_split_policy_utility"]["cz_fluid_mean_nb"]),
        float(m_win["channel_billed_afa"]["direct_voi"]["per_split_mean_nb"]),
    ]
    gl_vals = [
        3.435,
        float(m_last["estimands"]["primary_common_panel_bagged_voi"]["global_cmi"]["nb_per_1k"]),
        float(m_win["estimands"]["primary_common_panel_bagged_voi"]["global_cmi"]["nb_per_1k"]),
        float(m_win["estimands"]["union_panel_bagged_voi_diagnostic"]["rerouted_nb_glob"]),
        float(m_win["estimands"]["mean_single_split_policy_utility"]["global_cmi_mean_nb"]),
        float(m_win["channel_billed_afa"]["cmi"]["per_split_mean_nb"]),
    ]

    y_pos = np.arange(len(labels))
    h = 0.34
    ax1.barh(y_pos - h / 2, cz_vals, height=h, color=C_BLUE, alpha=0.9, edgecolor="black", lw=0.7, label="CZ-Fluid / Direct-VOI")
    ax1.barh(y_pos + h / 2, gl_vals, height=h, color=C_RED, alpha=0.82, edgecolor="black", lw=0.7, label="Global-CMI / Covert-CMI")
    ax1.axvline(3.435, color=C_GRAY, ls="--", lw=1.1, label="Stage-1 Baseline (+3.43)")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=8.1)
    ax1.invert_yaxis()
    ax1.set_xlabel("External Emory Net Benefit per 1,000 Encounters")
    ax1.set_title("(A) Sepsis Track B1b Estimands & Cost Ledger (N = 6,463)")
    ax1.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.92)
    ax1.grid(axis="x", alpha=0.22)

    # Panel B: Acquisition-Cost Multiplier Sweep (6h Window Common-Panel vs Single-Draw _last)
    sw_win = m_win["acquisition_cost_sweep"]
    sw_last = m_last["acquisition_cost_sweep"]
    mults = [float(d["cost_multiplier"]) for d in sw_win]
    ax2.plot(mults, [float(d["rerouted_cz_nb"]) for d in sw_win], "o-", color=C_BLUE, lw=2.2, label="CZ-Fluid Rerouted (6h Window)")
    ax2.plot(mults, [float(d["rerouted_glob_nb"]) for d in sw_win], "s--", color=C_RED, lw=1.9, label="Global-CMI Rerouted (6h Window)")
    ax2.plot(mults, [float(d["rerouted_cz_nb"]) for d in sw_last], "^:", color=C_TEAL, lw=1.8, label="CZ-Fluid Rerouted (Single-Draw _last)")
    ax2.axhline(3.435, color=C_GRAY, ls="-.", lw=1.1, label="Zero-Query Stage-1 (+3.43)")
    ax2.axvline(1.0, color="#2c3e50", ls=":", lw=1.0)
    ax2.text(1.06, 6.2, "Default CPT\nCost (1.0x)", fontsize=8.0, color="#2c3e50")
    ax2.set_xlabel("Laboratory Panel Cost Multiplier ($1.0\\times = 0.030\\, C_{\\mathrm{FP}}$)")
    ax2.set_ylabel("Rerouted Bagged VOI Net Benefit / 1,000")
    ax2.set_title("(B) Acquisition-Cost Sensitivity Sweep on Emory B1b")
    ax2.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.92)
    ax2.grid(True, alpha=0.22)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_sepsis_ledger_and_factorial.png", dpi=300)
    plt.close(fig)


def plot_fig4_horizons_shift_and_missingness(v3: dict) -> None:
    b3 = v3["block3_horizons_shift_missingness"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.3), dpi=300)

    # Panel A: F9 Event-Time Distribution at 6h Landmark & 2D Horizon Reality
    lead = b3["f9_w6h_pre_onset_lead_stratification"]
    bins = ["0_to_2h", "3_to_6h", "7_to_12h", "gt_12h"]
    blbls = ["0–2h\nPost-6h", "3–6h\nPost-6h", "7–12h\nPost-6h", ">12h Later\nin Stay"]
    cnts = [int(lead[k]["n_events"]) for k in bins]
    pcts = [100.0 * c / sum(cnts) for c in cnts]
    sens_cz = [float(lead[k]["sens_cz_fluid_pct"]) for k in bins]

    x = np.arange(len(bins))
    bars = ax1.bar(x, cnts, width=0.52, color=[C_ORANGE, C_ORANGE, C_TEAL, C_BLUE], alpha=0.88, edgecolor="black", lw=0.7)
    for i, b in enumerate(bars):
        ax1.text(
            b.get_x() + b.get_width() / 2.0,
            cnts[i] + 6,
            f"{cnts[i]} ({pcts[i]:.1f}%)\nSens {sens_cz[i]:.0f}%",
            ha="center",
            va="bottom",
            fontsize=8.0,
            fontweight="bold" if i == 3 else "normal",
        )
    ax1.set_xticks(x)
    ax1.set_xticklabels(blbls)
    ax1.set_ylim(0, 325)
    ax1.set_ylabel("External Emory Sepsis Events (Total = 349)")
    ax1.set_title("(A) W=6h Landmark Timing: 77.7% Occur >12h Later")
    ax1.grid(axis="y", alpha=0.22)

    # Panel B: F10 Prior-Shift Misspecification Sweep + Taxonomic Anchors on Track B2
    f10 = b3["f10_prevalence_shift_taxonomy_b2"]
    sw = f10["prevalence_misspecification_sweep"]
    errs = [float(d["error_pp"]) for d in sw]
    nbs = [float(d["nb_per_1k"]) for d in sw]
    ax2.plot(errs, nbs, "o-", color=C_BLUE, lw=2.2, label="Prior-Odds Shift Across Estimation Error")
    ax2.axhline(0.0, color="black", lw=1.0, ls="--", label="Treat-None / EM Collapse (NB = 0.00)")
    zs_nb = float(f10["disjoint_75pct_holdout_comparison"]["zero_shot_unadjusted"]["nb_per_1k"])
    hist_nb = float(f10["disjoint_75pct_holdout_comparison"]["historical_first25pct_prior"]["nb_per_1k"])
    ax2.axhline(zs_nb, color=C_RED, lw=1.5, ls=":", label=f"Zero-Shot Unadjusted ({zs_nb:+.2f})")
    ax2.scatter([-0.16], [hist_nb], color=C_TEAL, s=85, zorder=6, marker="*", edgecolor="black", label=f"25% Historical Prior on 75% Holdout ({hist_nb:+.2f})")
    ax2.set_xlabel("Target Prevalence Estimation Error (percentage points vs. 4.14%)")
    ax2.set_ylabel("Emory Track B2 Net Benefit / 1,000")
    ax2.set_title("(B) Prior-Shift Taxonomy & Prevalence Error Tolerance")
    ax2.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.92, fontsize=7.8)
    ax2.grid(True, alpha=0.22)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_horizons_shift_and_missingness.png", dpi=300)
    plt.close(fig)


def plot_fig5_yale_campuses_and_cfbh_governance(v3: dict) -> None:
    b4 = v3["block4_yale_and_sensitivity"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.3), dpi=300)

    # Panel A: Yale Cross-Campus Delta NB (CZ-Fluid minus Global-CMI) across Campus B and Untested Campus C
    esi_b = b4["f13_yale_cross_campus_esi45"]["campus_b_esi45"]["boot_cz_vs_global_cmi"]
    esi_c = b4["f13_yale_cross_campus_esi45"]["campus_c_esi45_untested"]["boot_cz_vs_global_cmi"]
    all_b = b4["f13_yale_cross_campus_allcomers"]["campus_b_allcomers"]["boot_cz_vs_global_cmi"]
    all_c = b4["f13_yale_cross_campus_allcomers"]["campus_c_allcomers_untested"]["boot_cz_vs_global_cmi"]

    items = [
        ("Campus A → B\nESI 4–5 (N=51,205)", esi_b, C_GRAY),
        ("Campus A → C (Untested)\nESI 4–5 (N=31,430)", esi_c, C_TEAL),
        ("Campus A → B\nAll-Comers (N=166,497)", all_b, C_BLUE),
        ("Campus A → C (Untested)\nAll-Comers (N=71,706)", all_c, C_PURPLE),
    ]
    yp = np.arange(len(items))
    means = [float(d["diff_mean"]) for _, d, _ in items]
    err_lo = [float(d["diff_mean"]) - float(d["ci95_low"]) for _, d, _ in items]
    err_hi = [float(d["ci95_high"]) - float(d["diff_mean"]) for _, d, _ in items]
    pvals = [float(d["p_two_sided_dh"]) for _, d, _ in items]

    ax1.axvline(0.0, color="black", lw=1.0, ls="--")
    for idx, (lbl, _, col) in enumerate(items):
        ax1.errorbar(
            means[idx],
            yp[idx],
            xerr=[[err_lo[idx]], [err_hi[idx]]],
            fmt="o",
            color=col,
            ms=7,
            lw=2.0,
            capsize=4,
        )
        ax1.text(
            means[idx] + err_hi[idx] + 0.03,
            yp[idx],
            f"{means[idx]:+.2f} (p={pvals[idx]:.4f})",
            va="center",
            fontsize=8.2,
            fontweight="bold" if pvals[idx] < 0.05 else "normal",
        )
    ax1.set_yticks(yp)
    ax1.set_yticklabels([x[0] for x in items], fontsize=8.4)
    ax1.invert_yaxis()
    ax1.set_xlim(-0.35, 1.25)
    ax1.set_xlabel(r"Paired $\Delta\mathrm{NB}_{1000}$ (CZ-Fluid $-$ Global-CMI, 95% CI)")
    ax1.set_title("(A) Yale Cross-Campus Transfer (Campuses B & C)")
    ax1.grid(axis="x", alpha=0.22)

    # Panel B: F15 Repeated-Subsample (R=200) Campus B CfBH Burn-In Governance Curve
    burn = b4["f15_campus_b_cfbh_repeated_burnin_curve"]
    x_lbls = ["0\n(Zero-Shot)", "250", "500", "1,000", "2,000", "4,000"]
    xb = np.arange(len(burn))
    fdrs = [100.0 * float(r["mean_fdp_empirical_fdr"]) for r in burn]
    recs = [float(r["mean_recall_pct"]) for r in burn]
    alts = [float(r["mean_alerts_per_100"]) for r in burn]

    ax2.plot(xb, fdrs, "o-", color=C_RED, lw=2.2, label=r"Empirical FDR $\mathbb{E}[\mathrm{FDP}]$ (%)")
    ax2.plot(xb, recs, "s--", color=C_BLUE, lw=2.0, label="Mean Recall / Sensitivity (%)")
    ax2.axhline(75.0, color=C_RED, ls=":", lw=1.3, label="Nominal FDR Target (75%)")
    for i, a_val in enumerate(alts):
        dx, dy = (18, -8) if i == 0 else ((0, 10) if i == 5 else (0, 8))
        ax2.annotate(
            f"{a_val:.1f}/100",
            (xb[i], fdrs[i]),
            textcoords="offset points",
            xytext=(dx, dy),
            ha="center",
            fontsize=7.6,
            color="#2c3e50",
        )
    ax2.set_xticks(xb)
    ax2.set_xticklabels(x_lbls)
    ax2.set_ylim(0, 108)
    ax2.set_xlabel("Local Campus B Burn-In Calibration Sample Size $n_{\\mathrm{burn}}$ (R = 200 Splits)")
    ax2.set_ylabel("Percentage (%)  [Annotations: Alerts / 100 Visits]")
    ax2.set_title("(B) Repeated-Sampling Conformal (CfBH) Burn-In Curve")
    ax2.legend(loc="center right", frameon=True, facecolor="white", framealpha=0.92, fontsize=7.9)
    ax2.grid(True, alpha=0.22)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_yale_campuses_and_cfbh_governance.png", dpi=300)
    plt.close(fig)


def main() -> None:
    v3 = json.loads(MERGED_V3.read_text())
    v2 = json.loads(V2_JSON.read_text())
    plot_fig1_geometry_and_frontier(v3, v2)
    plot_fig2_synthetic_factorial_and_regime_map(v3)
    plot_fig3_sepsis_ledger_and_factorial(v3)
    plot_fig4_horizons_shift_and_missingness(v3)
    plot_fig5_yale_campuses_and_cfbh_governance(v3)
    print("Generated all 5 publication figures in", FIG_DIR)


if __name__ == "__main__":
    main()
