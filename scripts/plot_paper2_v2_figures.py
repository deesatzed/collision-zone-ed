"""Generate 3 publication-grade 300-DPI figures for Paper 2 v2 / JAMIA submission.

Outputs inside `manuscript/`:
  1. `fig1_mechanism_frontier_and_beta.png`
  2. `fig2_sepsis_transfer_afa_and_landmarks.png`
  3. `fig3_calibration_and_yale_transfer.png`
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"

REPO_ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT_DIR = REPO_ROOT / "manuscript"
SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from collision_zone.three_predicates import bernoulli_kl_from_tau, schervish_collision_weight

plt.rcParams.update(
    {
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.edgecolor": "#334155",
        "axes.linewidth": 0.85,
        "axes.titlesize": 10.6,
        "axes.labelsize": 9.4,
        "xtick.labelsize": 8.4,
        "ytick.labelsize": 8.4,
        "legend.fontsize": 8.0,
    }
)


def plot_fig1(res: dict) -> Path:
    sim1 = res["sim1_1000_replicates"]
    front = res["tau_kappa_frontiers"]
    sim2 = res["archived_reference_summaries"]["sim2_leaf_starvation"]
    ya_low = res["archived_reference_summaries"]["yale_track_a_esi45_parity"]["arms"]
    ya_full = res["archived_reference_summaries"]["yale_track_a_full_560k_parity"]["arms"]
    b1b_int = res["sepsis_track_b1b_primary"]["internal_bidmc_test"]
    ta_int = res["sepsis_track_a_parity"]["internal_bidmc_test"]
    b4b = res["yale_strengthened"]["yale_b4_cross_campus_primary"]["b4b_allcomers_cross_campus"]["external_per_split"]
    b4a = res["yale_strengthened"]["yale_b4_cross_campus_primary"]["b4a_esi45_cross_campus"]["external_per_split"]

    fig, axes = plt.subplots(2, 2, figsize=(14.2, 9.6), dpi=300)

    # Panel A: Log-Loss Singularity vs Schervish Kernel & Stage-2 KL Gate
    ax = axes[0, 0]
    p_grid = np.linspace(0.002, 0.32, 600)
    tau = 0.0625
    w_log = 1.0 / (p_grid * (1.0 - p_grid))
    w_log_norm = w_log / np.max(w_log)
    w_cz = schervish_collision_weight(p_grid, tau_star=tau, kappa=24.0)
    w_cz_norm = w_cz / np.max(w_cz)
    pi_kl = np.exp(-20.0 * bernoulli_kl_from_tau(p_grid, tau))

    ax.axvspan(0.0, 3.0, color="#FEE2E2", alpha=0.45, label="Subthreshold TN Region (<3%)")
    ax.axvspan(3.5, 10.0, color="#DCFCE7", alpha=0.45, label=r"Collision Zone Around $\tau^*$")
    ax.plot(p_grid * 100, w_log_norm, "--", lw=2.2, color="#DC2626", label=r"Global Log-Loss: $1/[p(1-p)]$")
    ax.plot(p_grid * 100, w_cz_norm, "-", lw=2.5, color="#0284C7", label=r"Schervish Kernel $w_{\tau^*,24}(p)$")
    ax.plot(p_grid * 100, pi_kl, "-.", lw=2.2, color="#059669", label=r"Stage-2 KL Gate $\pi_{20}(p)$")
    ax.axvline(tau * 100, color="#0F172A", ls=":", lw=1.5, label=r"Threshold $\tau^* = 6.25\%$")
    ax.set_xlim(0, 32)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("Baseline / Posterior Risk p (%)")
    ax.set_ylabel("Normalized Curvature Weight / Gate")
    ax.set_title("A. Predicate 1 Kernel & Stage-2 KL Gate vs. Log-Loss Singularity", fontweight="bold")
    ax.legend(loc="upper right", frameon=True, framealpha=0.92)
    ax.grid(alpha=0.25)

    # Panel B: (tau*, kappa) Suppression Exponent vs Active Collision Mass Frontier (P2-13)
    ax = axes[0, 1]
    fr_specs = [
        ("yale_esi45_tau_0p04", r"Yale ESI 4-5 ($\tau^*=4.0\%$)", "#7C3AED", "o"),
        ("sepsis_bidmc_tau_0p0625", r"BIDMC Sepsis ($\tau^*=6.25\%$)", "#0284C7", "s"),
        ("yale_allcomers_tau_0p15", r"Yale All-Comers ($\tau^*=15.0\%$)", "#059669", "^"),
    ]
    for fkey, flab, fcol, fmk in fr_specs:
        fd = front[fkey]
        exps = np.array(fd["suppression_exponent_kappa_tau"], dtype=float)
        masses = np.array(fd["active_hessian_mass"], dtype=float)
        ax.plot(exps, masses, "-", lw=2.2, color=fcol, label=flab)
        # Mark solved operating point kappa*
        ax.scatter(
            [fd["solved_exponent_kappa_tau"]],
            [fd["solved_active_mass"]],
            color=fcol,
            marker=fmk,
            s=72,
            zorder=5,
            edgecolors="#0F172A",
            linewidths=0.9,
        )
    ax.axhline(240.0, color="#DC2626", ls="--", lw=1.4, label=r"Active-Mass Floor $M_{\min} = 240$")
    ax.set_yscale("log")
    ax.set_xlabel(r"Subthreshold Suppression Exponent ($\kappa \cdot \tau^*$)")
    ax.set_ylabel(r"Active Collision Hessian Mass $M(\kappa)$ [log scale]")
    ax.set_title(r"B. $(\tau^*, \kappa)$ Suppression-vs-Active-Mass Frontier (Markers = $\kappa^*$)", fontweight="bold")
    ax.legend(loc="upper right", frameon=True, framealpha=0.92)
    ax.grid(alpha=0.25, which="both")

    # Panel C: 1,000-Replicate Sim 1 Dilution Rescue & Null FWER with Exact 95% CIs (P2-5)
    ax = axes[1, 0]
    cats_c = [
        "True Tie-Breaker\nPicked (B=1)",
        "Pilot Controls:\nHigh-Signal FP %",
        "Sequential Null\nFWER (B=5)",
        "Pure All-Null\nStep FWER",
    ]
    gl_means = [
        sim1["global_cmi_tiebreaker_recovery"]["rate"] * 100.0,
        sim1["pilot_hsfp_share_global_neyman"]["mean"] * 100.0,
        sim1["global_cmi_b5_sequential_null_fwer"]["rate"] * 100.0,
        sim1["global_cmi_pure_all_null_step_fwer"]["rate"] * 100.0,
    ]
    gl_lo = [
        sim1["global_cmi_tiebreaker_recovery"]["ci95_low"] * 100.0,
        sim1["pilot_hsfp_share_global_neyman"]["ci95_low"] * 100.0,
        sim1["global_cmi_b5_sequential_null_fwer"]["ci95_low"] * 100.0,
        sim1["global_cmi_pure_all_null_step_fwer"]["ci95_low"] * 100.0,
    ]
    gl_hi = [
        sim1["global_cmi_tiebreaker_recovery"]["ci95_high"] * 100.0,
        sim1["pilot_hsfp_share_global_neyman"]["ci95_high"] * 100.0,
        sim1["global_cmi_b5_sequential_null_fwer"]["ci95_high"] * 100.0,
        sim1["global_cmi_pure_all_null_step_fwer"]["ci95_high"] * 100.0,
    ]

    cz_means = [
        sim1["cz_tiebreaker_recovery"]["rate"] * 100.0,
        sim1["pilot_hsfp_share_collision_neyman"]["mean"] * 100.0,
        sim1["cz_b5_sequential_null_fwer"]["rate"] * 100.0,
        sim1["cz_pure_all_null_step_fwer"]["rate"] * 100.0,
    ]
    cz_lo = [
        sim1["cz_tiebreaker_recovery"]["ci95_low"] * 100.0,
        sim1["pilot_hsfp_share_collision_neyman"]["ci95_low"] * 100.0,
        sim1["cz_b5_sequential_null_fwer"]["ci95_low"] * 100.0,
        sim1["cz_pure_all_null_step_fwer"]["ci95_low"] * 100.0,
    ]
    cz_hi = [
        sim1["cz_tiebreaker_recovery"]["ci95_high"] * 100.0,
        sim1["pilot_hsfp_share_collision_neyman"]["ci95_high"] * 100.0,
        sim1["cz_b5_sequential_null_fwer"]["ci95_high"] * 100.0,
        sim1["cz_pure_all_null_step_fwer"]["ci95_high"] * 100.0,
    ]

    xc = np.arange(len(cats_c))
    bw = 0.35
    gl_err = np.vstack([np.maximum(0, np.array(gl_means) - np.array(gl_lo)), np.maximum(0, np.array(gl_hi) - np.array(gl_means))])
    cz_err = np.vstack([np.maximum(0, np.array(cz_means) - np.array(cz_lo)), np.maximum(0, np.array(cz_hi) - np.array(cz_means))])

    ax.bar(xc - bw / 2, gl_means, width=bw, yerr=gl_err, capsize=3.5, color="#94A3B8", edgecolor="#334155", lw=0.6, label="Global-CMI (Global Neyman)")
    ax.bar(xc + bw / 2, cz_means, width=bw, yerr=cz_err, capsize=3.5, color="#059669", edgecolor="#064E3B", lw=0.6, label="CZ-Fluid (Collision-Zone + Holm)")
    ax.axhline(5.0, color="#DC2626", ls=":", lw=1.3, label=r"Nominal $\alpha = 5.0\%$ FWER Bound")
    for xi, gm, gh, cm, ch_v in zip(xc, gl_means, gl_hi, cz_means, cz_hi):
        y_gl = max(gh + 1.8, 6.8)
        y_cz = max(ch_v + 1.8, 6.8)
        ax.text(xi - bw / 2, y_gl, f"{gm:.1f}%", ha="center", fontsize=7.9, color="#334155",
                bbox={"boxstyle": "round,pad=0.06", "facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
        ax.text(xi + bw / 2, y_cz, f"{cm:.1f}%", ha="center", fontsize=7.9, fontweight="bold", color="#065F46",
                bbox={"boxstyle": "round,pad=0.06", "facecolor": "white", "alpha": 0.85, "edgecolor": "none"})

    null_g_mean = sim1["empirical_null_schervish_gain_step1"]["mean"]
    nb_diff = sim1["paired_nb_diff_cz_vs_global_cmi"]["mean"]
    ax.text(
        0.98,
        0.64,
        f"1,000-Rep Null Gain: {null_g_mean:+.2e} (<= 0)\nAuto-VOI Delta NB: {nb_diff:+.2f}/1k",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.0,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#F8FAFC", "edgecolor": "#CBD5E1"},
    )
    ax.set_xticks(xc)
    ax.set_xticklabels(cats_c)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Percentage (%, with 95% CIs)")
    ax.set_title("C. Synthetic Dilution Rescue & Null FWER (1,000 Replicates)", fontweight="bold")
    ax.legend(loc="upper right", frameon=True, framealpha=0.92)
    ax.grid(axis="y", alpha=0.25)

    # Panel D: Learned beta_CZ Self-Damping Regime Detector Across Synthetic & Clinical Regimes (P2-3)
    ax = axes[1, 1]
    regimes = [
        ("Sim 2A:\nHomog.", sim2["1_homogeneous"]["beta_cz"]["mean"], 1.96 * sim2["1_homogeneous"]["beta_cz"]["se"], "#94A3B8"),
        ("Sim 2C:\nHijack", sim2["3_tn_capacity_hijack"]["beta_cz"]["mean"], 1.96 * sim2["3_tn_capacity_hijack"]["beta_cz"]["se"], "#0284C7"),
        ("Sim 2B:\nInversion", sim2["2_tn_decoy_inversion"]["beta_cz"]["mean"], 1.96 * sim2["2_tn_decoy_inversion"]["beta_cz"]["se"], "#059669"),
        (
            "Yale ESI 4-5\n(1.1k ev.)",
            ya_low["paper2_cz_disc_fluid_s2"]["learned_beta_cz"]["mean"],
            1.96 * ya_low["paper2_cz_disc_fluid_s2"]["learned_beta_cz"]["se"],
            "#F59E0B",
        ),
        (
            "Yale B4a Ext.\nESI 4-5",
            b4a["paper2_cz_disc_fluid_s2"]["learned_beta_cz"]["mean"],
            1.96 * b4a["paper2_cz_disc_fluid_s2"]["learned_beta_cz"]["se"],
            "#D97706",
        ),
        (
            "BIDMC Sepsis\n6h Parity",
            ta_int["paper2_cz_disc_fluid_s2"]["learned_beta_cz"]["mean"],
            1.96 * ta_int["paper2_cz_disc_fluid_s2"]["learned_beta_cz"]["se"],
            "#0284C7",
        ),
        (
            "Yale B4b/560k\n(22k ev.)",
            ya_full["paper2_cz_disc_fluid_s2"]["learned_beta_cz"]["mean"],
            1.96 * ya_full["paper2_cz_disc_fluid_s2"]["learned_beta_cz"]["se"],
            "#059669",
        ),
    ]
    xd = np.arange(len(regimes))
    b_means = np.array([r[1] for r in regimes], dtype=float)
    b_ci = np.array([r[2] for r in regimes], dtype=float)
    b_lo_err = np.minimum(b_means, b_ci)
    b_hi_err = b_ci
    b_cols = [r[3] for r in regimes]

    ax.bar(xd, b_means, yerr=np.vstack([b_lo_err, b_hi_err]), capsize=3.5, width=0.58, color=b_cols, edgecolor="#334155", lw=0.6)
    ax.axvline(2.5, color="#64748B", ls="--", lw=1.1)
    for xi, bm, b_h in zip(xd, b_means, b_hi_err):
        ax.text(xi, bm + b_h + 0.035, f"{bm:.2f}", ha="center", fontsize=8.0, fontweight="bold")
    ax.text(1.0, 1.40, "Synthetic Regimes", ha="center", fontsize=8.5, style="italic", color="#475569")
    ax.text(4.6, 1.40, "Clinical Scale & Heterogeneity", ha="center", fontsize=8.5, style="italic", color="#475569")
    ax.set_xticks(xd)
    ax.set_xticklabels([r[0] for r in regimes], fontsize=7.9)
    ax.set_ylim(0, 1.54)
    ax.set_ylabel(r"Learned OOF Specialist Weight $\hat{\beta}_{\mathrm{CZ}}$ (95% CI)")
    ax.set_title(r"D. Learned $\hat{\beta}_{\mathrm{CZ}}$ as Self-Damping Collision Regime Detector", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout(pad=1.35)
    out = MANUSCRIPT_DIR / "fig1_mechanism_frontier_and_beta.png"
    plt.savefig(out, dpi=300)
    plt.close(fig)
    return out


def plot_fig2(res: dict) -> Path:
    b1b = res["sepsis_track_b1b_primary"]
    ta = res["sepsis_track_a_parity"]
    b2 = res["sepsis_track_b2_gap_and_recal_rescue"]
    sweep = res["sepsis_landmark_sweep"]["measured_bloodwork_cpt"]

    fig, axes = plt.subplots(2, 2, figsize=(14.4, 9.8), dpi=300)

    # Panel A: Emory B1b (Primary) & Track A — Per-Split Mean (+/- 95% CI) vs Symmetric Ensemble NB + Covert-AFA
    ax = axes[0, 0]
    arms_a = [
        ("Stage-1\nVitals", "stage1_base", "#CBD5E1"),
        ("Global-CMI\n(Global S2)", "paper1_global_cmi_global_s2", "#3B82F6"),
        ("Covert-AFA\n(ICML 2023)", "covert2023_greedy_cmi_afa", "#8B5CF6"),
        ("CZ-Fluid\n(Proposed)", "paper2_cz_disc_fluid_s2", "#059669"),
    ]
    x_groups = np.arange(len(arms_a))
    bw = 0.34

    def _get_split_mean_ci(cohort_dict: dict, key: str) -> tuple[float, float]:
        ps = cohort_dict["external_emory_per_split"]
        if key == "stage1_base":
            d = ps["paper1_global_cmi_global_s2"]["nb_base_0pct"]
        else:
            d = ps[key]["nb_voi_auto"]
        return float(d["mean"]), 1.96 * float(d["se"])

    b1b_sp = [_get_split_mean_ci(b1b, k) for _, k, _ in arms_a]
    b1b_ens = [float(b1b["external_emory_symmetric_ensemble"][k]["ens_nb_auto"]) for _, k, _ in arms_a]
    ta_sp = [_get_split_mean_ci(ta, k) for _, k, _ in arms_a]
    ta_ens = [float(ta["external_emory_symmetric_ensemble"][k]["ens_nb_auto"]) for _, k, _ in arms_a]

    ax.bar(
        x_groups - bw / 2,
        [m for m, _ in b1b_sp],
        yerr=[ci for _, ci in b1b_sp],
        capsize=3.5,
        width=bw,
        color=[c for _, _, c in arms_a],
        edgecolor="#0F172A",
        lw=0.7,
        label="Track B1b Per-Split Mean (+/-95% CI)",
    )
    ax.scatter(
        x_groups - bw / 2,
        b1b_ens,
        marker="D",
        s=52,
        color="#0F172A",
        zorder=6,
        label="Track B1b Symmetric Ensemble NB",
    )
    ax.bar(
        x_groups + bw / 2,
        [m for m, _ in ta_sp],
        yerr=[ci for _, ci in ta_sp],
        capsize=3.5,
        width=bw,
        color=[c for _, _, c in arms_a],
        alpha=0.48,
        hatch="//",
        edgecolor="#334155",
        lw=0.7,
        label="Track A Parity Per-Split (+/-95% CI)",
    )
    ax.scatter(
        x_groups + bw / 2,
        ta_ens,
        marker="o",
        s=50,
        facecolors="#FFFFFF",
        edgecolors="#0F172A",
        linewidths=1.4,
        zorder=6,
        label="Track A Symmetric Ensemble NB",
    )
    ax.axhline(0.0, color="#475569", lw=0.9)

    bt_b1b = b1b["emory_paired_bootstraps_10k"]["cz_fluid_vs_global_cmi"]
    bt_afa = b1b["emory_paired_bootstraps_10k"]["cz_fluid_vs_covert_afa"]
    ax.text(
        0.02,
        0.97,
        f"Primary B1b Ensemble Bootstraps (B=10,000):\n"
        f"CZ-Fluid vs Global-CMI: {bt_b1b['diff_mean']:+.2f} [{bt_b1b['ci95_low']:+.2f}, {bt_b1b['ci95_high']:+.2f}], p={bt_b1b['p_two_sided_dh']:.4f}\n"
        f"CZ-Fluid vs Covert-AFA: {bt_afa['diff_mean']:+.2f} [{bt_afa['ci95_low']:+.2f}, {bt_afa['ci95_high']:+.2f}], p={bt_afa['p_two_sided_dh']:.4f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.6,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#F8FAFC", "edgecolor": "#CBD5E1"},
    )
    ax.set_xticks(x_groups)
    ax.set_xticklabels([lbl for lbl, _, _ in arms_a])
    ax.set_ylim(-3.3, 8.2)
    ax.set_ylabel("Emory External Net Benefit (/1,000)")
    ax.set_title("A. Emory Zero-Shot Transfer vs. Global-CMI & Covert-AFA", fontweight="bold")
    ax.legend(loc="upper right", fontsize=7.2, frameon=True, framealpha=0.94)
    ax.grid(axis="y", alpha=0.25)

    # Panel B: Full Emory Vickers Decision Curve (Track B1b) Including Covert-AFA
    ax = axes[0, 1]
    dca = b1b["emory_decision_curve"]
    taus_pct = [t * 100.0 for t in dca["taus"]]
    ax.plot(taus_pct, dca["treat_all"], ":", lw=1.6, color="#94A3B8", label="Treat All")
    ax.axhline(0.0, ls="--", lw=1.2, color="#64748B", label="Treat None (0)")
    ax.plot(taus_pct, dca["stage1_base"], "-.", lw=1.9, color="#475569", label="Stage-1 Bedside Vitals")
    ax.plot(taus_pct, dca["paper1_global"], "-o", ms=4.5, lw=2.0, color="#3B82F6", label="Global-CMI + Global S2")
    if "covert_afa" in dca:
        ax.plot(taus_pct, dca["covert_afa"], "-^", ms=4.8, lw=2.0, color="#8B5CF6", label="Covert-AFA (Greedy CMI)")
    ax.plot(taus_pct, dca["paper2_fluid"], "-s", ms=5.2, lw=2.5, color="#059669", label="CZ-Fluid (Proposed)")
    ax.axvline(6.25, color="#0F172A", ls=":", lw=1.3, label=r"Contract $\tau^* = 6.25\%$")
    ax.set_ylim(-5.5, max(dca["paper2_fluid"]) + 3.0)
    ax.set_xlabel(r"Clinical Action Threshold $\tau$ (%)")
    ax.set_ylabel("Ensemble Net Benefit per 1,000 Patients")
    ax.set_title("B. Emory Vickers Decision Curve (Track B1b Measured CPT)", fontweight="bold")
    ax.legend(loc="upper right", frameon=True, framealpha=0.92)
    ax.grid(alpha=0.25)

    # Panel C: PhysioNet BIDMC -> Emory Landmark Sweep (4h / 6h / 8h / 12h, P2-14)
    ax = axes[1, 0]
    ws = [4, 6, 8, 12]
    w_keys = [f"{w}h" for w in ws]
    base_m = [sweep[wk]["external_emory_per_split"]["paper1_global_cmi_global_s2"]["nb_base_0pct"]["mean"] for wk in w_keys]
    base_ci = [1.96 * sweep[wk]["external_emory_per_split"]["paper1_global_cmi_global_s2"]["nb_base_0pct"]["se"] for wk in w_keys]
    gl_m = [sweep[wk]["external_emory_per_split"]["paper1_global_cmi_global_s2"]["nb_voi_auto"]["mean"] for wk in w_keys]
    gl_ci = [1.96 * sweep[wk]["external_emory_per_split"]["paper1_global_cmi_global_s2"]["nb_voi_auto"]["se"] for wk in w_keys]
    cz_m = [sweep[wk]["external_emory_per_split"]["paper2_cz_disc_fluid_s2"]["nb_voi_auto"]["mean"] for wk in w_keys]
    cz_ci = [1.96 * sweep[wk]["external_emory_per_split"]["paper2_cz_disc_fluid_s2"]["nb_voi_auto"]["se"] for wk in w_keys]
    cz_ens_w = [sweep[wk]["external_emory_symmetric_ensemble"]["paper2_cz_disc_fluid_s2"]["ens_nb_auto"] for wk in w_keys]
    gl_ens_w = [sweep[wk]["external_emory_symmetric_ensemble"]["paper1_global_cmi_global_s2"]["ens_nb_auto"] for wk in w_keys]

    xw = np.array(ws, dtype=float)
    ax.errorbar(xw - 0.18, base_m, yerr=base_ci, fmt="--o", ms=5, lw=1.8, capsize=3.5, color="#64748B", label="Stage-1 Vitals (Per-Split +/-95% CI)")
    ax.errorbar(xw, gl_m, yerr=gl_ci, fmt="-^", ms=5.5, lw=2.0, capsize=3.5, color="#3B82F6", label="Global-CMI (Per-Split +/-95% CI)")
    ax.errorbar(xw + 0.18, cz_m, yerr=cz_ci, fmt="-s", ms=5.8, lw=2.3, capsize=3.5, color="#059669", label="CZ-Fluid (Per-Split +/-95% CI)")
    ax.plot(xw + 0.18, cz_ens_w, "D", ms=6.2, color="#064E3B", label="CZ-Fluid Symmetric Ensemble")
    ax.plot(xw, gl_ens_w, "d", ms=5.8, color="#1D4ED8", label="Global-CMI Symmetric Ensemble")
    ax.axhline(0.0, color="#475569", lw=0.9)
    ax.set_xticks(ws)
    ax.set_xticklabels([f"{w}h Window\n(N={sweep[f'{w}h']['emory_rows']:,})" for w in ws])
    ax.set_ylabel("Emory External Net Benefit (/1,000)")
    ax.set_title("C. BIDMC -> Emory Observation Landmark Sweep (4h to 12h)", fontweight="bold")
    ax.legend(loc="best", fontsize=7.5, frameon=True, framealpha=0.92)
    ax.grid(alpha=0.25)

    # Panel D: Emory Track B2 Prevalence-Shift Calibration Curve & Intercept Recalibration Rescue (P2-8)
    ax = axes[1, 1]
    cal_raw = b2["calibration"]["emory_cz_fluid"]
    recal_info = b2["prevalence_shift_recalibration_rescue"]
    cal_after = recal_info["prior_odds_recalibrated_with_voi_rerouting"]["cz_fluid_calibration"]

    p_raw_c = np.array(cal_raw["curve_p"]) * 100.0
    y_raw_c = np.array(cal_raw["curve_y_loess"]) * 100.0
    p_aft_c = np.array(cal_after["curve_p"]) * 100.0
    y_aft_c = np.array(cal_after["curve_y_loess"]) * 100.0
    max_ax = min(25.0, max(float(np.max(p_raw_c)), float(np.max(y_raw_c)), 15.0) + 1.5)

    ax.plot([0, max_ax], [0, max_ax], ":", lw=1.6, color="#0F172A", label="Ideal 45-deg Calibration")
    ax.plot(
        p_raw_c,
        y_raw_c,
        "--o",
        ms=3.2,
        lw=2.1,
        color="#DC2626",
        label=(
            f"Zero-Shot Unrecalibrated (CITL={cal_raw['citl']:+.2f}, "
            f"ICI={cal_raw['ici_loess']:.3f}, NB={recal_info['raw_zero_shot']['ens_cz_fluid_nb']:+.2f})"
        ),
    )
    ens_nb_recal = recal_info["prior_odds_recalibrated_with_voi_rerouting"]["ens_cz_fluid_nb"]
    ax.plot(
        p_aft_c,
        y_aft_c,
        "-s",
        ms=3.2,
        lw=2.3,
        color="#059669",
        label=(
            f"Prior-Odds Intercept Shift {recal_info['prior_log_odds_offset']:+.2f} "
            f"(CITL={cal_after['citl']:+.2f}, ICI={cal_after['ici_loess']:.3f}, NB={ens_nb_recal:+.2f})"
        ),
    )
    ax.axvline(6.25, color="#475569", ls=":", lw=1.2)
    bt_b2_rec = recal_info["prior_odds_recalibrated_with_voi_rerouting"]["boot_cz_vs_base_10k"]
    ax.text(
        0.03,
        0.96,
        f"Prevalence Shift: BIDMC {b2['bidmc_prev']*100:.2f}% -> Emory {b2['emory_prev']*100:.2f}%\n"
        f"Pure Prior-Odds Shift (0 Emory labels): Delta_0 = {recal_info['prior_log_odds_offset']:+.2f}\n"
        f"Recalibrated CZ-Fluid vs Base: {bt_b2_rec['diff_mean']:+.2f} "
        f"[{bt_b2_rec['ci95_low']:+.2f}, {bt_b2_rec['ci95_high']:+.2f}]",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.7,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#F8FAFC", "edgecolor": "#CBD5E1"},
    )
    ax.set_xlim(0, max_ax)
    ax.set_ylim(0, max_ax)
    ax.set_xlabel("Predicted Sepsis Risk (%)")
    ax.set_ylabel("LOESS-Smoothed Observed Sepsis Rate (%)")
    ax.set_title("D. Track B2 Prevalence-Shift Miscalibration & Intercept Rescue", fontweight="bold")
    ax.legend(loc="lower right", fontsize=7.3, frameon=True, framealpha=0.92)
    ax.grid(alpha=0.25)

    plt.tight_layout(pad=1.35)
    out = MANUSCRIPT_DIR / "fig2_sepsis_transfer_afa_and_landmarks.png"
    plt.savefig(out, dpi=300)
    plt.close(fig)
    return out


def plot_fig3(res: dict) -> Path:
    b1b_cal = res["sepsis_track_b1b_primary"]["calibration"]
    b4 = res["yale_strengthened"]["yale_b4_cross_campus_primary"]
    b3 = res["yale_strengthened"]["yale_b3_fairness_blinded"]
    ya_low = res["archived_reference_summaries"]["yale_track_a_esi45_parity"]["arms"]
    ya_full = res["archived_reference_summaries"]["yale_track_a_full_560k_parity"]["arms"]

    fig = plt.figure(figsize=(14.6, 9.8), dpi=300)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.12], hspace=0.32, wspace=0.24)

    # Panel A: Flexible LOESS Calibration Curves Across Sepsis B1b & Yale Cross-Campus B4a/B4b
    ax_a = fig.add_subplot(gs[0, 0])
    c_emory = b1b_cal["emory_cz_fluid"]
    c_b4a = b4["b4a_esi45_cross_campus"]["calibration"]["cz_fluid"]
    c_b4b = b4["b4b_allcomers_cross_campus"]["calibration"]["cz_fluid"]

    ax_a.plot([0, 95], [0, 95], ":", lw=1.5, color="#0F172A", label="Ideal 45-deg Calibration")
    ax_a.plot(
        np.array(c_b4b["curve_p"]) * 100,
        np.array(c_b4b["curve_y_loess"]) * 100,
        "-",
        lw=2.3,
        color="#059669",
        label=f"Yale B4b Ext. All-Comers (CITL={c_b4b['citl']:+.2f}, Slope={c_b4b['cal_slope']:.2f}, ICI={c_b4b['ici_loess']:.3f})",
    )
    ax_a.plot(
        np.array(c_b4a["curve_p"]) * 100,
        np.array(c_b4a["curve_y_loess"]) * 100,
        "-.",
        lw=2.2,
        color="#7C3AED",
        label=f"Yale B4a Ext. ESI 4-5 (CITL={c_b4a['citl']:+.2f}, Slope={c_b4a['cal_slope']:.2f}, ICI={c_b4a['ici_loess']:.3f})",
    )
    ax_a.plot(
        np.array(c_emory["curve_p"]) * 100,
        np.array(c_emory["curve_y_loess"]) * 100,
        "--",
        lw=2.1,
        color="#0284C7",
        label=f"Emory B1b Ext. Sepsis (CITL={c_emory['citl']:+.2f}, Slope={c_emory['cal_slope']:.2f}, ICI={c_emory['ici_loess']:.3f})",
    )
    ax_a.set_xlim(0, 92)
    ax_a.set_ylim(0, 92)
    ax_a.set_xlabel("Predicted Probability (%)")
    ax_a.set_ylabel("LOESS-Smoothed Observed Event Rate (%)")
    ax_a.set_title("A. Flexible External Calibration Curves (Emory B1b & Yale B4)", fontweight="bold")
    ax_a.legend(loc="upper left", fontsize=7.4, frameon=True, framealpha=0.92)
    ax_a.grid(alpha=0.25)

    # Panel B: Yale ED All 5 Cohorts — Delta NB Gain Over Stage-1 Triage (Primary B4a/B4b vs Reference B3/Track A)
    ax_b = fig.add_subplot(gs[0, 1])
    yale_rows = [
        (
            "Primary B4a:\nCross-Campus\nESI 4-5 (51k)",
            b4["b4a_esi45_cross_campus"]["external_per_split"],
            b4["b4a_esi45_cross_campus"]["external_symmetric_ensemble"],
        ),
        (
            "Primary B4b:\nCross-Campus\nAll-Comers (166k)",
            b4["b4b_allcomers_cross_campus"]["external_per_split"],
            b4["b4b_allcomers_cross_campus"]["external_symmetric_ensemble"],
        ),
        (
            "Ref. B3:\nBlinded ESI 4-5\n(153k)",
            b3["per_split_arms"],
            None,
        ),
        (
            "Ref. Track A:\nParity ESI 4-5\n(153k)",
            ya_low,
            None,
        ),
        (
            "Ref. Track A:\nParity Full\n(560k)",
            ya_full,
            None,
        ),
    ]
    xb = np.arange(len(yale_rows))
    bw = 0.25

    def _gain_and_ci(arms_d: dict, arm_key: str) -> tuple[float, float]:
        if arm_key not in arms_d:
            return np.nan, 0.0
        b0 = arms_d["paper1_global_cmi_global_s2"]["nb_base_0pct"]
        m_a = arms_d[arm_key]["nb_voi_auto"]
        diff = float(m_a["mean"]) - float(b0["mean"])
        # Conservative SE propagation on per-split summary
        se = float(np.hypot(m_a["se"], b0["se"]))
        return diff, 1.96 * se

    gl_gains = [_gain_and_ci(r[1], "paper1_global_cmi_global_s2") for r in yale_rows]
    gf_gains = [
        _gain_and_ci(r[1], "ablation_global_cmi_fluid_s2")
        if "ablation_global_cmi_fluid_s2" in r[1]
        else _gain_and_ci(r[1], "paper1_global_cmi_global_s2")
        for r in yale_rows
    ]
    cz_gains = [_gain_and_ci(r[1], "paper2_cz_disc_fluid_s2") for r in yale_rows]

    ax_b.bar(
        xb - bw,
        [g for g, _ in gl_gains],
        yerr=[c for _, c in gl_gains],
        capsize=3,
        width=bw,
        color="#3B82F6",
        edgecolor="#334155",
        lw=0.6,
        label="Global-CMI + Global S2",
    )
    ax_b.bar(
        xb,
        [g for g, _ in gf_gains],
        yerr=[c for _, c in gf_gains],
        capsize=3,
        width=bw,
        color="#0284C7",
        edgecolor="#334155",
        lw=0.6,
        label="Global-CMI + Fluid S2",
    )
    ax_b.bar(
        xb + bw,
        [g for g, _ in cz_gains],
        yerr=[c for _, c in cz_gains],
        capsize=3,
        width=bw,
        color="#059669",
        edgecolor="#064E3B",
        lw=0.6,
        label="CZ-Fluid (Proposed)",
    )
    # Mark Symmetric Ensemble Delta NB on Primary B4a and B4b
    for idx_e in (0, 1):
        ens_d = yale_rows[idx_e][2]
        assert ens_d is not None
        b0_e = ens_d["stage1_base"]["ens_nb_auto"]
        ax_b.scatter(
            [idx_e - bw, idx_e, idx_e + bw],
            [
                ens_d["paper1_global_cmi_global_s2"]["ens_nb_auto"] - b0_e,
                ens_d["ablation_global_cmi_fluid_s2"]["ens_nb_auto"] - b0_e,
                ens_d["paper2_cz_disc_fluid_s2"]["ens_nb_auto"] - b0_e,
            ],
            marker="D",
            s=36,
            color="#0F172A",
            zorder=6,
            label="Primary B4 Ensemble Gain" if idx_e == 0 else None,
        )
    ax_b.axvline(1.5, color="#64748B", ls="--", lw=1.1)
    bt_b4b = b4["b4b_allcomers_cross_campus"]["paired_bootstraps_10k"]["cz_fluid_vs_global_cmi"]
    bt_b4a = b4["b4a_esi45_cross_campus"]["paired_bootstraps_10k"]["cz_fluid_vs_global_cmi"]
    ax_b.text(
        0.02,
        0.97,
        f"B4b Ext. All-Comers (166k): {bt_b4b['diff_mean']:+.2f} vs Glob-CMI (p={bt_b4b['p_two_sided_dh']:.3f})\n"
        f"B4a Ext. ESI 4-5 (51k): {bt_b4a['diff_mean']:+.2f} vs Glob-CMI (p={bt_b4a['p_two_sided_dh']:.3f})",
        transform=ax_b.transAxes,
        ha="left",
        va="top",
        fontsize=7.6,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "#F8FAFC", "edgecolor": "#CBD5E1"},
    )
    ax_b.set_xticks(xb)
    ax_b.set_xticklabels([r[0] for r in yale_rows], fontsize=7.6)
    ax_b.set_ylim(0, 8.25)
    ax_b.set_ylabel("Delta Net Benefit Over Stage-1 Triage (/1,000)")
    ax_b.set_title("B. All 5 Yale ED Benchmarks: Primary Cross-Campus B4 vs. Reference", fontweight="bold")
    ax_b.legend(loc="upper right", fontsize=7.3, frameon=True, framealpha=0.92)
    ax_b.grid(axis="y", alpha=0.25)

    # Panel C (Bottom Full-Width Dual Forest Plot): All 11 Fairness-Blinded Subgroups — Delta NB + CITL with 95% CIs
    gs_bot = gs[1, :].subgridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.14)
    ax_c1 = fig.add_subplot(gs_bot[0, 0])
    ax_c2 = fig.add_subplot(gs_bot[0, 1], sharey=ax_c1)

    sub_dict = b3["fairness_subgroups_11_with_cis_and_citl"]
    sub_b4a = b4["b4a_esi45_cross_campus"]["external_campus_b_subgroup_audit"]
    label_map = {
        "lang:English": "Language: English",
        "lang:Other": "Language: Non-English",
        "race:White or Caucasian": "Race: White / Caucasian",
        "race:Black or African American": "Race: Black / African American",
        "race:Other": "Race: Other",
        "race:Asian": "Race: Asian",
        "race:Patient Refused": "Race: Patient Refused",
        "insurance_status:Medicaid": "Insurance: Medicaid",
        "insurance_status:Commercial": "Insurance: Commercial",
        "insurance_status:Medicare": "Insurance: Medicare",
        "insurance_status:Other": "Insurance: Other / Self-Pay",
    }
    sorted_subs = sorted(sub_dict.items(), key=lambda kv: float(kv[1]["oof_ens_delta_nb"]), reverse=True)
    y_pos = np.arange(len(sorted_subs))
    names = [f"{label_map.get(k, k)} (n={v['oof_unique_n']:,})" for k, v in sorted_subs]

    d_means = np.array([float(v["oof_ens_delta_nb"]) for _, v in sorted_subs])
    d_lo = np.array([float(v["oof_ens_delta_ci95_low"]) for _, v in sorted_subs])
    d_hi = np.array([float(v["oof_ens_delta_ci95_high"]) for _, v in sorted_subs])
    d_err = np.vstack([np.maximum(0, d_means - d_lo), np.maximum(0, d_hi - d_means)])

    ax_c1.barh(
        y_pos,
        d_means,
        xerr=d_err,
        capsize=3,
        height=0.62,
        color="#059669",
        alpha=0.88,
        edgecolor="#064E3B",
        lw=0.6,
        label="Track B3 OOF Ensemble Delta NB (95% CI)",
    )
    # Overlay Campus B external subgroup gains where n >= 250
    ext_y, ext_d = [], []
    for yi, (k, _) in enumerate(sorted_subs):
        if k in sub_b4a:
            ext_y.append(yi)
            ext_d.append(float(sub_b4a[k]["delta_nb"]))
    if ext_y:
        ax_c1.scatter(ext_d, ext_y, marker="D", s=34, color="#0F172A", zorder=6, label="Primary B4a Campus B External Gain")

    ax_c1.axvline(0.0, color="#DC2626", ls="--", lw=1.2)
    ax_c1.set_yticks(y_pos)
    ax_c1.set_yticklabels(names, fontsize=8.2)
    ax_c1.invert_yaxis()
    ax_c1.set_xlabel("Subgroup Net Benefit Gain Over Stage-1 Triage (/1,000 Visits, 95% CI)")
    ax_c1.set_title("C1. Fairness-Blinded Subgroup Net Benefit Gain (11/11 Positive; 8/11 95% CIs > 0)", fontweight="bold")
    ax_c1.legend(loc="lower right", fontsize=7.6, frameon=True, framealpha=0.92)
    ax_c1.grid(axis="x", alpha=0.25)

    citl_m = np.array([float(v["citl_cz_fluid"]) for _, v in sorted_subs])
    citl_lo = np.array([float(v["citl_ci95"][0]) for _, v in sorted_subs])
    citl_hi = np.array([float(v["citl_ci95"][1]) for _, v in sorted_subs])
    citl_err = np.vstack([np.maximum(0, citl_m - citl_lo), np.maximum(0, citl_hi - citl_m)])

    ax_c2.errorbar(
        citl_m,
        y_pos,
        xerr=citl_err,
        fmt="o",
        ms=5.5,
        lw=1.7,
        capsize=3,
        color="#0284C7",
        label="Subgroup Calibration-in-the-Large (95% CI)",
    )
    ax_c2.axvline(0.0, color="#0F172A", ls=":", lw=1.4, label="Ideal CITL = 0.0")
    plt.setp(ax_c2.get_yticklabels(), visible=False)
    ax_c2.set_xlabel("Subgroup Calibration-in-the-Large (Log-Odds Intercept Offset)")
    ax_c2.set_title("C2. Subgroup Calibration-in-the-Large (CITL)", fontweight="bold")
    ax_c2.legend(loc="lower right", fontsize=7.6, frameon=True, framealpha=0.92)
    ax_c2.grid(axis="x", alpha=0.25)

    out = MANUSCRIPT_DIR / "fig3_calibration_and_yale_transfer.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    MANUSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    res_path = REPO_ROOT / "results" / "paper2_v2_strengthening_results.json"
    res = json.loads(res_path.read_text())
    f1 = plot_fig1(res)
    f2 = plot_fig2(res)
    f3 = plot_fig3(res)
    print("Generated 300-DPI figures:\n ", f1, "\n ", f2, "\n ", f3)


if __name__ == "__main__":
    main()
