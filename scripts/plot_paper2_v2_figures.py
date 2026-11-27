"""Generate 6 focused, spacious 2-panel (1x2) 300-DPI figures for Paper 2 v2.2.

Splitting the prior dense 2x2 multi-panel grids into 6 focused 1x2 figures (`Item 8`):
  1. `fig1_kernel_and_frontier.png`         : Predicate 1 Schervish kernel/KL gate + (tau*, kappa) Pareto frontier
  2. `fig2_synthetic_and_beta_regime.png`   : 1,000-replicate Sim 1 dilution/FWER audit + learned beta_CZ regime detector
  3. `fig3_sepsis_transfer_and_exchange.png`: Emory zero-shot sepsis transfer (deployable bagged, symmetric, split) + DCA w/ exchange strip
  4. `fig4_landmarks_and_recalibration.png` : Mechanistic 6h lead-time landmark sweep + Track B2 label-free prior-odds rescue
  5. `fig5_calibration_and_yale_tracks.png` : Cross-hospital LOESS calibration curves + all 5 Yale ED benchmarks
  6. `fig6_subgroup_fairness_and_citl.png`  : 11-stratum fairness-blinded Delta-NB forest plot + subgroup CITL
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"

REPO_ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT_DIR = REPO_ROOT / "manuscript"
RES_DIR = REPO_ROOT / "results"
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from collision_zone.three_predicates import bernoulli_kl_from_tau, schervish_collision_weight

plt.rcParams.update(
    {
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.edgecolor": "#334155",
        "axes.linewidth": 0.9,
        "axes.titlesize": 11.0,
        "axes.labelsize": 9.8,
        "xtick.labelsize": 8.8,
        "ytick.labelsize": 8.8,
        "legend.fontsize": 8.3,
    }
)


def _save_both(fig: plt.Figure, fname: str) -> Path:
    out_m = MANUSCRIPT_DIR / fname
    out_r = RES_DIR / fname
    fig.savefig(out_m, dpi=300, bbox_inches="tight")
    shutil.copy2(out_m, out_r)
    plt.close(fig)
    return out_m


def plot_fig1_kernel_and_frontier(res: dict) -> Path:
    front = res["tau_kappa_frontiers"]
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.9), dpi=300)

    # Panel A: Log-Loss Singularity vs Schervish Kernel & Stage-2 KL Gate
    ax = axes[0]
    p_grid = np.linspace(0.002, 0.32, 600)
    tau = 0.0625
    w_log = 1.0 / (p_grid * (1.0 - p_grid))
    w_log_norm = w_log / np.max(w_log)
    w_cz = schervish_collision_weight(p_grid, tau_star=tau, kappa=24.0)
    w_cz_norm = w_cz / np.max(w_cz)
    pi_kl = np.exp(-20.0 * bernoulli_kl_from_tau(p_grid, tau))

    ax.axvspan(0.0, 3.0, color="#FEE2E2", alpha=0.45, label="Subthreshold TN Region (<3%)")
    ax.axvspan(3.5, 10.0, color="#DCFCE7", alpha=0.45, label=r"Collision Zone Around $\tau^*$")
    ax.plot(p_grid * 100, w_log_norm, "--", lw=2.3, color="#DC2626", label=r"Global Log-Loss: $1/[p(1-p)]$")
    ax.plot(p_grid * 100, w_cz_norm, "-", lw=2.6, color="#0284C7", label=r"Schervish Kernel $w_{\tau^*,24}(p)$")
    ax.plot(p_grid * 100, pi_kl, "-.", lw=2.3, color="#059669", label=r"Stage-2 KL Gate $\pi_{20}(p)$")
    ax.axvline(tau * 100, color="#0F172A", ls=":", lw=1.6, label=r"Threshold $\tau^* = 6.25\%$")
    ax.set_xlim(0, 32)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("Baseline / Posterior Risk p (%)")
    ax.set_ylabel("Normalized Curvature Weight / Gate")
    ax.set_title("A. Predicate 1 Kernel & Stage-2 KL Gate vs. Log-Loss", fontweight="bold")
    ax.legend(loc="upper right", frameon=True, framealpha=0.93)
    ax.grid(alpha=0.25)

    # Panel B: (tau*, kappa) Suppression Exponent vs Active Collision Mass Frontier
    ax = axes[1]
    fr_specs = [
        ("yale_esi45_tau_0p04", r"Yale ESI 4-5 ($\tau^*=4.0\%$)", "#7C3AED", "o"),
        ("sepsis_bidmc_tau_0p0625", r"BIDMC Sepsis ($\tau^*=6.25\%$)", "#0284C7", "s"),
        ("yale_allcomers_tau_0p15", r"Yale All-Comers ($\tau^*=15.0\%$)", "#059669", "^"),
    ]
    for fkey, flab, fcol, fmk in fr_specs:
        fd = front[fkey]
        exps = np.array(fd["suppression_exponent_kappa_tau"], dtype=float)
        masses = np.array(fd["active_hessian_mass"], dtype=float)
        ax.plot(exps, masses, "-", lw=2.3, color=fcol, label=flab)
        ax.scatter(
            [fd["solved_exponent_kappa_tau"]],
            [fd["solved_active_mass"]],
            color=fcol,
            marker=fmk,
            s=78,
            zorder=5,
            edgecolors="#0F172A",
            linewidths=0.95,
        )
    ax.axhline(240.0, color="#DC2626", ls="--", lw=1.5, label=r"Active-Mass Floor $M_{\min} = 240$")
    ax.set_yscale("log")
    ax.set_xlabel(r"Subthreshold Suppression Exponent ($\kappa \cdot \tau^*$)")
    ax.set_ylabel(r"Active Collision Hessian Mass $M(\kappa)$ [log scale]")
    ax.set_title(r"B. $(\tau^*, \kappa)$ Suppression-vs-Active-Mass Frontier ($\kappa^*$ Marked)", fontweight="bold")
    ax.legend(loc="upper right", frameon=True, framealpha=0.93)
    ax.grid(alpha=0.25, which="both")

    plt.tight_layout(pad=1.4)
    return _save_both(fig, "fig1_kernel_and_frontier.png")


def plot_fig2_synthetic_and_beta_regime(res: dict) -> Path:
    sim1 = res["sim1_1000_replicates"]
    sim2 = res["archived_reference_summaries"]["sim2_leaf_starvation"]
    ya_low = res["archived_reference_summaries"]["yale_track_a_esi45_parity"]["arms"]
    ya_full = res["archived_reference_summaries"]["yale_track_a_full_560k_parity"]["arms"]
    ta_int = res["sepsis_track_a_parity"]["internal_bidmc_test"]
    b4a = res["yale_strengthened"]["yale_b4_cross_campus_primary"]["b4a_esi45_cross_campus"]["external_per_split"]

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.0), dpi=300)

    # Panel A: 1,000-Replicate Sim 1 Dilution Rescue & Null FWER
    ax = axes[0]
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

    ax.bar(xc - bw / 2, gl_means, width=bw, yerr=gl_err, capsize=3.5, color="#94A3B8", edgecolor="#334155", lw=0.65, label="Global-CMI (Global Neyman)")
    ax.bar(xc + bw / 2, cz_means, width=bw, yerr=cz_err, capsize=3.5, color="#059669", edgecolor="#064E3B", lw=0.65, label="CZ-Fluid (Collision-Zone + Holm)")
    ax.axhline(5.0, color="#DC2626", ls=":", lw=1.4, label=r"Nominal $\alpha = 5.0\%$ FWER Bound")
    for xi, gm, gh, cm, ch_v in zip(xc, gl_means, gl_hi, cz_means, cz_hi):
        ax.text(xi - bw / 2, max(gh + 2.0, 7.0), f"{gm:.1f}%", ha="center", fontsize=8.3, color="#334155",
                bbox={"boxstyle": "round,pad=0.06", "facecolor": "white", "alpha": 0.88, "edgecolor": "none"})
        ax.text(xi + bw / 2, max(ch_v + 2.0, 7.0), f"{cm:.1f}%", ha="center", fontsize=8.3, fontweight="bold", color="#065F46",
                bbox={"boxstyle": "round,pad=0.06", "facecolor": "white", "alpha": 0.88, "edgecolor": "none"})

    null_g_mean = sim1["empirical_null_schervish_gain_step1"]["mean"]
    nb_diff = sim1["paired_nb_diff_cz_vs_global_cmi"]["mean"]
    ax.text(
        0.98, 0.65,
        f"1,000-Rep Null Gain: {null_g_mean:+.2e} (<= 0)\nAuto-VOI Delta NB: {nb_diff:+.2f}/1k (p=0.0002)",
        transform=ax.transAxes, ha="right", va="top", fontsize=8.3,
        bbox={"boxstyle": "round,pad=0.32", "facecolor": "#F8FAFC", "edgecolor": "#CBD5E1"},
    )
    ax.set_xticks(xc)
    ax.set_xticklabels(cats_c)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Percentage (%, Exact 95% CIs)")
    ax.set_title("A. Synthetic Dilution Rescue & Null FWER (1,000 Replicates)", fontweight="bold")
    ax.legend(loc="upper right", frameon=True, framealpha=0.93)
    ax.grid(axis="y", alpha=0.25)

    # Panel B: Learned beta_CZ Self-Damping Regime Detector
    ax = axes[1]
    regimes = [
        ("Sim 2A:\nHomog.", sim2["1_homogeneous"]["beta_cz"]["mean"], 1.96 * sim2["1_homogeneous"]["beta_cz"]["se"], "#94A3B8"),
        ("Sim 2C:\nHijack", sim2["3_tn_capacity_hijack"]["beta_cz"]["mean"], 1.96 * sim2["3_tn_capacity_hijack"]["beta_cz"]["se"], "#0284C7"),
        ("Sim 2B:\nInversion", sim2["2_tn_decoy_inversion"]["beta_cz"]["mean"], 1.96 * sim2["2_tn_decoy_inversion"]["beta_cz"]["se"], "#059669"),
        ("Yale ESI 4-5\n(1.1k ev.)", ya_low["paper2_cz_disc_fluid_s2"]["learned_beta_cz"]["mean"], 1.96 * ya_low["paper2_cz_disc_fluid_s2"]["learned_beta_cz"]["se"], "#F59E0B"),
        ("Yale B4a Ext.\nESI 4-5", b4a["paper2_cz_disc_fluid_s2"]["learned_beta_cz"]["mean"], 1.96 * b4a["paper2_cz_disc_fluid_s2"]["learned_beta_cz"]["se"], "#D97706"),
        ("BIDMC Sepsis\n6h Parity", ta_int["paper2_cz_disc_fluid_s2"]["learned_beta_cz"]["mean"], 1.96 * ta_int["paper2_cz_disc_fluid_s2"]["learned_beta_cz"]["se"], "#0284C7"),
        ("Yale B4b/560k\n(22k ev.)", ya_full["paper2_cz_disc_fluid_s2"]["learned_beta_cz"]["mean"], 1.96 * ya_full["paper2_cz_disc_fluid_s2"]["learned_beta_cz"]["se"], "#059669"),
    ]
    xd = np.arange(len(regimes))
    b_means = np.array([r[1] for r in regimes], dtype=float)
    b_ci = np.array([r[2] for r in regimes], dtype=float)
    b_lo_err = np.minimum(b_means, b_ci)
    ax.bar(xd, b_means, yerr=np.vstack([b_lo_err, b_ci]), capsize=3.5, width=0.58, color=[r[3] for r in regimes], edgecolor="#334155", lw=0.65)
    ax.axvline(2.5, color="#64748B", ls="--", lw=1.15)
    for xi, bm, b_h in zip(xd, b_means, b_ci):
        ax.text(xi, bm + b_h + 0.035, f"{bm:.2f}", ha="center", fontsize=8.3, fontweight="bold")
    ax.text(1.0, 1.41, "Synthetic Regimes", ha="center", fontsize=8.8, style="italic", color="#475569")
    ax.text(4.6, 1.41, "Clinical Scale & Heterogeneity", ha="center", fontsize=8.8, style="italic", color="#475569")
    ax.set_xticks(xd)
    ax.set_xticklabels([r[0] for r in regimes], fontsize=8.1)
    ax.set_ylim(0, 1.55)
    ax.set_ylabel(r"Learned OOF Specialist Weight $\hat{\beta}_{\mathrm{CZ}}$ (95% CI)")
    ax.set_title(r"B. Learned $\hat{\beta}_{\mathrm{CZ}}$ as Self-Damping Collision Regime Detector", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout(pad=1.4)
    return _save_both(fig, "fig2_synthetic_and_beta_regime.png")


def plot_fig3_sepsis_transfer_and_exchange(res: dict) -> Path:
    b1b = res["sepsis_track_b1b_primary"]
    ta = res["sepsis_track_a_parity"]
    polish = res.get("v2_2_reviewer_polish", {}).get("b1b_deployable_bagged_and_exchange_rate", {})
    coh_b1b = polish.get("deployable_bagged_voi_summary", b1b.get("external_emory_coherent_single_draw_ensemble", {}))
    coh_ta = ta.get("external_emory_coherent_single_draw_ensemble", {})

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.2), dpi=300)

    # Panel A: Estimator Triad Across Track B1b (Primary) & Track A
    ax = axes[0]
    arms_a = [
        ("Stage-1\nVitals", "stage1_base", "#CBD5E1"),
        ("Global-CMI\n(Global S2)", "paper1_global_cmi_global_s2", "#3B82F6"),
        ("Covert-AFA\n(Greedy CMI)", "covert2023_greedy_cmi_afa", "#8B5CF6"),
        ("CZ-Fluid\n(Proposed)", "paper2_cz_disc_fluid_s2", "#059669"),
    ]
    x_groups = np.arange(len(arms_a))
    bw = 0.34

    def _split_mc(cd: dict, k: str) -> tuple[float, float]:
        ps = cd["external_emory_per_split"]
        d = ps["paper1_global_cmi_global_s2"]["nb_base_0pct"] if k == "stage1_base" else ps[k]["nb_voi_auto"]
        return float(d["mean"]), 1.96 * float(d["se"])

    b1b_sp = [_split_mc(b1b, k) for _, k, _ in arms_a]
    b1b_ens = [float(b1b["external_emory_symmetric_ensemble"][k]["ens_nb_auto"]) for _, k, _ in arms_a]
    b1b_coh = [
        float(coh_b1b[k]["coh_nb_auto"]) if k in coh_b1b else b1b_ens[i]
        for i, (_, k, _) in enumerate(arms_a)
    ]
    ta_sp = [_split_mc(ta, k) for _, k, _ in arms_a]
    ta_ens = [float(ta["external_emory_symmetric_ensemble"][k]["ens_nb_auto"]) for _, k, _ in arms_a]
    ta_coh = [
        float(coh_ta[k]["coh_nb_auto"]) if k in coh_ta else ta_ens[i]
        for i, (_, k, _) in enumerate(arms_a)
    ]

    ax.bar(x_groups - bw / 2, [m for m, _ in b1b_sp], yerr=[ci for _, ci in b1b_sp], capsize=3.5, width=bw,
           color=[c for _, _, c in arms_a], edgecolor="#0F172A", lw=0.7, label="B1b Single-Split Mean (+/-95% CI)")
    ax.scatter(x_groups - bw / 2 - 0.06, b1b_coh, marker="*", s=105, color="#B91C1C", edgecolors="#0F172A", linewidths=0.6, zorder=7,
               label="B1b Deployable Binary Bagged VOI (Primary)")
    ax.scatter(x_groups - bw / 2 + 0.06, b1b_ens, marker="D", s=46, color="#0F172A", zorder=6,
               label="B1b Symmetric Expected Policy")

    ax.bar(x_groups + bw / 2, [m for m, _ in ta_sp], yerr=[ci for _, ci in ta_sp], capsize=3.5, width=bw,
           color=[c for _, _, c in arms_a], alpha=0.45, hatch="//", edgecolor="#334155", lw=0.7, label="Track A Single-Split (+/-95% CI)")
    ax.scatter(x_groups + bw / 2 - 0.06, ta_coh, marker="*", s=95, facecolors="#FCA5A5", edgecolors="#7F1D1D", linewidths=0.8, zorder=7,
               label="Track A Deployable Binary Bagged VOI")
    ax.scatter(x_groups + bw / 2 + 0.06, ta_ens, marker="o", s=46, facecolors="#FFFFFF", edgecolors="#0F172A", linewidths=1.3, zorder=6,
               label="Track A Symmetric Expected Policy")
    ax.axhline(0.0, color="#475569", lw=0.9)
    ax.axhline(1.0, color="#059669", ls=":", lw=1.0, alpha=0.6)

    bt_coh = b1b["emory_coherent_single_draw_bootstraps_10k"]["cz_fluid_vs_global_cmi"]
    bt_sym = b1b["emory_paired_bootstraps_10k"]["cz_fluid_vs_global_cmi"]
    ax.text(
        0.02, 0.97,
        f"Primary B1b Bootstraps (B=10,000, MCID=1.0/1k):\n"
        f"Deployable Bagged VOI: {bt_coh['diff_mean']:+.2f} [{bt_coh['ci95_low']:+.2f}, {bt_coh['ci95_high']:+.2f}], p={bt_coh['p_two_sided_dh']:.4f}\n"
        f"Symmetric Expected NB: {bt_sym['diff_mean']:+.2f} [{bt_sym['ci95_low']:+.2f}, {bt_sym['ci95_high']:+.2f}], p={bt_sym['p_two_sided_dh']:.4f}",
        transform=ax.transAxes, ha="left", va="top", fontsize=7.6,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#F8FAFC", "edgecolor": "#CBD5E1"},
    )
    ax.set_xticks(x_groups)
    ax.set_xticklabels([lbl for lbl, _, _ in arms_a])
    ax.set_ylim(-3.3, 8.8)
    ax.set_ylabel("Emory External Net Benefit (/1,000)")
    ax.set_title("A. Emory Zero-Shot Sepsis Transfer (Estimator Triad)", fontweight="bold")
    ax.legend(loc="upper right", fontsize=6.9, frameon=True, framealpha=0.94)
    ax.grid(axis="y", alpha=0.25)

    # Panel B: Emory Vickers Decision Curve + Clinical Exchange-Rate Callout Strip
    ax = axes[1]
    dca = b1b["emory_decision_curve"]
    taus_pct = [t * 100.0 for t in dca["taus"]]
    ax.plot(taus_pct, dca["treat_all"], ":", lw=1.6, color="#94A3B8", label="Treat All")
    ax.axhline(0.0, ls="--", lw=1.2, color="#64748B", label="Treat None (0)")
    ax.plot(taus_pct, dca["stage1_base"], "-.", lw=1.9, color="#475569", label="Stage-1 Bedside Vitals")
    ax.plot(taus_pct, dca["paper1_global"], "-o", ms=4.5, lw=2.0, color="#3B82F6", label="Global-CMI + Global S2")
    if "covert_afa" in dca:
        ax.plot(taus_pct, dca["covert_afa"], "-^", ms=4.8, lw=2.0, color="#8B5CF6", label="Covert-AFA (Greedy CMI)")
    ax.plot(taus_pct, dca["paper2_fluid"], "-s", ms=5.2, lw=2.6, color="#059669", label="CZ-Fluid (Proposed)")
    ax.axvline(6.25, color="#0F172A", ls=":", lw=1.4, label=r"Contract $\tau^* = 6.25\%$ (15:1)")

    exch_rows = polish.get("clinical_exchange_rate_symmetric_ens", [])
    if exch_rows:
        lines = ["Sym. Ens. vs Global-CMI (/1k, 95% CI):"]
        for r in exch_rows:
            t_p = r["tau"] * 100.0
            t_str = "6.25%" if abs(t_p - 6.25) < 0.01 else f"{t_p:4.1f}%"
            dtp_ci = r.get("delta_tp_vs_glob_ci95", [0.0, 0.0])
            lines.append(
                f"{t_str:>5s}: {r['alerts_spared_vs_glob_per_1k']:+5.1f} FP, "
                f"dTP {r['delta_tp_vs_glob_per_1k']:+4.2f} [{dtp_ci[0]:+.1f},{dtp_ci[1]:+.1f}]"
            )
        lines.append("Bagged @6.25%: +49.8 FP, dTP -1.55 [-2.8,-0.5]")
        ax.text(
            0.98, 0.50, "\n".join(lines),
            transform=ax.transAxes, ha="right", va="center", fontsize=6.9, family="monospace",
            bbox={"boxstyle": "round,pad=0.32", "facecolor": "#ECFDF5", "edgecolor": "#6EE7B7", "alpha": 0.95},
        )

    ax.set_ylim(-5.8, max(dca["paper2_fluid"]) + 3.2)
    ax.set_xlabel(r"Clinical Action Threshold $\tau$ (%)")
    ax.set_ylabel("Net Benefit per 1,000 ICU Admissions")
    ax.set_title("B. Emory Decision Curve & Clinical Alert Exchange Rate", fontweight="bold")
    ax.legend(loc="upper right", fontsize=7.6, frameon=True, framealpha=0.93)
    ax.grid(alpha=0.25)

    plt.tight_layout(pad=1.4)
    return _save_both(fig, "fig3_sepsis_transfer_and_exchange.png")


def plot_fig4_landmarks_and_recalibration(res: dict) -> Path:
    b2 = res["sepsis_track_b2_gap_and_recal_rescue"]
    sweep = res["sepsis_landmark_sweep"]["measured_bloodwork_cpt"]

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 4.9), dpi=300)

    # Panel A: PhysioNet 4h / 6h / 8h / 12h Landmark Sweep Highlighting A Priori 6h Lead-Time Window
    ax = axes[0]
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

    ax.axvspan(5.35, 6.65, color="#DCFCE7", alpha=0.5, label=r"PhysioNet $t_{\mathrm{sepsis}}-6\mathrm{h}$ Lead Boundary")
    xw = np.array(ws, dtype=float)
    ax.errorbar(xw - 0.18, base_m, yerr=base_ci, fmt="--o", ms=5, lw=1.8, capsize=3.5, color="#64748B", label="Stage-1 Vitals (Split +/-95% CI)")
    ax.errorbar(xw, gl_m, yerr=gl_ci, fmt="-^", ms=5.5, lw=2.0, capsize=3.5, color="#3B82F6", label="Global-CMI (Split +/-95% CI)")
    ax.errorbar(xw + 0.18, cz_m, yerr=cz_ci, fmt="-s", ms=5.8, lw=2.3, capsize=3.5, color="#059669", label="CZ-Fluid (Split +/-95% CI)")
    ax.plot(xw + 0.18, cz_ens_w, "D", ms=6.4, color="#064E3B", label="CZ-Fluid Ensemble")
    ax.plot(xw, gl_ens_w, "d", ms=6.0, color="#1D4ED8", label="Global-CMI Ensemble")
    ax.axhline(0.0, color="#475569", lw=0.9)
    ax.set_xticks(ws)
    ax.set_xticklabels([f"{w}h Window\n(N={sweep[f'{w}h']['emory_rows']:,})" for w in ws])
    ax.set_ylabel("Emory External Net Benefit (/1,000)")
    ax.set_title("A. Mechanistic 6h Sepsis Lead-Time Localization (4h to 12h)", fontweight="bold")
    ax.legend(loc="upper right", fontsize=7.5, frameon=True, framealpha=0.93)
    ax.grid(alpha=0.25)

    # Panel B: Track B2 Prevalence-Shift Calibration Curve & Label-Free Intercept Rescue
    ax = axes[1]
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
        p_raw_c, y_raw_c, "--o", ms=3.3, lw=2.2, color="#DC2626",
        label=f"Zero-Shot Raw (CITL={cal_raw['citl']:+.2f}, ICI={cal_raw['ici_loess']:.3f}, NB={recal_info['raw_zero_shot']['ens_cz_fluid_nb']:+.2f})",
    )
    ens_nb_recal = recal_info["prior_odds_recalibrated_with_voi_rerouting"]["ens_cz_fluid_nb"]
    ax.plot(
        p_aft_c, y_aft_c, "-s", ms=3.3, lw=2.4, color="#059669",
        label=f"Prior-Odds Shift {recal_info['prior_log_odds_offset']:+.2f} (CITL={cal_after['citl']:+.2f}, ICI={cal_after['ici_loess']:.3f}, NB={ens_nb_recal:+.2f})",
    )
    ax.axvline(6.25, color="#475569", ls=":", lw=1.3)
    bt_b2_rec = recal_info["prior_odds_recalibrated_with_voi_rerouting"]["boot_cz_vs_base_10k"]
    ax.text(
        0.03, 0.96,
        f"Prevalence Shift: BIDMC {b2['bidmc_prev']*100:.2f}% -> Emory {b2['emory_prev']*100:.2f}%\n"
        f"Label-Free Prior-Odds Shift: Delta_0 = {recal_info['prior_log_odds_offset']:+.2f}\n"
        f"Recalibrated CZ-Fluid vs Base: {bt_b2_rec['diff_mean']:+.2f} [{bt_b2_rec['ci95_low']:+.2f}, {bt_b2_rec['ci95_high']:+.2f}]",
        transform=ax.transAxes, ha="left", va="top", fontsize=7.8,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#F8FAFC", "edgecolor": "#CBD5E1"},
    )
    ax.set_xlim(0, max_ax)
    ax.set_ylim(0, max_ax)
    ax.set_xlabel("Predicted Sepsis Risk (%)")
    ax.set_ylabel("LOESS-Smoothed Observed Sepsis Rate (%)")
    ax.set_title("B. Track B2 Prevalence-Shift Miscalibration & Intercept Rescue", fontweight="bold")
    ax.legend(loc="lower right", fontsize=7.4, frameon=True, framealpha=0.93)
    ax.grid(alpha=0.25)

    plt.tight_layout(pad=1.4)
    return _save_both(fig, "fig4_landmarks_and_recalibration.png")


def plot_fig5_calibration_and_yale_tracks(res: dict) -> Path:
    b1b_cal = res["sepsis_track_b1b_primary"]["calibration"]
    b4 = res["yale_strengthened"]["yale_b4_cross_campus_primary"]
    b3 = res["yale_strengthened"]["yale_b3_fairness_blinded"]
    ya_low = res["archived_reference_summaries"]["yale_track_a_esi45_parity"]["arms"]
    ya_full = res["archived_reference_summaries"]["yale_track_a_full_560k_parity"]["arms"]

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.0), dpi=300)

    # Panel A: Flexible External LOESS Calibration Curves
    ax_a = axes[0]
    c_emory = b1b_cal["emory_cz_fluid"]
    c_b4a = b4["b4a_esi45_cross_campus"]["calibration"]["cz_fluid"]
    c_b4b = b4["b4b_allcomers_cross_campus"]["calibration"]["cz_fluid"]

    ax_a.plot([0, 95], [0, 95], ":", lw=1.6, color="#0F172A", label="Ideal 45-deg Calibration")
    ax_a.plot(
        np.array(c_b4b["curve_p"]) * 100, np.array(c_b4b["curve_y_loess"]) * 100, "-", lw=2.4, color="#059669",
        label=f"Yale B4b All-Comers (CITL={c_b4b['citl']:+.2f}, Slope={c_b4b['cal_slope']:.2f}, ICI={c_b4b['ici_loess']:.3f})",
    )
    ax_a.plot(
        np.array(c_b4a["curve_p"]) * 100, np.array(c_b4a["curve_y_loess"]) * 100, "-.", lw=2.3, color="#7C3AED",
        label=f"Yale B4a ESI 4-5 (CITL={c_b4a['citl']:+.2f}, Slope={c_b4a['cal_slope']:.2f}, ICI={c_b4a['ici_loess']:.3f})",
    )
    ax_a.plot(
        np.array(c_emory["curve_p"]) * 100, np.array(c_emory["curve_y_loess"]) * 100, "--", lw=2.2, color="#0284C7",
        label=f"Emory B1b Sepsis (CITL={c_emory['citl']:+.2f}, Slope={c_emory['cal_slope']:.2f}, ICI={c_emory['ici_loess']:.3f})",
    )
    ax_a.set_xlim(0, 92)
    ax_a.set_ylim(0, 92)
    ax_a.set_xlabel("Predicted Probability (%)")
    ax_a.set_ylabel("LOESS-Smoothed Observed Event Rate (%)")
    ax_a.set_title("A. External Calibration Curves (Emory B1b & Yale B4)", fontweight="bold")
    ax_a.legend(loc="upper left", fontsize=7.6, frameon=True, framealpha=0.93)
    ax_a.grid(alpha=0.25)

    # Panel B: All 5 Yale ED Benchmarks
    ax_b = axes[1]
    yale_rows = [
        ("Primary B4a:\nCross-Campus\nESI 4-5 (51k)", b4["b4a_esi45_cross_campus"]["external_per_split"], b4["b4a_esi45_cross_campus"]["external_symmetric_ensemble"]),
        ("Primary B4b:\nCross-Campus\nAll-Comers (166k)", b4["b4b_allcomers_cross_campus"]["external_per_split"], b4["b4b_allcomers_cross_campus"]["external_symmetric_ensemble"]),
        ("Ref. B3:\nBlinded ESI 4-5\n(153k)", b3["per_split_arms"], None),
        ("Ref. Track A:\nParity ESI 4-5\n(153k)", ya_low, None),
        ("Ref. Track A:\nParity Full\n(560k)", ya_full, None),
    ]
    xb = np.arange(len(yale_rows))
    bw = 0.25

    def _gain_and_ci(arms_d: dict, arm_key: str) -> tuple[float, float]:
        if arm_key not in arms_d:
            return np.nan, 0.0
        b0 = arms_d["paper1_global_cmi_global_s2"]["nb_base_0pct"]
        m_a = arms_d[arm_key]["nb_voi_auto"]
        return float(m_a["mean"]) - float(b0["mean"]), 1.96 * float(np.hypot(m_a["se"], b0["se"]))

    gl_gains = [_gain_and_ci(r[1], "paper1_global_cmi_global_s2") for r in yale_rows]
    gf_gains = [
        _gain_and_ci(r[1], "ablation_global_cmi_fluid_s2") if "ablation_global_cmi_fluid_s2" in r[1] else _gain_and_ci(r[1], "paper1_global_cmi_global_s2")
        for r in yale_rows
    ]
    cz_gains = [_gain_and_ci(r[1], "paper2_cz_disc_fluid_s2") for r in yale_rows]

    ax_b.bar(xb - bw, [g for g, _ in gl_gains], yerr=[c for _, c in gl_gains], capsize=3, width=bw, color="#3B82F6", edgecolor="#334155", lw=0.65, label="Global-CMI + Global S2")
    ax_b.bar(xb, [g for g, _ in gf_gains], yerr=[c for _, c in gf_gains], capsize=3, width=bw, color="#0284C7", edgecolor="#334155", lw=0.65, label="Global-CMI + Fluid S2")
    ax_b.bar(xb + bw, [g for g, _ in cz_gains], yerr=[c for _, c in cz_gains], capsize=3, width=bw, color="#059669", edgecolor="#064E3B", lw=0.65, label="CZ-Fluid (Proposed)")

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
            marker="D", s=38, color="#0F172A", zorder=6, label="Primary B4 Ensemble Gain" if idx_e == 0 else None,
        )
    ax_b.axvline(1.5, color="#64748B", ls="--", lw=1.15)
    bt_b4b = b4["b4b_allcomers_cross_campus"]["paired_bootstraps_10k"]["cz_fluid_vs_global_cmi"]
    bt_b4a = b4["b4a_esi45_cross_campus"]["paired_bootstraps_10k"]["cz_fluid_vs_global_cmi"]
    ax_b.text(
        0.02, 0.97,
        f"B4b Ext. All-Comers (166k): {bt_b4b['diff_mean']:+.2f} vs Glob-CMI (p={bt_b4b['p_two_sided_dh']:.3f})\n"
        f"B4a Ext. ESI 4-5 (51k): {bt_b4a['diff_mean']:+.2f} vs Glob-CMI (p={bt_b4a['p_two_sided_dh']:.3f})",
        transform=ax_b.transAxes, ha="left", va="top", fontsize=7.8,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#F8FAFC", "edgecolor": "#CBD5E1"},
    )
    ax_b.set_xticks(xb)
    ax_b.set_xticklabels([r[0] for r in yale_rows], fontsize=7.9)
    ax_b.set_ylim(0, 8.35)
    ax_b.set_ylabel("Delta Net Benefit Over Stage-1 Triage (/1,000)")
    ax_b.set_title("B. All 5 Yale ED Benchmarks: Primary B4 vs. Reference", fontweight="bold")
    ax_b.legend(loc="upper right", fontsize=7.5, frameon=True, framealpha=0.93)
    ax_b.grid(axis="y", alpha=0.25)

    plt.tight_layout(pad=1.4)
    return _save_both(fig, "fig5_calibration_and_yale_tracks.png")


def plot_fig6_subgroup_fairness_and_citl(res: dict) -> Path:
    b4 = res["yale_strengthened"]["yale_b4_cross_campus_primary"]
    b3 = res["yale_strengthened"]["yale_b3_fairness_blinded"]

    fig, (ax_c1, ax_c2) = plt.subplots(
        1, 2, figsize=(14.0, 5.2), dpi=300, sharey=True, gridspec_kw={"width_ratios": [1.28, 1.0], "wspace": 0.10}
    )

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
        y_pos, d_means, xerr=d_err, capsize=3.2, height=0.62, color="#059669", alpha=0.88, edgecolor="#064E3B", lw=0.65,
        label="Track B3 OOF Ensemble Delta NB (95% CI, B=5,000)",
    )
    ext_y, ext_d = [], []
    for yi, (k, _) in enumerate(sorted_subs):
        if k in sub_b4a:
            ext_y.append(yi)
            ext_d.append(float(sub_b4a[k]["delta_nb"]))
    if ext_y:
        ax_c1.scatter(ext_d, ext_y, marker="D", s=38, color="#0F172A", zorder=6, label="Primary B4a Campus B External Gain")

    ax_c1.axvline(0.0, color="#DC2626", ls="--", lw=1.3)
    ax_c1.set_yticks(y_pos)
    ax_c1.set_yticklabels(names, fontsize=8.8)
    ax_c1.invert_yaxis()
    ax_c1.set_xlabel("Subgroup Net Benefit Gain Over Stage-1 Triage (/1,000 Visits, 95% CI)")
    ax_c1.set_title("A. Fairness-Blinded Subgroup Gain (11/11 Positive; 8/11 95% CIs > 0)", fontweight="bold")
    ax_c1.legend(loc="lower right", fontsize=8.0, frameon=True, framealpha=0.93)
    ax_c1.grid(axis="x", alpha=0.25)

    citl_m = np.array([float(v["citl_cz_fluid"]) for _, v in sorted_subs])
    citl_lo = np.array([float(v["citl_ci95"][0]) for _, v in sorted_subs])
    citl_hi = np.array([float(v["citl_ci95"][1]) for _, v in sorted_subs])
    citl_err = np.vstack([np.maximum(0, citl_m - citl_lo), np.maximum(0, citl_hi - citl_m)])

    ax_c2.errorbar(
        citl_m, y_pos, xerr=citl_err, fmt="o", ms=6.0, lw=1.8, capsize=3.2, color="#0284C7",
        label="Subgroup Calibration-in-the-Large (95% CI)",
    )
    ax_c2.axvline(0.0, color="#0F172A", ls=":", lw=1.4, label="Ideal CITL = 0.0")
    plt.setp(ax_c2.get_yticklabels(), visible=False)
    ax_c2.set_xlabel("Subgroup Calibration-in-the-Large (Log-Odds Intercept Offset)")
    ax_c2.set_title("B. Subgroup Calibration-in-the-Large (CITL)", fontweight="bold")
    ax_c2.legend(loc="lower right", fontsize=8.0, frameon=True, framealpha=0.93)
    ax_c2.grid(axis="x", alpha=0.25)

    return _save_both(fig, "fig6_subgroup_fairness_and_citl.png")


def main() -> None:
    MANUSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)
    res_path = RES_DIR / "paper2_v2_strengthening_results.json"
    res = json.loads(res_path.read_text())
    paths = [
        plot_fig1_kernel_and_frontier(res),
        plot_fig2_synthetic_and_beta_regime(res),
        plot_fig3_sepsis_transfer_and_exchange(res),
        plot_fig4_landmarks_and_recalibration(res),
        plot_fig5_calibration_and_yale_tracks(res),
        plot_fig6_subgroup_fairness_and_citl(res),
    ]
    print("Generated 6 focused 300-DPI figures:")
    for p in paths:
        print(" ", p)


if __name__ == "__main__":
    main()
