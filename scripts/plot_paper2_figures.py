"""Generate 4-panel publication-quality summary figure for Paper 2 Fluid Collision Backbone campaign."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from collision_zone.three_predicates import bernoulli_kl_from_tau, schervish_collision_weight

REPO_ROOT = Path(__file__).resolve().parents[1]
RES_DIR = Path(os.environ.get("COLLISION_ZONE_RES_DIR", str(REPO_ROOT / "results")))
OUT_DIR = Path(os.environ.get("COLLISION_ZONE_OUT_DIR", str(RES_DIR)))


def main() -> None:
    summary = json.loads((RES_DIR / "paper2_initial_campaign_summary.json").read_text())
    sim = summary["synthetic_dilution_sim"]
    sep = summary["sepsis_bidmc_to_emory"]
    yale_low = summary["yale_ed_esi45_lowacuity"]
    yale_full = summary["yale_ed_full_560k"]

    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.2), dpi=200)

    # Panel A: Schervish True-Negative Singularity vs. Proper Kernel & KL Gate
    ax = axes[0, 0]
    p_grid = np.linspace(0.002, 0.40, 500)
    tau = 0.08
    w_log = 1.0 / (p_grid * (1.0 - p_grid))
    w_log_norm = w_log / np.max(w_log)
    w_cz = schervish_collision_weight(p_grid, tau_star=tau, kappa=24.0)
    w_cz_norm = w_cz / np.max(w_cz)
    pi_kl = np.exp(-16.0 * bernoulli_kl_from_tau(p_grid, tau))

    ax.plot(p_grid * 100, w_log_norm, "--", lw=2.1, color="#DC2626", label="Global Log-Loss: 1/[p(1-p)]")
    ax.plot(p_grid * 100, w_cz_norm, "-", lw=2.4, color="#0284C7", label="Discovery Proper Kernel w_CZ(p)")
    ax.plot(p_grid * 100, pi_kl, "-.", lw=2.1, color="#059669", label="Stage-2 Fluid KL Gate pi_kappa*(p)")
    ax.axvline(tau * 100, color="#0F172A", ls=":", lw=1.6, label=f"Threshold tau* = {tau*100:.1f}%")
    ax.set_xlabel("Posterior Risk (%)")
    ax.set_ylabel("Normalized Weight / Gate")
    ax.set_title("A. Predicate 1 & Fluid KL Gate Around tau*", fontsize=10.8, fontweight="bold")
    ax.legend(fontsize=8.3, loc="upper right")
    ax.grid(alpha=0.25)

    # Panel B: Synthetic True-Negative Dilution Recovery & Pilot High-Signal FP Share
    ax = axes[0, 1]
    cats = ["True Collision\nChannel Picked", "Pilot Controls:\nHigh-Signal FP %"]
    g_vals = [
        sim["global_cmi_true_tiebreaker_rate"]["mean"] * 100.0,
        sim["pilot_hsfp_share_global_neyman"]["mean"] * 100.0,
    ]
    c_vals = [
        sim["collision_zone_true_tiebreaker_rate"]["mean"] * 100.0,
        sim["pilot_hsfp_share_collision_neyman"]["mean"] * 100.0,
    ]
    x = np.arange(len(cats))
    w = 0.34
    ax.bar(x - w / 2, g_vals, width=w, color="#94A3B8", label="Global-CMI (Global Neyman)")
    ax.bar(x + w / 2, c_vals, width=w, color="#059669", label="CZ-Fluid (Collision-Zone + HSFP)")
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("B. Rescuing Diluted Signal (30 Replicates)", fontsize=10.8, fontweight="bold")
    for xi, gv, cv in zip(x, g_vals, c_vals):
        ax.text(xi - w / 2, gv + 2.2, f"{gv:.1f}%", ha="center", fontsize=8.8)
        ax.text(xi + w / 2, cv + 2.2, f"{cv:.1f}%", ha="center", fontsize=8.8, fontweight="bold")
    ax.legend(fontsize=8.5, loc="upper right")
    ax.grid(axis="y", alpha=0.25)

    # Panel C: Zero-Shot External Hospital Sepsis Transfer (Beth Israel -> Emory, N=19,696)
    ax = axes[1, 0]
    ext = sep["external_emory_transfer"]
    arm_keys = [
        ("Triage Vitals\nOnly (0% q)", ext["paper1_global_cmi_global_s2"]["nb_base_0pct"], "#CBD5E1"),
        ("Global-CMI\n+ Global S2", ext["paper1_global_cmi_global_s2"]["nb_voi_auto"], "#3B82F6"),
        ("Ablation: CZ +\nNaive Downwt.", ext["ablation_cz_disc_naive_s2"]["nb_voi_auto"], "#F59E0B"),
        ("CZ-Fluid:\nKL Backbone", ext["paper2_cz_disc_fluid_s2"]["nb_voi_auto"], "#059669"),
    ]
    xc = np.arange(len(arm_keys))
    means_c = [item[1]["mean"] for item in arm_keys]
    ses_c = [item[1]["se"] for item in arm_keys]
    cols_c = [item[2] for item in arm_keys]
    ax.bar(xc, means_c, yerr=ses_c, capsize=4, width=0.58, color=cols_c, edgecolor="#334155", lw=0.6)
    ax.axhline(0.0, color="#475569", lw=1.0)
    ax.set_xticks(xc)
    ax.set_xticklabels([item[0] for item in arm_keys], fontsize=8.8)
    ax.set_ylabel("Net Benefit per 1,000 Patients (Auto VOI)")
    ax.set_title("C. Zero-Shot Emory Sepsis Transfer (N = 19,696)", fontsize=10.8, fontweight="bold")
    for xi, mv, sv in zip(xc, means_c, ses_c):
        offset_y = mv + sv + 0.16 if mv >= 0 else mv - sv - 0.35
        ax.text(xi, offset_y, f"{mv:+.2f}", ha="center", fontsize=8.8, fontweight="bold")
    ax.set_ylim(min(means_c) - 0.9, max(means_c) + 0.9)
    ax.grid(axis="y", alpha=0.25)

    # Panel D: Net Benefit Gain Over Stage-1 Triage Across BIDMC, Yale ESI 4-5 (153k), Yale Full (560k)
    ax = axes[1, 1]
    cohorts = [
        ("BIDMC Sepsis\n(Internal 20k)", sep["internal_bidmc_test"]),
        ("Yale ESI 4-5\n(Occult 153k)", yale_low["arms"]),
        ("Yale All-Comers\n(Full 560k)", yale_full["arms"]),
    ]
    xd = np.arange(len(cohorts))
    bw = 0.25
    g_p1 = [
        cdict["paper1_global_cmi_global_s2"]["nb_voi_auto"]["mean"]
        - cdict["paper1_global_cmi_global_s2"]["nb_base_0pct"]["mean"]
        for _, cdict in cohorts
    ]
    g_naive = [
        cdict["ablation_cz_disc_naive_s2"]["nb_voi_auto"]["mean"]
        - cdict["ablation_cz_disc_naive_s2"]["nb_base_0pct"]["mean"]
        for _, cdict in cohorts
    ]
    g_fluid = [
        cdict["paper2_cz_disc_fluid_s2"]["nb_voi_auto"]["mean"]
        - cdict["paper2_cz_disc_fluid_s2"]["nb_base_0pct"]["mean"]
        for _, cdict in cohorts
    ]
    ax.bar(xd - bw, g_p1, width=bw, color="#3B82F6", label="Global-CMI + Global S2")
    ax.bar(xd, g_naive, width=bw, color="#F59E0B", label="Ablation: CZ + Naive Downwt. S2")
    ax.bar(xd + bw, g_fluid, width=bw, color="#059669", label="CZ-Fluid: KL Backbone")
    ax.axhline(0.0, color="#475569", lw=0.9)
    ax.set_xticks(xd)
    ax.set_xticklabels([c[0] for c in cohorts], fontsize=9.0)
    ax.set_ylabel("Delta Net Benefit vs. Stage-1 Triage (/1,000)")
    ax.set_title("D. Eliminating Leaf Starvation (BIDMC & 3 Yale EDs)", fontsize=10.8, fontweight="bold")
    ax.legend(fontsize=8.2, loc="upper left")
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    out_png = OUT_DIR / "fig_paper2_three_predicates_ws4tb.png"
    out_pdf = OUT_DIR / "fig_paper2_three_predicates_ws4tb.pdf"
    plt.savefig(out_png, dpi=200)
    plt.savefig(out_pdf)
    plt.close(fig)
    print("Saved 4-panel figure:", out_png)


if __name__ == "__main__":
    main()
