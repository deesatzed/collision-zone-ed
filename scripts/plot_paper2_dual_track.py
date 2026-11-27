"""Render 4-panel Dual-Track (Track A Parity vs. Track B Hardened + Decision Curve + Fairness) figure."""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
RES_DIR = Path(os.environ.get("COLLISION_ZONE_RES_DIR", str(REPO_ROOT / "results")))
OUT_DIR = Path(os.environ.get("COLLISION_ZONE_OUT_DIR", str(RES_DIR)))


def main() -> None:
    track_a = json.loads((RES_DIR / "paper2_initial_campaign_summary.json").read_text())
    track_b = json.loads((RES_DIR / "paper2_track_b_hardened_summary.json").read_text())

    fig, axes = plt.subplots(2, 2, figsize=(14.4, 9.6), dpi=200)

    # Panel A: Sepsis Emory Zero-Shot Transfer — Track A Parity vs. Track B1b Measured-Assay CPT vs. Track B2 Gap
    ax = axes[0, 0]
    sep_a = track_a["sepsis_bidmc_to_emory"]["external_emory_transfer"]
    sep_b1 = track_b["b1b_sepsis_6h_measured_bloodwork_cpt"]["external_emory_transfer"]
    sep_b2 = track_b["b2_sepsis_4h_gap2h_strict_lead"]["external_emory_transfer"]

    labels_a = [
        "Track A:\nPhysioNet Parity (6h)",
        "Track B1b:\nMeasured Blood CPT (6h)",
        "Track B2:\nStrict 4h+2h Gap",
    ]
    b0_vals = [
        sep_a["paper1_global_cmi_global_s2"]["nb_base_0pct"]["mean"],
        sep_b1["paper1_global_cmi_global_s2"]["nb_base_0pct"]["mean"],
        sep_b2["paper1_global_cmi_global_s2"]["nb_base_0pct"]["mean"],
    ]
    p1_vals = [
        sep_a["paper1_global_cmi_global_s2"]["nb_voi_auto"]["mean"],
        sep_b1["paper1_global_cmi_global_s2"]["nb_voi_auto"]["mean"],
        sep_b2["paper1_global_cmi_global_s2"]["nb_voi_auto"]["mean"],
    ]
    p2_vals = [
        sep_a["paper2_cz_disc_fluid_s2"]["nb_voi_auto"]["mean"],
        sep_b1["paper2_cz_disc_fluid_s2"]["nb_voi_auto"]["mean"],
        sep_b2["paper2_cz_disc_fluid_s2"]["nb_voi_auto"]["mean"],
    ]
    xa = np.arange(len(labels_a))
    bw = 0.25
    ax.bar(xa - bw, b0_vals, width=bw, color="#CBD5E1", label="Stage-1 Bedside Vitals (0% q)")
    ax.bar(xa, p1_vals, width=bw, color="#3B82F6", label="Global-CMI + Global S2")
    ax.bar(xa + bw, p2_vals, width=bw, color="#059669", label="CZ-Fluid: KL Backbone")
    ax.axhline(0.0, color="#475569", lw=1.0)
    ax.set_xticks(xa)
    ax.set_xticklabels(labels_a, fontsize=8.8)
    ax.set_ylabel("Zero-Shot Emory Net Benefit (/1,000)")
    ax.set_title("A. Sepsis External Transfer: Parity vs. De-Confounded", fontsize=10.8, fontweight="bold")
    ax.legend(fontsize=8.1, loc="best")
    ax.grid(axis="y", alpha=0.25)

    # Panel B: Full Vickers Decision Curve Across Clinical Thresholds tau in [2%, 15%] (Emory B1b)
    ax = axes[0, 1]
    dca = track_b["b1b_sepsis_6h_measured_bloodwork_cpt"]["emory_decision_curve"]
    taus_pct = [t * 100.0 for t in dca["taus"]]
    ax.plot(taus_pct, dca["treat_all"], ":", lw=1.6, color="#94A3B8", label="Treat All")
    ax.axhline(0.0, ls="--", lw=1.2, color="#64748B", label="Treat None (0)")
    ax.plot(taus_pct, dca["stage1_base"], "-.", lw=2.0, color="#475569", label="Stage-1 Bedside Vitals")
    ax.plot(taus_pct, dca["paper1_global"], "-o", ms=4.5, lw=2.0, color="#3B82F6", label="Global-CMI + Global S2")
    ax.plot(taus_pct, dca["paper2_fluid"], "-s", ms=5.0, lw=2.4, color="#059669", label="CZ-Fluid: KL Backbone")
    ax.axvline(6.25, color="#0F172A", ls=":", lw=1.3)
    ax.set_ylim(-6.0, max(dca["paper2_fluid"]) + 3.0)
    ax.set_xlabel("Clinical Action Threshold tau (%)")
    ax.set_ylabel("Net Benefit per 1,000 Patients")
    ax.set_title("B. Emory External Vickers Decision Curve (Measured CPT)", fontsize=10.8, fontweight="bold")
    ax.legend(fontsize=8.0, loc="upper right")
    ax.grid(alpha=0.25)

    # Panel C: Yale ED ESI 4-5 Occult Admission — Track A Parity vs. Track B3 Blinded vs. Track B4 Cross-Campus
    ax = axes[1, 0]
    ya_low = track_a["yale_ed_esi45_lowacuity"]["arms"]
    yb3_low = track_b["b3_yale_esi45_fairness_patient_grouped"]["arms"]
    yb4_low = track_b["b4_yale_cross_campus_external_transfer"]["esi45_lowacuity_cross_campus"]["external_arms"]

    labels_c = [
        "Track A:\nHong Parity (153k)",
        "Track B3:\nFairness-Blinded",
        "Track B4: Zero-Shot\nCross-Campus ED",
    ]
    yc_b0 = [
        ya_low["paper1_global_cmi_global_s2"]["nb_base_0pct"]["mean"],
        yb3_low["paper1_global_cmi_global_s2"]["nb_base_0pct"]["mean"],
        yb4_low["paper1_global_cmi_global_s2"]["nb_base_0pct"]["mean"],
    ]
    yc_p1 = [
        ya_low["paper1_global_cmi_global_s2"]["nb_voi_auto"]["mean"],
        yb3_low["paper1_global_cmi_global_s2"]["nb_voi_auto"]["mean"],
        yb4_low["paper1_global_cmi_global_s2"]["nb_voi_auto"]["mean"],
    ]
    yc_p2 = [
        ya_low["paper2_cz_disc_fluid_s2"]["nb_voi_auto"]["mean"],
        yb3_low["paper2_cz_disc_fluid_s2"]["nb_voi_auto"]["mean"],
        yb4_low["paper2_cz_disc_fluid_s2"]["nb_voi_auto"]["mean"],
    ]
    xc = np.arange(len(labels_c))
    ax.bar(xc - bw, yc_b0, width=bw, color="#CBD5E1", label="Stage-1 Triage Only (0% q)")
    ax.bar(xc, yc_p1, width=bw, color="#3B82F6", label="Global-CMI + Global S2")
    ax.bar(xc + bw, yc_p2, width=bw, color="#059669", label="CZ-Fluid: KL Backbone")
    ax.set_xticks(xc)
    ax.set_xticklabels(labels_c, fontsize=8.8)
    ax.set_ylabel("Net Benefit per 1,000 Fast-Track Visits")
    ax.set_title("C. Yale ESI 4-5: Parity, Fairness-Blinded & Cross-Campus", fontsize=10.8, fontweight="bold")
    ax.legend(fontsize=8.1, loc="best")
    ax.grid(axis="y", alpha=0.25)

    # Panel D: Post-Hoc Subgroup Fairness Audit (All 11 Audited Demographics & Insurance Strata)
    ax = axes[1, 1]
    sub_dict = track_b["b3_yale_esi45_fairness_patient_grouped"]["fairness_subgroup_audit"]
    label_map = {
        "lang:English": "Lang: English",
        "lang:Other": "Lang: Non-English",
        "race:White or Caucasian": "Race: White",
        "race:Black or African American": "Race: Black",
        "race:Other": "Race: Other",
        "race:Asian": "Race: Asian",
        "race:Patient Refused": "Race: Refused",
        "insurance_status:Medicaid": "Ins: Medicaid",
        "insurance_status:Commercial": "Ins: Commercial",
        "insurance_status:Medicare": "Ins: Medicare",
        "insurance_status:Other": "Ins: Other",
    }
    sorted_subs = sorted(sub_dict.items(), key=lambda kv: kv[1]["delta_nb_gain"], reverse=True)
    s_names = [label_map.get(k, k) for k, _ in sorted_subs]
    s_gains = [v["delta_nb_gain"] for _, v in sorted_subs]
    yd = np.arange(len(s_names))
    colors_d = ["#059669" if g >= 0 else "#DC2626" for g in s_gains]
    ax.barh(yd, s_gains, color=colors_d, height=0.66, edgecolor="#334155", lw=0.5)
    ax.axvline(0.0, color="#475569", lw=1.0)
    ax.set_yticks(yd)
    ax.set_yticklabels(s_names, fontsize=8.3)
    ax.invert_yaxis()
    ax.set_xlabel("Net Benefit Gain Over Stage-1 Triage (/1,000 Visits)")
    ax.set_title("D. Fairness-Blinded Subgroup Gain (All 11 Strata)", fontsize=10.8, fontweight="bold")
    ax.grid(axis="x", alpha=0.25)

    plt.tight_layout()
    out_png = OUT_DIR / "fig_paper2_dual_track_hardened.png"
    out_pdf = OUT_DIR / "fig_paper2_dual_track_hardened.pdf"
    plt.savefig(out_png, dpi=200)
    plt.savefig(out_pdf)
    plt.close(fig)
    print("Saved Dual-Track 4-panel figure:", out_png)


if __name__ == "__main__":
    main()
