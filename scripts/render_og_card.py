"""Render a crisp 1200x630 Open Graph / Twitter social preview card for collision-zone-ed."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "docs" / "og-card.png"


def bern_kl(tau: float, p: np.ndarray) -> np.ndarray:
    pc = np.clip(p, 1e-6, 1.0 - 1e-6)
    return tau * np.log(tau / pc) + (1.0 - tau) * np.log((1.0 - tau) / (1.0 - pc))


def main() -> None:
    fig = plt.figure(figsize=(12.0, 6.3), dpi=100, facecolor="#F8F7F3")
    ax_bg = fig.add_axes([0, 0, 1, 1], facecolor="#F8F7F3")
    ax_bg.set_xlim(0, 1200)
    ax_bg.set_ylim(0, 630)
    ax_bg.axis("off")

    # Top editorial accent rule + footer divider
    ax_bg.plot([0, 1200], [624, 624], color="#065F46", lw=12)
    ax_bg.plot([52, 1148], [122, 122], color="#DDD8CC", lw=1.2)

    # Left column typography (strictly within x = 52 .. 680)
    ax_bg.text(
        52, 566, "collision-zone-ed  ·  Preprint & Open-Source Package",
        fontsize=12.5, fontweight="bold", color="#065F46", family="monospace"
    )
    ax_bg.text(
        52, 508, "Rescuing Diagnostic Signal",
        fontsize=26.5, fontweight="bold", color="#111827", family="serif"
    )
    ax_bg.text(
        52, 464, "from True-Negative Abundance",
        fontsize=26.5, fontweight="bold", color="#111827", family="serif"
    )
    ax_bg.text(
        52, 404,
        "High-Signal False Positives, Schervish Beta-Kernel Discovery,\n"
        "and KL-Gated Residual Learning in Acute Care",
        fontsize=13.2, color="#374151", linespacing=1.48
    )

    # Three crisp editorial metric cards (52..686, 200px wide each, 17px gap)
    metrics = [
        ("600,154", "Clinical Encounters\nAcross 5 Hospitals", 52),
        ("+1.38 NB", "Emory CPT Blood\nDraws (p = 0.017)", 269),
        ("11 / 11", "Fairness-Blinded\nSubgroups ΔNB > 0", 486),
    ]
    for val, lbl, x0 in metrics:
        box = FancyBboxPatch(
            (x0, 168), 200, 170,
            boxstyle="round,pad=0,rounding_size=10",
            facecolor="#FFFFFF", edgecolor="#DDD8CC", linewidth=1.2
        )
        ax_bg.add_patch(box)
        ax_bg.text(x0 + 18, 286, val, fontsize=21.5, fontweight="bold", color="#065F46", family="serif")
        ax_bg.text(x0 + 18, 208, lbl, fontsize=10.8, color="#4B5563", linespacing=1.42)

    # Right column embedded mathematical curve card (x = 716 .. 1148)
    card = FancyBboxPatch(
        (716, 168), 432, 384,
        boxstyle="round,pad=0,rounding_size=10",
        facecolor="#FFFFFF", edgecolor="#DDD8CC", linewidth=1.2
    )
    ax_bg.add_patch(card)

    ax = fig.add_axes([0.645, 0.345, 0.285, 0.455], facecolor="#FCFBF8")
    p_grid = np.linspace(0.003, 0.32, 300)
    tau = 0.0625
    w_log = 1.0 / (p_grid * (1.0 - p_grid))
    w_log = w_log / np.max(w_log)
    w_cz = np.exp(24.0 * (tau * np.log(p_grid / tau) + (1.0 - tau) * np.log((1.0 - p_grid) / (1.0 - tau))))
    pi_kl = np.exp(-16.0 * bern_kl(tau, p_grid))

    ax.plot(p_grid * 100, w_log, "--", lw=2.0, color="#B91C1C", label="Log-Loss 1/[p(1-p)]")
    ax.plot(p_grid * 100, w_cz, "-", lw=2.5, color="#1D4ED8", label="Schervish Kernel")
    ax.plot(p_grid * 100, pi_kl, "-.", lw=2.1, color="#047857", label="Fluid KL Gate")
    ax.axvline(tau * 100, color="#111827", ls=":", lw=1.4)
    ax.scatter([tau * 100], [1.0], color="#1D4ED8", edgecolor="#FFFFFF", s=45, zorder=5)
    ax.set_xlim(0, 32)
    ax.set_ylim(-0.04, 1.08)
    ax.set_xlabel("Predicted Risk p (%)", fontsize=9.8, color="#374151")
    ax.set_ylabel("Normalized Weight", fontsize=9.8, color="#374151")
    ax.set_title("Focus at Action Threshold τ* = 6.25%", fontsize=11.0, fontweight="bold", color="#111827", pad=8)
    ax.tick_params(labelsize=8.8, colors="#525D6D")
    ax.grid(alpha=0.25, color="#BDB6A7")
    ax.legend(fontsize=8.4, loc="upper right", frameon=True, facecolor="#FFFFFF", edgecolor="#DDD8CC")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#BDB6A7")

    # Footer: Author & Google Antigravity for Science credit
    ax_bg.text(
        52, 78, "Wayne A. Satz, MD  ·  Department of Emergency Medicine, Temple University",
        fontsize=13.5, fontweight="bold", color="#111827"
    )
    ax_bg.text(
        52, 42, "Supported in part by the Google Antigravity for Science program  ·  github.com/deesatzed/collision-zone-ed",
        fontsize=12.0, color="#4B5563"
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=100, facecolor="#F8F7F3")
    plt.close(fig)
    print("Saved OG social card:", OUT_PATH)


if __name__ == "__main__":
    main()
