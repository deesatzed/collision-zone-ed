"""Unit and mathematical verification tests for Three-Predicates & Fluid Collision Backbone."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score

from collision_zone import (
    AgnielDeconfounder,
    ChannelDef,
    ClinicalContract,
    FluidCollisionBackboneClassifier,
    bernoulli_kl_from_tau,
    collision_beta_params,
    collision_zone_relevance_weights,
    compute_decision_curve,
    empirical_martingale_voi,
    paired_bootstrap_nb_diff,
    run_sequential_discovery,
    sample_pilot,
    schervish_collision_proper_score,
    schervish_collision_weight,
    solve_ess_kappa,
)
from collision_zone.three_predicates import FastFrameEncoder, compute_oof_stage1


def test_schervish_score_is_strictly_proper_and_peaks_at_tau_star() -> None:
    tau = 0.10
    kappa = 16.0
    a, b = collision_beta_params(tau, kappa)
    mode = (a - 1.0) / (a + b - 2.0)
    assert abs(mode - tau) < 1e-12

    w_vals = schervish_collision_weight(np.array([0.001, tau, 0.50]), tau, kappa)
    assert w_vals[1] > 20.0 * w_vals[0]
    assert w_vals[1] > w_vals[2]

    for eta in [0.03, 0.10, 0.25, 0.60]:
        q_grid = np.linspace(0.005, 0.995, 991)
        s1 = schervish_collision_proper_score(np.ones_like(q_grid), q_grid, tau, kappa)
        s0 = schervish_collision_proper_score(np.zeros_like(q_grid), q_grid, tau, kappa)
        expected = eta * s1 + (1.0 - eta) * s0
        best_q = float(q_grid[int(np.argmax(expected))])
        assert abs(best_q - eta) < 0.002


def test_collision_neyman_concentrates_on_high_signal_false_positives() -> None:
    rng = np.random.default_rng(7)
    n = 10000
    is_cz = rng.uniform(size=n) < 0.10
    p_base = np.where(is_cz, rng.uniform(0.07, 0.18, size=n), rng.uniform(0.001, 0.015, size=n))
    y = rng.binomial(1, p_base)

    p_srs = sample_pilot(y, p_base, pilot_fraction=0.15, mode="srs", tau_star=0.10, seed=1)
    p_gney = sample_pilot(
        y, p_base, pilot_fraction=0.15, mode="global_neyman", tau_star=0.10, seed=1
    )
    p_cz = sample_pilot(
        y, p_base, pilot_fraction=0.15, mode="collision_neyman", tau_star=0.10, seed=1
    )

    assert p_cz.indices.size == 1500
    assert np.isclose(np.sum(p_cz.ht_weights), n, rtol=1e-6)
    assert (
        p_cz.high_signal_fp_share_of_controls
        > p_gney.high_signal_fp_share_of_controls
        > p_srs.high_signal_fp_share_of_controls
    )


def test_true_negative_dilution_rescue_and_null_control() -> None:
    rng = np.random.default_rng(123)
    n = 10000
    in_cz = rng.uniform(size=n) < 0.06
    z_cz_true = rng.normal(0.0, 1.0, size=n)
    z_tn_decoy = rng.normal(0.0, 1.0, size=n)

    true_logit = np.where(in_cz, -2.9 + 1.85 * z_cz_true, -4.2 + 1.08 * z_tn_decoy)
    y = rng.binomial(1, expit(true_logit))

    df = pd.DataFrame(
        {
            "base_flag": in_cz.astype(float) + rng.normal(0, 0.05, size=n),
            "base_noise": rng.normal(0, 1.0, size=n),
            "ch_cz_tiebreaker": np.where(in_cz, z_cz_true, rng.normal(0, 1.0, size=n)),
            "ch_tn_decoy": np.where(~in_cz, z_tn_decoy, rng.normal(0, 1.0, size=n)),
            "ch_pure_null_1": rng.normal(0, 1.0, size=n),
            "ch_pure_null_2": rng.normal(0, 1.0, size=n),
        }
    )
    base_cols = ["base_flag", "base_noise"]
    channels = [
        ChannelDef("ch_cz_tiebreaker", ("ch_cz_tiebreaker",), cost=1.0),
        ChannelDef("ch_tn_decoy", ("ch_tn_decoy",), cost=1.0),
        ChannelDef("ch_pure_null_1", ("ch_pure_null_1",), cost=1.0, is_null_stress=True),
        ChannelDef("ch_pure_null_2", ("ch_pure_null_2",), cost=1.0, is_null_stress=True),
    ]
    contract = ClinicalContract(
        cost_fp=1.0, cost_fn=9.0, target_precision=0.45, per_instance_query_cost=0.02
    )
    oof_p = compute_oof_stage1(df[base_cols], y, n_folds=4, seed=11)

    res_global = run_sequential_discovery(
        df,
        y,
        base_cols,
        channels,
        oof_p,
        contract,
        method="global_cmi",
        pilot_fraction=0.25,
        budget=1.0,
        seed=21,
    )
    res_cz = run_sequential_discovery(
        df,
        y,
        base_cols,
        channels,
        oof_p,
        contract,
        method="collision_zone",
        pilot_fraction=0.25,
        budget=1.0,
        seed=21,
    )

    assert res_global.selected_ids == ["ch_tn_decoy"]
    assert res_cz.selected_ids == ["ch_cz_tiebreaker"]
    res_cz_b4 = run_sequential_discovery(
        df,
        y,
        base_cols,
        channels,
        oof_p,
        contract,
        method="collision_zone",
        pilot_fraction=0.25,
        budget=4.0,
        seed=21,
    )
    assert "ch_pure_null_1" not in res_cz_b4.selected_ids
    assert "ch_pure_null_2" not in res_cz_b4.selected_ids


def test_relevance_weights_range() -> None:
    w = collision_zone_relevance_weights(
        np.array([0.001, 0.10, 0.90]), tau_star=0.10, kappa=14.0, floor=0.05
    )
    assert np.all(w >= 0.05)
    assert np.isclose(w[1], 1.05, atol=0.03)


def test_ess_kappa_solver_adapts_to_collision_scarcity() -> None:
    rng = np.random.default_rng(42)
    tau = 0.08
    # Abundant collision samples (30% near tau*) vs. ultra-scarce (0.5% near tau*)
    p_abundant = np.where(rng.random(3000) < 0.30, 0.08, 0.002)
    p_scarce = np.where(rng.random(3000) < 0.005, 0.08, 0.002)
    k_abundant = solve_ess_kappa(
        bernoulli_kl_from_tau(p_abundant, tau), kappa_max=18.0, min_effective_cz_samples=250.0
    )
    k_scarce = solve_ess_kappa(
        bernoulli_kl_from_tau(p_scarce, tau), kappa_max=18.0, min_effective_cz_samples=250.0
    )
    assert k_abundant == 18.0
    assert 0.0 < k_scarce < k_abundant


def test_fluid_collision_backbone_rescues_subthreshold_inversion() -> None:
    rng = np.random.default_rng(2026)
    tau = 0.08
    n_tr, n_te = 3500, 4000
    n = n_tr + n_te
    is_cz = rng.random(n) < 0.12
    ell1 = np.where(
        is_cz, logit(tau) + rng.normal(0.0, 0.4, n), -5.0 + rng.normal(0.0, 0.6, n)
    )
    z1 = rng.normal(0.0, 1.0, n)
    z2 = rng.normal(0.0, 1.0, n)
    x_nuis = rng.normal(0.0, 1.0, (n, 10))

    cz_gate = expit((ell1 - logit(0.035)) / 0.30)
    delta_true = cz_gate * (1.35 * z1 + 0.9 * ((z1 > 0.3) & (z2 > 0.3))) - (1.0 - cz_gate) * (
        0.85 * z1
    )
    y = rng.binomial(1, expit(ell1 + delta_true))

    x_all = np.column_stack([x_nuis, z1, z2]).astype(np.float32)
    z_s2 = np.column_stack([z1, z2]).astype(np.float32)

    fluid = FluidCollisionBackboneClassifier(tau_star=tau, min_cz_ess=200.0, seed=11).fit(
        x_all[:n_tr], z_s2[:n_tr], y[:n_tr], ell1[:n_tr]
    )
    p_fluid = fluid.predict_proba(x_all[n_tr:], z_s2[n_tr:], ell1[n_tr:])
    p_glob_only = expit(
        np.clip(
            fluid.platt_glob_[0] * fluid.m_glob_.predict(x_all[n_tr:], raw_score=True)  # type: ignore[union-attr]
            + fluid.platt_glob_[1],
            -20.0,
            20.0,
        )
    )
    assert fluid.beta_cz_ > 0.4
    assert average_precision_score(y[n_tr:], p_fluid) > average_precision_score(
        y[n_tr:], p_glob_only
    ) + 0.04


def test_agniel_deconfounder_blocks_informative_ordering_leakage() -> None:
    """Verify AgnielDeconfounder eliminates signal when lab ordering D->Y is confounded but assay Z is pure noise."""
    rng = np.random.default_rng(77)
    n = 4000
    x_b = rng.normal(0.0, 1.0, n)
    # Unobserved clinician bedside suspicion u drives BOTH lab ordering D and outcome Y!
    u_suspicion = rng.normal(0.0, 1.0, n)
    y = rng.binomial(1, expit(-2.2 + 0.7 * x_b + 1.8 * u_suspicion))
    drawn = rng.binomial(1, expit(-1.5 + 0.5 * x_b + 2.0 * u_suspicion))
    # Actual lab assay value depends only on x_b + noise (ZERO conditional info about u_suspicion or y!)
    z_raw = 0.8 * x_b + rng.normal(0.0, 1.0, n)
    z_obs = np.where(drawn == 1, z_raw, np.nan)

    df_base = pd.DataFrame({"xb": x_b})
    df_s2 = pd.DataFrame({"lab_confounded": z_obs})

    # Naive encoding leaks `drawn` via missingness indicator; AgnielDeconfounder + value-only blocks it
    deconf = AgnielDeconfounder().fit(df_base.iloc[:2500], df_s2.iloc[:2500])
    df_s2_clean = deconf.transform(df_base.iloc[2500:], df_s2.iloc[2500:], seed=9)
    enc_clean = FastFrameEncoder(include_missing_indicators=False).fit(df_s2_clean)
    mat_clean = enc_clean.transform(df_s2_clean)
    assert mat_clean.shape[1] == 1
    assert np.all(np.isfinite(mat_clean))
    # Verify conditional independence from ordering flag after adjusting for x_b
    resid_after = mat_clean[:, 0] - 0.8 * x_b[2500:]
    corr_with_drawn = abs(float(np.corrcoef(resid_after, drawn[2500:])[0, 1]))
    corr_with_y = abs(float(np.corrcoef(resid_after, y[2500:])[0, 1]))
    assert corr_with_drawn < 0.06
    assert corr_with_y < 0.06


def test_empirical_martingale_voi_and_paired_bootstrap() -> None:
    rng = np.random.default_rng(99)
    contract = ClinicalContract(
        cost_fp=1.0, cost_fn=9.0, target_precision=0.4, per_instance_query_cost=0.02
    )
    # Right-skewed bimodal lab distribution (80% small negative nudge, 20% +2.5 log-odds spike)
    raw_samples = np.where(rng.random(2000) < 0.20, 2.4, -0.35) + rng.normal(0.0, 0.1, 2000)
    q_grid = np.quantile(raw_samples, np.linspace(0.02, 0.98, 33))[None, :]  # 1 bin x 33 quantiles
    p1 = np.array([0.002, 0.095, 0.65])
    voi = empirical_martingale_voi(p1, np.zeros(3, dtype=int), q_grid, contract)
    # VOI peaks right near tau* = 0.10 and vanishes far away
    assert voi[1] > 10.0 * voi[0]
    assert voi[1] > 5.0 * voi[2]

    # Paired bootstrap & decision curve sanity check
    y_te = rng.binomial(1, 0.10, 1000)
    p_better = np.clip(0.08 + 0.25 * y_te + rng.normal(0, 0.02, 1000), 0.001, 0.999)
    p_worse = np.full(1000, 0.08)
    q_mask = np.zeros(1000, dtype=bool)
    boot = paired_bootstrap_nb_diff(
        y_te, p_better, q_mask, p_worse, q_mask, contract, n_boot=300, seed=5
    )
    assert boot["diff_mean"] > 0.0
    assert boot["ci95_low"] > 0.0
    assert boot["p_two_sided"] < 0.01

    dc = compute_decision_curve(y_te, p_better, q_mask, [0.05, 0.10, 0.20])
    assert len(dc["nb_model"]) == 3
