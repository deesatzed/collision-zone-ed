"""Expanded 24-test mathematical & statistical unit test suite for Paper 2 v2 (`P2-15`)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score

from collision_zone.three_predicates import (
    AgnielDeconfounder,
    ChannelDef,
    ClinicalContract,
    FastFrameEncoder,
    PilotSample,
    bernoulli_kl_from_tau,
    cfbh_selection,
    collision_beta_params,
    collision_zone_relevance_weights,
    compute_decision_curve,
    compute_oof_stage1,
    empirical_martingale_voi,
    evaluate_stage2_multi_test,
    fit_stage1_predictions,
    holm_adjust,
    log_likelihood_score,
    run_sequential_discovery,
    sample_pilot,
    schervish_collision_proper_score,
    schervish_collision_weight,
    solve_ess_kappa,
    two_phase_ht_mean_and_se,
    union_collision_gate,
    waterfill_neyman,
)
from collision_zone.three_predicates_v2 import (
    FluidCollisionBackboneClassifierV2,
    _unit_nb_vector,
    clinical_impact_counts_per_1k,
    clopper_pearson_ci,
    compute_calibration_metrics,
    compute_oof_stage1_v2,
    compute_tau_kappa_frontier,
    evaluate_covert_greedy_cmi_afa,
    evaluate_stage2_multi_test_v2,
    paired_bootstrap_interaction,
    paired_bootstrap_nb_diff_v2,
    prior_odds_recalibrate_probabilities,
    run_sequential_discovery_v2,
    solve_citl_intercept,
    standardized_net_benefit,
)


# 1
def test_01_beta_kernel_mode_and_boundary_vanishing() -> None:
    for tau in (0.04, 0.0625, 0.10, 0.15):
        a, b = collision_beta_params(tau, kappa=24.0)
        mode = (a - 1.0) / (a + b - 2.0)
        assert abs(mode - tau) < 1e-12
        w_ends = schervish_collision_weight(np.array([1e-9, tau, 1.0 - 1e-9]), tau, kappa=24.0)
        assert w_ends[0] < 1e-3
        assert w_ends[2] < 1e-3
        assert w_ends[1] > 1.0


# 2
def test_02_schervish_closed_form_matches_numerical_quadrature() -> None:
    tau = 0.08
    kappa = 18.0
    for p in (0.01, 0.08, 0.25, 0.70):
        s1_closed = float(schervish_collision_proper_score(np.array([1.0]), np.array([p]), tau, kappa)[0])
        s0_closed = float(schervish_collision_proper_score(np.array([0.0]), np.array([p]), tau, kappa)[0])
        s1_num, _ = quad(
            lambda t: -(1.0 - t) * float(schervish_collision_weight(np.array([t]), tau, kappa)[0]),
            p,
            1.0,
        )
        s0_num, _ = quad(
            lambda t: -t * float(schervish_collision_weight(np.array([t]), tau, kappa)[0]),
            0.0,
            p,
        )
        assert abs(s1_closed - s1_num) < 1e-9
        assert abs(s0_closed - s0_num) < 1e-9


# 3
def test_03_schervish_strict_properness_across_probabilities() -> None:
    tau = 0.0625
    kappa = 20.0
    q_grid = np.linspace(0.002, 0.998, 1993)
    s1 = schervish_collision_proper_score(np.ones_like(q_grid), q_grid, tau, kappa)
    s0 = schervish_collision_proper_score(np.zeros_like(q_grid), q_grid, tau, kappa)
    for eta in (0.015, 0.0625, 0.18, 0.55, 0.85):
        exp_score = eta * s1 + (1.0 - eta) * s0
        opt_q = float(q_grid[int(np.argmax(exp_score))])
        assert abs(opt_q - eta) < 0.0015


# 4
def test_04_bregman_divergence_nonnegative_and_zero_iff_equal() -> None:
    tau = 0.10
    kappa = 24.0
    for eta1, eta2 in [(0.05, 0.05), (0.05, 0.14), (0.20, 0.06)]:
        s_at_2 = eta2 * schervish_collision_proper_score(np.array([1]), np.array([eta2]), tau, kappa) + (
            1.0 - eta2
        ) * schervish_collision_proper_score(np.array([0]), np.array([eta2]), tau, kappa)
        s_at_1 = eta2 * schervish_collision_proper_score(np.array([1]), np.array([eta1]), tau, kappa) + (
            1.0 - eta2
        ) * schervish_collision_proper_score(np.array([0]), np.array([eta1]), tau, kappa)
        gain = float((s_at_2 - s_at_1)[0])
        if abs(eta1 - eta2) < 1e-12:
            assert abs(gain) < 1e-14
        else:
            assert gain > 1e-6


# 5
def test_05_subthreshold_curvature_suppression_vs_logloss_explosion() -> None:
    tau = 0.10
    kappa = 24.0
    delta = 1e-4
    eta_sub = 0.002
    # Log-loss Bregman gain explodes as 0.5 * delta^2 / (eta*(1-eta))
    ll_gain = float(
        eta_sub * np.log((eta_sub + delta) / eta_sub)
        + (1.0 - eta_sub) * np.log((1.0 - eta_sub - delta) / (1.0 - eta_sub))
    )
    # Wait: expected log-loss gain when true prob is eta2 = eta_sub + delta
    eta2 = eta_sub + delta
    ll_breg = float(
        eta2 * np.log(eta2 / eta_sub) + (1.0 - eta2) * np.log((1.0 - eta2) / (1.0 - eta_sub))
    )
    cz_breg = float(
        (
            eta2 * schervish_collision_proper_score(np.array([1]), np.array([eta2]), tau, kappa)
            + (1.0 - eta2) * schervish_collision_proper_score(np.array([0]), np.array([eta2]), tau, kappa)
            - eta2 * schervish_collision_proper_score(np.array([1]), np.array([eta_sub]), tau, kappa)
            - (1.0 - eta2)
            * schervish_collision_proper_score(np.array([0]), np.array([eta_sub]), tau, kappa)
        )[0]
    )
    assert ll_gain < 0.0
    assert ll_breg > 1000.0 * cz_breg


# 6
def test_06_waterfill_neyman_exact_budget_and_bounds() -> None:
    pop = np.array([5, 50, 200, 1500, 3000])
    disp = np.array([2.0, 1.5, 0.8, 0.05, 0.001])
    alloc = waterfill_neyman(pop, disp, n_pilot=400, min_per_stratum=2)
    assert int(np.sum(alloc)) == 400
    assert np.all(alloc >= np.minimum(pop, 2))
    assert np.all(alloc <= pop)
    assert alloc[0] == 5  # Census on tiny high-dispersion stratum


# 7
def test_07_ht_weights_sum_to_population_and_census_stratum_zero_v2() -> None:
    rng = np.random.default_rng(19)
    n = 2000
    p_base = rng.uniform(0.005, 0.25, size=n)
    y = rng.binomial(1, p_base)
    pilot = sample_pilot(y, p_base, pilot_fraction=0.20, mode="collision_neyman", tau_star=0.08, seed=3)
    assert pilot.indices.size == 400
    assert np.isclose(np.sum(pilot.ht_weights), float(n), rtol=1e-9)

    full_pilot = sample_pilot(y, p_base, pilot_fraction=1.0, mode="full", tau_star=0.08, seed=3)
    psi = rng.normal(0.05, 0.2, size=n)
    g_hat, se_tot = two_phase_ht_mean_and_se(psi, full_pilot)
    assert np.isclose(g_hat, float(np.mean(psi)), rtol=1e-12)
    # Under census sampling, V_II == 0 so se == sqrt(V_I) == sd(psi)/sqrt(N)
    assert np.isclose(se_tot, float(np.std(psi, ddof=1) / np.sqrt(n)), rtol=1e-10)


# 8
def test_08_two_phase_variance_reduces_identically_to_srs() -> None:
    rng = np.random.default_rng(23)
    n_tot = 5000
    n_p = 500
    psi = rng.normal(0.02, 0.5, size=n_p)
    pilot_srs = PilotSample(
        indices=np.arange(n_p),
        strata=np.zeros(n_p, dtype=int),
        ht_weights=np.full(n_p, n_tot / n_p, dtype=float),
        pop_counts={0: n_tot},
        sample_counts={0: n_p},
        n_total=n_tot,
        high_signal_fp_share_of_controls=0.1,
    )
    g_hat, se_tot = two_phase_ht_mean_and_se(psi, pilot_srs)
    expected_se = float(np.std(psi, ddof=1) / np.sqrt(n_p))
    assert np.isclose(g_hat, float(np.mean(psi)), rtol=1e-12)
    assert np.isclose(se_tot, expected_se, rtol=1e-12)


# 9
def test_09_hsfp_concentration_ordering() -> None:
    rng = np.random.default_rng(7)
    n = 8000
    is_cz = rng.uniform(size=n) < 0.08
    p_base = np.where(is_cz, rng.uniform(0.06, 0.16, size=n), rng.uniform(0.001, 0.012, size=n))
    y = rng.binomial(1, p_base)
    p_srs = sample_pilot(y, p_base, pilot_fraction=0.15, mode="srs", tau_star=0.10, seed=5)
    p_gn = sample_pilot(y, p_base, pilot_fraction=0.15, mode="global_neyman", tau_star=0.10, seed=5)
    p_cz = sample_pilot(y, p_base, pilot_fraction=0.15, mode="collision_neyman", tau_star=0.10, seed=5)
    assert (
        p_cz.high_signal_fp_share_of_controls
        > p_gn.high_signal_fp_share_of_controls
        > p_srs.high_signal_fp_share_of_controls
    )


# 10
def test_10_holm_step_down_properties() -> None:
    raw = [0.008, 0.021, 0.015, 0.40]
    adj = holm_adjust(raw)
    assert len(adj) == 4
    assert all(0.0 <= p <= 1.0 for p in adj)
    # Smallest p (0.008) multiplied by 4 = 0.032
    assert np.isclose(adj[0], 0.032)
    # Monotonicity along sorted raw p-values
    order = np.argsort(raw)
    sorted_adj = [adj[i] for i in order]
    assert sorted_adj == sorted(sorted_adj)
    assert holm_adjust([]) == []


# 11
def test_11_v2_wrappers_are_bit_exact_with_v1() -> None:
    rng = np.random.default_rng(123)
    n = 3000
    in_cz = rng.uniform(size=n) < 0.08
    z_cz = rng.normal(size=n)
    z_tn = rng.normal(size=n)
    y = rng.binomial(1, expit(np.where(in_cz, -2.7 + 1.8 * z_cz, -4.1 + 1.05 * z_tn)))
    df = pd.DataFrame(
        {
            "b1": in_cz.astype(float) + rng.normal(0, 0.05, n),
            "b2": rng.normal(0, 1, n),
            "ch_cz": np.where(in_cz, z_cz, rng.normal(0, 1, n)),
            "ch_tn": np.where(~in_cz, z_tn, rng.normal(0, 1, n)),
            "null_1": rng.normal(0, 1, n),
        }
    )
    base_cols = ["b1", "b2"]
    chs = [
        ChannelDef("ch_cz", ("ch_cz",), 1.0),
        ChannelDef("ch_tn", ("ch_tn",), 1.0),
        ChannelDef("null_1", ("null_1",), 1.0, is_null_stress=True),
    ]
    ct = ClinicalContract(1.0, 9.0, 0.45, 0.02)

    oof1 = compute_oof_stage1(df.iloc[:1800][base_cols], y[:1800], n_folds=4, seed=31)
    oof2 = compute_oof_stage1_v2(df.iloc[:1800][base_cols], y[:1800], n_folds=4, seed=31, n_jobs=1)
    assert np.max(np.abs(oof1 - oof2)) == 0.0

    d1 = run_sequential_discovery(
        df.iloc[:1800], y[:1800], base_cols, chs, oof1, ct, method="collision_zone", pilot_fraction=0.25, budget=2.0, seed=41
    )
    d2 = run_sequential_discovery_v2(
        df.iloc[:1800], y[:1800], base_cols, chs, oof2, ct, method="collision_zone", pilot_fraction=0.25, budget=2.0, n_jobs=1, seed=41
    )
    assert d1.selected_ids == d2.outcome.selected_ids
    for k in d1.step1_scores:
        assert np.isclose(d1.step1_scores[k], d2.outcome.step1_scores[k], rtol=1e-12)

    s1_cache = fit_stage1_predictions(
        df.iloc[:1800], y[:1800], df.iloc[1800:2400], y[1800:2400], [df.iloc[2400:]], base_cols, seed=51
    )
    ev1 = evaluate_stage2_multi_test(
        df.iloc[:1800],
        y[:1800],
        df.iloc[1800:2400],
        y[1800:2400],
        [(df.iloc[2400:], y[2400:])],
        base_cols,
        ["ch_cz"],
        oof1,
        ct,
        stage2_mode="fluid_backbone",
        precomputed_stage1=s1_cache,
        seed=51,
    )[0]
    ev2 = evaluate_stage2_multi_test_v2(
        df.iloc[:1800],
        y[:1800],
        df.iloc[1800:2400],
        y[1800:2400],
        [(df.iloc[2400:], y[2400:])],
        base_cols,
        ["ch_cz"],
        oof2,
        ct,
        stage2_mode="fluid_backbone",
        precomputed_stage1=s1_cache,
        n_jobs=1,
        seed=51,
    )[0]
    assert np.isclose(ev1["nb_voi_auto"], ev2["nb_voi_auto"], rtol=1e-12)
    assert np.isclose(ev1["learned_beta_cz"], ev2["learned_beta_cz"], rtol=1e-12)


# 12
def test_12_capacity_matched_null_gain_is_nonpositive_on_average() -> None:
    rng = np.random.default_rng(555)
    null_gains: list[float] = []
    ct = ClinicalContract(1.0, 9.0, 0.45, 0.02)
    base_cols = [f"b_{k}" for k in range(8)]
    for r in range(10):
        n = 2500
        xb = rng.normal(size=(n, 8))
        y = rng.binomial(1, expit(-2.6 + 0.9 * xb[:, 0] - 0.5 * xb[:, 1]))
        data = {f"b_{k}": xb[:, k] for k in range(8)}
        for j in (1, 2, 3):
            data[f"null_{j}"] = rng.normal(size=n)
        df = pd.DataFrame(data)
        chs = [
            ChannelDef(f"null_{j}", (f"null_{j}",), 1.0, is_null_stress=True) for j in (1, 2, 3)
        ]
        oof = compute_oof_stage1_v2(df[base_cols], y, n_folds=4, seed=10 + r, n_jobs=1)
        det = run_sequential_discovery_v2(
            df,
            y,
            base_cols,
            chs,
            oof,
            ct,
            method="collision_zone",
            pilot_fraction=0.25,
            budget=3.0,
            n_jobs=1,
            seed=30 + r,
        )
        null_gains.extend(det.pure_null_step1_gains)
    # Cross-fitted finite-sample GBDT variance makes null gains strictly <= 0 in expectation
    assert float(np.mean(null_gains)) <= 0.0


# 13
def test_13_monotone_ess_kappa_solver_accuracy() -> None:
    rng = np.random.default_rng(42)
    tau = 0.0625
    p = rng.beta(1.2, 18.0, size=4000)
    dkl = bernoulli_kl_from_tau(p, tau)
    k_star = solve_ess_kappa(dkl, kappa_max=25.0, min_effective_cz_samples=300.0)
    if 0.0 < k_star < 25.0:
        mass = float(np.sum(np.exp(-2.0 * k_star * dkl)))
        assert abs(mass - 300.0) < 2.0


# 14
def test_14_union_collision_gate_symmetry_and_peak() -> None:
    tau = 0.08
    f_tau = logit(tau)
    f_low = logit(0.002)
    g1 = union_collision_gate(np.array([f_tau, f_low]), np.array([f_low, f_tau]), tau, kappa=16.0)
    g_sym = union_collision_gate(np.array([f_low, f_tau]), np.array([f_tau, f_low]), tau, kappa=16.0)
    assert np.allclose(g1, g_sym)
    assert np.allclose(g1, 1.0, atol=1e-12)
    g_far = union_collision_gate(np.array([f_low]), np.array([f_low]), tau, kappa=16.0)
    assert 0.0 < float(g_far[0]) < 0.07


# 15
def test_15_leaf_starvation_newton_bound_shrinks_to_zero() -> None:
    rng = np.random.default_rng(91)
    lam = 3.5
    p = rng.uniform(0.01, 0.20, size=50)
    y = rng.binomial(1, p)
    for scale in (1.0, 1e-2, 1e-5):
        pi = np.full(50, scale)
        grad = pi * (p - y)
        hess = (pi**2) * p * (1.0 - p)
        gamma_hat = -float(np.sum(grad)) / (float(np.sum(hess)) + lam)
        theo_bound = float(np.sum(pi)) / lam
        assert abs(gamma_hat) <= theo_bound + 1e-12
    assert abs(gamma_hat) < 1e-3


# 16
def test_16_fluid_backbone_regime_detector_damps_homogeneous_and_opens_heterogeneous() -> None:
    rng = np.random.default_rng(2026)
    tau = 0.08
    n = 6000
    x0 = rng.normal(size=n)
    ell1 = -3.4 + 1.15 * x0
    z1 = rng.normal(size=n)
    x_nuis = rng.normal(size=(n, 3))
    x_all = np.column_stack([x0, x_nuis, z1]).astype(np.float32)
    z_s2 = z1.reshape(-1, 1).astype(np.float32)

    # Homogeneous physiology across all risk strata -> global backbone suffices, beta_cz damps low
    y_hom = rng.binomial(1, expit(ell1 + 0.85 * z1))
    m_hom = FluidCollisionBackboneClassifierV2(tau_star=tau, min_cz_ess=240.0, n_jobs=1, seed=7).fit(
        x_all, z_s2, y_hom, ell1
    )

    # Subthreshold slope inversion -> specialist residual opens beta_cz sharply
    cz_g = expit((ell1 - logit(0.04)) / 0.25)
    y_inv = rng.binomial(1, expit(ell1 + cz_g * 1.65 * z1 - (1.0 - cz_g) * 1.15 * z1))
    m_inv = FluidCollisionBackboneClassifierV2(tau_star=tau, min_cz_ess=240.0, n_jobs=1, seed=7).fit(
        x_all, z_s2, y_inv, ell1
    )
    assert m_hom.beta_cz_ < 0.45
    assert m_inv.beta_cz_ > 2.0 * m_hom.beta_cz_


# 17
def test_17_agniel_deconfounder_null_calibration_guarantee() -> None:
    rng = np.random.default_rng(77)
    n = 4000
    xb = rng.normal(size=n)
    u = rng.normal(size=n)
    y = rng.binomial(1, expit(-2.2 + 0.7 * xb + 1.9 * u))
    drawn = rng.binomial(1, expit(-1.4 + 0.5 * xb + 2.1 * u))
    z_raw = 0.75 * xb + rng.normal(size=n)
    z_obs = np.where(drawn == 1, z_raw, np.nan)

    df_b = pd.DataFrame({"xb": xb})
    df_s = pd.DataFrame({"lab": z_obs})
    dec = AgnielDeconfounder().fit(df_b.iloc[:2500], df_s.iloc[:2500])
    clean = dec.transform(df_b.iloc[2500:], df_s.iloc[2500:], seed=12)
    resid = clean["lab"].to_numpy() - 0.75 * xb[2500:]
    assert abs(float(np.corrcoef(resid, y[2500:])[0, 1])) < 0.06
    assert abs(float(np.corrcoef(resid, drawn[2500:])[0, 1])) < 0.06


# 18
def test_18_agniel_deconfounder_preserves_true_biological_signal_on_measured() -> None:
    rng = np.random.default_rng(88)
    n = 3000
    xb = rng.normal(size=n)
    bio = rng.normal(size=n)
    y = rng.binomial(1, expit(-2.0 + 0.5 * xb + 1.6 * bio))
    drawn = rng.binomial(1, 0.75, size=n)
    z_obs = np.where(drawn == 1, 0.4 * xb + bio, np.nan)

    df_b = pd.DataFrame({"xb": xb})
    df_s = pd.DataFrame({"lab": z_obs})
    dec = AgnielDeconfounder().fit(df_b.iloc[:2000], df_s.iloc[:2000])
    clean = dec.transform(df_b.iloc[2000:], df_s.iloc[2000:], seed=4)
    # Measured rows must remain untouched and strongly correlated with outcome Y
    meas_te = drawn[2000:] == 1
    assert np.allclose(clean["lab"].to_numpy()[meas_te], z_obs[2000:][meas_te])
    assert float(np.corrcoef(clean["lab"].to_numpy()[meas_te], y[2000:][meas_te])[0, 1]) > 0.25


# 19
def test_19_empirical_martingale_voi_nonnegative_and_peaks_at_threshold() -> None:
    rng = np.random.default_rng(101)
    ct = ClinicalContract(1.0, 15.0, 0.30, 0.03)  # tau* = 0.0625
    skewed_deltas = rng.exponential(scale=0.8, size=3000) - 0.8
    q_grid = np.quantile(skewed_deltas, np.linspace(0.02, 0.98, 33))[None, :]
    p1 = np.array([0.001, 0.0625, 0.50])
    voi = empirical_martingale_voi(p1, np.zeros(3, dtype=int), q_grid, ct)
    assert np.all(voi >= 0.0)
    assert voi[1] > 15.0 * voi[0]
    assert voi[1] > 5.0 * voi[2]


# 20
def test_20_conformal_bh_controls_fdr_on_exchangeable_splits() -> None:
    rng = np.random.default_rng(202)
    fdps: list[float] = []
    fdr_target = 0.35
    for r in range(40):
        n_cal, n_te = 2000, 2000
        s_all = rng.normal(size=n_cal + n_te)
        p_true = expit(-2.2 + 1.9 * s_all)
        y_all = rng.binomial(1, p_true)
        scores = p_true + rng.normal(0, 0.03, size=n_cal + n_te)
        res = cfbh_selection(
            y_all[:n_cal], scores[:n_cal], y_all[n_cal:], scores[n_cal:], fdr_target=fdr_target, seed=r
        )
        fdps.append(res["fdp"])
    assert float(np.mean(fdps)) <= fdr_target + 0.04


# 21
def test_21_davison_hinkley_bootstrap_never_returns_exact_zero_pvalue() -> None:
    rng = np.random.default_rng(303)
    ct = ClinicalContract(1.0, 15.0, 0.30, 0.03)
    y = rng.binomial(1, 0.08, size=1500)
    p_good = np.clip(0.05 + 0.50 * y, 1e-4, 0.999)
    p_bad = np.full(1500, 0.02)
    q = np.zeros(1500, dtype=bool)
    res = paired_bootstrap_nb_diff_v2(y, p_good, q, p_bad, q, ct, n_boot=2000, seed=9)
    assert res["diff_mean"] > 20.0
    assert res["p_two_sided_dh"] > 0.0
    assert np.isclose(res["p_two_sided_dh"], 2.0 / 2001.0)


# 22
def test_22_superadditivity_interaction_test_recovers_zero_under_pure_additivity() -> None:
    rng = np.random.default_rng(404)
    n = 2000
    u_base = rng.normal(2.0, 1.0, size=n)
    eff_cz = rng.normal(0.8, 0.5, size=n)
    eff_fl = rng.normal(0.6, 0.5, size=n)
    u_glob = u_base + rng.normal(0.2, 0.3, size=n)
    u_cz_glob = u_base + eff_cz
    u_glob_fl = u_base + eff_fl
    # Purely additive relative to Base: CZ_Fluid - Base == (CZ_Glob - Base) + (Glob_Fluid - Base)
    u_cz_fl = u_base + eff_cz + eff_fl + rng.normal(0.0, 0.2, size=n)
    inter = paired_bootstrap_interaction(u_base, u_glob, u_cz_glob, u_glob_fl, u_cz_fl, n_boot=2000, seed=11)
    assert inter["interaction_vs_base_ci95_low"] < 0.0 < inter["interaction_vs_base_ci95_high"]
    assert inter["interaction_vs_base_p_dh"] > 0.15


# 23
def test_23_calibration_metrics_and_prior_odds_intercept_rescue() -> None:
    rng = np.random.default_rng(505)
    n = 6000
    x = rng.normal(size=n)
    # Source site logit: intercept -2.5 (prev ~8%), target site shifted by -0.60 log-odds (prev ~4.7%)
    p_src_model = expit(-2.5 + 0.9 * x)
    y_tgt = rng.binomial(1, expit(-3.1 + 0.9 * x))

    cal_before = compute_calibration_metrics(y_tgt, p_src_model, n_boot=100, seed=1)
    assert float(cal_before["citl"]) < -0.40  # type: ignore[arg-type]
    assert float(cal_before["ici_loess"]) > 0.025  # type: ignore[arg-type]

    # Recalibrate using only scalar prevalences
    pi_src = float(np.mean(p_src_model))
    pi_tgt = float(np.mean(y_tgt))
    p_after, offset = prior_odds_recalibrate_probabilities(p_src_model, pi_src, pi_tgt)
    assert offset < -0.40
    cal_after = compute_calibration_metrics(y_tgt, p_after, n_boot=100, seed=2)
    assert abs(float(cal_after["citl"])) < abs(float(cal_before["citl"])) * 0.35  # type: ignore[arg-type]
    assert float(cal_after["ici_loess"]) < float(cal_before["ici_loess"]) * 0.50  # type: ignore[arg-type]

    # Exact marginal-risk CITL shift makes CITL == 0 to machine precision
    delta_exact = solve_citl_intercept(pi_tgt, p_src_model)
    p_exact = expit(logit(p_src_model) + delta_exact)
    assert abs(solve_citl_intercept(y_tgt, p_exact)) < 1e-7


# 24
def test_24_covert_afa_frontier_snb_and_clopper_pearson() -> None:
    cp = clopper_pearson_ci(33, 1000)
    assert 0.022 < cp["ci95_low"] < 0.033 < cp["ci95_high"] < 0.047

    ct = ClinicalContract(1.0, 15.0, 0.30, 0.03)
    rng = np.random.default_rng(606)
    n = 1500
    b1 = rng.normal(size=n)
    z1 = rng.normal(size=n)
    z_null = rng.normal(size=n)
    y = rng.binomial(1, expit(-2.7 + 0.7 * b1 + 1.5 * z1))
    df = pd.DataFrame({"b1": b1, "z1": z1, "z_null": z_null})
    chs = [ChannelDef("z1", ("z1",), 1.0), ChannelDef("z_null", ("z_null",), 1.0)]

    tr_df, ca_df, te_df = df.iloc[:800], df.iloc[800:1150], df.iloc[1150:]
    y_tr, y_ca, y_te = y[:800], y[800:1150], y[1150:]
    oof = compute_oof_stage1_v2(tr_df[["b1"]], y_tr, n_folds=4, seed=1, n_jobs=1)
    s1_cache = fit_stage1_predictions(tr_df, y_tr, ca_df, y_ca, [te_df], ["b1"], seed=2)
    traces: list[dict[str, np.ndarray]] = []
    ev = evaluate_covert_greedy_cmi_afa(
        tr_df,
        y_tr,
        ca_df,
        y_ca,
        [(te_df, y_te)],
        ["b1"],
        chs,
        oof,
        s1_cache,
        ct,
        budget=1.5,
        out_routed_traces=traces,
        n_masks_per_row=3,
        n_jobs=1,
        seed=3,
    )[0]
    assert ev["auroc_full_s2"] > 0.72
    counts = clinical_impact_counts_per_1k(
        y_te, traces[0]["p_auto_te"], traces[0]["auto_te"], ct, y_cal=y_ca, p_cal=s1_cache[0], seed=4
    )
    assert np.isclose(counts["snb"], standardized_net_benefit(counts["nb_per_1k"], float(np.mean(y_te))))

    fr = compute_tau_kappa_frontier(s1_cache[0], ct.tau_star, [0.0, 10.0, 20.0], min_cz_ess=80.0)
    assert fr["active_hessian_mass"][0] > fr["active_hessian_mass"][1] > fr["active_hessian_mass"][2]
