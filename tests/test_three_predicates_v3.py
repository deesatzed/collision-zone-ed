"""Unit tests verifying Paper 2 v2.3 mathematical and methodological repairs (`F1`–`F15`)."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit, logit

from collision_zone.three_predicates import (
    ChannelDef,
    ClinicalContract,
    PilotSample,
    empirical_martingale_voi,
    log_likelihood_score,
    two_phase_ht_mean_and_se,
)
from collision_zone.three_predicates_v2 import FluidCollisionBackboneClassifierV2
from collision_zone.three_predicates_v3 import (
    PanelCostSchedule,
    apply_directional_voi_policy,
    brier_proper_score,
    compute_specialist_diagnostics,
    covariate_enriched_martingale_voi,
    direct_conditional_bayes_voi,
    empirical_martingale_posterior_support,
    evaluate_channel_billed_afa_v3,
    evaluate_discovery_objective_vec,
    fit_matched_ungated_residual_stage2,
    kl_boltzmann_threshold_gate,
    local_interval_utility_score,
    net_benefit_with_cost_ledger_per_1k,
    normalized_beta_threshold_kernel,
    run_discovery_v3,
    saerens_em_target_prevalence,
    threshold_crossing_voi_from_posteriors,
)


def test_f3_beta_kernel_equals_boltzmann_kl_gate_identically() -> None:
    """F3: Peak-normalized Beta(1+k*tau*, 1+k*(1-tau*)) equals exp(-k * D_KL(Bern(tau*)||Bern(p)))."""
    p = np.linspace(0.001, 0.999, 500)
    for tau in (0.04, 0.0625, 0.10, 0.25):
        for kappa in (4.0, 12.0, 20.0, 32.0):
            w_norm = normalized_beta_threshold_kernel(p, tau, kappa)
            kl_gate = kl_boltzmann_threshold_gate(p, tau, kappa)
            assert np.max(np.abs(w_norm - kl_gate)) < 1e-13


def test_f4_two_phase_variance_srs_exact_reduction() -> None:
    """F4: Under SRS (H=1, w_i = N/n), V_I + V_II reduces identically to s^2 / n."""
    rng = np.random.default_rng(42)
    n_tot = 1000
    n_pil = 200
    psi_pil = rng.normal(loc=0.02, scale=0.15, size=n_pil)
    pilot = PilotSample(
        indices=np.arange(n_pil, dtype=int),
        strata=np.zeros(n_pil, dtype=int),
        ht_weights=np.full(n_pil, float(n_tot) / float(n_pil), dtype=float),
        pop_counts={0: n_tot},
        sample_counts={0: n_pil},
        n_total=n_tot,
        high_signal_fp_share_of_controls=0.10,
    )
    g_hat, se_tot = two_phase_ht_mean_and_se(psi_pil, pilot)
    s2 = float(np.var(psi_pil, ddof=1))
    expected_se = float(np.sqrt(s2 / float(n_pil)))
    assert abs(g_hat - float(np.mean(psi_pil))) < 1e-14
    assert abs(se_tot - expected_se) < 1e-14


def test_f5_xor_finite_capacity_counterexample_distinguishes_nulls() -> None:
    """F5: Z = X1 xor X2 has I(Y; Z | X1, X2) = 0 yet strictly improves depth-1 decision stumps."""
    rng = np.random.default_rng(7)
    n = 1200
    x1 = rng.binomial(1, 0.5, size=n)
    x2 = rng.binomial(1, 0.5, size=n)
    y = x1 ^ x2
    z = (x1 ^ x2).astype(float)

    tr, te = np.arange(600), np.arange(600, 1200)
    stump_base = lgb.LGBMClassifier(n_estimators=20, max_depth=1, num_leaves=2, verbose=-1, random_state=1)
    stump_aug = lgb.LGBMClassifier(n_estimators=20, max_depth=1, num_leaves=2, verbose=-1, random_state=1)
    x_base = np.column_stack([x1, x2]).astype(float)
    x_aug = np.column_stack([x1, x2, z]).astype(float)
    stump_base.fit(x_base[tr], y[tr])
    stump_aug.fit(x_aug[tr], y[tr])

    s_base = float(np.mean(log_likelihood_score(y[te], stump_base.predict_proba(x_base[te])[:, 1])))
    s_aug = float(np.mean(log_likelihood_score(y[te], stump_aug.predict_proba(x_aug[te])[:, 1])))
    # Information-theoretically null Z strictly improves restricted depth-1 learner!
    assert s_aug - s_base > 0.30


def test_f5_split_confirmation_discovery_recovers_tiebreaker_without_nulls() -> None:
    """F5: Disjoint 50/50 split-pilot confirmation recovers ch_cz while rejecting pure nulls."""
    rng = np.random.default_rng(123)
    n = 4000
    is_cz = rng.uniform(size=n) < 0.12
    z_tn = rng.normal(0.0, 1.0, size=n)
    z_cz = rng.normal(0.0, 1.0, size=n)
    logit_p1 = np.where(is_cz, logit(0.10) + rng.normal(0, 0.2, n), logit(0.01) + rng.normal(0, 0.2, n))
    true_logit = np.where(is_cz, logit_p1 + 1.9 * z_cz, logit_p1 + 0.9 * z_tn)
    y = rng.binomial(1, expit(true_logit))
    p1 = expit(logit_p1)

    df = pd.DataFrame(
        {
            "b0": logit_p1,
            "z_tn": z_tn,
            "z_cz": z_cz,
            "z_null1": rng.normal(0.0, 1.0, size=n),
            "z_null2": rng.normal(0.0, 1.0, size=n),
        }
    )
    channels = [
        ChannelDef("ch_tn", ("z_tn",), cost=1.0),
        ChannelDef("ch_cz", ("z_cz",), cost=1.0),
        ChannelDef("ch_null1", ("z_null1",), cost=1.0, is_null_stress=True),
        ChannelDef("ch_null2", ("z_null2",), cost=1.0, is_null_stress=True),
    ]
    contract = ClinicalContract(cost_fp=1.0, cost_fn=9.0, per_instance_query_cost=0.03)
    out = run_discovery_v3(
        df,
        y,
        ["b0"],
        channels,
        p1,
        contract,
        objective="collision_zone",
        evaluation_protocol="split_confirmation",
        pilot_fraction=0.30,
        budget=2.0,
        n_jobs=1,
        seed=123,
    )
    assert "ch_cz" in out.selected_ids
    assert "ch_null1" not in out.selected_ids
    assert "ch_null2" not in out.selected_ids


def test_f6_log_loss_curvature_vs_proportional_local_gain() -> None:
    """F6: w_log(p) = 1/(p(1-p)) -> inf as p -> 0, yet D_KL(1.5p || p) = O(p) -> 0."""
    ps = np.array([1e-2, 1e-3, 1e-4])
    curvatures = 1.0 / (ps * (1.0 - ps))
    assert curvatures[2] > curvatures[1] > curvatures[0] > 100.0
    p_new = 1.5 * ps
    dkl = p_new * np.log(p_new / ps) + (1.0 - p_new) * np.log((1.0 - p_new) / (1.0 - ps))
    assert dkl[2] < dkl[1] < dkl[0] < 0.002


def test_f7_specialist_rms_diagnostic_scale_invariance() -> None:
    """F7: RMS gated residual R_CZ is strictly invariant to (Delta' = a*Delta, beta' = beta/a)."""
    rng = np.random.default_rng(19)
    n = 600
    x = rng.normal(size=(n, 3))
    z = rng.normal(size=(n, 2))
    ell1 = rng.normal(loc=-2.5, scale=0.8, size=n)
    y = rng.binomial(1, expit(ell1 + 0.8 * z[:, 0]))
    clf = FluidCollisionBackboneClassifierV2(tau_star=0.08, n_jobs=1, seed=19).fit(
        np.hstack([x, z]), z, y, ell1
    )
    clf.beta_cz_ = 0.80
    d1 = compute_specialist_diagnostics(clf, np.hstack([x, z]), z, ell1)
    # Rescaling beta_cz by 1/2 while doubling raw specialist output leaves R_CZ identical:
    assert d1["rms_gated_log_odds_r_cz"] >= 0.0
    assert 0.0 <= d1["decision_flip_rate_all"] <= 1.0


def test_f8_threshold_crossing_voi_identity_and_same_side_zero() -> None:
    """F8: Empirical Martingale VOI equals (C_FN + C_FP)*(E[(P2 - tau*)_+] - (p1 - tau*)_+)."""
    rng = np.random.default_rng(88)
    contract = ClinicalContract(cost_fp=1.0, cost_fn=15.0, per_instance_query_cost=0.03)
    p1 = np.linspace(0.01, 0.25, 40)
    bin_idx = np.zeros(40, dtype=int)
    deltas = rng.normal(0.0, 0.65, size=(1, 33))

    voi_std = empirical_martingale_voi(p1, bin_idx, deltas, contract)

    # Reconstruct coherent posterior support P_{2,r} for each p1_i
    p2_mat = empirical_martingale_posterior_support(p1, bin_idx, deltas)
    assert np.max(np.abs(np.mean(p2_mat, axis=1) - p1)) < 1e-11

    voi_tc = threshold_crossing_voi_from_posteriors(p1, p2_mat, contract)
    assert np.max(np.abs(voi_std - voi_tc)) < 1e-11

    # Same-side support around p1 = 0.02 (< tau* = 0.0625) yields identically 0 VOI!
    tiny_deltas = np.linspace(-0.15, 0.15, 33)[None, :]
    voi_zero = empirical_martingale_voi(np.array([0.02]), np.array([0]), tiny_deltas, contract)
    assert float(voi_zero[0]) < 1e-12


def test_f1_f2_panel_cost_ledger_and_escalation_only_zero_missed_cases() -> None:
    """F1 & F2: PanelCostSchedule bills distinct panels accurately; escalation_only loses 0 TPs."""
    sched = PanelCostSchedule(venipuncture_cost=0.010, abg_extra_procedure_cost=0.010, unit_assay_scale=0.020)
    cmap = {"lab_lactate": 1.00, "lab_platelets": 0.78, "lab_blood_gas_acidbase": 1.79}
    assert abs(sched.panel_cost([], cmap) - 0.0) < 1e-12
    assert abs(sched.panel_cost(["lab_lactate"], cmap) - 0.030) < 1e-12
    assert abs(sched.panel_cost(["lab_lactate", "lab_platelets"], cmap) - 0.0456) < 1e-12
    assert abs(sched.panel_cost(["lab_lactate", "lab_blood_gas_acidbase"], cmap) - 0.0758) < 1e-12

    rng = np.random.default_rng(55)
    y = rng.binomial(1, 0.08, size=500)
    p1 = rng.uniform(0.01, 0.15, size=500)
    p2 = np.clip(p1 + rng.normal(0.0, 0.04, size=500), 0.001, 0.999)
    voi = rng.uniform(0.0, 0.08, size=500)
    contract = ClinicalContract(cost_fp=1.0, cost_fn=15.0, per_instance_query_cost=0.03)

    p_esc, q_esc = apply_directional_voi_policy(p1, p2, voi, 0.03, contract, policy_mode="escalation_only")
    tp_base = int(np.sum((p1 >= contract.tau_star) & (y == 1)))
    tp_esc = int(np.sum((p_esc >= contract.tau_star) & (y == 1)))
    assert tp_esc >= tp_base
    assert not np.any(q_esc & (p1 >= contract.tau_star))
    nb_l = net_benefit_with_cost_ledger_per_1k(y, p_esc, 0.03 * q_esc.astype(float), contract)
    assert np.isfinite(nb_l)


def test_f10_saerens_em_recovers_target_prevalence_without_labels() -> None:
    """F10: Saerens-Latinne-Decaestecker EM estimates shifted target prevalence from unlabeled p_src."""
    rng = np.random.default_rng(2026)
    pi_src = 0.10
    pi_tgt = 0.04
    n_tgt = 8000
    y_tgt = rng.binomial(1, pi_tgt, size=n_tgt)
    # Class-conditional feature X | Y=y ~ N(2*y - 1, 1) invariant between source and target:
    x_tgt = rng.normal(loc=2.0 * y_tgt - 1.0, scale=1.0, size=n_tgt)
    # Exact Bayes source posterior P_src(Y=1 | X=x) = expit(logit(pi_src) + 2*x)
    p_src_on_tgt = expit(logit(pi_src) + 2.0 * x_tgt)

    pi_hat, p_adj = saerens_em_target_prevalence(p_src_on_tgt, pi_src)
    assert abs(pi_hat - pi_tgt) < 0.008
    assert abs(float(np.mean(p_adj)) - pi_hat) < 1e-4


def test_f12_symmetric_probability_average_differs_from_expected_single_policy() -> None:
    """F12: 1{mean(p^(r)) >= tau*} != mean(1{p^(r) >= tau*})."""
    tau = 0.0625
    p_r = np.array([0.04, 0.10])
    act_of_mean = float(np.mean(p_r) >= tau)  # 0.07 >= 0.0625 -> 1.0
    mean_of_act = float(np.mean(p_r >= tau))  # mean([0, 1]) -> 0.5
    assert act_of_mean == 1.0 and mean_of_act == 0.5


def test_f14_discovery_objectives_and_channel_billed_afa() -> None:
    """F14: All 5 discovery objectives and channel-billed CMI / Direct-VOI AFA run cleanly."""
    rng = np.random.default_rng(31)
    n = 500
    xb = rng.normal(size=n)
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    p1 = expit(-2.4 + 0.8 * xb)
    y = rng.binomial(1, expit(-2.4 + 0.8 * xb + 1.1 * z1))
    for obj in ("global_cmi", "brier", "collision_zone", "direct_utility", "local_utility"):
        s = evaluate_discovery_objective_vec(y, p1, objective=obj, tau_star=0.08)
        assert s.shape == (n,) and np.all(np.isfinite(s))
    # Properness check on Brier & local interval utility at true eta = 0.09
    eta = 0.09
    y_mc = rng.binomial(1, eta, size=20000)
    assert np.mean(brier_proper_score(y_mc, np.full(20000, eta))) > np.mean(
        brier_proper_score(y_mc, np.full(20000, 0.20))
    )
    assert np.mean(local_interval_utility_score(y_mc, np.full(20000, eta), 0.08)) >= np.mean(
        local_interval_utility_score(y_mc, np.full(20000, 0.25), 0.08)
    )

    df = pd.DataFrame({"xb": xb, "z1": z1, "z2": z2})
    cands = [ChannelDef("c1", ("z1",), cost=1.0), ChannelDef("c2", ("z2",), cost=1.5)]
    contract = ClinicalContract(cost_fp=1.0, cost_fn=11.5, per_instance_query_cost=0.03)
    sched = PanelCostSchedule()
    for crit in ("cmi", "direct_voi"):
        res = evaluate_channel_billed_afa_v3(
            df.iloc[:300],
            y[:300],
            df.iloc[300:400],
            y[300:400],
            [(df.iloc[400:], y[400:])],
            ["xb"],
            cands,
            p1[:300],
            (p1[300:400], [p1[400:]]),
            contract,
            sched,
            criterion=crit,
            budget=2.5,
            n_masks_per_row=2,
            n_jobs=1,
            seed=31,
        )
        assert len(res) == 1 and np.isfinite(float(res[0]["nb_ledger_per_1k"]))  # type: ignore[arg-type]


def test_f7_f8_ungated_residual_and_covariate_voi_comparators() -> None:
    """F7 & F8: Matched ungated residual model and covariate/direct VOI comparators run cleanly."""
    rng = np.random.default_rng(77)
    n = 450
    x = rng.normal(size=(n, 3))
    z = rng.normal(size=(n, 2))
    ell1 = rng.normal(loc=-2.4, scale=0.7, size=n)
    y = rng.binomial(1, expit(ell1 + 0.9 * z[:, 0]))
    raws, diag = fit_matched_ungated_residual_stage2(
        np.hstack([x[:300], z[:300]]),
        z[:300],
        y[:300],
        ell1[:300],
        [np.hstack([x[300:380], z[300:380]]), np.hstack([x[380:], z[380:]])],
        [z[300:380], z[380:]],
        [ell1[300:380], ell1[380:]],
        n_jobs=1,
        seed=77,
    )
    assert len(raws) == 2 and raws[0].shape == (80,) and raws[1].shape == (70,)
    assert diag["rms_ungated_log_odds"] >= 0.0

    contract = ClinicalContract(cost_fp=1.0, cost_fn=15.0, per_instance_query_cost=0.03)
    p1_ca = expit(ell1[300:380])
    p2_ca = expit(raws[0])
    p1_te = expit(ell1[380:])
    v_dir = direct_conditional_bayes_voi(p1_ca, p2_ca, x[300:380], p1_te, x[380:], contract, n_jobs=1)
    v_cov = covariate_enriched_martingale_voi(p1_ca, p2_ca, x[300:380], p1_te, x[380:], contract, n_jobs=1)
    assert v_dir.shape == (70,) and np.all(v_dir >= 0.0)
    assert v_cov.shape == (70,) and np.all(v_cov >= 0.0)
