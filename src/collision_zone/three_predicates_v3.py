"""Paper 2 v2.3 Methodological & Evaluation Repairs (`F1`–`F15` of Editorial Review).

Implements:
  1. Unified Beta-KL Threshold Geometry (`F3`; Buja, Stuetzle & Shen, 2005):
       w_{tau*, kappa}(p) / w_{tau*, kappa}(tau*) == exp(-kappa * D_KL(Bern(tau*) || Bern(p))).
  2. Honest Disjoint Split-Pilot Discovery-Confirmation (`F5`):
       Splits pilot P = P_fit u P_conf within strata so candidate models are trained strictly on
       P_fit and tested via exact Horvitz-Thompson variance + Holm step-down on independent P_conf.
  3. Decisive Discovery-Objective Factorial (`F14`):
       Supports 'global_cmi' (log-loss), 'brier', 'collision_zone' (Beta kernel S_{tau*, kappa}),
       'direct_utility' (fixed-threshold Bayes utility), and 'local_utility' (interval-smoothed
       decision utility around tau*).
  4. Explicit Patient-Level Panel & Procedure Acquisition Ledger (`F1`, `F12`):
       C_i(A_i) = 1{A_i != empty} * c_proc(A_i) + sum_{m in A_i} c_assay,m,
       supporting Common-Panel Bagged VOI (primary), Union-Panel Bagged VOI, Single-Split Policy
       Mean, Fractionally Billed Ensemble, and per-patient channel-billed Covert/Utility AFA.
  5. VOI Threshold-Crossing Identity & Routing Comparators (`F2`, `F8`):
       VOI(x) = (C_FN + C_FP) * (E[(P_2 - tau*)_+ | x] - (P_1 - tau*)_+) under E[P_2 | x] = P_1,
       plus Escalation-Only, De-escalation-Only, Sensitivity-Constrained, Distance-to-Threshold,
       Entropy, Uncentered VOI, and Covariate-Enriched VOI routing.
  6. Scale-Invariant Specialist Diagnostics (`F7`):
       RMS gated log-odds contribution R_CZ = sqrt(mean((beta_CZ * pi_i * Delta_i)^2)) and
       collision-zone decision-flip rate.
  7. Genuinely Label-Free Saerens-Latinne-Decaestecker EM Prior-Shift Estimation (`F10`):
       Estimates target prevalence pi_tgt_hat strictly from unlabeled target predictions p_src(X_tgt).
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import norm

from collision_zone.three_predicates import (
    AgnielDeconfounder,
    ChannelDef,
    ClinicalContract,
    DiscoveryOutcome,
    DiscoveryStepLog,
    FastFrameEncoder,
    PilotSample,
    bernoulli_kl_from_tau,
    collision_zone_relevance_weights,
    empirical_martingale_voi,
    holm_adjust,
    log_likelihood_score,
    sample_pilot,
    schervish_collision_proper_score,
    schervish_collision_weight,
    solve_ess_kappa,
    two_phase_ht_mean_and_se,
    union_collision_gate,
)
from collision_zone.three_predicates_v2 import FluidCollisionBackboneClassifierV2

_EPS = 1e-12


# ---------------------------------------------------------------------------
# F3: Exact Beta-Density <-> Bernoulli-KL Gate Identity (Buja et al., 2005)
# ---------------------------------------------------------------------------


def normalized_beta_threshold_kernel(
    p: np.ndarray,
    tau_star: float,
    kappa: float,
) -> np.ndarray:
    """Return peak-normalized Beta(1 + kappa*tau*, 1 + kappa*(1-tau*)) weight w(p) / w(tau*)."""
    pv = np.clip(np.asarray(p, dtype=float), _EPS, 1.0 - _EPS)
    w_p = schervish_collision_weight(pv, tau_star, kappa)
    w_peak = float(schervish_collision_weight(np.array([tau_star]), tau_star, kappa)[0])
    return w_p / max(_EPS, w_peak)


def kl_boltzmann_threshold_gate(
    p: np.ndarray,
    tau_star: float,
    kappa: float,
) -> np.ndarray:
    """Return exp(-kappa * D_KL(Bern(tau*) || Bern(p))), algebraically identical to w(p)/w(tau*)."""
    pv = np.clip(np.asarray(p, dtype=float), _EPS, 1.0 - _EPS)
    return np.exp(-float(kappa) * bernoulli_kl_from_tau(pv, tau_star))


# ---------------------------------------------------------------------------
# F14: Alternative Discovery Scoring Objectives (Brier, Direct & Local Utility)
# ---------------------------------------------------------------------------


def brier_proper_score(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Strictly proper negative Brier score -(y - p)^2 <= 0 (higher is better)."""
    yv = np.asarray(y, dtype=float)
    pv = np.clip(np.asarray(p, dtype=float), _EPS, 1.0 - _EPS)
    return -((yv - pv) ** 2)


def direct_threshold_utility_score(
    y: np.ndarray,
    p: np.ndarray,
    tau_star: float,
) -> np.ndarray:
    """Pointwise fixed-threshold Bayes utility in TP-equivalent units (higher is better)."""
    yv = np.asarray(y, dtype=float)
    pv = np.asarray(p, dtype=float)
    act = (pv >= float(tau_star)).astype(float)
    odds = float(tau_star) / max(_EPS, 1.0 - float(tau_star))
    return act * yv - odds * act * (1.0 - yv)


def local_interval_utility_score(
    y: np.ndarray,
    p: np.ndarray,
    tau_star: float,
    half_width_frac: float = 0.50,
    n_grid: int = 17,
) -> np.ndarray:
    """Schervish local uniform-band utility integral around tau* in [tau*(1-hw), tau*(1+hw)].

    Strictly proper inside the local threshold band and constant outside, providing a smoothed
    decision-utility comparator without the Beta tail.
    """
    yv = np.asarray(y, dtype=float)
    pv = np.clip(np.asarray(p, dtype=float), _EPS, 1.0 - _EPS)
    lo = max(1e-3, float(tau_star) * (1.0 - half_width_frac))
    hi = min(1.0 - 1e-3, float(tau_star) * (1.0 + half_width_frac))
    taus = np.linspace(lo, hi, n_grid)
    # Schervish integrand: -y*(1-tau)*1{tau > p} - (1-y)*tau*1{tau <= p}
    diff_p = pv[:, None]
    y_col = yv[:, None]
    t_row = taus[None, :]
    loss_grid = y_col * (1.0 - t_row) * (t_row > diff_p) + (1.0 - y_col) * t_row * (t_row <= diff_p)
    return -np.mean(loss_grid, axis=1)


def evaluate_discovery_objective_vec(
    y: np.ndarray,
    p: np.ndarray,
    *,
    objective: str,
    tau_star: float,
    kappa: float = 24.0,
) -> np.ndarray:
    """Evaluate per-observation discovery score for F14 factorial comparison."""
    if objective == "collision_zone":
        return schervish_collision_proper_score(y, p, tau_star, kappa=kappa)
    if objective in ("global_cmi", "srs_global_cmi"):
        return log_likelihood_score(y, p)
    if objective == "brier":
        return brier_proper_score(y, p)
    if objective == "direct_utility":
        return direct_threshold_utility_score(y, p, tau_star)
    if objective == "local_utility":
        return local_interval_utility_score(y, p, tau_star)
    raise ValueError(f"Unknown discovery objective: {objective}")


# ---------------------------------------------------------------------------
# F5 & F14: Split-Pilot Confirmation Discovery & General Objective Discovery
# ---------------------------------------------------------------------------


def split_pilot_sample(
    pilot: PilotSample,
    y_full: np.ndarray,
    *,
    fit_fraction: float = 0.50,
    seed: int = 42,
) -> tuple[np.ndarray, PilotSample]:
    """Split a Phase-2 PilotSample within each stratum into disjoint P_fit and P_conf.

    Conditional on P_fit, P_conf is a stratified SRSWOR sample from the remaining frame units,
    so models fit strictly on P_fit produce conditionally fixed scores psi_i on P_conf.
    """
    rng = np.random.default_rng(seed)
    fit_parts: list[np.ndarray] = []
    conf_parts: list[np.ndarray] = []
    conf_strata_parts: list[np.ndarray] = []
    conf_w_parts: list[np.ndarray] = []
    conf_pop_counts: dict[int, int] = {}
    conf_sample_counts: dict[int, int] = {}

    for h_i, n_h_pop in pilot.pop_counts.items():
        mask_h = pilot.strata == h_i
        idx_h = pilot.indices[mask_h]
        n_h = int(idx_h.size)
        if n_h == 0:
            continue
        perm = rng.permutation(idx_h)
        if n_h >= 4:
            n_fit_h = max(2, min(n_h - 2, round(fit_fraction * n_h)))
        elif n_h >= 2:
            n_fit_h = n_h // 2
        else:
            n_fit_h = 0
        fit_h = np.sort(perm[:n_fit_h])
        conf_h = np.sort(perm[n_fit_h:])
        if fit_h.size > 0:
            fit_parts.append(fit_h)
        if conf_h.size > 0:
            n_conf_h = int(conf_h.size)
            conf_parts.append(conf_h)
            conf_strata_parts.append(np.full(n_conf_h, h_i, dtype=int))
            conf_w_parts.append(np.full(n_conf_h, float(n_h_pop) / float(n_conf_h), dtype=float))
            conf_pop_counts[int(h_i)] = int(n_h_pop)
            conf_sample_counts[int(h_i)] = n_conf_h

    fit_idx = np.sort(np.concatenate(fit_parts)) if fit_parts else pilot.indices[:1]
    conf_idx = np.concatenate(conf_parts)
    conf_st = np.concatenate(conf_strata_parts)
    conf_wt = np.concatenate(conf_w_parts)
    order = np.argsort(conf_idx)
    conf_idx, conf_st, conf_wt = conf_idx[order], conf_st[order], conf_wt[order]

    # Rescale confirmation HT weights so sum(w_conf) == n_total when any tiny stratum gave 0 conf rows
    w_sum = float(np.sum(conf_wt))
    if w_sum > 0 and abs(w_sum - pilot.n_total) > 1e-6:
        conf_wt = conf_wt * (float(pilot.n_total) / w_sum)

    conf_pilot = PilotSample(
        indices=conf_idx,
        strata=conf_st,
        ht_weights=conf_wt,
        pop_counts=conf_pop_counts,
        sample_counts=conf_sample_counts,
        n_total=pilot.n_total,
        high_signal_fp_share_of_controls=pilot.high_signal_fp_share_of_controls,
    )
    return fit_idx, conf_pilot


def run_discovery_v3(
    df_train: pd.DataFrame,
    y_train: np.ndarray,
    base_cols: list[str],
    channels: list[ChannelDef],
    oof_p_base: np.ndarray,
    contract: ClinicalContract,
    *,
    objective: str = "collision_zone",
    pilot_mode: str | None = None,
    evaluation_protocol: str = "cross_fitted",
    pilot_fraction: float = 0.20,
    budget: float = 3.2,
    kappa: float = 24.0,
    n_bins: int = 8,
    alpha: float = 0.05,
    use_holm_lcb: bool = True,
    deconfound_stage2: bool = False,
    n_jobs: int = 4,
    seed: int = 42,
) -> DiscoveryOutcome:
    """General Stage-A discovery supporting F14 objectives and F5 split-pilot confirmation.

    evaluation_protocol:
      - 'cross_fitted': 4-fold cross-fitting on full pilot P (high-power heuristic).
      - 'split_confirmation': disjoint 50/50 P = P_fit u P_conf where learners are fit strictly
        on P_fit and tested via conditional design-based HT inference + Holm on P_conf.
    """
    yv = np.asarray(y_train, dtype=int)
    tau = contract.tau_star
    if pilot_mode is None:
        pilot_mode = (
            "collision_neyman"
            if objective in ("collision_zone", "direct_utility", "local_utility")
            else ("srs" if objective == "srs_global_cmi" else "global_neyman")
        )
    pilot = sample_pilot(
        yv,
        oof_p_base,
        pilot_fraction=pilot_fraction,
        mode=pilot_mode,
        tau_star=tau,
        kappa=kappa,
        n_bins=n_bins,
        seed=seed,
    )

    p0_all = np.clip(oof_p_base, 1e-5, 1.0 - 1e-5)
    kappa_eff = (
        solve_ess_kappa(
            bernoulli_kl_from_tau(p0_all[pilot.indices], tau),
            kappa_max=kappa,
            min_effective_cz_samples=max(160.0, 0.22 * float(pilot.indices.size)),
        )
        if objective == "collision_zone"
        else float(kappa)
    )
    use_cz_fit_weights = objective in ("collision_zone", "direct_utility", "local_utility")

    all_ch_cols = list(dict.fromkeys(c for ch in channels for c in ch.columns))

    if evaluation_protocol == "split_confirmation":
        fit_idx, eval_pilot = split_pilot_sample(pilot, yv, fit_fraction=0.50, seed=seed + 13)
        eval_idx = eval_pilot.indices
        y_fit = yv[fit_idx]
        y_eval = yv[eval_idx]
        init_fit = logit(p0_all[fit_idx])
        init_eval = logit(p0_all[eval_idx])

        # Compute fit weights from stratum frequencies inside fit_idx
        w_fit = np.ones(fit_idx.size, dtype=float)
        if use_cz_fit_weights:
            w_fit = collision_zone_relevance_weights(
                p0_all[fit_idx], tau, kappa=kappa_eff, floor=0.05
            )

        enc_b = FastFrameEncoder().fit(df_train[base_cols].iloc[fit_idx])
        xb_fit = enc_b.transform(df_train[base_cols].iloc[fit_idx])
        xb_eval = enc_b.transform(df_train[base_cols].iloc[eval_idx])

        if deconfound_stage2 and all_ch_cols:
            deconf = AgnielDeconfounder().fit(
                df_train[base_cols].iloc[fit_idx], df_train[all_ch_cols].iloc[fit_idx]
            )
            s2_fit_df = deconf.transform(
                df_train[base_cols].iloc[fit_idx], df_train[all_ch_cols].iloc[fit_idx], seed=seed + 51
            )
            s2_eval_df = deconf.transform(
                df_train[base_cols].iloc[eval_idx], df_train[all_ch_cols].iloc[eval_idx], seed=seed + 52
            )
        else:
            s2_fit_df = df_train[all_ch_cols].iloc[fit_idx]
            s2_eval_df = df_train[all_ch_cols].iloc[eval_idx]

        ch_mats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for ch in channels:
            cols_c = list(ch.columns)
            enc_c = FastFrameEncoder(include_missing_indicators=not deconfound_stage2).fit(
                s2_fit_df[cols_c]
            )
            ch_mats[ch.channel_id] = (
                enc_c.transform(s2_fit_df[cols_c]),
                enc_c.transform(s2_eval_df[cols_c]),
            )

        def _eval_probs_for_set(sel_ids: list[str]) -> np.ndarray:
            x_tr = np.hstack([xb_fit] + [ch_mats[cid][0] for cid in sel_ids])
            x_va = np.hstack([xb_eval] + [ch_mats[cid][1] for cid in sel_ids])
            clf = lgb.LGBMClassifier(
                n_estimators=70,
                learning_rate=0.05,
                num_leaves=15,
                min_child_samples=max(12, min(25, y_fit.size // 12)),
                max_delta_step=1.0,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=seed + 101,
                n_jobs=n_jobs,
                verbose=-1,
            )
            clf.fit(x_tr, y_fit, sample_weight=w_fit, init_score=init_fit)
            raw_va = init_eval + clf.predict(x_va, raw_score=True)
            return np.clip(expit(np.clip(raw_va, -20.0, 20.0)), 1e-6, 1.0 - 1e-6)

    elif evaluation_protocol == "cross_fitted":
        eval_pilot = pilot
        p_idx = pilot.indices
        y_eval = yv[p_idx]
        p0_p = p0_all[p_idx]
        init_p = logit(p0_p)
        fit_weights = (
            pilot.ht_weights
            * collision_zone_relevance_weights(p0_p, tau, kappa=kappa_eff, floor=0.05)
            if use_cz_fit_weights
            else pilot.ht_weights.copy()
        )

        rng = np.random.default_rng(seed + 17)
        pos = rng.permutation(np.flatnonzero(y_eval == 1))
        neg = rng.permutation(np.flatnonzero(y_eval == 0))
        n_folds = 4
        folds = [np.concatenate([pos[k::n_folds], neg[k::n_folds]]) for k in range(n_folds)]
        df_p = df_train.iloc[p_idx].reset_index(drop=True)
        fold_base: list[tuple[np.ndarray, np.ndarray]] = []
        fold_ch: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
            ch.channel_id: [] for ch in channels
        }
        for k in range(n_folds):
            va = folds[k]
            tr = np.concatenate([folds[j] for j in range(n_folds) if j != k])
            enc_b = FastFrameEncoder().fit(df_p[base_cols].iloc[tr])
            fold_base.append(
                (enc_b.transform(df_p[base_cols].iloc[tr]), enc_b.transform(df_p[base_cols].iloc[va]))
            )
            if deconfound_stage2 and all_ch_cols:
                deconf = AgnielDeconfounder().fit(
                    df_p[base_cols].iloc[tr], df_p[all_ch_cols].iloc[tr]
                )
                s2_tr_df = deconf.transform(
                    df_p[base_cols].iloc[tr], df_p[all_ch_cols].iloc[tr], seed=seed + 50 + k
                )
                s2_va_df = deconf.transform(
                    df_p[base_cols].iloc[va], df_p[all_ch_cols].iloc[va], seed=seed + 90 + k
                )
            else:
                s2_tr_df = df_p[all_ch_cols].iloc[tr]
                s2_va_df = df_p[all_ch_cols].iloc[va]
            for ch in channels:
                cols_c = list(ch.columns)
                enc_c = FastFrameEncoder(include_missing_indicators=not deconfound_stage2).fit(
                    s2_tr_df[cols_c]
                )
                fold_ch[ch.channel_id].append(
                    (enc_c.transform(s2_tr_df[cols_c]), enc_c.transform(s2_va_df[cols_c]))
                )

        def _eval_probs_for_set(sel_ids: list[str]) -> np.ndarray:
            oof = np.zeros(y_eval.size, dtype=float)
            for k in range(n_folds):
                va = folds[k]
                tr = np.concatenate([folds[j] for j in range(n_folds) if j != k])
                x_tr = np.hstack([fold_base[k][0]] + [fold_ch[cid][k][0] for cid in sel_ids])
                x_va = np.hstack([fold_base[k][1]] + [fold_ch[cid][k][1] for cid in sel_ids])
                clf = lgb.LGBMClassifier(
                    n_estimators=70,
                    learning_rate=0.05,
                    num_leaves=15,
                    min_child_samples=25,
                    max_delta_step=1.0,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    random_state=seed + 100 + k,
                    n_jobs=n_jobs,
                    verbose=-1,
                )
                clf.fit(x_tr, y_eval[tr], sample_weight=fit_weights[tr], init_score=init_p[tr])
                raw_va = init_p[va] + clf.predict(x_va, raw_score=True)
                oof[va] = expit(np.clip(raw_va, -20.0, 20.0))
            return np.clip(oof, 1e-6, 1.0 - 1e-6)

    else:
        raise ValueError(f"Unknown evaluation_protocol: {evaluation_protocol}")

    selected: list[str] = []
    spent = 0.0
    steps: list[DiscoveryStepLog] = []
    step1_scores: dict[str, float] = {}

    cur_p = _eval_probs_for_set([])
    cur_s = evaluate_discovery_objective_vec(
        y_eval, cur_p, objective=objective, tau_star=tau, kappa=kappa_eff
    )

    step_num = 0
    while True:
        step_num += 1
        affordable = [
            ch
            for ch in channels
            if ch.channel_id not in selected and spent + ch.cost <= budget + 1e-9
        ]
        if not affordable:
            break

        cand_gains: list[float] = []
        cand_ses: list[float] = []
        cand_pvals: list[float] = []
        cand_probs: list[np.ndarray] = []

        for ch in affordable:
            p_aug = _eval_probs_for_set(selected + [ch.channel_id])
            s_aug = evaluate_discovery_objective_vec(
                y_eval, p_aug, objective=objective, tau_star=tau, kappa=kappa_eff
            )
            psi = s_aug - cur_s
            g_hat, se_tot = two_phase_ht_mean_and_se(psi, eval_pilot)
            z = g_hat / max(1e-12, se_tot)
            pval = float(1.0 - norm.cdf(z))
            cand_gains.append(g_hat)
            cand_ses.append(se_tot)
            cand_pvals.append(pval)
            cand_probs.append(p_aug)
            if step_num == 1:
                step1_scores[ch.channel_id] = g_hat

        p_holm = holm_adjust(cand_pvals)
        best_idx = -1
        best_ratio = -np.inf
        for idx_c, ch in enumerate(affordable):
            passes = (p_holm[idx_c] < alpha) if use_holm_lcb else (cand_gains[idx_c] > 0.0)
            if passes and cand_gains[idx_c] > 0.0:
                ratio = cand_gains[idx_c] / ch.cost
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_idx = idx_c

        if best_idx < 0:
            break

        winner = affordable[best_idx]
        selected.append(winner.channel_id)
        spent += winner.cost
        cur_p = cand_probs[best_idx]
        cur_s = evaluate_discovery_objective_vec(
            y_eval, cur_p, objective=objective, tau_star=tau, kappa=kappa_eff
        )
        steps.append(
            DiscoveryStepLog(
                step=step_num,
                selected_id=winner.channel_id,
                gain=cand_gains[best_idx],
                se_total=cand_ses[best_idx],
                z_stat=cand_gains[best_idx] / max(1e-12, cand_ses[best_idx]),
                p_holm=p_holm[best_idx],
                cost=winner.cost,
            )
        )

    return DiscoveryOutcome(
        method=f"{objective}_{evaluation_protocol}",
        pilot_fraction=pilot_fraction,
        selected_ids=selected,
        total_cost=spent,
        pilot_high_signal_fp_share=pilot.high_signal_fp_share_of_controls,
        steps=steps,
        step1_scores=step1_scores,
    )


# ---------------------------------------------------------------------------
# F1: Explicit Patient-Level Panel & Procedure Acquisition Cost Ledger
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PanelCostSchedule:
    """Explicit decomposition of procedure cost + per-assay laboratory processing cost.

    Costs are expressed in units of C_FP (false-positive action cost = 1.0 numeraire).
    Under default sepsis calibration (venipuncture = 0.010, unit_assay_scale = 0.020):
      - Single Lactate draw (CPT rel 1.00): 0.010 + 0.020 * 1.00 = 0.030 (exact parity with v2!).
      - Lactate + Platelets (1.00 + 0.78):  0.010 + 0.020 * 1.78 = 0.0456.
      - Lactate + Blood Gas (1.00 + 1.79 + extra ABG procedure 0.010): 0.010 + 0.010 + 0.020*2.79 = 0.0758.
    """

    venipuncture_cost: float = 0.010
    abg_extra_procedure_cost: float = 0.010
    unit_assay_scale: float = 0.020
    abg_channel_ids: tuple[str, ...] = ("lab_blood_gas_acidbase",)

    def panel_cost(self, channel_ids: list[str] | set[str], cost_map: dict[str, float]) -> float:
        uniq = sorted(set(channel_ids))
        if not uniq:
            return 0.0
        proc = float(self.venipuncture_cost)
        if any(cid in self.abg_channel_ids for cid in uniq):
            proc += float(self.abg_extra_procedure_cost)
        assays = sum(float(cost_map[cid]) for cid in uniq)
        return proc + float(self.unit_assay_scale) * assays

    def instance_cost_vector(
        self,
        mask_matrix: np.ndarray,
        channels: list[ChannelDef],
    ) -> np.ndarray:
        """Compute exact C_i(A_i) for an (N, M) binary acquisition mask matrix."""
        n = mask_matrix.shape[0]
        has_any = np.any(mask_matrix > 0.5, axis=1)
        costs_m = np.array([float(ch.cost) for ch in channels], dtype=float)
        abg_mask = np.zeros(n, dtype=bool)
        for j, ch in enumerate(channels):
            if ch.channel_id in self.abg_channel_ids:
                abg_mask |= mask_matrix[:, j] > 0.5
        c_vec = np.zeros(n, dtype=float)
        c_vec[has_any] = (
            float(self.venipuncture_cost)
            + float(self.abg_extra_procedure_cost) * abg_mask[has_any].astype(float)
            + float(self.unit_assay_scale) * (mask_matrix[has_any] @ costs_m)
        )
        return c_vec


def net_benefit_with_cost_ledger_per_1k(
    y: np.ndarray,
    p_routed: np.ndarray,
    instance_costs_cfp_units: np.ndarray,
    contract: ClinicalContract,
) -> float:
    """Exact Net Benefit per 1,000 arrivals using patient-specific cost ledger C_i (in C_FP units)."""
    yv = np.asarray(y, dtype=int)
    pv = np.asarray(p_routed, dtype=float)
    cv = np.asarray(instance_costs_cfp_units, dtype=float)
    act = pv >= contract.tau_star
    w_fp = contract.cost_fp / contract.cost_fn
    unit = (
        (act & (yv == 1)).astype(float)
        - w_fp * (act & (yv == 0)).astype(float)
        - (cv / contract.cost_fn)
    )
    return float(1000.0 * np.mean(unit))


def unit_nb_ledger_vector(
    y: np.ndarray,
    p_routed: np.ndarray,
    instance_costs_cfp_units: np.ndarray,
    contract: ClinicalContract,
) -> np.ndarray:
    """Per-instance Net Benefit contribution scaled by 1,000 for paired bootstrap inference."""
    yv = np.asarray(y, dtype=int)
    pv = np.asarray(p_routed, dtype=float)
    cv = np.asarray(instance_costs_cfp_units, dtype=float)
    act = pv >= contract.tau_star
    w_fp = contract.cost_fp / contract.cost_fn
    return 1000.0 * (
        (act & (yv == 1)).astype(float)
        - w_fp * (act & (yv == 0)).astype(float)
        - (cv / contract.cost_fn)
    )


# ---------------------------------------------------------------------------
# F7: Scale-Invariant Specialist Contribution Diagnostics
# ---------------------------------------------------------------------------


def compute_specialist_diagnostics(
    clf: FluidCollisionBackboneClassifierV2,
    x_all: np.ndarray,
    z_s2: np.ndarray,
    ell1_base: np.ndarray,
) -> dict[str, float]:
    """Compute scale-invariant RMS gated residual R_CZ and collision-zone decision-flip rate."""
    assert clf.m_glob_ is not None
    ell1 = np.clip(np.asarray(ell1_base, dtype=float), -20.0, 20.0)
    raw_g = clf.m_glob_.predict(x_all, raw_score=True)
    ag, bg = clf.platt_glob_
    f_glob = np.clip(ag * raw_g + bg, -20.0, 20.0)
    f_backbone = clf.alpha_0_ + clf.beta_glob_ * f_glob

    if clf.m_cz_ is None or clf.beta_cz_ <= 1e-4:
        return {
            "beta_cz": float(clf.beta_cz_),
            "rms_gated_log_odds_r_cz": 0.0,
            "rms_cz_zone_r_cz": 0.0,
            "decision_flip_rate_all": 0.0,
            "decision_flip_rate_cz": 0.0,
        }

    pi = union_collision_gate(ell1, f_glob, clf.tau_star, clf.kappa_)
    z_aug = np.column_stack([ell1, f_glob, z_s2]).astype(np.float32)
    d_cz = clf.m_cz_.predict(z_aug, raw_score=True)
    gated_term = clf.beta_cz_ * pi * d_cz
    f_total = f_backbone + gated_term

    p_back = expit(np.clip(f_backbone, -25.0, 25.0))
    p_tot = expit(np.clip(f_total, -25.0, 25.0))
    flips = (p_tot >= clf.tau_star) != (p_back >= clf.tau_star)
    cz_mask = pi >= 0.25

    return {
        "beta_cz": float(clf.beta_cz_),
        "rms_gated_log_odds_r_cz": float(np.sqrt(np.mean(gated_term**2))),
        "rms_cz_zone_r_cz": float(np.sqrt(np.mean(gated_term[cz_mask] ** 2)))
        if np.any(cz_mask)
        else 0.0,
        "decision_flip_rate_all": float(np.mean(flips)),
        "decision_flip_rate_cz": float(np.mean(flips[cz_mask])) if np.any(cz_mask) else 0.0,
    }


def fit_matched_ungated_residual_stage2(
    x_tr: np.ndarray,
    z_tr: np.ndarray,
    y_tr: np.ndarray,
    ell1_tr: np.ndarray,
    x_evals: list[np.ndarray],
    z_evals: list[np.ndarray],
    ell1_evals: list[np.ndarray],
    *,
    res_leaves: int = 8,
    res_trees: int = 50,
    n_jobs: int = 4,
    seed: int = 42,
) -> tuple[list[np.ndarray], dict[str, float]]:
    """F7 comparator: Matched-capacity ungated residual model (pi_i === 1.0, global log-loss stacking)."""
    from scipy.optimize import minimize
    from sklearn.model_selection import StratifiedKFold

    from collision_zone.three_predicates import fit_platt_calibrator
    from collision_zone.three_predicates_v2 import _fit_gated_residual_booster_v2

    yv = np.asarray(y_tr, dtype=int)
    ell1 = np.clip(np.asarray(ell1_tr, dtype=float), -20.0, 20.0)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    splits = list(skf.split(x_tr, yv))

    def _mk_glob(s_off: int) -> lgb.LGBMClassifier:
        return lgb.LGBMClassifier(
            n_estimators=140,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=30,
            max_delta_step=1.0,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=seed + s_off,
            n_jobs=n_jobs,
            verbose=-1,
        )

    oof_raw_g = np.zeros(yv.size, dtype=float)
    for k, (ti, vi) in enumerate(splits):
        mg = _mk_glob(10 + k)
        mg.fit(x_tr[ti], yv[ti])
        oof_raw_g[vi] = mg.predict(x_tr[vi], raw_score=True)

    ag, bg = fit_platt_calibrator(oof_raw_g, yv)
    oof_fg = np.clip(ag * oof_raw_g + bg, -20.0, 20.0)
    pi_ones = np.ones(yv.size, dtype=float)
    z_aug = np.column_stack([ell1, oof_fg, z_tr]).astype(np.float32)

    oof_d = np.zeros(yv.size, dtype=float)
    for k, (ti, vi) in enumerate(splits):
        mres = _fit_gated_residual_booster_v2(
            z_aug[ti],
            yv[ti],
            oof_fg[ti],
            pi_ones[ti],
            n_estimators=res_trees,
            num_leaves=res_leaves,
            n_jobs=n_jobs,
            seed=seed + 30 + k,
        )
        oof_d[vi] = mres.predict(z_aug[vi], raw_score=True)

    def _obj_ll(params: np.ndarray) -> float:
        a0, b_g, b_r = float(params[0]), float(params[1]), float(params[2])
        f = np.clip(a0 + b_g * oof_fg + b_r * oof_d, -25.0, 25.0)
        ll = -float(np.mean(log_likelihood_score(yv, expit(f))))
        return ll + 2e-4 * ((b_g - 1.0) ** 2 + b_r**2 + a0**2)

    opt = minimize(
        _obj_ll,
        x0=np.array([0.0, 1.0, 0.25]),
        bounds=[(-1.5, 1.5), (0.5, 1.3), (0.0, 1.5)],
        method="L-BFGS-B",
    )
    a0, b_g, b_r = float(opt.x[0]), float(opt.x[1]), float(opt.x[2])
    m_glob_full = _mk_glob(7).fit(x_tr, yv)
    m_res_full = (
        _fit_gated_residual_booster_v2(
            z_aug,
            yv,
            oof_fg,
            pi_ones,
            n_estimators=res_trees,
            num_leaves=res_leaves,
            n_jobs=n_jobs,
            seed=seed + 77,
        )
        if b_r > 1e-4
        else None
    )

    out_raws: list[np.ndarray] = []
    rms_list: list[float] = []
    for xe, ze, e1e in zip(x_evals, z_evals, ell1_evals):
        e1c = np.clip(np.asarray(e1e, dtype=float), -20.0, 20.0)
        fg = np.clip(ag * m_glob_full.predict(xe, raw_score=True) + bg, -20.0, 20.0)
        f_out = a0 + b_g * fg
        if m_res_full is not None and b_r > 1e-4:
            za = np.column_stack([e1c, fg, ze]).astype(np.float32)
            r_term = b_r * m_res_full.predict(za, raw_score=True)
            f_out = f_out + r_term
            rms_list.append(float(np.sqrt(np.mean(r_term**2))))
        else:
            rms_list.append(0.0)
        out_raws.append(f_out)

    return out_raws, {
        "beta_ungated": b_r,
        "rms_ungated_log_odds": float(np.mean(rms_list)) if rms_list else 0.0,
    }


# ---------------------------------------------------------------------------
# F8 & F2: Threshold-Crossing VOI Identity, Routing Comparators, & One-Sided Policies
# ---------------------------------------------------------------------------


def empirical_martingale_posterior_support(
    p1: np.ndarray,
    bin_indices: np.ndarray,
    bin_delta_quantiles: np.ndarray,
) -> np.ndarray:
    """Return the (N, K) mean-coherent empirical posterior support P_{2,r} = expit(logit(p1) + mu* + delta_r)."""
    pv = np.clip(np.asarray(p1, dtype=float), 1e-6, 1.0 - 1e-6)
    b_idx = np.asarray(bin_indices, dtype=int)
    u1 = logit(pv)
    d_mat = bin_delta_quantiles[b_idx, :]
    spread = np.std(d_mat, axis=1)
    mu = np.zeros(pv.size, dtype=float)
    for _ in range(18):
        z = np.clip(u1[:, None] + mu[:, None] + d_mat, -30.0, 30.0)
        q = expit(z)
        err = np.mean(q, axis=1) - pv
        if float(np.max(np.abs(err))) < 1e-12:
            break
        deriv = np.mean(q * (1.0 - q), axis=1)
        mu -= np.clip(err / np.maximum(1e-12, deriv), -2.5, 2.5)
    mu = np.where(spread <= 1e-12, 0.0, mu)
    return expit(np.clip(u1[:, None] + mu[:, None] + d_mat, -30.0, 30.0))


def threshold_crossing_voi_from_posteriors(
    p1: np.ndarray,
    p2_support: np.ndarray,
    contract: ClinicalContract,
) -> np.ndarray:
    """Evaluate VOI via F8 threshold-crossing identity:
       VOI(x) = (C_FN + C_FP) * (E[(P_2 - tau*)_+ | x] - (P_1 - tau*)_+)
    for an (N, K) matrix of mean-coherent posterior support points P_2 (mean(P_2, axis=1) == p1).
    """
    p1v = np.asarray(p1, dtype=float)
    tau = contract.tau_star
    scale = contract.cost_fn + contract.cost_fp
    pos_p2 = np.mean(np.maximum(0.0, p2_support - tau), axis=1)
    pos_p1 = np.maximum(0.0, p1v - tau)
    return np.maximum(0.0, scale * (pos_p2 - pos_p1))


def uncentered_empirical_voi(
    p1: np.ndarray,
    bin_indices: np.ndarray,
    bin_quantiles: np.ndarray,
    contract: ClinicalContract,
) -> np.ndarray:
    """Empirical VOI without martingale mean-centering (clips negative risk diffs at 0)."""
    p1v = np.clip(np.asarray(p1, dtype=float), 1e-6, 1.0 - 1e-6)
    ell1 = logit(p1v)
    cfn, cfp = contract.cost_fn, contract.cost_fp
    l1 = np.minimum(p1v * cfn, (1.0 - p1v) * cfp)
    deltas = bin_quantiles[bin_indices]  # (N, K)
    p2_raw = expit(np.clip(ell1[:, None] + deltas, -25.0, 25.0))
    l2 = np.mean(np.minimum(p2_raw * cfn, (1.0 - p2_raw) * cfp), axis=1)
    return np.maximum(0.0, l1 - l2)


def routing_priority_scores(
    p1: np.ndarray,
    bin_indices: np.ndarray,
    bin_quantiles: np.ndarray,
    contract: ClinicalContract,
    *,
    mode: str,
) -> np.ndarray:
    """Compute Stage-B acquisition priority scores across F8 comparators."""
    p1v = np.clip(np.asarray(p1, dtype=float), 1e-6, 1.0 - 1e-6)
    if mode == "empirical_martingale_voi":
        return empirical_martingale_voi(p1v, bin_indices, bin_quantiles, contract)
    if mode == "uncentered_empirical_voi":
        return uncentered_empirical_voi(p1v, bin_indices, bin_quantiles, contract)
    if mode == "distance_to_threshold":
        return -np.abs(p1v - contract.tau_star)
    if mode == "shannon_entropy":
        return -(p1v * np.log(p1v) + (1.0 - p1v) * np.log(1.0 - p1v))
    raise ValueError(f"Unknown routing mode: {mode}")


def direct_conditional_bayes_voi(
    p1_cal: np.ndarray,
    p2_cal: np.ndarray,
    xb_cal: np.ndarray,
    p1_te: np.ndarray,
    xb_te: np.ndarray,
    contract: ClinicalContract,
    *,
    n_jobs: int = 4,
    seed: int = 42,
) -> np.ndarray:
    """F8 comparator: Direct regression of realized Bayes point-risk reduction on (logit(p1), X_base)."""
    p1c = np.clip(np.asarray(p1_cal, dtype=float), 1e-6, 1.0 - 1e-6)
    p2c = np.clip(np.asarray(p2_cal, dtype=float), 1e-6, 1.0 - 1e-6)
    p1t = np.clip(np.asarray(p1_te, dtype=float), 1e-6, 1.0 - 1e-6)
    cfn, cfp = contract.cost_fn, contract.cost_fp
    l1 = np.minimum(p1c * cfn, (1.0 - p1c) * cfp)
    l2 = np.minimum(p2c * cfn, (1.0 - p2c) * cfp)
    y_gain = np.maximum(0.0, l1 - l2)
    x_ca = np.column_stack([logit(p1c), xb_cal]).astype(np.float32)
    x_te = np.column_stack([logit(p1t), xb_te]).astype(np.float32)
    reg = lgb.LGBMRegressor(
        n_estimators=55,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=max(15, min(30, p1c.size // 15)),
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=seed,
        n_jobs=n_jobs,
        verbose=-1,
    )
    reg.fit(x_ca, y_gain)
    return np.maximum(0.0, reg.predict(x_te))


def covariate_enriched_martingale_voi(
    p1_cal: np.ndarray,
    p2_cal: np.ndarray,
    xb_cal: np.ndarray,
    p1_te: np.ndarray,
    xb_te: np.ndarray,
    contract: ClinicalContract,
    *,
    n_jobs: int = 4,
    seed: int = 42,
) -> np.ndarray:
    """F8 comparator: Conditional update distribution using baseline covariates X_base in addition to p1."""
    p1c = np.clip(np.asarray(p1_cal, dtype=float), 1e-6, 1.0 - 1e-6)
    p2c = np.clip(np.asarray(p2_cal, dtype=float), 1e-6, 1.0 - 1e-6)
    p1t = np.clip(np.asarray(p1_te, dtype=float), 1e-6, 1.0 - 1e-6)
    delta_cal = logit(p2c) - logit(p1c)
    abs_dev = np.abs(delta_cal - float(np.median(delta_cal)))
    x_ca = np.column_stack([logit(p1c), xb_cal]).astype(np.float32)
    x_te = np.column_stack([logit(p1t), xb_te]).astype(np.float32)
    reg_scale = lgb.LGBMRegressor(
        n_estimators=50,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=max(15, min(30, p1c.size // 15)),
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=seed + 3,
        n_jobs=n_jobs,
        verbose=-1,
    )
    reg_scale.fit(x_ca, abs_dev)
    scale_ca = np.maximum(0.05, reg_scale.predict(x_ca))
    scale_te = np.maximum(0.05, reg_scale.predict(x_te))
    z_std = (delta_cal - float(np.mean(delta_cal))) / scale_ca
    z_quants = np.quantile(z_std, np.linspace(0.02, 0.98, 33))  # (33,)
    # Construct row-specific empirical support delta_mat (N_te, 33) = scale_te[:, None] * z_quants[None, :]
    d_mat = scale_te[:, None] * z_quants[None, :]
    u1 = logit(p1t)
    mu = np.zeros(p1t.size, dtype=float)
    for _ in range(16):
        z = np.clip(u1[:, None] + mu[:, None] + d_mat, -30.0, 30.0)
        q = expit(z)
        err = np.mean(q, axis=1) - p1t
        if float(np.max(np.abs(err))) < 1e-11:
            break
        deriv = np.mean(q * (1.0 - q), axis=1)
        mu -= np.clip(err / np.maximum(1e-12, deriv), -2.5, 2.5)
    p2_supp = expit(np.clip(u1[:, None] + mu[:, None] + d_mat, -30.0, 30.0))
    return threshold_crossing_voi_from_posteriors(p1t, p2_supp, contract)


def apply_directional_voi_policy(
    p1: np.ndarray,
    p2: np.ndarray,
    voi_scores: np.ndarray,
    panel_query_cost: float,
    contract: ClinicalContract,
    *,
    policy_mode: str = "two_sided",
    p1_cal: np.ndarray | None = None,
    p2_cal: np.ndarray | None = None,
    y_cal: np.ndarray | None = None,
    voi_cal: np.ndarray | None = None,
    max_rel_sens_drop: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply F2 directional / sensitivity-constrained VOI acquisition policy.

    Returns (p_routed, query_mask_bool).
    """
    tau = contract.tau_star
    p1v = np.asarray(p1, dtype=float)
    p2v = np.asarray(p2, dtype=float)
    vv = np.asarray(voi_scores, dtype=float)

    if policy_mode == "two_sided":
        q = vv > float(panel_query_cost)
        return np.where(q, p2v, p1v), q

    if policy_mode == "escalation_only":
        # Only query sub-threshold baseline patients; never de-escalate p1 >= tau*!
        q = (p1v < tau) & (vv > float(panel_query_cost))
        p_out = np.where(q, np.maximum(p1v, p2v), p1v)
        return p_out, q

    if policy_mode == "deescalation_only":
        q = (p1v >= tau) & (vv > float(panel_query_cost))
        p_out = np.where(q, np.minimum(p1v, p2v), p1v)
        return p_out, q

    if policy_mode == "sensitivity_constrained":
        # Choose upper p1 de-escalation ceiling tau_hi >= tau* on D_cal so sensitivity drop <= max_rel_sens_drop
        tau_hi = tau
        if p1_cal is not None and p2_cal is not None and y_cal is not None and voi_cal is not None:
            yc = np.asarray(y_cal, dtype=int)
            q_es_cal = (p1_cal < tau) & (voi_cal > float(panel_query_cost))
            p_es_cal = np.where(q_es_cal, np.maximum(p1_cal, p2_cal), p1_cal)
            tp_es = float(np.sum((p_es_cal >= tau) & (yc == 1)))
            pos_tot = max(1.0, float(np.sum(yc == 1)))
            for cand_hi in np.linspace(tau, min(0.50, 3.0 * tau), 25):
                q_try = ((p1_cal < cand_hi) & (voi_cal > float(panel_query_cost)))
                p_try = np.where(q_try, p2_cal, p1_cal)
                tp_try = float(np.sum((p_try >= tau) & (yc == 1)))
                if (tp_es - tp_try) / pos_tot <= max_rel_sens_drop:
                    tau_hi = float(cand_hi)
        q = (p1v < tau_hi) & (vv > float(panel_query_cost))
        p_out = np.where(q & (p1v < tau), np.maximum(p1v, p2v), np.where(q, p2v, p1v))
        return p_out, q

    raise ValueError(f"Unknown policy_mode: {policy_mode}")


# ---------------------------------------------------------------------------
# F10: Label-Free Saerens-Latinne-Decaestecker EM Prevalence Estimation
# ---------------------------------------------------------------------------


def saerens_em_target_prevalence(
    p_source_on_target: np.ndarray,
    pi_source: float,
    *,
    max_iter: int = 200,
    tol: float = 1e-8,
) -> tuple[float, np.ndarray]:
    """Estimate target prevalence pi_tgt and recalibrated probabilities strictly without target labels.

    Implements Saerens, Latinne & Decaestecker (2002) EM under label shift P_tgt(X|Y) = P_src(X|Y).
    """
    ps = np.clip(np.asarray(p_source_on_target, dtype=float).ravel(), 1e-6, 1.0 - 1e-6)
    pi_s = float(np.clip(pi_source, 1e-5, 1.0 - 1e-5))
    pi_t = pi_s

    for _ in range(max_iter):
        r1 = (pi_t / pi_s) * ps
        r0 = ((1.0 - pi_t) / (1.0 - pi_s)) * (1.0 - ps)
        post = r1 / np.maximum(_EPS, r1 + r0)
        pi_next = float(np.clip(np.mean(post), 1e-5, 1.0 - 1e-5))
        if abs(pi_next - pi_t) < tol:
            pi_t = pi_next
            break
        pi_t = pi_next

    delta_0 = float(logit(pi_t) - logit(pi_s))
    p_adj = np.clip(expit(logit(ps) + delta_0), 1e-6, 1.0 - 1e-6)
    return pi_t, p_adj


# ---------------------------------------------------------------------------
# F1 & F14: Patient-Level Channel-Billed Sequential AFA (CMI vs. Direct VOI)
# ---------------------------------------------------------------------------


def evaluate_channel_billed_afa_v3(
    df_train: pd.DataFrame,
    y_train: np.ndarray,
    df_cal: pd.DataFrame,
    y_cal: np.ndarray,
    test_sets: list[tuple[pd.DataFrame, np.ndarray]],
    base_cols: list[str],
    channels: list[ChannelDef],
    oof_p_base_train: np.ndarray,
    precomputed_stage1: tuple[np.ndarray, list[np.ndarray]],
    contract: ClinicalContract,
    cost_schedule: PanelCostSchedule,
    *,
    criterion: str = "cmi",
    budget: float = 3.2,
    n_masks_per_row: int = 5,
    n_jobs: int = 4,
    seed: int = 42,
) -> list[dict[str, object]]:
    """Sequential instancewise AFA with exact patient-level panel/procedure cost ledger.

    criterion:
      - 'cmi': Covert et al. (2023) conditional Bernoulli KL divergence D_KL(Bern(p_+) || Bern(p)).
      - 'direct_voi': Direct decision-theoretic Bayes risk reduction L*(p) - L*(p_+).
    """
    from collision_zone.three_predicates import apply_platt, fit_platt_calibrator

    y_tr = np.asarray(y_train, dtype=int)
    y_ca = np.asarray(y_cal, dtype=int)
    p1_cal, p1_tests = precomputed_stage1
    df_tests = [t[0] for t in test_sets]

    m_ch = len(channels)
    costs = np.array([float(ch.cost) for ch in channels], dtype=float)
    all_ch_cols = list(dict.fromkeys(c for ch in channels for c in ch.columns))

    enc_b = FastFrameEncoder().fit(df_train[base_cols])
    xb_tr = enc_b.transform(df_train[base_cols]).astype(np.float32)
    xb_ca = enc_b.transform(df_cal[base_cols]).astype(np.float32)
    xb_tests = [enc_b.transform(dft[base_cols]).astype(np.float32) for dft in df_tests]

    s2_tr_df = df_train[all_ch_cols]
    s2_ca_df = df_cal[all_ch_cols]
    s2_te_dfs = [dft[all_ch_cols] for dft in df_tests]

    ch_slices: list[tuple[int, int]] = []
    z_blocks_tr: list[np.ndarray] = []
    z_blocks_ca: list[np.ndarray] = []
    z_blocks_te: list[list[np.ndarray]] = [[] for _ in df_tests]
    col_cursor = 0
    for ch in channels:
        cols_c = list(ch.columns)
        enc_c = FastFrameEncoder(include_missing_indicators=True).fit(s2_tr_df[cols_c])
        b_tr = enc_c.transform(s2_tr_df[cols_c]).astype(np.float32)
        b_ca = enc_c.transform(s2_ca_df[cols_c]).astype(np.float32)
        w_c = b_tr.shape[1]
        ch_slices.append((col_cursor, col_cursor + w_c))
        col_cursor += w_c
        z_blocks_tr.append(b_tr)
        z_blocks_ca.append(b_ca)
        for i_t, df_s in enumerate(s2_te_dfs):
            z_blocks_te[i_t].append(enc_c.transform(df_s[cols_c]).astype(np.float32))

    z_full_tr = np.hstack(z_blocks_tr)
    z_full_ca = np.hstack(z_blocks_ca)
    z_full_tes = [np.hstack(blks) for blks in z_blocks_te]
    sentinel = np.float32(-999.0)

    def _build_masked_features(
        xb: np.ndarray, z_full: np.ndarray, mask_mat: np.ndarray
    ) -> np.ndarray:
        z_m = np.full_like(z_full, sentinel)
        for j, (c0, c1) in enumerate(ch_slices):
            obs_j = mask_mat[:, j] > 0.5
            if np.any(obs_j):
                z_m[obs_j, c0:c1] = z_full[obs_j, c0:c1]
        return np.hstack([xb, z_m, mask_mat.astype(np.float32)])

    rng = np.random.default_rng(seed + 303)

    def _sample_affordable_masks(n_rows: int, rng_local: np.random.Generator) -> np.ndarray:
        masks = np.zeros((n_rows, m_ch), dtype=np.float32)
        for i in range(n_rows):
            perm = rng_local.permutation(m_ch)
            target_k = int(rng_local.integers(1, min(4, m_ch) + 1))
            rem_b = budget
            picked = 0
            for j in perm:
                if picked >= target_k:
                    break
                if costs[j] <= rem_b + 1e-9:
                    masks[i, j] = 1.0
                    rem_b -= costs[j]
                    picked += 1
        return masks

    n_tr = y_tr.size
    ell1_tr = logit(np.clip(oof_p_base_train, 1e-5, 1.0 - 1e-5))
    x_rep_list: list[np.ndarray] = []
    y_rep_list: list[np.ndarray] = []
    init_rep_list: list[np.ndarray] = []
    for _ in range(n_masks_per_row):
        m_mat = _sample_affordable_masks(n_tr, rng)
        x_rep_list.append(_build_masked_features(xb_tr, z_full_tr, m_mat))
        y_rep_list.append(y_tr)
        init_rep_list.append(ell1_tr)

    clf_masked = lgb.LGBMClassifier(
        n_estimators=140,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=30,
        max_delta_step=1.0,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=seed + 41,
        n_jobs=n_jobs,
        verbose=-1,
    )
    clf_masked.fit(
        np.vstack(x_rep_list), np.concatenate(y_rep_list), init_score=np.concatenate(init_rep_list)
    )

    m_cal_rand = _sample_affordable_masks(y_ca.size, rng)
    raw_ca_masked = logit(p1_cal) + clf_masked.predict(
        _build_masked_features(xb_ca, z_full_ca, m_cal_rand), raw_score=True
    )
    platt_m = fit_platt_calibrator(raw_ca_masked, y_ca)

    def _predict_masked_p(
        xb: np.ndarray, z_full: np.ndarray, p1_vec: np.ndarray, mask_mat: np.ndarray
    ) -> np.ndarray:
        has_any = np.any(mask_mat > 0.5, axis=1)
        out = p1_vec.copy()
        if np.any(has_any):
            xm = _build_masked_features(xb[has_any], z_full[has_any], mask_mat[has_any])
            raw = logit(p1_vec[has_any]) + clf_masked.predict(xm, raw_score=True)
            out[has_any] = np.clip(apply_platt(raw, platt_m), 1e-6, 1.0 - 1e-6)
        return out

    sub_n = min(n_tr, 6000)
    sub_idx = rng.choice(n_tr, size=sub_n, replace=False) if sub_n < n_tr else np.arange(n_tr)
    xb_s, zf_s = xb_tr[sub_idx], z_full_tr[sub_idx]
    p1_s = np.clip(oof_p_base_train[sub_idx], 1e-5, 1.0 - 1e-5)

    m_zero = np.zeros((sub_n, m_ch), dtype=np.float32)
    m_part = np.zeros((sub_n, m_ch), dtype=np.float32)
    for i in range(sub_n):
        j0 = int(rng.integers(0, m_ch))
        if costs[j0] < budget - 0.5:
            m_part[i, j0] = 1.0

    states_m = np.vstack([m_zero, m_part])
    states_xb = np.vstack([xb_s, xb_s])
    states_zf = np.vstack([zf_s, zf_s])
    states_p1 = np.concatenate([p1_s, p1_s])
    p_cur_state = _predict_masked_p(states_xb, states_zf, states_p1, states_m)
    x_state_feat = np.column_stack(
        [
            logit(p_cur_state).astype(np.float32),
            _build_masked_features(states_xb, states_zf, states_m),
        ]
    )

    cfn, cfp = contract.cost_fn, contract.cost_fp
    value_regs: list[lgb.LGBMRegressor | None] = []
    for j in range(m_ch):
        valid = (states_m[:, j] < 0.5) & ((states_m @ costs) + costs[j] <= budget + 1e-9)
        if int(np.sum(valid)) < 40:
            value_regs.append(None)
            continue
        m_plus = states_m[valid].copy()
        m_plus[:, j] = 1.0
        p_plus = _predict_masked_p(states_xb[valid], states_zf[valid], states_p1[valid], m_plus)
        p_base_v = p_cur_state[valid]
        if criterion == "cmi":
            target_val = np.maximum(
                0.0,
                p_plus * np.log(p_plus / p_base_v)
                + (1.0 - p_plus) * np.log((1.0 - p_plus) / (1.0 - p_base_v)),
            )
        elif criterion == "direct_voi":
            # Expected decision-risk reduction using Bayes point risk
            l_cur = np.minimum(p_base_v * cfn, (1.0 - p_base_v) * cfp)
            l_plus = np.minimum(p_plus * cfn, (1.0 - p_plus) * cfp)
            target_val = np.maximum(0.0, l_cur - l_plus)
        else:
            raise ValueError(f"Unknown AFA criterion: {criterion}")

        reg_j = lgb.LGBMRegressor(
            n_estimators=55,
            learning_rate=0.06,
            num_leaves=15,
            min_child_samples=25,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=seed + 200 + j,
            n_jobs=n_jobs,
            verbose=-1,
        )
        reg_j.fit(x_state_feat[valid], target_val)
        value_regs.append(reg_j)

    def _rollout(
        xb: np.ndarray,
        z_full: np.ndarray,
        p1_vec: np.ndarray,
        lambda_stop: float,
        max_steps: int = 3,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = p1_vec.size
        mask_cur = np.zeros((n, m_ch), dtype=np.float32)
        spent_cur = np.zeros(n, dtype=float)
        p_cur = p1_vec.copy()
        active = np.ones(n, dtype=bool)
        step1_best = np.zeros(n, dtype=float)

        for step in range(max_steps):
            act_idx = np.flatnonzero(active)
            if act_idx.size == 0:
                break
            x_st = np.column_stack(
                [
                    logit(p_cur[act_idx]).astype(np.float32),
                    _build_masked_features(xb[act_idx], z_full[act_idx], mask_cur[act_idx]),
                ]
            )
            cur_cost_vec = cost_schedule.instance_cost_vector(mask_cur[act_idx], channels)
            ratios = np.full((act_idx.size, m_ch), -np.inf, dtype=float)
            for j, reg_j in enumerate(value_regs):
                if reg_j is None:
                    continue
                can_buy = (mask_cur[act_idx, j] < 0.5) & (
                    spent_cur[act_idx] + costs[j] <= budget + 1e-9
                )
                if not np.any(can_buy):
                    continue
                m_try = mask_cur[act_idx[can_buy]].copy()
                m_try[:, j] = 1.0
                inc_cost = np.maximum(
                    1e-6,
                    cost_schedule.instance_cost_vector(m_try, channels) - cur_cost_vec[can_buy],
                )
                pred_val = np.maximum(0.0, reg_j.predict(x_st[can_buy]))
                ratios[can_buy, j] = pred_val / inc_cost

            best_j = np.argmax(ratios, axis=1)
            best_r = ratios[np.arange(act_idx.size), best_j]
            if step == 0:
                step1_best[act_idx] = np.where(np.isfinite(best_r), best_r, 0.0)

            go = np.isfinite(best_r) & (best_r > lambda_stop)
            if not np.any(go):
                break
            go_rows = act_idx[go]
            go_ch = best_j[go]
            mask_cur[go_rows, go_ch] = 1.0
            spent_cur[go_rows] += costs[go_ch]
            p_cur[go_rows] = _predict_masked_p(
                xb[go_rows], z_full[go_rows], p1_vec[go_rows], mask_cur[go_rows]
            )
            active[~np.isin(np.arange(n), go_rows)] = False

        return p_cur, mask_cur, step1_best

    # Tune lambda_stop strictly on D_cal under the EXACT PanelCostSchedule ledger!
    _, _, cal_r1 = _rollout(xb_ca, z_full_ca, p1_cal, lambda_stop=1e9, max_steps=1)
    pos_r = cal_r1[cal_r1 > 0.0]
    if pos_r.size >= 10:
        cand_lambdas = np.unique(
            np.concatenate(
                [np.quantile(pos_r, [0.10, 0.25, 0.40, 0.55, 0.70, 0.80, 0.90, 0.96]), [1e9]]
            )
        )
    else:
        cand_lambdas = np.array([0.1, 0.5, 1.0, 1e9])

    best_lam = float(cand_lambdas[-1])
    best_cal_nb = -np.inf
    for lam in cand_lambdas:
        p_try, m_try, _ = _rollout(xb_ca, z_full_ca, p1_cal, float(lam))
        c_try = cost_schedule.instance_cost_vector(m_try, channels)
        nb_try = net_benefit_with_cost_ledger_per_1k(y_ca, p_try, c_try, contract)
        if nb_try > best_cal_nb:
            best_cal_nb = nb_try
            best_lam = float(lam)

    outputs: list[dict[str, object]] = []
    for i_t, (_, yt_raw) in enumerate(test_sets):
        y_te = np.asarray(yt_raw, dtype=int)
        p_te, m_te, _ = _rollout(xb_tests[i_t], z_full_tes[i_t], p1_tests[i_t], best_lam)
        c_te = cost_schedule.instance_cost_vector(m_te, channels)
        outputs.append(
            {
                "p_auto_te": p_te,
                "mask_te": m_te,
                "cost_vec_te": c_te,
                "queried_any_te": np.any(m_te > 0.5, axis=1),
                "nb_ledger_per_1k": net_benefit_with_cost_ledger_per_1k(
                    y_te, p_te, c_te, contract
                ),
                "mean_cost_cfp": float(np.mean(c_te)),
                "query_rate": float(np.mean(np.any(m_te > 0.5, axis=1))),
            }
        )
    return outputs
