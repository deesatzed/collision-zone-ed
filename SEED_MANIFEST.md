# Deterministic RNG Seed & Provenance Manifest (`collision-zone-ed` `v2.3.0`)

## 1. Algorithmic Provenance (`v1.0` Commit `8c8c4c8` $\to$ `v2.3.0` Commit `5362988`)

Core algorithmic defaults were committed in `src/collision_zone/three_predicates.py` and `scripts/run_paper2_track_b_hardened.py` at initial `v1.0` commit `8c8c4c8` (`2026-11-22T18:16:58-05:00`) prior to executing sensitivity grids or post-review repairs:

- **Stage-A Discovery Kernel Cap:** `kappa = 24.0` (`run_sequential_discovery`, `8c8c4c8`)
- **Stage-A Score Stratification Bins:** `n_bins = 8` (`B_score = 8`, `sample_pilot`, `8c8c4c8`)
- **Stage-A Relevance Floor (Naive Baseline):** `floor = 0.05` (`collision_zone_relevance_weights`, `8c8c4c8`)
- **Stage-2 Specialist Kernel Cap & Active-Mass Floor:** `kappa_max = min(20.0, kappa) = 20.0`, `min_cz_ess = 240.0` (`M_min = 240.0`, `FluidCollisionBackboneClassifier`, `8c8c4c8`)
- **Empirical Martingale VOI Discretization:** `12` Stage-1 quantile bins (`np.linspace(0.0, 1.0, 13)`), `K = 33` log-odds update quantiles (`np.linspace(0.02, 0.98, 33)`, `evaluate_stage2_multi_test`, `8c8c4c8`)
- **Primary Sepsis Observation Landmark (`6h`):** Selected in `8c8c4c8` (`sepsis_setA_bidmc_6h`, `sepsis_setB_emory_6h`). In `v2.3.0` (`5362988`), a 16-cell `Landmark x Future Horizon` audit (`4/6/8/12h x {next 2h, 6h, 12h, remaining stay}`) established that `271/349` (`77.7%`) of Emory `6h` remaining-stay events occur `> 12h` post-landmark, retiring the earlier `6h` Sepsis-3 pre-onset mechanistic interpretation.

All `v2.3.0` explicit `PanelCostSchedule` procedures, Common-Panel Bagged VOI repairs, `6x2` synthetic and `5x3` clinical discovery factorials, `36`-cell regime mapping, `16`-cell horizon matrices, historical prior-shift splits, and previously untested Yale Campus C (`N = 71,706` all-comers; `N = 31,430` `ESI 4-5`) evaluations are post-review methodological repairs committed at `5362988` (`2026-11-27`).

---

## 2. `v2.3.0` Modular Remediation & Panel-Ledger Suite (`scripts/run_paper2_v2_3_remediation.py`, Commit `5362988`)

| Block / Checkpoint JSON | Sub-Experiments & Replicate Counts | Deterministic RNG Seeds | Bootstrap Resamples (`B`) & Seeds |
| :--- | :--- | :--- | :--- |
| **`block1_synthetic.json`** | (i) `6x2` Discovery x Backbone Factorial (`300` reps, `N = 8,000`); (ii) `6`-Scenario FWER Stress Suite (`6 x 200` reps, Cross-Fit vs. `50/50` Split-Confirmation); (iii) Quadrature Oracle VOI Audit (`200` reps); (iv) `36`-Cell Regime Map (`36 x 40 = 1,440` runs) | Factorial: `10000 + r`; Stress: `20000 + 1000*scen + r`; VOI: `30000 + r`; Regime map: `40000 + 100*cell + r` | Exact Clopper–Pearson 95% CIs; Paired Davison–Hinkley `B = 10,000` (`seeds = 901..936`) |
| **`block2_sepsis_primary_ledger.json`** | BIDMC (`8,345`) $\to$ Emory (`6,463`) Track B1b (`6h` window summaries & single-draw `_last`): Common-Panel Bagged VOI, Union-Panel, Per-Split (`R = 10`), Frac. Ens., `5x3` Clinical Factorial, Ungated Residual S2, Channel-Billed AFA, Directional VOI Triad, Cost Sweep (`0x..5x`), Track A (`19,972 -> 19,696`), Track B1a | Full-source lock: `seed = 777`; `R = 10` splits: `700..709` (OOF `800..809`, disc. `900..909`, S2/AFA `1500..1509`) | `B = 10,000` Davison–Hinkley paired bootstraps (`seeds = 2101..2140`) |
| **`block3_horizons_shift_missingness.json`** | (i) `16`-Cell `4/6/8/12h x {2h, 6h, 12h, remaining stay}` Horizon Matrix (`R = 6` splits); (ii) SOFA Organ-Channel Exclusion Audit; (iii) Track B2 Prior-Shift Taxonomy (`25%` historical slice vs. `75%` untouched Emory holdout `N = 14,755`, unlabeled Saerens EM, error sweep); (iv) `14`-Channel Missingness & Availability Audit | Horizon splits: `700..705`; Track B2 `25/75` chronological/stratified split: `seed = 8200`; `R = 8` models: `700..707` | `B = 10,000` Davison–Hinkley paired bootstraps (`seeds = 3101..3125`) |
| **`block4_yale_and_sensitivity.json`** | (i) Yale Campus A $\to$ Campus B & Untested Campus C (`ESI 4-5` `N = 51,205` & `31,430`, `R = 6`; All-Comers `N = 166,497` & `71,706`, `R = 5`); (ii) `10` External Campus B+C Subgroups (`N = 82,635`); (iii) `R = 200` Campus B `CfBH` Burn-In Splits (`n_burn in {250..4000}`); (iv) `9` Distinct `(kappa_max, B_score)` Sepsis Sensitivity Cells (noting `M(kappa) > 320` renders `M_min in {160, 240, 320}` numerically identical to `<= 6e-8`) | Yale Campus A lock: `5050`; `R = 6/5` splits: `5100 + s`; `CfBH` `200` subsamples: `6000..6199`; `9`-cell grid: `600 + kappa_max + B_score` | Cohorts & `9`-grid: `B = 10,000` (`seeds = 4101..4130`); `10` Subgroups: `B = 5,000` (`seeds = 4200..4209`) |

---

## 3. Foundational `v2.0`–`v2.2` Seed Ledger (Preserved for Backward Reproducibility)

| Experiment / Suite | Script | Split / Replicate Seeds | Stage-1 / Discovery / Stage-2 Seeds | Bootstrap Resamples (`B`) & Seeds |
| :--- | :--- | :--- | :--- | :--- |
| **Sim 1: True-Negative Dilution (`1,000` reps)** | `scripts/run_paper2_v2_suite.py` | Data: `1000..1999`; split: `2000..2999` | OOF: `3000..3999`; Pilot/disc.: `4000..4999`; S2: `5000..5999` | Clopper–Pearson 95% CIs; `B = 10,000` (`seed = 42`) |
| **Sim 2: Stage-2 Leaf Starvation (`3 x 15` reps)** | `scripts/run_paper2_experiments.py` | Regimes `1..3`, reps `0..14`: `2100 + 100*reg + r` | Stage-1: `2500 + r`; Stage-2: `2600 + r` | Descriptive mean +/- SE across `15` draws |
| **Yale Track B3 (`Blinded ESI 4-5 OOF`, `N = 152,995`)** | `scripts/run_paper2_v2_suite.py` | Encounter split: `4100..4105` (note: `pat_mrn_id` in Hong et al. 2018 is encounter-level across all `560,486` rows) | OOF: `4200..4205`; Disc.: `4300..4305`; S2: `4400..4405` | `B = 10,000` (`seeds = 81..82`) |
| **UCI Public Benchmark Sanity Checks (`WDBC` & `Diabetes`)** | `scripts/run_paper2_v2_2_polish.py` | 3-way split: `3100..3109` | OOF: `3200..3209`; S1: `3300..3309`; AFA: `3400..3709` | Paired comparison across `10` splits |
