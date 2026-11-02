# COMP6009 Assignment Part B - Biogas Plant Intelligent Monitoring and Decision System

Bayesian inference, HMM filtering and expected-utility decision making for an anaerobic digester
with hidden state X ∈ {Stable, Stressed, Critical}. All algorithms are implemented directly with
numpy (no HMM or probabilistic-inference libraries).

## Folder structure

```text
.
├── main.py                 # Entry point: runs Tasks 1-3 and writes everything to results/
├── requirements.txt
├── src/
│   ├── biogas_model.py     # Model tables + bayes_posterior, hmm_filter, recommend_action
│   ├── sensitivity.py      # Sensitivity of the recommended actions to modelling assumptions
│   └── report_figures.py   # Combined figures used in the report
├── data/                   # Supplied Sensor 1 observation sequences (30 steps each)
│   ├── scenario_A_stable.csv
│   ├── scenario_B_deteriorating.csv
│   └── scenario_C_noisy.csv
├── assignment/             # Assignment specification
└── results/
    ├── tables/             # All numerical results (CSV)
    ├── figures/            # Per-task plots
    └── report_figures/     # Combined figures used in the report
```

## Setup and running

Requires Python 3.9+.

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

## Outputs

| Task | Files (`results/tables/`, `results/figures/`) |
| --- | --- |
| 1 — Bayesian inference | `task1_bayes_posteriors.csv` (unnormalised values, P(evidence), posterior for each Sensor 1 reading, each pH reading alone, and Warning + Abnormal pH), `task1_posteriors.png` |
| 2 — HMM filtering | `task2_hmm_scenario_{A,B,C}.csv` (predicted, unnormalised and filtered belief + most probable state per step), `task2_hmm_scenario_{A,B,C}.png` |
| 3 — Expected utility | `task3_decisions_scenario_{A,B,C}.csv` (EU of each action, recommended action, action for most probable state), `task3_decisions_scenario_{A,B,C}.png`, `task3_summary.csv` |
| Sensitivity | `sensitivity_analysis.csv` (action counts, switches, steps differing from baseline and mean belief for each setting and scenario) |

### Report figures (`results/report_figures/`)

| Figure | Report section |
| --- | --- |
| `fig_task1_evidence_combination.png` | 2: prior → Warning → Warning + Abnormal pH |
| `fig_task2_scenario_comparison.png` | 3: filtered belief for all three scenarios |
| `fig_task3_action_timeline.png` | 4: max-EU action vs most-probable-state action over time |
| `fig_task3_decision_regions.png` | 4: optimal-action regions on the belief simplex, with the HMM beliefs overlaid |
| `fig_sensitivity.png` | 3-4: how the recommended actions change when one modelling assumption is changed |

## Method summary

- **Task 1:** `P(X | e) ∝ P(X) · Π_i P(e_i | X)` (sensors conditionally independent given X), then normalised.
  The script also checks the hand-calculated Warning-only case against the implementation.
- **Task 2:** at t = 1 the prior is updated directly. For t > 1:
  1. prediction `b̄_t = b_{t-1} T`
  2. observation update `b̄_t ⊙ O[:, e_t]`
  3. normalisation
- **Task 3:** `EU(a) = Σ_x b_t(x) U(a, x)`. The recommended action is the argmax.
- **Sensitivity:** one assumption is changed at a time and Tasks 2-3 are rerun for every scenario:
  - utility: `U(Continue, Critical)` from -5 to -40 (baseline -20)
  - Sensor 1 reliability: `P'(e | x) = (1 - ε) P(e | x) + ε / 3` for ε = 0, 0.2, 0.4, 0.6
  - transition persistence: `T' = (1 - λ) T + λ I` for λ = 0, 0.2, 0.4, 0.6
