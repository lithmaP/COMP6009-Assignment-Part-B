# COMP6009 Assignment Part B - Biogas Plant Intelligent Monitoring and Decision System

Repository: <https://github.com/lithmaP/COMP6009-Assignment-Part-B>

## Overview

This project models an anaerobic biogas digester whose condition is a hidden state
X ∈ {Stable, Stressed, Critical}. It combines sensor evidence, tracks the state over time and
recommends an operating action. The objectives are to:

1. infer the digester state from Sensor 1 and pH (Sensor 2) readings using Bayes' rule (Task 1);
2. track the state over 30 time steps in three scenarios with HMM forward filtering (Task 2);
3. recommend Continue, Reduce Feed or Intervene at each step by maximising expected utility (Task 3);
4. test how sensitive the recommendations are to the modelling assumptions.

All algorithms are implemented directly with NumPy; no HMM or probabilistic-inference libraries
are used. The model tables (prior, sensor models, transition model and utilities) are defined in
`src/biogas_model.py`.

## Project structure

```text
.
├── main.py                 # Entry point: runs Tasks 1-3 and the sensitivity analysis
├── requirements.txt        # Pinned dependencies
├── src/
│   ├── biogas_model.py     # Model tables + bayes_posterior, hmm_filter, recommend_action
│   ├── sensitivity.py      # Sensitivity analysis of the recommended actions
│   └── report_figures.py   # Combined figures used in the report
├── data/                   # Supplied Sensor 1 observation sequences (30 steps each)
│   ├── scenario_A_stable.csv
│   ├── scenario_B_deteriorating.csv
│   └── scenario_C_noisy.csv
├── results/                # Generated outputs (see below)
├── assignment/             # Assignment specification (PDF)
├── report/                 # Report (PDF for submission, DOCX source)
└── submission/             # Exactly what is submitted
    ├── code/               # Code package (has its own README)
    └── report/             # Report PDF
```

## Prerequisites

- Python 3.12 or newer. The pinned NumPy 2.5.4 requires Python ≥ 3.12, so Python 3.11 and
  older cannot install the requirements.
- `pip` (included with Python).

The setup below has been tested in clean virtual environments with Python 3.12.3 and 3.13.18
on Linux.

## Installation and running

Run all commands from the project root (the folder containing `main.py`).

**1. Create a virtual environment**

| Platform | Command |
| --- | --- |
| Windows | `py -3 -m venv .venv` |
| macOS / Linux | `python3 -m venv .venv` |

**2. Activate it**

| Platform | Command |
| --- | --- |
| Windows (PowerShell) | `.venv\Scripts\Activate.ps1` |
| Windows (Command Prompt) | `.venv\Scripts\activate.bat` |
| macOS / Linux | `source .venv/bin/activate` |

**3. Install the dependencies**

```bash
pip install -r requirements.txt
```

This installs numpy 2.5.4, pandas 3.1.0 and matplotlib 3.11.2 (and their own dependencies).

**4. Run the project**

```bash
python main.py
```

The script prints the results of each task to the terminal and writes all tables and figures
to `results/`, creating the folders if needed. Existing files in `results/` are overwritten,
and a rerun with the pinned versions reproduces them.

## Assignment tasks

- **Task 1 - Bayesian inference:** `P(X | e) ∝ P(X) · Π_i P(e_i | X)`, assuming the sensors are
  conditionally independent given X, then normalised. Posteriors are computed for each Sensor 1
  reading, each pH reading, and Sensor 1 = Warning combined with Abnormal pH. The script also
  checks the hand-calculated Warning-only case against the implementation.
- **Task 2 - HMM filtering:** for each scenario, the belief at t = 1 is the prior updated with
  the first observation. For t > 1: prediction `b̄_t = b_{t-1} T`, observation update
  `b̄_t ⊙ O[:, e_t]`, then normalisation.
- **Task 3 - Expected utility:** `EU(a) = Σ_x b_t(x) U(a, x)` for each action using the filtered
  belief; the recommended action is the one with the highest expected utility. It is compared
  with the action that would be chosen for the most probable state alone.

## Sensitivity analysis

One modelling assumption is changed at a time and Tasks 2-3 are rerun for every scenario:

- **Utility:** `U(Continue, Critical)` set to -5, -10, -20, -30 and -40 (baseline -20).
- **Sensor 1 reliability:** `P'(e | x) = (1 - ε) P(e | x) + ε / 3` for ε = 0, 0.2, 0.4, 0.6.
- **Transition persistence:** `T' = (1 - λ) T + λ I` for λ = 0, 0.2, 0.4, 0.6.

For each setting and scenario the results record how often each action is recommended, the number
of action switches, the number of steps that differ from the baseline, the mean maximum belief
and the mean P(Critical).

## Outputs

All outputs are saved under `results/`.

**Tables (`results/tables/`)**

| File | Contents |
| --- | --- |
| `task1_bayes_posteriors.csv` | Unnormalised values, P(evidence), posterior and most probable state for each evidence case |
| `task2_hmm_scenario_{A,B,C}.csv` | Predicted, unnormalised and filtered belief, and most probable state, per time step |
| `task3_decisions_scenario_{A,B,C}.csv` | Filtered belief, EU of each action, recommended action and action for the most probable state, per time step |
| `task3_summary.csv` | Action counts, action switches and disagreements with the most-probable-state action per scenario |
| `all_scenarios_beliefs_and_decisions.csv` | All 90 time steps of scenarios A-C: filtered belief, most probable state, EU of each action and recommended action |
| `sensitivity_analysis.csv` | Sensitivity analysis results for each setting and scenario |

**Per-task figures (`results/figures/`)**

| File | Contents |
| --- | --- |
| `task1_posteriors.png` | Task 1 posteriors for each evidence case |
| `task2_hmm_scenario_{A,B,C}.png` | Filtered belief over time with the observations |
| `task3_decisions_scenario_{A,B,C}.png` | Expected utility of each action and the recommended action over time |

**Report figures (`results/report_figures/`)**

| File | Report section |
| --- | --- |
| `fig_task1_evidence_combination.png` | 2: prior → Warning → Warning + Abnormal pH |
| `fig_task2_scenario_comparison.png` | 3: filtered belief for all three scenarios |
| `fig_task3_action_timeline.png` | 4: max-EU action vs most-probable-state action over time |
| `fig_task3_decision_regions.png` | 4: optimal-action regions on the belief simplex, with the HMM beliefs overlaid |
| `fig_sensitivity.png` | 3-4: change in the recommended actions under each sensitivity setting |

In total, `python main.py` produces 10 CSV files and 12 PNG figures.
