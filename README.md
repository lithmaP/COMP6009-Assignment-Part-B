# COMP6009 Assignment Part B — Biogas Plant Intelligent Monitoring and Decision System

Bayesian inference, HMM filtering and expected-utility decision making for an anaerobic digester
with hidden state X ∈ {Stable, Stressed, Critical}. All algorithms are implemented directly with
numpy (no HMM or probabilistic-inference libraries).

## Files

| File | Contents |
|------|----------|
| `biogas_model.py` | Model tables (prior, sensor models, transition, utility) and the algorithms: `bayes_posterior`, `hmm_filter`, `recommend_action` |
| `main.py` | Runs Tasks 1–3, prints results and writes all tables/plots to `results/` |
| `scenario_*.csv` | Supplied Sensor 1 observation sequences (30 steps each) |
| `results/` | Generated outputs (see below) |

## Setup and running

Requires Python 3.9+.

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

## Outputs (`results/`)

| Task | Files |
|------|-------|
| 1 — Bayesian inference | `task1_bayes_posteriors.csv` (unnormalised values, P(evidence), posterior for each case), `task1_posteriors.png` |
| 2 — HMM filtering | `task2_hmm_scenario_{A,B,C}.csv` (predicted, unnormalised and filtered belief + most probable state per step), `task2_hmm_scenario_{A,B,C}.png` |
| 3 — Expected utility | `task3_decisions_scenario_{A,B,C}.csv` (EU of each action, recommended action, action for most probable state), `task3_decisions_scenario_{A,B,C}.png`, `task3_summary.csv` |

## Method summary

- **Task 1:** `P(X | e) ∝ P(X) · Π_i P(e_i | X)` (sensors conditionally independent given X), then normalised.
  The script also checks the hand-calculated Warning-only case against the implementation.
- **Task 2:** at t = 1 the prior is updated directly. For t > 1:
  1. prediction `b̄_t = b_{t-1} T`
  2. observation update `b̄_t ⊙ O[:, e_t]`
  3. normalisation
- **Task 3:** `EU(a) = Σ_x b_t(x) U(a, x)`. The recommended action is the argmax.
