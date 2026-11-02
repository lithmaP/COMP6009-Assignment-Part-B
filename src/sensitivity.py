"""Sensitivity of the recommended actions to the modelling assumptions.

Each analysis changes one part of the baseline model, reruns HMM filtering and the
expected-utility decision for every scenario, and compares the result with the baseline:
  1. Utility:    penalty for Continue when the digester is Critical (baseline -20)
  2. Sensor 1:   noise level eps, P'(e | x) = (1 - eps) P(e | x) + eps / 3
  3. Transition: persistence lam, T' = (1 - lam) T + lam I (states change less often)
"""
import numpy as np
import pandas as pd

from .biogas_model import (
    ACTIONS, PRIOR, SENSOR1, STATES, TRANSITION, UTILITY, hmm_filter, recommend_action,
)

CONTINUE_CRITICAL_PENALTIES = [-5, -10, -20, -30, -40]
SENSOR_NOISE_LEVELS = [0.0, 0.2, 0.4, 0.6]
PERSISTENCE_LEVELS = [0.0, 0.2, 0.4, 0.6]


def run(observations, transition=TRANSITION, sensor=SENSOR1, utility=UTILITY):
    """Filtered beliefs (n x 3) and recommended actions for one observation sequence."""
    beliefs = np.array([s["belief"] for s in hmm_filter(PRIOR, transition, sensor, observations)])
    actions = [recommend_action(b, utility)[0] for b in beliefs]
    return beliefs, actions


def summarise(analysis, parameter, value, scenario, beliefs, actions, baseline_actions):
    counts = pd.Series(actions).value_counts()
    return {
        "analysis": analysis,
        "parameter": parameter,
        "value": value,
        "scenario": scenario,
        **{a: int(counts.get(a, 0)) for a in ACTIONS},
        "n_switches": int(sum(a != b for a, b in zip(actions[1:], actions[:-1]))),
        "n_differs_from_baseline": int(sum(a != b for a, b in zip(actions, baseline_actions))),
        "mean_max_belief": float(beliefs.max(axis=1).mean()),
        "mean_P(Critical)": float(beliefs[:, STATES.index("Critical")].mean()),
    }


def sensitivity_analysis(observations_by_scenario):
    """Runs all three analyses. Returns one tidy DataFrame (one row per setting and scenario)."""
    rows = []
    for key, obs in observations_by_scenario.items():
        base_beliefs, base_actions = run(obs)

        for penalty in CONTINUE_CRITICAL_PENALTIES:
            utility = UTILITY.copy()
            utility[ACTIONS.index("Continue"), STATES.index("Critical")] = penalty
            actions = [recommend_action(b, utility)[0] for b in base_beliefs]
            rows.append(summarise("utility", "U(Continue, Critical)", penalty, key,
                                  base_beliefs, actions, base_actions))

        for eps in SENSOR_NOISE_LEVELS:
            sensor = (1 - eps) * SENSOR1 + eps / SENSOR1.shape[1]
            beliefs, actions = run(obs, sensor=sensor)
            rows.append(summarise("sensor_noise", "Sensor 1 noise eps", eps, key,
                                  beliefs, actions, base_actions))

        for lam in PERSISTENCE_LEVELS:
            transition = (1 - lam) * TRANSITION + lam * np.eye(len(STATES))
            beliefs, actions = run(obs, transition=transition)
            rows.append(summarise("transition", "Transition persistence lam", lam, key,
                                  beliefs, actions, base_actions))
    return pd.DataFrame(rows)
