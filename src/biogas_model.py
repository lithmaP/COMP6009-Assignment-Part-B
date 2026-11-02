"""Probability model and inference/decision algorithms for the biogas digester.

All inference is implemented directly with numpy (no HMM / Bayes libraries).
"""
import numpy as np

STATES = ["Stable", "Stressed", "Critical"]
OBS_S1 = ["Normal", "Warning", "Alarm"]
OBS_S2 = ["Normal pH", "Abnormal pH"]
ACTIONS = ["Continue", "Reduce Feed", "Intervene"]

# Prior P(X0)
PRIOR = np.array([0.70, 0.20, 0.10])

# Sensor 1 observation model P(E1 | X): rows = states, cols = OBS_S1
SENSOR1 = np.array([
    [0.80, 0.15, 0.05],
    [0.25, 0.60, 0.15],
    [0.05, 0.25, 0.70],
])

# Sensor 2 (pH) observation model P(E2 | X): rows = states, cols = OBS_S2
SENSOR2 = np.array([
    [0.90, 0.10],
    [0.50, 0.50],
    [0.15, 0.85],
])

# Transition model P(X_t+1 | X_t): rows = current state, cols = next state
TRANSITION = np.array([
    [0.80, 0.18, 0.02],
    [0.25, 0.60, 0.15],
    [0.05, 0.30, 0.65],
])

# Utility U(a, x): rows = ACTIONS, cols = STATES
UTILITY = np.array([
    [10, 2, -20],
    [5, 6, -5],
    [-4, 3, 8],
], dtype=float)

SENSORS = {
    "S1": (SENSOR1, OBS_S1),
    "S2": (SENSOR2, OBS_S2),
}


def normalise(v):
    total = v.sum()
    if total <= 0:
        raise ValueError("Evidence has zero probability under the model.")
    return v / total


def bayes_posterior(prior, evidence, sensors=SENSORS):
    """Posterior P(X | evidence) assuming sensors are conditionally independent given X.

    evidence: dict mapping sensor name -> observed value, e.g. {"S1": "Warning", "S2": "Abnormal pH"}.
    Returns (unnormalised, evidence_probability, posterior).
    """
    unnormalised = np.asarray(prior, dtype=float).copy()
    for name, value in evidence.items():
        table, labels = sensors[name]
        unnormalised = unnormalised * table[:, labels.index(value)]
    z = unnormalised.sum()
    return unnormalised, z, normalise(unnormalised)


def hmm_filter(prior, transition, emission, observations, obs_labels=OBS_S1):
    """Forward filtering P(X_t | e_1:t) for every t.

    t = 1: the prior is updated directly with the first observation.
    t > 1: Prediction -> Observation update -> Normalisation.
    Returns a list of dicts holding each intermediate step.
    """
    steps = []
    belief = np.asarray(prior, dtype=float)
    for t, obs in enumerate(observations, start=1):
        # 1. Prediction: sum_x P(X_t | x) b_{t-1}(x)
        predicted = belief if t == 1 else belief @ transition
        # 2. Observation update: multiply by P(e_t | X_t)
        unnormalised = predicted * emission[:, obs_labels.index(obs)]
        # 3. Normalisation
        belief = normalise(unnormalised)
        steps.append({
            "t": t,
            "observation": obs,
            "predicted": predicted,
            "unnormalised": unnormalised,
            "belief": belief,
        })
    return steps


def expected_utilities(belief, utility_table=UTILITY):
    """EU(a) = sum_x b(x) U(a, x) for every action."""
    return utility_table @ np.asarray(belief, dtype=float)


def recommend_action(belief, utility_table=UTILITY, actions=ACTIONS):
    """Return (best_action, expected utility per action)."""
    eu = expected_utilities(belief, utility_table)
    return actions[int(np.argmax(eu))], eu
