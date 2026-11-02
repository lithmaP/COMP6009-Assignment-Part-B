"""Runs Tasks 1-3 and writes all tables and plots to results/."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import report_figures
from src.biogas_model import (
    ACTIONS, OBS_S1, PRIOR, SENSOR1, STATES, TRANSITION, UTILITY,
    bayes_posterior, hmm_filter, recommend_action,
)

ROOT = Path(__file__).parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
SCENARIOS = {
    "A": ("Scenario A - Stable operation", "scenario_A_stable.csv"),
    "B": ("Scenario B - Deteriorating operation", "scenario_B_deteriorating.csv"),
    "C": ("Scenario C - Noisy monitoring", "scenario_C_noisy.csv"),
}
STATE_COLOURS = {"Stable": "tab:green", "Stressed": "tab:orange", "Critical": "tab:red"}
ACTION_COLOURS = {"Continue": "tab:green", "Reduce Feed": "tab:orange", "Intervene": "tab:red"}
OBS_MARKERS = {"Normal": ("o", "tab:green"), "Warning": ("^", "tab:orange"), "Alarm": ("X", "tab:red")}

# Action that is best if the state were known with certainty (argmax of each utility column)
ACTION_FOR_STATE = {s: ACTIONS[int(np.argmax(UTILITY[:, i]))] for i, s in enumerate(STATES)}


def fmt(v):
    return "[" + ", ".join(f"{x:.4f}" for x in v) + "]"


# ---------------------------------------------------------------- Task 1
def task1():
    print("=" * 70)
    print("TASK 1 - Bayesian inference")
    print("=" * 70)
    cases = [("S1=" + o, {"S1": o}) for o in OBS_S1]
    cases.append(("S1=Warning, S2=Abnormal pH", {"S1": "Warning", "S2": "Abnormal pH"}))

    rows = []
    for name, evidence in cases:
        unnorm, z, post = bayes_posterior(PRIOR, evidence)
        print(f"\n{name}")
        print(f"  unnormalised  = {fmt(unnorm)}")
        print(f"  P(evidence)   = {z:.5f}")
        print(f"  posterior     = {fmt(post)}  -> most probable: {STATES[int(np.argmax(post))]}")
        row = {"evidence": name, "P(evidence)": z}
        row.update({f"unnorm_{s}": u for s, u in zip(STATES, unnorm)})
        row.update({f"P({s})": p for s, p in zip(STATES, post)})
        row["most_probable"] = STATES[int(np.argmax(post))]
        rows.append(row)

    # Manual check for the worked example in the report (S1 = Warning)
    manual_unnorm = np.array([0.70 * 0.15, 0.20 * 0.60, 0.10 * 0.25])  # 0.105, 0.120, 0.025
    manual_post = manual_unnorm / manual_unnorm.sum()                 # / 0.25
    _, _, code_post = bayes_posterior(PRIOR, {"S1": "Warning"})
    assert np.allclose(manual_post, code_post), "Manual calculation disagrees with implementation"
    print(f"\nManual check (S1=Warning): {fmt(manual_post)} matches implementation: "
          f"{np.allclose(manual_post, code_post)}")

    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "task1_bayes_posteriors.csv", index=False, float_format="%.6f")

    # Bar chart comparing posteriors
    fig, ax = plt.subplots(figsize=(9, 4.5))
    labels = ["Prior"] + [r["evidence"].replace("S1=", "").replace(", S2=", " +\n") for r in rows]
    data = np.vstack([PRIOR] + [[r[f"P({s})"] for s in STATES] for r in rows])
    x = np.arange(len(labels))
    width = 0.26
    for i, s in enumerate(STATES):
        bars = ax.bar(x + (i - 1) * width, data[:, i], width, label=s, color=STATE_COLOURS[s])
        ax.bar_label(bars, fmt="%.2f", fontsize=7)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1.05)
    ax.set_title("Task 1: Prior vs posterior belief over digester state")
    ax.legend(title="State")
    fig.tight_layout()
    fig.savefig(FIGURES / "task1_posteriors.png", dpi=200)
    plt.close(fig)
    return df


# ---------------------------------------------------------------- Task 2
def plot_beliefs(df, title, path):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for s in STATES:
        ax.plot(df["time"], df[f"P({s})"], marker=".", color=STATE_COLOURS[s],
                label=f"P({s} | e$_{{1:t}}$)")
    for obs, (m, c) in OBS_MARKERS.items():
        sel = df["observation"] == obs
        ax.scatter(df.loc[sel, "time"], np.full(sel.sum(), -0.08), marker=m, color=c,
                   s=40, label=f"Observed: {obs}", clip_on=False)
    ax.set_xlim(0.5, len(df) + 0.5)
    ax.set_ylim(-0.12, 1.02)
    ax.set_xticks(df["time"])
    ax.set_xlabel("Time step t")
    ax.set_ylabel("Filtered probability")
    ax.set_title(f"Task 2: HMM filtered belief - {title}")
    ax.grid(alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def task2():
    print("\n" + "=" * 70)
    print("TASK 2 - HMM filtering")
    print("=" * 70)
    results = {}
    for key, (title, fname) in SCENARIOS.items():
        obs = pd.read_csv(DATA / fname)
        steps = hmm_filter(PRIOR, TRANSITION, SENSOR1, obs["observation"].tolist())

        rows = []
        for st in steps:
            row = {"time": st["t"], "observation": st["observation"]}
            row.update({f"pred_{s}": v for s, v in zip(STATES, st["predicted"])})
            row.update({f"unnorm_{s}": v for s, v in zip(STATES, st["unnormalised"])})
            row.update({f"P({s})": v for s, v in zip(STATES, st["belief"])})
            row["most_probable"] = STATES[int(np.argmax(st["belief"]))]
            rows.append(row)
        df = pd.DataFrame(rows)
        assert np.allclose(df[[f"P({s})" for s in STATES]].sum(axis=1), 1.0)

        df.to_csv(TABLES / f"task2_hmm_scenario_{key}.csv", index=False, float_format="%.6f")
        plot_beliefs(df, title, FIGURES / f"task2_hmm_scenario_{key}.png")
        results[key] = df

        print(f"\n{title}")
        print(df[["time", "observation"] + [f"P({s})" for s in STATES] + ["most_probable"]]
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return results


# ---------------------------------------------------------------- Task 3
def plot_decisions(df, title, path):
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True,
                                  gridspec_kw={"height_ratios": [4, 1]})
    for a in ACTIONS:
        ax.plot(df["time"], df[f"EU({a})"], marker=".", color=ACTION_COLOURS[a], label=f"EU({a})")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_ylabel("Expected utility")
    ax.set_title(f"Task 3: Expected utility of actions - {title}")
    ax.grid(alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)

    for t, a, o in zip(df["time"], df["recommended_action"], df["observation"]):
        ax2.bar(t, 1, width=1, color=ACTION_COLOURS[a], edgecolor="white")
        ax2.text(t, 0.5, o[0], ha="center", va="center", fontsize=7, color="white")
    ax2.set_yticks([])
    ax2.set_ylabel("Action", rotation=0, ha="right", va="center")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ACTION_COLOURS.values()]
    ax2.legend(handles, ACTION_COLOURS.keys(), loc="center left", bbox_to_anchor=(1.01, 0.5),
               fontsize=8, title="Recommended", title_fontsize=8)
    ax2.set_xlabel("Time step t   (letters = observation: N/W/A)")
    ax2.set_xticks(df["time"])
    ax2.set_xlim(0.5, len(df) + 0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def task3(hmm_results):
    print("\n" + "=" * 70)
    print("TASK 3 - Expected-utility decisions")
    print("=" * 70)
    summary = []
    for key, df in hmm_results.items():
        title = SCENARIOS[key][0]
        rows = []
        for _, r in df.iterrows():
            belief = r[[f"P({s})" for s in STATES]].to_numpy(dtype=float)
            action, eu = recommend_action(belief, UTILITY)
            row = {"time": r["time"], "observation": r["observation"]}
            row.update({f"P({s})": b for s, b in zip(STATES, belief)})
            row.update({f"EU({a})": v for a, v in zip(ACTIONS, eu)})
            row["recommended_action"] = action
            row["most_probable_state"] = r["most_probable"]
            row["action_for_most_probable_state"] = ACTION_FOR_STATE[r["most_probable"]]
            row["differs"] = action != row["action_for_most_probable_state"]
            rows.append(row)
        out = pd.DataFrame(rows)
        out["action_changed"] = out["recommended_action"].ne(out["recommended_action"].shift())
        out.loc[0, "action_changed"] = False
        out.to_csv(TABLES / f"task3_decisions_scenario_{key}.csv", index=False, float_format="%.6f")
        plot_decisions(out, title, FIGURES / f"task3_decisions_scenario_{key}.png")

        print(f"\n{title}")
        print(out[["time", "observation"] + [f"EU({a})" for a in ACTIONS]
                  + ["recommended_action", "most_probable_state", "differs"]]
              .to_string(index=False, float_format=lambda v: f"{v:.2f}"))
        switches = out[out["action_changed"]]
        print("  Action switches:", ", ".join(
            f"t={t}->{a}" for t, a in zip(switches["time"], switches["recommended_action"])) or "none")

        counts = out["recommended_action"].value_counts()
        summary.append({"scenario": key, **{a: int(counts.get(a, 0)) for a in ACTIONS},
                        "n_switches": int(out["action_changed"].sum()),
                        "n_differs_from_most_probable": int(out["differs"].sum())})
    pd.DataFrame(summary).to_csv(TABLES / "task3_summary.csv", index=False)
    print("\nSummary:\n" + pd.DataFrame(summary).to_string(index=False))


def main():
    for d in (TABLES, FIGURES, report_figures.REPORT_FIGURES):
        d.mkdir(parents=True, exist_ok=True)
    task1()
    hmm_results = task2()
    task3(hmm_results)
    report_figures.make_all()
    print(f"\nAll outputs written to {RESULTS}")


if __name__ == "__main__":
    main()
