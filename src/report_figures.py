"""Combined figures for the report, built from the CSVs written by main.py."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

from .biogas_model import ACTIONS, PRIOR, STATES, UTILITY, bayes_posterior

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
REPORT_FIGURES = RESULTS / "report_figures"
SCENARIO_TITLES = {
    "A": "Scenario A - Stable operation",
    "B": "Scenario B - Deteriorating operation",
    "C": "Scenario C - Noisy monitoring",
}
STATE_COLOURS = {"Stable": "tab:green", "Stressed": "tab:orange", "Critical": "tab:red"}
ACTION_COLOURS = {"Continue": "tab:green", "Reduce Feed": "tab:orange", "Intervene": "tab:red"}
OBS_MARKERS = {"Normal": ("o", "tab:green"), "Warning": ("^", "tab:orange"), "Alarm": ("X", "tab:red")}

# Simplex corners for (Stable, Stressed, Critical)
CORNERS = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])


def to_xy(beliefs):
    return np.asarray(beliefs) @ CORNERS


def load(task, key):
    name = "task2_hmm" if task == 2 else "task3_decisions"
    return pd.read_csv(TABLES / f"{name}_scenario_{key}.csv")


def fig_task1_evidence():
    """Prior -> Sensor 1 Warning -> Warning + Abnormal pH."""
    stages = [
        ("Prior P(X$_0$)", PRIOR),
        ("After S1 = Warning", bayes_posterior(PRIOR, {"S1": "Warning"})[2]),
        ("After S1 = Warning\n+ S2 = Abnormal pH",
         bayes_posterior(PRIOR, {"S1": "Warning", "S2": "Abnormal pH"})[2]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True)
    for ax, (title, p) in zip(axes, stages):
        bars = ax.bar(STATES, p, color=[STATE_COLOURS[s] for s in STATES])
        ax.bar_label(bars, fmt="%.3f", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Probability")
    fig.suptitle("Task 1: Effect of combining Sensor 1 and pH evidence", fontsize=12)
    fig.tight_layout()
    fig.savefig(REPORT_FIGURES / "fig_task1_evidence_combination.png", dpi=200)
    plt.close(fig)


def fig_task2_comparison():
    """All three scenarios' filtered beliefs stacked for comparison."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for ax, key in zip(axes, SCENARIO_TITLES):
        df = load(2, key)
        for s in STATES:
            ax.plot(df["time"], df[f"P({s})"], marker=".", color=STATE_COLOURS[s],
                    label=f"P({s} | e$_{{1:t}}$)")
        for obs, (m, c) in OBS_MARKERS.items():
            sel = df["observation"] == obs
            ax.scatter(df.loc[sel, "time"], np.full(sel.sum(), -0.1), marker=m, color=c,
                       s=30, label=f"Observed: {obs}", clip_on=False)
        ax.set_ylim(-0.15, 1.02)
        ax.set_ylabel("Probability")
        ax.set_title(SCENARIO_TITLES[key], fontsize=10)
        ax.grid(alpha=0.3)
    axes[-1].set_xticks(range(1, 31))
    axes[-1].set_xlim(0.5, 30.5)
    axes[-1].set_xlabel("Time step t")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=8, bbox_to_anchor=(0.5, 0))
    fig.suptitle("Task 2: HMM filtered belief P(X$_t$ | e$_{1:t}$) across scenarios", fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(REPORT_FIGURES / "fig_task2_scenario_comparison.png", dpi=200)
    plt.close(fig)


def fig_task3_action_timeline():
    """Per scenario: EU-optimal action vs action implied by the most probable state."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 4.8), sharex=True)
    for ax, key in zip(axes, SCENARIO_TITLES):
        df = load(3, key)
        rows = [("Max EU action", df["recommended_action"]),
                ("Most probable\nstate action", df["action_for_most_probable_state"])]
        for y, (_, actions) in enumerate(rows):
            for t, a in zip(df["time"], actions):
                ax.bar(t, 0.9, bottom=1 - y, width=1, color=ACTION_COLOURS[a], edgecolor="white")
        for t in df.loc[df["differs"], "time"]:
            ax.add_patch(plt.Rectangle((t - 0.5, 0), 1, 1.9, fill=False, ec="black", lw=1.5))
        for t, o in zip(df["time"], df["observation"]):
            ax.text(t, 1.45, o[0], ha="center", va="center", fontsize=7, color="white")
        ax.set_yticks([1.45, 0.45], [r[0] for r in rows], fontsize=8)
        ax.set_ylim(0, 1.95)
        ax.set_title(SCENARIO_TITLES[key], fontsize=10)
    axes[-1].set_xticks(range(1, 31))
    axes[-1].set_xlim(0.5, 30.5)
    axes[-1].set_xlabel("Time step t   (letters = observation N/W/A; black boxes = actions differ)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ACTION_COLOURS.values()]
    fig.legend(handles, ACTION_COLOURS.keys(), loc="upper right", ncol=3, fontsize=8)
    fig.suptitle("Task 3: Recommended action over time", fontsize=12, x=0.3)
    fig.tight_layout()
    fig.savefig(REPORT_FIGURES / "fig_task3_action_timeline.png", dpi=200)
    plt.close(fig)


def fig_task3_decision_regions():
    """Belief simplex coloured by the max-EU action, with most-probable-state boundaries."""
    n = 700
    xs, ys = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, np.sqrt(3) / 2, n))
    # Barycentric coordinates of each grid point
    p_crit = ys / (np.sqrt(3) / 2)
    p_stress = xs - 0.5 * p_crit
    p_stable = 1 - p_stress - p_crit
    bary = np.stack([p_stable, p_stress, p_crit], axis=-1)
    outside = (bary < -1e-9).any(axis=-1)

    best_action = np.argmax(bary @ UTILITY.T, axis=-1).astype(float)
    best_state = np.argmax(bary, axis=-1).astype(float)
    best_action[outside] = np.nan
    best_state[outside] = np.nan

    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = ListedColormap([ACTION_COLOURS[a] for a in ACTIONS])
    ax.contourf(xs, ys, best_action, levels=[-0.5, 0.5, 1.5, 2.5], cmap=cmap, alpha=0.25)
    ax.contour(xs, ys, best_action, levels=[0.5, 1.5], colors="black", linewidths=1.2)
    ax.contour(xs, ys, best_state, levels=[0.5, 1.5], colors="grey", linewidths=1.2,
               linestyles="dashed")

    markers = {"A": "o", "B": "s", "C": "^"}
    for key in SCENARIO_TITLES:
        df = load(2, key)
        pts = to_xy(df[[f"P({s})" for s in STATES]].to_numpy())
        ax.scatter(pts[:, 0], pts[:, 1], marker=markers[key], s=28, facecolor="white",
                   edgecolor="black", lw=0.8, label=f"HMM beliefs, scenario {key}", zorder=3)

    tri = np.vstack([CORNERS, CORNERS[:1]])
    ax.plot(tri[:, 0], tri[:, 1], color="black", lw=1)
    for (x, y), s, (dx, dy) in zip(CORNERS, STATES, [(-0.03, -0.04), (0.03, -0.04), (0, 0.03)]):
        ax.text(x + dx, y + dy, f"{s} = 1", ha="center", fontsize=10, weight="bold")

    handles = [plt.Rectangle((0, 0), 1, 1, color=ACTION_COLOURS[a], alpha=0.35) for a in ACTIONS]
    handles += [plt.Line2D([], [], color="black", lw=1.2),
                plt.Line2D([], [], color="grey", lw=1.2, ls="--")]
    labels = [f"Max EU: {a}" for a in ACTIONS] + ["Max-EU action boundary",
                                                  "Most-probable-state boundary"]
    h2, l2 = ax.get_legend_handles_labels()
    ax.legend(handles + h2, labels + l2, loc="upper left", fontsize=8)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Task 3: Optimal action regions over the belief simplex", fontsize=12, pad=30)
    fig.tight_layout()
    fig.savefig(REPORT_FIGURES / "fig_task3_decision_regions.png", dpi=200)
    plt.close(fig)


def fig_sensitivity():
    """How far the recommended actions move when one modelling assumption is changed."""
    df = pd.read_csv(TABLES / "sensitivity_analysis.csv")
    analyses = [("utility", "U(Continue, Critical)", -20),
                ("sensor_noise", "Sensor 1 noise $\\varepsilon$", 0.0),
                ("transition", "Transition persistence $\\lambda$", 0.0)]
    metrics = [("n_differs_from_baseline", "Steps with a different\naction from baseline"),
               ("n_switches", "Number of action\nswitches over 30 steps")]
    markers = {"A": "o", "B": "s", "C": "^"}
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.6), sharey="row")
    for col, (analysis, xlabel, baseline) in enumerate(analyses):
        part = df[df["analysis"] == analysis]
        for row, (metric, ylabel) in enumerate(metrics):
            ax = axes[row, col]
            for key in SCENARIO_TITLES:
                sel = part[part["scenario"] == key]
                ax.plot(sel["value"], sel[metric], marker=markers[key], label=f"Scenario {key}")
            ax.axvline(baseline, color="grey", ls="--", lw=1)
            ax.grid(alpha=0.3)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=9)
            if row == 1:
                ax.set_xlabel(xlabel)
        axes[0, col].set_title(["Utility model", "Sensor 1 reliability", "Transition model"][col],
                               fontsize=10)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color="grey", ls="--"))
    labels.append("Baseline model")
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, 0))
    fig.suptitle("Sensitivity of recommended actions to modelling assumptions", fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(REPORT_FIGURES / "fig_sensitivity.png", dpi=200)
    plt.close(fig)


def make_all():
    fig_task1_evidence()
    fig_task2_comparison()
    fig_task3_action_timeline()
    fig_task3_decision_regions()
    fig_sensitivity()


if __name__ == "__main__":
    REPORT_FIGURES.mkdir(parents=True, exist_ok=True)
    make_all()
