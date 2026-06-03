"""Evaluation result visualizations: classifier accuracy + hybrid confusion.

Reads the gold evaluation produced by src/evaluate/evaluate_hybrid.py
(data/processed/gold_hybrid_eval.csv) and writes two figures to
results/figures/:

    accuracy_comparison.png   -- overall + per-class accuracy for the rule,
                                 zero-shot, and hybrid classifiers
    hybrid_confusion_matrix.png -- gold vs hybrid-predicted label heatmap

Run from the project root: `python src/viz/results_overview.py`.
"""

import matplotlib

matplotlib.use("Agg")  # headless backend: save files without opening a window

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path

# Relative paths work because this script is run from the project root.
EVAL_PATH   = Path("data/processed/gold_hybrid_eval.csv")
FIGURES_DIR = Path("results/figures")

# One consistent colour per classifier, reused across every panel.
PALETTE = {"Rule": "#55A868", "Zero-shot": "#4C72B0", "Hybrid": "#DD8452"}

# The 8 canonical labels the gold set uses, abbreviated for compact axes.
LABEL_SHORT: dict[str, str] = {
    "Empirical Study":               "Empirical",
    "Tool Paper":                    "Tool",
    "Systematic Literature Review":  "SLR",
    "Case Study":                    "Case Study",
    "Survey":                        "Survey",
    "Experience Report":             "Exp. Report",
    "Theoretical Contribution":      "Theoretical",
    "Position Paper":                "Position",
    "Controlled Experiment":         "Controlled Exp.",
}


def load_eval() -> pd.DataFrame:
    """Load the gold hybrid evaluation produced by evaluate_hybrid.py."""
    if not EVAL_PATH.exists():
        raise SystemExit(
            f"{EVAL_PATH} not found. Run `python src/evaluate/evaluate_hybrid.py` first."
        )
    return pd.read_csv(EVAL_PATH)


def _per_class(df: pd.DataFrame) -> pd.DataFrame:
    """Per-class accuracy (%) for each classifier, sorted by class size desc."""
    grp = df.groupby("gold_label")
    out = pd.DataFrame(
        {
            "count":     grp.size(),
            "Rule":      grp["rule_correct"].mean() * 100,
            "Zero-shot": grp["zs_correct"].mean() * 100,
            "Hybrid":    grp["hybrid_correct"].mean() * 100,
        }
    ).sort_values("count", ascending=False)
    return out


def plot_overall(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Three bars: overall accuracy of each classifier across all gold papers."""
    n = len(df)
    accs = {
        "Rule":      df["rule_correct"].mean() * 100,
        "Zero-shot": df["zs_correct"].mean() * 100,
        "Hybrid":    df["hybrid_correct"].mean() * 100,
    }
    bars = ax.bar(list(accs), list(accs.values()),
                  color=[PALETTE[k] for k in accs])
    for bar, v in zip(bars, accs.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1, f"{v:.0f}%",
                ha="center", va="bottom", fontweight="bold")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Overall Accuracy  (n={n})", fontweight="bold")


def plot_per_class(pc: pd.DataFrame, ax: plt.Axes) -> None:
    """Grouped bars: rule vs zero-shot vs hybrid accuracy for each gold class."""
    labels = [f"{LABEL_SHORT.get(c, c)}\n(n={int(pc.loc[c, 'count'])})" for c in pc.index]
    x = range(len(pc))
    w = 0.27
    for i, clf in enumerate(["Rule", "Zero-shot", "Hybrid"]):
        ax.bar([p + (i - 1) * w for p in x], pc[clf], width=w,
               label=clf, color=PALETTE[clf])
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Per-Class Accuracy (rows = gold label)", fontweight="bold")
    ax.legend(title="Classifier")


def plot_confusion(df: pd.DataFrame) -> Path:
    """Heatmap of gold label vs hybrid prediction; save and return the path."""
    labels = [c for c in LABEL_SHORT if c in set(df["gold_label"]) | set(df["hybrid_predicted"])]
    cm = pd.crosstab(df["gold_label"], df["hybrid_predicted"]).reindex(
        index=labels, columns=labels, fill_value=0
    )
    short = [LABEL_SHORT.get(c, c) for c in labels]

    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=short, yticklabels=short,
                linewidths=0.5, linecolor="white", ax=ax)
    ax.set_xlabel("Hybrid predicted")
    ax.set_ylabel("Gold label")
    acc = df["hybrid_correct"].mean() * 100
    ax.set_title(f"Hybrid Confusion Matrix  (accuracy {acc:.0f}%, n={len(df)})",
                 fontweight="bold")
    plt.tight_layout()
    out = FIGURES_DIR / "hybrid_confusion_matrix.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    sns.set_theme(style="whitegrid")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_eval()
    pc = _per_class(df)

    # Figure 1: overall (narrow) + per-class (wide) side by side.
    fig, axes = plt.subplots(1, 2, figsize=(17, 6),
                             gridspec_kw={"width_ratios": [1, 3]})
    fig.suptitle("Paper-Type Classifier Accuracy on the Gold Set",
                 fontsize=14, fontweight="bold")
    plot_overall(df, axes[0])
    plot_per_class(pc, axes[1])
    plt.tight_layout()
    out1 = FIGURES_DIR / "accuracy_comparison.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out1}")

    # Figure 2: hybrid confusion matrix.
    out2 = plot_confusion(df)
    print(f"Saved: {out2}")


if __name__ == "__main__":
    main()
