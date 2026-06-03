"""Corpus paper-type distribution: overall mix + trend over time.

Reports the variety of paper types across the whole technical-debt corpus using
the hybrid classifier's predictions (data/processed/corpus_labeled_hybrid.csv,
produced by src/evaluate/label_corpus_hybrid.py). Writes one figure to
results/figures/corpus_type_distribution.png:

    Panel 1 -- overall type counts (horizontal bar, count + % of corpus)
    Panel 2 -- type counts per year (stacked bar), showing how the field's
               methodology mix grows and diversifies over time

Labels are PREDICTED (hybrid ~80% accurate on the gold set), so the distribution
is an estimate of the field's methodology mix, not exact hand-counted totals.

Run from the project root: `python src/viz/corpus_type_distribution.py`.
"""

import matplotlib

matplotlib.use("Agg")  # headless backend: save files without opening a window

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path

LABELED_PATH = Path("data/processed/corpus_labeled_hybrid.csv")
FIGURES_DIR  = Path("results/figures")

# Fixed colour per paper type, reused across both panels so a type reads the same
# everywhere. Ordered most→least common for a stable legend/stack order.
TYPE_ORDER = [
    "Empirical Study", "Tool Paper", "Case Study", "Theoretical Contribution",
    "Survey", "Position Paper", "Experience Report",
    "Systematic Literature Review", "Controlled Experiment",
]
TYPE_PALETTE = dict(zip(TYPE_ORDER, sns.color_palette("tab10", len(TYPE_ORDER))))

LABEL_SHORT: dict[str, str] = {
    "Empirical Study": "Empirical Study",
    "Tool Paper": "Tool Paper",
    "Case Study": "Case Study",
    "Theoretical Contribution": "Theoretical",
    "Survey": "Survey",
    "Position Paper": "Position Paper",
    "Experience Report": "Experience Report",
    "Systematic Literature Review": "SLR",
    "Controlled Experiment": "Controlled Exp.",
}


def load_labeled() -> pd.DataFrame:
    if not LABELED_PATH.exists():
        raise SystemExit(
            f"{LABELED_PATH} not found. Run "
            "`python src/evaluate/label_corpus_hybrid.py` first."
        )
    df = pd.read_csv(LABELED_PATH)
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    return df


def plot_overall(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Pie chart of paper-type share, slices coloured to match the trend panel."""
    counts = df["hybrid_label"].value_counts()
    counts = counts.reindex([t for t in TYPE_ORDER if t in counts.index])
    n = len(df)
    # Percentage on slices big enough to read (>=4%); the small slices crowd if
    # labelled on the pie, so all types are identified by the side legend instead.
    def _autopct(pct: float) -> str:
        return f"{pct:.0f}%" if pct >= 4 else ""
    wedges, _, _ = ax.pie(
        counts.values,
        colors=[TYPE_PALETTE[t] for t in counts.index],
        autopct=_autopct,
        startangle=90,
        counterclock=False,           # clockwise, largest first from 12 o'clock
        pctdistance=0.78,
        textprops={"fontsize": 9},
        wedgeprops={"linewidth": 0.5, "edgecolor": "white"},
    )
    ax.legend(
        wedges,
        [f"{LABEL_SHORT[t]} — {v} ({100 * v / n:.0f}%)" for t, v in counts.items()],
        title="Paper Type",
        loc="center right",
        bbox_to_anchor=(-0.05, 0.5),  # sit in the empty space to the left of the pie
        fontsize=8,
        title_fontsize=9,
    )
    ax.set_title(f"Overall Paper-Type Mix  (n={n})", fontweight="bold")


def plot_trend(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Stacked bar of paper counts per year, coloured by type."""
    pivot = (
        df.groupby(["year", "hybrid_label"]).size().unstack(fill_value=0)
        .reindex(columns=[t for t in TYPE_ORDER if t in df["hybrid_label"].unique()])
    )
    bottom = [0] * len(pivot)
    for t in pivot.columns:
        ax.bar(pivot.index, pivot[t], bottom=bottom,
               color=TYPE_PALETTE[t], label=LABEL_SHORT[t], width=0.85)
        bottom = [b + v for b, v in zip(bottom, pivot[t])]
    ax.set_xlabel("Year  (2026 partial)")
    ax.set_ylabel("Papers")
    ax.set_title("Paper-Type Mix Over Time", fontweight="bold")
    ax.set_xticks(pivot.index)
    ax.set_xticklabels(pivot.index, rotation=45, ha="right")
    ax.legend(title="Paper Type", bbox_to_anchor=(1.01, 1), loc="upper left",
              fontsize=9, title_fontsize=10)


def main() -> None:
    sns.set_theme(style="whitegrid")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_labeled()

    fig, axes = plt.subplots(1, 2, figsize=(18, 6.5),
                             gridspec_kw={"width_ratios": [1, 1.3]})
    fig.suptitle(
        "Technical-Debt Corpus: Paper-Type Variety "
        f"(hybrid-predicted, n={len(df)}, ~80% gold accuracy)",
        fontsize=14, fontweight="bold",
    )
    plot_overall(df, axes[0])
    plot_trend(df, axes[1])
    plt.tight_layout()

    out = FIGURES_DIR / "corpus_type_distribution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
