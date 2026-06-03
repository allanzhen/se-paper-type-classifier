"""Corpus overview visualizations: papers per year and top venues.

The corpus (data/processed/corpus.csv) is collected from DBLP and already
filtered to A/A* venues, so it has columns [title, year, venue, doi, url,
abstract, paper_id] -- there is no per-paper CORE-rank column to break down by.
This figure therefore reports the two dimensions that are present: publication
year (field growth) and venue concentration.

Writes results/figures/corpus_overview.png. Run from the project root:
`python src/viz/corpus_overview.py`.
"""

import matplotlib

matplotlib.use("Agg")  # headless backend: save files without opening a window

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path

# Relative paths work here because this script is run from the project root.
CORPUS_PATH  = Path("data/processed/corpus.csv")
FIGURES_DIR  = Path("results/figures")
TOP_N_VENUES = 15        # how many venues to show in the bar chart
BAR_COLOR    = "#4C72B0"  # single consistent colour for both panels
MAX_LABEL    = 32        # truncate long venue strings on the y-axis


def load_corpus() -> pd.DataFrame:
    """Load the corpus and coerce year to int.

    Drops rows with a missing year so the per-year groupby doesn't produce a
    spurious NaN bucket on the x-axis.
    """
    df = pd.read_csv(CORPUS_PATH)
    df = df.dropna(subset=["year"])
    # Year arrives as float from pandas (possible NaN before dropna); cast to int
    # so x-labels show "2015" not "2015.0".
    df["year"] = df["year"].astype(int)
    return df


def plot_year_distribution(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Bar chart of paper counts per year, with the count above each bar."""
    counts = df["year"].value_counts().sort_index()
    bars = ax.bar(counts.index, counts.values, color=BAR_COLOR, edgecolor="white")
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5, str(int(v)),
                ha="center", va="bottom", fontsize=7)
    ax.set_title("Papers per Year", fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Paper Count")
    ax.set_xticks(counts.index)
    ax.set_xticklabels(counts.index, rotation=45, ha="right")


def plot_venue_distribution(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Pie of the top N venues (+ an 'Other' wedge for the rest of the corpus).

    Including 'Other' keeps the pie a true part-of-whole: every wedge is a share
    of the full corpus (n), not of the top-N subtotal.
    """
    n = len(df)
    top = df["venue"].value_counts().head(TOP_N_VENUES)
    other = n - int(top.sum())

    labels = [
        (v if len(v) <= MAX_LABEL else v[: MAX_LABEL - 1] + "…")
        for v in top.index
    ]
    sizes  = list(top.values)
    colors = list(sns.color_palette("tab20", len(top)))
    if other > 0:                       # long tail of remaining venues
        labels.append(f"Other ({df['venue'].nunique() - len(top)} venues)")
        sizes.append(other)
        colors.append("#BBBBBB")        # neutral grey for the catch-all

    # Percentage only on slices big enough to read (>=4% of the corpus); the
    # small venues are identified by the side legend instead.
    def _autopct(pct: float) -> str:
        return f"{pct:.0f}%" if pct >= 4 else ""

    wedges, _, _ = ax.pie(
        sizes,
        colors=colors,
        autopct=_autopct,
        startangle=90,
        counterclock=False,           # clockwise, largest first from 12 o'clock
        pctdistance=0.8,
        textprops={"fontsize": 8},
        wedgeprops={"linewidth": 0.5, "edgecolor": "white"},
    )
    ax.legend(
        wedges,
        [f"{lab} — {sz}" for lab, sz in zip(labels, sizes)],
        title="Venue",
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),    # sit to the right of the pie
        fontsize=8,
        title_fontsize=9,
    )
    ax.set_title(f"Top {TOP_N_VENUES} Venues", fontweight="bold")


def main() -> None:
    sns.set_theme(style="whitegrid")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_corpus()

    # Two side-by-side panels: year trend and venue concentration.
    fig, axes = plt.subplots(1, 2, figsize=(16, 6),
                             gridspec_kw={"width_ratios": [1.1, 1]})
    fig.suptitle(
        f"Corpus Overview  (n={len(df)} papers, DBLP A/A* venues)",
        fontsize=14, fontweight="bold",
    )

    plot_year_distribution(df, axes[0])
    plot_venue_distribution(df, axes[1])

    plt.tight_layout()
    out_path = FIGURES_DIR / "corpus_overview.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
