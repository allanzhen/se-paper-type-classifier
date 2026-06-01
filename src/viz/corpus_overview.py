"""Corpus overview visualizations: year, venue, and CORE rank distributions."""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

# Relative paths work here because this script is run from the project root.
CORPUS_PATH  = Path("data/processed/corpus.csv")
FIGURES_DIR  = Path("results/figures")
TOP_N_VENUES = 15  # how many venues to show in the bar chart
PALETTE      = {"A": "#4C72B0", "A*": "#DD8452"}  # consistent colors across all three plots

# Maps the normalised venue key (from clean_corpus.py) to a short display
# label used on chart axes. Without this, axis labels show the full lowercase
# normalised string which is often too long to read on a chart.
VENUE_LABELS: dict[str, str] = {
    "access":                                        "IEEE Access",
    "software engineering":                          "ICSE",
    "journal of systems and software":               "JSS",
    "empirical software engineering":                "EMSE",
    "software maintenance and evolution":            "ICSME",
    "mining software repositories":                  "MSR",
    "transactions on software engineering":          "TSE",
    "empirical software engineering and measurement":"ESEM",
    "information and software technology":           "IST",
    "transactions on software engineering and methodology": "TOSEM",
    "evaluation and assessment in software engineering": "EASE",
    "software analysis evolution and reengineering": "SANER",
    "program comprehension":                         "ICPC",
    "hawaii international conference on system sciences": "HICSS",
    "european journal of operational research":      "EJOR",
}


def load_corpus() -> pd.DataFrame:
    """Load the processed corpus and coerce year to int.

    Drops rows with a missing year so downstream groupby operations don't
    produce a spurious NaN bucket on the year axis.
    """
    df = pd.read_csv(CORPUS_PATH)
    df = df.dropna(subset=["year"])
    # Year arrives as float from pandas (due to possible NaN before dropna);
    # cast to int so bar chart x-labels show "2015" not "2015.0".
    df["year"] = df["year"].astype(int)
    return df


def plot_year_distribution(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Draw a stacked bar chart of paper counts per year, coloured by CORE rank.

    Stacking A and A* bars lets readers see both the total volume per year
    and the rank breakdown in a single glance. Count labels above each bar
    prevent misreading bar heights on a dense x-axis.
    """
    # Build a (year x core_rank) pivot table so each column becomes one
    # stacked bar segment.
    pivot = df.groupby(["year", "core_rank"]).size().unstack(fill_value=0)

    # Guarantee both columns exist even if the corpus only contains one rank —
    # missing columns would cause a KeyError in the column reorder below.
    for rank in ("A", "A*"):
        if rank not in pivot.columns:
            pivot[rank] = 0
    pivot = pivot[["A", "A*"]]  # fix column order so A is always the bottom segment

    pivot.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=[PALETTE["A"], PALETTE["A*"]],
        edgecolor="white",   # thin white gap between bars aids readability
        legend=False,        # we draw a custom legend below for consistent styling
    )

    # Annotate each bar group with the total count above the top segment.
    # ax.containers[-1] holds the top stacked segment, so its bars give the
    # correct y position (top of the full stack) for the label.
    totals = pivot.sum(axis=1)
    for bar_group, total in zip(ax.containers[-1], totals):
        ax.text(
            bar_group.get_x() + bar_group.get_width() / 2,  # horizontal center of bar
            bar_group.get_y() + bar_group.get_height() + 0.5,  # just above the bar top
            str(int(total)),
            ha="center",
            va="bottom",
            fontsize=7,
        )

    ax.set_title("Papers per Year", fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Paper Count")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    # Manual legend using mpatches so colour and label match the PALETTE dict
    # rather than whatever matplotlib auto-generates.
    ax.legend(
        handles=[mpatches.Patch(color=PALETTE[r], label=r) for r in ("A", "A*")],
        loc="upper left",
        fontsize=8,
    )


def plot_venue_distribution(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Draw a horizontal bar chart of paper counts for the top N venues.

    Each bar is coloured by the venue's dominant CORE rank (A or A*).
    The chart is drawn bottom-to-top (highest count at the top) by reversing
    the sorted lists before plotting, which matches the natural reading direction
    for horizontal bar charts.
    """
    # Count papers per normalised venue and keep the top N.
    venue_counts = df["normalised_venue"].value_counts().head(TOP_N_VENUES)

    # Assign each venue the rank that appears most often for that venue —
    # a venue could theoretically span both ranks if it changed classification,
    # but idxmax() picks the dominant one for colouring.
    dominant_rank = {
        v: df[df["normalised_venue"] == v]["core_rank"].value_counts().idxmax()
        for v in venue_counts.index
    }
    colors = [PALETTE[dominant_rank[v]] for v in venue_counts.index]

    # Map normalised keys to short display labels; fall back to the raw key
    # for any venue not listed in VENUE_LABELS.
    labels = [VENUE_LABELS.get(v, v) for v in venue_counts.index]

    # Reverse all three lists so the highest-count venue appears at the top
    # of the horizontal bar chart (barh plots bottom-to-top by default).
    venues_rev = labels[::-1]
    counts_rev = venue_counts.values[::-1].tolist()
    colors_rev = colors[::-1]

    bars = ax.barh(venues_rev, counts_rev, color=colors_rev, edgecolor="white")

    # Annotate each bar with its exact count just to the right of the bar end.
    for bar, count in zip(bars, counts_rev):
        ax.text(
            bar.get_width() + 0.3,           # small offset to the right of the bar
            bar.get_y() + bar.get_height() / 2,  # vertical center of the bar
            str(count),
            va="center",
            fontsize=7,
        )

    ax.set_title(f"Top {TOP_N_VENUES} Venues", fontweight="bold")
    ax.set_xlabel("Paper Count")
    ax.legend(
        handles=[mpatches.Patch(color=PALETTE[r], label=r) for r in ("A", "A*")],
        loc="lower right",
        fontsize=8,
    )


def plot_core_distribution(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Draw a donut chart showing the A vs A* split across the whole corpus.

    A donut (wedgeprops width < 1) is used instead of a full pie so the
    chart reads as a part-of-whole comparison rather than an area estimate,
    which is perceptually more accurate for two-category data.
    """
    rank_counts = df["core_rank"].value_counts()
    ax.pie(
        rank_counts,
        labels=rank_counts.index,
        colors=[PALETTE[r] for r in rank_counts.index],
        autopct="%1.0f%%",    # show integer percentage inside each wedge
        startangle=90,         # start at 12 o'clock so the larger slice is on top
        wedgeprops={"width": 0.5},   # width < 1 creates the donut hole
        textprops={"fontsize": 10},
    )
    ax.set_title("CORE Rank Split", fontweight="bold")


def main() -> None:
    # Apply a clean seaborn theme so the three subplots share consistent grid
    # and font styling without per-plot configuration.
    sns.set_theme(style="whitegrid")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_corpus()

    # Three side-by-side panels: year trend, top venues, rank split.
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Corpus Overview  (n={len(df)} papers)", fontsize=14, fontweight="bold")

    plot_year_distribution(df, axes[0])
    plot_venue_distribution(df, axes[1])
    plot_core_distribution(df, axes[2])

    # tight_layout adjusts subplot spacing so titles and tick labels don't
    # overlap — call it after all plots are drawn, not before.
    plt.tight_layout()

    out_path = FIGURES_DIR / "corpus_overview.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")  # bbox_inches="tight" prevents label clipping
    print(f"Saved: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
