"""Corpus overview visualizations: year, venue, and CORE rank distributions."""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

CORPUS_PATH = Path("data/processed/corpus.csv")
FIGURES_DIR = Path("results/figures")
TOP_N_VENUES = 15
PALETTE = {"A": "#4C72B0", "A*": "#DD8452"}

VENUE_LABELS: dict[str, str] = {
    "access": "IEEE Access",
    "software engineering": "ICSE",
    "journal of systems and software": "JSS",
    "empirical software engineering": "EMSE",
    "software maintenance and evolution": "ICSME",
    "mining software repositories": "MSR",
    "transactions on software engineering": "TSE",
    "empirical software engineering and measurement": "ESEM",
    "information and software technology": "IST",
    "transactions on software engineering and methodology": "TOSEM",
    "evaluation and assessment in software engineering": "EASE",
    "software analysis evolution and reengineering": "SANER",
    "program comprehension": "ICPC",
    "hawaii international conference on system sciences": "HICSS",
    "european journal of operational research": "EJOR",
}


def load_corpus() -> pd.DataFrame:
    df = pd.read_csv(CORPUS_PATH)
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    return df


def plot_year_distribution(df: pd.DataFrame, ax: plt.Axes) -> None:
    pivot = df.groupby(["year", "core_rank"]).size().unstack(fill_value=0)
    for rank in ("A", "A*"):
        if rank not in pivot.columns:
            pivot[rank] = 0
    pivot = pivot[["A", "A*"]]

    pivot.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=[PALETTE["A"], PALETTE["A*"]],
        edgecolor="white",
        legend=False,
    )

    totals = pivot.sum(axis=1)
    for bar_group, total in zip(ax.containers[-1], totals):
        ax.text(
            bar_group.get_x() + bar_group.get_width() / 2,
            bar_group.get_y() + bar_group.get_height() + 0.5,
            str(int(total)),
            ha="center",
            va="bottom",
            fontsize=7,
        )

    ax.set_title("Papers per Year", fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Paper Count")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(
        handles=[mpatches.Patch(color=PALETTE[r], label=r) for r in ("A", "A*")],
        loc="upper left",
        fontsize=8,
    )


def plot_venue_distribution(df: pd.DataFrame, ax: plt.Axes) -> None:
    venue_counts = df["normalised_venue"].value_counts().head(TOP_N_VENUES)
    dominant_rank = {
        v: df[df["normalised_venue"] == v]["core_rank"].value_counts().idxmax()
        for v in venue_counts.index
    }
    colors = [PALETTE[dominant_rank[v]] for v in venue_counts.index]

    labels = [VENUE_LABELS.get(v, v) for v in venue_counts.index]
    venues_rev = labels[::-1]
    counts_rev = venue_counts.values[::-1].tolist()
    colors_rev = colors[::-1]

    bars = ax.barh(venues_rev, counts_rev, color=colors_rev, edgecolor="white")
    for bar, count in zip(bars, counts_rev):
        ax.text(
            bar.get_width() + 0.3,
            bar.get_y() + bar.get_height() / 2,
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
    rank_counts = df["core_rank"].value_counts()
    ax.pie(
        rank_counts,
        labels=rank_counts.index,
        colors=[PALETTE[r] for r in rank_counts.index],
        autopct="%1.0f%%",
        startangle=90,
        wedgeprops={"width": 0.5},
        textprops={"fontsize": 10},
    )
    ax.set_title("CORE Rank Split", fontweight="bold")


def main() -> None:
    sns.set_theme(style="whitegrid")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_corpus()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Corpus Overview  (n={len(df)} papers)", fontsize=14, fontweight="bold")

    plot_year_distribution(df, axes[0])
    plot_venue_distribution(df, axes[1])
    plot_core_distribution(df, axes[2])

    plt.tight_layout()
    out_path = FIGURES_DIR / "corpus_overview.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
