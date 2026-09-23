"""Render the feature-level LOFO and permutation-importance Figure 4."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_core import PUBLIC_TEST_COHORT, configure_matplotlib, write_json


LSM_FEATURES = {
    "hist_count",
    "last_lsm",
    "baseline_lsm",
    "hist_mean",
    "hist_median",
    "hist_std",
    "hist_iqr",
    "hist_range",
    "slope_kpa_per_day",
    "history_days",
    "prior_interval_days",
    "mean_prior_interval_days",
    "sd_prior_interval_days",
}
EVALUATION_ORDER = ["temporal_validation_2023", "temporal_test_2024"]
EVALUATION_LABELS = {
    "temporal_validation_2023": "test",
    "temporal_test_2024": "validation",
}
EVALUATION_COLORS = {
    "temporal_validation_2023": "#D17A22",
    "temporal_test_2024": "#376996",
}
METHOD_MARKERS = {
    "temporal_validation_2023": "s",
    "temporal_test_2024": "o",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render the feature-level importance Figure 4.")
    parser.add_argument("--result-dir", default=None)
    parser.add_argument("--article-figure-dir", default=None)
    return parser


def _default_result_dir() -> Path:
    return Path(__file__).resolve().parent / "result"


def _default_article_figure_dir() -> Path:
    return Path(__file__).resolve().parent / "文章" / "图片"


def _add_panel_label(axis, label: str) -> None:
    axis.text(
        -0.12,
        1.05,
        label,
        transform=axis.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
    )


def _plot_forest(
    axis,
    frame: pd.DataFrame,
    value_column: str,
    low_column: str,
    high_column: str,
    order: list[str],
    title: str,
    xlabel: str,
    note: str,
) -> None:
    y = np.arange(len(order), dtype=float)
    offsets = {"temporal_validation_2023": 0.15, "temporal_test_2024": -0.15}
    for evaluation_set in EVALUATION_ORDER:
        subset = frame.loc[frame["evaluation_set"].eq(evaluation_set)].set_index("feature").loc[order]
        values = subset[value_column].to_numpy(dtype=float)
        low = subset[low_column].to_numpy(dtype=float)
        high = subset[high_column].to_numpy(dtype=float)
        axis.errorbar(
            values,
            y + offsets[evaluation_set],
            xerr=[values - low, high - values],
            fmt=METHOD_MARKERS[evaluation_set],
            markersize=4.6,
            color=EVALUATION_COLORS[evaluation_set],
            ecolor=EVALUATION_COLORS[evaluation_set],
            elinewidth=1.1,
            capsize=2.5,
            alpha=0.95,
            label=EVALUATION_LABELS[evaluation_set],
        )
    axis.axvline(0, color="#374151", linewidth=1)
    axis.set_yticks(y, order)
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.grid(axis="x", color="#D1D5DB", linewidth=0.6, alpha=0.65)
    axis.set_axisbelow(True)
    axis.text(
        0.01,
        0.01,
        note,
        transform=axis.transAxes,
        fontsize=8,
        va="bottom",
        color="#374151",
    )
    for tick, feature in zip(axis.get_yticklabels(), order):
        tick.set_color("#376996" if feature in LSM_FEATURES else "#D17A22")


def render_feature_level_figure(result_dir: Path, article_figure_dir: Path) -> tuple[Path, Path]:
    configure_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    article_figure_dir.mkdir(parents=True, exist_ok=True)

    permutation = pd.read_csv(result_dir / "phase11_permutation_importance.csv")
    lofo = pd.read_csv(result_dir / "phase11_feature_level_lofo_mae_delta.csv")
    permutation = permutation.loc[permutation["evaluation_set"].isin(EVALUATION_ORDER)].copy()
    lofo = lofo.loc[lofo["evaluation_set"].isin(EVALUATION_ORDER)].copy()
    if permutation.empty or lofo.empty:
        raise RuntimeError("Feature-level result CSVs contain no validation/test rows.")
    features = permutation.loc[permutation["evaluation_set"].eq(PUBLIC_TEST_COHORT)].copy()
    order = (
        features.sort_values("permutation_mae_delta", ascending=False)["feature"].tolist()
    )
    if len(order) != 21 or len(set(order)) != 21:
        raise ValueError(f"Expected 21 unique features for the figure, found {len(order)}.")

    fig, axes = plt.subplots(1, 2, figsize=(15.0, 10.8), sharey=True, gridspec_kw={"wspace": 0.20})
    _plot_forest(
        axes[0],
        permutation,
        "permutation_mae_delta",
        "ci_2_5",
        "ci_97_5",
        order,
        "Permutation importance of the locked random forest",
        "MAE increase after feature permutation (kPa)",
        "Positive = greater model reliance",
    )
    _plot_forest(
        axes[1],
        lofo,
        "mae_delta_removed_minus_full",
        "ci_2_5",
        "ci_97_5",
        order,
        "Leave-one-feature-out (LOFO) ablation",
        "MAE change after feature removal (kPa)",
        "Positive = removal worsens MAE",
    )
    axes[0].invert_yaxis()
    axes[1].tick_params(axis="y", labelleft=False)
    fig.legend(
        handles=[
            Line2D([0], [0], marker="s", color=EVALUATION_COLORS["temporal_validation_2023"], linestyle="None", markersize=5, label=EVALUATION_LABELS["temporal_validation_2023"]),
            Line2D([0], [0], marker="o", color=EVALUATION_COLORS["temporal_test_2024"], linestyle="None", markersize=5, label=EVALUATION_LABELS["temporal_test_2024"]),
            Patch(facecolor="#376996", edgecolor="none", alpha=0.85, label="Longitudinal LSM feature"),
            Patch(facecolor="#D17A22", edgecolor="none", alpha=0.85, label="Clinical/laboratory feature"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.58, 0.955),
        ncol=4,
        frameon=False,
        fontsize=8,
    )
    _add_panel_label(axes[0], "A")
    _add_panel_label(axes[1], "B")
    fig.suptitle(
        "Feature-level contribution and model reliance in temporal hold-out cohorts",
        fontsize=14,
        y=0.995,
    )
    fig.text(
        0.5,
        0.005,
        "Points show estimates and horizontal bars show patient-bootstrap 95% CIs. "
        "LOFO refits the locked RF after removing one feature; permutation keeps the full model fixed. "
        "Correlated LSM summaries can substitute for one another.",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#374151",
    )
    fig.subplots_adjust(left=0.19, right=0.985, top=0.89, bottom=0.065)
    article_stem = article_figure_dir / "Fig4"
    fig.savefig(article_stem.with_suffix(".png"), dpi=320, bbox_inches="tight")
    fig.savefig(article_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    result_stem = result_dir / "publication_figure4_feature_level_importance"
    shutil.copy2(article_stem.with_suffix(".png"), result_stem.with_suffix(".png"))
    shutil.copy2(article_stem.with_suffix(".pdf"), result_stem.with_suffix(".pdf"))

    metadata_path = result_dir / "phase11_feature_level_importance_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["figure_outputs"] = [
        "文章/图片/Fig4.png",
        "文章/图片/Fig4.pdf",
        "publication_figure4_feature_level_importance.png",
        "publication_figure4_feature_level_importance.pdf",
    ]
    write_json(metadata, metadata_path)
    return article_stem.with_suffix(".png"), article_stem.with_suffix(".pdf")


def main() -> None:
    args = build_parser().parse_args()
    result_dir = Path(args.result_dir) if args.result_dir else _default_result_dir()
    article_figure_dir = (
        Path(args.article_figure_dir) if args.article_figure_dir else _default_article_figure_dir()
    )
    article_png, article_pdf = render_feature_level_figure(result_dir, article_figure_dir)
    print(f"Feature-level Figure 4 written to: {article_png} / {article_pdf}")


if __name__ == "__main__":
    main()
