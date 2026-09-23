"""Run prespecified subgroup, residual, slope, and exploratory clustering analyses."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, silhouette_score

from analysis_core import (
    PUBLIC_TEST_COHORT,
    RANDOM_SEED,
    configure_matplotlib,
    ensure_result_dir,
    write_csv,
    write_json,
)
from modeling_core import bootstrap_regression_interval


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate subgroups and exploratory LSM trajectories.")
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--cluster-bootstrap-replicates", type=int, default=100)
    return parser


def _default_result_dir() -> Path:
    return Path(__file__).resolve().parent / "result"


def _history_group(count: int) -> str:
    if count == 2:
        return "2"
    if count == 3:
        return "3"
    if count <= 5:
        return "4-5"
    return ">=6"


def _index_lsm_group(value: float) -> str:
    if value < 7:
        return "<7"
    if value < 12.5:
        return "7-<12.5"
    if value < 20:
        return "12.5-<20"
    return ">=20"


def _horizon_group(days: float) -> str:
    if days < 330:
        return "270-<330"
    if days < 390:
        return "330-<390"
    return "390-450"


def subgroup_results(
    prediction: pd.DataFrame,
    tasks: pd.DataFrame,
    bootstrap_replicates: int,
) -> pd.DataFrame:
    frame = prediction.merge(
        tasks[["patient_key", "index_date", "target_date", "last_lsm"]],
        on=["patient_key", "index_date", "target_date"],
        how="left",
        validate="one_to_one",
    )
    frame["history_group"] = frame["hist_count"].map(_history_group)
    frame["index_lsm_group"] = frame["last_lsm"].map(_index_lsm_group)
    frame["horizon_group"] = frame["horizon_days"].map(_horizon_group)

    rows = []
    expected_levels = {
        "history_group": ["2", "3", "4-5", ">=6"],
        "index_lsm_group": ["<7", "7-<12.5", "12.5-<20", ">=20"],
        "horizon_group": ["270-<330", "330-<390", "390-450"],
    }
    for definition, levels in expected_levels.items():
        for level in levels:
            subset = frame.loc[frame[definition].eq(level)]
            if subset.empty:
                rows.append(
                    {
                        "subgroup_definition": definition,
                        "subgroup": level,
                        "patients": 0,
                        "tasks": 0,
                        "mae_estimate": np.nan,
                        "mae_ci_2_5": np.nan,
                        "mae_ci_97_5": np.nan,
                        "rmse_estimate": np.nan,
                        "rmse_ci_2_5": np.nan,
                        "rmse_ci_97_5": np.nan,
                        "r2_estimate": np.nan,
                        "r2_ci_2_5": np.nan,
                        "r2_ci_97_5": np.nan,
                        "r2_interpretation": "not estimable: no patients",
                    }
                )
                continue
            interval = bootstrap_regression_interval(
                subset,
                "locked_ml_model",
                n_bootstrap=bootstrap_replicates,
                seed=RANDOM_SEED + len(rows),
            )
            rows.append(
                {
                    "subgroup_definition": definition,
                    "subgroup": level,
                    "patients": int(subset["patient_key"].nunique()),
                    "tasks": int(len(subset)),
                    **interval,
                    "r2_interpretation": (
                        "descriptive only: fewer than 20 patients"
                        if subset["patient_key"].nunique() < 20
                        else "reported with patient-level bootstrap interval"
                    ),
                }
            )
    return pd.DataFrame(rows)


def residual_review(prediction: pd.DataFrame, tasks: pd.DataFrame) -> pd.DataFrame:
    frame = prediction.merge(
        tasks[
            [
                "patient_key",
                "index_date",
                "target_date",
                "last_lsm",
                "slope_kpa_per_year",
            ]
        ],
        on=["patient_key", "index_date", "target_date"],
        how="left",
        validate="one_to_one",
    )
    frame["residual_kpa"] = frame["locked_ml_model"] - frame["target_lsm"]
    frame["absolute_error_kpa"] = frame["residual_kpa"].abs()
    frame["observed_change_from_index_kpa"] = frame["target_lsm"] - frame["last_lsm"]
    frame["absolute_observed_change_kpa"] = frame["observed_change_from_index_kpa"].abs()
    frame["history_group"] = frame["hist_count"].map(_history_group)
    frame["exploratory_large_change_ge_10_kpa"] = frame["absolute_observed_change_kpa"].ge(10.0)
    return frame.sort_values("absolute_error_kpa", ascending=False).reset_index(drop=True)


def residual_trajectories(residuals: pd.DataFrame, visits: pd.DataFrame) -> pd.DataFrame:
    """Return visit-level trajectories for the largest errors without raw identifiers."""

    selected = pd.concat(
        [
            residuals.head(20),
            residuals.loc[residuals["hist_count"].ge(6)].head(10),
        ],
        ignore_index=True,
    ).drop_duplicates("patient_key")
    trajectory = selected[
        [
            "patient_key",
            "index_date",
            "target_date",
            "hist_count",
            "target_lsm",
            "locked_ml_model",
            "absolute_error_kpa",
            "history_group",
        ]
    ].merge(
        visits[["patient_key", "visit_date", "lsm", "same_day_lsm_records"]],
        on="patient_key",
        how="left",
        validate="one_to_many",
    )
    trajectory = trajectory.loc[trajectory["visit_date"].le(trajectory["target_date"])].copy()
    trajectory["visit_role"] = np.select(
        [
            trajectory["visit_date"].eq(trajectory["target_date"]),
            trajectory["visit_date"].eq(trajectory["index_date"]),
            trajectory["visit_date"].lt(trajectory["index_date"]),
        ],
        ["target", "index", "available_history"],
        default="post_index_not_used_for_prediction",
    )
    return trajectory.sort_values(["absolute_error_kpa", "patient_key", "visit_date"], ascending=[False, True, True])


def history_group_review(residuals: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group in ["2", "3", "4-5", ">=6"]:
        frame = residuals.loc[residuals["history_group"].eq(group)]
        if frame.empty:
            rows.append(
                {
                    "history_group": group,
                    "patients": 0,
                    "tasks": 0,
                    "target_lsm_median": np.nan,
                    "target_lsm_q1": np.nan,
                    "target_lsm_q3": np.nan,
                    "index_lsm_median": np.nan,
                    "horizon_days_median": np.nan,
                    "horizon_days_q1": np.nan,
                    "horizon_days_q3": np.nan,
                    "absolute_observed_change_kpa_median": np.nan,
                    "large_change_ge_10_kpa_n": 0,
                    "ml_mae_kpa": np.nan,
                }
            )
            continue
        rows.append(
            {
                "history_group": group,
                "patients": int(frame["patient_key"].nunique()),
                "tasks": int(len(frame)),
                "target_lsm_median": float(frame["target_lsm"].median()),
                "target_lsm_q1": float(frame["target_lsm"].quantile(0.25)),
                "target_lsm_q3": float(frame["target_lsm"].quantile(0.75)),
                "index_lsm_median": float(frame["last_lsm"].median()),
                "horizon_days_median": float(frame["horizon_days"].median()),
                "horizon_days_q1": float(frame["horizon_days"].quantile(0.25)),
                "horizon_days_q3": float(frame["horizon_days"].quantile(0.75)),
                "absolute_observed_change_kpa_median": float(frame["absolute_observed_change_kpa"].median()),
                "large_change_ge_10_kpa_n": int(frame["exploratory_large_change_ge_10_kpa"].sum()),
                "ml_mae_kpa": float(frame["absolute_error_kpa"].mean()),
            }
        )
    return pd.DataFrame(rows)


def large_change_sensitivity(
    residuals: pd.DataFrame,
    bootstrap_replicates: int,
) -> pd.DataFrame:
    rows = []
    analysis_sets = {
        "all_2023_test_patients": residuals,
        "exclude_absolute_index_to_target_change_ge_10_kpa": residuals.loc[
            ~residuals["exploratory_large_change_ge_10_kpa"]
        ],
    }
    for set_number, (analysis_set, frame) in enumerate(analysis_sets.items()):
        for method in ["locked_ml_model", "last_observed_lsm"]:
            interval = bootstrap_regression_interval(
                frame,
                method,
                n_bootstrap=bootstrap_replicates,
                seed=RANDOM_SEED + 500 + set_number,
            )
            rows.append(
                {
                    "analysis_set": analysis_set,
                    "method": method,
                    "patients": int(frame["patient_key"].nunique()),
                    "tasks": int(len(frame)),
                    "excluded_large_change_patients": int(
                        residuals["exploratory_large_change_ge_10_kpa"].sum()
                    )
                    if set_number
                    else 0,
                    **interval,
                }
            )
    return pd.DataFrame(rows)


def slope_sensitivity(tasks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for threshold in [0.005, 0.01, 0.02]:
        slope = tasks["slope_kpa_per_day"]
        labels = np.select(
            [slope > threshold, slope < -threshold],
            ["increasing", "decreasing"],
            default="stable",
        )
        for label in ["decreasing", "stable", "increasing"]:
            subset = tasks.loc[labels == label]
            rows.append(
                {
                    "absolute_slope_threshold_kpa_per_day": threshold,
                    "equivalent_kpa_per_year": threshold * 365.25,
                    "trajectory_label": label,
                    "patients": int(subset["patient_key"].nunique()),
                    "tasks": int(len(subset)),
                    "median_target_lsm": float(subset["target_lsm"].median()) if len(subset) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def cluster_analysis(development: pd.DataFrame, bootstrap_replicates: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Describe train-set trajectories without calling clusters clinical stages."""

    feature_columns = ["baseline_lsm", "last_lsm", "hist_mean", "slope_kpa_per_year", "hist_std", "history_days"]
    values = development[feature_columns].copy()
    values = values.fillna(values.median(numeric_only=True))
    values = (values - values.mean()) / values.std(ddof=0).replace(0, 1)
    matrix = values.to_numpy(dtype=float)
    score_rows = []
    labels_by_k: dict[int, np.ndarray] = {}
    for k in range(2, 7):
        model = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=20)
        labels = model.fit_predict(matrix)
        labels_by_k[k] = labels
        score_rows.append(
            {
                "k": k,
                "silhouette": float(silhouette_score(matrix, labels)),
                "calinski_harabasz": float(calinski_harabasz_score(matrix, labels)),
                "davies_bouldin": float(davies_bouldin_score(matrix, labels)),
            }
        )
    scores = pd.DataFrame(score_rows)
    selected_k = int(scores.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0]["k"])
    reference_labels = labels_by_k[selected_k]
    rng = np.random.default_rng(RANDOM_SEED)
    stability_rows = []
    for bootstrap in range(bootstrap_replicates):
        sampled_rows = rng.choice(np.arange(len(matrix)), size=len(matrix), replace=True)
        unique_rows = np.unique(sampled_rows)
        if len(unique_rows) <= selected_k:
            continue
        model = KMeans(n_clusters=selected_k, random_state=RANDOM_SEED + bootstrap + 1, n_init=20)
        model.fit(matrix[sampled_rows])
        bootstrap_labels = model.predict(matrix[unique_rows])
        stability_rows.append(
            {
                "bootstrap": bootstrap + 1,
                "unique_patients": int(len(unique_rows)),
                "adjusted_rand_index": float(adjusted_rand_score(reference_labels[unique_rows], bootstrap_labels)),
            }
        )
    stability = pd.DataFrame(stability_rows)
    assignment = development[
        ["patient_key", "baseline_lsm", "last_lsm", "hist_mean", "slope_kpa_per_year", "target_lsm"]
    ].copy()
    assignment["cluster"] = [f"Cluster {label + 1}" for label in reference_labels]
    summary = (
        assignment.groupby("cluster", as_index=False)
        .agg(
            patients=("patient_key", "nunique"),
            median_baseline_lsm=("baseline_lsm", "median"),
            median_last_lsm=("last_lsm", "median"),
            median_hist_mean_lsm=("hist_mean", "median"),
            median_slope_kpa_per_year=("slope_kpa_per_year", "median"),
            median_future_lsm=("target_lsm", "median"),
        )
        .sort_values("cluster")
    )
    summary["selected_k"] = selected_k
    summary["stability_mean_ari"] = float(stability["adjusted_rand_index"].mean())
    summary["stability_sd_ari"] = float(stability["adjusted_rand_index"].std(ddof=1))
    summary["interpretation"] = "Exploratory continuous-LSM trajectory summary; not a pathology or risk-stratum result."
    return scores, stability, summary


def create_figures(
    prediction: pd.DataFrame,
    subgroup: pd.DataFrame,
    cluster_scores: pd.DataFrame,
    cluster_stability: pd.DataFrame,
    output_dir: Path,
) -> None:
    configure_matplotlib()
    import matplotlib.pyplot as plt

    test = prediction.loc[prediction["evaluation_set"].eq(PUBLIC_TEST_COHORT)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(test["target_lsm"], test["locked_ml_model"], alpha=0.7, color="#2A6F97")
    lower = min(test["target_lsm"].min(), test["locked_ml_model"].min())
    upper = max(test["target_lsm"].max(), test["locked_ml_model"].max())
    axes[0].plot([lower, upper], [lower, upper], linestyle="--", color="#8C1C13")
    axes[0].set_xlabel("Observed future LSM (kPa)")
    axes[0].set_ylabel("Locked ML prediction (kPa)")
    axes[0].set_title("2023 test-set performance")

    plot_data = subgroup.loc[subgroup["subgroup_definition"].eq("history_group")].copy()
    plot_data = plot_data.loc[plot_data["patients"].gt(0)]
    axes[1].errorbar(
        plot_data["subgroup"],
        plot_data["mae_estimate"],
        yerr=[
            plot_data["mae_estimate"] - plot_data["mae_ci_2_5"],
            plot_data["mae_ci_97_5"] - plot_data["mae_estimate"],
        ],
        fmt="o",
        color="#2A6F97",
        capsize=4,
    )
    axes[1].set_xlabel("Historical LSM records")
    axes[1].set_ylabel("MAE (kPa), bootstrap 95% CI")
    axes[1].set_title("2023 test subgroup performance")
    fig.tight_layout()
    fig.savefig(output_dir / "phase8_test_scatter_and_subgroup_mae.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(cluster_scores["k"], cluster_scores["silhouette"], marker="o", color="#2A6F97")
    axes[0].set_xlabel("Number of clusters (k)")
    axes[0].set_ylabel("Silhouette score")
    axes[0].set_title("Train-set cluster selection")
    axes[0].set_xticks(cluster_scores["k"])
    axes[1].hist(cluster_stability["adjusted_rand_index"], bins=15, color="#4F772D", edgecolor="white")
    axes[1].axvline(
        cluster_stability["adjusted_rand_index"].mean(),
        linestyle="--",
        color="#8C1C13",
        label=f"Mean = {cluster_stability['adjusted_rand_index'].mean():.2f}",
    )
    axes[1].set_xlabel("Bootstrap adjusted Rand index")
    axes[1].set_ylabel("Replicates")
    axes[1].set_title("Cluster stability")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "phase8_development_cluster_diagnostics.png", dpi=220)
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    input_dir = Path(args.input_dir) if args.input_dir else _default_result_dir()
    output_dir = ensure_result_dir(Path(args.output_dir) if args.output_dir else _default_result_dir())
    tasks = pd.read_csv(input_dir / "phase1_primary_tasks_one_per_patient.csv", parse_dates=["index_date", "target_date"])
    predictions = pd.read_csv(input_dir / "phase5_locked_temporal_predictions.csv", parse_dates=["index_date", "target_date"])
    visits = pd.read_csv(input_dir / "phase0_deidentified_visits.csv", parse_dates=["visit_date"])
    repeated_tasks = pd.read_csv(
        input_dir / "phase1_repeated_tasks_sensitivity.csv",
        parse_dates=["index_date", "target_date"],
    )
    test_prediction = predictions.loc[predictions["evaluation_set"].eq(PUBLIC_TEST_COHORT)].copy()
    test_tasks = tasks.loc[tasks["cohort"].eq(PUBLIC_TEST_COHORT)].copy()
    repeated_prediction = predictions.loc[
        predictions["evaluation_set"].eq(f"{PUBLIC_TEST_COHORT}_repeated_tasks_sensitivity")
    ].copy()
    repeated_test_tasks = repeated_tasks.loc[
        repeated_tasks["cohort"].eq(PUBLIC_TEST_COHORT)
    ].copy()
    development = tasks.loc[tasks["cohort"].eq("development_le_2022")].copy()

    subgroup = subgroup_results(test_prediction, test_tasks, args.bootstrap_replicates)
    residuals = residual_review(test_prediction, test_tasks)
    repeated_subgroup = subgroup_results(
        repeated_prediction,
        repeated_test_tasks,
        args.bootstrap_replicates,
    )
    repeated_residuals = residual_review(repeated_prediction, repeated_test_tasks)
    trajectories = residual_trajectories(repeated_residuals, visits)
    history_review = history_group_review(residuals)
    repeated_history_review = history_group_review(repeated_residuals)
    jump_sensitivity = large_change_sensitivity(residuals, args.bootstrap_replicates)
    slopes = slope_sensitivity(test_tasks)
    cluster_scores, cluster_stability, cluster_association = cluster_analysis(
        development,
        args.cluster_bootstrap_replicates,
    )

    write_csv(subgroup, output_dir / "phase8_temporal_test_subgroup_performance.csv")
    write_csv(
        repeated_subgroup,
        output_dir / "phase8_repeated_test_subgroup_performance.csv",
    )
    write_csv(residuals.head(20), output_dir / "phase8_temporal_test_largest_residuals.csv")
    ge6_residuals = repeated_residuals.loc[repeated_residuals["hist_count"].ge(6)].head(20)
    write_csv(ge6_residuals, output_dir / "phase8_repeated_test_ge6_history_largest_residuals.csv")
    # Keep the original planned filename synchronized rather than leaving stale output.
    write_csv(ge6_residuals, output_dir / "phase8_ge6_history_largest_residuals.csv")
    write_csv(trajectories, output_dir / "phase8_largest_residual_deidentified_trajectories.csv")
    write_csv(history_review, output_dir / "phase8_history_group_distribution_review.csv")
    write_csv(
        repeated_history_review,
        output_dir / "phase8_repeated_test_history_group_distribution_review.csv",
    )
    write_csv(jump_sensitivity, output_dir / "phase8_large_change_sensitivity.csv")
    write_csv(slopes, output_dir / "phase8_slope_threshold_sensitivity.csv")
    write_csv(cluster_scores, output_dir / "phase8_development_cluster_k_metrics.csv")
    write_csv(cluster_stability, output_dir / "phase8_development_cluster_bootstrap_stability.csv")
    write_csv(cluster_association, output_dir / "phase8_development_cluster_summary.csv")
    create_figures(predictions, subgroup, cluster_scores, cluster_stability, output_dir)
    write_csv(
        pd.DataFrame(
            [
                {
                    "requested_subgroup": "ALT abnormal status",
                    "status": "not_executed",
                    "reason": "The local ALT upper limit of normal was not supplied.",
                },
                {
                    "requested_subgroup": "AST abnormal status",
                    "status": "not_executed",
                    "reason": "The local AST upper limit of normal was not supplied.",
                },
            ]
        ),
        output_dir / "phase8_alt_ast_subgroup_gate.csv",
    )
    write_json(
        {
            "subgroups": "Evaluated only in the locked 2023 test set; R2 is descriptive for groups with <20 patients.",
            "repeated_task_sensitivity": "All eligible 2023 tasks from patients assigned to the test set were evaluated only after model lock. This secondary task-weighted analysis uses patient-level bootstrap and supplies the >=6-history review absent from the one-task-per-patient primary cohort.",
            "alt_ast_subgroups": "Not executed because local laboratory upper limits of normal were not supplied; no universal 40 U/L cutoff was assumed.",
            "index_lsm_subgroups": "Numeric descriptive bins only; they are not F2 stages or validated clinical risk strata.",
            "slope": "Continuous slope is the primary representation; +/-0.005, +/-0.01, and +/-0.02 kPa/day labels are sensitivity descriptions only.",
            "large_change_sensitivity": "An exploratory >=10 kPa absolute index-to-target change flag is used only for sensitivity analysis. It is not a clinical cutoff, and all observations remain in the primary analysis.",
            "clustering": "Train-set exploratory clustering uses historical LSM features only. Categories are neutral Cluster 1...k, are not pathology stages, and are not tested against unconfirmed LSM threshold categories.",
        },
        output_dir / "phase8_analysis_notes.json",
    )
    print("Subgroup, residual, slope, and exploratory clustering outputs written.")


if __name__ == "__main__":
    main()
