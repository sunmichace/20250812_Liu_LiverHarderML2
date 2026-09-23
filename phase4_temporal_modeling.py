"""Nested temporal model selection and locked temporal evaluation for continuous LSM."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_core import (
    COHORT_TO_SPLIT,
    PUBLIC_TEST_COHORT,
    PUBLIC_VALIDATION_COHORT,
    RANDOM_SEED,
    ensure_result_dir,
    write_csv,
    write_json,
)
from modeling_core import (
    CORE_FEATURE_COLUMNS,
    bootstrap_continuous_calibration,
    bootstrap_mae_gain,
    bootstrap_metric_intervals,
    choose_model_from_outer_results,
    clinical_range_audit,
    date_expanding_splits,
    make_pipeline,
    nested_temporal_evaluation,
    tune_candidate,
)


BASELINE_COLUMNS = ["last_lsm", "baseline_lsm", "hist_mean", "linear_trend_365"]
BASELINE_LABELS = {
    "last_lsm": "last_observed_lsm",
    "baseline_lsm": "baseline_lsm",
    "hist_mean": "historical_mean_lsm",
    "linear_trend_365": "linear_trend_extrapolation_365_days",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run nested temporal regression modeling for LSM.")
    parser.add_argument("--input-dir", default=None, help="Directory holding phase 1 task CSV files.")
    parser.add_argument("--output-dir", default=None, help="Directory for third-analysis results.")
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    return parser


def _default_result_dir() -> Path:
    return Path(__file__).resolve().parent / "result"


def _evaluate_dataset(
    name: str,
    frame: pd.DataFrame,
    pipeline,
    bootstrap_replicates: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output = frame[
        ["patient_key", "index_date", "target_date", "horizon_days", "target_lsm", "hist_count"]
    ].copy()
    for source, label in BASELINE_LABELS.items():
        output[label] = frame[source].to_numpy(dtype=float)
    output["locked_ml_model"] = pipeline.predict(frame[CORE_FEATURE_COLUMNS])
    output["evaluation_set"] = name
    output["split"] = (
        COHORT_TO_SPLIT["development_le_2022"]
        if name == "development_apparent"
        else COHORT_TO_SPLIT.get(name.split("_repeated_tasks_sensitivity", 1)[0])
    )

    methods = [*BASELINE_LABELS.values(), "locked_ml_model"]
    intervals = bootstrap_metric_intervals(
        output,
        methods,
        n_bootstrap=bootstrap_replicates,
        seed=RANDOM_SEED,
    )
    intervals["evaluation_set"] = name
    gains = bootstrap_mae_gain(
        output,
        "locked_ml_model",
        list(BASELINE_LABELS.values()),
        n_bootstrap=bootstrap_replicates,
        seed=RANDOM_SEED + 1,
    )
    gains["evaluation_set"] = name
    return output, intervals, gains


def main() -> None:
    args = build_parser().parse_args()
    input_dir = Path(args.input_dir) if args.input_dir else _default_result_dir()
    output_dir = ensure_result_dir(Path(args.output_dir) if args.output_dir else _default_result_dir())
    tasks = pd.read_csv(input_dir / "phase1_primary_tasks_one_per_patient.csv", parse_dates=["index_date", "target_date"])
    repeated_sensitivity = pd.read_csv(
        input_dir / "phase1_repeated_tasks_sensitivity.csv",
        parse_dates=["index_date", "target_date"],
    )
    development = tasks.loc[tasks["cohort"].eq("development_le_2022")].sort_values("index_date").reset_index(drop=True)
    public_test = tasks.loc[
        tasks["cohort"].eq(PUBLIC_TEST_COHORT)
    ].sort_values("index_date").reset_index(drop=True)
    public_validation = tasks.loc[
        tasks["cohort"].eq(PUBLIC_VALIDATION_COHORT)
    ].sort_values("index_date").reset_index(drop=True)
    repeated_public_test = repeated_sensitivity.loc[
        repeated_sensitivity["cohort"].eq(PUBLIC_TEST_COHORT)
    ].sort_values(["index_date", "patient_key"]).reset_index(drop=True)

    if development.empty or public_test.empty or public_validation.empty:
        raise RuntimeError("Train, validation, and test sets must all be nonempty.")
    if development["patient_key"].duplicated().any():
        raise RuntimeError("Train data must have one primary task per patient.")

    write_csv(clinical_range_audit(tasks), output_dir / "phase4_prespecified_range_audit.csv")
    outer_splits = date_expanding_splits(development, n_splits=4)
    outer_results, inner_results = nested_temporal_evaluation(
        development,
        CORE_FEATURE_COLUMNS,
        outer_splits,
    )
    selected_model = choose_model_from_outer_results(outer_results)

    full_development_inner_splits = date_expanding_splits(development, n_splits=4)
    selected_params, full_tuning = tune_candidate(
        development,
        CORE_FEATURE_COLUMNS,
        selected_model,
        full_development_inner_splits,
    )
    locked_pipeline = make_pipeline(selected_model, selected_params)
    locked_pipeline.fit(development[CORE_FEATURE_COLUMNS], development["target_lsm"])

    evaluations: list[pd.DataFrame] = []
    intervals: list[pd.DataFrame] = []
    gains: list[pd.DataFrame] = []
    calibration_results: list[pd.DataFrame] = []
    evaluation_sets = [
        ("development_apparent", development),
        (PUBLIC_TEST_COHORT, public_test),
        (PUBLIC_VALIDATION_COHORT, public_validation),
    ]
    if not repeated_public_test.empty:
        evaluation_sets.append(
            (f"{PUBLIC_TEST_COHORT}_repeated_tasks_sensitivity", repeated_public_test)
        )
    for name, frame in evaluation_sets:
        prediction, metric_interval, mae_gain = _evaluate_dataset(
            name,
            frame,
            locked_pipeline,
            args.bootstrap_replicates,
        )
        evaluations.append(prediction)
        intervals.append(metric_interval)
        gains.append(mae_gain)
        calibration = bootstrap_continuous_calibration(
            prediction,
            [*BASELINE_LABELS.values(), "locked_ml_model"],
            n_bootstrap=args.bootstrap_replicates,
            seed=RANDOM_SEED + 2,
        )
        calibration["evaluation_set"] = name
        calibration_results.append(calibration)

    write_csv(outer_results, output_dir / "phase4_nested_outer_temporal_cv.csv")
    write_csv(inner_results, output_dir / "phase4_nested_inner_tuning.csv")
    write_csv(full_tuning, output_dir / "phase4_locked_model_full_development_tuning.csv")
    write_csv(pd.concat(evaluations, ignore_index=True), output_dir / "phase5_locked_temporal_predictions.csv")
    write_csv(pd.concat(intervals, ignore_index=True), output_dir / "phase5_continuous_performance_bootstrap_ci.csv")
    write_csv(
        pd.concat(calibration_results, ignore_index=True),
        output_dir / "phase5_continuous_calibration_bootstrap_ci.csv",
    )
    write_csv(pd.concat(gains, ignore_index=True), output_dir / "phase6_ml_vs_baseline_mae_gain.csv")

    selection_summary = (
        outer_results.groupby("model", as_index=False)
        .agg(
            outer_folds=("outer_fold", "nunique"),
            mean_validation_mae=("mae", "mean"),
            sd_validation_mae=("mae", "std"),
            mean_validation_rmse=("rmse", "mean"),
            mean_validation_r2=("r2", "mean"),
        )
        .sort_values("mean_validation_mae")
    )
    write_csv(selection_summary, output_dir / "phase4_model_selection_summary.csv")
    write_json(
        {
            "feature_columns": CORE_FEATURE_COLUMNS,
            "feature_selection": "Protocol-defined for the 8.19 retrospective rerun; no correlation, VIF, importance, or test-set selection was used.",
            "outer_cv": "Four expanding date-block folds within the <=2022 train set; patients have one primary task only.",
            "inner_cv": "Expanding date-block folds within each outer training subset; preprocessing and hyperparameter tuning occur inside the training subset.",
            "candidate_models": ["ridge", "random_forest", "hist_gradient_boosting"],
            "selected_model": selected_model,
            "selected_parameters": selected_params,
            "selection_rule": "Lowest mean outer-fold MAE; ties favor lower protocol-defined complexity rank.",
            "locked_before_temporal_evaluation": True,
            "lock_scope": "The feature set, model candidates, and tuning rules were frozen in code before the formal 8.19 full rerun; this is not a preregistered prospective lock.",
            "historical_prospective_lock": False,
            "bootstrap_replicates": args.bootstrap_replicates,
            "baseline_methods": BASELINE_LABELS,
            "repeated_task_sensitivity": "The locked model was additionally evaluated on all eligible 2023 tasks from patients assigned to the public test set. It did not influence model selection; bootstrap resampling remained patient-level.",
            "platelet_excluded_from_ml": "Platelet completeness improved in the 8.19 workbook, but the ML feature definition was retained unchanged to avoid post-result feature expansion. Platelet is used only in the FIB-4 feasibility and descriptive analysis.",
        },
        output_dir / "phase4_model_lock.json",
    )

    print(f"Selected model: {selected_model} with parameters {selected_params}")
    print(f"Train tasks: {len(development)}, 2023 test: {len(public_test)}, 2024 validation: {len(public_validation)}")
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
