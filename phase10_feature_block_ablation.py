"""Feature-block ablation and incremental-value analysis for the locked RF.

This secondary analysis keeps the primary temporal design intact:

* one primary task per patient;
* train (index year <= 2022), 2023 test, and 2024 validation;
* all preprocessing is performed inside the fitted pipeline;
* the random-forest algorithm and hyperparameters are fixed to the primary locked
  model (max_features=0.7, min_samples_leaf=3); and
* the 2023 test and 2024 validation cohorts are never used for fitting.

The purpose is to isolate the information contributed by longitudinal LSM features
and clinical/laboratory features. It is not a new model-selection exercise and does
not replace the primary locked analysis.
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_core import (
    ANALYSIS_PROTOCOL_VERSION,
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
    bootstrap_metric_intervals,
    date_expanding_splits,
    make_pipeline,
    regression_metrics,
)
from phase9_reporting import append_csv_provenance, write_release_manifest, write_result_catalog


LOCKED_RF_PARAMS = {
    "model__max_features": 0.7,
    "model__min_samples_leaf": 3,
}

CLINICAL_FEATURE_COLUMNS = [
    "age",
    "cap",
    "alt",
    "ast",
    "total_bilirubin",
    "alt_lag_days",
    "ast_lag_days",
    "total_bilirubin_lag_days",
]

LONGITUDINAL_LSM_FEATURE_COLUMNS = [
    column for column in CORE_FEATURE_COLUMNS if column not in CLINICAL_FEATURE_COLUMNS
]

# Ordered for stable output and manuscript table construction.
FEATURE_BLOCKS = OrderedDict(
    [
        ("lsm_history_rf", LONGITUDINAL_LSM_FEATURE_COLUMNS),
        ("clinical_only_rf", CLINICAL_FEATURE_COLUMNS),
        (
            "last_lsm_plus_clinical_rf",
            ["last_lsm", *CLINICAL_FEATURE_COLUMNS],
        ),
        ("full_rf", CORE_FEATURE_COLUMNS),
    ]
)

BASELINE_METHOD = "last_observed_lsm"


def _split_label(evaluation_set: str) -> str | None:
    """Return the canonical public split for an internal evaluation-set key."""

    if evaluation_set == "development_apparent":
        return "train"
    return COHORT_TO_SPLIT.get(evaluation_set.split("_repeated_tasks_sensitivity", 1)[0])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run locked random-forest feature-block ablation for continuous LSM."
    )
    parser.add_argument("--input-dir", default=None, help="Directory holding phase 1 task CSV files.")
    parser.add_argument("--output-dir", default=None, help="Directory for third-analysis results.")
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    return parser


def _default_result_dir() -> Path:
    return Path(__file__).resolve().parent / "result"


def _source_sha(tasks: pd.DataFrame) -> str | None:
    values = tasks.get("provenance_source_workbook_sha256", pd.Series(dtype="string")).dropna()
    return str(values.iloc[0]) if len(values) else None


def _assert_feature_sets(tasks: pd.DataFrame) -> None:
    missing = {
        block: sorted(set(features) - set(tasks.columns))
        for block, features in FEATURE_BLOCKS.items()
    }
    missing = {block: values for block, values in missing.items() if values}
    if missing:
        raise KeyError(f"Feature block columns are missing from task table: {missing}")
    if len(set(CORE_FEATURE_COLUMNS)) != len(CORE_FEATURE_COLUMNS):
        raise ValueError("CORE_FEATURE_COLUMNS contains duplicate columns.")
    if set(CLINICAL_FEATURE_COLUMNS) & set(LONGITUDINAL_LSM_FEATURE_COLUMNS):
        raise ValueError("Clinical and longitudinal feature blocks overlap.")


def _fit_block_and_predict(
    development: pd.DataFrame,
    frame: pd.DataFrame,
    features: list[str],
) -> np.ndarray:
    pipeline = make_pipeline("random_forest", LOCKED_RF_PARAMS)
    pipeline.fit(development[features], development["target_lsm"])
    return pipeline.predict(frame[features]).astype(float)


def _prediction_frame(
    name: str,
    development: pd.DataFrame,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    output = frame[
        ["patient_key", "index_date", "target_date", "horizon_days", "target_lsm", "hist_count"]
    ].copy()
    output[BASELINE_METHOD] = frame["last_lsm"].to_numpy(dtype=float)
    for block, features in FEATURE_BLOCKS.items():
        output[block] = _fit_block_and_predict(development, frame, features)
    output["evaluation_set"] = name
    output["split"] = _split_label(name)
    return output


def _paired_mae_gain(
    prediction_frame: pd.DataFrame,
    reference_method: str,
    comparison_method: str,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float | int | str]:
    """Return reference MAE minus comparison MAE using patient-level bootstrap."""

    if prediction_frame["patient_key"].duplicated().any():
        raise ValueError("Primary feature-block analysis requires one task per patient.")
    y = prediction_frame["target_lsm"].to_numpy(dtype=float)
    reference_error = np.abs(y - prediction_frame[reference_method].to_numpy(dtype=float))
    comparison_error = np.abs(y - prediction_frame[comparison_method].to_numpy(dtype=float))
    point = float(np.mean(reference_error - comparison_error))

    patients = prediction_frame["patient_key"].drop_duplicates().to_numpy()
    grouped_indices = {
        patient: prediction_frame.index[prediction_frame["patient_key"].eq(patient)].to_numpy()
        for patient in patients
    }
    rng = np.random.default_rng(seed)
    bootstrap_values: list[float] = []
    for _ in range(n_bootstrap):
        sampled_patients = rng.choice(patients, size=len(patients), replace=True)
        sampled_indices = np.concatenate([grouped_indices[patient] for patient in sampled_patients])
        bootstrap_values.append(
            float(np.mean((reference_error - comparison_error)[sampled_indices]))
        )
    interval = np.quantile(np.asarray(bootstrap_values), [0.025, 0.975])
    return {
        "reference_method": reference_method,
        "comparison_method": comparison_method,
        "patients": int(prediction_frame["patient_key"].nunique()),
        "tasks": int(len(prediction_frame)),
        "mae_gain_reference_minus_comparison": point,
        "ci_2_5": float(interval[0]),
        "ci_97_5": float(interval[1]),
        "bootstrap_replicates": int(n_bootstrap),
        "interpretation": "positive values favor the comparison method; CI excluding zero supports an improvement",
    }


def _performance(
    prediction_frame: pd.DataFrame,
    evaluation_set: str,
    n_bootstrap: int,
) -> pd.DataFrame:
    methods = [BASELINE_METHOD, *FEATURE_BLOCKS.keys()]
    intervals = bootstrap_metric_intervals(
        prediction_frame,
        methods,
        n_bootstrap=n_bootstrap,
        seed=RANDOM_SEED + 101,
    )
    intervals.insert(0, "evaluation_set", evaluation_set)
    intervals.insert(1, "split", _split_label(evaluation_set))
    intervals.insert(2, "patients", int(prediction_frame["patient_key"].nunique()))
    intervals.insert(3, "tasks", int(len(prediction_frame)))
    return intervals


def _development_cv(development: pd.DataFrame) -> pd.DataFrame:
    """Evaluate fixed block models in the four expanding train folds."""

    rows: list[dict[str, float | int | str]] = []
    for fold_number, (train_index, validation_index) in enumerate(
        date_expanding_splits(development, n_splits=4), start=1
    ):
        train = development.iloc[train_index]
        validation = development.iloc[validation_index]
        baseline_metrics = regression_metrics(
            validation["target_lsm"], validation["last_lsm"]
        )
        rows.append(
            {
                "outer_fold": fold_number,
                "evaluation_set": "development_le_2022",
                "split": "train",
                "method": BASELINE_METHOD,
                "feature_count": 0,
                "train_tasks": int(len(train)),
                "validation_tasks": int(len(validation)),
                "train_end": str(pd.to_datetime(train["index_date"]).max().date()),
                "validation_start": str(pd.to_datetime(validation["index_date"]).min().date()),
                **baseline_metrics,
            }
        )
        for block, features in FEATURE_BLOCKS.items():
            pipeline = make_pipeline("random_forest", LOCKED_RF_PARAMS)
            pipeline.fit(train[features], train["target_lsm"])
            prediction = pipeline.predict(validation[features])
            rows.append(
                {
                    "outer_fold": fold_number,
                    "evaluation_set": "development_le_2022",
                    "split": "train",
                    "method": block,
                    "feature_count": len(features),
                    "train_tasks": int(len(train)),
                    "validation_tasks": int(len(validation)),
                    "train_end": str(pd.to_datetime(train["index_date"]).max().date()),
                    "validation_start": str(pd.to_datetime(validation["index_date"]).min().date()),
                    **regression_metrics(validation["target_lsm"], prediction),
                }
            )
    return pd.DataFrame(rows)


def _load_primary_tasks(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tasks = pd.read_csv(
        input_dir / "phase1_primary_tasks_one_per_patient.csv",
        parse_dates=["index_date", "target_date"],
    )
    _assert_feature_sets(tasks)
    development = (
        tasks.loc[tasks["cohort"].eq("development_le_2022")]
        .sort_values("index_date")
        .reset_index(drop=True)
    )
    test = (
        tasks.loc[tasks["cohort"].eq(PUBLIC_TEST_COHORT)]
        .sort_values("index_date")
        .reset_index(drop=True)
    )
    validation = (
        tasks.loc[tasks["cohort"].eq(PUBLIC_VALIDATION_COHORT)]
        .sort_values("index_date")
        .reset_index(drop=True)
    )
    if any(frame.empty for frame in [development, validation, test]):
        raise RuntimeError("Train, validation, and test sets must all be nonempty.")
    if tasks["patient_key"].duplicated().any():
        raise ValueError("Primary task table must contain one task per patient.")
    if set(development["patient_key"]) & set(validation["patient_key"]):
        raise ValueError("Train and validation patients overlap.")
    if set(development["patient_key"]) & set(test["patient_key"]):
        raise ValueError("Train and test patients overlap.")
    if set(validation["patient_key"]) & set(test["patient_key"]):
        raise ValueError("Validation and test patients overlap.")
    return development, validation, test


def main() -> None:
    args = build_parser().parse_args()
    input_dir = Path(args.input_dir) if args.input_dir else _default_result_dir()
    output_dir = ensure_result_dir(Path(args.output_dir) if args.output_dir else _default_result_dir())
    development, validation, test = _load_primary_tasks(input_dir)
    source_tasks = pd.concat([development, validation, test], ignore_index=True)

    frames = [
        ("development_apparent", development),
        (PUBLIC_TEST_COHORT, test),
        (PUBLIC_VALIDATION_COHORT, validation),
    ]
    predictions = [
        _prediction_frame(name, development, frame)
        for name, frame in frames
    ]
    prediction_table = pd.concat(predictions, ignore_index=True)

    # Verify that the full feature block reproduces the primary locked model.
    lock_check: dict[str, object] = {"performed": False}
    existing_prediction_path = input_dir / "phase5_locked_temporal_predictions.csv"
    if existing_prediction_path.is_file():
        existing = pd.read_csv(
            existing_prediction_path,
            parse_dates=["index_date", "target_date"],
            usecols=[
                "patient_key",
                "index_date",
                "target_date",
                "evaluation_set",
                "locked_ml_model",
            ],
        )
        current = prediction_table.merge(
            existing,
            on=["patient_key", "index_date", "target_date", "evaluation_set"],
            how="inner",
            validate="one_to_one",
        )
        if len(current) != len(prediction_table):
            raise ValueError("Full-model lock check did not match every primary prediction row.")
        max_abs_diff = float(np.max(np.abs(current["full_rf"] - current["locked_ml_model"])))
        if max_abs_diff > 1e-10:
            raise AssertionError(
                f"Full feature-block model does not reproduce the locked RF; max difference={max_abs_diff}."
            )
        lock_check = {
            "performed": True,
            "matched_rows": int(len(current)),
            "max_absolute_prediction_difference": max_abs_diff,
        }

    performance = pd.concat(
        [
            _performance(frame, name, args.bootstrap_replicates)
            for name, frame in zip(
                [name for name, _ in frames], predictions, strict=True
            )
        ],
        ignore_index=True,
    )

    # Comparisons most directly relevant to incremental value. Positive values
    # favor the comparison method because they are reference MAE minus comparison MAE.
    comparison_pairs = [
        (BASELINE_METHOD, "lsm_history_rf"),
        (BASELINE_METHOD, "last_lsm_plus_clinical_rf"),
        (BASELINE_METHOD, "full_rf"),
        ("lsm_history_rf", "full_rf"),
        ("clinical_only_rf", "full_rf"),
        ("last_lsm_plus_clinical_rf", "full_rf"),
    ]
    gain_rows: list[dict[str, object]] = []
    for evaluation_set, frame in zip(
        [name for name, _ in frames], predictions, strict=True
    ):
        for pair_number, (reference, comparison) in enumerate(comparison_pairs):
            row = _paired_mae_gain(
                frame,
                reference,
                comparison,
                n_bootstrap=args.bootstrap_replicates,
                seed=RANDOM_SEED + 201 + pair_number,
            )
            row["evaluation_set"] = evaluation_set
            row["split"] = _split_label(evaluation_set)
            gain_rows.append(row)
    gains = pd.DataFrame(gain_rows)

    development_cv = _development_cv(development)

    write_csv(
        prediction_table,
        output_dir / "phase10_feature_block_ablation_predictions.csv",
    )
    write_csv(
        performance,
        output_dir / "phase10_feature_block_ablation_performance.csv",
    )
    write_csv(
        gains,
        output_dir / "phase10_feature_block_ablation_mae_gain.csv",
    )
    write_csv(
        development_cv,
        output_dir / "phase10_feature_block_ablation_development_cv.csv",
    )

    metadata = {
        "analysis": "feature_block_ablation_and_incremental_value",
        "analysis_protocol_version": ANALYSIS_PROTOCOL_VERSION,
        "source_workbook_sha256": _source_sha(source_tasks),
        "random_seed": RANDOM_SEED,
        "bootstrap_replicates": args.bootstrap_replicates,
        "primary_task_rule": "one earliest eligible task per patient; train <=2022, test 2023, validation 2024",
        "data_split_labels": {
            "train": "development_le_2022",
            "test": "temporal_validation_2023",
            "validation": "temporal_test_2024",
        },
        "model": "random_forest",
        "hyperparameters": LOCKED_RF_PARAMS,
        "model_selection": "No ablation-specific model selection; primary locked RF algorithm and parameters were reused for every fitted block.",
        "preprocessing": "Prespecified range cleaning and median imputation with missingness indicators are fitted inside each pipeline using train data only for locked evaluations.",
        "feature_blocks": FEATURE_BLOCKS,
        "baseline_method": BASELINE_METHOD,
        "incremental_value_definition": "reference MAE minus comparison MAE; positive values favor the comparison method",
        "lock_reproduction_check": lock_check,
        "cohort_sizes": {
            "train": int(len(development)),
            "validation": int(len(validation)),
            "test": int(len(test)),
            "development": int(len(development)),
            "temporal_validation_2023": int(len(test)),
            "temporal_test_2024": int(len(validation)),
        },
        "patient_overlap": False,
        "outputs": [
            "phase10_feature_block_ablation_predictions.csv",
            "phase10_feature_block_ablation_performance.csv",
            "phase10_feature_block_ablation_mae_gain.csv",
            "phase10_feature_block_ablation_development_cv.csv",
        ],
    }
    write_json(metadata, output_dir / "phase10_feature_block_ablation_metadata.json")

    # Keep secondary outputs auditable in the same way as the formal result
    # files. This also updates result_catalog.csv and the release manifest so
    # the new analysis is included in the repository's traceability package.
    formal_metadata = json.loads(
        (output_dir / "analysis_metadata.json").read_text(encoding="utf-8")
    )
    primary_tasks = pd.read_csv(
        input_dir / "phase1_primary_tasks_one_per_patient.csv",
        parse_dates=["index_date", "target_date"],
    )
    append_csv_provenance(output_dir, formal_metadata, primary_tasks)
    write_result_catalog(output_dir, formal_metadata, primary_tasks)
    write_release_manifest(output_dir, formal_metadata)

    print("Feature-block ablation completed.")
    print(
        f"Train={len(development)}, test={len(test)}, validation={len(validation)}; "
        f"full-model lock check={lock_check}"
    )
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
