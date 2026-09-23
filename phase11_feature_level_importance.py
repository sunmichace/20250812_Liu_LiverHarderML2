"""Feature-level leave-one-out ablation and permutation importance for locked RF.

This secondary analysis answers a question that block-level ablation cannot answer:
which individual predictors contribute marginal information, and which predictors
the locked full model relies on in temporal hold-out cohorts.  The primary design
is unchanged: one task per patient, train-set fitting only, and 2023/2024
time-separated evaluation.  The primary random-forest algorithm and hyperparameters
are reused without feature-level model selection.

Two complementary estimands are reported:

* Leave-one-feature-out (LOFO): refit the locked RF after removing one feature.
  The paired MAE change is ``MAE_without_feature - MAE_full``; positive values mean
  that removing the feature worsens prediction.
* Permutation importance: retain the full model, permute one feature across patients
  within an evaluation cohort, and measure the paired MAE increase.  Importance is
  averaged over repeated permutations and accompanied by patient-level bootstrap CIs.

Correlated predictors can substitute for one another.  Therefore a small individual
LOFO or permutation value does not prove that a feature is clinically irrelevant;
it may indicate redundancy with other longitudinal LSM summaries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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
from modeling_core import CORE_FEATURE_COLUMNS, bootstrap_metric_intervals, make_pipeline, regression_metrics
from phase9_reporting import append_csv_provenance, write_release_manifest, write_result_catalog


LOCKED_RF_PARAMS = {
    "model__max_features": 0.7,
    "model__min_samples_leaf": 3,
}

BASELINE_METHOD = "last_observed_lsm"
FULL_METHOD = "full_rf"
LOFO_PREFIX = "without_"


def _split_label(evaluation_set: str) -> str | None:
    """Return the canonical public split for an internal evaluation-set key."""

    if evaluation_set == "development_apparent":
        return "train"
    return COHORT_TO_SPLIT.get(evaluation_set)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run feature-level LOFO ablation and permutation importance for locked RF."
    )
    parser.add_argument("--input-dir", default=None, help="Directory holding phase 1 task CSV files.")
    parser.add_argument("--output-dir", default=None, help="Directory for third-analysis results.")
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument(
        "--permutation-repeats",
        type=int,
        default=100,
        help="Number of within-cohort permutations per feature and evaluation set.",
    )
    return parser


def _default_result_dir() -> Path:
    return Path(__file__).resolve().parent / "result"


def _source_sha(tasks: pd.DataFrame) -> str | None:
    values = tasks.get("provenance_source_workbook_sha256", pd.Series(dtype="string")).dropna()
    return str(values.iloc[0]) if len(values) else None


def _load_primary_tasks(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tasks = pd.read_csv(
        input_dir / "phase1_primary_tasks_one_per_patient.csv",
        parse_dates=["index_date", "target_date"],
    )
    missing = sorted(set(CORE_FEATURE_COLUMNS) - set(tasks.columns))
    if missing:
        raise KeyError(f"Primary task table is missing locked features: {missing}")
    if tasks["patient_key"].duplicated().any():
        raise ValueError("Primary task table must contain one task per patient.")
    frames = {
        "development_le_2022": tasks.loc[tasks["cohort"].eq("development_le_2022")]
        .sort_values("index_date")
        .reset_index(drop=True),
        "temporal_validation_2023": tasks.loc[tasks["cohort"].eq("temporal_validation_2023")]
        .sort_values("index_date")
        .reset_index(drop=True),
        "temporal_test_2024": tasks.loc[tasks["cohort"].eq("temporal_test_2024")]
        .sort_values("index_date")
        .reset_index(drop=True),
    }
    if any(frame.empty for frame in frames.values()):
        raise RuntimeError("Train, validation, and test sets must all be nonempty.")
    names = list(frames)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            if set(frames[left_name]["patient_key"]) & set(frames[right_name]["patient_key"]):
                raise ValueError(f"Patient overlap between {left_name} and {right_name}.")
    # Return the public validation and test sets in their canonical roles:
    # 2023 is test and 2024 is validation for this manuscript revision.
    return frames["development_le_2022"], frames[PUBLIC_VALIDATION_COHORT], frames[PUBLIC_TEST_COHORT]


def _lofo_method(feature: str) -> str:
    return f"{LOFO_PREFIX}{feature}"


def _fit_full_and_lofo_models(development: pd.DataFrame) -> dict[str, Any]:
    models: dict[str, Any] = {}
    full = make_pipeline("random_forest", LOCKED_RF_PARAMS)
    full.fit(development[CORE_FEATURE_COLUMNS], development["target_lsm"])
    models[FULL_METHOD] = full
    for feature in CORE_FEATURE_COLUMNS:
        features = [column for column in CORE_FEATURE_COLUMNS if column != feature]
        model = make_pipeline("random_forest", LOCKED_RF_PARAMS)
        model.fit(development[features], development["target_lsm"])
        models[_lofo_method(feature)] = (model, features)
    return models


def _prediction_frame(
    evaluation_set: str,
    frame: pd.DataFrame,
    models: dict[str, Any],
) -> pd.DataFrame:
    output = frame[
        ["patient_key", "index_date", "target_date", "horizon_days", "target_lsm", "hist_count"]
    ].copy()
    output[BASELINE_METHOD] = frame["last_lsm"].to_numpy(dtype=float)
    full_model = models[FULL_METHOD]
    output[FULL_METHOD] = full_model.predict(frame[CORE_FEATURE_COLUMNS]).astype(float)
    for feature in CORE_FEATURE_COLUMNS:
        model, features = models[_lofo_method(feature)]
        output[_lofo_method(feature)] = model.predict(frame[features]).astype(float)
    output["evaluation_set"] = evaluation_set
    output["split"] = _split_label(evaluation_set)
    return output


def _paired_delta_ci(
    prediction_frame: pd.DataFrame,
    reduced_method: str,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float | int | str]:
    """Patient-level CI for MAE(reduced) minus MAE(full)."""

    if prediction_frame["patient_key"].duplicated().any():
        raise ValueError("Feature-level analysis requires one task per patient.")
    y = prediction_frame["target_lsm"].to_numpy(dtype=float)
    full_error = np.abs(y - prediction_frame[FULL_METHOD].to_numpy(dtype=float))
    reduced_error = np.abs(y - prediction_frame[reduced_method].to_numpy(dtype=float))
    paired_delta = reduced_error - full_error
    patients = prediction_frame["patient_key"].to_numpy()
    rng = np.random.default_rng(seed)
    bootstrap_values = np.empty(n_bootstrap, dtype=float)
    for replicate in range(n_bootstrap):
        sampled = rng.integers(0, len(paired_delta), size=len(paired_delta))
        bootstrap_values[replicate] = float(np.mean(paired_delta[sampled]))
    interval = np.quantile(bootstrap_values, [0.025, 0.975])
    return {
        "feature": reduced_method.removeprefix(LOFO_PREFIX),
        "full_method": FULL_METHOD,
        "ablation_method": reduced_method,
        "patients": int(len(patients)),
        "tasks": int(len(prediction_frame)),
        "mae_delta_removed_minus_full": float(np.mean(paired_delta)),
        "ci_2_5": float(interval[0]),
        "ci_97_5": float(interval[1]),
        "bootstrap_replicates": int(n_bootstrap),
        "interpretation": "positive values mean removing the feature worsens MAE; CI excluding zero supports marginal contribution",
    }


def _permutation_importance_for_feature(
    model: Any,
    frame: pd.DataFrame,
    feature: str,
    permutation_repeats: int,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float | int | str]:
    """Compute paired MAE increase after within-cohort feature permutation."""

    if frame["patient_key"].duplicated().any():
        raise ValueError("Permutation importance requires one task per patient.")
    n_rows = len(frame)
    y = frame["target_lsm"].to_numpy(dtype=float)
    base_prediction = model.predict(frame[CORE_FEATURE_COLUMNS]).astype(float)
    base_error = np.abs(y - base_prediction)
    original_values = frame[feature].to_numpy(copy=True)
    repeated = pd.concat(
        [frame[CORE_FEATURE_COLUMNS].reset_index(drop=True)] * permutation_repeats,
        ignore_index=True,
    )
    permutation_rng = np.random.default_rng(seed)
    for repeat in range(permutation_repeats):
        start = repeat * n_rows
        stop = start + n_rows
        repeated.loc[start:stop - 1, feature] = original_values[permutation_rng.permutation(n_rows)]
    permuted_prediction = model.predict(repeated[CORE_FEATURE_COLUMNS]).astype(float).reshape(
        permutation_repeats, n_rows
    )
    delta = np.abs(y[None, :] - permuted_prediction) - base_error[None, :]
    patient_level_delta = delta.mean(axis=0)
    point = float(patient_level_delta.mean())
    bootstrap_rng = np.random.default_rng(seed + 1_000_003)
    bootstrap_values = np.empty(n_bootstrap, dtype=float)
    for replicate in range(n_bootstrap):
        sampled = bootstrap_rng.integers(0, n_rows, size=n_rows)
        bootstrap_values[replicate] = float(patient_level_delta[sampled].mean())
    interval = np.quantile(bootstrap_values, [0.025, 0.975])
    repeat_estimates = delta.mean(axis=1)
    return {
        "feature": feature,
        "patients": int(n_rows),
        "tasks": int(n_rows),
        "base_method": FULL_METHOD,
        "base_mae": float(base_error.mean()),
        "permutation_mae_delta": point,
        "ci_2_5": float(interval[0]),
        "ci_97_5": float(interval[1]),
        "permutation_repeats": int(permutation_repeats),
        "permutation_sd": float(np.std(repeat_estimates, ddof=1)) if permutation_repeats > 1 else 0.0,
        "bootstrap_replicates": int(n_bootstrap),
        "interpretation": "positive values mean the locked model relies on the feature; correlated features may yield attenuated importance",
    }


def _permutation_importance(
    models: dict[str, Any],
    frames: list[tuple[str, pd.DataFrame]],
    permutation_repeats: int,
    n_bootstrap: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    full_model = models[FULL_METHOD]
    for evaluation_index, (evaluation_set, frame) in enumerate(frames):
        for feature_index, feature in enumerate(CORE_FEATURE_COLUMNS):
            row = _permutation_importance_for_feature(
                full_model,
                frame,
                feature,
                permutation_repeats=permutation_repeats,
                n_bootstrap=n_bootstrap,
                seed=RANDOM_SEED + 10_000 + evaluation_index * 1_000 + feature_index,
            )
            row["evaluation_set"] = evaluation_set
            row["split"] = _split_label(evaluation_set)
            row["feature_order"] = feature_index + 1
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = build_parser().parse_args()
    if args.bootstrap_replicates < 1:
        raise ValueError("--bootstrap-replicates must be positive.")
    if args.permutation_repeats < 1:
        raise ValueError("--permutation-repeats must be positive.")
    input_dir = Path(args.input_dir) if args.input_dir else _default_result_dir()
    output_dir = ensure_result_dir(Path(args.output_dir) if args.output_dir else _default_result_dir())
    development, validation, test = _load_primary_tasks(input_dir)
    source_tasks = pd.concat([development, validation, test], ignore_index=True)
    models = _fit_full_and_lofo_models(development)
    frames = [
        ("development_apparent", development),
        (PUBLIC_TEST_COHORT, test),
        (PUBLIC_VALIDATION_COHORT, validation),
    ]
    prediction_table = pd.concat(
        [_prediction_frame(evaluation_set, frame, models) for evaluation_set, frame in frames],
        ignore_index=True,
    )

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
        max_abs_diff = float(np.max(np.abs(current[FULL_METHOD] - current["locked_ml_model"])))
        if max_abs_diff > 1e-10:
            raise AssertionError(
                f"Full feature-level model does not reproduce locked RF; max difference={max_abs_diff}."
            )
        lock_check = {
            "performed": True,
            "matched_rows": int(len(current)),
            "max_absolute_prediction_difference": max_abs_diff,
        }

    methods = [FULL_METHOD, *[_lofo_method(feature) for feature in CORE_FEATURE_COLUMNS]]
    performance = pd.concat(
        [
            bootstrap_metric_intervals(
                prediction_table.loc[prediction_table["evaluation_set"].eq(evaluation_set)],
                methods,
                n_bootstrap=args.bootstrap_replicates,
                seed=RANDOM_SEED + 30_000 + evaluation_index,
            ).assign(
                evaluation_set=evaluation_set,
                split=_split_label(evaluation_set),
                patients=int(frame["patient_key"].nunique()),
                tasks=int(len(frame)),
            )
            for evaluation_index, (evaluation_set, frame) in enumerate(frames)
        ],
        ignore_index=True,
    )
    performance = performance[
        [
            "evaluation_set",
            "split",
            "patients",
            "tasks",
            "method",
            "metric",
            "estimate",
            "ci_2_5",
            "ci_97_5",
            "bootstrap_replicates",
        ]
    ]

    lofo_rows: list[dict[str, Any]] = []
    for evaluation_index, (evaluation_set, frame) in enumerate(frames):
        prediction_frame = prediction_table.loc[
            prediction_table["evaluation_set"].eq(evaluation_set)
        ].reset_index(drop=True)
        for feature_index, feature in enumerate(CORE_FEATURE_COLUMNS):
            row = _paired_delta_ci(
                prediction_frame,
                _lofo_method(feature),
                n_bootstrap=args.bootstrap_replicates,
                seed=RANDOM_SEED + 40_000 + evaluation_index * 1_000 + feature_index,
            )
            row["evaluation_set"] = evaluation_set
            row["split"] = _split_label(evaluation_set)
            row["feature_order"] = feature_index + 1
            lofo_rows.append(row)
    lofo = pd.DataFrame(lofo_rows)
    permutation = _permutation_importance(
        models,
        frames,
        permutation_repeats=args.permutation_repeats,
        n_bootstrap=args.bootstrap_replicates,
    )

    write_csv(prediction_table, output_dir / "phase11_feature_level_predictions.csv")
    write_csv(performance, output_dir / "phase11_feature_level_performance.csv")
    write_csv(lofo, output_dir / "phase11_feature_level_lofo_mae_delta.csv")
    write_csv(permutation, output_dir / "phase11_permutation_importance.csv")

    metadata = {
        "analysis": "feature_level_lofo_and_permutation_importance",
        "analysis_protocol_version": ANALYSIS_PROTOCOL_VERSION,
        "source_workbook_sha256": _source_sha(source_tasks),
        "random_seed": RANDOM_SEED,
        "bootstrap_replicates": args.bootstrap_replicates,
        "permutation_repeats": args.permutation_repeats,
        "primary_task_rule": "one earliest eligible task per patient; train <=2022, test 2023, validation 2024",
        "data_split_labels": {
            "train": "development_le_2022",
            "test": "temporal_validation_2023",
            "validation": "temporal_test_2024",
        },
        "model": "random_forest",
        "hyperparameters": LOCKED_RF_PARAMS,
        "model_selection": "No feature-level model selection; primary locked RF algorithm and parameters were reused for the full and every leave-one-feature-out model.",
        "preprocessing": "Prespecified range cleaning and fold-local median imputation with missingness indicators; all fitted models use train data only.",
        "features": CORE_FEATURE_COLUMNS,
        "lofo_definition": "MAE without feature minus MAE full; positive values mean removal worsens prediction.",
        "permutation_definition": "Within-evaluation-cohort patient-level feature permutation; importance is permuted MAE minus unpermuted full-model MAE.",
        "correlation_caveat": "Individual importance can be attenuated when correlated longitudinal summaries substitute for one another; low marginal importance is not proof of clinical irrelevance.",
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
            "phase11_feature_level_predictions.csv",
            "phase11_feature_level_performance.csv",
            "phase11_feature_level_lofo_mae_delta.csv",
            "phase11_permutation_importance.csv",
        ],
    }
    write_json(metadata, output_dir / "phase11_feature_level_importance_metadata.json")

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

    print("Feature-level LOFO and permutation importance completed.")
    print(
        f"Train={len(development)}, test={len(test)}, validation={len(validation)}; "
        f"features={len(CORE_FEATURE_COLUMNS)}, permutation_repeats={args.permutation_repeats}; "
        f"full-model lock check={lock_check}"
    )
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
