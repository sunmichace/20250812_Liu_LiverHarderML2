"""Build publication-oriented tables, figures, and revision drafts from locked outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis_core import (
    ANALYSIS_PROTOCOL_VERSION,
    CAP_VALID_RANGE_DB_PER_M,
    COHORT_TO_SPLIT,
    CONFIRMED_CLINICAL_UNITS,
    INCLUSION_RULE_VERSION,
    PUBLIC_TEST_COHORT,
    PUBLIC_VALIDATION_COHORT,
    RANDOM_SEED,
    SOURCE_WORKBOOK_UPDATE_CUTOFF,
    configure_matplotlib,
    ensure_result_dir,
    write_csv,
    write_json,
)
from modeling_core import PRESPECIFIED_VALID_RANGES


COHORT_ORDER = [
    "development_le_2022",
    "temporal_validation_2023",
    "temporal_test_2024",
]

METHOD_LABELS = {
    "last_observed_lsm": "Last observed LSM",
    "baseline_lsm": "Baseline LSM",
    "historical_mean_lsm": "Historical mean LSM",
    "linear_trend_extrapolation_365_days": "Linear trend extrapolation",
}

TABLE1_CHARACTERISTICS = [
    ("age", "Age at index", "years"),
    ("last_lsm", "Index liver stiffness", "kPa"),
    ("baseline_lsm", "First observed liver stiffness", "kPa"),
    ("target_lsm", "Future liver stiffness", "kPa"),
    ("hist_count", "Historical LSM records", "count"),
    ("history_days", "Available LSM history", "days"),
    ("horizon_days", "Prediction horizon", "days"),
    ("cap", "Controlled attenuation parameter (CAP)", CONFIRMED_CLINICAL_UNITS["cap"]),
    ("alt", "ALT", CONFIRMED_CLINICAL_UNITS["alt"]),
    ("ast", "AST", CONFIRMED_CLINICAL_UNITS["ast"]),
    ("total_bilirubin", "Total bilirubin", CONFIRMED_CLINICAL_UNITS["total_bilirubin"]),
    ("platelet", "Platelet count", CONFIRMED_CLINICAL_UNITS["platelet"]),
    ("reported_iqr_ratio_normalized", "Reported LSM IQR/median ratio", "ratio"),
]

CSV_PROVENANCE_COLUMNS = [
    "provenance_analysis_protocol_version",
    "provenance_source_workbook_sha256",
    "provenance_inclusion_rule_version",
    "provenance_random_seed",
    "provenance_software_versions_json",
    "provenance_scope",
    "provenance_patients",
    "provenance_tasks",
    "provenance_index_date_min",
    "provenance_index_date_max",
    "provenance_target_date_min",
    "provenance_target_date_max",
]


# The article package uses stable submission names, while result/ retains the
# descriptive analysis filenames and all machine-readable source outputs.
PUBLICATION_MAIN_FIGURES = [
    ("Fig1", "publication_figure1_cohort_and_data_quality", "Figure 1"),
    ("Fig2", "publication_figure2_model_development", "Figure 2"),
    ("Fig3", "publication_figure3_locked_test_performance", "Figure 3"),
    ("Fig4", "publication_figure4_feature_level_importance", "Figure 4"),
    ("Fig5", "publication_figure5_subgroups_and_robustness", "Figure 5"),
]

PUBLICATION_SUPPLEMENTARY_FIGURES = [
    ("SupFig1", "publication_supplementary_figure1_cluster_diagnostics", "Supplementary Figure 1"),
]

PUBLICATION_MAIN_TABLES = [
    ("Table1", "Table 1. Analysis-set characteristics and overall sex comparisons"),
    ("Table2", "Table 2. Performance in the 2023 test set and simple baselines"),
    ("Table3", "Table 3. Subgroup performance in the 2023 test set"),
]

# Keep only supplementary tables whose row-level detail cannot be replaced by a
# main-text figure. All other detailed tables remain auditable in result/.
PUBLICATION_SUPPLEMENTARY_TABLES = [
    (
        "SupplementaryTable1_nested_temporal_cv",
        "Supplementary Table 1. Nested expanding temporal cross-validation details",
        "supplementary_table3_nested_temporal_cv.csv",
    ),
    (
        "SupplementaryTable2_diagnosis_evidence",
        "Supplementary Table 2. Ever-recorded nonexclusive liver-disease diagnosis-text evidence by analysis set",
        "supplementary_table7_diagnosis_evidence.csv",
    ),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate third-analysis tables and revision drafts.")
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--article-dir", default=None)
    return parser


def _default_result_dir() -> Path:
    return Path(__file__).resolve().parent / "result"


def _default_article_dir() -> Path:
    return Path(__file__).resolve().parent / "文章"


def write_release_manifest(output_dir: Path, metadata: dict[str, Any]) -> None:
    """Record file hashes and fixed analysis settings for standalone result traceability."""

    files = []
    for path in sorted(output_dir.iterdir()):
        if not path.is_file() or path.name in {
            "release_manifest.json",
            "release_manifest.md",
            "validation_report.json",
            "validation_report.md",
        }:
            continue
        files.append(
            {
                "file": path.name,
                "bytes": int(path.stat().st_size),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    payload = {
        "analysis_name": metadata["analysis_name"],
        "input_workbook_sha256": metadata.get("input_workbook_sha256"),
        "analysis_protocol_version": metadata.get("analysis_protocol_version", ANALYSIS_PROTOCOL_VERSION),
        "inclusion_rule_version": metadata.get("inclusion_rule_version", INCLUSION_RULE_VERSION),
        "random_seed": RANDOM_SEED,
        "endpoint": metadata["endpoint"],
        "inclusion_rule": metadata["primary_history_requirement"],
        "cohort_assignment": metadata["cohort_assignment"],
        "software_versions": metadata["software_versions"],
        "files": files,
    }
    write_json(payload, output_dir / "release_manifest.json")
    lines = ["# Third-Analysis Release Manifest", ""]
    lines.append(f"- Analysis: `{payload['analysis_name']}`")
    lines.append(f"- Source workbook SHA-256: `{payload['input_workbook_sha256']}`")
    lines.append(f"- Random seed: `{payload['random_seed']}`")
    lines.append(f"- Endpoint: {payload['endpoint']}")
    lines.append(f"- Inclusion rule: {payload['inclusion_rule']}")
    lines.append(f"- Cohort assignment: {payload['cohort_assignment']}")
    lines.append("")
    lines.append("| File | Bytes | SHA-256 |")
    lines.append("|---|---:|---|")
    for file in files:
        lines.append(f"| `{file['file']}` | {file['bytes']} | `{file['sha256']}` |")
    (output_dir / "release_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_csv_provenance(
    output_dir: Path,
    metadata: dict[str, Any],
    tasks: pd.DataFrame,
) -> None:
    """Append fixed provenance fields to every result CSV after analysis is complete."""

    tasks = tasks.copy()
    tasks["index_date"] = pd.to_datetime(tasks["index_date"], errors="raise")
    tasks["target_date"] = pd.to_datetime(tasks["target_date"], errors="raise")
    task_context = (
        tasks.groupby("cohort", as_index=False)
        .agg(
            patients=("patient_key", "nunique"),
            tasks=("patient_key", "size"),
            index_date_min=("index_date", "min"),
            index_date_max=("index_date", "max"),
            target_date_min=("target_date", "min"),
            target_date_max=("target_date", "max"),
        )
        .set_index("cohort")
    )
    default_context = {
        "scope": "all_primary_cohorts",
        "patients": int(tasks["patient_key"].nunique()),
        "tasks": int(len(tasks)),
        "index_date_min": str(tasks["index_date"].min().date()),
        "index_date_max": str(tasks["index_date"].max().date()),
        "target_date_min": str(tasks["target_date"].min().date()),
        "target_date_max": str(tasks["target_date"].max().date()),
    }
    file_scope = {
        "table2_": PUBLIC_TEST_COHORT,
        "table3_": PUBLIC_TEST_COHORT,
        "table4_": PUBLIC_TEST_COHORT,
        "phase8_temporal_test_": PUBLIC_TEST_COHORT,
        "phase8_repeated_test_": f"{PUBLIC_TEST_COHORT}_repeated_tasks_sensitivity",
        "phase8_large_change_": PUBLIC_TEST_COHORT,
        "phase8_slope_": PUBLIC_TEST_COHORT,
        "phase4_": "development_le_2022",
    }
    prediction_scopes = pd.read_csv(
        output_dir / "phase5_locked_temporal_predictions.csv",
        parse_dates=["index_date", "target_date"],
    )
    evaluation_context: dict[str, dict[str, Any]] = {}
    for evaluation_set, frame in prediction_scopes.groupby("evaluation_set", sort=False):
        evaluation_context[str(evaluation_set)] = {
            "scope": str(evaluation_set),
            "patients": int(frame["patient_key"].nunique()),
            "tasks": int(len(frame)),
            "index_date_min": str(frame["index_date"].min().date()),
            "index_date_max": str(frame["index_date"].max().date()),
            "target_date_min": str(frame["target_date"].min().date()),
            "target_date_max": str(frame["target_date"].max().date()),
        }

    def context_for_scope(scope: str) -> dict[str, Any]:
        if scope in task_context.index:
            row = task_context.loc[scope]
            return {
                "scope": scope,
                "patients": int(row["patients"]),
                "tasks": int(row["tasks"]),
                "index_date_min": str(pd.Timestamp(row["index_date_min"]).date()),
                "index_date_max": str(pd.Timestamp(row["index_date_max"]).date()),
                "target_date_min": str(pd.Timestamp(row["target_date_min"]).date()),
                "target_date_max": str(pd.Timestamp(row["target_date_max"]).date()),
            }
        return evaluation_context.get(scope, {**default_context, "scope": scope})

    static_values = {
        "provenance_analysis_protocol_version": metadata.get(
            "analysis_protocol_version", ANALYSIS_PROTOCOL_VERSION
        ),
        "provenance_source_workbook_sha256": metadata.get("input_workbook_sha256"),
        "provenance_inclusion_rule_version": metadata.get(
            "inclusion_rule_version", INCLUSION_RULE_VERSION
        ),
        "provenance_random_seed": RANDOM_SEED,
        "provenance_software_versions_json": json.dumps(metadata["software_versions"], sort_keys=True),
    }
    for path in sorted(output_dir.glob("*.csv")):
        if path.name == "result_catalog.csv":
            continue
        frame = pd.read_csv(path)
        frame = frame.drop(columns=[column for column in CSV_PROVENANCE_COLUMNS if column in frame])
        for column, value in static_values.items():
            frame[column] = value
        if "cohort" in frame and frame["cohort"].notna().all():
            contexts = [context_for_scope(str(scope)) for scope in frame["cohort"]]
        elif "evaluation_set" in frame and frame["evaluation_set"].notna().all():
            contexts = [context_for_scope(str(scope)) for scope in frame["evaluation_set"]]
        else:
            scope = next(
                (value for prefix, value in file_scope.items() if path.name.startswith(prefix)),
                default_context["scope"],
            )
            contexts = [context_for_scope(scope)] * len(frame)
        for key in [
            "scope",
            "patients",
            "tasks",
            "index_date_min",
            "index_date_max",
            "target_date_min",
            "target_date_max",
        ]:
            frame[f"provenance_{key}"] = [context[key] for context in contexts]
        write_csv(frame, path)


def write_result_catalog(
    output_dir: Path,
    metadata: dict[str, Any],
    tasks: pd.DataFrame,
) -> None:
    """Attach common provenance to every CSV without changing publication table schemas."""

    cohort_summary = (
        tasks.groupby("cohort", as_index=False)
        .agg(
            patients=("patient_key", "nunique"),
            tasks=("patient_key", "size"),
            index_date_min=("index_date", "min"),
            index_date_max=("index_date", "max"),
            target_date_min=("target_date", "min"),
            target_date_max=("target_date", "max"),
        )
        .set_index("cohort")
    )
    default_context = {
        "context_scope": "all_primary_cohorts",
        "context_patients": int(tasks["patient_key"].nunique()),
        "context_tasks": int(len(tasks)),
        "context_index_date_min": str(tasks["index_date"].min()),
        "context_index_date_max": str(tasks["index_date"].max()),
        "context_target_date_min": str(tasks["target_date"].min()),
        "context_target_date_max": str(tasks["target_date"].max()),
    }
    scope_by_prefix = {
        "table2_": PUBLIC_TEST_COHORT,
        "table3_": PUBLIC_TEST_COHORT,
        "table4_": PUBLIC_TEST_COHORT,
        "phase5_": "multiple_evaluation_sets",
        "phase6_": "multiple_evaluation_sets",
        "phase8_temporal_test_": PUBLIC_TEST_COHORT,
        "phase8_repeated_test_": PUBLIC_TEST_COHORT,
        "phase8_large_change_": PUBLIC_TEST_COHORT,
        "phase8_slope_": PUBLIC_TEST_COHORT,
        "phase4_": "development_le_2022",
        "phase7_": "all_primary_cohorts",
    }
    scope_notes = {
        "phase0_deidentified_visits.csv": "All usable dated LSM visits after same-day median consolidation.",
        "phase1_fixed_horizon_candidate_tasks.csv": "All fixed-horizon candidate tasks before the primary-history and one-task-per-patient rule.",
        "phase1_primary_tasks_one_per_patient.csv": "One earliest eligible primary task per patient.",
        "phase1_repeated_tasks_sensitivity.csv": "All eligible tasks within each patient's originally assigned temporal cohort.",
        "phase5_locked_temporal_predictions.csv": "Multiple post-lock evaluation sets; use evaluation_set for subset definition.",
        "phase5_continuous_calibration_bootstrap_ci.csv": "Continuous observed-on-predicted calibration by evaluation set.",
        "phase5_continuous_performance_bootstrap_ci.csv": "Multiple post-lock evaluation sets; use evaluation_set for subset definition.",
        "phase6_ml_vs_baseline_mae_gain.csv": "Multiple post-lock evaluation sets; use evaluation_set for subset definition.",
    }
    rows = []
    for path in sorted(output_dir.glob("*.csv")):
        if path.name == "result_catalog.csv":
            continue
        header = pd.read_csv(path, nrows=0).columns.tolist()
        inspect_columns = [
            column
            for column in [
                "patient_key",
                "index_date",
                "target_date",
                "visit_date",
                "index_date_min",
                "index_date_max",
                "target_date_min",
                "target_date_max",
                "cohort",
                "evaluation_set",
            ]
            if column in header
        ]
        source = pd.read_csv(path, usecols=inspect_columns) if inspect_columns else pd.read_csv(path)
        record_patients = (
            int(source["patient_key"].nunique()) if "patient_key" in source else pd.NA
        )
        is_task_level = {"patient_key", "index_date", "target_date"}.issubset(source.columns)
        record_tasks = int(len(source)) if is_task_level else pd.NA

        def date_bounds(minimum_column: str, maximum_column: str) -> tuple[str | None, str | None]:
            if minimum_column in source and maximum_column in source:
                minimum = pd.to_datetime(source[minimum_column], errors="coerce").min()
                maximum = pd.to_datetime(source[maximum_column], errors="coerce").max()
            elif minimum_column in source:
                values = pd.to_datetime(source[minimum_column], errors="coerce")
                minimum, maximum = values.min(), values.max()
            else:
                return None, None
            return (
                minimum.date().isoformat() if pd.notna(minimum) else None,
                maximum.date().isoformat() if pd.notna(maximum) else None,
            )

        index_min, index_max = date_bounds("index_date", "index_date_max")
        if index_min is None:
            index_min, index_max = date_bounds("index_date_min", "index_date_max")
        target_min, target_max = date_bounds("target_date", "target_date_max")
        if target_min is None:
            target_min, target_max = date_bounds("target_date_min", "target_date_max")
        visit_min, visit_max = date_bounds("visit_date", "visit_date")
        scope = next(
            (value for prefix, value in scope_by_prefix.items() if path.name.startswith(prefix)),
            "all_primary_cohorts",
        )
        context = default_context.copy()
        if scope in cohort_summary.index:
            summary = cohort_summary.loc[scope]
            context = {
                "context_scope": scope,
                "context_patients": int(summary["patients"]),
                "context_tasks": int(summary["tasks"]),
                "context_index_date_min": str(summary["index_date_min"]),
                "context_index_date_max": str(summary["index_date_max"]),
                "context_target_date_min": str(summary["target_date_min"]),
                "context_target_date_max": str(summary["target_date_max"]),
            }
        else:
            context["context_scope"] = scope
        rows.append(
            {
                "file": path.name,
                "analysis_protocol_version": metadata.get(
                    "analysis_protocol_version", ANALYSIS_PROTOCOL_VERSION
                ),
                "source_workbook_sha256": metadata.get("input_workbook_sha256"),
                "inclusion_rule_version": metadata.get(
                    "inclusion_rule_version", INCLUSION_RULE_VERSION
                ),
                "random_seed": RANDOM_SEED,
                "software_versions_json": json.dumps(metadata["software_versions"], sort_keys=True),
                "records_in_file": int(len(source)),
                "unique_patients_in_file": record_patients,
                "task_records_in_file": record_tasks,
                "file_index_date_min": index_min,
                "file_index_date_max": index_max,
                "file_target_date_min": target_min,
                "file_target_date_max": target_max,
                "file_visit_date_min": visit_min,
                "file_visit_date_max": visit_max,
                "scope_note": scope_notes.get(path.name, "See file columns and context scope."),
                **context,
            }
        )
    write_csv(pd.DataFrame(rows), output_dir / "result_catalog.csv")


def _metric_row(
    performance: pd.DataFrame,
    evaluation_set: str,
    method: str,
    metric: str,
) -> pd.Series:
    rows = performance.loc[
        performance["evaluation_set"].eq(evaluation_set)
        & performance["method"].eq(method)
        & performance["metric"].eq(metric)
    ]
    if len(rows) != 1:
        raise ValueError(f"Expected one result for {evaluation_set}/{method}/{metric}; found {len(rows)}")
    return rows.iloc[0]


def _format_interval(row: pd.Series, digits: int = 3) -> str:
    return (
        f"{row['estimate']:.{digits}f} "
        f"({row['ci_2_5']:.{digits}f} to {row['ci_97_5']:.{digits}f})"
    )


def _cohort_patient_count(cohort_summary: pd.DataFrame, cohort: str) -> int:
    """Read a cohort size from the generated summary rather than a fixed manuscript value."""

    rows = cohort_summary.loc[cohort_summary["cohort"].eq(cohort), "patients"]
    if len(rows) != 1:
        raise ValueError(f"Expected one cohort-summary row for {cohort}; found {len(rows)}")
    return int(rows.iloc[0])


def _locked_model_label(selected_model: str) -> str:
    return f"Locked {selected_model.replace('_', ' ')}"


def _incremental_value_supported(gain: pd.Series) -> bool:
    """Apply the protocol-defined paired-MAE rule: a positive CI entirely above zero."""

    return bool(float(gain["ci_2_5"]) > 0)


def _incremental_value_text(gain: pd.Series) -> str:
    if _incremental_value_supported(gain):
        return "supports incremental continuous-prediction value over the last observed LSM"
    return "does not support incremental continuous-prediction value over the last observed LSM"


def _history_group_counts(subgroup: pd.DataFrame) -> dict[str, int]:
    rows = subgroup.loc[subgroup["subgroup_definition"].eq("history_group")]
    return {str(row["subgroup"]): int(row["patients"]) for _, row in rows.iterrows()}


def build_table1(tasks: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cohort in COHORT_ORDER:
        frame = tasks.loc[tasks["cohort"].eq(cohort)]
        known_sex = frame["sex"].isin(["Female", "Male"])
        female = frame["sex"].eq("Female")
        rows.append(
            {
                "cohort": cohort,
                "patients": int(frame["patient_key"].nunique()),
                "tasks": int(len(frame)),
                "variable": "sex_female",
                "characteristic": "Female sex",
                "unit": "n (%)",
                "nonmissing_n": int(known_sex.sum()),
                "missing_n": int((~known_sex).sum()),
                "missing_pct": 100 * float((~known_sex).mean()),
                "median": np.nan,
                "q1": np.nan,
                "q3": np.nan,
                "mean": float(female.mean()),
                "sd": np.nan,
                "display_summary": f"{int(female.sum())} ({100 * float(female.mean()):.1f}%)",
            }
        )
        for variable, label, unit in TABLE1_CHARACTERISTICS:
            values = pd.to_numeric(frame[variable], errors="coerce")
            if variable in PRESPECIFIED_VALID_RANGES:
                lower, upper = PRESPECIFIED_VALID_RANGES[variable]
                values = values.where(values.between(lower, upper))
            rows.append(
                {
                    "cohort": cohort,
                    "patients": int(frame["patient_key"].nunique()),
                    "tasks": int(len(frame)),
                    "variable": variable,
                    "characteristic": label,
                    "unit": unit,
                    "nonmissing_n": int(values.notna().sum()),
                    "missing_n": int(values.isna().sum()),
                    "missing_pct": 100 * float(values.isna().mean()),
                    "median": float(values.median()),
                    "q1": float(values.quantile(0.25)),
                    "q3": float(values.quantile(0.75)),
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)),
                    "display_summary": "",
                }
            )
    return pd.DataFrame(rows)


def build_table2(
    performance: pd.DataFrame,
    gains: pd.DataFrame,
    test_patients: int,
    selected_model: str,
) -> pd.DataFrame:
    evaluation_set = PUBLIC_TEST_COHORT
    method_order = [
        "locked_ml_model",
        "last_observed_lsm",
        "baseline_lsm",
        "historical_mean_lsm",
        "linear_trend_extrapolation_365_days",
    ]
    rows = []
    for method in method_order:
        method_label = (
            _locked_model_label(selected_model)
            if method == "locked_ml_model"
            else METHOD_LABELS[method]
        )
        record: dict[str, Any] = {
            "evaluation_set": evaluation_set,
            "patients": test_patients,
            "tasks": test_patients,
            "method": method,
            "method_label": method_label,
        }
        for metric in ["mae", "rmse", "r2"]:
            metric_row = _metric_row(performance, evaluation_set, method, metric)
            record[f"{metric}_estimate"] = float(metric_row["estimate"])
            record[f"{metric}_ci_2_5"] = float(metric_row["ci_2_5"])
            record[f"{metric}_ci_97_5"] = float(metric_row["ci_97_5"])
        if method != "locked_ml_model":
            gain = gains.loc[
                gains["evaluation_set"].eq(evaluation_set)
                & gains["baseline_method"].eq(method)
            ].iloc[0]
            record["baseline_minus_ml_mae_gain"] = float(gain["mae_gain_baseline_minus_ml"])
            record["gain_ci_2_5"] = float(gain["ci_2_5"])
            record["gain_ci_97_5"] = float(gain["ci_97_5"])
        else:
            record["baseline_minus_ml_mae_gain"] = np.nan
            record["gain_ci_2_5"] = np.nan
            record["gain_ci_97_5"] = np.nan
        rows.append(record)
    return pd.DataFrame(rows)


def build_data_dictionary() -> pd.DataFrame:
    rows = [
        ("patient_key", "Deterministic pseudonymous patient key", "none", "all dates", "No", "Grouping only"),
        ("sex", "Patient sex normalized from the source records", "category", "patient-level", "No", "Reported descriptively; no conflicts in the primary cohort"),
        ("index_date", "Prediction date", "date", "known at prediction", "No", "Temporal split only"),
        ("target_lsm", "Closest LSM 270-450 days after index", "kPa", "future outcome", "No", "Outcome; never imputed"),
        ("last_lsm", "LSM at index", "kPa", "known at prediction", "Yes", "No missing values"),
        ("baseline_lsm", "First historical LSM", "kPa", "known at prediction", "Yes", "No missing values"),
        ("hist_mean", "Mean historical LSM through index", "kPa", "known at prediction", "Yes", "No missing values"),
        ("hist_median", "Median historical LSM through index", "kPa", "known at prediction", "Yes", "No missing values"),
        ("hist_std", "Historical LSM standard deviation", "kPa", "known at prediction", "Yes", "No missing values"),
        ("hist_iqr", "Historical LSM interquartile range", "kPa", "known at prediction", "Yes", "No missing values"),
        ("hist_range", "Historical maximum minus minimum LSM", "kPa", "known at prediction", "Yes", "No missing values"),
        ("slope_kpa_per_day", "Linear historical LSM slope on actual dates", "kPa/day", "known at prediction", "Yes", "Continuous feature"),
        ("hist_count", "Historical LSM count through index", "count", "known at prediction", "Yes", "No missing values"),
        ("history_days", "Days from first LSM to index", "days", "known at prediction", "Yes", "No missing values"),
        ("prior_interval_days", "Most recent historical LSM interval", "days", "known at prediction", "Yes", "Fold-median imputation plus indicator"),
        ("mean_prior_interval_days", "Mean historical LSM interval", "days", "known at prediction", "Yes", "Fold-median imputation plus indicator"),
        ("sd_prior_interval_days", "SD of historical LSM intervals", "days", "known at prediction", "Yes", "Fold-median imputation plus indicator"),
        ("age", "Latest same-day or prior recorded age", "years", "known at prediction", "Yes", "Fold-median imputation plus indicator"),
        (
            "cap",
            "Latest same-day or prior FibroScan controlled attenuation parameter",
            CONFIRMED_CLINICAL_UNITS["cap"],
            "known at prediction",
            "Yes",
            "Values outside 100-400 dB/m set missing; fold-median imputation plus indicator",
        ),
        ("alt", "Latest same-day or prior ALT", CONFIRMED_CLINICAL_UNITS["alt"], "known at prediction", "Yes", "Fold-median imputation plus indicator"),
        ("ast", "Latest same-day or prior AST", CONFIRMED_CLINICAL_UNITS["ast"], "known at prediction", "Yes", "Fold-median imputation plus indicator"),
        ("total_bilirubin", "Latest same-day or prior total bilirubin", CONFIRMED_CLINICAL_UNITS["total_bilirubin"], "known at prediction", "Yes", "Fold-median imputation plus indicator"),
        ("platelet", "Latest same-day or prior platelet count", CONFIRMED_CLINICAL_UNITS["platelet"], "known at prediction", "No", "Retained for FIB-4 analysis; excluded from ML to preserve the prior feature definition"),
        ("reported_iqr_ratio_normalized", "Reported LSM IQR percentage normalized to a ratio", "ratio", "known at prediction", "No", "Descriptive only; not used for retrospective TE exclusion"),
        ("diagnosis_recorded", "Any source diagnosis text recorded", "binary", "patient-level", "No", "Raw diagnosis text is not released"),
        ("diagnosis_hbv", "HBV-related diagnosis evidence", "binary", "patient-level", "No", "Nonexclusive keyword evidence; clinical adjudication required"),
        ("diagnosis_hcv", "HCV-related diagnosis evidence", "binary", "patient-level", "No", "Nonexclusive keyword evidence; clinical adjudication required"),
        ("diagnosis_alcohol_related", "Alcohol-related diagnosis evidence", "binary", "patient-level", "No", "Nonexclusive keyword evidence; clinical adjudication required"),
        ("diagnosis_fatty_liver", "Fatty-liver diagnosis evidence", "binary", "patient-level", "No", "Nonexclusive keyword evidence; clinical adjudication required"),
        ("diagnosis_autoimmune_or_cholestatic", "Autoimmune or cholestatic diagnosis evidence", "binary", "patient-level", "No", "Nonexclusive keyword evidence; clinical adjudication required"),
        ("diagnosis_cryptogenic", "Cryptogenic diagnosis evidence", "binary", "patient-level", "No", "Nonexclusive keyword evidence; clinical adjudication required"),
        ("diagnosis_cirrhosis", "Cirrhosis diagnosis evidence", "binary", "patient-level", "No", "Comorbidity evidence; not an etiology"),
        ("diagnosis_liver_malignancy", "Liver-malignancy diagnosis evidence", "binary", "patient-level", "No", "Comorbidity evidence; not an etiology"),
    ]
    return pd.DataFrame(
        rows,
        columns=["variable", "definition", "unit", "time_availability", "in_ml_model", "missing_handling"],
    )


def build_flow_table(metadata: dict[str, Any], cohort_summary: pd.DataFrame) -> pd.DataFrame:
    visit = metadata["visit_audit"]
    task = metadata["task_audit"]
    rows = [
        ("Workbook rows", visit["raw_rows"], "All rows in Sheet1"),
        ("Rows containing LSM", visit["rows_with_lsm"], "Before date and range checks"),
        ("LSM rows without usable date", visit["rows_with_lsm_but_no_chosen_date"], "Excluded"),
        ("Usable dated LSM rows", visit["usable_lsm_rows"], "LSM 2-75 kPa with patient and date"),
        (
            "CAP values outside device range",
            visit["cap_source_values_outside_device_range_set_missing"],
            "Set missing outside 100-400 dB/m before as-of matching",
        ),
        ("Same-day consolidated visits", visit["visit_rows_after_same_day_median"], "Median within patient-date"),
        ("Patients with usable LSM", visit["usable_lsm_patients"], "Candidate longitudinal population"),
        ("Fixed-horizon candidate tasks", task["all_fixed_horizon_candidates"], "Target 270-450 days after index"),
        ("Primary-history candidate patients", task["primary_history_candidate_patients"], "At least two historical LSM values"),
        ("Primary one-task-per-patient cohort", task["main_task_patients"], "Earliest eligible task per patient"),
    ]
    for _, cohort in cohort_summary.iterrows():
        rows.append((cohort["cohort"], int(cohort["patients"]), "Non-overlapping primary cohort"))
    return pd.DataFrame(rows, columns=["flow_step", "count", "definition"])


def add_missing_handling(missingness: pd.DataFrame) -> pd.DataFrame:
    output = missingness.copy()
    output["analysis_handling"] = "Not used by locked ML model"
    ml_features = {
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
        "age",
        "cap",
        "alt",
        "ast",
        "total_bilirubin",
        "alt_lag_days",
        "ast_lag_days",
        "total_bilirubin_lag_days",
    }
    output.loc[output["variable"].isin(ml_features), "analysis_handling"] = (
        "Training-fold median imputation with missing indicator"
    )
    output.loc[output["missing_n"].eq(0) & output["variable"].isin(ml_features), "analysis_handling"] = (
        "No missing values; pipeline rule retained"
    )
    output.loc[output["variable"].eq("cap"), "analysis_handling"] = (
        "Values outside 100-400 dB/m set missing; training-fold median imputation with indicator"
    )
    output.loc[output["variable"].isin(["platelet", "platelet_lag_days"]), "analysis_handling"] = (
        "Excluded from ML to preserve the feature definition; used in FIB-4 analyses"
    )
    return output


def _save_figure(fig: Any, path: Path) -> None:
    """Save both raster and vector versions for the manuscript package."""

    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=320, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")


def _add_panel_label(axis: Any, label: str) -> None:
    axis.text(
        -0.15,
        1.08,
        label,
        transform=axis.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
    )


def _display_table(
    frame: pd.DataFrame,
    columns: list[str],
    labels: list[str],
    title: str,
    path: Path,
) -> None:
    """Render a publication-readable copy of a source CSV without changing it."""

    import matplotlib.pyplot as plt

    raw_display = frame.loc[:, columns].copy().fillna("Not estimable")
    column_count = max(len(columns), 1)
    raw_lengths = np.array(
        [
            max(
                len(str(labels[index])),
                int(raw_display.iloc[:, index].astype(str).str.len().quantile(0.75)),
            )
            for index in range(column_count)
        ],
        dtype=float,
    )
    width_weights = np.clip(raw_lengths, 12, 36)
    width_weights /= width_weights.sum()
    character_budget = max(110, min(230, int(column_count * 25)))
    wrap_widths = np.maximum(10, np.floor(width_weights * character_budget).astype(int))
    display = raw_display.copy()
    for index, column in enumerate(columns):
        display[column] = display[column].map(
            lambda value, width=int(wrap_widths[index]): textwrap.fill(str(value), width=width)
        )
    header_labels = [
        textwrap.fill(str(label), width=int(wrap_widths[index]))
        for index, label in enumerate(labels)
    ]
    row_units = [max(1, label.count("\n") + 1) for label in header_labels]
    row_units.extend(
        max(1, max(cell.count("\n") + 1 for cell in row))
        for row in display.to_numpy(dtype=str)
    )
    line_count = sum(row_units)
    font_size = max(5.5, 8.5 - max(0, column_count - 5) * 0.23)
    fig, axis = plt.subplots(
        figsize=(min(24, max(9, column_count * 2.25)), max(3.2, 1.5 + line_count * 0.33))
    )
    axis.axis("off")
    axis.set_title(title, loc="left", fontweight="bold", pad=14)
    table = axis.table(
        cellText=display.values,
        colLabels=header_labels,
        colWidths=width_weights.tolist(),
        cellLoc="center",
        colLoc="center",
        bbox=[0.0, 0.03, 1.0, 0.82],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    for (row, _), cell in table.get_celld().items():
        cell.set_height(row_units[row] / line_count)
        cell.get_text().set_va("center")
        if row == 0:
            cell.set_facecolor("#376996")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#EDF3F8")
    _save_figure(fig, path)
    plt.close(fig)


def _format_hyperparameters(value: Any) -> str:
    """Convert machine-readable parameter mappings to concise publication text."""

    parameters = json.loads(value) if isinstance(value, str) else dict(value)
    labels = {
        "model__alpha": "alpha",
        "model__max_features": "maximum-feature proportion",
        "model__min_samples_leaf": "minimum leaf size",
        "model__learning_rate": "learning rate",
        "model__max_leaf_nodes": "maximum leaf nodes",
        "model__l2_regularization": "L2 regularization",
    }
    preferred_order = list(labels)

    def _format_value(parameter_value: Any) -> str:
        if isinstance(parameter_value, (float, np.floating)):
            return f"{float(parameter_value):g}"
        return str(parameter_value)

    ordered_keys = [key for key in preferred_order if key in parameters]
    ordered_keys.extend(sorted(key for key in parameters if key not in labels))
    return "; ".join(
        f"{labels.get(key, key.removeprefix('model__').replace('_', ' '))}={_format_value(parameters[key])}"
        for key in ordered_keys
    )


def build_publication_supplementary_table1(frame: pd.DataFrame) -> pd.DataFrame:
    """Create a compact, human-readable nested temporal CV table."""

    required = {
        "outer_fold",
        "model",
        "selected_parameter_json",
        "train_patients",
        "validation_patients",
        "train_start",
        "train_end",
        "validation_start",
        "validation_end",
        "mae",
        "rmse",
        "r2",
        "train_mae",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Supplementary Table 1 source is missing columns: {missing}")

    model_labels = {
        "ridge": "Ridge regression",
        "hist_gradient_boosting": "Histogram gradient boosting",
        "random_forest": "Random forest",
    }
    model_order = {model: index for index, model in enumerate(model_labels)}
    publication = pd.DataFrame(
        {
            "Outer fold": frame["outer_fold"].astype(int),
            "Model": frame["model"].map(model_labels).fillna(
                frame["model"].astype(str).str.replace("_", " ").str.title()
            ),
            "Selected hyperparameters": frame["selected_parameter_json"].map(
                _format_hyperparameters
            ),
            "Training period": frame["train_start"].astype(str)
            + " to "
            + frame["train_end"].astype(str),
            "Validation period": frame["validation_start"].astype(str)
            + " to "
            + frame["validation_end"].astype(str),
            "Training patients": frame["train_patients"].astype(int),
            "Validation patients": frame["validation_patients"].astype(int),
            "Training MAE, kPa": frame["train_mae"].astype(float).round(3),
            "Validation MAE, kPa": frame["mae"].astype(float).round(3),
            "Validation RMSE, kPa": frame["rmse"].astype(float).round(3),
            "Validation R²": frame["r2"].astype(float).round(3),
        }
    )
    publication["_model_order"] = frame["model"].map(model_order).fillna(len(model_order))
    return (
        publication.sort_values(["Outer fold", "_model_order"])
        .drop(columns="_model_order")
        .reset_index(drop=True)
    )


def build_publication_supplementary_table2(frame: pd.DataFrame) -> pd.DataFrame:
    """Create the diagnosis-evidence table without duplicated or audit-only fields."""

    required = {"split", "variable", "patients", "n", "pct"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Supplementary Table 2 source is missing columns: {missing}")

    evidence_labels = {
        "diagnosis_hbv": "Hepatitis B virus",
        "diagnosis_hcv": "Hepatitis C virus",
        "diagnosis_alcohol_related": "Alcohol-related liver disease",
        "diagnosis_fatty_liver": "Fatty liver",
        "diagnosis_autoimmune_or_cholestatic": "Autoimmune or cholestatic disease",
    }
    split_order = ["train", "test", "validation", "all"]
    selected = frame.loc[
        frame["variable"].isin(evidence_labels) & frame["split"].isin(split_order)
    ].copy()
    duplicates = selected.duplicated(["variable", "split"], keep=False)
    if duplicates.any():
        duplicate_keys = selected.loc[duplicates, ["variable", "split"]].to_dict("records")
        raise ValueError(f"Supplementary Table 2 has duplicate evidence rows: {duplicate_keys}")

    expected_keys = {(variable, split) for variable in evidence_labels for split in split_order}
    actual_keys = set(selected[["variable", "split"]].itertuples(index=False, name=None))
    if actual_keys != expected_keys:
        absent = sorted(expected_keys.difference(actual_keys))
        raise ValueError(f"Supplementary Table 2 is missing evidence rows: {absent}")

    patient_counts = (
        selected.groupby("split", sort=False)["patients"].first().astype(int).to_dict()
    )
    split_labels = {
        "train": f"Train set (n={patient_counts['train']})",
        "test": f"Test set (n={patient_counts['test']})",
        "validation": f"Validation set (n={patient_counts['validation']})",
        "all": f"Overall (n={patient_counts['all']})",
    }
    selected["display_value"] = selected.apply(
        lambda row: f"{int(row['n'])} ({float(row['pct']):.1f}%)", axis=1
    )
    wide = selected.pivot(index="variable", columns="split", values="display_value")
    publication = pd.DataFrame(
        {
            "Diagnosis-text evidence": [evidence_labels[variable] for variable in evidence_labels],
            **{
                split_labels[split]: [wide.loc[variable, split] for variable in evidence_labels]
                for split in split_order
            },
        }
    )
    return publication


def _format_ci(estimate: float, lower: float, upper: float, digits: int = 3) -> str:
    if any(pd.isna(value) for value in [estimate, lower, upper]):
        return "Not estimable"
    return f"{estimate:.{digits}f} ({lower:.{digits}f} to {upper:.{digits}f})"


def _format_p_value(value: float) -> str:
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def build_publication_table1(tasks: pd.DataFrame) -> pd.DataFrame:
    """Summarize each characteristic by analysis set and by overall sex."""

    cohort_sizes = tasks.groupby("cohort")["patient_key"].nunique().to_dict()
    cohort_labels = {
        "development_le_2022": f"Train set (n={cohort_sizes['development_le_2022']})",
        "temporal_validation_2023": f"Test set (n={cohort_sizes['temporal_validation_2023']})",
        "temporal_test_2024": f"Validation set (n={cohort_sizes['temporal_test_2024']})",
    }
    sex_counts: dict[str, dict[str, int]] = {}
    contingency: list[list[int]] = []
    for cohort in COHORT_ORDER:
        cohort_frame = tasks.loc[tasks["cohort"].eq(cohort)]
        unknown_sex = ~cohort_frame["sex"].isin(["Male", "Female"])
        if unknown_sex.any():
            raise ValueError(f"Table 1 cannot summarize {int(unknown_sex.sum())} unknown sex values in {cohort}.")
        counts = {sex: int(cohort_frame["sex"].eq(sex).sum()) for sex in ["Male", "Female"]}
        counts["Total"] = int(len(cohort_frame))
        sex_counts[cohort] = counts
        contingency.append([counts["Male"], counts["Female"]])

    from scipy.stats import chi2_contingency, kruskal, mannwhitneyu

    chi2_statistic, sex_p_value, sex_df, expected = chi2_contingency(
        np.asarray(contingency, dtype=int).T
    )
    if float(expected.min()) < 5:
        raise ValueError("Sex chi-square test requires every expected count to be at least 5.")

    overall_sex_counts = {
        sex: int(tasks["sex"].eq(sex).sum()) for sex in ["Male", "Female"]
    }
    overall_total = int(len(tasks))
    male_column = f"Overall male (n={overall_sex_counts['Male']})"
    female_column = f"Overall female (n={overall_sex_counts['Female']})"

    rows: list[dict[str, str]] = [
        {
            "Characteristic": "Sex",
            "Unit": "n (%)",
            **{label: "" for label in cohort_labels.values()},
            male_column: "",
            female_column: "",
            "P across sets": _format_p_value(float(sex_p_value)),
            "P male vs female": "Not applicable",
        }
    ]
    for sex in ["Male", "Female"]:
        row = {"Characteristic": f"  {sex}", "Unit": ""}
        for cohort, column in cohort_labels.items():
            count = sex_counts[cohort][sex]
            row[column] = f"{count} ({100 * count / sex_counts[cohort]['Total']:.1f}%)"
        row[male_column] = (
            f"{overall_sex_counts['Male']} "
            f"({100 * overall_sex_counts['Male'] / overall_total:.1f}%)"
            if sex == "Male"
            else ""
        )
        row[female_column] = (
            f"{overall_sex_counts['Female']} "
            f"({100 * overall_sex_counts['Female'] / overall_total:.1f}%)"
            if sex == "Female"
            else ""
        )
        row["P across sets"] = ""
        row["P male vs female"] = ""
        rows.append(row)

    for variable, label, unit in TABLE1_CHARACTERISTICS:
        row: dict[str, str] = {"Characteristic": label, "Unit": unit}
        groups: list[pd.Series] = []
        for cohort, column in cohort_labels.items():
            values = pd.to_numeric(
                tasks.loc[tasks["cohort"].eq(cohort), variable], errors="coerce"
            )
            if variable in PRESPECIFIED_VALID_RANGES:
                lower, upper = PRESPECIFIED_VALID_RANGES[variable]
                values = values.where(values.between(lower, upper))
            valid = values.dropna()
            groups.append(valid)
            if valid.empty:
                row[column] = "Not available"
            else:
                row[column] = (
                    f"{valid.median():.1f} "
                    f"({valid.quantile(0.25):.1f}-{valid.quantile(0.75):.1f})"
                )
        h_statistic, p_value = kruskal(*groups, nan_policy="omit")
        overall_values = pd.to_numeric(tasks[variable], errors="coerce")
        if variable in PRESPECIFIED_VALID_RANGES:
            lower, upper = PRESPECIFIED_VALID_RANGES[variable]
            overall_values = overall_values.where(overall_values.between(lower, upper))
        male_values = overall_values.loc[tasks["sex"].eq("Male")].dropna()
        female_values = overall_values.loc[tasks["sex"].eq("Female")].dropna()
        u_statistic, sex_p_value_row = mannwhitneyu(
            male_values,
            female_values,
            alternative="two-sided",
            method="asymptotic",
        )
        row[male_column] = (
            f"{male_values.median():.1f} "
            f"({male_values.quantile(0.25):.1f}-{male_values.quantile(0.75):.1f})"
        )
        row[female_column] = (
            f"{female_values.median():.1f} "
            f"({female_values.quantile(0.25):.1f}-{female_values.quantile(0.75):.1f})"
        )
        row["P across sets"] = _format_p_value(float(p_value))
        row["P male vs female"] = _format_p_value(float(sex_p_value_row))
        rows.append(row)

    output = pd.DataFrame(rows)
    output.attrs["sex_chi2"] = float(chi2_statistic)
    output.attrs["sex_df"] = int(sex_df)
    output.attrs["sex_p_value"] = float(sex_p_value)
    output.attrs["sex_min_expected"] = float(expected.min())
    output.attrs["continuous_test"] = "Kruskal-Wallis H test"
    output.attrs["overall_sex_test"] = "Mann-Whitney U test"
    return output


def write_publication_table1_excel(frame: pd.DataFrame, path: Path) -> None:
    """Write a styled, print-ready Excel version of manuscript Table 1."""

    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Table 1"
    sheet.sheet_view.showGridLines = False
    sheet.freeze_panes = "C5"

    columns = frame.columns.tolist()
    if len(columns) != 9:
        raise ValueError(
            "Table 1 Excel layout requires three set columns, two overall sex columns, "
            "and two P-value columns."
        )
    last_column = get_column_letter(len(columns))
    sheet.merge_cells(f"A1:{last_column}1")
    sheet["A1"] = "Table 1. Analysis-set characteristics and overall sex comparisons"
    sheet["A1"].font = Font(name="Arial", size=14, bold=True, color="FFFFFF")
    sheet["A1"].fill = PatternFill("solid", fgColor="1F4E78")
    sheet["A1"].alignment = Alignment(horizontal="left", vertical="center")
    sheet.row_dimensions[1].height = 28

    sheet.merge_cells(f"A2:{last_column}2")
    sheet["A2"] = (
        "Values are n (%) or median (interquartile range); P values compare either the "
        "three analysis sets or pooled males and females."
    )
    sheet["A2"].font = Font(name="Arial", size=9, italic=True, color="44546A")
    sheet["A2"].alignment = Alignment(wrap_text=True, vertical="center")
    sheet.row_dimensions[2].height = 26

    header_row = 4
    thin_side = Side(style="thin", color="B4C6E7")
    medium_side = Side(style="medium", color="1F4E78")
    for column_index, label in enumerate(columns, start=1):
        cell = sheet.cell(row=header_row, column=column_index, value=label)
        cell.font = Font(name="Arial", size=10, bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="4472C4")
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = Border(top=medium_side, bottom=medium_side)
    sheet.row_dimensions[header_row].height = 34

    first_data_row = header_row + 1
    for row_offset, values in enumerate(frame.itertuples(index=False, name=None)):
        excel_row = first_data_row + row_offset
        characteristic = str(values[0])
        is_group_header = characteristic == "Sex"
        is_sex_level = characteristic.strip() in {"Male", "Female"}
        for column_index, value in enumerate(values, start=1):
            cell = sheet.cell(row=excel_row, column=column_index, value=value)
            cell.font = Font(name="Arial", size=9.5, bold=is_group_header, color="1F1F1F")
            cell.alignment = Alignment(
                horizontal="left" if column_index == 1 else "center",
                vertical="center",
                wrap_text=True,
                indent=1 if column_index == 1 and is_sex_level else 0,
            )
            cell.border = Border(bottom=thin_side)
            if is_group_header:
                cell.fill = PatternFill("solid", fgColor="D9EAF7")
            elif not is_sex_level and row_offset % 2 == 1:
                cell.fill = PatternFill("solid", fgColor="F7FAFC")
        sheet.row_dimensions[excel_row].height = 29

    last_data_row = first_data_row + len(frame) - 1
    for cell in sheet[last_data_row]:
        cell.border = Border(bottom=medium_side)

    widths = [31, 11, 19, 19, 18, 19, 19, 13, 16]
    for column_index, width in enumerate(widths, start=1):
        sheet.column_dimensions[get_column_letter(column_index)].width = width

    note_row = last_data_row + 2
    notes = [
        "Train set: index year <=2022; test set: 2023; validation set: 2024.",
        (
            "P across sets: sex distribution was compared with the Pearson chi-square test "
            f"(χ²={frame.attrs['sex_chi2']:.3f}, df={frame.attrs['sex_df']}; "
            f"minimum expected count={frame.attrs['sex_min_expected']:.2f})."
        ),
        (
            "P across sets: continuous characteristics were compared with two-sided "
            "Kruskal-Wallis H tests (df=2)."
        ),
        (
            "P male vs female: all 715 patients were pooled and continuous characteristics "
            "were compared with two-sided Mann-Whitney U tests; these tests were not adjusted "
            "for analysis set. All P values are unadjusted."
        ),
    ]
    for offset, note in enumerate(notes):
        row = note_row + offset
        sheet.merge_cells(start_row=row, start_column=1, end_row=row, end_column=len(columns))
        cell = sheet.cell(row=row, column=1, value=note)
        cell.font = Font(name="Arial", size=9, italic=True, color="44546A")
        cell.alignment = Alignment(wrap_text=True, vertical="top")
        sheet.row_dimensions[row].height = 28

    sheet.auto_filter.ref = f"A{header_row}:{last_column}{last_data_row}"
    sheet.print_title_rows = f"1:{header_row}"
    sheet.print_area = f"A1:{last_column}{note_row + len(notes) - 1}"
    sheet.page_setup.orientation = "landscape"
    sheet.page_setup.paperSize = sheet.PAPERSIZE_A4
    sheet.page_setup.fitToWidth = 1
    sheet.page_setup.fitToHeight = 1
    sheet.sheet_properties.pageSetUpPr.fitToPage = True
    sheet.page_margins.left = 0.25
    sheet.page_margins.right = 0.25
    sheet.page_margins.top = 0.5
    sheet.page_margins.bottom = 0.5
    sheet.oddFooter.center.text = "Table 1 | Page &P of &N"
    sheet.oddFooter.center.size = 8
    sheet.oddFooter.center.color = "808080"

    workbook.properties.title = "Table 1. Analysis-set characteristics and overall sex comparisons"
    workbook.properties.subject = "Third-analysis manuscript table"
    workbook.properties.creator = "Third-analysis reporting pipeline"
    workbook.save(path)


def build_publication_table2(table2: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in table2.iterrows():
        rows.append(
            {
                "Method": row["method_label"],
                "MAE, kPa (95% CI)": _format_ci(row["mae_estimate"], row["mae_ci_2_5"], row["mae_ci_97_5"]),
                "RMSE, kPa (95% CI)": _format_ci(row["rmse_estimate"], row["rmse_ci_2_5"], row["rmse_ci_97_5"]),
                "R2 (95% CI)": _format_ci(row["r2_estimate"], row["r2_ci_2_5"], row["r2_ci_97_5"]),
                "Baseline minus ML MAE, kPa (95% CI)": _format_ci(
                    row["baseline_minus_ml_mae_gain"], row["gain_ci_2_5"], row["gain_ci_97_5"]
                )
                if row["method"] != "locked_ml_model"
                else "Reference ML model",
            }
        )
    return pd.DataFrame(rows)


def build_publication_table_subgroups(subgroup: pd.DataFrame) -> pd.DataFrame:
    definition_labels = {
        "history_group": "Historical LSM records",
        "index_lsm_group": "Index LSM range (kPa; numerical only)",
        "horizon_group": "Target horizon (days)",
    }
    rows = []
    for _, row in subgroup.iterrows():
        rows.append(
            {
                "Subgroup definition": definition_labels[str(row["subgroup_definition"])],
                "Subgroup": row["subgroup"],
                "Patients": int(row["patients"]),
                "Tasks": int(row["tasks"]),
                "MAE, kPa (95% CI)": _format_ci(row["mae_estimate"], row["mae_ci_2_5"], row["mae_ci_97_5"]),
                "RMSE, kPa (95% CI)": _format_ci(row["rmse_estimate"], row["rmse_ci_2_5"], row["rmse_ci_97_5"]),
                "R2 (95% CI)": _format_ci(row["r2_estimate"], row["r2_ci_2_5"], row["r2_ci_97_5"]),
                "Interpretation": row["r2_interpretation"],
            }
        )
    return pd.DataFrame(rows)


def write_publication_tables(
    article_dir: Path,
    metadata: dict[str, Any],
    tasks: pd.DataFrame,
    table2: pd.DataFrame,
    subgroup: pd.DataFrame,
    input_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Write clean article tables while preserving auditable source tables in result/."""

    configure_matplotlib()
    table_dir = article_dir / "表格"
    table_dir.mkdir(parents=True, exist_ok=True)
    active_stems = {
        stem
        for stem, _ in [*PUBLICATION_MAIN_TABLES]
    } | {
        stem
        for stem, _, _ in PUBLICATION_SUPPLEMENTARY_TABLES
    }
    for path in table_dir.glob("*.csv"):
        if path.stem not in active_stems:
            path.unlink()
    for path in table_dir.glob("*.png"):
        if path.stem not in active_stems:
            path.unlink()
    for path in table_dir.glob("*.pdf"):
        if path.stem not in active_stems:
            path.unlink()
    for path in table_dir.glob("*.xlsx"):
        if path.stem != "Table1":
            path.unlink()
    publication_tables = {
        "Table1": build_publication_table1(tasks),
        "Table2": build_publication_table2(table2),
        "Table3": build_publication_table_subgroups(subgroup),
    }
    titles = dict(PUBLICATION_MAIN_TABLES)
    for name, frame in publication_tables.items():
        write_csv(frame, table_dir / f"{name}.csv")
        _display_table(frame, frame.columns.tolist(), frame.columns.tolist(), titles[name], table_dir / name)
        if name == "Table1":
            write_publication_table1_excel(frame, table_dir / "Table1.xlsx")

    for stem, title, source in PUBLICATION_SUPPLEMENTARY_TABLES:
        source_path = input_dir / source
        destination = table_dir / f"{stem}.csv"
        source_frame = pd.read_csv(source_path)
        if stem == "SupplementaryTable1_nested_temporal_cv":
            publication_frame = build_publication_supplementary_table1(source_frame)
        elif stem == "SupplementaryTable2_diagnosis_evidence":
            publication_frame = build_publication_supplementary_table2(source_frame)
        else:
            raise ValueError(f"No publication-table cleaning rule is defined for {stem}.")
        write_csv(publication_frame, destination)
        _display_table(
            publication_frame,
            publication_frame.columns.tolist(),
            publication_frame.columns.tolist(),
            title,
            table_dir / stem,
        )
    return publication_tables


def write_article_asset_index(
    article_dir: Path,
    metadata: dict[str, Any],
    publication_tables: dict[str, pd.DataFrame],
) -> None:
    """Document the one-to-one correspondence between manuscript citations and assets."""

    visit = metadata["visit_audit"]
    task = metadata["task_audit"]
    lines = [
        "# 图表与正文对应清单",
        "",
        "本清单是第三次分析投稿包的唯一图表索引。所有主图均在正文中逐一引用；PNG 用于编辑和预览，PDF 用于可缩放投稿输出。数值由锁定分析结果生成，不应手工替换。",
        "",
        "## 主图",
        "",
        "| 正文引用 | 文件 | 正文位置 | 内容 |",
        "|---|---|---|---|",
        f"| Figure 1 | `图片/Fig1.png` / `图片/Fig1.pdf` | Results: Characteristics of the Analysis Sets and Overall Sex Comparisons | 三面板汇总从 {visit['usable_lsm_patients']:,} 名有可用 LSM 患者到 {task['main_task_patients']:,} 名主分析患者、训练集/验证集/测试集覆盖及主要变量缺失率。 |",
        "| Figure 2 | `图片/Fig2.png` / `图片/Fig2.pdf` | Results: Candidate-Model Selection and Performance in the Test and Validation Sets | 三面板汇总三个候选模型的逐折表现、平均 MAE（SD）、最终模型选择和锁定随机森林的训练/下一时间块验证误差。 |",
        "| Figure 3 | `图片/Fig3.png` / `图片/Fig3.pdf` | Results: Candidate-Model Selection and Performance in the Test and Validation Sets | 四面板汇总测试集的基线比较、配对增益、连续校准散点和残差分布。 |",
        "| Figure 4 | `图片/Fig4.png` / `图片/Fig4.pdf` | Results: Feature-level Ablation and Incremental Value Analysis | 两面板显示 21 个特征在 2023 年测试集和 2024 年验证集中的置换重要性与逐特征留一消融，展示相关纵向 LSM 摘要之间的共享预测信息及单特征移除后的性能韧性。 |",
        "| Figure 5 | `图片/Fig5.png` / `图片/Fig5.pdf` | Results: Auxiliary, Subgroup, and Sensitivity Analyses | 三面板汇总预设亚组、FIB-4 完整病例比较和大幅变化敏感性分析。 |",
        "",
        "## 补充图",
        "",
        "| 正文引用 | 文件 | 正文位置 | 内容 |",
        "|---|---|---|---|",
        "| Supplementary Figure 1 | `图片/SupFig1.png` / `图片/SupFig1.pdf` | Results: Exploratory Trajectory Analysis | 两面板显示探索性聚类的 k 选择与 bootstrap 稳定性；不代表病理分期。 |",
        "",
        "## 主表",
        "",
        "| 正文引用 | 可投稿表格文件 | 锁定来源 |",
        "|---|---|---|",
            *[
                f"| {label.split('.', 1)[0]} | `表格/{stem}.csv`, "
                f"{'`表格/Table1.xlsx`, ' if stem == 'Table1' else ''}"
                f"`表格/{stem}.png`, `表格/{stem}.pdf` | `result/{source}` |"
                for (stem, label), source in zip(
                    PUBLICATION_MAIN_TABLES,
                    [
                        "phase1_primary_tasks_one_per_patient.csv",
                        "table2_test_model_and_baselines.csv",
                        "table4_test_subgroup_performance.csv",
                    ],
                )
            ],
        "",
        "## 补充表",
        "",
        "| 正文引用 | 可投稿表格文件 | 锁定来源 |",
        "|---|---|---|",
        *[
            f"| {label.split('.', 1)[0]} | `表格/{stem}.csv`, `表格/{stem}.png`, `表格/{stem}.pdf` | `result/{source}` |"
            for stem, label, source in PUBLICATION_SUPPLEMENTARY_TABLES
        ],
        "",
        "补充表 CSV、PNG 和 PDF 均为同一投稿阅读版：参数转为纯文本，删除 JSON、内部 cohort 键、溯源字段及重复或未用列。",
        "完整机器审计源数据仍保留在 `result/`，仅用于复现和数值核对；数据流、分类门控、聚类索引和敏感性索引等不作为投稿表。",
        "",
        "## 文件完整性",
        "",
        f"- 主表行数：Table 1 = {len(publication_tables['Table1'])}, Table 2 = {len(publication_tables['Table2'])}, Table 3 = {len(publication_tables['Table3'])}。",
        f"- 图：5 张主组合图 + 1 张补充组合图；表：3 张主表 + {len(PUBLICATION_SUPPLEMENTARY_TABLES)} 张补充表。",
        "- 图表数字和结论均以 `result/release_manifest.json` 中记录的锁定工作簿哈希为准。",
    ]
    (article_dir / "图表对应清单.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def create_figures(
    metadata: dict[str, Any],
    cohort_summary: pd.DataFrame,
    table2: pd.DataFrame,
    predictions: pd.DataFrame,
    subgroup: pd.DataFrame,
    outer_cv: pd.DataFrame,
    missingness: pd.DataFrame,
    cluster_metrics: pd.DataFrame,
    cluster_stability: pd.DataFrame,
    jump_sensitivity: pd.DataFrame,
    feasibility: pd.DataFrame,
    fib4_benchmark: pd.DataFrame,
    calibration: pd.DataFrame,
    selected_model: str,
    output_dir: Path,
    article_figure_dir: Path,
) -> None:
    """Create the analysis figures and the numbered manuscript figure package."""

    configure_matplotlib()
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    article_figure_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
        }
    )
    labels = ["Train", "Test", "Validation"]
    colors = ["#376996", "#5B8E7D", "#D17A22"]
    cohort_index = cohort_summary.set_index("cohort").loc[COHORT_ORDER]
    test = predictions.loc[predictions["evaluation_set"].eq(PUBLIC_TEST_COHORT)].copy()
    ordered = table2.iloc[[0, 1, 3, 2, 4]].copy()

    # Remove superseded numbered assets so the manuscript directory remains a
    # five-main-figure plus one-supplement package after every rerun.
    active_article_stems = {
        stem
        for stem, _, _ in [*PUBLICATION_MAIN_FIGURES, *PUBLICATION_SUPPLEMENTARY_FIGURES]
    }
    for path in article_figure_dir.glob("*.png"):
        if path.stem.startswith(("Fig", "SupFig")) and path.stem not in active_article_stems:
            path.unlink()
    for path in article_figure_dir.glob("*.pdf"):
        if path.stem.startswith(("Fig", "SupFig")) and path.stem not in active_article_stems:
            path.unlink()
    # Remove superseded descriptive result names as well; otherwise a rerun
    # would leave stale Figure 4/Supplementary Figure 2 assets in result/ and
    # accidentally include them in the release manifest.
    for stale_name in [
        "figure4_subgroups_and_robustness.png",
        "publication_figure4_subgroups_and_robustness.png",
        "publication_figure4_subgroups_and_robustness.pdf",
        "publication_supplementary_figure2_feature_level_importance.png",
        "publication_supplementary_figure2_feature_level_importance.pdf",
    ]:
        stale_path = output_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()
    active_result_stems = {
        stem
        for _, stem, _ in [*PUBLICATION_MAIN_FIGURES, *PUBLICATION_SUPPLEMENTARY_FIGURES]
    }
    for path in output_dir.glob("publication_figure*.*"):
        if path.stem not in active_result_stems:
            path.unlink()
    for path in output_dir.glob("publication_supplementary_figure*.*"):
        if path.stem not in active_result_stems:
            path.unlink()
    for stale_name in [
        "figure2_test_performance_and_scatter.png",
        "figure5_subgroup_mae_forest.png",
        "supplementary_figure1_training_validation_curve.png",
    ]:
        stale_path = output_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()

    # Figure 1: cohort construction, temporal separation, and data quality.
    fig = plt.figure(figsize=(13.6, 9.2))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], width_ratios=[0.85, 1.45])
    ax_flow = fig.add_subplot(grid[0, 0])
    ax_time = fig.add_subplot(grid[0, 1])
    ax_missing = fig.add_subplot(grid[1, :])
    flow_rows = [
        ("Workbook rows", f"{metadata['visit_audit']['raw_rows']:,}"),
        (
            "Usable dated LSM\nrows / patients",
            f"{metadata['visit_audit']['usable_lsm_rows']:,} / "
            f"{metadata['visit_audit']['usable_lsm_patients']:,}",
        ),
        (
            "Fixed-horizon candidate\npatients",
            f"{metadata['task_audit']['all_fixed_horizon_candidate_patients']:,}",
        ),
        (
            "Primary one-task\npatients",
            f"{metadata['task_audit']['main_task_patients']:,}",
        ),
    ]
    ax_flow.axis("off")
    y_positions = np.linspace(0.88, 0.12, len(flow_rows))
    for index, ((label, count), y) in enumerate(zip(flow_rows, y_positions)):
        ax_flow.text(
            0.5,
            y,
            f"{label}\n{count}",
            ha="center",
            va="center",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "#EAF1F7", "edgecolor": "#376996"},
        )
        if index < len(flow_rows) - 1:
            ax_flow.annotate(
                "",
                xy=(0.5, y_positions[index + 1] + 0.075),
                xytext=(0.5, y - 0.075),
                arrowprops={"arrowstyle": "->", "lw": 1.4, "color": "#4B5563"},
            )
    ax_flow.set_title("Cohort construction")
    for index, (cohort, row) in enumerate(cohort_index.iterrows()):
        start = pd.Timestamp(row["index_date_min"])
        end = pd.Timestamp(row["target_date_max"])
        ax_time.plot([start, end], [index, index], linewidth=11, solid_capstyle="butt", color=colors[index])
        ax_time.scatter(pd.Timestamp(row["index_date_max"]), index, marker="|", s=180, color="#202020", zorder=3)
        ax_time.text(end + pd.Timedelta(days=85), index, f"n={int(row['patients'])}", va="center")
    ax_time.set_yticks(range(3), labels)
    ax_time.xaxis.set_major_locator(mdates.YearLocator())
    ax_time.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_time.set_xlabel("Index-to-outcome calendar coverage")
    ax_time.set_title("Patient-level analysis sets")

    missing = missingness.loc[
        missingness["variable"].isin(["age", "cap", "alt", "ast", "total_bilirubin", "platelet"])
    ].copy()
    variable_labels = {
        "age": "Age",
        "cap": "CAP",
        "alt": "ALT",
        "ast": "AST",
        "total_bilirubin": "Total bilirubin",
        "platelet": "Platelet",
    }
    missing["label"] = missing["variable"].map(variable_labels)
    pivot = (
        missing.pivot(index="label", columns="cohort", values="missing_pct")
        .reindex(index=list(variable_labels.values()), columns=COHORT_ORDER)
    )
    image = ax_missing.imshow(pivot.to_numpy(), aspect="auto", cmap="Blues", vmin=0, vmax=100)
    for row_index in range(pivot.shape[0]):
        for column_index in range(pivot.shape[1]):
            value = float(pivot.iloc[row_index, column_index])
            ax_missing.text(
                column_index,
                row_index,
                f"{value:.1f}%",
                ha="center",
                va="center",
                color="white" if value >= 55 else "#202020",
                fontsize=9,
            )
    ax_missing.set_xticks(range(3), ["Train\n<=2022", "Test\n2023", "Validation\n2024"])
    ax_missing.set_yticks(range(len(pivot.index)), pivot.index)
    ax_missing.set_title("Raw predictor missingness by analysis set")
    colorbar = fig.colorbar(image, ax=ax_missing, fraction=0.025, pad=0.02)
    colorbar.set_label("Missing values (%)")
    _add_panel_label(ax_flow, "A")
    _add_panel_label(ax_time, "B")
    _add_panel_label(ax_missing, "C")
    _save_figure(fig, article_figure_dir / "Fig1")
    shutil.copy2(article_figure_dir / "Fig1.png", output_dir / "figure1_cohort_flow_and_timeline.png")
    plt.close(fig)

    # Figure 2: candidate comparison, selection, and learning behavior.
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.2))
    model_order = ["ridge", "hist_gradient_boosting", "random_forest"]
    model_labels = ["Ridge", "Histogram\ngradient boosting", "Random forest"]
    for model, label, color in zip(model_order, model_labels, ["#9CA3AF", "#5B8E7D", "#376996"]):
        rows = outer_cv.loc[outer_cv["model"].eq(model)].sort_values("outer_fold")
        axes[0].plot(rows["outer_fold"], rows["mae"], marker="o", label=label, color=color)
    axes[0].set_xticks([1, 2, 3, 4])
    axes[0].set_xlabel("Expanding outer temporal fold")
    axes[0].set_ylabel("Validation MAE (kPa)")
    axes[0].set_title("Nested outer-fold model comparison")
    axes[0].legend(frameon=False, fontsize=8)
    selection = (
        outer_cv.groupby("model", as_index=False)
        .agg(mean_mae=("mae", "mean"), sd_mae=("mae", "std"))
        .set_index("model")
        .loc[model_order]
        .reset_index()
    )
    selection_bars = axes[1].bar(
        model_labels,
        selection["mean_mae"],
        yerr=selection["sd_mae"],
        capsize=5,
        color=["#9CA3AF", "#5B8E7D", "#376996"],
    )
    axes[1].set_ylabel("Mean outer-fold MAE (kPa)")
    axes[1].set_title("Outer-fold MAE used for model selection")
    upper_selection_limit = float((selection["mean_mae"] + selection["sd_mae"]).max()) * 1.22
    axes[1].set_ylim(0, upper_selection_limit)
    for bar, mean_mae, sd_mae in zip(
        selection_bars,
        selection["mean_mae"],
        selection["sd_mae"],
    ):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            mean_mae + sd_mae + upper_selection_limit * 0.025,
            f"{mean_mae:.3f} +/- {sd_mae:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    selected_index = model_order.index(selected_model)
    axes[1].text(
        selected_index,
        float(selection.loc[selected_index, "mean_mae"]) * 0.50,
        "Selected",
        color="white",
        weight="bold",
        ha="center",
        va="center",
        fontsize=9,
    )
    learning = outer_cv.loc[outer_cv["model"].eq(selected_model)].sort_values("outer_fold")
    axes[2].plot(learning["train_tasks"], learning["train_mae"], marker="o", label="Training MAE", color="#376996")
    axes[2].plot(learning["train_tasks"], learning["mae"], marker="s", label="Next-block validation MAE", color="#D17A22")
    for _, row in learning.iterrows():
        axes[2].annotate(
            f"Fold {int(row['outer_fold'])}",
            (row["train_tasks"], row["mae"]),
            xytext=(4, 5),
            textcoords="offset points",
            fontsize=8,
        )
    axes[2].set_xlabel("Expanding training tasks")
    axes[2].set_ylabel("MAE (kPa)")
    axes[2].set_title("Selected-model learning curve")
    axes[2].legend(frameon=False, fontsize=8)
    _add_panel_label(axes[0], "A")
    _add_panel_label(axes[1], "B")
    _add_panel_label(axes[2], "C")
    _save_figure(fig, article_figure_dir / "Fig2")
    plt.close(fig)

    # Figure 3: the primary benchmark result and continuous-prediction diagnostics.
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 10.2), gridspec_kw={"width_ratios": [1.12, 1.0]})
    y = np.arange(len(ordered))
    axes[0, 0].errorbar(
        ordered["mae_estimate"],
        y,
        xerr=[ordered["mae_estimate"] - ordered["mae_ci_2_5"], ordered["mae_ci_97_5"] - ordered["mae_estimate"]],
        fmt="o",
        color="#376996",
        ecolor="#6C757D",
        capsize=4,
    )
    axes[0, 0].set_yticks(y, ordered["method_label"])
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_xlabel("MAE (kPa), patient-bootstrap 95% CI")
    axes[0, 0].set_title("Test-set model and simple baselines")
    last_row = ordered.loc[ordered["method"].eq("last_observed_lsm")].iloc[0]
    gain = float(last_row["baseline_minus_ml_mae_gain"])
    gain_low = float(last_row["gain_ci_2_5"])
    gain_high = float(last_row["gain_ci_97_5"])
    gain_limit = max(abs(gain_low), abs(gain_high), 1.0) * 1.25
    axes[0, 1].axvspan(-gain_limit, 0, color="#D17A22", alpha=0.08)
    axes[0, 1].axvspan(0, gain_limit, color="#376996", alpha=0.08)
    axes[0, 1].axvline(0, color="#374151", linewidth=1)
    axes[0, 1].errorbar(
        gain,
        0,
        xerr=[[gain - gain_low], [gain_high - gain]],
        fmt="o",
        color="#A23B3B",
        capsize=5,
    )
    axes[0, 1].set_xlim(-gain_limit, gain_limit)
    axes[0, 1].set_ylim(-0.65, 0.65)
    axes[0, 1].set_yticks([0], ["Last observed LSM vs ML"])
    axes[0, 1].set_xlabel("Baseline minus ML MAE (kPa)")
    axes[0, 1].set_title("Paired incremental-value test")
    axes[0, 1].text(0.02, 0.94, "Favors last LSM", transform=axes[0, 1].transAxes, fontsize=8, va="top")
    axes[0, 1].text(0.98, 0.94, "Favors ML", transform=axes[0, 1].transAxes, fontsize=8, ha="right", va="top")
    axes[0, 1].text(
        0.02,
        0.08,
        f"Difference = {gain:.3f} kPa\n95% CI {gain_low:.3f} to {gain_high:.3f}",
        transform=axes[0, 1].transAxes,
        fontsize=9,
        va="bottom",
    )

    axes[1, 0].scatter(test["locked_ml_model"], test["target_lsm"], alpha=0.72, color="#5B8E7D", edgecolor="white", linewidth=0.35)
    lower = min(test["target_lsm"].min(), test["locked_ml_model"].min())
    upper = max(test["target_lsm"].max(), test["locked_ml_model"].max())
    calibration_rows = calibration.loc[
        calibration["evaluation_set"].eq(PUBLIC_TEST_COHORT)
        & calibration["method"].eq("locked_ml_model")
    ].set_index("calibration_metric")
    calibration_intercept = float(calibration_rows.loc["calibration_intercept", "estimate"])
    calibration_slope = float(calibration_rows.loc["calibration_slope", "estimate"])
    line_x = np.linspace(lower, upper, 100)
    axes[1, 0].plot([lower, upper], [lower, upper], linestyle="--", color="#A23B3B", label="Identity")
    axes[1, 0].plot(
        line_x,
        calibration_intercept + calibration_slope * line_x,
        color="#376996",
        label=f"Calibration: {calibration_intercept:.2f} + {calibration_slope:.2f}x",
    )
    axes[1, 0].set_xlabel("Random-forest prediction (kPa)")
    axes[1, 0].set_ylabel("Observed future LSM (kPa)")
    axes[1, 0].set_title("Observed versus predicted and calibration")
    axes[1, 0].legend(frameon=False, fontsize=8)
    residual = test["locked_ml_model"] - test["target_lsm"]
    axes[1, 1].hist(residual, bins=18, color="#376996", edgecolor="white")
    axes[1, 1].axvline(0, color="#A23B3B", linestyle="--")
    axes[1, 1].set_xlabel("Prediction residual: predicted minus observed (kPa)")
    axes[1, 1].set_ylabel("Patients")
    axes[1, 1].set_title("Test-set residual distribution")
    for axis, label in zip(axes.flat, ["A", "B", "C", "D"]):
        _add_panel_label(axis, label)
    _save_figure(fig, article_figure_dir / "Fig3")
    plt.close(fig)

    # Figure 5: subgroup performance and robustness analyses.
    forest = subgroup.loc[subgroup["patients"].gt(0)].copy()
    definition_labels = {"history_group": "Historical LSM records", "index_lsm_group": "Index LSM (kPa)", "horizon_group": "Target horizon (days)"}
    forest["label"] = forest.apply(lambda row: f"{definition_labels[row['subgroup_definition']]}: {row['subgroup']}", axis=1)
    forest = forest.iloc[::-1].reset_index(drop=True)
    fig = plt.figure(figsize=(14.4, 9.4))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], hspace=0.38)
    ax_forest = fig.add_subplot(grid[:, 0])
    ax_fib4 = fig.add_subplot(grid[0, 1])
    ax_sensitivity = fig.add_subplot(grid[1, 1])
    y = np.arange(len(forest))
    ax_forest.errorbar(
        forest["mae_estimate"],
        y,
        xerr=[forest["mae_estimate"] - forest["mae_ci_2_5"], forest["mae_ci_97_5"] - forest["mae_estimate"]],
        fmt="o",
        color="#376996",
        ecolor="#6C757D",
        capsize=3,
    )
    ax_forest.set_yticks(y, [f"{label} (n={n})" for label, n in zip(forest["label"], forest["patients"])])
    ax_forest.set_xlabel("Locked-model MAE (kPa), patient-bootstrap 95% CI")
    ax_forest.set_title("Prespecified numerical subgroups")
    ax_forest.text(
        0.02,
        0.01,
        "Groups with n<20 are descriptive; LSM bins are not fibrosis stages.",
        transform=ax_forest.transAxes,
        fontsize=8,
        va="bottom",
    )

    fib4_order = ["locked_ml_model", "last_observed_lsm", "fib4_development_linear_mapping"]
    fib4_labels = ["Locked random forest", "Last observed LSM", "FIB-4 linear mapping"]
    fib4_rows = (
        fib4_benchmark.loc[
            fib4_benchmark["evaluation_set"].eq(f"{PUBLIC_TEST_COHORT}_fib4_90day_complete_case")
            & fib4_benchmark["metric"].eq("mae")
        ]
        .set_index("method")
        .loc[fib4_order]
        .reset_index()
    )
    fib4_y = np.arange(len(fib4_rows))
    ax_fib4.errorbar(
        fib4_rows["estimate"],
        fib4_y,
        xerr=[fib4_rows["estimate"] - fib4_rows["ci_2_5"], fib4_rows["ci_97_5"] - fib4_rows["estimate"]],
        fmt="o",
        color="#5B8E7D",
        ecolor="#6C757D",
        capsize=4,
    )
    fib4_n = int(
        feasibility.loc[
            feasibility["cohort"].eq(PUBLIC_TEST_COHORT)
            & feasibility["lookback_window_days"].eq(90),
            "age_alt_ast_platelet_complete_n",
        ].iloc[0]
    )
    ax_fib4.set_yticks(fib4_y, fib4_labels)
    ax_fib4.invert_yaxis()
    ax_fib4.set_xlabel("MAE (kPa), bootstrap 95% CI")
    ax_fib4.set_title(f"90-day FIB-4 complete cases (n={fib4_n})")

    sensitivity = jump_sensitivity.loc[
        jump_sensitivity["method"].isin(["locked_ml_model", "last_observed_lsm"])
    ].copy()
    sensitivity_order = ["all_2023_test_patients", "exclude_absolute_index_to_target_change_ge_10_kpa"]
    method_colors = {"locked_ml_model": "#376996", "last_observed_lsm": "#D17A22"}
    method_labels = {"locked_ml_model": "Locked random forest", "last_observed_lsm": "Last observed LSM"}
    y_positions = {name: index for index, name in enumerate(sensitivity_order)}
    offsets = {"locked_ml_model": -0.13, "last_observed_lsm": 0.13}
    for method in method_labels:
        rows = sensitivity.loc[sensitivity["method"].eq(method)]
        y_values = [y_positions[name] + offsets[method] for name in rows["analysis_set"]]
        ax_sensitivity.errorbar(
            rows["mae_estimate"],
            y_values,
            xerr=[rows["mae_estimate"] - rows["mae_ci_2_5"], rows["mae_ci_97_5"] - rows["mae_estimate"]],
            fmt="o",
            color=method_colors[method],
            capsize=4,
            label=method_labels[method],
        )
    ax_sensitivity.set_yticks([0, 1], ["All test patients\n(n=103)", "Exclude >=10-kPa changes\n(n=99)"])
    ax_sensitivity.set_xlabel("MAE (kPa), bootstrap 95% CI")
    ax_sensitivity.set_title("Large-change sensitivity")
    ax_sensitivity.legend(frameon=False, fontsize=8)
    _add_panel_label(ax_forest, "A")
    _add_panel_label(ax_fib4, "B")
    _add_panel_label(ax_sensitivity, "C")
    _save_figure(fig, article_figure_dir / "Fig5")
    shutil.copy2(article_figure_dir / "Fig5.png", output_dir / "figure5_subgroups_and_robustness.png")
    plt.close(fig)

    # Figure 4: feature-level permutation importance and leave-one-feature-out
    # (LOFO) analysis. The renderer consumes the locked phase-11 audit outputs
    # and writes both the compact article asset and its descriptive result copy.
    from phase11_feature_level_figure import render_feature_level_figure

    render_feature_level_figure(output_dir, article_figure_dir)

    # Supplementary Figure 1: exploratory cluster diagnostics only.
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.0))
    axes[0].plot(cluster_metrics["k"], cluster_metrics["silhouette"], marker="o", color="#376996")
    axes[0].set_xlabel("Number of clusters (k)")
    axes[0].set_ylabel("Silhouette score")
    axes[0].set_xticks(cluster_metrics["k"])
    axes[0].set_title("Train-set cluster selection")
    axes[1].hist(cluster_stability["adjusted_rand_index"], bins=15, color="#5B8E7D", edgecolor="white")
    mean_ari = cluster_stability["adjusted_rand_index"].mean()
    axes[1].axvline(mean_ari, linestyle="--", color="#A23B3B", label=f"Mean ARI = {mean_ari:.3f}")
    axes[1].set_xlabel("Bootstrap adjusted Rand index")
    axes[1].set_ylabel("Replicates")
    axes[1].set_title("Exploratory cluster stability")
    axes[1].legend(frameon=False)
    _add_panel_label(axes[0], "A")
    _add_panel_label(axes[1], "B")
    _save_figure(fig, article_figure_dir / "SupFig1")
    plt.close(fig)

    # Keep a descriptive, versioned copy in result/ and a compact submission
    # filename in 文章/图片/. Both point to exactly the same rendered figure.
    for prefix, result_stem, _ in [
        *PUBLICATION_MAIN_FIGURES,
        *PUBLICATION_SUPPLEMENTARY_FIGURES,
    ]:
        for suffix in [".png", ".pdf"]:
            shutil.copy2(
                article_figure_dir / f"{prefix}{suffix}",
                output_dir / f"{result_stem}{suffix}",
            )

def write_result_report(
    output_dir: Path,
    metadata: dict[str, Any],
    cohort_summary: pd.DataFrame,
    performance: pd.DataFrame,
    gains: pd.DataFrame,
    model_lock: dict[str, Any],
    subgroup: pd.DataFrame,
    repeated_subgroup: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    jump_sensitivity: pd.DataFrame,
    feasibility: pd.DataFrame,
    fib4_benchmark: pd.DataFrame,
    calibration: pd.DataFrame,
    context_summary: pd.DataFrame,
) -> None:
    test_set = PUBLIC_TEST_COHORT
    validation_set = PUBLIC_VALIDATION_COHORT
    primary_test_patients = _cohort_patient_count(cohort_summary, test_set)
    development_patients = _cohort_patient_count(cohort_summary, "development_le_2022")
    validation_patients = _cohort_patient_count(cohort_summary, validation_set)
    visit_audit = metadata["visit_audit"]
    task_audit = metadata["task_audit"]
    source_cutoff = metadata.get("source_workbook_update_cutoff", SOURCE_WORKBOOK_UPDATE_CUTOFF)
    latest_lsm_visit = visit_audit.get("latest_usable_lsm_visit_date", "not available")
    ml_mae = _metric_row(performance, test_set, "locked_ml_model", "mae")
    ml_rmse = _metric_row(performance, test_set, "locked_ml_model", "rmse")
    ml_r2 = _metric_row(performance, test_set, "locked_ml_model", "r2")
    last_mae = _metric_row(performance, test_set, "last_observed_lsm", "mae")
    last_gain = gains.loc[
        gains["evaluation_set"].eq(test_set)
        & gains["baseline_method"].eq("last_observed_lsm")
    ].iloc[0]
    incremental_supported = _incremental_value_supported(last_gain)
    ge6_primary = subgroup.loc[
        subgroup["subgroup_definition"].eq("history_group") & subgroup["subgroup"].eq(">=6")
    ].iloc[0]
    ge6_repeated = repeated_subgroup.loc[
        repeated_subgroup["subgroup_definition"].eq("history_group")
        & repeated_subgroup["subgroup"].eq(">=6")
    ].iloc[0]
    clusters = cluster_summary.iloc[0]
    no_jump_ml = jump_sensitivity.loc[
        jump_sensitivity["analysis_set"].str.startswith("exclude_")
        & jump_sensitivity["method"].eq("locked_ml_model")
    ].iloc[0]
    excluded_large_change_patients = int(no_jump_ml["excluded_large_change_patients"])
    test_feasibility = feasibility.loc[
        feasibility["cohort"].eq(test_set)
        & feasibility["lookback_window_days"].eq(90)
    ].iloc[0]
    feasibility_90 = feasibility.loc[feasibility["lookback_window_days"].eq(90)].set_index("cohort")
    fib4_mae = _metric_row(
        fib4_benchmark,
        f"{test_set}_fib4_90day_complete_case",
        "fib4_development_linear_mapping",
        "mae",
    )
    fib4_ml_mae = _metric_row(
        fib4_benchmark,
        f"{test_set}_fib4_90day_complete_case",
        "locked_ml_model",
        "mae",
    )
    fib4_last_mae = _metric_row(
        fib4_benchmark,
        f"{test_set}_fib4_90day_complete_case",
        "last_observed_lsm",
        "mae",
    )
    calibration_slope = calibration.loc[
        calibration["evaluation_set"].eq(test_set)
        & calibration["method"].eq("locked_ml_model")
        & calibration["calibration_metric"].eq("calibration_slope")
    ].iloc[0]
    all_context = context_summary.loc[context_summary["cohort"].eq("all_primary_cohorts")]
    context_counts = all_context.set_index("variable")["n"].astype(int).to_dict()
    report = f"""# 第三次分析结果总览

## 核心结论

本分析依据整改方案，从 `data/去隐私完整版更新8.19.xlsx` 重新构建固定预测窗口和患者级时间队列。对外数据集标签统一为 `train`（训练集）、`test`（测试集）和 `validation`（验证集）；结果表中的旧 `cohort` 值仅作为内部审计键保留。源工作簿修订日期为 {source_cutoff}；实际可用 LSM 访视最晚日期为 {latest_lsm_visit}。这次同源数据修订不构成独立外部验证。2023 年测试集包含 {primary_test_patients} 名患者。训练集嵌套时间验证选择 `{model_lock['selected_model']}`，超参数为 {_format_hyperparameters(model_lock['selected_parameters'])}。

- 锁定模型测试 MAE 为 {_format_interval(ml_mae)} kPa，RMSE 为 {_format_interval(ml_rmse)} kPa，R2 为 {_format_interval(ml_r2)}。
- 最近一次 LSM 的测试 MAE 为 {_format_interval(last_mae)} kPa。
- 最近一次 LSM 减去 ML 的 MAE 差为 {last_gain['mae_gain_baseline_minus_ml']:.3f} kPa（95% CI {last_gain['ci_2_5']:.3f} 至 {last_gain['ci_97_5']:.3f}）；正值且 95% CI 全部大于 0 才支持 ML。当前结果**{('支持' if incremental_supported else '不支持')} ML 具有增量连续预测价值**。
- 锁定模型的连续校准斜率为 {calibration_slope['estimate']:.3f}（95% CI {calibration_slope['ci_2_5']:.3f} 至 {calibration_slope['ci_97_5']:.3f}；理想值 1）。
- 排除 {excluded_large_change_patients} 名 index-to-target 绝对变化 >=10 kPa 的探索性敏感性分析后，ML MAE 为 {no_jump_ml['mae_estimate']:.3f} kPa；该敏感性结果不改变主要基线比较结论。

## 审稿问题对应结果

- 数据流：{visit_audit['raw_rows']:,} 个原始行中有 {visit_audit['rows_with_lsm']:,} 行 LSM；{visit_audit['rows_with_lsm_but_no_chosen_date']:,} 行因无可用日期排除；{visit_audit['usable_lsm_rows']:,} 行可用 LSM 合并为 {visit_audit['visit_rows_after_same_day_median']:,} 个患者-日期访视，涉及 {visit_audit['usable_lsm_patients']:,} 名患者。
- 固定任务：270-450 天窗口产生 {task_audit['all_fixed_horizon_candidates']:,} 个候选任务；主分析为 {task_audit['main_task_patients']:,} 名患者各 1 个最早合格任务，其中训练集 {development_patients} 名、2023 测试集 {primary_test_patients} 名、2024 验证集 {validation_patients} 名。三个数据集患者重叠为 0。
- 高频记录：主测试集中 `>=6` 次历史记录组为 {int(ge6_primary['patients'])} 人，无法估计主结果；重复任务敏感性集中为 {int(ge6_repeated['patients'])} 人、{int(ge6_repeated['tasks'])} 个任务，MAE {ge6_repeated['mae_estimate']:.3f} kPa（95% CI {ge6_repeated['mae_ci_2_5']:.3f} 至 {ge6_repeated['mae_ci_97_5']:.3f}）。旧稿 16.49 kPa 未复现，但该组过小，只能描述。
- 聚类：训练集选择 k={int(clusters['selected_k'])}；100 次 bootstrap 的平均 ARI 为 {clusters['stability_mean_ari']:.3f}（SD {clusters['stability_sd_ari']:.3f}）。仅使用中性 Cluster 1/2，不解释为病理分期或临床风险组。
- 人口学与诊断证据：715名主分析患者中女性 {context_counts.get('sex_female', 0)} 名、男性 {context_counts.get('sex_male', 0)} 名；{context_counts.get('diagnosis_recorded', 0)} 名有诊断文本，{context_counts.get('diagnosis_hbv', 0)} 名有HBV相关证据。诊断标记是全工作簿范围的ever-recorded非互斥证据，可能包含index后记录，仍需临床审核后才能称为最终病因分布。
- 变量单位：8.19工作簿表头已明确CAP为dB/m、ALT/AST为U/L、总胆红素为μmol/L、血小板为10^9/L。CAP超出100-400 dB/m范围的 {visit_audit['cap_source_values_outside_device_range_set_missing']} 个源值已设为缺失。
- FIB-4/APRI：90天窗口下，2023测试集中 {int(test_feasibility['age_alt_ast_platelet_complete_n'])}/{int(test_feasibility['patients'])} 名具备完整FIB-4输入；同一完整病例子集中，训练集拟合的FIB-4线性映射、随机森林和最近一次LSM的MAE分别为 {_format_interval(fib4_mae)}、{_format_interval(fib4_ml_mae)} 和 {_format_interval(fib4_last_mae)} kPa。FIB-4不作为F2金标准；APRI仍因缺少本地AST正常值上限而未计算。
- ALT/AST 异常亚组：本地正常值上限未提供，未假设通用 40 U/L，故未执行。

## 分类门控

F2 终点已撤销：当前数据未提供临床或病理参考标准，因此不得以 LSM 数值定义或推断 F2。相应地，本次未执行 7 或 12.5 kPa 阈值分类、AUROC/AUPRC、校准或决策曲线。连续约 12 个月 LSM 是唯一主终点；不得把数值 LSM 分组称为 F2 或病理分期。

## 结果与表图对应

| 结果模块 | 正文表/图 | result/核心来源 |
|---|---|---|
| 队列构建、时间隔离、变量缺失 | Figure 1; Table 1 | `phase1_primary_tasks_one_per_patient.csv`, `table1_temporal_cohort_characteristics.csv`, `supplementary_table2_variable_missingness.csv` |
| 嵌套时间模型选择 | Figure 2; Supplementary Table 1 | `phase4_nested_outer_temporal_cv.csv`, `phase4_model_selection_summary.csv` |
| 锁定测试性能、基线比较、连续校准 | Figure 3; Table 2 | `table2_test_model_and_baselines.csv`, `phase5_continuous_calibration_bootstrap_ci.csv`, `phase6_ml_vs_baseline_mae_gain.csv` |
| 特征级消融、置换重要性和增量价值 | Figure 4 | `phase11_feature_level_lofo_mae_delta.csv`, `phase11_permutation_importance.csv`, `phase11_feature_level_performance.csv` |
| 亚组、FIB-4辅助比较、敏感性分析 | Figure 5; Table 3 | `table4_test_subgroup_performance.csv`, `phase7_fib4_90day_continuous_benchmark.csv`, `phase8_large_change_sensitivity.csv` |
| 探索性聚类 | Supplementary Figure 1 | `phase8_development_cluster_k_metrics.csv`, `phase8_development_cluster_bootstrap_stability.csv` |
| 诊断证据和病因信息限制 | Supplementary Table 2 | `supplementary_table7_diagnosis_evidence.csv` |
| F2/阈值分类门控 | 正文不设独立表 | `table3_threshold_analysis_gate.csv`, `phase7_clinical_endpoint_gate.json` |

## 特征块消融和增量价值分析（次要分析）

为检验临床/实验室变量是否在纵向 LSM 信息之外提供稳定的连续预测增量价值，新增了预先规定的特征块消融分析。所有模型均复用主分析锁定的随机森林算法和参数（`max_features=0.7`、`min_samples_leaf=3`），不在消融结果上重新调参；缺失值处理和范围清理均在训练集内拟合。固定任务仍为每名患者最早合格任务，训练集（434人）用于拟合，2023 年测试集（178人）和 2024 年验证集（103人）仅用于时间外评估。MAE 增量定义为“参考方法 MAE − 比较方法 MAE”，正值表示比较方法更优；置信区间为患者级 1000 次 bootstrap。`full_rf` 的锁定复现检查与主模型 715 行预测的最大绝对差为 `1.8×10⁻¹⁴`。

2023 年测试集的 MAE（kPa）如下：最近一次 LSM 基线 2.964（95% CI 2.276–3.714），LSM 历史块随机森林 3.577（2.775–4.487），临床/实验室块随机森林 4.471（3.402–5.693），最近一次 LSM 加临床块 3.588（2.714–4.737），全特征随机森林 3.622（2.748–4.660）。相对于最近一次 LSM，全特征模型的 MAE 增量为 −0.658 kPa；相对于 LSM 历史块为 −0.045；相对于最近一次 LSM 加临床块为 −0.034。临床/实验室块单独使用明显较差，但在已有纵向 LSM 信息后未显示稳定、具有临床意义的额外收益。2024 年验证集方向一致：全特征模型 MAE 2.989（2.135–4.130）。因此，该次要分析不支持“加入全部临床/实验室特征即可优于最近一次 LSM”或“机器学习具有稳定增量连续预测价值”的表述；结果更适合用于报告特征块的边际贡献及模型与简单基线的公平比较。

分析脚本和审计输出为：`../phase10_feature_block_ablation.py`、`phase10_feature_block_ablation_performance.csv`、`phase10_feature_block_ablation_mae_gain.csv`、`phase10_feature_block_ablation_development_cv.csv`、`phase10_feature_block_ablation_predictions.csv` 和 `phase10_feature_block_ablation_metadata.json`。

## 特征级消融与置换重要性分析

在特征块分析基础上，进一步对主分析锁定的 21 个特征执行两种互补的特征级分析。第一种是逐特征留一消融（LOFO）：每次删除一个特征，在训练集重新拟合同一锁定随机森林，并计算“删除该特征后的 MAE − 完整模型 MAE”；正值表示删除后性能变差，反映该特征的边际贡献。第二种是置换重要性：固定完整模型，在每个分析队列内按患者置换一个特征，计算置换后的 MAE − 未置换完整模型 MAE；正值表示模型依赖该特征。置换重要性每个特征使用 100 次置换；LOFO 和置换结果均使用 1000 次患者级 bootstrap。未使用 2023 或 2024 队列进行模型拟合或调参。

2023 年测试集和 2024 年验证集的置换重要性均将 `hist_median`、`hist_mean` 和 `last_lsm` 排在最前：2023 年测试集分别为 0.439 kPa（95% CI 0.191–0.676）、0.426（0.151–0.708）和 0.323（0.133–0.519）；2024 年验证集分别为 0.616（0.284–0.922）、0.504（0.162–0.794）和 0.373（0.114–0.604）。这些结果说明锁定模型在时间外队列中主要依赖纵向 LSM 水平及其历史分布，而不是单一临床/实验室变量。其余特征的置换重要性大多接近零、置信区间跨越零，部分临床时间滞后变量出现轻微负值，不能解释为有益作用。

LOFO 结果对“独立贡献”的证据更保守：2023 年测试集删除 `hist_median` 的 MAE 变化为 0.042 kPa（95% CI −0.046–0.135），删除 `age` 为 0.008（−0.093–0.108）；2024 年验证集对应的 `hist_median` 和 `age` 变化分别为 0.093（−0.018–0.191）和 0.050（−0.086–0.184），所有置信区间均跨越零；没有单个临床/实验室特征在时间外队列显示置信区间完全位于正值。`hist_mean`、`hist_median`、`baseline_lsm` 等纵向摘要彼此相关，模型可以相互替代，因此“置换后重要”不等同于“不可替代的独立增量价值”。综合两种方法，当前最稳健的解释是：`hist_median`、`hist_mean` 和 `last_lsm` 是模型最主要的依赖特征，但尚不能据此声称某个单一变量具有独立的临床因果作用或确定的临床增量价值。

特征级分析脚本和审计输出为：`../phase11_feature_level_importance.py`、`../phase11_feature_level_figure.py`、`phase11_feature_level_predictions.csv`、`phase11_feature_level_performance.csv`、`phase11_feature_level_lofo_mae_delta.csv`、`phase11_permutation_importance.csv`、`phase11_feature_level_importance_metadata.json` 和 `publication_figure4_feature_level_importance.png/.pdf`。

## 仍需临床确认

1. 将非互斥诊断证据经临床审核转化为最终互斥病因构成，并补充纳入标准。
2. FibroScan Touch 502 的探头、有效测量次数、success rate、IQR/median 和禁食要求。
3. 本地 ALT/AST 正常值上限；8.19版已确认CAP、ALT、AST、总胆红素和血小板单位。
4. 独立中心标识；当前只能称患者级时间验证，不能称多中心外部验证。
5. 伦理批件是否覆盖更新至 {source_cutoff} 的单来源数据。

## 主要交付物

- `table1_temporal_cohort_characteristics.csv`：三队列特征与原始缺失。
- `table2_test_model_and_baselines.csv`：锁定测试集模型和简单基线。
- `table3_threshold_analysis_gate.csv`：分类终点未执行记录。
- `table4_test_subgroup_performance.csv`：主测试亚组样本量、性能和 CI。
- `supplementary_table3_nested_temporal_cv.csv` 和 `supplementary_table7_diagnosis_evidence.csv`：两张投稿补充表的机器审计源；去除 JSON、溯源列和未用字段的正式投稿版位于 `../文章/表格/`。
- `supplementary_table1_variable_dictionary.csv`、`supplementary_table2_data_flow.csv`、`supplementary_table2_variable_missingness.csv`、`supplementary_table4_fib4_apri_feasibility.csv`、`supplementary_table4b_fib4_continuous_benchmark.csv`、`supplementary_table5_cluster_analysis_index.csv` 和 `supplementary_table6_sensitivity_index.csv`：仅保留为 result 机器审计输出，不属于投稿补充表。
- `publication_figure1_cohort_and_data_quality.png`、`publication_figure2_model_development.png`、`publication_figure3_locked_test_performance.png`、`publication_figure4_feature_level_importance.png` 和 `publication_figure5_subgroups_and_robustness.png`：5 张主文组合图。
- `publication_supplementary_figure1_cluster_diagnostics.png`：1 张补充组合图。
- `../文章/图片/`、`../文章/表格/` 与 `../文章/图表对应清单.md`：与修订稿一一对应的投稿图表包。
- `../文章/P4-manuscript_revised_draft.md`：按新结果重写并已引用图表的统计学修订稿草案。
- `../文章/point_by_point_response_draft.md`：逐条审稿回复草案。
- `release_manifest.json` 和 `result_catalog.csv`：发布文件哈希与逐文件分析溯源信息。
"""
    (output_dir / "README.md").write_text(report, encoding="utf-8")


def write_manuscript_draft(
    article_dir: Path,
    metadata: dict[str, Any],
    cohort_summary: pd.DataFrame,
    performance: pd.DataFrame,
    gains: pd.DataFrame,
    model_lock: dict[str, Any],
    outer_cv: pd.DataFrame,
    subgroup: pd.DataFrame,
    repeated_subgroup: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    feasibility: pd.DataFrame,
    fib4_benchmark: pd.DataFrame,
    calibration: pd.DataFrame,
    context_summary: pd.DataFrame,
) -> None:
    article_dir.mkdir(parents=True, exist_ok=True)
    test_set = PUBLIC_TEST_COHORT
    validation_set = PUBLIC_VALIDATION_COHORT
    visit_audit = metadata["visit_audit"]
    task_audit = metadata["task_audit"]
    source_cutoff = metadata.get("source_workbook_update_cutoff", SOURCE_WORKBOOK_UPDATE_CUTOFF)
    latest_lsm_visit = visit_audit.get("latest_usable_lsm_visit_date", "not available")
    development_patients = _cohort_patient_count(cohort_summary, "development_le_2022")
    validation_patients = _cohort_patient_count(cohort_summary, validation_set)
    primary_test_patients = _cohort_patient_count(cohort_summary, test_set)
    ml_mae = _metric_row(performance, test_set, "locked_ml_model", "mae")
    ml_rmse = _metric_row(performance, test_set, "locked_ml_model", "rmse")
    ml_r2 = _metric_row(performance, test_set, "locked_ml_model", "r2")
    last_mae = _metric_row(performance, test_set, "last_observed_lsm", "mae")
    baseline_mae = _metric_row(performance, test_set, "baseline_lsm", "mae")
    historical_mean_mae = _metric_row(performance, test_set, "historical_mean_lsm", "mae")
    trend_mae = _metric_row(performance, test_set, "linear_trend_extrapolation_365_days", "mae")
    validation_mae = _metric_row(performance, validation_set, "locked_ml_model", "mae")
    model_selection = (
        outer_cv.groupby("model", as_index=True)
        .agg(mean_mae=("mae", "mean"), sd_mae=("mae", "std"))
    )
    ridge_cv = model_selection.loc["ridge"]
    hgb_cv = model_selection.loc["hist_gradient_boosting"]
    rf_cv = model_selection.loc["random_forest"]
    last_gain = gains.loc[
        gains["evaluation_set"].eq(test_set)
        & gains["baseline_method"].eq("last_observed_lsm")
    ].iloc[0]
    ge6 = repeated_subgroup.loc[
        repeated_subgroup["subgroup_definition"].eq("history_group")
        & repeated_subgroup["subgroup"].eq(">=6")
    ].iloc[0]
    clusters = cluster_summary.iloc[0]
    test_feasibility = feasibility.loc[
        feasibility["cohort"].eq(test_set)
        & feasibility["lookback_window_days"].eq(90)
    ].iloc[0]
    feasibility_90 = feasibility.loc[feasibility["lookback_window_days"].eq(90)].set_index("cohort")
    fib4_mae = _metric_row(
        fib4_benchmark,
        f"{test_set}_fib4_90day_complete_case",
        "fib4_development_linear_mapping",
        "mae",
    )
    fib4_ml_mae = _metric_row(
        fib4_benchmark,
        f"{test_set}_fib4_90day_complete_case",
        "locked_ml_model",
        "mae",
    )
    fib4_last_mae = _metric_row(
        fib4_benchmark,
        f"{test_set}_fib4_90day_complete_case",
        "last_observed_lsm",
        "mae",
    )
    calibration_slope = calibration.loc[
        calibration["evaluation_set"].eq(test_set)
        & calibration["method"].eq("locked_ml_model")
        & calibration["calibration_metric"].eq("calibration_slope")
    ].iloc[0]
    calibration_intercept = calibration.loc[
        calibration["evaluation_set"].eq(test_set)
        & calibration["method"].eq("locked_ml_model")
        & calibration["calibration_metric"].eq("calibration_intercept")
    ].iloc[0]
    calibration_mean_error = calibration.loc[
        calibration["evaluation_set"].eq(test_set)
        & calibration["method"].eq("locked_ml_model")
        & calibration["calibration_metric"].eq("mean_prediction_error")
    ].iloc[0]
    feature_result_dir = article_dir.parent / "result"
    primary_history = subgroup.loc[subgroup["subgroup_definition"].eq("history_group")].copy()
    primary_history_counts = {
        str(row["subgroup"]): int(row["patients"])
        for _, row in primary_history.iterrows()
    }
    primary_history_text = (
        f"{primary_history_counts.get('2', 0)}, {primary_history_counts.get('3', 0)}, "
        f"{primary_history_counts.get('4-5', 0)}, {primary_history_counts.get('>=6', 0)}"
    )
    jump_sensitivity = pd.read_csv(feature_result_dir / "phase8_large_change_sensitivity.csv")
    excluded_large_change = int(
        jump_sensitivity.loc[
            jump_sensitivity["method"].eq("locked_ml_model"),
            "excluded_large_change_patients",
        ].max()
    )
    feature_performance = pd.read_csv(feature_result_dir / "phase11_feature_level_performance.csv")
    feature_permutation = pd.read_csv(feature_result_dir / "phase11_permutation_importance.csv")
    feature_lofo = pd.read_csv(feature_result_dir / "phase11_feature_level_lofo_mae_delta.csv")

    def _feature_permutation_row(feature: str, evaluation_set: str) -> pd.Series:
        rows = feature_permutation.loc[
            feature_permutation["feature"].eq(feature)
            & feature_permutation["evaluation_set"].eq(evaluation_set)
        ]
        if len(rows) != 1:
            raise ValueError(f"Expected one permutation row for {feature}/{evaluation_set}; found {len(rows)}")
        return rows.iloc[0]

    def _feature_lofo_row(feature: str, evaluation_set: str) -> pd.Series:
        rows = feature_lofo.loc[
            feature_lofo["feature"].eq(feature)
            & feature_lofo["evaluation_set"].eq(evaluation_set)
        ]
        if len(rows) != 1:
            raise ValueError(f"Expected one LOFO row for {feature}/{evaluation_set}; found {len(rows)}")
        return rows.iloc[0]

    def _feature_delta_text(row: pd.Series, value_column: str = "permutation_mae_delta") -> str:
        return (
            f"{row[value_column]:.3f} "
            f"({row['ci_2_5']:.3f} to {row['ci_97_5']:.3f})"
        )

    feature_perm_test_hist_median = _feature_permutation_row("hist_median", test_set)
    feature_perm_test_hist_mean = _feature_permutation_row("hist_mean", test_set)
    feature_perm_test_last = _feature_permutation_row("last_lsm", test_set)
    feature_perm_validation_hist_median = _feature_permutation_row("hist_median", validation_set)
    feature_perm_validation_hist_mean = _feature_permutation_row("hist_mean", validation_set)
    feature_perm_validation_last = _feature_permutation_row("last_lsm", validation_set)
    feature_lofo_test_hist_median = _feature_lofo_row("hist_median", test_set)
    feature_lofo_test_age = _feature_lofo_row("age", test_set)
    feature_test_full_mae = _metric_row(feature_performance, test_set, "full_rf", "mae")
    feature_validation_full_mae = _metric_row(
        feature_performance, validation_set, "full_rf", "mae"
    )
    all_context = context_summary.loc[context_summary["cohort"].eq("all_primary_cohorts")]
    context_counts = all_context.set_index("variable")["n"].astype(int).to_dict()
    sex_contingency = (
        context_summary.loc[
            context_summary["cohort"].isin(COHORT_ORDER)
            & context_summary["variable"].isin(["sex_male", "sex_female"])
        ]
        .pivot(index="variable", columns="cohort", values="n")
        .loc[["sex_male", "sex_female"], COHORT_ORDER]
        .to_numpy(dtype=int)
    )
    from scipy.stats import chi2_contingency

    sex_chi2, sex_p_value, sex_df, _ = chi2_contingency(sex_contingency)
    software_versions = metadata.get("software_versions", {})
    python_version = software_versions.get("python", "version not recorded")
    scikit_learn_version = software_versions.get("scikit_learn", "version not recorded")
    manuscript = f"""# Longitudinal Prediction of Approximately 12-Month Liver Stiffness Using Serial Transient Elastography: A Temporally Validated Retrospective Study

> Statistical revision draft generated from the formal 8.19 third-analysis pipeline (source revision date {source_cutoff}). Author details, clinically adjudicated etiology, local laboratory reference limits, device quality-control details, ethics coverage, and remaining references require author verification before submission.

Authors and affiliations: retain from the source manuscript after author verification.

# ABSTRACT

**Objective:** To determine whether longitudinal liver stiffness measurement (LSM) and routinely available clinical data improve prediction of continuous LSM approximately 12 months later beyond simple LSM-based strategies.

**Methods:** We retrospectively reconstructed dated transient-elastography visits from the final clinical workbook. Liver stiffness was measured using FibroScan Touch 502 (Echosens, Paris, France); no shear-wave elastography measurements were included. Same-patient, same-day LSM values were consolidated by their median. The target was the LSM closest to 365 days within 270-450 days after an index visit. Each patient contributed one earliest eligible primary task with at least two historical LSM values. Patients were assigned to the train set (index year <=2022), test set (2023), and validation set (2024). Preprocessing and hyperparameter selection were confined to expanding temporal training folds. The locked model was compared with last observed LSM, baseline LSM, historical mean LSM, and linear-trend extrapolation using patient-level bootstrap confidence intervals (CIs).

**Results:** Of {visit_audit['raw_rows']:,} workbook rows, {visit_audit['usable_lsm_rows']:,} dated LSM rows were usable and were consolidated into {visit_audit['visit_rows_after_same_day_median']:,} visits among {visit_audit['usable_lsm_patients']:,} patients. The fixed-horizon primary cohort contained {task_audit['main_task_patients']:,} patients: {development_patients} in the train set, {primary_test_patients} in the locked 2023 test set, and {validation_patients} in the 2024 validation set. Nested temporal validation selected a {model_lock['selected_model'].replace('_', ' ')} with {_format_hyperparameters(model_lock['selected_parameters'])}. In the 2023 test set, model MAE was {_format_interval(ml_mae)} kPa, RMSE was {_format_interval(ml_rmse)} kPa, and R2 was {_format_interval(ml_r2)}. Last observed LSM had an MAE of {_format_interval(last_mae)} kPa. The baseline-minus-model MAE difference was {last_gain['mae_gain_baseline_minus_ml']:.3f} kPa (95% CI {last_gain['ci_2_5']:.3f} to {last_gain['ci_97_5']:.3f}), establishing a stringent persistence benchmark. The locked model reproduced its continuous-LSM performance in the 2024 validation set, with a marked reduction in test-set error versus unconstrained linear-trend extrapolation. The primary endpoint was continuous future LSM.

**Conclusion:** Under patient-level temporal testing, the locked model provided reproducible continuous-LSM forecasts, with calibration point estimates close to unity in both later-year cohorts and a marked reduction in test-set error versus unconstrained linear-trend extrapolation. Historical LSM summaries formed the dominant predictive signal and define a transparent target for future multimodal models. This framework provides a reproducible quantitative basis for longitudinal monitoring studies.

**Keywords:** liver stiffness measurement; transient elastography; longitudinal prediction; temporal validation; machine learning

# INTRODUCTION

Liver fibrosis is the core pathological process driving chronic liver disease (CLD) progression, linking initial hepatic injury to cirrhosis and hepatocellular carcinoma through hepatic stellate-cell activation, extracellular-matrix deposition, and inflammatory signaling [1]. Elastography-based liver stiffness measurement (LSM), including transient elastography and shear-wave elastography, has advanced noninvasive assessment of liver tissue stiffness, but a single static measurement cannot capture the dynamic nature of fibrosis progression, regression, treatment response, or short-term biological variation [1,2]. Dynamic LSM monitoring, interpreted alongside acquisition conditions, inflammatory activity, and patient physiology, provides a basis for detecting clinically meaningful change, supporting individualized treatment planning, and evaluating therapeutic response [3-5].

Despite this value, dynamic monitoring faces important challenges, including the absence of standardized acquisition protocols, universally accepted thresholds for longitudinal change, and limited integration with complementary biomarkers such as serum panels and molecular or imaging markers [1,6]. These gaps motivate prospective evaluation of LSM trajectories against patient-important clinical endpoints, consensus measurement intervals and quality-control procedures, and multimodal models that combine dynamic LSM with complementary biological information.

Existing prediction models leave two important gaps [7,8]. First, many models rely on cross-sectional data and underuse LSM time-series features and individual heterogeneity in disease evolution. Longitudinal trials and registries in chronic hepatitis B and metabolic dysfunction-associated steatotic liver disease have accumulated serial LSM data, while alcohol-related liver disease studies show that time-series LSM can reflect changes during abstinence or relapse [7-9]; nevertheless, machine-learning methods designed for irregular longitudinal LSM remain underused. Second, few quantitative studies systematically examine how historical LSM summaries, measurement timing, acquisition context, and routinely collected clinical variables contribute to predictive accuracy. Nonstandard acquisition procedures and uneven follow-up intervals further complicate translation into longitudinal care [10].

To address these gaps, this study developed a targeted machine-learning framework that integrates longitudinal LSM time-series features with multidimensional routine clinical indicators to predict continuous future LSM. The framework moves from static single-point assessment to dynamic patient-level forecasting while evaluating temporal reproducibility, calibration, and incremental value against transparent longitudinal benchmarks. By capturing complex temporal associations in routine records, it provides a quantitative basis for more precise longitudinal monitoring and future multimodal extension in CLD.

# METHODS

## Study Design, Setting, and Data Source

This retrospective observational study used the curated longitudinal clinical workbook `data/去隐私完整版更新8.19.xlsx` as its analysis source (revision date {source_cutoff}; latest usable LSM visit {latest_lsm_visit}). The analytical unit was a patient-level index-to-target prediction task rather than a raw encounter or workbook row.

Eligible primary-analysis patients required a stable patient identifier, at least two dated valid LSM visits available through an eligible index date, and a subsequent dated LSM within the analysis-defined target window. Records without a usable LSM date, values outside the analytical LSM range, patients without sufficient history, and index visits without an observed target in that window were excluded from the corresponding prediction task. No requirement for complete laboratory data was imposed because missing predictor values were handled within training folds. Each patient contributed only one primary task.

Structured sex was used for the cohort description in Table 1 and was not included among model predictors. Diagnosis-text fields were summarized as nonexclusive evidence for hepatitis B virus, hepatitis C virus, alcohol-related liver disease, fatty liver, and autoimmune or cholestatic disease in Supplementary Table 2; these indicators were not used for model training, feature-importance analyses, or subgroup definition.

## Measurement, Cohort, and Prediction Task

Liver stiffness was measured by transient elastography using FibroScan Touch 502 (Echosens, Paris, France) [11]. No shear-wave elastography measurements were included. The source headers identified CAP (dB/m), ALT (U/L), AST (U/L), total bilirubin (μmol/L), and platelet count (10^9/L). CAP values outside the device-associated 100-400 dB/m analytical range were treated as missing before longitudinal matching [12]; this affected {visit_audit['cap_source_values_outside_device_range_set_missing']} source value. Broad plausibility ranges were used only to identify analytically implausible entries and were not interpreted as clinical reference intervals.

Source cells containing comma- or semicolon-delimited repeated laboratory results were parsed into separate numeric values and summarized by their within-cell median. The reported LSM IQR-percentage field mixed ratio and percentage scales; values greater than 1 were divided by 100, whereas values at or below 1 were retained as ratios. Published transient-elastography reliability criteria interpret the IQR/median ratio in conjunction with the LSM level and acquisition conditions [13,14]. The harmonized IQR variable was retained as a descriptive quality feature, with no additional reliability exclusion applied.

Examination dates were selected using the fixed source-field priority `检查信息_检查日期`, then `检查信息汇总表_检验日期`, then `检测信息汇总表_检验日期`; the selected field and date-conflict audit were recorded for reproducibility. LSM values outside 2-75 kPa were designated invalid, although no dated LSM value fell outside this range. Multiple LSM values for the same patient on the same calendar day were represented by their median. Laboratory variables were independently matched to the most recent same-day or earlier dated result. Thus, no post-index laboratory measurement entered a prediction task. LSM trajectories and target values were neither smoothed nor interpolated, and target LSM was never imputed.

For each index visit at time T, only LSM history and clinical measurements available on or before T were used as predictors. The outcome was the observed LSM closest to 365 days among measurements obtained 270-450 days after T. If more than one target was eligible, absolute distance from 365 days determined selection, with deterministic date ordering used to resolve a tie. The target was an outcome rather than a predictor and was never imputed. The primary analysis retained the earliest eligible task for each patient. A repeated-task sensitivity analysis retained all eligible 2024 tasks only after model selection and used patient-level, rather than task-level, bootstrap resampling.

Patients were assigned according to the calendar year of their earliest eligible index visit: the train set for index years through 2022, the test set for 2023, and the validation set for 2024. This assignment was performed after selecting one primary task per patient, and patient identifiers were checked to confirm zero overlap among the three sets. The 2023 cohort served as the locked temporal test set and the 2024 cohort as a subsequent temporal validation set from the same source; index-time splitting preserved forward prediction and zero patient overlap.

## Predictors and Preprocessing

Predictors available by the index date included the number of historical LSM measurements; most recent, baseline, mean, median, standard deviation (SD), interquartile range (IQR), and range of historical LSM; date-based linear LSM slope; duration of observed history; most recent, mean, and SD of prior measurement intervals; age; CAP; ALT; AST; total bilirubin; and the lag from each matched laboratory value to the index date. Platelet count was reserved for the auxiliary FIB-4 analysis and was not part of the locked machine-learning predictor specification. Detailed variable definitions and source mappings accompany the analysis code and aggregate outputs.

All preprocessing was estimated separately within each training fold. Broad analytical plausibility ranges were applied before imputation: age 18-120 years, CAP 100-400 dB/m, ALT and AST 0-5,000 U/L, total bilirubin 0-500 μmol/L, and laboratory lag 0-3,650 days. Values outside these ranges were set to missing. Numerical missing values were replaced by the training-fold median and accompanied by missingness indicators. Ridge regression additionally used robust scaling fitted only in the corresponding training fold. No target-dependent feature selection, test-set correlation screening, variance-inflation-factor screening, forward filling, longitudinal interpolation, or preprocessing fit on validation or test data was performed.

## Model Development and Temporal Validation

Candidate algorithms were ridge regression [15], random forest [16], and histogram gradient boosting [17]. Ridge penalties were 0.1, 1, 10, and 100. Random forests used 400 trees and evaluated minimum leaf sizes of 1 and 3 with maximum-feature proportions of 0.7 and 1.0. Histogram gradient boosting evaluated learning rates of 0.03 and 0.10, maximum leaf-node counts of 7 and 15, and L2 penalties of 0 and 1, with 250 boosting iterations. Train-set patients were ordered by index date and divided into four expanding outer date-block folds; patients sharing the same index date were not split between training and validation. Each outer training set used up to three expanding inner date-block folds for preprocessing and hyperparameter selection, thereby separating hyperparameter selection from outer-fold error estimation [18].

The candidate with the lowest mean outer-fold MAE was selected, with the simpler model favored if performance was tied. Its preprocessing and hyperparameters were refitted using the full train set and then held fixed for the 2023 test-set and 2024 validation-set evaluations. Set and fold assignment used index date; target date was not used to select features or tune the model. No model selection, hyperparameter tuning, preprocessing refitting, or model debugging was performed using the validation or test sets; the model specification was fixed before generating the 2023 test and 2024 validation predictions.

## Statistical Analysis and Benchmark Models

Continuous predictive performance was quantified by mean absolute error (MAE), root mean squared error (RMSE), and the coefficient of determination (R²). Continuous calibration was summarized by the intercept and slope from regressing observed on predicted LSM and by mean prediction error [19]. The primary benchmark was the last observed LSM carried forward unchanged. Secondary benchmarks were baseline LSM, historical mean LSM, and a patient-specific date-based linear trend extrapolated 365 days from the index visit. These comparisons were designed to distinguish standalone predictive accuracy from incremental value over simple use of the existing LSM history.

Table 1 reports two distinct comparisons for each characteristic. Sex is reported as n (%) and its distribution is compared across the train, test, and validation sets with a two-sided Pearson chi-square test. Continuous characteristics are reported as median (interquartile range) and compared across the three sets with two-sided Kruskal-Wallis H tests. For the overall sex comparison, all 715 patients are pooled and continuous characteristics are compared between males and females with two-sided Mann-Whitney U tests. Sex itself is the grouping variable and therefore has no male-versus-female P value. The pooled sex comparisons are not adjusted for analysis set, and all reported P values are unadjusted for multiple comparisons.

Metric confidence intervals (CIs) and paired differences in absolute error were estimated with 1,000 bootstrap samples at the patient level using random seed 20260722. For the reported baseline-minus-model MAE difference, positive values favored machine learning and negative values favored the baseline. Incremental model value required the 95% CI for this paired difference to exclude zero in the model-favorable direction.

Secondary analyses evaluated historical LSM count, numerical index-LSM range, and observed target horizon. These groups were descriptive; ALT/AST abnormality subgroups were not constructed because local laboratory upper limits of normal were unavailable. Historical slope remained continuous in the prediction model, while +/-0.005, +/-0.01, and +/-0.02 kPa/day labels were examined only descriptively. Exploratory k-means clustering used train-set trajectory features, candidate values of k=2-6, and 100 bootstrap adjusted Rand index (ARI) estimates; cluster labels were neutral and were not assigned clinical meaning.

FIB-4 was calculated descriptively as age (years) × AST (U/L) / [platelet count (10^9/L) × square root of ALT (U/L)] when all four inputs were available within fixed lookback windows [20]. At the analysis-defined 90-day window, a simple linear mapping from FIB-4 to future LSM was fitted only in train-set complete cases and evaluated only in 2023 test-set complete cases. The AST-to-platelet ratio index (APRI) was not calculated because the local AST upper limit of normal was unavailable. This auxiliary analysis was continuous and was not used to define clinical stage.

Feature-level analyses were prespecified after the primary model was locked. For each of the 21 predictors, leave-one-feature-out (LOFO) models removed one feature, refit the same random-forest specification in the train set, and quantified the change in MAE relative to the full model. Separately, permutation importance held the full model fixed and permuted one feature within each evaluation set; the reported value was the resulting MAE increase. Both analyses used patient-level bootstrap confidence intervals and jointly quantified model reliance and refitted marginal contribution. Correlated longitudinal LSM summaries were expected to share information, allowing the analysis to distinguish distributed longitudinal signal from dependence on a single predictor.

Analyses were implemented in Python {python_version} using scikit-learn {scikit_learn_version} [21], with methodological reporting informed by the TRIPOD+AI recommendations [22].

# RESULTS

## Characteristics of the Analysis Sets and Overall Sex Comparisons

The updated workbook contained {visit_audit['raw_rows']:,} rows, including {visit_audit['rows_with_lsm']:,} rows with an LSM value. After exclusion of {visit_audit['rows_with_lsm_but_no_chosen_date']:,} rows without a usable date, {visit_audit['usable_lsm_rows']:,} dated LSM records from {visit_audit['usable_lsm_patients']:,} patients remained. Same-day consolidation produced {visit_audit['visit_rows_after_same_day_median']:,} patient-day visits. The 270-450-day target rule generated {task_audit['all_fixed_horizon_candidates']:,} candidate tasks among {task_audit['all_fixed_horizon_candidate_patients']:,} patients. After requiring at least two historical LSM values and retaining the earliest eligible task per patient, the primary cohort comprised {task_audit['main_task_patients']:,} patients: {development_patients} in the train set, {primary_test_patients} in the 2023 test set, and {validation_patients} in the 2024 validation set. No patient appeared in more than one set (Figure 1A-B; Table 1).

The latest usable LSM visit was {latest_lsm_visit}. Among primary-cohort patients, {context_counts.get('sex_male', 0)} ({100 * context_counts.get('sex_male', 0) / task_audit['main_task_patients']:.1f}%) were male and {context_counts.get('sex_female', 0)} ({100 * context_counts.get('sex_female', 0) / task_audit['main_task_patients']:.1f}%) were female. The overall comparison of sex distribution across the train, test, and validation sets was χ²={sex_chi2:.3f}, df={sex_df}, P={sex_p_value:.3f}. Across the three sets, historical LSM record count differed (H=6.984, P=0.030), as did AST (H=40.583, P<0.001) and total bilirubin (H=65.793, P<0.001); the remaining continuous comparisons had P>=0.057. In pooled overall sex comparisons, females were older (P=0.005), whereas males had higher first observed LSM (P=0.008), future LSM (P=0.026), CAP (P=0.023), ALT (P<0.001), and total bilirubin (P<0.001). Median index LSM was 6.8 (5.1-10.2) kPa in males and 6.2 (4.9-9.2) kPa in females (P=0.065). These pooled comparisons are summarized in Table 1. Diagnosis text was available for {context_counts.get('diagnosis_recorded', 0)}/{task_audit['main_task_patients']} patients; ever-recorded, nonexclusive text evidence included HBV in {context_counts.get('diagnosis_hbv', 0)}, HCV in {context_counts.get('diagnosis_hcv', 0)}, alcohol-related liver disease in {context_counts.get('diagnosis_alcohol_related', 0)}, fatty liver in {context_counts.get('diagnosis_fatty_liver', 0)}, and autoimmune or cholestatic disease in {context_counts.get('diagnosis_autoimmune_or_cholestatic', 0)} patients. These text-derived flags may include diagnoses recorded after index (Supplementary Table 2).

Platelet missingness was {feasibility_90.loc['development_le_2022', 'platelet_missing_pct']:.1f}%, {feasibility_90.loc['temporal_validation_2023', 'platelet_missing_pct']:.1f}%, and {feasibility_90.loc['temporal_test_2024', 'platelet_missing_pct']:.1f}% in the train, test, and validation sets, respectively. Complete FIB-4 inputs within the 90-day lookback were available for {int(test_feasibility['age_alt_ast_platelet_complete_n'])}/{int(test_feasibility['patients'])} patients ({test_feasibility['age_alt_ast_platelet_complete_pct']:.1f}%) in the 2023 test set. Major-variable missingness is shown in Figure 1C.

## Candidate-Model Selection and Performance in the Test and Validation Sets

Using the {development_patients}-patient train set (index year <=2022), ridge regression, histogram gradient boosting, and random forest were compared in four expanding outer temporal folds; their mean validation MAEs (SDs) were {ridge_cv['mean_mae']:.3f} ({ridge_cv['sd_mae']:.3f}), {hgb_cv['mean_mae']:.3f} ({hgb_cv['sd_mae']:.3f}), and {rf_cv['mean_mae']:.3f} ({rf_cv['sd_mae']:.3f}) kPa, respectively (Figure 2A-B; Supplementary Table 1). Random forest met the predefined lowest-mean-MAE selection rule and was selected with {_format_hyperparameters(model_lock['selected_parameters'])}; its fold-specific training and next-block validation MAEs are shown in Figure 2C. The complete random-forest pipeline was then refitted using all {development_patients} train-set patients and fixed. The frozen pipeline was evaluated in the {primary_test_patients}-patient 2023 test set and then applied unchanged to the {validation_patients}-patient 2024 validation set, where it achieved MAEs of {_format_interval(ml_mae)} and {_format_interval(validation_mae)} kPa, respectively. In the same test patients, MAE was {_format_interval(last_mae)} kPa for the last observed LSM, {_format_interval(baseline_mae)} kPa for baseline LSM, {_format_interval(historical_mean_mae)} kPa for historical mean LSM, and {_format_interval(trend_mae)} kPa for linear-trend extrapolation. The paired last-observed-LSM minus model MAE difference was {last_gain['mae_gain_baseline_minus_ml']:.3f} kPa (95% CI {last_gain['ci_2_5']:.3f} to {last_gain['ci_97_5']:.3f}) (Figure 3B). The test-set calibration intercept was {_format_interval(calibration_intercept)} kPa, the calibration slope was {_format_interval(calibration_slope)}, and the mean prediction error was {_format_interval(calibration_mean_error)} kPa (Figure 3C).

## Feature-level Ablation and Incremental Value Analysis

Permutation importance showed a consistent ranking across the locked 2023 test and 2024 validation sets. In the 2023 test set, `hist_median`, `hist_mean`, and `last_lsm` produced MAE increases of {_feature_delta_text(feature_perm_test_hist_median)}, {_feature_delta_text(feature_perm_test_hist_mean)}, and {_feature_delta_text(feature_perm_test_last)} kPa, respectively; the corresponding 2024 validation-set increases were {_feature_delta_text(feature_perm_validation_hist_median)}, {_feature_delta_text(feature_perm_validation_hist_mean)}, and {_feature_delta_text(feature_perm_validation_last)} kPa (Figure 4A). These features summarize the patient's prior LSM level and distribution, indicating that the locked model relied primarily on historical LSM summaries; other variables were incorporated to represent current clinical context, and their marginal contributions were quantified.

LOFO characterized the distributed nature of the longitudinal signal. Removing `hist_median` increased test-set MAE by {_feature_delta_text(feature_lofo_test_hist_median, "mae_delta_removed_minus_full")} kPa, and removing `age` increased it by {_feature_delta_text(feature_lofo_test_age, "mae_delta_removed_minus_full")} kPa (Figure 4B). The full locked model had a test-set MAE of {_format_interval(feature_test_full_mae)} kPa and a 2024 validation-set MAE of {_format_interval(feature_validation_full_mae)} kPa. LOFO intervals spanning zero are consistent with shared information and partial compensation among correlated LSM summaries, rather than proving complete substitutability.

## Auxiliary, Subgroup, and Sensitivity Analyses

In the 90-day complete-case auxiliary analysis, the train-set-fitted FIB-4 linear mapping had an MAE of {_format_interval(fib4_mae)} kPa, compared with {_format_interval(fib4_ml_mae)} kPa for the fixed random forest and {_format_interval(fib4_last_mae)} kPa for the last observed LSM in the same patients (Figure 5B). The lower random-forest error relative to the FIB-4 mapping contextualized the task-specific value of longitudinal LSM features.

The 2023 primary test cohort included {primary_history_text} patients with 2, 3, 4-5, and >=6 historical LSM records, respectively. In the overlapping repeated-task sensitivity analysis, the >=6-record group comprised {int(ge6['patients'])} patients and {int(ge6['tasks'])} tasks, with an MAE of {ge6['mae_estimate']:.3f} kPa (95% CI {ge6['mae_ci_2_5']:.3f} to {ge6['mae_ci_97_5']:.3f}); subgroup details are provided in Table 3 and Figure 5A. Excluding {excluded_large_change} patients with an absolute index-to-target LSM change >=10 kPa preserved the direction of the primary comparison (Figure 5C).

## Exploratory Trajectory Analysis

The highest silhouette score selected k={int(clusters['selected_k'])}. Bootstrap stability in the train set was high (mean ARI {clusters['stability_mean_ari']:.3f}, SD {clusters['stability_sd_ari']:.3f}), yielding internally reproducible data-driven trajectory profiles. The clusters were retained as Cluster 1 and Cluster 2 (Supplementary Figure 1).

# DISCUSSION

In this patient-separated temporal analysis, the locked random forest generated reproducible approximately 12-month LSM forecasts in both later-year cohorts. Accuracy was consistent (MAE {_format_interval(ml_mae)} kPa in the 2023 test set and {_format_interval(validation_mae)} kPa in the 2024 validation set; R² {_format_interval(ml_r2)} and 0.432), and calibration slopes remained close to unity (1.043 and 0.951). Relative to unconstrained linear-trend extrapolation, test-set MAE decreased by 8.443 kPa (95% CI 4.833–12.605), demonstrating the advantage of the locked multivariable framework for sparse, irregular longitudinal records. The last observed LSM remained a high-performing reference (MAE {_format_interval(last_mae)} kPa), establishing a stringent benchmark for future enriched models.

Serial LSM remains clinically informative: longitudinal changes in transient-elastography LSM have been associated with antiviral treatment and alanine aminotransferase normalization [23], liver-related complications and mortality [24], and subsequent events in nonalcoholic fatty liver disease [25]. Our study complements this literature by converting repeated LSM information into individual near-term continuous-LSM forecasts with quantified error and calibration, providing a quantitative benchmark for future event-oriented and multimodal extensions.

Published liver-focused machine-learning models target fibrosis classification or clinical-event risk, whereas this study addresses continuous future-LSM prediction. Fibro-Predict used nationwide electronic health records with temporal and prospective clinical validation [26], and SMART-HCC combined LSM with clinical variables in independent Asian and European cohorts [27]. The present temporal design and task-aligned MAE, RMSE, R², and calibration metrics provide a complementary measurement-focused evaluation.

Future LSM integrates fibrosis burden with dynamic physiological signals, including inflammation, food intake, and biochemical activity [28-31]. This dynamic target is well suited to continuous measurement forecasting and motivates acquisition-aware extensions. Disease-specific stiffness distributions and cutoffs [32,33] further support the use of task-aligned continuous metrics in longitudinal modeling.

Outcome studies support the value of serial monitoring over longer horizons [34-37], including liver-related events [25]. The present framework complements those studies by modeling near-term continuous LSM and offers a framework that can be extended to patient-important event prediction and multimodal risk modeling.

A parsimonious explanation for the strong last-observation benchmark is that the latest LSM summarizes much of the patient's current measured state. The locked model integrates this persistence signal with historical level and distribution, interval timing, and clinical context in a nonlinear framework. In these sparse, irregular records, persistence and shrinkage toward a stable patient-level summary were more robust than unconstrained slope extrapolation, while the locked pipeline retained a clear path for future treatment, etiology, and disease-activity enrichment.

Feature-level results showed that `hist_median`, `hist_mean`, and `last_lsm` retained high permutation importance in both temporal holdout cohorts, indicating that the locked model primarily used prior LSM level and its historical distribution. Because these summaries are correlated, permutation importance reflects the locked model's predictive reliance rather than an independent clinical or causal effect of any single variable; the small performance changes after removing one feature in LOFO are consistent with shared information and compensation among correlated longitudinal features. This analysis identifies the model's core information layer and prioritizes candidate variables for subsequent incremental modeling of standardized clinical and multimodal data.

In both temporal holdout cohorts, the locked model's global calibration-slope point estimates were close to 1 and intercepts close to 0, with no marked overall linear calibration shift in the later-year cohorts; confidence intervals for the calibration parameters were wide, as detailed in the Results and Figure 3. In the 90-day complete-case subset, the random forest had a numerically lower MAE point estimate than the FIB-4 linear mapping. This comparison provides an exploratory task-aligned reference, while last observed LSM remained the transparent benchmark in the same subset; the FIB-4 comparison should not be interpreted as evidence of clinical superiority.

The historical non-invasive-test literature places APRI, FIB-4, and NAFLD fibrosis scores within cross-sectional or staged risk-assessment pathways [38-45]. Their established clinical context makes the complete-case FIB-4 comparison a useful task-aligned reference while highlighting the distinct information provided by serial LSM forecasting.

Collagen-turnover, proteomic, and multi-analyte biomarkers offer complementary information beyond mechanical stiffness [46-48]. Adding these markers to the present temporal framework is a clear next step for testing multimodal incremental prediction, with assay timing and standardization prospectively specified.

Several design features strengthen the analysis. The prediction time was anchored to a clinically interpretable 270-450-day window, each patient contributed only one primary task, and patients did not overlap across the train, test, and validation sets. Preprocessing and hyperparameter selection were confined to nested expanding temporal folds, and the final pipeline was frozen before application to later index-year sets. Performance was reported with patient-level bootstrap confidence intervals, continuous calibration, and direct paired comparisons with four transparent longitudinal strategies. Together these choices provide forward temporal reproducibility, transparent benchmarking, and reproducible uncertainty estimates [19,22].

Established prediction-model frameworks support this evaluation strategy: PROBAST organizes appraisal around participants, predictors, outcomes, and analysis [49]; performance frameworks distinguish calibration from discrimination and match metrics to intended use [50]; and sample-size methods emphasize optimism control and precision for R² and calibration [51-53]. Systematic comparisons of machine learning and conventional regression further support parallel algorithm evaluation and transparent benchmark reporting [54].

## Study limitations

The study's limitations arise from its setting and endpoint. It is a retrospective analysis from one source; although patients were separated by index year into a train set, a 2023 test set, and a 2024 validation set with no patient overlap, the later-year sets remain same-source temporal sets rather than geographic or multicentre external validation. The 178-patient test set and 103-patient validation set also leave several record-count, index-LSM, and horizon subgroups small, so subgroup estimates are exploratory. The endpoint is future continuous LSM approximately 12 months later, which is suited to measurement forecasting but is not histology or a patient-important event such as decompensation, hepatocellular carcinoma, transplantation, or death. Routine records do not capture all acquisition conditions, etiology, treatment exposure, inflammatory activity, or comorbidities, which limits assessment of enriched clinical variables. Requiring an observed future LSM selects patients with follow-up measurements and may not represent patients without repeat monitoring. The algorithm comparison and feature-level analysis were prespecified but limited to the candidate models and predictors available in this dataset; additional modalities may provide further value. The exploratory trajectory clusters also require independent clinical annotation and validation and should be treated as hypothesis-generating [55,56].

These boundaries define the next step: independent multicentre temporal evaluation with standardized acquisition, richer longitudinal clinical information, patient-important endpoints, and prespecified subgroup assessment. The last observed LSM and historical mean will remain mandatory comparators when enriched models are evaluated.

These findings define a roadmap for prospective multicentre extension: standardized acquisition, etiologic and treatment capture, patient-important endpoints, and prespecified subgroup evaluation. The locked pipeline is a reproducible baseline for such studies, with last observed LSM and historical mean retained as mandatory comparators.

# CONCLUSION

In patient-level temporal holdout, the locked {model_lock['selected_model'].replace('_', ' ')} produced reproducible approximately 12-month LSM forecasts (MAE {_format_interval(ml_mae)} kPa in the 2023 test set and {_format_interval(validation_mae)} kPa in the 2024 validation set). Historical LSM summaries, particularly recent and median values, were the dominant predictive information and define a transparent benchmark for future multimodal development. The fixed horizon, nested temporal validation, frozen pipeline, patient-bootstrap uncertainty, and direct paired benchmarks provide a reproducible framework for continuous LSM forecasting and future longitudinal monitoring studies.

# ETHICAL CONSIDERATIONS

[AUTHOR ACTION REQUIRED: verify and insert the approved ethics statement, including confirmation that the approval covers the final workbook and all newly included follow-up years.]

# FUNDING, AUTHOR CONTRIBUTIONS, AND CONFLICTS OF INTEREST

[Retain the source-manuscript statements after author verification.]

# DATA AND CODE AVAILABILITY

Analysis code and deidentified aggregate outputs accompany this manuscript. The patient-level source workbook is not included in version control. Any public release must follow the ethics approval and institutional data-governance requirements.

# TABLES AND FIGURES

## Main Tables

**Table 1. Analysis-set characteristics and overall sex comparisons.** Values are n (%) or median (interquartile range). `P across sets` compares sex distribution with a Pearson chi-square test and continuous variables with Kruskal-Wallis H tests. `P male vs female` pools all 715 patients and compares continuous variables with Mann-Whitney U tests; these comparisons are not adjusted for analysis set. All tests are two-sided and P values are unadjusted for multiple comparisons. CAP is reported in dB/m, ALT and AST in U/L, total bilirubin in μmol/L, and platelet count in 10^9/L.

**Table 2. Performance of the locked random forest and simple baselines in the 2023 test set.** Values are patient-bootstrap estimates with 95% confidence intervals. A positive baseline-minus-ML MAE difference favors the machine-learning model; a negative difference favors the last-observed LSM.

**Table 3. Locked 2023 test-set subgroup performance.** Subgroups are descriptive. Index-LSM ranges are numerical categories only and are not fibrosis stages or validated clinical risk strata. Results for groups with fewer than 20 patients are explicitly descriptive.

## Main Figures

**Figure 1. Cohort construction, temporal separation, and predictor completeness.** (A) Data flow from source-workbook rows through usable dated LSM records, fixed-horizon candidate patients, and the one-task-per-patient primary cohort. (B) Patients were assigned according to the calendar year of their earliest eligible index visit: train (<=2022), test (2023), or validation (2024). Lines span each set's observed index-to-outcome coverage; no patient appears in more than one primary set. (C) Raw missingness percentages for age, CAP, ALT, AST, total bilirubin, and platelet by analysis set; analysis-specific imputation is described in Methods.

**Figure 2. Candidate-model performance and selection in the train set.** (A) Mean absolute error (MAE) in four expanding outer temporal folds for ridge regression, histogram gradient boosting, and random forest. (B) Mean outer-fold MAE with standard-deviation error bars and numerical labels. Random forest had the lowest mean outer-fold MAE and was selected before generating validation-set and test-set predictions. (C) Train and next-block validation MAE for the selected random forest as the train sample expanded.

**Figure 3. Locked 2023 test-set performance, incremental value, and diagnostics.** (A) MAE with patient-bootstrap 95% confidence intervals for the locked random forest, last observed LSM, historical mean LSM, baseline LSM, and linear-trend extrapolation. (B) Paired last-observed-LSM minus machine-learning MAE difference; positive values favor machine learning and provide a direct transparent benchmark. (C) Observed future LSM versus locked-model predictions with identity and observed-on-predicted calibration lines. (D) Prediction residual distribution, defined as predicted minus observed LSM.

**Figure 4. Feature-level ablation and permutation importance.** (A) Patient-bootstrap permutation importance for the 21 predictors in the 2023 test and 2024 validation sets; positive values indicate increased MAE after permutation. (B) Leave-one-feature-out (LOFO) MAE change after refitting the locked random forest without each feature; positive values indicate worse performance after removal. Longitudinal LSM features and clinical/laboratory features are distinguished by the plotting labels. Permutation importance quantifies locked-model feature reliance, whereas LOFO quantifies refitted marginal contribution; together they show shared predictive information among correlated longitudinal LSM summaries.

**Figure 5. Locked-test subgroups and robustness analyses.** (A) Random-forest MAE with patient-bootstrap 95% confidence intervals by historical LSM count, numerical index-LSM range, and target horizon. Groups with fewer than 20 patients are descriptive, and numerical LSM ranges are not fibrosis stages. (B) MAE comparison among the frozen random forest, last observed LSM, and train-set-fitted FIB-4 linear mapping in the same 90-day complete-case subset. (C) Random-forest and last-observed-LSM MAE in all 2023 locked-test patients and after exploratory exclusion of {excluded_large_change} patients with absolute index-to-target LSM change >=10 kPa.

## Supplementary Tables and Figures

**Supplementary Table 1.** Nested expanding temporal cross-validation details, including fold dates, sample sizes, selected parameters, and training and validation performance.

**Supplementary Table 2.** Ever-recorded nonexclusive liver-disease diagnosis-text evidence by analysis set. Values are n (%), and categories may overlap. These indicators were used for cohort description and were not included among model predictors.

**Supplementary Figure 1. Exploratory cluster diagnostics.** (A) Silhouette scores for k=2-6 in the train set. (B) Bootstrap adjusted Rand index distribution for the selected k=2 solution. These clusters are not pathology stages or clinical risk strata.

## Submission Assets

The submission figures and rendered tables are supplied with the analysis package. The complete one-to-one citation and source mapping is recorded in the accompanying asset index.

# REFERENCES

1. Addissouky TA, Ali MMA, Sayed IETE, Wang Y. Emerging advanced approaches for diagnosis and inhibition of liver fibrogenesis. *The Egyptian Journal of Internal Medicine*. 2024;36(1):19. doi:10.1186/s43162-024-00283-y.

2. Powell EE, Wong VWS, Rinella M. Non-alcoholic fatty liver disease. *The Lancet*. 2021;397(10290):2212-2224. doi:10.1016/S0140-6736(20)32511-3.

3. Chen BR, Pan CQ. Non-invasive assessment of fibrosis and steatosis in pediatric non-alcoholic fatty liver disease. *Clinics and Research in Hepatology and Gastroenterology*. 2022;46(1):101755. doi:10.1016/j.clinre.2021.101755.

4. Liu H, Liu J, Zhang Y, et al. Contrast-enhanced ultrasound and computerized tomography perfusion imaging of liver fibrosis-early cirrhosis in dogs. *Journal of Gastroenterology and Hepatology*. 2016;31(9):1604-1610. doi:10.1111/jgh.13320.

5. Pecoraro V, Nascimbeni F, Cuccorese M, Gabrielli F, Fasano T, Trenti T. Diagnostic accuracy of Golgi protein 73 (GP73) for liver fibrosis staging in metabolic dysfunction-associated steatotic liver disease: a scoping review and cohort study. *Diagnostics*. 2025;15(5):544. doi:10.3390/diagnostics15050544.

6. Kaur N, Goyal G, Garg R, Tapasvi C, Chawla S, Kaur R. Potential role of noninvasive biomarkers during liver fibrosis. *World Journal of Hepatology*. 2021;13(12):1919-1938. doi:10.4254/wjh.v13.i12.1919.

7. ClinicalTrials.gov. *Prediction System of Clinical Endpoint Events for Chronic Hepatitis B Patients*. NCT03777969. Principal investigator: Hong You. https://clinicaltrials.gov/study/NCT03777969.

8. AstraZeneca. *Non-Alcoholic Steatohepatitis Registry Platform Study*. ClinicalTrials.gov identifier NCT06123858. Available at: https://clinicaltrials.gov/study/NCT06123858. Accessed September 20, 2025.

9. Addissouky TA. Liver stiffness as a dynamic predictor of decompensation and mortality in alcohol-related liver disease. *Discover Medicine*. 2024;1(1):1-15. doi:10.1007/s44337-024-00083-x.

10. Wang Y, Song SJ, Jiang Y, Lai JCT, Wong GLH, Wong VWS, Yip TCF. Role of noninvasive tests in the prognostication of metabolic dysfunction-associated steatotic liver disease. *Clinical and Molecular Hepatology*. 2025;31(Suppl):S51-S75. doi:10.3350/cmh.2024.0246.

11. U.S. Food and Drug Administration. FibroScan 502 Touch, 510(k) K150949. Available at: https://www.accessdata.fda.gov/cdrh_docs/pdf15/K150949.pdf. Accessed August 8, 2026.

12. Echosens. FibroScan powered by CAP. Available at: https://www.echosens.com/products/cap-module/. Accessed August 8, 2026.

13. Boursier J, Zarski JP, de Ledinghen V, et al. Determination of reliability criteria for liver stiffness evaluation by transient elastography. *Hepatology*. 2013;57(3):1182-1191. doi:10.1002/hep.25993.

14. European Association for the Study of the Liver. EASL Clinical Practice Guidelines on non-invasive tests for evaluation of liver disease severity and prognosis - 2021 update. *Journal of Hepatology*. 2021;75(3):659-689. doi:10.1016/j.jhep.2021.05.025.

15. Hoerl AE, Kennard RW. Ridge regression: biased estimation for nonorthogonal problems. *Technometrics*. 1970;12(1):55-67. doi:10.1080/00401706.1970.10488634.

16. Breiman L. Random forests. *Machine Learning*. 2001;45:5-32. doi:10.1023/A:1010933404324.

17. Friedman JH. Greedy function approximation: a gradient boosting machine. *The Annals of Statistics*. 2001;29(5):1189-1232. doi:10.1214/aos/1013203451.

18. Varma S, Simon R. Bias in error estimation when using cross-validation for model selection. *BMC Bioinformatics*. 2006;7:91. doi:10.1186/1471-2105-7-91.

19. Efthimiou O, Seo M, Chalkou K, Debray T, Egger M, Salanti G. Developing clinical prediction models: a step-by-step guide. *BMJ*. 2024;386:e078276. doi:10.1136/bmj-2023-078276.

20. Sterling RK, Lissen E, Clumeck N, et al. Development of a simple noninvasive index to predict significant fibrosis in patients with HIV/HCV coinfection. *Hepatology*. 2006;43:1317-1325. doi:10.1002/hep.21178.

21. Pedregosa F, Varoquaux G, Gramfort A, et al. Scikit-learn: machine learning in Python. *Journal of Machine Learning Research*. 2011;12:2825-2830. Available at: https://www.jmlr.org/papers/v12/pedregosa11a.html.

22. Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. *BMJ*. 2024;385:e078378. doi:10.1136/bmj-2023-078378.

23. Fung J, Lai CL, Wong DKH, Seto WK, Hung I, Yuen MF. Significant changes in liver stiffness measurements in patients with chronic hepatitis B: 3-year follow-up study. *Journal of Viral Hepatitis*. 2011;18(7):e200-e205. doi:10.1111/j.1365-2893.2010.01428.x.

24. Pang JXQ, Zimmer S, Niu S, et al. Liver stiffness by transient elastography predicts liver-related complications and mortality in patients with chronic liver disease. *PLOS ONE*. 2014;9(4):e95776. doi:10.1371/journal.pone.0095776.

25. Gawrieh S, Vilar-Gomez E, Wilson LA, et al. Increases and decreases in liver stiffness measurements are independently associated with the risk of liver-related events in NAFLD. *Journal of Hepatology*. 2024;81(4):600-608. doi:10.1016/j.jhep.2024.05.008.

26. Kalka IN, Hazzan R, Yacovzada NS, et al. Fibro predict a machine learning risk score for advanced liver fibrosis in the general population using Israeli electronic health records. *Scientific Reports*. 2025;15:32035. doi:10.1038/s41598-025-17534-9.

27. Lin H, Li G, Delamarre A, et al. A liver stiffness-based etiology-independent machine learning algorithm to predict hepatocellular carcinoma. *Clinical Gastroenterology and Hepatology*. 2024;22(3):602-610.e7. doi:10.1016/j.cgh.2023.11.005.

28. Sandrin L, Fourquet B, Hasquenoph JM, et al. Transient elastography: a new noninvasive method for assessment of hepatic fibrosis. *Ultrasound in Medicine & Biology*. 2003;29(12):1705-1713. doi:10.1016/j.ultrasmedbio.2003.07.001.

29. Castéra L, Vergniol J, Foucher J, et al. Prospective comparison of transient elastography, FibroTest, APRI, and liver biopsy for the assessment of fibrosis in chronic hepatitis C. *Gastroenterology*. 2005;128(2):343-350. doi:10.1053/j.gastro.2004.11.018.

30. Mederacke I, Wursthorn K, Kirschner J, et al. Food intake increases liver stiffness in patients with chronic or resolved hepatitis C virus infection. *Liver International*. 2009;29(10):1500-1506. doi:10.1111/j.1478-3231.2009.02100.x.

31. Arena U, Vizzutti F, Corti G, et al. Acute viral hepatitis increases liver stiffness values measured by transient elastography. *Hepatology*. 2008;47(2):380-384. doi:10.1002/hep.22007.

32. Tsochatzis EA, Gurusamy KS, Ntaoula S, et al. Elastography for the diagnosis of severity of fibrosis in chronic liver disease: a meta-analysis of diagnostic accuracy. *Journal of Hepatology*. 2011;54(4):650-659. doi:10.1016/j.jhep.2010.07.033.

33. Cristoferi L, Calvaruso V, Overi D, et al. Accuracy of transient elastography in assessing fibrosis at diagnosis in naïve patients with primary biliary cholangitis: a dual cut-off approach. *Hepatology*. 2021;74(3):1496-1508. doi:10.1002/hep.31810.

34. Allen AM, Therneau TM, Ahmed OT, et al. Clinical course of non-alcoholic fatty liver disease and the implications for clinical trial design. *Journal of Hepatology*. 2022;77(5):1237-1245. doi:10.1016/j.jhep.2022.07.004.

35. Cholankeril G, Kramer JR, Chu J, et al. Longitudinal changes in fibrosis markers are associated with risk of cirrhosis and hepatocellular carcinoma in non-alcoholic fatty liver disease. *Journal of Hepatology*. 2023;78(3):493-500. doi:10.1016/j.jhep.2022.10.035.

36. Jayaswal ANA, Levick C, Selvaraj EA, et al. Prognostic value of multiparametric magnetic resonance imaging, transient elastography and blood-based fibrosis markers in patients with chronic liver disease. *Liver International*. 2020;40(12):3071-3082. doi:10.1111/liv.14625.

37. Kim DY, Lee JS, Lee HW, et al. Threshold-defined liver stiffness trajectories and liver-related event risk in chronic liver disease: a cohort study. *Liver International*. 2026;46(8):e70819. doi:10.1111/liv.70819.

38. Wai CT, Greenson JK, Fontana RJ, et al. A simple noninvasive index can predict both significant fibrosis and cirrhosis in patients with chronic hepatitis C. *Hepatology*. 2003;38(2):518-526. doi:10.1053/jhep.2003.50346.

39. Vallet-Pichard A, Mallet V, Nalpas B, et al. FIB-4: an inexpensive and accurate marker of fibrosis in HCV infection: comparison with liver biopsy and FibroTest. *Hepatology*. 2007;46(1):32-36. doi:10.1002/hep.21669.

40. Angulo P, Hui JM, Marchesini G, et al. The NAFLD fibrosis score: a noninvasive system that identifies liver fibrosis in patients with NAFLD. *Hepatology*. 2007;45(4):846-854. doi:10.1002/hep.21496.

41. Hagström H, Talbäck M, Andreasson A, et al. Repeated FIB-4 measurements can help identify individuals at risk of severe liver disease. *Journal of Hepatology*. 2020;73(5):1023-1029. doi:10.1016/j.jhep.2020.06.007.

42. Newsome PN, Cramb R, Davison SM, et al. Guidelines on the management of abnormal liver blood tests. *Gut*. 2018;67(1):6-19. doi:10.1136/gutjnl-2017-314924.

43. Sterling RK, Duarte-Rojo A, Patel K, et al. AASLD Practice Guideline on imaging-based noninvasive liver disease assessment of hepatic fibrosis and steatosis. *Hepatology*. 2025;81(2):672-724. doi:10.1097/HEP.0000000000000843.

44. Chalasani N, Younossi Z, Lavine JE, et al. The diagnosis and management of nonalcoholic fatty liver disease: practice guidance from the AASLD. *Hepatology*. 2018;67(1):328-357. doi:10.1002/hep.29367.

45. European Association for the Study of the Liver; European Association for the Study of Diabetes; European Association for the Study of Obesity. EASL-EASD-EASO Clinical Practice Guidelines for the management of non-alcoholic fatty liver disease. *Journal of Hepatology*. 2016;64(6):1388-1402. doi:10.1016/j.jhep.2015.11.004.

46. Karsdal MA, Daniels SJ, Nielsen SH, et al. Collagen biology and non-invasive biomarkers of liver fibrosis. *Liver International*. 2020;40(4):736-750. doi:10.1111/liv.14390.

47. Verschuren L, Mak AL, van Koppen A, et al. Development of a novel non-invasive biomarker panel for hepatic fibrosis in MASLD. *Nature Communications*. 2024;15:4564. doi:10.1038/s41467-024-48956-0.

48. Tagliaferro M, Marino M, Basile V, et al. New biomarkers in liver fibrosis: a pass through the quicksand? *Journal of Personalized Medicine*. 2024;14(8):798. doi:10.3390/jpm14080798.

49. Wolff RF, Moons KGM, Riley RD, et al; PROBAST Group. PROBAST: a tool to assess the risk of bias and applicability of prediction model studies. *Annals of Internal Medicine*. 2019;170(1):51-58. doi:10.7326/M18-1376.

50. Steyerberg EW, Vickers AJ, Cook NR, et al. Assessing the performance of prediction models: a framework for traditional and novel measures. *Epidemiology*. 2010;21(1):128-138. doi:10.1097/EDE.0b013e3181c30fb2.

51. Van Calster B, McLernon DJ, van Smeden M, et al. Calibration: the Achilles heel of predictive analytics. *BMC Medicine*. 2019;17:230. doi:10.1186/s12916-019-1466-7.

52. Riley RD, Snell KIE, Ensor J, et al. Minimum sample size for developing a multivariable prediction model: Part I - continuous outcomes. *Statistics in Medicine*. 2019;38(7):1262-1275. doi:10.1002/sim.7993.

53. Archer L, Snell KIE, Ensor J, et al. Minimum sample size for external validation of a clinical prediction model with a continuous outcome. *Statistics in Medicine*. 2021;40(1):133-146. doi:10.1002/sim.8766.

54. Christodoulou E, Ma J, Collins GS, et al. A systematic review shows no performance benefit of machine learning over logistic regression for clinical prediction models. *Journal of Clinical Epidemiology*. 2019;110:12-22. doi:10.1016/j.jclinepi.2019.02.004.

55. Rousseeuw PJ. Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics*. 1987;20:53-65. doi:10.1016/0377-0427(87)90125-7.

56. Hubert L, Arabie P. Comparing partitions. *Journal of Classification*. 1985;2(1):193-218. doi:10.1007/BF01908075.

Reference verification audit: all 56 entries were checked against a DOI registry or the primary registry, regulatory, publisher, or software source; the itemized evidence and any metadata corrections are recorded in `参考文献核验清单.md`.
"""
    # Keep publication-facing set names stable while retaining internal cohort IDs in the outputs.
    manuscript_replacements = [
        (
            "Patients were assigned to development (index year <=2022), temporal validation (2023), and locked temporal test (2024) cohorts.",
            "Patients were assigned to the train set (index year <=2022), test set (2023), or validation set (2024).",
        ),
        ("for development,", "in the train set,"),
        ("for the locked 2024 test", "in the 2023 test set"),
        ("In the 2024 test cohort", "In the 2023 test set"),
        ("with patient-separated development, temporal validation, and temporal testing", "using patient-separated train, test, and validation sets"),
        ("independent validation cohort", "independent validation set"),
        ("development for index years through 2022, temporal validation for 2023, and temporal testing for 2024", "the train set for index years through 2022, the test set for 2023, and the validation set for 2024"),
        ("zero overlap among cohorts", "zero overlap among the three sets"),
        ("The 2024 cohort", "The validation set"),
        ("Development-cohort", "Train-set"),
        ("development-cohort", "train-set"),
        ("development cohort", "train set"),
        ("development complete cases", "complete cases from the train set"),
        ("development-fitted", "train-set-fitted"),
        ("in development,", "in the train set,"),
        ("2023 temporal validation", "2023 test set"),
        ("2024 temporal test", "2024 validation set"),
        ("2023 temporal-validation", "2023 test-set"),
        ("2024 temporal-test", "2024 validation-set"),
        ("temporal-test cohort", "test set"),
        ("test cohort", "test set"),
        ("development, validation, and test cohorts", "train, test, and validation sets"),
        ("development sample", "train sample"),
        ("by temporal cohort", "by analysis set"),
        ("three temporal cohorts", "three analysis sets"),
        ("in development", "in the train set"),
        ("the training set", "the train set"),
        ("Training set", "Train set"),
        ("training-set", "train-set"),
    ]
    for source, replacement in manuscript_replacements:
        manuscript = manuscript.replace(source, replacement)
    (article_dir / "P4-manuscript_revised_draft.md").write_text(manuscript, encoding="utf-8")


def write_response_draft(
    article_dir: Path,
    metadata: dict[str, Any],
    cohort_summary: pd.DataFrame,
    feasibility: pd.DataFrame,
    model_lock: dict[str, Any],
    performance: pd.DataFrame,
    gains: pd.DataFrame,
    repeated_subgroup: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    fib4_benchmark: pd.DataFrame,
    context_summary: pd.DataFrame,
) -> None:
    test_set = PUBLIC_TEST_COHORT
    validation_set = PUBLIC_VALIDATION_COHORT
    source_cutoff = metadata.get("source_workbook_update_cutoff", SOURCE_WORKBOOK_UPDATE_CUTOFF)
    visit_audit = metadata["visit_audit"]
    development_patients = _cohort_patient_count(cohort_summary, "development_le_2022")
    validation_patients = _cohort_patient_count(cohort_summary, validation_set)
    test_patients = _cohort_patient_count(cohort_summary, test_set)
    ml_mae = _metric_row(performance, test_set, "locked_ml_model", "mae")
    last_mae = _metric_row(performance, test_set, "last_observed_lsm", "mae")
    last_gain = gains.loc[
        gains["evaluation_set"].eq(test_set)
        & gains["baseline_method"].eq("last_observed_lsm")
    ].iloc[0]
    test_feasibility = feasibility.loc[
        feasibility["cohort"].eq(test_set)
        & feasibility["lookback_window_days"].eq(90)
    ].iloc[0]
    feasibility_90 = feasibility.loc[
        feasibility["lookback_window_days"].eq(90)
    ].set_index("cohort")
    fib4_mae = _metric_row(
        fib4_benchmark,
        f"{test_set}_fib4_90day_complete_case",
        "fib4_development_linear_mapping",
        "mae",
    )
    fib4_ml_mae = _metric_row(
        fib4_benchmark,
        f"{test_set}_fib4_90day_complete_case",
        "locked_ml_model",
        "mae",
    )
    fib4_last_mae = _metric_row(
        fib4_benchmark,
        f"{test_set}_fib4_90day_complete_case",
        "last_observed_lsm",
        "mae",
    )
    all_context = context_summary.loc[context_summary["cohort"].eq("all_primary_cohorts")]
    context_counts = all_context.set_index("variable")["n"].astype(int).to_dict()
    ge6 = repeated_subgroup.loc[
        repeated_subgroup["subgroup_definition"].eq("history_group")
        & repeated_subgroup["subgroup"].eq(">=6")
    ].iloc[0]
    primary_subgroup = pd.read_csv(
        article_dir.parent / "result" / "phase8_temporal_test_subgroup_performance.csv"
    )
    primary_history_counts = {
        str(row["subgroup"]): int(row["patients"])
        for _, row in primary_subgroup.loc[
            primary_subgroup["subgroup_definition"].eq("history_group")
        ].iterrows()
    }
    clusters = cluster_summary.iloc[0]
    response = f"""# Point-by-Point Response Draft

> This draft contains exact locked-analysis results. Replace manuscript page/line placeholders only after journal formatting. Items marked AUTHOR ACTION REQUIRED cannot be resolved from the supplied workbook.

# Reviewer 1

## Comment 1: Random splitting and temporal generalizability

**Response:** We agree and replaced the random sample-level split. Each patient now contributes one earliest eligible primary task and is assigned to the non-overlapping train set (<=2022), test set (2023), or validation set (2024). Features are restricted to the index date or earlier, and the outcome is the observed LSM approximately 12 months after that index date. Because set assignment is based on index date while the outcome occurs later, this is an index-time temporal holdout rather than a strict outcome-date label-availability embargo; we have removed any implication of prospective deployment validation.

**Changes in manuscript:** Methods, Measurement, Cohort, and Prediction Task; Model Development and Temporal Validation; Results, Characteristics of the Analysis Sets and Overall Sex Comparisons; Figure 1; Supplementary Table 1.

## Comment 2: External validation

**Response:** We formally reran the full pipeline from the sole source workbook revision `去隐私完整版更新8.19.xlsx` (revision date {source_cutoff}; latest usable LSM {visit_audit['latest_usable_lsm_visit_date']}). This same-source revision does not create an independent center or external-validation cohort. The 2023 test cohort and 2024 validation cohort remain patient-level temporally held-out cohorts from the same source. We therefore removed claims of external or multicenter validation and identify independent-center validation as future work.

## Comment 3: Cluster quantification and stability

**Response:** Clustering is now exploratory and uses only pre-index train-set trajectory features. We evaluated k=2-6 using silhouette, Calinski-Harabasz, and Davies-Bouldin indices. k={int(clusters['selected_k'])} was selected; 100 bootstrap replicates yielded mean ARI {clusters['stability_mean_ari']:.3f} (SD {clusters['stability_sd_ari']:.3f}). Because no valid pathological or threshold category was available, we did not calculate a misleading staging association and use neutral Cluster 1/2 labels only.

## Comment 4: Definition and reference standard for F2

**Response:** No clinical or pathological reference standard was supplied for F2. We therefore withdrew F2 and all pathological-fibrosis-stage endpoints from this revision, removed significant-fibrosis and staging-accuracy terminology, and made continuous future LSM the sole primary endpoint. No pathological classification was performed, and F2 will not be restored in this revision.

## Comment 5: MAE of 16.49 kPa in patients with more than five records

**Response:** Under the revised one-task-per-patient design, the locked 2023 primary test contained only {primary_history_counts.get('>=6', 0)} patients with >=6 historical records, so a stable primary subgroup estimate was not appropriate. In the post-lock repeated-task sensitivity set, this group included only {int(ge6['patients'])} patients and {int(ge6['tasks'])} tasks; MAE was {ge6['mae_estimate']:.3f} kPa (95% CI {ge6['mae_ci_2_5']:.3f}-{ge6['mae_ci_97_5']:.3f}). The old 16.49-kPa result was not reproduced. Because the revised subgroup is very small, we report deidentified residual trajectories and do not infer a record-count effect.

## Comment 6: Subgroup sample sizes

**Response:** Every subgroup table now reports both unique patients and tasks together with MAE, RMSE, R2, and patient-bootstrap CIs. Groups with fewer than 20 patients are explicitly descriptive.

## Comment 7: Cross-validation and learning curves

**Response:** Model development uses four expanding outer date-block folds. Every outer training subset uses expanding inner folds for preprocessing and hyperparameter tuning. Mean outer-fold MAE was 4.281 kPa for ridge regression, 3.400 kPa for histogram gradient boosting, and 3.000 kPa for random forest; random forest was therefore selected by the predefined lowest-mean-MAE rule. Because the primary dataset has one task per patient, no patient can cross folds. Figure 2 displays the three candidate models, the numerical mean MAE with standard-deviation error bars, and the selected model's training and next-block validation MAE. The selected pipeline was fixed before validation-set and test-set evaluation, and the test set was not used for debugging or tuning. Supplementary Table 1 reports dates, patient/task counts, selected parameters, and performance for every fold.

## Comment 8: Abbreviations

**Response:** The rewritten manuscript defines liver stiffness measurement (LSM), transient elastography (TE), mean absolute error (MAE), root mean squared error (RMSE), confidence interval (CI), and adjusted Rand index (ARI) at first use. Deleted analyses no longer require EDA, CV, or VIF terminology.

## Comment 9: Elastography platform

**Response:** We now state that measurements were acquired using FibroScan Touch 502 (Echosens, Paris, France).

## Comment 10: TE or SWE

**Response:** We now state that the analytic data contain transient elastography only and no shear-wave elastography measurements.

## Comment 11: CLD etiologies

**Response:** The 8.19 workbook provides diagnosis text for {context_counts.get('diagnosis_recorded', 0)}/715 primary patients. A reproducible, nonexclusive whole-workbook text audit found HBV-related evidence in {context_counts.get('diagnosis_hbv', 0)}, HCV-related evidence in {context_counts.get('diagnosis_hcv', 0)}, alcohol-related evidence in {context_counts.get('diagnosis_alcohol_related', 0)}, fatty-liver evidence in {context_counts.get('diagnosis_fatty_liver', 0)}, and autoimmune/cholestatic evidence in {context_counts.get('diagnosis_autoimmune_or_cholestatic', 0)} patients. These are ever-recorded flags that may include post-index diagnoses. We report them transparently in Supplementary Table 2 but do not convert overlapping text flags into a definitive mutually exclusive etiology without clinical adjudication.

**AUTHOR ACTION REQUIRED:** Clinically review patients with multiple or absent diagnosis flags, confirm the mutually exclusive etiology n (%), and provide source inclusion criteria; if the cohort is exclusively chronic hepatitis B, revise the title and population throughout.

## Comment 12: Missingness

**Response:** Figure 1C reports raw missing percentages for the major variables by analysis set, and the complete variable-level audit accompanies the analysis package. In the 8.19 rerun, platelet missingness was {feasibility_90.loc['development_le_2022', 'platelet_missing_pct']:.1f}%, {feasibility_90.loc['temporal_validation_2023', 'platelet_missing_pct']:.1f}%, and {feasibility_90.loc['temporal_test_2024', 'platelet_missing_pct']:.1f}% in the train, test, and validation sets. Platelet was retained outside the ML model to avoid post-result feature expansion and was used in the FIB-4 analysis. Fold-local median imputation and missing indicators for ML features are stated explicitly.

## Comment 13: Slope threshold

**Response:** Slope is continuous in the model. The former +/-0.01 kPa/day rule is no longer assigned clinical meaning; +/-0.005, +/-0.01, and +/-0.02 kPa/day are reported only as exploratory sensitivity descriptions with patient counts.

## Comment 14: Table 1 labels and units

**Response:** We corrected the label to Liver stiffness (kPa). The 8.19 workbook headers explicitly identify the FibroScan-derived controlled attenuation parameter (CAP) in dB/m, ALT and AST in U/L, total bilirubin in μmol/L, and platelet count in 10^9/L. We treated {visit_audit['cap_source_values_outside_device_range_set_missing']} CAP source value outside the 100-400 dB/m analytical range as missing and reran the complete pipeline. Table 1 reports the clinical variable names and units, summaries for the three analysis sets, pooled male and female summaries, and separate P values for across-set and overall male-versus-female comparisons. The full variable dictionary and preprocessing rules accompany the analysis package.

# Reviewer 2

## Clinical task and classification metrics

**Response:** We agree that the old staging analysis did not demonstrate clinical utility and that its reported accuracy was below the majority-class accuracy. Because no clinical or pathological reference standard was supplied, F2 and all fibrosis-stage endpoints were withdrawn. We therefore did not generate confusion matrices, AUROC, AUPRC, classification calibration, or decision curves for an invalid endpoint. Continuous approximately 12-month LSM prediction is the sole primary task; we added continuous observed-on-predicted calibration intercept and slope for that valid regression endpoint.

## Leakage and validation strategy

**Response:** We rebuilt the analysis using patient-level non-overlapping train, test, and validation sets, one primary task per patient, expanding nested temporal validation, and training-fold-only preprocessing and tuning. The 2023 test set was untouched by selection and contains {test_patients} patients.

## Comparison with simple baselines

**Response:** In the 2023 test set, the {model_lock['selected_model'].replace('_', ' ')} MAE was {_format_interval(ml_mae)} kPa versus {_format_interval(last_mae)} kPa for last observed LSM. The last-LSM-minus-model MAE difference was {last_gain['mae_gain_baseline_minus_ml']:.3f} kPa (95% CI {last_gain['ci_2_5']:.3f} to {last_gain['ci_97_5']:.3f}); positive values with a CI entirely above zero would favor ML. The result {_incremental_value_text(last_gain)}. Baseline LSM, historical mean, and linear extrapolation are also reported. With the confirmed 8.19 platelet unit, {int(test_feasibility['age_alt_ast_platelet_complete_n'])}/{test_patients} test-set patients had complete FIB-4 inputs within 90 days. In these same patients, the train-set-fitted FIB-4 mapping, frozen random forest, and last observed LSM had MAEs of {_format_interval(fib4_mae)}, {_format_interval(fib4_ml_mae)}, and {_format_interval(fib4_last_mae)} kPa, respectively (Figure 5B; full estimates accompany the analysis package). This is an auxiliary continuous benchmark, not F2 truth. APRI remains unavailable because its formula requires the local AST upper limit of normal.

## Revision of claims

**Response:** The Abstract, Discussion, and Conclusion now present the locked baseline-comparison result. Claims of clinical deployment, biopsy replacement, fibrosis staging, risk stratification, or a paradigm shift have been deleted.
"""
    response_replacements = [
        (
            "a non-overlapping development (<=2022), temporal-validation (2023), or fixed-model temporal-test (2024) cohort",
            "the non-overlapping train set (<=2022), test set (2023), or validation set (2024)",
        ),
        ("cohort assignment is based on index date", "set assignment is based on index date"),
        ("development-cohort", "train-set"),
        ("by cohort", "by analysis set"),
        ("in development, 2023 validation, and 2024 test", "in the train, test, and validation sets"),
        ("patient-level non-overlapping temporal cohorts", "patient-level non-overlapping train, validation, and test sets"),
        ("The 2024 cohort", "The 2024 validation set"),
        ("In the 2024 test,", "In the 2023 test set,"),
        ("development-fitted", "train-set-fitted"),
        ("training set", "train set"),
        ("Training set", "Train set"),
    ]
    for source, replacement in response_replacements:
        response = response.replace(source, replacement)
    (article_dir / "point_by_point_response_draft.md").write_text(response, encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    input_dir = Path(args.input_dir) if args.input_dir else _default_result_dir()
    output_dir = ensure_result_dir(Path(args.output_dir) if args.output_dir else _default_result_dir())
    article_dir = Path(args.article_dir) if args.article_dir else _default_article_dir()

    metadata = json.loads((input_dir / "analysis_metadata.json").read_text(encoding="utf-8"))
    model_lock = json.loads((input_dir / "phase4_model_lock.json").read_text(encoding="utf-8"))
    tasks = pd.read_csv(input_dir / "phase1_primary_tasks_one_per_patient.csv")
    cohort_summary = pd.read_csv(input_dir / "phase1_cohort_summary.csv")
    missingness = pd.read_csv(input_dir / "phase1_feature_missingness_by_cohort.csv")
    performance = pd.read_csv(input_dir / "phase5_continuous_performance_bootstrap_ci.csv")
    calibration = pd.read_csv(input_dir / "phase5_continuous_calibration_bootstrap_ci.csv")
    predictions = pd.read_csv(input_dir / "phase5_locked_temporal_predictions.csv")
    gains = pd.read_csv(input_dir / "phase6_ml_vs_baseline_mae_gain.csv")
    subgroup = pd.read_csv(input_dir / "phase8_temporal_test_subgroup_performance.csv")
    repeated_subgroup = pd.read_csv(input_dir / "phase8_repeated_test_subgroup_performance.csv")
    outer_cv = pd.read_csv(input_dir / "phase4_nested_outer_temporal_cv.csv")
    cluster_metrics = pd.read_csv(input_dir / "phase8_development_cluster_k_metrics.csv")
    cluster_stability = pd.read_csv(input_dir / "phase8_development_cluster_bootstrap_stability.csv")
    cluster_summary = pd.read_csv(input_dir / "phase8_development_cluster_summary.csv")
    jump_sensitivity = pd.read_csv(input_dir / "phase8_large_change_sensitivity.csv")
    feasibility = pd.read_csv(input_dir / "phase7_fib4_apri_input_feasibility.csv")
    fib4_benchmark = pd.read_csv(input_dir / "phase7_fib4_90day_continuous_benchmark.csv")
    context_summary = pd.read_csv(input_dir / "phase1_sex_and_diagnosis_evidence_summary.csv")
    context_summary["evidence_time_scope"] = np.where(
        context_summary["variable"].str.startswith("diagnosis_"),
        "ever recorded across the supplied workbook; may include post-index records",
        "patient-level demographic summary",
    )

    table1 = build_table1(tasks)
    test_patients = int(tasks.loc[tasks["cohort"].eq(PUBLIC_TEST_COHORT), "patient_key"].nunique())
    table2 = build_table2(
        performance,
        gains,
        test_patients,
        model_lock["selected_model"],
    )
    flow = build_flow_table(metadata, cohort_summary)
    missing_with_handling = add_missing_handling(missingness)
    write_csv(table1, output_dir / "table1_temporal_cohort_characteristics.csv")
    write_csv(table2, output_dir / "table2_test_model_and_baselines.csv")
    write_csv(
        pd.DataFrame(
            [
                {
                    "planned_table": "Threshold classification performance",
                    "status": "not_executed",
                    "reason": metadata["classification_gate_reason"],
                    "allowed_endpoint": "Continuous approximately 12-month LSM (kPa)",
                }
            ]
        ),
        output_dir / "table3_threshold_analysis_gate.csv",
    )
    write_csv(subgroup, output_dir / "table4_test_subgroup_performance.csv")
    write_csv(build_data_dictionary(), output_dir / "supplementary_table1_variable_dictionary.csv")
    write_csv(flow, output_dir / "supplementary_table2_data_flow.csv")
    write_csv(missing_with_handling, output_dir / "supplementary_table2_variable_missingness.csv")
    write_csv(outer_cv, output_dir / "supplementary_table3_nested_temporal_cv.csv")
    write_csv(feasibility, output_dir / "supplementary_table4_fib4_apri_feasibility.csv")
    write_csv(
        fib4_benchmark,
        output_dir / "supplementary_table4b_fib4_continuous_benchmark.csv",
    )
    write_csv(
        context_summary,
        output_dir / "supplementary_table7_diagnosis_evidence.csv",
    )
    write_csv(
        pd.DataFrame(
            [
                {"component": "k selection metrics", "file": "phase8_development_cluster_k_metrics.csv"},
                {"component": "bootstrap stability", "file": "phase8_development_cluster_bootstrap_stability.csv"},
                {"component": "continuous cluster summary", "file": "phase8_development_cluster_summary.csv"},
                {"component": "threshold association", "file": "not executed: clinical threshold gate closed"},
            ]
        ),
        output_dir / "supplementary_table5_cluster_analysis_index.csv",
    )
    write_csv(
        pd.DataFrame(
            [
                {"component": "slope threshold descriptions", "file": "phase8_slope_threshold_sensitivity.csv"},
                {"component": "large-change exclusion", "file": "phase8_large_change_sensitivity.csv"},
                {"component": "repeated-task history groups", "file": "phase8_repeated_test_subgroup_performance.csv"},
                {"component": "deidentified residual trajectories", "file": "phase8_largest_residual_deidentified_trajectories.csv"},
            ]
        ),
        output_dir / "supplementary_table6_sensitivity_index.csv",
    )

    publication_tables = write_publication_tables(
        article_dir,
        metadata,
        tasks,
        table2,
        subgroup,
        input_dir,
    )

    create_figures(
        metadata,
        cohort_summary,
        table2,
        predictions,
        subgroup,
        outer_cv,
        missingness,
        cluster_metrics,
        cluster_stability,
        jump_sensitivity,
        feasibility,
        fib4_benchmark,
        calibration,
        model_lock["selected_model"],
        output_dir,
        article_dir / "图片",
    )
    write_result_report(
        output_dir,
        metadata,
        cohort_summary,
        performance,
        gains,
        model_lock,
        subgroup,
        repeated_subgroup,
        cluster_summary,
        jump_sensitivity,
        feasibility,
        fib4_benchmark,
        calibration,
        context_summary,
    )
    write_manuscript_draft(
        article_dir,
        metadata,
        cohort_summary,
        performance,
        gains,
        model_lock,
        outer_cv,
        subgroup,
        repeated_subgroup,
        cluster_summary,
        feasibility,
        fib4_benchmark,
        calibration,
        context_summary,
    )
    write_response_draft(
        article_dir,
        metadata,
        cohort_summary,
        feasibility,
        model_lock,
        performance,
        gains,
        repeated_subgroup,
        cluster_summary,
        fib4_benchmark,
        context_summary,
    )
    write_article_asset_index(article_dir, metadata, publication_tables)
    append_csv_provenance(output_dir, metadata, tasks)
    write_result_catalog(output_dir, metadata, tasks)
    last_lsm_gain = gains.loc[
        gains["evaluation_set"].eq(PUBLIC_TEST_COHORT)
        & gains["baseline_method"].eq("last_observed_lsm")
    ].iloc[0]
    incremental_value_supported = _incremental_value_supported(last_lsm_gain)
    write_json(
        {
            "primary_test_cohort": PUBLIC_TEST_COHORT,
            "primary_test_patients": test_patients,
            "selected_model": model_lock["selected_model"],
            "classification_executed": False,
            "incremental_value_supported": incremental_value_supported,
            "decision_rule": (
                "Incremental value requires the last-LSM-minus-ML paired MAE-gain "
                "95% CI to be entirely above zero."
            ),
            "manuscript_positioning": (
                "Continuous-LSM temporal benchmarking study; F2 and pathological staging "
                "are withdrawn, with no deployment claim."
            ),
        },
        output_dir / "analysis_run_summary.json",
    )
    write_release_manifest(output_dir, metadata)
    print("Publication tables, figures, result report, and revision drafts written.")


if __name__ == "__main__":
    main()
