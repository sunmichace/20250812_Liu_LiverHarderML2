"""Build de-identified visits and fixed-horizon tasks from the final workbook."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from analysis_core import (
    ANALYSIS_PROTOCOL_VERSION,
    AnalysisSettings,
    CONFIRMED_CLINICAL_UNITS,
    INCLUSION_RULE_VERSION,
    SOURCE_WORKBOOK_UPDATE_CUTOFF,
    build_candidate_tasks,
    build_visits,
    cohort_summary,
    ensure_result_dir,
    load_raw_workbook,
    missingness_by_cohort,
    patient_context_summary,
    patient_overlap_check,
    primary_feature_columns,
    select_analysis_tasks,
    select_repeated_task_sensitivity,
    software_versions,
    write_csv,
    write_json,
)


def _sha256_file(path: Path) -> str:
    """Hash the source workbook without materializing it a second time in memory."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare de-identified longitudinal LSM visits and fixed-horizon tasks."
    )
    parser.add_argument("--input", default=None, help="Path to the final de-identified Excel workbook.")
    parser.add_argument("--output-dir", default=None, help="Directory for third-analysis results.")
    parser.add_argument(
        "--enable-threshold-classification",
        action="store_true",
        help=(
            "Disabled by protocol: F2 and threshold classification were withdrawn because "
            "the supplied data contain no clinical reference endpoint."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.enable_threshold_classification:
        raise RuntimeError(
            "Threshold classification is disabled in this revision. The F2 endpoint was "
            "withdrawn because no clinical or pathological reference endpoint was supplied."
        )
    settings = AnalysisSettings(
        data_path=Path(args.input) if args.input else AnalysisSettings.data_path,
        result_dir=Path(args.output_dir) if args.output_dir else AnalysisSettings.result_dir,
        classification_enabled=bool(args.enable_threshold_classification),
    )
    output_dir = ensure_result_dir(settings.result_dir)
    input_sha256 = _sha256_file(settings.data_path)

    raw = load_raw_workbook(settings)
    visits, visit_audit = build_visits(raw, settings)
    # The raw frame includes identifiers only while visits/tasks are derived;
    # release it before writing any output to minimize sensitive-data lifetime.
    del raw
    candidates = build_candidate_tasks(visits, settings)
    main_tasks, primary_test_cohort, task_audit = select_analysis_tasks(
        candidates, settings
    )
    repeated_sensitivity = select_repeated_task_sensitivity(candidates, main_tasks, settings)

    write_csv(visits, output_dir / "phase0_deidentified_visits.csv")
    updated_field_rows = [
        {"field": field, **availability}
        for field, availability in visit_audit.get("updated_workbook_field_availability", {}).items()
    ]
    write_csv(
        pd.DataFrame(updated_field_rows),
        output_dir / "phase0_updated_workbook_field_availability.csv",
    )
    write_csv(candidates, output_dir / "phase1_fixed_horizon_candidate_tasks.csv")
    write_csv(main_tasks, output_dir / "phase1_primary_tasks_one_per_patient.csv")
    # This legacy file represented an overlapping subset of the 2024 validation
    # set rather than an independent validation set.
    (output_dir / "phase1_latest_followup_sensitivity_tasks.csv").unlink(missing_ok=True)
    write_csv(
        repeated_sensitivity,
        output_dir / "phase1_repeated_tasks_sensitivity.csv",
    )
    write_csv(
        patient_context_summary(main_tasks),
        output_dir / "phase1_sex_and_diagnosis_evidence_summary.csv",
    )
    write_csv(cohort_summary(main_tasks), output_dir / "phase1_cohort_summary.csv")
    write_csv(
        missingness_by_cohort(main_tasks, primary_feature_columns()),
        output_dir / "phase1_feature_missingness_by_cohort.csv",
    )
    overlap = patient_overlap_check(main_tasks)
    write_csv(overlap, output_dir / "phase1_patient_overlap_check.csv")
    if not overlap.empty and overlap["overlapping_patients"].gt(0).any():
        raise RuntimeError("Patient overlap detected across primary temporal cohorts.")

    metadata = {
        "analysis_name": "third_analysis_temporal_lsm_revision",
        "analysis_protocol_version": ANALYSIS_PROTOCOL_VERSION,
        "inclusion_rule_version": INCLUSION_RULE_VERSION,
        "input_workbook": str(settings.data_path),
        "input_sheet": "Sheet1",
        "input_workbook_sha256": input_sha256,
        "input_workbook_bytes": int(settings.data_path.stat().st_size),
        "source_workbook_update_cutoff": SOURCE_WORKBOOK_UPDATE_CUTOFF,
        "source_workbook_update_statement": (
            "The source workbook is the 2026-08-19 revision. The actual latest usable "
            "LSM date is derived from its records and reported separately; this revision "
            "does not create an independent external-validation cohort."
        ),
        "confirmed_clinical_units": CONFIRMED_CLINICAL_UNITS,
        "unit_confirmation_basis": {
            "cap": (
                "The 8.19 workbook header explicitly identifies FibroScan CAP in dB/m."
            ),
            "alt_ast_total_bilirubin": (
                "The 8.19 workbook headers explicitly identify ALT/AST in U/L and total "
                "bilirubin in umol/L, reported typographically as μmol/L."
            ),
            "platelet": "The 8.19 workbook header explicitly identifies platelet count in 10^9/L.",
        },
        "multivalue_cell_rule": (
            "Comma-, Chinese-comma-, semicolon-, Chinese-semicolon-, or pipe-delimited "
            "measurements in one source cell are parsed separately and aggregated by median."
        ),
        "patient_identifier_handling": "Raw patient identifiers and names are not written to outputs; deterministic Pxxxxxx keys are used.",
        "endpoint": "Continuous LSM (kPa) at the closest visit 270-450 days after index, targeting 365 days.",
        "primary_history_requirement": "At least two historical LSM visits at or before index, so each primary task has at least three LSM visits including target.",
        "cohort_assignment": "One earliest eligible primary task per patient; assignment by its index year.",
        "data_split_labels": {
            "train": {"internal_cohort": "development_le_2022", "index_year": "<=2022"},
            "test": {"internal_cohort": "temporal_validation_2023", "index_year": "2023"},
            "validation": {"internal_cohort": "temporal_test_2024", "index_year": "2024"},
        },
        "primary_test_cohort": primary_test_cohort,
        "classification_enabled": settings.classification_enabled,
        "classification_gate_reason": settings.classification_gate_reason,
        "f2_endpoint_status": (
            "Withdrawn: no clinical or pathological reference standard was supplied; "
            "F2 is not an endpoint in this revision."
        ),
        "apri_gate_reason": settings.apri_gate_reason,
        "visit_audit": visit_audit,
        "task_audit": task_audit,
        "repeated_task_sensitivity": {
            "definition": "All eligible tasks retained only within each patient's originally assigned calendar cohort; not used for model selection or the primary result.",
            "patients": int(repeated_sensitivity["patient_key"].nunique()),
            "tasks": int(len(repeated_sensitivity)),
            "temporal_test_2024_patients": int(
                repeated_sensitivity.loc[
                    repeated_sensitivity["cohort"].eq("temporal_test_2024"), "patient_key"
                ].nunique()
            ),
            "temporal_test_2024_tasks": int(
                repeated_sensitivity["cohort"].eq("temporal_test_2024").sum()
            ),
        },
        "software_versions": software_versions(),
    }
    write_json(metadata, output_dir / "analysis_metadata.json")

    print(f"Prepared {len(visits):,} visits from {visits['patient_key'].nunique():,} patients.")
    print(f"Built {len(candidates):,} candidate fixed-horizon tasks.")
    print(f"Selected {len(main_tasks):,} primary tasks from {main_tasks['patient_key'].nunique():,} patients.")
    print(f"Primary test set: {primary_test_cohort}")
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
