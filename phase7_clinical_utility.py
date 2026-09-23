"""Record the clinical-classification gate and score-completeness feasibility.

This revision does not manufacture pathological F2 labels from LSM. The F2
endpoint is withdrawn because no clinical or pathological reference endpoint
was supplied, so the script records an explicit no-analysis decision.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from analysis_core import (
    CONFIRMED_CLINICAL_UNITS,
    PUBLIC_TEST_COHORT,
    RANDOM_SEED,
    ensure_result_dir,
    write_csv,
    write_json,
)
from modeling_core import (
    PRESPECIFIED_VALID_RANGES,
    bootstrap_mae_gain,
    bootstrap_metric_intervals,
)


FIB4_LOOKBACK_WINDOWS_DAYS = [0, 30, 90, 180, 365, 730, 3650]
FIB4_BENCHMARK_WINDOW_DAYS = 90


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Document clinical threshold-analysis eligibility.")
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument(
        "--enable-threshold-classification",
        action="store_true",
        help="Reserved for a future run after required clinical metadata are independently verified.",
    )
    return parser


def _default_result_dir() -> Path:
    return Path(__file__).resolve().parent / "result"


def _fib4_values(tasks: pd.DataFrame, lookback_days: int) -> tuple[pd.Series, pd.Series]:
    numeric = {
        column: pd.to_numeric(tasks[column], errors="coerce")
        for column in [
            "age",
            "alt",
            "ast",
            "platelet",
            "alt_lag_days",
            "ast_lag_days",
            "platelet_lag_days",
        ]
    }
    complete = (
        numeric["age"].between(*PRESPECIFIED_VALID_RANGES["age"])
        & numeric["alt"].gt(0)
        & numeric["alt"].le(PRESPECIFIED_VALID_RANGES["alt"][1])
        & numeric["ast"].gt(0)
        & numeric["ast"].le(PRESPECIFIED_VALID_RANGES["ast"][1])
        & numeric["platelet"].between(*PRESPECIFIED_VALID_RANGES["platelet"])
        & numeric["alt_lag_days"].between(0, lookback_days)
        & numeric["ast_lag_days"].between(0, lookback_days)
        & numeric["platelet_lag_days"].between(0, lookback_days)
    )
    fib4 = pd.Series(np.nan, index=tasks.index, dtype=float)
    fib4.loc[complete] = (
        numeric["age"].loc[complete] * numeric["ast"].loc[complete]
    ) / (
        numeric["platelet"].loc[complete]
        * np.sqrt(numeric["alt"].loc[complete])
    )
    return complete, fib4


def build_fib4_feasibility(tasks: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    rows = []
    benchmark_values = pd.Series(np.nan, index=tasks.index, dtype=float)
    for lookback_days in FIB4_LOOKBACK_WINDOWS_DAYS:
        complete, fib4 = _fib4_values(tasks, lookback_days)
        if lookback_days == FIB4_BENCHMARK_WINDOW_DAYS:
            benchmark_values = fib4
        for cohort, frame in tasks.groupby("cohort", sort=False):
            cohort_complete = complete.loc[frame.index]
            cohort_fib4 = fib4.loc[frame.index].dropna()
            apri_inputs = (
                pd.to_numeric(frame["ast"], errors="coerce").gt(0)
                & pd.to_numeric(frame["platelet"], errors="coerce").gt(0)
                & pd.to_numeric(frame["ast_lag_days"], errors="coerce").between(0, lookback_days)
                & pd.to_numeric(frame["platelet_lag_days"], errors="coerce").between(0, lookback_days)
            )
            rows.append(
                {
                    "cohort": cohort,
                    "lookback_window_days": lookback_days,
                    "patients": int(frame["patient_key"].nunique()),
                    "tasks": int(len(frame)),
                    "age_alt_ast_platelet_complete_n": int(cohort_complete.sum()),
                    "age_alt_ast_platelet_complete_pct": 100 * float(cohort_complete.mean()),
                    "fib4_median": float(cohort_fib4.median()) if len(cohort_fib4) else np.nan,
                    "fib4_q1": float(cohort_fib4.quantile(0.25)) if len(cohort_fib4) else np.nan,
                    "fib4_q3": float(cohort_fib4.quantile(0.75)) if len(cohort_fib4) else np.nan,
                    "ast_platelet_complete_n": int(apri_inputs.sum()),
                    "ast_platelet_complete_pct": 100 * float(apri_inputs.mean()),
                    "platelet_missing_n": int(frame["platelet"].isna().sum()),
                    "platelet_missing_pct": 100 * float(frame["platelet"].isna().mean()),
                }
            )
    return pd.DataFrame(rows), benchmark_values


def run_fib4_benchmark(
    tasks: pd.DataFrame,
    fib4: pd.Series,
    predictions: pd.DataFrame,
    bootstrap_replicates: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float | int | str]]:
    frame = tasks.copy()
    frame["fib4_90day"] = fib4
    development = frame.loc[
        frame["cohort"].eq("development_le_2022") & frame["fib4_90day"].notna()
    ].copy()
    test = frame.loc[
        frame["cohort"].eq(PUBLIC_TEST_COHORT) & frame["fib4_90day"].notna()
    ].copy()
    if len(development) < 20 or len(test) < 20:
        raise RuntimeError("FIB-4 benchmark requires at least 20 complete patients in train and test sets.")

    model = LinearRegression()
    model.fit(development[["fib4_90day"]], development["target_lsm"])
    test_prediction = predictions.loc[
        predictions["evaluation_set"].eq(PUBLIC_TEST_COHORT),
        [
            "patient_key",
            "index_date",
            "target_date",
            "target_lsm",
            "last_observed_lsm",
            "locked_ml_model",
        ],
    ]
    output = test[
        ["patient_key", "index_date", "target_date", "target_lsm", "fib4_90day"]
    ].merge(
        test_prediction,
        on=["patient_key", "index_date", "target_date", "target_lsm"],
        how="left",
        validate="one_to_one",
    )
    output["fib4_development_linear_mapping"] = model.predict(output[["fib4_90day"]])
    methods = [
        "fib4_development_linear_mapping",
        "last_observed_lsm",
        "locked_ml_model",
    ]
    metrics = bootstrap_metric_intervals(
        output,
        methods,
        n_bootstrap=bootstrap_replicates,
        seed=RANDOM_SEED + 700,
    )
    benchmark_evaluation_set = f"{PUBLIC_TEST_COHORT}_fib4_90day_complete_case"
    metrics["evaluation_set"] = benchmark_evaluation_set
    gains = bootstrap_mae_gain(
        output,
        "locked_ml_model",
        ["fib4_development_linear_mapping", "last_observed_lsm"],
        n_bootstrap=bootstrap_replicates,
        seed=RANDOM_SEED + 701,
    )
    gains["evaluation_set"] = benchmark_evaluation_set
    details: dict[str, float | int | str] = {
        "lookback_window_days": FIB4_BENCHMARK_WINDOW_DAYS,
        "development_complete_patients": int(development["patient_key"].nunique()),
        "test_complete_patients": int(test["patient_key"].nunique()),
        "linear_mapping_intercept": float(model.intercept_),
        "linear_mapping_slope": float(model.coef_[0]),
        "fit_data": "train set only (internal cohort key: development_le_2022)",
        "evaluation_data": "2023 public test-set complete cases only (internal cohort key: temporal_validation_2023)",
    }
    return output, metrics, gains, details


def main() -> None:
    args = build_parser().parse_args()
    input_dir = Path(args.input_dir) if args.input_dir else _default_result_dir()
    output_dir = ensure_result_dir(Path(args.output_dir) if args.output_dir else _default_result_dir())
    tasks = pd.read_csv(input_dir / "phase1_primary_tasks_one_per_patient.csv")
    metadata = json.loads((input_dir / "analysis_metadata.json").read_text(encoding="utf-8"))
    if args.enable_threshold_classification:
        raise RuntimeError(
            "Threshold classification is disabled: the F2 endpoint was withdrawn because no "
            "clinical or pathological reference endpoint was supplied."
        )
    if metadata.get("classification_enabled"):
        raise RuntimeError("Classification-enabled metadata is inconsistent with the current clinical gate.")

    feasibility, fib4_90day = build_fib4_feasibility(tasks)
    write_csv(feasibility, output_dir / "phase7_fib4_apri_input_feasibility.csv")
    predictions = pd.read_csv(input_dir / "phase5_locked_temporal_predictions.csv")
    benchmark_predictions, benchmark_metrics, benchmark_gains, benchmark_details = run_fib4_benchmark(
        tasks,
        fib4_90day,
        predictions,
        args.bootstrap_replicates,
    )
    write_csv(
        benchmark_predictions,
        output_dir / "phase7_fib4_90day_complete_case_predictions.csv",
    )
    write_csv(
        benchmark_metrics,
        output_dir / "phase7_fib4_90day_continuous_benchmark.csv",
    )
    write_csv(
        benchmark_gains,
        output_dir / "phase7_fib4_90day_benchmark_mae_gain.csv",
    )

    gate = {
        "classification_executed": False,
        "requested_enable_flag": False,
        "decision": "Do not run LSM-threshold or F2 classification, classification calibration, ROC/PR curves, or decision-curve analysis in this run.",
        "rationale": [
            "The F2 endpoint was withdrawn because no clinical or pathological reference standard was supplied.",
            "The 8.19 workbook supplies nonexclusive diagnosis evidence but not a clinically adjudicated mutually exclusive etiology, so a common TE threshold cannot be assumed.",
            "No etiology-specific FibroScan transient-elastography guideline or validation reference fixing 7 and 12.5 kPa was supplied before analysis.",
            "TE probe and acquisition-quality metadata remain unconfirmed.",
        ],
        "allowed_current_endpoint": "Continuous approximately 12-month future LSM (kPa).",
        "forbidden_current_terms": [
            "F2",
            ">=F2",
            "significant fibrosis",
            "fibrosis staging accuracy",
            "pathological fibrosis classification",
        ],
        "f2_endpoint_status": "Withdrawn; do not restore F2 or pathological-fibrosis claims in this revision.",
        "future_nonpathological_option": "Only a separately prespecified nonpathological LSM risk-stratum study could be considered after written confirmation of etiology, TE conditions, and a matching threshold reference; it must not be called F2.",
        "confirmed_units": CONFIRMED_CLINICAL_UNITS,
        "fib4_apri": {
            "fib4_calculated": "FIB-4 was calculated descriptively when age, ALT, AST, and platelet count were complete within fixed lookback windows; no FIB-4 threshold was used as F2 truth.",
            "fib4_formula": "age (years) * AST (U/L) / [platelet (10^9/L) * sqrt(ALT (U/L))]",
            "fib4_formula_source": "Sterling et al., Hepatology 2006;43:1317-1325, doi:10.1002/hep.21178",
            "continuous_benchmark": benchmark_details,
            "apri_not_calculated": "The local AST upper limit of normal was not supplied; APRI cannot be calculated reproducibly.",
        },
    }
    write_json(gate, output_dir / "phase7_clinical_endpoint_gate.json")

    test_90day = feasibility.loc[
        feasibility["cohort"].eq(PUBLIC_TEST_COHORT)
        & feasibility["lookback_window_days"].eq(FIB4_BENCHMARK_WINDOW_DAYS)
    ].iloc[0]
    fib4_mae = benchmark_metrics.loc[
        benchmark_metrics["method"].eq("fib4_development_linear_mapping")
        & benchmark_metrics["metric"].eq("mae")
    ].iloc[0]
    report = f"""# 临床分类分析门控记录

## 本次执行结论

本次第三次分析**未执行** 7 kPa、12.5 kPa 的二分类/三分类，未计算 AUROC、AUPRC、分类校准、决策曲线或所谓 F2 分期准确率。F2 终点因未提供临床或病理参考标准而已撤销；连续约 12 个月 LSM（kPa）是本次唯一的主终点。

这不是遗漏：当前 Excel 未提供临床或病理参考标准，不能定义 F2。任何数值 LSM 阈值也不能被写成病理分期。

## 缺少的确认

1. 队列诊断证据虽已提取，但仍需临床团队完成互斥病因判定并确认是否为纯 CHB 队列。
2. 与该病因、FibroScan Touch 502 TE 条件匹配的、分析前固定的阈值文献或指南。
3. TE 探头、有效测量次数、success rate、IQR/median 和禁食信息。
4. APRI 所需的本地 AST 正常值上限。

8.19工作簿已明确CAP为dB/m、ALT/AST为U/L、总胆红素为μmol/L、血小板为10^9/L；这些单位不再列为待确认项。

## FIB-4辅助分析

按固定回溯窗口报告FIB-4输入完整性。90天窗口下，2023测试集有{int(test_90day['age_alt_ast_platelet_complete_n'])}/{int(test_90day['patients'])}名患者可计算FIB-4。使用仅在训练集拟合的线性映射后，该完整病例子集的FIB-4辅助连续预测MAE为{fib4_mae['estimate']:.3f} kPa（95% CI {fib4_mae['ci_2_5']:.3f}-{fib4_mae['ci_97_5']:.3f}）。该分析只用于与连续未来LSM任务作辅助比较，不使用FIB-4阈值、不定义F2，也不将FIB-4当作病理金标准。

FIB-4公式依据Sterling等2006年原始研究（doi:10.1002/hep.21178）。APRI因缺少本地AST正常值上限仍未计算。
"""
    (output_dir / "phase7_clinical_classification_not_executed.md").write_text(report, encoding="utf-8")
    print("Clinical threshold classification was intentionally not executed; gate record written.")


if __name__ == "__main__":
    main()
