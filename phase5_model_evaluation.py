#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 5: 模型评估与验证
基于纵向历史数据预测最后一次肝脏弹性值的完整分析方案
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import TimeSeriesSplit, learning_curve
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Set English display
plt.rcParams['font.family'] = 'DejaVu Sans'

print("========== Phase 5: 模型评估与验证 ==========")
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("目标: 全面评估模型性能，验证预测能力")

# Step 1: 加载数据和模型
print("\n1. 加载数据和训练好的模型")

features_df = pd.read_csv('data/processed/comprehensive_features_final_new.csv')
longitudinal_df = pd.read_csv('data/processed/phase1_longitudinal_data.csv')
performance_df = pd.read_csv('data/processed/phase4_model_performance_comparison.csv')

# 准备数据
feature_cols = [col for col in features_df.columns if col not in ['patient_id', 'sample_id', 'target_liver_stiffness']]
X = features_df[feature_cols]
y = features_df['target_liver_stiffness']

print(f"数据集大小: {X.shape[0]} 样本, {X.shape[1]} 特征")
print(f"目标变量范围: [{y.min():.2f}, {y.max():.2f}] kPa")

# 加载最佳模型
best_model_name = performance_df.iloc[0]['Model'].lower()
best_model = joblib.load(f'data/processed/models/{best_model_name}_model.pkl')
print(f"最佳模型: {best_model_name}")

# Step 2: 时间序列交叉验证
print("\n2. 时间序列交叉验证")

def time_series_cv_evaluation(model, X, y, cv_splits=5):
    """时间序列交叉验证评估"""
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    
    cv_results = {
        'fold': [],
        'train_size': [],
        'test_size': [],
        'train_mae': [],
        'test_mae': [],
        'train_r2': [],
        'test_r2': []
    }
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train_cv, X_test_cv = X.iloc[train_idx], X.iloc[test_idx]
        y_train_cv, y_test_cv = y.iloc[train_idx], y.iloc[test_idx]
        
        # 训练模型
        model.fit(X_train_cv, y_train_cv)
        
        # 预测
        y_train_pred = model.predict(X_train_cv)
        y_test_pred = model.predict(X_test_cv)
        
        # 评估
        train_mae = mean_absolute_error(y_train_cv, y_train_pred)
        test_mae = mean_absolute_error(y_test_cv, y_test_pred)
        train_r2 = r2_score(y_train_cv, y_train_pred)
        test_r2 = r2_score(y_test_cv, y_test_pred)
        
        cv_results['fold'].append(fold + 1)
        cv_results['train_size'].append(len(train_idx))
        cv_results['test_size'].append(len(test_idx))
        cv_results['train_mae'].append(train_mae)
        cv_results['test_mae'].append(test_mae)
        cv_results['train_r2'].append(train_r2)
        cv_results['test_r2'].append(test_r2)
    
    return pd.DataFrame(cv_results)

cv_results_df = time_series_cv_evaluation(best_model, X, y)

print("时间序列交叉验证结果:")
print(cv_results_df.round(3))

print(f"\n交叉验证统计:")
print(f"测试MAE: {cv_results_df['test_mae'].mean():.3f} ± {cv_results_df['test_mae'].std():.3f}")
print(f"测试R²: {cv_results_df['test_r2'].mean():.3f} ± {cv_results_df['test_r2'].std():.3f}")

# Step 3: 学习曲线分析
print("\n3. 学习曲线分析")

def plot_learning_curve(model, X, y):
    """绘制学习曲线"""
    train_sizes = np.linspace(0.1, 1.0, 10)
    
    train_sizes_abs, train_scores, val_scores = learning_curve(
        model, X, y,
        train_sizes=train_sizes,
        cv=TimeSeriesSplit(n_splits=3),
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )
    
    train_scores_mean = -train_scores.mean(axis=1)
    train_scores_std = train_scores.std(axis=1)
    val_scores_mean = -val_scores.mean(axis=1)
    val_scores_std = val_scores.std(axis=1)
    
    return train_sizes_abs, train_scores_mean, train_scores_std, val_scores_mean, val_scores_std

train_sizes, train_mae_mean, train_mae_std, val_mae_mean, val_mae_std = plot_learning_curve(best_model, X, y)

# Step 4: 预测区间分析
print("\n4. 预测区间分析")

def calculate_prediction_intervals(y_true, y_pred, confidence=0.95):
    """计算预测区间"""
    residuals = y_true - y_pred
    residuals_std = np.std(residuals)
    
    # 基于残差分布的预测区间
    from scipy.stats import t
    n = len(residuals)
    t_val = t.ppf((1 + confidence) / 2, n - 1)
    
    margin = t_val * residuals_std
    lower_bound = y_pred - margin
    upper_bound = y_pred + margin
    
    # 计算区间覆盖率
    coverage = np.mean((y_true >= lower_bound) & (y_true <= upper_bound))
    
    return lower_bound, upper_bound, coverage, margin

# 使用交叉验证的最后一折进行预测区间分析
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

lower_bound, upper_bound, coverage, margin = calculate_prediction_intervals(y_test, y_pred)

print(f"预测区间分析:")
print(f"- 95%置信区间覆盖率: {coverage:.1%}")
print(f"- 平均预测区间宽度: {2 * margin:.2f} kPa")
print(f"- 区间半宽: {margin:.2f} kPa")

# Step 5: 临床相关指标评估
print("\n5. 临床相关指标评估")

def clinical_evaluation(y_true, y_pred):
    """临床相关指标评估"""
    
    # 基本误差指标
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # 临床阈值准确性
    thresholds = [1.0, 1.5, 2.0, 3.0]  # kPa
    accuracy_within_threshold = {}
    
    for threshold in thresholds:
        within_threshold = np.abs(y_true - y_pred) <= threshold
        accuracy = np.mean(within_threshold)
        accuracy_within_threshold[f'±{threshold} kPa'] = accuracy
    
    # 纤维化分级预测准确性
    def liver_stiffness_grade(values):
        """肝脏纤维化分级"""
        grades = np.zeros_like(values)
        grades[(values >= 7) & (values < 12.5)] = 1  # F1
        grades[(values >= 12.5) & (values < 17.5)] = 2  # F2-F3
        grades[values >= 17.5] = 3  # F4
        return grades
    
    true_grades = liver_stiffness_grade(y_true)
    pred_grades = liver_stiffness_grade(y_pred)
    grade_accuracy = np.mean(true_grades == pred_grades)
    
    # 高风险患者识别
    high_risk_threshold = 12.5  # kPa
    true_high_risk = y_true >= high_risk_threshold
    pred_high_risk = y_pred >= high_risk_threshold
    
    from sklearn.metrics import precision_score, recall_score, f1_score
    precision = precision_score(true_high_risk, pred_high_risk, zero_division=0)
    recall = recall_score(true_high_risk, pred_high_risk, zero_division=0)
    f1 = f1_score(true_high_risk, pred_high_risk, zero_division=0)
    
    results = {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'grade_accuracy': grade_accuracy,
        'high_risk_precision': precision,
        'high_risk_recall': recall,
        'high_risk_f1': f1,
        'accuracy_within_threshold': accuracy_within_threshold
    }
    
    return results

clinical_results = clinical_evaluation(y_test.values, y_pred)

print("临床评估结果:")
print(f"- MAE: {clinical_results['mae']:.3f} kPa")
print(f"- RMSE: {clinical_results['rmse']:.3f} kPa") 
print(f"- MAPE: {clinical_results['mape']:.1f}%")
print(f"- 纤维化分级准确率: {clinical_results['grade_accuracy']:.1%}")

print(f"\n误差阈值准确率:")
for threshold, accuracy in clinical_results['accuracy_within_threshold'].items():
    print(f"- {threshold}: {accuracy:.1%}")

print(f"\n高风险患者识别:")
print(f"- 精确率: {clinical_results['high_risk_precision']:.3f}")
print(f"- 召回率: {clinical_results['high_risk_recall']:.3f}")
print(f"- F1得分: {clinical_results['high_risk_f1']:.3f}")

# Step 6: 不同患者亚组评估
print("\n6. 不同患者亚组评估")

def subgroup_analysis(features_df, y_true, y_pred):
    """不同亚组的预测性能分析"""
    
    results_df = pd.DataFrame({
        'patient_id': features_df['patient_id'],
        'y_true': y_true,
        'y_pred': y_pred,
        'abs_error': np.abs(y_true - y_pred)
    })
    
    # 添加基线特征
    baseline_features = ['baseline_value', 'hist_count', 'total_follow_up_years']
    for feature in baseline_features:
        if feature in features_df.columns:
            results_df[feature] = features_df[feature].values
    
    subgroup_results = {}
    
    # 按基线肝脏弹性值分组
    if 'baseline_value' in results_df.columns:
        baseline_groups = pd.cut(results_df['baseline_value'], 
                               bins=[0, 7, 12.5, 20, 100], 
                               labels=['Normal(<7)', 'Mild(7-12.5)', 'Moderate(12.5-20)', 'Severe(>20)'])
        results_df['baseline_group'] = baseline_groups
        
        for group in baseline_groups.cat.categories:
            group_data = results_df[results_df['baseline_group'] == group]
            if len(group_data) > 0:
                subgroup_results[f'Baseline_{group}'] = {
                    'count': len(group_data),
                    'mae': group_data['abs_error'].mean(),
                    'r2': r2_score(group_data['y_true'], group_data['y_pred']) if len(group_data) > 1 else 0
                }
    
    # 按历史记录数分组
    if 'hist_count' in results_df.columns:
        count_groups = pd.cut(results_df['hist_count'], 
                            bins=[1, 2, 3, 5, 100], 
                            labels=['2_times', '3_times', '4-5_times', '>5_times'])
        results_df['count_group'] = count_groups
        
        for group in count_groups.cat.categories:
            group_data = results_df[results_df['count_group'] == group]
            if len(group_data) > 0:
                subgroup_results[f'Records_{group}'] = {
                    'count': len(group_data),
                    'mae': group_data['abs_error'].mean(),
                    'r2': r2_score(group_data['y_true'], group_data['y_pred']) if len(group_data) > 1 else 0
                }
    
    return subgroup_results

# 获取测试集对应的特征数据
test_indices = X_test.index
test_features = features_df.loc[test_indices]

subgroup_results = subgroup_analysis(test_features, y_test.values, y_pred)

print("不同患者亚组预测性能:")
for group, metrics in subgroup_results.items():
    print(f"{group}: n={metrics['count']}, MAE={metrics['mae']:.3f}, R²={metrics['r2']:.3f}")

# Step 7: 综合可视化
print("\n7. 生成综合评估可视化")

fig, axes = plt.subplots(3, 2, figsize=(16, 18))

# 7.1 Cross-validation results
ax = axes[0, 0]
x_pos = np.arange(len(cv_results_df))
ax.bar(x_pos - 0.2, cv_results_df['train_mae'], 0.4, label='Training MAE', alpha=0.7)
ax.bar(x_pos + 0.2, cv_results_df['test_mae'], 0.4, label='Validation MAE', alpha=0.7)
ax.set_xlabel('Cross-Validation Fold')
ax.set_ylabel('MAE (kPa)')
ax.set_title('Time Series Cross-Validation Results')
ax.set_xticks(x_pos)
ax.set_xticklabels([f'Fold {i}' for i in cv_results_df['fold']])
ax.legend()
ax.grid(True, alpha=0.3)

# 7.2 Learning curve
ax = axes[0, 1]
ax.plot(train_sizes, train_mae_mean, 'o-', label='Training MAE', linewidth=2)
ax.fill_between(train_sizes, train_mae_mean - train_mae_std, 
                train_mae_mean + train_mae_std, alpha=0.1)
ax.plot(train_sizes, val_mae_mean, 'o-', label='Validation MAE', linewidth=2)
ax.fill_between(train_sizes, val_mae_mean - val_mae_std, 
                val_mae_mean + val_mae_std, alpha=0.1)
ax.set_xlabel('Training Sample Size')
ax.set_ylabel('MAE (kPa)')
ax.set_title('Learning Curve')
ax.legend()
ax.grid(True, alpha=0.3)

# 7.3 Prediction intervals visualization
ax = axes[1, 0]
# Select subset of samples for display
sample_indices = np.random.choice(len(y_test), min(50, len(y_test)), replace=False)
sample_y_test = y_test.iloc[sample_indices]
sample_y_pred = y_pred[sample_indices]
sample_lower = lower_bound[sample_indices]
sample_upper = upper_bound[sample_indices]

sorted_indices = np.argsort(sample_y_test)
ax.fill_between(range(len(sorted_indices)), 
               sample_lower[sorted_indices], 
               sample_upper[sorted_indices], 
               alpha=0.3, label='95% Prediction Interval')
ax.plot(range(len(sorted_indices)), sample_y_test.iloc[sorted_indices], 'ro', label='Actual Values', markersize=6)
ax.plot(range(len(sorted_indices)), sample_y_pred[sorted_indices], 'bo', label='Predicted Values', markersize=6)
ax.set_xlabel('Samples (sorted by actual values)')
ax.set_ylabel('Liver Stiffness (kPa)')
ax.set_title('Prediction Intervals Display')
ax.legend()
ax.grid(True, alpha=0.3)

# 7.4 Error distribution analysis
ax = axes[1, 1]
errors = y_test.values - y_pred
ax.hist(errors, bins=25, alpha=0.7, edgecolor='black', density=True)
ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error Line')
ax.axvline(x=np.mean(errors), color='orange', linestyle='--', linewidth=2, label=f'Mean Error ({np.mean(errors):.2f})')
ax.set_xlabel('Prediction Error (Actual - Predicted) kPa')
ax.set_ylabel('Density')
ax.set_title('Prediction Error Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

# 7.5 Clinical threshold accuracy
ax = axes[2, 0]
thresholds = list(clinical_results['accuracy_within_threshold'].keys())
accuracies = list(clinical_results['accuracy_within_threshold'].values())
bars = ax.bar(thresholds, accuracies, alpha=0.7)
ax.set_ylabel('Accuracy Rate')
ax.set_title('Accuracy at Different Error Thresholds')
ax.grid(True, alpha=0.3)

# Add value labels
for bar, acc in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
           f'{acc:.1%}', ha='center', va='bottom')

# 7.6 Subgroup performance comparison
ax = axes[2, 1]
if subgroup_results:
    groups = list(subgroup_results.keys())
    mae_values = [subgroup_results[group]['mae'] for group in groups]
    
    bars = ax.bar(range(len(groups)), mae_values, alpha=0.7)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=45, ha='right')
    ax.set_ylabel('MAE (kPa)')
    ax.set_title('Prediction Performance by Patient Subgroups')
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(mae_values):
        ax.text(i, v + 0.05, f'{v:.2f}', ha='center', va='bottom')

plt.tight_layout()
# Save both PNG and PDF formats
plt.savefig('data/processed/phase5_model_evaluation_comprehensive.png', dpi=300, bbox_inches='tight')
plt.savefig('data/processed/phase5_model_evaluation_comprehensive.pdf', dpi=300, bbox_inches='tight')
plt.show()

# Step 8: 保存评估结果
print("\n8. 保存评估结果")

# 保存交叉验证结果
cv_results_df.to_csv('data/processed/phase5_cross_validation_results.csv', index=False)

# 保存临床评估结果
clinical_results_df = pd.DataFrame([clinical_results])
clinical_results_df.to_csv('data/processed/phase5_clinical_evaluation_results.csv', index=False)

# 保存亚组分析结果
if subgroup_results:
    subgroup_df = pd.DataFrame(subgroup_results).T
    subgroup_df.to_csv('data/processed/phase5_subgroup_analysis_results.csv')

print("保存的文件:")
print("- data/processed/phase5_cross_validation_results.csv: 交叉验证结果")
print("- data/processed/phase5_clinical_evaluation_results.csv: 临床评估结果")
print("- data/processed/phase5_subgroup_analysis_results.csv: 亚组分析结果")
print("- data/processed/phase5_model_evaluation_comprehensive.png: 综合评估图表")

# Step 9: 模型解释性分析
print("\n9. 模型解释性分析")

# 特征重要性总结
feature_importance_df = pd.read_csv('data/processed/phase4_feature_importance_ranking.csv')
top_features = feature_importance_df.head(10)

print("模型解释性:")
print("最重要的预测因子:")
for i, (_, row) in enumerate(top_features.iterrows()):
    print(f"{i+1:2d}. {row['feature']}: {row['importance']:.4f}")

# 预测可靠性评估
reliability_analysis = {
    'high_confidence': np.sum(np.abs(y_test.values - y_pred) <= 1.0),
    'medium_confidence': np.sum((np.abs(y_test.values - y_pred) > 1.0) & (np.abs(y_test.values - y_pred) <= 2.0)),
    'low_confidence': np.sum(np.abs(y_test.values - y_pred) > 2.0),
    'total_predictions': len(y_test)
}

print(f"\n预测可靠性分析:")
print(f"- 高置信度 (误差≤1.0 kPa): {reliability_analysis['high_confidence']}/{reliability_analysis['total_predictions']} ({reliability_analysis['high_confidence']/reliability_analysis['total_predictions']:.1%})")
print(f"- 中等置信度 (误差1.0-2.0 kPa): {reliability_analysis['medium_confidence']}/{reliability_analysis['total_predictions']} ({reliability_analysis['medium_confidence']/reliability_analysis['total_predictions']:.1%})")
print(f"- 低置信度 (误差>2.0 kPa): {reliability_analysis['low_confidence']}/{reliability_analysis['total_predictions']} ({reliability_analysis['low_confidence']/reliability_analysis['total_predictions']:.1%})")

# Step 10: 最终评估报告
print("\n========== Phase 5 模型评估与验证摘要报告 ==========")

print(f"\n模型性能总览:")
print(f"- 最佳模型: {best_model_name.upper()}")
print(f"- 交叉验证MAE: {cv_results_df['test_mae'].mean():.3f} ± {cv_results_df['test_mae'].std():.3f} kPa")
print(f"- 交叉验证R²: {cv_results_df['test_r2'].mean():.3f} ± {cv_results_df['test_r2'].std():.3f}")

print(f"\n临床实用性:")
print(f"- ±1.0 kPa准确率: {clinical_results['accuracy_within_threshold']['±1.0 kPa']:.1%}")
print(f"- ±2.0 kPa准确率: {clinical_results['accuracy_within_threshold']['±2.0 kPa']:.1%}")
print(f"- 纤维化分级准确率: {clinical_results['grade_accuracy']:.1%}")
print(f"- 高风险患者识别F1: {clinical_results['high_risk_f1']:.3f}")

print(f"\n预测区间评估:")
print(f"- 95%置信区间覆盖率: {coverage:.1%}")
print(f"- 平均区间宽度: {2 * margin:.2f} kPa")

print(f"\n模型可靠性:")
print(f"- 高置信度预测: {reliability_analysis['high_confidence']/reliability_analysis['total_predictions']:.1%}")
print(f"- 可接受误差预测 (≤2.0 kPa): {(reliability_analysis['high_confidence'] + reliability_analysis['medium_confidence'])/reliability_analysis['total_predictions']:.1%}")

print(f"\n关键预测因子:")
for i, (_, row) in enumerate(top_features.head(5).iterrows()):
    print(f"- {row['feature']}")

print(f"\n适用性评估:")
if subgroup_results:
    best_subgroup = min(subgroup_results.keys(), key=lambda x: subgroup_results[x]['mae'])
    worst_subgroup = max(subgroup_results.keys(), key=lambda x: subgroup_results[x]['mae'])
    print(f"- 表现最佳亚组: {best_subgroup} (MAE: {subgroup_results[best_subgroup]['mae']:.3f})")
    print(f"- 表现最差亚组: {worst_subgroup} (MAE: {subgroup_results[worst_subgroup]['mae']:.3f})")

print(f"\n完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("Phase 5 模型评估与验证完成！")