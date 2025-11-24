#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速超参数调优测试：验证优化效果
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

plt.rcParams['font.family'] = 'DejaVu Sans'

print("========== 快速超参数调优测试 ==========")
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# 1. 加载数据
features_df = pd.read_csv('data/processed/comprehensive_features_final_new.csv')
current_model = joblib.load('data/processed/models/best_model_gradientboosting.pkl')

feature_cols = [col for col in features_df.columns if col not in ['patient_id', 'sample_id', 'target_liver_stiffness']]
X = features_df[feature_cols]
y = features_df['target_liver_stiffness']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 基准性能
baseline_pred = current_model.predict(X_test)
baseline_mae = mean_absolute_error(y_test, baseline_pred)
baseline_r2 = r2_score(y_test, baseline_pred)

print(f"\n基准性能: MAE={baseline_mae:.3f}, R²={baseline_r2:.3f}")

# 2. 快速参数搜索
print("\n执行快速参数优化...")

# 简化参数网格
param_grids = {
    'GradientBoosting': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.05, 0.1, 0.15],
        'max_depth': [4, 5, 6],
        'min_samples_split': [5, 10],
        'subsample': [0.8, 0.9, 1.0]
    },
    'RandomForest': {
        'n_estimators': [200, 300],
        'max_depth': [10, 15, 20],
        'min_samples_split': [5, 10],
        'min_samples_leaf': [2, 4]
    }
}

results = []

# GradientBoosting优化
print("优化GradientBoosting...")
gb = GradientBoostingRegressor(random_state=42)
gb_search = GridSearchCV(gb, param_grids['GradientBoosting'], cv=3, scoring='r2', n_jobs=-1)
gb_search.fit(X_train, y_train)

gb_pred = gb_search.predict(X_test)
gb_mae = mean_absolute_error(y_test, gb_pred)
gb_r2 = r2_score(y_test, gb_pred)

results.append({
    'Model': 'Optimized_GradientBoosting',
    'MAE': gb_mae,
    'R²': gb_r2,
    'Improvement(%)': ((gb_r2 - baseline_r2) / baseline_r2) * 100
})

print(f"最佳GB参数: {gb_search.best_params_}")
print(f"GB结果: MAE={gb_mae:.3f}, R²={gb_r2:.3f}")

# RandomForest优化
print("优化RandomForest...")
rf = RandomForestRegressor(random_state=42)
rf_search = GridSearchCV(rf, param_grids['RandomForest'], cv=3, scoring='r2', n_jobs=-1)
rf_search.fit(X_train, y_train)

rf_pred = rf_search.predict(X_test)
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)

results.append({
    'Model': 'Optimized_RandomForest',
    'MAE': rf_mae,
    'R²': rf_r2,
    'Improvement(%)': ((rf_r2 - baseline_r2) / baseline_r2) * 100
})

print(f"最佳RF参数: {rf_search.best_params_}")
print(f"RF结果: MAE={rf_mae:.3f}, R²={rf_r2:.3f}")

# 3. 简单集成
print("\n测试简单集成...")
ensemble_pred = (gb_pred + rf_pred) / 2
ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
ensemble_r2 = r2_score(y_test, ensemble_pred)

results.append({
    'Model': 'Simple_Ensemble',
    'MAE': ensemble_mae,
    'R²': ensemble_r2,
    'Improvement(%)': ((ensemble_r2 - baseline_r2) / baseline_r2) * 100
})

# 4. 结果汇总
results_df = pd.DataFrame([
    {'Model': 'Baseline_GradientBoosting', 'MAE': baseline_mae, 'R²': baseline_r2, 'Improvement(%)': 0.0}
] + results)

results_df = results_df.sort_values('R²', ascending=False)

print(f"\n========== 快速优化结果 ==========")
print(results_df.round(3))

# 5. 可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# R²对比
ax1 = axes[0]
bars = ax1.bar(range(len(results_df)), results_df['R²'], 
               color=['red' if 'Baseline' in model else 'skyblue' for model in results_df['Model']],
               edgecolor='black')
ax1.set_title('R² Comparison')
ax1.set_ylabel('R² Score')
ax1.set_xticks(range(len(results_df)))
ax1.set_xticklabels(results_df['Model'], rotation=45, ha='right')

for bar, r2 in zip(bars, results_df['R²']):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{r2:.3f}', ha='center', va='bottom', fontweight='bold')

# 改进百分比
ax2 = axes[1]
improvement_data = results_df[results_df['Model'] != 'Baseline_GradientBoosting']
bars = ax2.bar(range(len(improvement_data)), improvement_data['Improvement(%)'], 
               color='gold', edgecolor='black')
ax2.set_title('R² Improvement')
ax2.set_ylabel('Improvement (%)')
ax2.set_xticks(range(len(improvement_data)))
ax2.set_xticklabels(improvement_data['Model'], rotation=45, ha='right')

for bar, imp in zip(bars, improvement_data['Improvement(%)']):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{imp:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('optimization/quick_optimization_test.png', dpi=300, bbox_inches='tight')
plt.savefig('optimization/quick_optimization_test.pdf', dpi=300, bbox_inches='tight')
plt.show()

# 6. 保存最佳模型
best_model_name = results_df.iloc[0]['Model']
best_r2 = results_df.iloc[0]['R²']

print(f"\n最佳模型: {best_model_name}")
print(f"R²改进: {((best_r2 - baseline_r2) / baseline_r2) * 100:.1f}%")

if best_model_name == 'Optimized_GradientBoosting':
    best_model = gb_search.best_estimator_
elif best_model_name == 'Optimized_RandomForest':
    best_model = rf_search.best_estimator_
else:
    best_model = gb_search.best_estimator_  # 默认选择GB

# 保存
os.makedirs('optimization/models', exist_ok=True)
joblib.dump(best_model, 'optimization/models/quick_optimized_model.pkl')

if best_r2 > 0.4:
    print("🎉 已达到良好水平 (R² > 0.4)")
elif best_r2 > 0.35:
    print("✅ 达到可接受水平 (R² > 0.35)")
else:
    print("📈 有所改进，继续优化")

print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")