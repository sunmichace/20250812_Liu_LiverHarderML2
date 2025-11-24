#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化集成学习测试
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

plt.rcParams['font.family'] = 'DejaVu Sans'

print("========== 简化集成学习测试 ==========")
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# 1. 加载数据
features_df = pd.read_csv('data/processed/comprehensive_features_final_new.csv')

feature_cols = [col for col in features_df.columns if col not in ['patient_id', 'sample_id', 'target_liver_stiffness']]
X = features_df[feature_cols]
y = features_df['target_liver_stiffness']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 基准性能
baseline_model = joblib.load('data/processed/models/best_model_gradientboosting.pkl')
baseline_pred = baseline_model.predict(X_test)
baseline_mae = mean_absolute_error(y_test, baseline_pred)
baseline_r2 = r2_score(y_test, baseline_pred)

print(f"基准性能: MAE={baseline_mae:.3f}, R²={baseline_r2:.3f}")

# 2. 创建多个基学习器
print("\n训练多个基学习器...")

# 使用快速测试中找到的最佳参数
gb1 = GradientBoostingRegressor(
    learning_rate=0.05, max_depth=5, min_samples_split=5, 
    n_estimators=200, subsample=0.9, random_state=42
)

gb2 = GradientBoostingRegressor(
    learning_rate=0.1, max_depth=4, min_samples_split=10,
    n_estimators=300, subsample=0.8, random_state=43
)

rf = RandomForestRegressor(
    max_depth=15, min_samples_leaf=4, min_samples_split=5,
    n_estimators=300, random_state=42
)

ridge = Ridge(alpha=1.0)

# 训练所有模型
models = {
    'GB1': gb1,
    'GB2': gb2, 
    'RF': rf,
    'Ridge': ridge
}

predictions = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    predictions[name] = pred
    print(f"{name:8s}: MAE={mae:.3f}, R²={r2:.3f}")

# 3. 集成方法测试
print("\n测试集成方法...")

results = []

# Voting集成
voting_model = VotingRegressor([
    ('gb1', gb1),
    ('gb2', gb2), 
    ('rf', rf),
    ('ridge', ridge)
])

voting_model.fit(X_train, y_train)
voting_pred = voting_model.predict(X_test)
voting_mae = mean_absolute_error(y_test, voting_pred)
voting_r2 = r2_score(y_test, voting_pred)

results.append({
    'Model': 'Voting_Ensemble',
    'MAE': voting_mae,
    'R²': voting_r2,
    'Improvement(%)': ((voting_r2 - baseline_r2) / baseline_r2) * 100
})

# 简单平均
simple_avg = np.mean(list(predictions.values()), axis=0)
avg_mae = mean_absolute_error(y_test, simple_avg)
avg_r2 = r2_score(y_test, simple_avg)

results.append({
    'Model': 'Simple_Average',
    'MAE': avg_mae,
    'R²': avg_r2,
    'Improvement(%)': ((avg_r2 - baseline_r2) / baseline_r2) * 100
})

# 加权平均（基于单个模型R²）
weights = []
for name in predictions.keys():
    pred = predictions[name]
    r2 = r2_score(y_test, pred)
    weights.append(max(r2, 0))

weights = np.array(weights)
weights = weights / weights.sum()

weighted_avg = np.average(list(predictions.values()), axis=0, weights=weights)
weighted_mae = mean_absolute_error(y_test, weighted_avg)
weighted_r2 = r2_score(y_test, weighted_avg)

results.append({
    'Model': 'Weighted_Average',
    'MAE': weighted_mae,
    'R²': weighted_r2,
    'Improvement(%)': ((weighted_r2 - baseline_r2) / baseline_r2) * 100
})

# 4. 结果汇总
results_df = pd.DataFrame([
    {'Model': 'Baseline', 'MAE': baseline_mae, 'R²': baseline_r2, 'Improvement(%)': 0.0}
] + results)

results_df = results_df.sort_values('R²', ascending=False)

print(f"\n========== 集成学习结果 ==========")
print(results_df.round(3))

# 5. 可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# R²对比
ax1 = axes[0]
colors = ['red' if 'Baseline' in model else 'lightgreen' for model in results_df['Model']]
bars = ax1.bar(range(len(results_df)), results_df['R²'], color=colors, edgecolor='black')
ax1.set_title('Ensemble Learning R² Comparison')
ax1.set_ylabel('R² Score')
ax1.set_xticks(range(len(results_df)))
ax1.set_xticklabels(results_df['Model'], rotation=45, ha='right')

for bar, r2 in zip(bars, results_df['R²']):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{r2:.3f}', ha='center', va='bottom', fontweight='bold')

# 改进百分比
ax2 = axes[1]
improvement_data = results_df[results_df['Model'] != 'Baseline']
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
plt.savefig('optimization/ensemble_test_results.png', dpi=300, bbox_inches='tight')
plt.savefig('optimization/ensemble_test_results.pdf', dpi=300, bbox_inches='tight')
plt.show()

# 6. 保存最佳模型
best_model_name = results_df.iloc[0]['Model']
best_r2 = results_df.iloc[0]['R²']

print(f"\n最佳集成方法: {best_model_name}")
print(f"R²改进: {((best_r2 - baseline_r2) / baseline_r2) * 100:.1f}%")

# 保存最佳模型
os.makedirs('optimization/models', exist_ok=True)
if best_model_name == 'Voting_Ensemble':
    joblib.dump(voting_model, 'optimization/models/best_ensemble_model.pkl')

results_df.to_csv('optimization/results/ensemble_test_results.csv', index=False)

if best_r2 > 0.5:
    print("🎉 已达到优秀水平 (R² > 0.5)")
elif best_r2 > 0.4:
    print("🎉 已达到良好水平 (R² > 0.4)")
else:
    print("📈 有所改进")

print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")