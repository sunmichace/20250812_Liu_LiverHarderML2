#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 4: 时间序列建模（简化版）
基于纵向历史数据预测最后一次肝脏弹性值的完整分析方案
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 机器学习相关
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

# Set English display
plt.rcParams['font.family'] = 'DejaVu Sans'

print("========== Phase 4: 时间序列建模（简化版） ==========")
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Step 1: 加载特征数据
print("\n1. 加载特征数据")
features_df = pd.read_csv('data/processed/comprehensive_features_final_new.csv')

print(f"特征数据形状: {features_df.shape}")

# 准备建模数据
feature_cols = [col for col in features_df.columns if col not in ['patient_id', 'sample_id', 'target_liver_stiffness']]
X = features_df[feature_cols]
y = features_df['target_liver_stiffness']

print(f"输入特征维度: {X.shape}")
print(f"目标变量: 均值={y.mean():.2f}, 标准差={y.std():.2f}")

# Step 2: 数据分割
print("\n2. 数据分割")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"训练集: {X_train.shape[0]}, 测试集: {X_test.shape[0]}")

# Step 3: 训练核心模型
print("\n3. 训练核心模型")

models = {}
results = {}

def evaluate_model(name, model):
    """评估模型性能"""
    print(f"\n训练 {name}...")
    model.fit(X_train, y_train)
    
    # 预测
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # 计算指标
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    print(f"{name} 结果:")
    print(f"  训练 MAE: {train_mae:.3f}, R²: {train_r2:.3f}")
    print(f"  测试 MAE: {test_mae:.3f}, R²: {test_r2:.3f}, RMSE: {test_rmse:.3f}")
    
    return {
        'model': model,
        'test_mae': test_mae,
        'test_r2': test_r2,
        'test_rmse': test_rmse,
        'predictions': y_test_pred
    }

# 3.1 线性回归
models['Linear'] = LinearRegression()
results['Linear'] = evaluate_model('Linear Regression', models['Linear'])

# 3.2 Ridge回归  
models['Ridge'] = Ridge(alpha=1.0)
results['Ridge'] = evaluate_model('Ridge Regression', models['Ridge'])

# 3.3 随机森林
models['RandomForest'] = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
results['RandomForest'] = evaluate_model('Random Forest', models['RandomForest'])

# 3.4 梯度提升
models['GradientBoosting'] = GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=6)
results['GradientBoosting'] = evaluate_model('Gradient Boosting', models['GradientBoosting'])

# Step 4: 模型比较
print("\n4. 模型比较")

comparison_data = []
for name, result in results.items():
    comparison_data.append({
        'Model': name,
        'Test MAE': result['test_mae'],
        'Test RMSE': result['test_rmse'], 
        'Test R²': result['test_r2']
    })

comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.sort_values('Test MAE')
print("\n模型性能比较:")
print(comparison_df.round(3))

# Step 5: 可视化
print("\n5. 生成可视化结果")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Model MAE comparison
ax = axes[0, 0]
bars = ax.bar(comparison_df['Model'], comparison_df['Test MAE'])
ax.set_ylabel('MAE (kPa)')
ax.set_title('Model MAE Comparison')
ax.tick_params(axis='x', rotation=45)
for i, v in enumerate(comparison_df['Test MAE']):
    ax.text(i, v + 0.01, f'{v:.2f}', ha='center', va='bottom')
ax.grid(True, alpha=0.3)

# Model R² comparison
ax = axes[0, 1]
r2_sorted = comparison_df.sort_values('Test R²', ascending=False)
bars = ax.bar(r2_sorted['Model'], r2_sorted['Test R²'])
ax.set_ylabel('R²')
ax.set_title('Model R² Comparison') 
ax.tick_params(axis='x', rotation=45)
for i, v in enumerate(r2_sorted['Test R²']):
    ax.text(i, v + 0.01, f'{v:.2f}', ha='center', va='bottom')
ax.grid(True, alpha=0.3)

# Best model prediction vs actual
best_model_name = comparison_df.iloc[0]['Model']
best_predictions = results[best_model_name]['predictions']

ax = axes[1, 0]
ax.scatter(y_test, best_predictions, alpha=0.6, s=50)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', alpha=0.8)
ax.set_xlabel('Actual Values (kPa)')
ax.set_ylabel('Predicted Values (kPa)')
ax.set_title(f'{best_model_name} - Prediction vs Actual')
ax.grid(True, alpha=0.3)

# Residual distribution
ax = axes[1, 1]
residuals = y_test.values - best_predictions
ax.hist(residuals, bins=20, alpha=0.7, edgecolor='black')
ax.set_xlabel('Residuals (kPa)')
ax.set_ylabel('Frequency')
ax.set_title(f'{best_model_name} - Residual Distribution')
ax.axvline(x=0, color='red', linestyle='--', alpha=0.8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
# Save both PNG and PDF formats
plt.savefig('data/processed/phase4_model_comparison_simple.png', dpi=300, bbox_inches='tight')
plt.savefig('data/processed/phase4_model_comparison_simple.pdf', dpi=300, bbox_inches='tight')
plt.show()

# Step 6: 特征重要性（最佳树模型）
print("\n6. 特征重要性分析")

tree_models = ['RandomForest', 'GradientBoosting']
best_tree_name = None
best_tree_mae = float('inf')

for name in tree_models:
    if results[name]['test_mae'] < best_tree_mae:
        best_tree_mae = results[name]['test_mae']
        best_tree_name = name

if best_tree_name:
    best_tree_model = models[best_tree_name]
    feature_importance = best_tree_model.feature_importances_
    
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print(f"\n{best_tree_name} 特征重要性 (Top 15):")
    for i, (_, row) in enumerate(importance_df.head(15).iterrows()):
        print(f"{i+1:2d}. {row['feature']:<30}: {row['importance']:.4f}")
    
    # 可视化特征重要性
    plt.figure(figsize=(12, 8))
    top_features = importance_df.head(20)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance Score')
    plt.title(f'{best_tree_name} - Feature Importance (Top 20)')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # Save both PNG and PDF formats
    plt.savefig('data/processed/phase4_feature_importance_simple.png', dpi=300, bbox_inches='tight')
    plt.savefig('data/processed/phase4_feature_importance_simple.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 保存重要性结果
    importance_df.to_csv('data/processed/phase4_feature_importance_ranking.csv', index=False)

# Step 7: 保存模型和结果
print("\n7. 保存结果")

# 创建模型目录
os.makedirs('data/processed/models', exist_ok=True)

# 保存最佳模型
best_model = models[best_model_name]
joblib.dump(best_model, f'data/processed/models/best_model_{best_model_name.lower()}.pkl')

# 保存所有模型
for name, model in models.items():
    joblib.dump(model, f'data/processed/models/{name.lower()}_model.pkl')

# 保存性能比较
comparison_df.to_csv('data/processed/phase4_model_performance_comparison.csv', index=False)

print("保存的文件:")
print("- data/processed/phase4_model_performance_comparison.csv: 模型性能比较")
print("- data/processed/phase4_feature_importance_ranking.csv: 特征重要性排序")
print("- data/processed/models/: 训练好的模型文件")

# Step 8: 摘要报告
print("\n========== Phase 4 建模摘要报告 ==========")

print(f"\n数据集信息:")
print(f"- 训练样本: {X_train.shape[0]}")
print(f"- 测试样本: {X_test.shape[0]}")  
print(f"- 特征数量: {X_train.shape[1]}")

print(f"\n模型性能排名:")
for i, (_, row) in enumerate(comparison_df.iterrows()):
    print(f"{i+1}. {row['Model']}: MAE={row['Test MAE']:.3f}, R²={row['Test R²']:.3f}")

print(f"\n最佳模型: {best_model_name}")
print(f"- 测试MAE: {results[best_model_name]['test_mae']:.3f} kPa")
print(f"- 测试R²: {results[best_model_name]['test_r2']:.3f}")

print(f"\n临床评估:")
good_models = sum(1 for _, row in comparison_df.iterrows() if row['Test MAE'] <= 2.0)
excellent_models = sum(1 for _, row in comparison_df.iterrows() if row['Test MAE'] <= 1.0)
print(f"- MAE ≤ 2.0 kPa (良好): {good_models}/{len(comparison_df)} 个模型")
print(f"- MAE ≤ 1.0 kPa (优秀): {excellent_models}/{len(comparison_df)} 个模型")

print(f"\n完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("Phase 4 时间序列建模完成！")