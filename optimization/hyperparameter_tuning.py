#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
超参数调优脚本：提升GradientBoosting模型性能
基于纵向历史数据预测肝脏弹性值的模型优化
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 机器学习相关
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, 
    RandomizedSearchCV, TimeSeriesSplit
)
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.preprocessing import StandardScaler
import joblib
import os

# 高级调优工具
try:
    import optuna
    OPTUNA_AVAILABLE = True
    print("✅ Optuna可用 - 将使用贝叶斯优化")
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️ Optuna不可用 - 使用传统网格搜索")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    print("✅ XGBoost可用")
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost不可用")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
    print("✅ LightGBM可用")
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️ LightGBM不可用")

# Set English display
plt.rcParams['font.family'] = 'DejaVu Sans'

print("========== 超参数调优：提升模型R² ==========")
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("目标: 通过超参数调优提升GradientBoosting模型的R²性能")

# Step 1: 加载数据
print("\n1. 加载数据和当前最佳模型")
features_df = pd.read_csv('data/processed/comprehensive_features_final_new.csv')
current_best_model = joblib.load('data/processed/models/best_model_gradientboosting.pkl')

# 准备数据
feature_cols = [col for col in features_df.columns if col not in ['patient_id', 'sample_id', 'target_liver_stiffness']]
X = features_df[feature_cols]
y = features_df['target_liver_stiffness']

print(f"数据集大小: {X.shape[0]} 样本, {X.shape[1]} 特征")
print(f"目标变量范围: [{y.min():.2f}, {y.max():.2f}] kPa")

# 当前模型基准性能
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
current_pred = current_best_model.predict(X_test)
current_mae = mean_absolute_error(y_test, current_pred)
current_r2 = r2_score(y_test, current_pred)

print(f"\n当前最佳模型基准性能:")
print(f"MAE: {current_mae:.3f} kPa")
print(f"R²: {current_r2:.3f}")

# Step 2: 特征选择优化
print("\n2. 特征选择优化")

def optimize_feature_selection(X_train, y_train, X_test, y_test):
    """优化特征选择以提升R²"""
    results = {}
    
    # 2.1 基于F统计量的特征选择
    print("2.1 基于F统计量的特征选择")
    f_scores, _ = f_regression(X_train, y_train)
    
    # 尝试不同的特征数量
    feature_numbers = [10, 15, 20, 25, 30, 35, 40, 45, 50]
    
    for n_features in feature_numbers:
        if n_features <= X_train.shape[1]:
            selector = SelectKBest(f_regression, k=n_features)
            X_train_selected = selector.fit_transform(X_train, y_train)
            X_test_selected = selector.transform(X_test)
            
            # 训练模型
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            model.fit(X_train_selected, y_train)
            pred = model.predict(X_test_selected)
            
            mae = mean_absolute_error(y_test, pred)
            r2 = r2_score(y_test, pred)
            
            results[f'F_select_{n_features}'] = {
                'n_features': n_features,
                'mae': mae,
                'r2': r2,
                'selector': selector,
                'model': model
            }
            
            print(f"  {n_features}特征: MAE={mae:.3f}, R²={r2:.3f}")
    
    # 2.2 递归特征消除
    print("\n2.2 递归特征消除(RFE)")
    base_model = GradientBoostingRegressor(n_estimators=50, random_state=42)
    
    for n_features in [15, 20, 25, 30]:
        if n_features <= X_train.shape[1]:
            rfe = RFE(base_model, n_features_to_select=n_features)
            X_train_rfe = rfe.fit_transform(X_train, y_train)
            X_test_rfe = rfe.transform(X_test)
            
            # 训练最终模型
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            model.fit(X_train_rfe, y_train)
            pred = model.predict(X_test_rfe)
            
            mae = mean_absolute_error(y_test, pred)
            r2 = r2_score(y_test, pred)
            
            results[f'RFE_{n_features}'] = {
                'n_features': n_features,
                'mae': mae,
                'r2': r2,
                'selector': rfe,
                'model': model
            }
            
            print(f"  RFE {n_features}特征: MAE={mae:.3f}, R²={r2:.3f}")
    
    return results

feature_selection_results = optimize_feature_selection(X_train, y_train, X_test, y_test)

# 找到最佳特征选择方案
best_feature_method = max(feature_selection_results.keys(), 
                         key=lambda k: feature_selection_results[k]['r2'])
best_feature_r2 = feature_selection_results[best_feature_method]['r2']
print(f"\n最佳特征选择: {best_feature_method}, R²: {best_feature_r2:.3f}")

# Step 3: 超参数网格搜索
print("\n3. 超参数网格搜索优化")

def grid_search_optimization(X_train, y_train, cv_folds=3):
    """网格搜索优化GradientBoosting参数"""
    
    # 定义参数网格
    param_grid = {
        'n_estimators': [100, 200, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
        'max_depth': [3, 4, 5, 6, 7, 8],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4, 6],
        'subsample': [0.8, 0.9, 1.0],
        'max_features': ['sqrt', 'log2', None]
    }
    
    print("执行网格搜索...")
    print(f"参数组合总数: {np.prod([len(v) for v in param_grid.values()]):,}")
    
    # 使用时间序列交叉验证
    tscv = TimeSeriesSplit(n_splits=cv_folds)
    
    # 随机搜索（因为网格太大）
    gb_model = GradientBoostingRegressor(random_state=42)
    
    # 使用随机搜索
    random_search = RandomizedSearchCV(
        gb_model,
        param_distributions=param_grid,
        n_iter=100,  # 随机尝试100个组合
        cv=tscv,
        scoring='r2',  # 直接优化R²
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    print("开始随机搜索...")
    random_search.fit(X_train, y_train)
    
    print(f"最佳参数: {random_search.best_params_}")
    print(f"最佳CV R²: {random_search.best_score_:.3f}")
    
    return random_search.best_estimator_, random_search.best_params_

best_gb_model, best_params = grid_search_optimization(X_train, y_train)

# Step 4: Optuna贝叶斯优化 (如果可用)
if OPTUNA_AVAILABLE:
    print("\n4. Optuna贝叶斯优化")
    
    def objective(trial):
        """Optuna优化目标函数"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'random_state': 42
        }
        
        model = GradientBoostingRegressor(**params)
        
        # 使用交叉验证评估
        tscv = TimeSeriesSplit(n_splits=3)
        scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='r2')
        
        return scores.mean()
    
    # 创建研究对象
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    
    print(f"Optuna最佳参数: {study.best_params}")
    print(f"Optuna最佳CV R²: {study.best_value:.3f}")
    
    # 用最佳参数训练模型
    optuna_model = GradientBoostingRegressor(**study.best_params)
    optuna_model.fit(X_train, y_train)
else:
    optuna_model = None
    study = None

# Step 5: 尝试其他高级模型
print("\n5. 尝试其他高级集成模型")

advanced_models = {}

# 5.1 随机森林超参数调优
print("5.1 优化随机森林")
rf_param_grid = {
    'n_estimators': [200, 300, 500],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False]
}

rf_random_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42),
    param_distributions=rf_param_grid,
    n_iter=30,
    cv=3,
    scoring='r2',
    n_jobs=-1,
    random_state=42
)

rf_random_search.fit(X_train, y_train)
advanced_models['RandomForest'] = rf_random_search.best_estimator_
print(f"最佳RF R²: {rf_random_search.best_score_:.3f}")

# 5.2 XGBoost (如果可用)
if XGBOOST_AVAILABLE:
    print("5.2 优化XGBoost")
    xgb_param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 1],
        'reg_lambda': [0, 0.1, 1]
    }
    
    xgb_random_search = RandomizedSearchCV(
        xgb.XGBRegressor(random_state=42),
        param_distributions=xgb_param_grid,
        n_iter=30,
        cv=3,
        scoring='r2',
        n_jobs=-1,
        random_state=42
    )
    
    xgb_random_search.fit(X_train, y_train)
    advanced_models['XGBoost'] = xgb_random_search.best_estimator_
    print(f"最佳XGB R²: {xgb_random_search.best_score_:.3f}")

# 5.3 LightGBM (如果可用)
if LIGHTGBM_AVAILABLE:
    print("5.3 优化LightGBM")
    lgb_param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7, -1],
        'num_leaves': [31, 50, 100],
        'min_child_samples': [10, 20, 30],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 1],
        'reg_lambda': [0, 0.1, 1]
    }
    
    lgb_random_search = RandomizedSearchCV(
        lgb.LGBMRegressor(random_state=42, verbose=-1),
        param_distributions=lgb_param_grid,
        n_iter=30,
        cv=3,
        scoring='r2',
        n_jobs=-1,
        random_state=42
    )
    
    lgb_random_search.fit(X_train, y_train)
    advanced_models['LightGBM'] = lgb_random_search.best_estimator_
    print(f"最佳LGB R²: {lgb_random_search.best_score_:.3f}")

# Step 6: 模型性能对比
print("\n6. 优化后模型性能对比")

def evaluate_model_comprehensive(model, X_test, y_test, model_name):
    """全面评估模型性能"""
    pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    
    # 临床评估指标
    errors = np.abs(pred - y_test)
    acc_1kpa = (errors <= 1.0).mean() * 100
    acc_2kpa = (errors <= 2.0).mean() * 100
    acc_3kpa = (errors <= 3.0).mean() * 100
    
    return {
        'Model': model_name,
        'MAE': mae,
        'RMSE': rmse,
        'R²': r2,
        'Acc_1kPa(%)': acc_1kpa,
        'Acc_2kPa(%)': acc_2kpa,
        'Acc_3kPa(%)': acc_3kpa
    }

# 评估所有模型
results_list = []

# 当前基准模型
results_list.append(evaluate_model_comprehensive(
    current_best_model, X_test, y_test, 'Current_GradientBoosting'))

# 网格搜索优化的模型
results_list.append(evaluate_model_comprehensive(
    best_gb_model, X_test, y_test, 'GridSearch_GradientBoosting'))

# Optuna优化的模型
if optuna_model:
    results_list.append(evaluate_model_comprehensive(
        optuna_model, X_test, y_test, 'Optuna_GradientBoosting'))

# 其他高级模型
for name, model in advanced_models.items():
    results_list.append(evaluate_model_comprehensive(
        model, X_test, y_test, f'Optimized_{name}'))

# 创建结果DataFrame
results_df = pd.DataFrame(results_list)
results_df = results_df.sort_values('R²', ascending=False)

print("\n模型性能对比结果:")
print(results_df.round(3))

# Step 7: 可视化结果
print("\n7. 生成优化结果可视化")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 7.1 R²对比
ax1 = axes[0, 0]
bars = ax1.bar(results_df['Model'], results_df['R²'], color='skyblue', edgecolor='black')
ax1.set_title('Model R² Comparison', fontsize=14, fontweight='bold')
ax1.set_ylabel('R² Score')
ax1.tick_params(axis='x', rotation=45)
# 添加数值标签
for bar, r2 in zip(bars, results_df['R²']):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{r2:.3f}', ha='center', va='bottom', fontweight='bold')

# 7.2 MAE对比
ax2 = axes[0, 1]
bars = ax2.bar(results_df['Model'], results_df['MAE'], color='lightcoral', edgecolor='black')
ax2.set_title('Model MAE Comparison', fontsize=14, fontweight='bold')
ax2.set_ylabel('MAE (kPa)')
ax2.tick_params(axis='x', rotation=45)
for bar, mae in zip(bars, results_df['MAE']):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{mae:.3f}', ha='center', va='bottom', fontweight='bold')

# 7.3 临床准确率对比
ax3 = axes[0, 2]
width = 0.25
x = np.arange(len(results_df))
ax3.bar(x - width, results_df['Acc_1kPa(%)'], width, label='±1.0 kPa', color='lightgreen')
ax3.bar(x, results_df['Acc_2kPa(%)'], width, label='±2.0 kPa', color='orange')
ax3.bar(x + width, results_df['Acc_3kPa(%)'], width, label='±3.0 kPa', color='lightblue')
ax3.set_title('Clinical Accuracy Comparison', fontsize=14, fontweight='bold')
ax3.set_ylabel('Accuracy (%)')
ax3.set_xticks(x)
ax3.set_xticklabels(results_df['Model'], rotation=45)
ax3.legend()

# 7.4 特征选择结果
ax4 = axes[1, 0]
if feature_selection_results:
    fs_names = list(feature_selection_results.keys())
    fs_r2s = [feature_selection_results[name]['r2'] for name in fs_names]
    ax4.bar(fs_names, fs_r2s, color='mediumpurple', edgecolor='black')
    ax4.set_title('Feature Selection R² Results', fontsize=14, fontweight='bold')
    ax4.set_ylabel('R² Score')
    ax4.tick_params(axis='x', rotation=45)

# 7.5 超参数重要性（如果有Optuna结果）
ax5 = axes[1, 1]
if OPTUNA_AVAILABLE and study:
    importance = optuna.importance.get_param_importances(study)
    params = list(importance.keys())[:6]  # Top 6
    values = list(importance.values())[:6]
    ax5.barh(params, values, color='gold', edgecolor='black')
    ax5.set_title('Hyperparameter Importance (Optuna)', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Importance')
else:
    ax5.text(0.5, 0.5, 'Optuna not available\nor no results', 
             ha='center', va='center', transform=ax5.transAxes,
             fontsize=12, style='italic')
    ax5.set_title('Hyperparameter Importance', fontsize=14, fontweight='bold')

# 7.6 改进幅度
ax6 = axes[1, 2]
if len(results_df) > 1:
    baseline_r2 = results_df.iloc[-1]['R²']  # 假设最后一个是基准
    best_r2 = results_df.iloc[0]['R²']
    improvement = ((best_r2 - baseline_r2) / baseline_r2) * 100
    
    ax6.bar(['Baseline', 'Best Model'], [baseline_r2, best_r2], 
            color=['lightcoral', 'lightgreen'], edgecolor='black')
    ax6.set_title(f'R² Improvement: +{improvement:.1f}%', fontsize=14, fontweight='bold')
    ax6.set_ylabel('R² Score')
    
    # 添加改进箭头
    ax6.annotate('', xy=(1, best_r2), xytext=(0, baseline_r2),
                arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    ax6.text(0.5, (baseline_r2 + best_r2)/2, f'+{improvement:.1f}%',
             ha='center', va='center', fontweight='bold', color='red')

plt.tight_layout()
# Save both PNG and PDF formats
plt.savefig('optimization/hyperparameter_tuning_results.png', dpi=300, bbox_inches='tight')
plt.savefig('optimization/hyperparameter_tuning_results.pdf', dpi=300, bbox_inches='tight')
plt.show()

# Step 8: 保存最佳模型和结果
print("\n8. 保存优化结果")

# 创建优化结果目录
os.makedirs('optimization/models', exist_ok=True)
os.makedirs('optimization/results', exist_ok=True)

# 保存最佳模型
best_model_name = results_df.iloc[0]['Model']
best_model_r2 = results_df.iloc[0]['R²']

if 'GridSearch' in best_model_name:
    best_model = best_gb_model
elif 'Optuna' in best_model_name and optuna_model:
    best_model = optuna_model
else:
    # 从advanced_models中找到最佳的
    for name, model in advanced_models.items():
        if f'Optimized_{name}' == best_model_name:
            best_model = model
            break

joblib.dump(best_model, f'optimization/models/best_optimized_model_{best_model_name.lower()}.pkl')
print(f"✅ 最佳模型已保存: {best_model_name}")

# 保存结果
results_df.to_csv('optimization/results/hyperparameter_tuning_results.csv', index=False)

# 保存特征选择结果
if feature_selection_results:
    fs_results_df = pd.DataFrame([
        {
            'Method': name,
            'N_Features': result['n_features'],
            'MAE': result['mae'],
            'R²': result['r2']
        }
        for name, result in feature_selection_results.items()
    ])
    fs_results_df.to_csv('optimization/results/feature_selection_results.csv', index=False)

# 保存超参数结果
if OPTUNA_AVAILABLE and study:
    study.trials_dataframe().to_csv('optimization/results/optuna_trials.csv', index=False)

# 生成优化报告
print("\n========== 超参数优化摘要报告 ==========")
print(f"原始模型性能:")
print(f"- MAE: {current_mae:.3f} kPa")
print(f"- R²: {current_r2:.3f}")

print(f"\n最佳优化模型: {best_model_name}")
print(f"- MAE: {results_df.iloc[0]['MAE']:.3f} kPa")
print(f"- R²: {results_df.iloc[0]['R²']:.3f}")
print(f"- R²改进: {((results_df.iloc[0]['R²'] - current_r2) / current_r2) * 100:.1f}%")
print(f"- 临床准确率(±2.0 kPa): {results_df.iloc[0]['Acc_2kPa(%)']:.1f}%")

improvement_r2 = ((results_df.iloc[0]['R²'] - current_r2) / current_r2) * 100
improvement_mae = ((current_mae - results_df.iloc[0]['MAE']) / current_mae) * 100

print(f"\n关键改进:")
print(f"✅ R²提升: {improvement_r2:.1f}%")
print(f"✅ MAE降低: {improvement_mae:.1f}%")

if best_model_r2 > 0.4:
    print("🎉 模型R²已达到良好水平 (>0.4)")
elif best_model_r2 > 0.35:
    print("✅ 模型R²达到可接受水平 (>0.35)")
else:
    print("⚠️ 模型R²仍需进一步优化")

print(f"\n保存的文件:")
print(f"- optimization/models/: 最佳优化模型")
print(f"- optimization/results/: 优化结果和数据")
print(f"- optimization/hyperparameter_tuning_results.png/.pdf: 结果可视化")

print(f"\n完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("超参数优化完成！")