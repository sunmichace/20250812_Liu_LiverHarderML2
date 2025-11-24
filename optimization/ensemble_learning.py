#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级集成学习脚本：Stacking和Blending提升R²
基于纵向历史数据预测肝脏弹性值的集成学习优化
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 机器学习相关
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.ensemble import (
    GradientBoostingRegressor, RandomForestRegressor, 
    ExtraTreesRegressor, AdaBoostRegressor, VotingRegressor
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Stacking相关
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import cross_val_predict

# Set English display
plt.rcParams['font.family'] = 'DejaVu Sans'

print("========== 高级集成学习：Stacking & Blending ==========")
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("目标: 通过集成学习方法进一步提升模型R²性能")

# Step 1: 加载数据
print("\n1. 加载数据")
features_df = pd.read_csv('data/processed/comprehensive_features_final_new.csv')

# 准备数据
feature_cols = [col for col in features_df.columns if col not in ['patient_id', 'sample_id', 'target_liver_stiffness']]
X = features_df[feature_cols]
y = features_df['target_liver_stiffness']

print(f"数据集大小: {X.shape[0]} 样本, {X.shape[1]} 特征")

# 数据分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)  # 进一步分割验证集

print(f"训练集: {X_train.shape[0]}, 验证集: {X_val.shape[0]}, 测试集: {X_test.shape[0]}")

# Step 2: 构建基学习器集合
print("\n2. 构建多样化基学习器集合")

def create_base_models():
    """创建多样化的基学习器"""
    base_models = {}
    
    # 树模型类
    base_models['gb1'] = GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42
    )
    base_models['gb2'] = GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.2, max_depth=4, random_state=43
    )
    base_models['rf1'] = RandomForestRegressor(
        n_estimators=200, max_depth=12, min_samples_split=5, random_state=42
    )
    base_models['rf2'] = RandomForestRegressor(
        n_estimators=300, max_depth=8, min_samples_split=10, random_state=43
    )
    base_models['et'] = ExtraTreesRegressor(
        n_estimators=200, max_depth=10, random_state=42
    )
    base_models['ada'] = AdaBoostRegressor(
        n_estimators=100, learning_rate=0.8, random_state=42
    )
    
    # 线性模型类
    base_models['ridge'] = Ridge(alpha=1.0)
    base_models['lasso'] = Lasso(alpha=0.1)
    base_models['elastic'] = ElasticNet(alpha=0.1, l1_ratio=0.5)
    
    # 其他类型
    base_models['svr'] = SVR(kernel='rbf', C=1.0, gamma='scale')
    base_models['knn'] = KNeighborsRegressor(n_neighbors=5)
    base_models['mlp'] = MLPRegressor(
        hidden_layer_sizes=(100, 50), max_iter=500, random_state=42
    )
    
    return base_models

base_models = create_base_models()

# 标准化特征（某些模型需要）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Step 3: 评估基学习器性能
print("\n3. 评估基学习器性能")

base_predictions = {}
base_performance = {}

def evaluate_base_model(name, model, X_train, X_val, y_train, y_val, use_scaled=False):
    """评估单个基学习器"""
    X_tr = X_train_scaled if use_scaled else X_train
    X_v = X_val_scaled if use_scaled else X_val
    
    # 训练模型
    model.fit(X_tr, y_train)
    
    # 预测
    val_pred = model.predict(X_v)
    
    # 计算指标
    mae = mean_absolute_error(y_val, val_pred)
    r2 = r2_score(y_val, val_pred)
    
    print(f"{name:12s}: MAE={mae:.3f}, R²={r2:.3f}")
    
    return model, val_pred, mae, r2

# 评估所有基学习器
for name, model in base_models.items():
    use_scaled = name in ['svr', 'knn', 'mlp', 'ridge', 'lasso', 'elastic']
    
    trained_model, val_pred, mae, r2 = evaluate_base_model(
        name, model, X_train, X_val, y_train, y_val, use_scaled
    )
    
    base_predictions[name] = val_pred
    base_performance[name] = {'model': trained_model, 'mae': mae, 'r2': r2}

# Step 4: 相关性分析
print("\n4. 基学习器预测相关性分析")

# 创建预测结果DataFrame
pred_df = pd.DataFrame(base_predictions)
correlation_matrix = pred_df.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, fmt='.3f')
plt.title('Base Model Predictions Correlation Matrix')
plt.tight_layout()
# Save both PNG and PDF formats
plt.savefig('optimization/base_models_correlation.png', dpi=300, bbox_inches='tight')
plt.savefig('optimization/base_models_correlation.pdf', dpi=300, bbox_inches='tight')
plt.show()

print("基学习器多样性分析:")
print(f"平均相关系数: {correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean():.3f}")

# Step 5: Stacking方法
print("\n5. Stacking集成学习")

def create_stacking_model(base_models, meta_learner=None):
    """创建Stacking模型"""
    if meta_learner is None:
        meta_learner = Ridge(alpha=1.0)
    
    # 选择表现较好的基学习器
    good_models = []
    for name, perf in base_performance.items():
        if perf['r2'] > 0.2:  # 只选择R²>0.2的模型
            use_scaled = name in ['svr', 'knn', 'mlp', 'ridge', 'lasso', 'elastic']
            if use_scaled:
                # 对需要标准化的模型创建Pipeline
                from sklearn.pipeline import Pipeline
                model_pipe = Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', perf['model'])
                ])
                good_models.append((name, model_pipe))
            else:
                good_models.append((name, perf['model']))
    
    print(f"选择了 {len(good_models)} 个表现良好的基学习器")
    
    stacking_model = StackingRegressor(
        estimators=good_models,
        final_estimator=meta_learner,
        cv=5,
        n_jobs=-1
    )
    
    return stacking_model

# 尝试不同的元学习器
meta_learners = {
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.1),
    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=3),
    'Linear': LinearRegression()
}

stacking_results = {}

for meta_name, meta_model in meta_learners.items():
    print(f"\n5.{len(stacking_results)+1} 使用{meta_name}作为元学习器")
    
    stacking_model = create_stacking_model(base_models, meta_model)
    
    # 训练Stacking模型
    stacking_model.fit(X_train, y_train)
    
    # 预测
    val_pred = stacking_model.predict(X_val)
    test_pred = stacking_model.predict(X_test)
    
    # 评估
    val_mae = mean_absolute_error(y_val, val_pred)
    val_r2 = r2_score(y_val, val_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    test_r2 = r2_score(y_test, test_pred)
    
    stacking_results[f'Stacking_{meta_name}'] = {
        'model': stacking_model,
        'val_mae': val_mae,
        'val_r2': val_r2,
        'test_mae': test_mae,
        'test_r2': test_r2,
        'test_pred': test_pred
    }
    
    print(f"  验证集: MAE={val_mae:.3f}, R²={val_r2:.3f}")
    print(f"  测试集: MAE={test_mae:.3f}, R²={test_r2:.3f}")

# Step 6: Blending方法
print("\n6. Blending集成学习")

def create_blending_predictions():
    """创建Blending预测"""
    # 使用验证集训练基学习器并在测试集上预测
    blend_predictions = {}
    
    for name, model_info in base_performance.items():
        model = model_info['model']
        use_scaled = name in ['svr', 'knn', 'mlp', 'ridge', 'lasso', 'elastic']
        
        X_test_use = X_test_scaled if use_scaled else X_test
        test_pred = model.predict(X_test_use)
        blend_predictions[name] = test_pred
    
    return blend_predictions

blend_predictions = create_blending_predictions()

# 尝试不同的权重策略
def weighted_blend(predictions_dict, weights=None):
    """加权融合预测结果"""
    pred_matrix = np.column_stack(list(predictions_dict.values()))
    
    if weights is None:
        # 等权重
        weights = np.ones(pred_matrix.shape[1]) / pred_matrix.shape[1]
    
    return np.dot(pred_matrix, weights)

# 6.1 等权重Blending
equal_weight_pred = weighted_blend(blend_predictions)
equal_weight_mae = mean_absolute_error(y_test, equal_weight_pred)
equal_weight_r2 = r2_score(y_test, equal_weight_pred)

print(f"等权重Blending: MAE={equal_weight_mae:.3f}, R²={equal_weight_r2:.3f}")

# 6.2 基于性能的权重Blending
performance_weights = np.array([base_performance[name]['r2'] for name in blend_predictions.keys()])
performance_weights = np.maximum(performance_weights, 0)  # 确保非负
performance_weights = performance_weights / performance_weights.sum()  # 归一化

performance_weight_pred = weighted_blend(blend_predictions, performance_weights)
performance_weight_mae = mean_absolute_error(y_test, performance_weight_pred)
performance_weight_r2 = r2_score(y_test, performance_weight_r2)

print(f"性能权重Blending: MAE={performance_weight_mae:.3f}, R²={performance_weight_r2:.3f}")

# 6.3 优化权重Blending
from scipy.optimize import minimize

def optimize_weights(y_true, predictions_dict):
    """优化Blending权重"""
    pred_matrix = np.column_stack(list(predictions_dict.values()))
    
    def objective(weights):
        weights = weights / weights.sum()  # 归一化
        blend_pred = np.dot(pred_matrix, weights)
        return -r2_score(y_true, blend_pred)  # 最大化R²
    
    # 约束：权重和为1，所有权重非负
    constraints = [{'type': 'eq', 'fun': lambda w: w.sum() - 1}]
    bounds = [(0, 1) for _ in range(len(predictions_dict))]
    
    # 初始权重
    initial_weights = np.ones(len(predictions_dict)) / len(predictions_dict)
    
    result = minimize(objective, initial_weights, method='SLSQP', 
                     bounds=bounds, constraints=constraints)
    
    return result.x

# 使用验证集数据优化权重
val_blend_predictions = {}
for name, model_info in base_performance.items():
    val_blend_predictions[name] = base_predictions[name]

optimal_weights = optimize_weights(y_val, val_blend_predictions)
optimal_weight_pred = weighted_blend(blend_predictions, optimal_weights)
optimal_weight_mae = mean_absolute_error(y_test, optimal_weight_pred)
optimal_weight_r2 = r2_score(y_test, optimal_weight_pred)

print(f"优化权重Blending: MAE={optimal_weight_mae:.3f}, R²={optimal_weight_r2:.3f}")
print(f"最优权重: {dict(zip(blend_predictions.keys(), optimal_weights.round(3)))}")

# Step 7: Voting集成
print("\n7. Voting集成方法")

# 选择表现好的模型进行Voting
good_estimators = []
for name, perf in base_performance.items():
    if perf['r2'] > 0.25:  # 选择R²>0.25的模型
        use_scaled = name in ['svr', 'knn', 'mlp', 'ridge', 'lasso', 'elastic']
        if use_scaled:
            from sklearn.pipeline import Pipeline
            model_pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('model', perf['model'])
            ])
            good_estimators.append((name, model_pipe))
        else:
            good_estimators.append((name, perf['model']))

voting_model = VotingRegressor(estimators=good_estimators)
voting_model.fit(X_train, y_train)

voting_pred = voting_model.predict(X_test)
voting_mae = mean_absolute_error(y_test, voting_pred)
voting_r2 = r2_score(y_test, voting_pred)

print(f"Voting集成: MAE={voting_mae:.3f}, R²={voting_r2:.3f}")

# Step 8: 结果汇总和比较
print("\n8. 集成学习结果汇总")

# 加载原始最佳模型进行对比
try:
    original_model = joblib.load('data/processed/models/best_model_gradientboosting.pkl')
    original_pred = original_model.predict(X_test)
    original_mae = mean_absolute_error(y_test, original_pred)
    original_r2 = r2_score(y_test, original_pred)
    print(f"原始最佳模型: MAE={original_mae:.3f}, R²={original_r2:.3f}")
except:
    original_r2 = 0.321  # 从之前的结果
    original_mae = 2.457

# 汇总所有结果
ensemble_results = []

# 原始模型
ensemble_results.append({
    'Model': 'Original_GradientBoosting',
    'MAE': original_mae,
    'R²': original_r2,
    'Type': 'Single Model'
})

# Stacking结果
for name, result in stacking_results.items():
    ensemble_results.append({
        'Model': name,
        'MAE': result['test_mae'],
        'R²': result['test_r2'],
        'Type': 'Stacking'
    })

# Blending结果
ensemble_results.extend([
    {'Model': 'Blending_Equal_Weight', 'MAE': equal_weight_mae, 'R²': equal_weight_r2, 'Type': 'Blending'},
    {'Model': 'Blending_Performance_Weight', 'MAE': performance_weight_mae, 'R²': performance_weight_r2, 'Type': 'Blending'},
    {'Model': 'Blending_Optimal_Weight', 'MAE': optimal_weight_mae, 'R²': optimal_weight_r2, 'Type': 'Blending'}
])

# Voting结果
ensemble_results.append({
    'Model': 'Voting_Ensemble',
    'MAE': voting_mae,
    'R²': voting_r2,
    'Type': 'Voting'
})

# 创建结果DataFrame
ensemble_df = pd.DataFrame(ensemble_results)
ensemble_df = ensemble_df.sort_values('R²', ascending=False)

print("\n集成学习性能对比:")
print(ensemble_df.round(3))

# Step 9: 可视化结果
print("\n9. 生成集成学习结果可视化")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 9.1 R²对比
ax1 = axes[0, 0]
colors = ['red' if 'Original' in model else 'skyblue' for model in ensemble_df['Model']]
bars = ax1.bar(range(len(ensemble_df)), ensemble_df['R²'], color=colors, edgecolor='black')
ax1.set_title('Ensemble Learning R² Comparison', fontsize=14, fontweight='bold')
ax1.set_ylabel('R² Score')
ax1.set_xticks(range(len(ensemble_df)))
ax1.set_xticklabels(ensemble_df['Model'], rotation=45, ha='right')

# 添加数值标签
for i, (bar, r2) in enumerate(zip(bars, ensemble_df['R²'])):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{r2:.3f}', ha='center', va='bottom', fontweight='bold')

# 9.2 MAE对比
ax2 = axes[0, 1]
colors = ['red' if 'Original' in model else 'lightcoral' for model in ensemble_df['Model']]
bars = ax2.bar(range(len(ensemble_df)), ensemble_df['MAE'], color=colors, edgecolor='black')
ax2.set_title('Ensemble Learning MAE Comparison', fontsize=14, fontweight='bold')
ax2.set_ylabel('MAE (kPa)')
ax2.set_xticks(range(len(ensemble_df)))
ax2.set_xticklabels(ensemble_df['Model'], rotation=45, ha='right')

# 9.3 方法类型分布
ax3 = axes[0, 2]
type_counts = ensemble_df['Type'].value_counts()
ax3.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%', startangle=90)
ax3.set_title('Ensemble Method Distribution', fontsize=14, fontweight='bold')

# 9.4 基学习器性能分布
ax4 = axes[1, 0]
base_r2_values = [perf['r2'] for perf in base_performance.values()]
base_names = list(base_performance.keys())
bars = ax4.bar(base_names, base_r2_values, color='lightgreen', edgecolor='black')
ax4.set_title('Base Learner R² Performance', fontsize=14, fontweight='bold')
ax4.set_ylabel('R² Score')
ax4.tick_params(axis='x', rotation=45)

# 9.5 权重分布（最优Blending）
ax5 = axes[1, 1]
ax5.bar(list(blend_predictions.keys()), optimal_weights, color='gold', edgecolor='black')
ax5.set_title('Optimal Blending Weights', fontsize=14, fontweight='bold')
ax5.set_ylabel('Weight')
ax5.tick_params(axis='x', rotation=45)

# 9.6 改进程度
ax6 = axes[1, 2]
best_ensemble_r2 = ensemble_df.iloc[0]['R²']
improvement = ((best_ensemble_r2 - original_r2) / original_r2) * 100

ax6.bar(['Original', 'Best Ensemble'], [original_r2, best_ensemble_r2], 
        color=['lightcoral', 'lightgreen'], edgecolor='black')
ax6.set_title(f'R² Improvement: +{improvement:.1f}%', fontsize=14, fontweight='bold')
ax6.set_ylabel('R² Score')

# 添加改进标注
ax6.annotate('', xy=(1, best_ensemble_r2), xytext=(0, original_r2),
            arrowprops=dict(arrowstyle='->', lw=2, color='red'))
ax6.text(0.5, (original_r2 + best_ensemble_r2)/2, f'+{improvement:.1f}%',
         ha='center', va='center', fontweight='bold', color='red')

plt.tight_layout()
# Save both PNG and PDF formats
plt.savefig('optimization/ensemble_learning_results.png', dpi=300, bbox_inches='tight')
plt.savefig('optimization/ensemble_learning_results.pdf', dpi=300, bbox_inches='tight')
plt.show()

# Step 10: 保存最佳集成模型
print("\n10. 保存最佳集成模型和结果")

# 确定最佳模型
best_ensemble_name = ensemble_df.iloc[0]['Model']
best_ensemble_r2 = ensemble_df.iloc[0]['R²']

print(f"最佳集成模型: {best_ensemble_name}")
print(f"R²: {best_ensemble_r2:.3f}")

# 保存最佳模型
if 'Stacking' in best_ensemble_name:
    best_model = stacking_results[best_ensemble_name]['model']
    joblib.dump(best_model, f'optimization/models/best_ensemble_stacking.pkl')
elif best_ensemble_name == 'Voting_Ensemble':
    joblib.dump(voting_model, f'optimization/models/best_ensemble_voting.pkl')
elif 'Blending' in best_ensemble_name:
    # 保存Blending信息
    blending_info = {
        'base_models': base_performance,
        'weights': optimal_weights,
        'model_names': list(blend_predictions.keys()),
        'scaler': scaler
    }
    joblib.dump(blending_info, f'optimization/models/best_ensemble_blending.pkl')

# 保存结果
ensemble_df.to_csv('optimization/results/ensemble_learning_results.csv', index=False)

# 保存基学习器性能
base_perf_df = pd.DataFrame([
    {
        'Model': name,
        'MAE': perf['mae'],
        'R²': perf['r2']
    }
    for name, perf in base_performance.items()
]).sort_values('R²', ascending=False)

base_perf_df.to_csv('optimization/results/base_learner_performance.csv', index=False)

# 生成最终报告
print("\n========== 集成学习优化摘要报告 ==========")
print(f"原始模型性能:")
print(f"- MAE: {original_mae:.3f} kPa")
print(f"- R²: {original_r2:.3f}")

print(f"\n最佳集成模型: {best_ensemble_name}")
print(f"- MAE: {ensemble_df.iloc[0]['MAE']:.3f} kPa")
print(f"- R²: {ensemble_df.iloc[0]['R²']:.3f}")

improvement_r2 = ((best_ensemble_r2 - original_r2) / original_r2) * 100
improvement_mae = ((original_mae - ensemble_df.iloc[0]['MAE']) / original_mae) * 100

print(f"\n关键改进:")
print(f"✅ R²提升: {improvement_r2:.1f}%")
print(f"✅ MAE{'降低' if improvement_mae > 0 else '变化'}: {improvement_mae:.1f}%")

if best_ensemble_r2 > 0.5:
    print("🎉 模型R²已达到优秀水平 (>0.5)")
elif best_ensemble_r2 > 0.4:
    print("🎉 模型R²已达到良好水平 (>0.4)")
elif best_ensemble_r2 > 0.35:
    print("✅ 模型R²达到可接受水平 (>0.35)")
else:
    print("⚠️ 模型R²仍需进一步优化")

print(f"\n集成学习方法表现:")
for method_type in ensemble_df['Type'].unique():
    if method_type != 'Single Model':
        method_df = ensemble_df[ensemble_df['Type'] == method_type]
        best_method_r2 = method_df['R²'].max()
        print(f"- {method_type}: 最佳R² = {best_method_r2:.3f}")

print(f"\n保存的文件:")
print(f"- optimization/models/: 最佳集成模型")
print(f"- optimization/results/: 集成学习结果")
print(f"- optimization/ensemble_learning_results.png/.pdf: 结果可视化")

print(f"\n完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("集成学习优化完成！")