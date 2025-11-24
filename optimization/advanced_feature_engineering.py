#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征工程增强脚本：高级特征构造提升R²
基于纵向历史数据预测肝脏弹性值的高级特征工程
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 机器学习和特征工程相关
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, RFE
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, QuantileTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import KMeans
import joblib
import os

# 高级特征工程
try:
    from sklearn.ensemble import IsolationForest
    ISOLATION_AVAILABLE = True
except ImportError:
    ISOLATION_AVAILABLE = False

try:
    from tsfresh import extract_features, select_features
    from tsfresh.utilities.dataframe_functions import impute
    TSFRESH_AVAILABLE = True
    print("✅ tsfresh可用 - 将使用时间序列特征提取")
except ImportError:
    TSFRESH_AVAILABLE = False
    print("⚠️ tsfresh不可用 - 使用手工特征工程")

# Set English display
plt.rcParams['font.family'] = 'DejaVu Sans'

print("========== 高级特征工程：提升模型R² ==========")
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("目标: 通过高级特征工程技术提升模型R²性能")

# Step 1: 加载数据
print("\n1. 加载原始数据")
longitudinal_df = pd.read_csv('data/processed/phase1_longitudinal_data.csv')
prediction_df = pd.read_csv('data/processed/phase1_prediction_dataset.csv')
current_features_df = pd.read_csv('data/processed/comprehensive_features_final_new.csv')

longitudinal_df['examination_date'] = pd.to_datetime(longitudinal_df['examination_date'])

print(f"纵向数据: {longitudinal_df.shape}")
print(f"预测样本: {prediction_df.shape}")
print(f"当前特征: {current_features_df.shape}")

# 当前模型基准性能
feature_cols = [col for col in current_features_df.columns if col not in ['patient_id', 'sample_id', 'target_liver_stiffness']]
X_current = current_features_df[feature_cols]
y = current_features_df['target_liver_stiffness']

X_train, X_test, y_train, y_test = train_test_split(X_current, y, test_size=0.2, random_state=42)

baseline_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
baseline_model.fit(X_train, y_train)
baseline_pred = baseline_model.predict(X_test)
baseline_mae = mean_absolute_error(y_test, baseline_pred)
baseline_r2 = r2_score(y_test, baseline_pred)

print(f"\n当前基准性能:")
print(f"MAE: {baseline_mae:.3f} kPa, R²: {baseline_r2:.3f}")

# Step 2: 高级统计特征
print("\n2. 构建高级统计特征")

def extract_advanced_statistical_features(sample_row, longitudinal_data):
    """提取高级统计特征"""
    patient_id = sample_row['patient_id']
    sample_id = sample_row['sample_id']
    target_value = sample_row['target_liver_stiffness']
    historical_count = sample_row['historical_count']
    
    # 获取患者数据
    patient_data = longitudinal_data[longitudinal_data['patient_id'] == patient_id].copy()
    patient_data = patient_data.sort_values('examination_date')
    
    # 获取历史数据（用于预测的部分）
    historical_data = patient_data.iloc[:historical_count]
    liver_values = historical_data['liver_stiffness'].values
    
    if len(liver_values) < 2:
        return {}
    
    features = {}
    
    # 2.1 高阶统计矩
    features['hist_moment3'] = np.mean((liver_values - np.mean(liver_values))**3)  # 三阶矩
    features['hist_moment4'] = np.mean((liver_values - np.mean(liver_values))**4)  # 四阶矩
    
    # 2.2 分位数特征
    features['hist_q10'] = np.percentile(liver_values, 10)
    features['hist_q25'] = np.percentile(liver_values, 25)
    features['hist_q75'] = np.percentile(liver_values, 75)
    features['hist_q90'] = np.percentile(liver_values, 90)
    features['hist_iqr_ratio'] = (features['hist_q75'] - features['hist_q25']) / (features['hist_q90'] - features['hist_q10'] + 1e-8)
    
    # 2.3 变异系数和鲁棒性指标
    features['hist_robust_std'] = np.median(np.abs(liver_values - np.median(liver_values)))  # MAD
    features['hist_robust_cv'] = features['hist_robust_std'] / (np.median(liver_values) + 1e-8)
    
    # 2.4 分布形状特征
    mean_val = np.mean(liver_values)
    features['hist_left_tail'] = np.mean(liver_values[liver_values < mean_val])
    features['hist_right_tail'] = np.mean(liver_values[liver_values > mean_val])
    features['hist_tail_ratio'] = features['hist_right_tail'] / (features['hist_left_tail'] + 1e-8)
    
    # 2.5 局部极值特征
    if len(liver_values) >= 3:
        local_maxima = []
        local_minima = []
        for i in range(1, len(liver_values)-1):
            if liver_values[i] > liver_values[i-1] and liver_values[i] > liver_values[i+1]:
                local_maxima.append(liver_values[i])
            elif liver_values[i] < liver_values[i-1] and liver_values[i] < liver_values[i+1]:
                local_minima.append(liver_values[i])
        
        features['local_maxima_count'] = len(local_maxima)
        features['local_minima_count'] = len(local_minima)
        features['local_extrema_ratio'] = len(local_maxima) / (len(local_minima) + 1)
        features['avg_local_maxima'] = np.mean(local_maxima) if local_maxima else np.mean(liver_values)
        features['avg_local_minima'] = np.mean(local_minima) if local_minima else np.mean(liver_values)
    
    return features

# Step 3: 时间序列高级特征
print("\n3. 构建时间序列高级特征")

def extract_time_series_features(sample_row, longitudinal_data):
    """提取时间序列特征"""
    patient_id = sample_row['patient_id']
    historical_count = sample_row['historical_count']
    
    patient_data = longitudinal_data[longitudinal_data['patient_id'] == patient_id].copy()
    patient_data = patient_data.sort_values('examination_date')
    historical_data = patient_data.iloc[:historical_count]
    
    if len(historical_data) < 2:
        return {}
    
    features = {}
    liver_values = historical_data['liver_stiffness'].values
    dates = historical_data['examination_date'].values
    
    # 时间间隔分析
    time_diffs = np.diff(dates).astype('timedelta64[D]').astype(int)
    
    # 3.1 时间相关特征
    features['time_regularity'] = 1.0 / (np.std(time_diffs) + 1)  # 时间规律性
    features['time_acceleration'] = np.std(np.diff(time_diffs)) if len(time_diffs) > 1 else 0  # 时间加速度
    
    # 3.2 趋势复杂度特征
    if len(liver_values) >= 3:
        # 二阶差分（加速度）
        first_diff = np.diff(liver_values)
        second_diff = np.diff(first_diff)
        features['trend_acceleration_mean'] = np.mean(second_diff)
        features['trend_acceleration_std'] = np.std(second_diff)
        
        # 趋势变化点检测
        sign_changes = np.sum(np.diff(np.sign(first_diff)) != 0)
        features['trend_change_points'] = sign_changes
        features['trend_stability'] = 1.0 / (sign_changes + 1)
    
    # 3.3 周期性特征（如果数据足够长）
    if len(liver_values) >= 6:
        # 使用傅里叶变换检测周期性
        fft_vals = np.fft.fft(liver_values)
        fft_freqs = np.fft.fftfreq(len(liver_values))
        
        # 主频率成分
        dominant_freq_idx = np.argmax(np.abs(fft_vals[1:len(fft_vals)//2])) + 1
        features['dominant_frequency'] = np.abs(fft_freqs[dominant_freq_idx])
        features['dominant_amplitude'] = np.abs(fft_vals[dominant_freq_idx])
        
        # 频谱能量分布
        total_energy = np.sum(np.abs(fft_vals)**2)
        low_freq_energy = np.sum(np.abs(fft_vals[:len(fft_vals)//4])**2)
        features['low_freq_energy_ratio'] = low_freq_energy / total_energy
    
    # 3.4 自回归特征
    if len(liver_values) >= 4:
        # 简单自回归模型拟合
        X_ar = np.column_stack([liver_values[:-2], liver_values[1:-1]])  # lag-1 and lag-2
        y_ar = liver_values[2:]
        
        if len(y_ar) > 0:
            # 最小二乘回归
            coeffs = np.linalg.lstsq(X_ar, y_ar, rcond=None)[0]
            features['ar1_coeff'] = coeffs[0] if len(coeffs) > 0 else 0
            features['ar2_coeff'] = coeffs[1] if len(coeffs) > 1 else 0
            
            # 自相关
            autocorr_1 = np.corrcoef(liver_values[:-1], liver_values[1:])[0,1]
            features['autocorr_lag1'] = autocorr_1 if not np.isnan(autocorr_1) else 0
    
    return features

# Step 4: 多变量交互特征
print("\n4. 构建多变量交互特征")

def extract_multivariate_features(sample_row, longitudinal_data):
    """提取多变量交互特征"""
    patient_id = sample_row['patient_id']
    historical_count = sample_row['historical_count']
    
    patient_data = longitudinal_data[longitudinal_data['patient_id'] == patient_id].copy()
    patient_data = patient_data.sort_values('examination_date')
    historical_data = patient_data.iloc[:historical_count]
    
    if len(historical_data) < 2:
        return {}
    
    features = {}
    
    # 4.1 肝功能指标协同特征
    liver_stiffness = historical_data['liver_stiffness'].values
    
    # ALT相关
    if 'ALT' in historical_data.columns:
        alt_values = historical_data['ALT'].dropna().values
        if len(alt_values) > 1:
            # ALT与肝脏弹性的相关性
            if len(alt_values) == len(liver_stiffness):
                corr_alt = np.corrcoef(liver_stiffness, alt_values)[0,1]
                features['liver_alt_correlation'] = corr_alt if not np.isnan(corr_alt) else 0
            
            # ALT变化与肝脏弹性变化的关系
            if len(alt_values) > 2:
                alt_changes = np.diff(alt_values)
                liver_changes = np.diff(liver_stiffness[:len(alt_changes)])
                if len(alt_changes) == len(liver_changes):
                    change_corr = np.corrcoef(alt_changes, liver_changes)[0,1]
                    features['alt_liver_change_correlation'] = change_corr if not np.isnan(change_corr) else 0
    
    # AST相关
    if 'AST' in historical_data.columns:
        ast_values = historical_data['AST'].dropna().values
        if len(ast_values) > 1 and len(ast_values) == len(liver_stiffness):
            corr_ast = np.corrcoef(liver_stiffness, ast_values)[0,1]
            features['liver_ast_correlation'] = corr_ast if not np.isnan(corr_ast) else 0
    
    # 4.2 多指标综合评分
    # 构建综合肝功能评分
    score_components = []
    
    if 'ALT' in historical_data.columns:
        alt_mean = historical_data['ALT'].mean()
        if not np.isnan(alt_mean):
            alt_score = min(alt_mean / 40.0, 3.0)  # 正常值上限约40，最高分3
            score_components.append(alt_score)
    
    if 'AST' in historical_data.columns:
        ast_mean = historical_data['AST'].mean()
        if not np.isnan(ast_mean):
            ast_score = min(ast_mean / 40.0, 3.0)
            score_components.append(ast_score)
    
    liver_mean = np.mean(liver_stiffness)
    liver_score = min(liver_mean / 10.0, 5.0)  # 肝脏弹性评分
    score_components.append(liver_score)
    
    if score_components:
        features['comprehensive_liver_score'] = np.mean(score_components)
        features['liver_score_std'] = np.std(score_components)
    
    # 4.3 病毒学特征（如果有）
    if 'HBV_DNA' in historical_data.columns:
        hbv_values = historical_data['HBV_DNA'].dropna().values
        if len(hbv_values) > 0:
            features['hbv_positive_ratio'] = np.mean(hbv_values > 0)
            features['hbv_mean_level'] = np.mean(hbv_values)
            
            # HBV与肝脏弹性关系
            if len(hbv_values) == len(liver_stiffness):
                hbv_liver_corr = np.corrcoef(liver_stiffness, hbv_values)[0,1]
                features['hbv_liver_correlation'] = hbv_liver_corr if not np.isnan(hbv_liver_corr) else 0
    
    return features

# Step 5: 构建所有高级特征
print("\n5. 批量提取高级特征")

def extract_all_advanced_features(prediction_data, longitudinal_data):
    """批量提取所有高级特征"""
    all_features = []
    
    for idx, row in prediction_data.iterrows():
        if idx % 500 == 0:
            print(f"处理进度: {idx}/{len(prediction_data)}")
        
        sample_features = {
            'patient_id': row['patient_id'],
            'sample_id': row['sample_id'],
            'target_liver_stiffness': row['target_liver_stiffness']
        }
        
        # 提取各类高级特征
        try:
            stat_features = extract_advanced_statistical_features(row, longitudinal_data)
            ts_features = extract_time_series_features(row, longitudinal_data)
            mv_features = extract_multivariate_features(row, longitudinal_data)
            
            sample_features.update(stat_features)
            sample_features.update(ts_features)
            sample_features.update(mv_features)
            
        except Exception as e:
            print(f"处理样本 {row['sample_id']} 时出错: {e}")
            continue
        
        all_features.append(sample_features)
    
    return pd.DataFrame(all_features)

# 提取高级特征
advanced_features_df = extract_all_advanced_features(prediction_df, longitudinal_df)
print(f"\n高级特征提取完成: {advanced_features_df.shape}")

# Step 6: 特征组合和变换
print("\n6. 特征组合和变换")

# 6.1 合并当前特征和高级特征
merged_features = current_features_df.merge(
    advanced_features_df[['patient_id', 'sample_id'] + [col for col in advanced_features_df.columns 
                                                        if col not in ['patient_id', 'sample_id', 'target_liver_stiffness']]],
    on=['patient_id', 'sample_id'],
    how='left'
)

print(f"合并后特征数: {merged_features.shape[1] - 3}")  # 减去ID和target列

# 6.2 多项式特征（选择重要特征）
print("生成多项式交互特征...")
important_features = ['hist_mean', 'hist_median', 'last_hist_value', 'baseline_value', 'hist_std']
important_cols = [col for col in important_features if col in merged_features.columns]

if important_cols:
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    poly_features = poly.fit_transform(merged_features[important_cols].fillna(0))
    poly_feature_names = [f"poly_{name}" for name in poly.get_feature_names_out(important_cols)]
    
    poly_df = pd.DataFrame(poly_features, columns=poly_feature_names, index=merged_features.index)
    
    # 只保留交互项（不保留原特征和平方项）
    interaction_cols = [col for col in poly_df.columns if ' ' in col and col.count(' ') == 1]
    poly_df = poly_df[interaction_cols]
    
    merged_features = pd.concat([merged_features, poly_df], axis=1)
    print(f"添加多项式特征后: {merged_features.shape[1] - 3} 特征")

# 6.3 聚类特征
print("生成聚类特征...")
numeric_cols = merged_features.select_dtypes(include=[np.number]).columns
numeric_cols = [col for col in numeric_cols if col not in ['patient_id', 'sample_id', 'target_liver_stiffness']]

if len(numeric_cols) > 5:
    # 选择前20个最重要的特征进行聚类
    cluster_data = merged_features[numeric_cols[:20]].fillna(0)
    
    # K-means聚类
    for n_clusters in [3, 5, 8]:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(cluster_data)
        merged_features[f'cluster_{n_clusters}'] = cluster_labels
        
        # 到各聚类中心的距离
        distances = kmeans.transform(cluster_data)
        for i in range(n_clusters):
            merged_features[f'dist_to_cluster_{n_clusters}_{i}'] = distances[:, i]

print(f"添加聚类特征后: {merged_features.shape[1] - 3} 特征")

# 6.4 PCA特征
print("生成PCA特征...")
if len(numeric_cols) > 10:
    pca_data = merged_features[numeric_cols].fillna(0)
    
    # 标准化
    scaler = StandardScaler()
    pca_data_scaled = scaler.fit_transform(pca_data)
    
    # PCA
    pca = PCA(n_components=min(10, len(numeric_cols)))
    pca_features = pca.fit_transform(pca_data_scaled)
    
    # 添加PCA特征
    for i in range(pca_features.shape[1]):
        merged_features[f'pca_{i}'] = pca_features[:, i]
    
    print(f"PCA解释方差比: {pca.explained_variance_ratio_[:5].round(3)}")

print(f"最终特征数: {merged_features.shape[1] - 3}")

# Step 7: 特征选择和评估
print("\n7. 特征选择和模型评估")

# 准备数据
feature_cols_new = [col for col in merged_features.columns 
                   if col not in ['patient_id', 'sample_id', 'target_liver_stiffness']]
X_enhanced = merged_features[feature_cols_new].fillna(0)
y_enhanced = merged_features['target_liver_stiffness']

# 处理无穷大值
X_enhanced = X_enhanced.replace([np.inf, -np.inf], 0)

X_train_enh, X_test_enh, y_train_enh, y_test_enh = train_test_split(
    X_enhanced, y_enhanced, test_size=0.2, random_state=42
)

print(f"增强特征集大小: {X_enhanced.shape}")

# 7.1 基于方差的特征选择
from sklearn.feature_selection import VarianceThreshold
var_selector = VarianceThreshold(threshold=0.01)
X_train_var = var_selector.fit_transform(X_train_enh)
X_test_var = var_selector.transform(X_test_enh)

print(f"方差过滤后特征数: {X_train_var.shape[1]}")

# 7.2 基于相关性的特征选择
correlation_selector = SelectKBest(f_regression, k=min(100, X_train_var.shape[1]))
X_train_corr = correlation_selector.fit_transform(X_train_var, y_train_enh)
X_test_corr = correlation_selector.transform(X_test_var)

selected_features = var_selector.get_feature_names_out(feature_cols_new)[correlation_selector.get_support()]
print(f"相关性选择后特征数: {X_train_corr.shape[1]}")

# 7.3 训练增强模型
print("\n训练增强特征模型...")

# GradientBoosting
gb_enhanced = GradientBoostingRegressor(
    n_estimators=200, 
    learning_rate=0.1, 
    max_depth=6,
    min_samples_split=10,
    random_state=42
)
gb_enhanced.fit(X_train_corr, y_train_enh)

# 预测
pred_enhanced = gb_enhanced.predict(X_test_corr)
mae_enhanced = mean_absolute_error(y_test_enh, pred_enhanced)
r2_enhanced = r2_score(y_test_enh, pred_enhanced)

print(f"增强特征模型性能:")
print(f"MAE: {mae_enhanced:.3f} kPa")
print(f"R²: {r2_enhanced:.3f}")

# RandomForest对比
rf_enhanced = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    min_samples_split=5,
    random_state=42
)
rf_enhanced.fit(X_train_corr, y_train_enh)

pred_rf_enhanced = rf_enhanced.predict(X_test_corr)
mae_rf_enhanced = mean_absolute_error(y_test_enh, pred_rf_enhanced)
r2_rf_enhanced = r2_score(y_test_enh, pred_rf_enhanced)

print(f"随机森林增强模型:")
print(f"MAE: {mae_rf_enhanced:.3f} kPa")
print(f"R²: {r2_rf_enhanced:.3f}")

# Step 8: 结果对比和可视化
print("\n8. 生成特征工程结果对比")

# 对比结果
results = [
    {'Model': 'Baseline_GB', 'Features': len(feature_cols), 'MAE': baseline_mae, 'R²': baseline_r2},
    {'Model': 'Enhanced_GB', 'Features': X_train_corr.shape[1], 'MAE': mae_enhanced, 'R²': r2_enhanced},
    {'Model': 'Enhanced_RF', 'Features': X_train_corr.shape[1], 'MAE': mae_rf_enhanced, 'R²': r2_rf_enhanced}
]

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('R²', ascending=False)

print("\n特征工程结果对比:")
print(results_df.round(3))

# 可视化
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 8.1 性能对比
ax1 = axes[0, 0]
x_pos = range(len(results_df))
bars1 = ax1.bar([x - 0.2 for x in x_pos], results_df['MAE'], 0.4, label='MAE', color='lightcoral')
ax2 = ax1.twinx()
bars2 = ax2.bar([x + 0.2 for x in x_pos], results_df['R²'], 0.4, label='R²', color='skyblue')

ax1.set_xlabel('Model')
ax1.set_ylabel('MAE (kPa)', color='red')
ax2.set_ylabel('R² Score', color='blue')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(results_df['Model'], rotation=45)
ax1.set_title('Model Performance Comparison')

# 添加数值标签
for bar, val in zip(bars1, results_df['MAE']):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{val:.3f}', ha='center', va='bottom')

for bar, val in zip(bars2, results_df['R²']):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{val:.3f}', ha='center', va='bottom')

# 8.2 特征数量对比
ax3 = axes[0, 1]
ax3.bar(results_df['Model'], results_df['Features'], color='lightgreen', edgecolor='black')
ax3.set_title('Number of Features Used')
ax3.set_ylabel('Feature Count')
ax3.tick_params(axis='x', rotation=45)

# 8.3 R²改进
ax4 = axes[0, 2]
baseline_r2_val = results_df[results_df['Model'] == 'Baseline_GB']['R²'].iloc[0]
best_r2_val = results_df.iloc[0]['R²']
improvement = ((best_r2_val - baseline_r2_val) / baseline_r2_val) * 100

ax4.bar(['Baseline', 'Best Enhanced'], [baseline_r2_val, best_r2_val], 
        color=['lightcoral', 'lightgreen'], edgecolor='black')
ax4.set_title(f'R² Improvement: +{improvement:.1f}%')
ax4.set_ylabel('R² Score')

# 添加改进箭头
ax4.annotate('', xy=(1, best_r2_val), xytext=(0, baseline_r2_val),
            arrowprops=dict(arrowstyle='->', lw=2, color='red'))
ax4.text(0.5, (baseline_r2_val + best_r2_val)/2, f'+{improvement:.1f}%',
         ha='center', va='center', fontweight='bold', color='red')

# 8.4 特征重要性（前15个）
ax5 = axes[1, 0]
if hasattr(gb_enhanced, 'feature_importances_'):
    feature_importance = gb_enhanced.feature_importances_
    top_indices = np.argsort(feature_importance)[-15:]
    top_features = [selected_features[i] for i in top_indices]
    top_importance = feature_importance[top_indices]
    
    ax5.barh(range(len(top_features)), top_importance, color='gold', edgecolor='black')
    ax5.set_yticks(range(len(top_features)))
    ax5.set_yticklabels([f.replace('poly_', '').replace('hist_', '')[:15] for f in top_features])
    ax5.set_title('Top 15 Feature Importance')
    ax5.set_xlabel('Importance')

# 8.5 预测散点图
ax6 = axes[1, 1]
ax6.scatter(y_test_enh, pred_enhanced, alpha=0.6, s=30)
ax6.plot([y_test_enh.min(), y_test_enh.max()], [y_test_enh.min(), y_test_enh.max()], 'r--', alpha=0.7)
ax6.set_xlabel('Actual Values')
ax6.set_ylabel('Predicted Values')
ax6.set_title(f'Enhanced Model Predictions\n(R² = {r2_enhanced:.3f})')
ax6.grid(True, alpha=0.3)

# 8.6 残差分布
ax7 = axes[1, 2]
residuals = pred_enhanced - y_test_enh
ax7.hist(residuals, bins=30, alpha=0.7, color='purple', edgecolor='black')
ax7.set_xlabel('Residuals (kPa)')
ax7.set_ylabel('Frequency')
ax7.set_title('Residual Distribution')
ax7.axvline(x=0, color='red', linestyle='--', alpha=0.7)
ax7.grid(True, alpha=0.3)

plt.tight_layout()
# Save both PNG and PDF formats
plt.savefig('optimization/advanced_feature_engineering_results.png', dpi=300, bbox_inches='tight')
plt.savefig('optimization/advanced_feature_engineering_results.pdf', dpi=300, bbox_inches='tight')
plt.show()

# Step 9: 保存增强特征和模型
print("\n9. 保存增强特征工程结果")

# 保存增强特征数据
enhanced_features_final = merged_features.copy()
enhanced_features_final.to_csv('optimization/results/enhanced_features_dataset.csv', index=False)

# 保存最佳模型
best_model = gb_enhanced if r2_enhanced > r2_rf_enhanced else rf_enhanced
best_model_name = 'GradientBoosting' if r2_enhanced > r2_rf_enhanced else 'RandomForest'
best_r2 = max(r2_enhanced, r2_rf_enhanced)

joblib.dump(best_model, f'optimization/models/best_enhanced_features_model.pkl')

# 保存特征选择器
joblib.dump({
    'variance_selector': var_selector,
    'correlation_selector': correlation_selector,
    'selected_features': selected_features
}, 'optimization/models/feature_selectors.pkl')

# 保存结果
results_df.to_csv('optimization/results/feature_engineering_results.csv', index=False)

# 特征重要性
if hasattr(best_model, 'feature_importances_'):
    feature_importance_df = pd.DataFrame({
        'Feature': selected_features,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    feature_importance_df.to_csv('optimization/results/enhanced_feature_importance.csv', index=False)

# 生成最终报告
print("\n========== 高级特征工程摘要报告 ==========")
print(f"基准模型性能:")
print(f"- 特征数: {len(feature_cols)}")
print(f"- MAE: {baseline_mae:.3f} kPa")
print(f"- R²: {baseline_r2:.3f}")

print(f"\n最佳增强模型: {best_model_name}")
print(f"- 特征数: {X_train_corr.shape[1]} (优化后)")
print(f"- MAE: {results_df.iloc[0]['MAE']:.3f} kPa")
print(f"- R²: {results_df.iloc[0]['R²']:.3f}")

improvement_r2 = ((best_r2 - baseline_r2) / baseline_r2) * 100
improvement_mae = ((baseline_mae - results_df.iloc[0]['MAE']) / baseline_mae) * 100

print(f"\n关键改进:")
print(f"✅ R²提升: {improvement_r2:.1f}%")
print(f"✅ MAE{'降低' if improvement_mae > 0 else '变化'}: {improvement_mae:.1f}%")

print(f"\n新增特征类型:")
print(f"- 高级统计特征: 矩特征、分位数特征、鲁棒性指标")
print(f"- 时间序列特征: 趋势复杂度、周期性、自相关")
print(f"- 多变量交互: 生化指标协同、综合评分")
print(f"- 组合特征: 多项式交互、聚类特征、PCA特征")

if best_r2 > 0.5:
    print("🎉 模型R²已达到优秀水平 (>0.5)")
elif best_r2 > 0.4:
    print("🎉 模型R²已达到良好水平 (>0.4)")
elif best_r2 > 0.35:
    print("✅模型R²达到可接受水平 (>0.35)")

print(f"\n保存的文件:")
print(f"- optimization/results/enhanced_features_dataset.csv: 增强特征数据集")
print(f"- optimization/models/best_enhanced_features_model.pkl: 最佳增强模型")
print(f"- optimization/results/enhanced_feature_importance.csv: 特征重要性")

print(f"\n完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("高级特征工程完成！")