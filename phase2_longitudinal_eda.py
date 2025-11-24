#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2: 纵向EDA - 时间序列探索
基于纵向历史数据预测最后一次肝脏弹性值的完整分析方案
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set English display
plt.rcParams['font.family'] = 'DejaVu Sans'

print("========== Phase 2: 纵向EDA - 时间序列探索 ==========")
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("目标: 深入分析时间序列模式，发现预测性特征")

# Step 1: 加载预处理数据
print("\n1. 加载预处理数据")
longitudinal_df = pd.read_csv('data/processed/phase1_longitudinal_data.csv')
time_series_stats_df = pd.read_csv('data/processed/phase1_time_series_statistics.csv')
prediction_df = pd.read_csv('data/processed/phase1_prediction_dataset.csv')

# 转换日期格式
longitudinal_df['examination_date'] = pd.to_datetime(longitudinal_df['examination_date'])
time_series_stats_df['baseline_date'] = pd.to_datetime(time_series_stats_df['baseline_date'])
time_series_stats_df['final_date'] = pd.to_datetime(time_series_stats_df['final_date'])

print(f"纵向数据记录数: {len(longitudinal_df)}")
print(f"患者数: {len(time_series_stats_df)}")
print(f"预测任务数: {len(prediction_df)}")

# Step 2: 时间序列模式分析
print("\n2. 时间序列模式分析")

# 2.1 个体轨迹可视化
print("2.1 个体轨迹可视化")

# 选择不同模式的典型患者
patients_by_trend = []

for _, patient in time_series_stats_df.iterrows():
    patient_id = patient['patient_id']
    patient_data = longitudinal_df[longitudinal_df['patient_id'] == patient_id].copy()
    patient_data = patient_data.sort_values('examination_date')
    
    if len(patient_data) >= 3:  # 至少3个点才能看趋势
        liver_values = patient_data['liver_stiffness'].values
        times = patient_data['days_from_baseline'].values
        
        # 计算线性趋势
        slope, _, r_value, _, _ = stats.linregress(times, liver_values)
        
        patients_by_trend.append({
            'patient_id': patient_id,
            'slope': slope,
            'r_squared': r_value**2,
            'baseline_value': liver_values[0],
            'final_value': liver_values[-1],
            'total_change': liver_values[-1] - liver_values[0],
            'record_count': len(patient_data),
            'follow_up_days': max(times)
        })

trend_df = pd.DataFrame(patients_by_trend)

# 分类患者
trend_df['trend_category'] = 'Stable'
trend_df.loc[trend_df['slope'] > 0.01, 'trend_category'] = 'Increasing'  
trend_df.loc[trend_df['slope'] < -0.01, 'trend_category'] = 'Decreasing'

print(f"趋势分类统计:")
print(trend_df['trend_category'].value_counts())

# 可视化典型轨迹
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 选择典型患者
categories = ['Increasing', 'Decreasing', 'Stable']
colors = ['red', 'blue', 'green']

for i, (cat, color) in enumerate(zip(categories, colors)):
    cat_patients = trend_df[trend_df['trend_category'] == cat]
    
    # 选择典型患者（按r_squared排序选前5个）
    if len(cat_patients) > 0:
        top_patients = cat_patients.nlargest(min(5, len(cat_patients)), 'r_squared')
        
        ax = axes[0, i]
        for _, patient in top_patients.iterrows():
            patient_data = longitudinal_df[longitudinal_df['patient_id'] == patient['patient_id']].copy()
            patient_data = patient_data.sort_values('examination_date')
            
            times = patient_data['days_from_baseline'].values / 365.25  # 转换为年
            liver_values = patient_data['liver_stiffness'].values
            
            ax.plot(times, liver_values, 'o-', alpha=0.7, linewidth=2, markersize=6)
        
        ax.set_title(f'{cat} Trend (n={len(cat_patients)})')
        ax.set_xlabel('Time (years)')
        ax.set_ylabel('Liver Stiffness (kPa)')
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('white')

# 2.2 群体趋势分析
print("2.2 群体趋势分析")

# Baseline level groups
baseline_groups = pd.cut(time_series_stats_df['baseline_liver_stiffness'], 
                        bins=[0, 7, 12.5, 20, 100], 
                        labels=['Normal(<7)', 'Mild(7-12.5)', 'Moderate(12.5-20)', 'Severe(>20)'])

time_series_stats_df['baseline_group'] = baseline_groups

# Baseline group trajectory comparison
for i, group in enumerate(['Normal(<7)', 'Mild(7-12.5)', 'Moderate(12.5-20)']):
    if i >= 3: break  # Only show first 3 groups
    
    ax = axes[1, i]
    group_patients = time_series_stats_df[time_series_stats_df['baseline_group'] == group]['patient_id']
    
    # Randomly select some patients from this group
    sample_patients = np.random.choice(group_patients, min(20, len(group_patients)), replace=False)
    
    for patient_id in sample_patients:
        patient_data = longitudinal_df[longitudinal_df['patient_id'] == patient_id].copy()
        patient_data = patient_data.sort_values('examination_date')
        
        if len(patient_data) >= 2:
            times = patient_data['days_from_baseline'].values / 365.25
            liver_values = patient_data['liver_stiffness'].values
            ax.plot(times, liver_values, 'o-', alpha=0.5, linewidth=1, markersize=4)
    
    ax.set_title(f'{group} Baseline Group (n={len(group_patients)})')
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Liver Stiffness (kPa)')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
# Save both PNG and PDF formats
plt.savefig('data/processed/phase2_individual_trajectories.png', dpi=300, bbox_inches='tight')
plt.savefig('data/processed/phase2_individual_trajectories.pdf', dpi=300, bbox_inches='tight')
plt.show()

# Step 3: 预测性特征发现
print("\n3. 预测性特征发现")

# 3.1 历史模式与最终值相关性分析
print("3.1 历史模式与最终值相关性分析")

# 创建扩展的预测特征
enhanced_prediction_data = []

for _, row in prediction_df.iterrows():
    patient_id = row['patient_id']
    patient_data = longitudinal_df[longitudinal_df['patient_id'] == patient_id].copy()
    patient_data = patient_data.sort_values('examination_date')
    
    historical_data = patient_data.iloc[:-1]  # 历史数据
    target_value = patient_data.iloc[-1]['liver_stiffness']  # 目标值
    
    if len(historical_data) >= 1:
        liver_values = historical_data['liver_stiffness'].values
        times = historical_data['days_from_baseline'].values
        
        # 基础统计特征
        features = {
            'patient_id': patient_id,
            'target_liver_stiffness': target_value,
            'baseline_value': liver_values[0],
            'historical_count': len(liver_values),
            'hist_mean': np.mean(liver_values),
            'hist_std': np.std(liver_values) if len(liver_values) > 1 else 0,
            'hist_min': np.min(liver_values),
            'hist_max': np.max(liver_values),
            'hist_range': np.max(liver_values) - np.min(liver_values),
            'hist_cv': np.std(liver_values) / np.mean(liver_values) if np.mean(liver_values) > 0 else 0
        }
        
        # 趋势特征
        if len(liver_values) >= 2:
            slope, _, r_value, _, _ = stats.linregress(times, liver_values)
            features.update({
                'trend_slope': slope,
                'trend_r_squared': r_value**2,
                'last_value': liver_values[-1],
                'early_late_ratio': liver_values[-1] / liver_values[0] if liver_values[0] > 0 else 1,
                'total_change': liver_values[-1] - liver_values[0],
                'change_rate': (liver_values[-1] - liver_values[0]) / (max(times) / 365.25) if max(times) > 0 else 0
            })
        else:
            features.update({
                'trend_slope': 0, 'trend_r_squared': 0, 'last_value': liver_values[0],
                'early_late_ratio': 1, 'total_change': 0, 'change_rate': 0
            })
        
        # 波动性特征
        if len(liver_values) >= 3:
            # 变化率标准差
            changes = np.diff(liver_values)
            features.update({
                'volatility': np.std(changes),
                'max_jump': np.max(np.abs(changes)),
                'positive_changes': np.sum(changes > 0),
                'negative_changes': np.sum(changes < 0),
                'change_direction_consistency': abs(np.sum(changes > 0) - np.sum(changes < 0)) / len(changes)
            })
        else:
            features.update({
                'volatility': 0, 'max_jump': 0, 'positive_changes': 0, 
                'negative_changes': 0, 'change_direction_consistency': 0
            })
        
        enhanced_prediction_data.append(features)

enhanced_pred_df = pd.DataFrame(enhanced_prediction_data)

# 3.2 特征与目标值相关性分析
print("3.2 特征与目标值相关性分析")

# 计算相关性
numeric_features = ['baseline_value', 'hist_mean', 'hist_std', 'hist_range', 'hist_cv',
                   'trend_slope', 'trend_r_squared', 'last_value', 'early_late_ratio',
                   'total_change', 'change_rate', 'volatility', 'max_jump']

correlation_with_target = {}
for feature in numeric_features:
    if feature in enhanced_pred_df.columns:
        corr = enhanced_pred_df[feature].corr(enhanced_pred_df['target_liver_stiffness'])
        correlation_with_target[feature] = corr

# 排序并可视化
corr_df = pd.DataFrame(list(correlation_with_target.items()), columns=['Feature', 'Correlation'])
corr_df = corr_df.sort_values('Correlation', key=abs, ascending=False)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Correlation bar chart
ax = axes[0, 0]
bars = ax.barh(corr_df['Feature'], corr_df['Correlation'])
ax.set_title('Feature Correlation with Target')
ax.set_xlabel('Pearson Correlation Coefficient')
ax.grid(True, alpha=0.3)

# Add colors to bars
for i, bar in enumerate(bars):
    if corr_df.iloc[i]['Correlation'] > 0:
        bar.set_color('red')
    else:
        bar.set_color('blue')

print("Feature importance ranking:")
for i, row in corr_df.head(10).iterrows():
    print(f"{row['Feature']}: {row['Correlation']:.3f}")

# Most important feature scatter plots
top_features = corr_df['Feature'].head(3).tolist()

for i, feature in enumerate(top_features[:3]):
    ax = axes[0, 1] if i == 0 else axes[1, 0] if i == 1 else axes[1, 1]
    
    x = enhanced_pred_df[feature]
    y = enhanced_pred_df['target_liver_stiffness']
    
    ax.scatter(x, y, alpha=0.6, s=50)
    
    # Add regression line
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    ax.plot(x, p(x), "r--", alpha=0.8, linewidth=2)
    
    ax.set_xlabel(f'{feature}')
    ax.set_ylabel('Target Liver Stiffness (kPa)')
    ax.set_title(f'{feature} vs Target (r={correlation_with_target[feature]:.3f})')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
# Save both PNG and PDF formats
plt.savefig('data/processed/phase2_predictive_features.png', dpi=300, bbox_inches='tight')
plt.savefig('data/processed/phase2_predictive_features.pdf', dpi=300, bbox_inches='tight')
plt.show()

# Step 4: 临床亚组分析
print("\n4. 临床亚组分析")

# 4.1 病程模式聚类
print("4.1 基于轨迹形状的患者聚类")

# 准备聚类特征
cluster_features = enhanced_pred_df[['baseline_value', 'hist_mean', 'trend_slope', 
                                   'hist_cv', 'change_rate', 'volatility']].fillna(0)

# 标准化特征
scaler = StandardScaler()
cluster_features_scaled = scaler.fit_transform(cluster_features)

# K-means聚类
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(cluster_features_scaled)

enhanced_pred_df['trajectory_cluster'] = cluster_labels

# 分析各聚类的特征
print("聚类结果分析:")
cluster_analysis = enhanced_pred_df.groupby('trajectory_cluster').agg({
    'baseline_value': ['mean', 'std'],
    'trend_slope': ['mean', 'std'], 
    'hist_cv': ['mean', 'std'],
    'target_liver_stiffness': ['mean', 'std'],
    'patient_id': 'count'
}).round(3)

cluster_analysis.columns = ['_'.join(col).strip() for col in cluster_analysis.columns]
print(cluster_analysis)

# 可视化聚类结果
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 不同聚类的轨迹展示
for cluster_id in range(min(4, n_clusters)):
    ax = axes[cluster_id // 2, cluster_id % 2] if cluster_id < 4 else None
    if ax is None: continue
    
    cluster_patients = enhanced_pred_df[enhanced_pred_df['trajectory_cluster'] == cluster_id]['patient_id'].tolist()
    
    # 随机选择该聚类的部分患者展示轨迹
    sample_patients = np.random.choice(cluster_patients, min(15, len(cluster_patients)), replace=False)
    
    colors = plt.cm.Set1(cluster_id)
    
    for patient_id in sample_patients:
        patient_data = longitudinal_df[longitudinal_df['patient_id'] == patient_id].copy()
        patient_data = patient_data.sort_values('examination_date')
        
        times = patient_data['days_from_baseline'].values / 365.25
        liver_values = patient_data['liver_stiffness'].values
        ax.plot(times, liver_values, 'o-', color=colors, alpha=0.7, linewidth=2, markersize=4)
    
    ax.set_title(f'Cluster {cluster_id} (n={len(cluster_patients)})')
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Liver Stiffness (kPa)')
    ax.grid(True, alpha=0.3)

# Cluster feature comparison
ax = axes[1, 2]
cluster_means = enhanced_pred_df.groupby('trajectory_cluster')[['baseline_value', 'trend_slope', 'hist_cv', 'target_liver_stiffness']].mean()
cluster_means.T.plot(kind='bar', ax=ax, width=0.8)
ax.set_title('Cluster Feature Comparison')
ax.set_xlabel('Features')
ax.set_ylabel('Mean Value')
ax.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
# Save both PNG and PDF formats
plt.savefig('data/processed/phase2_trajectory_clusters.png', dpi=300, bbox_inches='tight')
plt.savefig('data/processed/phase2_trajectory_clusters.pdf', dpi=300, bbox_inches='tight')
plt.show()

# Step 5: 时间依赖性分析
print("\n5. 时间依赖性分析")

# 5.1 不同时间间隔的预测价值
print("5.1 时间间隔与预测价值分析")

# 分析预测跨度与预测难度的关系
span_analysis = []

for _, row in enhanced_pred_df.iterrows():
    patient_id = row['patient_id']
    patient_data = longitudinal_df[longitudinal_df['patient_id'] == patient_id].copy()
    patient_data = patient_data.sort_values('examination_date')
    
    if len(patient_data) >= 2:
        span_days = patient_data.iloc[-1]['days_from_baseline']
        span_years = span_days / 365.25
        
        baseline_value = patient_data.iloc[0]['liver_stiffness']
        target_value = patient_data.iloc[-1]['liver_stiffness']
        
        # 计算预测误差（使用基线值作为简单预测）
        prediction_error = abs(target_value - baseline_value)
        
        span_analysis.append({
            'patient_id': patient_id,
            'span_years': span_years,
            'span_days': span_days,
            'baseline_prediction_error': prediction_error,
            'baseline_value': baseline_value,
            'target_value': target_value,
            'historical_count': len(patient_data) - 1
        })

span_df = pd.DataFrame(span_analysis)

# 按时间跨度分组分析
span_groups = pd.cut(span_df['span_years'], bins=[0, 1, 2, 3, 10], labels=['<1 year', '1-2 years', '2-3 years', '>3 years'])
span_df['span_group'] = span_groups

span_group_stats = span_df.groupby('span_group').agg({
    'baseline_prediction_error': ['mean', 'std'],
    'baseline_value': 'mean',
    'target_value': 'mean',
    'patient_id': 'count'
}).round(3)

print("不同预测时间跨度的预测难度:")
print(span_group_stats)

# 可视化时间依赖性
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Prediction error vs time span
ax = axes[0]
ax.scatter(span_df['span_years'], span_df['baseline_prediction_error'], alpha=0.6)
ax.set_xlabel('Prediction Time Span (years)')
ax.set_ylabel('Baseline Prediction Error (kPa)')
ax.set_title('Prediction Time Span vs Prediction Difficulty')
ax.grid(True, alpha=0.3)

# Error distribution by different time span groups
ax = axes[1]
span_groups_data = [span_df[span_df['span_group'] == group]['baseline_prediction_error'].dropna() 
                   for group in ['<1 year', '1-2 years', '2-3 years', '>3 years']]
ax.boxplot(span_groups_data, labels=['<1yr', '1-2yr', '2-3yr', '>3yr'])
ax.set_xlabel('Prediction Time Span')
ax.set_ylabel('Baseline Prediction Error (kPa)')
ax.set_title('Prediction Error Distribution by Time Span')
ax.grid(True, alpha=0.3)

# Historical record count vs prediction error
ax = axes[2]
ax.scatter(span_df['historical_count'], span_df['baseline_prediction_error'], alpha=0.6)
ax.set_xlabel('Number of Historical Records')
ax.set_ylabel('Baseline Prediction Error (kPa)')
ax.set_title('Historical Data Volume vs Prediction Difficulty')
ax.grid(True, alpha=0.3)

plt.tight_layout()
# Save both PNG and PDF formats
plt.savefig('data/processed/phase2_time_dependency_analysis.png', dpi=300, bbox_inches='tight')
plt.savefig('data/processed/phase2_time_dependency_analysis.pdf', dpi=300, bbox_inches='tight')
plt.show()

# Step 6: 保存EDA结果
print("\n6. 保存EDA结果")

# 保存增强的预测数据集
enhanced_pred_df.to_csv('data/processed/phase2_enhanced_prediction_features.csv', index=False)

# 保存趋势分析结果
trend_df.to_csv('data/processed/phase2_trajectory_trend_analysis.csv', index=False)

# 保存时间跨度分析结果
span_df.to_csv('data/processed/phase2_time_span_analysis.csv', index=False)

print("保存文件:")
print("- data/processed/phase2_enhanced_prediction_features.csv: 增强的预测特征数据")
print("- data/processed/phase2_trajectory_trend_analysis.csv: 轨迹趋势分析结果")
print("- data/processed/phase2_time_span_analysis.csv: 时间跨度分析结果")

# Step 7: EDA摘要报告
print("\n========== Phase 2 EDA摘要报告 ==========")

print(f"\n时间序列模式分析:")
print(f"- 上升趋势患者: {len(trend_df[trend_df['trend_category']=='Increasing'])} ({len(trend_df[trend_df['trend_category']=='Increasing'])/len(trend_df)*100:.1f}%)")
print(f"- 下降趋势患者: {len(trend_df[trend_df['trend_category']=='Decreasing'])} ({len(trend_df[trend_df['trend_category']=='Decreasing'])/len(trend_df)*100:.1f}%)")
print(f"- 稳定趋势患者: {len(trend_df[trend_df['trend_category']=='Stable'])} ({len(trend_df[trend_df['trend_category']=='Stable'])/len(trend_df)*100:.1f}%)")

print(f"\n预测性特征发现:")
print("最重要的预测特征:")
for i, (_, row) in enumerate(corr_df.head(5).iterrows()):
    print(f"  {i+1}. {row['Feature']}: r={row['Correlation']:.3f}")

print(f"\n轨迹聚类分析:")
for i in range(n_clusters):
    cluster_size = len(enhanced_pred_df[enhanced_pred_df['trajectory_cluster'] == i])
    cluster_mean_target = enhanced_pred_df[enhanced_pred_df['trajectory_cluster'] == i]['target_liver_stiffness'].mean()
    print(f"- 聚类 {i}: {cluster_size} 患者, 平均目标值 {cluster_mean_target:.2f} kPa")

print(f"\n时间依赖性分析:")
print("预测时间跨度分布:")
for group in ['<1年', '1-2年', '2-3年', '>3年']:
    if group in span_df['span_group'].cat.categories:
        count = len(span_df[span_df['span_group'] == group])
        avg_error = span_df[span_df['span_group'] == group]['baseline_prediction_error'].mean()
        print(f"- {group}: {count} 患者, 平均预测误差 {avg_error:.2f} kPa")

print(f"\n完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("Phase 2 纵向EDA完成！")