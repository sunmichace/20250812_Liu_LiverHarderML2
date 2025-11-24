#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 3: 时间序列特征工程 - 基于2984个预测样本
基于纵向历史数据预测下一次肝脏弹性值的完整分析方案
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set English display
plt.rcParams['font.family'] = 'DejaVu Sans'

print("========== Phase 3: 时间序列特征工程 (2984样本版) ==========")
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("目标: 基于2984个预测样本构建全面的时间序列特征集")

# Step 1: 加载数据
print("\n1. 加载数据")
longitudinal_df = pd.read_csv('data/processed/phase1_longitudinal_data.csv')
prediction_df = pd.read_csv('data/processed/phase1_prediction_dataset.csv')

# 转换日期格式
longitudinal_df['examination_date'] = pd.to_datetime(longitudinal_df['examination_date'])

print(f"纵向数据记录数: {len(longitudinal_df)}")
print(f"预测样本数: {len(prediction_df)}")

# Step 2: 构建时间序列特征提取函数
print("\n2. 构建时间序列特征集")

def extract_comprehensive_features(sample_row, longitudinal_data):
    """为单个预测样本提取全面的时间序列特征"""
    
    patient_id = sample_row['patient_id']
    sample_id = sample_row['sample_id']
    target_value = sample_row['target_liver_stiffness']
    historical_count = sample_row['historical_count']
    
    # 获取患者完整数据
    patient_data = longitudinal_data[longitudinal_data['patient_id'] == patient_id].copy()
    patient_data = patient_data.sort_values('examination_date')
    
    if len(patient_data) < 2:
        return None
    
    # 根据historical_count确定该样本的历史数据范围
    historical_data = patient_data.iloc[:historical_count]
    
    # 基础数据
    liver_values = historical_data['liver_stiffness'].values
    times = historical_data['days_from_baseline'].values
    dates = historical_data['examination_date'].values
    
    # 其他生理指标历史数据
    alt_values = historical_data['ALT'].fillna(historical_data['ALT'].median()).values
    ast_values = historical_data['AST'].fillna(historical_data['AST'].median()).values  
    bilirubin_values = historical_data['total_bilirubin'].fillna(historical_data['total_bilirubin'].median()).values
    hbv_dna_values = historical_data['HBV_DNA'].fillna(0).values
    
    features = {
        'patient_id': patient_id, 
        'sample_id': sample_id,
        'target_liver_stiffness': target_value
    }
    
    # ========== 1. 基础历史统计特征 ==========
    features.update({
        'hist_count': len(liver_values),
        'hist_mean': np.mean(liver_values),
        'hist_median': np.median(liver_values),
        'hist_std': np.std(liver_values) if len(liver_values) > 1 else 0,
        'hist_min': np.min(liver_values),
        'hist_max': np.max(liver_values),
        'hist_range': np.max(liver_values) - np.min(liver_values),
        'hist_cv': np.std(liver_values) / np.mean(liver_values) if np.mean(liver_values) > 0 and len(liver_values) > 1 else 0,
        'hist_iqr': np.percentile(liver_values, 75) - np.percentile(liver_values, 25),
        'hist_skewness': stats.skew(liver_values) if len(liver_values) >= 3 else 0,
        'hist_kurtosis': stats.kurtosis(liver_values) if len(liver_values) >= 4 else 0
    })
    
    # ========== 2. 时间趋势特征 ==========
    if len(liver_values) >= 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(times, liver_values)
        features.update({
            'trend_slope': slope,
            'trend_intercept': intercept,
            'trend_r_squared': r_value**2,
            'trend_p_value': p_value,
            'trend_std_err': std_err,
        })
        
        # 二次趋势
        if len(liver_values) >= 3:
            coeffs = np.polyfit(times, liver_values, 2)
            features.update({
                'trend_quadratic_a': coeffs[0],
                'trend_quadratic_b': coeffs[1], 
                'trend_quadratic_c': coeffs[2]
            })
        else:
            features.update({'trend_quadratic_a': 0, 'trend_quadratic_b': slope, 'trend_quadratic_c': intercept})
    else:
        features.update({
            'trend_slope': 0, 'trend_intercept': liver_values[0], 'trend_r_squared': 0,
            'trend_p_value': 1, 'trend_std_err': 0,
            'trend_quadratic_a': 0, 'trend_quadratic_b': 0, 'trend_quadratic_c': liver_values[0]
        })
    
    # ========== 3. 时间窗口特征 ==========
    if len(liver_values) >= 2:
        mid_point = len(liver_values) // 2
        early_values = liver_values[:mid_point+1]
        late_values = liver_values[mid_point:]
        
        features.update({
            'early_mean': np.mean(early_values),
            'late_mean': np.mean(late_values),
            'middle_mean': np.mean(liver_values[mid_point-1:mid_point+2]) if mid_point > 0 and len(liver_values) > 2 else np.mean(liver_values),
            'recent_mean': np.mean(liver_values[-2:]) if len(liver_values) >= 2 else liver_values[-1],
        })
    else:
        features.update({
            'early_mean': liver_values[0], 'late_mean': liver_values[0],
            'middle_mean': liver_values[0], 'recent_mean': liver_values[0]
        })
    
    # ========== 4. 变化率特征 ==========
    if len(liver_values) >= 2:
        changes = np.diff(liver_values)
        features.update({
            'change_mean': np.mean(changes),
            'change_std': np.std(changes),
            'change_max_abs': np.max(np.abs(changes)),
            'change_positive_count': np.sum(changes > 0),
            'change_negative_count': np.sum(changes < 0),
        })
        
        # 近期趋势(最后2-3个点)
        if len(liver_values) >= 3:
            recent_values = liver_values[-3:]
            recent_times = times[-3:]
            recent_slope, _, recent_r2, _, _ = stats.linregress(recent_times, recent_values)
            features.update({
                'recent_trend_slope': recent_slope,
                'recent_trend_r2': recent_r2,
                'recent_vs_overall_trend': recent_slope - features['trend_slope']
            })
        else:
            features.update({'recent_trend_slope': 0, 'recent_trend_r2': 0, 'recent_vs_overall_trend': 0})
    else:
        features.update({
            'change_mean': 0, 'change_std': 0, 'change_max_abs': 0,
            'change_positive_count': 0, 'change_negative_count': 0,
            'recent_trend_slope': 0, 'recent_trend_r2': 0, 'recent_vs_overall_trend': 0
        })
    
    # ========== 5. 基线相关特征 ==========
    baseline_value = liver_values[0]
    features.update({
        'baseline_value': baseline_value,
        'baseline_clinical_stage': 1 if baseline_value < 7 else (2 if baseline_value < 12.5 else 3),
        'last_hist_value': liver_values[-1],
        'baseline_to_last_change': liver_values[-1] - baseline_value,
        'baseline_to_last_ratio': liver_values[-1] / baseline_value if baseline_value > 0 else 1
    })
    
    # ========== 6. 时间编码特征 ==========
    if len(times) > 1:
        time_intervals = np.diff(times)
        features.update({
            'detection_frequency': np.mean(time_intervals),
            'detection_regularity': np.std(time_intervals) / np.mean(time_intervals) if np.mean(time_intervals) > 0 else 0,
            'total_follow_up_days': times[-1] - times[0],
        })
    else:
        features.update({
            'detection_frequency': 0, 'detection_regularity': 0, 'total_follow_up_days': 0
        })
    
    # 季节性特征（基于最后一次检查日期）
    if len(dates) > 0:
        last_date = pd.to_datetime(dates[-1])
        day_of_year = last_date.timetuple().tm_yday
        features.update({
            'season_sin': np.sin(2 * np.pi * day_of_year / 365.25),
            'season_cos': np.cos(2 * np.pi * day_of_year / 365.25),
        })
    else:
        features.update({'season_sin': 0, 'season_cos': 0})
    
    # ========== 7. 多变量联合特征 ==========
    features.update({
        'alt_mean': np.mean(alt_values),
        'ast_mean': np.mean(ast_values),
        'alt_abnormal_count': np.sum(alt_values > 40),  # ALT正常值通常<40
        'ast_abnormal_count': np.sum(ast_values > 35),  # AST正常值通常<35
        'bilirubin_mean': np.mean(bilirubin_values),
        'hbv_dna_mean': np.mean(hbv_dna_values),
        'hbv_dna_positive_rate': np.mean(hbv_dna_values > 0),
    })
    
    # ========== 8. 序列形状特征 ==========
    if len(liver_values) >= 3:
        # 峰值和谷值
        from scipy.signal import argrelextrema
        peaks = argrelextrema(liver_values, np.greater)[0]
        valleys = argrelextrema(liver_values, np.less)[0]
        
        features.update({
            'peak_count': len(peaks),
            'valley_count': len(valleys),
            'peak_value': np.max(liver_values[peaks]) if len(peaks) > 0 else liver_values.max(),
            'valley_value': np.min(liver_values[valleys]) if len(valleys) > 0 else liver_values.min(),
        })
    else:
        features.update({
            'peak_count': 0, 'valley_count': 0,
            'peak_value': liver_values.max(), 'valley_value': liver_values.min()
        })
    
    # ========== 9. 相对变化特征 ==========
    if baseline_value > 0:
        features.update({
            'rel_change_mean': np.mean(liver_values) / baseline_value,
            'rel_change_max': np.max(liver_values) / baseline_value,
            'rel_change_current': liver_values[-1] / baseline_value
        })
    else:
        features.update({'rel_change_mean': 1, 'rel_change_max': 1, 'rel_change_current': 1})
    
    return features

# Step 3: 批量提取特征
print("\n3. 批量提取特征")
comprehensive_features = []
failed_count = 0

for idx, sample_row in prediction_df.iterrows():
    if idx % 500 == 0:
        print(f"处理进度: {idx}/{len(prediction_df)}")
    
    try:
        features = extract_comprehensive_features(sample_row, longitudinal_df)
        if features is not None:
            comprehensive_features.append(features)
        else:
            failed_count += 1
    except Exception as e:
        print(f"样本 {sample_row['sample_id']} 特征提取失败: {e}")
        failed_count += 1

comprehensive_features_df = pd.DataFrame(comprehensive_features)

print(f"成功提取特征的样本数: {len(comprehensive_features_df)}")
print(f"特征提取失败的样本数: {failed_count}")
print(f"总特征数量: {len(comprehensive_features_df.columns)-3}")  # 减去patient_id, sample_id, target

# Step 4: 特征质量检查
print("\n4. 特征质量检查")

# 4.1 缺失值统计
print("4.1 缺失值统计:")
missing_counts = comprehensive_features_df.isnull().sum()
missing_features = missing_counts[missing_counts > 0]
print(f"有缺失值的特征数: {len(missing_features)}")
if len(missing_features) > 0:
    print(missing_features.head(10))

# 4.2 异常值检查
print("\n4.2 异常值检查:")
numeric_cols = comprehensive_features_df.select_dtypes(include=[np.number]).columns
numeric_cols = [col for col in numeric_cols if col != 'target_liver_stiffness']

outlier_counts = {}
for col in numeric_cols[:10]:  # 只检查前10个特征
    if comprehensive_features_df[col].std() > 0:
        z_scores = np.abs(stats.zscore(comprehensive_features_df[col].fillna(comprehensive_features_df[col].mean())))
        outlier_count = np.sum(z_scores > 3)
        outlier_counts[col] = f"{outlier_count} ({outlier_count/len(comprehensive_features_df)*100:.1f}%)"

print("前10个特征的异常值统计:")
for col, count in outlier_counts.items():
    print(f"{col}: {count}")

# 4.3 特征分布可视化
print("\n4.3 特征分布可视化")
plt.figure(figsize=(20, 15))

# 选择主要特征进行可视化
main_features = ['hist_mean', 'trend_slope', 'hist_cv', 'baseline_value', 
                'change_max_abs', 'recent_trend_slope', 'detection_frequency', 'target_liver_stiffness']

for i, feature in enumerate(main_features):
    plt.subplot(2, 4, i+1)
    if feature in comprehensive_features_df.columns:
        data = comprehensive_features_df[feature].dropna()
        if feature == 'target_liver_stiffness':
            plt.hist(data, bins=30, alpha=0.7, color='red', edgecolor='black')
            plt.title(f'Target Value Distribution')
        else:
            plt.hist(data, bins=30, alpha=0.7, edgecolor='black')
            plt.title(f'{feature}')
        plt.xlabel('Value')
        plt.ylabel('Frequency')

plt.tight_layout()
# Save both PNG and PDF formats
plt.savefig('data/processed/phase3_feature_distributions.png', dpi=300, bbox_inches='tight')
plt.savefig('data/processed/phase3_feature_distributions.pdf', dpi=300, bbox_inches='tight')
plt.show()

# Step 5: 特征相关性分析
print("\n5. 特征相关性分析")
# 计算与目标变量的相关性
feature_cols = [col for col in comprehensive_features_df.columns 
                if col not in ['patient_id', 'sample_id', 'target_liver_stiffness']]

correlations = {}
for col in feature_cols:
    if comprehensive_features_df[col].dtype in ['int64', 'float64']:
        corr = comprehensive_features_df[col].corr(comprehensive_features_df['target_liver_stiffness'])
        if not np.isnan(corr):
            correlations[col] = corr

# 按相关性排序
correlation_df = pd.DataFrame(list(correlations.items()), columns=['Feature', 'Correlation'])
correlation_df = correlation_df.sort_values('Correlation', key=abs, ascending=False)

print("特征重要性排序（Top 20）:")
for i, (_, row) in enumerate(correlation_df.head(20).iterrows()):
    print(f"{i+1:2d}. {row['Feature']:<25}: {row['Correlation']:6.3f}")

# 特征相关性可视化
plt.figure(figsize=(16, 8))

# 左图：Top 20特征相关性
plt.subplot(1, 2, 1)
top_20 = correlation_df.head(20)
bars = plt.barh(range(len(top_20)), top_20['Correlation'])
plt.yticks(range(len(top_20)), top_20['Feature'])
plt.xlabel('Correlation Coefficient')
plt.title('Top 20 Feature Correlations with Target')
plt.grid(True, alpha=0.3)

# 颜色编码
for i, bar in enumerate(bars):
    if top_20.iloc[i]['Correlation'] > 0:
        bar.set_color('red')
    else:
        bar.set_color('blue')

# 右图：重要特征相关性矩阵
plt.subplot(1, 2, 2)
important_features = top_20['Feature'].head(10).tolist() + ['target_liver_stiffness']
corr_matrix = comprehensive_features_df[important_features].corr()
sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, 
            square=True, fmt='.2f', cbar_kws={"shrink": .8})
plt.title('Important Feature Correlation Matrix')

plt.tight_layout()
# Save both PNG and PDF formats
plt.savefig('data/processed/phase3_feature_correlations.png', dpi=300, bbox_inches='tight')
plt.savefig('data/processed/phase3_feature_correlations.pdf', dpi=300, bbox_inches='tight')
plt.show()

# Step 6: 特征变换和标准化
print("\n6. 特征变换和标准化")

# 6.1 处理缺失值
comprehensive_features_clean = comprehensive_features_df.copy()
for col in feature_cols:
    if comprehensive_features_clean[col].isnull().sum() > 0:
        if comprehensive_features_clean[col].dtype in ['int64', 'float64']:
            comprehensive_features_clean[col].fillna(comprehensive_features_clean[col].median(), inplace=True)
        else:
            comprehensive_features_clean[col].fillna(comprehensive_features_clean[col].mode()[0], inplace=True)

# 6.2 高变异特征处理
print("6.2 高变异特征处理:")
high_variance_features = []
for col in feature_cols:
    if comprehensive_features_clean[col].dtype in ['int64', 'float64']:
        col_range = comprehensive_features_clean[col].max() - comprehensive_features_clean[col].min()
        if col_range > 20:  # 阈值可调
            high_variance_features.append(col)
            print(f"{col}: 范围 {col_range:.2f}")

print(f"高变异特征数量: {len(high_variance_features)}")

# 6.3 特征标准化
scaler = RobustScaler()  # 使用RobustScaler对异常值更稳健
feature_data = comprehensive_features_clean[feature_cols]
feature_data_scaled = pd.DataFrame(
    scaler.fit_transform(feature_data),
    columns=feature_cols,
    index=feature_data.index
)

# 添加标识列
comprehensive_features_final = pd.concat([
    comprehensive_features_clean[['patient_id', 'sample_id', 'target_liver_stiffness']],
    feature_data_scaled
], axis=1)

print(f"最终特征集形状: {comprehensive_features_final.shape}")

# Step 7: 保存特征工程结果
print("\n7. 保存特征工程结果")

# 确保目录存在
import os
os.makedirs('data/processed', exist_ok=True)

# 保存数据
comprehensive_features_df.to_csv('data/processed/comprehensive_features_raw_new.csv', index=False)
comprehensive_features_clean.to_csv('data/processed/comprehensive_features_clean_new.csv', index=False)
comprehensive_features_final.to_csv('data/processed/comprehensive_features_final_new.csv', index=False)

# 保存特征信息
feature_info = pd.DataFrame({
    'feature_name': feature_cols,
    'correlation_with_target': [correlations.get(col, 0) for col in feature_cols],
    'missing_count': [comprehensive_features_df[col].isnull().sum() for col in feature_cols],
    'data_type': [str(comprehensive_features_df[col].dtype) for col in feature_cols]
})
feature_info.to_csv('data/processed/feature_information_new.csv', index=False)

# 保存标准化器
joblib.dump(scaler, 'data/processed/phase3_feature_scaler.pkl')

print("保存文件:")
print("- data/processed/comprehensive_features_raw_new.csv: 原始特征数据")
print("- data/processed/comprehensive_features_clean_new.csv: 清洗后特征数据")
print("- data/processed/comprehensive_features_final_new.csv: 最终标准化特征数据")
print("- data/processed/feature_information_new.csv: 特征信息表")
print("- data/processed/phase3_feature_scaler.pkl: 特征标准化器")

# Step 8: 摘要报告
print("\n========== Phase 3 特征工程摘要报告 (2984样本版) ==========")

print(f"\n特征集统计:")
print(f"- 总样本数: {len(comprehensive_features_final)}")
print(f"- 总特征数: {len(feature_cols)}")
print(f"- 数值型特征数: {len([col for col in feature_cols if comprehensive_features_clean[col].dtype in ['int64', 'float64']])}")

print(f"\n数据质量:")
print(f"- 缺失值处理: 完成")
print(f"- 异常值处理: {len(high_variance_features)} 个高变异特征")
print(f"- 特征标准化: 完成")

# 预测性评估
high_corr_features = len([corr for corr in correlations.values() if abs(corr) > 0.3])
medium_corr_features = len([corr for corr in correlations.values() if 0.1 < abs(corr) <= 0.3])

print(f"\n预测性评估:")
print(f"- 高相关性特征 (|r|>0.3): {high_corr_features}")
print(f"- 中等相关性特征 (0.1<|r|≤0.3): {medium_corr_features}")
if len(correlation_df) > 0:
    print(f"- 最强预测特征: {correlation_df.iloc[0]['Feature']} (r={correlation_df.iloc[0]['Correlation']:.3f})")

print(f"\n完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("Phase 3 时间序列特征工程完成！")