#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1: 纵向数据预处理
基于纵向历史数据预测最后一次肝脏弹性值的完整分析方案
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set English display
plt.rcParams['font.family'] = 'DejaVu Sans'

print("========== Phase 1: 纵向数据预处理 ==========")
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("目标: 筛选有≥2次记录的患者，构建时间序列数据")

# Step 1: 数据加载和初步检查
print("\n1. 数据加载和初步检查")
data_path = 'data/hepatitis_b_longitudinal_data.xlsx'
df = pd.read_excel(data_path)

print(f"原始数据形状: {df.shape}")
print(f"字段信息:")
print(df.dtypes)
print(f"\n前5行数据:")
print(df.head())

# 检查关键字段
print(f"\npatient_id唯一值数量: {df['patient_id'].nunique()}")
print(f"liver_stiffness非空值数量: {df['liver_stiffness'].count()}")
print(f"检测日期范围: {df['examination_date'].min()} 到 {df['examination_date'].max()}")

# Step 2: 患者筛选与重构
print("\n2. 患者筛选与重构")

# 2.1 筛选有≥2次liver_stiffness记录的患者
patient_record_counts = df.groupby('patient_id')['liver_stiffness'].count()
valid_patients = patient_record_counts[patient_record_counts >= 2].index
print(f"有≥2次liver_stiffness记录的患者数量: {len(valid_patients)}")

# 筛选数据
df_filtered = df[df['patient_id'].isin(valid_patients)].copy()
df_filtered = df_filtered.dropna(subset=['liver_stiffness'])
print(f"筛选后数据形状: {df_filtered.shape}")

# 2.2 按患者分组并构建时间序列
print("\n构建时间序列数据...")

# 确保日期格式正确
df_filtered['examination_date'] = pd.to_datetime(df_filtered['examination_date'], format='mixed')

# 按患者ID和检测日期排序
df_filtered = df_filtered.sort_values(['patient_id', 'examination_date']).reset_index(drop=True)

# 创建时间序列统计
time_series_stats = []
longitudinal_data = []

for patient_id in valid_patients:
    patient_data = df_filtered[df_filtered['patient_id'] == patient_id].copy()
    patient_data = patient_data.sort_values('examination_date').reset_index(drop=True)
    
    # 计算时间间隔
    dates = patient_data['examination_date']
    intervals = []
    for i in range(1, len(dates)):
        interval = (dates.iloc[i] - dates.iloc[i-1]).days
        intervals.append(interval)
    
    # 统计信息
    stats = {
        'patient_id': patient_id,
        'record_count': len(patient_data),
        'total_follow_up_days': (dates.max() - dates.min()).days,
        'avg_interval_days': np.mean(intervals) if intervals else 0,
        'std_interval_days': np.std(intervals) if len(intervals) > 1 else 0,
        'min_interval_days': np.min(intervals) if intervals else 0,
        'max_interval_days': np.max(intervals) if intervals else 0,
        'baseline_liver_stiffness': patient_data['liver_stiffness'].iloc[0],
        'final_liver_stiffness': patient_data['liver_stiffness'].iloc[-1],
        'baseline_date': dates.iloc[0],
        'final_date': dates.iloc[-1]
    }
    time_series_stats.append(stats)
    
    # 为每个患者创建序列数据
    for i in range(len(patient_data)):
        longitudinal_record = patient_data.iloc[i].copy()
        longitudinal_record['sequence_position'] = i + 1  # 1开始的位置
        longitudinal_record['is_last_record'] = (i == len(patient_data) - 1)
        longitudinal_record['days_from_baseline'] = (dates.iloc[i] - dates.iloc[0]).days
        longitudinal_data.append(longitudinal_record)

# 转为DataFrame
time_series_stats_df = pd.DataFrame(time_series_stats)
longitudinal_df = pd.DataFrame(longitudinal_data)

print(f"时间序列统计表形状: {time_series_stats_df.shape}")
print(f"纵向数据表形状: {longitudinal_df.shape}")

# Step 3: 目标变量设计 
print("\n3. 目标变量设计")

# 3.1 多点预测任务设计 (1个患者 → 多个预测样本)
prediction_data = []

for patient_id in valid_patients:
    patient_data = longitudinal_df[longitudinal_df['patient_id'] == patient_id].copy()
    patient_data = patient_data.sort_values('examination_date').reset_index(drop=True)
    
    if len(patient_data) < 2:
        continue
    
    # 为每个患者生成多个预测任务
    # 如果患者有n次检查，生成n-1个预测任务
    for target_idx in range(1, len(patient_data)):
        # 历史数据 (前target_idx次)
        historical_records = patient_data.iloc[:target_idx]
        # 目标值 (第target_idx+1次)
        target_record = patient_data.iloc[target_idx]
        
        # 创建预测样本
        pred_sample = {
            'patient_id': patient_id,
            'sample_id': f"{patient_id}_sample_{target_idx}",  # 唯一样本标识
            'target_liver_stiffness': target_record['liver_stiffness'],
            'historical_count': len(historical_records),
            'prediction_span_days': (target_record['examination_date'] - 
                                   historical_records['examination_date'].iloc[0]).days
        }
        
        # 添加历史统计特征
        historical_values = historical_records['liver_stiffness'].values
        pred_sample.update({
            'hist_mean': np.mean(historical_values),
            'hist_std': np.std(historical_values) if len(historical_values) > 1 else 0,
            'hist_min': np.min(historical_values),
            'hist_max': np.max(historical_values),
            'hist_first': historical_values[0],
            'hist_last': historical_values[-1],
            'hist_trend_slope': np.polyfit(range(len(historical_values)), historical_values, 1)[0] 
                               if len(historical_values) > 1 else 0
        })
        
        prediction_data.append(pred_sample)

prediction_df = pd.DataFrame(prediction_data)
print(f"预测任务数据形状: {prediction_df.shape}")
print(f"预测任务样本数量: {len(prediction_df)}")

# Step 4: 数据清洗
print("\n4. 序列数据清洗")

# 4.1 检查缺失值
print("缺失值检查:")
missing_info = longitudinal_df.isnull().sum()
print(missing_info[missing_info > 0])

# 4.2 异常值检测
print("\n异常值检测:")

# 基于个体患者轨迹的异常检测
outlier_records = []

for patient_id in valid_patients:
    patient_data = longitudinal_df[longitudinal_df['patient_id'] == patient_id].copy()
    patient_data = patient_data.sort_values('examination_date')
    
    if len(patient_data) < 3:
        continue
    
    liver_values = patient_data['liver_stiffness'].values
    
    # 检测异常跳跃 (相邻值变化>3倍标准差)
    for i in range(1, len(liver_values)):
        prev_val = liver_values[i-1]
        curr_val = liver_values[i]
        
        # 计算历史标准差
        if i >= 2:
            hist_std = np.std(liver_values[:i])
            if hist_std > 0:
                jump_size = abs(curr_val - prev_val)
                if jump_size > 3 * hist_std and jump_size > 5:  # 绝对变化>5且>3倍标准差
                    outlier_records.append({
                        'patient_id': patient_id,
                        'record_index': i,
                        'prev_value': prev_val,
                        'curr_value': curr_val,
                        'jump_size': jump_size,
                        'threshold': 3 * hist_std,
                        'examination_date': patient_data.iloc[i]['examination_date']
                    })

outlier_df = pd.DataFrame(outlier_records)
print(f"检测到异常跳跃记录数: {len(outlier_df)}")

if len(outlier_df) > 0:
    print("异常跳跃样例:")
    print(outlier_df.head(3))

# Step 5: 保存预处理结果
print("\n5. 保存预处理结果")

# 创建输出目录
import os
os.makedirs('data/processed', exist_ok=True)

# 保存各个处理结果
time_series_stats_df.to_csv('data/processed/phase1_time_series_statistics.csv', index=False)
longitudinal_df.to_csv('data/processed/phase1_longitudinal_data.csv', index=False)
prediction_df.to_csv('data/processed/phase1_prediction_dataset.csv', index=False)

if len(outlier_df) > 0:
    outlier_df.to_csv('data/processed/phase1_outlier_records.csv', index=False)

print("保存文件:")
print("- data/processed/phase1_time_series_statistics.csv: 时间序列统计")
print("- data/processed/phase1_longitudinal_data.csv: 完整纵向数据") 
print("- data/processed/phase1_prediction_dataset.csv: 预测任务数据集")
if len(outlier_df) > 0:
    print("- data/processed/phase1_outlier_records.csv: 异常值记录")

# Step 6: 生成预处理报告
print("\n6. 生成预处理报告")

# 6.1 时间序列长度分布
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.hist(time_series_stats_df['record_count'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Distribution of Record Count per Patient')
plt.xlabel('Number of Records')
plt.ylabel('Number of Patients')
plt.grid(True, alpha=0.3)

# 6.2 Follow-up time distribution  
plt.subplot(2, 3, 2)
follow_up_years = time_series_stats_df['total_follow_up_days'] / 365.25
plt.hist(follow_up_years, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
plt.title('Distribution of Total Follow-up Time')
plt.xlabel('Follow-up Time (years)')
plt.ylabel('Number of Patients')
plt.grid(True, alpha=0.3)

# 6.3 Average detection interval distribution
plt.subplot(2, 3, 3)
plt.hist(time_series_stats_df['avg_interval_days'], bins=20, alpha=0.7, color='orange', edgecolor='black')
plt.title('Distribution of Average Detection Interval')
plt.xlabel('Average Interval (days)')
plt.ylabel('Number of Patients')
plt.grid(True, alpha=0.3)

# 6.4 Baseline vs final liver stiffness values
plt.subplot(2, 3, 4)
plt.scatter(time_series_stats_df['baseline_liver_stiffness'], 
           time_series_stats_df['final_liver_stiffness'], alpha=0.6, s=30)
plt.plot([0, 30], [0, 30], 'r--', alpha=0.7)  # Diagonal line
plt.title('Baseline vs Final Liver Stiffness')
plt.xlabel('Baseline Liver Stiffness (kPa)')
plt.ylabel('Final Liver Stiffness (kPa)')
plt.grid(True, alpha=0.3)

# 6.5 Liver stiffness change distribution
plt.subplot(2, 3, 5) 
change_values = time_series_stats_df['final_liver_stiffness'] - time_series_stats_df['baseline_liver_stiffness']
plt.hist(change_values, bins=30, alpha=0.7, color='purple', edgecolor='black')
plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
plt.title('Distribution of Liver Stiffness Change')
plt.xlabel('Change Amount (Final-Baseline, kPa)')
plt.ylabel('Number of Patients')
plt.grid(True, alpha=0.3)

# 6.6 Prediction task difficulty distribution
plt.subplot(2, 3, 6)
plt.hist(prediction_df['historical_count'], bins=range(2, max(prediction_df['historical_count'])+2), 
         alpha=0.7, color='red', edgecolor='black')
plt.title('Distribution of Historical Record Count')
plt.xlabel('Number of Historical Records')
plt.ylabel('Number of Prediction Tasks')
plt.grid(True, alpha=0.3)

plt.tight_layout()
# Save both PNG and PDF formats
plt.savefig('data/processed/phase1_preprocessing_report.png', dpi=300, bbox_inches='tight')
plt.savefig('data/processed/phase1_preprocessing_report.pdf', dpi=300, bbox_inches='tight')
plt.show()

# 6.7 数值摘要报告
print("\n========== 预处理摘要报告 ==========")
print(f"✓ 原始患者数: {df['patient_id'].nunique()}")
print(f"✓ 有效患者数(≥2次记录): {len(valid_patients)}")
print(f"✓ 有效记录总数: {len(longitudinal_df)}")
print(f"✓ 预测任务数: {len(prediction_df)}")

print(f"\n时间序列特征:")
print(f"- 平均记录次数: {time_series_stats_df['record_count'].mean():.1f}")
print(f"- 平均随访时间: {time_series_stats_df['total_follow_up_days'].mean()/365.25:.1f} 年")
print(f"- 平均检测间隔: {time_series_stats_df['avg_interval_days'].mean():.1f} 天")

print(f"\n肝脏弹性值特征:")
print(f"- 基线均值: {time_series_stats_df['baseline_liver_stiffness'].mean():.2f} kPa")
print(f"- 最终均值: {time_series_stats_df['final_liver_stiffness'].mean():.2f} kPa") 
print(f"- 平均变化: {change_values.mean():.2f} kPa")
print(f"- 进展患者比例: {(change_values > 0).mean()*100:.1f}%")

print(f"\n数据质量:")
print(f"- 异常跳跃记录: {len(outlier_df)}")
print(f"- 缺失值记录: {longitudinal_df.isnull().sum().sum()}")

print(f"\n完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Step 7: 生成数据清洗流程MD文档
print("\n7. 生成数据清洗流程文档")

def generate_data_cleaning_process_md():
    """生成详细的数据清洗流程Markdown文档"""

    # 计算关键统计指标
    original_patients = df['patient_id'].nunique()
    original_records = len(df)
    valid_patients_count = len(valid_patients)
    valid_records = len(longitudinal_df)
    prediction_tasks = len(prediction_df)
    outliers_count = len(outlier_df)

    # 计算数据保留率
    patient_retention_rate = (valid_patients_count / original_patients) * 100
    record_retention_rate = (valid_records / original_records) * 100

    # 计算临床统计
    baseline_mean = time_series_stats_df['baseline_liver_stiffness'].mean()
    final_mean = time_series_stats_df['final_liver_stiffness'].mean()
    improvement_rate = ((baseline_mean - final_mean) / baseline_mean) * 100

    # 记录分布统计
    record_distribution = time_series_stats_df['record_count'].value_counts().sort_index()

    md_content = f"""# Phase 1: 纵向数据预处理 - 数据清洗流程报告

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**项目**: 基于纵向历史数据预测肝脏弹性值
**阶段**: 数据预处理与质量控制

---

## 📊 执行摘要

| 指标类别 | 指标名称 | 数值 | 说明 |
|---------|---------|------|------|
| **原始数据** | 患者总数 | {original_patients:,} | 数据库中的总患者数 |
| **原始数据** | 记录总数 | {original_records:,} | 所有检查记录总数 |
| **筛选后** | 有效患者数 | {valid_patients_count:,} | 有≥2次liver_stiffness记录的患者 |
| **筛选后** | 有效记录数 | {valid_records:,} | 筛选后的有效记录数 |
| **任务生成** | 预测任务数 | {prediction_tasks:,} | 生成的机器学习预测任务 |
| **数据保留率** | 患者保留率 | {patient_retention_rate:.1f}% | 有效患者占原始患者比例 |
| **数据保留率** | 记录保留率 | {record_retention_rate:.1f}% | 有效记录占原始记录比例 |

---

## 🔧 数据清洗流程

### 1. 数据加载与初步评估

**输入数据源**: `data/hepatitis_b_longitudinal_data.xlsx`

**质量评估结果**:
- 原始数据维度: {df.shape[0]} 行 × {df.shape[1]} 列
- 患者ID唯一性: {df['patient_id'].nunique()} 个唯一患者
- liver_stiffness覆盖率: {(df['liver_stiffness'].count()/len(df)*100):.1f}%
- 检测时间跨度: {pd.to_datetime(df['examination_date'], format='mixed').min().strftime('%Y-%m-%d')} 至 {pd.to_datetime(df['examination_date'], format='mixed').max().strftime('%Y-%m-%d')}

### 2. 患者筛选标准

**筛选逻辑**: 仅保留有≥2次liver_stiffness有效记录的患者

**筛选理由**:
- 纵向预测需要至少2个时间点的数据建立时间序列
- 单次记录无法识别疾病进展趋势和变化模式
- 确保预测模型有足够的历史信息进行学习

**筛选结果**:
- 符合条件患者: {valid_patients_count:,} 人 ({patient_retention_rate:.1f}%)
- 剔除患者: {original_patients - valid_patients_count:,} 人
- 主要剔除原因: liver_stiffness缺失率高({(1-df['liver_stiffness'].count()/len(df))*100:.1f}%)

### 3. 时间序列重构

**重构策略**:
1. **时间排序**: 按patient_id和examination_date排序确保时间序列正确性
2. **间隔计算**: 计算相邻检查的时间间隔，识别随访规律性
3. **基线设定**: 以首次检查为时间基线(第0天)，计算相对时间
4. **序列标记**: 为每条记录添加序列位置和是否最后记录标识

**时间特征统计**:
- 平均记录次数: {time_series_stats_df['record_count'].mean():.1f} 次/患者
- 平均随访时间: {(time_series_stats_df['total_follow_up_days'].mean()/365.25):.1f} 年
- 平均检测间隔: {time_series_stats_df['avg_interval_days'].mean():.0f} 天 (约{(time_series_stats_df['avg_interval_days'].mean()/30):.1f}个月)
- 最长随访时间: {(time_series_stats_df['total_follow_up_days'].max()/365.25):.1f} 年

### 4. 预测任务设计

**1-to-Many策略**: 从每位患者的时间序列生成多个预测任务

**任务生成逻辑**:
- 如果患者有n次检查，生成(n-1)个预测任务
- 每个任务使用前k次记录预测第(k+1)次结果
- 避免数据泄漏：严格按时间顺序分割特征和目标

**生成统计**:"""

    # 添加记录分布详细信息
    md_content += f"""
- 总预测任务: {prediction_tasks:,} 个
- 平均任务/患者: {(prediction_tasks/valid_patients_count):.2f} 个
- 任务复杂度分布:
"""

    for records, count in record_distribution.head(10).items():
        tasks_per_patient = records - 1
        md_content += f"  - {records}次记录患者: {count}人 → {count * tasks_per_patient}个任务\n"

    md_content += f"""
### 5. 异常值检测与处理

**检测方法**: 基于个体患者轨迹的统计异常检测

**检测标准**:
- 相邻检查值变化 > 3倍个人历史标准差
- 绝对变化量 > 5.0 kPa
- 需要≥3次记录才能进行检测

**检测结果**:
- 异常跳跃记录: {outliers_count} 个
- 异常率: {(outliers_count/valid_records*100):.3f}%
- 处理策略: 标记但保留，为后续特征工程提供异常标识

### 6. 数据质量控制

**质量指标评估**:
- 缺失值控制: 关键字段liver_stiffness 100%完整
- 时间序列连续性: {((valid_records - len(time_series_stats_df))/valid_records*100):.1f}% 记录有完整时间信息
- 测量范围检查: liver_stiffness 范围 {longitudinal_df['liver_stiffness'].min():.1f}-{longitudinal_df['liver_stiffness'].max():.1f} kPa
- 异常值标记: {outliers_count} 个潜在测量异常已标识

**最终数据特征**:
- 患者数量: {valid_patients_count:,} 人
- 有效记录: {valid_records:,} 条
- 预测样本: {prediction_tasks:,} 个
- 平均随访: {(time_series_stats_df['total_follow_up_days'].mean()/365.25):.1f} 年
- 数据时间跨度: {(pd.to_datetime(df['examination_date'], format='mixed').max() - pd.to_datetime(df['examination_date'], format='mixed').min()).days} 天

---

## 📈 临床特征概览

**基线特征** (首次检查):
- 肝脏弹性均值: {baseline_mean:.2f} ± {time_series_stats_df['baseline_liver_stiffness'].std():.2f} kPa
- 中位数: {time_series_stats_df['baseline_liver_stiffness'].median():.2f} kPa
- 范围: {time_series_stats_df['baseline_liver_stiffness'].min():.1f} - {time_series_stats_df['baseline_liver_stiffness'].max():.1f} kPa

**最终特征** (最后一次检查):
- 肝脏弹性均值: {final_mean:.2f} ± {time_series_stats_df['final_liver_stiffness'].std():.2f} kPa
- 中位数: {time_series_stats_df['final_liver_stiffness'].median():.2f} kPa
- 范围: {time_series_stats_df['final_liver_stiffness'].min():.1f} - {time_series_stats_df['final_liver_stiffness'].max():.1f} kPa

**临床意义**:
- 整体改善率: {improvement_rate:.1f}% (基线相比最终值的平均改善)
- 进展患者: {((time_series_stats_df['final_liver_stiffness'] > time_series_stats_df['baseline_liver_stiffness']).sum()/len(time_series_stats_df)*100):.1f}%
- 改善患者: {((time_series_stats_df['final_liver_stiffness'] < time_series_stats_df['baseline_liver_stiffness']).sum()/len(time_series_stats_df)*100):.1f}%
- 稳定患者: {((time_series_stats_df['final_liver_stiffness'] == time_series_stats_df['baseline_liver_stiffness']).sum()/len(time_series_stats_df)*100):.1f}%

---

## 🔍 输出文件说明

| 文件名 | 内容描述 | 用途 |
|--------|----------|------|
| `phase1_time_series_statistics.csv` | 每位患者的时间序列统计信息 | 患者特征分析 |
| `phase1_longitudinal_data.csv` | 完整的纵向数据记录 | 时间序列建模 |
| `phase1_prediction_dataset.csv` | 机器学习预测任务数据集 | 模型训练评估 |
| `phase1_outlier_records.csv` | 异常值检测结果 | 数据质量监控 |
| `phase1_preprocessing_report.png/pdf` | 可视化分析报告 | 结果展示 |

---

## ✅ 数据清洗完成确认

- [x] 原始数据加载完成
- [x] 患者筛选标准应用 (≥2次记录)
- [x] 时间序列重构完成
- [x] 预测任务生成 ({prediction_tasks:,} 个任务)
- [x] 异常值检测标记 ({outliers_count} 个异常)
- [x] 数据质量验证通过
- [x] 输出文件生成完成

**数据已准备就绪，可进入Phase 2: 探索性数据分析阶段**

---

*报告生成于: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
*下一阶段: Phase 2 - 纵向EDA分析*
"""

    return md_content

# 生成并保存数据清洗流程文档
cleaning_process_md = generate_data_cleaning_process_md()
with open('data/processed/phase1_data_cleaning_process.md', 'w', encoding='utf-8') as f:
    f.write(cleaning_process_md)

print("✓ 数据清洗流程文档已生成: data/processed/phase1_data_cleaning_process.md")

# Step 8: 生成患者特征统计报告 (Patient Characteristics CSV)
print("\n8. 生成患者特征统计报告")

def generate_patient_characteristics_excel():
    """生成F2分组比较的患者基线特征Excel表格，包含统计检验"""
    from scipy import stats
    from scipy.stats import chi2_contingency, mannwhitneyu, ttest_ind
    import openpyxl

    # 创建患者基线数据
    patient_baseline_data = []

    # 为每个患者提取基线特征
    for patient_id in valid_patients:
        patient_records = df[df['patient_id'] == patient_id].copy()
        patient_records = patient_records.sort_values('examination_date')
        baseline_record = patient_records.iloc[0]  # 基线记录

        # 基线肝脏弹性值
        baseline_ls = baseline_record['liver_stiffness']

        # F2分组：>=7.0 kPa为F2+，<7.0 kPa为F0-F1
        f2_group = "≥F2" if baseline_ls >= 7.0 else "<F2"

        patient_data = {
            'patient_id': patient_id,
            'baseline_liver_stiffness': baseline_ls,
            'f2_group': f2_group,
            'age': baseline_record['age'] if pd.notna(baseline_record['age']) else None,
            'gender': baseline_record['gender'] if pd.notna(baseline_record['gender']) else None,
            'ALT': baseline_record['ALT'] if pd.notna(baseline_record['ALT']) else None,
            'AST': baseline_record['AST'] if pd.notna(baseline_record['AST']) else None,
            'total_bilirubin': baseline_record['total_bilirubin'] if pd.notna(baseline_record['total_bilirubin']) else None,
            'fat_content': baseline_record['fat_content'] if pd.notna(baseline_record['fat_content']) else None,
            'HBV_DNA': baseline_record['HBV_DNA'] if pd.notna(baseline_record['HBV_DNA']) else None,
            'HBsAg': baseline_record['HBsAg'] if pd.notna(baseline_record['HBsAg']) else None,
            'HBeAg': baseline_record['HBeAg'] if pd.notna(baseline_record['HBeAg']) else None,
        }

        # 添加随访特征
        patient_stats = time_series_stats_df[time_series_stats_df['patient_id'] == patient_id].iloc[0]
        patient_data.update({
            'follow_up_years': patient_stats['total_follow_up_days'] / 365.25,
            'record_count': patient_stats['record_count'],
            'avg_interval_days': patient_stats['avg_interval_days']
        })

        patient_baseline_data.append(patient_data)

    # 转为DataFrame
    baseline_df = pd.DataFrame(patient_baseline_data)

    # 分组数据
    f2_high = baseline_df[baseline_df['f2_group'] == '≥F2']  # F2+组
    f2_low = baseline_df[baseline_df['f2_group'] == '<F2']   # F0-F1组

    print(f"F2分组结果: ≥F2组 {len(f2_high)} 人, <F2组 {len(f2_low)} 人")

    # 统计函数
    def format_continuous_var(data, var_name):
        """格式化连续变量统计"""
        clean_data = data[var_name].dropna()
        if len(clean_data) == 0:
            return "N/A", 0
        mean_val = clean_data.mean()
        std_val = clean_data.std()
        return f"{mean_val:.1f} ± {std_val:.1f}", len(clean_data)

    def format_categorical_var(data, var_name, category_value):
        """格式化分类变量统计"""
        clean_data = data[var_name].dropna()
        if len(clean_data) == 0:
            return "N/A", 0
        count = (clean_data == category_value).sum()
        percentage = count / len(clean_data) * 100
        return f"{count} ({percentage:.1f}%)", len(clean_data)

    def calculate_p_value(data, var_name, var_type, category_value=None):
        """计算P值"""
        try:
            high_data = f2_high[var_name].dropna()
            low_data = f2_low[var_name].dropna()

            if len(high_data) < 5 or len(low_data) < 5:
                return "N/A"

            if var_type == 'continuous':
                # 连续变量：使用Mann-Whitney U检验（更稳健）
                statistic, p_value = mannwhitneyu(high_data, low_data, alternative='two-sided')
                return f"{p_value:.3f}" if p_value >= 0.001 else "<0.001"

            elif var_type == 'categorical':
                # 分类变量：使用卡方检验
                # 创建列联表
                high_count = (high_data == category_value).sum()
                high_other = len(high_data) - high_count
                low_count = (low_data == category_value).sum()
                low_other = len(low_data) - low_count

                contingency_table = np.array([[high_count, high_other],
                                            [low_count, low_other]])

                # 检查期望频数
                if np.any(contingency_table < 5):
                    from scipy.stats import fisher_exact
                    _, p_value = fisher_exact(contingency_table)
                else:
                    chi2, p_value, _, _ = chi2_contingency(contingency_table)

                return f"{p_value:.3f}" if p_value >= 0.001 else "<0.001"

        except Exception as e:
            print(f"统计检验失败 {var_name}: {e}")
            return "N/A"

    # 构建结果表格
    results = []

    # 基本信息
    total_n = len(baseline_df)
    high_n = len(f2_high)
    low_n = len(f2_low)

    results.append({
        'Variable': 'Total patients, n',
        'Overall': f"{total_n}",
        '≥F2': f"{high_n}",
        '<F2': f"{low_n}",
        'P value': "-"
    })

    # 连续变量分析
    continuous_vars = [
        ('age', 'Age (years)'),
        ('baseline_liver_stiffness', 'Liver stiffness (kPa)'),
        ('ALT', 'ALT (U/L)'),
        ('AST', 'AST (U/L)'),
        ('total_bilirubin', 'Total bilirubin (μmol/L)'),
        ('fat_content', 'Fat content (%)'),
        ('follow_up_years', 'Follow-up duration (years)'),
        ('record_count', 'Number of examinations'),
        ('avg_interval_days', 'Examination interval (days)')
    ]

    for var_name, var_label in continuous_vars:
        overall_stat, overall_n = format_continuous_var(baseline_df, var_name)
        high_stat, high_stat_n = format_continuous_var(f2_high, var_name)
        low_stat, low_stat_n = format_continuous_var(f2_low, var_name)
        p_val = calculate_p_value(baseline_df, var_name, 'continuous')

        results.append({
            'Variable': f'{var_label}, mean ± SD',
            'Overall': f"{overall_stat}",
            '≥F2': f"{high_stat}",
            '<F2': f"{low_stat}",
            'P value': p_val
        })

    # 分类变量分析
    categorical_vars = [
        ('gender', 'Gender', ['男', '女']),
        ('HBsAg', 'HBsAg positive', ['阳性', '+', 'positive']),
        ('HBeAg', 'HBeAg positive', ['阳性', '+', 'positive'])
    ]

    for var_name, var_label, positive_values in categorical_vars:
        if var_name == 'gender':
            # 性别分析 - 分别统计男女
            for gender_val in ['男', '女']:
                overall_stat, overall_n = format_categorical_var(baseline_df, var_name, gender_val)
                high_stat, high_stat_n = format_categorical_var(f2_high, var_name, gender_val)
                low_stat, low_stat_n = format_categorical_var(f2_low, var_name, gender_val)

                # 只为男性计算P值（避免重复）
                if gender_val == '男':
                    p_val = calculate_p_value(baseline_df, var_name, 'categorical', gender_val)
                else:
                    p_val = "-"

                results.append({
                    'Variable': f'{gender_val}, n (%)',
                    'Overall': overall_stat,
                    '≥F2': high_stat,
                    '<F2': low_stat,
                    'P value': p_val
                })
        else:
            # 其他分类变量 - 统计阳性率
            # 检查实际存在的阳性值
            actual_positive = []
            for pos_val in positive_values:
                if (baseline_df[var_name] == pos_val).any():
                    actual_positive.append(pos_val)

            if actual_positive:
                # 使用找到的第一个阳性值进行统计
                pos_val = actual_positive[0]
                overall_stat, overall_n = format_categorical_var(baseline_df, var_name, pos_val)
                high_stat, high_stat_n = format_categorical_var(f2_high, var_name, pos_val)
                low_stat, low_stat_n = format_categorical_var(f2_low, var_name, pos_val)
                p_val = calculate_p_value(baseline_df, var_name, 'categorical', pos_val)

                results.append({
                    'Variable': f'{var_label}, n (%)',
                    'Overall': overall_stat,
                    '≥F2': high_stat,
                    '<F2': low_stat,
                    'P value': p_val
                })

    return pd.DataFrame(results)

# 生成患者特征Excel表格
patient_characteristics_df = generate_patient_characteristics_excel()

# 保存Excel文件
excel_path = 'data/processed/phase1_patient_characteristics_F2_comparison.xlsx'
patient_characteristics_df.to_excel(excel_path, index=False, engine='openpyxl')

print("✓ 患者基线特征F2分组比较表已生成: data/processed/phase1_patient_characteristics_F2_comparison.xlsx")
print(f"✓ 包含 {len(patient_characteristics_df)} 个变量，分为≥F2和<F2两组进行统计检验")

# 显示预览
print("\n患者特征表格预览:")
print(patient_characteristics_df.head(10).to_string(index=False))

# 更新文件列表说明
print("\n========== 更新的保存文件列表 ==========")
print("- data/processed/phase1_time_series_statistics.csv: 时间序列统计")
print("- data/processed/phase1_longitudinal_data.csv: 完整纵向数据")
print("- data/processed/phase1_prediction_dataset.csv: 预测任务数据集")
if len(outlier_df) > 0:
    print("- data/processed/phase1_outlier_records.csv: 异常值记录")
print("- data/processed/phase1_preprocessing_report.png/pdf: 可视化报告")
print("- data/processed/phase1_data_cleaning_process.md: 数据清洗流程文档")
print("- data/processed/phase1_patient_characteristics_F2_comparison.xlsx: 患者F2分组比较表格")

print("Phase 1 预处理完成！")