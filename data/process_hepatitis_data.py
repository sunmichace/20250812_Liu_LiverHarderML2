#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重新处理慢性乙型肝炎数据：纵向合并、英文表头、前缀序号
保留所有行，不进行任何过滤
"""

import pandas as pd
import numpy as np
from datetime import datetime

def create_column_mapping():
    """创建中英文表头映射"""
    column_mapping = {
        # 基础信息字段
        '序号': 'patient_id',
        '姓名': 'patient_name', 
        '患': 'patient_name',  # 验证集1的字段名
        '检查日期': 'examination_date',
        '性别': 'gender',
        '患者自然信息_性别': 'gender',  # 验证集1的字段名
        '年龄': 'age',
        '患者自然信息_年龄': 'age',  # 验证集1的字段名
        
        # 目标变量和检测结果
        '肝脏弹性纤维化水平': 'liver_stiffness',
        '脂肪含量': 'fat_content',
        
        # 肝功能指标
        '谷丙转氨酶': 'ALT',  # Alanine Aminotransferase
        '谷草转氨酶': 'AST',  # Aspartate Aminotransferase
        '总胆红素': 'total_bilirubin',
        
        # 乙肝病毒标志物
        '乙肝表面抗原': 'HBsAg',  # Hepatitis B surface Antigen
        '乙肝表面抗体': 'HBsAb',  # Hepatitis B surface Antibody
        '乙肝e抗原': 'HBeAg',    # Hepatitis B e Antigen
        '乙肝e抗体': 'HBeAb',    # Hepatitis B e Antibody
        '乙肝核心抗体IgG': 'HBcAb_IgG',  # Hepatitis B core Antibody IgG
        '乙型肝炎DNA测定': 'HBV_DNA'  # Hepatitis B Virus DNA
    }
    
    return column_mapping

def load_and_standardize_data():
    """加载并标准化原始数据"""
    print("=== 加载原始数据 ===")
    
    # 加载三个数据文件
    data_path = 'data/ori'
    train_data = pd.read_excel(f'{data_path}/训练集.xlsx')
    val1_data = pd.read_excel(f'{data_path}/验证集1.xlsx') 
    val2_data = pd.read_excel(f'{data_path}/验证集2.xlsx')
    
    print(f"训练集原始: {train_data.shape}")
    print(f"验证集1原始: {val1_data.shape}")
    print(f"验证集2原始: {val2_data.shape}")
    
    # 获取列映射
    column_mapping = create_column_mapping()
    
    # 标准化验证集1的列名（处理特殊命名）
    val1_data = val1_data.rename(columns={
        '患': '姓名',
        '患者自然信息_年龄': '年龄',
        '患者自然信息_性别': '性别'
    })
    
    # 确保所有数据集都有相同的列
    standard_columns = [
        '序号', '姓名', '检查日期', '性别', '年龄', '肝脏弹性纤维化水平',
        '脂肪含量', '谷丙转氨酶', '谷草转氨酶', '总胆红素', 
        '乙肝表面抗原', '乙肝表面抗体', '乙肝e抗原', '乙肝e抗体', 
        '乙肝核心抗体IgG', '乙型肝炎DNA测定'
    ]
    
    # 重新排列列顺序
    train_data = train_data[standard_columns]
    val1_data = val1_data[standard_columns] 
    val2_data = val2_data[standard_columns]
    
    print("列名标准化完成")
    
    return train_data, val1_data, val2_data, column_mapping

def create_prefixed_patient_ids(data, dataset_prefix):
    """创建带前缀的患者ID"""
    print(f"为{dataset_prefix}数据集创建前缀序号...")
    
    # 处理序号字段，确保为数值类型
    data['序号'] = pd.to_numeric(data['序号'], errors='coerce')
    
    # 创建带前缀的patient_id
    def create_prefixed_id(seq_num):
        if pd.notna(seq_num):
            return f"{dataset_prefix}_{int(seq_num):06d}"
        else:
            return f"{dataset_prefix}_UNKNOWN"
    
    data['prefixed_id'] = data['序号'].apply(create_prefixed_id)
    
    return data

def merge_longitudinal_data(train_data, val1_data, val2_data):
    """纵向合并所有数据"""
    print("\n=== 纵向合并数据 ===")
    
    # 为每个数据集添加前缀
    train_data = create_prefixed_patient_ids(train_data, 'TR')
    val1_data = create_prefixed_patient_ids(val1_data, 'V1') 
    val2_data = create_prefixed_patient_ids(val2_data, 'V2')
    
    # 添加数据源标识
    train_data['data_source'] = 'Training'
    val1_data['data_source'] = 'Validation1'
    val2_data['data_source'] = 'Validation2'
    
    # 纵向合并所有数据
    merged_data = pd.concat([train_data, val1_data, val2_data], 
                           ignore_index=True, sort=False)
    
    print(f"合并后总记录数: {len(merged_data)}")
    print(f"来源分布:")
    print(merged_data['data_source'].value_counts())
    
    return merged_data

def translate_to_english(merged_data, column_mapping):
    """将表头翻译为英文"""
    print("\n=== 翻译表头为英文 ===")
    
    # 更新序号字段为前缀ID
    merged_data['序号'] = merged_data['prefixed_id']
    
    # 应用列名映射
    merged_data = merged_data.rename(columns=column_mapping)
    
    # 删除临时字段
    if 'prefixed_id' in merged_data.columns:
        merged_data = merged_data.drop('prefixed_id', axis=1)
    
    # 重新排列列顺序（英文列名）
    english_columns = [
        'patient_id', 'patient_name', 'examination_date', 'gender', 'age', 
        'liver_stiffness', 'fat_content', 'ALT', 'AST', 'total_bilirubin',
        'HBsAg', 'HBsAb', 'HBeAg', 'HBeAb', 'HBcAb_IgG', 'HBV_DNA',
        'data_source'
    ]
    
    merged_data = merged_data[english_columns]
    
    print("表头翻译完成")
    print("最终列名:", list(merged_data.columns))
    
    return merged_data

def generate_summary_report(merged_data):
    """生成数据摘要报告"""
    
    report = f"""# Longitudinal Hepatitis B Liver Stiffness Dataset

## Data Overview
- **Total Records**: {len(merged_data):,}
- **Processing Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Data Structure**: Longitudinal (multiple visits per patient)

## Data Source Distribution
"""
    
    source_dist = merged_data['data_source'].value_counts()
    for source, count in source_dist.items():
        pct = count / len(merged_data) * 100
        report += f"- **{source}**: {count:,} records ({pct:.1f}%)\n"
    
    report += f"""

## Patient ID Format
- **Training Set**: TR_XXXXXX (e.g., TR_000001, TR_000002)
- **Validation Set 1**: V1_XXXXXX (e.g., V1_000004, V1_000005) 
- **Validation Set 2**: V2_XXXXXX (e.g., V2_000002, V2_000003)

## Data Fields (English Headers)
| Field | Description | Type |
|-------|-------------|------|
| patient_id | Patient identifier with dataset prefix | String |
| patient_name | Patient name | String |
| examination_date | Date of examination | DateTime |
| gender | Patient gender | String |
| age | Patient age at examination | Numeric |
| liver_stiffness | Liver fibrosis stiffness level (kPa) | Numeric |
| fat_content | Liver fat content | Numeric |
| ALT | Alanine Aminotransferase | Numeric |
| AST | Aspartate Aminotransferase | Numeric |
| total_bilirubin | Total bilirubin level | Numeric |
| HBsAg | Hepatitis B surface Antigen | Numeric |
| HBsAb | Hepatitis B surface Antibody | Numeric |
| HBeAg | Hepatitis B e Antigen | Numeric |
| HBeAb | Hepatitis B e Antibody | Numeric |
| HBcAb_IgG | Hepatitis B core Antibody IgG | Numeric |
| HBV_DNA | Hepatitis B Virus DNA level | Numeric |
| data_source | Source dataset identifier | String |

## Data Quality Summary
- **Complete Dataset**: All original rows preserved
- **No Filtering**: All records included regardless of missing values
- **Unique Patient IDs**: Each record has traceable patient identifier
- **Longitudinal Structure**: Multiple time points per patient maintained

## Missing Value Distribution
"""
    
    missing_stats = merged_data.isnull().sum().sort_values(ascending=False)
    for field, missing_count in missing_stats.items():
        if missing_count > 0:
            pct = missing_count / len(merged_data) * 100
            report += f"- **{field}**: {missing_count:,} missing ({pct:.1f}%)\n"
    
    report += "\n## Target Variable (liver_stiffness) Statistics\n"
    
    if merged_data['liver_stiffness'].notna().sum() > 0:
        target_stats = merged_data['liver_stiffness'].describe()
        report += f"""- **Count**: {target_stats['count']:.0f} non-missing values
- **Mean**: {target_stats['mean']:.2f} kPa
- **Std**: {target_stats['std']:.2f} kPa
- **Range**: [{target_stats['min']:.1f}, {target_stats['max']:.1f}] kPa
- **Median**: {target_stats['50%']:.2f} kPa
"""
    
    report += """
## Usage Notes
- This is a longitudinal dataset with multiple observations per patient
- Patient IDs with same prefix and base number represent repeated measurements
- For cross-sectional analysis, consider using only one record per patient
- For longitudinal modeling, the time series structure should be preserved
"""
    
    return report

def main():
    """主函数"""
    print("开始重新处理肝炎数据...")
    print("=" * 50)
    
    try:
        # 1. 加载并标准化原始数据
        train_data, val1_data, val2_data, column_mapping = load_and_standardize_data()
        
        # 2. 纵向合并所有数据
        merged_data = merge_longitudinal_data(train_data, val1_data, val2_data)
        
        # 3. 翻译表头为英文
        final_data = translate_to_english(merged_data, column_mapping)
        
        # 4. 保存最终数据
        output_file = 'data/hepatitis_b_longitudinal_data.xlsx'
        final_data.to_excel(output_file, index=False)
        print(f"\n最终数据已保存至: {output_file}")
        
        # 5. 生成摘要报告
        report = generate_summary_report(final_data)
        with open('hepatitis_b_dataset_summary.md', 'w', encoding='utf-8') as f:
            f.write(report)
        print("数据摘要报告已保存至: hepatitis_b_dataset_summary.md")
        
        # 6. 显示最终结果预览
        print("\n=== 最终数据预览 ===")
        print(f"数据形状: {final_data.shape}")
        print("\n前5行数据:")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(final_data.head())
        
        print("\n数据处理完成！")
        
        return final_data
        
    except Exception as e:
        print(f"数据处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = main()