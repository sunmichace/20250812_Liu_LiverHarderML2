#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型优化主执行脚本：协调所有优化方法
基于纵向历史数据预测肝脏弹性值的综合优化方案
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import subprocess
import sys
import os
import joblib

# Set English display
plt.rcParams['font.family'] = 'DejaVu Sans'

print("========== 综合模型优化：提升R²性能 ==========")
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("目标: 通过多种优化方法综合提升GradientBoosting模型的R²性能")

# Step 1: 获取基准性能
print("\n1. 获取当前模型基准性能")

try:
    # 加载当前最佳模型
    current_model = joblib.load('data/processed/models/best_model_gradientboosting.pkl')
    features_df = pd.read_csv('data/processed/comprehensive_features_final_new.csv')
    
    feature_cols = [col for col in features_df.columns if col not in ['patient_id', 'sample_id', 'target_liver_stiffness']]
    X = features_df[feature_cols]
    y = features_df['target_liver_stiffness']
    
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, r2_score
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    baseline_pred = current_model.predict(X_test)
    baseline_mae = mean_absolute_error(y_test, baseline_pred)
    baseline_r2 = r2_score(y_test, baseline_pred)
    
    print(f"当前基准性能:")
    print(f"- MAE: {baseline_mae:.3f} kPa")
    print(f"- R²: {baseline_r2:.3f}")
    print(f"- 特征数: {len(feature_cols)}")
    
except Exception as e:
    print(f"无法加载基准模型: {e}")
    baseline_mae = 2.457  # 从README中获取的值
    baseline_r2 = 0.321
    print(f"使用默认基准值: MAE={baseline_mae:.3f}, R²={baseline_r2:.3f}")

# Step 2: 执行优化方法
print("\n2. 执行各种优化方法")

optimization_results = {}

# 2.1 超参数调优
print("\n2.1 执行超参数调优...")
try:
    result = subprocess.run([sys.executable, 'optimization/hyperparameter_tuning.py'], 
                          capture_output=True, text=True, timeout=1800)  # 30分钟超时
    
    if result.returncode == 0:
        print("✅ 超参数调优完成")
        # 尝试读取结果
        try:
            hp_results = pd.read_csv('optimization/results/hyperparameter_tuning_results.csv')
            best_hp_r2 = hp_results['R²'].max()
            optimization_results['Hyperparameter_Tuning'] = {
                'R²': best_hp_r2,
                'MAE': hp_results.loc[hp_results['R²'].idxmax(), 'MAE'],
                'Status': 'Success'
            }
            print(f"   最佳R²: {best_hp_r2:.3f}")
        except:
            optimization_results['Hyperparameter_Tuning'] = {'Status': 'Completed_NoResults'}
    else:
        print(f"⚠️ 超参数调优失败: {result.stderr}")
        optimization_results['Hyperparameter_Tuning'] = {'Status': 'Failed'}

except subprocess.TimeoutExpired:
    print("⚠️ 超参数调优超时，跳过")
    optimization_results['Hyperparameter_Tuning'] = {'Status': 'Timeout'}
except Exception as e:
    print(f"⚠️ 超参数调优出错: {e}")
    optimization_results['Hyperparameter_Tuning'] = {'Status': 'Error'}

# 2.2 高级特征工程
print("\n2.2 执行高级特征工程...")
try:
    result = subprocess.run([sys.executable, 'optimization/advanced_feature_engineering.py'], 
                          capture_output=True, text=True, timeout=1800)
    
    if result.returncode == 0:
        print("✅ 高级特征工程完成")
        try:
            fe_results = pd.read_csv('optimization/results/feature_engineering_results.csv')
            best_fe_r2 = fe_results['R²'].max()
            optimization_results['Feature_Engineering'] = {
                'R²': best_fe_r2,
                'MAE': fe_results.loc[fe_results['R²'].idxmax(), 'MAE'],
                'Status': 'Success'
            }
            print(f"   最佳R²: {best_fe_r2:.3f}")
        except:
            optimization_results['Feature_Engineering'] = {'Status': 'Completed_NoResults'}
    else:
        print(f"⚠️ 高级特征工程失败: {result.stderr}")
        optimization_results['Feature_Engineering'] = {'Status': 'Failed'}

except subprocess.TimeoutExpired:
    print("⚠️ 高级特征工程超时，跳过")
    optimization_results['Feature_Engineering'] = {'Status': 'Timeout'}
except Exception as e:
    print(f"⚠️ 高级特征工程出错: {e}")
    optimization_results['Feature_Engineering'] = {'Status': 'Error'}

# 2.3 集成学习
print("\n2.3 执行集成学习...")
try:
    result = subprocess.run([sys.executable, 'optimization/ensemble_learning.py'], 
                          capture_output=True, text=True, timeout=2400)  # 40分钟超时
    
    if result.returncode == 0:
        print("✅ 集成学习完成")
        try:
            ensemble_results = pd.read_csv('optimization/results/ensemble_learning_results.csv')
            best_ensemble_r2 = ensemble_results['R²'].max()
            optimization_results['Ensemble_Learning'] = {
                'R²': best_ensemble_r2,
                'MAE': ensemble_results.loc[ensemble_results['R²'].idxmax(), 'MAE'],
                'Status': 'Success'
            }
            print(f"   最佳R²: {best_ensemble_r2:.3f}")
        except:
            optimization_results['Ensemble_Learning'] = {'Status': 'Completed_NoResults'}
    else:
        print(f"⚠️ 集成学习失败: {result.stderr}")
        optimization_results['Ensemble_Learning'] = {'Status': 'Failed'}

except subprocess.TimeoutExpired:
    print("⚠️ 集成学习超时，跳过")
    optimization_results['Ensemble_Learning'] = {'Status': 'Timeout'}
except Exception as e:
    print(f"⚠️ 集成学习出错: {e}")
    optimization_results['Ensemble_Learning'] = {'Status': 'Error'}

# Step 3: 汇总和对比结果
print("\n3. 汇总优化结果")

# 收集所有成功的结果
successful_results = []

# 基准结果
successful_results.append({
    'Method': 'Baseline_GradientBoosting',
    'R²': baseline_r2,
    'MAE': baseline_mae,
    'Improvement_R2(%)': 0.0,
    'Status': 'Baseline'
})

# 添加成功的优化结果
for method, result in optimization_results.items():
    if isinstance(result, dict) and 'R²' in result:
        improvement = ((result['R²'] - baseline_r2) / baseline_r2) * 100
        successful_results.append({
            'Method': method,
            'R²': result['R²'],
            'MAE': result['MAE'],
            'Improvement_R2(%)': improvement,
            'Status': result['Status']
        })

# 创建结果DataFrame
results_summary_df = pd.DataFrame(successful_results)
if len(results_summary_df) > 1:
    results_summary_df = results_summary_df.sort_values('R²', ascending=False)

print("\n优化方法结果汇总:")
if len(results_summary_df) > 0:
    print(results_summary_df.round(3))
else:
    print("没有成功的优化结果")

# Step 4: 寻找最佳组合方法（如果有多个成功的方法）
print("\n4. 寻找最佳组合策略")

if len(successful_results) > 2:  # 至少有1个成功的优化方法
    print("尝试组合多种优化方法的结果...")
    
    # 4.1 加载各种优化后的模型
    optimized_models = {}
    
    # 尝试加载超参数优化模型
    try:
        hp_model_files = [f for f in os.listdir('optimization/models') if 'optimized' in f.lower()]
        if hp_model_files:
            hp_model = joblib.load(f'optimization/models/{hp_model_files[0]}')
            optimized_models['Hyperparameter_Optimized'] = hp_model
    except:
        pass
    
    # 尝试加载特征工程模型
    try:
        fe_model = joblib.load('optimization/models/best_enhanced_features_model.pkl')
        optimized_models['Feature_Enhanced'] = fe_model
    except:
        pass
    
    # 尝试加载集成学习模型
    try:
        ensemble_files = [f for f in os.listdir('optimization/models') if 'ensemble' in f.lower()]
        if ensemble_files:
            ensemble_model = joblib.load(f'optimization/models/{ensemble_files[0]}')
            optimized_models['Ensemble'] = ensemble_model
    except:
        pass
    
    if optimized_models:
        print(f"找到 {len(optimized_models)} 个优化模型")
        
        # 4.2 简单的模型平均（如果可能）
        try:
            if len(optimized_models) >= 2:
                print("尝试模型平均...")
                
                # 使用相同的测试集
                model_predictions = {}
                
                for name, model in optimized_models.items():
                    try:
                        if name == 'Feature_Enhanced':
                            # 特征工程模型需要特殊处理
                            selectors = joblib.load('optimization/models/feature_selectors.pkl')
                            X_test_fe = selectors['variance_selector'].transform(X_test.fillna(0))
                            X_test_fe = selectors['correlation_selector'].transform(X_test_fe)
                            pred = model.predict(X_test_fe)
                        else:
                            pred = model.predict(X_test)
                        
                        model_predictions[name] = pred
                        
                        # 单独评估
                        mae_single = mean_absolute_error(y_test, pred)
                        r2_single = r2_score(y_test, pred)
                        print(f"   {name}: MAE={mae_single:.3f}, R²={r2_single:.3f}")
                        
                    except Exception as e:
                        print(f"   无法评估模型 {name}: {e}")
                
                # 平均预测
                if len(model_predictions) >= 2:
                    avg_pred = np.mean(list(model_predictions.values()), axis=0)
                    avg_mae = mean_absolute_error(y_test, avg_pred)
                    avg_r2 = r2_score(y_test, avg_pred)
                    
                    print(f"\n模型平均结果:")
                    print(f"   MAE: {avg_mae:.3f} kPa")
                    print(f"   R²: {avg_r2:.3f}")
                    
                    # 添加到结果中
                    avg_improvement = ((avg_r2 - baseline_r2) / baseline_r2) * 100
                    successful_results.append({
                        'Method': 'Model_Average',
                        'R²': avg_r2,
                        'MAE': avg_mae,
                        'Improvement_R2(%)': avg_improvement,
                        'Status': 'Combined'
                    })
        
        except Exception as e:
            print(f"模型组合失败: {e}")

# Step 5: 最终结果可视化
print("\n5. 生成最终优化结果可视化")

# 更新结果DataFrame
results_summary_df = pd.DataFrame(successful_results)
results_summary_df = results_summary_df.sort_values('R²', ascending=False)

# 创建综合可视化
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 5.1 R²对比
ax1 = axes[0, 0]
colors = ['red' if 'Baseline' in method else 'skyblue' if 'Average' in method else 'lightgreen' 
          for method in results_summary_df['Method']]
bars = ax1.bar(range(len(results_summary_df)), results_summary_df['R²'], color=colors, edgecolor='black')
ax1.set_title('Optimization Methods R² Comparison', fontsize=14, fontweight='bold')
ax1.set_ylabel('R² Score')
ax1.set_xticks(range(len(results_summary_df)))
ax1.set_xticklabels(results_summary_df['Method'], rotation=45, ha='right')

# 添加数值标签
for i, (bar, r2) in enumerate(zip(bars, results_summary_df['R²'])):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{r2:.3f}', ha='center', va='bottom', fontweight='bold')

# 5.2 改进百分比
ax2 = axes[0, 1]
improvement_data = results_summary_df[results_summary_df['Method'] != 'Baseline_GradientBoosting']
if len(improvement_data) > 0:
    bars = ax2.bar(range(len(improvement_data)), improvement_data['Improvement_R2(%)'], 
                   color='gold', edgecolor='black')
    ax2.set_title('R² Improvement Percentage', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Improvement (%)')
    ax2.set_xticks(range(len(improvement_data)))
    ax2.set_xticklabels(improvement_data['Method'], rotation=45, ha='right')
    
    # 添加数值标签
    for bar, imp in zip(bars, improvement_data['Improvement_R2(%)']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{imp:.1f}%', ha='center', va='bottom', fontweight='bold')

# 5.3 MAE对比
ax3 = axes[0, 2]
bars = ax3.bar(range(len(results_summary_df)), results_summary_df['MAE'], 
               color=colors, edgecolor='black')
ax3.set_title('MAE Comparison', fontsize=14, fontweight='bold')
ax3.set_ylabel('MAE (kPa)')
ax3.set_xticks(range(len(results_summary_df)))
ax3.set_xticklabels(results_summary_df['Method'], rotation=45, ha='right')

# 5.4 优化方法成功率
ax4 = axes[1, 0]
status_counts = pd.Series([result.get('Status', 'Unknown') for result in optimization_results.values()]).value_counts()
colors_pie = ['lightgreen' if 'Success' in status else 'lightcoral' if 'Failed' in status 
              else 'lightyellow' for status in status_counts.index]
ax4.pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%', 
        colors=colors_pie, startangle=90)
ax4.set_title('Optimization Methods Success Rate', fontsize=14, fontweight='bold')

# 5.5 最佳改进
ax5 = axes[1, 1]
if len(results_summary_df) > 1:
    best_method = results_summary_df.iloc[0]
    baseline_method = results_summary_df[results_summary_df['Method'].str.contains('Baseline')].iloc[0]
    
    ax5.bar(['Baseline', 'Best Method'], [baseline_method['R²'], best_method['R²']], 
            color=['lightcoral', 'lightgreen'], edgecolor='black')
    
    improvement_pct = best_method['Improvement_R2(%)']
    ax5.set_title(f'Best Improvement: +{improvement_pct:.1f}%', fontsize=14, fontweight='bold')
    ax5.set_ylabel('R² Score')
    
    # 添加改进箭头
    ax5.annotate('', xy=(1, best_method['R²']), xytext=(0, baseline_method['R²']),
                arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    ax5.text(0.5, (baseline_method['R²'] + best_method['R²'])/2, f'+{improvement_pct:.1f}%',
             ha='center', va='center', fontweight='bold', color='red')

# 5.6 总结信息
ax6 = axes[1, 2]
ax6.axis('off')

if len(results_summary_df) > 1:
    best_result = results_summary_df.iloc[0]
    summary_text = f"""
    优化总结:
    
    基准性能:
    • R² = {baseline_r2:.3f}
    • MAE = {baseline_mae:.3f} kPa
    
    最佳优化方法:
    • {best_result['Method'].replace('_', ' ')}
    • R² = {best_result['R²']:.3f}
    • MAE = {best_result['MAE']:.3f} kPa
    • 改进 = +{best_result['Improvement_R2(%)']:.1f}%
    
    成功优化方法数: {len(improvement_data)}
    """
else:
    summary_text = """
    优化总结:
    
    基准性能:
    • R² = {:.3f}
    • MAE = {:.3f} kPa
    
    优化结果:
    • 所有优化方法均未成功
    • 建议检查数据和参数设置
    """.format(baseline_r2, baseline_mae)

ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=11,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

plt.tight_layout()
# Save both PNG and PDF formats
plt.savefig('optimization/comprehensive_optimization_results.png', dpi=300, bbox_inches='tight')
plt.savefig('optimization/comprehensive_optimization_results.pdf', dpi=300, bbox_inches='tight')
plt.show()

# Step 6: 保存综合结果
print("\n6. 保存综合优化结果")

# 保存结果汇总
results_summary_df.to_csv('optimization/results/comprehensive_optimization_summary.csv', index=False)

# 保存最佳模型（如果有改进）
if len(results_summary_df) > 1:
    best_method = results_summary_df.iloc[0]['Method']
    
    if best_method != 'Baseline_GradientBoosting':
        print(f"最佳优化方法: {best_method}")
        
        # 尝试复制最佳模型
        try:
            if 'Hyperparameter' in best_method:
                source_files = [f for f in os.listdir('optimization/models') if 'optimized' in f.lower()]
            elif 'Feature' in best_method:
                source_files = ['best_enhanced_features_model.pkl']
            elif 'Ensemble' in best_method:
                source_files = [f for f in os.listdir('optimization/models') if 'ensemble' in f.lower()]
            elif 'Average' in best_method:
                source_files = []  # 模型平均不需要单独保存
            else:
                source_files = []
            
            if source_files:
                import shutil
                shutil.copy(f'optimization/models/{source_files[0]}', 
                           'optimization/models/final_best_optimized_model.pkl')
                print(f"✅ 最佳模型已保存为 final_best_optimized_model.pkl")
        
        except Exception as e:
            print(f"⚠️ 无法保存最佳模型: {e}")

# 生成最终报告
print("\n========== 综合优化最终报告 ==========")
print(f"优化任务完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

print(f"\n基准模型性能:")
print(f"- R²: {baseline_r2:.3f}")
print(f"- MAE: {baseline_mae:.3f} kPa")

if len(results_summary_df) > 1:
    best_result = results_summary_df.iloc[0]
    print(f"\n最佳优化结果:")
    print(f"- 方法: {best_result['Method'].replace('_', ' ')}")
    print(f"- R²: {best_result['R²']:.3f}")
    print(f"- MAE: {best_result['MAE']:.3f} kPa")
    print(f"- R²改进: +{best_result['Improvement_R2(%)']:.1f}%")
    
    # 性能分级
    final_r2 = best_result['R²']
    if final_r2 > 0.5:
        print("🎉 模型R²已达到优秀水平 (>0.5)")
    elif final_r2 > 0.4:
        print("🎉 模型R²已达到良好水平 (>0.4)")
    elif final_r2 > 0.35:
        print("✅ 模型R²达到可接受水平 (>0.35)")
    elif final_r2 > baseline_r2:
        print("📈 模型R²有所提升，继续优化空间")
    else:
        print("⚠️ 模型性能未显著改善，建议重新评估数据和方法")
else:
    print("\n⚠️ 所有优化方法均未成功")
    print("建议:")
    print("- 检查数据质量和完整性")
    print("- 调整优化参数范围")
    print("- 尝试其他算法或框架")

print(f"\n执行的优化方法:")
for method, result in optimization_results.items():
    status = result.get('Status', 'Unknown') if isinstance(result, dict) else str(result)
    print(f"- {method.replace('_', ' ')}: {status}")

print(f"\n生成的文件:")
print(f"- optimization/comprehensive_optimization_results.png/.pdf: 综合结果可视化")
print(f"- optimization/results/comprehensive_optimization_summary.csv: 结果汇总")
print(f"- optimization/models/: 优化后的模型文件")

print(f"\n🎯 总体评估:")
successful_count = sum(1 for result in optimization_results.values() 
                      if isinstance(result, dict) and result.get('Status') == 'Success')
total_count = len(optimization_results)

print(f"- 成功率: {successful_count}/{total_count} ({successful_count/total_count*100:.1f}%)")

if successful_count > 0:
    print("- 综合优化取得积极成果")
    print("- 建议将最佳模型应用于实际预测")
else:
    print("- 优化效果有限，建议深入分析原因")
    print("- 可考虑数据增强或算法创新")

print("\n综合模型优化完成！")