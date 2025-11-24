#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 4: 时间序列建模
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
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression, RFE

# 深度学习相关  
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available, will skip deep learning models")

# 时间序列模型
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Statsmodels not available, will skip traditional time series models")

# XGBoost和LightGBM
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available")

import joblib
import os

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("========== Phase 4: 时间序列建模 ==========")
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("目标: 构建和训练多种预测模型")

# Step 1: 加载特征数据
print("\n1. 加载特征数据")
features_df = pd.read_csv('data/processed/comprehensive_features_final_new.csv')
longitudinal_df = pd.read_csv('data/processed/phase1_longitudinal_data.csv')

print(f"特征数据形状: {features_df.shape}")
print(f"特征数量: {features_df.shape[1] - 2}")  # 减去patient_id和target

# 准备建模数据
feature_cols = [col for col in features_df.columns if col not in ['patient_id', 'sample_id', 'target_liver_stiffness']]
X = features_df[feature_cols]
y = features_df['target_liver_stiffness']

print(f"输入特征维度: {X.shape}")
print(f"目标变量分布: 均值={y.mean():.2f}, 标准差={y.std():.2f}, 范围=[{y.min():.2f}, {y.max():.2f}]")

# Step 2: 数据分割
print("\n2. 数据分割")

# 随机分割（传统方法）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"训练集大小: {X_train.shape[0]}")
print(f"测试集大小: {X_test.shape[0]}")

# 时间序列分割（用于交叉验证）
tscv = TimeSeriesSplit(n_splits=5)
print(f"时间序列交叉验证: {tscv.n_splits} 折")

# Step 3: 基础机器学习模型
print("\n3. 基础机器学习模型")

models = {}
model_results = {}

def evaluate_model(name, model, X_train, X_test, y_train, y_test, cv_score=None):
    """评估模型性能"""
    # 训练模型
    model.fit(X_train, y_train)
    
    # 预测
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # 计算指标
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    try:
        test_mape = mean_absolute_percentage_error(y_test, y_test_pred) * 100
    except:
        test_mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100
    
    results = {
        'model': model,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'test_mape': test_mape,
        'cv_score': cv_score,
        'predictions': y_test_pred
    }
    
    print(f"{name}:")
    print(f"  训练 MAE: {train_mae:.3f}, RMSE: {train_rmse:.3f}, R²: {train_r2:.3f}")
    print(f"  测试 MAE: {test_mae:.3f}, RMSE: {test_rmse:.3f}, R²: {test_r2:.3f}, MAPE: {test_mape:.1f}%")
    if cv_score is not None:
        print(f"  CV MAE: {-cv_score:.3f}")
    
    return results

# 3.1 线性回归
print("3.1 线性回归模型")
lr = LinearRegression()
cv_scores = cross_val_score(lr, X_train, y_train, cv=tscv, scoring='neg_mean_absolute_error')
model_results['Linear Regression'] = evaluate_model('Linear Regression', lr, X_train, X_test, y_train, y_test, cv_scores.mean())
models['Linear Regression'] = lr

# 3.2 Ridge回归
print("\n3.2 Ridge回归（L2正则化）")
ridge = Ridge(alpha=1.0)
cv_scores = cross_val_score(ridge, X_train, y_train, cv=tscv, scoring='neg_mean_absolute_error')
model_results['Ridge'] = evaluate_model('Ridge Regression', ridge, X_train, X_test, y_train, y_test, cv_scores.mean())
models['Ridge'] = ridge

# 3.3 Lasso回归
print("\n3.3 Lasso回归（L1正则化）")
lasso = Lasso(alpha=0.1)
cv_scores = cross_val_score(lasso, X_train, y_train, cv=tscv, scoring='neg_mean_absolute_error')
model_results['Lasso'] = evaluate_model('Lasso Regression', lasso, X_train, X_test, y_train, y_test, cv_scores.mean())
models['Lasso'] = lasso

# 3.4 随机森林
print("\n3.4 随机森林")
rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
cv_scores = cross_val_score(rf, X_train, y_train, cv=tscv, scoring='neg_mean_absolute_error')
model_results['Random Forest'] = evaluate_model('Random Forest', rf, X_train, X_test, y_train, y_test, cv_scores.mean())
models['Random Forest'] = rf

# 3.5 梯度提升
print("\n3.5 梯度提升")
gbr = GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=6)
cv_scores = cross_val_score(gbr, X_train, y_train, cv=tscv, scoring='neg_mean_absolute_error')
model_results['Gradient Boosting'] = evaluate_model('Gradient Boosting', gbr, X_train, X_test, y_train, y_test, cv_scores.mean())
models['Gradient Boosting'] = gbr

# 3.6 支持向量回归
print("\n3.6 支持向量回归")
svr = SVR(kernel='rbf', C=10, gamma='scale')
cv_scores = cross_val_score(svr, X_train, y_train, cv=tscv, scoring='neg_mean_absolute_error')
model_results['SVR'] = evaluate_model('Support Vector Regression', svr, X_train, X_test, y_train, y_test, cv_scores.mean())
models['SVR'] = svr

# Step 4: 高级集成模型
print("\n4. 高级集成模型")

# 4.1 XGBoost
if XGBOOST_AVAILABLE:
    print("4.1 XGBoost")
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        subsample=0.8,
        colsample_bytree=0.8
    )
    cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=tscv, scoring='neg_mean_absolute_error')
    model_results['XGBoost'] = evaluate_model('XGBoost', xgb_model, X_train, X_test, y_train, y_test, cv_scores.mean())
    models['XGBoost'] = xgb_model

# 4.2 LightGBM  
if LIGHTGBM_AVAILABLE:
    print("\n4.2 LightGBM")
    lgb_model = lgb.LGBMRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        subsample=0.8,
        colsample_bytree=0.8,
        verbose=-1
    )
    cv_scores = cross_val_score(lgb_model, X_train, y_train, cv=tscv, scoring='neg_mean_absolute_error')
    model_results['LightGBM'] = evaluate_model('LightGBM', lgb_model, X_train, X_test, y_train, y_test, cv_scores.mean())
    models['LightGBM'] = lgb_model

# Step 5: 深度学习序列模型
print("\n5. 深度学习序列模型")

if TENSORFLOW_AVAILABLE:
    print("5.1 构建LSTM模型数据")
    
    # 为LSTM准备序列数据
    def prepare_lstm_data(longitudinal_data, features_data, sequence_length=5):
        """准备LSTM训练数据"""
        lstm_X = []
        lstm_y = []
        patient_ids = []
        
        for _, row in features_data.iterrows():
            patient_id = row['patient_id']
            target_value = row['target_liver_stiffness']
            
            # 获取该患者的时间序列数据
            patient_data = longitudinal_data[longitudinal_data['patient_id'] == patient_id].copy()
            patient_data = patient_data.sort_values('examination_date')
            
            if len(patient_data) >= 2:  # 至少需要2个点
                # 历史肝脏弹性值序列（不包括最后一个）
                liver_sequence = patient_data['liver_stiffness'].iloc[:-1].values
                
                # 如果序列长度不够，进行填充
                if len(liver_sequence) < sequence_length:
                    # 用第一个值向前填充
                    padded_sequence = np.full(sequence_length, liver_sequence[0])
                    padded_sequence[-len(liver_sequence):] = liver_sequence
                    liver_sequence = padded_sequence
                else:
                    # 如果超长，取最后sequence_length个
                    liver_sequence = liver_sequence[-sequence_length:]
                
                lstm_X.append(liver_sequence.reshape(-1, 1))  # LSTM需要3D输入
                lstm_y.append(target_value)
                patient_ids.append(patient_id)
        
        return np.array(lstm_X), np.array(lstm_y), patient_ids
    
    # 准备LSTM数据
    sequence_length = 5
    lstm_X, lstm_y, lstm_patient_ids = prepare_lstm_data(
        pd.read_csv('data/processed/longitudinal_data.csv'), 
        features_df, 
        sequence_length
    )
    
    print(f"LSTM数据形状: X={lstm_X.shape}, y={lstm_y.shape}")
    
    # 分割LSTM数据
    lstm_X_train, lstm_X_test, lstm_y_train, lstm_y_test = train_test_split(
        lstm_X, lstm_y, test_size=0.2, random_state=42
    )
    
    # 构建LSTM模型
    def create_lstm_model(input_shape):
        model = Sequential([
            Input(shape=input_shape),
            LSTM(50, return_sequences=True, dropout=0.2),
            LSTM(50, dropout=0.2),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model
    
    lstm_model = create_lstm_model((sequence_length, 1))
    
    print("5.2 训练LSTM模型")
    history = lstm_model.fit(
        lstm_X_train, lstm_y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )
    
    # LSTM预测和评估
    lstm_y_pred = lstm_model.predict(lstm_X_test, verbose=0).flatten()
    
    lstm_mae = mean_absolute_error(lstm_y_test, lstm_y_pred)
    lstm_rmse = np.sqrt(mean_squared_error(lstm_y_test, lstm_y_pred))
    lstm_r2 = r2_score(lstm_y_test, lstm_y_pred)
    lstm_mape = np.mean(np.abs((lstm_y_test - lstm_y_pred) / lstm_y_test)) * 100
    
    model_results['LSTM'] = {
        'model': lstm_model,
        'test_mae': lstm_mae,
        'test_rmse': lstm_rmse,
        'test_r2': lstm_r2,
        'test_mape': lstm_mape,
        'predictions': lstm_y_pred,
        'history': history
    }
    
    print(f"LSTM模型:")
    print(f"  测试 MAE: {lstm_mae:.3f}, RMSE: {lstm_rmse:.3f}, R²: {lstm_r2:.3f}, MAPE: {lstm_mape:.1f}%")

# Step 6: 传统时间序列模型
print("\n6. 传统时间序列模型")

if STATSMODELS_AVAILABLE:
    print("6.1 ARIMA和指数平滑模型")
    
    # 选择几个典型患者进行时间序列建模示例
    ts_results = []
    sample_patients = features_df['patient_id'].sample(10, random_state=42).tolist()
    
    for patient_id in sample_patients[:3]:  # 只测试前3个以节省时间
        try:
            patient_data = longitudinal_df[longitudinal_df['patient_id'] == patient_id].copy()
            patient_data = patient_data.sort_values('examination_date')
            
            if len(patient_data) >= 4:  # 至少需要4个点
                liver_series = patient_data['liver_stiffness'].iloc[:-1]  # 历史数据
                target_value = patient_data['liver_stiffness'].iloc[-1]   # 目标值
                
                # 简单ARIMA模型
                try:
                    arima_model = ARIMA(liver_series, order=(1,0,1))
                    arima_fit = arima_model.fit()
                    arima_pred = arima_fit.forecast(steps=1)[0]
                    arima_error = abs(arima_pred - target_value)
                    ts_results.append(('ARIMA', patient_id, arima_error, arima_pred, target_value))
                except:
                    pass
                
                # 指数平滑
                try:
                    exp_model = ExponentialSmoothing(liver_series, trend='add', seasonal=None)
                    exp_fit = exp_model.fit()
                    exp_pred = exp_fit.forecast(steps=1)[0]
                    exp_error = abs(exp_pred - target_value)
                    ts_results.append(('ExpSmooth', patient_id, exp_error, exp_pred, target_value))
                except:
                    pass
        except:
            pass
    
    if ts_results:
        ts_df = pd.DataFrame(ts_results, columns=['Model', 'PatientID', 'Error', 'Prediction', 'Target'])
        ts_summary = ts_df.groupby('Model')['Error'].agg(['mean', 'std', 'count'])
        print("传统时间序列模型样本测试结果:")
        print(ts_summary)

# Step 7: 模型比较和可视化
print("\n7. 模型比较和可视化")

# 7.1 性能比较表
print("7.1 模型性能比较")
comparison_data = []

for name, results in model_results.items():
    if name != 'LSTM':  # LSTM结果格式不同
        comparison_data.append({
            'Model': name,
            'Test MAE': results['test_mae'],
            'Test RMSE': results['test_rmse'],
            'Test R²': results['test_r2'],
            'Test MAPE': results['test_mape'],
            'CV MAE': -results['cv_score'] if results['cv_score'] is not None else np.nan
        })

if 'LSTM' in model_results:
    comparison_data.append({
        'Model': 'LSTM',
        'Test MAE': model_results['LSTM']['test_mae'],
        'Test RMSE': model_results['LSTM']['test_rmse'],
        'Test R²': model_results['LSTM']['test_r2'],
        'Test MAPE': model_results['LSTM']['test_mape'],
        'CV MAE': np.nan
    })

comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.sort_values('Test MAE')
print(comparison_df.round(3))

# 7.2 可视化模型比较
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# MAE比较
ax = axes[0, 0]
mae_data = comparison_df[['Model', 'Test MAE']].sort_values('Test MAE')
bars = ax.bar(range(len(mae_data)), mae_data['Test MAE'])
ax.set_xticks(range(len(mae_data)))
ax.set_xticklabels(mae_data['Model'], rotation=45, ha='right')
ax.set_ylabel('MAE (kPa)')
ax.set_title('模型MAE比较')
ax.grid(True, alpha=0.3)

# 为条形图添加数值标签
for i, v in enumerate(mae_data['Test MAE']):
    ax.text(i, v + 0.01, f'{v:.2f}', ha='center', va='bottom')

# R²比较
ax = axes[0, 1]
r2_data = comparison_df[['Model', 'Test R²']].sort_values('Test R²', ascending=False)
bars = ax.bar(range(len(r2_data)), r2_data['Test R²'])
ax.set_xticks(range(len(r2_data)))
ax.set_xticklabels(r2_data['Model'], rotation=45, ha='right')
ax.set_ylabel('R²')
ax.set_title('模型R²比较')
ax.grid(True, alpha=0.3)

# 预测vs实际值散点图（最佳模型）
best_model_name = comparison_df.iloc[0]['Model']
best_results = model_results[best_model_name]

ax = axes[1, 0]
ax.scatter(y_test, best_results['predictions'], alpha=0.6, s=50)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', alpha=0.8)
ax.set_xlabel('实际值 (kPa)')
ax.set_ylabel('预测值 (kPa)')
ax.set_title(f'{best_model_name} - 预测vs实际')
ax.grid(True, alpha=0.3)

# 残差分布
ax = axes[1, 1]
residuals = y_test - best_results['predictions']
ax.hist(residuals, bins=20, alpha=0.7, edgecolor='black')
ax.set_xlabel('残差 (kPa)')
ax.set_ylabel('频次')
ax.set_title(f'{best_model_name} - 残差分布')
ax.axvline(x=0, color='red', linestyle='--', alpha=0.8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
# Save both PNG and PDF formats
plt.savefig('data/processed/phase4_model_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('data/processed/phase4_model_comparison.pdf', dpi=300, bbox_inches='tight')
plt.show()

# Step 8: 特征重要性分析
print("\n8. 特征重要性分析")

# 使用最佳的基于树的模型分析特征重要性
tree_based_models = ['Random Forest', 'Gradient Boosting']
if 'XGBoost' in models:
    tree_based_models.append('XGBoost')
if 'LightGBM' in models:
    tree_based_models.append('LightGBM')

if tree_based_models:
    best_tree_model_name = None
    best_tree_mae = float('inf')
    
    for model_name in tree_based_models:
        if model_name in model_results and model_results[model_name]['test_mae'] < best_tree_mae:
            best_tree_mae = model_results[model_name]['test_mae']
            best_tree_model_name = model_name
    
    if best_tree_model_name:
        best_tree_model = models[best_tree_model_name]
        
        if hasattr(best_tree_model, 'feature_importances_'):
            feature_importance = best_tree_model.feature_importances_
            feature_names = feature_cols
            
            # 创建重要性DataFrame并排序
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            
            print(f"{best_tree_model_name} 特征重要性 (Top 15):")
            for i, (_, row) in enumerate(importance_df.head(15).iterrows()):
                print(f"{i+1:2d}. {row['feature']:<30}: {row['importance']:.4f}")
            
            # 可视化特征重要性
            plt.figure(figsize=(12, 8))
            top_features = importance_df.head(20)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('重要性得分')
            plt.title(f'{best_tree_model_name} - 特征重要性 (Top 20)')
            plt.gca().invert_yaxis()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            # Save both PNG and PDF formats
            plt.savefig('data/processed/phase4_feature_importance.png', dpi=300, bbox_inches='tight')
            plt.savefig('data/processed/phase4_feature_importance.pdf', dpi=300, bbox_inches='tight')
            plt.show()
            
            # 保存特征重要性
            importance_df.to_csv('data/processed/feature_importance_ranking.csv', index=False)

# Step 9: 模型保存
print("\n9. 保存训练好的模型")

# 创建模型保存目录
os.makedirs('data/processed/models', exist_ok=True)

# 保存最佳模型
best_model_name = comparison_df.iloc[0]['Model']
best_model = models[best_model_name] if best_model_name != 'LSTM' else model_results['LSTM']['model']

if best_model_name == 'LSTM':
    if TENSORFLOW_AVAILABLE:
        best_model.save('data/processed/models/best_lstm_model.h5')
        print(f"保存最佳模型 ({best_model_name}) 到: data/processed/models/best_lstm_model.h5")
else:
    joblib.dump(best_model, f'data/processed/models/best_model_{best_model_name.replace(" ", "_").lower()}.pkl')
    print(f"保存最佳模型 ({best_model_name}) 到: data/processed/models/best_model_{best_model_name.replace(' ', '_').lower()}.pkl")

# 保存所有传统ML模型
for name, model in models.items():
    filename = f"data/processed/models/{name.replace(' ', '_').lower()}_model.pkl"
    joblib.dump(model, filename)

# 保存模型性能比较结果
comparison_df.to_csv('data/processed/model_performance_comparison.csv', index=False)

print("保存的模型文件:")
for name in models.keys():
    print(f"- {name.replace(' ', '_').lower()}_model.pkl")

# Step 10: 建模摘要报告
print("\n========== Phase 4 时间序列建模摘要报告 ==========")

print(f"\n建模数据集:")
print(f"- 训练样本数: {X_train.shape[0]}")
print(f"- 测试样本数: {X_test.shape[0]}")
print(f"- 特征维度: {X_train.shape[1]}")

print(f"\n模型性能排名:")
for i, (_, row) in enumerate(comparison_df.head(5).iterrows()):
    print(f"{i+1}. {row['Model']}: MAE={row['Test MAE']:.3f}, R²={row['Test R²']:.3f}")

print(f"\n最佳模型: {best_model_name}")
best_results = model_results[best_model_name] if best_model_name != 'LSTM' else model_results['LSTM']
print(f"- 测试MAE: {best_results['test_mae']:.3f} kPa")
print(f"- 测试RMSE: {best_results['test_rmse']:.3f} kPa")  
print(f"- 测试R²: {best_results['test_r2']:.3f}")
print(f"- 测试MAPE: {best_results['test_mape']:.1f}%")

print(f"\n模型类型统计:")
print(f"- 线性模型: 3 个 (Linear, Ridge, Lasso)")
print(f"- 集成模型: {len([x for x in models.keys() if x in ['Random Forest', 'Gradient Boosting', 'XGBoost', 'LightGBM']])} 个")
print(f"- 核方法: 1 个 (SVR)")
print(f"- 深度学习: {'1 个 (LSTM)' if 'LSTM' in model_results else '0 个'}")

print(f"\n临床意义评估:")
mae_threshold_1 = sum(1 for _, row in comparison_df.iterrows() if row['Test MAE'] <= 1.0)
mae_threshold_2 = sum(1 for _, row in comparison_df.iterrows() if row['Test MAE'] <= 2.0)
print(f"- MAE ≤ 1.0 kPa (优秀): {mae_threshold_1} 个模型")
print(f"- MAE ≤ 2.0 kPa (良好): {mae_threshold_2} 个模型")

r2_threshold = sum(1 for _, row in comparison_df.iterrows() if row['Test R²'] >= 0.8)
print(f"- R² ≥ 0.8 (强预测能力): {r2_threshold} 个模型")

print(f"\n完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("Phase 4 时间序列建模完成！")