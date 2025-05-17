import pandas as pd
import numpy as np

# 读取原始数据
print("读取原始数据...")
df = pd.read_csv('data/ETT/ETTh1.csv')
print("\n数据集基本信息:")
print(f"数据集大小: {df.shape}")
print("\n前5行数据:")
print(df.head())
print("\n数据统计信息:")
print(df.describe())

# 读取预测结果
print("\n读取预测结果...")
pred = np.load('results/long_term_forecast_TimeXer_TimesNet_ETTh1_ftM_sl96_ll48_pl6_dm16_nh8_el2_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0/pred.npy')
true = np.load('results/long_term_forecast_TimeXer_TimesNet_ETTh1_ftM_sl96_ll48_pl6_dm16_nh8_el2_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0/true.npy')

print("\n预测数据形状:", pred.shape)
print("真实值数据形状:", true.shape)

# 检查非零值
print("\n检查预测值中的非零值:")
print(f"预测值中非零值的数量: {np.count_nonzero(pred)}")
print(f"预测值中零值的数量: {pred.size - np.count_nonzero(pred)}")
print(f"预测值的范围: {np.min(pred)} 到 {np.max(pred)}")

print("\n检查真实值中的非零值:")
print(f"真实值中非零值的数量: {np.count_nonzero(true)}")
print(f"真实值中零值的数量: {true.size - np.count_nonzero(true)}")
print(f"真实值的范围: {np.min(true)} 到 {np.max(true)}")

# 检查第一个样本的所有特征
print("\n第一个样本的所有特征值:")
print("预测值:")
print(pred[0])
print("\n真实值:")
print(true[0]) 