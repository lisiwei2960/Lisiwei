import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(true - pred))


def MSE(pred, true):
    return np.mean((true - pred) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    # 避免除以 0 或非常小的数，确保结果不失控
    epsilon = 1e-5
    return np.mean(np.abs((true - pred) / (true + epsilon)))


def MSPE(pred, true):
    epsilon = 1e-5
    return np.mean(np.square((true - pred) / (true + epsilon)))


def metric(pred, true):
    """
    计算预测结果的评估指标
    Args:
        pred: 预测值，shape为(batch_size, pred_len, feature_dim)
        true: 真实值，shape为(batch_size, pred_len, feature_dim)
    Returns:
        mae: 平均绝对误差
        mse: 均方误差
        rmse: 均方根误差
        mape: 平均绝对百分比误差
        mspe: 均方百分比误差
    """
    # 确保输入是numpy数组
    pred = np.array(pred)
    true = np.array(true)
    
    # 计算每个特征的指标
    mae_list = []
    mse_list = []
    rmse_list = []
    mape_list = []
    mspe_list = []
    
    # 对每个特征分别计算指标
    for i in range(pred.shape[2]):  # 遍历每个特征
        pred_i = pred[:, :, i]
        true_i = true[:, :, i]
        
        # 计算MAE
        mae = mean_absolute_error(true_i.flatten(), pred_i.flatten())
        mae_list.append(mae)
        
        # 计算MSE
        mse = mean_squared_error(true_i.flatten(), pred_i.flatten())
        mse_list.append(mse)
        
        # 计算RMSE
        rmse = np.sqrt(mse)
        rmse_list.append(rmse)
        
        # 计算MAPE
        mape = np.mean(np.abs((true_i - pred_i) / (true_i + 1e-5)))
        mape_list.append(mape)
        
        # 计算MSPE
        mspe = np.mean(np.square((true_i - pred_i) / (true_i + 1e-5)))
        mspe_list.append(mspe)
    
    # 计算所有特征的平均指标
    mae = np.mean(mae_list)
    mse = np.mean(mse_list)
    rmse = np.mean(rmse_list)
    mape = np.mean(mape_list)
    mspe = np.mean(mspe_list)
    
    # 打印每个特征的具体指标
    feature_names = ['HUFL', 'MUFL', 'LUFL']
    print('\n各特征评估指标:')
    for i, name in enumerate(feature_names):
        print(f'\n{name}:')
        print(f'MAE: {mae_list[i]:.4f}')
        print(f'MSE: {mse_list[i]:.4f}')
        print(f'RMSE: {rmse_list[i]:.4f}')
        print(f'MAPE: {mape_list[i]:.4f}')
        print(f'MSPE: {mspe_list[i]:.4f}')
    
    return mae, mse, rmse, mape, mspe
