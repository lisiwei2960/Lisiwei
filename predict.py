import torch
import numpy as np
import pandas as pd
from models.TimesNet import Model
from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import argparse
import os

def get_args():
    parser = argparse.ArgumentParser()
    
    # 基本配置
    parser.add_argument('--task_name', type=str, default='long_term_forecast',
                      help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, default=0, help='status')
    parser.add_argument('--model_id', type=str, default='test', help='model id')
    parser.add_argument('--model', type=str, default='TimesNet', help='model name')
    
    # 数据加载器
    parser.add_argument('--data', type=str, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                      help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                      help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    
    # 预测任务
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    
    # 模型定义
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=3, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=3, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=3, help='output size')
    parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
    parser.add_argument('--factor', type=int, default=3, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                      help='whether to use distilling in encoder, using this argument means not using distilling',
                      default=True)
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                      help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length for TimeXer')
    parser.add_argument('--use_norm', action='store_true', help='whether to use normalization in TimeXer')
    
    # 优化器
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    
    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() else False
    
    return args

def main(args=None, progress_callback=None):
    """运行预测
    Args:
        args: 预测参数，如果为None则使用默认参数
        progress_callback: 进度回调函数，接收(percent, message)
    """
    if args is None:
        args = get_args()
    
    # 加载数据
    data_set, data_loader = data_provider(args, flag='test')
    total_batches = len(data_loader)
    
    # 设置设备
    device = torch.device('cuda:0' if args.use_gpu else 'cpu')
    
    # 加载模型 - 启用SE和Hybrid模块
    model = Model(args, use_se=True, use_hybrid=True).to(device)
    model.load_state_dict(torch.load(args.checkpoints, map_location=device))
    model.eval()
    
    # 创建结果保存目录
    dataset_name = os.path.splitext(os.path.basename(args.data_path))[0]
    # 动态hour_str和pl参数
    hour_str = f"{args.pred_len}h"
    folder_path = f'./test_results/{dataset_name}_long_term_forecast_ETTh1_{hour_str}_TimesNet_ETTh1_ftM_sl96_ll48_pl{args.pred_len}_dm16_nh8_el2_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # 预测
    preds = []
    trues = []
    
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(data_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            
            # 解码器输入
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
            
            # 预测
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            
            # 保存结果
            pred = outputs.detach().cpu().numpy()
            true = batch_y[:, -args.pred_len:, :].detach().cpu().numpy()
            
            preds.append(pred)
            trues.append(true)
            
            # 进度回调
            if progress_callback is not None:
                percent = int((i + 1) / total_batches * 70) + 20  # 20~90之间
                progress_callback(percent, f'模型预测中...（进度{percent}%）')
    
    # 合并结果
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    
    # 计算指标
    mae, mse, rmse, mape, mspe = metric(preds, trues)
    print('Test MAE: {:.4f}'.format(mae))
    print('Test MSE: {:.4f}'.format(mse))
    print('Test RMSE: {:.4f}'.format(rmse))
    print('Test MAPE: {:.4f}'.format(mape))
    print('Test MSPE: {:.4f}'.format(mspe))
    
    # 保存预测结果
    np.save(os.path.join(folder_path, 'prediction.npy'), preds)
    np.save(os.path.join(folder_path, 'groundtruth.npy'), trues)
    
    # 保存数据集的统计信息
    if hasattr(data_set, 'scaler'):
        dataset_stats = {
            'mean': data_set.scaler.mean_,
            'std': data_set.scaler.scale_
        }
        np.save(os.path.join(folder_path, 'dataset_stats.npy'), dataset_stats)

    # 返回指标
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'mape': mape,
        'mspe': mspe
    }

if __name__ == '__main__':
    args = get_args()
    main(args) 