import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import os
import json

def format_datetime(dt_str):
    """格式化日期时间字符串"""
    try:
        dt = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S.%f')
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except:
        return dt_str

def get_dataset_last_time():
    """获取数据集最后一个时间点"""
    try:
        # 读取ETTh1.csv文件
        df = pd.read_csv('data/ETT/ETTh1.csv')
        # 将date列转换为datetime类型
        df['date'] = pd.to_datetime(df['date'])
        # 返回最后一个时间点
        return df['date'].iloc[-1]
    except Exception as e:
        print(f"警告：无法读取数据集最后时间点: {str(e)}")
        return datetime.now()

def analyze_and_visualize(pred_path, true_path, output_dir):
    """分析预测结果并生成可视化图表
    
    Args:
        pred_path: 预测值npy文件路径
        true_path: 真实值npy文件路径
        output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载预测数据和真实值
    predictions = np.load(pred_path)
    groundtruth = np.load(true_path)

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    # 数据基本信息
    print("预测数据形状:", predictions.shape)
    print("真实值数据形状:", groundtruth.shape)
    print("标准化后的预测数据范围:", np.min(predictions), "到", np.max(predictions))
    print("标准化后的真实值数据范围:", np.min(groundtruth), "到", np.max(groundtruth))

    # 从数据集加载均值和标准差
    try:
        dataset_stats = np.load(os.path.join(os.path.dirname(pred_path), 'dataset_stats.npy'), allow_pickle=True)
        stats_dict = dataset_stats.item()  # 获取字典对象
        means = stats_dict['mean']  # 获取均值数组
        stds = stats_dict['std']   # 获取标准差数组
        print("\n加载的数据集统计信息:")
        print("均值:", means)
        print("标准差:", stds)
    except Exception as e:
        print(f"\n警告：加载数据集统计信息失败: {str(e)}")
        print("使用原始数据集计算统计信息...")
        # 读取原始数据集
        df = pd.read_csv('data/ETT/ETTh1.csv')
        # 只选择HUFL, MUFL, LUFL三列
        selected_columns = ['HUFL', 'MUFL', 'LUFL']
        means = df[selected_columns].mean().values
        stds = df[selected_columns].std().values
        print("计算的统计信息:")
        print("均值:", means)
        print("标准差:", stds)

    # 反标准化数据
    predictions_original = np.zeros_like(predictions)
    groundtruth_original = np.zeros_like(groundtruth)
    
    # 对每个特征分别进行反标准化
    for i in range(predictions.shape[-1]):
        # 反标准化
        predictions_original[..., i] = predictions[..., i] * stds[i] + means[i]
        groundtruth_original[..., i] = groundtruth[..., i] * stds[i] + means[i]
        
        # 确保值非负
        predictions_original[..., i] = np.maximum(predictions_original[..., i], 0)
        groundtruth_original[..., i] = np.maximum(groundtruth_original[..., i], 0)
    
    # 打印反标准化后的数据范围
    print("\n反标准化后的数据范围:")
    for i in range(predictions.shape[-1]):
        print(f"特征 {i+1}:")
        print(f"预测数据范围: {np.min(predictions_original[..., i]):.2f} 到 {np.max(predictions_original[..., i]):.2f}")
        print(f"真实值数据范围: {np.min(groundtruth_original[..., i]):.2f} 到 {np.max(groundtruth_original[..., i]):.2f}")

    # 获取数据集最后时间点
    start_time = get_dataset_last_time()
    # 创建预测时间点（从数据集最后时间点开始）
    times = [start_time + timedelta(hours=i) for i in range(1, predictions.shape[1] + 1)]
    time_labels = [t.strftime('%Y-%m-%d %H:%M') for t in times]

    # 定义特征名称映射
    feature_names = {
        0: 'HUFL (高压设备负荷)',
        1: 'MUFL (中压设备负荷)',
        2: 'LUFL (低压设备负荷)'
    }

    # 选择要显示的特征
    sample_idx = 0   # 第一个样本
    for feature_idx in range(3):  # 3个特征
        # 获取选定特征的统计指标
        cur_pred = predictions_original[sample_idx, :, feature_idx]
        cur_true = groundtruth_original[sample_idx, :, feature_idx]
        
        # 计算统计指标（仅针对选定特征）
        mean_pred = np.mean(cur_pred)
        std_pred = np.std(cur_pred)
        max_pred = np.max(cur_pred)
        min_pred = np.min(cur_pred)
        
        # 计算预测误差（仅针对选定特征）
        mae = np.mean(np.abs(cur_pred - cur_true))
        mse = np.mean((cur_pred - cur_true) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((cur_true - cur_pred) / (cur_true + 1e-5))) * 100

        # 创建图表
        plt.figure(figsize=(15, 8))

        # 绘制预测值
        plt.plot(range(len(time_labels)), predictions_original[sample_idx, :, feature_idx], 'b-', 
                label='预测值', linewidth=2, marker='o')
        # 绘制真实值
        plt.plot(range(len(time_labels)), groundtruth_original[sample_idx, :, feature_idx], 'r--', 
               label='真实值', linewidth=2, marker='s')

        # 标注预测值
        for i, pred in enumerate(predictions_original[sample_idx, :, feature_idx]):
            plt.annotate(f'{pred:.2f}', (i, pred), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=8, color='blue')
        # 标注真实值
        for i, true in enumerate(groundtruth_original[sample_idx, :, feature_idx]):
            plt.annotate(f'{true:.2f}', (i, true), textcoords="offset points", 
                        xytext=(0,-15), ha='center', fontsize=8, color='red')

        # 添加网格
        plt.grid(True, linestyle='--', alpha=0.7)

        # 设置标题和标签
        plt.title(f'未来{predictions.shape[1]}小时负荷预测结果\n{feature_names[feature_idx]}', fontsize=14, pad=20)
        plt.xlabel('预测时间', fontsize=12)
        plt.ylabel('负荷值', fontsize=12)

        # 设置x轴刻度
        plt.xticks(range(len(time_labels)), time_labels, rotation=45)

        # 添加统计信息文本框
        stats_text = f'统计信息 ({feature_names[feature_idx]}):\n' \
                    f'预测平均值: {mean_pred:.2f}\n' \
                    f'预测标准差: {std_pred:.2f}\n' \
                    f'预测最大值: {max_pred:.2f}\n' \
                    f'预测最小值: {min_pred:.2f}'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
                verticalalignment='top', fontsize=10)

        # 图例只保留预测值
        plt.legend(loc='upper right', fontsize=10)

        # 调整布局
        plt.tight_layout()

        # 保存图表
        viz_path = os.path.join(output_dir, f'prediction_vs_groundtruth_feature_{feature_idx}.png')
        print(f"\n保存{feature_names[feature_idx]}的对比图到: {viz_path}")
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()

        # 保存详细数据到CSV
        csv_path = os.path.join(output_dir, f'prediction_vs_groundtruth_feature_{feature_idx}.csv')
        print(f"保存{feature_names[feature_idx]}的分析结果到: {csv_path}")
        results_df = pd.DataFrame({
            'Time': time_labels,
            'Prediction': predictions_original[sample_idx, :, feature_idx],
            'Groundtruth': groundtruth_original[sample_idx, :, feature_idx],
            'Absolute_Error': np.abs(predictions_original[sample_idx, :, feature_idx] - groundtruth_original[sample_idx, :, feature_idx])
        })
        results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

        print(f"\n{feature_names[feature_idx]}的预测误差统计:")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAPE: {mape:.4f}%")

    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'mean_pred': mean_pred,
        'std_pred': std_pred,
        'max_pred': max_pred,
        'min_pred': min_pred
    }

if __name__ == '__main__':
    # 如果直接运行此脚本，使用命令行参数
    import argparse
    parser = argparse.ArgumentParser(description='分析预测结果并生成可视化图表')
    parser.add_argument('--pred_path', type=str, required=True, help='预测值npy文件路径')
    parser.add_argument('--true_path', type=str, required=True, help='真实值npy文件路径')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    args = parser.parse_args()
    
    analyze_and_visualize(args.pred_path, args.true_path, args.output_dir) 