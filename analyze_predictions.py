import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import os

def analyze_and_visualize(pred_path, true_path, output_dir):
    """分析预测结果并生成可视化图表
    
    Args:
        pred_path: 预测值npy文件路径
        true_path: 真实值npy文件路径
        output_dir: 输出目录
    """
    # 加载预测数据和真实值
    predictions = np.load(pred_path)
    groundtruth = np.load(true_path)

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    # 数据基本信息
    print("预测数据形状:", predictions.shape)
    print("真实值数据形状:", groundtruth.shape)
    print("预测数据范围:", np.min(predictions), "到", np.max(predictions))
    print("真实值数据范围:", np.min(groundtruth), "到", np.max(groundtruth))

    # 反标准化函数（使用数据集的均值和标准差）
    def inverse_normalize(data, mean=0, std=1):
        return data * std + mean

    # 假设我们知道原始数据的均值和标准差（需要从训练数据中获取）
    means = [0] * 7
    stds = [1] * 7

    # 反标准化数据
    predictions_original = np.zeros_like(predictions)
    groundtruth_original = np.zeros_like(groundtruth)
    for i in range(7):  # 对每个特征
        predictions_original[..., i] = inverse_normalize(predictions[..., i], means[i], stds[i])
        groundtruth_original[..., i] = inverse_normalize(groundtruth[..., i], means[i], stds[i])

    # 创建时间索引（每个预测点间隔1小时，总共6小时）
    start_time = datetime.now()
    times = [start_time + timedelta(hours=i) for i in range(6)]  # 改为6小时
    time_labels = [t.strftime('%H:%M') for t in times]

    # 计算统计指标
    mean_pred = np.mean(predictions_original)
    std_pred = np.std(predictions_original)
    max_pred = np.max(predictions_original)
    min_pred = np.min(predictions_original)

    # 计算预测误差
    mae = np.mean(np.abs(predictions_original - groundtruth_original))
    mse = np.mean((predictions_original - groundtruth_original) ** 2)
    rmse = np.sqrt(mse)

    # 创建图表
    plt.figure(figsize=(15, 8))
    feature_idx = 0  # 可以修改这个索引来查看不同特征
    sample_idx = 0   # 可以修改这个索引来查看不同样本

    # 绘制预测结果和真实值
    plt.plot(time_labels, predictions_original[sample_idx, :, feature_idx], 'b-', 
             label='预测值', linewidth=2, marker='o')
    plt.plot(time_labels, groundtruth_original[sample_idx, :, feature_idx], 'r--', 
             label='真实值', linewidth=2, marker='s')

    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.7)

    # 设置标题和标签
    plt.title(f'未来6小时负荷预测结果对比\n(特征 {feature_idx + 1})', fontsize=14, pad=20)
    plt.xlabel('预测时间', fontsize=12)
    plt.ylabel('负荷值', fontsize=12)

    # 旋转x轴标签
    plt.xticks(rotation=45)

    # 添加统计信息文本框
    stats_text = f'统计信息:\n' \
                 f'预测平均值: {mean_pred:.2f}\n' \
                 f'预测标准差: {std_pred:.2f}\n' \
                 f'预测最大值: {max_pred:.2f}\n' \
                 f'预测最小值: {min_pred:.2f}\n' \
                 f'MAE: {mae:.4f}\n' \
                 f'RMSE: {rmse:.4f}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
             verticalalignment='top', fontsize=10)

    # 添加图例
    plt.legend(loc='upper right', fontsize=10)

    # 调整布局
    plt.tight_layout()

    # 保存图表
    viz_path = os.path.join(output_dir, 'prediction_vs_groundtruth.png')
    print(f"保存对比图到: {viz_path}")
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 保存详细数据到CSV
    csv_path = os.path.join(output_dir, 'prediction_vs_groundtruth.csv')
    print(f"保存分析结果到: {csv_path}")
    results_df = pd.DataFrame({
        'Time': time_labels,
        'Prediction': predictions_original[sample_idx, :, feature_idx],
        'Groundtruth': groundtruth_original[sample_idx, :, feature_idx],
        'Absolute_Error': np.abs(predictions_original[sample_idx, :, feature_idx] - groundtruth_original[sample_idx, :, feature_idx])
    })
    results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    print(f"\n预测结果已保存到 {csv_path}")
    print(f"可视化图表已保存到 {viz_path}")
    print(f"\n预测误差统计:")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

    # 确保文件已正确保存
    if not os.path.exists(viz_path):
        print(f"警告：对比图文件未能成功保存: {viz_path}")
    if not os.path.exists(csv_path):
        print(f"警告：分析结果文件未能成功保存: {csv_path}")

    return {
        'mae': mae,
        'rmse': rmse,
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