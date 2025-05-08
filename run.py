import argparse
import os
import torch
import torch.backends
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_imputation import Exp_Imputation
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
from utils.print_args import print_args
import random
import numpy as np

if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--task_name', type=str,  default='imputation',
                        help='任务名称, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int,  default=1, help='是否训练，训练还是预测')
    parser.add_argument('--model_id', type=str, default='ETTh1_mask_0.125', help='模型 ID，用于标识不同的实验')
    parser.add_argument('--model', type=str, default='TimesNet',
                        help='模型名称, options: [Autoformer, Transformer, TimesNet]')

    # data loader
    parser.add_argument('--data', type=str, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTm1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='预测任务类型:[M, S, MS]; M:多测多, S:单测单, MS:多测单')
    parser.add_argument('--target', type=str, default='OT', help='目标特征列名，用于单变量预测任务')
    parser.add_argument('--freq', type=str, default='h',
                        help='时间特征编码的频率, options:[s:秒, t:分, h:时, d:天, b:工作日, w:周, m:月],也可以使用更详细的频率 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='模型检查点的保存路径')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='输入序列的长度')
    parser.add_argument('--label_len', type=int, default=0, help='起始标记的长度')
    parser.add_argument('--pred_len', type=int, default=0, help='预测序列的长度')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='M4 数据集的子集类型')
    parser.add_argument('--inverse', action='store_true', help='是否对输出数据进行逆变换', default=False)

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.125, help='数据掩码比例')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='先验异常比例（百分比）')

    # model define
    parser.add_argument('--expand', type=int, default=2, help='Mamba 模型的扩展因子')
    parser.add_argument('--d_conv', type=int, default=4, help='Mamba 模型的卷积核大小')
    parser.add_argument('--top_k', type=int, default=3, help='TimesBlock 模型的 top-k 参数')
    parser.add_argument('--num_kernels', type=int, default=6, help='Inception 模型的卷积核数量')
    parser.add_argument('--enc_in', type=int, default=7, help='编码器的输入维度')
    parser.add_argument('--dec_in', type=int, default=7, help='解码器的输入维度')
    parser.add_argument('--c_out', type=int, default=7, help='输出维度')
    parser.add_argument('--d_model', type=int, default=16, help='模型的维度')
    parser.add_argument('--n_heads', type=int, default=8, help='多头注意力机制的头数')
    parser.add_argument('--e_layers', type=int, default=2, help='编码器的层数')
    parser.add_argument('--d_layers', type=int, default=1, help='解码器的层数')
    parser.add_argument('--d_ff', type=int, default=32, help='全连接层的维度')
    parser.add_argument('--moving_avg', type=int, default=25, help='移动平均的窗口大小')
    parser.add_argument('--factor', type=int, default=3, help='注意力因子')
    parser.add_argument('--distil', action='store_false',help='是否在编码器中使用蒸馏技术',default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout 比例')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='时间特征编码方式，可选值包括:[固定时间编码timeF, 固定编码fixed, 学习编码learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='激活函数')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='通道独立性设置，1 表示通道独立，0 表示通道依赖')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='序列分解方法，支持 moving_avg 或 dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='是否使用归一化')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='下采样层数')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='下采样窗口大小')
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='下采样方法，支持 avg、max 或 conv')
    parser.add_argument('--seg_len', type=int, default=48,
                        help='SegRNN 模型的片段长度')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='数据加载的线程数')
    parser.add_argument('--itr', type=int, default=1, help='实验次数')
    parser.add_argument('--train_epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='训练数据的批次大小')
    parser.add_argument('--patience', type=int, default=5, help='早停法的耐心值')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='优化器的学习率')
    parser.add_argument('--des', type=str, default='Exp', help='实验描述')
    parser.add_argument('--loss', type=str, default='MSE', help='损失函数')
    parser.add_argument('--lradj', type=str, default='type1', help='学习率调整方式')
    parser.add_argument('--use_amp', type=int, choices=[0, 1], help='是否使用自动混合精度训练（0: 只用 FP32；1: 使用 AMP）', default=0)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')  # cuda or mps
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='投影器的隐藏层维度（列表）')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='投影器的隐藏层数')

    # metrics (dtw)
    parser.add_argument('--use_dtw', type=bool, default=False,
                        help='是否使用 DTW 指标（DTW 计算耗时，除非必要否则不建议使用）')

    # Augmentation
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="数据增强的次数")
    parser.add_argument('--seed', type=int, default=2, help="随机种子")
    parser.add_argument('--jitter', default=False, action="store_true", help="是否使用抖动增强")
    parser.add_argument('--scaling', default=True, action="store_true", help="是否使用缩放增强")
    parser.add_argument('--permutation', default=False, action="store_true",
                        help="是否使用等长排列增强")
    parser.add_argument('--randompermutation', default=False, action="store_true",
                        help="是否使用随机长度排列增强")
    parser.add_argument('--magwarp', default=False, action="store_true", help="是否使用幅度扭曲增强")
    parser.add_argument('--timewarp', default=False, action="store_true", help="是否使用时间扭曲增强")
    parser.add_argument('--windowslice', default=False, action="store_true", help="是否使用窗口切片增强")
    parser.add_argument('--windowwarp', default=False, action="store_true", help="是否使用窗口扭曲增强")
    parser.add_argument('--rotation', default=False, action="store_true", help="是否使用旋转增强")
    parser.add_argument('--spawner', default=False, action="store_true", help="是否使用 SPAWNER 增强")
    parser.add_argument('--dtwwarp', default=False, action="store_true", help="是否使用 DTW 扭曲增强")
    parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="是否使用形状 DTW 扭曲增强")
    parser.add_argument('--wdba', default=False, action="store_true", help="是否使用加权 DBA 增强")
    parser.add_argument('--discdtw', default=False, action="store_true",
                        help="是否使用判别式 DTW 增强")
    parser.add_argument('--discsdtw', default=False, action="store_true",
                        help="是否使用判别式形状 DTW 增强")
    parser.add_argument('--extra_tag', type=str, default="", help="额外的标签信息")

    # TimeXer
    parser.add_argument('--patch_len', type=int, default=16, help='TimeXer 模型的补丁长度')
    parser.add_argument('--use_se', action='store_true', help='启用 SE 通道注意力')
    parser.add_argument('--use_hybrid', action='store_true', help='启用混合卷积-注意力机制')

    args = parser.parse_args()
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
        print('Using GPU')
    else:
        if hasattr(torch.backends, "mps"):
            args.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        else:
            args.device = torch.device("cpu")
        print('Using cpu or mps')

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)

    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == 'short_term_forecast':
        Exp = Exp_Short_Term_Forecast
    elif args.task_name == 'imputation':
        Exp = Exp_Imputation
    elif args.task_name == 'anomaly_detection':
        Exp = Exp_Anomaly_Detection
    elif args.task_name == 'classification':
        Exp = Exp_Classification
    else:
        Exp = Exp_Long_Term_Forecast

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args)  # set experiments
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.expand,
                args.d_conv,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii)

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            if args.gpu_type == 'mps':
                torch.backends.mps.empty_cache()
            elif args.gpu_type == 'cuda':
                torch.cuda.empty_cache()
    else:
        exp = Exp(args)  # set experiments
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.expand,
            args.d_conv,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        if args.gpu_type == 'mps':
            torch.backends.mps.empty_cache()
        elif args.gpu_type == 'cuda':
            torch.cuda.empty_cache()
