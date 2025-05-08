import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridBlock(nn.Module):
    """优化的混合卷积-注意力模块
    
    特性：
    1. 简化的多尺度卷积特征提取
    2. 动态自适应融合机制
    3. 优化的维度处理
    4. 预标准化和增强的激活函数
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(HybridBlock, self).__init__()
        
        # 可学习的融合参数，初始值偏向卷积特征
        self.fusion_weight = nn.Parameter(torch.tensor(0.7))
        
        # 标准化层
        self.norm = nn.LayerNorm(d_model)
        
        # 简化的卷积网络 - 减少参数量
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
            nn.GELU()
        )
        
        # 自注意力机制
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=n_heads, 
            dropout=dropout,
            batch_first=True
        )
        
        # dropout层
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        
        # 保存原始输入用于残差连接
        identity = x
        
        # 预标准化
        x_norm = self.norm(x)
        
        # 1. 卷积特征提取路径
        x_conv = x_norm.transpose(1, 2)  # [batch, d_model, seq_len]
        x_conv = self.conv(x_conv)
        x_conv = x_conv.transpose(1, 2)  # [batch, seq_len, d_model]
        
        # 2. 注意力特征提取路径
        x_attn, _ = self.self_attn(x_norm, x_norm, x_norm)
        
        # 3. 融合两种特征 - 使用可学习权重进行简单融合
        x_fused = self.fusion_weight * x_conv + (1 - self.fusion_weight) * x_attn
        
        # 4. 添加残差连接和dropout
        return identity + self.dropout(x_fused)
