import torch
import torch.nn as nn
import math


class AttentionLayer(nn.Module):
    """
    注意力层：实现Transformer中的自注意力机制
    """

    def __init__(self, d_model):
        super(AttentionLayer, self).__init__()
        # 层归一化，规范化输入特征，提升训练稳定性和收敛速度
        self.norm = nn.LayerNorm(d_model)

        # 线性层，将输入特征映射为查询（Q）、键（K）、值（V）
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)

        # 输出线性层，将注意力输出映射回原始特征维度
        self.out_projection = nn.Linear(d_model, d_model)

    def forward(self, x):
        # 输入x形状为 [B, T, D]
        # B: batch size，T: 时间步长，D: 特征维度
        B, T, D = x.shape

        # 通过线性层生成查询Q，形状仍为 [B, T, D]
        queries = self.query_projection(x)  # [B, T, D]

        # 通过线性层生成键K，随后转置维度1和2，变为 [B, D, T]
        # 转置是为了后续矩阵乘法计算注意力分数
        keys = self.key_projection(x).transpose(1, 2)  # [B, D, T]

        # 通过线性层生成值V，形状为 [B, T, D]
        values = self.value_projection(x)  # [B, T, D]

        # 计算缩放点积注意力权重：
        # 先计算 Q 与 K 的矩阵乘积，形状为 [B, T, T]
        # 除以 sqrt(D) 进行缩放，防止数值过大导致梯度消失或爆炸
        # 最后对最后一个维度（时间步）做softmax，得到注意力权重分布
        attn = torch.softmax(torch.matmul(queries, keys) / math.sqrt(D), -1)  # [B, T, T]

        # 用注意力权重加权值V，得到加权和，形状为 [B, T, D]
        # 并与输入x做残差连接，帮助梯度流动和模型训练
        out = torch.matmul(attn, values) + x  # [B, T, D]

        # 对加权和结果先做层归一化，再通过输出线性层映射
        # 最后再加上残差连接out，实现Transformer中的残差结构
        return self.out_projection(self.norm(out)) + out, attn


# --------------------- embed ------------------------------------
import torch
import torch.nn as nn
import math


class PositionalEmbedding(nn.Module):
    """
    位置嵌入层：为序列添加位置信息，使模型感知时间顺序
    """

    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # 预计算位置编码
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False  # 位置编码不参与训练

        # 计算位置索引
        position = torch.arange(0, max_len).float().unsqueeze(1)
        # 计算衰减因子
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        # 偶数维度用正弦函数，奇数维度用余弦函数
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)  # 注册为缓冲区，不被视为模型参数

    def forward(self, data=None, idx=None):
        # 两种获取位置编码的方式：
        # 1. 根据输入数据长度获取前N个位置的编码
        if data is not None:
            p = self.pe[:data].unsqueeze(0)  # [1, T, D]
        # 2. 根据索引获取特定位置的编码
        else:
            # 为每个样本按索引选取位置编码
            p = self.pe.unsqueeze(0).repeat(idx.shape[0], 1, 1)[torch.arange(idx.shape[0])[:, None], idx, :]
        return p


class TokenEmbedding(nn.Module):
    """
    令牌嵌入层：将输入特征通过卷积映射到模型维度
    """

    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        # 根据PyTorch版本设置填充方式
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        # 1D卷积层：将输入特征维度c_in映射到d_model
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=padding,
            padding_mode='circular',  # 循环填充，保持序列长度
            bias=False
        )
        # 初始化卷积层权重
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        # x: [B, T, C] -> 转置为[B, C, T]进行卷积 -> 再转置回[B, T, D]
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class DataEmbedding(nn.Module):
    """
    数据嵌入层：组合令牌嵌入和位置嵌入
    """

    def __init__(self, c_in, d_model, dropout=0.05):
        super(DataEmbedding, self).__init__()
        # 特征值嵌入
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        # 位置嵌入
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        # Dropout层防止过拟合
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # 组合特征嵌入和位置嵌入，并应用dropout
        x = self.value_embedding(x) + self.position_embedding(data=x.shape[1])
        return self.dropout(x)