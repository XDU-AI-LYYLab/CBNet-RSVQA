import psutil
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import math
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchmetrics
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision import models
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models import resnet152, ResNet152_Weights
from transformers import RobertaModel


class CBAM(nn.Module):
    """
    完整版的卷积块注意力模块 (Convolutional Block Attention Module)
    包含串联的通道注意力模块 (Channel Attention) 和空间注意力模块 (Spatial Attention)
    """

    # 初始化函数
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()

        # ================== 1. 通道注意力模块 (Channel Attention Module) ==================
        # 平均池化和最大池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 共享的全连接层 (MLP)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
        )

        # ================== 2. 空间注意力模块 (Spatial Attention Module) ==================
        # 空间注意力的卷积层
        self.spatial_conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False
        )

        # ================== 共享的激活函数 ==================
        self.sigmoid = nn.Sigmoid()

    # 前向传播函数
    def forward(self, x):
        # 假设输入 x 的形状为: [B, C, H, W], 例如 [32, 768, 8, 8]
        b, c, h, w = x.size()

        # ================== 1. 通道注意力分支 ==================
        # 1.1 全局池化
        avg_out = self.avg_pool(x).view(b, c)  # -> [32, 768, 1, 1] -> [32, 768]
        max_out = self.max_pool(x).view(b, c)  # -> [32, 768, 1, 1] -> [32, 768]

        # 1.2 送入共享MLP
        avg_out = self.fc(avg_out)  # -> [32, 768]
        max_out = self.fc(max_out)  # -> [32, 768]

        # 1.3 逐元素相加并通过sigmoid获得通道注意力权重
        channel_att = self.sigmoid(avg_out + max_out).view(
            b, c, 1, 1
        )  # -> [32, 768] -> [32, 768, 1, 1]

        # 1.4 将通道注意力权重应用到原始特征图上
        x_channel_refined = (
            x * channel_att
        )  # [32, 768, 8, 8] * [32, 768, 1, 1] -> [32, 768, 8, 8]
        # '*'号在这里执行的是广播(broadcasting)乘法

        # ================== 2. 空间注意力分支 ==================
        # 2.1 在通道维度上进行最大池化和平均池化
        # torch.max返回(values, indices), 我们只需要值
        max_pool_out, _ = torch.max(
            x_channel_refined, dim=1, keepdim=True
        )  # -> [32, 1, 8, 8]
        avg_pool_out = torch.mean(
            x_channel_refined, dim=1, keepdim=True
        )  # -> [32, 1, 8, 8]

        # 2.2 将两个池化结果在通道维度上拼接
        pooled_features = torch.cat(
            [max_pool_out, avg_pool_out], dim=1
        )  # -> [32, 2, 8, 8]

        # 2.3 通过卷积层降维并用sigmoid获得空间注意力权重
        spatial_att = self.sigmoid(
            self.spatial_conv(pooled_features)
        )  # -> [32, 1, 8, 8]

        # 2.4 将空间注意力权重应用到通道注意力处理后的特征图上
        x_final_refined = (
            x_channel_refined * spatial_att
        )  # [32, 768, 8, 8] * [32, 1, 8, 8] -> [32, 768, 8, 8]

        return x_final_refined


class ImageEncoder(nn.Module):
    def __init__(self, feature_dim=768, freeze_level=3):
        super().__init__()
        # 加载预训练resnet152（ImageNet权重）
        # self.resnet152 = resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)
        # 1. 先创建一个不带预训练权重的ResNet152结构
        self.resnet152 = resnet152(weights=None)

        # 2. 定义本地权重文件的路径
        resnet_weights_path = "./offline_models/resnet152-imagenet1k-v1.pth"

        # 3. 加载本地权重文件
        print(f"✅ Loading local ResNet-152 weights from: {resnet_weights_path}")
        state_dict = torch.load(resnet_weights_path)

        # 4. 将权重加载到模型结构中
        self.resnet152.load_state_dict(state_dict)

        # 提取骨干网络的不同层 （conv1→bn1→relu→maxpool→layer1→layer2→layer3→layer4）
        self.conv1 = self.resnet152.conv1
        self.bn1 = self.resnet152.bn1
        self.relu = self.resnet152.relu
        self.maxpool = self.resnet152.maxpool
        self.layer1 = self.resnet152.layer1  # 输出通道: 256（空间尺寸64×64）
        self.layer2 = self.resnet152.layer2  # 输出通道: 512（32×32）
        self.layer3 = self.resnet152.layer3  # 输出通道: 1024（16×16）
        self.layer4 = self.resnet152.layer4  # 输出通道: 2048（8×8）

        # 特征投影模块
        self.q1_proj = nn.Sequential(
            nn.Conv2d(256, feature_dim, 3, 1, 1, bias=False),  # 256->768
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(),
        )
        self.q2_proj = nn.Sequential(
            nn.Conv2d(512, feature_dim, 3, 1, 1, bias=False),  # 512→768
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(),
        )
        self.q3_proj = nn.Sequential(
            nn.Conv2d(1024, feature_dim, 3, 1, 1, bias=False),  # 1024→768
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(),
        )
        self.q4_proj = nn.Sequential(
            nn.Conv2d(2048, feature_dim, 3, 1, 1, bias=False),  # 2048→768
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(),
        )

        # 通道注意力模块（增强layer4特征的关键通道）
        # self.cbam1 = CBAM(in_channels=256)
        # self.cbam2 = CBAM(in_channels=512)
        self.cbam3 = CBAM(in_channels=768)
        self.cbam4 = CBAM(in_channels=768)

        # 冻结指定层级参数
        self._freeze_layers(freeze_level)

    def _freeze_layers(self, freeze_level=3):
        """
        根据freeze_level冻结ResNet152的指定层（仅当freeze_level=3时，仅微调layer3和layer4）
        freeze_level规则：
            0: 冻结所有层级（所有参数不可训练）
            1: 仅训练conv1、bn1、relu、maxpool、layer1（其他层冻结）
            2: 仅训练conv1、bn1、relu、maxpool、layer1、layer2（其他层冻结）
            3: 仅训练layer3、layer4（其他层冻结）
            4: 训练全部层级（所有参数可训练）
        """
        # 定义各层组对应的最小训练level（freeze_level需≥该值才会解冻）
        layer_rules = {
            "conv1_group": 4,  # 包含conv1、bn1、relu、maxpool（仅freeze_level≥4时解冻）
            "layer1": 4,  # 第一层残差块（仅freeze_level≥4时解冻）
            "layer2": 4,  # 第二层残差块（仅freeze_level≥4时解冻）
            "layer3": 3,  # 第三层残差块（freeze_level≥3时解冻）
            "layer4": 3,  # 第四层残差块（freeze_level≥3时解冻）
        }

        # 遍历各层组，根据freeze_level设置参数是否可训练
        # 1. 处理conv1相关层组（conv1、bn1、relu、maxpool）
        for param in self.conv1.parameters():
            param.requires_grad = freeze_level >= layer_rules["conv1_group"]
        for param in self.bn1.parameters():
            param.requires_grad = freeze_level >= layer_rules["conv1_group"]
        for param in self.relu.parameters():
            param.requires_grad = freeze_level >= layer_rules["conv1_group"]
        for param in self.maxpool.parameters():
            param.requires_grad = freeze_level >= layer_rules["conv1_group"]

        # 2. 处理layer1
        for param in self.layer1.parameters():
            param.requires_grad = freeze_level >= layer_rules["layer1"]

        # 3. 处理layer2
        for param in self.layer2.parameters():
            param.requires_grad = freeze_level >= layer_rules["layer2"]

        # 4. 处理layer3
        for param in self.layer3.parameters():
            param.requires_grad = freeze_level >= layer_rules["layer3"]

        # 5. 处理layer4（ResNet152的第四层残差块）
        for param in self.layer4.parameters():
            param.requires_grad = freeze_level >= layer_rules["layer4"]

    def forward(self, image):
        """
        输入：image [B, 3, 224, 224]（TIF格式转RGB后的图像）
        输出：q1_feat [B, 768, 64, 64], q4_feat [B, 768, 8, 8], global_feat [B, 768]
        """
        # 1. 前置卷积层（conv1→bn1→relu→maxpool）
        x = self.conv1(image)  # [B, 64, 112, 112]
        x = self.bn1(x)  # [B, 64, 112, 112]
        x = self.relu(x)  # [B, 64, 112, 112]
        x = self.maxpool(x)  # [B, 64, 56, 56]

        # 2. 自适应平均池化（统一空间尺寸为64×64，适配后续特征提取）
        x = nn.AdaptiveAvgPool2d((64, 64))(x)  # [B, 64, 64, 64]

        # 3. 提取各层骨干特征
        l1 = self.layer1(x)  # [B, 256, 64, 64]（低层次纹理）
        l2 = self.layer2(l1)  # [B, 512, 32, 32]（中等语义）
        l3 = self.layer3(l2)  # [B, 1024, 16, 16]（高层次语义）
        l4 = self.layer4(l3)  # [B, 2048, 8, 8]（最高层次语义）

        # 特征投影
        q1_feat = self.q1_proj(l1)  # [B, 768, 64, 64]
        # q2_feat = self.q2_proj(l2)  # [B, 768, 32, 32]
        q3_feat = self.q3_proj(l3)  # [B, 768, 16, 16]
        q4_feat = self.q4_proj(l4)  # [B, 768, 8, 8]

        # 5. 通道注意力增强（仅对layer4特征增强，抑制背景噪声）
        q3_feat = self.cbam3(q3_feat)  # [B, 768, 64, 64]
        q4_feat = self.cbam4(q4_feat)  # [B, 768, 8, 8]
        batch = q1_feat.shape[0]

        global_feat = F.adaptive_avg_pool2d(q4_feat, (1, 1)).view(batch, -1)  # [B, 768]

        return q1_feat, q3_feat, global_feat


class TextEncoder(nn.Module):
    def __init__(self, in_dim=768, out_dim=256, shared_encoder=None):
        super().__init__()
        # 2. 直接使用传入的共享模型
        self.bert = shared_encoder
        # 使用1D池化替代2D池化（适配序列数据）
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=3)  # 压缩序列长度为1/3
        self.tanh = nn.Tanh()
        # GRU输入维度改为256（与卷积输出一致）
        self.gru = nn.GRU(
            768, hidden_size=768, num_layers=4, batch_first=True, dropout=0.2
        )
        # LayerNorm维度改为256（与卷积输出一致）
        self.ln1 = nn.LayerNorm((in_dim,))  # BERT输出保持768维
        self.ln2 = nn.LayerNorm((in_dim,))  # 卷积输出256维

        # 卷积层输出保持256维（out_dim=256）
        self.bigram = nn.Conv1d(in_dim, out_dim, 2, stride=1, padding=1, dilation=2)
        self.trigram = nn.Conv1d(in_dim, out_dim, 3, stride=1, padding=2, dilation=2)
        self.quagram = nn.Conv1d(in_dim, out_dim, 4, stride=1, padding=3, dilation=2)

    def forward(self, token_id, token_seg, token_mask):
        # 1. BERT编码（输出768维）
        q1 = self.bert(
            token_id, attention_mask=token_mask, token_type_ids=token_seg
        ).last_hidden_state
        q1 = self.ln1(q1)  # [batch, seq_len, 768]

        # 2. 特征增强（卷积提取多尺度特征）
        temp = q1.permute(0, 2, 1)  # [batch, 768, seq_len]（调整维度顺序适配Conv1d）

        # 卷积后输出形状：[batch, 256, seq_len]（通道数256，序列长度不变）
        qb = self.tanh(self.bigram(temp))  # [batch, 256, seq_len]
        qt = self.tanh(self.trigram(temp))  # [batch, 256, seq_len]
        qq = self.tanh(self.quagram(temp))  # [batch, 256, seq_len]

        # 3. 拼接通道维（关键修正！）
        # 直接在通道维（dim=1）拼接，得到 [batch, 256 * 3=768, seq_len]
        q_cat = torch.cat([qb, qt, qq], dim=1)  # [batch, 768, seq_len]

        # 4. 序列长度压缩（可选，根据任务需求）
        # 使用1D池化压缩序列长度（seq_len → seq_len//3）
        # q_pooled = self.maxpool(q_cat)  # [batch, 768, seq_len//3]

        # 5. 调整维度顺序并归一化
        q2 = q_cat.transpose(1, 2)  # [batch, seq_len//3, 768]
        q2 = self.ln2(q2)  # 归一化维度768（与输入一致）

        # 6. GRU处理（输入768维）
        q3, _ = self.gru(self.tanh(q2))  # [batch, seq_len//3, 768]
        q3 = self.tanh(q3)

        return q1, q3


class TextInAttention(nn.Module):
    def __init__(self, in_dim=768, out_dim=768):
        """
        自定义注意力模块，目标是通过显式的特征交互捕捉序列中的长距离依赖。

        Args:
            in_dim (int): 输入特征的维度（词级特征的维度，如 RoBERTa 的 768 维）
            out_dim (int): 输出特征的维度（经注意力增强后的特征维度）
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # 线性变换层：将输入特征投影到目标维度（in_dim → out_dim）
        self.W = nn.Linear(in_dim, out_dim)

        # 注意力分数计算层：输入为 2*out_dim（查询+键的拼接），输出为 1（未归一化的注意力分数）
        self.a = nn.Linear(2 * out_dim, 1)

        # 输出变换层：对注意力加权后的特征进行二次变换（out_dim → out_dim）
        self.Wo = nn.Linear(out_dim, out_dim)

        # 激活函数与正则化
        self.elu = nn.ELU()  # 指数线性单元，引入非线性
        self.softmax = nn.Softmax(dim=2)  # 对序列长度维度归一化注意力分数
        self.dropout = nn.Dropout(0.2)  # 防止过拟合

    def forward(self, input_h):
        """
        前向传播：通过显式特征交互计算增强后的特征。

        Args:
            input_h (Tensor): 输入特征，形状为 [batch_size, seq_len, in_dim]

        Returns:
            Tensor: 增强后的特征，形状为 [batch_size, seq_len, out_dim]
        """
        # --------------------------
        # 步骤1：特征投影与初始化
        # --------------------------
        # 输入特征通过线性层 W 投影到 out_dim 维度，并添加 Dropout 正则化
        h = self.W(self.dropout(input_h))  # 形状：[batch_size, seq_len, out_dim]
        B, L, D = h.size()  # 批次大小 B，序列长度 L，特征维度 D=out_dim

        # --------------------------
        # 步骤2：构造注意力交互矩阵（查询 Q 和键 K）
        # --------------------------
        # 目标：生成每个位置与其他所有位置的交互对（形状 [B, L, L, D]）

        # 方法：通过 unsqueeze + expand 显式复制特征到所有位置
        # - 查询 Q：每个位置 i 的特征复制到所有位置 j（包括自己），形状 [B, L, L, D]
        Q = h.unsqueeze(2).expand(B, L, L, D)  # [B, L, 1, D] → [B, L, L, D]

        # - 键 K：每个位置 j 的特征复制到所有位置 i（包括自己），形状 [B, L, L, D]
        K = h.unsqueeze(1).expand(B, L, L, D)  # [B, 1, L, D] → [B, L, L, D]

        # --------------------------
        # 步骤3：计算注意力分数
        # --------------------------
        # 拼接查询 Q 和键 K（在特征维度拼接），得到 [B, L, L, 2D]
        input_concat = torch.cat([Q, K], dim=-1)  # 形状：[B, L, L, 2D]

        # 通过线性层 a 计算原始注意力分数（未归一化），并添加 Dropout
        e_raw = self.a(self.dropout(input_concat))  # 形状：[B, L, L, 1]

        # 移除最后一维（大小为1），并通过 ELU 激活引入非线性
        e = self.elu(e_raw.squeeze(3))  # 形状：[B, L, L]

        # 对序列长度维度（dim=2）应用 Softmax，得到归一化的注意力权重
        att = self.softmax(e)  # 形状：[B, L, L]（每个位置 i 对其他位置 j 的注意力权重）

        # --------------------------
        # 步骤4：注意力加权与残差连接
        # --------------------------
        # 注意力权重 att [B, L, L] 与输入特征 h [B, L, D] 加权求和
        # - 矩阵乘法：att 的每个元素 att[i][j] 表示位置 i 对位置 j 的注意力权重
        # - 加权和形状：[B, L, D]（每个位置 i 的特征是所有位置 j 的特征按权重加权后的和）
        weighted_h = torch.bmm(att, h)  # 形状：[B, L, D]

        # 残差连接：保留原始特征 h，避免梯度消失
        output_h = weighted_h + h  # 形状：[B, L, D]

        # --------------------------
        # 步骤5：输出特征增强
        # --------------------------
        # 对加权后的特征通过线性层 Wo 二次变换，并添加 ELU 激活
        output_h = self.elu(self.Wo(self.dropout(output_h)))  # 形状：[B, L, out_dim]

        return output_h


class PositionalEncoding2D(nn.Module):
    """生成二维空间位置编码（可适应任意网格大小，使用可学习参数并适配GPU）"""

    def __init__(self, grid_size, embed_dim, device):
        super().__init__()
        self.grid_size = grid_size  # 输入特征图的空间尺寸（H=W=grid_size）
        self.embed_dim = embed_dim  # 位置编码的嵌入维度
        self.device = device  # 目标设备（如 "cuda:0" 或 "cpu"）

        # 初始化可学习的位置编码参数（一维基础向量，形状：[embed_dim//2]）
        # 关键：参数形状修正为一维向量，便于后续扩展为二维网格
        self.embedding_h = nn.Parameter(
            torch.empty(embed_dim // 2, device=device)  # 直接在目标设备初始化
        )
        self.embedding_w = nn.Parameter(
            torch.empty(embed_dim // 2, device=device)  # 直接在目标设备初始化
        )
        nn.init.normal_(self.embedding_h, std=0.02)  # 初始化（在目标设备上）
        nn.init.normal_(self.embedding_w, std=0.02)  # 初始化（在目标设备上）

    def forward(self, height, width):
        """
        生成位置编码（输出形状：[1, height*width, embed_dim]，位于目标设备）

        Args:
            height: 输入图像的高度（需等于 grid_size）
            width: 输入图像的宽度（需等于 grid_size）

        Returns:
            位置编码张量 [1, height*width, embed_dim]（位于目标设备）
        """
        # 验证输入尺寸与 grid_size 一致（确保位置编码与特征图空间对齐）
        assert (
            height == self.grid_size
        ), f"输入高度 {height} 与 grid_size {self.grid_size} 不匹配"
        assert (
            width == self.grid_size
        ), f"输入宽度 {width} 与 grid_size {self.grid_size} 不匹配"

        device = self.device
        grid_size = self.grid_size

        # --------------------------
        # 步骤1：生成可学习的位置编码基（形状适配）
        # --------------------------
        # 高度方向：[1, embed_dim//2] → [grid_size, embed_dim//2]（直接扩展高度维度）
        pos_h_base = self.embedding_h.repeat(grid_size, 1)  # 无需unsqueeze，直接重复
        # 宽度方向：[1, embed_dim//2] → [grid_size, embed_dim//2]（直接扩展宽度维度）
        pos_w_base = self.embedding_w.repeat(grid_size, 1)  # 无需unsqueeze，直接重复

        # --------------------------
        # 步骤2：拼接高度和宽度编码基（每个空间位置 (i,j) 对应一个 embed_dim 维编码）
        # --------------------------
        # 扩展维度以便拼接：高度编码扩展宽度维度，宽度编码扩展高度维度
        pos_h_expanded = pos_h_base.unsqueeze(2).expand(
            -1, -1, grid_size
        )  # [grid_size, embed_dim//2, grid_size]
        pos_w_expanded = pos_w_base.unsqueeze(2).expand(
            -1, -1, grid_size
        )  # [grid_size, embed_dim//2, grid_size]
        pos_enc = torch.cat(
            [pos_h_expanded, pos_w_expanded], dim=2
        )  # [grid_size, grid_size, embed_dim]

        # --------------------------
        # 步骤3：展平为空间序列（形状：[1, grid_size*grid_size, embed_dim]）
        # --------------------------
        # 调整维度顺序为 [embed_dim, grid_size, grid_size] → 展平高度和宽度维度
        pos_enc_flat = pos_enc.permute(2, 0, 1).reshape(-1, self.embed_dim)
        # 添加批次维度 → [1, grid_size*grid_size, embed_dim]
        pos_enc_final = pos_enc_flat.unsqueeze(0).to(device)  # 确保在目标设备上

        return pos_enc_final


class ImageSelfAttention(nn.Module):
    """适配图像特征的自注意力模块（可处理任意大小的特征图）"""

    def __init__(self, in_dim, out_dim, pos_embed_dim, grid_size):  # 768,768,64,64
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.pos_embed_dim = pos_embed_dim

        # 1. 投影层：输入特征 → 目标维度
        self.W = nn.Linear(in_dim + pos_embed_dim, out_dim)

        # 2. 二维位置编码
        self.pos_enc = PositionalEncoding2D(
            grid_size=grid_size, embed_dim=pos_embed_dim, device="cuda:0"
        )

        # 3. 注意力计算层
        self.a = nn.Linear(2 * out_dim, 1)  # 注意力分数计算
        self.Wo = nn.Linear(out_dim, out_dim)  # 输出变换
        self.elu = nn.ELU()
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(0.2)

    def forward(self, image_feat):
        """
        输入：image_feat [batch, in_dim, H, W]（二维图像特征）
        输出：enhanced_feat [batch, out_dim, H*W]（增强后的图像特征）
        """
        B, C, H, W = image_feat.shape

        # 展平图像特征：[B, C, H, W] → [B, H*W, C]
        image_flat = image_feat.transpose(1, 2).reshape(B, H * W, C)

        # 生成位置编码（确保height=H, width=W与grid_size一致）
        pos_enc = self.pos_enc(height=H, width=W)  # [1, H*W, embed_dim]
        # 拼接特征与位置编码（关键修正：维度对齐）
        # image_flat: [B, H*W, C]，pos_enc: [1, H*W, embed_dim]
        # 重复pos_enc以匹配batch维度，并拼接最后一维
        pos_enc = pos_enc.repeat(B, 1, 1)  # [B, H*W, embed_dim]
        fused_feat = torch.cat([image_flat, pos_enc], dim=-1)  # [B, H*W, C+embed_dim]

        # 后续自注意力计算（保持不变）
        h = self.W(self.dropout(fused_feat))  # [B, H*W, out_dim]

        # --------------------------
        # 步骤3：自注意力计算
        # --------------------------
        # 特征投影到 out_dim 维度
        h = self.W(self.dropout(fused_feat))  # [B, H*W, out_dim]
        L = h.size(1)  # 序列长度 L=H*W

        # 生成查询 Q 和键 K
        Q = K = h  # [B, H*W, out_dim]

        # 计算注意力分数
        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(
            self.out_dim
        )  # [B, H*W, H*W]

        # 归一化注意力权重
        att = self.softmax(scores)  # [B, H*W, H*W]

        # 加权求和（注意力权重 × 特征）
        weighted_h = torch.bmm(att, h)  # [B, H*W, out_dim]

        # 残差连接 + 输出变换
        enhanced_feat = weighted_h + h
        enhanced_feat = self.elu(enhanced_feat)
        enhanced_feat = self.dropout(enhanced_feat)

        enhanced_feat = self.Wo(enhanced_feat)  # [batch, H*W, out_dim]

        return enhanced_feat  # [batch, H*W, out_dim]


class TextGuidedCrossAttention(nn.Module):
    def __init__(
        self, text_dim=768, img_dim=768, hidden_dim=768, num_heads=8, dropout=0.2
    ):
        super().__init__()
        self.text_dim = text_dim 
        self.img_dim = img_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.img_proj = nn.Linear(img_dim, hidden_dim)

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_feat, img_feat):
        """
        正确的前向传播：文本引导图像的注意力。
        Args:
            text_feat (Tensor): 文本特征，形状 [B, T, D_text]
            img_feat (Tensor): 图像特征，形状 [B, S, D_img]
        Returns:
            enhanced_feat (Tensor): 增强后的图像特征，形状 [B, S, H]
        """
        # 维度映射：H=hidden_dim
        text_hidden = self.text_proj(text_feat)  # [B, T, H]
        img_hidden = self.img_proj(img_feat)  # [B, S, H]

        # multihead_attn的输出 enhanced_feat 的序列长度将与 key/value (文本) 的序列长度 T 相同。
        # 它的输出是增强后的文本特征，形状是 [B, T, H]
        enhanced_feat, _ = self.multihead_attn(
            query=img_hidden,  # Query: [B, S, H] <- query应该由被增强的模态提供
            key=text_hidden,  # Key:   [B, T, H] <- key/value由引导的模态提供
            value=text_hidden,  # Value: [B, T, H]
        )
        # enhanced_feat 的形状是 [B, S, H]
        # 残差连接： enhanced_feat 与原始的 img_hidden 相加
        enhanced_feat = enhanced_feat + img_hidden
        enhanced_feat = self.layer_norm(enhanced_feat)
        enhanced_feat = self.dropout(enhanced_feat)

        return enhanced_feat


class Fusion(nn.Module):
    def __init__(self, in_dim=768):
        super().__init__()
        self.w = nn.Linear(2 * in_dim, in_dim)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, v1, v3):
        v1 = v1.mean(dim=1)
        v3 = v3.mean(dim=1)
        h_cat = torch.cat([v1, v3], dim=1)
        h = self.w(h_cat)
        h = self.elu(h)
        h = self.dropout(h)
        return h


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, num_answers):
        super().__init__()
        self.hidden_layers = nn.ModuleList()
        prev_dim = in_dim

        # 动态构建带残差连接的隐藏层
        for dim in hidden_dims:
            # 主路径：线性层 -> GELU -> BatchNorm -> Dropout
            main = nn.Sequential(
                nn.Linear(prev_dim, dim),
                nn.GELU(),
                nn.BatchNorm1d(dim),
                nn.Dropout(0.3),
            )

            # 快捷连接维度匹配
            shortcut = nn.Identity()
            if prev_dim != dim:
                shortcut = nn.Linear(prev_dim, dim)

            # 将主路径和快捷连接打包为残差块
            self.hidden_layers.append(nn.ModuleList([main, shortcut]))
            prev_dim = dim
        self.final_dropout = nn.Dropout(0.5)
        # 输出层（无残差连接）
        self.output_layer = nn.Linear(prev_dim, num_answers)

    def forward(self, x):
        # 逐层应用残差连接
        for main, shortcut in self.hidden_layers:
            x = main(x) + shortcut(x)  # 残差相加
        x = self.final_dropout(x)
        return self.output_layer(x)


class VariableFactorPlateauScheduler(ReduceLROnPlateau):
    """
    一个直接继承自 ReduceLROnPlateau 的自定义调度器。
    它增加了在每次学习率降低后，动态更新下一次降低因子(factor)的功能。
    """

    def __init__(
        self, optimizer, factors, mode="max", patience=5, min_lr=1e-7, **kwargs
    ):
        """
        Args:
            optimizer: 优化器。
            factors (list): 一个包含每次降低时要使用的因子的列表, e.g., [0.5, 0.1, 0.2]。
            mode, patience, etc.: ReduceLROnPlateau 的标准参数。
        """
        if not factors:
            raise ValueError("factors 列表不能为空。")

        self.factors = factors
        self.factor_index = 0

        # ✨ 关键修改1: 调用父类(ReduceLROnPlateau)的 __init__ 方法
        # 使用因子列表中的第一个因子进行初始化
        super().__init__(
            optimizer,
            factor=self.factors[0],
            mode=mode,
            patience=patience,
            min_lr=min_lr,
            **kwargs,
        )

    # ✨ 关键修改2: 重写 step 方法
    def step(self, metrics, epoch=None):
        # 在执行父类的 step 之前，记录当前的学习率
        # PyTorch 1.x 和 2.x 的 `param_groups` 行为略有不同，用 `_get_lr()` 更安全
        old_lr = self._get_lr()[0]

        # 调用父类 ReduceLROnPlateau 的原始 step 方法
        # 它会根据 metrics 判断是否需要降低学习率，并实际执行降低操作
        super().step(metrics, epoch)

        # 在执行之后，获取新的学习率
        new_lr = self._get_lr()[0]

        # 检查学习率是否真的发生了变化
        if new_lr < old_lr:
            print(f"\n学习率已从 {old_lr:.7f} 降低到 {new_lr:.7f}。")
            # 如果降低了，我们就为下一次降低做准备
            self.factor_index += 1

            # 获取下一个要使用的因子
            # 如果因子列表用完了，就一直使用最后一个
            next_factor_index = min(self.factor_index, len(self.factors) - 1)
            next_factor = self.factors[next_factor_index]

            # 更新下一次将要使用的 factor 属性
            # 注意：这里我们修改的是父类的 self.factor 属性
            self.factor = next_factor
            print(f"下一次学习率降低的因子已更新为: {next_factor}\n")

    # _get_lr() 是 ReduceLROnPlateau 的一个内部辅助方法，我们直接使用它
    def _get_lr(self):
        return [param_group["lr"] for param_group in self.optimizer.param_groups]


class VQAModel(pl.LightningModule):
    def __init__(
        self,
        batch_size=None,
        lr=None,
        number_outputs=None,
        backbone_name="resnet152",
        # shared_encoder=RobertaModel.from_pretrained("roberta-base"),
        shared_encoder=RobertaModel.from_pretrained("./offline_models/roberta-base"),
    ):
        super(VQAModel, self).__init__()
        self.img_encode = ImageEncoder()
        num_answers = number_outputs
        # 2. 将共享模型传递给 TextEncoder
        self.ques_encode = TextEncoder(shared_encoder=shared_encoder)
        self.inattv1 = ImageSelfAttention(768, 768, 64, 64)
        # self.inattv2=ImageSelfAttention(768,768,64,32)
        self.inattv3 = ImageSelfAttention(768, 768, 64, 16)
        self.inattq1 = TextInAttention()
        # self.inattq2=TextInAttention()
        self.inattq3 = TextInAttention()
        self.cross_attn1 = TextGuidedCrossAttention()
        # self.cross_attn2 = TextGuidedCrossAttention()
        self.cross_attn3 = TextGuidedCrossAttention()
        # self.cross_attn1_ = ImageGuidedCrossAttention()
        # self.cross_attn3_ = ImageGuidedCrossAttention()
        self.fus = Fusion()
        # self.fus2 = Fusion()
        # self.mlp = MLP(768+768, 512,num_answers=num_answers)
        # self.mlp = MLP(768 + 768, [1280, 1024, 512], num_answers=num_answers)
        self.mlp = MLP(768, [512], num_answers=num_answers)
        self.save_hyperparameters(ignore=["shared_encoder"])
        self.number_outputs = number_outputs
        # self.loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.loss = F.cross_entropy
        self.lr = lr
        self.batch_size = batch_size
        # 准确率指标
        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=number_outputs
        )
        self.valid_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=number_outputs
        )

        # 用于手动记录每个epoch的损失和准确率
        self.all_train_losses = []
        self.all_valid_losses = []
        self.all_train_accs = []
        self.all_valid_accs = []
        self.all_learning_rates = []

        # 用于存储每个step的损失，以便在epoch结束时计算平均值
        # self.step_train_losses = []
        # self.step_valid_losses = []

        # 用于你的原始分类别准确率统计
        self.validation_step_outputs = []
        self.backbone_name = backbone_name

    def forward(self, img, ques_id, ques_seg, ques_mask):
        img1, img3, global_img = self.img_encode(img)
        ques1, ques3 = self.ques_encode(ques_id, ques_seg, ques_mask)
        v1 = self.inattv1(img1)
        # v2=self.inattv2(img2)
        v3 = self.inattv3(img3)
        q1 = self.inattq1(ques1)
        # q2=self.inattq2(ques2)
        q3 = self.inattq3(ques3)
        v1_enhance = self.cross_attn1(q1, v1)
        # v2_enhance= self.cross_attn1(q2, v2)
        v3_enhance = self.cross_attn3(q3, v3)
        # v1_enhance_ = self.cross_attn1_(v1, q1)
        # v3_enhance_ = self.cross_attn3_(v3, q3)
        # h1 = self.fus1(v1_enhance, v1_enhance_)
        # h2 = self.fus2(v3_enhance, v3_enhance_)
        h = self.fus(v1_enhance, v3_enhance)
        h = h + global_img
        # h_combined = torch.cat(
        #     [h, global_img], dim=1
        # )  # shape: (batch_size, in_dim + in_dim + img_dim)
        # ans = self.mlp(h_combined)
        ans = self.mlp(h)
        return ans

    def configure_optimizers(self):
        """
        配置优化器和学习率调度器。
        使用 AdamW 优化器和 ReduceLROnPlateau 学习率调度器。
        该调度器会监控验证集损失 (valid_loss)，当损失在3个epoch内不再下降时，
        会自动将学习率乘以0.1。
        """
        # 1. 定义不同参数组的学习率
        # 这里的 self.lr 可以通过学习率查找器获得，例如 3e-4
        initial_lr = self.lr
        new_module_lr = initial_lr
        backbone_lr = initial_lr / 10
        low_lr = initial_lr / 100
        # 2. 创建一个包含特定学习率的参数字典列表
        optimizer_params = [
            # 浅层网络：使用较低的学习率
            # {"params": self.img_encode.conv1.parameters(), "lr": new_module_lr},
            # {"params": self.img_encode.bn1.parameters(), "lr": new_module_lr},
            # {"params": self.img_encode.relu.parameters(), "lr": new_module_lr},
            # {"params": self.img_encode.maxpool.parameters(), "lr": new_module_lr},
            # {'params': self.img_encode.layer1.parameters(), 'lr': self.lr},
            # {'params': self.img_encode.layer2.parameters(), 'lr': self.lr},
            # 深层网络：使用中等的学习率
            {"params": self.img_encode.layer3.parameters(), "lr": new_module_lr},
            {"params": self.img_encode.layer4.parameters(), "lr": new_module_lr},
            # 新增模块：使用较高的学习率
            {"params": self.img_encode.cbam3.parameters(), "lr": new_module_lr},
            {"params": self.img_encode.cbam4.parameters(), "lr": new_module_lr},
            {"params": self.inattv1.parameters(), "lr": new_module_lr},
            {"params": self.inattv3.parameters(), "lr": new_module_lr},
            {"params": self.inattq1.parameters(), "lr": new_module_lr},
            {"params": self.inattq3.parameters(), "lr": new_module_lr},
            {"params": self.cross_attn1.parameters(), "lr": new_module_lr},
            {"params": self.cross_attn3.parameters(), "lr": new_module_lr},
            {"params": self.fus.parameters(), "lr": new_module_lr},
            {"params": self.mlp.parameters(), "lr": new_module_lr},
        ]
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.lr,
            weight_decay=1e-4,
        )

        # --- 【核心修改区域】 ---
        # 定义您想要的降低因子顺序
        # 第一次降低 * 0.5，第二次 * 0.1, 第三次 * 0.2
        # 如果之后还有降低，将一直使用最后一个因子 (0.2)
        lr_factors = [0.5]

        # 使用我们自定义的 VariableFactorPlateauScheduler
        scheduler = VariableFactorPlateauScheduler(
            optimizer,
            factors=lr_factors,  # 传入因子列表
            mode="max",  # 监控指标越大越好
            patience=5,  # 5个epoch不提升则触发
            min_lr=1e-7,
        )
        # --- 【修改结束】 ---

        # 返回PyTorch Lightning要求的特定格式字典
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "valid_OA",  # 监控 "valid_AA" 指标
                "interval": "epoch",
                "frequency": 1,
            },
        }

    # def configure_optimizers(self):
    #     """
    #     配置优化器和学习率调度器。
    #     使用 AdamW 优化器和 CosineAnnealingLR 学习率调度器。
    #     """
    #     optimizer = torch.optim.AdamW(
    #         filter(lambda p: p.requires_grad, self.parameters()),
    #         lr=self.lr,
    #         weight_decay=1e-4,
    #     )

    #     # --- 关键修改在这里 ---
    #     # T_max 不再直接等于 trainer.max_epochs，
    #     # 而是我们根据上次实验观察到的、一个预估的实际收敛轮数。
    #     # 根据您上次在第25轮停止的经验，设置一个25-30之间的值是非常合理的。
    #     estimated_convergence_epochs = 25

    #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #         optimizer,
    #         T_max=estimated_convergence_epochs,  # <--- 使用预估值
    #         eta_min=1e-7,
    #     )

    #     return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    # # 1. 为不同的参数组定义基础学习率
    # # 你可以根据实验结果调整这些值。
    # base_lr = self.lr # 这个值根据你之前的描述应该是 1e-5
    # deep_layer_lr = 1e-4  # 用于 layer3 和 layer4
    # shallow_layer_lr = 1e-5 # 用于 layer1 和 layer2
    # new_module_lr = 1e-3 # 用于所有新增的模块，如 inattv, inattq, cross_attn, fus, mlp

    # 2. 创建一个包含特定学习率的参数字典列表
    # optimizer_params = [
    #     # 浅层网络：使用较低的学习率
    #     {'params': self.img_encode.conv1.parameters(), 'lr': self.lr},
    #     {'params': self.img_encode.bn1.parameters(), 'lr': self.lr},
    #     {'params': self.img_encode.relu.parameters(), 'lr': self.lr},
    #     {'params': self.img_encode.maxpool.parameters(), 'lr': self.lr},
    #     {'params': self.img_encode.layer1.parameters(), 'lr': self.lr},
    #     {'params': self.img_encode.layer2.parameters(), 'lr': self.lr},
    #
    #     # 深层网络：使用中等的学习率
    #     {'params': self.img_encode.layer3.parameters(), 'lr': self.lr},
    #     {'params': self.img_encode.layer4.parameters(), 'lr': self.lr},
    #
    #     # 新增模块：使用较高的学习率
    #     {'params': self.inattv1.parameters(), 'lr': self.lr},
    #     {'params': self.inattv3.parameters(), 'lr': self.lr},
    #     {'params': self.inattq1.parameters(), 'lr': self.lr},
    #     {'params': self.inattq3.parameters(), 'lr': self.lr},
    #     {'params': self.cross_attn1.parameters(), 'lr': self.lr},
    #     {'params': self.cross_attn3.parameters(), 'lr': self.lr},
    #     {'params': self.fus.parameters(), 'lr': self.lr},
    #     {'params': self.mlp.parameters(), 'lr': self.lr},
    # ]

    # # 3. 使用新的参数组创建优化器
    # # 这样就不需要再使用 filter(lambda p: p.requires_grad, self.parameters()) 来筛选了
    # optimizer = torch.optim.Adam(optimizer_params)
    #
    # # 4. 创建学习率调度器
    # # 调度器现在会按照你的 rule 函数，对每个参数组的基础学习率进行相应的衰减。
    # scheduler = LambdaLR(optimizer, lr_lambda=rule)
    #
    # return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def training_step(self, batch, batch_idx):
        pixel_values, input_ids, token_type_ids, attention_mask, answer = batch
        pred = self(pixel_values, input_ids, token_type_ids, attention_mask)
        self.train_acc(pred, answer)
        train_loss = self.loss(pred, answer)
        self.log("train_loss", train_loss, on_step=False, on_epoch=True, prog_bar=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            (
                pixel_values,
                input_ids,
                token_type_ids,
                attention_mask,
                answer,
                question_type,
                img_id,
                question,
                answer_str,
            ) = batch
            pred = self(pixel_values, input_ids, token_type_ids, attention_mask)

            self.valid_acc(pred, answer)
            valid_loss = self.loss(pred, answer)
            # self.step_valid_losses.append(valid_loss.item())  # 手动记录损失

            # PyTorch Lightning 的回调需要 self.log()
            self.log("valid_loss", valid_loss, on_epoch=True, prog_bar=True)
            self.log("valid_acc", self.valid_acc, on_epoch=True, prog_bar=True)

            pred_arg = torch.argmax(pred, axis=1)
            for i in range(pred.shape[0]):
                if pred_arg[i] == answer[i]:
                    self.validation_step_outputs.append([1, question_type[i]])
                else:
                    self.validation_step_outputs.append([0, question_type[i]])

    def on_validation_epoch_end(self):
        # --- 新增逻辑：统一收集用于最终绘图的数据 ---
        # 此时，整个epoch的训练和验证都已结束，所有指标都已在callback_metrics中最终确定

        # 1. 收集本轮的平均训练损失和验证损失
        train_loss = self.trainer.callback_metrics.get("train_loss", 0.0)
        valid_loss = self.trainer.callback_metrics.get("valid_loss", 0.0)
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]

        self.all_train_losses.append(
            train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss
        )
        self.all_valid_losses.append(
            valid_loss.item() if isinstance(valid_loss, torch.Tensor) else valid_loss
        )

        # 2. 收集本轮的平均训练准确率和验证准确率 (与您之前的逻辑类似)
        # 注意：训练准确率需要从 self.all_train_accs 获取，因为它在 on_training_epoch_end 中计算
        # 为了逻辑统一，我们也可以在这里重新计算
        train_acc = self.train_acc.compute()  # 确保在 on_training_epoch_end 重置前获取
        valid_acc = self.valid_acc.compute()
        self.all_train_accs.append(train_acc.item())
        self.all_valid_accs.append(valid_acc.item())
        self.all_learning_rates.append(current_lr)

        # 3. 在所有计算和存储完成后，统一重置所有指标，为下一轮做准备
        self.train_acc.reset()
        self.valid_acc.reset()

        # 2. 计算各个任务的准确率（保留你的原始逻辑）
        if len(self.validation_step_outputs) > 0:
            outputs = np.stack(self.validation_step_outputs)
            total_rural_urban, total_presence, total_count, total_comp = 0, 0, 0, 0
            right_rural_urban, right_presence, right_count, right_comp = 0, 0, 0, 0
            for i in range(outputs.shape[0]):
                if outputs[i][1] == "comp":
                    total_comp += 1
                    if outputs[i][0] == "1":
                        right_comp += 1
                elif outputs[i][1] == "presence":
                    total_presence += 1
                    if outputs[i][0] == "1":
                        right_presence += 1
                elif outputs[i][1] == "count":
                    total_count += 1
                    if outputs[i][0] == "1":
                        right_count += 1
                else:
                    total_rural_urban += 1
                    if outputs[i][0] == "1":
                        right_rural_urban += 1

            acc_rural_urban = (
                right_rural_urban / total_rural_urban if total_rural_urban > 0 else 0
            )
            acc_presence = right_presence / total_presence if total_presence > 0 else 0
            acc_count = right_count / total_count if total_count > 0 else 0
            acc_comp = right_comp / total_comp if total_comp > 0 else 0
            right = right_rural_urban + right_presence + right_count + right_comp
            total = total_rural_urban + total_presence + total_count + total_comp
            AA = (acc_rural_urban + acc_presence + acc_count + acc_comp) / 4
            OA = right / total
            self.log("valid_AA", AA, prog_bar=True)
            self.log("valid_OA", AA, prog_bar=True)
        else:
            acc_rural_urban, acc_presence, acc_count, acc_comp = 0, 0, 0, 0
            AA, OA = 0, 0

        # 3. 打印所有轮次的历史记录和本轮详细指标
        print("\n--- All Epochs Summary ---")
        for i, (train_l, valid_l, train_a, valid_a, lr) in enumerate(
            zip(
                self.all_train_losses,
                self.all_valid_losses,
                self.all_train_accs,
                self.all_valid_accs,
                self.all_learning_rates,
            )
        ):
            print(
                f"Epoch {i}: Train Loss={train_l:.4f}, Valid Loss={valid_l:.4f}, Train Acc={train_a:.4f}, Valid Acc={valid_a:.4f}, LR={lr:.8f}"
            )
        print("--------------------------\n")

        print(f"Current Epoch {self.current_epoch} Task Accuracies:")
        print(f"  - OA: {OA:.4f}")
        print(f"  - AA: {AA:.4f}")
        print(
            f"  - R/U: {acc_rural_urban:.4f}, Presence: {acc_presence:.4f}, Count: {acc_count:.4f}, Comp: {acc_comp:.4f}, Train Loss={train_loss:.4f}, Valid Loss={valid_loss:.4f}"
        )

        # 4. 保存当前最好的模型（根据AA指标）
        if not hasattr(self, "best_valid_OA"):
            self.best_valid_OA = 0.0
        if OA > self.best_valid_OA:
            self.best_valid_OA = OA
            self.best_valid_AA = AA
            self.best_acc_rural_urban = acc_rural_urban
            self.best_acc_presence = acc_presence
            self.best_acc_count = acc_count
            self.best_acc_comp = acc_comp
            torch.save(self.state_dict(), "best_model_lr_wt_q.ckpt")
            print("✨ New best OA model saved!")
            self.print(
                f"[Best OA Updated] OA: {OA:.4f}, AA: {AA:.4f}, "
                f"R/U: {acc_rural_urban:.4f}, Presence: {acc_presence:.4f}, "
                f"Count: {acc_count:.4f}, Comp: {acc_comp:.4f}"
            )
        # # 4. 保存当前最好的模型（根据AA指标）
        # if not hasattr(self, "best_valid_Loss"):
        #     self.best_valid_Loss = 0.35
        # if valid_loss < self.best_valid_Loss:
        #     self.best_valid_OA_LOSS = OA
        #     self.best_valid_AA_LOSS = AA
        #     self.best_acc_rural_urban_LOSS = acc_rural_urban
        #     self.best_acc_presence_LOSS = acc_presence
        #     self.best_acc_count_LOSS = acc_count
        #     self.best_acc_comp_LOSS = acc_comp
        #     #self.best_valid_Loss = valid_loss
        #     torch.save(self.state_dict(), "best_model_lr_loss_wt5.ckpt")
        #     print("✨ New best LOSS model saved!")
        #     self.print(
        #         f"[Best AA Updated] OA: {OA:.4f}, AA: {AA:.4f}, "
        #         f"R/U: {acc_rural_urban:.4f}, Presence: {acc_presence:.4f}, "
        #         f"Count: {acc_count:.4f}, Comp: {acc_comp:.4f}"
        #     )
        # # 5. 打印全程最佳结果
        self.print(
            f"[Summary] Best OA: {self.best_valid_OA:.4f}, "
            f"AA: {self.best_valid_AA:.4f}, "
            f"R/U: {self.best_acc_rural_urban:.4f}, "
            f"Presence: {self.best_acc_presence:.4f}, "
            f"Count: {self.best_acc_count:.4f}, "
            f"Comp: {self.best_acc_comp:.4f}, "
            f"Model Path: best_model_lr_wt_q.ckpt"
        )
        # # 5. 打印全程最佳结果
        # self.print(
        #     f"[Summary] Best OA_LOSS: {self.best_valid_OA_LOSS:.4f}, "
        #     f"AA_LOSS: {self.best_valid_AA_LOSS:.4f}, "
        #     f"R/U_LOSS: {self.best_acc_rural_urban_LOSS:.4f}, "
        #     f"Presence_LOSS: {self.best_acc_presence_LOSS:.4f}, "
        #     f"Count_LOSS: {self.best_acc_count_LOSS:.4f}, "
        #     f"Comp_LOSS: {self.best_acc_comp_LOSS:.4f}, "
        #     f"LOSS: {self.best_valid_Loss:.4f}, "
        #     f"Model Path: best_model_lr_loss_wt5.ckpt"
        # )
        self.validation_step_outputs.clear()

    def on_fit_end(self):
        """
        在整个训练过程结束时被调用。
        绘制并保存训练损失和验证损失的变化曲线。
        """
        print("\n--- 绘制并保存损失曲线 ---")

        # 绘制训练和验证损失曲线
        plt.figure(figsize=(10, 6))
        plt.plot(self.all_train_losses, label="Training Loss")
        plt.plot(self.all_valid_losses, label="Validation Loss")
        plt.title("Training and Validation Loss Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

        # 创建保存图片的目录
        plot_dir = "loss_plots_lr_wt_q"
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        # 保存图片到指定目录
        plot_path = os.path.join(plot_dir, "loss_curve_q.png")
        plt.savefig(plot_path)
        plt.close()  # 关闭绘图窗口以释放内存

        print(f"✅ 损失曲线已保存到: {plot_path}")
