from __future__ import annotations

import torch
import torch.nn as nn


class SliceEncoder(nn.Module):
    """单张 gated 切片的共享 CNN 编码器。

    gate_0 / gate_1 / gate_2 会复用同一个编码器，这样注意力融合前，
    每张切片提取到的是可比较的特征。
    """

    def __init__(self, in_channels: int = 1, feature_dim: int = 128) -> None:
        super().__init__()
        # 逐步下采样切片图像，保留紧凑的全局形状信息。
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        # 将 CNN 输出映射到注意力模块和分类器需要的特征维度。
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(self.features(x))


class SliceAttentionClassifier(nn.Module):
    """根据同一物体的多张 gated 切片进行分类。

    输入张量形状应为 [B, S, C, H, W]：
    B 表示 batch 大小，S 表示切片数量，C 表示图像通道数。
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        feature_dim: int = 128,
        attention_hidden_dim: int = 64,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.encoder = SliceEncoder(in_channels=in_channels, feature_dim=feature_dim)
        # 为每张切片生成一个分数，再通过 softmax 转换成切片权重。
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, attention_hidden_dim),
            nn.Tanh(),
            nn.Linear(attention_hidden_dim, 1),
        )
        # 分类器接收注意力加权后的整体物体特征。
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 5:
            raise ValueError(f"Expected input with shape [B, S, C, H, W], got {tuple(x.shape)}")

        batch_size, num_slices, channels, height, width = x.shape
        # 将所有切片合并成一个大 batch，一次性送入共享 CNN 编码器。
        flat = x.view(batch_size * num_slices, channels, height, width)
        slice_features = self.encoder(flat).view(batch_size, num_slices, -1)

        # 注意力权重在切片维度上和为 1，可用于观察模型更依赖哪个 gate。
        attention_logits = self.attention(slice_features).squeeze(-1)
        attention_weights = torch.softmax(attention_logits, dim=1)

        # 加权求和，把 S 个切片特征融合成一个物体级特征。
        fused = torch.sum(slice_features * attention_weights.unsqueeze(-1), dim=1)
        logits = self.classifier(fused)
        return logits, attention_weights, slice_features
