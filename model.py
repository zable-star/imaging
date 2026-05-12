from __future__ import annotations

import torch
import torch.nn as nn


class SliceEncoder(nn.Module):
    def __init__(self, out_dim: int = 128) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(64, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x).flatten(1)
        return self.proj(feat)


class SliceAttentionClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        feature_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = SliceEncoder(out_dim=feature_dim)
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor):
        # x: [B, S, 1, H, W]
        bsz, num_slices, channels, height, width = x.shape
        flat = x.view(bsz * num_slices, channels, height, width)
        slice_features = self.encoder(flat).view(bsz, num_slices, -1)

        attn_logits = self.attention(slice_features).squeeze(-1)
        attn_weights = torch.softmax(attn_logits, dim=1)
        fused = torch.sum(slice_features * attn_weights.unsqueeze(-1), dim=1)

        logits = self.classifier(fused)
        return logits, attn_weights, slice_features

