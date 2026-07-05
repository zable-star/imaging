from __future__ import annotations

import torch
import torch.nn as nn


FUSION_MODES = ["attention", "mean", "concat", "attention_residual"]


class SliceEncoder(nn.Module):
    """Shared CNN encoder for one gated slice."""

    def __init__(self, in_channels: int = 1, feature_dim: int = 128) -> None:
        super().__init__()
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
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(self.features(x))


class SliceAttentionClassifier(nn.Module):
    """Classify an object from one or more gated slices.

    The input shape is [B, S, C, H, W], where S is the number of slices.
    fusion_mode controls how per-slice features are combined:
    - attention: learned softmax weights over slices.
    - mean: uniform average of slice features.
    - concat: concatenate all slice features before classification.
    - attention_residual: attention fusion plus a projected concat residual.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        feature_dim: int = 128,
        attention_hidden_dim: int = 64,
        dropout: float = 0.25,
        fusion_mode: str = "attention",
        num_slices: int | None = None,
    ) -> None:
        super().__init__()
        if fusion_mode not in FUSION_MODES:
            raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")
        if fusion_mode in {"concat", "attention_residual"} and num_slices is None:
            raise ValueError(f"num_slices is required when fusion_mode='{fusion_mode}'")

        self.fusion_mode = fusion_mode
        self.num_slices = num_slices
        self.feature_dim = feature_dim
        self.encoder = SliceEncoder(in_channels=in_channels, feature_dim=feature_dim)
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, attention_hidden_dim),
            nn.Tanh(),
            nn.Linear(attention_hidden_dim, 1),
        )
        self.residual_projection = (
            nn.Sequential(
                nn.LayerNorm(feature_dim * num_slices),
                nn.Linear(feature_dim * num_slices, feature_dim),
                nn.ReLU(inplace=True),
            )
            if fusion_mode == "attention_residual"
            else None
        )

        classifier_input_dim = feature_dim * num_slices if fusion_mode == "concat" else feature_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_input_dim),
            nn.Dropout(dropout),
            nn.Linear(classifier_input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 5:
            raise ValueError(f"Expected input with shape [B, S, C, H, W], got {tuple(x.shape)}")

        batch_size, num_slices, channels, height, width = x.shape
        flat = x.view(batch_size * num_slices, channels, height, width)
        slice_features = self.encoder(flat).view(batch_size, num_slices, -1)

        if self.fusion_mode in {"attention", "attention_residual"}:
            attention_logits = self.attention(slice_features).squeeze(-1)
            attention_weights = torch.softmax(attention_logits, dim=1)
            fused = torch.sum(slice_features * attention_weights.unsqueeze(-1), dim=1)
            if self.fusion_mode == "attention_residual":
                if self.num_slices is not None and num_slices != self.num_slices:
                    raise ValueError(f"Expected {self.num_slices} slices for attention_residual fusion, got {num_slices}")
                residual = self.residual_projection(slice_features.reshape(batch_size, num_slices * self.feature_dim))
                fused = fused + residual
        elif self.fusion_mode == "mean":
            attention_weights = self._uniform_weights(batch_size, num_slices, slice_features)
            fused = slice_features.mean(dim=1)
        else:
            if self.num_slices is not None and num_slices != self.num_slices:
                raise ValueError(f"Expected {self.num_slices} slices for concat fusion, got {num_slices}")
            attention_weights = self._uniform_weights(batch_size, num_slices, slice_features)
            fused = slice_features.reshape(batch_size, num_slices * self.feature_dim)

        logits = self.classifier(fused)
        return logits, attention_weights, slice_features

    @staticmethod
    def _uniform_weights(batch_size: int, num_slices: int, reference: torch.Tensor) -> torch.Tensor:
        return torch.full(
            (batch_size, num_slices),
            fill_value=1.0 / num_slices,
            dtype=reference.dtype,
            device=reference.device,
        )
