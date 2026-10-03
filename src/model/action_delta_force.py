from typing import Optional, Sequence

import torch
import torch.nn as nn

from src.model.unet import DoubleConv, Down


class ActionConditionedDeltaForceNet(nn.Module):
    """
    Deterministic force-dynamics critic.

    The model encodes a wrist-camera observation history with a U-Net-style
    encoder, encodes a candidate action trajectory with an MLP, and predicts a
    force-delta trajectory. It is intended for action reranking, not action
    generation.
    """

    def __init__(
        self,
        image_channels: int,
        action_dim: int,
        pred_horizon: int,
        force_dim: int = 1,
        low_dim_dim: int = 0,
        output_horizon: Optional[int] = None,
        encoder_channels: Sequence[int] = (32, 64, 128, 256),
        force_spatial_size: int = 4,
        action_hidden_dim: int = 256,
        low_dim_hidden_dim: int = 128,
        fusion_hidden_dim: int = 512,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.image_channels = int(image_channels)
        self.action_dim = int(action_dim)
        self.pred_horizon = int(pred_horizon)
        self.force_dim = int(force_dim)
        self.low_dim_dim = int(low_dim_dim)
        self.output_horizon = int(output_horizon or pred_horizon)

        self.use_visual = self.image_channels > 0
        if self.use_visual:
            channels = list(encoder_channels)
            self.in_conv = DoubleConv(self.image_channels, channels[0])
            self.down_blocks = nn.ModuleList(
                [Down(channels[i], channels[i + 1]) for i in range(len(channels) - 1)]
            )
            bottleneck_channels = channels[-1] * 2
            self.bottleneck = DoubleConv(channels[-1], bottleneck_channels)
            self.visual_pool = nn.AdaptiveAvgPool2d(
                (force_spatial_size, force_spatial_size)
            )
            visual_dim = bottleneck_channels * force_spatial_size * force_spatial_size
        else:
            self.in_conv = None
            self.down_blocks = nn.ModuleList()
            self.bottleneck = None
            self.visual_pool = None
            visual_dim = 0

        action_in_dim = self.pred_horizon * self.action_dim
        self.action_encoder = nn.Sequential(
            nn.Linear(action_in_dim, action_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(action_hidden_dim, action_hidden_dim),
            nn.ReLU(inplace=True),
        )

        low_dim_out_dim = 0
        if self.low_dim_dim > 0:
            self.low_dim_encoder = nn.Sequential(
                nn.Linear(self.low_dim_dim, low_dim_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(low_dim_hidden_dim, low_dim_hidden_dim),
                nn.ReLU(inplace=True),
            )
            low_dim_out_dim = low_dim_hidden_dim
        else:
            self.low_dim_encoder = None

        self.head = nn.Sequential(
            nn.Linear(visual_dim + action_hidden_dim + low_dim_out_dim, fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_hidden_dim, self.output_horizon * self.force_dim),
        )

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        if not self.use_visual:
            return image.new_zeros((image.shape[0], 0))
        x = self.in_conv(image)
        for down in self.down_blocks:
            x = down(x)
        x = self.bottleneck(x)
        return self.visual_pool(x).flatten(start_dim=1)

    def forward(
        self,
        image: torch.Tensor,
        action_delta: torch.Tensor,
        low_dim: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            image: (B, C, H, W), usually stacked wrist RGB history.
            action_delta: (B, pred_horizon, action_dim).
            low_dim: optional flattened low-dimensional observation history.

        Returns:
            Predicted force-change target with shape
            (B, output_horizon, force_dim).
        """
        visual = self.encode_image(image)
        action_feat = self.action_encoder(action_delta.flatten(start_dim=1))
        feats = [visual, action_feat]
        if self.low_dim_encoder is not None:
            if low_dim is None:
                raise ValueError("low_dim input is required for this checkpoint")
            feats.append(self.low_dim_encoder(low_dim.flatten(start_dim=1)))
        out = self.head(torch.cat(feats, dim=1))
        return out.view(-1, self.output_horizon, self.force_dim)
