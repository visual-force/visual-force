from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_conv(x)


class UNet(nn.Module):
    """
    UNet encoder with a force-regression head on the bottleneck.

    When encoder_only=True (default for force prediction) the decoder is not
    built, and forward() returns only the force prediction tensor.
    When encoder_only=False the full U-Net decoder is included and forward()
    returns (stress_map, force_pred).  force_dim must be set in both cases.
    """

    def __init__(
        self,
        in_channels: int = 3,
        encoder_channels: Sequence[int] = (32, 64, 128, 256),
        force_dim: int = 1,
        force_hidden_dim: int = 256,
        force_dropout: float = 0.3,
        force_pooling: str = "avg",
        force_spatial_size: int = 4,
        encoder_only: bool = True,
        # kept for backward-compat when encoder_only=False
        stress_out_channels: int = 1,
    ) -> None:
        super().__init__()

        channels = list(encoder_channels)
        self.encoder_only = encoder_only
        self.force_pooling = force_pooling

        self.in_conv     = DoubleConv(in_channels, channels[0])
        self.down_blocks = nn.ModuleList(
            [Down(channels[i], channels[i + 1]) for i in range(len(channels) - 1)]
        )
        self.bottleneck  = DoubleConv(channels[-1], channels[-1] * 2)

        # Force regression head on top of bottleneck
        bottleneck_channels = channels[-1] * 2
        if force_pooling == "avg":
            self.force_pool = nn.AdaptiveAvgPool2d((1, 1))
            force_in_dim = bottleneck_channels
        elif force_pooling == "spatial":
            self.force_pool = nn.AdaptiveAvgPool2d((force_spatial_size, force_spatial_size))
            force_in_dim = bottleneck_channels * force_spatial_size * force_spatial_size
        else:
            raise ValueError(f"Unknown force_pooling: {force_pooling!r}")

        self.force_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(force_in_dim, force_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(force_dropout),
            nn.Linear(force_hidden_dim, force_dim),
        )

        if not encoder_only:
            decoder_spec: List[Tuple[int, int, int]] = []
            in_c = channels[-1] * 2
            for skip_c in reversed(channels):
                decoder_spec.append((in_c, skip_c, skip_c))
                in_c = skip_c
            self.up_blocks   = nn.ModuleList([_Up(ic, sc, oc) for ic, sc, oc in decoder_spec])
            self.stress_head = nn.Conv2d(channels[0], stress_out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        skips: List[torch.Tensor] = []
        x = self.in_conv(x)
        skips.append(x)
        for down in self.down_blocks:
            x = down(x)
            skips.append(x)
        bottleneck = self.bottleneck(x)

        force_pred = self.force_head(self.force_pool(bottleneck))

        if self.encoder_only:
            return force_pred

        x = bottleneck
        for up, skip in zip(self.up_blocks, reversed(skips)):
            x = up(x, skip)
        stress_pred = self.stress_head(x)
        return stress_pred, force_pred


class _Up(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv((in_channels // 2) + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


def build_unet(**kwargs) -> UNet:
    return UNet(**kwargs)
