"""HA-RUnet: Hybrid Attention-based Residual 3D U-Net for brain tumor segmentation.

Encoder/decoder built from residual blocks with Squeeze-and-Excitation (channel
attention); skip connections pass through additive attention gates (spatial
attention) before being concatenated in the decoder.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock3D(nn.Module):
    """Squeeze-and-Excitation: re-weights channels using global context."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(self.pool(x))


class ResidualSEBlock3D(nn.Module):
    """Two 3x3x3 conv layers + SE, with a projected identity shortcut."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
        )
        self.se = SEBlock3D(out_ch)
        self.shortcut = (
            nn.Identity()
            if in_ch == out_ch and stride == 1
            else nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.InstanceNorm3d(out_ch, affine=True),
            )
        )
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        return self.act(self.se(self.body(x)) + self.shortcut(x))


class AttentionGate3D(nn.Module):
    """Additive attention gate (Oktay et al.) applied to encoder skip features."""

    def __init__(self, skip_ch, gate_ch, inter_ch):
        super().__init__()
        self.w_skip = nn.Conv3d(skip_ch, inter_ch, 1, bias=False)
        self.w_gate = nn.Conv3d(gate_ch, inter_ch, 1, bias=True)
        self.psi = nn.Sequential(nn.Conv3d(inter_ch, 1, 1), nn.Sigmoid())
        self.act = nn.ReLU(inplace=True)

    def forward(self, skip, gate):
        g = F.interpolate(self.w_gate(gate), size=skip.shape[2:], mode="trilinear", align_corners=False)
        alpha = self.psi(self.act(self.w_skip(skip) + g))
        return skip * alpha


class UpBlock3D(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2)
        self.gate = AttentionGate3D(skip_ch, in_ch, max(skip_ch // 2, 4))
        self.block = ResidualSEBlock3D(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        skip = self.gate(skip, x)
        x = self.up(x)
        return self.block(torch.cat([x, skip], dim=1))


class HARUNet(nn.Module):
    """
    Args:
        in_channels: MRI modalities (BraTS: T1, T1ce, T2, FLAIR -> 4).
        num_classes: output channels. For BraTS we predict the three
            overlapping regions WT / TC / ET with a sigmoid (3 channels).
        base: width of the first stage; later stages double it.
    """

    def __init__(self, in_channels=4, num_classes=3, base=16, depth=4):
        super().__init__()
        widths = [base * 2**i for i in range(depth + 1)]
        self.stem = ResidualSEBlock3D(in_channels, widths[0])
        self.down = nn.ModuleList(
            ResidualSEBlock3D(widths[i], widths[i + 1], stride=2) for i in range(depth)
        )
        self.up = nn.ModuleList(
            UpBlock3D(widths[i + 1], widths[i], widths[i]) for i in reversed(range(depth))
        )
        self.head = nn.Conv3d(widths[0], num_classes, 1)

    def forward(self, x):
        skips = [self.stem(x)]
        for layer in self.down:
            skips.append(layer(skips[-1]))
        x = skips.pop()
        for layer in self.up:
            x = layer(x, skips.pop())
        return self.head(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    net = HARUNet()
    y = net(torch.randn(1, 4, 64, 64, 64))
    print("output:", tuple(y.shape), "| parameters:", f"{count_parameters(net):,}")
