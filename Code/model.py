"""HA-RUnet: Hybrid Attention-based Residual 3D U-Net for brain tumor segmentation (BraTS-2020).

Architecture (input 4 x 128^3: FLAIR, T1, T2, T1ce):

    encoder   3D conv block 128^3x32 -> [2x2x2 max-pool + residual block] 64^3x64 -> 32^3x128 -> 16^3x256
              -> 8^3x512 -> bottleneck 4^3x512 (two residual blocks) + SE
    skips     level 128^3 is a plain skip connection; levels 64^3 ... 8^3 pass through
              Attention Modules 1-4 (residual attention: trunk branch + encoder-decoder soft-mask branch)
    decoder   2x2x2 up-sampling, concatenation with the attended skip, residual block, Squeeze-Excitation
    output    1x1x1 convolution -> 128^3 x num_classes

Residual block: identity mapping + three pre-activation (BN -> ReLU -> Conv) units, bottleneck by default
(1x1x1 -> 3x3x3 -> 1x1x1) to keep the model lightweight.
Attention module: p residual blocks, trunk of t residual blocks, soft mask M in [0, 1];
output = Residual((1 + M) * Trunk)  (attention residual learning).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock3D(nn.Module):
    """Identity mapping + three pre-activation BN -> ReLU -> Conv units.

    bottleneck=True (default, lightweight): the units are 1x1x1 (reduce to out/4) -> 3x3x3 -> 1x1x1 (expand),
    as in pre-activation bottleneck ResNets / Residual Attention Networks.
    bottleneck=False: three full 3x3x3 convolutions at `out_ch` channels.
    """

    def __init__(self, in_ch, out_ch, bottleneck=True):
        super().__init__()
        if bottleneck:
            mid = max(out_ch // 4, 8)
            plan = [(in_ch, mid, 1), (mid, mid, 3), (mid, out_ch, 1)]
        else:
            plan = [(in_ch, out_ch, 3), (out_ch, out_ch, 3), (out_ch, out_ch, 3)]
        layers = []
        for a, b, k in plan:
            layers += [nn.BatchNorm3d(a), nn.ReLU(inplace=True), nn.Conv3d(a, b, k, padding=k // 2, bias=False)]
        self.body = nn.Sequential(*layers)
        self.identity = nn.Identity() if in_ch == out_ch else nn.Conv3d(in_ch, out_ch, 1, bias=False)

    def forward(self, x):
        return self.body(x) + self.identity(x)


class SqueezeExcitation3D(nn.Module):
    """Global average pooling -> FC -> ReLU -> FC -> sigmoid channel weights."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc = nn.Sequential(nn.Linear(channels, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, channels), nn.Sigmoid())

    def forward(self, x):
        w = self.fc(x.mean(dim=(2, 3, 4)))
        return x * w[:, :, None, None, None]


class SoftMaskBranch3D(nn.Module):
    """Encoder-decoder (hourglass) of depth D with skip connections, ending in 1x1x1 convs and a sigmoid."""

    def __init__(self, ch, depth, r=1):
        super().__init__()
        self.depth = depth
        self.down = nn.ModuleList(ResidualBlock3D(ch, ch) for _ in range(depth))
        self.skips = nn.ModuleList(ResidualBlock3D(ch, ch) for _ in range(max(depth - 1, 0)))
        self.middle = nn.Sequential(*[ResidualBlock3D(ch, ch) for _ in range(2 * r)])
        self.up = nn.ModuleList(ResidualBlock3D(ch, ch) for _ in range(depth))
        self.head = nn.Sequential(nn.BatchNorm3d(ch), nn.ReLU(inplace=True), nn.Conv3d(ch, ch, 1, bias=False),
                                  nn.BatchNorm3d(ch), nn.ReLU(inplace=True), nn.Conv3d(ch, ch, 1, bias=False), nn.Sigmoid())

    def forward(self, x):
        size, skips = x.shape[2:], []
        for i in range(self.depth):
            x = self.down[i](F.max_pool3d(x, 2))
            if i < self.depth - 1:
                skips.append(self.skips[i](x))
        x = self.middle(x)
        for i in reversed(range(self.depth)):
            target = skips[i - 1].shape[2:] if i > 0 else size
            x = F.interpolate(x, size=target, mode="trilinear", align_corners=False)
            if i > 0:
                x = x + skips[i - 1]
            x = self.up[i](x)
        return self.head(x)


class AttentionModule3D(nn.Module):
    def __init__(self, ch, depth, p=1, t=2, r=1):
        super().__init__()
        self.pre = nn.Sequential(*[ResidualBlock3D(ch, ch) for _ in range(p)])
        self.trunk = nn.Sequential(*[ResidualBlock3D(ch, ch) for _ in range(t)])
        self.mask = SoftMaskBranch3D(ch, depth, r)
        self.post = nn.Sequential(*[ResidualBlock3D(ch, ch) for _ in range(p)])

    def forward(self, x):
        x = self.pre(x)
        return self.post((1 + self.mask(x)) * self.trunk(x))


class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.block(x)


class HARUNet(nn.Module):
    """
    Args:
        in_channels: MRI modalities (4).
        num_classes: 3 overlapping regions WT / TC / ET with sigmoid (default), or 4 classes with softmax.
        widths: channels per level; the paper uses (32, 64, 128, 256, 512).
        mask_depths: soft-mask encoder-decoder depth D for Attention Modules 1-4 (deeper at higher resolution).
        attention / se: switch the modules off to reproduce the ablation
            (Residual U-Net -> + Attention -> + Attention + SE).
    """

    def __init__(self, in_channels=4, num_classes=3, widths=(32, 64, 128, 256, 512), mask_depths=(3, 2, 1, 1),
                 attention=True, se=True):
        super().__init__()
        self.stem = ConvBlock3D(in_channels, widths[0])
        self.encoder = nn.ModuleList(ResidualBlock3D(widths[i], widths[i + 1]) for i in range(len(widths) - 1))
        self.bottleneck = nn.Sequential(ResidualBlock3D(widths[-1], widths[-1]), ResidualBlock3D(widths[-1], widths[-1]))
        self.bottleneck_se = SqueezeExcitation3D(widths[-1]) if se else nn.Identity()
        self.attention = nn.ModuleList(
            AttentionModule3D(widths[i + 1], mask_depths[i]) if attention else nn.Identity()
            for i in range(len(widths) - 1))
        self.up = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.se = nn.ModuleList()
        # decoder levels from 8^3 (widths[-1]) up to 64^3 (widths[1]), then the 128^3 conv block
        for i in reversed(range(1, len(widths))):
            in_ch = widths[-1] if i == len(widths) - 1 else widths[i + 1]
            self.up.append(nn.ConvTranspose3d(in_ch, widths[i], 2, stride=2))
            self.decoder.append(ResidualBlock3D(2 * widths[i], widths[i]))
            self.se.append(SqueezeExcitation3D(widths[i]) if se else nn.Identity())
        self.up_top = nn.ConvTranspose3d(widths[1], widths[0], 2, stride=2)
        self.decoder_top = ConvBlock3D(2 * widths[0], widths[0])
        self.head = nn.Conv3d(widths[0], num_classes, 1)

    def forward(self, x):
        top = self.stem(x)
        feats, h = [], top
        for enc in self.encoder:
            h = enc(F.max_pool3d(h, 2))
            feats.append(h)  # 64^3x64, 32^3x128, 16^3x256, 8^3x512
        h = self.bottleneck_se(self.bottleneck(F.max_pool3d(h, 2)))  # 4^3x512
        for k, (up, dec, se) in enumerate(zip(self.up, self.decoder, self.se)):
            level = len(feats) - 1 - k
            skip = self.attention[level](feats[level])
            h = se(dec(torch.cat([up(h), skip], dim=1)))
        h = self.decoder_top(torch.cat([self.up_top(h), top], dim=1))
        return self.head(h)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    for name, kw in [("Residual U-Net", dict(attention=False, se=False)),
                     ("+ Attention", dict(se=False)), ("HA-RUnet (+ Attention + SE)", {})]:
        print(f"{name:30s} {count_parameters(HARUNet(**kw)):>12,} parameters")
    net = HARUNet(widths=(8, 16, 32, 64, 128))
    y = net(torch.randn(1, 4, 64, 64, 64))
    print("forward check (reduced widths, 64^3 input):", tuple(y.shape))
