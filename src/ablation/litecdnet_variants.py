from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.models import MobileNet_V2_Weights, mobilenet_v2
except ImportError:
    from torchvision.models import mobilenet_v2

    MobileNet_V2_Weights = None


class SEBlock(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(1, in_channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)


class LiteContextModule(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=3,
                dilation=3,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        out = self.fuse(torch.cat([b1, b2], dim=1))
        return out + x


class LiteContextModuleTriple(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, dilation=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=3, dilation=3, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=5, dilation=5, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        out = self.fuse(torch.cat([b1, b2, b3], dim=1))
        return out + x


class IdentityContext(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class LearnableDiffFusion(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        return self.fuse(torch.cat([f1, f2], dim=1))


class LearnableDiffFusionWithAbs(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(f1 - f2)
        return self.fuse(torch.cat([f1, f2, diff], dim=1))


class AbsDiffFusion(nn.Module):
    def __init__(self, channels: int | None = None):
        super().__init__()

    def forward(self, f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        return torch.abs(f1 - f2)


class AddDecoderBlock(nn.Module):
    def __init__(self, in_channels_high: int, in_channels_low: int, out_channels: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_low = nn.Conv2d(in_channels_low, out_channels, kernel_size=1, bias=False)
        self.conv_high = nn.Conv2d(in_channels_high, out_channels, kernel_size=1, bias=False)
        self.conv_fuse = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.attention = SEBlock(out_channels)

    def forward(self, high_feat: torch.Tensor, low_feat: torch.Tensor) -> torch.Tensor:
        high_feat = self.conv_high(self.up(high_feat))
        low_feat = self.conv_low(low_feat)
        fuse = self.conv_fuse(high_feat + low_feat)
        return self.attention(fuse)


class ConcatDecoderBlock(nn.Module):
    def __init__(self, in_channels_high: int, in_channels_low: int, out_channels: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_low = nn.Conv2d(in_channels_low, out_channels, kernel_size=1, bias=False)
        self.conv_high = nn.Conv2d(in_channels_high, out_channels, kernel_size=1, bias=False)
        self.conv_fuse = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.attention = SEBlock(out_channels)

    def forward(self, high_feat: torch.Tensor, low_feat: torch.Tensor) -> torch.Tensor:
        high_feat = self.conv_high(self.up(high_feat))
        low_feat = self.conv_low(low_feat)
        fuse = self.conv_fuse(torch.cat([high_feat, low_feat], dim=1))
        return self.attention(fuse)


class LiteCDNetAblation(nn.Module):
    def __init__(
        self,
        output_nc: int = 2,
        fusion_mode: str = "learnable",
        context_mode: str = "lite",
        decoder_mode: str = "add",
        deep_supervision: bool = True,
        use_pretrained_backbone: bool = True,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision

        if use_pretrained_backbone and MobileNet_V2_Weights is not None:
            encoder = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1).features
        elif use_pretrained_backbone:
            encoder = mobilenet_v2(pretrained=True).features
        else:
            if MobileNet_V2_Weights is not None:
                encoder = mobilenet_v2(weights=None).features
            else:
                encoder = mobilenet_v2(pretrained=False).features

        self.enc1 = encoder[0:2]
        self.enc2 = encoder[2:4]
        self.enc3 = encoder[4:7]
        self.enc4 = encoder[7:14]
        self.enc5 = encoder[14:18]

        decoder_cls = {
            "add": AddDecoderBlock,
            "concat": ConcatDecoderBlock,
        }.get(decoder_mode)
        if decoder_cls is None:
            raise ValueError(f"Unsupported decoder mode: {decoder_mode}")

        self.fusion5, self.fusion4, self.fusion3, self.fusion2, self.fusion1 = self._build_fusion_modules(fusion_mode)
        self.diff_c5, self.diff_c4, self.diff_c3, self.diff_c2, self.diff_c1 = self._build_context_modules(context_mode)

        self.dec4 = decoder_cls(320, 96, 96)
        self.dec3 = decoder_cls(96, 32, 64)
        self.dec2 = decoder_cls(64, 24, 32)
        self.dec1 = decoder_cls(32, 16, 16)

        self.pred_c4 = nn.Conv2d(96, output_nc, kernel_size=1)
        self.pred_c3 = nn.Conv2d(64, output_nc, kernel_size=1)
        self.pred_c2 = nn.Conv2d(32, output_nc, kernel_size=1)
        self.pred_c1 = nn.Conv2d(16, output_nc, kernel_size=1)

    @staticmethod
    def _build_fusion_modules(fusion_mode: str):
        if fusion_mode == "learnable":
            cls = LearnableDiffFusion
            return cls(320), cls(96), cls(32), cls(24), cls(16)
        if fusion_mode == "abs_diff":
            cls = AbsDiffFusion
            return cls(320), cls(96), cls(32), cls(24), cls(16)
        if fusion_mode == "c45_abs_concat":
            return (
                LearnableDiffFusionWithAbs(320),
                LearnableDiffFusionWithAbs(96),
                LearnableDiffFusion(32),
                LearnableDiffFusion(24),
                LearnableDiffFusion(16),
            )
        raise ValueError(f"Unsupported fusion mode: {fusion_mode}")

    @staticmethod
    def _build_context_modules(context_mode: str):
        if context_mode == "lite":
            return (
                LiteContextModule(320),
                LiteContextModule(96),
                LiteContextModule(32),
                LiteContextModule(24),
                LiteContextModule(16),
            )
        if context_mode == "identity":
            return IdentityContext(), IdentityContext(), IdentityContext(), IdentityContext(), IdentityContext()
        if context_mode == "c45_triple":
            return (
                LiteContextModuleTriple(320),
                LiteContextModuleTriple(96),
                LiteContextModule(32),
                LiteContextModule(24),
                LiteContextModule(16),
            )
        raise ValueError(f"Unsupported context mode: {context_mode}")

    def extract_features(self, x: torch.Tensor):
        f1 = self.enc1(x)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)
        f4 = self.enc4(f3)
        f5 = self.enc5(f4)
        return f1, f2, f3, f4, f5

    def forward(self, img1: torch.Tensor, img2: torch.Tensor):
        f1_1, f1_2, f1_3, f1_4, f1_5 = self.extract_features(img1)
        f2_1, f2_2, f2_3, f2_4, f2_5 = self.extract_features(img2)

        d5 = self.diff_c5(self.fusion5(f1_5, f2_5))
        d4 = self.diff_c4(self.fusion4(f1_4, f2_4))
        d3 = self.diff_c3(self.fusion3(f1_3, f2_3))
        d2 = self.diff_c2(self.fusion2(f1_2, f2_2))
        d1 = self.diff_c1(self.fusion1(f1_1, f2_1))

        out4 = self.dec4(d5, d4)
        out3 = self.dec3(out4, d3)
        out2 = self.dec2(out3, d2)
        out1 = self.dec1(out2, d1)

        h, w = img1.size(2), img1.size(3)
        final_mask = F.interpolate(self.pred_c1(out1), size=(h, w), mode="bilinear", align_corners=True)

        if self.training and self.deep_supervision:
            mask4 = F.interpolate(self.pred_c4(out4), size=(h, w), mode="bilinear", align_corners=True)
            mask3 = F.interpolate(self.pred_c3(out3), size=(h, w), mode="bilinear", align_corners=True)
            mask2 = F.interpolate(self.pred_c2(out2), size=(h, w), mode="bilinear", align_corners=True)
            return final_mask, mask2, mask3, mask4

        return final_mask


def build_ablation_model(args) -> LiteCDNetAblation:
    return LiteCDNetAblation(
        output_nc=args.n_class,
        fusion_mode=args.fusion_mode,
        context_mode=args.context_mode,
        decoder_mode=args.decoder_mode,
        deep_supervision=args.deep_supervision,
        use_pretrained_backbone=getattr(args, "use_pretrained_backbone", True),
    )
