import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

# ==========================================
# 1. 差异引导注意力
# ==========================================
class DiffGuidedAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1)
        )

    def forward(self, x1, x2):
        diff = torch.abs(x1 - x2)
        att = torch.sigmoid(self.conv(diff))
        return att

# ==========================================
# 2. CoDEM（改进版）
# ==========================================
class CoDEM2(nn.Module):
    def __init__(self, channel_dim):
        super().__init__()
        self.conv_fuse = nn.Sequential(
            nn.Conv2d(2 * channel_dim, channel_dim, 3, padding=1),
            nn.GroupNorm(8, channel_dim),
            nn.ReLU(inplace=True)
        )
        self.att = DiffGuidedAttention(channel_dim)

    def forward(self, x1, x2):
        diff = torch.abs(x1 - x2)
        att = self.att(x1, x2)
        diff_enhanced = diff * (1 + att)
        fuse = self.conv_fuse(torch.cat([x1, x2], dim=1))
        return diff_enhanced + fuse

# ==========================================
# 3. FPN（严格对齐版本）
# ==========================================
class FPN(nn.Module):
    def __init__(self, channels, out_c=64):
        super().__init__()
        self.lateral = nn.ModuleList([
            nn.Conv2d(c, out_c, 1) for c in channels
        ])

    def forward(self, d2, d3, d4, d5):
        p5 = self.lateral[3](d5)

        p4 = self.lateral[2](d4) + F.interpolate(
            p5, size=d4.shape[-2:], mode='bilinear', align_corners=False)

        p3 = self.lateral[1](d3) + F.interpolate(
            p4, size=d3.shape[-2:], mode='bilinear', align_corners=False)

        p2 = self.lateral[0](d2) + F.interpolate(
            p3, size=d2.shape[-2:], mode='bilinear', align_corners=False)

        return p2, p3, p4, p5

# ==========================================
# 4. Decoder
# ==========================================
class Decoder(nn.Module):
    def __init__(self, c=64, out_c=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.ReLU(inplace=True)
        )
        # 这里的输出通道数由外部的 output_nc 决定
        self.cls = nn.Conv2d(c, out_c, 1)

    def forward(self, p2):
        x = self.conv(p2)
        return self.cls(x)

# ==========================================
# 5. 主模型 A2Net_v2
# ==========================================
class A2Net_v2(nn.Module):
    # 完美适配你的 define_G 接口：input_nc=3, output_nc=2
    def __init__(self, input_nc=3, output_nc=2, **kwargs):
        super().__init__()

        # 主干网络
        backbone = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        # 如果你的 input_nc 不是3，这里需要修改第一层卷积，通常CD任务默认是3
        features = backbone.features

        # ===== 标准分层 =====
        self.layer1 = nn.Sequential(*features[:2])    # 16
        self.layer2 = nn.Sequential(*features[2:4])   # 24
        self.layer3 = nn.Sequential(*features[4:7])   # 32/64/96
        self.layer4 = nn.Sequential(*features[7:14])  # 160
        self.layer5 = nn.Sequential(*features[14:])   # 1280

        # 降维
        self.reduce_l3 = nn.Conv2d(96, 32, 1)
        self.reduce_l4 = nn.Conv2d(160, 96, 1)
        self.reduce_l5 = nn.Sequential(
            nn.Conv2d(1280, 320, 1),
            nn.GroupNorm(8, 320),
            nn.ReLU(inplace=True)
        )

        # 差分模块
        self.diff2 = CoDEM2(24)
        self.diff3 = CoDEM2(32)
        self.diff4 = CoDEM2(96)
        self.diff5 = CoDEM2(320)

        # FPN
        self.fpn = FPN([24, 32, 96, 320], 64)

        # 动态传入 output_nc
        self.decoder = Decoder(64, output_nc)

        # 边缘分支永远输出单通道 (1)
        self.edge_head = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, x1, x2):
        # ===== Backbone (已修复第一版的严重Bug) =====
        # 分支 1
        feat1_l1 = self.layer1(x1)
        feat1_l2 = self.layer2(feat1_l1)
        feat1_l3 = self.layer3(feat1_l2)
        feat1_l4 = self.layer4(feat1_l3)
        feat1_l5 = self.layer5(feat1_l4)

        f1_l2 = feat1_l2
        f1_l3 = self.reduce_l3(feat1_l3)
        f1_l4 = self.reduce_l4(feat1_l4)
        f1_l5 = self.reduce_l5(feat1_l5)

        # 分支 2
        feat2_l1 = self.layer1(x2)
        feat2_l2 = self.layer2(feat2_l1)
        feat2_l3 = self.layer3(feat2_l2)
        feat2_l4 = self.layer4(feat2_l3)
        feat2_l5 = self.layer5(feat2_l4)

        f2_l2 = feat2_l2
        f2_l3 = self.reduce_l3(feat2_l3)
        f2_l4 = self.reduce_l4(feat2_l4)
        f2_l5 = self.reduce_l5(feat2_l5)

        # ===== 差分 =====
        d2 = self.diff2(f1_l2, f2_l2)
        d3 = self.diff3(f1_l3, f2_l3)
        d4 = self.diff4(f1_l4, f2_l4)
        d5 = self.diff5(f1_l5, f2_l5)

        # ===== FPN =====
        p2, p3, p4, p5 = self.fpn(d2, d3, d4, d5)

        # ===== 双头输出 =====
        out = self.decoder(p2)
        edge = self.edge_head(p2)

        # 恢复到原图分辨率
        out = F.interpolate(out, size=x1.shape[-2:], mode='bilinear', align_corners=False)
        edge = F.interpolate(edge, size=x1.shape[-2:], mode='bilinear', align_corners=False)

        # 因为你外部框架设置了 --deep_supervision=True，
        # 所以 trainer 会接收这个元组/列表，拆分为 main_loss 和 edge_loss
        return out, edge