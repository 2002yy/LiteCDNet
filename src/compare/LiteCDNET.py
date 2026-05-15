import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
except ImportError:
    from torchvision.models import mobilenet_v2
    MobileNet_V2_Weights = None

# ==========================================
# 1. 轻量级通道注意力模块 (类似 SNUNet 的 ECAM，但更轻)
# ==========================================
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        weight = self.sigmoid(avg_out + max_out)
        return x * weight

# ==========================================
# 2. 轻量级多尺度上下文模块 (改进 A2Net 的 TFFM)
# 使用深度可分离空洞卷积，参数量锐减，感受野变大
# ==========================================
class LiteContextModule(nn.Module):
    def __init__(self, channels):
        super(LiteContextModule, self).__init__()
        # 分支1: 局部细节 (dilaion=1)
        self.branch1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        # 分支2: 较大感受野 (dilation=3)
        self.branch2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=3, dilation=3, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        # 融合层 (1x1 Conv)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        out = self.fuse(torch.cat([b1, b2], dim=1))
        return out + x  # 残差连接

# ==========================================
# 3. 解码器融合块
# ==========================================
class DecoderBlock(nn.Module):
    def __init__(self, in_channels_high, in_channels_low, out_channels):
        super(DecoderBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # 将低层特征调整维度
        self.conv_low = nn.Conv2d(in_channels_low, out_channels, kernel_size=1, bias=False)
        # 将高层特征调整维度
        self.conv_high = nn.Conv2d(in_channels_high, out_channels, kernel_size=1, bias=False)
        
        # 融合后的特征处理
        self.conv_fuse = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.attention = SEBlock(out_channels)

    def forward(self, high_feat, low_feat):
        high_feat = self.conv_high(self.up(high_feat))
        low_feat = self.conv_low(low_feat)
        
        # 使用相加代替拼接，极大减少参数量
        fuse = high_feat + low_feat 
        fuse = self.conv_fuse(fuse)
        fuse = self.attention(fuse) # 通道注意力过滤无用特征
        return fuse

# ==========================================
# 4. 整体网络架构 (LiteCDNet)
# ==========================================
# 新增一个极其轻量的特征差分融合模块
class DiffFusion(nn.Module):
    def __init__(self, channels):
        super(DiffFusion, self).__init__()
        # 使用 1x1 卷积将拼接后的通道数降维，比纯 abs() 效果好得多，参数极少
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, f1, f2):
        # 拼接特征而不是直接相减，保留更多语义信息
        x = torch.cat([f1, f2], dim=1)
        return self.fuse(x)

# 修改后的 LiteCDNet
class LiteCDNet(nn.Module):
    def __init__(self, output_nc=2):
        super(LiteCDNet, self).__init__()
        
        if MobileNet_V2_Weights is not None:
            weights = MobileNet_V2_Weights.IMAGENET1K_V1
            mobilenet = mobilenet_v2(weights=weights).features
        else:
            mobilenet = mobilenet_v2(pretrained=True).features
            
        self.enc1 = mobilenet[0:2]   # 1/2 scale, 16 channels (必须保留)
        self.enc2 = mobilenet[2:4]   # 1/4 scale, 24 channels
        self.enc3 = mobilenet[4:7]   # 1/8 scale, 32 channels
        self.enc4 = mobilenet[7:14]  # 1/16 scale, 96 channels
        self.enc5 = mobilenet[14:18] # 1/32 scale, 320 channels

        # 时序融合模块 (替换直接相减)
        self.fusion5 = DiffFusion(320)
        self.fusion4 = DiffFusion(96)
        self.fusion3 = DiffFusion(32)
        self.fusion2 = DiffFusion(24)
        self.fusion1 = DiffFusion(16) # 新增 1/2 尺度融合

        # 时序差异特征处理模块
        self.diff_c5 = LiteContextModule(320)
        self.diff_c4 = LiteContextModule(96)
        self.diff_c3 = LiteContextModule(32)
        self.diff_c2 = LiteContextModule(24)
        self.diff_c1 = LiteContextModule(16) # 新增

        # 解码器 (逐层上采样，补全 1/2 尺度)
        self.dec4 = DecoderBlock(320, 96, 96)
        self.dec3 = DecoderBlock(96, 32, 64)
        self.dec2 = DecoderBlock(64, 24, 32)
        self.dec1 = DecoderBlock(32, 16, 16) # 新增，输出 16 通道

        # 预测头
        self.pred_c4 = nn.Conv2d(96, output_nc, kernel_size=1)
        self.pred_c3 = nn.Conv2d(64, output_nc, kernel_size=1)
        self.pred_c2 = nn.Conv2d(32, output_nc, kernel_size=1)
        self.pred_c1 = nn.Conv2d(16, output_nc, kernel_size=1) # 最终预测头

    def extract_features(self, x):
        f1 = self.enc1(x)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)
        f4 = self.enc4(f3)
        f5 = self.enc5(f4)
        return f1, f2, f3, f4, f5  # 恢复返回 f1

    def forward(self, img1, img2):
        f1_1, f1_2, f1_3, f1_4, f1_5 = self.extract_features(img1)
        f2_1, f2_2, f2_3, f2_4, f2_5 = self.extract_features(img2)

        # 融合 + 上下文提取
        d5 = self.diff_c5(self.fusion5(f1_5, f2_5))
        d4 = self.diff_c4(self.fusion4(f1_4, f2_4))
        d3 = self.diff_c3(self.fusion3(f1_3, f2_3))
        d2 = self.diff_c2(self.fusion2(f1_2, f2_2))
        d1 = self.diff_c1(self.fusion1(f1_1, f2_1))

        # 解码阶段
        out4 = self.dec4(d5, d4)   # 1/16 scale
        out3 = self.dec3(out4, d3) # 1/8 scale
        out2 = self.dec2(out3, d2) # 1/4 scale
        out1 = self.dec1(out2, d1) # 1/2 scale (高分辨率特征恢复)

        # 预测与上采样
        H, W = img1.size(2), img1.size(3)
        
        if self.training:
            mask4 = F.interpolate(self.pred_c4(out4), size=(H, W), mode='bilinear', align_corners=True)
            mask3 = F.interpolate(self.pred_c3(out3), size=(H, W), mode='bilinear', align_corners=True)
            mask2 = F.interpolate(self.pred_c2(out2), size=(H, W), mode='bilinear', align_corners=True)
            mask1 = F.interpolate(self.pred_c1(out1), size=(H, W), mode='bilinear', align_corners=True)
            return mask1, mask2, mask3, mask4
        else:
            mask1 = self.pred_c1(out1)
            # 推理时只需将 1/2 尺度放大两倍到原图
            mask1 = F.interpolate(mask1, size=(H, W), mode='bilinear', align_corners=True)
            return mask1

# ==========================================
# 5. 测试与参数量/计算量分析
# ==========================================
if __name__ == '__main__':
    # 初始化模型
    model = LiteCDNet(output_nc=2)
    model.eval() # 切换到推理模式
    
    # 论文中计算 FLOPs 的标准输入通常为：Batch=1, C=3, H=256, W=256
    dummy_t1 = torch.randn(1, 3, 256, 256)
    dummy_t2 = torch.randn(1, 3, 256, 256)
    
    print("="*50)
    print("🚀 开始分析模型 (Input Size: 1 x 3 x 256 x 256)")
    print("="*50)
    
    # 尝试使用 thop 计算真实的 FLOPs 和 参数量
    try:
        from thop import profile, clever_format
        
        # profile 模型，关闭 verbose 避免刷屏
        macs, params = profile(model, inputs=(dummy_t1, dummy_t2), verbose=False)
        
        # 格式化输出 (自动转为 M, G)
        macs_str, params_str = clever_format([macs, params], "%.2f")
        
        # 论文中通常把 MACs 直接写成 FLOPs（或 1 MAC ≈ 2 FLOPs，这里按学界惯例输出）
        print(f"📊 模型参数量 (Params) : {params_str}")
        print(f"⚡ 计算复杂度 (MACs)   : {macs_str}  (论文表中可直接填此数值)")
        
        # 补充严格意义上的 FLOPs
        print(f"⚡ 理论浮点运算(FLOPs)  : {macs * 2 / 1e9:.2f} G")
        
    except ImportError:
        print("⚠️ 未安装 'thop' 库，无法计算 FLOPs！")
        print("👉 请在终端执行: pip install thop")
        print("-" * 50)
        # 备用方案：仅计算参数量
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"📊 模型参数量 (Params) : {total_params / 1e6:.2f} M")

    print("="*50)
    
    # 测试前向传播是否正常
    with torch.no_grad():
        output = model(dummy_t1, dummy_t2)
        print(f"✅ 前向传播测试通过，输出形状: {output.shape}") 
    print("="*50)