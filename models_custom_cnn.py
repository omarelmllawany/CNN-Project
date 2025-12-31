"""
CNN مخصص مع وحدات انتباه - تنفيذ يدوي
"""

import torch
import torch.nn as nn
from .models_attention import CBAM

class ConvBlock(nn.Module):
    """كتلة تلافيفية مع انتباه"""
    def __init__(self, in_channels, out_channels, use_attention=True):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 
                              kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                              kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # وحدة الانتباه
        self.attention = CBAM(out_channels) if use_attention else nn.Identity()
        
        # وصلة التخطي (إذا كانت الأبعاد مختلفة)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 
                         kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # تطبيق الانتباه
        out = self.attention(out)
        
        # إضافة وصلة التخطي
        identity = self.shortcut(identity)
        out += identity
        out = self.relu(out)
        
        return out

class CustomCNN(nn.Module):
    """شبكة CNN مخصصة مع وحدات انتباه متعددة"""
    def __init__(self, num_classes=10, input_channels=3):
        super().__init__()
        
        # الطبقة الأولى
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        # كتل التلافيف مع انتباه
        self.block1 = ConvBlock(32, 64, use_attention=True)
        self.pool1 = nn.MaxPool2d(2)
        
        self.block2 = ConvBlock(64, 128, use_attention=True)
        self.pool2 = nn.MaxPool2d(2)
        
        self.block3 = ConvBlock(128, 256, use_attention=True)
        
        # طبقة التجميع النهائية
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # مصنف
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # تهيئة الأوزان
        self._initialize_weights()
    
    def _initialize_weights(self):
        """تهيئة أوزان النموذج"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # طبقة أولى
        x = self.conv1(x)
        
        # كتل مع انتباه
        x = self.block1(x)
        x = self.pool1(x)
        
        x = self.block2(x)
        x = self.pool2(x)
        
        x = self.block3(x)
        
        # تجميع نهائي
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        
        # تصنيف
        x = self.classifier(x)
        
        return x
    
    def extract_features(self, x):
        """استخراج المميزات من طبقات متعددة"""
        features = []
        
        x = self.conv1(x)
        features.append(x)  # مميزات منخفضة المستوى
        
        x = self.block1(x)
        x = self.pool1(x)
        features.append(x)  # مميزات متوسطة المستوى
        
        x = self.block2(x)
        x = self.pool2(x)
        features.append(x)  # مميزات عالية المستوى
        
        x = self.block3(x)
        features.append(x)  # مميزات نهائية
        
        return features

# اختبار النموذج
if __name__ == "__main__":
    model = CustomCNN(num_classes=10)
    print("بنية CustomCNN:")
    print(model)
    
    # اختبار تمرير البيانات
    test_input = torch.randn(4, 3, 64, 64)
    output = model(test_input)
    print(f"\nإدخال: {test_input.shape}")
    print(f"إخراج: {output.shape}")
    
    # حساب عدد الباراميترات
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nإجمالي الباراميترات: {total_params:,}")
    print(f"الباراميترات القابلة للتدريب: {trainable_params:,}")
    
    print("\n✓ النموذج يعمل بشكل صحيح!")