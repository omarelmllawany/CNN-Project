"""
وحدات الانتباه - التنفيذ اليدوي
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ChannelAttention(nn.Module):
    """انتباه القنوات - تنفيذ يدوي"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        
        # متوسط القنوات
        avg_out = self.avg_pool(x).view(batch_size, channels)
        avg_out = self.fc(avg_out).view(batch_size, channels, 1, 1)
        
        # أقصى قيمة للقنوات
        max_out = self.max_pool(x).view(batch_size, channels)
        max_out = self.fc(max_out).view(batch_size, channels, 1, 1)
        
        # الجمع والتطبيع
        attention = self.sigmoid(avg_out + max_out)
        
        return x * attention

class SpatialAttention(nn.Module):
    """انتباه مكاني - تنفيذ يدوي"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, 
                             padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # تجميع على مستوى القنوات
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # دمج النتائج
        combined = torch.cat([avg_out, max_out], dim=1)
        
        # تطبيق الـ convolution
        attention = self.conv(combined)
        attention = self.sigmoid(attention)
        
        return x * attention

class CBAM(nn.Module):
    """وحدة الانتباه المزدوجة (قنوي + مكاني)"""
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def forward(self, x):
        # انتباه القنوات أولاً
        x = self.channel_attention(x)
        # ثم الانتباه المكاني
        x = self.spatial_attention(x)
        return x

class MultiHeadAttention(nn.Module):
    """انتباه متعدد الرؤوس للـ Transformer"""
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim يجب أن يقبل القسمة على num_heads"
        
        # طبقات خطية للاستعلام والمفتاح والقيمة
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # عامل القياس
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # إسقاط خطي
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)
        
        # إعادة تشكيل للرؤوس المتعددة
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # حساب الانتباه
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # تطبيق الأوزان على القيم
        output = torch.matmul(attention_weights, V)
        
        # إعادة التشكيل
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, -1, self.embed_dim)
        
        # الإسقاط النهائي
        output = self.out_proj(output)
        
        return output, attention_weights

# اختبار الوحدات
if __name__ == "__main__":
    # اختبار ChannelAttention
    x = torch.randn(4, 64, 32, 32)
    ca = ChannelAttention(64)
    out = ca(x)
    print(f"ChannelAttention: Input {x.shape} -> Output {out.shape}")
    
    # اختبار CBAM
    cbam = CBAM(64)
    out = cbam(x)
    print(f"CBAM: Input {x.shape} -> Output {out.shape}")
    
    # اختبار MultiHeadAttention
    mha = MultiHeadAttention(embed_dim=512, num_heads=8)
    q = torch.randn(4, 16, 512)
    out, attn = mha(q, q, q)
    print(f"MultiHeadAttention: Input {q.shape} -> Output {out.shape}")
    
    print("\n✓ جميع وحدات الانتباه تعمل بشكل صحيح!")