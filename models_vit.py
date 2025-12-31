"""
Vision Transformer - تنفيذ يدوي مع تحسينات
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .models_attention import MultiHeadAttention

class PatchEmbedding(nn.Module):
    """تحويل الصورة إلى بقع (patches)"""
    def __init__(self, img_size=64, patch_size=8, in_channels=3, embed_dim=128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # استخدام convolution لاستخراج البقع
        self.projection = nn.Conv2d(in_channels, embed_dim,
                                   kernel_size=patch_size, stride=patch_size)
        
        # تسوية
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        # x: [batch, channels, height, width]
        x = self.projection(x)  # [batch, embed_dim, num_patches_h, num_patches_w]
        x = x.flatten(2)  # [batch, embed_dim, num_patches]
        x = x.transpose(1, 2)  # [batch, num_patches, embed_dim]
        x = self.norm(x)
        return x

class PositionalEncoding(nn.Module):
    """ترميز موضعي قابل للتعلم"""
    def __init__(self, num_patches, embed_dim, dropout=0.1):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # توسيع token التصنيف
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        
        # إضافة token التصنيف للبقع
        x = torch.cat([cls_tokens, x], dim=1)
        
        # إضافة الترميز الموضعي
        x = x + self.pos_embedding
        x = self.dropout(x)
        
        return x

class TransformerBlock(nn.Module):
    """كتلة Transformer واحدة"""
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        
        # الانتباه متعدد الرؤوس
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        
        # تسوية
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # شبكة تغذية أمامية
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # انتباه مع وصلة تخطي
        residual = x
        x = self.norm1(x)
        attn_output, attn_weights = self.attention(x, x, x)
        x = residual + self.dropout(attn_output)
        
        # MLP مع وصلة تخطي
        residual = x
        x = self.norm2(x)
        mlp_output = self.mlp(x)
        x = residual + self.dropout(mlp_output)
        
        return x, attn_weights

class VisionTransformer(nn.Module):
    """محول الرؤية الكامل"""
    def __init__(self, img_size=64, patch_size=8, in_channels=3,
                 num_classes=10, embed_dim=128, depth=4, 
                 num_heads=4, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        
        # استخراج البقع
        self.patch_embed = PatchEmbedding(img_size, patch_size, 
                                         in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # الترميز الموضعي
        self.pos_embed = PositionalEncoding(num_patches, embed_dim, dropout)
        
        # كتل Transformer
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # تسوية نهائية
        self.norm = nn.LayerNorm(embed_dim)
        
        # مصنف
        self.head = nn.Linear(embed_dim, num_classes)
        
        # تهيئة الأوزان
        self._initialize_weights()
    
    def _initialize_weights(self):
        """تهيئة أوزان النموذج"""
        # تهيئة token التصنيف
        nn.init.trunc_normal_(self.pos_embed.cls_token, std=0.02)
        
        # تهيئة الترميز الموضعي
        nn.init.trunc_normal_(self.pos_embed.pos_embedding, std=0.02)
        
        # تهيئة المصنف
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        
        # تهيئة الطبقات الخطية
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x, return_attention=False):
        # استخراج البقع
        x = self.patch_embed(x)
        
        # إضافة الترميز الموضعي
        x = self.pos_embed(x)
        
        # تمرير خلال كتل Transformer
        attention_weights = []
        for block in self.blocks:
            x, attn = block(x)
            if return_attention:
                attention_weights.append(attn)
        
        # تسوية نهائية
        x = self.norm(x)
        
        # استخدام token التصنيف للتصنيف
        cls_token = x[:, 0]
        
        # تصنيف
        x = self.head(cls_token)
        
        if return_attention:
            return x, attention_weights
        return x

# اختبار النموذج
if __name__ == "__main__":
    model = VisionTransformer(
        img_size=64,
        patch_size=8,
        num_classes=10,
        embed_dim=128,
        depth=4,
        num_heads=4
    )
    
    print("بنية VisionTransformer:")
    print(model)
    
    # اختبار تمرير البيانات
    test_input = torch.randn(4, 3, 64, 64)
    output = model(test_input)
    print(f"\nإدخال: {test_input.shape}")
    print(f"إخراج: {output.shape}")
    
    # اختبار مع إرجاع الانتباه
    output, attention = model(test_input, return_attention=True)
    print(f"\nعدد خرائط الانتباه: {len(attention)}")
    print(f"شكل خرائط الانتباه: {attention[0].shape}")
    
    # حساب عدد الباراميترات
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nإجمالي الباراميترات: {total_params:,}")
    print(f"الباراميترات القابلة للتدريب: {trainable_params:,}")
    
    print("\n✓ النموذج يعمل بشكل صحيح!")