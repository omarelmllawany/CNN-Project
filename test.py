"""
سكربت الاختبار
"""
import torch
import torch.nn as nn
import os
import sys

# إضافة المسارات
sys.path.append('./models')
sys.path.append('./utils')

def test_models():
    """اختبار جميع النماذج"""
    print("="*60)
    print("اختبار نماذج المشروع")
    print("="*60)
    
    # اختبار الانتباه
    print("\n1. اختبار وحدات الانتباه...")
    try:
        from models.attention import ChannelAttention, SpatialAttention, CBAM
        
        # اختبار ChannelAttention
        ca = ChannelAttention(64)
        test_input = torch.randn(4, 64, 32, 32)
        output = ca(test_input)
        print(f"  ✓ ChannelAttention: {test_input.shape} -> {output.shape}")
        
        # اختبار CBAM
        cbam = CBAM(64)
        output = cbam(test_input)
        print(f"  ✓ CBAM: {test_input.shape} -> {output.shape}")
        
    except Exception as e:
        print(f"  ✗ خطأ في اختبار الانتباه: {e}")
    
    # اختبار CNN
    print("\n2. اختبار Custom CNN...")
    try:
        from models.custom_cnn import CustomCNN
        
        model = CustomCNN(num_classes=10)
        test_input = torch.randn(4, 3, 64, 64)
        output = model(test_input)
        
        params = sum(p.numel() for p in model.parameters())
        print(f"  ✓ CustomCNN: {test_input.shape} -> {output.shape}")
        print(f"    الباراميترات: {params:,}")
        
    except Exception as e:
        print(f"  ✗ خطأ في اختبار CNN: {e}")
    
    # اختبار Vision Transformer
    print("\n3. اختبار Vision Transformer...")
    try:
        from models.vit import VisionTransformer
        
        model = VisionTransformer(
            img_size=64,
            patch_size=8,
            num_classes=10,
            embed_dim=128,
            depth=4,
            num_heads=4
        )
        
        test_input = torch.randn(4, 3, 64, 64)
        output = model(test_input)
        
        params = sum(p.numel() for p in model.parameters())
        print(f"  ✓ VisionTransformer: {test_input.shape} -> {output.shape}")
        print(f"    الباراميترات: {params:,}")
        
    except Exception as e:
        print(f"  ✗ خطأ في اختبار Transformer: {e}")
    
    # اختبار البيانات
    print("\n4. اختبار مجموعة البيانات...")
    try:
        from utils.dataset import create_dataloaders
        
        train_loader, val_loader, test_loader = create_dataloaders(
            data_dir='./data',
            batch_size=4,
            img_size=64
        )
        
        print(f"  ✓ تم إنشاء DataLoaders")
        print(f"    التدريب: {len(train_loader.dataset)} صورة")
        print(f"    التحقق: {len(val_loader.dataset)} صورة")
        print(f"    الاختبار: {len(test_loader.dataset)} صورة")
        
        # اختبار دفعة واحدة
        for images, labels in train_loader:
            print(f"  ✓ دفعة عينة: {images.shape}, {labels.shape}")
            break
            
    except Exception as e:
        print(f"  ✗ خطأ في اختبار البيانات: {e}")
    
    # اختبار التدريب
    print("\n5. اختبار دالة التدريب...")
    try:
        import torch.nn as nn
        import torch.optim as optim
        
        # نموذج مصغر للاختبار
        test_model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 10)
        )
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(test_model.parameters(), lr=0.001)
        
        test_input = torch.randn(4, 3, 64, 64)
        test_target = torch.randint(0, 10, (4,))
        
        # تمرير أمامي
        output = test_model(test_input)
        loss = criterion(output, test_target)
        
        # انتشار عكسي
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"  ✓ دالة التدريب تعمل")
        print(f"    الفقد: {loss.item():.4f}")
        
    except Exception as e:
        print(f"  ✗ خطأ في اختبار التدريب: {e}")
    
    print("\n" + "="*60)
    print("انتهى الاختبار!")
    print("="*60)
    
    # تعليمات التشغيل
    print("\nتعليمات التشغيل:")
    print("1. تنزيل البيانات: python download_data.py")
    print("2. تدريب CNN: python train.py --model cnn --epochs 10")
    print("3. تدريب Transformer: python train.py --model vit --epochs 10")
    print("4. التوقع: python main.py --mode predict --model cnn --image test.jpg")
    print("5. استخدام كلا النموذجين: python main.py --mode ensemble --image test.jpg")

if __name__ == "__main__":
    test_models()