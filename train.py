"""
سكربت التدريب الرئيسي
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import argparse
import os
import time
import json
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from models.models_custom_cnn import CustomCNN
from models.models_vit import VisionTransformer
from utils.utils_dataset import create_dataloaders

class Trainer:
    """مدرب النماذج"""
    
    def __init__(self, model_type='cnn', num_classes=10, device=None):
        self.model_type = model_type
        self.num_classes = num_classes
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # إنشاء النموذج
        self.model = self._create_model()
        self.model.to(self.device)
        
        # تاريخ التدريب
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'learning_rates': []
        }
        
        # مجلد الحفظ
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.save_dir = f'./results/{model_type}_{timestamp}'
        os.makedirs(self.save_dir, exist_ok=True)
        
        print(f"النموذج: {model_type}")
        print(f"الجهاز: {self.device}")
        print(f"مجلد الحفظ: {self.save_dir}")
    
    def _create_model(self):
        """إنشاء النموذج المطلوب"""
        if self.model_type == 'cnn':
            print("إنشاء Custom CNN...")
            return CustomCNN(num_classes=self.num_classes)
        elif self.model_type == 'vit':
            print("إنشاء Vision Transformer...")
            return VisionTransformer(num_classes=self.num_classes)
        else:
            raise ValueError(f"نموذج غير معروف: {self.model_type}")
    
    def train_epoch(self, train_loader, criterion, optimizer):
        """تدريب عصر واحد"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='تدريب', leave=False)
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # التمرير الأمامي
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            
            # الانتشار العكسي
            optimizer.zero_grad()
            loss.backward()
            
            # تقييد الانحدار
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # إحصائيات
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # تحديث progress bar
            current_loss = running_loss / total
            current_acc = 100. * correct / total
            pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'acc': f'{current_acc:.2f}%'
            })
        
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader, criterion):
        """التحقق من دقة النموذج"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='تحقق', leave=False)
            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                current_loss = running_loss / total
                current_acc = 100. * correct / total
                pbar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'acc': f'{current_acc:.2f}%'
                })
        
        val_loss = running_loss / total
        val_acc = 100. * correct / total
        
        return val_loss, val_acc
    
    def train(self, train_loader, val_loader, 
              num_epochs=30, learning_rate=0.001,
              weight_decay=1e-4, scheduler_type='step'):
        """عملية التدريب الرئيسية"""
        
        print(f"\nبدء التدريب لـ {num_epochs} عصر...")
        
        # المعايير والمحسن
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # جدولة معدل التعلم
        if scheduler_type == 'step':
            scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        elif scheduler_type == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
        else:
            scheduler = None
        
        أفضل_dacc = 0.0
        أفضل_epoch = 0
        
        for epoch in range(num_epochs):
            print(f"\n{'='*60}")
            print(f"عصر {epoch+1}/{num_epochs}")
            print(f"{'='*60}")
            
            start_time = time.time()
            
            # التدريب
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            
            # التحقق
            val_loss, val_acc = self.validate(val_loader, criterion)
            
            # وقت العصر
            epoch_time = time.time() - start_time
            
            # حفظ النتائج
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(optimizer.param_groups[0]['lr'])
            
            # تحديث جدولة معدل التعلم
            if scheduler:
                scheduler.step()
            
            # عرض النتائج
            print(f"\nنتائج العصر:")
            print(f"  فقدان التدريب: {train_loss:.4f}")
            print(f"  دقة التدريب:   {train_acc:.2f}%")
            print(f"  فقدان التحقق:   {val_loss:.4f}")
            print(f"  دقة التحقق:     {val_acc:.2f}%")
            print(f"  معدل التعلم:    {optimizer.param_groups[0]['lr']:.6f}")
            print(f"  الوقت:          {epoch_time:.1f} ثانية")
            
            # حفظ أفضل نموذج
            if val_acc > أفضل_dacc:
                أفضل_dacc = val_acc
                أفضل_epoch = epoch + 1
                
                model_path = os.path.join(self.save_dir, 'أفضل_نموذج.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                }, model_path)
                
                print(f"\n✓ حفظ أفضل نموذج (دقة: {val_acc:.2f}%)")
            
            # حفظ checkpoint دوري
            if (epoch + 1) % 5 == 0:
                checkpoint_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch+1}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'history': self.history,
                }, checkpoint_path)
        
        # حفظ تاريخ التدريب
        self._save_history()
        
        print(f"\n{'='*60}")
        print(f"انتهى التدريب!")
        print(f"أفضل دقة تحقق: {أفضل_dacc:.2f}% (عصر {أفضل_epoch})")
        print(f"{'='*60}")
        
        return أفضل_dacc
    
    def _save_history(self):
        """حفظ تاريخ التدريب"""
        # حفظ كـ JSON
        history_path = os.path.join(self.save_dir, 'training_history.json')
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=4, ensure_ascii=False)
        
        # رسم المخططات
        self._plot_training_history()
    
    def _plot_training_history(self):
        """رسم مخططات التدريب"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # فقدان التدريب والتحقق
        axes[0, 0].plot(self.history['train_loss'], 'b-', label='فقدان التدريب')
        axes[0, 0].plot(self.history['val_loss'], 'r-', label='فقدان التحقق')
        axes[0, 0].set_xlabel('العصر')
        axes[0, 0].set_ylabel('الفقدان')
        axes[0, 0].set_title('فقدان التدريب والتحقق')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # دقة التدريب والتحقق
        axes[0, 1].plot(self.history['train_acc'], 'b-', label='دقة التدريب')
        axes[0, 1].plot(self.history['val_acc'], 'r-', label='دقة التحقق')
        axes[0, 1].set_xlabel('العصر')
        axes[0, 1].set_ylabel('الدقة (%)')
        axes[0, 1].set_title('دقة التدريب والتحقق')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # معدل التعلم
        axes[1, 0].plot(self.history['learning_rates'], 'g-')
        axes[1, 0].set_xlabel('العصر')
        axes[1, 0].set_ylabel('معدل التعلم')
        axes[1, 0].set_title('تغير معدل التعلم')
        axes[1, 0].grid(True)
        
        # الفرق بين التدريب والتحقق
        diff_acc = [t - v for t, v in zip(self.history['train_acc'], self.history['val_acc'])]
        axes[1, 1].plot(diff_acc, 'purple')
        axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[1, 1].set_xlabel('العصر')
        axes[1, 1].set_ylabel('الفرق في الدقة (%)')
        axes[1, 1].set_title('فرق الدقة (تدريب - تحقق)')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # حفظ المخطط
        plot_path = os.path.join(self.save_dir, 'training_plots.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ تم حفظ مخططات التدريب في: {plot_path}")
    
    def test(self, test_loader):
        """اختبار النموذج النهائي"""
        print(f"\n{'='*60}")
        print("اختبار النموذج")
        print(f"{'='*60}")
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc='اختبار')
            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                
                all_predictions.extend(predicted.cpu().numpy().tolist())
                all_targets.extend(targets.cpu().numpy().tolist())
                
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                current_acc = 100. * correct / total
                pbar.set_postfix({'acc': f'{current_acc:.2f}%'})
        
        test_acc = 100. * correct / total
        
        print(f"\nنتائج الاختبار:")
        print(f"  الدقة: {test_acc:.2f}%")
        print(f"  الصحيح: {correct}/{total}")
        
        # حفظ نتائج الاختبار
        test_results = {
            'test_accuracy': test_acc,
            'correct': correct,
            'total': total,
            'predictions': all_predictions,
            'targets': all_targets
        }
        
        results_path = os.path.join(self.save_dir, 'test_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, indent=4, ensure_ascii=False)
        
        print(f"✓ تم حفظ نتائج الاختبار في: {results_path}")
        
        return test_acc

def main():
    """الدالة الرئيسية"""
    parser = argparse.ArgumentParser(description='تدريب نماذج CNN وTransformer')
    
    parser.add_argument('--model', type=str, default='cnn',
                       choices=['cnn', 'vit'],
                       help='نوع النموذج (cnn أو vit)')
    parser.add_argument('--epochs', type=int, default=20,
                       help='عدد العصور للتدريب')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='حجم الدفعة')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='معدل التعلم')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='مجلد البيانات')
    parser.add_argument('--img_size', type=int, default=64,
                       help='حجم الصور')
    
    args = parser.parse_args()
    
    print("="*60)
    print("نظام تصنيف صور الأقمار الصناعية")
    print("="*60)
    
    # تحميل البيانات
    print(f"\nجاري تحميل البيانات من {args.data_dir}...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        img_size=args.img_size
    )
    
    print(f"بيانات التدريب: {len(train_loader.dataset)} صورة")
    print(f"بيانات التحقق: {len(val_loader.dataset)} صورة")
    print(f"بيانات الاختبار: {len(test_loader.dataset)} صورة")
    
    # إنشاء المدرب
    trainer = Trainer(
        model_type=args.model,
        num_classes=10,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )
    
    # التدريب
    best_val_acc = trainer.train(
        train_loader, val_loader,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=1e-4,
        scheduler_type='step'
    )
    
    # اختبار النموذج
    test_acc = trainer.test(test_loader)
    
    print(f"\n{'='*60}")
    print("ملخص النتائج:")
    print(f"  أفضل دقة تحقق: {best_val_acc:.2f}%")
    print(f"  دقة الاختبار:   {test_acc:.2f}%")
    print(f"{'='*60}")
    
    print(f"\nيمكنك العثور على النتائج في: {trainer.save_dir}")

if __name__ == "__main__":
    main()