"""
التطبيق الرئيسي
"""
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import matplotlib
# Allow selecting backend via environment variable `MATPLOTLIB_BACKEND` before importing pyplot.
backend = os.environ.get('MATPLOTLIB_BACKEND')
if backend:
    try:
        matplotlib.use(backend)
    except Exception:
        pass
else:
    # If no backend specified, try to use TkAgg when tkinter is available (common on Windows)
    try:
        import tkinter  # type: ignore
        matplotlib.use('TkAgg')
    except Exception:
        # Leave matplotlib to pick a default backend; if none suitable, user can set MATPLOTLIB_BACKEND
        pass
import matplotlib.pyplot as plt
import argparse
import os
import sys

# إضافة المسارات
# تأكد من استيراد الحزمة `models` من المجلد الحالي (مجلد المشروع)
# لا نحتاج لإضافة ./models إلى sys.path لأن `models` هو package في المجلد الجذري
from models.models_custom_cnn import CustomCNN
from models.models_vit import VisionTransformer

class SatelliteClassifier:
    """مصنف صور الأقمار الصناعية"""
    
    CLASS_NAMES = [
        'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
        'Industrial', 'Pasture', 'PermanentCrop', 'Residential',
        'River', 'SeaLake'
    ]
    
    def __init__(self, model_type='cnn', model_path=None):
        self.model_type = model_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # إنشاء النموذج
        if model_type == 'cnn':
            self.model = CustomCNN(num_classes=10)
        elif model_type == 'vit':
            self.model = VisionTransformer(num_classes=10)
        else:
            raise ValueError(f"نموذج غير معروف: {model_type}")
        
        # تحميل الأوزان
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"تم تحميل النموذج من {model_path}")
        else:
            print("تحذير: استخدام نموذج غير مدرب")
        
        self.model.to(self.device)
        self.model.eval()
        
        # تحويلات الصور
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image_path):
        """توقع فئة الصورة"""
        # تحميل الصورة
        try:
            image = Image.open(image_path).convert('RGB')
        except:
            print(f"خطأ: لا يمكن تحميل الصورة {image_path}")
            return None
        
        # حفظ للعرض
        original_image = np.array(image)
        
        # تطبيق التحويلات
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # التوقع
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
        
        # النتائج
        result = {
            'class_index': predicted_idx.item(),
            'class_name': self.CLASS_NAMES[predicted_idx.item()],
            'confidence': confidence.item() * 100,
            'probabilities': probabilities.cpu().numpy()[0].tolist()
        }
        
        return result, original_image
    
    def predict_batch(self, image_paths):
        """توقع مجموعة من الصور"""
        results = []
        originals = []
        
        for img_path in image_paths:
            result, original = self.predict(img_path)
            if result:
                results.append(result)
                originals.append(original)
        
        return results, originals
    
    def visualize_prediction(self, image_path, save_path=None, show=True):
        """تصور التوقع مع الاحتمالات

        Args:
            image_path: path to input image
            save_path: if provided, save the figure to this path
            show: whether to call `plt.show()` (default True)
        """
        result, original_image = self.predict(image_path)

        if not result:
            return None

        # إنشاء الشكل
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # عرض الصورة الأصلية
        axes[0].imshow(original_image)
        axes[0].set_title(f'الصورة المدخلة\nالتوقع: {result["class_name"]}\nالثقة: {result["confidence"]:.1f}%')
        axes[0].axis('off')
        
        # عرض الاحتمالات
        probabilities = result['probabilities']
        class_indices = np.arange(len(self.CLASS_NAMES))
        
        # ألوان الأعمدة
        colors = ['red' if i == result['class_index'] else 'blue' 
                 for i in class_indices]
        
        bars = axes[1].barh(class_indices, probabilities, color=colors)
        axes[1].set_yticks(class_indices)
        axes[1].set_yticklabels(self.CLASS_NAMES, fontsize=9)
        axes[1].set_xlabel('الاحتمالية')
        axes[1].set_title('احتمالية الفئات')
        axes[1].set_xlim(0, 1)
        
        # إضافة القيم
        for bar, prob in zip(bars, probabilities):
            axes[1].text(min(prob + 0.01, 0.99), bar.get_y() + bar.get_height()/2,
                        f'{prob:.3f}', va='center', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"تم حفظ التصور في: {save_path}")

        if show:
            plt.show()

        plt.close(fig)

        return result
    
    def ensemble_predict(self, image_path, cnn_path=None, vit_path=None):
        """توقع باستخدام كلا النموذجين"""
        # إنشاء المصنفين
        cnn_classifier = SatelliteClassifier('cnn', cnn_path)
        vit_classifier = SatelliteClassifier('vit', vit_path)
        
        # التوقعات
        cnn_result, _ = cnn_classifier.predict(image_path)
        vit_result, _ = vit_classifier.predict(image_path)
        
        if not cnn_result or not vit_result:
            print("خطأ في التوقع")
            return None
        
        print(f"\nنتائج CNN:")
        print(f"  الفئة: {cnn_result['class_name']}")
        print(f"  الثقة: {cnn_result['confidence']:.1f}%")
        
        print(f"\nنتائج Transformer:")
        print(f"  الفئة: {vit_result['class_name']}")
        print(f"  الثقة: {vit_result['confidence']:.1f}%")
        
        # المتوسط المرجح
        avg_confidence = (cnn_result['confidence'] + vit_result['confidence']) / 2
        
        # إذا اتفق النموذجان
        if cnn_result['class_index'] == vit_result['class_index']:
            final_class = cnn_result['class_index']
            print(f"\nالنماذج متفقان!")
        else:
            # اختيار النموذج الأعلى ثقة
            if cnn_result['confidence'] > vit_result['confidence']:
                final_class = cnn_result['class_index']
                print(f"\nCNN له ثقة أعلى")
            else:
                final_class = vit_result['class_index']
                print(f"\nTransformer له ثقة أعلى")
        
        final_result = {
            'class_index': final_class,
            'class_name': self.CLASS_NAMES[final_class],
            'confidence': avg_confidence,
            'cnn_prediction': cnn_result['class_name'],
            'cnn_confidence': cnn_result['confidence'],
            'vit_prediction': vit_result['class_name'],
            'vit_confidence': vit_result['confidence']
        }
        
        print(f"\nالنتيجة النهائية: {final_result['class_name']} ({final_result['confidence']:.1f}%)")
        
        return final_result

def main():
    parser = argparse.ArgumentParser(description='تصنيف صور الأقمار الصناعية')

    parser.add_argument('--mode', type=str, default='predict',
                        choices=['predict', 'ensemble', 'train'],
                        help='وضع التشغيل')
    parser.add_argument('--model', type=str, default='cnn',
                        choices=['cnn', 'vit'],
                        help='نوع النموذج')
    parser.add_argument('--image', type=str,
                        help='مسار الصورة للتوقع')
    parser.add_argument('--model_path', type=str,
                        help='مسار أوزان النموذج')
    parser.add_argument('--save_path', type=str, default=None,
                        help='مسار حفظ التصور (إذا أردت الحفظ بدلاً من العرض)')
    parser.add_argument('--no_show', action='store_true',
                        help='إذا تم تمريرها، لن يتم استدعاء plt.show() (مفيد للـ headless)')
    parser.add_argument('--cnn_path', type=str, default='./results/cnn_best.pth',
                        help='مسار نموذج CNN')
    parser.add_argument('--vit_path', type=str, default='./results/vit_best.pth',
                        help='مسار نموذج Transformer')

    args = parser.parse_args()

    print("="*60)
    print("نظام تصنيف صور الأقمار الصناعية")
    print("="*60)

    if args.mode == 'train':
        print("تشغيل التدريب...")
        os.system(f"python train.py --model {args.model}")

    elif args.mode == 'predict':
        if not args.image:
            print("يرجى تحديد مسار الصورة باستخدام --image")
            return

        print(f"\nالتوقع باستخدام {args.model}...")
        classifier = SatelliteClassifier(args.model, args.model_path)
        # show unless user requested no_show; pass save_path if provided
        result = classifier.visualize_prediction(args.image, save_path=args.save_path, show=not args.no_show)

        if result:
            print(f"\nالنتيجة: {result['class_name']} ({result['confidence']:.1f}%)")

    elif args.mode == 'ensemble':
        if not args.image:
            print("يرجى تحديد مسار الصورة باستخدام --image")
            return

        print("\nالتوقع باستخدام كلا النموذجين...")
        classifier = SatelliteClassifier('cnn')
        result = classifier.ensemble_predict(args.image, args.cnn_path, args.vit_path)

    else:
        print("وضع غير معروف")

if __name__ == "__main__":
    main()