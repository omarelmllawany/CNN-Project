"""
تنزيل مجموعة بيانات EuroSAT
"""

import os
import zipfile
import requests
from tqdm import tqdm
import shutil

def download_eurosat():
    """تنزيل مجموعة بيانات EuroSAT"""
    print("="*60)
    print("تنزيل مجموعة بيانات EuroSAT")
    print("="*60)
    
    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)
    
    # رابط تنزيل EuroSAT
    url = "http://madm.dfki.de/files/sentinel/EuroSAT.zip"
    zip_path = os.path.join(data_dir, "EuroSAT.zip")
    
    # تنزيل الملف
    print("\nجاري تنزيل مجموعة البيانات...")
    try:
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(zip_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, 
                     desc='Downloading', ncols=80) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    f.write(data)
                    pbar.update(len(data))
        
        print("✓ تم تنزيل الملف بنجاح")
        
        # استخراج الملف
        print("\nجاري استخراج الملفات...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        
        print("✓ تم استخراج الملفات بنجاح")
        
        # إعادة تنظيم المجلدات
        print("\nجاري تنظيم الملفات...")
        eurosat_path = os.path.join(data_dir, "EuroSAT")
        if os.path.exists(eurosat_path):
            # تحقق من الهيكل
            subdirs = [d for d in os.listdir(eurosat_path) 
                      if os.path.isdir(os.path.join(eurosat_path, d))]
            
            if len(subdirs) == 10:  # 10 فئات
                print("✓ الهيكل صحيح")
            else:
                # قد تحتاج إلى إعادة التنظيم
                for subdir in subdirs:
                    if subdir.endswith(".jpg"):
                        continue
                    # هذه مجلدات الفئات
                    print(f"  فئة: {subdir}")
        
        # حذف ملف الزيب
        os.remove(zip_path)
        print("✓ تم حذف ملف الزيب")
        
        print(f"\n✓ تم إعداد مجموعة البيانات في: {data_dir}")
        return True
        
    except Exception as e:
        print(f"\n✗ خطأ في التنزيل: {e}")
        print("\nيمكنك تنزيل البيانات يدوياً:")
        print("1. تفضل إلى: http://madm.dfki.de/files/sentinel/EuroSAT.zip")
        print("2. نزل الملف")
        print("3. استخرجه إلى: ./data/EuroSAT/")
        return False

def prepare_cifar10():
    """تحضير CIFAR-10 كبديل"""
    print("\n" + "="*60)
    print("تحضير CIFAR-10 كبديل مؤقت")
    print("="*60)
    
    import torchvision
    import torchvision.transforms as transforms
    
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    
    print("جاري تنزيل CIFAR-10...")
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    
    print(f"✓ تم تنزيل CIFAR-10")
    print(f"  بيانات التدريب: {len(train_dataset)} صورة")
    print(f"  بيانات الاختبار: {len(test_dataset)} صورة")
    
    # إنشاء مجلد EuroSAT وهمي ليتوافق مع الكود
    eurosat_dir = "./data/EuroSAT"
    os.makedirs(eurosat_dir, exist_ok=True)
    
    with open(os.path.join(eurosat_dir, "README.txt"), "w") as f:
        f.write("CIFAR-10 used as temporary dataset\n")
    
    print("\nملاحظة: تم استخدام CIFAR-10 كبديل مؤقت")
    print("يمكنك استخدام EuroSAT الحقيقي لاحقاً")
    return True

if __name__ == "__main__":
    print("إعداد مجموعة البيانات")
    print("1. محاولة تنزيل EuroSAT")
    print("2. استخدام CIFAR-10 كبديل")
    
    choice = input("\nاختر (1 أو 2): ").strip()
    
    if choice == "1":
        success = download_eurosat()
        if not success:
            print("\nالانتقال إلى الخيار 2...")
            prepare_cifar10()
    else:
        prepare_cifar10()
    
    print("\n" + "="*60)
    print("تم إعداد البيانات بنجاح!")
    print("يمكنك الآن تشغيل: python train.py")
    print("="*60)