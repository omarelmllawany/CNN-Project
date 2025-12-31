"""
معالجة وتحضير البيانات
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

class EuroSATDataset(Dataset):
    """مجموعة بيانات EuroSAT"""
    
    # فئات EuroSAT
    CLASSES = [
        'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
        'Industrial', 'Pasture', 'PermanentCrop', 'Residential',
        'River', 'SeaLake'
    ]
    
    def __init__(self, root_dir, split='train', transform=None, 
                 train_size=0.7, val_size=0.15, seed=42):
        """
        Args:
            root_dir: المسار الرئيسي للبيانات
            split: 'train', 'val', أو 'test'
            transform: تحويلات البيانات
            train_size: نسبة التدريب
            val_size: نسبة التحقق
            seed: seed عشوائي
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # تحميل البيانات
        self.images, self.labels = self._load_data()
        
        # تقسيم البيانات
        self._split_data(train_size, val_size, seed)
        
        print(f"{split} set: {len(self.images)} images")
    
    def _load_data(self):
        """تحميل الصور والتسميات"""
        images = []
        labels = []
        
        eurosat_path = os.path.join(self.root_dir, "EuroSAT")
        
        # إذا لم يكن EuroSAT موجوداً، استخدم CIFAR-10 كبديل
        if not os.path.exists(eurosat_path):
            print("EuroSAT not found. Creating placeholder data...")
            # إنشاء بيانات وهمية للاختبار
            for class_idx in range(10):
                for i in range(100):
                    images.append(f"dummy_{class_idx}_{i}.jpg")
                    labels.append(class_idx)
            return images, labels
        
        # تحميل البيانات الحقيقية
        for class_idx, class_name in enumerate(self.CLASSES):
            class_dir = os.path.join(eurosat_path, class_name)
            
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.tif')):
                        img_path = os.path.join(class_dir, img_name)
                        images.append(img_path)
                        labels.append(class_idx)
            else:
                # بحث في المجلدات الفرعية
                for root, dirs, files in os.walk(eurosat_path):
                    if class_name in root:
                        for img_name in files:
                            if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.tif')):
                                img_path = os.path.join(root, img_name)
                                images.append(img_path)
                                labels.append(class_idx)
        
        return images, labels
    
    def _split_data(self, train_size, val_size, seed):
        """تقسيم البيانات"""
        if len(self.images) == 0:
            return
        
        # تقسيم التدريب والباقي
        train_idx, temp_idx = train_test_split(
            range(len(self.images)),
            train_size=train_size,
            stratify=self.labels,
            random_state=seed
        )
        
        # تقسيم الباقي إلى تحقق واختبار
        val_ratio = val_size / (1 - train_size)
        val_idx, test_idx = train_test_split(
            temp_idx,
            train_size=val_ratio,
            stratify=[self.labels[i] for i in temp_idx],
            random_state=seed
        )
        
        # اختيار البيانات بناءً على split
        if self.split == 'train':
            indices = train_idx
        elif self.split == 'val':
            indices = val_idx
        else:  # 'test'
            indices = test_idx
        
        self.images = [self.images[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # إذا كانت بيانات وهمية، إنشاء صورة عشوائية
        if "dummy" in str(self.images[idx]):
            # إنشاء صورة عشوائية
            img_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            image = Image.fromarray(img_array)
        else:
            # تحميل الصورة الحقيقية
            image = Image.open(self.images[idx]).convert('RGB')
        
        label = self.labels[idx]
        
        # تطبيق التحويلات
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_distribution(self):
        """الحصول على توزيع الفئات"""
        from collections import Counter
        return dict(Counter(self.labels))

def get_transforms(augment=True, img_size=64):
    """الحصول على تحويلات البيانات"""
    if augment:
        # تحويلات التدريب (مع زيادة البيانات)
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        # تحويلات التحقق/الاختبار
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

def create_dataloaders(data_dir='./data', batch_size=32, img_size=64):
    """إنشاء DataLoaders"""
    
    # تحويلات مختلفة لكل مجموعة
    train_transform = get_transforms(augment=True, img_size=img_size)
    val_transform = get_transforms(augment=False, img_size=img_size)
    test_transform = get_transforms(augment=False, img_size=img_size)
    
    # مجموعات البيانات
    train_dataset = EuroSATDataset(
        data_dir, split='train', transform=train_transform
    )
    
    val_dataset = EuroSATDataset(
        data_dir, split='val', transform=val_transform
    )
    
    test_dataset = EuroSATDataset(
        data_dir, split='test', transform=test_transform
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True  # num_workers=0 لتجنب المشاكل في Windows
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

# اختبار مجموعة البيانات
if __name__ == "__main__":
    print("اختبار مجموعة البيانات...")
    
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir='./data', batch_size=4, img_size=64
    )
    
    # الحصول على دفعة عينة
    for images, labels in train_loader:
        print(f"شكل الصور: {images.shape}")
        print(f"شكل التسميات: {labels.shape}")
        print(f"قيم التسميات: {labels}")
        
        # توزيع الفئات
        dataset = train_loader.dataset
        class_dist = dataset.get_class_distribution()
        print(f"\nتوزيع الفئات: {class_dist}")
        
        break
    
    print("\n✓ مجموعة البيانات تعمل بشكل صحيح!")