import cv2
import albumentations as A
import os
import numpy as np

# Definisikan pipeline augmentasi
augmentations = A.Compose([
    A.Rotate(limit=10, p=0.5),          # Rotasi kecil hingga 10 derajat
    A.HorizontalFlip(p=0.5),            # Flip horizontal
    A.RandomBrightnessContrast(p=0.3),  # Ubah kecerahan dan kontras
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=5, p=0.5),  # Geser dan skala kecil
    A.GaussNoise(p=0.2),                # Tambahkan noise kecil
])

# Fungsi untuk augmentasi gambar
def augment_image(image_path, output_dir, num_augmentations=5):
    # Baca gambar
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Nama file tanpa ekstensi
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Simpan gambar asli
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_original.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    # Lakukan augmentasi
    for i in range(num_augmentations):
        augmented = augmentations(image=image)['image']
        output_path = os.path.join(output_dir, f"{base_name}_aug_{i}.jpg")
        cv2.imwrite(output_path, cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))

# Proses semua subfolder
input_dir = "Uang_Baru"  # Folder utama
output_base_dir = "Uang_Baru_Augmented"  # Folder output utama
os.makedirs(output_base_dir, exist_ok=True)

# Iterasi melalui setiap subfolder
for class_folder in os.listdir(input_dir):
    class_path = os.path.join(input_dir, class_folder)
    if os.path.isdir(class_path):
        output_class_dir = os.path.join(output_base_dir, class_folder)
        os.makedirs(output_class_dir, exist_ok=True)
        
        # Proses setiap gambar di subfolder
        for filename in os.listdir(class_path):
            if filename.endswith((".jpg", ".png")):
                image_path = os.path.join(class_path, filename)
                augment_image(image_path, output_class_dir, num_augmentations=5)

print("Augmentasi selesai!")