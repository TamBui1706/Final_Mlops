import cv2
import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import os

# --- CẤU HÌNH ---
# Đường dẫn ảnh bạn đã cung cấp
IMAGE_PATH = r"E:\MLOps\Final\RiceLeafsDisease\train\healthy\healthy (1).jpg"

# --- ĐỊNH NGHĨA TRANSFORM ---
# FIX LỖI: Thay height=..., width=... bằng size=(height, width) cho bản Albumentations mới
transform = A.Compose([
    # Resize ảnh về 256x256 trước
    A.Resize(height=256, width=256),
    
    # FIX: Dùng tham số 'size' thay vì height/width riêng lẻ
    A.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0), p=1.0),
    
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.7),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
])

def visualize_augmentation(image_path, num_examples=5):
    """
    Hiển thị ảnh gốc và các biến thể sau khi augmentation
    """
    # 1. Kiểm tra và Đọc ảnh
    if os.path.exists(image_path):
        # Đọc ảnh bằng OpenCV (BGR)
        image = cv2.imread(image_path)
        # Chuyển sang RGB để hiển thị đúng màu trên Matplotlib
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(f"✅ Đã đọc ảnh từ: {image_path}")
    else:
        print(f"⚠️ Không tìm thấy file: '{image_path}'")
        print("-> Đang tạo ảnh giả lập để demo...")
        image = np.zeros((500, 500, 3), dtype=np.uint8)
        image[:] = [34, 139, 34]  # Màu xanh lá
        cv2.circle(image, (250, 250), 50, (139, 69, 19), -1) # Đốm nâu giả

    # 2. Tạo lưới hiển thị (1 dòng, N+1 cột)
    # Cột đầu tiên là ảnh gốc, các cột sau là ảnh Augment
    fig, axes = plt.subplots(1, num_examples + 1, figsize=(20, 6))

    # --- HIỂN THỊ ẢNH GỐC ---
    axes[0].imshow(image)
    axes[0].set_title("Original Image\n(Ảnh gốc)", fontweight='bold', color='blue')
    axes[0].axis('off')

    # --- HIỂN THỊ CÁC ẢNH ĐÃ AUGMENT ---
    for i in range(num_examples):
        try:
            # Áp dụng augmentation
            augmented = transform(image=image)['image']
            
            # Hiển thị
            axes[i+1].imshow(augmented)
            axes[i+1].set_title(f"Augmented {i+1}\n(Đã xử lý)")
            axes[i+1].axis('off')
        except Exception as e:
            print(f"Lỗi khi augment lần {i+1}: {e}")

    plt.tight_layout()
    plt.suptitle(f"Minh họa Data Augmentation (Albumentations)", fontsize=16, y=1.05)
    plt.show()

# --- CHẠY CHƯƠNG TRÌNH ---
if __name__ == "__main__":
    visualize_augmentation(IMAGE_PATH, num_examples=5)