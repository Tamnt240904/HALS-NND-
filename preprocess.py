#!/usr/bin/env python3
"""
Script tiền xử lý dataset Imagenette
- Tải dataset từ Kaggle
- Giải nén
- Gộp train/val và đổi tên thư mục theo class name
- Tạo subset dữ liệu (10 ảnh/class)
- Đổi tên ảnh trong subset thành 1.jpeg, 2.jpeg...
"""

import os
import shutil
import subprocess
import random
import kagglehub


def run_command(cmd):
    """Chạy shell command"""
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def count_images(root):
    """Đếm số lượng ảnh trong thư mục"""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
    count = 0

    for dirpath, dirnames, filenames in os.walk(root):
        for f in filenames:
            if os.path.splitext(f.lower())[1] in exts:
                count += 1

    return count


def create_subset(src_root, dst_root, num_samples=10):
    """
    Tạo folder data/subset chứa một lượng nhỏ ảnh ngẫu nhiên từ dataset gốc
    """
    print(f"\n=== Bước 8: Tạo subset ({num_samples} ảnh/class) ===")
    
    if not os.path.exists(dst_root):
        os.makedirs(dst_root)

    print(f"Source: {src_root}")
    print(f"Destination: {dst_root}")

    for class_folder in os.listdir(src_root):
        src_class_path = os.path.join(src_root, class_folder)
        
        if os.path.isdir(src_class_path):
            dst_class_path = os.path.join(dst_root, class_folder)
            os.makedirs(dst_class_path, exist_ok=True)
            
            files = [f for f in os.listdir(src_class_path) if os.path.isfile(os.path.join(src_class_path, f))]
            selected_files = random.sample(files, min(len(files), num_samples))
            
            for file_name in selected_files:
                shutil.copy2(os.path.join(src_class_path, file_name), 
                             os.path.join(dst_class_path, file_name))
            
            print(f"  - Đã copy {len(selected_files)} ảnh cho class: {class_folder}")

    print(f"-> Đã tạo xong data/subset.")


def rename_subset_images(root_dir):
    """
    Đổi tên các file trong subset thành 1.jpeg, 2.jpeg...
    """
    print("\n=== Bước 9: Đổi tên ảnh trong subset (1.jpeg, 2.jpeg...) ===")
    
    # Duyệt qua từng folder con (từng class)
    for class_folder in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_folder)
        
        # Kiểm tra xem có phải là folder không
        if os.path.isdir(class_path):
            # Lấy danh sách tất cả các file trong folder đó
            files = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
            
            # Sắp xếp danh sách file
            files.sort()
            
            print(f"  - Đang xử lý: {class_folder} ({len(files)} ảnh)")
            
            for idx, filename in enumerate(files):
                # Tạo tên mới: 1.jpeg, 2.jpeg, ...
                new_name = f"{idx + 1}.jpeg"
                
                old_file_path = os.path.join(class_path, filename)
                new_file_path = os.path.join(class_path, new_name)
                
                # Thực hiện đổi tên
                if old_file_path != new_file_path:
                    # Trong trường hợp file đích đã tồn tại (ví dụ chạy lại script), force overwrite hoặc bỏ qua tùy logic
                    # Ở đây dùng os.rename cơ bản
                    os.rename(old_file_path, new_file_path)

    print("-> Hoàn tất đổi tên.")


def main():
    # 1. Tạo thư mục data
    print("=== Bước 1: Tạo thư mục data ===")
    os.makedirs("data", exist_ok=True)

    # 2. Download dataset từ Kaggle
    print("\n=== Bước 2: Download dataset ===")
    path = kagglehub.dataset_download("jhoward/imagenette-160-px")
    print(f"Path to dataset files: {path}")
    
    if os.path.exists("./data/imagenette"):
        shutil.rmtree("./data/imagenette")
    run_command(f"mv {path} ./data/imagenette")

    # 3. Giải nén
    print("\n=== Bước 3: Giải nén dataset ===")
    run_command("tar -xvf data/imagenette/imagenette-160.tgz -C data/imagenette --strip-components 1")

    # 4. Đếm số ảnh trong train
    print("\n=== Bước 4: Đếm số ảnh ===")
    root_folder = "data/imagenette/train"
    print(f"Số lượng ảnh: {count_images(root_folder)}")

    # 5. Gộp và đổi tên thư mục
    print("\n=== Bước 5: Gộp train/val và đổi tên ===")
    
    class_map = {
        'n01440764': 'tench',
        'n02102040': 'English_springer',
        'n02979186': 'cassette_player',
        'n03000684': 'chain_saw',
        'n03028079': 'church',
        'n03394916': 'French_horn',
        'n03417042': 'garbage_truck',
        'n03425413': 'gas_pump',
        'n03445777': 'golf_ball',
        'n03888257': 'parachute'
    }

    base_dir = 'data/imagenette'
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')

    print(f"Bắt đầu gộp và đổi tên tại: {base_dir}")

    for class_code, class_name in class_map.items():
        source_train = os.path.join(train_dir, class_code)
        source_val = os.path.join(val_dir, class_code)
        target_dir = os.path.join(base_dir, class_name)
        
        os.makedirs(target_dir, exist_ok=True)
        
        if os.path.isdir(source_train):
            for filename in os.listdir(source_train):
                shutil.copy(os.path.join(source_train, filename), os.path.join(target_dir, filename))
        
        if os.path.isdir(source_val):
            for filename in os.listdir(source_val):
                shutil.copy(os.path.join(source_val, filename), os.path.join(target_dir, filename))
                
        print(f"[XONG] Đã gộp {class_code} -> {class_name}")

    # 6. Di chuyển thư mục train/val gốc
    print("\n=== Bước 6: Lưu trữ thư mục gốc ===")
    os.makedirs("data/original", exist_ok=True)
    
    if os.path.exists("data/imagenette/train"):
        run_command("mv data/imagenette/train data/original/")
    if os.path.exists("data/imagenette/val"):
        run_command("mv data/imagenette/val data/original/")

    # 7. Xóa file nén
    print("\n=== Bước 7: Xóa file nén ===")
    if os.path.exists("data/imagenette/imagenette-160.tgz"):
        run_command("rm -rf data/imagenette/imagenette-160.tgz")

    # 8. Tạo subset
    subset_root = os.path.join('data', 'subset')
    # Xóa subset cũ nếu có để đảm bảo sạch sẽ trước khi tạo mới và đổi tên
    if os.path.exists(subset_root):
        shutil.rmtree(subset_root)
        
    src_root = os.path.join('data', 'imagenette')
    create_subset(src_root, subset_root, num_samples=10)

    # 9. Đổi tên ảnh trong subset
    rename_subset_images(subset_root)

    print("\n✅ Hoàn tất! Dữ liệu đã sẵn sàng.")


if __name__ == "__main__":
    main()