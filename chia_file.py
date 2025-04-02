import os
import random

# Đường dẫn đến thư mục images và labels
images_folder = r"C:\Users\OS\Desktop\ActionProject\datasets\images"  
labels_folder = r"C:\Users\OS\Desktop\ActionProject\datasets\labels"  
output_folder = r"C:\Users\OS\Desktop\ActionProject\datasets"       

# Tỷ lệ chia tập train và val (có thể thay đổi)
train_ratio = 0.90  # 80% cho train, 20% cho val

# Đảm bảo thư mục đầu ra tồn tại
os.makedirs(output_folder, exist_ok=True)

# Lấy danh sách file ảnh
image_files = [f for f in os.listdir(images_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Kiểm tra file nhãn tương ứng và tạo danh sách hợp lệ
valid_image_files = []
for img_file in image_files:
    label_file = os.path.splitext(img_file)[0] + ".txt"
    label_path = os.path.join(labels_folder, label_file)
    if os.path.exists(label_path):
        valid_image_files.append(img_file)
    else:
        print(f"⚠️ Không tìm thấy file nhãn cho ảnh: {img_file}")

# Số lượng file hợp lệ
total_images = len(valid_image_files)
print(f"✅ Tổng số ảnh hợp lệ: {total_images}")

# Xáo trộn danh sách ảnh để chia ngẫu nhiên
random.shuffle(valid_image_files)

# Tính số lượng ảnh cho tập train và val
train_size = int(total_images * train_ratio)
train_files = valid_image_files[:train_size]
val_files = valid_image_files[train_size:]

# Định dạng đường dẫn theo yêu cầu
train_paths = [f"./images/{img_file}" for img_file in train_files]
val_paths = [f"./images/{img_file}" for img_file in val_files]

# Ghi vào file train2017.txt
train_txt_path = os.path.join(output_folder, "train2017.txt")
with open(train_txt_path, 'w') as f:
    f.write('\n'.join(train_paths))
print(f"✅ Đã tạo file {train_txt_path} với {len(train_paths)} đường dẫn.")

# Ghi vào file val2017.txt
val_txt_path = os.path.join(output_folder, "val2017.txt")
with open(val_txt_path, 'w') as f:
    f.write('\n'.join(val_paths))
print(f"✅ Đã tạo file {val_txt_path} với {len(val_paths)} đường dẫn.")