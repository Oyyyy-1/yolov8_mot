import os
import random
import shutil
from sklearn.model_selection import train_test_split

def split_dataset(images_dir, labels_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):

    # 获取所有图像文件
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    random.seed(seed)
    random.shuffle(image_files)  # 随机打乱图像文件列表

    # 划分训练集、验证集和测试集
    train_files, temp_files = train_test_split(image_files, train_size=train_ratio, random_state=seed)
    val_files, test_files = train_test_split(temp_files, train_size=val_ratio / (val_ratio + test_ratio),
                                             random_state=seed)

    # 创建子文件夹
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)

    # 复制文件到对应子文件夹，原数据不修改
    def copy_files(file_list, split_name):
        for img_file in file_list:
            # 图像文件路径
            src_img = os.path.join(images_dir, img_file)
            dst_img = os.path.join(output_dir, 'images', split_name, img_file)
            # 对应的标签文件路径
            label_file = img_file.replace('.jpg', '.txt')
            src_label = os.path.join(labels_dir, label_file)
            dst_label = os.path.join(output_dir, 'labels', split_name, label_file)

            # 复制图像和标签文件到新文件夹
            if os.path.exists(src_label):  # 确保标签文件存在
                shutil.copy2(src_img, dst_img)
                shutil.copy2(src_label, dst_label)
            else:
                print(f"警告: {label_file} 不存在，跳过 {img_file}")

    copy_files(train_files, 'train')
    copy_files(val_files, 'val')
    copy_files(test_files, 'test')

    # 打印划分结果
    print(f"训练集: {len(train_files)} 张图像")
    print(f"验证集: {len(val_files)} 张图像")
    print(f"测试集: {len(test_files)} 张图像")

images_dir = r'D:\py\mot_data\MOT20-01\img1'
labels_dir = r'D:\py\mot_data\labels\MOT20-01\yolo_label'

# images_dir = r'D:\py\mot_data\all_data\images'  # 包括增强数据做划分
# labels_dir = r'D:\py\mot_data\all_data\labels'
output_dir = r'D:\py\mot_data\before_enhance'

if __name__ == "__main__":
    split_dataset(images_dir, labels_dir, output_dir)