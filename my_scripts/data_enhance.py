import os
import cv2
import numpy as np
from albumentations import (
    Compose, HorizontalFlip, GaussNoise, RandomBrightnessContrast,
    CLAHE, Affine
)
from albumentations.pytorch import ToTensorV2
from albumentations import BboxParams
import random


# 修正 bbox 坐标的辅助函数（与之前类似）
def fix_bbox(bbox):
    """
    修正 bbox 使其符合 YOLO 格式要求：所有值在 [0,1] 内
    bbox 格式: [x_center, y_center, width, height]
    同时确保 x_min, x_max, y_min, y_max 均在 [0,1] 内。
    """
    x_center, y_center, width, height = bbox

    # 计算当前的 x_min, x_max, y_min, y_max
    x_min = x_center - width / 2
    x_max = x_center + width / 2
    y_min = y_center - height / 2
    y_max = y_center + height / 2

    # 调整 x 坐标
    if x_max > 1.0:
        diff = x_max - 1.0
        width = max(0.0, width - diff)
        x_center = 1.0 - width / 2
    if x_min < 0.0:
        diff = -x_min
        width = max(0.0, width - diff)
        x_center = width / 2

    # 调整 y 坐标
    if y_max > 1.0:
        diff = y_max - 1.0
        height = max(0.0, height - diff)
        y_center = 1.0 - height / 2
    if y_min < 0.0:
        diff = -y_min
        height = max(0.0, height - diff)
        y_center = height / 2

    # 最终确保 x_center, y_center, width, height 均在 [0,1] 内
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    width = max(0.0, min(1.0, width))
    height = max(0.0, min(1.0, height))

    return [x_center, y_center, width, height]



# 预处理标签：遍历标签目录，对所有标签进行修正，并输出到新目录
# def preprocess_labels(input_labels_dir, output_labels_dir):
#     if not os.path.exists(output_labels_dir):
#         os.makedirs(output_labels_dir)
#
#     label_files = [f for f in os.listdir(input_labels_dir) if f.endswith('.txt')]
#     for label_file in label_files:
#         input_label_path = os.path.join(input_labels_dir, label_file)
#         output_label_path = os.path.join(output_labels_dir, label_file)
#         with open(input_label_path, 'r') as fin, open(output_label_path, 'w') as fout:
#             lines = fin.readlines()
#             new_lines = []
#             for line in lines:
#                 parts = line.strip().split()
#                 if len(parts) != 5:
#                     continue  # 跳过格式不正确的行
#                 class_id = parts[0]
#                 bbox = list(map(float, parts[1:]))
#                 fixed = fix_bbox(bbox)
#                 new_line = f"{class_id} {fixed[0]:.6f} {fixed[1]:.6f} {fixed[2]:.6f} {fixed[3]:.6f}\n"
#                 new_lines.append(new_line)
#             fout.writelines(new_lines)
#         print(f"[INFO] 预处理完毕: {label_file}")


# 获取数据增强策略（不变）
def get_augmentation():
    return Compose([
        HorizontalFlip(p=0.5),
        GaussNoise(p=0.5, var_limit=(6.4, 25.6)),
        RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.1, p=0.25),
        CLAHE(p=0.25),
        Affine(p=0.25, keep_ratio=True, translate_percent=(-0.15, 0.15), scale=(0.6, 1.4)),
        ToTensorV2()
    ], bbox_params=BboxParams(format='yolo', label_fields=['labels']))


# 数据增强阶段，与之前类似，但使用预处理后的标签
def apply_augmentation(image_path, label_path, augmentations, images_output_dir, labels_output_dir, i, enhance_unripe):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with open(label_path, 'r') as f:
        labels = f.readlines()

    bboxes = []
    label_ids = []
    for label in labels:
        parts = label.strip().split()
        class_id = int(parts[0])
        # 此处根据需求判断是否需要增强某些类别（例如 enhance_unripe）
        if class_id == 1 and not enhance_unripe:
            continue
        else:
            x_center, y_center, width, height = map(float, parts[1:])
            bboxes.append([x_center, y_center, width, height])
            label_ids.append(class_id)

    if len(label_ids) > 0:
        augmented = augmentations(image=image, bboxes=bboxes, labels=label_ids)
        augmented_image = augmented['image']
        augmented_bboxes = augmented['bboxes']
        augmented_labels = augmented['labels']

        # 如果需要在增强后再做一次修正（可选），可以添加如下：
        for j in range(len(augmented_bboxes)):
            augmented_bboxes[j] = fix_bbox(augmented_bboxes[j])

        head, tail = os.path.splitext(os.path.basename(image_path))
        image_file_name = f"{head}_{i}{tail}"
        label_file_name = f"{head}_{i}.txt"

        output_image_path = os.path.join(images_output_dir, image_file_name)
        augmented_image = augmented_image.numpy().transpose(1, 2, 0)
        augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_image_path, augmented_image)

        augmented_labels_str = []
        for j, bbox in enumerate(augmented_bboxes):
            class_id = int(augmented_labels[j])
            x_center, y_center, width, height = bbox
            augmented_labels_str.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        output_label_path = os.path.join(labels_output_dir, label_file_name)
        with open(output_label_path, 'w') as f:
            f.writelines(augmented_labels_str)


def augment_dataset(images_dir, input_labels_dir, output_dir, augmentations, num_augmentations=7, p=0.6):
    # 输出目录设置
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images_output_dir = os.path.join(output_dir, 'images')
    labels_output_dir = os.path.join(output_dir, 'labels')
    if not os.path.exists(images_output_dir):
        os.makedirs(images_output_dir)
    if not os.path.exists(labels_output_dir):
        os.makedirs(labels_output_dir)

    label_files = [f for f in os.listdir(input_labels_dir) if f.endswith('.txt')]
    random.shuffle(label_files)
    selected_num = int(len(label_files) * p)
    selected_label_files = label_files[:selected_num]

    count = 0
    for label_file in selected_label_files:
        label_path = os.path.join(input_labels_dir, label_file)
        if count >= 100:
            enhance_unripe = False
        else:
            with open(label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if line.strip().startswith('1'):
                        count += 1
            enhance_unripe = True

        image_path = os.path.join(images_dir, label_file.replace('.txt', '.jpg'))
        if os.path.exists(image_path):
            for i in range(num_augmentations):
                apply_augmentation(image_path, label_path, augmentations, images_output_dir, labels_output_dir, i,
                                   enhance_unripe)
        else:
            image_path = os.path.join(images_dir, label_file.replace('.txt', '.png'))
            if os.path.exists(image_path):
                for i in range(num_augmentations):
                    apply_augmentation(image_path, label_path, augmentations, images_output_dir, labels_output_dir, i,
                                       enhance_unripe)


def main():
    # 原始数据目录
    images_dir = r'D:\py\mot_data\MOT20-01\img1'
    # original_labels_dir = r'D:\py\mot_data\labels\MOT20-01\yolo_label'
    # 预处理后的标签目录
    preprocessed_labels_dir = r'D:\py\mot_data\labels\MOT20-01\preprocessed_labels'
    # 增强后数据保存目录
    output_dir = r'D:\py\mot_data\enhance_data'

    # 预处理标签文件：修正所有超出范围的 bbox
    # preprocess_labels(original_labels_dir, preprocessed_labels_dir)

    # 获取增强策略
    augmentations = get_augmentation()

    # 使用预处理后的标签进行数据增强
    augment_dataset(images_dir, preprocessed_labels_dir, output_dir, augmentations, num_augmentations=1)


if __name__ == "__main__":
    main()
