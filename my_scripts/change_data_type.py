import os
import cv2
# 根目录
root_dir = r"D:\py\mot_data"
dataset_name = "MOT20-01"

# 设置输入输出路径
image_dir = os.path.join(root_dir, dataset_name, "img1")  # 图片存放路径
gt_file = os.path.join(root_dir, dataset_name, "gt", "gt.txt")  # 标注文件路径
label_dir = os.path.join(root_dir, "labels", dataset_name)  # YOLO 格式标签存放路径

# 创建标签存放文件夹
os.makedirs(label_dir, exist_ok=True)

# 读取 gt.txt 并转换为 YOLO 格式
with open(gt_file, "r") as f:
    lines = f.readlines()

for line in lines:
    data = line.strip().split(',')
    frame_id, obj_id, x, y, w, h, conf, cls, visibility = map(float, data)

    # 只保留行人（person 类别）
    if cls != 1:
        continue

    # 生成对应的图片文件名
    img_path = os.path.join(image_dir, f"{int(frame_id):06d}.jpg")
    if not os.path.exists(img_path):
        continue

    # 读取图片尺寸
    img = cv2.imread(img_path)
    img_h, img_w = img.shape[:2]

    # 转换为 YOLO 格式
    center_x = (x + w / 2) / img_w
    center_y = (y + h / 2) / img_h
    width = w / img_w
    height = h / img_h

    # 生成 YOLO 标签路径
    label_dir1=os.path.join(label_dir,"yolo_label")
    label_path = os.path.join(label_dir1, f"{int(frame_id):06d}.txt")

    # 写入 YOLO 格式标签
    with open(label_path, "a") as label_file:
        label_file.write(f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")

print("MOT20-01 数据集转换完成！")
