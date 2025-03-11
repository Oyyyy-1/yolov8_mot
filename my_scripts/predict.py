from ultralytics import YOLO

# 加载模型
model = YOLO(r'C:\Users\86178\Desktop\结果\3080\train4\weights\best.pt')

# 对测试集进行推理  myscripts/runs/detect （检测后的带框结果图和标签）
# results = model.predict(source=r'D:\py\mot_data\all_data\images\test', save=True, save_txt=False,
#                         show_labels=True, show_conf=True)

results = model.predict(source=r'D:\py\mot_data\all_data\images\test', save=True, save_txt=False,
                        show_labels=False, show_conf=True, line_thickness=2)
# 查看结果
for result in results:
    print(result)