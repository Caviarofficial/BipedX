import os
import yaml
from ultralytics import YOLO

# -------------------- 第一步：检查图片路径是否存在 --------------------
train_path = '/mnt/e/BIpedX/第三周任务提交/Image_training_dataset/images/train'
val_path = '/mnt/e/BIpedX/第三周任务提交/Image_training_dataset/images/val'

print(f"检查训练集路径: {train_path}")
if os.path.exists(train_path):
    images = os.listdir(train_path)
    print(f"✅ 找到 {len(images)} 张训练图片")
else:
    print(f"❌ 训练集路径不存在！请检查路径是否正确。")
    exit()

print(f"检查验证集路径: {val_path}")
if os.path.exists(val_path):
    images = os.listdir(val_path)
    print(f"✅ 找到 {len(images)} 张验证图片")
else:
    print(f"❌ 验证集路径不存在！请检查路径是否正确。")
    exit()

# -------------------- 第二步：定义数据集配置字典 --------------------
data = {
    'train': train_path,
    'val': val_path,
    'nc': 12,
    'names': ['painter', 'cipher machine', 'antique merchant', 'lizard', 'prospector',
              'wax sculptor', 'painting', 'mercenary', 'priest', 'lawyer',
              'little girl', 'acrobat']
}

# -------------------- 第三步：将字典写入一个临时的 YAML 文件 --------------------
yaml_file = 'temp_dataset.yaml'
with open(yaml_file, 'w') as f:
    yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
print(f"已生成临时配置文件: {yaml_file}")

# -------------------- 第四步：加载模型并开始训练 --------------------
model = YOLO('yolov8n.pt')  # 模型文件会自动下载，如已存在则直接使用

results = model.train(
    data=yaml_file,        # 关键：传入的是临时 YAML 文件的路径
    epochs=10,             # 先用 10 轮测试，成功后可以增加
    imgsz=640,
    batch=8,
    device='cpu',
    workers=2,
    project='runs/train',
    name='my_first_train',
    exist_ok=True,
    verbose=True
)