from ultralytics import YOLO

# 加载你训练好的模型
model = YOLO('/home/caviar/my_yolo_project/runs/detect/runs/train/my_first_train/weights/best.pt')

# 对一张图片进行预测
results = model('/mnt/e/BIpedX/第三周任务提交/Image_training_dataset/test_image.png')  # 替换为你的图片路径

# 显示结果（如果有图形界面）
results[0].show()

# 保存结果图片
results[0].save('result.jpg')