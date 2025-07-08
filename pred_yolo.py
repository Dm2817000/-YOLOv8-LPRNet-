from ultralytics import YOLO
import cv2

# 1. 加载训练好的模型
model = YOLO(r'E:\translation\ultralytics-main\runs\detect\train3\weights\best.pt')  # 替换为你的实际路径

# 2. 预测单张图片
results = model.predict(
    source=r'E:\translation\ultralytics-main\pred.png',  # 图片路径
    conf=0.25,  # 置信度阈值 (可调整)
    save=True,  # 自动保存结果
    show_labels=True,  # 显示标签
    show_conf=True,  # 显示置信度
    line_width=1  # 边界框线宽
)

# 3. 高级：实时显示预测结果（可选）
for result in results:
    # 获取带标注的图像（numpy数组）
    annotated_frame = result.plot()

    # 显示图像
    cv2.imshow("YOLOv8 Prediction", annotated_frame)
    cv2.waitKey(0)  # 按任意键关闭

cv2.destroyAllWindows()