# 基于 YOLOv8 和 LPRNet 的车牌识别系统

本项目旨在实现一个高效的车牌检测与识别系统。利用 YOLOv8 进行车牌区域的检测，并结合 LPRNet 进行车牌字符的精确识别。此外，项目提供了一个用户友好的图形用户界面（GUI），方便用户上传图片进行实时识别。

## 1. 项目概览

* **车牌检测:** 使用 **YOLOv8n** (Nano 版本) 进行训练，以实现快速准确的车牌边界框检测。
* **车牌识别:** 采用 **LPRNet** (License Plate Recognition Network) 对检测到的车牌区域进行字符识别。
* **训练脚本:** 提供了 YOLOv8 和 LPRNet 的训练脚本，支持自定义数据集和训练参数。
* **GUI 界面:** 提供一个基于 Tkinter 的图形界面，用户可以方便地上传图片，查看车牌检测框和识别结果。

## 2. 环境搭建

在运行本项目之前，请确保你的系统满足以下要求并安装必要的依赖。

## 3. 数据集准备
本项目使用的车牌数据集是 CCPD2020 数据集，CCPD2020 数据集是一个大规模的中国车牌数据集，包含多种天气、光照条件下的车牌图像。

## 4. 模型训练
### 4.1 YOLOv8 训练
编辑 train_yolo.py 中的 data="data-bvn.yaml"，确保 data-bvn.yaml 文件正确配置了你的 YOLOv8 格式的数据集路径（训练集、验证集和类别信息）。

运行 YOLOv8 训练：
python train_yolo.py

### 4.2 LPRNet 训练
编辑 train_lpr_try620.py，确认 train_img_dirs, val_img_dirs, test_img_dirs 指向处理好的 CCPD2020 数据集的对应目录。

注意： LPRNet 训练脚本中集成了日志记录、精度计算、学习率调度器（ReduceLROnPlateau）和早停机制，以优化训练过程和结果可视化。

运行 LPRNet 训练：

python train_lpr_try620.py


## 5. 使用 GUI 进行车牌识别

### 5.1 配置模型路径和字体

打开 GUI 识别主程序文件（例如 `jicheng_sys.py`），更新以下配置参数为你的模型文件和中文字体文件的**绝对路径**：


#### YOLOv8 模型路径

YOLO_MODEL_PATH = r"E:\translation\autodl_results\yolo11n测车牌结果\weights\best.pt" # 替换为你的YOLOv8模型路径

#### LPRNet 模型路径

LPRNET_MODEL_PATH = r"E:\translation\LPRNet_Pytorch-master\weights\Final_LPRNet_model.pth" # 替换为你的LPRNet模型路径


#### 用于在图像上绘制中文的字体路径 (非常重要!)

FONT_PATH = r"E:\translation\LPRNet_Pytorch-master\data\NotoSansCJK-Regular.ttc" # 替换为你的中文字体路径

#### 如果没有NotoSansCJK-Regular.ttc，在Windows上可以尝试

FONT_PATH = "C:\Windows\Fonts\simsun.ttc"


### 5.2 GUI 界面操作
标题: 界面顶部显示 "基于YOLOv8和LPRNet的车牌号识别系统"。

上传图片: 点击界面中的 "上传图片" 按钮，选择一张包含车牌的图片文件。

状态提示: 界面下方的状态栏会实时显示操作进度，例如“正在加载模型...”、“正在进行车牌检测和识别...”、“车牌识别完成！”等。

进度条: 在识别过程中，进度条会动态更新，表示当前任务的完成比例。

图片显示: 选择图片后，左侧区域会显示被处理的图片。如果检测到车牌，车牌位置会用绿色矩形框标注，并在框上方显眼地显示红色的车牌识别结果和YOLOv8置信度。

识别结果: 右侧的“识别结果”区域会详细列出所有检测到的车牌的识别结果、YOLOv8检测置信度和精确的边界框坐标。

# 6. 最终呈现效果
![image](https://github.com/user-attachments/assets/df66b641-e7d4-4b71-b55d-9737f423bb4b)
![image](https://github.com/user-attachments/assets/0be2e6c8-b659-497d-b933-1d41101e882c)
![c59e0d6fa9bd4bdf4fcb5ab5a35e933](https://github.com/user-attachments/assets/7785a09a-2c02-4416-afe0-e77455da65d0)
