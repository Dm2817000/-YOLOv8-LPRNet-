import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk # 导入 ImageTk
import os
import sys
import tkinter as tk
from tkinter import filedialog, ttk, messagebox # 导入 ttk for progressbar

# --- LPRNet 特定的导入和定义 ---
# 动态添加 LPRNet 项目根目录到 Python 路径
# 请根据您的实际LPRNet_Pytorch-master文件夹路径进行修改
lprnet_root = r"E:/translation/LPRNet_Pytorch-master"
if lprnet_root not in sys.path:
    sys.path.append(lprnet_root)
if os.path.join(lprnet_root, 'model') not in sys.path:
    sys.path.append(os.path.join(lprnet_root, 'model'))
if os.path.join(lprnet_root, 'data') not in sys.path:
    sys.path.append(os.path.join(lprnet_root, 'data'))

# 确保这些模块可以被找到
try:
    from model.LPRNet import build_lprnet
    from data.load_data import CHARS # 确保 CHARS 列表与您的 LPRNet 训练时使用的字符集一致
except ImportError as e:
    print(f"LPRNet 模块导入失败，请检查 lprnet_root 路径和文件结构是否正确: {e}")
    # 在GUI中，我们不直接sys.exit，而是弹出错误框
    messagebox.showerror("LPRNet 导入错误", f"LPRNet 模块导入失败，请检查 lprnet_root 路径和文件结构是否正确: {e}")
    sys.exit(1) # 如果是关键错误，仍然退出

# --- YOLOv8 特定的导入 ---
try:
    from ultralytics import YOLO
except ImportError as e:
    print(f"ultralytics 库导入失败，请确保您已安装: pip install ultralytics -> {e}")
    messagebox.showerror("Ultralytics 导入错误", f"ultralytics 库导入失败，请确保您已安装: pip install ultralytics -> {e}")
    sys.exit(1)

# --- 配置参数 ---
# YOLOv8 模型路径
YOLO_MODEL_PATH = r"E:\translation\autodl_results\yolo11n测车牌结果\weights\best.pt"
# LPRNet 模型路径
LPRNET_MODEL_PATH = r"E:\translation\LPRNet_Pytorch-master\weights\Final_LPRNet_model.pth"

# LPRNet 输入图像尺寸 (宽度, 高度)
LPR_INPUT_SIZE = [94, 24]
# LPRNet 最大车牌长度 (与训练时保持一致)
LPR_MAX_LEN = 8

# 用于在图像上绘制中文的字体路径
# 这是一个非常重要的参数，请确保路径正确且字体支持中文
FONT_PATH = r"E:\translation\LPRNet_Pytorch-master\data\NotoSansCJK-Regular.ttc"
if not os.path.exists(FONT_PATH):
    print(f"警告: 字体文件未找到于 {FONT_PATH}。中文字符显示可能不正确。")
    # 如果字体不存在，尝试使用系统默认字体，但可能不支持中文
    # 在Windows上，simsun.ttc可能存在，但不是所有系统都有
    FONT_PATH = "simsun.ttc" # 备用字体，可能需要根据您的系统调整
    if not os.path.exists(FONT_PATH):
        messagebox.showwarning("字体文件警告", f"字体文件未找到于 {FONT_PATH}，且备用字体也未找到。中文字符显示可能不正确。")

# --- 全局模型变量，只加载一次 ---
yolo_model = None
lprnet_model = None
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_models_once():
    """只加载一次模型，避免重复加载。"""
    global yolo_model, lprnet_model, device

    if yolo_model is None:
        print(f"正在从 {YOLO_MODEL_PATH} 加载 YOLOv8 模型...")
        try:
            yolo_model = YOLO(YOLO_MODEL_PATH)
            print("YOLOv8 模型加载成功。")
        except Exception as e:
            messagebox.showerror("模型加载失败", f"加载 YOLOv8 模型失败: {e}")
            return False

    if lprnet_model is None:
        print(f"正在从 {LPRNET_MODEL_PATH} 加载 LPRNet 模型...")
        try:
            lprnet_model = build_lprnet(lpr_max_len=LPR_MAX_LEN, phase=False, class_num=len(CHARS), dropout_rate=0)
            lprnet_model.load_state_dict(torch.load(LPRNET_MODEL_PATH, map_location=device))
            lprnet_model.to(device)
            lprnet_model.eval() # 设置 LPRNet 为评估模式
            print("LPRNet 模型加载成功。")
        except Exception as e:
            messagebox.showerror("模型加载失败", f"加载 LPRNet 模型失败: {e}")
            return False
    return True

# --- LPRNet 辅助函数 (保持不变) ---

def cv2ImgAddText(img, text, pos, textColor=(255, 0, 0), textSize=20):
    """
    使用 PIL 在 OpenCV 图像上添加中文文本。
    img: OpenCV 图像 (numpy 数组, HWC, BGR)
    text: 要添加的文本
    pos: 文本的左上角坐标 (x, y)
    textColor: 文本颜色 (B, G, R)
    textSize: 文本大小
    """
    if isinstance(img, np.ndarray):
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        img_pil = img

    draw = ImageDraw.Draw(img_pil)
    try:
        fontText = ImageFont.truetype(FONT_PATH, textSize, encoding="utf-8")
    except IOError:
        print(f"错误: 无法加载字体文件 '{FONT_PATH}'。将使用 PIL 默认字体。")
        fontText = ImageFont.load_default()

    pil_textColor = (textColor[2], textColor[1], textColor[0]) # BGR to RGB for PIL

    draw.text(pos, text, pil_textColor, font=fontText)

    return cv2.cvtColor(np.asarray(img_pil), cv2.COLOR_RGB2BGR)

def preprocess_lpr_image(img):
    """
    对 LPRNet 输入图像进行预处理。
    img: OpenCV 图像 (numpy 数组, HWC, BGR)
    """
    img = cv2.resize(img, tuple(LPR_INPUT_SIZE))
    img = img.astype('float32')
    img = (img - 127.5) * 0.0078125
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img).unsqueeze(0)

def decode_lprnet_output(prebs):
    """
    将 LPRNet 的原始输出概率解码为车牌字符串。
    prebs: LPRNet 对于单张图像的输出 (C, T) 形状的 numpy 数组
    """
    preb_label = []
    for j in range(prebs.shape[1]):
        preb_label.append(np.argmax(prebs[:, j], axis=0))

    no_repeat_blank_label = []
    pre_c = preb_label[0]
    if pre_c != len(CHARS) - 1:
        no_repeat_blank_label.append(pre_c)

    for c in preb_label:
        if (pre_c == c) or (c == len(CHARS) - 1):
            if c == len(CHARS) - 1:
                pre_c = c
            continue
        no_repeat_blank_label.append(c)
        pre_c = c
    
    decoded_text = ""
    for c_idx in no_repeat_blank_label:
        if 0 <= c_idx < len(CHARS):
            decoded_text += CHARS[c_idx]
        else:
            decoded_text += "?"
    return decoded_text

# --- 修改后的主识别功能，用于GUI ---
def process_image_for_gui(image_path, progress_callback=None, status_callback=None):
    """
    加载图像，使用 YOLOv8 检测车牌，然后使用 LPRNet 识别字符，并返回处理后的图像和结果。
    image_path: 输入图像的完整路径
    progress_callback: 用于更新进度的函数 (接受0-100的整数)
    status_callback: 用于更新状态文本的函数 (接受字符串)
    返回: (处理后的图像 numpy 数组, 识别的车牌信息列表)
    """
    if not load_models_once():
        return None, None # 模型加载失败

    if status_callback: status_callback("正在加载图像...")
    img_array = np.fromfile(image_path, dtype=np.uint8)
    original_image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if original_image is None:
        messagebox.showerror("图像加载失败", f"错误: 无法从路径 {image_path} 加载图像。请检查路径和文件完整性。")
        return None, None

    display_image = original_image.copy()
    recognized_plates_info = []

    if status_callback: status_callback("正在使用 YOLOv8 执行车牌检测...")
    if progress_callback: progress_callback(10) # 进度 10%

    results = yolo_model(original_image, verbose=False)
    if progress_callback: progress_callback(50) # 进度 50%

    # 处理检测结果
    total_detections = sum(len(r.boxes.xyxy) for r in results)
    processed_detections = 0

    if status_callback: status_callback("正在使用 LPRNet 识别车牌字符...")
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        confidences = r.boxes.conf.cpu().numpy()

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            confidence = confidences[i]

            if confidence < 0.1: # 您可以根据需要设置置信度阈值来过滤低置信度的检测
                continue

            cropped_lp = original_image[max(0, y1):min(original_image.shape[0], y2), max(0, x1):min(original_image.shape[1], x2)]
            
            if cropped_lp.shape[0] == 0 or cropped_lp.shape[1] == 0:
                print(f"警告: 裁剪的车牌区域为空或尺寸为零 ({x1},{y1},{x2},{y2})。跳过识别。")
                continue

            lpr_input = preprocess_lpr_image(cropped_lp).to(device)

            with torch.no_grad():
                lpr_output = lprnet_model(lpr_input)
                prebs = lpr_output.cpu().detach().numpy()[0]

            plate_number = decode_lprnet_output(prebs)
            recognized_plates_info.append((plate_number, confidence, (x1, y1, x2, y2)))

            # 在显示图像上绘制边界框和识别结果
            cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2) # 绿色矩形

            text_to_display = f"{plate_number} ({confidence:.2f})"
            
            # 使用PIL的ImageFont.truetype来估算文本尺寸
            # 创建一个临时的PIL图片和Draw对象来获取字体尺寸
            temp_img_pil = Image.fromarray(np.zeros((50, 50, 3), dtype=np.uint8))
            temp_draw = ImageDraw.Draw(temp_img_pil)
            try:
                fontText_box = ImageFont.truetype(FONT_PATH, 25, encoding="utf-8")
            except IOError:
                fontText_box = ImageFont.load_default()
            
            bbox = temp_draw.textbbox((0, 0), text_to_display, font=fontText_box)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            text_pos_x = x1
            text_pos_y = y1 - text_height - 5
            if text_pos_y < 0:
                text_pos_y = y2 + 5 

            display_image = cv2ImgAddText(display_image, 
                                            text_to_display, 
                                            (text_pos_x, text_pos_y), 
                                            (0, 0, 255), # 红色字体 (BGR 格式)
                                            15) # 稍微增大字体大小

            print(f"检测到的车牌: {plate_number} (置信度: {confidence:.2f}) 坐标: {x1, y1, x2, y2}")
            
            processed_detections += 1
            if progress_callback:
                # 假设识别每个车牌的LPRNet部分占总进度的剩余50%
                progress_val = 50 + int((processed_detections / total_detections) * 50) if total_detections > 0 else 100
                progress_callback(progress_val)

    if progress_callback: progress_callback(100) # 完成
    if status_callback: status_callback("检测和识别完成。")
    return display_image, recognized_plates_info

# --- GUI 界面的实现 ---

class LicensePlateRecognizerGUI:
    def __init__(self, master):
        self.master = master
        master.title("基于YOLOv8和LPRNet的车牌号识别系统")
        master.geometry("1200x800") # 初始窗口大小
        master.configure(bg="#2E2E2E") # 深灰色背景

        # --- 标题 ---
        self.title_label = tk.Label(master, 
                                    text="基于YOLOv8和LPRNet的车牌号识别系统", 
                                    font=("Microsoft YaHei UI", 24, "bold"), 
                                    fg="#4CAF50", # 绿色
                                    bg="#2E2E2E")
        self.title_label.pack(pady=20)

        # --- 主容器 ---
        self.main_frame = tk.Frame(master, bg="#3C3C3C", bd=5, relief=tk.GROOVE) # 深一点的灰色，带边框
        self.main_frame.pack(padx=20, pady=10, fill=tk.BOTH, expand=True)

        # --- 左侧：图像显示区 ---
        self.image_frame = tk.Frame(self.main_frame, bg="#4A4A4A") # 更深的灰色
        self.image_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.image_label = tk.Label(self.image_frame, bg="#4A4A4A", text="请点击“上传图片”按钮选择图片进行识别",
                                     font=("Microsoft YaHei UI", 14), fg="#EEEEEE")
        self.image_label.pack(expand=True)
        
        self.current_image_path = None # 存储当前图片路径

        # --- 右侧：控制和结果显示区 ---
        self.control_frame = tk.Frame(self.main_frame, bg="#3C3C3C", width=300)
        self.control_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.Y)
        self.control_frame.pack_propagate(False) # 防止子控件调整frame大小

        # 上传图片按钮
        self.upload_button = tk.Button(self.control_frame, 
                                       text="上传图片", 
                                       command=self.upload_image,
                                       font=("Microsoft YaHei UI", 16, "bold"),
                                       bg="#007BFF", fg="white", # 蓝色背景，白色文字
                                       activebackground="#0056b3", activeforeground="white",
                                       relief=tk.RAISED, bd=3, cursor="hand2")
        self.upload_button.pack(pady=20, ipadx=20, ipady=10)

        # 状态提示
        self.status_label = tk.Label(self.control_frame, 
                                     text="等待上传图片...", 
                                     font=("Microsoft YaHei UI", 12), 
                                     fg="#FFD700", # 金色
                                     bg="#3C3C3C")
        self.status_label.pack(pady=10)

        # 进度条
        self.progress_bar = ttk.Progressbar(self.control_frame, 
                                            orient="horizontal", 
                                            length=250, 
                                            mode="determinate",
                                            style="Custom.Horizontal.TProgressbar") # 自定义样式
        self.progress_bar.pack(pady=10)
        self.progress_bar["value"] = 0 # 初始值

        # 自定义进度条样式
        s = ttk.Style()
        s.theme_use('clam') # 使用 'clam' 主题作为基础
        s.configure("Custom.Horizontal.TProgressbar", 
                    troughcolor='#555555', # 进度条凹槽颜色
                    background='#4CAF50', # 进度条填充色 (绿色)
                    bordercolor='#4CAF50', # 边框颜色
                    lightcolor='#4CAF50', # 亮色
                    darkcolor='#4CAF50') # 暗色
        
        # 识别结果显示区
        self.result_frame = tk.LabelFrame(self.control_frame, 
                                          text="识别结果", 
                                          font=("Microsoft YaHei UI", 14, "bold"), 
                                          fg="#4CAF50", bg="#3C3C3C", bd=2, relief=tk.SOLID)
        self.result_frame.pack(pady=20, padx=10, fill=tk.BOTH, expand=True)

        self.result_text = tk.Text(self.result_frame, 
                                   wrap=tk.WORD, 
                                   height=10, 
                                   font=("Consolas", 12), 
                                   bg="#2A2A2A", fg="#FFFFFF", # 深黑背景，白色文字
                                   insertbackground="white", relief=tk.FLAT) # 无边框
        self.result_text.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)
        self.result_text.insert(tk.END, "识别到的车牌号将在此处显示。\n")
        self.result_text.config(state=tk.DISABLED) # 初始设置为只读

        # 加载模型一次（可选，可以在程序启动时加载）
        # self.status_label.config(text="正在加载模型，请稍候...")
        # master.update_idletasks() # 强制更新UI
        # if not load_models_once():
        #     master.destroy() # 如果模型加载失败，关闭窗口
        # self.status_label.config(text="等待上传图片...")


    def upload_image(self):
        """处理图片上传按钮点击事件"""
        file_path = filedialog.askopenfilename(
            title="选择图片文件",
            filetypes=[("图像文件", "*.jpg;*.jpeg;*.png;*.bmp"), ("所有文件", "*.*")]
        )
        if file_path:
            self.current_image_path = file_path
            self.status_label.config(text=f"已选择图片：{os.path.basename(file_path)}")
            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "正在处理图片，请稍候...\n")
            self.result_text.config(state=tk.DISABLED)
            self.progress_bar["value"] = 0
            
            # 使用 threading 来避免UI卡死
            import threading
            process_thread = threading.Thread(target=self.run_recognition, args=(file_path,))
            process_thread.start()
        else:
            self.status_label.config(text="未选择图片。")

    def update_progress(self, value):
        """更新进度条"""
        self.progress_bar["value"] = value
        self.master.update_idletasks() # 强制更新GUI

    def update_status(self, text):
        """更新状态文本"""
        self.status_label.config(text=text)
        self.master.update_idletasks() # 强制更新GUI

    def run_recognition(self, image_path):
        """在单独线程中运行识别过程"""
        self.upload_button.config(state=tk.DISABLED) # 禁用按钮，避免重复点击
        self.update_status("正在加载模型...")
        # 确保模型只加载一次
        if not load_models_once():
            self.update_status("模型加载失败，请检查路径。")
            self.upload_button.config(state=tk.NORMAL)
            return

        self.update_status("正在进行车牌检测和识别...")
        
        processed_image, recognized_plates = process_image_for_gui(
            image_path,
            progress_callback=self.update_progress,
            status_callback=self.update_status
        )
        
        if processed_image is not None and recognized_plates is not None:
            # 显示处理后的图片
            self.display_image_on_gui(processed_image)
            
            # 显示识别结果
            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete(1.0, tk.END)
            if recognized_plates:
                self.result_text.insert(tk.END, "--- 检测与识别结果 ---\n\n")
                for idx, (plate_num, conf, coords) in enumerate(recognized_plates):
                    x1, y1, x2, y2 = coords
                    self.result_text.insert(tk.END, f"车牌 {idx+1}:\n")
                    self.result_text.insert(tk.END, f"  识别结果: {plate_num}\n")
                    self.result_text.insert(tk.END, f"  YOLOv8置信度: {conf:.2f}\n")
                    self.result_text.insert(tk.END, f"  检测框坐标: ({x1},{y1},{x2},{y2})\n")
                    self.result_text.insert(tk.END, "-------------------------\n")
                self.update_status("车牌识别完成！")
            else:
                self.result_text.insert(tk.END, "未检测到或未识别到任何车牌。\n")
                self.update_status("未检测到车牌。")
            self.result_text.config(state=tk.DISABLED)
        else:
            self.update_status("图片处理失败。")
            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "处理图片时发生错误。\n")
            self.result_text.config(state=tk.DISABLED)

        self.upload_button.config(state=tk.NORMAL) # 重新启用按钮
        self.progress_bar["value"] = 0 # 重置进度条

    def display_image_on_gui(self, cv_image):
        """
        在GUI的Label中显示OpenCV图像。
        cv_image: OpenCV 格式的图像 (BGR numpy array)
        """
        # 将OpenCV BGR图像转换为PIL RGB图像
        img_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        # 调整图像大小以适应显示区域
        max_width = self.image_frame.winfo_width()
        max_height = self.image_frame.winfo_height()
        if max_width == 1 or max_height == 1: # 初始winfo_width/height可能为1
            max_width = 700 # 设定一个默认最大值
            max_height = 600

        img_width, img_height = img_pil.size
        ratio = min(max_width / img_width, max_height / img_height)
        new_width = int(img_width * ratio)
        new_height = int(img_height * ratio)

        img_pil = img_pil.resize((new_width, new_height), Image.LANCZOS) # 使用高品质缩放算法

        # 转换为Tkinter PhotoImage
        self.tk_image = ImageTk.PhotoImage(img_pil)
        
        # 更新Label以显示图像
        self.image_label.config(image=self.tk_image, text="") # 移除文本提示
        self.image_label.image = self.tk_image # 保持引用，防止垃圾回收

        # 居中图像
        self.image_label.pack_forget() # 先取消打包
        self.image_label.pack(expand=True, padx=5, pady=5)


# --- 运行 GUI ---
if __name__ == "__main__":
    root = tk.Tk()
    app = LicensePlateRecognizerGUI(root)
    # 首次启动时加载模型
    # 注意：如果模型文件过大，这可能会导致GUI启动时卡顿。
    # 更好的做法是在后台线程加载，或者在用户点击“上传图片”时再加载（如代码中已实现）
    # app.update_status("正在加载模型，请稍候...")
    # root.update_idletasks()
    # if not load_models_once():
    #     messagebox.showerror("模型加载失败", "程序启动失败，请检查模型路径。")
    #     root.destroy()
    # else:
    #     app.update_status("模型加载完成，请上传图片。")
    root.mainloop()