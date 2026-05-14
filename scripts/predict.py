import cv2
import os
import numpy as np
from pathlib import Path
from ultralytics import YOLO    
from math import ceil
from tqdm import tqdm

# ================= 🚨 配置区 🚨 =================
MODEL_PATH = 'runs/runs/Model_Comparison/exp_yolov8s-seg_mass_crack/weights/best.pt'
INPUT_DIR = '../tests'
OUTPUT_FILE = '../prediction.jpg'
COLS = 4  # 你希望每一行放几组对比（1组 = 原图+预测）
BASE_HEIGHT = 480  # 【核心修复】统一所有行的高度为 480 像素
# ===============================================

def generate_big_grid():
    model = YOLO(MODEL_PATH)
    exts = ('.jpg', '.jpeg', '.png')
    files = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith(exts)])
    
    if not files:
        print("❌ 没找到图片！")
        return

    # 1. 计算单组对比图的基准宽度
    # 我们以 BASE_HEIGHT 为准，等比例缩放图片
    test_img = cv2.imread(os.path.join(INPUT_DIR, files[0]))
    h_orig, w_orig = test_img.shape[:2]
    aspect_ratio = w_orig / h_orig
    base_width = int(BASE_HEIGHT * aspect_ratio)
    
    single_compare_w = 2 * base_width + 10 # 两张图 + 10px白线
    single_compare_h = BASE_HEIGHT
    
    # 2. 计算大图尺寸
    rows = ceil(len(files) / COLS)
    grid_w = COLS * single_compare_w
    grid_h = rows * single_compare_h
    
    # 创建黑色大背景
    canvas = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

    print(f"🚀 正在处理 {len(files)} 张图，生成 {rows}x{COLS} 的对齐平铺图...")

    for idx, name in enumerate(tqdm(files)):
        img_path = os.path.join(INPUT_DIR, name)
        orig = cv2.imread(img_path)
        if orig is None: continue
        
        # 强制缩放到基准高度，保持比例
        orig = cv2.resize(orig, (base_width, BASE_HEIGHT))
        
        # 预测
        # results = model.predict(img_path, conf=0.15, retina_masks=True, verbose=False)
        results = model.predict(
            img_path, 
            conf=0.05,          # 极低阈值：把那些“似是而非”的裂缝全抓回来
            iou=0.8,            # 允许更多重叠：解决裂缝断断续续的问题
            retina_masks=True,  # 开启高清模式：让边缘贴合得更紧
            imgsz=960,          # 放大看图：960 是 32 的倍数，4060 推理这个尺寸很轻松
            verbose=False
        )
        pred = results[0].plot(labels=False, boxes=True)
        # 预测图也强制缩放到同样大小
        pred = cv2.resize(pred, (base_width, BASE_HEIGHT))
        
        # 拼接单组对比（现在高度绝对都是 BASE_HEIGHT 了）
        divider = np.ones((BASE_HEIGHT, 10, 3), dtype=np.uint8) * 255
        pair = np.hstack((orig, divider, pred))
        
        # 计算在画布上的坐标
        r = idx // COLS
        c = idx % COLS
        y_start = r * single_compare_h
        x_start = c * single_compare_w
        
        # 塞进画布
        canvas[y_start:y_start+BASE_HEIGHT, x_start:x_start+single_compare_w] = pair

    # 3. 保存
    cv2.imwrite(OUTPUT_FILE, canvas)
    print(f"\n✅ 修复完成！大平铺图已生成：{os.path.abspath(OUTPUT_FILE)}")

if __name__ == "__main__":
    generate_big_grid()