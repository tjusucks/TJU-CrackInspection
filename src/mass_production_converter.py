import cv2
import os
import shutil
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm

def process_mask_to_polygon(mask_path, img_path, txt_path):
    """
    核心算法逻辑
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return False

    h, w = mask.shape
    # 针对 JPG 格式的 Mask，阈值 127 是安全的
    _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 【优化 1：形态学连接】连接细微断裂
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lines = []
    for contour in contours:
        if cv2.contourArea(contour) < 20:
            continue

        # 【优化 2：多边形简化】Douglas-Peucker 算法降维
        epsilon = 0.002 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        polygon = []
        for point in approx:
            x = max(0.0, min(1.0, point[0][0] / w))
            y = max(0.0, min(1.0, point[0][1] / h))
            polygon.append(f"{x:.6f} {y:.6f}")

        if len(polygon) >= 3:
            lines.append(f"0 " + " ".join(polygon))

    # 写入 TXT 标签
    with open(txt_path, 'w') as f:
        f.write("\n".join(lines))
    return True

def run_mass_conversion():
    # --- 路径设置 ---
    RAW_IMG_DIR = Path("/mnt/data/Data/Desktop/Homework/TJU-CrackInspection/datasets/Crack_Segmentation_Dataset/images")
    RAW_MASK_DIR = Path("/mnt/data/Data/Desktop/Homework/TJU-CrackInspection/datasets/Crack_Segmentation_Dataset/masks")
    OUTPUT_DIR = Path("/mnt/data/Data/Desktop/Homework/TJU-CrackInspection/datasets/YOLO_MASS_CRACK")

    # 1. 扫描所有图片
    all_imgs = list(RAW_IMG_DIR.glob("*.jpg"))
    if not all_imgs:
        print(f"未找到图片，请检查路径: {RAW_IMG_DIR}")
        return

    # 2. 随机打乱并截取前 10,000 张
    random.seed(42)
    random.shuffle(all_imgs)
    target_imgs = all_imgs[:10000]

    # 3. 按 8:1:1 划分
    total = len(target_imgs)
    train_idx = int(total * 0.8)
    val_idx = train_idx + int(total * 0.1)

    splits = {
        'train': target_imgs[:train_idx],
        'val': target_imgs[train_idx:val_idx],
        'test': target_imgs[val_idx:]
    }

    for phase, img_list in splits.items():
        print(f"\n📦 正在处理 {phase} 集 (共 {len(img_list)} 张)...")
        dst_img_dir = OUTPUT_DIR / "images" / phase
        dst_lbl_dir = OUTPUT_DIR / "labels" / phase
        os.makedirs(dst_img_dir, exist_ok=True)
        os.makedirs(dst_lbl_dir, exist_ok=True)

        for img_path in tqdm(img_list):
            # 因为文件名完全对应，直接在 Masks 文件夹找同名 jpg
            mask_path = RAW_MASK_DIR / img_path.name
            txt_path = dst_lbl_dir / (img_path.stem + ".txt")

            if process_mask_to_polygon(mask_path, img_path, txt_path):
                # 只有转换成功才搬运原图
                shutil.copy(img_path, dst_img_dir / img_path.name)

    print(f"\n✅ 全量转换任务完成！数据已就位: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_mass_conversion()
