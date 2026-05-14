import cv2
import glob
import os
import random
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

MODEL_PATH = 'runs/runs/Model_Comparison/exp_yolov8s-seg_mass_crack/weights/best.pt'
VAL_IMAGES_DIR = '../val'
OUTPUT_PATH = '../yolov8_segmentation_grid.png' 

def plot_mass_comparison_grid(model, img_paths, rows=10, cols=15):
    """
    Generates a grid of original images and their corresponding YOLOv8 segmentation predictions.
    """
    num_imgs = rows * cols
    if len(img_paths) < num_imgs:
        sample_imgs = img_paths
    else:
        sample_imgs = random.sample(img_paths, num_imgs)
        
    print(f"Selected {len(sample_imgs)} images for visualization (rows={rows}, cols={cols}).")

    fig, axes = plt.subplots(nrows=rows, ncols=cols * 2, figsize=(10 * cols, 6 * rows))
    
    axes = axes.flatten()
    
    for i, img_path in enumerate(sample_imgs):
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # YOLO prediction
        results = model.predict(
            img_path, 
            conf=0.15,          # 极低阈值：把那些“似是而非”的裂缝全抓回来
            iou=0.7,            # 允许更多重叠：解决裂缝断断续续的问题
            retina_masks=True,  # 开启高清模式：让边缘贴合得更紧
            imgsz=960,          # 放大看图：960 是 32 的倍数，4060 推理这个尺寸很轻松
            verbose=False
        )
        pred_res = results[0].plot()
        pred_rgb = cv2.cvtColor(pred_res, cv2.COLOR_BGR2RGB)
        
        orig_idx = i * 2
        pred_idx = orig_idx + 1
        
        axes[orig_idx].imshow(img_rgb)
        axes[orig_idx].set_title(f"Original: {os.path.basename(img_path)}", fontsize=16, pad=10)
        axes[orig_idx].axis('off')
        
        axes[pred_idx].imshow(pred_rgb)
        axes[pred_idx].set_title("YOLOv8 Prediction", fontsize=16, pad=10, color='darkred')
        axes[pred_idx].axis('off')
        
    plt.tight_layout()
    
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
    print(f"Saved to current directory: {OUTPUT_PATH}")
    
    plt.show()

if __name__ == "__main__":
    model = YOLO(MODEL_PATH)
    all_imgs = glob.glob(os.path.join(VAL_IMAGES_DIR, "*.jpg"))
    
    if not all_imgs:
        print("No images found, please check the VAL_IMAGES_DIR path!")
    else:
        # You can modify rows and cols as needed
        plot_mass_comparison_grid(model, all_imgs, rows=4, cols=3)