from ultralytics import YOLO
import time
import torch

# ----- YOLOv8 -----
# MODEL_NAME = 'yolov8n-seg.pt'
MODEL_NAME = 'yolov8s-seg.pt'
# MODEL_NAME = 'yolov8m-seg.pt'

# ----- YOLO11 -----
# MODEL_NAME = 'yolo11n-seg.pt'
# MODEL_NAME = 'yolo11s-seg.pt'
# MODEL_NAME = 'yolo11m-seg.pt'

# =====================================================================
# Hyperparameters
# =====================================================================
DATA_YAML = '../datasets/YOLO_MASS_CRACK/mass_crack.yaml'
EPOCHS = 100
IMG_SIZE = 640
BATCH_SIZE = 8
PROJECT_DIR = '../runs/Model_Comparison'

def run_single_experiment():
    run_name = f"exp_{MODEL_NAME.split('.')[0]}_mass_crack"

    print("\n" + "="*60)
    print(f"Current Model: {MODEL_NAME}")
    print(f"Saving at directory: {PROJECT_DIR}/{run_name}")
    print("="*60)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        model = YOLO(MODEL_NAME)

        start_time = time.time()
        results = model.train(
            data=DATA_YAML,
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            project=PROJECT_DIR,
            name=run_name,
            workers=4,
            patience=20,
            close_mosaic=10,
            overlap_mask=True,
            device='0'
        )
        train_time = (time.time() - start_time) / 60

        print("\n" + "="*60)
        print(f"Training completed: {MODEL_NAME}")
        print(f"Training time: {train_time:.2f} minutes")
        print(f"Mask mAP@0.5:      {results.seg.map50:.4f}")
        print(f"Mask mAP@0.5-0.95: {results.seg.map:.4f}")
        print("="*60)

    except Exception as e:
        print(f"Error during training: {e}")

if __name__ == "__main__":
    run_single_experiment()
