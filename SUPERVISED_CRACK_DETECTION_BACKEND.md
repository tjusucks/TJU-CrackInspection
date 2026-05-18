# SOM + Supervised Crack Detection Backend Integration

本文档说明如何在后端中接入当前的裂缝检测流程：

1. SOM 生成疑似裂缝 mask。
2. 读取有监督模型 `joblib`，过滤砖缝、拼接缝、污渍、边缘等误检区域。
3. 输出裂缝检测结果、mask、叠加图、裂缝尺寸。

当前模型是一个候选区域二分类器：

- `1`: 裂缝
- `0`: 非裂缝

训练好的模型路径：

```bash
/home/cyaegha/som_component_annotation/som_crack_classifier.joblib
```

人工标注文件：

```bash
/home/cyaegha/som_component_annotation/lable.csv
```

测试集评估文件：

```bash
/home/cyaegha/som_component_annotation/evaluation_report.txt
/home/cyaegha/som_component_annotation/evaluation_report.json
/home/cyaegha/som_component_annotation/test_predictions.csv
```

当前测试集结果：

```text
Accuracy: 0.8777
AUC: 0.9245
Crack precision: 0.6522
Crack recall: 0.8182
Crack F1: 0.7258
Confusion Matrix [[TN, FP], [FN, TP]]
[[199, 24], [10, 45]]
```

注意：这个准确率是“候选区域级别”的分类准确率，不是整图级准确率，也不是像素级 IoU。

## 代码文件

后端最常用的文件：

```text
detect_crack_size.py            # 单张图片检测入口，输出检测结果和尺寸
som_crack_seg.py                # SOM 候选裂缝分割
som_component_classifier.py     # 读取 joblib，提取候选区域特征，过滤误检
crack_size_measure.py           # 裂缝长度、宽度、面积测量
run_detect_crack_size.sh        # 单图检测 shell 封装
```

训练和评估相关文件：

```text
make_som_annotation_set.py      # 生成候选区域标注集
train_som_crack_classifier.py   # 训练 joblib 模型，拆分训练/验证/测试集
serve_som_annotation.sh         # 本地 HTTP 服务，方便浏览器标注
```

批量检测相关文件：

```text
som_supervised_detect.py        # 批量 SOM + 有监督过滤
som_wall_block_pipeline.py      # 切砖 + SOM + 有监督过滤
crack_run_report.py             # 批量运行总体报告
```

## 最小依赖

在当前环境中使用的是 conda 环境：

```bash
conda run -n crack python ...
```

最小 Python 依赖：

```text
opencv-python
numpy
scipy
scikit-image
scikit-learn
joblib
minisom
```

如果后端独立部署，需要确保这些包已经安装，并且能访问模型文件：

```bash
/home/cyaegha/som_component_annotation/som_crack_classifier.joblib
```

## 单张图片命令行调用

推荐后端最先接入这个脚本：

```bash
cd /home/cyaegha/TJU-CrackInspection

conda run -n crack python detect_crack_size.py \
  --image /path/to/input.jpg \
  --joblib /home/cyaegha/som_component_annotation/som_crack_classifier.joblib \
  --output-dir /path/to/output_dir
```

如果知道物理比例尺，例如 `1 px = 0.1 mm`：

```bash
conda run -n crack python detect_crack_size.py \
  --image /path/to/input.jpg \
  --joblib /home/cyaegha/som_component_annotation/som_crack_classifier.joblib \
  --output-dir /path/to/output_dir \
  --mm-per-pixel 0.1
```

可选参数：

```text
--image / --input             输入图片
--joblib / --model            有监督模型 joblib 文件
--output-dir                  输出目录
--classifier-threshold        分类阈值，默认读取 joblib 内保存的 threshold
--mm-per-pixel                像素到毫米比例，不传则只输出像素尺寸
--min-area                    过滤很小的裂缝连通域，默认 20 px
--no-classifier               不使用有监督模型，只使用原始 SOM mask
```

封装脚本也可以直接用：

```bash
cd /home/cyaegha/TJU-CrackInspection

./run_detect_crack_size.sh \
  /path/to/input.jpg \
  /path/to/output_dir
```

## 单图输出文件

假设输入图片名是：

```text
test.jpg
```

输出目录会包含：

```text
test_mask.png              # 有监督过滤后的裂缝 mask
test_raw_som_mask.png      # 原始 SOM mask
test_size_overlay.png      # 检测结果叠加图，包含裂缝编号和尺寸
test_crack_sizes.json      # 完整 JSON 结果
test_crack_sizes.csv       # 裂缝尺寸表格
```

## JSON 输出结构

`*_crack_sizes.json` 顶层结构：

```json
{
  "input_image": "/path/to/input.jpg",
  "classifier_model": "/path/to/som_crack_classifier.joblib",
  "classifier_threshold": 0.35,
  "raw_crack_pixels": 18473,
  "kept_crack_pixels": 15044,
  "mask_path": "/path/to/output/test_mask.png",
  "raw_mask_path": "/path/to/output/test_raw_som_mask.png",
  "overlay_path": "/path/to/output/test_size_overlay.png",
  "csv_path": "/path/to/output/test_crack_sizes.csv",
  "summary": {},
  "measurements": [],
  "component_predictions": []
}
```

`summary` 字段：

```json
{
  "image_width_px": 448,
  "image_height_px": 448,
  "mm_per_pixel": null,
  "crack_count": 1,
  "has_crack": true,
  "total_area_px": 15044,
  "total_length_px": 3677.77,
  "average_width_px": 4.09,
  "max_width_px": 27.57,
  "crack_area_ratio": 0.0749,
  "total_area_mm2": null,
  "total_length_mm": null,
  "average_width_mm": null,
  "max_width_mm": null
}
```

`measurements` 中每个裂缝对象：

```json
{
  "crack_id": 1,
  "area_px": 15044,
  "length_px": 3677.77,
  "avg_width_px": 4.09,
  "max_width_px": 27.57,
  "bbox_width_px": 244,
  "bbox_height_px": 430,
  "bbox_min_row": 0,
  "bbox_min_col": 204,
  "bbox_max_row": 430,
  "bbox_max_col": 448,
  "orientation_deg": 15.03,
  "eccentricity": 0.954,
  "endpoint_count": 148,
  "length_mm": null,
  "avg_width_mm": null,
  "max_width_mm": null,
  "area_mm2": null
}
```

`component_predictions` 是有监督模型对 SOM 候选连通域的判断：

```json
{
  "component_label": 1,
  "probability": 0.96,
  "kept": true,
  "area": 15044,
  "bbox": [0, 204, 430, 448]
}
```

## 尺寸计算规则

长度：

- 对最终裂缝 mask 做骨架化。
- 水平/垂直相邻骨架点按 `1 px`。
- 斜向相邻骨架点按 `sqrt(2) px`。
- 所有骨架连接长度求和，得到 `length_px`。

宽度：

- `avg_width_px = area_px / length_px`
- `max_width_px` 使用距离变换估计局部最大宽度。

面积：

- `area_px` 是裂缝 mask 像素数。
- 如果传入 `--mm-per-pixel`，则：
  - `length_mm = length_px * mm_per_pixel`
  - `avg_width_mm = avg_width_px * mm_per_pixel`
  - `area_mm2 = area_px * mm_per_pixel^2`

## 后端直接 Python 调用

如果后端和算法在同一个 Python 环境中，推荐直接 import，而不是通过 subprocess。

示例：

```python
from pathlib import Path
import json
import cv2

from som_crack_seg import read_image, segment_image_bgr
from som_component_classifier import load_classifier_bundle, filter_mask_with_classifier
from crack_size_measure import (
    measure_crack_components,
    summarize_measurements,
    draw_measurement_overlay,
)


def detect_one_image(image_path, model_path, output_dir, mm_per_pixel=None, threshold=None):
    image_path = Path(image_path).expanduser()
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    image_bgr = read_image(image_path)
    if image_bgr is None:
        raise ValueError(f"Failed to read image: {image_path}")

    raw_mask_bool, _, _ = segment_image_bgr(image_bgr)

    bundle = load_classifier_bundle(model_path)
    final_mask_bool, component_predictions = filter_mask_with_classifier(
        image_bgr,
        raw_mask_bool,
        bundle,
        threshold=threshold,
    )

    measurements = measure_crack_components(
        final_mask_bool,
        mm_per_pixel=mm_per_pixel,
        min_area=20,
    )
    summary = summarize_measurements(
        measurements,
        image_bgr.shape,
        mm_per_pixel=mm_per_pixel,
    )

    stem = image_path.stem
    mask_path = output_dir / f"{stem}_mask.png"
    overlay_path = output_dir / f"{stem}_size_overlay.png"
    json_path = output_dir / f"{stem}_crack_sizes.json"

    cv2.imwrite(str(mask_path), final_mask_bool.astype("uint8") * 255)
    overlay = draw_measurement_overlay(image_bgr, final_mask_bool, measurements, mm_per_pixel)
    cv2.imwrite(str(overlay_path), overlay)

    result = {
        "input_image": str(image_path),
        "mask_path": str(mask_path),
        "overlay_path": str(overlay_path),
        "summary": summary,
        "measurements": measurements,
        "component_predictions": component_predictions,
    }
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
```

后端建议：

- 服务启动时只加载一次 joblib。
- 每次请求只做图片读取、SOM、分类过滤、尺寸测量。
- 输出路径最好按任务 ID 隔离，避免并发覆盖。

## FastAPI 接口建议

推荐接口：

```text
POST /api/crack/detect-size
Content-Type: multipart/form-data
```

请求参数：

```text
file: 图片文件
mm_per_pixel: 可选，像素到毫米比例
classifier_threshold: 可选，裂缝概率阈值
```

响应示例：

```json
{
  "success": true,
  "has_crack": true,
  "crack_count": 1,
  "summary": {
    "total_length_px": 3677.77,
    "average_width_px": 4.09,
    "max_width_px": 27.57
  },
  "measurements": [],
  "files": {
    "mask": "/static/results/task_id/test_mask.png",
    "overlay": "/static/results/task_id/test_size_overlay.png",
    "json": "/static/results/task_id/test_crack_sizes.json",
    "csv": "/static/results/task_id/test_crack_sizes.csv"
  }
}
```

## 阈值说明

模型输出 `probability` 表示该候选区域是裂缝的概率。

- 阈值低：召回率更高，漏检少，但误检更多。
- 阈值高：误检少，但可能漏检。

当前模型默认阈值保存在 joblib 内：

```text
0.35
```

后端可以暴露一个高级参数 `classifier_threshold`，但普通用户建议使用默认值。

## 批量检测

只做 SOM + 有监督过滤：

```bash
cd /home/cyaegha/TJU-CrackInspection

CLASSIFIER_MODEL=/home/cyaegha/som_component_annotation/som_crack_classifier.joblib \
./run_som_supervised_detect.sh \
  /path/to/images_or_image \
  /path/to/output_dir
```

切砖 + SOM + 有监督过滤：

```bash
cd /home/cyaegha/TJU-CrackInspection

CLASSIFIER_MODEL=/home/cyaegha/som_component_annotation/som_crack_classifier.joblib \
./run_som_wall_block_pipeline.sh \
  /path/to/images_or_image \
  /path/to/output_dir
```

批量运行会输出：

```text
index.json
run_report.json
run_report.txt
```

`run_report.txt` 中包括处理成功率、检测出裂缝的比例、过滤掉的像素比例等。

## 重新训练模型

如果人工标注更新，需要重新训练：

```bash
cd /home/cyaegha/TJU-CrackInspection

conda run -n crack python train_som_crack_classifier.py \
  --labels-csv /home/cyaegha/som_component_annotation/lable.csv \
  --model-out /home/cyaegha/som_component_annotation/som_crack_classifier.joblib \
  --test-size 0.2 \
  --val-size 0.2
```

训练后会更新：

```text
som_crack_classifier.joblib
evaluation_report.json
evaluation_report.txt
test_predictions.csv
```

## 已知限制

1. SOM 是无监督候选生成，若 SOM 完全没覆盖某条裂缝，有监督模型无法凭空找回。
2. 当前准确率是候选区域级别；整图级和像素级指标需要额外真值标签。
3. 物理尺寸依赖 `mm_per_pixel`，没有比例尺时只能输出像素尺寸。
4. `length_px` 是骨架路径长度估计，适合比较裂缝相对长度，不等价于人工测量的工程长度。
5. 大图会在 SOM 阶段内部缩放到最长边 `1600` 以内处理，再映射回原图。

## 推荐后端落地流程

1. 上传图片保存到临时任务目录。
2. 调用 `detect_crack_size.py` 或直接 import Python 函数。
3. 读取 `*_crack_sizes.json`。
4. 将 `summary` 和 `measurements` 返回给前端。
5. 将 `*_size_overlay.png` 作为检测结果图展示。
6. 如需要下载表格，提供 `*_crack_sizes.csv`。
