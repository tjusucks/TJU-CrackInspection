from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import cv2
import numpy as np
from skimage.measure import label, regionprops

from som_crack_seg import IMAGE_EXTS, read_image, segment_image_bgr


DEFAULT_SOURCE_DIR = Path("~/crack_datasets/images_denoise").expanduser()
DEFAULT_OUTPUT_DIR = Path("~/som_component_annotation").expanduser()


def iter_images(source_dir: Path):
    return sorted(
        path
        for path in source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def make_component_overlay(image_bgr, component_mask):
    overlay = image_bgr.copy()
    color = np.zeros_like(image_bgr)
    color[component_mask] = (0, 0, 255)
    return cv2.addWeighted(overlay, 1.0, color, 0.55, 0)


def crop_with_padding(image, minr, minc, maxr, maxc, padding):
    h, w = image.shape[:2]
    y1 = clamp(minr - padding, 0, h)
    x1 = clamp(minc - padding, 0, w)
    y2 = clamp(maxr + padding, 0, h)
    x2 = clamp(maxc + padding, 0, w)
    return image[y1:y2, x1:x2], (y1, x1, y2, x2)


def region_to_record(
    candidate_id,
    image_path,
    image_bgr,
    labeled_mask,
    region,
    output_dir,
    padding,
):
    component_mask = labeled_mask == region.label
    minr, minc, maxr, maxc = region.bbox

    crop, crop_box = crop_with_padding(image_bgr, minr, minc, maxr, maxc, padding)
    overlay_full = make_component_overlay(image_bgr, component_mask)
    overlay_crop, _ = crop_with_padding(overlay_full, minr, minc, maxr, maxc, padding)
    mask_crop, _ = crop_with_padding(
        (component_mask.astype(np.uint8) * 255),
        minr,
        minc,
        maxr,
        maxc,
        padding,
    )

    stem = f"{candidate_id:06d}_{image_path.stem}_r{int(region.label):03d}"
    crop_rel = Path("crops") / f"{stem}_crop.jpg"
    overlay_rel = Path("overlays") / f"{stem}_overlay.jpg"
    mask_rel = Path("masks") / f"{stem}_mask.png"

    cv2.imwrite(str(output_dir / crop_rel), crop)
    cv2.imwrite(str(output_dir / overlay_rel), overlay_crop)
    cv2.imwrite(str(output_dir / mask_rel), mask_crop)

    bbox_h = maxr - minr
    bbox_w = maxc - minc
    major = float(region.major_axis_length)
    minor = float(region.minor_axis_length)

    return {
        "candidate_id": f"{candidate_id:06d}",
        "label": "",
        "image_path": str(image_path),
        "crop_path": str(crop_rel),
        "overlay_path": str(overlay_rel),
        "mask_path": str(mask_rel),
        "bbox_min_row": int(minr),
        "bbox_min_col": int(minc),
        "bbox_max_row": int(maxr),
        "bbox_max_col": int(maxc),
        "crop_min_row": int(crop_box[0]),
        "crop_min_col": int(crop_box[1]),
        "crop_max_row": int(crop_box[2]),
        "crop_max_col": int(crop_box[3]),
        "area": int(region.area),
        "bbox_width": int(bbox_w),
        "bbox_height": int(bbox_h),
        "eccentricity": float(region.eccentricity),
        "major_axis_length": major,
        "minor_axis_length": minor,
        "major_minor_ratio": major / max(minor, 1e-6),
        "orientation": float(region.orientation),
    }


def collect_candidates(args):
    source_dir = Path(args.source_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ["crops", "overlays", "masks"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    image_paths = iter_images(source_dir)
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(image_paths)

    records = []
    processed_images = 0
    failed_images = 0

    for image_path in image_paths:
        if len(records) >= args.target_count:
            break

        image_bgr = read_image(image_path)
        if image_bgr is None:
            failed_images += 1
            continue

        try:
            mask_bool, _, _ = segment_image_bgr(image_bgr)
        except Exception as exc:
            print(f"FAILED {image_path}: {exc}")
            failed_images += 1
            continue

        labeled_mask = label(mask_bool)
        regions = []
        for region in regionprops(labeled_mask):
            if region.area < args.min_area:
                continue
            minr, minc, maxr, maxc = region.bbox
            if (maxr - minr) < args.min_bbox_side and (maxc - minc) < args.min_bbox_side:
                continue
            regions.append(region)

        regions.sort(key=lambda r: r.area, reverse=True)
        regions = regions[: args.max_components_per_image]

        for region in regions:
            if len(records) >= args.target_count:
                break
            candidate_id = len(records) + 1
            record = region_to_record(
                candidate_id,
                image_path,
                image_bgr,
                labeled_mask,
                region,
                output_dir,
                args.padding,
            )
            records.append(record)

        processed_images += 1
        print(
            f"{processed_images:04d} images | {len(records):04d}/{args.target_count} candidates | {image_path.name}"
        )

    return output_dir, records, processed_images, failed_images


def write_csv(output_dir, records):
    csv_path = output_dir / "candidates.csv"
    if not records:
        csv_path.write_text("", encoding="utf-8")
        return csv_path

    fieldnames = list(records[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    return csv_path


def write_html(output_dir, records):
    html_path = output_dir / "annotate.html"
    records_json = json.dumps(records, ensure_ascii=False)
    storage_key = f"som_component_labels_{len(records)}_{records[0]['candidate_id'] if records else 'empty'}_{records[-1]['candidate_id'] if records else 'empty'}"
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>SOM 裂缝候选区域标注</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, sans-serif; margin: 0; background: #f5f5f5; }}
    header {{ position: sticky; top: 0; z-index: 2; background: #202124; color: white; padding: 12px 18px; display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }}
    button {{ border: 0; padding: 8px 11px; border-radius: 6px; cursor: pointer; font-weight: 600; }}
    .yes {{ background: #1a7f37; color: white; }}
    .no {{ background: #b42318; color: white; }}
    .unsure {{ background: #d97706; color: white; }}
    .clear {{ background: #e5e7eb; color: #111827; }}
    .export {{ background: #2563eb; color: white; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 12px; padding: 12px; }}
    .card {{ background: white; border: 3px solid transparent; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,.12); }}
    .card[data-label="1"] {{ border-color: #1a7f37; }}
    .card[data-label="0"] {{ border-color: #b42318; }}
    .card[data-label="-1"] {{ border-color: #d97706; }}
    .meta {{ font-size: 12px; color: #374151; padding: 8px 10px; line-height: 1.45; word-break: break-all; }}
    .candidate-img {{ width: 100%; display: block; background: #ddd; }}
    .actions {{ display: flex; gap: 6px; padding: 8px 10px 10px; flex-wrap: wrap; }}
    .hidden {{ display: none; }}
    code {{ background: rgba(255,255,255,.15); padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <header>
    <strong>SOM 裂缝候选区域标注</strong>
    <span id="stats"></span>
    <button class="export" onclick="exportCsv()">导出 labels.csv</button>
    <button class="clear" onclick="setFilter('all')">全部</button>
    <button class="clear" onclick="setFilter('todo')">未标注</button>
    <button class="clear" onclick="setFilter('1')">裂缝</button>
    <button class="clear" onclick="setFilter('0')">非裂缝</button>
    <span>快捷键：<code>1</code> 裂缝，<code>0</code> 非裂缝，<code>?</code> 不确定，<code>n</code> 下一个未标注</span>
  </header>
  <main class="grid" id="grid"></main>
  <script>
    const records = {records_json};
    const storageKey = {json.dumps(storage_key)};
    const labels = JSON.parse(localStorage.getItem(storageKey) || "{{}}");
    let filter = "all";
    let activeIndex = 0;

    function getLabel(id) {{
      return labels[id] ?? "";
    }}

    function setLabel(id, value) {{
      if (value === "") delete labels[id];
      else labels[id] = value;
      localStorage.setItem(storageKey, JSON.stringify(labels));
      render();
    }}

    function labelName(value) {{
      if (value === "1") return "裂缝";
      if (value === "0") return "非裂缝";
      if (value === "-1") return "不确定";
      return "未标注";
    }}

    function visible(record) {{
      const lb = getLabel(record.candidate_id);
      if (filter === "all") return true;
      if (filter === "todo") return lb === "";
      return lb === filter;
    }}

    function setFilter(value) {{
      filter = value;
      render();
    }}

    function render() {{
      const grid = document.getElementById("grid");
      grid.innerHTML = "";
      let done = 0, yes = 0, no = 0, unsure = 0;
      for (const r of records) {{
        const lb = getLabel(r.candidate_id);
        if (lb !== "") done++;
        if (lb === "1") yes++;
        if (lb === "0") no++;
        if (lb === "-1") unsure++;
        if (!visible(r)) continue;

        const card = document.createElement("section");
        card.className = "card";
        card.dataset.label = lb;
        card.dataset.id = r.candidate_id;
        card.innerHTML = `
          <img class="candidate-img" src="${{r.overlay_path}}" loading="lazy">
          <div class="meta">
            <b>#${{r.candidate_id}}</b> ${{labelName(lb)}}<br>
            area=${{r.area}}, ratio=${{Number(r.major_minor_ratio).toFixed(2)}}, ecc=${{Number(r.eccentricity).toFixed(2)}}<br>
            ${{r.image_path}}
          </div>
          <div class="actions">
            <button class="yes" onclick="setLabel('${{r.candidate_id}}','1')">裂缝 1</button>
            <button class="no" onclick="setLabel('${{r.candidate_id}}','0')">非裂缝 0</button>
            <button class="unsure" onclick="setLabel('${{r.candidate_id}}','-1')">不确定 ?</button>
            <button class="clear" onclick="setLabel('${{r.candidate_id}}','')">清除</button>
          </div>
        `;
        grid.appendChild(card);
      }}
      document.getElementById("stats").textContent =
        `总数 ${{records.length}} | 已标 ${{done}} | 裂缝 ${{yes}} | 非裂缝 ${{no}} | 不确定 ${{unsure}}`;
    }}

    function csvEscape(value) {{
      const text = String(value ?? "");
      if (/[",\\n]/.test(text)) return `"${{text.replaceAll('"', '""')}}"`;
      return text;
    }}

    function exportCsv() {{
      const keys = Object.keys(records[0]).concat(["annotated_label"]);
      const lines = [keys.join(",")];
      for (const r of records) {{
        const row = keys.map(k => csvEscape(k === "annotated_label" ? getLabel(r.candidate_id) : r[k]));
        lines.push(row.join(","));
      }}
      const blob = new Blob([lines.join("\\n")], {{ type: "text/csv;charset=utf-8" }});
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = "labels.csv";
      a.click();
    }}

    function focusNextTodo() {{
      const idx = records.findIndex((r, i) => i > activeIndex && getLabel(r.candidate_id) === "");
      const next = idx >= 0 ? idx : records.findIndex(r => getLabel(r.candidate_id) === "");
      if (next >= 0) {{
        activeIndex = next;
        const el = document.querySelector(`[data-id="${{records[next].candidate_id}}"]`);
        if (el) el.scrollIntoView({{ behavior: "smooth", block: "center" }});
      }}
    }}

    document.addEventListener("keydown", (event) => {{
      const visibleCards = Array.from(document.querySelectorAll(".card"));
      const center = window.innerHeight / 2;
      let best = null, bestDist = Infinity;
      for (const card of visibleCards) {{
        const rect = card.getBoundingClientRect();
        const dist = Math.abs((rect.top + rect.bottom) / 2 - center);
        if (dist < bestDist) {{ best = card; bestDist = dist; }}
      }}
      if (!best) return;
      const id = best.dataset.id;
      activeIndex = records.findIndex(r => r.candidate_id === id);
      if (event.key === "1") setLabel(id, "1");
      if (event.key === "0") setLabel(id, "0");
      if (event.key === "?") setLabel(id, "-1");
      if (event.key.toLowerCase() === "n") focusNextTodo();
    }});

    render();
  </script>
</body>
</html>
"""
    html_path.write_text(html_text, encoding="utf-8")
    return html_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate an annotation set from SOM candidate crack components."
    )
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--target-count", type=int, default=1500)
    parser.add_argument("--max-components-per-image", type=int, default=8)
    parser.add_argument("--min-area", type=int, default=20)
    parser.add_argument("--min-bbox-side", type=int, default=5)
    parser.add_argument("--padding", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true", default=True)
    parser.add_argument("--no-shuffle", action="store_false", dest="shuffle")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir, records, processed_images, failed_images = collect_candidates(args)
    csv_path = write_csv(output_dir, records)
    html_path = write_html(output_dir, records)

    print("Done")
    print(f"Candidates: {len(records)}")
    print(f"Processed images: {processed_images}")
    print(f"Failed images: {failed_images}")
    print(f"CSV: {csv_path}")
    print(f"HTML: {html_path}")


if __name__ == "__main__":
    main()
