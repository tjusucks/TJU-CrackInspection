from __future__ import annotations

import math

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize


def skeleton_length_px(skeleton):
    skeleton = skeleton.astype(bool)
    if skeleton.sum() <= 1:
        return float(skeleton.sum())

    horizontal = np.logical_and(skeleton[:, :-1], skeleton[:, 1:]).sum()
    vertical = np.logical_and(skeleton[:-1, :], skeleton[1:, :]).sum()
    diag1 = np.logical_and(skeleton[:-1, :-1], skeleton[1:, 1:]).sum()
    diag2 = np.logical_and(skeleton[:-1, 1:], skeleton[1:, :-1]).sum()
    return float(horizontal + vertical + math.sqrt(2.0) * (diag1 + diag2))


def endpoint_count(skeleton):
    skeleton = skeleton.astype(bool)
    if skeleton.sum() == 0:
        return 0
    neighbors = ndi.convolve(
        skeleton.astype(np.uint8),
        np.ones((3, 3), dtype=np.uint8),
        mode="constant",
        cval=0,
    )
    return int(np.logical_and(skeleton, neighbors == 2).sum())


def _scaled(value, mm_per_pixel):
    return None if mm_per_pixel is None else float(value) * float(mm_per_pixel)


def measure_crack_components(mask_bool, mm_per_pixel=None, min_area=1):
    mask_bool = mask_bool.astype(bool)
    labeled = label(mask_bool)
    measurements = []

    for index, region in enumerate(regionprops(labeled), start=1):
        if region.area < min_area:
            continue

        minr, minc, maxr, maxc = region.bbox
        component = labeled == region.label
        crop = component[minr:maxr, minc:maxc]
        skeleton_crop = skeletonize(crop)
        length_px = skeleton_length_px(skeleton_crop)
        area_px = int(region.area)
        avg_width_px = area_px / max(length_px, 1.0)
        distance = cv2.distanceTransform(crop.astype(np.uint8), cv2.DIST_L2, 5)
        max_width_px = float(distance.max() * 2.0)
        bbox_width_px = int(maxc - minc)
        bbox_height_px = int(maxr - minr)

        measurements.append(
            {
                "crack_id": int(index),
                "component_label": int(region.label),
                "area_px": area_px,
                "length_px": length_px,
                "avg_width_px": avg_width_px,
                "max_width_px": max_width_px,
                "bbox_width_px": bbox_width_px,
                "bbox_height_px": bbox_height_px,
                "bbox_min_row": int(minr),
                "bbox_min_col": int(minc),
                "bbox_max_row": int(maxr),
                "bbox_max_col": int(maxc),
                "orientation_deg": float(np.degrees(region.orientation)),
                "eccentricity": float(region.eccentricity),
                "endpoint_count": endpoint_count(skeleton_crop),
                "length_mm": _scaled(length_px, mm_per_pixel),
                "avg_width_mm": _scaled(avg_width_px, mm_per_pixel),
                "max_width_mm": _scaled(max_width_px, mm_per_pixel),
                "bbox_width_mm": _scaled(bbox_width_px, mm_per_pixel),
                "bbox_height_mm": _scaled(bbox_height_px, mm_per_pixel),
                "area_mm2": None
                if mm_per_pixel is None
                else float(area_px) * float(mm_per_pixel) * float(mm_per_pixel),
            }
        )

    measurements.sort(key=lambda item: item["length_px"], reverse=True)
    for index, item in enumerate(measurements, start=1):
        item["crack_id"] = index
    return measurements


def summarize_measurements(measurements, image_shape, mm_per_pixel=None):
    h, w = image_shape[:2]
    total_area_px = sum(item["area_px"] for item in measurements)
    total_length_px = sum(item["length_px"] for item in measurements)
    max_width_px = max((item["max_width_px"] for item in measurements), default=0.0)
    avg_width_px = total_area_px / max(total_length_px, 1.0)

    return {
        "image_width_px": int(w),
        "image_height_px": int(h),
        "mm_per_pixel": None if mm_per_pixel is None else float(mm_per_pixel),
        "crack_count": int(len(measurements)),
        "has_crack": bool(len(measurements) > 0),
        "total_area_px": int(total_area_px),
        "total_length_px": float(total_length_px),
        "average_width_px": float(avg_width_px),
        "max_width_px": float(max_width_px),
        "crack_area_ratio": total_area_px / float(max(h * w, 1)),
        "total_area_mm2": None
        if mm_per_pixel is None
        else float(total_area_px) * float(mm_per_pixel) * float(mm_per_pixel),
        "total_length_mm": _scaled(total_length_px, mm_per_pixel),
        "average_width_mm": _scaled(avg_width_px, mm_per_pixel),
        "max_width_mm": _scaled(max_width_px, mm_per_pixel),
    }


def draw_measurement_overlay(image_bgr, mask_bool, measurements, mm_per_pixel=None):
    overlay = image_bgr.copy()
    color = np.zeros_like(image_bgr)
    color[mask_bool.astype(bool)] = (0, 0, 255)
    overlay = cv2.addWeighted(overlay, 1.0, color, 0.45, 0)

    for item in measurements:
        x1 = item["bbox_min_col"]
        y1 = item["bbox_min_row"]
        x2 = item["bbox_max_col"]
        y2 = item["bbox_max_row"]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)
        if mm_per_pixel is None:
            text = f"#{item['crack_id']} L={item['length_px']:.1f}px W={item['avg_width_px']:.1f}px"
        else:
            text = f"#{item['crack_id']} L={item['length_mm']:.1f}mm W={item['avg_width_mm']:.1f}mm"
        cv2.putText(
            overlay,
            text,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return overlay
