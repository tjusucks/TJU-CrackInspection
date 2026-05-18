from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from som_component_classifier import (
    FEATURE_NAMES,
    extract_component_features,
    features_to_array,
    save_classifier_bundle,
)
from som_crack_seg import read_image


DEFAULT_ANNOTATION_DIR = Path("~/som_component_annotation").expanduser()


def find_default_labels_csv(annotation_dir: Path):
    for name in ("labels.csv", "label.csv", "lable.csv"):
        path = annotation_dir / name
        if path.exists():
            return path
    return annotation_dir / "labels.csv"


def load_rows(labels_csv: Path):
    with labels_csv.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader)


def read_component_mask(row, labels_csv: Path, image_shape):
    base_dir = labels_csv.parent
    mask_path = Path(row["mask_path"])
    if not mask_path.is_absolute():
        mask_path = base_dir / mask_path

    mask_crop = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_crop is None:
        raise ValueError(f"Failed to read mask: {mask_path}")

    full_mask = np.zeros(image_shape[:2], dtype=bool)
    y1 = int(row["crop_min_row"])
    x1 = int(row["crop_min_col"])
    y2 = int(row["crop_max_row"])
    x2 = int(row["crop_max_col"])

    crop_h = min(y2 - y1, mask_crop.shape[0])
    crop_w = min(x2 - x1, mask_crop.shape[1])
    full_mask[y1 : y1 + crop_h, x1 : x1 + crop_w] = mask_crop[:crop_h, :crop_w] > 0
    return full_mask


def build_dataset(labels_csv: Path):
    rows = load_rows(labels_csv)
    feature_dicts = []
    labels = []
    groups = []
    used_rows = []
    skipped = {"blank": 0, "uncertain": 0, "failed": 0}

    image_cache = {}
    for row in rows:
        annotated = str(row.get("annotated_label", "")).strip()
        if annotated == "":
            skipped["blank"] += 1
            continue
        if annotated not in {"0", "1"}:
            skipped["uncertain"] += 1
            continue

        image_path = Path(row["image_path"]).expanduser()
        try:
            if image_path not in image_cache:
                image_cache[image_path] = read_image(image_path)
            image_bgr = image_cache[image_path]
            if image_bgr is None:
                raise ValueError(f"Failed to read image: {image_path}")
            component_mask = read_component_mask(row, labels_csv, image_bgr.shape)
            features = extract_component_features(image_bgr, component_mask)
        except Exception as exc:
            skipped["failed"] += 1
            print(f"SKIP candidate {row.get('candidate_id')} | {exc}")
            continue

        feature_dicts.append(features)
        labels.append(int(annotated))
        groups.append(str(image_path))
        used_rows.append(row)

    x = features_to_array(feature_dicts, FEATURE_NAMES)
    y = np.asarray(labels, dtype=np.int64)
    return x, y, np.asarray(groups), used_rows, skipped


def split_dataset(x, y, groups, test_size, seed):
    unique_groups = np.unique(groups)
    class_counts = np.bincount(y, minlength=2)
    if len(unique_groups) >= 2 and class_counts.min() >= 2:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(splitter.split(x, y, groups))
        if len(np.unique(y[train_idx])) == 2 and len(np.unique(y[test_idx])) == 2:
            return train_idx, test_idx

    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=y if np.bincount(y, minlength=2).min() >= 2 else None,
    )
    return train_idx, test_idx


def best_f1_threshold(y_true, probabilities):
    best_threshold = 0.5
    best_score = -1.0
    for threshold in np.linspace(0.2, 0.8, 61):
        pred = (probabilities >= threshold).astype(np.int64)
        score = f1_score(y_true, pred, zero_division=0)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold, best_score


def split_fit_validation(train_idx, y, val_size, seed):
    train_idx = np.asarray(train_idx)
    y_train = y[train_idx]
    if len(train_idx) < 10 or val_size <= 0:
        return train_idx, train_idx

    class_counts = np.bincount(y_train, minlength=2)
    stratify = y_train if class_counts.min() >= 2 else None
    fit_idx, val_idx = train_test_split(
        train_idx,
        test_size=val_size,
        random_state=seed,
        stratify=stratify,
    )
    if len(np.unique(y[fit_idx])) < 2 or len(np.unique(y[val_idx])) < 2:
        return train_idx, train_idx
    return fit_idx, val_idx


def metrics_dict(y_true, y_pred, probabilities):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    try:
        auc = float(roc_auc_score(y_true, probabilities))
    except ValueError:
        auc = None

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auc": auc,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "matrix_order": "[[TN, FP], [FN, TP]]",
            "matrix": matrix.tolist(),
        },
        "non_crack": {
            "precision": float(precision[0]),
            "recall": float(recall[0]),
            "f1": float(f1[0]),
            "support": int((y_true == 0).sum()),
        },
        "crack": {
            "precision": float(precision[1]),
            "recall": float(recall[1]),
            "f1": float(f1[1]),
            "support": int((y_true == 1).sum()),
        },
    }


def write_evaluation_files(
    model_out,
    report,
    used_rows,
    test_idx,
    y_true,
    y_pred,
    probabilities,
):
    output_dir = model_out.parent
    json_path = output_dir / "evaluation_report.json"
    txt_path = output_dir / "evaluation_report.txt"
    pred_csv_path = output_dir / "test_predictions.csv"

    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "SOM Supervised Crack Classifier Evaluation",
        "",
        f"Labels CSV: {report['labels_csv']}",
        f"Model: {report['model_out']}",
        f"Seed: {report['seed']}",
        f"Threshold: {report['threshold']:.3f}",
        f"Samples: {report['samples']['total']}",
        f"Train samples: {report['samples']['train']}",
        f"Validation samples: {report['samples']['validation']}",
        f"Test samples: {report['samples']['test']}",
        f"Positive crack samples: {report['samples']['positive']}",
        f"Negative non-crack samples: {report['samples']['negative']}",
        "",
        "Test Metrics",
        f"Accuracy: {report['test_metrics']['accuracy']:.4f}",
        f"AUC: {report['test_metrics']['auc']}",
        f"Crack precision: {report['test_metrics']['crack']['precision']:.4f}",
        f"Crack recall: {report['test_metrics']['crack']['recall']:.4f}",
        f"Crack F1: {report['test_metrics']['crack']['f1']:.4f}",
        f"Non-crack precision: {report['test_metrics']['non_crack']['precision']:.4f}",
        f"Non-crack recall: {report['test_metrics']['non_crack']['recall']:.4f}",
        f"Non-crack F1: {report['test_metrics']['non_crack']['f1']:.4f}",
        "",
        "Confusion Matrix [[TN, FP], [FN, TP]]",
        str(report["test_metrics"]["confusion_matrix"]["matrix"]),
    ]
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    fieldnames = [
        "candidate_id",
        "image_path",
        "true_label",
        "predicted_label",
        "crack_probability",
        "is_correct",
        "crop_path",
        "overlay_path",
        "mask_path",
    ]
    with pred_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, true_label, pred_label, probability in zip(
            test_idx,
            y_true,
            y_pred,
            probabilities,
        ):
            row = used_rows[int(idx)]
            writer.writerow(
                {
                    "candidate_id": row.get("candidate_id", ""),
                    "image_path": row.get("image_path", ""),
                    "true_label": int(true_label),
                    "predicted_label": int(pred_label),
                    "crack_probability": float(probability),
                    "is_correct": int(true_label == pred_label),
                    "crop_path": row.get("crop_path", ""),
                    "overlay_path": row.get("overlay_path", ""),
                    "mask_path": row.get("mask_path", ""),
                }
            )

    return json_path, txt_path, pred_csv_path


def parse_args():
    default_labels = find_default_labels_csv(DEFAULT_ANNOTATION_DIR)
    parser = argparse.ArgumentParser(description="Train a supervised crack/non-crack filter for SOM components.")
    parser.add_argument("--labels-csv", default=str(default_labels))
    parser.add_argument(
        "--model-out",
        default=str(DEFAULT_ANNOTATION_DIR / "som_crack_classifier.joblib"),
    )
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.2,
        help="Validation ratio inside the training split, used only to choose the probability threshold.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trees", type=int, default=400)
    return parser.parse_args()


def main():
    args = parse_args()
    labels_csv = Path(args.labels_csv).expanduser()
    model_out = Path(args.model_out).expanduser()

    x, y, groups, used_rows, skipped = build_dataset(labels_csv)
    if len(y) == 0:
        raise ValueError(f"No labeled samples found in {labels_csv}")
    if len(np.unique(y)) < 2:
        raise ValueError("Need both positive crack samples and negative non-crack samples.")

    train_idx, test_idx = split_dataset(x, y, groups, args.test_size, args.seed)
    fit_idx, val_idx = split_fit_validation(train_idx, y, args.val_size, args.seed)

    model = RandomForestClassifier(
        n_estimators=args.trees,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=args.seed,
        n_jobs=-1,
    )
    model.fit(x[fit_idx], y[fit_idx])

    val_prob = model.predict_proba(x[val_idx])[:, 1]
    test_prob = model.predict_proba(x[test_idx])[:, 1]
    threshold, threshold_f1 = (
        (float(args.threshold), None)
        if args.threshold is not None
        else best_f1_threshold(y[val_idx], val_prob)
    )
    val_pred = (val_prob >= threshold).astype(np.int64)
    test_pred = (test_prob >= threshold).astype(np.int64)
    val_metrics = metrics_dict(y[val_idx], val_pred, val_prob)
    test_metrics = metrics_dict(y[test_idx], test_pred, test_prob)

    bundle = {
        "model": model,
        "feature_names": FEATURE_NAMES,
        "threshold": threshold,
        "labels_csv": str(labels_csv),
        "positive_label": 1,
        "negative_label": 0,
        "training_summary": {
            "samples": int(len(y)),
            "positive": int((y == 1).sum()),
            "negative": int((y == 0).sum()),
            "train_samples": int(len(fit_idx)),
            "validation_samples": int(len(val_idx)),
            "test_samples": int(len(test_idx)),
            "skipped": skipped,
            "threshold": threshold,
            "threshold_f1": None if threshold_f1 is None else float(threshold_f1),
            "test_metrics": test_metrics,
        },
    }
    save_classifier_bundle(bundle, model_out)

    importances = sorted(
        zip(FEATURE_NAMES, model.feature_importances_),
        key=lambda item: item[1],
        reverse=True,
    )[:12]
    report = {
        "labels_csv": str(labels_csv),
        "model_out": str(model_out),
        "seed": int(args.seed),
        "test_size": float(args.test_size),
        "val_size": float(args.val_size),
        "threshold": float(threshold),
        "threshold_source": "argument" if args.threshold is not None else "validation_best_f1",
        "validation_best_f1": None if threshold_f1 is None else float(threshold_f1),
        "samples": {
            "total": int(len(y)),
            "positive": int((y == 1).sum()),
            "negative": int((y == 0).sum()),
            "train": int(len(fit_idx)),
            "validation": int(len(val_idx)),
            "test": int(len(test_idx)),
            "skipped": skipped,
        },
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "top_feature_importances": [
            {"name": name, "importance": float(importance)}
            for name, importance in importances
        ],
    }
    eval_json, eval_txt, pred_csv = write_evaluation_files(
        model_out=model_out,
        report=report,
        used_rows=used_rows,
        test_idx=test_idx,
        y_true=y[test_idx],
        y_pred=test_pred,
        probabilities=test_prob,
    )

    print("Done")
    print(f"Labels: {labels_csv}")
    print(f"Model: {model_out}")
    print(f"Samples: {len(y)} | crack={int((y == 1).sum())} | non_crack={int((y == 0).sum())}")
    print(f"Skipped: {json.dumps(skipped, ensure_ascii=False)}")
    print(f"Threshold: {threshold:.3f}")
    print(f"Evaluation JSON: {eval_json}")
    print(f"Evaluation TXT: {eval_txt}")
    print(f"Test predictions CSV: {pred_csv}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y[test_idx], test_pred, labels=[0, 1]))
    print(classification_report(y[test_idx], test_pred, target_names=["non_crack", "crack"], zero_division=0))
    print("Top feature importances:")
    for name, importance in importances:
        print(f"  {name}: {importance:.4f}")


if __name__ == "__main__":
    main()
