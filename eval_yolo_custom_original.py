import os
import glob
import json
from pathlib import Path

import yaml
import cv2
import numpy as np

# ==============================
# 可調整參數
# ==============================
PROJECT_ROOT = Path(__file__).resolve().parent

YAML_PATH = PROJECT_ROOT / "aortic_valve_colab.yaml"

GT_LABEL_DIR = PROJECT_ROOT / "datasets" / "test" / "labels"
PRED_LABEL_DIR = PROJECT_ROOT / "runs" / "detect" / "val" / "labels"
IMAGE_DIR = PROJECT_ROOT / "datasets" / "test" / "images"

# IoU threshold 用來判定 TP / FP / FN
IOU_THRESHOLD = 0.5

# 可視化輸出資料夾
VIS_OUTPUT_DIR = PROJECT_ROOT / "eval_vis"

# 摘要報告輸出
SUMMARY_NDJSON = PROJECT_ROOT / "eval_summary.ndjson"
GLOBAL_SUMMARY_JSON = PROJECT_ROOT / "eval_global_summary.json"


# ==============================
# 工具函式
# ==============================
def load_class_names(yaml_path):
    if not yaml_path.exists():
        print(f"[WARN] YAML file not found: {yaml_path}, will use numeric class ids only.")
        return {}
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    # YOLO 通常 names 是 list 或 dict
    names = data.get("names", {})
    if isinstance(names, dict):
        # {0: "cls0", 1: "cls1", ...}
        return {int(k): v for k, v in names.items()}
    elif isinstance(names, list):
        return {i: name for i, name in enumerate(names)}
    else:
        return {}


def xywhn_to_xyxy(x, y, w, h, img_w, img_h):
    """ YOLO normalized xywh -> pixel xyxy """
    cx = x * img_w
    cy = y * img_h
    bw = w * img_w
    bh = h * img_h
    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2
    return [x1, y1, x2, y2]


def box_iou(box1, box2):
    """
    box: [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    if inter_area <= 0:
        return 0.0

    area1 = max(0.0, (box1[2] - box1[0])) * max(0.0, (box1[3] - box1[1]))
    area2 = max(0.0, (box2[2] - box2[0])) * max(0.0, (box2[3] - box2[1]))

    union = area1 + area2 - inter_area
    if union <= 0:
        return 0.0

    return inter_area / union


def read_yolo_label_file(path, is_pred=False):
    """
    讀取 YOLO txt 標籤檔。
    GT 通常:  class x y w h
    Pred 通常: class x y w h conf
    回傳: list of dict: {class_id, x, y, w, h, conf(optional)}
    """
    boxes = []
    if not path.exists():
        return boxes
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            class_id = int(float(parts[0]))
            x = float(parts[1])
            y = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
            conf = float(parts[5]) if (is_pred and len(parts) >= 6) else None
            boxes.append(
                {
                    "class_id": class_id,
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "conf": conf,
                }
            )
    return boxes


def match_predictions_to_gt(gt_boxes, pred_boxes, img_w, img_h, iou_thres=0.5):
    """
    以 IoU 貪婪匹配預測與 GT。
    回傳:
      matches: list of (gt_idx, pred_idx, iou)
      unmatched_gt_indices: set
      unmatched_pred_indices: set
      duplicate_gt_indices: set  # 被多個預測匹配到的 GT
    """
    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
        return [], set(), set(), set()

    # 預先轉成 pixel 坐標
    gt_xyxy = []
    for b in gt_boxes:
        gt_xyxy.append(xywhn_to_xyxy(b["x"], b["y"], b["w"], b["h"], img_w, img_h))
    pred_xyxy = []
    for b in pred_boxes:
        pred_xyxy.append(xywhn_to_xyxy(b["x"], b["y"], b["w"], b["h"], img_w, img_h))

    gt_cnt = len(gt_boxes)
    pred_cnt = len(pred_boxes)

    iou_matrix = np.zeros((gt_cnt, pred_cnt), dtype=float)

    # 只計算同 class 的 iou，其餘保持 0
    for gi in range(gt_cnt):
        for pi in range(pred_cnt):
            if gt_boxes[gi]["class_id"] != pred_boxes[pi]["class_id"]:
                iou_matrix[gi, pi] = 0.0
            else:
                iou_matrix[gi, pi] = box_iou(gt_xyxy[gi], pred_xyxy[pi])

    matches = []
    used_gt = set()
    used_pred = set()
    duplicate_gt = set()

    # 貪婪匹配：把所有 (gi, pi) 按 iou 排序，由大到小
    all_pairs = [
        (gi, pi, iou_matrix[gi, pi])
        for gi in range(gt_cnt)
        for pi in range(pred_cnt)
        if iou_matrix[gi, pi] >= iou_thres
    ]
    all_pairs.sort(key=lambda x: x[2], reverse=True)

    for gi, pi, iou in all_pairs:
        if pi in used_pred:
            continue
        if gi in used_gt:
            # 這代表這個 GT 已經配過一次，又被另一個 pred 配到
            duplicate_gt.add(gi)
            # 這個 pred 算是多餘的偵測
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        matches.append((gi, pi, iou))

    unmatched_gt = set(range(gt_cnt)) - used_gt
    unmatched_pred = set(range(pred_cnt)) - used_pred

    return matches, unmatched_gt, unmatched_pred, duplicate_gt


def draw_boxes(
    img,
    gt_boxes,
    pred_boxes,
    matches,
    unmatched_gt_idx,
    unmatched_pred_idx,
    img_w,
    img_h,
    class_names,
):
    """
    在影像上畫出 GT 與預測框
    - GT: 綠色 (0,255,0)
    - Pred TP: 藍色 (255,0,0)
    - Pred FP: 紅色 (0,0,255)
    """
    # 建立 pred_idx -> matched_gt_idx map
    pred_to_gt = {}
    for gi, pi, _ in matches:
        pred_to_gt[pi] = gi

    # 畫 GT 框
    for gi, box in enumerate(gt_boxes):
        x1, y1, x2, y2 = xywhn_to_xyxy(box["x"], box["y"], box["w"], box["h"], img_w, img_h)
        cls_id = box["class_id"]
        cls_name = class_names.get(cls_id, str(cls_id))
        color = (0, 255, 0)  # green
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=2)
        cv2.putText(
            img,
            f"GT:{cls_name}",
            (int(x1), max(0, int(y1) - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )

    # 畫 Pred 框
    for pi, box in enumerate(pred_boxes):
        x1, y1, x2, y2 = xywhn_to_xyxy(box["x"], box["y"], box["w"], box["h"], img_w, img_h)
        cls_id = box["class_id"]
        conf = box["conf"]
        cls_name = class_names.get(cls_id, str(cls_id))
        if pi in pred_to_gt:
            color = (255, 0, 0)  # blue for TP
            tag = "TP"
        else:
            color = (0, 0, 255)  # red for FP
            tag = "FP"

        label = f"{tag}:{cls_name}"
        if conf is not None:
            label += f" {conf:.2f}"

        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=2)
        cv2.putText(
            img,
            label,
            (int(x1), min(img_h - 5, int(y1) + 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )

    return img


# ==============================
# 主流程
# ==============================
def main():
    class_names = load_class_names(YAML_PATH)
    VIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    gt_label_files = sorted(glob.glob(str(GT_LABEL_DIR / "*.txt")))
    if not gt_label_files:
        print(f"[ERROR] No GT label files found in {GT_LABEL_DIR}")
        return

    image_level_records = []

    global_tp = 0
    global_fp = 0
    global_fn = 0
    global_duplicate_dets = 0

    for gt_path_str in gt_label_files:
        gt_path = Path(gt_path_str)
        stem = gt_path.stem  # e.g. image name without extension

        # 找影像檔案（支援 jpg/png/jpeg）
        img_file = None
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            candidate = IMAGE_DIR / f"{stem}{ext}"
            if candidate.exists():
                img_file = candidate
                break

        if img_file is None:
            print(f"[WARN] Image for {stem} not found under {IMAGE_DIR}, skip visualization.")
            img = None
            img_w = img_h = None
        else:
            img = cv2.imread(str(img_file))
            if img is None:
                print(f"[WARN] Failed to read image {img_file}")
                img_w = img_h = None
            else:
                img_h, img_w = img.shape[:2]

        gt_boxes = read_yolo_label_file(gt_path, is_pred=False)

        pred_path = PRED_LABEL_DIR / gt_path.name
        pred_boxes = read_yolo_label_file(pred_path, is_pred=True)

        if img_w is None or img_h is None:
            # 若沒影像，IoU 仍可以計算嗎？(理論上不行，因為沒有寬高)
            # 這邊保守處理：略過此圖的 IoU 評估與可視化，但仍可做簡單統計（只看數量），
            # 不過這樣沒法做重疊程度分析，所以直接跳過此圖。
            print(f"[WARN] No valid image size for {stem}, skip IoU/matching.")
            continue

        # IoU 匹配
        matches, unmatched_gt, unmatched_pred, duplicate_gt = match_predictions_to_gt(
            gt_boxes, pred_boxes, img_w, img_h, IOU_THRESHOLD
        )

        tp = len(matches)
        fn = len(unmatched_gt)
        fp = len(unmatched_pred)
        duplicate_count = len(duplicate_gt)

        global_tp += tp
        global_fp += fp
        global_fn += fn
        global_duplicate_dets += duplicate_count

        # 圖像可視化
        if img is not None:
            vis_img = img.copy()
            vis_img = draw_boxes(
                vis_img,
                gt_boxes,
                pred_boxes,
                matches,
                unmatched_gt,
                unmatched_pred,
                img_w,
                img_h,
                class_names,
            )
            out_path = VIS_OUTPUT_DIR / f"{stem}_vis.jpg"
            cv2.imwrite(str(out_path), vis_img)

        # 每張圖的詳細紀錄（會寫入 NDJSON）
        record = {
            "image_id": stem,
            "num_gt": len(gt_boxes),
            "num_pred": len(pred_boxes),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "duplicate_gt_count": duplicate_count,
            "iou_threshold": IOU_THRESHOLD,
            "gt_missing_indices": sorted(list(unmatched_gt)),
            "pred_spurious_indices": sorted(list(unmatched_pred)),
            # 可選：列出匹配對以及對應 IoU
            "matches": [
                {
                    "gt_index": int(gi),
                    "pred_index": int(pi),
                    "iou": float(iou),
                    "class_id": int(gt_boxes[gi]["class_id"]),
                }
                for gi, pi, iou in matches
            ],
        }

        image_level_records.append(record)

    # 輸出 NDJSON（每行一個 image 的 JSON）
    with open(SUMMARY_NDJSON, "w", encoding="utf-8") as f:
        for rec in image_level_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 全局統計
    precision = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0.0
    recall = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0.0

    global_summary = {
        "iou_threshold": IOU_THRESHOLD,
        "total_images": len(image_level_records),
        "total_tp": global_tp,
        "total_fp": global_fp,
        "total_fn": global_fn,
        "total_duplicate_gt": global_duplicate_dets,
        "precision": precision,
        "recall": recall,
    }

    with open(GLOBAL_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(global_summary, f, ensure_ascii=False, indent=2)

    print("=== Evaluation Finished ===")
    print(f"Images evaluated: {global_summary['total_images']}")
    print(f"TP: {global_tp}, FP: {global_fp}, FN: {global_fn}, Duplicated GT: {global_duplicate_dets}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}")
    print(f"Per-image NDJSON: {SUMMARY_NDJSON}")
    print(f"Global summary JSON: {GLOBAL_SUMMARY_JSON}")
    print(f"Visualization images saved to: {VIS_OUTPUT_DIR}")


if __name__ == "__main__":
    main()