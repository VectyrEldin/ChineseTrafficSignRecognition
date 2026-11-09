#!/usr/bin/env python3
"""eval_by_size.py —— 按论文 Table 3 的尺寸分档评测 + 背景误检 + 翻转一致性诊断。

论文 (Zhu et al., IEEE Conference on Computer Vision and Pattern Recognition 2016)
用 COCO 式尺寸分档证明其网络在小标志上远优于
Fast R-CNN: small(0,32] 的 recall 0.87 vs 0.24。本项目的 train.py 只打印
聚合 mAP, 43.7% 的 small 框被平均掉，无法定位小目标短板，也无法与论文对照。

三类诊断：
  1) size_bins   : 按 GT 等效边长 ≤32 / 32-96 / >96 px 分档统计
                   recall@IoU0.5、precision@IoU0.5 (微平均，非 COCO mAP) ;
                   同时给出每个 GT 在 imgsz 下的等效边长。
  2) background  : 纯背景图的 FPPI (每图误检框数) 与图像级误检率，
                   把一次性的 9.7% 误检变成可回归指标。
  3) flip        : 原图 / 水平翻转图的 top-1 类别一致率，用于量化
                   fliplr=0.5 是否给方向类/数值类制造标签噪声。

用法：
  uv run python eval_by_size.py \
      --weights runs/detect/tt100k45_yolo26s/weights/best.pt \
      --data ../data/tt100k45_yolo/tt100k45.yaml --split test \
      --background-dir ../data/tt100k_2021/other \
      --annotation ../data/tt100k_2021/annotations_all.json \
      --flip-limit 50
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = SCRIPT_DIR / "eval_by_size.json"

# 论文 §5 / Table 3 的阈值，单位为「原始 2048×2048 图」中的等效边长像素
SIZE_BINS = (
    ("small(0,32]", 0.0, 32.0),
    ("medium(32,96]", 32.0, 96.0),
    ("large(96,400]", 96.0, float("inf")),
)

# 与 export_onnx.py 的 nms=False 导出、train.py 的训练内 val 保持同一检测头。
# 不传 nms 时 Ultralytics 默认用 one-to-many+NMS 头，评测会与部署口径错开。
DEPLOY_NMS = False


# ==================== 数据 ====================
def load_dataset(data_yaml: Path, split: str) -> tuple[list[Path], list[Path], list[str]]:
    cfg_data: Any = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    if not isinstance(cfg_data, dict):
        raise TypeError(
            f"YAML 文件根节点必须是字典，但 {data_yaml} 加载出了 {type(cfg_data).__name__}"
        )

    cfg = cfg_data

    names = list(cfg.get("names", []))
    entry = cfg[split]
    if cfg.get("path"):
        root = Path(cfg["path"])
        root = (data_yaml.parent / root).resolve() if not root.is_absolute() else root
    else:
        # 与 Ultralytics check_det_dataset 一致：path 缺省时根 = yaml 所在目录
        root = data_yaml.parent.resolve()
    img_root = (root / entry).resolve()
    images = (
        sorted(
            p for p in img_root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
        if img_root.is_dir()
        else []
    )
    if not images and img_root.is_file():
        # 支持 train: train_balanced.txt（类感知重采样清单）
        lines = [l.strip() for l in img_root.read_text(encoding="utf-8").splitlines() if l.strip()]
        images = sorted({(img_root.parent / l.replace("./", "", 1)).resolve() for l in lines})
    return images, [to_label_path(p) for p in images], names


def to_label_path(image_path: Path) -> Path:
    """与 Ultralytics img2label_paths 一致：把最后一段 /images/ 换成 /labels/。"""
    sep = os.sep
    replaced = str(image_path).replace(f"{sep}images{sep}", f"{sep}labels{sep}")
    return Path(replaced).with_suffix(".txt")


def read_gt(label_path: Path, width: int, height: int) -> list[tuple[int, list[float]]]:
    """YOLO 归一化标注 → 原图像素框 (x1,y1,x2,y2)。

    同时兼容两种行格式：
      · 5 列检测行      `cls cx cy w h`
      · >6 列 segment 行 `cls x1 y1 x2 y2 ...` (convert_tt100k.py --emit-segments)
    segment 行由多边形的 min/max 反算外接框，与 Ultralytics segments2boxes 同口径。
    """
    if not label_path.is_file():
        return []
    out = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        cid = int(p[0])
        if len(p) > 6:
            pts = [float(v) for v in p[1:]]
            xs, ys = pts[0::2], pts[1::2]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            out.append((cid, [x1 * width, y1 * height, x2 * width, y2 * height]))
            continue
        cx, cy, w, h = (float(v) for v in p[1:5])
        x1, y1 = (cx - w / 2) * width, (cy - h / 2) * height
        x2, y2 = (cx + w / 2) * width, (cy + h / 2) * height
        out.append((cid, [x1, y1, x2, y2]))
    return out


def iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (
        max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        + max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        - inter
    )
    return inter / ua if ua > 0 else 0.0


def bin_of(side: float) -> str:
    for name, lo, hi in SIZE_BINS:
        if lo <= side <= hi:
            return name
    return SIZE_BINS[-1][0]


# ==================== 推理 ====================
def predict(model, image_or_path, imgsz: int, conf: float) -> list[tuple[int, float, list[float]]]:
    results = model.predict(
        source=image_or_path, conf=conf, imgsz=imgsz, nms=DEPLOY_NMS, verbose=False, stream=False
    )
    first = next(iter(results), None)
    if first is None or first.boxes is None:
        return []
    rows = np.asarray(first.boxes.data.cpu().numpy()).tolist()
    detections = [(round(r[5]), float(r[4]), [float(v) for v in r[:4]]) for r in rows]
    detections.sort(key=lambda d: -d[1])
    return detections


def match(
    gt: list[tuple[int, list[float]]],
    detections: list[tuple[int, float, list[float]]],
    thr: float = 0.5,
) -> tuple[set[int], set[int]]:
    """类别感知 + IoU 贪心匹配，返回 (已匹配 GT 下标, 已匹配 det 下标)。"""
    pairs = []
    for gi, (gc, gb) in enumerate(gt):
        for di, (dc, _, db) in enumerate(detections):
            if gc != dc:
                continue
            v = iou(gb, db)
            if v >= thr:
                pairs.append((v, gi, di))
    pairs.sort(reverse=True)
    used_g: set[int] = set()
    used_d: set[int] = set()
    for _, gi, di in pairs:
        if gi in used_g or di in used_d:
            continue
        used_g.add(gi)
        used_d.add(di)
    return used_g, used_d


# ==================== 1) 尺寸分档 ====================
def _safe_div(numerator: float, denominator: float, digits: int = 4) -> float | None:
    """安全除法，避免在每个调用点写 `if denom else None`。"""
    if denominator:
        return round(numerator / denominator, digits)
    return None


# ==================== 推理 ====================
def _process_single_image(
    img_path: Path, lbl_path: Path, model, imgsz: int, conf: float
) -> tuple[list, list, float] | None:  # 修改返回类型
    """处理单张图片，返回 (gt_boxes, detections, net_scale)，失败返回 None。"""
    img = cv2.imread(str(img_path))
    if img is None:
        return None  # 直接返回 None，不再用 0.0 做哨兵
    h, w = img.shape[:2]
    net_scale = imgsz / max(w, h)
    gt = read_gt(lbl_path, w, h)
    detections = predict(model, str(img_path), imgsz, conf)
    return gt, detections, net_scale


def _update_stats_for_image(
    gt: list, detections: list, used_g: set[int], net_scale: float, stats: dict, per_class: dict
) -> None:
    """将单张图片的 GT 和检测框累加到统计字典中。"""
    for gi, (cid, box) in enumerate(gt):
        side = math.sqrt(max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]))
        b = bin_of(side)
        stats[b]["gt"] += 1
        stats[b]["net_px_sum"] += side * net_scale
        per_class[cid]["gt"] += 1
        if gi in used_g:
            stats[b]["tp"] += 1
            per_class[cid]["tp"] += 1

    for _, _, detection_box in detections:
        side = math.sqrt(
            max(0.0, detection_box[2] - detection_box[0])
            * max(0.0, detection_box[3] - detection_box[1])
        )
        stats[bin_of(side)]["det"] += 1


def _compute_size_stats(stats: dict) -> dict:
    """根据累计统计计算最终的尺寸分档指标。"""
    out = {}
    for name, st in stats.items():
        gt_n, tp, det = st["gt"], st["tp"], st["det"]
        out[name] = {
            "gt_boxes": gt_n,
            "matched": tp,
            "detections": det,
            "recall_iou50": _safe_div(tp, gt_n),
            "precision_iou50": _safe_div(tp, det),
            "mean_net_equivalent_px": _safe_div(st["net_px_sum"], gt_n, digits=2),
        }
    return out


def _compute_class_recall(per_class: dict, names: list[str]) -> dict:
    """计算各类别的 recall (仅返回合法类别)。"""
    result = {}
    for c, v in sorted(per_class.items()):
        if 0 <= c < len(names):
            result[names[c]] = {
                "gt": v["gt"],
                "recall_iou50": _safe_div(v["tp"], v["gt"]),
            }
    return result


# ==================== 1) 尺寸分档 ====================
def eval_sizes(model, images, labels, names, imgsz: int, conf: float, limit: int) -> dict:
    stats: dict[str, dict] = {
        name: {"gt": 0, "tp": 0, "det": 0, "net_px_sum": 0.0} for name, _, _ in SIZE_BINS
    }
    per_class: dict[int, dict] = defaultdict(lambda: {"gt": 0, "tp": 0})
    n = min(len(images), limit) if limit > 0 else len(images)
    print(f"\n[1/3] 尺寸分档评测：{n} 张图 ...")

    for idx, (img_path, lbl_path) in enumerate(zip(images[:n], labels[:n])):
        result = _process_single_image(img_path, lbl_path, model, imgsz, conf)
        if result is None:  # 清晰的错误状态判断，无浮点比较
            continue
        gt, detections, net_scale = result

        used_g, _ = match(gt, detections)
        _update_stats_for_image(gt, detections, used_g, net_scale, stats, per_class)

        if (idx + 1) % 200 == 0:
            print(f"      {idx + 1}/{n} ...")

    return {
        "split_images": n,
        "imgsz": imgsz,
        "conf": conf,
        "bins": _compute_size_stats(stats),
        "per_class": _compute_class_recall(per_class, names),
    }


# ==================== 2) 背景 FPPI ====================
def eval_background(
    model, dirs: list[Path], annotation: Path | None, imgsz: int, conf: float, limit: int
) -> dict:
    registered: set[str] = set()
    if annotation and annotation.is_file():
        ann = json.loads(annotation.read_text(encoding="utf-8"))
        for img_data in ann.get("imgs", {}).values():
            registered.add(Path(str(img_data.get("path", ""))).name)
            registered.add(f"{img_data.get('id')}.jpg")

    candidates = sorted(
        {p for d in dirs if d.is_dir() for p in d.glob("*.jpg") if p.name not in registered},
        key=lambda p: hashlib.sha256(p.name.encode()).hexdigest(),
    )
    chosen = candidates[:limit] if limit > 0 else candidates
    print(f"\n[2/3] 背景误检评测：{len(chosen)} 张纯背景图 ...")
    if not chosen:
        return {"images": 0, "note": "未找到无标注背景图"}

    boxes = 0
    images_with_fp = 0
    confidence_scores: list[float] = []
    for p in chosen:
        detections = predict(model, str(p), imgsz, conf)
        boxes += len(detections)
        if detections:
            images_with_fp += 1
            confidence_scores.append(detections[0][1])
    return {
        "images": len(chosen),
        "total_false_boxes": boxes,
        "fppi": round(boxes / len(chosen), 4),
        "image_fp_rate": round(images_with_fp / len(chosen), 4),
        "mean_top_conf_when_fp": round(sum(confidence_scores) / len(confidence_scores), 4)
        if confidence_scores
        else None,
        "conf": conf,
    }


# ==================== 3) 翻转一致性 ====================
def eval_flip(model, images, names, imgsz: int, conf: float, limit: int) -> dict:
    chosen = images[:limit] if limit > 0 else images
    print(f"\n[3/3] 翻转一致性诊断：{len(chosen)} 张图 ...")
    compared = 0
    disagree = 0
    per_class: Counter[str] = Counter()
    per_class_total: Counter[str] = Counter()

    for img_path in chosen:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        a = predict(model, img, imgsz, conf)
        b = predict(model, cv2.flip(img, 1), imgsz, conf)
        if not a or not b:
            continue
        compared += 1
        ca, cb = a[0][0], b[0][0]
        na = names[ca] if 0 <= ca < len(names) else str(ca)
        per_class_total[na] += 1
        if ca != cb:
            disagree += 1
            per_class[na] += 1

    top = sorted(
        (
            {
                "class": k,
                "flips": v,
                "compared": per_class_total[k],
                "disagree_rate": round(v / per_class_total[k], 4),
            }
            for k, v in per_class.items()
        ),
        key=lambda x: -x["flips"],
    )[:12]
    return {
        "compared_images": compared,
        "top1_disagreement_rate": round(disagree / compared, 4) if compared else None,
        "note": "top-1 类别在水平翻转后改变的比例；高值说明 fliplr 会制造标签噪声",
        "top_classes": top,
    }


# ==================== main ====================
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="尺寸分档 / 背景误检 / 翻转一致性评测")
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--data", required=True, type=Path, help="数据集 yaml")
    ap.add_argument("--split", default="test", choices=("train", "val", "test"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--limit", type=int, default=0, help="分档评测最多图片数，0=全部")
    ap.add_argument("--bg-dir", action="append", default=[], help="纯背景图目录，可重复指定")
    ap.add_argument(
        "--annotation",
        type=Path,
        default=None,
        help="annotations_all.json，用于排除已登记的有标注图",
    )
    ap.add_argument("--bg-limit", type=int, default=300)
    ap.add_argument("--flip-limit", type=int, default=50)
    ap.add_argument("--skip-flip", action="store_true")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if not args.weights.is_file():
        raise SystemExit(f"[FAIL] 权重不存在: {args.weights}")

    from ultralytics import YOLO

    images, labels, names = load_dataset(args.data, args.split)
    print(
        f"[INFO] split={args.split} 图片={len(images)} 类别={len(names)} "
        f"imgsz={args.imgsz} conf={args.conf}"
    )
    if not images:
        raise SystemExit("[FAIL] 该 split 未找到图片")

    model = YOLO(str(args.weights))
    result: dict = {
        "weights": str(args.weights),
        "data": str(args.data),
        "split": args.split,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "num_classes": len(names),
        "size_bins": eval_sizes(model, images, labels, names, args.imgsz, args.conf, args.limit),
    }
    if args.bg_dir:
        result["background"] = eval_background(
            model,
            [Path(d) for d in args.bg_dir],
            args.annotation,
            args.imgsz,
            args.conf,
            args.bg_limit,
        )
    if not args.skip_flip:
        result["flip_consistency"] = eval_flip(
            model,
            images,
            names,
            args.imgsz,
            args.conf,
            args.flip_limit,
        )

    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 68)
    print("尺寸分档（论文 Table 3 口径, recall/precision 为 IoU0.5 微平均）")
    print("=" * 68)
    for name, st in result["size_bins"]["bins"].items():
        print(
            f"  {name:16s} GT={st['gt_boxes']:6d}  TP={st['matched']:6d}  "
            f"R={st['recall_iou50']}  P={st['precision_iou50']}  "
            f"网络等效边长={st['mean_net_equivalent_px']} px"
        )
    if "background" in result:
        bg = result["background"]
        print(
            f"\n背景误检: {bg['images']} 图  FPPI={bg.get('fppi')}  "
            f"图像级误检率={bg.get('image_fp_rate')}"
        )
    if "flip_consistency" in result:
        print(f"\n翻转 top-1 不一致率: {result['flip_consistency']['top1_disagreement_rate']}")
    print(f"\n✅ 报告: {args.out}")


if __name__ == "__main__":
    main()
