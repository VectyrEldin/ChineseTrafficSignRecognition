#!/usr/bin/env python3
"""
verify_parity.py —— 生成 Python 端与 ONNX 端的检测基准，
用于和 Java 端结果逐条比对

输出 JSON: scripts/parity_baseline.json
  {
    "image": "10098.jpg",
    "pt":   [ {"cls": 0, "name": "i2", "conf": 0.93, "box": [x1,y1,x2,y2]}, ... ],
    "onnx": [ ... ]
  }

用法（默认跟随 FPPI 择优结果 deploy_candidate.json 的推荐权重，
无择优结果时回退 best.pt; ONNX 默认取权重同目录同名 .onnx):
  uv run python verify_parity.py ../data/tt100k48_yolo/images/val/10098.jpg

复现旧 45 类模型时显式指定权重与 names:
  uv run python verify_parity.py <图片> \
      --weights runs/detect/tt100k45_yolo26s/weights/best.pt \
      --onnx runs/detect/tt100k45_yolo26s/weights/best.onnx \
      --names ../data/tt100k45_yolo/tt100k45.names
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics.engine.results import Results

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RUN_DIR = SCRIPT_DIR / "runs" / "detect" / "tt100k48_yolo26s"


def default_weights() -> str:
    """与导出链路同源：优先 select_best.py 写出的推荐权重（可能是 last.pt）。"""
    candidate = RUN_DIR / "deploy_candidate.json"
    if candidate.is_file():
        try:
            rec = json.loads(candidate.read_text(encoding="utf-8")).get("recommended_weights")
        except json.JSONDecodeError, OSError:
            rec = None
        if rec:
            p = Path(rec)
            if not p.is_absolute():
                p = SCRIPT_DIR / p
            if p.is_file():
                return str(p)
    return str(RUN_DIR / "weights" / "best.pt")


DEFAULT_WEIGHTS = default_weights()
DEFAULT_ONNX = str(Path(DEFAULT_WEIGHTS).with_suffix(".onnx"))
DEFAULT_NAMES = str(PROJECT_ROOT / "data" / "tt100k48_yolo" / "tt100k48.names")
BASELINE_PATH = SCRIPT_DIR / "parity_baseline.json"


def letterbox(
    image: np.ndarray, size: int = 640, pad: int = 114
) -> tuple[np.ndarray, float, int, int]:
    """与 Java 端 + Ultralytics LetterBox 一致的 letterbox: 等比缩放 + 灰边填充。

    INTER_LINEAR 与 Ultralytics 训练(warpAffine)、val(dataset.py LetterBox)、
    predict(LetterBox) 的默认插值一致；此前用 INTER_AREA 属部署侧偏移。
    round() 为 round-to-even, 对应 Java 的 Math.rint() (不是 Math.round)。
    """
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    new_width, new_height = round(width * scale), round(height * scale)
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    top = (size - new_height) // 2
    left = (size - new_width) // 2
    canvas = cv2.copyMakeBorder(
        resized,
        top,
        size - new_height - top,
        left,
        size - new_width - left,
        cv2.BORDER_CONSTANT,
        value=(pad, pad, pad),
    )
    return canvas, scale, left, top


def restore_box(
    box: list[float] | np.ndarray,
    scale: float,
    pad_x: int,
    pad_y: int,
    original_w: int,
    original_h: int,
) -> list[float]:
    """把 letterbox 空间的框还原到原图空间（与 Java 端后处理器一致）。"""
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    safe_scale = scale or 1.0
    x1 = max(0.0, min(original_w, (x1 - pad_x) / safe_scale))
    y1 = max(0.0, min(original_h, (y1 - pad_y) / safe_scale))
    x2 = max(0.0, min(original_w, (x2 - pad_x) / safe_scale))
    y2 = max(0.0, min(original_h, (y2 - pad_y) / safe_scale))
    return [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]


def make_detection(class_id: int, names: list[str], score: float, box: list[float]) -> dict:
    """构造一条标准检测结果。"""
    name = names[class_id] if 0 <= class_id < len(names) else "?"
    return {
        "cls": class_id,
        "name": name,
        "conf": round(float(score), 4),
        "box": box,
    }


def parse_flat_output(
    values: np.ndarray,
    names: list[str],
    conf: float,
    scale: float,
    pad_x: int,
    pad_y: int,
    original_w: int,
    original_h: int,
) -> list[dict]:
    """形态 A：端到端输出 [1, N, 6]，每行 [x1, y1, x2, y2, score, class_id]。"""
    detections = []
    for row in values[0]:
        x1, y1, x2, y2, score, raw_id = (float(v) for v in row[:6])
        if score < conf:
            continue
        box = restore_box([x1, y1, x2, y2], scale, pad_x, pad_y, original_w, original_h)
        detections.append(make_detection(round(raw_id), names, score, box))
    return detections


def collect_head_candidates(
    values: np.ndarray, conf: float
) -> tuple[list[list[float]], list[float], list[int]]:
    """形态 B：从 [1, C, A] 原始头收集高于阈值的候选框。"""
    matrix = values[0].T
    boxes: list[list[float]] = []
    scores: list[float] = []
    class_ids: list[int] = []
    for row in matrix:
        class_id = int(np.argmax(row[4:]))
        score = float(row[4 + class_id])
        if score < conf:
            continue
        center_x, center_y, box_w, box_h = (float(v) for v in row[:4])
        boxes.append(
            [
                center_x - box_w / 2,
                center_y - box_h / 2,
                center_x + box_w / 2,
                center_y + box_h / 2,
            ]
        )
        scores.append(score)
        class_ids.append(class_id)
    return boxes, scores, class_ids


def parse_head_output(
    values: np.ndarray,
    names: list[str],
    conf: float,
    scale: float,
    pad_x: int,
    pad_y: int,
    original_w: int,
    original_h: int,
) -> list[dict]:
    """形态 B：转置后做 NMS，再还原到原图空间。"""
    boxes, scores, class_ids = collect_head_candidates(values, conf)
    if not boxes:
        return []
    keep = cv2.dnn.NMSBoxes(boxes, scores, conf, 0.60)
    detections = []
    for index in np.array(keep).flatten():
        position = int(index)
        box = restore_box(boxes[position], scale, pad_x, pad_y, original_w, original_h)
        detections.append(make_detection(int(class_ids[position]), names, scores[position], box))
    return detections


def run_input_parity(image: np.ndarray, size: int) -> dict:
    """输入张量级对拍: Java/ONNX 端 letterbox 必须等价于 Ultralytics LetterBox。

    此前只比对最终检测框，两套不同预处理也可能碰巧给出接近的结果；
    逐元素比较输入张量才能真正证明端到端预处理一致。
    """
    from ultralytics.data.augment import LetterBox

    canvas, scale, pad_x, pad_y = letterbox(image, size)
    java_tensor = (
        np.ascontiguousarray(
            cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None],
            dtype=np.float32,
        )
        / 255.0
    )

    lb = LetterBox(
        new_shape=(size, size),
        auto=False,
        stride=32,
        scaleup=False,
        interpolation=cv2.INTER_LINEAR,
    )
    padded = lb(image=image)
    if not isinstance(padded, np.ndarray):
        raise TypeError(f"LetterBox 返回 {type(padded).__name__}，预期 np.ndarray")
    ref_tensor = (
        np.ascontiguousarray(
            cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None],
            dtype=np.float32,
        )
        / 255.0
    )

    result = {
        "imgsz": size,
        "scale": scale,
        "pad_x": pad_x,
        "pad_y": pad_y,
        "shape_java": list(java_tensor.shape),
        "shape_ultralytics": list(ref_tensor.shape),
    }
    if java_tensor.shape != ref_tensor.shape:
        result.update(
            ok=False, max_abs_diff=None, note="shape 不一致：请核对 letterbox 的 round / pad 计算"
        )
        return result

    diff = np.abs(java_tensor - ref_tensor)
    max_diff = float(diff.max())

    # ── 关键修改：用 epsilon 容差替代浮点相等比较 ──
    EPS = 1e-6
    is_close = max_diff < EPS

    result.update(
        ok=is_close,
        max_abs_diff=max_diff,
        note=(
            "完全一致 (INTER_LINEAR + 114 填充 + round-to-even)"
            if is_close
            else "存在逐像素差异: 检查插值、round 方式与填充灰度"
        ),
    )
    return result


def run_onnx(
    onnx_path: Path,
    image: np.ndarray,
    names: list[str],
    conf: float = 0.25,
    size: int = 640,
) -> list[dict]:
    """用 onnxruntime 复现 Java 端流程: letterbox → 推理 → 解析。"""
    import onnxruntime

    canvas, scale, pad_x, pad_y = letterbox(image, size)
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0

    session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    raw = session.run(None, {session.get_inputs()[0].name: tensor})[0]
    values = np.asarray(raw)
    print(f"[ONNX] 输出 shape = {values.shape}")

    original_h, original_w = image.shape[:2]
    common = (names, conf, scale, pad_x, pad_y, original_w, original_h)
    shape = values.shape
    if len(shape) == 3 and shape[-1] == 6:
        detections = parse_flat_output(values, *common)
    elif len(shape) == 3 and shape[1] in (
        4 + len(names),
        5 + len(names),
        6 + len(names),
    ):
        detections = parse_head_output(values, *common)
    else:
        print(f"⚠️ 未知输出布局 {shape}，请人工解析")
        detections = []
    detections.sort(key=lambda item: -item["conf"])
    return detections


def extract_pt_detections(result: Results, names: list[str]) -> list[dict]:
    """把 Ultralytics Results 的框转换成标准检测结果。"""
    boxes = result.boxes
    if boxes is None:
        return []
    rows = np.asarray(boxes.cpu().numpy().data).tolist()
    detections = []
    for row in rows:
        x1, y1, x2, y2, score, raw_id = (float(v) for v in row[:6])
        box = [round(v, 1) for v in (x1, y1, x2, y2)]
        detections.append(make_detection(int(raw_id), names, score, box))
    return detections


def run_pt_baseline(
    weights: str, image_path: Path, names: list[str], conf: float, size: int = 640
) -> list[dict]:
    """用 PyTorch 权重生成基准；推理失败时打印告警并返回空列表。"""
    try:
        from ultralytics import YOLO

        model = YOLO(weights)
        predictions = model.predict(
            source=str(image_path),
            conf=conf,
            imgsz=size,
            nms=False,  # ONNX 以 nms=False 导出，PyTorch 基准必须用同一 e2e 头
            verbose=False,
            stream=False,
        )
        first = next(iter(predictions), None)
        if not isinstance(first, Results):
            raise TypeError("PyTorch 推理未返回 Results 对象")
        return extract_pt_detections(first, names)
    except (OSError, RuntimeError, ValueError, TypeError, ImportError, IndexError) as exc:
        print(f"[WARN] .pt 推理失败: {exc}")
        return []


def print_report(pytorch_detections: list[dict], onnx_detections: list[dict]) -> None:
    """逐条打印两侧基准，便于人工核对。"""
    print("\n" + "=" * 64)
    print(f"PyTorch(.pt) 结果 — {len(pytorch_detections)} 个框")
    for item in pytorch_detections:
        print(
            f"  cls={item['cls']:>2} {item['name']:<8} conf={item['conf']:.4f}  box={item['box']}"
        )
    print(f"\nONNX 结果 — {len(onnx_detections)} 个框")
    for item in onnx_detections:
        print(
            f"  cls={item['cls']:>2} {item['name']:<8} conf={item['conf']:.4f}  box={item['box']}"
        )
    print("=" * 64)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="生成 PyTorch 与 ONNX 检测基准")
    ap.add_argument("image", help="验证集图片路径")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--onnx", default=DEFAULT_ONNX)
    ap.add_argument("--names", default=DEFAULT_NAMES)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument(
        "--size",
        type=int,
        default=640,
        help="网络输入边长，必须与导出 imgsz / AppConfig.INPUT_SIZE 一致",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    image = cv2.imread(str(image_path))
    if image is None:
        sys.exit(f"读图失败: {image_path}")

    names = [
        line.strip()
        for line in Path(args.names).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(
        f"[INFO] 图片: {image_path.name}  "
        f"尺寸: {image.shape[1]}x{image.shape[0]}  类别数: {len(names)}"
    )
    print(f"[INFO] 权重: {args.weights}")
    print(f"[INFO] ONNX: {args.onnx}")

    input_parity = run_input_parity(image, args.size)
    print("\n[INPUT-PARITY] Java/ONNX letterbox vs Ultralytics LetterBox")
    print(
        f"  ok={input_parity['ok']}  max_abs_diff={input_parity['max_abs_diff']}"
        f"  {input_parity['note']}"
    )
    if not input_parity["ok"]:
        print("  [WARN] 输入张量不一致，下方 pt/onnx 框对比不能作为端到端对齐证据")

    pytorch_detections = run_pt_baseline(args.weights, image_path, names, args.conf, args.size)
    onnx_detections = run_onnx(Path(args.onnx), image, names, args.conf, args.size)
    print_report(pytorch_detections, onnx_detections)

    payload = {
        "image": image_path.name,
        "conf": args.conf,
        "imgsz": args.size,
        "input_parity": input_parity,
        "pt": pytorch_detections,
        "onnx": onnx_detections,
    }
    BASELINE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ 基准已保存: {BASELINE_PATH}")
    print("   用它和 Java 端结果表格逐条比对")


if __name__ == "__main__":
    main()
