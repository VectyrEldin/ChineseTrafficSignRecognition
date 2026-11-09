#!/usr/bin/env python3
"""select_best.py —— 把背景误检 FPPI 纳入模型选择（复合择优）。

动机: Ultralytics 的 best.pt 只按 fitness = 0.1*mAP50 + 0.9*mAP50-95 选出，
而本项目 val/test 分桶都是 0 背景图, mAP 完全不度量误检。实测部署痛点是
300 张无标志背景图 9.7% 图像级误检、100 张未收录类图 61% 被高分检出。
因此训练结束后比较候选权重 (默认 best.pt + last.pt), 用复合分数择优：

    score = mAP50 - fppi_weight * FPPI

mAP50 用 Ultralytics 默认 val 口径 (conf=0.001, 不传 conf)，并固定 nms=False
与 export_onnx.py 的 NMS-free 部署头一致; FPPI 用部署阈值 conf=0.25 在纯背景图上
测得（与 Java 端 0.25 一致），两个口径各自与生产/评估惯例保持一致，不混用。

train.py 默认 save_period=20 会额外存周期 checkpoint; 加 --all-checkpoints
可把它们一并纳入择优（候选变多，评估耗时线性增长，默认关闭）。

输出 deploy_candidate.json 并给出 recommended_weights, 供
export_onnx.py --weights 使用；不改写 best.pt、不触碰已部署的 best.onnx。

用法：
  uv run python select_best.py --run-dir runs/detect/tt100k48_yolo26s
  uv run python select_best.py --candidates a/weights/best.pt b/weights/best.pt \
      --imgsz 640 --fppi-weight 1.0 --bg-limit 300
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_DATA = PROJECT_ROOT / "data" / "tt100k48_yolo" / "tt100k48.yaml"
DEFAULT_BG = PROJECT_ROOT / "data" / "tt100k_2021" / "other"
DEFAULT_ANN = PROJECT_ROOT / "data" / "tt100k_2021" / "annotations_all.json"
DEFAULT_RUN_DIR = SCRIPT_DIR / "runs" / "detect" / "tt100k48_yolo26s"

# 同目录脚本：复用 eval_by_size.py 已实现的背景 FPPI 评测，避免口径漂移
sys.path.insert(0, str(SCRIPT_DIR))
from eval_by_size import eval_background


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="背景 FPPI 复合择优（best.pt vs last.pt）")
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR,
                    help="训练产物目录；默认 candidates 为其 weights/ 下的 best/last")
    ap.add_argument("--candidates", nargs="+", type=Path, default=[],
                    help="显式候选权重列表，指定后忽略 --run-dir 的默认候选")
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA, help="数据集 yaml")
    ap.add_argument("--bg-dir", action="append", default=[],
                    help="纯背景图目录，可重复指定；缺省用 data/tt100k_2021/other")
    ap.add_argument("--annotation", type=Path, default=DEFAULT_ANN,
                    help="annotations_all.json，用于排除已登记的有标注图")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25,
                    help="FPPI 所用的部署置信度阈值（Java 端默认 0.25）")
    ap.add_argument("--bg-limit", type=int, default=300)
    ap.add_argument("--fppi-weight", type=float, default=1.0,
                    help="score = mAP50 - fppi_weight * FPPI；越大越看重抑制误检")
    ap.add_argument("--all-checkpoints", action="store_true",
                    help="把 save_period 生成的 epoch*.pt 也纳入择优（默认只比 best/last）")
    ap.add_argument("--out", type=Path, default=None,
                    help="输出 JSON，默认 <run-dir>/deploy_candidate.json")
    return ap.parse_args()


def collect_candidates(args: argparse.Namespace) -> list[Path]:
    if args.candidates:
        paths = [p.resolve() for p in args.candidates]
    else:
        weights = args.run_dir / "weights"
        paths = [weights / "best.pt", weights / "last.pt"]
        if args.all_checkpoints:
            def epoch_num(p: Path) -> int:
                digits = "".join(c for c in p.stem if c.isdigit())
                return int(digits) if digits else -1
            paths.extend(sorted(weights.glob("epoch*.pt"), key=epoch_num))
    # 去重且保持顺序（显式 --candidates 也可能重复）
    seen: set[str] = set()
    unique: list[Path] = []
    for p in paths:
        key = str(p)
        if key not in seen:
            seen.add(key)
            unique.append(p)
    found = [p for p in unique if p.is_file()]
    for m in (str(p) for p in unique if not p.is_file()):
        print(f"[WARN] 候选权重不存在，跳过: {m}")
    return found


def evaluate(weights: Path, args: argparse.Namespace, bg_dirs: list[Path]) -> dict:
    """单候选：val 口径 mAP + 部署阈值口径 FPPI，两者不混用。"""
    import torch
    from ultralytics import YOLO

    model = YOLO(str(weights))
    # 不传 conf：沿用 Ultralytics val 的 0.001 口径，与训练内 val 可比
    # nms=False：与 train.py 训练内 val / export_onnx.py 导出的 e2e 头同口径
    metrics = model.val(
        data=str(args.data),
        imgsz=args.imgsz,
        nms=False,
        project=str(SCRIPT_DIR / "runs" / "detect"),
        name="fppi_select",
        exist_ok=True,
        plots=False,
        verbose=False,
    )
    map50 = round(float(metrics.box.map50), 5)
    map5095 = round(float(metrics.box.map), 5)

    bg = eval_background(
        model, bg_dirs, args.annotation if args.annotation.is_file() else None,
        args.imgsz, args.conf, args.bg_limit,
    )
    fppi = bg.get("fppi")
    if fppi is None:
        print(f"[WARN] {weights.name}: 未取得 FPPI（{bg.get('note', '无背景图')}），按 0 计")
        fppi = 0.0

    score = round(map50 - args.fppi_weight * float(fppi), 5)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "weights": str(weights),
        "map50": map50,
        "map50_95": map5095,
        "fppi": fppi,
        "background_images": bg.get("images", 0),
        "image_fp_rate": bg.get("image_fp_rate"),
        "score": score,
    }


def main() -> None:
    args = parse_args()
    candidates = collect_candidates(args)
    if not candidates:
        raise SystemExit("[FAIL] 没有可评估的候选权重")

    bg_dirs = [Path(d) for d in args.bg_dir] if args.bg_dir else [DEFAULT_BG]
    bg_dirs = [d for d in bg_dirs if d.is_dir()]
    if not bg_dirs:
        print("[WARN] 背景图目录不可用，FPPI 将按 0 计（退化为纯 mAP50 择优）")

    print(f"[INFO] 候选 {len(candidates)} 个  imgsz={args.imgsz}  "
          f"conf(FPPI)={args.conf}  fppi_weight={args.fppi_weight}")
    rows = []
    for w in candidates:
        print(f"\n[EVAL] {w}")
        rows.append(evaluate(w, args, bg_dirs))

    rows.sort(key=lambda r: -r["score"])
    result = {
        "score_formula": "mAP50 - fppi_weight * FPPI",
        "fppi_weight": args.fppi_weight,
        "map50_conf": "Ultralytics val default (0.001)",
        "fppi_conf": args.conf,
        "imgsz": args.imgsz,
        "nms_head": False,
        "all_checkpoints": args.all_checkpoints,
        "candidates": rows,
        "recommended_weights": rows[0]["weights"],
        "note": "不改写 best.pt；导出时用 export_onnx.py --weights <recommended_weights>",
    }
    out = args.out or (args.run_dir / "deploy_candidate.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print(" 候选权重对比 (score 越高越优)")
    print("=" * 72)
    for r in rows:
        print(f"  mAP50={r['map50']:.5f}  FPPI={r['fppi']:.4f}  "
              f"score={r['score']:.5f}  {r['weights']}")
    print(f"\n 推荐部署权重: {rows[0]['weights']}")
    print(f" 报告: {out}")


if __name__ == "__main__":
    main()
