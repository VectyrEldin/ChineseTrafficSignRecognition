#!/usr/bin/env python3
"""ablation.py —— 单变量短训消融链路。

**默认 dry-run**：只打印全部计划，不写文件、不启动训练、不覆盖任何已部署产物。
加 --execute 才真正执行；产物全部落在独立的 runs/detect/ablate_*/ 目录下
（exist_ok=True，重跑同名组会覆盖该组旧产物）。

为什么需要它：lr0 / mosaic / degrees / cls_pw 等调整均属**概率性收益**，
官方文档明确「用默认值微调已是强基线，仅在有具体理由时调整」。正确做法是
一次只动一个变量、用统一指标判定，再决定是否进入全量重训。

实现方式：train.py 已是单一默认配置（不再接受 --profile / --set），本脚本
import train 复用 build_train_kwargs / check_class_order / is_resampled，
在最终 kwargs 上施加单变量覆盖——基线配置的唯一权威来源仍是 train.py。
旧版 official_loss 组已移除：当前 train.py 的 box/cls/dfl 本来就是官方原值。

判定口径（全部与部署对齐）：
  · small(0,32] 档 recall@IoU0.5 —— 论文 Table 3 分档
  · mAP50 看训练产物 results.csv（eval_by_size 不产出）
  · 纯背景图 FPPI —— 聚合 mAP 不度量这一维
  权重一律用 NMS-free 头评测，与导出一致。

五组标准消融 (每组只改一个变量, baseline 为当前 train.py 默认):
  baseline       无覆盖
  cls_pw_050     cls_pw=0.5 (默认 0.25 起步，检验更强逆频率是否有益)
  lr0_001        lr0=0.001 (当前 0.002)
  mosaic_050     mosaic=0.5 (默认 0.99182；mosaic 增实例但进一步压缩小目标)
  geometry       论文 §4.2 视角增强 degrees=4 shear=1 perspective=0.0005
                 （train.py 默认禁止整图几何变换，本组为刻意偏离验证）

两个可选组（需要额外前置，默认不启用）：
  --tile-size N  追加 tile 切片训练组（ultralytics tile 接受 int，源像素 tile 边长）
  --segments-data PATH
                 追加 polygon_copy_paste 组：数据来自
                 `convert_tt100k.py --emit-segments` 的 9 列 segment 标签，
                 copy_paste=0.3 才会真正执行（bbox-only 标签下不生效）。

用法：
  uv run python ablation.py                          # dry-run, 展开全部计划
  uv run python ablation.py --groups baseline,lr0_001
  uv run python ablation.py --epochs 30 --execute    # 真正跑
  uv run python ablation.py --tile-size 960 --execute
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import TypedDict

import train as train_mod

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
PY = sys.executable
RUNS = SCRIPT_DIR / "runs" / "detect"
DEFAULT_DATA = ROOT / "data" / "tt100k48_yolo" / "tt100k48.yaml"
DEFAULT_BG = ROOT / "data" / "tt100k_2021" / "other"
DEFAULT_ANN = ROOT / "data" / "tt100k_2021" / "annotations_all.json"
DEFAULT_OUT = SCRIPT_DIR / "runs" / "ablation_summary.json"

# 每组只放相对 baseline 的单变量覆盖，作用于 train.build_train_kwargs 的最终 dict
Group = tuple[str, str, dict[str, object]]
GROUPS: list[Group] = [
    ("baseline", "当前 train.py 默认（对照组）", {}),
    ("cls_pw_050", "cls_pw 0.25 → 0.5", {"cls_pw": 0.5}),
    ("lr0_001", "lr0 0.002 → 0.001", {"lr0": 0.001}),
    ("mosaic_050", "mosaic 0.99182 → 0.5", {"mosaic": 0.5}),
    (
        "geometry",
        "论文 §4.2 视角增强（刻意偏离默认的禁止几何变换）",
        {"degrees": 4.0, "shear": 1.0, "perspective": 0.0005},
    ),
]


class EvalMetrics(TypedDict, total=False):
    """eval_by_size.json 中提取的评测指标。"""

    small_recall: float | None
    small_precision: float | None
    fppi: float | None
    map50: float | None
    error: str | None


def py(script: str, *args: object) -> list[str]:
    return [PY, str(SCRIPT_DIR / script), *[str(a) for a in args]]


def build_groups(args: argparse.Namespace) -> list[Group]:
    groups = list(GROUPS)
    if args.tile_size > 0:
        groups.append(
            (
                "tile",
                f"切片训练 tile={args.tile_size}（源像素）",
                {"tile": args.tile_size},
            )
        )
    if args.segments_data is not None:
        groups.append(
            (
                "polygon_copy_paste",
                "9 列 segment 标签 + copy_paste=0.3",
                {"copy_paste": 0.3},
            )
        )
    if args.groups:
        wanted = {g.strip() for g in args.groups.split(",") if g.strip()}
        known = {g[0] for g in groups}
        unknown = wanted - known
        if unknown:
            raise SystemExit(f"[FAIL] 未知消融组 {sorted(unknown)}；可用: {sorted(known)}")
        groups = [g for g in groups if g[0] in wanted]
    return groups


def group_data(args: argparse.Namespace, name: str) -> Path:
    """polygon_copy_paste 组必须用带 segment 标签的独立数据集。"""
    if name == "polygon_copy_paste" and args.segments_data is not None:
        return Path(args.segments_data)
    return Path(args.data)


def build_kwargs(
    args: argparse.Namespace, name: str, overrides: dict[str, object], data: Path
) -> dict:
    """复用 train.py 的唯一默认配置，再施加本组单变量覆盖。"""
    ns = SimpleNamespace(
        data=str(data),
        imgsz=args.imgsz,
        epochs=args.epochs,
        patience=20,
        time=0.0,
        batch=args.batch,
        workers=4,
        cache=False,
        device=args.device,
        run_name=f"ablate_{name}",
        smoke=False,
        resume=False,
    )
    # dry-run 时数据集可能尚未转换，缺失则按非重采样处理
    resampled = train_mod.is_resampled(data) if data.is_file() else False
    kwargs = train_mod.build_train_kwargs(ns, resampled)
    kwargs.update(overrides)
    return kwargs


def run_training(
    args: argparse.Namespace, name: str, overrides: dict[str, object], data: Path
) -> bool:
    """执行一组训练；结束后释放显存供下一组使用。返回 best.pt 是否产出。"""
    import torch
    from ultralytics import YOLO

    train_mod.check_class_order(data)
    kwargs = build_kwargs(args, name, overrides, data)
    if args.device != "cpu" and not torch.cuda.is_available():
        print("    [WARN] CUDA 不可用，回退 CPU（会非常慢）")
        kwargs["device"] = "cpu"
    print(f"    覆盖: {overrides or '无（baseline）'}")
    model = YOLO(train_mod.DEFAULT_WEIGHTS)
    try:
        model.train(**kwargs)
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return (RUNS / f"ablate_{name}" / "weights" / "best.pt").is_file()


def eval_command(args: argparse.Namespace, name: str, data: Path) -> tuple[list[str], Path]:
    run_dir = RUNS / f"ablate_{name}"
    out = run_dir / f"ablation_{name}.json"
    cmd = py(
        "eval_by_size.py",
        "--weights",
        run_dir / "weights" / "best.pt",
        "--data",
        data,
        "--split",
        "test",
        "--imgsz",
        args.imgsz,
        "--limit",
        args.eval_limit,
        "--bg-dir",
        args.bg_dir,
        "--annotation",
        args.annotation,
        "--bg-limit",
        args.bg_limit,
        "--skip-flip",
        "--out",
        out,
    )
    return cmd, out


def read_metrics(out: Path) -> EvalMetrics:
    """从 eval_by_size.json 抽出判定用的三项指标；缺失时返回 None 值。"""
    empty: EvalMetrics = {
        "small_recall": None,
        "small_precision": None,
        "fppi": None,
        "map50": None,
        "error": None,
    }
    if not out.is_file():
        return {**empty, "error": "评测报告未生成"}
    try:
        raw = json.loads(out.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {**empty, "error": f"报告解析失败: {exc}"}
    small = raw.get("size_bins", {}).get("bins", {}).get("small(0,32]", {})
    return {
        "small_recall": small.get("recall_iou50"),
        "small_precision": small.get("precision_iou50"),
        "fppi": raw.get("background", {}).get("fppi"),
        "map50": None,
        "error": None,
    }


def fmt(value: float | None) -> str:
    """格式化浮点指标，None 时显示 n/a。"""
    return "  n/a " if value is None else f"{value:.4f}"


def _print_banner_and_plan(
    args: argparse.Namespace,
    groups: list[Group],
) -> list[dict]:
    """打印消融计划；返回 dry-run 阶段预填充的结果列表。"""
    print("=" * 76)
    print(" TrafficSignRecognition — 单变量消融链路")
    print("=" * 76)
    print(f" 模式     : {'EXECUTE（会训练并写文件）' if args.execute else 'DRY-RUN（只读计划）'}")
    print(
        f" 组数={len(groups)}  imgsz={args.imgsz}  batch={args.batch}  "
        f"epochs={args.epochs}  eval_limit={args.eval_limit}"
    )
    print(" 判定指标 : small(0,32] recall + precision + 背景 FPPI（同一 NMS-free 头）")
    print("=" * 76)

    results: list[dict] = []
    for name, note, overrides in groups:
        data = group_data(args, name)
        kwargs = build_kwargs(args, name, overrides, data)
        eval_cmd, out = eval_command(args, name, data)
        print(f"\n[{name}] {note}")
        print(f"   data : {data}")
        print(
            f"   train: imgsz={kwargs['imgsz']} batch={kwargs['batch']} "
            f"epochs={kwargs['epochs']} run_name={kwargs['name']}"
        )
        print(f"   覆盖 : {overrides or '无'}")
        print("   eval : " + " ".join(f'"{c}"' if " " in str(c) else str(c) for c in eval_cmd))
        results.append(
            {
                "group": name,
                "note": note,
                "overrides": overrides,
                "data": str(data),
                "report": str(out),
                **read_metrics(out),
            }
        )
    return results


def _execute_runs(
    args: argparse.Namespace,
    groups: list[Group],
) -> None:
    """依次执行每组的训练和评测。"""
    print("\n[EXECUTE] 开始执行 ...")
    for name, note, overrides in groups:
        data = group_data(args, name)
        if not data.is_file():
            print(f"[FAIL] {name} 数据集不存在: {data}，跳过该组")
            continue
        eval_cmd, _ = eval_command(args, name, data)

        print(f"\n>>> [{name}] 训练 ...")
        try:
            ok = run_training(args, name, overrides, data)
        except (RuntimeError, OSError, ValueError) as exc:
            # 训练期可预期失败（CUDA OOM / 数据或配置错误）隔离在本组；其余异常直接抛出暴露 bug
            print(f"[FAIL] {name} 训练异常: {exc}，跳过该组评测")
            continue
        if not ok:
            print(f"[FAIL] {name} 未产出 best.pt，跳过该组评测")
            continue

        print(f">>> [{name}] 评测 ...")
        if (rc := subprocess.run(eval_cmd, cwd=str(SCRIPT_DIR), check=False).returncode) != 0:
            print(f"[WARN] {name} 评测失败 (rc={rc})，权重仍保留")


def _collect_and_save_results(
    args: argparse.Namespace,
    groups: list[Group],
    out_path: Path,
) -> list[dict]:
    """重新读取每组评测报告并写入汇总 JSON。"""
    results: list[dict] = []
    for name, note, overrides in groups:
        data = group_data(args, name)
        _, out = eval_command(args, name, data)
        results.append(
            {
                "group": name,
                "note": note,
                "overrides": overrides,
                "data": str(data),
                "report": str(out),
                **read_metrics(out),
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "imgsz": args.imgsz,
                "epochs": args.epochs,
                "metric_note": "small(0,32] recall/precision @ IoU0.5；FPPI @ conf=0.25",
                "groups": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return results


def _print_summary_table(results: list[dict]) -> None:
    """打印结果对比表。"""
    print("\n" + "=" * 76)
    print(f" {'组':<20} {'small R':>9} {'small P':>9} {'FPPI':>9}")
    print("=" * 76)
    for r in results:
        print(
            f" {r['group']:<20} {fmt(r['small_recall']):>9} "
            f"{fmt(r['small_precision']):>9} {fmt(r['fppi']):>9}"
        )
        if r.get("error"):
            print(f"     [WARN] {r['error']}")


def main() -> None:
    ap = argparse.ArgumentParser(description="单变量短训消融（默认 dry-run）")
    ap.add_argument("--execute", action="store_true", help="真正执行；不加则只打印计划")
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA, help="数据集 yaml")
    ap.add_argument("--imgsz", type=int, default=640, help="网络输入边长；默认与当前部署基线一致")
    ap.add_argument("--batch", type=int, default=16, help="批大小")
    ap.add_argument("--epochs", type=int, default=30, help="每组短训轮数")
    ap.add_argument("--device", default="0", help="'0' / 'cpu'")
    ap.add_argument("--groups", default="", help="逗号分隔的组名过滤；默认全部标准组")
    ap.add_argument(
        "--tile-size", type=int, default=0, help="追加 tile 切片训练组（源像素边长，0=不启用）"
    )
    ap.add_argument(
        "--segments-data",
        type=Path,
        default=None,
        help="追加 polygon_copy_paste 组用的 segment 数据集 yaml",
    )
    ap.add_argument("--eval-limit", type=int, default=0, help="分档评测最多图片数，0=全部")
    ap.add_argument("--bg-dir", type=Path, default=DEFAULT_BG, help="纯背景图库")
    ap.add_argument("--bg-limit", type=int, default=300, help="FPPI 评测背景图上限")
    ap.add_argument(
        "--annotation", type=Path, default=DEFAULT_ANN, help="TT100K 原始标注（背景图判空用）"
    )
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    groups = build_groups(args)
    if args.segments_data is not None and not args.segments_data.is_file():
        raise SystemExit(f"[FAIL] --segments-data 不存在: {args.segments_data}")

    # 1. 展示计划（dry-run 和 execute 都需要）
    _print_banner_and_plan(args, groups)

    # 2. dry-run 提前返回
    if not args.execute:
        print("\n[DRY-RUN] 未执行任何步骤。确认后加 --execute 运行。")
        return

    # 3. 执行
    _execute_runs(args, groups)

    # 4. 汇总 & 展示
    results = _collect_and_save_results(args, groups, args.out)
    _print_summary_table(results)
    print(f"\n 汇总报告: {args.out}")
    print(" 判定建议: small R 上升且 FPPI 不升为有效；单项改善需与 baseline 差值")
    print("           大于评测噪声再采纳，并在全量 100 轮上复验。")


if __name__ == "__main__":
    main()
