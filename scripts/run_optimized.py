#!/usr/bin/env python3
"""run_optimized.py —— 把论文对照优化项串成一条可复现链路。

默认 **dry-run**：只打印完整计划，不写任何文件、不启动训练。
加 --execute 才真正执行。这样既能一次性落实全部建议，又不会在未经确认时
覆盖已部署的 best.onnx / names。

链路 (默认 960 效果对照；单变量概率性项走 ablation.py):
  1) convert_tt100k.py  --split-mode group --class-balanced
       · 按序列分组划分，消除近重复帧泄漏（现为转换脚本默认）
       · 逆频率类感知重采样，不依赖 marks 模板
       · 背景图比例 0.30（部署痛点是背景误检，高于转换默认 0.10）
       · 可选 --segments: 生成 9 列 segment 行，使 copy_paste 真正生效
       · yaml 省略 path, 任意工作目录可运行
  2) eval_by_size.py    在当前部署的 48 类模型上建立尺寸分档 / FPPI / 翻转基线 (只读, 640)
       · 评测统一用 NMS-free 部署头（与导出一致）
  3) train.py           以当前唯一默认配置重训（--imgsz / --batch / --epochs 可调）
       · cls_pw 检测到重采样清单时自动置 0, 避免对稀有类双重补偿
  4) select_best.py     背景 FPPI 择优，产出 deploy_candidate.json
       · train.py 不再自动择优，本链路显式执行
  5) 同步 Java names (先备份 .bak)
  6) export_onnx.py     按 deploy_candidate.json 的推荐权重导出 (非无条件 best.pt)
  7) verify_parity.py   输入张量逐元素对拍；权重/ONNX 与第 6 步同源
  8) Gradle 编译 + 单测
  9) eval_by_size.py    在新权重上复测 (含背景 FPPI), 与基线对比

用法（查看计划）：
  uv run python run_optimized.py                          # dry-run, 展开全部命令
  uv run python run_optimized.py --imgsz 640 --batch 16   # 对照 640 口径的代价
真正执行 (960 / 100 轮约 5-6 小时重训):
  uv run python run_optimized.py --execute
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
PY = sys.executable

ANN = ROOT / "data" / "tt100k_2021" / "annotations_all.json"
SRC_TRAIN = ROOT / "data" / "tt100k_2021" / "train"
SRC_TEST = ROOT / "data" / "tt100k_2021" / "test"
BG_DIR = ROOT / "data" / "tt100k_2021" / "other"
# 基线 = 当前部署的 48 类模型（last.pt，与 model_manifest.json 的 source_weights 一致）
BASELINE_WEIGHTS = SCRIPT_DIR / "runs" / "detect" / "tt100k48_yolo26s" / "weights" / "last.pt"
BASELINE_DATA = ROOT / "data" / "tt100k48_yolo" / "tt100k48.yaml"
JAVA_NAMES = ROOT / "app" / "src" / "main" / "resources" / "config" / "tt100k48.names"
RUN_NAME = "tt100k48_opt"
# 这三个 callable 只读文件、仅构造命令，dry-run 时可安全展开为真实命令
SAFE_DRY_RUN = frozenset({"export_step", "parity_step", "eval_step"})
StepCmd = Callable[[], list[str] | int | Path] | list[str]


def py(script: str, *args: object) -> list[str]:
    return [PY, str(SCRIPT_DIR / script), *[str(a) for a in args]]


def first_val_image(out: Path) -> Path:
    images = sorted((out / "images" / "val").glob("*.jpg"))
    return images[0] if images else out / "images" / "val" / "<first>.jpg"


def build_steps(
    out: Path, imgsz: int, epochs: int, batch: int, force: bool, segments: bool = False
) -> list[tuple]:
    """返回 (标题, 命令或可调用对象, cwd, 说明) 列表。"""
    new_weights = SCRIPT_DIR / "runs" / "detect" / RUN_NAME / "weights" / "best.pt"
    names = out / "tt100k48.names"
    yaml_path = out / "tt100k48.yaml"

    conv = py(
        "convert_tt100k.py",
        "--annotation",
        ANN,
        "--image-dir",
        SRC_TRAIN,
        "--test-image-dir",
        SRC_TEST,
        "--output",
        out,
        "--link",
        "--split-mode",
        "group",
        "--class-balanced",
        "--background-ratio",
        0.30,
    )
    if segments:
        conv.append("--emit-segments")
    if force:
        conv.append("--overwrite")

    def sync_java_names() -> int:
        if JAVA_NAMES.is_file():
            shutil.copy2(JAVA_NAMES, JAVA_NAMES.with_suffix(".names.bak"))
        shutil.copy2(names, JAVA_NAMES)
        print(f"       已同步 {JAVA_NAMES}（旧文件备份为 .names.bak）")
        return 0

    def resolve_weights() -> Path:
        """读取第 ④ 步 select_best.py 产出的 FPPI 复合择优报告；缺失时退回 best.pt。"""
        report = SCRIPT_DIR / "runs" / "detect" / RUN_NAME / "deploy_candidate.json"
        if report.is_file():
            try:
                chosen = json.loads(report.read_text(encoding="utf-8")).get("recommended_weights")
                if chosen and Path(chosen).is_file():
                    return Path(chosen)
            except (OSError, ValueError) as e:
                print(f"       [WARN] 解析 {report.name} 失败，退回 best.pt: {e}")
        return new_weights

    def export_step() -> list[str]:
        """按复合择优结果导出，而不是无条件使用 best.pt。"""
        return py(
            "export_onnx.py", "--weights", resolve_weights(), "--names", names, "--imgsz", imgsz
        )

    def parity_step() -> list[str]:
        w = resolve_weights()
        return py(
            "verify_parity.py",
            "@FIRST_VAL@",
            "--weights",
            w,
            "--onnx",
            w.with_suffix(".onnx"),
            "--names",
            names,
            "--size",
            imgsz,
        )

    def eval_step() -> list[str]:
        return py(
            "eval_by_size.py",
            "--weights",
            resolve_weights(),
            "--data",
            yaml_path,
            "--split",
            "test",
            "--imgsz",
            imgsz,
            "--bg-dir",
            BG_DIR,
            "--annotation",
            ANN,
            "--out",
            SCRIPT_DIR / "eval_optimized.json",
        )

    gradle = ROOT / ("gradlew.bat" if os.name == "nt" else "gradlew")

    return [
        (
            "① 重建数据集 (group 划分 + 类感知重采样 + yaml 去 path)",
            conv,
            SCRIPT_DIR,
            "输出到独立目录，不覆盖 data/tt100k48_yolo; 若目录已存在请先删或加 --force",
        ),
        (
            "② 当前部署模型基线评测（尺寸分档 / FPPI / 翻转一致性）",
            py(
                "eval_by_size.py",
                "--weights",
                BASELINE_WEIGHTS,
                "--data",
                BASELINE_DATA,
                "--split",
                "test",
                "--imgsz",
                640,
                "--bg-dir",
                BG_DIR,
                "--annotation",
                ANN,
                "--out",
                SCRIPT_DIR / "eval_baseline.json",
            ),
            SCRIPT_DIR,
            "当前部署的 48 类模型为 640 口径；与重训结果逐项对比",
        ),
        (
            "③ 重训（train.py 当前唯一默认配置）",
            py(
                "train.py",
                "--data",
                yaml_path,
                "--run-name",
                RUN_NAME,
                "--imgsz",
                imgsz,
                "--batch",
                batch,
                "--epochs",
                epochs,
            ),
            SCRIPT_DIR,
            "cls_pw 在本链路的重采样清单下自动置 0; 训练后由第 ④ 步显式 FPPI 择优",
        ),
        (
            "④ 背景 FPPI 择优（select_best.py）",
            py(
                "select_best.py",
                "--run-dir",
                SCRIPT_DIR / "runs" / "detect" / RUN_NAME,
                "--data",
                yaml_path,
                "--imgsz",
                imgsz,
            ),
            SCRIPT_DIR,
            "产出 deploy_candidate.json; 后续导出/对拍/复测均按其推荐权重",
        ),
        (
            "⑤ 同步 Java names (先备份)",
            sync_java_names,
            SCRIPT_DIR,
            "48 行 names 必须与新模型一致，否则 class id 45-47 会被静默丢弃",
        ),
        (
            "⑥ 导出 ONNX + 输出契约（按 FPPI 复合择优结果选权重）",
            export_step,
            SCRIPT_DIR,
            (
                "deploy_candidate.json 推荐谁就导出谁，不再无条件用 best.pt;"
                "manifest 写入 output_layout / coordinate_space / 预处理契约"
            ),
        ),
        (
            "⑦ 端到端 parity (含输入张量逐元素对拍)",
            parity_step,
            SCRIPT_DIR,
            "权重/ONNX 路径与第 ⑥ 步同源; INTER_LINEAR 对齐后 max_abs_diff 应为 0",
        ),
        (
            "⑧ Gradle 编译 + 单元测试",
            [str(gradle), ":app:compileJava", ":app:test", "-q"],
            ROOT,
            "验证契约读取、切片推理与类别数动态化未破坏现有测试",
        ),
        (
            "⑨ 新模型复测并落报告 （含背景 FPPI, 与基线同口径）",
            eval_step,
            SCRIPT_DIR,
            "与 eval_baseline.json 对比 small 档 recall 与 FPPI; 权重与第 ⑥ 步同源",
        ),
    ]


def _print_header(out: Path, args: argparse.Namespace) -> None:
    """打印运行计划头部信息。"""
    print("=" * 72)
    print(" TrafficSignRecognition — 论文对照优化链路")
    print("=" * 72)
    print(
        f" 模式     : {'EXECUTE (会写入文件并启动重训)' if args.execute else 'DRY-RUN (只读计划)'}"
    )
    print(f" 输出目录 : {out}")
    print(
        f" imgsz={args.imgsz}  batch={args.batch}  epochs={args.epochs}  "
        f"segment 标签={args.segments}  训练目录={RUN_NAME}"
    )
    print()
    print(
        " ⚠️  --execute 会重训约 5-6 小时 (960 输入 / 100 轮), 覆盖已部署的 best.onnx 与 Java names。"
    )
    print("     建议先单独执行第 ② 步建立基线，确认收益预期后再决定是否重训。")
    print("=" * 72)


def _print_step(title: str, cmd: StepCmd, cwd: Path, note: str) -> None:
    """打印单个步骤的命令或动作说明（dry-run 与 execute 前的共用展示逻辑）。"""
    print(f"\n{title}")
    print(f"   cwd : {cwd}")
    if callable(cmd) and getattr(cmd, "__name__", "") not in SAFE_DRY_RUN:
        # 有写入副作用的 callable 在 dry-run 下绝不调用
        print(f"   act : {cmd.__name__}()  （仅在 --execute 时执行）")
    else:
        # 只读、仅构造命令的 callable 在 dry-run 下可展开，便于审计导出权重
        resolved = cmd() if callable(cmd) else cmd
        if isinstance(resolved, (list, tuple)):
            print("   cmd : " + " ".join(f'"{c}"' if " " in str(c) else str(c) for c in resolved))
        else:
            # 非列表/元组（如 int 等）直接打印，避免非可迭代对象触发迭代报错
            print(f"   cmd : {resolved}")
    print(f"   note: {note}")


def _execute_steps(steps: list[tuple[str, StepCmd, Path, str]], out: Path) -> None:
    """校验前置依赖并依次执行各步骤。"""
    missing = [
        p for p in (ANN, SRC_TRAIN, SRC_TEST, BASELINE_WEIGHTS, BASELINE_DATA) if not p.exists()
    ]
    if missing:
        raise SystemExit("[FAIL] 缺少前置文件:\n  " + "\n  ".join(map(str, missing)))

    print("\n[EXECUTE] 开始执行 ...")
    for title, cmd, cwd, _ in steps:
        print(f"\n>>> {title}")
        resolved = cmd() if callable(cmd) else cmd
        if isinstance(resolved, list):
            argv = [
                str(first_val_image(out)) if str(c) == "@FIRST_VAL@" else str(c) for c in resolved
            ]
            rc = subprocess.run(argv, cwd=str(cwd), check=False).returncode
        else:
            rc = resolved
        if rc != 0:
            raise SystemExit(
                f"[FAIL] 步骤失败 (rc={rc}): {title}\n"
                f"       已完成的步骤不会自动回滚；请按 README 的回滚说明处理。"
            )
    print("\n✅ 全部步骤完成。请对比 eval_baseline.json 与 eval_optimized.json。")


def main() -> None:
    ap = argparse.ArgumentParser(description="论文对照优化项一键链路（默认 dry-run）")
    ap.add_argument("--execute", action="store_true", help="真正执行；不加则只打印计划")
    ap.add_argument(
        "--imgsz",
        type=int,
        default=960,
        help="训练/导出输入边长；默认 960 走效果对照，"
        "改回 640 可复现当前部署口径（代价是小目标分辨率下降）",
    )
    ap.add_argument(
        "--batch",
        type=int,
        default=4,
        help="批大小；960 输入在 8GB 显存的安全值为 4，640 输入可改 16（train.py 默认值）",
    )
    ap.add_argument(
        "--epochs", type=int, default=100, help="训练轮数；默认与 train.py 的 100 轮一致"
    )
    ap.add_argument("--output", type=Path, default=ROOT / "data" / "tt100k48_yolo_opt")
    ap.add_argument(
        "--force", action="store_true", help="目标数据集目录已存在时清空 images/ 与 labels/ 重建"
    )
    ap.add_argument(
        "--segments",
        action="store_true",
        help="生成 9 列 segment 标签，使 copy_paste 增强真正生效（需消融验证）",
    )
    args = ap.parse_args()

    out = args.output.resolve()
    steps = build_steps(out, args.imgsz, args.epochs, args.batch, args.force, args.segments)

    _print_header(out, args)

    for step in steps:
        _print_step(*step)

    if not args.execute:
        print("\n[DRY-RUN] 未执行任何步骤。确认后加 --execute 运行完整链路。")
        print("           仅运行只读基线评测：")
        print(
            "           uv run python eval_by_size.py --weights "
            "runs/detect/tt100k48_yolo26s/weights/last.pt "
            "--data ../data/tt100k48_yolo/tt100k48.yaml --split test"
        )
        return

    _execute_steps(steps, out)


if __name__ == "__main__":
    main()
