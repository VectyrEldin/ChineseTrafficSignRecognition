#!/usr/bin/env python3
"""
train.py —— YOLO26s 微调 TT100K 48 类交通标志（时间效率优先，唯一默认配置）

默认参数（依据见各参数处注释）：
  imgsz=640, batch=16, epochs=100, patience=20, workers=4, cache=False,
  MuSGD, lr0=0.002, lrf=0.01, cos_lr=True,
  fliplr=0, translate=0.1, scale=[0.8,1.5], close_mosaic=10,
  mixup=0, copy_paste=0, cls_pw=0.25 (重采样清单自动置 0), hsv_v=0.3,
  其余与官方 yolo26s.pt train_args 完全一致: momentum=0.94751,
  weight_decay=0.00027, warmup=(0.98745, 0.54064, 0.05684),
  box=9.83241, cls=0.64896, dfl=0.95824, mosaic=0.99182,
  hsv_h=0.01315, hsv_s=0.35348; degrees/shear/perspective=0 (禁止整图几何变换)
  multi_scale=0, nms=False, save_json=False, plots=False, save_period=-1

实测参考: 45 类基线 640/100 轮约 2.7h, 第 100 轮 mAP50 已达 200 轮终值的 99.4%。
patience=20 允许平台期提前早停；--time 可设硬限时(到点停止, best.pt 照常保存)。
训练后不自动跑 FPPI 择优（固定开销）；需要时手动执行 select_best.py。

用法：
  uv run python train.py --smoke          # 1 epoch 冒烟测试（先跑这个！）
  uv run python train.py                  # 默认：约 2.5-3.0h @640
  uv run python train.py --epochs 70      # 更激进：约 1.9-2.2h
  uv run python train.py --time 2.0       # 硬限时 2 小时
  uv run python train.py --batch 8        # 显存被其他程序占用时退回 8

前置条件：
  · 先运行 convert_tt100k.py --output ../data/tt100k48_yolo 生成数据集
  · Windows 下若 venv 导入 numpy 报错，清空全局 PYTHONPATH 或统一用 uv run

产物目录与 export_onnx.py / verify_parity.py 默认路径一致:tt100k48_yolo26s/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# ==================== 默认配置 ====================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_DATA = str(PROJECT_ROOT / "data" / "tt100k48_yolo" / "tt100k48.yaml")
DEFAULT_WEIGHTS = str(SCRIPT_DIR / "yolo26s.pt")  # 缺失时会从 ultralytics 下载
NAME = "tt100k48_yolo26s"


def is_resampled(data_yaml: Path) -> bool:
    """yaml 的 train 指向 .txt 清单 (train_balanced.txt)时数据集已做逆频率
    重采样；再叠 cls_pw 会对稀有类二次补偿，此时 cls_pw 自动置 0。"""
    raw = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    # 显式确保加载结果是字典，避免 YAML 根节点为 list 等非字典类型时
    # 触发 AttributeError，同时消除类型检查器对 .get() 的警告
    if not isinstance(raw, dict):
        return False
    cfg: dict = raw
    train_val = cfg.get("train", "")
    return isinstance(train_val, str) and train_val.lower().endswith(".txt")


# ==================== 前置校验 ====================
def check_class_order(data_yaml: Path) -> list[str]:
    """确认 yaml 的 names 顺序与 .names 文件逐行一致（类别顺序三处一致的最后一道岗）。"""
    if not data_yaml.is_file():
        sys.exit(
            f"[FAIL] 数据集配置不存在: {data_yaml}\n       请先执行 convert_tt100k.py"
        )

    raw = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    # 显式确保加载结果是字典，消除类型检查器对 .get() 的警告
    if not isinstance(raw, dict):
        sys.exit(
            f"[FAIL] 数据集配置格式错误（期望字典结构，实际为 {type(raw).__name__}）: {data_yaml}"
        )
    cfg: dict = raw

    names = list(cfg.get("names", []))
    nc = int(cfg.get("nc", len(names)))

    names_file = data_yaml.with_suffix(".names")
    if not names_file.is_file():
        raise FileNotFoundError(f"缺少类别文件: {names_file}")
    lines = [
        l.strip()
        for l in names_file.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    expected = len(lines)

    if len(names) != expected:
        raise ValueError(f"yaml names 有 {len(names)} 项，期望 {expected}")
    if nc != expected:
        raise ValueError(f"yaml nc={nc}，期望 {expected}")
    if len(set(names)) != expected:
        raise ValueError("yaml names 存在重复项")
    if lines != names:
        diff = next(
            ((i, a, b) for i, (a, b) in enumerate(zip(names, lines)) if a != b),
            None,
        )
        raise ValueError(
            f"类别顺序不一致: {names_file} 与 {data_yaml}\n"
            f"首个分歧位置: {diff}\n"
            f"修复: 重跑 convert_tt100k.py --emit-names-to ... 让两者同源"
        )
    print(f"[OK] 类别顺序校验通过: {len(names)} 类，与 {names_file.name} 逐行一致")
    return names


# ==================== 主流程 ====================
def build_train_kwargs(args, resampled: bool) -> dict:
    kwargs = {
        # ── 数据 ──
        "data": args.data,
        "imgsz": args.imgsz,
        # ── 训练轮次与早停 ──
        # 100 轮已达 200 轮终值的 99.4%；patience=20 让平台期提前停
        "epochs": args.epochs,
        "patience": args.patience,
        "time": args.time if args.time > 0 else None,
        "seed": 0,
        "batch": args.batch,
        "workers": args.workers,
        # Ultralytics 要求缓存体积×2 的空闲 RAM (check_cache_ram),
        # 16GB 机器过不了检查会被静默拒绝; 默认关闭, 大内存机器可 --cache
        "cache": args.cache,
        # ── 优化器（已在本数据上实测验证的组合，不做未验证的替换）──
        "optimizer": "MuSGD",
        "lr0": 0.002,
        "lrf": 0.01,
        # 以下 5 项与官方 yolo26s.pt train_args 完全一致（任务无关的优化器参数）
        "momentum": 0.94751,
        "weight_decay": 0.00027,
        "warmup_epochs": 0.98745,
        "warmup_momentum": 0.54064,
        "warmup_bias_lr": 0.05684,
        "cos_lr": True,
        # ── 数据增强（交通标志有方向性）──
        # fliplr 只翻框不改 class id：i2r/i4l、pl*/ph* 数值族被镜像后标签不变，
        # 制造互斥标签，必须关闭
        # 禁止整图旋转/剪切/透视：方向性标志 + 小目标旋转后框膨胀；
        # 官方 train_args 的 0.00012/0.00136/0.00074 是超参搜索残留，语义上即 0
        "degrees": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.0,
        "translate": 0.1,
        # 640 下约四成框 <10px，缩放增强只放大不缩小
        "scale": [0.8, 1.5],
        "mosaic": 0.99182,  # 官方原值
        # Ultralytics 框架默认值；最后 10 轮关 mosaic 做收尾微调
        "close_mosaic": 10,
        "mixup": 0.0,      # 半透明重影干扰限速数字类
        "copy_paste": 0.0,  # bbox-only 标签无 segments，CopyPaste 不会执行
        # 损失权重与 dfl/hsv 取官方 yolo26s.pt train_args 原值；
        # 仅 hsv_v 例外：官方 0.194，因 TT100K 光照/天气变化大取 0.3。
        # 颜色=大类语义（黄警告/红禁令/蓝指示），官方低 hsv_h/s 正合适。零时间成本。
        "box": 9.83241,
        "cls": 0.64896,
        "dfl": 0.95824,
        "hsv_h": 0.01315,
        "hsv_s": 0.35348,
        "hsv_v": 0.3,
        # 重采样清单已做逆频率，避免双重补偿
        "cls_pw": 0.0 if resampled else 0.25,
        # 多尺度会随机放大到 800，拖慢且抬显存峰值，时间优先关闭
        "multi_scale": 0.0,
        # ── 训练/部署检测头一致性 ──
        # 训练内 val 与 best.pt 判定走 NMS-free 头，与 export_onnx.py 导出口径一致
        "nms": False,
        # ── 时间优先：关掉一切非必要的固定开销 ──
        "save_json": False,
        "plots": False,
        "save_period": -1,
        "tile": False,
        # ── 运行 ──
        "device": args.device,
        "project": str(SCRIPT_DIR / "runs" / "detect"),
        "name": args.run_name if not args.smoke else f"{args.run_name}_smoke",
        "exist_ok": True,
        "resume": args.resume,
        "verbose": True,
        "val": True,
        "save": True,
    }
    if args.smoke:
        kwargs.update(
            epochs=1,
            mosaic=0.0,
            close_mosaic=0,
            cache=False,
            val=False,
            fraction=0.02,
        )
    return kwargs


def main() -> None:
    ap = argparse.ArgumentParser(description="YOLO26s 训练 TT100K 48 类（时间效率优先）")
    ap.add_argument("--data", default=DEFAULT_DATA, help="数据集 yaml")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS, help="预训练权重")
    ap.add_argument("--run-name", default=NAME, help="训练输出目录名")
    ap.add_argument("--epochs", type=int, default=100, help="训练轮数")
    ap.add_argument("--patience", type=int, default=20, help="早停耐心轮数")
    ap.add_argument("--batch", type=int, default=16,
                    help="批大小; 640+16 在 8GB 显存已验证不 OOM, "
                         "显存被其他程序占用时退回 8")
    ap.add_argument("--imgsz", type=int, default=640, help="网络输入边长")
    ap.add_argument("--workers", type=int, default=4,
                    help="dataloader 进程数; Windows spawn 每进程约 0.4-0.5GB, "
                         "8 进程曾在轮次边界内存耗尽, 4 为 45 类基线验证值")
    ap.add_argument("--device", default="0", help="'0' / '0,1' / 'cpu'")
    ap.add_argument("--cache", action="store_true",
                    help="开启内存缓存 (需缓存体积×2 的空闲 RAM, 16GB 机器会被框架拒绝)")
    ap.add_argument("--smoke", action="store_true", help="1 epoch 冒烟测试")
    ap.add_argument("--resume", action="store_true", help="从 last.pt 断点续训")
    ap.add_argument("--time", type=float, default=0.0,
                    help="最长训练小时数,0 表示不限时")
    args = ap.parse_args()
    if args.smoke and args.resume:
        ap.error("--smoke 与 --resume 不能同时使用（续训应作用于正式训练）")

    print("=" * 64)
    print(" TrafficSignDetector — YOLO26s 训练（时间效率优先）")
    print("=" * 64)

    # ① 类别顺序校验（失败直接退出，不做无谓训练）
    names = check_class_order(Path(args.data))

    # ② 重采样探测：决定 cls_pw 是否生效
    resampled = is_resampled(Path(args.data))
    if resampled:
        print("[CFG] cls_pw 自动置 0: 重采样清单已做逆频率，避免双重补偿")

    # ③ 环境与依赖自检
    import torch

    print(
        f"[ENV] Python {sys.version.split()[0]}  "
        f"torch {torch.__version__}  "
        f"CUDA available={torch.cuda.is_available()}"
    )
    if torch.cuda.is_available():
        print(
            f"[ENV] GPU: {torch.cuda.get_device_name(0)}  "
            f"显存 {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )
    if args.device != "cpu" and not torch.cuda.is_available():
        print("[WARN] device 指定为 GPU 但 CUDA 不可用，自动回退 CPU (会非常慢)")
        args.device = "cpu"

    import ultralytics
    from ultralytics import YOLO

    print(f"[ENV] ultralytics {ultralytics.__version__}")

    # ④ 加载模型（断点续训必须从 last.pt 加载）
    run_name = args.run_name if not args.smoke else f"{args.run_name}_smoke"
    weights = (
        SCRIPT_DIR / "runs" / "detect" / run_name / "weights" / "last.pt"
        if args.resume
        else Path(args.weights)
    )
    if args.resume and not weights.is_file():
        sys.exit(f"[FAIL] 断点续训需要: {weights}")
    print(f"\n[LOAD] 权重: {weights}")
    model = YOLO(str(weights))

    # ⑤ 组装超参
    kwargs = build_train_kwargs(args, resampled)
    print("\n[CFG] 关键超参:")
    for k in (
        "data", "epochs", "patience", "time", "batch", "imgsz", "workers",
        "optimizer", "lr0", "lrf", "cos_lr", "momentum", "weight_decay",
        "warmup_epochs", "box", "cls", "dfl",
        "fliplr", "scale", "mosaic", "close_mosaic", "mixup",
        "hsv_h", "hsv_s", "hsv_v", "cls_pw", "multi_scale", "nms", "cache",
    ):
        print(f"      {k:12s} = {kwargs.get(k)}")

    # ⑥ 开训
    print("\n[TRAIN] 开始训练 ...\n")
    model.train(**kwargs)

    # ⑦ 结果汇总
    run_dir = SCRIPT_DIR / "runs" / "detect" / run_name
    best_pt = run_dir / "weights" / "best.pt"
    last_pt = run_dir / "weights" / "last.pt"

    print("\n" + "=" * 64)
    print(" 训练完成")
    print("=" * 64)
    print(f"  最佳权重 : {best_pt.resolve() if best_pt.exists() else '(未生成)'}")
    print(f"  末轮权重 : {last_pt.resolve() if last_pt.exists() else '(未生成)'}")
    print(f"  类别数   : {len(names)}")

    if best_pt.exists() and not args.smoke:
        print("\n[EVAL] 加载最佳权重进行验证 ...")
        best_model = YOLO(str(best_pt))
        metrics = best_model.val(
            data=args.data,
            imgsz=kwargs["imgsz"],  # 与训练分辨率一致，否则独立 val 会退回默认 640
            nms=False,              # 与训练内 val / ONNX 导出的 NMS-free 头同口径
            project=str(SCRIPT_DIR / "runs" / "detect"),
            name=f"{run_name}_val",
            verbose=False,
        )
        print(f"  mAP@0.5      : {metrics.box.map50:.4f}")
        print(f"  mAP@0.5:0.95 : {metrics.box.map:.4f}")
        print(f"  Precision    : {metrics.box.mp:.4f}")
        print(f"  Recall       : {metrics.box.mr:.4f}")

    print("\n[NEXT] 需要背景误检择优时手动执行: "
          f"python select_best.py --run-dir \"{run_dir}\" --data \"{args.data}\" "
          f"--imgsz {kwargs['imgsz']}")


# Windows + workers>0 必须保留此保护：dataloader 用 spawn，
# 没有 main guard 会无限递归创建子进程
if __name__ == "__main__":
    main()