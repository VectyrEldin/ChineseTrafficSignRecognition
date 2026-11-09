#!/usr/bin/env python3
"""
convert_tt100k.py —— tt100k 2021 → YOLO(ultralytics) 格式转换

适配目录结构：
  tt100k_2021/
  ├── annotations_all.json
  ├── train/*.jpg
  └── test/*.jpg

用法：
  uv run python convert_tt100k.py \
      --annotation ../data/tt100k_2021/annotations_all.json \
      --image-dir  ../data/tt100k_2021/train \
      --test-image-dir ../data/tt100k_2021/test \
      --output     ../data/tt100k48_yolo \
      --link

注意：上述示例路径相对于 scripts/ 目录，请在该目录下执行；
      也可直接传入绝对路径。

默认行为：在频次前 45 类之后追加 io/po/wo 三大父类、自动定位 other 图库、
          按 group 划分，
          并按 DEFAULT_BACKGROUND_RATIO 纳入背景图 (空 label)。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

try:
    import yaml
except ImportError:
    sys.exit("缺少 pyyaml, 请先执行: uv add pyyaml")


# ==================== 常量 ====================
DEFAULT_MIN_SAMPLES = 100
DEFAULT_MAX_CLASSES = 45
TRAIN_RATIO_PCT = 80
NAMES_FILE = "tt100k48.names"
CLASS_ORDER_FILE = "tt100k48_class_order.json"
YAML_FILE = "tt100k48.yaml"
LABEL_GLOB = "*.txt"

# 48 类方案默认值：类别码首字母即官方大类
SUPERCLASSES = ("io", "po", "wo")
SUPERCLASS_BY_PREFIX = {"i": "io", "p": "po", "w": "wo"}
# Ultralytics 建议背景图占训练集 0-10%; 0.10×候选池 ≈ train 总数的 9%
DEFAULT_BACKGROUND_RATIO = 0.10


# ==================== 通用工具 ====================
def require(condition: bool, message: str) -> None:
    """显式校验 : python -O 不会剥离，类别顺序等关键约束必须用它"""
    if not condition:
        raise ValueError(message)


def bbox_to_coordinate(bbox: Any) -> tuple[float, float, float, float] | None:
    try:
        if isinstance(bbox, dict):
            xmin = float(bbox["xmin"])
            ymin = float(bbox["ymin"])
            xmax = float(bbox["xmax"])
            ymax = float(bbox["ymax"])
        else:
            xmin, ymin, xmax, ymax = (float(v) for v in bbox[:4])
    except KeyError, TypeError, ValueError, IndexError:
        return None
    if xmax <= xmin or ymax <= ymin:
        return None
    return xmin, ymin, xmax, ymax


def _get_image_wh(img_data: dict, img_path: Path | None) -> tuple[float, float]:
    """优先从标注读宽高，没有则从图片文件读"""
    img_w = float(img_data.get("width", 0))
    img_h = float(img_data.get("height", 0))
    if img_w > 0 and img_h > 0:
        return img_w, img_h

    if img_path is None or not img_path.is_file():
        return 0.0, 0.0

    try:
        with Image.open(img_path) as im:
            return float(im.width), float(im.height)
    except OSError, Image.UnidentifiedImageError:
        print(f"  [WARN] 无法读取图片尺寸: {img_path}")
        return 0.0, 0.0  # ← 关键：except 里也要 return


def load_annotation(path: Path) -> dict:
    if not path.is_file():
        sys.exit(f"[FAIL] 标注文件不存在: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "imgs" not in data:
        sys.exit(f"[FAIL] 标注文件缺少 'imgs' 字段: {path}")
    return data


def deterministic_is_train(img_id: str, split_mode: str = "hash", group_size: int = 16) -> bool:
    """决定图片进入 train 还是 val。
    group : 把相邻 group_size 个连续 id 视为同一拍摄序列，对组号求哈希。
    """
    if split_mode == "group":
        try:
            key = str(int(str(img_id)) // group_size)
        except ValueError:
            key = str(img_id)
    else:
        key = str(img_id)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) % 100) < TRAIN_RATIO_PCT


def find_image(image_dir: Path, img_id: str) -> Path | None:
    p = image_dir / f"{img_id}.jpg"
    return p if p.is_file() else None


# ==================== 分桶 ====================
def _is_train_path(path: str) -> bool:
    """按路径段判断，避免路径中任意 train 子串误命中"""
    parts = [p.lower() for p in str(path).replace("\\", "/").split("/") if p]
    return "train" in parts


def _path_segment(img_data: dict) -> str:
    """取 path 的首段 (train / test / other)，用于三段式图库查找"""
    parts = str(img_data.get("path", "")).replace("\\", "/").split("/")
    return parts[0].lower() if parts and parts[0] else ""


def _get_bucket(
    img_id: str,
    img_data: dict,
    other_to_train: bool = False,
    split_mode: str = "hash",
    group_size: int = 16,
) -> str:
    """根据 path 字段分桶 : train→train/val(8:2), test→test

    other_to_train=True 时, other 目录的带框图并入 train,
    使 val/test 组成保持不变, mAP 与既有结果仍可比。
    """
    if _is_train_path(img_data.get("path", "")):
        return "train" if deterministic_is_train(img_id, split_mode, group_size) else "val"
    if other_to_train and _path_segment(img_data) == "other":
        return "train"
    return "test"


# ==================== 类别筛选 ====================
def select_classes(ann: dict, min_samples: int, max_classes: int):
    counter: Counter[str] = Counter()
    for img_data in ann["imgs"].values():
        for obj in img_data.get("objects", []):
            cat = obj.get("category")
            if cat is not None:
                counter[str(cat)] += 1

    qualified = [(c, n) for c, n in counter.items() if n >= min_samples]
    qualified.sort(key=lambda x: (-x[1], x[0]))
    selected = qualified[:max_classes]

    require(
        len(selected) == max_classes,
        (
            f"符合 min_samples={min_samples} 的类别只有 {len(selected)} 个，"
            f"不足 {max_classes}。请降低 --min-samples 后重跑。"
        ),
    )

    old_to_new = {old_id: new_id for new_id, (old_id, _) in enumerate(selected)}
    return selected, old_to_new


def select_classes_extended(
    ann: dict, min_samples: int, max_classes: int
) -> tuple[list[str], dict[str, int], dict[str, int]]:
    """返回 (names, raw_to_new, counts)。

    在频次前 max_classes 类之后追加 io/po/wo 三个大类，并让全部原始类别码都有归属，
    使未见标志不再被丢弃，而是落到大类出口。
    """
    selected, old_to_new = select_classes(ann, min_samples, max_classes)
    names = [c for c, _ in selected]
    raw_to_new = dict(old_to_new)

    counter: Counter[str] = Counter()
    for img_data in ann["imgs"].values():
        for obj in img_data.get("objects", []):
            cat = obj.get("category")
            if cat is not None:
                counter[str(cat)] += 1

    for offset, name in enumerate(SUPERCLASSES, start=len(names)):
        require(name not in raw_to_new, f"大类 {name} 已存在于具体类别中")
        raw_to_new[name] = offset
    names.extend(SUPERCLASSES)

    for cat in counter:
        if cat not in raw_to_new:
            require(cat[0] in SUPERCLASS_BY_PREFIX, f"无法归类的大类前缀: {cat}")
            raw_to_new[cat] = raw_to_new[SUPERCLASS_BY_PREFIX[cat[0]]]

    mapped: Counter[str] = Counter()
    for cat, n in counter.items():
        mapped[names[raw_to_new[cat]]] += n
    return names, raw_to_new, dict(mapped)


# ==================== 写文件 ====================
def write_names_file(path: Path, selected: list[tuple[str, int]]) -> list[str]:
    names = [old_id for old_id, _ in selected]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for n in names:
            f.write(f"{n}\n")
    return names


def write_yaml(path: Path, names: list[str], train_entry: str = "images/train") -> None:
    """写数据集 yaml。

    刻意不写 `path:` : Ultralytics check_det_dataset 在 path 缺省时会用
    yaml 文件所在目录作为数据集根 (YAML.load(append_filename=True) 会写入
    yaml_file。旧的 `path: ../data/...` 是相对「当前工作目录」解析的，
    只有从 scripts/ 启动才正确，从项目根启动会解析到 Z:/data/... 并失败。
    省略 path 后，从任意目录运行、以及整目录搬迁都成立。
    """
    dict_lines = [f"  - {name}" for name in names]
    content = (
        f"train: {train_entry}\nval: images/val\ntest: images/test\n\n"
        f"nc: {len(names)}\nnames:\n" + "\n".join(dict_lines) + "\n"
    )
    path.write_text(content, encoding="utf-8")


def _parse_label_file(txt_path: Path) -> list[int]:
    """解析单个 label 文件，返回类别 ID 列表。"""
    class_ids: list[int] = []
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if parts:
            class_ids.append(int(parts[0]))
    return class_ids


def _collect_label_stats(
    label_dir: Path, image_dir: Path
) -> tuple[Counter[int], dict[str, list[int]]]:
    """扫描 train labels, 返回 (类别计数, 每图类别列表)。"""
    counts: Counter[int] = Counter()
    per_image: dict[str, list[int]] = {}

    for txt in sorted(label_dir.glob(LABEL_GLOB)):
        class_ids = _parse_label_file(txt)
        if not class_ids:
            continue
        if not (image_dir / f"{txt.stem}.jpg").is_file():
            continue
        per_image[txt.stem] = class_ids
        for cid in class_ids:
            counts[cid] += 1

    return counts, per_image


def write_balanced_train_list(output_dir: Path, max_repeat: int) -> tuple[Path, dict]:
    """类感知重采样：把含稀有类的训练图重复列入 train_balanced.txt。

    论文 §4.2 把 100-1000 实例的类补到 1000; 本项目无法用 marks 模板做
    模板增广（仅覆盖 96/201 类），改用不依赖模板的「逆频率平方根重复采样」。
    按图片中出现的最稀有类别决定重复次数，上限 max_repeat,逐轮交错写出
    以免同一张图连续出现。路径用 ./ 前缀,Ultralytics 会相对 txt 所在目录解析，
    img2label_paths 也能正确把 /images/ 映射到 /labels/。
    """
    label_dir = output_dir / "labels" / "train"
    image_dir = output_dir / "images" / "train"

    # ① 收集统计（已抽离，无嵌套惩罚）
    counts, per_image = _collect_label_stats(label_dir, image_dir)

    # ② 计算每张图的重复次数
    max_count = max(counts.values(), default=1)
    repeats: dict[str, int] = {}
    for stem, class_ids in per_image.items():
        valid = [counts[c] for c in class_ids if counts[c] > 0]
        rarest = min(valid) if valid else 1
        factor = (max_count / rarest) ** 0.5
        repeats[stem] = max(1, min(max_repeat, round(factor)))

    # ③ 逐轮交错写出
    lines: list[str] = []
    for round_idx in range(max_repeat):
        for stem in sorted(per_image):
            if repeats[stem] > round_idx:
                lines.append(f"./images/train/{stem}.jpg")

    # ④ 写文件 & 统计
    list_path = output_dir / "train_balanced.txt"
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    stats = {
        "images": len(per_image),
        "list_entries": len(lines),
        "max_repeat": max_repeat,
        "max_class_count": max_count,
        "min_class_count": min(counts.values(), default=0),
        "repeat_histogram": dict(Counter(repeats.values())),
    }
    return list_path, stats


def sha256_of_names(names: list[str]) -> str:
    """对类别顺序求指纹 —— 后续可与 Java 等后端比对，确认类别顺序同源"""
    payload = "\n".join(names).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def write_class_order_json(path: Path, names: list[str], selected: list[tuple[str, int]]) -> str:
    """
    落一份机器可读的类别顺序清单（含 sha256 指纹）。
    """
    digest = sha256_of_names(names)
    payload = {
        "num_classes": len(names),
        "sha256_16": digest,
        "names": names,
        "counts": dict(selected),
        "note": "行号即类别 ID; 修改顺序 = 必须重新训练",
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return digest


# ==================== 转换 ====================
# _process_single_image 的返回码，取代原来含糊的 (bool, bool)
class _SkipReason:
    WROTE = "wrote"  # 正常写出 label + 拷图
    NO_RETENTION = "no_retention"  # 无保留类别 / 无有效标注 → 明确跳过，不写空 label
    BAD_GEOM = "bad_geom"  # 尺寸非法或 bbox 全部越界
    MISSING_IMG = "missing_img"  # label 已写但图片没找到


def _convert_single_object(obj, img_w, img_h, old_to_new, num_classes, emit_segments=False):
    old_cat = str(obj.get("category", ""))
    new_cat = old_to_new.get(old_cat)
    if new_cat is None:
        return None  # 不在保留类别内，丢弃
    coordinate = bbox_to_coordinate(obj.get("bbox"))
    if coordinate is None:
        return None
    xmin, ymin, xmax, ymax = coordinate
    cx = ((xmin + xmax) / 2.0) / img_w
    cy = ((ymin + ymax) / 2.0) / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))
    if w <= 0.0 or h <= 0.0:
        return None
    require(0 <= new_cat < num_classes, "类别 ID 越界")
    x1, y1 = max(0.0, cx - w / 2), max(0.0, cy - h / 2)
    x2, y2 = min(1.0, cx + w / 2), min(1.0, cy + h / 2)
    if emit_segments:
        # Ultralytics data/utils.py::verify_image_label 把 len(row)>6 的行判为 segment，
        # 并从第 1 列起解析多边形点（x[1:].reshape(-1,2)），再 segments2boxes 反算框。
        # 因此 segment 行必须是 9 列 `cls x1 y1 x2 y1 x2 y2 x1 y2`，不能带 bbox 前缀；
        # 若写 13 列，cx/cy/w/h 会被当成顶点，框会被静默改错（实测 0.5/0.5/0.1/0.2
        # 会变成 0.325/0.4/0.45/0.4）。四角矩形反算出的框与 bbox 完全一致。
        corners = ((x1, y1), (x2, y1), (x2, y2), (x1, y2))
        return f"{new_cat} " + " ".join(f"{x:.6f} {y:.6f}" for x, y in corners)
    return f"{new_cat} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def _convert_objects_to_yolo_lines(
    img_data, img_w, img_h, old_to_new, num_classes, emit_segments=False
):
    """只返回「保留类别且几何合法」的 YOLO 行；空列表 = 这张图不该有 label"""
    lines = []
    for obj in img_data.get("objects", []) or []:
        line = _convert_single_object(obj, img_w, img_h, old_to_new, num_classes, emit_segments)
        if line:
            lines.append(line)
    return lines


def _copy_image(img_id, image_dir, image_out_dir, use_hard_link):
    src = find_image(image_dir, img_id)
    if src is None:
        return False
    dst = image_out_dir / src.name
    if dst.exists():
        return True
    if use_hard_link:
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    else:
        shutil.copy2(src, dst)
    return True


# ==================== 处理单张图片 ====================
def _process_single_image(
    img_id,
    img_data,
    image_dir,
    test_image_dir,
    output_dir,
    old_to_new,
    num_classes,
    use_hard_link,
    other_image_dir=None,
    other_to_train=False,
    split_mode="hash",
    group_size=16,
    emit_segments=False,
):
    """
    返回 (_SkipReason, img_copied)。
    —— 关键：只有当图片里至少有一个「保留类别」的有效框时才写 label,
       否则返回 NO_RETENTION / BAD_GEOM, 绝不生成空 label 文件，
       避免训练时出现「有图无框 / 空标签」告警。
    """
    if other_image_dir is None:
        path = img_data.get("path", "")
        search_dir = image_dir if _is_train_path(path) else (test_image_dir or image_dir)
    else:
        segment = _path_segment(img_data)
        search_dir = {
            "train": image_dir,
            "test": test_image_dir or image_dir,
            "other": other_image_dir,
        }.get(segment, image_dir)
    src = find_image(search_dir, img_id)

    img_w, img_h = _get_image_wh(img_data, src)
    if img_w <= 0 or img_h <= 0:
        return _SkipReason.BAD_GEOM, False

    lines = _convert_objects_to_yolo_lines(
        img_data, img_w, img_h, old_to_new, num_classes, emit_segments
    )
    # 明确跳过：没有任何保留类别的有效标注 → 不写 label、不拷图
    if not lines:
        return _SkipReason.NO_RETENTION, False

    bucket = _get_bucket(
        img_id,
        img_data,
        other_to_train=other_to_train,
        split_mode=split_mode,
        group_size=group_size,
    )
    label_dir = output_dir / "labels" / bucket
    image_out_dir = output_dir / "images" / bucket
    label_dir.mkdir(parents=True, exist_ok=True)
    image_out_dir.mkdir(parents=True, exist_ok=True)

    (label_dir / f"{img_id}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    img_copied = _copy_image(img_id, search_dir, image_out_dir, use_hard_link)
    if not img_copied:
        # label 已落盘但图缺失：训练会报错，这里回滚 label 并标记为缺失
        (label_dir / f"{img_id}.txt").unlink(missing_ok=True)
        return _SkipReason.MISSING_IMG, False

    return _SkipReason.WROTE, True


def convert_all(
    ann,
    image_dir,
    test_image_dir,
    output_dir,
    old_to_new,
    num_classes,
    use_hard_link,
    other_image_dir=None,
    other_to_train=False,
    split_mode="hash",
    group_size=16,
    emit_segments=False,
):
    n_img = n_lbl = 0
    missing = 0  # 图片找不到（label 也已回滚，不会留下孤儿标签）
    no_retention = 0  # 明确跳过：无保留类别 / 无有效标注
    bad_geom = 0  # 尺寸非法
    for img_id, img_data in ann["imgs"].items():
        reason, wrote_img = _process_single_image(
            img_id,
            img_data,
            image_dir,
            test_image_dir,
            output_dir,
            old_to_new,
            num_classes,
            use_hard_link,
            other_image_dir,
            other_to_train,
            split_mode,
            group_size,
            emit_segments,
        )
        if reason == _SkipReason.WROTE:
            n_lbl += 1
            if wrote_img:
                n_img += 1
        elif reason == _SkipReason.MISSING_IMG:
            missing += 1
        elif reason == _SkipReason.NO_RETENTION:
            no_retention += 1
        elif reason == _SkipReason.BAD_GEOM:
            bad_geom += 1

    if no_retention:
        print(f"  [INFO] {no_retention} 张图片无保留类别的有效标注，已明确跳过 (未生成空 label)")
    if bad_geom:
        print(f"  [WARN] {bad_geom} 张图片尺寸非法或框全部无效，已跳过")
    if missing:
        print(f"  [WARN] {missing} 张图片在图库目录下未找到，已跳过 (并回滚对应 label)")
    return n_img, n_lbl


def write_background(
    ann: dict,
    image_dirs: dict[str, Path],
    output_dir: Path,
    ratio: float,
    use_hard_link: bool,
) -> int:
    """把「无标注记录」的图作为空 label 背景图写进 train。

    采样按 (目录段, 文件名) 的 sha256 排序，保证可复现。
    跨目录同名的 jpg 若目标已存在则跳过，避免覆盖带框图。
    """
    registered: set[tuple[str, str]] = set()
    for img_data in ann["imgs"].values():
        parts = str(img_data.get("path", "")).replace("\\", "/").split("/")
        key = (parts[0].lower(), parts[-1]) if len(parts) >= 2 else ("", parts[-1])
        registered.add(key)

    candidates = sorted(
        (segment, path)
        for segment, directory in image_dirs.items()
        for path in directory.glob("*.jpg")
        if (segment, path.name) not in registered
    )
    take = int(len(candidates) * ratio) if ratio > 0 else 0
    chosen = sorted(
        candidates,
        key=lambda x: hashlib.sha256(f"{x[0]}/{x[1].name}".encode()).hexdigest(),
    )[:take]

    image_out = output_dir / "images" / "train"
    label_out = output_dir / "labels" / "train"
    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)

    written = 0
    for _, src in chosen:
        dst = image_out / src.name
        if dst.exists():
            continue
        if use_hard_link:
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
        else:
            shutil.copy2(src, dst)
        (label_out / f"{src.stem}.txt").write_text("", encoding="utf-8")
        written += 1
    return written


# ==================== 校验 ====================
def verify(output_dir: Path, names: list[str], num_classes: int) -> None:
    """三处顺序一致性校验: names 文件 / yaml / 实际 label 中的 class id"""
    print("\n" + "=" * 60)
    print("顺序校验（三处一致）")
    print("=" * 60)

    # ① names 文件自身
    names_file = output_dir / NAMES_FILE
    lines = [l.strip() for l in names_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    require(len(lines) == num_classes, f"names 文件 {len(lines)} 行，期望 {num_classes}")
    require(len(set(lines)) == num_classes, "names 文件存在重复类别名")
    require(lines == names, "names 文件与内存中的 selected 顺序不一致（写入异常）")
    print(f"  [OK] {NAMES_FILE}: {len(lines)} 行，无重复")

    # ② yaml 的 names 列表
    yaml_path = output_dir / YAML_FILE
    y = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert isinstance(y, dict) and y is not None, f"yaml 文件为空或格式错误: {yaml_path}"

    require(y["nc"] == num_classes, f"yaml nc={y['nc']}，期望 {num_classes}")
    require(y["names"] == lines, "yaml names 与 .names 文件顺序不一致！")
    print(f"  [OK] {YAML_FILE}: nc={y['nc']}, names 顺序与 .names 一致")

    # ③ 实际 label 文件中的 class id
    used: set[int] = set()
    max_id = -1
    for txt in (output_dir / "labels").rglob(LABEL_GLOB):  # <-- 使用常量
        for line in txt.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            cid = int(line.split()[0])
            require(0 <= cid < num_classes, f"{txt} 中出现越界 class id={cid}")
            used.add(cid)
            max_id = max(max_id, cid)
    require(
        max_id == num_classes - 1,
        (
            f"最大 class id={max_id}，期望 {num_classes - 1}; "
            f"未被使用的类别: {sorted(set(range(num_classes)) - used)}"
        ),
    )
    print(f"  [OK] label class id 范围 [0, {max_id}], {num_classes} 类全部有样本")

    # ④ 图片/标签数量匹配
    for bucket in ("train", "val", "test"):
        image_dir, label_dir = (
            output_dir / "images" / bucket,
            output_dir / "labels" / bucket,
        )
        if not image_dir.exists():
            continue
        ni = len(list(image_dir.glob("*")))  # 图片格式多样，保持 *
        nl = len(list(label_dir.glob(LABEL_GLOB)))  # <-- 使用常量
        require(
            ni == nl,
            f"{bucket}: 图片 {ni} 张 != 标签 {nl} 个（有图无标签会让训练静默丢数据）",
        )
        print(f"  [OK] {bucket:5s}: 图片 {ni} 张 / 标签 {nl} 个")

    print("  ✅ 全部校验通过，类别顺序已锁定")


# ==================== main ====================
# ==================== main 拆分出来的私有函数 ====================
def _parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotation", required=True, type=Path)
    ap.add_argument("--image-dir", required=True, type=Path)
    ap.add_argument("--test-image-dir", type=Path, default=None)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES)
    ap.add_argument("--max-classes", type=int, default=DEFAULT_MAX_CLASSES)
    ap.add_argument("--link", action="store_true")
    ap.add_argument("--emit-names-to", type=Path, default=None)
    ap.add_argument(
        "--other-image-dir", type=Path, default=None, help="other 目录；默认自动从标注文件同级推断"
    )
    ap.add_argument(
        "--superclass", action="store_true", help="兼容保留: 48 类方案已默认启用，无需显式传入"
    )
    ap.add_argument(
        "--background-ratio",
        type=float,
        default=DEFAULT_BACKGROUND_RATIO,
        help="无标注背景图纳入 train 的比例（默认 0.10），0 表示关闭",
    )
    ap.add_argument(
        "--split-mode",
        choices=("hash", "group"),
        default="group",
        help="group=按相邻 id 分组划分(默认，消除近重复帧泄漏)；hash=逐图哈希(旧行为，逐字节复现)",
    )
    ap.add_argument(
        "--emit-segments",
        action="store_true",
        help="在 bbox 后追加四角多边形，使 Ultralytics copy_paste 真正生效",
    )
    ap.add_argument(
        "--group-size", type=int, default=16, help="group 模式下每组连续的 id 数量（默认 16）"
    )
    ap.add_argument(
        "--class-balanced", action="store_true", help="生成 train_balanced.txt 做逆频率类感知重采样"
    )
    ap.add_argument(
        "--max-repeat", type=int, default=3, help="--class-balanced 时单张图片最多重复次数"
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="清空已有的 images/ 与 labels/ 后重建（不可撤销）",
    )
    args = ap.parse_args()
    return args


def _prepare_output(args: argparse.Namespace) -> None:
    """处理 --overwrite 逻辑，确保输出目录可用。"""
    if args.overwrite:
        for sub in ("images", "labels"):
            shutil.rmtree(args.output / sub, ignore_errors=True)
    elif any(
        (args.output / s).is_dir() and any((args.output / s).iterdir())
        for s in ("images", "labels")
    ):
        sys.exit("[FAIL] 输出目录已有数据；请加 --overwrite 或换一个 --output")
    args.output.mkdir(parents=True, exist_ok=True)


def _select_and_print_classes(
    args: argparse.Namespace, ann: dict
) -> tuple[list[str], dict[str, int], list[tuple[str, int]]]:
    """筛选类别、打印映射表、写出 names/yaml/class_order.json。"""
    names, raw_to_new, counts = select_classes_extended(ann, args.min_samples, args.max_classes)
    selected = [(name, counts[name]) for name in names]
    print(f"[1/6] 类别筛选完成: 保留 {len(names)} 类（含 3 个大类）")

    # other 目录自动推断
    other_dir = args.other_image_dir
    if other_dir is None:
        guess = args.annotation.parent / "other"
        if guess.is_dir():
            other_dir = guess
            print(f"      [INFO] 自动使用 other 图库: {other_dir}")

    # 打印映射表
    print(f"\n{'新ID':>4} | {'原类别ID':<20} | {'样本数':>7}")
    print("-" * 40)
    for new_id, (old_id, cnt) in enumerate(selected):
        print(f"{new_id:>4} | {old_id:<20} | {cnt:>7}")

    # 写元数据文件
    write_names_file(args.output / NAMES_FILE, selected)
    write_yaml(args.output / YAML_FILE, names)
    digest = write_class_order_json(args.output / CLASS_ORDER_FILE, names, selected)
    print("[2/6] 写出 names / yaml / class_order.json")
    print(f"      类别顺序指纹 sha256[:16] = {digest}")

    return names, raw_to_new, selected


def _run_conversion(
    args: argparse.Namespace,
    ann: dict,
    raw_to_new: dict[str, int],
    names: list[str],
) -> None:
    """执行标注转换并打印统计。"""
    other_dir = args.other_image_dir
    if other_dir is None:
        guess = args.annotation.parent / "other"
        other_dir = guess if guess.is_dir() else None

    print("[3/6] 转换标注...")
    n_img, n_lbl = convert_all(
        ann,
        args.image_dir,
        args.test_image_dir,
        args.output,
        raw_to_new,
        len(names),
        args.link,
        other_dir,
        True,
        args.split_mode,
        args.group_size,
        args.emit_segments,
    )
    print(
        f"      图片 {n_img} 张 / 标签 {n_lbl} 个  "
        f"(划分模式={args.split_mode}, 多边形标签={args.emit_segments})"
    )


def _write_background_if_needed(args: argparse.Namespace, ann: dict) -> None:
    """按需写入背景图。"""
    if args.background_ratio <= 0:
        return

    image_dirs = {"train": args.image_dir}
    if args.test_image_dir is not None:
        image_dirs["test"] = args.test_image_dir

    other_dir = args.other_image_dir
    if other_dir is None:
        guess = args.annotation.parent / "other"
        other_dir = guess if guess.is_dir() else None
    if other_dir is not None:
        image_dirs["other"] = other_dir

    n_bg = write_background(ann, image_dirs, args.output, args.background_ratio, args.link)
    print(f"      背景图 {n_bg} 张（空 label，仅 train）")


def _write_balanced_list_if_needed(
    args: argparse.Namespace, output_dir: Path, names: list[str]
) -> None:
    """按需生成类平衡采样列表。"""
    if not args.class_balanced:
        return
    list_path, stats = write_balanced_train_list(output_dir, args.max_repeat)
    write_yaml(output_dir / YAML_FILE, names, train_entry=list_path.name)
    print(
        f"[4/6] 类感知重采样: {list_path.name} "
        f"{stats['images']} 张 → {stats['list_entries']} 条 "
        f"(重复分布 {stats['repeat_histogram']})"
    )
    print(f"      yaml train 已指向 {list_path.name}")


def _copy_names_if_needed(args: argparse.Namespace, output_dir: Path) -> None:
    """按需同步 names 文件到指定位置。"""
    if not args.emit_names_to:
        return
    args.emit_names_to.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output_dir / NAMES_FILE, args.emit_names_to)
    print(f"[4/6] 已同步 names → {args.emit_names_to}")


# ==================== main ====================
def main() -> None:
    args = _parse_args()
    _prepare_output(args)

    ann = load_annotation(args.annotation)
    names, raw_to_new, _ = _select_and_print_classes(args, ann)

    _run_conversion(args, ann, raw_to_new, names)

    if args.background_ratio > 0:
        _write_background_if_needed(args, ann)

    _write_balanced_list_if_needed(args, args.output, names)
    _copy_names_if_needed(args, args.output)

    print("[5/6] 校验...")
    verify(args.output, names, len(names))

    print("\n[6/6] ✅ 转换完成")


if __name__ == "__main__":
    main()
