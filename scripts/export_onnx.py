#!/usr/bin/env python3
"""
export_onnx.py —— 导出 YOLO26s → ONNX, 并用 onnxruntime 做 dummy 推理验证
(已重构以降低认知复杂度)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

# ==================== 默认路径 ====================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_WEIGHTS = str(
    SCRIPT_DIR / "runs" / "detect" / "tt100k48_yolo26s" / "weights" / "best.pt"
)
DEFAULT_NAMES = str(PROJECT_ROOT / "data" / "tt100k48_yolo" / "tt100k48.names")
JAVA_MODEL_DST = (
    PROJECT_ROOT / "app" / "src" / "main" / "resources" / "models" / "best.onnx"
)
JAVA_NAMES_DST = (
    PROJECT_ROOT / "app" / "src" / "main" / "resources" / "config" / "tt100k48.names"
)
# 覆盖部署前的旧文件备份放 resources 之外，避免 .bak 被打进 jar / 安装包
DEPLOY_BACKUP_DIR = PROJECT_ROOT / "backup" / "deployed"
IMGSZ = 640  # 与 train.py 默认 imgsz 一致；Java 端由 manifest 自动适配

# 预处理契约：必须与 Ultralytics LetterBox / Java YoloPreprocessor 完全一致。
# Ultralytics 训练(RandomPerspective.cv2.warpAffine)、val(dataset.py LetterBox)、
# predict(LetterBox) 默认均为 cv2.INTER_LINEAR；Java 侧曾误用 INTER_AREA。
PREPROCESS_CONTRACT = {
    "resize_interpolation": "INTER_LINEAR",
    "round_mode": "rint",          # Python round() / Java Math.rint()（同为 round-to-even）
    "pad_value": 114,
    "color_order": "RGB",
    "normalization": "div255",
    "letterbox": "center",
}
# 端到端 NMS-free 输出的列语义，写入 manifest 后 Java 端不再靠 shape 猜测。
END2END_COLUMNS = ["x1", "y1", "x2", "y2", "score", "class_id"]


# ==================== 工具 ====================
def load_names(path: Path) -> list[str]:
    """类别数不硬编码：以同目录 class_order.json 的 num_classes 为期望值。"""
    if not path.is_file():
        raise FileNotFoundError(f"类别文件缺失: {path}")
    names = [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not names:
        raise ValueError(f"类别文件为空: {path}")
    order_file = path.with_name(path.stem + "_class_order.json")
    if order_file.is_file():
        expected = json.loads(order_file.read_text(encoding="utf-8")).get("num_classes")
        if expected is not None and len(names) != expected:
            raise ValueError(
                f"类别数应为 {expected}（来自 {order_file.name}），实际 {len(names)}"
            )
    if len(set(names)) != len(names):
        raise ValueError("类别文件存在重复项")
    return names


def dim_to_list(dims) -> list:
    out = []
    for d in dims:
        if d.HasField("dim_value") and d.dim_value > 0:
            out.append(int(d.dim_value))
        elif d.dim_param:
            out.append(str(d.dim_param))
        else:
            out.append("?")
    return out


def probe_coord_scale(arr: np.ndarray, size: int) -> str:
    if arr.ndim < 2 or arr.shape[-1] < 4:
        return "unknown"
    box_part = arr[..., :4]
    finite = box_part[np.isfinite(box_part)]
    if finite.size == 0:
        return "unknown"
    mx = float(np.max(np.abs(finite)))
    if mx <= 2.0:
        return "normalized(0~1)"
    return f"pixel(0~{int(size)})"


def detect_layout(shape: tuple, nc: int) -> dict:
    info = {"shape": list(shape), "layout": "unknown", "need_nms": None, "note": ""}
    if len(shape) == 3:
        _b, n, c = shape
        if c == 6:
            info.update(
                layout="end2end_nms_free",
                need_nms=False,
                note=(
                    f"端到端输出：{n} 个已去重框，每行 [x1, y1, x2, y2, score, class_id]。"
                    f"Java 端禁止再做 NMS，直接按行读取并按 score 阈值过滤。"
                ),
            )
        elif c > 1000 and n in (4 + nc, 5 + nc, 6 + nc):
            info.update(
                layout="raw_head_transposed",
                need_nms=True,
                note=(
                    f"原始头输出：{c} 个候选框 * {n} 通道，需先转置为 ({c}, {n}), "
                    f"前 4 列为 cx,cy,w,h (或 x1,y1,x2,y2)，其余为 {nc} 个类别分。"
                    f"Java 端必须自己实现 NMS。"
                ),
            )
        else:
            info["note"] = (
                "⚠️ 未知布局，请人工阅读 shape 含义后手工修改 Java 端 PostProcessor。"
            )
    elif len(shape) == 2:
        info["note"] = "二维输出，请人工确认布局。"
    return info


# ==================== 参数解析 ====================
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="导出 YOLO26s 为 ONNX 并验证输出格式")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--names", default=DEFAULT_NAMES)
    ap.add_argument("--opset", type=lambda x: int(x), default=12)
    ap.add_argument("--imgsz", type=lambda x: int(x), default=IMGSZ)
    ap.add_argument("--skip-simplify", action="store_true")
    ap.add_argument("--no-copy", action="store_true")
    ap.add_argument(
        "--dst",
        type=Path,
        default=JAVA_MODEL_DST,
        help="Java 资源目录中的目标 onnx 路径",
    )
    return ap.parse_args()


# ==================== 步骤函数 ====================
def export_model(args: argparse.Namespace, weights: Path) -> tuple[Path, int]:
    from ultralytics import YOLO

    print("\n[1/6] 导出 ONNX ...")
    model = YOLO(str(weights))
    used_opset = args.opset
    try:
        exported = model.export(
            format="onnx", imgsz=args.imgsz, opset=used_opset,
            simplify=not args.skip_simplify, dynamic=False, nms=False,
        )
    except (ValueError, RuntimeError) as e:
        print(f"[WARN] opset={used_opset} 导出失败: {e}，回退 opset=17")
        used_opset = 17
        exported = model.export(
            format="onnx", imgsz=args.imgsz, opset=used_opset,
            simplify=not args.skip_simplify, dynamic=False, nms=False,
        )
    onnx_path = Path(str(exported))
    print(f"      已导出: {onnx_path} (实际 opset={used_opset})")
    return onnx_path, used_opset


def validate_and_collect_io(onnx_path: Path) -> dict:
    """步骤 2 & 3: ONNX 校验并收集静态图 IO 元信息"""
    import onnx

    print("\n[2/6] onnx.checker 校验 ...")
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print("      ✅ 模型结构合法")

    print("\n[3/6] 读取静态图 IO 元信息 ...")
    static_io = {"inputs": [], "outputs": []}
    for inp in onnx_model.graph.input:
        static_io["inputs"].append(
            {
                "name": inp.name,
                "shape": dim_to_list(inp.type.tensor_type.shape.dim),
                "dtype": onnx.TensorProto.DataType.Name(inp.type.tensor_type.elem_type),
            }
        )
    for out in onnx_model.graph.output:
        static_io["outputs"].append(
            {
                "name": out.name,
                "shape": dim_to_list(out.type.tensor_type.shape.dim),
                "dtype": onnx.TensorProto.DataType.Name(out.type.tensor_type.elem_type),
            }
        )
    for i in static_io["inputs"]:
        print(f"      IN  : {i['name']:<20} {i['shape']}  {i['dtype']}")
    for o in static_io["outputs"]:
        print(f"      OUT : {o['name']:<20} {o['shape']}  {o['dtype']}")
    return static_io


def run_dummy_inference(onnx_path: Path, args: argparse.Namespace, nc: int):
    """步骤 4: onnxruntime dummy 推理"""
    import onnxruntime as ort

    print("\n[4/6] onnxruntime dummy 推理 ...")
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    rt_inputs = [
        {"name": i.name, "shape": i.shape, "type": i.type} for i in session.get_inputs()
    ]
    rt_outputs_meta = [
        {"name": o.name, "shape": o.shape, "type": o.type}
        for o in session.get_outputs()
    ]

    input_name = session.get_inputs()[0].name

    # 使用 numpy.random.Generator 并提供固定种子，确保每次运行生成相同的 dummy 输入。
    # 必须是 [0, 1) 均匀分布：真实预处理为 RGB/255，standard_normal(N(0,1)) 含负值
    # 与 >1 的值，属分布外输入，其输出统计/坐标尺度推断不能作为 Java 端依据。
    rng = np.random.default_rng(seed=42)
    dummy = rng.random((1, 3, args.imgsz, args.imgsz), dtype=np.float32)

    outputs = session.run(None, {input_name: dummy})

    print(f"      输入节点 : {input_name}  shape={dummy.shape}  dtype={dummy.dtype}")
    print(f"      输出数量 : {len(outputs)}")

    runtime_outputs = []
    for i, out in enumerate(outputs):
        arr = np.asarray(out)
        meta = detect_layout(tuple(arr.shape), nc)
        scale = probe_coord_scale(arr, args.imgsz)
        finite = arr[np.isfinite(arr)]
        rec = {
            "index": i,
            "name": rt_outputs_meta[i]["name"]
            if i < len(rt_outputs_meta)
            else f"output{i}",
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "min": float(finite.min()) if finite.size else None,
            "max": float(finite.max()) if finite.size else None,
            "mean": float(finite.mean()) if finite.size else None,
            "coord_scale_guess": scale,
            "layout_guess": meta["layout"],
            "need_nms_guess": meta["need_nms"],
            "layout_note": meta["note"],
        }
        runtime_outputs.append(rec)
        print(
            f"      输出[{i}]: name={rec['name']}  shape={rec['shape']}  dtype={rec['dtype']}"
        )
        print(
            f"                 min={rec['min']:.4f}  max={rec['max']:.4f}  mean={rec['mean']:.4f}"
        )
        print(
            f"                 布局推断: {rec['layout_guess']}  (need_nms={rec['need_nms_guess']})"
        )
        print(f"                 坐标尺度: {rec['coord_scale_guess']}")

    return session, rt_inputs, runtime_outputs


def run_real_image_validation(
    session, input_name: str, args: argparse.Namespace, nc: int
) -> dict:
    """步骤 5: 真实图片推理验证。

    与步骤 4 的 dummy 不同，这里走的是完整真实预处理，因此输出的
    坐标尺度 / 布局推断才是 manifest 契约与 Java 端的权威依据。
    """
    print("\n[5/6] 真实图片推理验证 ...")
    real_check: dict = {"done": False, "note": "", "coord_scale": None,
                        "layout": None, "shape": None}
    try:
        import cv2

        val_dir = PROJECT_ROOT / "data" / "tt100k48_yolo" / "images" / "val"
        imgs = sorted(val_dir.glob("*.jpg"))[:1] if val_dir.is_dir() else []
        if not imgs:
            real_check["note"] = "未找到验证集图片，跳过"
            print(f"      {real_check['note']}")
            return real_check

        img = cv2.imread(str(imgs[0]))
        if img is None:
            real_check["note"] = "无法读取验证图片"
            return real_check

        from ultralytics.data.augment import LetterBox

        # scaleup=False / INTER_LINEAR 与 Ultralytics val、predict 的 LetterBox 一致
        lb = LetterBox(
            new_shape=(args.imgsz, args.imgsz), auto=False, stride=32,
            scaleup=False, interpolation=cv2.INTER_LINEAR,
        )
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        padded = lb(image=rgb)
        # LetterBox.__call__ 的返回标注为 dict[str, Any] | np.ndarray；
        # 此处按实际返回 ndarray 的分支收窄，避免类型检查器误判为 dict。
        if not isinstance(padded, np.ndarray):
            raise TypeError(
                f"LetterBox 返回 {type(padded).__name__}，预期 np.ndarray"
            )
        x = np.ascontiguousarray(
            padded.transpose(2, 0, 1)[None], dtype=np.float32
        ) / 255.0
        out = session.run(None, {input_name: x})[0]
        arr = np.asarray(out)
        layout = detect_layout(tuple(arr.shape), nc)
        coord_scale = probe_coord_scale(arr, args.imgsz)
        real_check.update(
            shape=list(arr.shape), layout=layout["layout"],
            need_nms=layout["need_nms"], coord_scale=coord_scale,
        )
        print(f"      测试图: {imgs[0].name}")
        print(f"      输出 shape: {arr.shape}")
        print(f"      布局(权威): {layout['layout']}  坐标尺度(权威): {coord_scale}")

        if arr.ndim == 3 and arr.shape[-1] == 6:
            rows = arr[0]
            valid = rows[rows[..., 4] > 0.25]
            print(f"      score>0.25 的框数: {len(valid)}")
            if len(valid):
                print(f"      样例: {np.round(valid[0], 3).tolist()}")
            real_check.update(done=True, note=f"{len(valid)} 个框 score>0.25")
        else:
            real_check.update(done=True, note="非 6 列布局，仅确认可推理")

    # 仅捕获可预期的特定异常，消除 blind exception 警告
    except (ImportError, OSError, RuntimeError, ValueError, TypeError) as e:
        real_check["note"] = f"真实图验证失败（不影响导出）: {e}"
        print(f"      [WARN] {real_check['note']}")
    return real_check


def build_output_contract(
    runtime_outputs: list[dict], real_check: dict, args: argparse.Namespace, nc: int
) -> dict:
    """把「实测输出格式」固化为机器可读契约，写入 model_manifest.json。

    Java 端据此校验布局与坐标空间，不再靠 shape / 最大绝对值启发式猜测。
    优先采用真实图片推理的结果；真实图不可用时才回退到 dummy。
    """
    out = runtime_outputs[0] if runtime_outputs else {}
    layout = real_check.get("layout") or out.get("layout_guess") or "unknown"
    scale = real_check.get("coord_scale") or out.get("coord_scale_guess") or "unknown"
    shape = real_check.get("shape") or out.get("shape")

    if scale == "normalized(0~1)":
        coordinate_space = "normalized_0_1"
    elif str(scale).startswith("pixel"):
        coordinate_space = "pixel_imgsz"
    else:
        coordinate_space = "unknown"

    columns = END2END_COLUMNS if layout == "end2end_nms_free" else []
    need_nms = layout == "raw_head_transposed"

    return {
        "output_layout": layout,
        "coordinate_space": coordinate_space,
        "columns": columns,
        "output_shape": shape,
        "need_nms": need_nms,
        "imgsz": args.imgsz,
        "num_classes": nc,
        "preprocess": PREPROCESS_CONTRACT,
        "contract_source": "real_image" if real_check.get("done") else "dummy_fallback",
        "contract_note": (
            "输出已端到端去重, Java 端禁止再做 NMS; 仅跨切片合并时允许去重。"
            if layout == "end2end_nms_free" else "按 onnx_output_format.txt 手工核对解析分支。"
        ),
    }


def generate_reports(report: dict, args: argparse.Namespace):
    """步骤 6: 写出报告"""
    print("\n[6/6] 写出 onnx_output_format.txt / .json ...")
    json_path = SCRIPT_DIR / "onnx_output_format.json"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    L = []
    A = L.append
    A("=" * 68)
    A("YOLO26s ONNX 输出格式说明（由 export_onnx.py 自动生成）")
    A("=" * 68)
    A("")
    A(f"源权重    : {report['source_weights']}")
    A(f"ONNX 文件 : {report['onnx_path']}")
    A(
        f"opset={report['opset']}  imgsz={report['imgsz']}  dynamic=False  simplify={not args.skip_simplify}"
    )
    A(f"类别数    : {report['num_classes']}")
    A("")
    A("── 输入 ──")
    for i in report["runtime_inputs"]:
        A(f"  name : {i['name']}")
        A(f"  shape: {i['shape']}")
        A(f"  type : {i['type']}")
    A("")
    A("── 输出 (dummy 仅用于确认 shape；坐标尺度与布局以『真实图片验证』为准)──")
    for r in report["runtime_outputs"]:
        A(f"  输出[{r['index']}] name : {r['name']}")
        A(f"            shape: {r['shape']}")
        A(f"            dtype: {r['dtype']}")
        A(f"            min={r['min']}  max={r['max']}  mean={r['mean']}")
        A(f"            布局推断: {r['layout_guess']}  need_nms={r['need_nms_guess']}")
        A(f"            坐标尺度: {r['coord_scale_guess']}")
        A(f"            说明: {r['layout_note']}")
        A("")
    A("── 真实图片验证（坐标尺度 / 布局的权威来源）──")
    rc = report["real_image_check"]
    A(f"  {rc.get('note', '')}")
    A(f"  实测布局     : {rc.get('layout')}")
    A(f"  实测坐标尺度 : {rc.get('coord_scale')}")
    A(f"  实测 shape   : {rc.get('shape')}")
    A("")
    c = report["output_contract"]
    A("── 输出契约（写入 model_manifest.json, Java 端据此校验，不再猜测）──")
    A(f"  output_layout : {c['output_layout']}")
    A(f"  coordinate_space : {c['coordinate_space']}")
    A(f"  columns       : {c['columns']}")
    A(f"  need_nms      : {c['need_nms']}")
    A(f"  contract_from : {c['contract_source']}")
    A(f"  预处理        : {c['preprocess']}")
    A("")
    A("=" * 68)
    A("⚠️ 给 Java 端的重要提醒")
    A("=" * 68)
    A("1. YOLO26 是 NMS-free 架构。若上方输出 shape 为 (1, 300, 6),")
    A("   则每行 = [x1, y1, x2, y2, score, class_id], Java 端【禁止】再做 NMS,")
    A("   【禁止】照搬 YOLOv8 的 (1,84,8400) 解析代码。")
    A("2. 若上方输出 shape 为 (1, 4+nc, 8400)，则为原始头输出，需转置后自写 NMS。")
    A("3. 框坐标尺度以『坐标尺度』一行为准: pixel 则需除以 640 再乘原图宽高；")
    A("   normalized 则直接乘原图宽高。")
    A("4. 由于预处理使用 letterbox (保持长宽比 + 灰边填充), Java 端必须把框坐标")
    A("   从 640 * 640 的 letterbox 空间反变换回原图空间 (减去 pad、除以 scale)。")
    A("5. 类别 ID 直接索引 names 文件 (--names) 的行号，顺序必须与训练时一致。")
    A("")
    A("── 类别顺序（行号即类别 ID, 供 Java ClassNames 核对）──")
    for i, n in enumerate(report["class_names"]):
        A(f"  {i:>2}: {n}")
    A("")
    A("=" * 68)

    txt_path = SCRIPT_DIR / "onnx_output_format.txt"
    txt_path.write_text("\n".join(L), encoding="utf-8")
    print(f"      ✅ {txt_path.resolve()}")
    print(f"      ✅ {json_path.resolve()}")


def copy_to_java_resources(onnx_path: Path, dst: Path, report: dict) -> None:
    """步骤 7: 拷贝到 Java 资源目录，并写出版本清单"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        DEPLOY_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dst, DEPLOY_BACKUP_DIR / (dst.name + ".bak"))
    shutil.copy2(onnx_path, dst)

    contract = report["output_contract"]
    manifest = {
        "model_sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
        "opset": report["opset"],
        "imgsz": report["imgsz"],
        "num_classes": report["num_classes"],
        "class_names_sha256_16": hashlib.sha256(
            "\n".join(report["class_names"]).encode("utf-8")
        ).hexdigest()[:16],
        "source_weights": report["source_weights"],
        "output_layout": contract["output_layout"],
        "coordinate_space": contract["coordinate_space"],
        "columns": contract["columns"],
        "output_shape": contract["output_shape"],
        "need_nms": contract["need_nms"],
        "preprocess": contract["preprocess"],
        "contract_source": contract["contract_source"],
    }
    (dst.parent / "model_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n[COPY] 已拷贝: {dst.resolve()}（旧文件备份于 {DEPLOY_BACKUP_DIR}）")


# ==================== 主流程 ====================
def main() -> None:
    args = parse_args()
    # 统一解析为绝对路径：报告中记录的源权重路径不应随调用时的工作目录变化
    weights = Path(args.weights).resolve()

    if not weights.is_file():
        sys.exit(f"[FAIL] 权重不存在: {weights}\n       请先执行 train.py 完成训练")

    names = load_names(Path(args.names))
    nc = len(names)

    print("=" * 68)
    print(" TrafficSignDetector — ONNX 导出与验证")
    print("=" * 68)
    print(f"  权重   : {weights}")
    print(f"  类别数 : {nc}（来自 {args.names}）")
    print(f"  请求 opset: {args.opset}   imgsz: {args.imgsz}   dynamic: False")

    # 线性编排各个步骤
    onnx_path, used_opset = export_model(args, weights)
    static_io = validate_and_collect_io(onnx_path)
    session, rt_inputs, runtime_outputs = run_dummy_inference(onnx_path, args, nc)
    real_check = run_real_image_validation(session, rt_inputs[0]["name"], args, nc)
    contract = build_output_contract(runtime_outputs, real_check, args, nc)
    print(f"      输出契约: {contract['output_layout']} / {contract['coordinate_space']}"
          f" ({contract['contract_source']})")

    # 组装报告数据
    report = {
        "source_weights": str(weights),
        "onnx_path": str(onnx_path),
        "opset": used_opset,
        "imgsz": args.imgsz,
        "dynamic": False,
        "num_classes": nc,
        "class_names": names,
        "runtime_inputs": rt_inputs,
        "runtime_outputs": runtime_outputs,
        "static_io": static_io,
        "real_image_check": real_check,
        "output_contract": contract,
    }

    generate_reports(report, args)

    if not args.no_copy:
        if args.dst == JAVA_MODEL_DST:
            java_names = JAVA_NAMES_DST
            if java_names.is_file():
                java_nc = len(
                    [l for l in java_names.read_text(encoding="utf-8").splitlines() if l.strip()]
                )
                if java_nc != nc:
                    sys.exit(
                        f"[FAIL] Java 端 names 行数 {java_nc} 与模型类别数 {nc} 不一致，"
                        "已中止覆盖。\n"
                        f"       先同步：uv run python convert_tt100k.py ... "
                        f"--emit-names-to {java_names}\n"
                        "       或改用 --no-copy 仅导出到 runs/ 不覆盖 Java 资源。"
                    )
        copy_to_java_resources(onnx_path, args.dst, report)

    print("\n" + "=" * 68)
    print(" 导出完成 ✅")
    print(" 下一步: 按 onnx_output_format.txt 的实测 shape 编写/调整")
    print("         Java 端 PostProcessor.java (第五部分第 7 个文件)")
    print("=" * 68)


if __name__ == "__main__":
    main()
