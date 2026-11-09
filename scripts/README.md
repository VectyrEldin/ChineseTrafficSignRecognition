# TrafficSignRecognition — 训练与导出脚本

本目录是 JavaFX 桌面端「交通标志识别」项目的 Python 侧工具集，负责 TT100K 数据集转换（**48 类**）、YOLO26s 微调与 ONNX 导出。依赖与环境由 `uv` 管理（`pyproject.toml` + `uv.lock`），项目为**非打包项目**（`[tool.uv] package = false`），只需安装依赖、不构建本项目。

> **配置口径（2026-11-07 更新）**：`train.py` 现为**单一默认配置**（时间效率优先，`imgsz=640 / batch=16 / epochs=100`），已移除旧的 `--profile` / `--set` / `--skip-fppi` / `--deep-select` 等开关。下游 `export_onnx.py` / `verify_parity.py` / `select_best.py` 的默认尺寸同步为 **640**。历史多 profile 设计的说明见文末「历史备注」。

## 环境

- Python 3.14（`.python-version`），`requires-python = ">=3.14"`
- GPU 构建：`torch` / `torchvision` 通过 `[tool.uv.sources]` 绑定到 `pytorch-cu134` 显式索引
- `ultralytics>=8.4.204`：Platform 的 `yolo26s` 要求 `>=8.4.203`
- `faster-coco-eval>=1.6.7`：`save_json=True` 计算 `mAP_small/medium/large(B)` 所需，已写入 `pyproject.toml` / `uv.lock`（当前 `train.py` 默认 `save_json=False`，需要分档指标时用 `eval_by_size.py`）
- 首次使用：`uv sync`；检查锁文件：`uv lock --check`；全量升级依赖：`uv lock --upgrade`

**`PYTHONPATH` 污染排查**：若 `uv run python ...` 报
`No module named 'numpy._core._multiarray_umath'` 或 `OpenCV bindings requires "numpy"`，
是外部 `site-packages` 抢在 venv 之前被导入，不是项目依赖问题。先执行
`echo $PYTHONPATH`（PowerShell：`$env:PYTHONPATH`）确认；临时清空即可：

```bash
PYTHONPATH= uv run python train.py --smoke      # Bash
$env:PYTHONPATH=""; uv run python train.py --smoke   # PowerShell
```

切换 CUDA 版本时，请修改 `pyproject.toml` 中 `[[tool.uv.index]]` 的 `name` 与 `url`（保持 `explicit = true` 与 `[tool.uv.sources]` 一致），然后执行 `uv lock --upgrade-package torch --upgrade-package torchvision`。**不要用 `--index-url` 临时指定**：它会替换默认索引且不写入配置。

## 四条主要命令（默认 48 类方案）

```bash
# 1) 数据集转换（相对 scripts/ 目录；默认 48 类 + other 图库 + 10% 背景图）
uv run python convert_tt100k.py \
    --annotation ../data/tt100k_2021/annotations_all.json \
    --image-dir  ../data/tt100k_2021/train \
    --test-image-dir ../data/tt100k_2021/test \
    --output     ../data/tt100k48_yolo \
    --link

# 2) 训练（单一默认配置：imgsz=640 / batch=16 / 100 轮，约 2.5–3.0 h）
uv run python train.py --smoke          # 1 epoch + 2% 数据冒烟（先跑这个！）
uv run python train.py                  # 全量训练，产物 runs/detect/tt100k48_yolo26s/
uv run python train.py --epochs 70      # 更激进：约 1.9–2.2 h
uv run python train.py --time 2.0       # 硬限时 2 h（到点停止，best.pt 照常保存）
uv run python train.py --batch 8        # 显存被其他程序占用时退回 8
uv run python train.py --resume         # 从 last.pt 断点续训

# 3) 导出 ONNX 并验证（默认 imgsz=640；不带 --no-copy 会同步到 Java 资源目录）
#    如需背景误检择优，先看 runs/detect/tt100k48_yolo26s/deploy_candidate.json
uv run python export_onnx.py --no-copy
uv run python export_onnx.py

# 4) 背景 FPPI 择优（训练后手动执行，比较 best.pt / last.pt）
uv run python select_best.py --run-dir runs/detect/tt100k48_yolo26s
```

`train.py` 可用参数：`--data / --weights / --run-name / --epochs / --patience / --batch / --imgsz / --workers / --device / --cache / --smoke / --resume / --time`。`--smoke` 与 `--resume` 互斥。

`--overwrite`（`convert_tt100k.py`）会清空 `images/` 与 `labels/` 后重建，属不可撤销操作，日常重跑建议换用新的 `--output` 目录。

### 默认行为的构成

- `convert_tt100k.py`：频次前 45 类之后追加 `io` / `po` / `wo`（其他指示 / 禁令 / 警告标志），未收录类别按类别码首字母归并；自动定位 `other` 图库并把带框图并入 `train`；默认 `group` 划分、按 `--background-ratio`（默认 0.10）写入背景图空 label。**前 45 个 class ID 与旧版完全一致**；`--emit-segments` 可改用 9 列 segment 行。
- `train.py`：单一默认配置（详见下节）；`--data` 默认 `data/tt100k48_yolo/tt100k48.yaml`，`--run-name` 默认 `tt100k48_yolo26s`，与旧 45 类训练产物目录分离。默认 `imgsz=640 / batch=16`，下游导出与 parity 默认尺寸同步为 640。
- `export_onnx.py` / `verify_parity.py`：默认 names 与验证图路径指向 48 类产物；`verify_parity.py` 默认权重跟随 `deploy_candidate.json` 的推荐权重（无择优结果时回退 `best.pt`），ONNX 默认取权重同目录同名 `.onnx`。默认 `imgsz` / `--size` 为 640（Java 端 `INPUT_SIZE` 由 manifest 自动适配）。

### 当前默认配置（时间效率优先）

`train.py` 只有一套默认参数，每个偏离官方 `yolo26s.pt train_args` 的取值都在源码对应行注明依据。核心项：

| 项 | 默认值 | 依据 |
|---|---|---|
| `imgsz` / `batch` | 640 / 16 | 640+16 在 8 GB 显存已验证不 OOM；时间优先 |
| `epochs` / `patience` | 100 / 20 | 45 类基线第 100 轮已达 200 轮终值 mAP50 的 99.4%；patience=20 允许平台期早停 |
| `optimizer` / `lr0` / `lrf` | MuSGD / 0.002 / 0.01 | 本数据上实测验证的组合，不做未验证替换 |
| `cos_lr` | True | 100 轮 cosine 终点 LR=2e-5 |
| `fliplr` / `flipud` | 0 / 0 | 镜像保留 class id，对 i2r/i4l、pl* 数值族制造互斥标签 |
| `degrees` / `shear` / `perspective` | 0 | 方向性标志 + 小目标旋转后框膨胀，禁止整图几何变换 |
| `scale` | [0.8, 1.5] | 640 下约四成框 <10 px，缩放增强以放大为主（0.8 倍仍会轻微缩小） |
| `mixup` / `copy_paste` | 0 / 0 | 半透明重影干扰限速数字类；bbox-only 标签无 segments，CopyPaste 不执行 |
| `hsv_v` | 0.3 | 官方 0.194；TT100K 光照/天气变化大取 0.3 |
| `cls_pw` | 0.25（重采样清单自动置 0） | 官方起步值；重采样已做逆频率，避免双重补偿 |
| `mosaic` / `close_mosaic` | 0.99182 / 10 | 官方原值；最后 10 轮关 mosaic 收尾微调 |
| `multi_scale` | 0 | 随机放大会拖慢并抬显存峰值，时间优先关闭 |
| `nms` | False | 训练内 val / best.pt 判定走 NMS-free 头，与 ONNX 导出口径一致 |
| `save_json` / `plots` / `save_period` | False / False / -1 | 关掉一切非必要固定开销 |

`momentum / weight_decay / warmup_* / box / cls / dfl / hsv_h / hsv_s` 均取官方 `yolo26s.pt train_args` 原值（任务无关项）。

**重采样探测**：当 `--data` 的 yaml `train` 指向 `.txt` 清单（`train_balanced.txt`）时，`train.py` 判定数据集已做逆频率重采样，自动把 `cls_pw` 置 0，避免对稀有类二次补偿。

**实测参考**（45 类基线 `results.csv`，640 / batch8 / 200 轮 / 共 5.10 h）：

| 轮数 | mAP50 | 占终值 | 累计耗时 |
|---:|---:|---:|---:|
| 43 | 0.78929 | 95.0% | 1.22 h |
| **100** | **0.83375** | **99.4%** | **2.74 h** |
| 200 | 0.83839 | 100% | 5.10 h |

后 100 轮只换来 +0.0046 mAP50（0.55%）却多花 2.36 h，收益拐点在约 100 轮——这是当前默认 `epochs=100` 的依据。`--time` 可设硬限时（到点停止，`best.pt` 照常保存）。训练后不自动跑 FPPI 择优（固定开销），需要时手动执行 `select_best.py`。

时间预算速查（本机 RTX 4060 Laptop 8GB）：冒烟 `--smoke` 约 1–2 min → 默认 640/100 约 2.5–3.0 h → `--epochs 70` 约 1.9–2.2 h。

### 模型选择：把背景误检纳入

`val` / `test` 分桶都是 0 背景图，mAP 完全不度量误检；实测旧模型 300 张背景图 9.7% 图像级误检、100 张未收录类图 61% 被高分检出。训练结束后手动执行 `select_best.py`：比较 `best.pt` 与 `last.pt`，按 `score = mAP50 - fppi_weight * FPPI` 择优（默认权重 1.0），输出 `<run-dir>/deploy_candidate.json`。mAP50 用 Ultralytics val 默认口径（conf=0.001）并固定 `nms=False`，FPPI 用部署阈值 conf=0.25 在纯背景图上测得，两者不混用。`eval_by_size.py` / `verify_parity.py` 的推理也已统一到 NMS-free 头。

```bash
uv run python select_best.py --run-dir runs/detect/tt100k48_yolo26s
uv run python select_best.py --run-dir runs/detect/tt100k48_yolo26s --all-checkpoints
uv run python export_onnx.py --weights <recommended_weights>
```

脚本不改写 `best.pt`、不触碰已部署 `best.onnx`。`--all-checkpoints` 需要训练时留有周期 checkpoint（当前默认 `save_period=-1` 不产出周期 checkpoint，默认候选只有 best/last）。

### 旧 45 类产物

`convert_tt100k.py` 已移除 `--legacy`，不再支持重新生成 45 类数据集。旧数据集 `data/tt100k45_yolo/`（文件名仍为 `tt100k45.*`）与旧权重 `runs/detect/tt100k45_yolo26s/` 保留，可显式指定使用：

```bash
uv run python train.py --data ../data/tt100k45_yolo/tt100k45.yaml \
    --run-name tt100k45_yolo26s_repro
uv run python export_onnx.py --no-copy --imgsz 640 \
    --weights runs/detect/tt100k45_yolo26s/weights/best.pt \
    --names ../data/tt100k45_yolo/tt100k45.names
```

`--superclass` 参数保留仅为兼容旧命令，48 类方案已默认启用。

### Java 端同步（重要）

当前部署的是 **48 类**模型（FPPI 择优推荐 `last.pt` 导出），`app/src/main/resources/config/tt100k48.names` 为 48 行，与 `model_manifest.json`（`num_classes=48`）一致。Java 启动时会把模型 SHA-256 与类别指纹与 manifest 比对，失配即告警。

`export_onnx.py` 在覆盖 Java 资源前会校验 names 行数与模型类别数，不一致即中止并提示。确认要切换时，先同步 names 再导出：

```bash
cp ../data/tt100k48_yolo/tt100k48.names ../app/src/main/resources/config/tt100k48.names
uv run python export_onnx.py --weights <recommended_weights>
```

同步后 `class_order.json` 指纹改变，需与 `model_manifest.json` 重新核对（根目录 `verify_all.ps1` 可自动完成全链路核对）。重训不可使用 `--resume`（旧 `last.pt` 的 nc 已失效）。覆盖部署前的旧文件备份保存在项目根 `backup/deployed/`（不再放 resources 内，避免被打进安装包），回滚时拷回原位即可。

## 类别顺序契约（关键）

类别 ID 是 `.names` 文件的**行号**，类别数由该文件行数决定（代码不硬编码类别数），以下四处必须完全同源：

1. `data/tt100k48_yolo/tt100k48.names`（默认；旧 45 类数据集在 `data/tt100k45_yolo/`，文件名仍为 `tt100k45.*`）
2. `data/tt100k48_yolo/tt100k48.yaml` 的 `names` 列表
3. label 文件首列的 class id
4. `app/src/main/resources/config/tt100k48.names`（Java 端 `ClassNames`；随已部署模型切换）

`data/tt100k48_yolo/tt100k48_class_order.json` 记录类别顺序的 `sha256_16` 指纹；`export_onnx.py` 写出的 `app/src/main/resources/models/model_manifest.json` 含模型 sha256、实际 opset、类别指纹，部署前请核对两者一致。训练脚本在启动前会做 `names` 与 `yaml` 的交叉校验，失败即中止。

## ONNX 输出约定

YOLO26 为 NMS-free，实测输出 `(1, 300, 6)`，每行 `[x1, y1, x2, y2, score, class_id]`，坐标为 imgsz×imgsz letterbox 空间的像素值（当前默认 640）。Java 端**禁止再做 NMS**，并需把坐标从 letterbox 空间反变换回原图空间。唯一的权威依据是 `onnx_output_format.txt` / `.json` 的实测结果。

导出时若 `opset=12` 失败会自动回退到 17，报告与 manifest 记录的是**实际生效值**，不要假设一定是 12。

## 部署一致性契约（已默认生效）

- `export_onnx.py`：dummy 输入为 `[0,1)` 均匀分布（与真实预处理同分布）；坐标尺度与布局以**真实图片推理**为权威；`model_manifest.json` 含 `output_layout` / `coordinate_space` / `columns` / `need_nms` / `preprocess` 契约。
- Java `ModelContract` 启动时读取 manifest 并校验布局、坐标空间、类别数与插值，不再靠 shape 与「最大绝对值 ≤2」启发式猜测（不一致直接拒绝推理）。
- `YoloPreprocessor.java` 与 `verify_parity.py` 统一为 `INTER_LINEAR`，缩放取整为 round-to-even（Java `Math.rint` ↔ Python `round`）；`verify_parity.py` 做**输入张量逐元素对拍**（`input_parity.max_abs_diff` 应为 0）。
- `AppConfig.INPUT_SIZE` / `PAD_GRAY` 由 manifest 驱动，换 imgsz 重训后 Java 自动适配。
- **检测头统一**：`train.py` 训练内 val / 独立 val、`eval_by_size.py`、`select_best.py`、`verify_parity.py` 的 PyTorch 基准全部固定 `nms=False`，与导出的 NMS-free `(1,300,6)` 头一致。与旧 45 类基线的 one-to-many 头数字不可直接比较，重建基线须同口径重测。
- 数据集 yaml 不写 `path:`：Ultralytics 以 yaml 所在目录为根，从任意工作目录运行都正确。

## 评估校准（不改动权重）

```bash
uv run python eval_by_size.py \
    --weights runs/detect/tt100k48_yolo26s/weights/last.pt \
    --data ../data/tt100k48_yolo/tt100k48.yaml --split test \
    --bg-dir ../data/tt100k_2021/other \
    --annotation ../data/tt100k_2021/annotations_all.json
```

输出 `eval_by_size.json`：论文 Table 3 口径的尺寸分档 recall/precision（≤32 / 32–96 / >96 px，并给出每档在 imgsz 下的等效边长）、逐类 recall、纯背景图 FPPI 与图像级误检率、水平翻转 top-1 类别不一致率。分档指标是 IoU@0.5 微平均，不是 COCO mAP；聚合 mAP 仍以 `model.val()` 为准。默认 `--imgsz 640`。

**数据侧可选开关**（`convert_tt100k.py`）：

- `--split-mode group` 为默认：把相邻 16 个 id 视为同一拍摄序列后整体划分，避免连续帧近重复泄漏；复现旧数据集用 `--split-mode hash`。
- `--background-ratio` 默认 0.10：背景误检是部署痛点，加入纯背景负样本；需要更多负样本时可显式调高（如 0.30）。
- `--emit-segments`：把 5 列 bbox 行改写成 9 列 segment 行 `cls x1 y1 x2 y1 x2 y2 x1 y2`（bbox 四角矩形近似；源标注只有 bbox，无 pixel mask）。
- `--class-balanced`：生成 `train_balanced.txt` 并把 yaml 的 `train` 指向它，按逆频率平方根重复含稀有类的图片（`--max-repeat` 默认 3）；`train.py` 检测到后自动把 `cls_pw` 置 0，避免双重补偿。

**免重训的精度选项**：Java 启动加 `-Dtsd.tiled=true` 启用 `TiledDetector`：全局图 + 2×2 重叠切片，每块独立缩放到 imgsz，等效提高小目标分辨率，权重/ONNX/names 全部不变。代价是 5 次推理，适合离线单图分析，不适合摄像头实时流；默认关闭。

## 版本控制

`runs/`、`*.pt`、`*.onnx`、`train.log`、`onnx_output_format.*`、`.venv/` 均已在 `.gitignore` 中排除，请勿提交。`uv.lock`、`.python-version`、`pyproject.toml` 与全部脚本应纳入版本控制。`eval_baseline.json`、`eval_optimized.json`、`eval_by_size.json` 已加入 `.gitignore`；`deploy_candidate.json` 默认写在 `runs/` 下，同样已被忽略。

## 代码检查

```bash
uv run ruff check .
uv run ruff format --check .
```

## 编排脚本（默认 dry-run）

两个编排脚本均已适配当前 `train.py` 的单一默认配置，不再使用已移除的 `--profile` / `--set`。默认只打印完整计划，不写文件、不启动训练；加 `--execute` 才真正执行。

### run_optimized.py —— 论文对照优化一键链路

```bash
uv run python run_optimized.py                  # dry-run，展开全部命令
uv run python run_optimized.py --imgsz 640      # 对照 640 口径
uv run python run_optimized.py --execute        # 真正执行（960 重训，耗时数小时）
```

9 步链路：① `convert_tt100k.py`（group 划分 + 类感知重采样，输出独立目录 `data/tt100k48_yolo_opt`，加 `--force` 覆盖重建）→ ② 现有 48 类模型基线评测（尺寸分档 / FPPI / 翻转）→ ③ `train.py` 重训（`--imgsz` / `--batch` / `--epochs` 透传；960 下默认 batch=4，8GB 显存安全值）→ ④ `select_best.py` 背景 FPPI 复合择优 → ⑤ 同步 Java names（先备份 `.bak`）→ ⑥ 按 `deploy_candidate.json` 推荐权重导出 ONNX → ⑦ 端到端 parity → ⑧ Gradle 编译 + 单测 → ⑨ 新权重同口径复测。`--segments` 生成 9 列 segment 标签使 copy_paste 可用。

### ablation.py —— 单变量短训消融

```bash
uv run python ablation.py                                   # dry-run
uv run python ablation.py --groups baseline,lr0_001         # 只跑指定组
uv run python ablation.py --epochs 30 --execute             # 真正执行
```

直接 `import train` 复用 `build_train_kwargs` 的权威默认配置，每组只在最终 kwargs 上覆盖一个变量，进程内训练并在组间清理 GPU 显存。标准 5 组：`baseline` / `cls_pw_050` / `lr0_001` / `mosaic_050` / `geometry`（论文 §4.2 视角增强，刻意偏离默认的禁止几何变换）。可选组：`--tile-size N` 追加切片训练组；`--segments-data PATH` 追加 `polygon_copy_paste` 组（数据须来自 `convert_tt100k.py --emit-segments`）。默认 640 / 30 轮短训，统一在末尾用 `eval_by_size.py` 同口径评测（small 档 recall/precision + 背景 FPPI），汇总写 `runs/ablation_summary.json`。

## 历史备注

- 旧的多 profile 设计（`tuned` / `fast` / `constrained` / `optimized`，960/4 默认、`--set` 单变量覆盖、训练后自动 FPPI 择优、`--deep-select`）已从 `train.py` 移除，改为上文单一默认配置。此备注仅用于解释旧产物目录（`runs/detect/*_constrained/`、`*_opt/` 等）的来历。