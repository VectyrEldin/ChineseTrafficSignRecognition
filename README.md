# TrafficSignRecognition 交通标志识别

基于 **YOLO26s** 的中国交通标志检测与识别系统：JavaFX 桌面端（ONNX Runtime 推理）+ Python 训练/导出管线（Ultralytics）。在 TT100K 数据集上以 **48 类**方案微调，NMS-free 端到端输出。

![License](https://img.shields.io/badge/license-AGPL--3.0-blue)
![Java](https://img.shields.io/badge/Java-25-orange)
![Python](https://img.shields.io/badge/Python-3.14-green)

## 功能特性

- **图片检测 / 摄像头实时检测 / 拍照检测**，检测结果表格展示类别、中文描述、置信度与位置
- 置信度阈值实时调节（0.05–0.90，默认 0.25）、标签显示开关
- **48 类中国交通标志**（完整类别见 [tt100k48.names](app/src/main/resources/config/tt100k48.names)）
- **NMS-free 端到端推理**：ONNX 输出 `(1, 300, 6)`，Java 端不做 NMS
- **契约驱动部署**：启动时按 `model_manifest.json` 校验模型 SHA-256、类别指纹、预处理与坐标空间，失配即拒绝推理
- 可选切片推理（启动加 `-Dtsd.tiled=true`）：全局图 + 2×2 重叠切片提升小目标召回，代价为 5 倍推理耗时，适合离线单图分析
- Windows 一键打包（jpackage 应用镜像 / exe 安装包）

## 项目结构

```
├── app/                JavaFX 桌面端（Java 25 / Gradle）
│   └── src/main/
│       ├── java/com/tsd/   core（预处理/推理/后处理/切片）、ui、config、model
│       └── resources/      fxml、config（names + 中文描述）、models（manifest）
├── scripts/            Python 训练/导出/评测脚本（uv 管理，详见 scripts/README.md）
├── data/               数据集（不入库，自行下载，见下文）
├── libs/               OpenCV jar + native dll（不入库，见下文）
├── package/icon/       打包图标
├── package_win.ps1     jpackage 一键打包
├── verify_env.ps1      环境自检
└── verify_all.ps1      全链路一致性核对（模型 / names / manifest）
```

## 环境要求

**桌面端**

- JDK 25（设置 `JAVA_HOME`；Gradle 由 wrapper 自动下载）
- Windows x64（构建目前只配套 win64 OpenCV native；其他平台需自备对应 native 并调整配置）
- 推理走 ONNX Runtime CPU，**无需 GPU**

**训练（可选）**

- Python 3.14 + [uv](https://docs.astral.sh/uv/)
- NVIDIA GPU（CUDA 版 PyTorch，8 GB 显存可训；实测 RTX 4060 Laptop，默认配置约 2.5–3 小时）

**打包 exe 安装包（可选）**：PowerShell 7 + WiX Toolset

## 快速开始

### 1. 放置大文件（不随仓库分发）

| 文件 | 来源 | 放置位置 |
|---|---|---|
| OpenCV 4.14.0 jar | [opencv.org/releases](https://opencv.org/releases/) 下载 Windows 包并解压 | `build/java/opencv-4140.jar` → `libs/opencv-4140.jar` |
| OpenCV native dll | 同上 | `build/java/x64/opencv_java4140.dll` → `libs/win64/opencv_java4140.dll` |
| `best.onnx`（约 38 MB） | 本仓库 **Releases** 页面 | `app/src/main/resources/models/best.onnx` |
| TT100K 2021（约 18 GB，仅训练需要） | [官网](https://cg.cs.tsinghua.edu.cn/traffic-sign/) 下载 | `data/tt100k_2021/` |

模型 manifest 已随仓库提供，放入 `best.onnx` 后启动时会自动校验 SHA-256 与类别指纹。

### 2. 自检与运行

```powershell
.\verify_env.ps1        # 环境自检（JDK / uv / OpenCV / GPU）
.\gradlew.bat run       # 启动桌面端
.\gradlew.bat test      # 单元测试
.\verify_all.ps1        # 全链路一致性核对
```

### 3. 打包 Windows 应用

```powershell
.\package_win.ps1 app   # 应用镜像（免安装，不需要 WiX）
.\package_win.ps1 exe   # exe 安装包（需 WiX Toolset，脚本头部有一次性配置说明）
```

输出目录：`app/build/package/dist/`。

## 训练与复现（可选）

完整说明见 [scripts/README.md](scripts/README.md)。主链路：

```bash
cd scripts
uv sync

# 1) 数据集转换（TT100K → 48 类 YOLO 格式）
uv run python convert_tt100k.py --annotation ../data/tt100k_2021/annotations_all.json --image-dir ../data/tt100k_2021/train --test-image-dir ../data/tt100k_2021/test --output ../data/tt100k48_yolo --link

# 2) 训练（先冒烟，再全量）
uv run python train.py --smoke
uv run python train.py

# 3) 背景误检择优 + 导出 ONNX（自动同步到 Java 资源目录）+ 对拍验证
uv run python select_best.py --run-dir runs/detect/tt100k48_yolo26s
uv run python export_onnx.py
uv run python verify_parity.py
```

关键设计（NMS-free 输出约定、类别顺序契约、Java↔Python 预处理逐元素对拍、FPPI 择优）均在 [scripts/README.md](scripts/README.md) 中详述。

## 许可证

本项目以 **AGPL-3.0** 发布，见 [LICENSE](LICENSE)。

原因：项目基于 Ultralytics YOLO26s（AGPL-3.0），其许可要求衍生作品（**包括训练所得模型权重**）同样以 AGPL-3.0 分发。

| 第三方组件 | 许可证 |
|---|---|
| Ultralytics YOLO26s（含预训练与微调权重） | AGPL-3.0 |
| TT100K 数据集 | CC BY-NC 2.0（**仅非商业使用**，不随仓库分发） |
| JavaFX | GPLv2 + Classpath Exception |
| ONNX Runtime | MIT |
| OpenCV | Apache-2.0 |

**注意**：AGPL-3.0 的强 copyleft 叠加 TT100K 的非商业限制，意味着本项目整体仅适合学习、研究与非商业用途。商业闭源使用需另行取得 [Ultralytics Enterprise 授权](https://www.ultralytics.com/license)及数据集权利人的许可。

## 致谢

- [TT100K](https://cg.cs.tsinghua.edu.cn/traffic-sign/)：Zhu et al., *Traffic-Sign Detection and Classification in the Wild*, CVPR 2016
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)