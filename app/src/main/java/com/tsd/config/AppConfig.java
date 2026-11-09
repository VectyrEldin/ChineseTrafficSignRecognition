package com.tsd.config;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * 全局配置：路径、模型参数、平台判定。
 * 所有文件路径统一使用 java.nio.file.Path。
 */
public final class AppConfig {

    private AppConfig() {}

    // ==================== 应用信息 ====================
    public static final String APP_TITLE = "交通标志检测系统 — YOLO26s / TT100K";
    public static final double WINDOW_WIDTH  = 1280;
    public static final double WINDOW_HEIGHT = 800;

    public static final String CSS_PATH    = "/css/style.css";
    public static final String NAMES_PATH  = "/config/tt100k48.names";

    // ==================== 推理参数 ====================
    /** 输出契约：由 export_onnx.py 写入 model_manifest.json，启动时读取 */
    public static final ModelContract MODEL_CONTRACT = ModelContract.load();
    /** 网络输入尺寸：由契约驱动（导出 imgsz），契约缺失时回退 640 */
    public static final int INPUT_SIZE = MODEL_CONTRACT.imgsz();
    /** letterbox 填充灰度值：由契约驱动（导出 preprocess.pad_value），缺失时 114 */
    public static final float PAD_GRAY = (float) MODEL_CONTRACT.padValue();
    /**
     * 切片推理开关：-Dtsd.tiled=true 启用（免重训提升小目标有效分辨率）。
     * 默认关闭，保持既有单图推理流程与耗时不变。
     */
    public static final boolean TILED_INFERENCE = Boolean.getBoolean("tsd.tiled");
    /** 切片网格边长（2 = 2x2 共 4 块 + 1 张全局图） */
    public static final int TILE_GRID = 2;
    /** 相邻切片的重叠比例，避免标志被切在边界上 */
    public static final float TILE_OVERLAP = 0.20f;
    /** 默认置信度阈值 */
    public static final float DEFAULT_CONF_THRESHOLD = 0.25f;
    /** IOU 阈值（仅在原始头输出需要自写 NMS 时使用） */
    public static final float IOU_THRESHOLD = 0.60f;
    /** 单张图最多保留的检测框（端到端输出通常已 ≤300） */
    public static final int MAX_DETECTIONS = 300;
    /** 过小的框直接丢弃（相对原图短边比例），抑制噪声 */
    public static final float MIN_BOX_RATIO = 0.002f;

    // ==================== app.dir ====================
    /**
     * 应用根目录：优先取 -Dapp.dir（build.gradle 的 run 任务注入），
     * 其次回退到 user.dir。OpenCV native 库按此定位。
     */
    public static Path appDir() {
        String dir = System.getProperty("app.dir");
        Path candidate = (dir != null && !dir.isBlank()) ? Paths.get(dir) : null;
        if (candidate != null && Files.isDirectory(candidate)) {
            return candidate.toAbsolutePath().normalize();
        }
        return Paths.get(System.getProperty("user.dir", ".")).toAbsolutePath().normalize();
    }

    /** libs 目录：app.dir/libs */
    public static Path libsDir() {
        return appDir().resolve("libs");
    }

    // ==================== 平台判定 ====================
    public enum Platform {
        WIN_X64("win64", "opencv_java4140.dll"),
        LINUX_X64("linux64", "libopencv_java4140.so"),
        MAC_X64("mac64", "libopencv_java4140.dylib"),
        MAC_ARM64("mac-arm64", "libopencv_java4140.dylib");

        public final String dir;
        public final String libName;

        Platform(String dir, String libName) {
            this.dir = dir;
            this.libName = libName;
        }
    }

    /** 依据 os.name + os.arch 判定当前平台 */
    public static Platform currentPlatform() {
        String os = System.getProperty("os.name", "").toLowerCase();
        String arch = System.getProperty("os.arch", "").toLowerCase();
        if (os.contains("win")) {
            return Platform.WIN_X64;
        }
        if (os.contains("mac") || os.contains("darwin")) {
            return (arch.contains("aarch64") || arch.contains("arm")) ? Platform.MAC_ARM64 : Platform.MAC_X64;
        }
        return Platform.LINUX_X64;   // linux / 其他 Unix 一律按 linux-x64
    }

    /** native 库在 app.dir 下的绝对路径，如 libs/win64/opencv_java4140.dll */
    public static Path nativeLibPath() {
        Platform p = currentPlatform();
        return libsDir().resolve(p.dir).resolve(p.libName);
    }

    /** native 库在 classpath 资源中的路径，如 /native/win64/opencv_java4140.dll */
    public static String nativeResourcePath() {
        Platform p = currentPlatform();
        return "/native/" + p.dir + "/" + p.libName;
    }
}