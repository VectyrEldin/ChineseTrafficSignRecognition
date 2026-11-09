package com.tsd.core;

import com.tsd.config.AppConfig;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.attribute.PosixFilePermission;
import java.util.Set;

import java.nio.file.StandardCopyOption;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * OpenCV native 库加载器。
 * <p>
 * 三级回退策略（硬性约束 9）：
 *   ① app.dir/libs/&lt;平台&gt;/opencv_java4140.dll   —— 开发与打包部署的主路径（安装包外挂 libs/）
 *   ② java.library.path / System.loadLibrary         —— 系统已装 OpenCV
 *   ③ classpath 资源 /native/&lt;平台&gt;/... 提取到临时目录 —— 保留回退，安装包不再内嵌该资源
 */
public final class OpenCVLoader {

    private static final Logger LOGGER = Logger.getLogger(OpenCVLoader.class.getName());

    private static volatile boolean loaded = false;
    private static volatile String loadedFrom = "(未加载)";

    private OpenCVLoader() {}

    /** 幂等：重复调用无副作用 */
    public static synchronized void init() {
        if (loaded) return;

        // ① app.dir 下的 native
        Path p = AppConfig.nativeLibPath();
        if (Files.isRegularFile(p)) {
            try {
                System.load(p.toAbsolutePath().normalize().toString());
                loaded = true;
                loadedFrom = p.toString();
                return;
            } catch (UnsatisfiedLinkError e) {
                LOGGER.log(Level.WARNING, "[OpenCV] app.dir 加载失败: {0}，尝试回退", e.getMessage());
            }
        } else {
            LOGGER.log(Level.WARNING, "[OpenCV] 未找到: {0}，尝试回退", p);
        }

        // ② 系统库路径
        try {
            System.loadLibrary("opencv_java4140");
            loaded = true;
            loadedFrom = "java.library.path";
            return;
        } catch (UnsatisfiedLinkError e) {
            LOGGER.log(Level.WARNING, "[OpenCV] loadLibrary 失败: {0}，尝试 classpath 提取", e.getMessage());
        }

        // ③ classpath 资源提取到临时目录
        try {
            Path tmp = extractFromClasspath();
            System.load(tmp.toAbsolutePath().toString());
            loaded = true;
            loadedFrom = tmp.toString();
        } catch (IOException | RuntimeException t) {
            throw new IllegalStateException("OpenCV native 库全部加载路径均失败，最后错误: " + t.getMessage(), t);
        }
    }

    /** 从 classpath 资源提取 native 到临时文件（进程退出时尝试删除） */
    private static Path extractFromClasspath() throws IOException {
        String res = AppConfig.nativeResourcePath();
        try (InputStream in = OpenCVLoader.class.getResourceAsStream(res)) {
            if (in == null) {
                throw new IllegalStateException("classpath 中不存在资源: " + res);
            }
            // 在用户主目录下创建私有缓存目录，避开系统公共临时目录
            Path privateDir = Paths.get(System.getProperty("user.home"), ".tsd-opencv-cache");
            Files.createDirectories(privateDir);
            try {
                // 设置目录权限为仅当前用户可访问（POSIX 系统）
                Files.setPosixFilePermissions(privateDir, Set.of(
                        PosixFilePermission.OWNER_READ,
                        PosixFilePermission.OWNER_WRITE,
                        PosixFilePermission.OWNER_EXECUTE
                ));
            } catch (UnsupportedOperationException _) {
                // Windows 不支持 POSIX 权限，依赖默认 ACL 隔离
            }
            // 在私有目录中原子性地创建临时文件
            Path tempFile = Files.createTempFile(privateDir, "lib", AppConfig.currentPlatform().libName);
            tempFile.toFile().deleteOnExit();
            // 覆盖空文件（私有目录中无符号链接风险）
            Files.copy(in, tempFile, StandardCopyOption.REPLACE_EXISTING);
            return tempFile;
        }
    }

    public static boolean isLoaded() {
        return loaded;
    }

    /** 供状态栏显示 */
    public static String status() {
        return loaded
                ? "OpenCV 4.14.0 已加载 ← " + loadedFrom
                : "OpenCV 未加载（解码功能不可用）";
    }
}