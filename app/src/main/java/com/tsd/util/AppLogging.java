package com.tsd.util;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.logging.FileHandler;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.logging.SimpleFormatter;

/**
 * 应用日志落盘配置。
 * <p>
 * 打包后的 GUI 程序由 javaw 启动、没有控制台，JUL 默认的 ConsoleHandler
 * 输出到 stderr 会被丢弃，导致现场问题无法排障。这里向 root logger 追加一个
 * FileHandler，并保留 ConsoleHandler（开发态仍可在 IDE / 终端看到日志）。
 * <p>
 * 日志目录放在用户主目录而非 app.dir：per-machine 安装时 app.dir 位于
 * Program Files，进程通常没有写权限。这与 OpenCVLoader、OrtInference
 * 已有的用户目录策略一致。
 */
public final class AppLogging {

    private static final Logger LOGGER = Logger.getLogger(AppLogging.class.getName());

    private static final String LOG_DIR = ".tsd-logs";
    private static final int MAX_BYTES = 2 * 1024 * 1024;   // 单个文件 2 MB
    private static final int MAX_FILES = 3;                 // 最多 3 个滚动文件

    private static boolean initialized = false;

    private AppLogging() {}

    /** 幂等；任何失败都不阻断应用启动 */
    public static synchronized void init() {
        if (initialized) return;
        try {
            Path dir = Paths.get(System.getProperty("user.home", "."), LOG_DIR);
            Files.createDirectories(dir);

            // %g 为滚动编号，放在扩展名之前，保证产物是 tsd0.log / tsd1.log
            FileHandler fh = new FileHandler(
                    dir.resolve("tsd%g.log").toString(), MAX_BYTES, MAX_FILES, true);
            fh.setEncoding("UTF-8");
            fh.setLevel(Level.INFO);
            fh.setFormatter(new SimpleFormatter());

            Logger.getLogger("").addHandler(fh);
            initialized = true;
            LOGGER.info(() -> "[AppLogging] 日志目录: " + dir.toAbsolutePath());
        } catch (IOException | RuntimeException e) {
            LOGGER.log(Level.SEVERE, "[AppLogging] 日志初始化失败（不影响启动）", e);
        }
    }
}
