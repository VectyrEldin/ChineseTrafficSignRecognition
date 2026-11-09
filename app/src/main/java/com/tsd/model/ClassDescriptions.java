package com.tsd.model;

import com.tsd.config.AppConfig;

import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Properties;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * 交通标志类别代号 → 中文描述。
 * <p>
 * 加载策略与 {@link ClassNames} 一致：优先外部文件（便于打包后替换），
 * 其次 classpath 资源。描述表属于辅助展示信息，加载失败不阻断应用启动，
 * 此时 {@link #descriptionOf(String)} 回退为原始类别代号。
 */
public final class ClassDescriptions {

    private static final Logger LOGGER = Logger.getLogger(ClassDescriptions.class.getName());

    private static final String RESOURCE_PATH = "/config/labels_cn.properties";
    private static final String EXTERNAL_PATH = "config/labels_cn.properties";

    private final Properties props;

    private ClassDescriptions(Properties props) {
        this.props = props;
    }

    /** 加载描述表；任何 I/O 失败都返回空表，由调用方回退显示 */
    public static ClassDescriptions load() {
        Path ext = AppConfig.appDir().resolve(EXTERNAL_PATH);
        if (Files.isRegularFile(ext)) {
            try (Reader r = Files.newBufferedReader(ext, StandardCharsets.UTF_8)) {
                ClassDescriptions d = new ClassDescriptions(read(r));
                LOGGER.info(() -> "[Labels] 中文描述表(外部): " + ext + " 共 " + d.size() + " 条");
                return d;
            } catch (IOException e) {
                LOGGER.log(Level.WARNING, "[Labels] 外部描述表读取失败，回退 classpath", e);
            }
        }

        try (InputStream in = ClassDescriptions.class.getResourceAsStream(RESOURCE_PATH)) {
            if (in != null) {
                try (Reader r = new InputStreamReader(in, StandardCharsets.UTF_8)) {
                    ClassDescriptions d = new ClassDescriptions(read(r));
                    LOGGER.info(() -> "[Labels] 中文描述表(classpath): 共 " + d.size() + " 条");
                    return d;
                }
            }
        } catch (IOException e) {
            LOGGER.log(Level.WARNING, "[Labels] classpath 描述表读取失败", e);
        }

        LOGGER.warning("[Labels] 未找到 " + RESOURCE_PATH + "，中文描述列将显示类别代号");
        return new ClassDescriptions(new Properties());
    }

    /** load(Reader) 按字符读取，UTF-8 由 Reader 保证（避免 load(InputStream) 的 ISO-8859-1 限制） */
    private static Properties read(Reader r) throws IOException {
        Properties p = new Properties();
        p.load(r);
        return p;
    }

    /** 按类别代号取中文描述；未匹配时回退为原代号 */
    public String descriptionOf(String code) {
        if (code == null || code.isBlank()) return "";
        return props.getProperty(code, code);
    }

    public int size() { return props.size(); }
}
