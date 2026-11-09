package com.tsd.model;

import com.tsd.config.AppConfig;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.List;
import static java.util.logging.Level.WARNING;

/**
 * 类别名称表：行号即类别 ID，类别数由本表行数决定。
 * <p>
 * 顺序必须与训练时 yaml 的 names 完全一致，
 * 因此这里额外算出 sha256[:16] 指纹，加载时与 model_manifest.json 比对，
 * 并与 Python 端 class_order.json 的 sha256_16 同源。
 */
public final class ClassNames {

    private static final java.util.logging.Logger LOGGER =
            java.util.logging.Logger.getLogger(ClassNames.class.getName());

    private final List<String> names;
    private final String fingerprint;

    private ClassNames(List<String> names) {
        this.names = List.copyOf(names);
        this.fingerprint = sha256Hex16(String.join("\n", this.names));
    }

    /** 加载：优先外部文件（便于打包后替换），其次 classpath 资源 */
    public static ClassNames load() throws IOException {
        Path ext = AppConfig.appDir().resolve("config").resolve("tt100k48.names");
        ClassNames cn;
        if (Files.isRegularFile(ext)) {
            cn = new ClassNames(readLines(Files.newBufferedReader(ext, StandardCharsets.UTF_8)));
        } else {
            try (InputStream in = ClassNames.class.getResourceAsStream(AppConfig.NAMES_PATH)) {
                if (in == null) {
                    throw new IOException("找不到类别文件。请确认存在 "
                            + AppConfig.NAMES_PATH + " 且已由 convert_tt100k.py 生成。");
                }
                cn = new ClassNames(readLines(new BufferedReader(
                        new InputStreamReader(in, StandardCharsets.UTF_8))));
            }
        }
        verifyAgainstContract(cn);
        return cn;
    }

    /** 替换外挂 names 后若与已部署模型失配，类别名会静默错位，必须在启动时告警 */
    private static void verifyAgainstContract(ClassNames cn) {
        if (!AppConfig.MODEL_CONTRACT.present()) return;

        String expected = AppConfig.MODEL_CONTRACT.classNamesSha25616();
        if (expected != null && !expected.equals(cn.fingerprint)) {
            // 使用内置格式化：{0} 和 {1} 为占位符，通过 Object 数组传入参数
            LOGGER.log(WARNING, "[ClassNames] 类别指纹与 model_manifest.json 不一致：实际 {0}，期望 {1}。names 与已部署模型可能失配，类别名将错位。",
                    new Object[]{cn.fingerprint, expected});
        }

        int nc = AppConfig.MODEL_CONTRACT.numClasses();
        if (nc > 0 && nc != cn.names.size()) {
            LOGGER.log(WARNING, "[ClassNames] 类别数与 model_manifest.json 不一致：实际 {0}，期望 {1}",
                    new Object[]{cn.names.size(), nc});
        }
    }

    public static ClassNames from(List<String> names) {
        return new ClassNames(names);
    }

    private static List<String> readLines(BufferedReader r) throws IOException {
        List<String> out = new ArrayList<>();
        try (BufferedReader br = r) {
            String line;
            while ((line = br.readLine()) != null) {
                String t = line.trim();
                if (!t.isEmpty() && !t.startsWith("#")) out.add(t);
            }
        }
        return out;
    }

    public int size() { return names.size(); }

    /** 按类别 ID 取名称；越界返回占位串，不抛异常（避免一条脏数据崩掉整次推理） */
    public String nameOf(int classId) {
        if (classId < 0 || classId >= names.size()) return "unknown#" + classId;
        return names.get(classId);
    }

    /** 类别顺序指纹，与 Python 端 class_order.json 的 sha256_16 比对 */
    public String fingerprint() { return fingerprint; }

    private static String sha256Hex16(String s) {
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            byte[] d = md.digest(s.getBytes(StandardCharsets.UTF_8));
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < 8; i++) sb.append(String.format("%02x", d[i]));
            return sb.toString();
        } catch (NoSuchAlgorithmException _) {
            return "n/a";
        }
    }
}