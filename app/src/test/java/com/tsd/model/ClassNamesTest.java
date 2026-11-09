package com.tsd.model;

import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * 类别表跨端一致性回归测试：防止 Java 端 names 与已部署模型悄悄失配。
 *
 * <p>此前把 45 硬编码在测试里，切换到 48 类模型时测试会误报。
 * 现在以 model_manifest.json 为唯一真相源（由 export_onnx.py 生成），
 * 再据此选择对应的 Python 端 class_order.json 做指纹比对。
 */
class ClassNamesTest {

    private static Path projectRoot() {
        String root = System.getProperty("project.root");
        assertNotNull(root, "缺少 project.root 系统属性（应由 Gradle test 任务注入）");
        return Path.of(root);
    }

    private static String manifest() throws IOException {
        Path p = projectRoot().resolve("app/src/main/resources/models/model_manifest.json");
        assertTrue(Files.isRegularFile(p), "缺少模型清单: " + p);
        return Files.readString(p, StandardCharsets.UTF_8);
    }

    @Test
    void namesMatchDeployedModel() throws IOException {
        String mf = manifest();
        int expected = Integer.parseInt(jsonNumClasses(mf)); // 修改调用点
        ClassNames cn = ClassNames.load();
        assertEquals(expected, cn.size(), "names 行数必须与 manifest num_classes 一致");
        assertEquals(jsonString(mf, "class_names_sha256_16"), cn.fingerprint(),
                "names 指纹必须与 manifest 一致；重训换模型后需同步 names");
    }

    @Test
    void fingerprintMatchesPythonSide() throws IOException {
        String mf = manifest();
        int expected = Integer.parseInt(jsonNumClasses(mf)); // 修改调用点
        String dir = expected == 45 ? "tt100k45_yolo"
                : expected == 48 ? "tt100k48_yolo" : null;
        assertNotNull(dir, "manifest num_classes=" + expected + "，没有对应的 Python 类别清单目录");

        ClassNames cn = ClassNames.load();
        String orderFile = expected == 45 ? "tt100k45_class_order.json" : "tt100k48_class_order.json";
        Path json = projectRoot().resolve("data/" + dir + "/" + orderFile);
        assertTrue(Files.isRegularFile(json), "缺少 Python 端类别清单: " + json);

        String text = Files.readString(json, StandardCharsets.UTF_8);
        assertEquals(String.valueOf(expected), jsonNumClasses(text), // 修改调用点
                "Python 端类别数必须与 manifest 一致");
        assertEquals(jsonString(text, "sha256_16"), cn.fingerprint(),
                "Java / Python / manifest 三处类别顺序必须同源");
    }

    /** 从 JSON 文本中提取 num_classes 数值（专门化方法，消除始终为同一参数的警告） */
    private static String jsonNumClasses(String text) {
        Matcher m = Pattern.compile("\"num_classes\"\\s*:\\s*([0-9]+)").matcher(text);
        assertTrue(m.find(), "JSON 中缺少数值字段 num_classes");
        return m.group(1);
    }

    /** 固定格式清单，用只读正则提取，避免为测试额外引入 JSON 依赖 */
    private static String jsonString(String text, String key) {
        Matcher m = Pattern.compile("\"" + key + "\"\\s*:\\s*\"([^\"]+)\"").matcher(text);
        assertTrue(m.find(), "JSON 中缺少字符串字段 " + key);
        return m.group(1);
    }
}
