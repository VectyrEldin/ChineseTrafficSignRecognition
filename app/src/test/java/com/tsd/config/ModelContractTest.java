package com.tsd.config;

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

/** 输出契约解析回归：manifest 字段必须被 Java 端正确读取。 */
class ModelContractTest {

    @Test
    void contractFieldsMatchManifest() throws IOException {
        String root = System.getProperty("project.root");
        assertNotNull(root, "缺少 project.root 系统属性");
        Path mf = Path.of(root, "app/src/main/resources/models/model_manifest.json");
        assertTrue(Files.isRegularFile(mf), "缺少模型清单: " + mf);
        String text = Files.readString(mf, StandardCharsets.UTF_8);

        System.setProperty("tsd.manifest", mf.toAbsolutePath().toString());
        try {
            ModelContract c = ModelContract.load();
            assertTrue(c.present(), "契约应成功载入");

            assertEquals(Integer.parseInt(number(text, "num_classes")), c.numClasses());

            assertEquals(string(text, "output_layout"), c.outputLayout());

            assertEquals(expectedCoordinateSpace(text), c.coordinateSpace());

            // 如果 manifest 有这两个字段才校验，没有就跳过（验证默认值走的是单元测试）
            if (text.contains("\"resize_interpolation\"")) {
                assertEquals(string(text, "resize_interpolation"), c.interpolation());
            }
            if (text.contains("\"pad_value\"")) {
                assertEquals(Integer.parseInt(number(text, "pad_value")), c.padValue());
            }
        } finally {
            System.clearProperty("tsd.manifest");
        }
    }

    /**
     * 根据 manifest 文本推断期望的坐标空间枚举值。
     * 覆盖全部三种情况，避免测试逻辑遗漏。
     */
    private static ModelContract.CoordinateSpace expectedCoordinateSpace(String text) {
        String raw = string(text, "coordinate_space");
        return switch (raw) {
            case "pixel_imgsz"    -> ModelContract.CoordinateSpace.PIXEL_IMGSZ;
            case "normalized_0_1" -> ModelContract.CoordinateSpace.NORMALIZED_0_1;
            default               -> ModelContract.CoordinateSpace.UNKNOWN;
        };
    }

    private static String string(String text, String key) {
        Matcher m = Pattern.compile("\"" + key + "\"\\s*:\\s*\"([^\"]+)\"").matcher(text);
        assertTrue(m.find(), "缺少字符串字段 " + key);
        return m.group(1);
    }

    private static String number(String text, String key) {
        Matcher m = Pattern.compile("\"" + key + "\"\\s*:\\s*([0-9]+)").matcher(text);
        assertTrue(m.find(), "缺少数值字段 " + key);
        return m.group(1);
    }
}
