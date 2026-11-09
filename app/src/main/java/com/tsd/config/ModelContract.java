package com.tsd.config;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.logging.Logger;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * 模型输出契约：读取 export_onnx.py 写出的 model_manifest.json。
 *
 * <p>此前 Java 端靠「输出 shape」猜布局、靠「前 4 列最大绝对值 ≤ 2」猜坐标空间，
 * 换模型 / 换 ONNX operator set / 换架构时可能静默误解析。契约把这两项变成显式声明：
 * 启动时校验，不一致直接报错而不是给出错误的检测框。
 *
 * <p>本项目不引入 JSON 第三方依赖，manifest 由本项目自己生成、格式固定，
 * 因此用于 ClassNamesTest 相同的轻量键值提取即可（只取标量字段）。
 */
public final class ModelContract {

    private static final Logger LOGGER = Logger.getLogger(ModelContract.class.getName());
    private static final String RESOURCE = "/models/model_manifest.json";

    public enum CoordinateSpace {
        PIXEL_IMGSZ, NORMALIZED_0_1, UNKNOWN
    }

    private final boolean present;
    private final String raw;
    private final int imgsz;
    private final int numClasses;
    private final String outputLayout;
    private final CoordinateSpace coordinateSpace;
    private final boolean needNms;
    private final String interpolation;
    private final int padValue;

    private ModelContract(boolean present, String raw) {
        this.present = present;
        this.raw = raw;

        // 提取局部变量，避免重复解析和自动拆箱警告
        Integer imgszVal = integer("imgsz");
        this.imgsz = imgszVal == null ? 640 : imgszVal;

        Integer numClassesVal = integer("num_classes");
        this.numClasses = numClassesVal == null ? -1 : numClassesVal;

        String outputLayoutVal = string("output_layout");
        this.outputLayout = outputLayoutVal == null ? "unknown" : outputLayoutVal;

        this.coordinateSpace = parseCoordinateSpace(string("coordinate_space"));

        this.needNms = parseNeedNms();

        String interpolationVal = string("resize_interpolation");
        this.interpolation = interpolationVal == null ? "INTER_LINEAR" : interpolationVal;

        Integer padValueVal = integer("pad_value");
        this.padValue = padValueVal == null ? 114 : padValueVal;
    }

    /** 载入契约：优先 -Dtsd.manifest 指定文件，其次 classpath 资源；缺失时返回空契约。 */
    public static ModelContract load() {
        String text = null;
        String sp = System.getProperty("tsd.manifest");
        if (sp != null && !sp.isBlank()) {
            Path p = Paths.get(sp);
            if (Files.isRegularFile(p)) {
                try {
                    text = Files.readString(p, StandardCharsets.UTF_8);
                } catch (IOException e) {
                    LOGGER.warning("无法读取 -Dtsd.manifest 指定的契约: " + e.getMessage());
                }
            }
        }
        if (text == null) {
            try (InputStream in = ModelContract.class.getResourceAsStream(RESOURCE)) {
                if (in != null) {
                    text = new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8))
                            .lines().reduce("", (a, b) -> a + b + "\n");
                }
            } catch (IOException e) {
                LOGGER.warning("无法读取 classpath 契约: " + e.getMessage());
            }
        }
        if (text == null || text.isBlank()) {
            LOGGER.warning("未找到 model_manifest.json，回退到启发式解析（旧行为）");
            return new ModelContract(false, "");
        }
        ModelContract c = new ModelContract(true, text);
        LOGGER.info(() -> "[Contract] layout=" + c.outputLayout
                + " coord=" + c.coordinateSpace() + " imgsz=" + c.imgsz
                + " interpolation=" + c.interpolation);
        return c;
    }

    private static CoordinateSpace parseCoordinateSpace(String s) {
        if ("pixel_imgsz".equals(s)) return CoordinateSpace.PIXEL_IMGSZ;
        if ("normalized_0_1".equals(s)) return CoordinateSpace.NORMALIZED_0_1;
        return CoordinateSpace.UNKNOWN;
    }

    private String string(String key) {
        Matcher m = Pattern.compile("\"" + Pattern.quote(key) + "\"\\s*:\\s*\"([^\"]*)\"")
                .matcher(raw);
        return m.find() ? m.group(1) : null;
    }

    private Integer integer(String key) {
        Matcher m = Pattern.compile("\"" + Pattern.quote(key) + "\"\\s*:\\s*(-?\\d+)")
                .matcher(raw);
        return m.find() ? Integer.valueOf(m.group(1)) : null;
    }

    /**
     * 解析 need_nms 字段，缺失时默认 false。
     */
    private boolean parseNeedNms() {
        Matcher m = Pattern.compile(
                "\"" + Pattern.quote("need_nms") + "\"\\s*:\\s*(true|false)"
        ).matcher(raw);
        return m.find() && Boolean.parseBoolean(m.group(1));
    }

    public boolean present() { return present; }
    /** manifest 声明的模型 SHA-256；缺失返回 null */
    public String modelSha256() { return string("model_sha256"); }
    /** manifest 声明的类别顺序指纹 sha256[:16]；缺失返回 null */
    public String classNamesSha25616() { return string("class_names_sha256_16"); }
    public int imgsz() { return imgsz; }
    public int numClasses() { return numClasses; }
    public String outputLayout() { return outputLayout; }
    public CoordinateSpace coordinateSpace() { return coordinateSpace; }
    public boolean needNms() { return needNms; }
    public String interpolation() { return interpolation; }
    public int padValue() { return padValue; }
}
