package com.tsd.core;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;
import com.tsd.config.AppConfig;
import org.jetbrains.annotations.NotNull;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.attribute.PosixFilePermission;
import java.nio.file.attribute.PosixFilePermissions;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.Set;
import java.lang.reflect.Array;
import java.nio.FloatBuffer;
import java.nio.file.*;
import java.util.Collections;
import java.util.Map;
import java.util.Arrays;
import java.util.logging.Logger;
import java.util.logging.Level;

/**
 * ONNX Runtime 推理封装（CPU）。
 * <p>
 * 模型可能位于 jar 内部资源，ORT 需要真实文件路径，
 * 因此构造时若为资源则先提取到临时文件。
 */
public final class OrtInference implements AutoCloseable {

    private static final Logger LOGGER = Logger.getLogger(OrtInference.class.getName());

    /**
     * 通用张量容器：扁平数组 + shape，按行主序索引
     */
    public record Tensor(long[] shape, float[] data) {

        public int rank() {
            return shape.length;
        }

        /**
         * 便捷：把 rank=3 的 Tensor 按 [i][j][k] 取值
         */
        public float get3(int i, int j, int k) {
            int pos = (i * (int) shape[1] + j) * (int) shape[2] + k;
            return data[pos];
        }

        public String shapeString() {
            StringBuilder sb = new StringBuilder("[");
            for (int i = 0; i < shape.length; i++) {
                if (i > 0) sb.append(", ");
                sb.append(shape[i]);
            }
            return sb.append("]").toString();
        }

        // 重写 equals：使用 Arrays.equals 比较数组内容，而非引用
        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            Tensor tensor = (Tensor) o;
            return Arrays.equals(shape, tensor.shape) && Arrays.equals(data, tensor.data);
        }

        // 重写 hashCode：基于数组内容计算哈希，保持与 equals 的一致性
        @Override
        public int hashCode() {
            int result = Arrays.hashCode(shape);
            result = 31 * result + Arrays.hashCode(data);
            return result;
        }

        // 重写 toString：输出数组实际内容，避免打印对象地址
        @Override
        @NotNull
        public String toString() {
            int maxPrint = 10; // 避免 data 过长导致日志爆炸
            StringBuilder dataStr = new StringBuilder("[");
            for (int i = 0; i < Math.min(data.length, maxPrint); i++) {
                if (i > 0) dataStr.append(", ");
                dataStr.append(data[i]);
            }
            if (data.length > maxPrint) dataStr.append(", ...(total ").append(data.length).append(")");
            dataStr.append("]");
            return "Tensor[shape=" + shapeString() + ", data=" + dataStr + "]";
        }
    }

    private final OrtEnvironment env;
    private final OrtSession session;
    private final String inputName;

    /**
     * @param modelPath 模型文件路径；若为 null 则使用 classpath 资源 /models/best.onnx
     */
    public OrtInference(Path modelPath) throws OrtException, NoSuchFileException {
        this.env = OrtEnvironment.getEnvironment();

        Path real = modelPath;
        if (real == null || !Files.isRegularFile(real)) {
            // ① 系统属性指定的路径（调试用：-Dtsd.model=D:/xxx/best.onnx）
            String sp = System.getProperty("tsd.model");
            Path spPath = (sp != null) ? Paths.get(sp) : null; // 只调用一次 Paths.get
            if (spPath != null && Files.isRegularFile(spPath)) {
                real = spPath; // 复用变量
            } else {
                // ② 安装目录下的 models/best.onnx（jpackage 后与 app 同级）
                Path ext = AppConfig.appDir().resolve("models").resolve("best.onnx");
                real = Files.isRegularFile(ext) ? ext : extractModelFromClasspath();
            }
        }

        int threads = Math.clamp(Runtime.getRuntime().availableProcessors(), 1, 8);
        verifyModelSha(real);
        try (OrtSession.SessionOptions opts = new OrtSession.SessionOptions()) {
            opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
            opts.setIntraOpNumThreads(threads);
            this.session = env.createSession(real.toAbsolutePath().toString(), opts);
        }

        this.inputName = session.getInputNames().iterator().next();

        LOGGER.log(Level.INFO, "[ORT] 模型: {0}", real);
        LOGGER.log(Level.INFO, "[ORT] 输入节点: {0}  线程数: {1}", new Object[]{inputName, threads});
        for (String on : session.getOutputNames()) {
            LOGGER.log(Level.INFO, "[ORT] 输出节点: {0}", on);
        }
    }

    /** 挂模型可能被替换，导致契约失配（输出布局/类别数），加载时比对 manifest 声明的 SHA-256 */
    private static void verifyModelSha(Path model) {
        if (!AppConfig.MODEL_CONTRACT.present()) return;
        String expected = AppConfig.MODEL_CONTRACT.modelSha256();
        if (expected == null || expected.isBlank()) return;
        try {
            String actual = sha256OfFile(model);
            if (!expected.equalsIgnoreCase(actual)) {
                // 使用 {0}, {1}, {2} 占位符，避免字符串拼接
                LOGGER.log(Level.WARNING,
                        "[ORT] 模型 SHA-256 与 model_manifest.json 不一致：{0} 实际 {1}，期望 {2}。模型与契约可能失配，解析结果不可信。",
                        new Object[]{model, actual, expected});
            }
        } catch (IOException | NoSuchAlgorithmException e) {
            // 同样使用占位符；如果希望打印异常堆栈，可将 e 作为最后一个参数传入
            LOGGER.log(Level.WARNING,
                    "[ORT] 无法计算模型 SHA-256，跳过契约校验: {0}",
                    e.getMessage());
        }
    }

    /** 模型缓存目录：放在用户主目录，避免 per-machine 安装时 Program Files 不可写 */
    private static Path cacheDir() {
        return Paths.get(System.getProperty("user.home", "."), ".tsd-models");
    }

    /** 从 classpath 资源提取模型到可复用的缓存文件（jar 内运行时必需） */
    private static Path extractModelFromClasspath() throws NoSuchFileException {
        try (InputStream in = OrtInference.class.getResourceAsStream("/models/best.onnx")) {
            if (in == null) {
                throw new NoSuchFileException("/models/best.onnx", null,
                        "找不到模型：请把 best.onnx 放到 app/src/main/resources/models/ 下，或用构造参数指定路径");
            }
            return cacheModel(in.readAllBytes());
        } catch (IOException _) {
            throw new NoSuchFileException("读取模型资源流失败");
        }
    }

    /**
     * 模型以固定文件名缓存，仅当 SHA-256 与 classpath 资源不一致时才覆盖写入。
     * 旧实现每次启动 createTempFile + deleteOnExit，ORT 内存映射会导致删除失败并逐次累积数十 MB。
     */
    private static Path cacheModel(byte[] model) throws NoSuchFileException {
        try {
            String hash = HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(model));
            Path dir = cacheDir();
            Path cached = dir.resolve("best.onnx");

            if (Files.isRegularFile(cached) && hash.equals(sha256OfFile(cached))) {
                LOGGER.info(() -> "[ORT] 复用已缓存模型: " + cached);
                return cached;
            }

            Files.createDirectories(dir);
            secureDir(dir);
            Files.write(cached, model);
            cleanupLegacy(dir);
            LOGGER.info(() -> "[ORT] 模型已写入缓存: " + cached + " sha256[:16]=" + hash.substring(0, 16));
            return cached;
        } catch (NoSuchAlgorithmException | IOException _) {
            throw new NoSuchFileException("创建模型缓存文件失败");
        }
    }

    private static String sha256OfFile(Path p) throws IOException, NoSuchAlgorithmException {
        MessageDigest md = MessageDigest.getInstance("SHA-256");
        try (InputStream in = Files.newInputStream(p)) {
            byte[] buf = new byte[64 * 1024];
            int n;
            while ((n = in.read(buf)) != -1) md.update(buf, 0, n);
        }
        return HexFormat.of().formatHex(md.digest());
    }

    private static void secureDir(Path dir) throws IOException {
        // POSIX 系统限制为仅当前用户可读写执行；Windows 依赖默认 ACL
        try {
            Set<PosixFilePermission> perms = PosixFilePermissions.fromString("rwx------");
            Files.setPosixFilePermissions(dir, perms);
        } catch (UnsupportedOperationException _) {
            // 非 POSIX 系统忽略
        }
    }

    /**
     * 安静地删除文件，忽略因文件被占用等原因导致的删除失败
     */
    private static void safeDelete(Path path) {
        try {
            Files.deleteIfExists(path);
        } catch (IOException _) {
            // 被占用时留待下次启动清理，不影响本次推理
        }
    }

    /** 清理旧版随机命名残留（仅匹配本应用生成的文件名前缀） */
    private static void cleanupLegacy(Path dir) {
        try (DirectoryStream<Path> ds = Files.newDirectoryStream(dir, "tsd-model-*.onnx")) {
            for (Path p : ds) {
                safeDelete(p); // 调用新方法，消除嵌套 try 块
            }
        } catch (IOException _) {
            // 目录不可读时忽略
        }
    }

    /** 执行一次推理，返回第一个输出张量 */
    public Tensor infer(float[] chwData, long[] shape) throws OrtException {
        try (OnnxTensor tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(chwData), shape)) {
            Map<String, OnnxTensor> inputs = Collections.singletonMap(inputName, tensor);
            try (OrtSession.Result result = session.run(inputs)) {
                OnnxValue value = result.get(0);
                long[] outShape = readShape(value);
                float[] flat;
                if (value instanceof OnnxTensor ot) {
                    // 直读 native buffer：一次拷贝，避免 getValue() 嵌套数组 + 逐元素装箱
                    FloatBuffer fb = ot.getFloatBuffer();
                    flat = new float[fb.remaining()];
                    fb.get(flat);
                } else {
                    flat = flatten(value.getValue());
                }
                return new Tensor(outShape, flat);
            }
        }
    }

    /** 便捷重载：直接吃 YoloPreprocessor 的结果 */
    public Tensor infer(YoloPreprocessor.Result prep) throws OrtException {
        return infer(prep.data, prep.shape);
    }

    /** 读取输出 shape；失败时按嵌套数组结构推断 */
    private static long[] readShape(OnnxValue v) throws OrtException {
        try {
            Object info = v.getInfo();
            if (info instanceof TensorInfo ti) {
                return ti.getShape();
            }
        } catch (Exception _) { // 将 Throwable 改为 Exception
            // 某些版本 getInfo() 不可用，走下方推断
        }
        return inferShape(v.getValue());
    }

    private static long[] inferShape(Object arr) {
        if (arr == null || !arr.getClass().isArray()) return new long[]{0};
        long len = Array.getLength(arr);
        if (len == 0) return new long[]{0};
        Object first = Array.get(arr, 0);
        if (first != null && first.getClass().isArray()) {
            long[] sub = inferShape(first);
            long[] out = new long[sub.length + 1];
            out[0] = len;
            System.arraycopy(sub, 0, out, 1, sub.length);
            return out;
        }
        return new long[]{len};
    }

    /** 递归摊平任意维度的 float 数组 */
    private static float[] flatten(Object v) {
        java.util.List<Float> buf = new java.util.ArrayList<>();
        flattenInto(v, buf);
        float[] out = new float[buf.size()];
        for (int i = 0; i < out.length; i++) out[i] = buf.get(i);
        return out;
    }

    private static void flattenInto(Object v, java.util.List<Float> buf) {
        if (v == null) return;
        if (v instanceof float[] fa) {
            for (float f : fa) buf.add(f);
        } else if (v.getClass().isArray()) {
            int n = Array.getLength(v);
            for (int i = 0; i < n; i++) flattenInto(Array.get(v, i), buf);
        } else if (v instanceof Number num) {
            buf.add(num.floatValue());
        }
    }

    @Override
    public void close() throws OrtException {
        session.close();
    }
}