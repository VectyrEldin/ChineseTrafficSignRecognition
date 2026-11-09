package com.tsd.core;

import com.tsd.config.AppConfig;
import com.tsd.config.ModelContract;
import com.tsd.model.Detection;
import static com.tsd.config.ModelContract.CoordinateSpace.*;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.logging.Logger;
import java.util.logging.Level;

/**
 * YOLO26 输出解析。
 *
 * <p><b>⚠️ 必须按 scripts/onnx_output_format.txt 的实测 shape 来解析</b>，
 * YOLO26 是 NMS-free（端到端）架构，严禁照搬 YOLOv8 的 (1, 84, 8400) 解析代码。
 *
 * <p>本类根据实测 shape 自动二选一：
 * <ol>
 *   <li><b>端到端 NMS-free</b>：shape = [1, N, 6]，每行 [x1, y1, x2, y2, score, classId]
 *       → 自动探测坐标尺度（归一化/像素），classId 四舍五入取整并过滤非法类别，直接按置信度过滤，<b>不做 NMS</b></li>
 *   <li><b>原始头输出</b>：shape = [1, 4+nc, A]（如 [1,49,8400]）
 *       → 转置为 [A, 4+nc]，前 4 列按 cx,cy,w,h 处理并转换为边界框坐标 (x1, y1, x2, y2)，取最大类别分，<b>自写 NMS</b></li>
 * </ol>
 *
 * <p>两种分支最后都做 letterbox 逆变换，把坐标还原到原图空间。
 */
public final class PostProcessor {

    private static final Logger LOGGER = Logger.getLogger(PostProcessor.class.getName());

    private PostProcessor() {}

    /**
     * 解析输出 Tensor。
     *
     * @param out          OrtInference 返回的输出 Tensor
     * @param prep         预处理结果（含 scale / padX / padY / 原图尺寸）
     * @param confThresh   置信度阈值
     * @param numClasses   类别数（由 ClassNames.size() 传入，不硬编码）
     */
    public static List<Detection> parse(OrtInference.Tensor out,
                                        YoloPreprocessor.Result prep,
                                        float confThresh,
                                        int numClasses) {
        return parse(out, prep, confThresh, numClasses, null);
    }

    /**
     * 契约驱动解析：布局与坐标空间由 model_manifest.json 显式声明，
     * 不再靠 shape / 最大绝对值启发式猜测；不一致直接抛错而非静默给出错误框。
     */
    public static List<Detection> parse(OrtInference.Tensor out,
                                        YoloPreprocessor.Result prep,
                                        float confThresh,
                                        int numClasses,
                                        ModelContract contract) {
        if (LOGGER.isLoggable(Level.INFO)) {
            LOGGER.log(Level.INFO, "[Post] 输出 shape = {0}", out.shapeString());
        }
        boolean useContract = contract != null && contract.present();
        if (useContract) {
            validateAgainstContract(out, contract, numClasses);
        }

        Boolean normalizedOverride = null;
        if (useContract) {
            normalizedOverride = switch (contract.coordinateSpace()) {
                case PIXEL_IMGSZ    -> Boolean.FALSE;
                case NORMALIZED_0_1 -> Boolean.TRUE;
                case UNKNOWN        -> null;
            };
        }

        List<Detection> raw;
        if (isEnd2End(out)) {
            raw = parseEnd2End(out, confThresh, numClasses, normalizedOverride);
        } else if (isRawHead(out, numClasses)) {
            raw = parseRawHead(out, confThresh, numClasses, prep.netSize);
        } else {
            // 若 onnx_output_format.txt 里是别的 shape，需在此补充对应解析分支，切勿沿用 YOLOv8 假设。
            throw new IllegalStateException(
                    "无法识别的 YOLO26 输出 shape: " + out.shapeString()
                            + "（numClasses=" + numClasses + "）。"
                            + "请打开 scripts/onnx_output_format.txt 核对实测输出，"
                            + "并在 PostProcessor 中补充对应解析分支。");
        }

        // letterbox 逆变换 + 裁剪 + 小框过滤
        List<Detection> result = new ArrayList<>(raw.size());
        int minSide = Math.min(prep.origWidth, prep.origHeight);
        float minBox = (float) minSide * AppConfig.MIN_BOX_RATIO;

        for (Detection d : raw) {
            float x1 = (d.getX1() - (float) prep.padX) / prep.scale;
            float y1 = (d.getY1() - (float) prep.padY) / prep.scale;
            float x2 = (d.getX2() - (float) prep.padX) / prep.scale;
            float y2 = (d.getY2() - (float) prep.padY) / prep.scale;

            // 裁剪到原图边界
            x1 = clamp(x1, (float) prep.origWidth);
            y1 = clamp(y1, (float) prep.origHeight);
            x2 = clamp(x2, (float) prep.origWidth);
            y2 = clamp(y2, (float) prep.origHeight);

            if (x2 - x1 < minBox || y2 - y1 < minBox) continue;   // 抑制噪声小框

            result.add(new Detection(x1, y1, x2, y2, d.getClassId(), d.getConfidence(), d.getLabel()));
        }

        result.sort(Comparator.comparingDouble(Detection::getConfidence).reversed());
        if (result.size() > AppConfig.MAX_DETECTIONS) {
            result = new ArrayList<>(result.subList(0, AppConfig.MAX_DETECTIONS));
        }
        return result;
    }

    // ==================================================================
    // 形态 A：端到端 NMS-free，[1, N, 6]
    // ==================================================================
    private static boolean isEnd2End(OrtInference.Tensor out) {
        return out.rank() == 3 && out.shape()[2] == 6 && out.shape()[1] > 0 && out.shape()[1] < 10000;
    }

    /**
     * 解析 [1, N, 6]：每行 = [x1, y1, x2, y2, score, classId]。
     * <p>已根据 scripts/onnx_output_format.txt 实测结果核对并实现：
     * <ul>
     *   <li>输出 shape 为 [1, N, 6]（N 为检测框数量）；</li>
     *   <li>坐标尺度自动探测：若最大值 ≤ 2 视为归一化，自动转换为网络像素坐标；</li>
     *   <li>第 6 列为 classId，通过四舍五入取整，非法类别将被丢弃；</li>
     *   <li>输出已经过端到端去重，此处【绝对不要】再调用 NMS。</li>
     * </ul>
     */
    private static List<Detection> parseEnd2End(OrtInference.Tensor out,
                                                float confThresh,
                                                int numClasses,
                                                Boolean normalizedOverride) {
        List<Detection> list = new ArrayList<>();
        int n = (int) out.shape()[1];

        // 契约优先；契约缺失时才用「最大绝对值 ≤ 2」的启发式兜底。
        // 整张表仅扫描一次，避免循环内重复全量遍历。
        boolean normalized = normalizedOverride != null
                ? normalizedOverride : detectNormalized(out, n);

        float coordinateScale = normalized ? (float) AppConfig.INPUT_SIZE : 1.0f;

        for (int i = 0; i < n; i++) {
            float score = out.get3(0, i, 4);
            float x1 = out.get3(0, i, 0);
            float y1 = out.get3(0, i, 1);
            float x2 = out.get3(0, i, 2);
            float y2 = out.get3(0, i, 3);
            int cls = Math.round(out.get3(0, i, 5));

            if (score < confThresh || cls < 0 || cls >= numClasses || x2 <= x1 || y2 <= y1) {
                continue;
            }

            if (normalized) {
                // 归一化坐标 → 网络像素坐标（后续 letterbox 逆变换会再除 scale）
                // ✅ 修改：对应变量名同步更新
                x1 *= coordinateScale;
                y1 *= coordinateScale;
                x2 *= coordinateScale;
                y2 *= coordinateScale;
            }

            list.add(new Detection(x1, y1, x2, y2, cls, score, null));
        }
        return list;
    }

    /** 启动时对拍实测 Tensor 与 manifest 契约，避免换模型后静默误解析。 */
    private static void validateAgainstContract(
            OrtInference.Tensor out, ModelContract c, int numClasses) {
        if (c.numClasses() > 0 && c.numClasses() != numClasses) {
            throw new IllegalStateException(
                    String.format("模型契约 num_classes=%d 与 names 行数 %d 不一致，拒绝推理",
                            c.numClasses(), numClasses));
        }
        String layout = c.outputLayout();
        if ("end2end_nms_free".equals(layout)) {
            if (!isEnd2End(out)) {
                throw new IllegalStateException(
                        String.format("契约声明 end2end_nms_free，但实测 shape=%s", out.shapeString()));
            }
            if (c.needNms()) {
                throw new IllegalStateException("契约冲突：端到端输出禁止再做 NMS");
            }
        } else if ("raw_head_transposed".equals(layout) && !isRawHead(out, numClasses)) {
            throw new IllegalStateException(
                    String.format("契约声明 raw_head_transposed，但实测 shape=%s", out.shapeString()));
        } else if (!"unknown".equals(layout)) {
            LOGGER.log(Level.WARNING, "[Contract] 未识别的 output_layout: {0}，跳过 shape 校验", layout);
        }
        if (c.coordinateSpace() == UNKNOWN) {
            LOGGER.warning("[Contract] 未声明 coordinate_space，回退启发式探测");
        }
        if (!"INTER_LINEAR".equalsIgnoreCase(c.interpolation())) {
            LOGGER.log(Level.WARNING, "[Contract] manifest 插值={0}，Java 端固定 INTER_LINEAR，请重新导出或核对",
                    c.interpolation());
        }
    }

    private static boolean detectNormalized(OrtInference.Tensor out, int n) {
        double mx = 0;
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < 4; k++) mx = Math.max(mx, Math.abs(out.get3(0, i, k)));
        }
        return mx > 0 && mx <= 2.0;
    }

    // ==================================================================
    // 形态 B：原始头输出 [1, 4+nc, A]，需转置 + 自写 NMS
    // ==================================================================
    private static boolean isRawHead(OrtInference.Tensor out, int nc) {
        if (out.rank() != 3) return false;
        long a = out.shape()[1];
        long b = out.shape()[2];
        return (a == 4 + nc || a == 5 + nc || a == 6 + nc) && b > 1000;
    }

    /**
     * 解析 [1, C, A]（C = 4+nc，A = 8400 等）：转置为 [A, C]，
     * 前 4 列 cx,cy,w,h（网络像素），随后 nc 列为类别分。
     * <p>已根据实测结果核对并实现：
     * <ul>
     *   <li>坐标尺度自动探测：若框坐标最大值 ≤ 2 视为归一化，自动乘网络尺寸转换为像素；</li>
     *   <li>前 4 列按 cx,cy,w,h 处理并转换为边界框坐标 (x1, y1, x2, y2)；</li>
     *   <li>原始头输出未经去重，此处执行类别无关 NMS。</li>
     * </ul>
     */
    private static List<Detection> parseRawHead(OrtInference.Tensor out,
                                                float confThresh,
                                                int numClasses,
                                                int netSize) {

        int a = (int) out.shape()[2];
        int boxDim = 4;

        List<Detection> candidates = new ArrayList<>();

        for (int i = 0; i < a; i++) {
            // 找最大类别分
            int bestCls = -1;
            float bestScore = -1f;
            for (int k = 0; k < numClasses; k++) {
                float s = out.get3(0, boxDim + k, i);
                if (s > bestScore) { bestScore = s; bestCls = k; }
            }
            if (bestCls < 0 || bestScore < confThresh) continue;

            float cx = out.get3(0, 0, i);
            float cy = out.get3(0, 1, i);
            float bw = out.get3(0, 2, i);
            float bh = out.get3(0, 3, i);

            // 归一化探测：整张输出最大值 ≤ 2 则认为框坐标也是归一化的
            if (Math.max(Math.max(Math.abs(cx), Math.abs(cy)),
                    Math.max(Math.abs(bw), Math.abs(bh))) <= 2.0f) {
                cx *= netSize * 1.0f; cy *= netSize * 1.0f; bw *= netSize * 1.0f; bh *= netSize * 1.0f;
            }

            float x1 = cx - bw / 2f;
            float y1 = cy - bh / 2f;
            float x2 = cx + bw / 2f;
            float y2 = cy + bh / 2f;
            candidates.add(new Detection(x1, y1, x2, y2, bestCls, bestScore, null));
        }

        return nms(candidates, AppConfig.IOU_THRESHOLD);
    }

    // ==================================================================
    // 工具
    // ==================================================================
    /** 类别无关 NMS（若需按类分别做，用 nmsClassAware） */
    public static List<Detection> nms(List<Detection> boxes, float iouThresh) {
        List<Detection> sorted = new ArrayList<>(boxes);
        sorted.sort(Comparator.comparingDouble(Detection::getConfidence).reversed());

        List<Detection> kept = new ArrayList<>();
        boolean[] suppressed = new boolean[sorted.size()];

        for (int i = 0; i < sorted.size(); i++) {
            if (suppressed[i]) continue;
            Detection a = sorted.get(i);
            kept.add(a);
            for (int j = i + 1; j < sorted.size(); j++) {
                if (suppressed[j]) continue;
                if (iou(a, sorted.get(j)) > iouThresh) suppressed[j] = true;
            }
        }
        return kept;
    }

    /** 按类 NMS：只在同类内抑制，同杆组合标志的不同类框重叠时不会互相误杀 */
    public static List<Detection> nmsClassAware(List<Detection> boxes, float iouThresh) {
        java.util.Map<Integer, List<Detection>> byClass = new java.util.HashMap<>();
        for (Detection d : boxes) {
            byClass.computeIfAbsent(d.getClassId(), _ -> new ArrayList<>()).add(d);
        }
        List<Detection> kept = new ArrayList<>();
        for (List<Detection> group : byClass.values()) {
            kept.addAll(nms(group, iouThresh));
        }
        return kept;
    }

    public static float iou(Detection a, Detection b) {
        float ix1 = Math.max(a.getX1(), b.getX1());
        float iy1 = Math.max(a.getY1(), b.getY1());
        float ix2 = Math.min(a.getX2(), b.getX2());
        float iy2 = Math.min(a.getY2(), b.getY2());
        float iw = Math.max(0, ix2 - ix1);
        float ih = Math.max(0, iy2 - iy1);
        float inter = iw * ih;
        float ua = a.area() + b.area() - inter;
        return ua <= 0 ? 0 : inter / ua;
    }

    private static float clamp(float v, float hi) {
        return Math.clamp(v, 0.0f, hi);
    }
}