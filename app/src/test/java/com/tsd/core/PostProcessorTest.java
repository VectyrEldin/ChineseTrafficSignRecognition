package com.tsd.core;

import com.tsd.config.AppConfig;
import com.tsd.config.ModelContract;
import com.tsd.model.Detection;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** 与 com.tsd.core 同包，便于构造包可见的 YoloPreprocessor.Result。 */
class PostProcessorTest {

    /** 640x640 原图、无填充、scale=1：网络坐标即原图坐标 */
    private static YoloPreprocessor.Result prep() {
        return new YoloPreprocessor.Result(new float[0], 640, 640, 1.0f, 0, 0, 640);
    }

    private static OrtInference.Tensor end2End(float... row) {
        return new OrtInference.Tensor(new long[]{1, 1, 6}, row);
    }

    @Test
    void pixelCoordinatesArePreserved() {
        List<Detection> ds = PostProcessor.parse(
                end2End(64f, 128f, 192f, 256f, 0.9f, 3f), prep(), 0.25f, 45);
        assertEquals(1, ds.size());
        assertEquals(64f, ds.getFirst().getX1(), 0.01f);
        assertEquals(256f, ds.getFirst().getY2(), 0.01f);
    }

    @Test
    void normalizedCoordinatesAreScaledToNetworkPixels() {
        // 修复前：归一化坐标未放大，会被 MIN_BOX_RATIO 当成噪声小框过滤，结果为空
        List<Detection> ds = PostProcessor.parse(
                end2End(0.1f, 0.2f, 0.3f, 0.4f, 0.9f, 3f), prep(), 0.25f, 45);
        assertEquals(1, ds.size(), "归一化坐标必须转换为网络像素坐标");
        assertEquals(AppConfig.INPUT_SIZE * 0.1f, ds.getFirst().getX1(), 0.01f);
        assertEquals(AppConfig.INPUT_SIZE * 0.3f, ds.getFirst().getX2(), 0.01f);
    }

    @Test
    void confidenceAndIllegalClassAreFiltered() {
        assertTrue(PostProcessor.parse(
                end2End(10f, 10f, 100f, 100f, 0.1f, 3f), prep(), 0.25f, 45).isEmpty());
        assertTrue(PostProcessor.parse(
                end2End(10f, 10f, 100f, 100f, 0.9f, 99f), prep(), 0.25f, 45).isEmpty());
    }

    @Test
    void contractDrivenParseUsesDeclaredCoordinateSpace() {
        ModelContract c = ModelContract.load();
        assertTrue(c.present(), "classpath 中应有 model_manifest.json");
        List<Detection> ds = PostProcessor.parse(
                end2End(64f, 128f, 192f, 256f, 0.9f, 3f),
                prep(), 0.25f, c.numClasses(), c);
        assertEquals(1, ds.size());
        assertEquals(64f, ds.getFirst().getX1(), 0.01f);
    }

    @Test
    void contractRejectsLayoutMismatch() {
        ModelContract c = ModelContract.load();
        assertTrue(c.present());

        // 契约声明 end2end_nms_free，却给一个原始头形状的 Tensor ：必须拒绝而非静默误解析
        OrtInference.Tensor rawHead = new OrtInference.Tensor(
                new long[]{1, 4 + c.numClasses(), 8400},
                new float[(4 + c.numClasses()) * 8400]);

        // 将参数准备移出 lambda，确保 lambda 体内只有一个调用
        YoloPreprocessor.Result prepResult = prep();
        int numClasses = c.numClasses();

        org.junit.jupiter.api.Assertions.assertThrows(IllegalStateException.class,
                () -> PostProcessor.parse(rawHead, prepResult, 0.25f, numClasses, c));
    }

    @Test
    void iouAndNms() {
        Detection a = new Detection(0, 0, 100, 100, 1, 0.9f, "a");
        Detection b = new Detection(2, 2, 102, 102, 1, 0.8f, "b");
        assertEquals(1.0f, PostProcessor.iou(a, a), 1e-6f);
        assertTrue(PostProcessor.iou(a, b) > 0.9f);
        List<Detection> kept = PostProcessor.nms(List.of(a, b), 0.5f);
        assertEquals(1, kept.size());
        assertEquals(0.9f, kept.getFirst().getConfidence(), 1e-6f);
    }

    @Test
    void classAwareNmsKeepsOverlappingDifferentClasses() {
        // 同杆组合标志：不同类的高重叠框不应互相抑制
        Detection a = new Detection(0, 0, 100, 100, 1, 0.9f, "a");
        Detection b = new Detection(2, 2, 102, 102, 2, 0.8f, "b");
        List<Detection> kept = PostProcessor.nmsClassAware(List.of(a, b), 0.5f);
        assertEquals(2, kept.size(), "不同类的重叠框都必须保留");
    }

    @Test
    void classAwareNmsSuppressesSameClassDuplicates() {
        Detection a = new Detection(0, 0, 100, 100, 1, 0.9f, "a");
        Detection b = new Detection(2, 2, 102, 102, 1, 0.8f, "b");
        List<Detection> kept = PostProcessor.nmsClassAware(List.of(a, b), 0.5f);
        assertEquals(1, kept.size());
        assertEquals(0.9f, kept.getFirst().getConfidence(), 1e-6f);
    }
}
