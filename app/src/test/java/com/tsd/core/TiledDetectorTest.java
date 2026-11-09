package com.tsd.core;

import com.tsd.config.AppConfig;
import com.tsd.model.Detection;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Rect;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * 切片几何回归 + 真实模型冒烟：切片偏移回映错误会把框平移到图外，
 * 这里用覆盖性/边界断言与端到端冒烟兜底。
 */
class TiledDetectorTest {

    @BeforeAll
    static void loadNative() {
        OpenCVLoader.init();
    }

    private static void assertCovers(List<Rect> rects, int width, int height) {
        boolean[][] covered = new boolean[height][width];
        for (Rect r : rects) {
            assertTrue(r.x >= 0 && r.y >= 0, "切片越界: " + r);
            assertTrue(r.x + r.width <= width && r.y + r.height <= height, "切片越界: " + r);
            for (int y = r.y; y < r.y + r.height; y++) {
                for (int x = r.x; x < r.x + r.width; x++) {
                    covered[y][x] = true;
                }
            }
        }
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                assertTrue(covered[y][x], "像素未被任何切片覆盖: " + x + "," + y);
            }
        }
    }

    @Test
    void tilesCoverWholeImage() {
        List<Rect> rects = TiledDetector.tileRects(100, 80, 2, 0.20f);
        assertEquals(4, rects.size());
        assertCovers(rects, 100, 80);
    }

    @Test
    void adjacentTilesOverlap() {
        // baseW=500, ov=100：col0 右缘 600，col1 左缘 400 → 重叠 200
        List<Rect> rects = TiledDetector.tileRects(1000, 1000, 2, 0.20f);
        assertEquals(4, rects.size());
        assertEquals(new Rect(0, 0, 600, 600), rects.get(0));
        assertEquals(new Rect(400, 0, 600, 600), rects.get(1));
        assertEquals(new Rect(0, 400, 600, 600), rects.get(2));
        assertEquals(new Rect(400, 400, 600, 600), rects.get(3));
    }

    @Test
    void nonMultipleDimensionsStillCover() {
        List<Rect> rects = TiledDetector.tileRects(101, 53, 2, 0.20f);
        assertCovers(rects, 101, 53);
    }

    @Test
    void tinyImageProducesNoTiles() {
        assertTrue(TiledDetector.tileRects(4, 4, 2, 0.20f).isEmpty());
    }

    @Test
    void singleGridReturnsWholeImage() {
        List<Rect> rects = TiledDetector.tileRects(640, 480, 1, 0.20f);
        assertEquals(List.of(new Rect(0, 0, 640, 480)), rects);
    }

    @Test
    void detectWithRealModelKeepsBoxesInBounds() throws Exception {
        String root = System.getProperty("project.root");
        assumeTrue(root != null, "缺少 project.root 系统属性");
        Path model = Path.of(root, "app/src/main/resources/models/best.onnx");
        assumeTrue(Files.isRegularFile(model), "部署模型不存在，跳过冒烟测试");

        int nc = AppConfig.MODEL_CONTRACT.numClasses();
        assumeTrue(nc > 0, "契约缺类别数，跳过冒烟测试");

        Mat noise = new Mat(640, 640, CvType.CV_8UC3);
        Core.randu(noise, 0, 255);
        try (OrtInference inference = new OrtInference(model)) {
            List<Detection> ds = TiledDetector.detect(noise, inference, 0.25f, nc);
            assertTrue(ds.size() <= AppConfig.MAX_DETECTIONS);
            for (Detection d : ds) {
                assertTrue(d.getX1() >= 0 && d.getY1() >= 0, "框越界（负坐标）: " + d);
                assertTrue(d.getX2() <= 640 + 1e-3 && d.getY2() <= 640 + 1e-3,
                        "切片偏移回映把框平移到图外: " + d);
                assertTrue(d.getConfidence() >= 0.25f);
                assertTrue(d.getClassId() >= 0 && d.getClassId() < nc);
            }
        } finally {
            noise.release();
        }
    }
}