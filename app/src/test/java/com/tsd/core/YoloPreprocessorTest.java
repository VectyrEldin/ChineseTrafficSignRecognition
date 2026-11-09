package com.tsd.core;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * letterbox 几何与像素布局回归：坐标逆变换的正确性完全依赖这里的
 * scale / padX / padY，奇数 padding 分配与 round-to-even 是最易回归的两处。
 */
class YoloPreprocessorTest {

    private static final float PAD = 114f / 255f;
    private static final float DELTA = 1e-6f;

    @BeforeAll
    static void loadNative() {
        OpenCVLoader.init();
    }

    private static Mat solid(int w, int h, double b, double g, double r) {
        return new Mat(h, w, CvType.CV_8UC3, new Scalar(b, g, r));
    }

    @Test
    void squareImageHasNoPadding() {
        Mat mat = solid(640, 640, 10, 20, 30);
        try {
            YoloPreprocessor.Result r = YoloPreprocessor.preprocess(mat, 640);
            assertEquals(1.0f, r.scale, DELTA);
            assertEquals(0, r.padX);
            assertEquals(0, r.padY);
            assertEquals(640, r.origWidth);
            assertEquals(640, r.origHeight);
            assertArrayEquals(new long[]{1, 3, 640, 640}, r.shape);
            assertEquals(3 * 640 * 640, r.data.length);
            int plane = 640 * 640;
            // BGR(10,20,30) → NCHW 的 R/G/B 平面分别为 30/20/10
            assertEquals(30f / 255f, r.data[0], DELTA);
            assertEquals(20f / 255f, r.data[plane], DELTA);
            assertEquals(10f / 255f, r.data[plane * 2], DELTA);
        } finally {
            mat.release();
        }
    }

    @Test
    void wideImagePadsVertically() {
        Mat mat = solid(1280, 640, 10, 20, 30);
        try {
            YoloPreprocessor.Result r = YoloPreprocessor.preprocess(mat, 640);
            assertEquals(0.5f, r.scale, DELTA);
            assertEquals(0, r.padX);
            assertEquals(160, r.padY);
            // 首行是灰边，中心行是图像内容
            assertEquals(PAD, r.data[0], DELTA);
            assertEquals(30f / 255f, r.data[320 * 640], DELTA);
        } finally {
            mat.release();
        }
    }

    @Test
    void oddPaddingSplitPutsExtraRowOnBottom() {
        // 639 → 640 差 1 px：top=0, bottom=1（与 Ultralytics //2 一致）
        Mat mat = solid(640, 639, 10, 20, 30);
        try {
            YoloPreprocessor.Result r = YoloPreprocessor.preprocess(mat, 640);
            assertEquals(0, r.padY);
            assertEquals(30f / 255f, r.data[0], DELTA);
            assertEquals(PAD, r.data[639 * 640], DELTA);
        } finally {
            mat.release();
        }
    }

    @Test
    void resizeUsesRoundToEvenNotHalfUp() {
        // 25 * 2.5 = 62.5：Math.rint → 62（padY=289）；Math.round → 63（padY=288）
        Mat mat = solid(256, 25, 10, 20, 30);
        try {
            YoloPreprocessor.Result r = YoloPreprocessor.preprocess(mat, 640);
            assertEquals(2.5f, r.scale, DELTA);
            assertEquals(289, r.padY);
        } finally {
            mat.release();
        }
    }

    @Test
    void grayscaleMatchesEquivalentBgr() {
        Mat gray = new Mat(100, 120, CvType.CV_8UC1, new Scalar(100));
        Mat bgr = solid(120, 100, 100, 100, 100);
        try {
            float[] fromGray = YoloPreprocessor.preprocess(gray, 640).data;
            float[] fromBgr = YoloPreprocessor.preprocess(bgr, 640).data;
            assertArrayEquals(fromBgr, fromGray);
        } finally {
            gray.release();
            bgr.release();
        }
    }

    @Test
    void bgraMatchesBgr() {
        Mat bgraMat = new Mat(100, 120, CvType.CV_8UC4, new Scalar(10, 20, 30, 255));
        Mat bgr = solid(120, 100, 10, 20, 30);
        try {
            float[] fromBgra = YoloPreprocessor.preprocess(bgraMat, 640).data;
            float[] fromBgr = YoloPreprocessor.preprocess(bgr, 640).data;
            assertArrayEquals(fromBgr, fromBgra);
        } finally {
            bgraMat.release();
            bgr.release();
        }
    }

    @Test
    void emptyInputThrows() {
        Mat empty = new Mat();
        try {
            assertThrows(IllegalArgumentException.class,
                    () -> YoloPreprocessor.preprocess(empty, 640));
        } finally {
            empty.release();
        }
    }

    @Test
    void unsupportedChannelCountThrows() {
        Mat twoChannel = new Mat(32, 32, CvType.CV_8UC2);
        try {
            assertThrows(IllegalArgumentException.class,
                    () -> YoloPreprocessor.preprocess(twoChannel, 640));
        } finally {
            twoChannel.release();
        }
    }

    @Test
    void tallImagePadsHorizontally() {
        Mat mat = solid(640, 1280, 10, 20, 30);
        try {
            YoloPreprocessor.Result r = YoloPreprocessor.preprocess(mat, 640);
            assertEquals(0.5f, r.scale, DELTA);
            assertEquals(160, r.padX);
            assertEquals(0, r.padY);
            // 首列是灰边，中心列是图像内容
            assertEquals(PAD, r.data[0], DELTA);
            assertEquals(30f / 255f, r.data[320], DELTA);
        } finally {
            mat.release();
        }
    }
}