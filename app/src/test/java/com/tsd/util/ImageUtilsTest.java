package com.tsd.util;

import javafx.scene.image.Image;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

/** 像素直写转换回归：BGRA 字节序弄错会让整幅画面红蓝互换。 */
class ImageUtilsTest {

    @TempDir
    Path tempDir;

    @BeforeAll
    static void loadNative() {
        com.tsd.core.OpenCVLoader.init();
    }

    @Test
    void bgrPixelOrderIsPreserved() {
        Mat mat = new Mat(4, 6, CvType.CV_8UC3, new Scalar(10, 20, 30));
        try {
            Image img = ImageUtils.matToImage(mat);
            assertNotNull(img);
            assertEquals(6, img.getWidth());
            assertEquals(4, img.getHeight());
            // ARGB = A=255, R=30, G=20, B=10
            assertEquals(0xFF1E140A, img.getPixelReader().getArgb(0, 0));
            assertEquals(0xFF1E140A, img.getPixelReader().getArgb(5, 3));
        } finally {
            mat.release();
        }
    }

    @Test
    void grayscaleIsExpandedToColor() {
        Mat mat = new Mat(4, 6, CvType.CV_8UC1, new Scalar(100));
        try {
            Image img = ImageUtils.matToImage(mat);
            assertNotNull(img);
            assertEquals(0xFF646464, img.getPixelReader().getArgb(0, 0));
        } finally {
            mat.release();
        }
    }

    @Test
    void bgraIsAcceptedDirectly() {
        Mat mat = new Mat(4, 6, CvType.CV_8UC4, new Scalar(10, 20, 30, 255));
        try {
            Image img = ImageUtils.matToImage(mat);
            assertNotNull(img);
            assertEquals(0xFF1E140A, img.getPixelReader().getArgb(0, 0));
        } finally {
            mat.release();
        }
    }

    @Test
    void nullAndEmptyReturnNull() {
        assertNull(ImageUtils.matToImage(null));
        Mat empty = new Mat();
        try {
            assertNull(ImageUtils.matToImage(empty));
        } finally {
            empty.release();
        }
    }

    @Test
    void readImageSafeSupportsChinesePath() throws IOException {
        Path dir = Files.createDirectory(tempDir.resolve("图片测试"));
        Path file = dir.resolve("标志.png");
        Mat mat = new Mat(8, 8, CvType.CV_8UC3, new Scalar(10, 20, 30));
        MatOfByte buf = new MatOfByte();
        try {
            Imgcodecs.imencode(".png", mat, buf);
            Files.write(file, buf.toArray());
        } finally {
            buf.release();
            mat.release();
        }
        Mat decoded = ImageUtils.readImageSafe(file);
        try {
            assertEquals(8, decoded.cols());
            assertEquals(8, decoded.rows());
            double[] pixel = decoded.get(0, 0);
            assertEquals(10, pixel[0], 1);
            assertEquals(20, pixel[1], 1);
            assertEquals(30, pixel[2], 1);
        } finally {
            decoded.release();
        }
    }

    @Test
    void readImageSafeRejectsCorruptFile() throws IOException {
        Path file = tempDir.resolve("corrupt.jpg");
        Files.write(file, new byte[]{1, 2, 3, 4});
        assertThrows(IOException.class, () -> ImageUtils.readImageSafe(file));
    }
}