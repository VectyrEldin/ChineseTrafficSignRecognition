package com.tsd.util;

import javafx.scene.image.Image;
import javafx.scene.image.PixelFormat;
import javafx.scene.image.WritableImage;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.NoSuchFileException;

/**
 * 图像工具：
 *   1) 中文路径安全读取（硬性约束 8）
 *   2) OpenCV Mat → JavaFX Image
 */
public final class ImageUtils {

    private ImageUtils() {}

    /**
     * 安全读取图片。
     * <p>
     * 改用 Java NIO 读取文件字节流（完美支持中文路径），
     * 再通过 OpenCV 内存解码，彻底避免在公共可写目录创建临时文件。
     *
     * @param path 图片文件路径
     * @return 解码后的 Mat 对象
     * @throws NoSuchFileException 当文件不存在时抛出
     * @throws IOException 当文件读取失败或 OpenCV 解码失败时抛出
     */
    public static Mat readImageSafe(Path path) throws IOException {
        if (path == null || !Files.isRegularFile(path)) {
            throw new NoSuchFileException("文件不存在: " + path);
        }

        byte[] fileContent = Files.readAllBytes(path);
        MatOfByte matOfByte = new MatOfByte(fileContent);
        Mat mat;
        try {
            mat = Imgcodecs.imdecode(matOfByte, Imgcodecs.IMREAD_COLOR);
        } finally {
            matOfByte.release();
        }

        if (mat.empty()) {
            mat.release();
            throw new IOException("OpenCV 解码失败，文件可能已损坏或格式不支持: " + path);
        }
        return mat;
    }

    /**
     * OpenCV Mat → JavaFX Image：像素直写，零编码。
     * <p>
     * 现在统一转 BGRA 后经 PixelWriter 一次写入，30fps 预览无编码开销。
     */
    public static Image matToImage(Mat bgr) {
        if (bgr == null || bgr.empty()) return null;

        Mat bgraMat;
        Mat converted = null;
        if (bgr.channels() == 4) {
            bgraMat = bgr;
        } else {
            int code = bgr.channels() == 1 ? Imgproc.COLOR_GRAY2BGRA : Imgproc.COLOR_BGR2BGRA;
            converted = new Mat();
            Imgproc.cvtColor(bgr, converted, code);
            bgraMat = converted;
        }
        try {
            int w = bgraMat.cols();
            int h = bgraMat.rows();
            byte[] pixels = new byte[w * h * 4];
            bgraMat.get(0, 0, pixels);
            WritableImage image = new WritableImage(w, h);
            image.getPixelWriter().setPixels(
                    0, 0, w, h, PixelFormat.getByteBgraInstance(), pixels, 0, w * 4);
            return image;
        } finally {
            if (converted != null) converted.release();
        }
    }

}