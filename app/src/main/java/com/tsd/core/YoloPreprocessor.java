package com.tsd.core;

import com.tsd.config.AppConfig;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

/**
 * YOLO 预处理：letterbox + BGR→RGB + 归一化 + HWC→NCHW。
 * <p>
 * letterbox 参数（scale / padX / padY）会随结果返回，
 * 供 PostProcessor 把框从 640×640 网络空间逆变换回原图空间。
 * 填充灰度固定 114，与 ultralytics 保持一致（用 0 会导致精度下降）。
 */
public final class YoloPreprocessor {

    private YoloPreprocessor() {}

    /** 预处理结果：既含输入 Tensor，也含逆变换所需的几何信息 */
    public static final class Result {
        /** NCHW float 数据，长度 = 3*size*size */
        public final float[] data;
        public final long[] shape;
        /** 原图宽高 */
        public final int origWidth;
        public final int origHeight;
        /** letterbox 缩放比例 */
        public final float scale;
        /** 左右/上下填充像素（网络坐标系） */
        public final int padX;
        public final int padY;
        /** 网络输入边长 */
        public final int netSize;

        Result(float[] data, int origWidth, int origHeight,
               float scale, int padX, int padY, int netSize) {
            this.data = data;
            this.shape = new long[]{1, 3, netSize, netSize};
            this.origWidth = origWidth;
            this.origHeight = origHeight;
            this.scale = scale;
            this.padX = padX;
            this.padY = padY;
            this.netSize = netSize;
        }
    }

    /**
     * 执行预处理。
     *
     * @param bgr   OpenCV 读入的 BGR 图像（非空、通道数 3 或 4）
     * @param size  网络输入边长，通常 640
     */
    public static Result preprocess(Mat bgr, int size) {
        if (bgr == null || bgr.empty()) {
            throw new IllegalArgumentException("输入图像为空");
        }

        // ---- 0. 统一成 3 通道 BGR（PNG 可能带 alpha）----
        Mat src = bgr;
        Mat converted = null;
        if (src.channels() == 4) {
            converted = new Mat();
            Imgproc.cvtColor(src, converted, Imgproc.COLOR_BGRA2BGR);
            src = converted;
        } else if (src.channels() == 1) {
            converted = new Mat();
            Imgproc.cvtColor(src, converted, Imgproc.COLOR_GRAY2BGR);
            src = converted;
        } else if (src.channels() != 3) {
            throw new IllegalArgumentException("不支持的通道数: " + src.channels());
        }

        int ow = src.cols();
        int oh = src.rows();

        // ---- 1. letterbox：等比缩放 ----
        // 与 Ultralytics LetterBox.get_params / Python round() 对齐：
        // 两者都是 round-to-even，Math.round 是 half-up，极端 .5 情形会差 1 px。
        float scale = Math.min((float) size / ow, (float) size / oh);
        int nw = (int) Math.rint(ow * scale);
        int nh = (int) Math.rint(oh * scale);

        // INTER_LINEAR 与 Ultralytics 训练(warpAffine)、val、predict 的 LetterBox
        // 默认插值一致；此前用 INTER_AREA 会造成训练/部署预处理偏移，小标志受害最大。
        Mat resized = new Mat();
        Imgproc.resize(src, resized, new Size(nw, nh), 0, 0, Imgproc.INTER_LINEAR);
        if (converted != null) {
            converted.release();
        }

        int top    = (size - nh) / 2;
        int bottom = size - nh - top;
        int left   = (size - nw) / 2;
        int right  = size - nw - left;

        Mat canvas = new Mat();
        Core.copyMakeBorder(resized, canvas, top, bottom, left, right,
                Core.BORDER_CONSTANT, new Scalar(AppConfig.PAD_GRAY,
                        AppConfig.PAD_GRAY, AppConfig.PAD_GRAY));

        // ---- 2. BGR → RGB ----
        Mat rgb = new Mat();
        Imgproc.cvtColor(canvas, rgb, Imgproc.COLOR_BGR2RGB);

        // ---- 3. 取像素并做 /255 归一化 + HWC→NCHW ----
        if (rgb.type() != CvType.CV_8UC3) {
            rgb.convertTo(rgb, CvType.CV_8UC3);
        }
        int plane = size * size;
        byte[] pixels = new byte[plane * 3];
        rgb.get(0, 0, pixels);          // 一次性取出全部 BGR/RGB 字节，避免逐像素 get() 极慢

        float[] chw = new float[plane * 3];
        for (int i = 0, hw = 0; hw < plane; hw++, i += 3) {
            chw[hw]                 = (pixels[i]     & 0xFF) / 255.0f;   // R
            chw[plane + hw]         = (pixels[i + 1] & 0xFF) / 255.0f;   // G
            chw[plane * 2 + hw]     = (pixels[i + 2] & 0xFF) / 255.0f;   // B
        }

        resized.release();
        canvas.release();
        rgb.release();

        return new Result(chw, ow, oh, scale, left, top, size);
    }

}