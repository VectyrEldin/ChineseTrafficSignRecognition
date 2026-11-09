package com.tsd.core;

import ai.onnxruntime.OrtException;
import com.tsd.config.AppConfig;
import com.tsd.model.Detection;
import org.opencv.core.Mat;
import org.opencv.core.Rect;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.logging.Logger;
import java.util.logging.Level;

/**
 * 免重训的小目标增强推理：全局图 + 重叠切片，各自缩放到同一 imgsz 后推理。
 *
 * <p>TT100K 源图为 2048×2048，直接缩到 640 时约 43.6% 的标志不足 10 px。
 * 2×2 切片把每块的有效缩放系数提高约 2 倍，等价于提高小目标的输入分辨率，
 * 但模型权重、ONNX 文件、类别顺序全部不变。
 *
 * <p>注意：YOLO26 端到端输出在每个切片内已去重，此处按类 NMS 只用于消除
 * 「同一切片边界被相邻切片重复检出」的跨切片重复，不是对端到端输出再做 NMS。
 *
 * <p>默认关闭（-Dtsd.tiled=true 启用）。启用后推理次数为 1 + grid² 次，
 * CPU 时延约线性上升，适合离线单图分析，不适合摄像头实时流。
 */
public final class TiledDetector {

    private static final Logger LOGGER = Logger.getLogger(TiledDetector.class.getName());

    private TiledDetector() {}

    public static List<Detection> detect(
            Mat bgr, OrtInference inference, float conf, int numClasses) throws OrtException {
        int size = AppConfig.INPUT_SIZE;

        // ① 全局图：保证大目标和跨切片目标不丢失
        YoloPreprocessor.Result global = YoloPreprocessor.preprocess(bgr, size);
        List<Detection> all = new ArrayList<>(PostProcessor.parse(
                inference.infer(global), global, conf, numClasses,
                AppConfig.MODEL_CONTRACT));

        // ② 重叠切片
        int grid = Math.max(1, AppConfig.TILE_GRID);
        int tiles = 0;

        for (Rect r : tileRects(bgr.cols(), bgr.rows(), grid, AppConfig.TILE_OVERLAP)) {
            Mat tile = new Mat(bgr, r);
            try {
                YoloPreprocessor.Result prep = YoloPreprocessor.preprocess(tile, size);
                List<Detection> local = PostProcessor.parse(
                        inference.infer(prep), prep, conf, numClasses,
                        AppConfig.MODEL_CONTRACT);
                for (Detection d : local) {
                    all.add(new Detection(
                            d.getX1() + r.x, d.getY1() + r.y,
                            d.getX2() + r.x, d.getY2() + r.y,
                            d.getClassId(), d.getConfidence(), d.getLabel()));
                }
                tiles++;
            } finally {
                tile.release();
            }
        }

        // ③ 跨切片去重（按类，避免同杆组合标志的不同类框互相抑制）+ 全局限流
        List<Detection> merged = PostProcessor.nmsClassAware(all, AppConfig.IOU_THRESHOLD);
        merged.sort(Comparator.comparingDouble(Detection::getConfidence).reversed());
        if (merged.size() > AppConfig.MAX_DETECTIONS) {
            merged = new ArrayList<>(merged.subList(0, AppConfig.MAX_DETECTIONS));
        }

        // 仅在 INFO 级别启用时才执行字符串格式化，避免无谓的性能损耗
        if (LOGGER.isLoggable(Level.INFO)) {
            LOGGER.info(String.format(
                    "[Tiled] grid=%dx%d tiles=%d 合并前=%d 合并后=%d",
                    grid, grid, tiles, all.size(), merged.size()));
        }

        return merged;
    }

    /** 切片几何：grid×grid 网格 + 双向重叠，边缘裁剪到图内；过小的切片（<8px）跳过 */
    static List<Rect> tileRects(int width, int height, int grid, float overlap) {
        int baseW = (width + grid - 1) / grid;
        int baseH = (height + grid - 1) / grid;
        int ovW = (int) (baseW * overlap);
        int ovH = (int) (baseH * overlap);
        List<Rect> rects = new ArrayList<>();
        for (int row = 0; row < grid; row++) {
            for (int col = 0; col < grid; col++) {
                int x0 = Math.max(0, col * baseW - ovW);
                int y0 = Math.max(0, row * baseH - ovH);
                int x1 = Math.min(width, (col + 1) * baseW + ovW);
                int y1 = Math.min(height, (row + 1) * baseH + ovH);
                if (x1 - x0 < 8 || y1 - y0 < 8) continue;
                rects.add(new Rect(x0, y0, x1 - x0, y1 - y0));
            }
        }
        return rects;
    }
}
