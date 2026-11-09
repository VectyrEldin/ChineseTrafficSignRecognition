package com.tsd.ui;

import com.tsd.model.Detection;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.image.Image;
import javafx.scene.paint.Color;
import javafx.scene.text.Font;
import javafx.scene.text.FontWeight;

import java.util.ArrayList;
import java.util.List;

/**
 * 画板：显示原图 + 检测框 + 类别标签 + 置信度。
 * 内部按"适应窗口"缩放，因此绘制坐标 = 原图坐标 × displayScale。
 */
public class DetectionCanvas extends Canvas {

    private Image image;
    private List<Detection> detections = new ArrayList<>();
    private boolean showLabel = true;

    public DetectionCanvas() {
        super();
        // 尺寸跟随父容器
        widthProperty().addListener(_ -> draw());
        heightProperty().addListener(_ -> draw());
    }

    public void setImage(Image image) {
        this.image = image;
        this.detections = new ArrayList<>();
        draw();
    }

    public void setDetections(List<Detection> detections) {
        this.detections = detections == null ? new ArrayList<>() : detections;
        draw();
    }

    public void setShowLabel(boolean showLabel) {
        this.showLabel = showLabel;
        draw();
    }

    public void clear() {
        this.image = null;
        this.detections = new ArrayList<>();
        draw();
    }

    /** 重绘：图片等比适应 + 叠加所有框 */
    public void draw() {
        GraphicsContext gc = getGraphicsContext2D();
        double w = getWidth();
        double h = getHeight();
        gc.clearRect(0, 0, w, h);

        // 背景
        gc.setFill(Color.web("#1e1e24"));
        gc.fillRect(0, 0, w, h);

        if (image == null) {
            gc.setFill(Color.web("#8a8a95"));
            gc.setFont(Font.font("Microsoft YaHei", 16));
            gc.fillText("请打开一张图片（支持拖拽到窗口）", 24, 40);
            return;
        }

        // 等比缩放并居中
        double scale = Math.min(w / image.getWidth(), h / image.getHeight());
        if (scale <= 0) return;
        double dw = image.getWidth() * scale;
        double dh = image.getHeight() * scale;
        double ox = (w - dw) / 2.0;
        double oy = (h - dh) / 2.0;

        gc.drawImage(image, ox, oy, dw, dh);

        // 画框
        gc.setLineWidth(2.0);
        gc.setFont(Font.font("Microsoft YaHei", FontWeight.BOLD, 13));

        for (Detection d : detections) {
            Color c = colorOf(d.getClassId());

            double x = ox + d.getX1() * scale;
            double y = oy + d.getY1() * scale;
            double bw = (d.getX2() - d.getX1()) * scale;
            double bh = (d.getY2() - d.getY1()) * scale;

            gc.setStroke(c);
            gc.strokeRect(x, y, bw, bh);

            if (!showLabel) continue;

            String text = String.format("%s %.0f%%", d.getLabel(), d.getConfidence() * 100);
            double tw = gc.getFont().getSize() * 0.62 * text.length() + 10;
            double th = gc.getFont().getSize() + 8;
            double ty = (y - th >= 0) ? (y - th) : y;   // 顶部越界则画在框内

            gc.setFill(c);
            gc.fillRect(x, ty, tw, th);
            gc.setFill(Color.WHITE);
            gc.fillText(text, x + 5, ty + th - 6);
        }
    }

    /** 按类别 ID 生成稳定且区分度高的颜色 */
    public static Color colorOf(int classId) {
        float hue = ((classId * 47) % 360) / 360.0f;
        return Color.hsb(hue * 360, 0.75, 0.95);
    }
}