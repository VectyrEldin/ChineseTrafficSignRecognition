package com.tsd.model;

/**
 * 单个检测结果：框坐标（原图空间）+ 类别 + 置信度。
 */
public class Detection {

    private final float x1;
    private final float y1;
    private final float x2;
    private final float y2;
    private final int classId;
    private final float confidence;
    private String label;

    public Detection(float x1, float y1, float x2, float y2,
                     int classId, float confidence, String label) {
        this.x1 = x1;
        this.y1 = y1;
        this.x2 = x2;
        this.y2 = y2;
        this.classId = classId;
        this.confidence = confidence;
        this.label = label;
    }

    public float getX1() { return x1; }
    public float getY1() { return y1; }
    public float getX2() { return x2; }
    public float getY2() { return y2; }
    public int getClassId() { return classId; }
    public float getConfidence() { return confidence; }

    public float getWidth()  { return x2 - x1; }
    public float getHeight() { return y2 - y1; }
    public float area()      { return Math.max(0, getWidth()) * Math.max(0, getHeight()); }

    public String getLabel() { return label == null ? ("#" + classId) : label; }
    public void setLabel(String label) { this.label = label; }

    /** 表格展示用的坐标文本 */
    public String getBoxText() {
        return String.format("(%.0f, %.0f) - (%.0f, %.0f)", x1, y1, x2, y2);
    }

    /** 置信度百分比文本 */
    public String getConfidenceText() {
        return String.format("%.1f%%", confidence * 100);
    }

    @Override
    public String toString() {
        return String.format("%s #%d conf=%.3f %s", getLabel(), classId, confidence, getBoxText());
    }
}