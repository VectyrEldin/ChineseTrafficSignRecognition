package com.tsd.model;

import org.junit.jupiter.api.Test;

import java.io.IOException;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;

/** 中文描述表覆盖度与关键映射回归测试。 */
class ClassDescriptionsTest {

    @Test
    void coversAll45Classes() throws IOException {
        ClassNames cn = ClassNames.load();
        ClassDescriptions d = ClassDescriptions.load();
        assertEquals(48, d.size(), "中文描述表应覆盖 45 类 + 3 个归并类");
        for (int i = 0; i < cn.size(); i++) {
            String code = cn.nameOf(i);
            assertNotEquals(code, d.descriptionOf(code), "类别 " + code + " 缺少中文描述");
        }
    }

    @Test
    void knownMappingsAreCorrect() {
        ClassDescriptions d = ClassDescriptions.load();
        assertEquals("限制速度为 40 km/h", d.descriptionOf("pl40"));
        assertEquals("禁止某两种车辆驶入标志", d.descriptionOf("p13"));
        assertEquals("非机动车行驶标志", d.descriptionOf("i2r"));
        assertEquals("指示标志", d.descriptionOf("io"));
        assertEquals("禁令标志", d.descriptionOf("po"));
        assertEquals("警告标志", d.descriptionOf("wo"));
    }

    @Test
    void unknownCodeFallsBackToItself() {
        assertEquals("not_a_class", ClassDescriptions.load().descriptionOf("not_a_class"));
    }
}
