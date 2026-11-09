package com.tsd;

import javafx.application.Application;

/**
 * 打包专用启动器（不继承 Application）。
 * <p>
 * JDK 启动器会检查主类是否继承 Application；若继承且 JavaFX 只存在于
 * classpath（jpackage 非模块化打包的现实情况），会抛
 * "JavaFX runtime components are missing"。
 * 用这个包装类绕开该检查，再反射式 launch 真正的 Main。
 * <p>
 * 注意：开发环境用 gradlew.bat run（javafxplugin 走 module-path）不受影响，
 * 因此 build.gradle 的 application.mainClass 仍保持 com.tsd.Main，
 * 只有 jpackage 的 --main-class 指向这里。
 */
public final class Launcher {

    private Launcher() {}

    static void main(String[] args) {
        // 可选：让 OpenCVLoader 在 app.dir 未注入时也能定位到安装目录
        if (System.getProperty("app.dir") == null) {
            String appDir = System.getProperty("user.dir");
            if (appDir != null && !appDir.isBlank()) {
                System.setProperty("app.dir", appDir);
            }
        }
        Application.launch(Main.class, args);
    }
}
