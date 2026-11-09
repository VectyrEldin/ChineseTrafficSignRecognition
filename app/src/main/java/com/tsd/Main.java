package com.tsd;

import ai.onnxruntime.OrtException;
import com.tsd.config.AppConfig;
import com.tsd.core.OpenCVLoader;
import com.tsd.core.OrtInference;
import com.tsd.model.ClassNames;
import com.tsd.ui.MainController;
import com.tsd.util.AppLogging;
import javafx.application.Application;
import javafx.application.Platform;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.stage.Stage;

import java.io.IOException;
import java.nio.file.NoSuchFileException;
import java.util.Objects;

import java.util.logging.Level;
import java.util.logging.Logger;
/**
 * 程序入口。
 * <p>
 * 职责：加载 OpenCV native → 载入 FXML → 显示主窗口 → 后台加载模型并注入控制器。
 * 推理引擎的所有权归本类：创建后注入控制器借用，窗口关闭时统一关闭。
 * 注意：JavaFX 25 + JDK 25 下，非模块化应用启动 Application 子类是安全的。
 */
public class Main extends Application {

    private static final Logger LOGGER = Logger.getLogger(Main.class.getName());

    // 仅在 FX 线程读写
    private MainController controller;
    private OrtInference engine;
    // 加载线程与 FX 线程共享：窗口已关闭后加载才完成时，由加载侧直接关闭引擎
    private volatile boolean stopped;

    @Override
    public void start(Stage stage) throws IOException {
        // 尽早安装文件 Handler，覆盖后续全部业务日志（含安装包无控制台场景）
        AppLogging.init();

        // ---- 1. 加载 OpenCV native（三级回退，失败不阻断启动，由界面提示）----
        try {
            OpenCVLoader.init();
            LOGGER.info(() -> "[Main] " + OpenCVLoader.status());
        } catch (UnsatisfiedLinkError | Exception t) {
            LOGGER.log(Level.SEVERE, "[Main] OpenCV 加载失败，解码功能将不可用", t);
        }

        // ---- 2. 加载界面 ----
        FXMLLoader loader = new FXMLLoader(
                Objects.requireNonNull(Main.class.getResource("/fxml/main.fxml"),
                        "找不到 /fxml/main.fxml"));
        Parent root = loader.load();

        Scene scene = new Scene(root, AppConfig.WINDOW_WIDTH, AppConfig.WINDOW_HEIGHT);
        scene.getStylesheets().add(
                Objects.requireNonNull(Main.class.getResource(AppConfig.CSS_PATH)).toExternalForm());

        stage.setTitle(AppConfig.APP_TITLE);
        stage.setScene(scene);
        stage.setMinWidth(960);
        stage.setMinHeight(640);
        controller = loader.getController();
        stage.setOnHidden(_ -> shutdown());
        stage.show();

        // ---- 3. 后台加载模型与类别表，完成后注入控制器 ----
        startModelLoader();
    }

    private void startModelLoader() {
        Thread t = new Thread(() -> {
            try {
                ClassNames names = ClassNames.load();
                LOGGER.info(() -> "[Main] 类别数 = " + names.size()
                        + "  顺序指纹 sha256[:16] = " + names.fingerprint());
                OrtInference loaded = createEngine();
                if (stopped) {
                    closeEngine(loaded);
                    return;
                }
                Platform.runLater(() -> {
                    if (stopped) {
                        closeEngine(loaded);
                    } else {
                        engine = loaded;
                        controller.setModel(loaded, names);
                    }
                });
            } catch (Exception e) {
                LOGGER.log(Level.SEVERE, "[Main] 模型/类别加载失败", e);
                Platform.runLater(() -> controller.showModelLoadError(e.getMessage()));
            }
        }, "tsd-model-loader");
        t.setDaemon(true);
        t.start();
    }

    /** 引擎所有权通过返回值移交调用方，由 shutdown() 统一关闭 */
    private static OrtInference createEngine() throws OrtException, NoSuchFileException {
        return new OrtInference(null);
    }

    private void shutdown() {
        stopped = true;
        // 先停控制器并等在途 session.run 退出，再关闭引擎，避免关闭正在执行的 native session
        controller.shutdown();
        if (engine != null) {
            closeEngine(engine);
            engine = null;
        }
    }

    private static void closeEngine(OrtInference e) {
        try {
            e.close();
        } catch (Exception ex) {
            LOGGER.log(Level.FINE, "[Main] 关闭推理引擎时发生异常（已忽略）", ex);
        }
    }

    @Override
    public void stop() {
        LOGGER.info("[Main] 退出");
    }

}