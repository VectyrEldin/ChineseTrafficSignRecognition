package com.tsd.ui;

import com.tsd.config.AppConfig;
import com.tsd.core.OpenCVLoader;
import com.tsd.core.OrtInference;
import com.tsd.core.PostProcessor;
import com.tsd.core.TiledDetector;
import com.tsd.core.YoloPreprocessor;
import com.tsd.model.ClassDescriptions;
import com.tsd.model.ClassNames;
import com.tsd.model.Detection;
import com.tsd.util.ImageUtils;
import javafx.application.Platform;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.concurrent.Task;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.control.*;
import javafx.scene.control.cell.PropertyValueFactory;
import javafx.scene.image.Image;
import javafx.scene.input.Dragboard;
import javafx.scene.input.TransferMode;
import javafx.scene.layout.StackPane;
import javafx.stage.FileChooser;
import javafx.stage.Stage;
import org.opencv.core.Mat;
import org.opencv.videoio.VideoCapture;

import java.io.File;
import java.nio.file.Path;
import java.util.List;
import java.util.ResourceBundle;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.net.URL;

/**
 * 主界面控制器。
 * 功能：打开图片 / 拖拽 / 摄像头拍照 → 推理 → 画框 + 结果表格 + 置信度阈值调节。
 */
public class MainController implements Initializable {

    // ==================== FXML 注入 ====================
    @FXML private StackPane canvasPane;
    @FXML private DetectionCanvas canvas;
    @FXML private Button openButton;
    @FXML private Button cameraButton;
    @FXML private Button captureButton;
    @FXML private Button clearButton;
    @FXML private Slider confSlider;
    @FXML private Label confLabel;
    @FXML private Label statusLabel;
    @FXML private CheckBox labelCheckBox;
    @FXML private TableView<Detection> resultTable;
    @FXML private TableColumn<Detection, String> colClass;
    @FXML private TableColumn<Detection, String> colDesc;
    @FXML private TableColumn<Detection, Number> colId;
    @FXML private TableColumn<Detection, String> colConf;
    @FXML private TableColumn<Detection, String> colBox;

    // 日志记录器
    private static final Logger LOGGER = Logger.getLogger(MainController.class.getName());

    // ==================== 运行时状态 ====================
    // 以下运行时状态仅在 FX 线程读写；worker 计算结果经 Platform.runLater 发布回 FX 线程后赋值
    // inference/classNames 由应用层（Main）加载并经 setModel 注入，所有权归应用层
    private OrtInference inference;
    private ClassNames classNames;
    private ClassDescriptions descriptions;
    private Mat currentMat;                 // 当前展示的原图（BGR）
    // 以滑块最低阈值推理一次的完整结果；滑块拖动只做过滤，不再重推理。
    // 仅在 FX 线程读写（runDetection 的所有调用点与 Task 回调均在 FX 线程）
    private List<Detection> cachedDetections;
    private Mat cachedFor;                  // 缓存对应的原图（引用相等判定）
    private final ObservableList<Detection> results = FXCollections.observableArrayList();
    private final ExecutorService worker = Executors.newSingleThreadExecutor(r -> {
        Thread t = new Thread(r, "tsd-infer");
        t.setDaemon(true);
        return t;
    });
    private static final String BUTTON_PRESSED_STYLE_CLASS = "button-pressed";

    // 摄像头
    private final AtomicBoolean cameraRunning = new AtomicBoolean(false);
    private final AtomicBoolean previewPending = new AtomicBoolean(false); // 至多保留一帧待渲染
    private VideoCapture camera;
    private ScheduledExecutorService cameraScheduler;

    // ==================== 初始化 ====================

    /** 模型就绪后由应用层在 FX 线程注入；引擎所有权归应用层，本控制器只借用不关闭 */
    public void setModel(OrtInference engine, ClassNames names) {
        this.inference = engine;
        this.classNames = names;
        statusLabel.setText("就绪 | 类别 " + names.size() + " 类 | 指纹 "
                + names.fingerprint() + " | " + OpenCVLoader.status());
    }

    /** 模型/类别加载失败时由应用层在 FX 线程调用 */
    public void showModelLoadError(String message) {
        statusLabel.setText("模型/类别加载失败：" + message);
    }

    @Override
    public void initialize(URL location, ResourceBundle resources) {
        // 中文描述表体量很小，同步加载即可，避免 cellValueFactory 读到 null
        descriptions = ClassDescriptions.load();

        // 按钮事件统一在代码中绑定（FXML 不再声明 onAction），避免双重来源互相覆盖
        openButton.setOnAction(_ -> onOpenImage());
        clearButton.setOnAction(_ -> onClear());
        cameraButton.setOnAction(_ -> onToggleCamera());
        captureButton.setOnAction(_ -> onCapture());

        // 画布尺寸跟随父容器
        canvas.widthProperty().bind(canvasPane.widthProperty());
        canvas.heightProperty().bind(canvasPane.heightProperty());

        // 初始化表格与控件（已提取为独立方法以降低认知复杂度）
        initResultTable();
        initControls();

        // 状态栏
        statusLabel.setText(OpenCVLoader.isLoaded()
                ? "就绪 | " + OpenCVLoader.status()
                : "警告：OpenCV 未加载");

        // ---- 模型与类别表由应用层（Main）后台加载，经 setModel 注入 ----

        setupDragAndDrop();
        openButton.setCursor(javafx.scene.Cursor.HAND);
        cameraButton.setCursor(javafx.scene.Cursor.HAND);
        captureButton.setCursor(javafx.scene.Cursor.HAND);
        clearButton.setCursor(javafx.scene.Cursor.HAND);
    }

    /** 初始化结果表格的列与渲染 */
    private void initResultTable() {
        colId.setCellValueFactory(new PropertyValueFactory<>("classId"));
        colClass.setCellValueFactory(d ->
                new javafx.beans.property.SimpleStringProperty(d.getValue().getLabel()));
        colDesc.setCellValueFactory(d ->
                new javafx.beans.property.SimpleStringProperty(
                        descriptions.descriptionOf(d.getValue().getLabel())));
        // 中文描述较长，列宽不足时通过 Tooltip 展示全文
        colDesc.setCellFactory(_ -> new TableCell<>() {
            @Override
            protected void updateItem(String item, boolean empty) {
                super.updateItem(item, empty);
                if (empty || item == null) {
                    setText(null);
                    setTooltip(null);
                } else {
                    setText(item);
                    Tooltip tip = new Tooltip(item);
                    tip.setWrapText(true);
                    tip.setMaxWidth(360);
                    setTooltip(tip);
                }
            }
        });
        colConf.setCellValueFactory(d ->
                new javafx.beans.property.SimpleStringProperty(d.getValue().getConfidenceText()));
        colBox.setCellValueFactory(d ->
                new javafx.beans.property.SimpleStringProperty(d.getValue().getBoxText()));

        // 使用独立的单元格类，消除内联 Lambda 与条件分支的复杂度
        colConf.setCellFactory(_ -> new ConfidenceTableCell());
        resultTable.setItems(results);

        // 将行工厂设置提取到独立方法，进一步降低认知复杂度
        initResultTableRowFactory();
    }

    /** 置信度列单元格（处理文本与颜色渲染） */
    private static class ConfidenceTableCell extends TableCell<Detection, String> {
        @Override
        protected void updateItem(String item, boolean empty) {
            super.updateItem(item, empty);
            if (empty || item == null) { setText(null); setStyle(""); return; }
            setText(item);
            double v = Double.parseDouble(item.replace("%", ""));
            javafx.scene.paint.Color textColor;
            if (v >= 70) {
                textColor = javafx.scene.paint.Color.web("#4ade80");
            } else if (v >= 40) {
                textColor = javafx.scene.paint.Color.web("#fbbf24");
            } else {
                textColor = javafx.scene.paint.Color.web("#f87171");
            }
            setTextFill(textColor);
        }
    }

    /** 初始化结果表格的行工厂（奇数行与选中行样式） */
    private void initResultTableRowFactory() {
        resultTable.setRowFactory(_ -> {
            TableRow<Detection> row = new TableRow<>();

            // 1. 奇数行样式
            row.indexProperty().addListener((_, _, newIdx) -> {
                row.getStyleClass().remove("table-row-odd");
                if (newIdx.intValue() % 2 != 0) {
                    row.getStyleClass().add("table-row-odd");
                }
            });

            // 2. 选中行样式
            row.selectedProperty().addListener((_, _, isNowSelected) -> {
                if (Boolean.TRUE.equals(isNowSelected)) {
                    row.getStyleClass().add("table-row-selected");
                } else {
                    row.getStyleClass().remove("table-row-selected");
                }
            });

            return row;
        });
    }

    /** 初始化置信度滑块与复选框控件 */
    private void initControls() {
        confSlider.setMin(0.05);
        confSlider.setMax(0.90);
        confSlider.setValue(AppConfig.DEFAULT_CONF_THRESHOLD);
        confLabel.setText(String.format("%.2f", AppConfig.DEFAULT_CONF_THRESHOLD));
        confSlider.valueProperty().addListener((_, _, nv) -> {
            confLabel.setText(String.format("%.2f", nv.doubleValue()));
            // 已有缓存则按新阈值过滤（毫秒级）；检测进行中不重复提交，
            // Task 完成时会按滑块最新值刷新
            applyConfFilter();
        });

        labelCheckBox.setSelected(true);
        labelCheckBox.selectedProperty().addListener(
                (_, _, nv) -> canvas.setShowLabel(nv));

        setupButtonPressedStyle(openButton);
        setupButtonPressedStyle(cameraButton);
        setupButtonPressedStyle(captureButton);
        setupButtonPressedStyle(clearButton);
    }

    /** 通过代码动态管理按钮按下样式，替代 CSS 伪类 */
    private void setupButtonPressedStyle(Button button) {
        button.setOnMousePressed(_ -> {
            if (!button.getStyleClass().contains(BUTTON_PRESSED_STYLE_CLASS)) {
                button.getStyleClass().add(BUTTON_PRESSED_STYLE_CLASS);
            }
        });
        button.setOnMouseReleased(_ -> button.getStyleClass().remove(BUTTON_PRESSED_STYLE_CLASS));
        button.setOnMouseExited(_ -> button.getStyleClass().remove(BUTTON_PRESSED_STYLE_CLASS)); // 防止鼠标移出时卡住
    }

    // ==================== 交互 ====================
    @FXML
    private void onOpenImage() {
        FileChooser fc = new FileChooser();
        fc.setTitle("选择交通标志图片");
        fc.getExtensionFilters().addAll(
                new FileChooser.ExtensionFilter("图片", "*.jpg", "*.jpeg", "*.png", "*.bmp"),
                new FileChooser.ExtensionFilter("所有文件", "*.*"));
        File f = fc.showOpenDialog(stage());
        if (f != null) loadAndDetect(f.toPath());
    }

    @FXML
    private void onClear() {
        results.clear();
        cachedDetections = null;
        cachedFor = null;
        if (currentMat != null) { currentMat.release(); currentMat = null; }
        canvas.clear();
        statusLabel.setText("已清空");
    }

    @FXML
    private void onToggleCamera() {
        if (cameraRunning.get()) stopCamera(); else startCamera();
    }

    @FXML
    private void onCapture() {
        if (!cameraRunning.get()) { statusLabel.setText("请先开启摄像头"); return; }
        Mat frame = new Mat();
        if (camera != null && camera.read(frame) && !frame.empty()) {
            setCurrentMat(frame);
            runDetection(frame);
        } else {
            frame.release();
            statusLabel.setText("抓帧失败");
        }
    }

    private void setupDragAndDrop() {
        canvasPane.setOnDragOver(e -> {
            if (e.getGestureSource() != canvasPane && e.getDragboard().hasFiles()) {
                e.acceptTransferModes(TransferMode.COPY);
            }
            e.consume();
        });
        canvasPane.setOnDragDropped(e -> {
            Dragboard db = e.getDragboard();
            if (db.hasFiles()) {
                File f = db.getFiles().getFirst();
                loadAndDetect(f.toPath());
                e.setDropCompleted(true);
            }
            e.consume();
        });
    }

    private Stage stage() {
        return (Stage) canvasPane.getScene().getWindow();
    }

    // ==================== 核心流程 ====================
    /** 读图 → 显示 → 推理 */
    private void loadAndDetect(Path path) {
        statusLabel.setText("读取中: " + path.getFileName());
        worker.submit(() -> {
            try {
                if (!OpenCVLoader.isLoaded()) throw new IllegalStateException("OpenCV 未加载");
                Mat mat = ImageUtils.readImageSafe(path);   // 中文路径安全
                if (mat.empty()) throw new IllegalStateException("无法解码该图片");

                // runDetection 读取滑块/更新状态栏，必须回到 FX 线程执行；
                // 推理本身仍由其中的 Task 提交到 worker
                Platform.runLater(() -> {
                    setCurrentMat(mat);
                    runDetection(mat);
                });
            } catch (Exception e) {
                Platform.runLater(() -> statusLabel.setText("读取失败：" + e.getMessage()));
            }
        });
    }

    private void setCurrentMat(Mat mat) {
        if (currentMat != null && currentMat != mat) currentMat.release();
        if (cachedFor != mat) {
            cachedDetections = null;
            cachedFor = null;
        }
        currentMat = mat;
        Image img = ImageUtils.matToImage(mat);
        canvas.setImage(img);
        canvas.draw();
    }

    /** 按当前阈值过滤缓存结果并刷新展示（FX 线程，毫秒级，不重推理） */
    private void applyConfFilter() {
        if (cachedDetections == null || currentMat == null || cachedFor != currentMat) {
            return;
        }
        float conf = (float) confSlider.getValue();
        List<Detection> ds = cachedDetections.stream()
                .filter(d -> d.getConfidence() >= conf)
                .toList();
        results.setAll(ds);
        canvas.setDetections(ds);
        statusLabel.setText(
                String.format("检测完成：%d 个目标 | 阈值 %.2f", ds.size(), conf));
    }

    /** 推理（在 worker 线程内执行，避免卡 UI）；本方法必须在 FX 线程调用 */
    private void runDetection(Mat mat) {
        // 字段仅限 FX 线程访问；取局部快照经闭包传给 worker（executor.submit 保证安全发布）
        OrtInference engine = inference;
        ClassNames names = classNames;
        if (engine == null || names == null) {
            statusLabel.setText("模型尚未就绪，请稍候");
            return;
        }
        // 以滑块最低阈值推理并缓存，之后滑块只过滤；
        // 提高阈值等价于先过滤再截断，低阈值缓存不会丢失高分框
        float minConf = (float) confSlider.getMin();

        // FX 线程可能在本任务执行期间 release currentMat（换图/清空），
        // 推理必须持有私有副本，否则 native 内存 use-after-free 直接崩溃
        Mat owned = mat.clone();
        Task<List<Detection>> task = new Task<>() {
            @Override
            protected List<Detection> call() throws Exception {
                try {
                    long t0 = System.currentTimeMillis();
                    String mode;
                    List<Detection> ds;
                    if (AppConfig.TILED_INFERENCE) {
                        mode = "tiled";
                        ds = TiledDetector.detect(owned, engine, minConf, names.size());
                    } else {
                        mode = "single";
                        YoloPreprocessor.Result prep =
                                YoloPreprocessor.preprocess(owned, AppConfig.INPUT_SIZE);
                        OrtInference.Tensor out = engine.infer(prep);
                        ds = PostProcessor.parse(out, prep, minConf, names.size(),
                                AppConfig.MODEL_CONTRACT);
                    }
                    long elapsed = System.currentTimeMillis() - t0;
                    // 填类别名
                    for (Detection d : ds) d.setLabel(names.nameOf(d.getClassId()));
                    LOGGER.info(() -> String.format(
                            "[Detect] mode=%s imgsz=%d 总耗时 %dms | 框数 %d",
                            mode, AppConfig.INPUT_SIZE, elapsed, ds.size()));
                    return ds;
                } finally {
                    owned.release();
                }
            }
        };

        task.setOnRunning(_ -> statusLabel.setText("检测中..."));
        task.setOnSucceeded(_ -> {
            cachedDetections = task.getValue();
            cachedFor = mat;
            applyConfFilter();

            if (cachedFor == currentMat && LOGGER.isLoggable(Level.INFO)) {
                StringBuilder sb = new StringBuilder("[RESULT] ");
                for (Detection d : cachedDetections) {
                    sb.append(String.format(
                            "cls=%d %s conf=%.4f box=[%.1f,%.1f,%.1f,%.1f] | ",
                            d.getClassId(), d.getLabel(), d.getConfidence(),
                            d.getX1(), d.getY1(), d.getX2(), d.getY2()
                    ));
                }
                LOGGER.info(sb.toString());
            }
        });
        task.setOnFailed(_ -> {
            Throwable ex = task.getException();
            LOGGER.log(java.util.logging.Level.SEVERE, "检测失败", ex);
            statusLabel.setText("检测失败：" + ex.getMessage());
        });

        worker.submit(task);
    }

    // ==================== 摄像头 ====================
    private void startCamera() {
        if (!OpenCVLoader.isLoaded()) { statusLabel.setText("OpenCV 未加载，无法使用摄像头"); return; }
        camera = new VideoCapture(0);
        if (!camera.isOpened()) {
            statusLabel.setText("无法打开摄像头（索引 0）");
            camera.release();
            return;
        }
        cameraRunning.set(true);
        previewPending.set(false);
        cameraButton.setText("关闭摄像头");
        captureButton.setDisable(false);
        statusLabel.setText("摄像头已开启，点击『拍照检测』抓取当前帧");

        // 使用 ScheduledExecutorService 替代 while 循环 + Thread.sleep，消除忙等待警告
        cameraScheduler = Executors.newSingleThreadScheduledExecutor(r -> {
            Thread t = new Thread(r, "tsd-camera");
            t.setDaemon(true);
            return t;
        });
        cameraScheduler.scheduleAtFixedRate(() -> {
            Mat frame = new Mat(); // 每次执行独立创建，避免跨帧复用
            try {
                // UI 繁忙时至多保留一帧待渲染，避免 Platform.runLater 任务积压
                if (camera.read(frame) && !frame.empty() && previewPending.compareAndSet(false, true)) {
                    Image img = ImageUtils.matToImage(frame);
                    Platform.runLater(() -> {
                        try {
                            canvas.setImage(img);
                            canvas.draw();
                        } finally {
                            previewPending.set(false);
                        }
                    });
                }
            } finally {
                frame.release(); // 确保资源立即释放
            }
        }, 0, 33, TimeUnit.MILLISECONDS);
    }

    private void stopCamera() {
        cameraRunning.set(false);
        // 优雅关闭调度器，中断可能的等待
        if (cameraScheduler != null) {
            cameraScheduler.shutdownNow();
            cameraScheduler = null;
        }
        if (camera != null) { camera.release(); camera = null; }
        cameraButton.setText("开启摄像头");
        captureButton.setDisable(true);
        statusLabel.setText("摄像头已关闭");
    }

    public void shutdown() {
        stopCamera();
        worker.shutdownNow();
        // 等在途 session.run 退出后返回，应用层（Main）随后 close 引擎才不会
        // 关闭正在执行的 native session 导致崩溃
        try {
            if (!worker.awaitTermination(2, TimeUnit.SECONDS)) {
                LOGGER.warning("[Shutdown] worker 未在 2s 内终止");
            }
        } catch (InterruptedException _) {
            Thread.currentThread().interrupt();
        }
    }


}