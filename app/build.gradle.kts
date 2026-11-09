import org.gradle.api.artifacts.Configuration
import org.gradle.api.file.DuplicatesStrategy
import org.gradle.api.tasks.Copy
import org.gradle.api.tasks.JavaExec
import org.gradle.api.tasks.compile.JavaCompile
import org.gradle.api.tasks.testing.Test
import org.gradle.jvm.toolchain.JavaLanguageVersion

plugins {
    java
    application
}

val opencvVersion = "4140"
val onnxRuntimeVersion = "1.31.1"
val junitVersion = "6.1.3"
val javafxVersion = "25.0.5"
val annotationsVersion = "26.1.0"

group = "com.tsd"
version = "1.0.0"

// ===== JavaFX 平台 classifier（含 native 库的 jar）=====
// 用 Provider 读取系统属性：配置缓存会把它登记为构建输入，换机器/换系统自动失效
// 显式声明为 String
val os: String = providers.systemProperty("os.name").get().lowercase()
val arch: String = providers.systemProperty("os.arch").get().lowercase()

val javafxPlatform: String = when {
    os.contains("win") -> "win"
    os.contains("mac") || os.contains("darwin") ->
        if (arch.contains("aarch64") || arch.contains("arm64")) "mac-aarch64" else "mac"
    arch.contains("aarch64") || arch.contains("arm64") -> "linux-aarch64"
    else -> "linux"
}
// Windows 上固定写 val javafxPlatform = "win" 也可以

// 独立 configuration：只装 JavaFX 模块 jar，绝不进 runtimeClasspath
val javafxSdk: Configuration = configurations.create("javafxSdk") {
    isCanBeConsumed = false
    isCanBeResolved = true
    isTransitive = false
}

// ===== JDK 25 =====
java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(25))
    }
    // 本项目非模块化，关掉 Gradle 的自动 module-path 推导，完全由下面手动控制
    modularity.inferModulePath.set(false)
}

// 仓库统一由 settings.gradle.kts 的 dependencyResolutionManagement 声明

dependencies {
    // JavaFX：只进 javafxSdk（编译 + module-path），坐标形如 org.openjfx:javafx-controls:25.0.5:win
    // 显式列全 4 个，因为 --add-modules 解析时 module-path 上必须有 base/graphics 实体 jar
    listOf("javafx-base", "javafx-graphics", "javafx-controls", "javafx-fxml").forEach { m ->
        add("javafxSdk", "org.openjfx:$m:$javafxVersion:$javafxPlatform")
    }

    implementation(files(rootProject.file("libs/opencv-$opencvVersion.jar")))
    implementation("com.microsoft.onnxruntime:onnxruntime:$onnxRuntimeVersion")

    testImplementation("org.junit.jupiter:junit-jupiter:$junitVersion")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher:$junitVersion")

    compileOnly("org.jetbrains:annotations:$annotationsVersion")
}

repositories {
    maven("https://maven.aliyun.com/repository/public/")
    mavenLocal()
    mavenCentral()
}

// 让 javac 看得见 JavaFX API；只影响 compileClasspath，不会进 run 的 classpath
configurations.named("compileClasspath") {
    extendsFrom(javafxSdk)
}
// ImageUtilsTest 等直接构造 javafx.scene.image.Image，测试编译/运行期同样需要
configurations.named("testCompileClasspath") {
    extendsFrom(javafxSdk)
}
configurations.named("testRuntimeClasspath") {
    extendsFrom(javafxSdk)
}

application {
    mainClass.set("com.tsd.Main")
    applicationDefaultJvmArgs = listOf(
        "--enable-native-access=ALL-UNNAMED",
        "-Dfile.encoding=UTF-8"
    )
}

tasks.withType<JavaCompile>().configureEach {
    options.encoding = "UTF-8"
    options.compilerArgs.add("-parameters")
}

tasks.withType<Test>().configureEach {
    useJUnitPlatform()
    jvmArgs("--enable-native-access=ALL-UNNAMED")
    systemProperty("project.root", rootProject.projectDir.absolutePath)
    // OpenCVLoader 主路径需要 app.dir/libs/<平台>/opencv_java4140.dll
    systemProperty("app.dir", rootProject.projectDir.absolutePath)
    testLogging {
        events("passed", "skipped", "failed")
    }
}

// ===== run：全部在配置阶段求值，配置缓存友好 =====
tasks.named<JavaExec>("run") {
    dependsOn("processResources")
    systemProperty("app.dir", rootProject.projectDir.absolutePath)

    // files(javafxSdk).asPath 在配置阶段算成 String，写进 jvmArgs（List<String>，可序列化）
    // 主类 + opencv + onnxruntime 留在 classpath，JavaFX 单独走 module-path
    jvmArgs(
        "--module-path", files(javafxSdk).asPath,
        "--add-modules", "javafx.controls,javafx.fxml"
    )
}

// ===== jpackage 依赖汇总：必须补上 JavaFX =====
val copyRuntimeDeps: TaskProvider<Copy> = tasks.register<Copy>("copyRuntimeDeps") {
    group = "build"
    description = "Copies runtime dependencies into a single directory for jpackage."
    from(configurations.runtimeClasspath)
    from(javafxSdk)
    into(layout.buildDirectory.dir("libs-runtime"))
}

tasks.jar {
    manifest {
        //attributes("Main-Class" to "com.tsd.Main")
        attributes["Main-Class"] = "com.tsd.Launcher"
    }
    duplicatesStrategy = DuplicatesStrategy.EXCLUDE
    // 模型由 packageInput 的外挂 models/ 目录提供（OrtInference 优先读外挂），
    // jar 内不再重复携带；manifest 契约文件必须保留在 jar 内。
    // 开发期 gradlew run 走 processResources 输出目录，不受此排除影响。
    exclude("models/*.onnx", "models/*.bak")
}

// ============================================================
// 打包准备：把所有 jar 与外挂资源汇总到 build/package/input/
// jpackage 会把 --input 目录下的【全部内容】复制进安装目录的 app/ 里，
// 所以模型、类别文件、OpenCV native 一股脑放进去即可
// ============================================================

// ============================================================
// 打包准备：把 jar、依赖与外挂资源汇总到 build/package/input/
// 所有输入均在配置阶段声明，无执行期 Project / Task 访问
// ============================================================

tasks.register<Copy>("packageInput") {
    group = "distribution"
    description = "Aggregates all inputs required by jpackage into the --input directory."

    // 主 jar：Provider 自带任务依赖，无需显式 dependsOn
    from(tasks.named<Jar>("jar").flatMap { it.archiveFile })

    // 运行时依赖 + JavaFX 模块 jar
    from(configurations.runtimeClasspath)
    from(javafxSdk)

    // 外挂资源（libs 在根项目，必须显式 rootProject.file）
    from(rootProject.file("libs")) {
        include("win64/**", "linux64/**", "mac64/**", "mac-arm64/**")
        into("libs")
    }
    from("src/main/resources/models") { into("models") }
    from("src/main/resources/config") { into("config") }

    into(layout.buildDirectory.dir("package/input"))

    // 如需打印，只能用不捕获任何脚本对象的纯字符串常量
    doLast { println("打包输入目录: app/build/package/input") }
}