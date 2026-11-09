<#
.SYNOPSIS
  TrafficSignRecognition 一键打包 (PowerShell 7 / jpackage)
.EXAMPLE
  .\package_win.ps1 app    # 仅生成应用镜像，不需要 WiX
  .\package_win.ps1 exe    # 生成 exe 安装包，需要 WiX Toolset
.NOTES
  保存为 UTF-8 (PS7 下 BOM 可选)。
  WiX 前置条件（一次性）：
    wix eula accept wix7
    wix extension add -g WixToolset.Util.wixext
    wix extension add -g WixToolset.UI.wixext
  jlink 运行时缺失时自动生成（配方见 [2/4]），gradlew clean 后重跑即可重建。
  $UpgradeUuid 一经固定不可更改，否则后续版本无法原地升级。
#>
param(
    [ValidateSet('app', 'exe')]
    [string]$Type = 'app',
    [string]$Root = $PSScriptRoot
)

$ErrorActionPreference = 'Stop'
# 原生 exe（gradle/jlink/jpackage）退出码由 $LASTEXITCODE 显式检查，不受 EAP 影响
$PSNativeCommandUseErrorActionPreference = $false

$Name        = '交通标志识别'
$InstallDir  = 'TrafficSignRecognition'   # 安装目录与安装包文件名保持英文
$Version     = '1.0.0'                    # 须与 app/build.gradle.kts 的 version 一致
$MainJar     = "app-$Version.jar"
$UpgradeUuid = '395c751d-83d8-4cb2-8175-28f90328796b'  # 固定升级码，保证跨版本原地升级
$Vendor      = 'TSD'
$MainClass   = 'com.tsd.Launcher'
$InputDir    = 'app\build\package\input'
$RuntimeDir  = 'app\build\package\runtime'
$DistDir     = 'app\build\package\dist'
$IconPath    = 'package\icon\app.ico'
$Description = '基于 YOLO26s 的中国交通标志识别系统'
$MenuGroup   = '交通标志识别'

# 优先 JAVA_HOME，缺失时回退 PATH
function Resolve-JdkTool([string]$ToolName) {
    if ($env:JAVA_HOME) {
        $p = Join-Path $env:JAVA_HOME "bin\$ToolName.exe"
        if (Test-Path $p) { return $p }
    }
    $cmd = Get-Command "$ToolName.exe" -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    throw "未找到 $ToolName.exe，请设置 JAVA_HOME 或将 JDK bin 目录加入 PATH。"
}

$savedJavaToolOptions = $env:JAVA_TOOL_OPTIONS

Push-Location $Root
try {
    if (-not (Test-Path 'app\build.gradle.kts')) {
        throw "未在 $Root 找到 app\build.gradle.kts，请用 -Root 指定项目根目录。"
    }

    Write-Host '===== [1/4] 汇总打包输入目录 ====='
    & .\gradlew.bat packageInput --quiet
    if ($LASTEXITCODE -ne 0) { throw "Gradle packageInput 失败 (exit $LASTEXITCODE)" }
    if (-not (Test-Path (Join-Path $InputDir $MainJar))) {
        throw "打包输入中未找到 $MainJar，请核对脚本 `$Version 与 app/build.gradle.kts 的 version 是否一致。"
    }

    # 注意：不可用 $args，那是 PowerShell 自动变量
    $jArgs = @(
        '--name', $Name
        '--app-version', $Version
        '--vendor', $Vendor
        '--description', $Description
        '--copyright', 'Copyright (c) 2026'
        '--input', $InputDir
        '--main-jar', $MainJar
        '--main-class', $MainClass
        '--dest', $DistDir
        '--java-options', '-Dfile.encoding=UTF-8'
        '--java-options', '-Dapp.dir=$APPDIR'   # 单引号：$APPDIR 由 jpackage 启动器展开
        '--java-options', '-Xmx2g'
        '--java-options', '--enable-native-access=ALL-UNNAMED'
    )

    Write-Host '===== [2/4] 检查运行时镜像 ====='
    if (-not (Test-Path (Join-Path $RuntimeDir 'bin\java.exe'))) {
        Write-Host '  [INFO] 未找到 jlink 运行时，自动生成精简镜像 ...'
        # 模块清单由 jdeps 分析打包输入 jar 得出（依赖变更后可重新核对）：
        #   jdeps --multi-release 25 --ignore-missing-deps --print-module-deps <input>\*.jar
        # jdk.unsupported：ONNX Runtime 内部使用 sun.misc.Unsafe；jdk.localedata 仅保留 zh,en
        if (Test-Path $RuntimeDir) { Remove-Item $RuntimeDir -Recurse -Force }  # 清理不完整残留
        $jlink = Resolve-JdkTool 'jlink'
        & $jlink --add-modules java.base,java.desktop,java.logging,java.scripting,jdk.jfr,jdk.unsupported,jdk.localedata --include-locales=zh,en --strip-debug --no-header-files --no-man-pages --compress zip-9 --output $RuntimeDir
        if ($LASTEXITCODE -ne 0) { throw "jlink 失败 (exit $LASTEXITCODE)" }
    }
    Write-Host "  [OK] 使用 jlink 运行时: $RuntimeDir"
    $jArgs += @('--runtime-image', $RuntimeDir)

    Write-Host '===== [3/4] 检查图标 ====='
    if (Test-Path $IconPath) {
        $jArgs += @('--icon', $IconPath)
        Write-Host "  [OK] 图标: $IconPath"
    } else {
        Write-Host '  [WARN] 未找到图标，使用默认图标'
    }

    Write-Host "===== [4/4] 执行 jpackage (type=$Type) ====="
    if ($Type -eq 'exe') {
        $jArgs += @(
            '--type', 'exe'
            '--install-dir', $InstallDir
            '--win-dir-chooser'
            '--win-menu'
            '--win-menu-group', $MenuGroup
            '--win-shortcut'
            '--win-per-user-install'
            '--win-upgrade-uuid', $UpgradeUuid
        )
        if (-not (Get-Command wix.exe -ErrorAction SilentlyContinue)) {
            throw '未找到 wix.exe，请安装 WiX Toolset（JDK 25 支持 4/5/6/7；本机为 7.0.1）'
        }
        $outExe = Join-Path $DistDir "$InstallDir-$Version.exe"
        if (Test-Path $outExe) {
            $prev = Join-Path $DistDir "$Name-$Version-previous.exe"
            Move-Item $outExe $prev -Force
            Write-Host "  [INFO] 已备份旧产物 -> $prev"
        }
        # 中文 MSI 需 Codepage 936：让 jpackage 选中 MsiInstallerStrings_zh_CN.wxl
        $env:JAVA_TOOL_OPTIONS = '-Duser.language=zh -Duser.country=CN'
    } else {
        $jArgs += @('--type', 'app-image')
        if (Test-Path (Join-Path $DistDir $Name)) {
            Write-Host '  [WARN] 同名 app-image 目录已存在，如需重生成请先手动删除'
        }
    }

    $jpackage = Resolve-JdkTool 'jpackage'
    & $jpackage @jArgs
    if ($LASTEXITCODE -ne 0) { throw "jpackage 失败 (exit $LASTEXITCODE)" }

    if ($Type -eq 'exe') {
        # jpackage 按 --name 生成中文安装包名，重命名为英文（外层安装器改名安全）
        Rename-Item (Join-Path $DistDir "$Name-$Version.exe") "$InstallDir-$Version.exe"
        Write-Host "  [OK] 安装包已重命名 -> $InstallDir-$Version.exe"
    }

    Write-Host ''
    Write-Host "===== 打包完成  输出目录: $DistDir ====="
    Get-ChildItem $DistDir -Filter '*.exe' |
        Select-Object Name, @{n = 'MB'; e = { [math]::Round($_.Length / 1MB, 2) } }, LastWriteTime
}
finally {
    if ($Type -eq 'exe') {
        if ($null -ne $savedJavaToolOptions) {
            $env:JAVA_TOOL_OPTIONS = $savedJavaToolOptions
        } else {
            Remove-Item Env:\JAVA_TOOL_OPTIONS -ErrorAction SilentlyContinue
        }
    }
    Pop-Location
}
