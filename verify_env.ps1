# TrafficSignDetector 环境自检脚本 (PowerShell)
# 如果提示无法运行，请先执行：Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
$ErrorActionPreference = "Continue"

# 设置 UTF-8 编码
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "============================================================"
Write-Host "  TrafficSignDetector 环境自检"
Write-Host "============================================================"

Write-Host ""
Write-Host "[1/6] JDK"
java -version 2>&1
if ($LASTEXITCODE -ne 0) { Write-Host "  [FAIL] java 不可用" } else { Write-Host "  [OK]" }

Write-Host ""
Write-Host "[2/6] JAVA_HOME = $env:JAVA_HOME"
if (-not $env:JAVA_HOME) { Write-Host "  [WARN] JAVA_HOME 未设置" } else { Write-Host "  [OK]" }

Write-Host ""
Write-Host "[3/6] Gradle Wrapper"
& .\gradlew.bat -v 2>&1 | Select-String -Pattern "Gradle", "JVM"
if ($LASTEXITCODE -ne 0) { Write-Host "  [FAIL] gradlew 不可用" } else { Write-Host "  [OK]" }

Write-Host ""
Write-Host "[4/6] uv"
uv --version 2>&1
if ($LASTEXITCODE -ne 0) { Write-Host "  [FAIL] uv 不可用" } else { Write-Host "  [OK]" }

Write-Host ""
Write-Host "[5/6] Python (uv 管理的 3.14)"
if (Test-Path "scripts") {
    Push-Location scripts
    uv run python -c "import sys;print('  Python', sys.version)" 2>&1
    Pop-Location
} else {
    Write-Host "  [WARN] scripts 目录不存在，跳过 Python 检查"
}

Write-Host ""
Write-Host "[6/6] NVIDIA Driver / CUDA"
nvidia-smi --query-gpu=name,driver_version --format=csv 2>&1
nvidia-smi 2>&1 | Select-String -Pattern "CUDA Version"

Write-Host ""
Write-Host "============================================================"
Write-Host "  检查 libs\win64\opencv_java4140.dll 是否存在:"
if (Test-Path "libs\win64\opencv_java4140.dll") {
    Write-Host "  [OK] OpenCV native 已就位"
} else {
    Write-Host "  [MISS] 请复制 opencv_java4140.dll"
}
if (Test-Path "libs\opencv-4140.jar") {
    Write-Host "  [OK] OpenCV jar 已就位"
} else {
    Write-Host "  [MISS] 请复制 opencv-4140.jar"
}
if ($Error.Count -gt 0) {
    Write-Host ""
    Write-Warning "检查过程中发生了一些错误，请查看上方输出。"
}
Read-Host -Prompt "按 Enter 键继续..."