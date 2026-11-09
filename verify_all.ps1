<#
.SYNOPSIS
  TrafficSignDetector 全链路自检 (PowerShell 7)
.EXAMPLE
  .\check_all.ps1
  .\check_all.ps1 -NoPause
.NOTES
  假设脚本位于项目根目录；放在别处时用 -Root <项目根目录绝对路径>
#>
param(
    [string]$Root = $PSScriptRoot,
    [switch]$NoPause
)

$ErrorActionPreference = 'Continue'
$script:issues = [System.Collections.Generic.List[string]]::new()

function Section([string]$Title) {
    Write-Host ''
    Write-Host ('=' * 60)
    Write-Host " $Title"
    Write-Host ('=' * 60)
}
function Ok([string]$Message)   { Write-Host "  [OK]   $Message" -ForegroundColor Green }
function Warn([string]$Message) { Write-Host "  [MISS] $Message" -ForegroundColor Yellow; $script:issues.Add($Message) }
function Bad([string]$Message)  { Write-Host "  [FAIL] $Message" -ForegroundColor Red;   $script:issues.Add($Message) }

function Get-ClassNames([string]$Path) {
    # 与 Java ClassNames.readLines() 一致：trim，丢弃空行与 # 注释行
    @(
        Get-Content -LiteralPath $Path -Encoding UTF8 |
            ForEach-Object { $_.Trim() } |
            Where-Object { $_ -and -not $_.StartsWith('#') }
    )
}
function Get-NameFingerprint([string[]]$Names) {
    # 与 convert_tt100k.sha256_of_names / ClassNames.sha256Hex16 一致
    $bytes = [System.Text.Encoding]::UTF8.GetBytes(($Names -join "`n"))
    $hash  = [System.Security.Cryptography.SHA256]::HashData($bytes)
    -join ($hash[0..7] | ForEach-Object { $_.ToString('x2') })
}

Push-Location $Root
try {
    if (-not (Test-Path 'app\build.gradle.kts')) {
        throw "未在 $Root 找到 app\build.gradle.kts，请用 -Root 指定项目根目录。"
    }

    Write-Host '============================================================'
    Write-Host ' TrafficSignDetector 全链路自检'
    Write-Host '============================================================'

    Section '[1] Java 构建'
    & .\gradlew.bat build --quiet
    if ($LASTEXITCODE -ne 0) {
        Bad "Gradle build 失败 (exit $LASTEXITCODE)；去掉 --quiet 重跑可查看详细错误"
    } else { Ok '构建通过' }

    Section '[2] 模型清单 (manifest)'
    $manifestPath = 'app\src\main\resources\models\model_manifest.json'
    $nc = 0
    $manifestFingerprint = $null
    $manifestModelSha = $null
    if (Test-Path $manifestPath) {
        $manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $nc = [int]$manifest.num_classes
        $manifestFingerprint = $manifest.class_names_sha256_16
        $manifestModelSha = $manifest.model_sha256
        Ok "manifest: num_classes=$nc  类别指纹=$manifestFingerprint"
    } else {
        Bad "$manifestPath 不存在，请先执行 scripts\export_onnx.py"
    }

    Section '[3] 类别文件'
    $javaFingerprint = $null
    $javaCount = 0
    if ($nc -gt 0) {
        # 类别数与数据集标签由 manifest 驱动，不再硬编码 45/48
        $tag = "tt100k$nc"
        $javaNames = "app\src\main\resources\config\$tag.names"
        $dataNames = "data\${tag}_yolo\$tag.names"

        if (Test-Path $javaNames) {
            $j = Get-ClassNames $javaNames
            $javaCount = $j.Count
            $javaFingerprint = Get-NameFingerprint $j
            Ok "Java 端有效类别数: $javaCount (manifest 期望 $nc)"
            if ($javaCount -ne $nc) { Bad "Java 端类别数 $javaCount 与 manifest num_classes=$nc 不一致" }
            if ($manifestFingerprint -and $javaFingerprint -cne $manifestFingerprint) {
                Bad "Java 端类别指纹与 manifest class_names_sha256_16 不一致"
            }
        } else { Warn "$javaNames 不存在" }

        if (Test-Path $dataNames) {
            $d = Get-ClassNames $dataNames
            Ok "数据端有效类别数: $($d.Count) (manifest 期望 $nc)"
            if ($d.Count -ne $nc) { Bad "数据端类别数 $($d.Count) 与 manifest num_classes=$nc 不一致" }
            if ($null -ne $javaFingerprint) {
                if (($j -join "`n") -ceq ($d -join "`n")) { Ok '两端类别顺序一致' }
                else { Bad 'Java 端与数据端类别顺序或内容不一致' }
            }
        } else { Warn "$dataNames 不存在" }
    } else { Warn 'manifest 缺失或 num_classes 非法，跳过类别文件检查' }

    Section '[4] 模型文件'
    $model = 'app\src\main\resources\models\best.onnx'
    if (Test-Path $model) {
        $mi = Get-Item -LiteralPath $model
        Ok ("best.onnx 已就位 ({0:N1} MB)" -f ($mi.Length / 1MB))
        if ($manifestModelSha) {
            $sha = (Get-FileHash -LiteralPath $model -Algorithm SHA256).Hash.ToLower()
            if ($sha -ceq $manifestModelSha.ToLower()) { Ok 'best.onnx SHA-256 与 manifest 一致' }
            else { Bad 'best.onnx SHA-256 与 manifest model_sha256 不一致：模型被替换但未重新导出' }
        }
    } else { Warn 'best.onnx 不存在，请执行 scripts\export_onnx.py' }

    Section '[5] OpenCV native / jar'
    foreach ($p in @('libs\win64\opencv_java4140.dll', 'libs\opencv-4140.jar')) {
        if (Test-Path $p) { Ok "$p 已就位" } else { Warn "$p 不存在" }
    }

    Section '[6] ONNX 输出格式说明'
    $fmt = 'scripts\onnx_output_format.txt'
    if (Test-Path $fmt) {
        Ok 'onnx_output_format.txt 已生成，shape 相关行:'
        $hits = @(Select-String -LiteralPath $fmt -SimpleMatch 'shape:')
        if ($hits.Count -eq 0) { Warn '文件中没有 shape: 行' }
        $hits | Select-Object -First 10 | ForEach-Object { Write-Host "      $($_.Line.Trim())" }
    } else { Warn 'onnx_output_format.txt 不存在，请执行 scripts\export_onnx.py' }

    Section '[7] Python 侧指纹与 Java 端自动比对'
    if ($nc -gt 0) {
        $tag = "tt100k$nc"
        $order = "data\${tag}_yolo\${tag}_class_order.json"
        if (Test-Path $order) {
            $meta = Get-Content -LiteralPath $order -Raw -Encoding UTF8 | ConvertFrom-Json
            Ok ("Python 端指纹: {0}  类别数: {1}" -f $meta.sha256_16, $meta.num_classes)
            if ($null -ne $javaFingerprint) {
                Ok ("Java 端指纹  : {0}  类别数: {1}" -f $javaFingerprint, $javaCount)
                if ($javaFingerprint -ceq $meta.sha256_16 -and $javaCount -eq $meta.num_classes) {
                    Ok '两端类别顺序与数量一致 (sha256[:16] 相同)'
                } else {
                    Bad '两端指纹或类别数不一致：重跑 convert_tt100k.py 并同步 names 到 app\src\main\resources\config\'
                }
            } else { Warn 'Java 端 names 缺失，无法比对' }
        } else { Warn "$order 不存在，请执行 scripts\convert_tt100k.py" }
    } else { Warn 'manifest 缺失，跳过 Python 侧指纹比对' }

    Section '[8] Python / uv 工具链'
    if (Get-Command uv -ErrorAction SilentlyContinue) {
        Ok ("uv: {0}" -f ((& uv --version 2>&1) | Select-Object -First 1))
    } else { Warn 'uv 不在 PATH' }
    if (Test-Path 'scripts\pyproject.toml') { Ok 'scripts\pyproject.toml 已就位' }
    else { Warn 'scripts\pyproject.toml 不存在' }

    Section '汇总'
    if ($script:issues.Count -eq 0) {
        Write-Host '  全部检查通过' -ForegroundColor Green
    } else {
        Write-Warning "存在 $($script:issues.Count) 项异常:"
        $script:issues | ForEach-Object { Write-Host "    - $_" -ForegroundColor Yellow }
    }
    Write-Host ''
    Write-Host '  提示: 运行 .\gradlew.bat run 后控制台会打印 Java 端指纹；'
    Write-Host '        本脚本已在 [7] 自动完成与 Python 端的比对，无需人工核对。'
}
finally {
    Pop-Location
}

if (-not $NoPause) { Read-Host '按 Enter 键继续...' }
