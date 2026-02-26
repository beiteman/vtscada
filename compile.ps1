$ErrorActionPreference = "Stop"

Write-Host "============================================================"
Write-Host "  VTScada Extension Build Script (PowerShell)"
Write-Host "============================================================"
Write-Host ""

function Require-Tool($name) {
    if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
        Write-Error "Required tool '$name' not found in PATH."
    }
}

function Download-File($dir, $url, $fileName) {
    if (-not $fileName) { return }

    $path = Join-Path $dir $fileName

    if (-not (Test-Path $path)) {
        Write-Host "Downloading $fileName ..."
        Invoke-WebRequest -Uri $url -OutFile $path
        Write-Host "Downloaded $fileName"
    }
    else {
        Write-Host "$fileName already exists."
    }
}

function Download-Model($dir, $files) {
    if (-not (Test-Path $dir)) {
        Write-Host "Creating directory $dir"
        New-Item -ItemType Directory -Path $dir | Out-Null
    }

    foreach ($file in $files) {
        Download-File $dir $file.url $file.name
    }
}

# ------------------------------------------------------------
# Tool checks
# ------------------------------------------------------------
Require-Tool "node"
Require-Tool "npm"
Require-Tool "npx"

# ------------------------------------------------------------
# Model definitions
# ------------------------------------------------------------
$enModelDir = "resource/all-MiniLM-L6-v2"
$zhModelDir = "resource/text2vec-base-chinese"

$enFiles = @(
    @{ url="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx"; name="model.onnx" },
    @{ url="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer_config.json"; name="tokenizer_config.json" },
    @{ url="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json"; name="tokenizer.json" }
)

$zhFiles = @(
    @{ url="https://huggingface.co/shibing624/text2vec-base-chinese/resolve/main/onnx/model.onnx"; name="model.onnx" },
    @{ url="https://huggingface.co/shibing624/text2vec-base-chinese/resolve/main/onnx/tokenizer_config.json"; name="tokenizer_config.json" },
    @{ url="https://huggingface.co/shibing624/text2vec-base-chinese/resolve/main/onnx/tokenizer.json"; name="tokenizer.json" }
)

# ------------------------------------------------------------
# Download models
# ------------------------------------------------------------
Download-Model $enModelDir $enFiles
Download-Model $zhModelDir $zhFiles

# ------------------------------------------------------------
# Build index if needed
# ------------------------------------------------------------
$indexFiles = @(
    "resource/index.en.json",
    "resource/index.zh-cn.json",
    "resource/index.zh-tw.json"
)

$missing = $false
foreach ($file in $indexFiles) {
    if (-not (Test-Path $file)) {
        $missing = $true
    }
}

if ($missing) {
    Write-Host "Index files missing. Building index..."
    npm install
    npx tsx build.ts
    Write-Host "Index built successfully."
}
else {
    Write-Host "All index files already present."
}

# ------------------------------------------------------------
# Clean build
# ------------------------------------------------------------
if (Test-Path "dist") {
    Write-Host "Removing dist folder..."
    Remove-Item -Recurse -Force "dist"
}

npm install
npm run compile
npx vsce package

Write-Host ""
Write-Host "============================================================"
Write-Host "  BUILD SUCCESSFUL"
Write-Host "============================================================"