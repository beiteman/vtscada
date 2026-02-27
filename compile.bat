@echo off
setlocal EnableExtensions EnableDelayedExpansion

chcp 65001 >nul

echo ============================================================
echo   VTScada Extension Build Script
echo ============================================================
echo.

:: ------------------------------------------------------------
:: TOOL CHECKS
:: ------------------------------------------------------------
call :require_tool curl
call :require_tool node
call :require_tool npm
call :require_tool npx

:: ------------------------------------------------------------
:: PARAMETERS
:: ------------------------------------------------------------
set "EN_MODEL_DIR=resource\all-MiniLM-L6-v2"
set "ZH_MODEL_DIR=resource\text2vec-base-chinese"

set "EN_INDEX_PATH=resource\index.en.json"
set "ZH_CN_INDEX_PATH=resource\index.zh-cn.json"
set "ZH_TW_INDEX_PATH=resource\index.zh-tw.json"

:: ------------------------------------------------------------
:: DOWNLOAD MODELS
:: ------------------------------------------------------------
call :download_model ^
 "%EN_MODEL_DIR%" ^
 "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx" "model.onnx" ^
 "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer_config.json" "tokenizer_config.json" ^
 "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json" "tokenizer.json"

call :download_model ^
 "%ZH_MODEL_DIR%" ^
 "https://huggingface.co/shibing624/text2vec-base-chinese/resolve/main/onnx/model.onnx" "model.onnx" ^
 "https://huggingface.co/shibing624/text2vec-base-chinese/resolve/main/onnx/tokenizer_config.json" "tokenizer_config.json" ^
 "https://huggingface.co/shibing624/text2vec-base-chinese/resolve/main/onnx/tokenizer.json" "tokenizer.json"

:: ------------------------------------------------------------
:: BUILD INDEX IF MISSING
:: ------------------------------------------------------------
echo.
echo Checking index files...

set "MISSING_INDEX=0"
if not exist "%EN_INDEX_PATH%" set "MISSING_INDEX=1"
if not exist "%ZH_CN_INDEX_PATH%" set "MISSING_INDEX=1"
if not exist "%ZH_TW_INDEX_PATH%" set "MISSING_INDEX=1"

if "%MISSING_INDEX%"=="1" (
    echo Index files missing. Building index...
    call npm install || goto :error
    call npx tsx build.ts || goto :error
    echo Index built successfully.
) else (
    echo All index files already present.
)

:: ------------------------------------------------------------
:: CLEAN BUILD
:: ------------------------------------------------------------
echo.
echo Starting clean build...

if exist dist (
    echo Removing dist folder...
    rmdir /s /q dist || goto :error
)

call npm install || goto :error
call npm run compile || goto :error
call npx vsce package || goto :error

echo.
echo ============================================================
echo   BUILD SUCCESSFUL
echo ============================================================
exit /b 0

:: ============================================================
:: FUNCTIONS
:: ============================================================

:require_tool
where %1 >nul 2>nul
if errorlevel 1 (
    echo ERROR: Required tool "%1" not found in PATH.
    echo Please install it and ensure it is available.
    exit /b 1
)
exit /b 0


:download_model
set "MODEL_DIR=%~1"
set "URL1=%~2"
set "FILE1=%~3"
set "URL2=%~4"
set "FILE2=%~5"
set "URL3=%~6"
set "FILE3=%~7"

if not exist "%MODEL_DIR%" (
    echo Creating directory %MODEL_DIR%
    mkdir "%MODEL_DIR%" || goto :error
)

call :download_file "%MODEL_DIR%" "%URL1%" "%FILE1%"
call :download_file "%MODEL_DIR%" "%URL2%" "%FILE2%"
call :download_file "%MODEL_DIR%" "%URL3%" "%FILE3%"

exit /b 0


:download_file
set "MODEL_DIR=%~1"
set "URL=%~2"
set "FILE_NAME=%~3"

if "%FILE_NAME%"=="" exit /b 0

if not exist "%MODEL_DIR%\%FILE_NAME%" (
    echo Downloading %FILE_NAME% ...
    curl -L --fail --silent --show-error "%URL%" -o "%MODEL_DIR%\%FILE_NAME%"
    if errorlevel 1 (
        echo ERROR: Failed to download %FILE_NAME%
        exit /b 1
    )
    echo Downloaded %FILE_NAME%
) else (
    echo %FILE_NAME% already exists.
)

exit /b 0


:error
echo.
echo ============================================================
echo   BUILD FAILED
echo ============================================================
exit /b 1