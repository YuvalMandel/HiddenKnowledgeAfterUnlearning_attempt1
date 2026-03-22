@echo off
setlocal
REM Reproduce the dual-confidence hidden-knowledge figure package.
REM Usage:
REM   run_generate_hidden_knowledge_dual_confidence_figures.cmd [INPUT_DIR] [OUTPUT_DIR]
REM Defaults:
REM   INPUT_DIR  = current folder
REM   OUTPUT_DIR = .\hidden_knowledge_dual_confidence_outputs

set INPUT_DIR=%~1
if "%INPUT_DIR%"=="" set INPUT_DIR=.

set OUTPUT_DIR=%~2
if "%OUTPUT_DIR%"=="" set OUTPUT_DIR=%CD%\hidden_knowledge_dual_confidence_outputs

python "%~dp0generate_hidden_knowledge_dual_confidence_figures.py" --input-dir "%INPUT_DIR%" --output-dir "%OUTPUT_DIR%"

if errorlevel 1 (
  echo Generation failed.
  exit /b 1
)

echo Done. Outputs written to:
echo   %OUTPUT_DIR%
endlocal
