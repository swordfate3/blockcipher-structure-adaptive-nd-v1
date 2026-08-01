@echo off
setlocal EnableExtensions

set PHYSICAL_GPU=%~1
if "%PHYSICAL_GPU%"=="" exit /b 5
if not "%PHYSICAL_GPU%"=="0" if not "%PHYSICAL_GPU%"=="1" exit /b 5

set RUN_ID=i1_runtime_spn_trainable_post_expert_adapter_k1by13_present_r7_16pair_2048_seed2_seed3_20260801
set RUNS_ROOT=G:\lxy\blockcipher-structure-adaptive-nd-runs
set RUN_ROOT=%RUNS_ROOT%\%RUN_ID%
set SOURCE_ROOT=%RUN_ROOT%\source
set LOG_DIR=%RUN_ROOT%\logs
set OUTPUT_ROOT=%RUN_ROOT%\output
set READINESS_ROOT=%RUN_ROOT%\remote_readiness
set PLAN=configs\experiment\innovation1\innovation1_runtime_spn_trainable_post_expert_adapter_k1by13_present_r7_16pair_2048_seed2_seed3.csv
set PY=F:\Anaconda\envs\DWT\torch310\python.exe
set CUDA_VISIBLE_DEVICES=%PHYSICAL_GPU%
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set PYTHONPATH=%SOURCE_ROOT%\src

if not exist "%RUN_ROOT%" mkdir "%RUN_ROOT%"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
if exist "%LOG_DIR%\%RUN_ID%_failed.marker" del /q "%LOG_DIR%\%RUN_ID%_failed.marker"
if exist "%OUTPUT_ROOT%\results.jsonl" goto existing_results

cd /d "%SOURCE_ROOT%" || goto failed
for /f "delims=" %%S in ('git status --porcelain') do goto dirty_source
git rev-parse HEAD > "%LOG_DIR%\%RUN_ID%_git_revision.txt" 2>&1 || goto failed
fc /b "%LOG_DIR%\%RUN_ID%_git_revision.txt" "%RUN_ROOT%\source_expected_commit.txt" > nul || goto source_revision_mismatch
git status --short --branch > "%LOG_DIR%\%RUN_ID%_git_status_before_run.txt" 2>&1 || goto failed
nvidia-smi > "%LOG_DIR%\%RUN_ID%_gpu_info.txt" 2>&1 || goto failed
"%PY%" -c "import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 1; print('torch', torch.__version__); print('cuda', torch.version.cuda); print('visible_count', torch.cuda.device_count()); print('device0', torch.cuda.get_device_name(0))" > "%LOG_DIR%\%RUN_ID%_torch_info.txt" 2> "%LOG_DIR%\%RUN_ID%_torch_info_stderr.txt" || goto failed

"%PY%" scripts\run-runtime-spn-trainable-post-expert-adapter-k1by13 ^
  --plan "%PLAN%" ^
  --output-root "%READINESS_ROOT%" ^
  --device cuda ^
  --readiness-only ^
  > "%LOG_DIR%\%RUN_ID%_readiness_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_readiness_stderr.txt"
if errorlevel 1 goto failed

echo started>"%LOG_DIR%\%RUN_ID%_started.marker"
"%PY%" scripts\run-runtime-spn-trainable-post-expert-adapter-k1by13 ^
  --plan "%PLAN%" ^
  --output-root "%OUTPUT_ROOT%" ^
  --device cuda ^
  > "%LOG_DIR%\%RUN_ID%_train_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_train_stderr.txt"
if errorlevel 1 goto failed

"%PY%" scripts\plot-runtime-spn-trainable-post-expert-adapter-k1by13 ^
  --gate "%OUTPUT_ROOT%\gate.json" ^
  --output "%OUTPUT_ROOT%\curves.svg" ^
  > "%LOG_DIR%\%RUN_ID%_plot_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_plot_stderr.txt"
if errorlevel 1 goto failed

set RESULT_LINES=0
for /f "tokens=3" %%L in ('find /c /v "" "%OUTPUT_ROOT%\results.jsonl"') do set RESULT_LINES=%%L
if not "%RESULT_LINES%"=="8" goto incomplete_results
"%PY%" -c "import json,pathlib,sys; root=pathlib.Path(r'%OUTPUT_ROOT%'); gate=json.loads((root/'gate.json').read_text(encoding='utf-8')); validation=json.loads((root/'validation.json').read_text(encoding='utf-8')); ok=gate.get('status') in {'pass','hold'} and validation.get('status') == 'pass' and validation.get('result_rows') == 8; sys.exit(0 if ok else 1)" || goto invalid_gate

echo visual_qa_pending>"%OUTPUT_ROOT%\visual_qa_pending.marker"
echo raw_archive_ready>"%LOG_DIR%\%RUN_ID%_raw_ready.marker"
echo done>"%LOG_DIR%\%RUN_ID%_done.marker"
exit /b 0

:dirty_source
echo dirty_source>"%LOG_DIR%\%RUN_ID%_failed.marker"
exit /b 2
:source_revision_mismatch
echo source_revision_mismatch>"%LOG_DIR%\%RUN_ID%_failed.marker"
exit /b 6
:existing_results
echo existing_results>"%LOG_DIR%\%RUN_ID%_failed.marker"
exit /b 7
:incomplete_results
echo incomplete_results>"%LOG_DIR%\%RUN_ID%_failed.marker"
exit /b 4
:invalid_gate
echo invalid_gate>"%LOG_DIR%\%RUN_ID%_failed.marker"
exit /b 8
:failed
echo failed>"%LOG_DIR%\%RUN_ID%_failed.marker"
exit /b 1
