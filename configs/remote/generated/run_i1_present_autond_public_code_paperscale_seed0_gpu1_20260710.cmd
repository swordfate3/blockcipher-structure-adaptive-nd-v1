@echo off
setlocal EnableExtensions EnableDelayedExpansion

set RUN_ID=i1_present_autond_public_code_paperscale_seed0_gpu1_20260710
set BRANCH=main
set PROJECT_ROOT=G:\lxy\blockcipher-structure-adaptive-nd-runs\%RUN_ID%\source
set RUN_ROOT=G:\lxy\blockcipher-structure-adaptive-nd-runs\%RUN_ID%
set LOG_DIR=%RUN_ROOT%\logs
set RESULTS_DIR=%RUN_ROOT%\results
set ARCHIVE_DIR=%RUN_ROOT%\results_archive\%RUN_ID%
set DATASET_CACHE_ROOT=%RUN_ROOT%\dataset_cache
set PYTHON_EXE=F:\Anaconda\envs\DWT\torch310\python.exe

if not exist "%RUN_ROOT%" mkdir "%RUN_ROOT%"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
if not exist "%RESULTS_DIR%" mkdir "%RESULTS_DIR%"
if not exist "%ARCHIVE_DIR%" mkdir "%ARCHIVE_DIR%"
if not exist "%DATASET_CACHE_ROOT%" mkdir "%DATASET_CACHE_ROOT%"

echo run_id=%RUN_ID% > "%LOG_DIR%\%RUN_ID%_launch_env.txt"
echo project_root=%PROJECT_ROOT% >> "%LOG_DIR%\%RUN_ID%_launch_env.txt"
echo run_root=%RUN_ROOT% >> "%LOG_DIR%\%RUN_ID%_launch_env.txt"
echo dataset_cache_root=%DATASET_CACHE_ROOT% >> "%LOG_DIR%\%RUN_ID%_launch_env.txt"
echo device=cuda:1 >> "%LOG_DIR%\%RUN_ID%_launch_env.txt"

nvidia-smi > "%LOG_DIR%\%RUN_ID%_gpu_info.txt" 2>&1
"%PYTHON_EXE%" -c "import torch; print('torch', torch.__version__); print('cuda', torch.version.cuda); print('available', torch.cuda.is_available()); print('count', torch.cuda.device_count()); print('device1', torch.cuda.get_device_name(1) if torch.cuda.device_count() > 1 else 'NA')" > "%LOG_DIR%\%RUN_ID%_torch_info.txt" 2> "%LOG_DIR%\%RUN_ID%_torch_info_stderr.txt"
if errorlevel 1 goto failed

if not exist "%PROJECT_ROOT%\.git" goto failed
cd /d "%PROJECT_ROOT%" || goto failed
git config --global --add safe.directory G:/lxy/blockcipher-structure-adaptive-nd-runs/%RUN_ID%/source
git status --short --branch > "%LOG_DIR%\%RUN_ID%_git_status_before_run.txt" 2>&1
git fetch origin %BRANCH% > "%LOG_DIR%\%RUN_ID%_git_fetch_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_git_fetch_stderr.txt" || goto failed
git checkout %BRANCH% > "%LOG_DIR%\%RUN_ID%_git_checkout_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_git_checkout_stderr.txt" || goto failed
git pull --ff-only origin %BRANCH% >> "%LOG_DIR%\%RUN_ID%_git_fetch_stdout.txt" 2>> "%LOG_DIR%\%RUN_ID%_git_fetch_stderr.txt" || goto failed
git rev-parse HEAD > "%LOG_DIR%\%RUN_ID%_git_revision.txt" 2>&1

"%PYTHON_EXE%" scripts\check-remote-readiness --config configs\remote\innovation1_spn_present_autond_public_code_paperscale_seed0_gpu1_20260710.json > "%LOG_DIR%\%RUN_ID%_readiness.txt" 2> "%LOG_DIR%\%RUN_ID%_readiness_stderr.txt" || goto failed
echo started > "%LOG_DIR%\%RUN_ID%_started.marker"

"%PYTHON_EXE%" scripts\train ^
  --plan configs\experiment\innovation1\innovation1_spn_present_autond_public_code_paperscale_seed0.csv ^
  --epochs 40 ^
  --batch-size 5000 ^
  --hidden-bits 32 ^
  --device cuda:1 ^
  --learning-rate 0.001 ^
  --optimizer adam ^
  --amsgrad ^
  --optimizer-state-transition carry_across_stages ^
  --weight-decay 0 ^
  --loss mse ^
  --lr-scheduler none ^
  --checkpoint-metric val_loss ^
  --restore-best-checkpoint ^
  --early-stopping-patience 0 ^
  --early-stopping-min-delta 0.0 ^
  --train-eval-interval 0 ^
  --pretrain-round-sequence "[5,6,7,8]" ^
  --pretrain-epochs 40 ^
  --train-samples-total 10000000 ^
  --validation-samples-total 1000000 ^
  --final-test-samples-total 1000000 ^
  --final-test-repeats 5 ^
  --dataset-label-mode random_labels_total ^
  --sample-structure independent_pairs ^
  --negative-mode random_ciphertext ^
  --key-rotation-interval 1 ^
  --dataset-cache-root "%DATASET_CACHE_ROOT%" ^
  --dataset-cache-chunk-size 8192 ^
  --dataset-cache-workers 4 ^
  --output "%RESULTS_DIR%\%RUN_ID%.jsonl" ^
  --progress-output "%LOG_DIR%\%RUN_ID%_progress.jsonl" ^
  > "%LOG_DIR%\%RUN_ID%_train_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_train_stderr.txt"
if errorlevel 1 goto failed

set RESULT_LINES=0
for /f "tokens=3" %%L in ('find /c /v "" "%RESULTS_DIR%\%RUN_ID%.jsonl"') do set RESULT_LINES=%%L
echo result_lines=%RESULT_LINES% > "%LOG_DIR%\%RUN_ID%_result_gate.txt"
echo expected_rows=1 >> "%LOG_DIR%\%RUN_ID%_result_gate.txt"
if not "%RESULT_LINES%"=="1" goto failed

"%PYTHON_EXE%" scripts\validate-results ^
  --plan configs\experiment\innovation1\innovation1_spn_present_autond_public_code_paperscale_seed0.csv ^
  --results "%RESULTS_DIR%\%RUN_ID%.jsonl" ^
  --expected-rows 1 ^
  --output "%RESULTS_DIR%\%RUN_ID%_validation.json" ^
  > "%LOG_DIR%\%RUN_ID%_validate_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_validate_stderr.txt"
if errorlevel 1 goto failed

copy "%RESULTS_DIR%\%RUN_ID%.jsonl" "%ARCHIVE_DIR%\" > nul
copy "%RESULTS_DIR%\%RUN_ID%_validation.json" "%ARCHIVE_DIR%\" > nul
copy "%LOG_DIR%\%RUN_ID%_git_revision.txt" "%ARCHIVE_DIR%\" > nul
copy "%LOG_DIR%\%RUN_ID%_git_status_before_run.txt" "%ARCHIVE_DIR%\" > nul
copy "%LOG_DIR%\%RUN_ID%_gpu_info.txt" "%ARCHIVE_DIR%\" > nul
copy "%LOG_DIR%\%RUN_ID%_torch_info.txt" "%ARCHIVE_DIR%\" > nul
copy "%LOG_DIR%\%RUN_ID%_result_gate.txt" "%ARCHIVE_DIR%\" > nul
echo done > "%LOG_DIR%\%RUN_ID%_done.marker"
exit /b 0

:failed
echo failed > "%LOG_DIR%\%RUN_ID%_failed.marker"
exit /b 1
