@echo off
setlocal EnableExtensions

set PHYSICAL_GPU=%~1
if "%PHYSICAL_GPU%"=="" exit /b 5
if not "%PHYSICAL_GPU%"=="1" exit /b 5

set RUN_ID=i1_uknit_r6_last2_neural_scale_k1br_262144_seed3_20260730
set RUNS_ROOT=G:\lxy\blockcipher-structure-adaptive-nd-runs
set RUN_ROOT=%RUNS_ROOT%\%RUN_ID%
set SOURCE_ROOT=%RUN_ROOT%\source
set LOG_DIR=%RUN_ROOT%\logs
set RESULTS_DIR=%RUN_ROOT%\results
set CHECKPOINT_DIR=%RUN_ROOT%\checkpoints
set CACHE_ROOT=%RUN_ROOT%\cache
set ARCHIVE_DIR=%SOURCE_ROOT%\results_archive\%RUN_ID%
set PLAN=configs\experiment\innovation1\innovation1_uknit_r6_last2_neural_scale_k1br_262144_seed3.csv
set REMOTE_CONFIG=configs\remote\innovation1_uknit_r6_last2_neural_scale_k1br_262144_seed3_gpu1_20260730.json
set PY=F:\Anaconda\envs\DWT\torch310\python.exe
set GITHUB_SSH_KEY=C:/Users/1304Lijinlin/.ssh/github_blockcipher_20260612_result_pusher_ed25519
set GIT_SSH_COMMAND=ssh -i %GITHUB_SSH_KEY% -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new
set CUDA_VISIBLE_DEVICES=%PHYSICAL_GPU%
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if not exist "%RUN_ROOT%" mkdir "%RUN_ROOT%"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
if not exist "%RESULTS_DIR%" mkdir "%RESULTS_DIR%"
if not exist "%CHECKPOINT_DIR%" mkdir "%CHECKPOINT_DIR%"
if not exist "%CACHE_ROOT%" mkdir "%CACHE_ROOT%"

cd /d "%SOURCE_ROOT%" || goto failed
for /f "delims=" %%S in ('git status --porcelain') do goto dirty_source
git rev-parse HEAD > "%LOG_DIR%\%RUN_ID%_git_revision.txt" 2>&1 || goto failed
fc /b "%LOG_DIR%\%RUN_ID%_git_revision.txt" "%RUN_ROOT%\source_expected_commit.txt" > nul || goto source_revision_mismatch
git status --short --branch > "%LOG_DIR%\%RUN_ID%_git_status_before_run.txt" 2>&1 || goto failed
nvidia-smi > "%LOG_DIR%\%RUN_ID%_gpu_info.txt" 2>&1 || goto failed
"%PY%" -c "import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 1; print('torch', torch.__version__); print('cuda', torch.version.cuda); print('visible_count', torch.cuda.device_count()); print('device0', torch.cuda.get_device_name(0))" > "%LOG_DIR%\%RUN_ID%_torch_info.txt" 2> "%LOG_DIR%\%RUN_ID%_torch_info_stderr.txt" || goto failed
"%PY%" scripts\check-remote-readiness --config "%REMOTE_CONFIG%" > "%LOG_DIR%\%RUN_ID%_readiness.txt" 2> "%LOG_DIR%\%RUN_ID%_readiness_stderr.txt" || goto failed

echo started>"%LOG_DIR%\%RUN_ID%_started.marker"
"%PY%" scripts\train ^
  --plan "%PLAN%" ^
  --epochs 10 ^
  --batch-size 64 ^
  --hidden-bits 32 ^
  --device cuda ^
  --learning-rate 0.0001 ^
  --optimizer adam ^
  --optimizer-state-transition reset_each_stage ^
  --weight-decay 0.00001 ^
  --loss mse ^
  --lr-scheduler none ^
  --checkpoint-metric val_auc ^
  --restore-best-checkpoint ^
  --early-stopping-patience 0 ^
  --early-stopping-min-delta 0.0 ^
  --train-eval-interval 1 ^
  --checkpoint-output-dir "%CHECKPOINT_DIR%" ^
  --dataset-cache-root "%CACHE_ROOT%" ^
  --dataset-cache-chunk-size 1024 ^
  --dataset-cache-workers 1 ^
  --progress-output "%LOG_DIR%\progress.jsonl" ^
  --output "%RESULTS_DIR%\results.jsonl" ^
  > "%LOG_DIR%\%RUN_ID%_train_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_train_stderr.txt"
if errorlevel 1 goto failed

"%PY%" scripts\validate-results --plan "%PLAN%" --results "%RESULTS_DIR%\results.jsonl" --expected-rows 3 --output "%RESULTS_DIR%\validation-plan.json" > "%LOG_DIR%\%RUN_ID%_validation_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_validation_stderr.txt"
if errorlevel 1 goto failed
"%PY%" scripts\gate-uknit-r6-last2-neural-scale-k1br --plan "%PLAN%" --results "%RESULTS_DIR%\results.jsonl" --progress "%LOG_DIR%\progress.jsonl" --source-commit-file "%LOG_DIR%\%RUN_ID%_git_revision.txt" --expected-source-commit-file "%RUN_ROOT%\source_expected_commit.txt" --output-root "%RESULTS_DIR%" > "%LOG_DIR%\%RUN_ID%_gate_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_gate_stderr.txt"
if errorlevel 1 goto failed

set RESULT_LINES=0
for /f "tokens=3" %%L in ('find /c /v "" "%RESULTS_DIR%\results.jsonl"') do set RESULT_LINES=%%L
echo result_lines=%RESULT_LINES% > "%LOG_DIR%\%RUN_ID%_result_gate.txt"
echo expected_rows=3 >> "%LOG_DIR%\%RUN_ID%_result_gate.txt"
if not "%RESULT_LINES%"=="3" goto incomplete_results

"%PY%" scripts\package-uknit-r6-last2-neural-scale-k1br --run-root "%RUN_ROOT%" --source-root "%SOURCE_ROOT%" --source-commit-file "%LOG_DIR%\%RUN_ID%_git_revision.txt" --expected-source-commit-file "%RUN_ROOT%\source_expected_commit.txt" --archive-root "%ARCHIVE_DIR%" > "%LOG_DIR%\%RUN_ID%_package_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_package_stderr.txt"
if errorlevel 1 goto failed

git config user.name "remote-experiment"
git config user.email "remote-experiment@local.invalid"
git checkout -B results/%RUN_ID% > "%LOG_DIR%\%RUN_ID%_result_branch_checkout.txt" 2>&1 || goto result_sync_failed
git add "results_archive\%RUN_ID%" || goto result_sync_failed
git commit -m "results: %RUN_ID% remote r6 diagnostic" > "%LOG_DIR%\%RUN_ID%_result_branch_commit.txt" 2>&1 || goto result_sync_failed
git push origin HEAD:refs/heads/results/%RUN_ID% > "%LOG_DIR%\%RUN_ID%_result_branch_push.txt" 2>&1 || goto result_sync_failed
echo pushed>"%LOG_DIR%\%RUN_ID%_result_branch_pushed.marker"
echo done>"%LOG_DIR%\%RUN_ID%_done.marker"
exit /b 0

:result_sync_failed
echo result_branch_push_failed>"%LOG_DIR%\%RUN_ID%_result_sync_failed.marker"
echo raw_archive_ready>"%LOG_DIR%\%RUN_ID%_raw_ready.marker"
echo done_with_raw_fallback>"%LOG_DIR%\%RUN_ID%_done.marker"
exit /b 0

:dirty_source
echo dirty_source>"%LOG_DIR%\%RUN_ID%_failed.marker"
exit /b 2

:source_revision_mismatch
echo source_revision_mismatch>"%LOG_DIR%\%RUN_ID%_failed.marker"
exit /b 6

:incomplete_results
echo incomplete_results>"%LOG_DIR%\%RUN_ID%_failed.marker"
exit /b 4

:failed
echo failed>"%LOG_DIR%\%RUN_ID%_failed.marker"
exit /b 1
