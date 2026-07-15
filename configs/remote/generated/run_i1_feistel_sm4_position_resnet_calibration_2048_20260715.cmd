@echo off
setlocal EnableExtensions EnableDelayedExpansion

set PHYSICAL_GPU=%~1
if "%PHYSICAL_GPU%"=="" goto invalid_arguments

set RUN_ID=i1_feistel_sm4_position_resnet_calibration_2048_seed0
set RUNS_ROOT=G:\lxy\blockcipher-structure-adaptive-nd-runs
set RUN_ROOT=%RUNS_ROOT%\%RUN_ID%
set SOURCE_ROOT=%RUN_ROOT%\source
set LOG_DIR=%RUN_ROOT%\logs
set RESULTS_DIR=%RUN_ROOT%\results
set ARCHIVE_DIR=%SOURCE_ROOT%\results_archive\%RUN_ID%
set PLAN=configs\experiment\innovation1\innovation1_feistel_sm4_position_resnet_calibration_2048_seed0.csv
set REMOTE_CONFIG=configs\remote\innovation1_feistel_sm4_position_resnet_calibration_2048_seed0_gpu1_20260715.json
set PY=F:\Anaconda\envs\DWT\torch310\python.exe
set GITHUB_SSH_KEY=C:/Users/1304Lijinlin/.ssh/github_blockcipher_20260612_result_pusher_ed25519
set GIT_SSH_COMMAND=ssh -i %GITHUB_SSH_KEY% -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new
set CUDA_VISIBLE_DEVICES=%PHYSICAL_GPU%
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if not exist "%RUN_ROOT%" mkdir "%RUN_ROOT%"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
if not exist "%RESULTS_DIR%" mkdir "%RESULTS_DIR%"

cd /d "%SOURCE_ROOT%" || goto failed
for /f "delims=" %%S in ('git status --porcelain') do goto dirty_source
git rev-parse HEAD > "%LOG_DIR%\%RUN_ID%_git_revision.txt" 2>&1 || goto failed
fc /b "%LOG_DIR%\%RUN_ID%_git_revision.txt" "%RUN_ROOT%\source_expected_commit.txt" > nul || goto source_revision_mismatch
git status --short --branch > "%LOG_DIR%\%RUN_ID%_git_status_before_run.txt" 2>&1 || goto failed
nvidia-smi > "%LOG_DIR%\%RUN_ID%_gpu_info.txt" 2>&1 || goto failed
"%PY%" -c "import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 1; print('torch', torch.__version__); print('cuda', torch.version.cuda); print('available', torch.cuda.is_available()); print('visible_count', torch.cuda.device_count()); print('device0', torch.cuda.get_device_name(0))" > "%LOG_DIR%\%RUN_ID%_torch_info.txt" 2> "%LOG_DIR%\%RUN_ID%_torch_info_stderr.txt" || goto failed
"%PY%" scripts\check-remote-readiness --config "%REMOTE_CONFIG%" > "%LOG_DIR%\%RUN_ID%_readiness.txt" 2> "%LOG_DIR%\%RUN_ID%_readiness_stderr.txt" || goto failed

echo started>"%LOG_DIR%\%RUN_ID%_started.marker"
"%PY%" scripts\train ^
  --plan "%PLAN%" ^
  --epochs 10 ^
  --batch-size 128 ^
  --hidden-bits 32 ^
  --device cuda ^
  --train-eval-interval 1 ^
  --progress-output "%LOG_DIR%\progress.jsonl" ^
  --output "%RESULTS_DIR%\results.jsonl" ^
  > "%LOG_DIR%\%RUN_ID%_train_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_train_stderr.txt"
if errorlevel 1 goto failed

"%PY%" scripts\validate-results ^
  --plan "%PLAN%" ^
  --results "%RESULTS_DIR%\results.jsonl" ^
  --expected-rows 4 ^
  --output "%RESULTS_DIR%\validation.json" ^
  > "%LOG_DIR%\%RUN_ID%_validation_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_validation_stderr.txt"
if errorlevel 1 goto failed

"%PY%" scripts\gate-feistel-sm4 ^
  --plan "%PLAN%" ^
  --results "%RESULTS_DIR%\results.jsonl" ^
  --samples-per-class 2048 ^
  --seeds 0 ^
  --epochs 10 ^
  --final-repeats 3 ^
  --position-calibration ^
  --output "%RESULTS_DIR%\gate.json" ^
  > "%LOG_DIR%\%RUN_ID%_gate_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_gate_stderr.txt"
if errorlevel 1 goto failed

set RESULT_LINES=0
for /f "tokens=3" %%L in ('find /c /v "" "%RESULTS_DIR%\results.jsonl"') do set RESULT_LINES=%%L
echo result_lines=%RESULT_LINES% > "%LOG_DIR%\%RUN_ID%_result_gate.txt"
echo expected_rows=4 >> "%LOG_DIR%\%RUN_ID%_result_gate.txt"
if not "%RESULT_LINES%"=="4" goto incomplete_results

"%PY%" scripts\plot-results ^
  --results "%RESULTS_DIR%\results.jsonl" ^
  --output "%RESULTS_DIR%\curves.svg" ^
  --history-csv "%RESULTS_DIR%\history.csv" ^
  --title "Innovation 1 Feistel SM4 position ResNet calibration" ^
  > "%LOG_DIR%\%RUN_ID%_plot_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_plot_stderr.txt"
if errorlevel 1 goto plot_deferred
goto plot_done

:plot_deferred
echo plot_deferred_to_local>"%LOG_DIR%\%RUN_ID%_plot_deferred.marker"
if exist "%RESULTS_DIR%\curves.svg" del /Q "%RESULTS_DIR%\curves.svg"

:plot_done
if not exist "%ARCHIVE_DIR%" mkdir "%ARCHIVE_DIR%"
copy /Y "%RESULTS_DIR%\results.jsonl" "%ARCHIVE_DIR%\results.jsonl" > nul || goto failed
copy /Y "%RESULTS_DIR%\validation.json" "%ARCHIVE_DIR%\validation.json" > nul || goto failed
copy /Y "%RESULTS_DIR%\gate.json" "%ARCHIVE_DIR%\gate.json" > nul || goto failed
copy /Y "%LOG_DIR%\progress.jsonl" "%ARCHIVE_DIR%\progress.jsonl" > nul || goto failed
copy /Y "%LOG_DIR%\%RUN_ID%_git_revision.txt" "%ARCHIVE_DIR%\git_revision.txt" > nul || goto failed
copy /Y "%LOG_DIR%\%RUN_ID%_git_status_before_run.txt" "%ARCHIVE_DIR%\git_status_before_run.txt" > nul || goto failed
copy /Y "%LOG_DIR%\%RUN_ID%_gpu_info.txt" "%ARCHIVE_DIR%\gpu_info.txt" > nul || goto failed
copy /Y "%LOG_DIR%\%RUN_ID%_torch_info.txt" "%ARCHIVE_DIR%\torch_info.txt" > nul || goto failed
copy /Y "%LOG_DIR%\%RUN_ID%_result_gate.txt" "%ARCHIVE_DIR%\result_gate.txt" > nul || goto failed
copy /Y "%REMOTE_CONFIG%" "%ARCHIVE_DIR%\remote_config.json" > nul || goto failed
copy /Y "%PLAN%" "%ARCHIVE_DIR%\plan.csv" > nul || goto failed
if exist "%RESULTS_DIR%\history.csv" copy /Y "%RESULTS_DIR%\history.csv" "%ARCHIVE_DIR%\history.csv" > nul
if exist "%RESULTS_DIR%\curves.svg" copy /Y "%RESULTS_DIR%\curves.svg" "%ARCHIVE_DIR%\curves.svg" > nul
if exist "%LOG_DIR%\%RUN_ID%_plot_deferred.marker" copy /Y "%LOG_DIR%\%RUN_ID%_plot_deferred.marker" "%ARCHIVE_DIR%\plot_deferred.marker" > nul
echo * -text>"%ARCHIVE_DIR%\.gitattributes"
"%PY%" -c "import hashlib,pathlib; root=pathlib.Path(r'%ARCHIVE_DIR%'); files=sorted(p for p in root.rglob('*') if p.is_file() and p.name!='SHA256SUMS'); (root/'SHA256SUMS').write_text('\n'.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.relative_to(root).as_posix() for p in files)+'\n',encoding='utf-8')" || goto failed

git config user.name "remote-experiment"
git config user.email "remote-experiment@local.invalid"
git checkout -B results/%RUN_ID% > "%LOG_DIR%\%RUN_ID%_result_branch_checkout.txt" 2>&1 || goto failed
git add "results_archive\%RUN_ID%" || goto failed
git commit -m "results: %RUN_ID% calibration" > "%LOG_DIR%\%RUN_ID%_result_branch_commit.txt" 2>&1 || goto failed
git push origin HEAD:refs/heads/results/%RUN_ID% > "%LOG_DIR%\%RUN_ID%_result_branch_push.txt" 2>&1 || goto failed
git rev-parse HEAD > "%LOG_DIR%\%RUN_ID%_result_branch_revision.txt" 2>&1 || goto failed
echo pushed>"%LOG_DIR%\%RUN_ID%_result_branch_pushed.marker"
echo done>"%LOG_DIR%\%RUN_ID%_done.marker"
exit /b 0

:dirty_source
echo dirty_source>"%LOG_DIR%\%RUN_ID%_failed.marker"
echo Remote run-owned source clone is dirty.>"%LOG_DIR%\%RUN_ID%_failure_reason.txt"
exit /b 2

:source_revision_mismatch
echo source_revision_mismatch>"%LOG_DIR%\%RUN_ID%_failed.marker"
echo Remote source HEAD does not match the launch-pinned commit.>"%LOG_DIR%\%RUN_ID%_failure_reason.txt"
exit /b 6

:incomplete_results
echo incomplete_results>"%LOG_DIR%\%RUN_ID%_failed.marker"
exit /b 4

:invalid_arguments
exit /b 5

:failed
echo failed>"%LOG_DIR%\%RUN_ID%_failed.marker"
exit /b 1
