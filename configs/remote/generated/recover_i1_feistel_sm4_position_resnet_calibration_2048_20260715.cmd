@echo off
setlocal EnableExtensions DisableDelayedExpansion

set RECOVERY_COMMIT=%~1
if "%RECOVERY_COMMIT%"=="" exit /b 2

for %%I in ("%~dp0..\..\..") do set RECOVERY_SOURCE=%%~fI
set RUN_ID=i1_feistel_sm4_position_resnet_calibration_2048_seed0
set RUNS_ROOT=G:\lxy\blockcipher-structure-adaptive-nd-runs
set RUN_ROOT=%RUNS_ROOT%\%RUN_ID%
set TRAIN_SOURCE=%RUN_ROOT%\source
set LOG_DIR=%RUN_ROOT%\logs
set RESULTS_DIR=%RUN_ROOT%\results
set ARCHIVE_DIR=%TRAIN_SOURCE%\results_archive\%RUN_ID%
set PLAN=%TRAIN_SOURCE%\configs\experiment\innovation1\innovation1_feistel_sm4_position_resnet_calibration_2048_seed0.csv
set REMOTE_CONFIG=%TRAIN_SOURCE%\configs\remote\innovation1_feistel_sm4_position_resnet_calibration_2048_seed0_gpu1_20260715.json
set PY=F:\Anaconda\envs\DWT\torch310\python.exe
set GITHUB_SSH_KEY=C:/Users/1304Lijinlin/.ssh/github_blockcipher_20260612_result_pusher_ed25519
set GIT_SSH_COMMAND=ssh -i %GITHUB_SSH_KEY% -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new

cd /d "%RECOVERY_SOURCE%" || goto recovery_failed
for /f "delims=" %%S in ('git status --porcelain') do goto recovery_failed
for /f "delims=" %%H in ('git rev-parse HEAD') do set ACTUAL_RECOVERY_COMMIT=%%H
if /I not "%ACTUAL_RECOVERY_COMMIT%"=="%RECOVERY_COMMIT%" goto recovery_failed

echo started>"%LOG_DIR%\%RUN_ID%_recovery_started.marker"
if not exist "%RESULTS_DIR%\results.jsonl" goto recovery_failed
if not exist "%RESULTS_DIR%\validation.json" goto recovery_failed
if not exist "%RESULTS_DIR%\gate.json" goto recovery_failed
if not exist "%ARCHIVE_DIR%" mkdir "%ARCHIVE_DIR%"

"%PY%" "%RECOVERY_SOURCE%\scripts\validate-results" ^
  --plan "%PLAN%" ^
  --results "%RESULTS_DIR%\results.jsonl" ^
  --expected-rows 4 ^
  --output "%RESULTS_DIR%\validation.recovery.json" ^
  > "%LOG_DIR%\%RUN_ID%_recovery_validation_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_recovery_validation_stderr.txt" || goto recovery_failed

"%PY%" "%RECOVERY_SOURCE%\scripts\gate-feistel-sm4" ^
  --plan "%PLAN%" ^
  --results "%RESULTS_DIR%\results.jsonl" ^
  --samples-per-class 2048 ^
  --seeds 0 ^
  --epochs 10 ^
  --final-repeats 3 ^
  --position-calibration ^
  --output "%RESULTS_DIR%\gate.recovery.json" ^
  > "%LOG_DIR%\%RUN_ID%_recovery_gate_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_recovery_gate_stderr.txt" || goto recovery_failed

copy /Y "%RESULTS_DIR%\results.jsonl" "%ARCHIVE_DIR%\results.jsonl" > nul || goto recovery_failed
copy /Y "%RESULTS_DIR%\validation.json" "%ARCHIVE_DIR%\validation.json" > nul || goto recovery_failed
copy /Y "%RESULTS_DIR%\validation.recovery.json" "%ARCHIVE_DIR%\validation.recovery.json" > nul || goto recovery_failed
copy /Y "%RESULTS_DIR%\gate.json" "%ARCHIVE_DIR%\gate.json" > nul || goto recovery_failed
copy /Y "%RESULTS_DIR%\gate.recovery.json" "%ARCHIVE_DIR%\gate.recovery.json" > nul || goto recovery_failed
copy /Y "%LOG_DIR%\progress.jsonl" "%ARCHIVE_DIR%\progress.jsonl" > nul || goto recovery_failed
copy /Y "%LOG_DIR%\%RUN_ID%_git_revision.txt" "%ARCHIVE_DIR%\git_revision.txt" > nul || goto recovery_failed
copy /Y "%LOG_DIR%\%RUN_ID%_git_status_before_run.txt" "%ARCHIVE_DIR%\git_status_before_run.txt" > nul || goto recovery_failed
copy /Y "%LOG_DIR%\%RUN_ID%_gpu_info.txt" "%ARCHIVE_DIR%\gpu_info.txt" > nul || goto recovery_failed
copy /Y "%LOG_DIR%\%RUN_ID%_torch_info.txt" "%ARCHIVE_DIR%\torch_info.txt" > nul || goto recovery_failed
copy /Y "%LOG_DIR%\%RUN_ID%_result_gate.txt" "%ARCHIVE_DIR%\result_gate.txt" > nul || goto recovery_failed
copy /Y "%REMOTE_CONFIG%" "%ARCHIVE_DIR%\remote_config.json" > nul || goto recovery_failed
copy /Y "%PLAN%" "%ARCHIVE_DIR%\plan.csv" > nul || goto recovery_failed
if exist "%LOG_DIR%\%RUN_ID%_plot_deferred.marker" copy /Y "%LOG_DIR%\%RUN_ID%_plot_deferred.marker" "%ARCHIVE_DIR%\plot_deferred.marker" > nul
echo %RECOVERY_COMMIT%>"%ARCHIVE_DIR%\recovery_commit.txt"
echo * -text>"%ARCHIVE_DIR%\.gitattributes"
"%PY%" -c "import hashlib,pathlib; root=pathlib.Path(r'%ARCHIVE_DIR%'); files=sorted(p for p in root.rglob('*') if p.is_file() and not p.name == 'SHA256SUMS'); (root/'SHA256SUMS').write_text('\n'.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.relative_to(root).as_posix() for p in files)+'\n',encoding='utf-8')" > "%LOG_DIR%\%RUN_ID%_recovery_hash_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_recovery_hash_stderr.txt" || goto recovery_failed

set SOURCE_COMMIT=
set /p SOURCE_COMMIT=<"%RUN_ROOT%\source_expected_commit.txt"
if "%SOURCE_COMMIT%"=="" goto recovery_failed
cd /d "%TRAIN_SOURCE%" || goto recovery_failed
git config user.name "remote-experiment"
git config user.email "remote-experiment@local.invalid"
git checkout -B results/%RUN_ID% "%SOURCE_COMMIT%" > "%LOG_DIR%\%RUN_ID%_recovery_branch_checkout.txt" 2>&1 || goto recovery_failed
git add "results_archive\%RUN_ID%" || goto recovery_failed
git commit -m "results: %RUN_ID% recovered calibration" > "%LOG_DIR%\%RUN_ID%_recovery_branch_commit.txt" 2>&1 || goto recovery_failed
git push origin HEAD:refs/heads/results/%RUN_ID% > "%LOG_DIR%\%RUN_ID%_recovery_branch_push.txt" 2>&1 || goto recovery_failed
git rev-parse HEAD > "%LOG_DIR%\%RUN_ID%_result_branch_revision.txt" 2>&1 || goto recovery_failed
echo pushed>"%LOG_DIR%\%RUN_ID%_result_branch_pushed.marker"
echo done>"%LOG_DIR%\%RUN_ID%_done.marker"
exit /b 0

:recovery_failed
echo failed>"%LOG_DIR%\%RUN_ID%_recovery_failed.marker"
exit /b 1
