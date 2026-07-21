@echo off
setlocal EnableExtensions

set REPO_URL=git@github.com:swordfate3/blockcipher-structure-adaptive-nd-v1.git
set RUNS_ROOT=G:\lxy\blockcipher-structure-adaptive-nd-runs
set LAUNCH_LOG_DIR=%RUNS_ROOT%\launcher_logs
set GITHUB_SSH_KEY=C:/Users/1304Lijinlin/.ssh/github_blockcipher_20260612_result_pusher_ed25519
set GIT_SSH_COMMAND=ssh -i %GITHUB_SSH_KEY% -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new
set RUN_ID=i2_output_prediction_op10_present_r3_easy_bit_confirm_gpu0_20260721
set SOURCE_RUN_ID=i2_output_prediction_op9_present_r3_kimura_lstm_2p17_seed0_gpu0_20260721
set TASK_NAME=I2_OP10_PRESENT_R3_EASY_BIT_GPU0
set SOURCE_COMMIT=%~1
if "%SOURCE_COMMIT%"=="" exit /b 2

set SOURCE_LOG_DIR=%RUNS_ROOT%\%SOURCE_RUN_ID%\logs
if not exist "%SOURCE_LOG_DIR%\%SOURCE_RUN_ID%_done.marker" exit /b 4
if not exist "%SOURCE_LOG_DIR%\%SOURCE_RUN_ID%_result_branch_pushed.marker" exit /b 4
if not exist "%RUNS_ROOT%" mkdir "%RUNS_ROOT%"
if not exist "%LAUNCH_LOG_DIR%" mkdir "%LAUNCH_LOG_DIR%"

set RUN_ROOT=%RUNS_ROOT%\%RUN_ID%
set SOURCE_ROOT=%RUN_ROOT%\source
if not exist "%RUN_ROOT%" mkdir "%RUN_ROOT%"
if exist "%SOURCE_ROOT%\.git" (
  cd /d "%SOURCE_ROOT%" || exit /b 1
  for /f "delims=" %%S in ('git status --porcelain') do exit /b 1
  git fetch origin || exit /b 1
) else (
  if exist "%SOURCE_ROOT%" exit /b 3
  git clone --no-checkout "%REPO_URL%" "%SOURCE_ROOT%" || exit /b 1
)
cd /d "%SOURCE_ROOT%" || exit /b 1
git config --global --add safe.directory G:/lxy/blockcipher-structure-adaptive-nd-runs/%RUN_ID%/source
git checkout --detach "%SOURCE_COMMIT%" || exit /b 1
for /f "delims=" %%S in ('git status --porcelain') do exit /b 1
for /f "delims=" %%H in ('git rev-parse HEAD') do set ACTUAL_COMMIT=%%H
if /I not "%ACTUAL_COMMIT%"=="%SOURCE_COMMIT%" exit /b 1
git rev-parse HEAD > "%RUN_ROOT%\source_expected_commit.txt" || exit /b 1

set RUN_CMD=%SOURCE_ROOT%\configs\remote\generated\run_i2_output_prediction_op10_present_r3_easy_bit_confirm_gpu0_20260721.cmd
schtasks /Create /TN "%TASK_NAME%" /SC ONCE /ST 23:59 /RU SYSTEM /RL HIGHEST /TR "cmd.exe /c %RUN_CMD% 0" /F > "%LAUNCH_LOG_DIR%\%RUN_ID%_schedule_create.txt" 2>&1 || exit /b 1
schtasks /Run /I /TN "%TASK_NAME%" > "%LAUNCH_LOG_DIR%\%RUN_ID%_schedule_run.txt" 2>&1 || exit /b 1
schtasks /Query /TN "%TASK_NAME%" /V /FO LIST > "%LAUNCH_LOG_DIR%\%RUN_ID%_schedule_query.txt" 2>&1
echo launched>"%LAUNCH_LOG_DIR%\%RUN_ID%_launched.marker"
exit /b 0
