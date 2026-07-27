@echo off
setlocal EnableExtensions

set SOURCE_COMMIT=%~1
set PHYSICAL_GPU=%~2
if "%SOURCE_COMMIT%"=="" exit /b 2
if "%PHYSICAL_GPU%"=="" exit /b 2

set REPO_URL=git@github.com:swordfate3/blockcipher-structure-adaptive-nd-v1.git
set RUN_ID=i1_rct3_rectangle80_runtime_e4_scale_262144_seed0_20260727
set RUNS_ROOT=G:\lxy\blockcipher-structure-adaptive-nd-runs
set SCHEDULE_ROOT=G:\lxy\scheduled-runs
set LAUNCH_LOG_DIR=%RUNS_ROOT%\launcher_logs
set GITHUB_SSH_KEY=C:/Users/1304Lijinlin/.ssh/github_blockcipher_20260612_result_pusher_ed25519
set GIT_SSH_COMMAND=ssh -i %GITHUB_SSH_KEY% -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new
set TASK_NAME=I1_RCT3_RECTANGLE80_S0_GPU%PHYSICAL_GPU%
set SCHEDULE_CMD=%SCHEDULE_ROOT%\i1_rct3_rectangle80_seed0.cmd

if not exist "%RUNS_ROOT%" mkdir "%RUNS_ROOT%"
if not exist "%SCHEDULE_ROOT%" mkdir "%SCHEDULE_ROOT%"
if not exist "%LAUNCH_LOG_DIR%" mkdir "%LAUNCH_LOG_DIR%"

call :prepare_source "%SOURCE_COMMIT%" || exit /b 1

set RUN_CMD=%RUNS_ROOT%\%RUN_ID%\source\configs\remote\generated\run_i1_rct3_rectangle80_runtime_e4_scale_262144_seed0_20260727.cmd
>"%SCHEDULE_CMD%" echo @echo off
>>"%SCHEDULE_CMD%" echo call "%RUN_CMD%" %PHYSICAL_GPU%

schtasks /Create /TN "%TASK_NAME%" /SC ONCE /ST 23:59 /RU SYSTEM /RL HIGHEST /TR "cmd.exe /c %SCHEDULE_CMD%" /F > "%LAUNCH_LOG_DIR%\%RUN_ID%_schedule_create.txt" 2>&1 || exit /b 1
schtasks /Run /I /TN "%TASK_NAME%" > "%LAUNCH_LOG_DIR%\%RUN_ID%_schedule_run.txt" 2>&1 || exit /b 1
schtasks /Change /TN "%TASK_NAME%" /DISABLE > "%LAUNCH_LOG_DIR%\%RUN_ID%_schedule_disable.txt" 2>&1 || goto schedule_disable_failed
echo disabled>"%LAUNCH_LOG_DIR%\%RUN_ID%_schedule_disabled.marker"
schtasks /Query /TN "%TASK_NAME%" /V /FO LIST > "%LAUNCH_LOG_DIR%\%RUN_ID%_schedule_query.txt" 2>&1
echo launched>"%LAUNCH_LOG_DIR%\%RUN_ID%_launched.marker"
exit /b 0

:schedule_disable_failed
schtasks /End /TN "%TASK_NAME%" > "%LAUNCH_LOG_DIR%\%RUN_ID%_schedule_disable_cleanup.txt" 2>&1
schtasks /Delete /TN "%TASK_NAME%" /F >> "%LAUNCH_LOG_DIR%\%RUN_ID%_schedule_disable_cleanup.txt" 2>&1
exit /b 1

:prepare_source
set EXPECTED_COMMIT=%~1
set RUN_ROOT=%RUNS_ROOT%\%RUN_ID%
set SOURCE_ROOT=%RUN_ROOT%\source
if not exist "%RUN_ROOT%" mkdir "%RUN_ROOT%"
if exist "%SOURCE_ROOT%\.git" (
  cd /d "%SOURCE_ROOT%" || exit /b 1
  for /f "delims=" %%S in ('git status --porcelain') do exit /b 1
  git fetch origin || exit /b 1
) else (
  if exist "%SOURCE_ROOT%" rmdir /s /q "%SOURCE_ROOT%"
  git clone --no-checkout "%REPO_URL%" "%SOURCE_ROOT%" || exit /b 1
)
cd /d "%SOURCE_ROOT%" || exit /b 1
git checkout --detach "%EXPECTED_COMMIT%" || exit /b 1
for /f "delims=" %%S in ('git status --porcelain') do exit /b 1
for /f "delims=" %%H in ('git rev-parse HEAD') do set ACTUAL_COMMIT=%%H
if /I not "%ACTUAL_COMMIT%"=="%EXPECTED_COMMIT%" exit /b 1
git rev-parse HEAD > "%RUN_ROOT%\source_expected_commit.txt" || exit /b 1
git rev-parse HEAD > "%RUN_ROOT%\source_revision_before_schedule.txt" || exit /b 1
exit /b 0
