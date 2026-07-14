@echo off
setlocal EnableExtensions

set REPO_URL=git@github.com:swordfate3/blockcipher-structure-adaptive-nd-v1.git
set RUNS_ROOT=G:\lxy\blockcipher-structure-adaptive-nd-runs
set LAUNCH_LOG_DIR=%RUNS_ROOT%\launcher_logs
set GITHUB_SSH_KEY=C:/Users/1304Lijinlin/.ssh/github_blockcipher_20260612_result_pusher_ed25519
set GIT_SSH_COMMAND=ssh -i %GITHUB_SSH_KEY% -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new
set SEED2_ID=i1_gift64_cross_spn_target_adaptation_r4_65536_seed2
set SEED3_ID=i1_gift64_cross_spn_target_adaptation_r4_65536_seed3
set TASK2=I1_E4_R4_GIFT64_SEED2_GPU0
set TASK3=I1_E4_R4_GIFT64_SEED3_GPU1
set SOURCE_COMMIT=%~1
if "%SOURCE_COMMIT%"=="" exit /b 2

if not exist "%RUNS_ROOT%" mkdir "%RUNS_ROOT%"
if not exist "%LAUNCH_LOG_DIR%" mkdir "%LAUNCH_LOG_DIR%"

call :prepare_source %SEED2_ID% %SOURCE_COMMIT% || exit /b 1
call :prepare_source %SEED3_ID% %SOURCE_COMMIT% || exit /b 1

set CMD2=%RUNS_ROOT%\%SEED2_ID%\source\configs\remote\generated\run_i1_gift64_cross_spn_target_adaptation_r4_65536_20260715.cmd
set CMD3=%RUNS_ROOT%\%SEED3_ID%\source\configs\remote\generated\run_i1_gift64_cross_spn_target_adaptation_r4_65536_20260715.cmd

schtasks /Create /TN "%TASK2%" /SC ONCE /ST 23:59 /RU SYSTEM /RL HIGHEST /TR "cmd.exe /c %CMD2% 2 0" /F > "%LAUNCH_LOG_DIR%\%SEED2_ID%_schedule_create.txt" 2>&1 || exit /b 1
schtasks /Create /TN "%TASK3%" /SC ONCE /ST 23:59 /RU SYSTEM /RL HIGHEST /TR "cmd.exe /c %CMD3% 3 1" /F > "%LAUNCH_LOG_DIR%\%SEED3_ID%_schedule_create.txt" 2>&1 || exit /b 1
schtasks /Run /I /TN "%TASK2%" > "%LAUNCH_LOG_DIR%\%SEED2_ID%_schedule_run.txt" 2>&1 || exit /b 1
schtasks /Run /I /TN "%TASK3%" > "%LAUNCH_LOG_DIR%\%SEED3_ID%_schedule_run.txt" 2>&1 || exit /b 1
schtasks /Query /TN "%TASK2%" /V /FO LIST > "%LAUNCH_LOG_DIR%\%SEED2_ID%_schedule_query.txt" 2>&1
schtasks /Query /TN "%TASK3%" /V /FO LIST > "%LAUNCH_LOG_DIR%\%SEED3_ID%_schedule_query.txt" 2>&1
echo launched>"%LAUNCH_LOG_DIR%\i1_gift64_cross_spn_target_adaptation_r4_65536_launched.marker"
exit /b 0

:prepare_source
set RUN_ID=%~1
set EXPECTED_COMMIT=%~2
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
