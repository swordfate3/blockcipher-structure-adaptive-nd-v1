@echo off
setlocal EnableExtensions

set REPO_URL=git@github.com:swordfate3/blockcipher-structure-adaptive-nd-v1.git
set RUNS_ROOT=G:\lxy\blockcipher-structure-adaptive-nd-runs
set LAUNCH_LOG_DIR=%RUNS_ROOT%\launcher_logs
set GITHUB_SSH_KEY=C:/Users/1304Lijinlin/.ssh/github_blockcipher_20260612_result_pusher_ed25519
set GIT_SSH_COMMAND=ssh -i %GITHUB_SSH_KEY% -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new
set SEED0_ID=i1_gift64_cross_spn_typed_transfer_r3_65536_seed0
set SEED1_ID=i1_gift64_cross_spn_typed_transfer_r3_65536_seed1
set TASK0=I1_E4_R3_GIFT64_SEED0_GPU0
set TASK1=I1_E4_R3_GIFT64_SEED1_GPU1
set SOURCE_COMMIT=%~1
if "%SOURCE_COMMIT%"=="" exit /b 2

if not exist "%RUNS_ROOT%" mkdir "%RUNS_ROOT%"
if not exist "%LAUNCH_LOG_DIR%" mkdir "%LAUNCH_LOG_DIR%"

call :prepare_source %SEED0_ID% %SOURCE_COMMIT% || exit /b 1
call :prepare_source %SEED1_ID% %SOURCE_COMMIT% || exit /b 1

set CMD0=%RUNS_ROOT%\%SEED0_ID%\source\configs\remote\generated\run_i1_gift64_cross_spn_typed_transfer_r3_65536_20260714.cmd
set CMD1=%RUNS_ROOT%\%SEED1_ID%\source\configs\remote\generated\run_i1_gift64_cross_spn_typed_transfer_r3_65536_20260714.cmd

schtasks /Create /TN "%TASK0%" /SC ONCE /ST 23:59 /RU SYSTEM /RL HIGHEST /TR "cmd.exe /c %CMD0% 0 0" /F > "%LAUNCH_LOG_DIR%\%SEED0_ID%_schedule_create.txt" 2>&1 || exit /b 1
schtasks /Create /TN "%TASK1%" /SC ONCE /ST 23:59 /RU SYSTEM /RL HIGHEST /TR "cmd.exe /c %CMD1% 1 1" /F > "%LAUNCH_LOG_DIR%\%SEED1_ID%_schedule_create.txt" 2>&1 || exit /b 1
schtasks /Run /I /TN "%TASK0%" > "%LAUNCH_LOG_DIR%\%SEED0_ID%_schedule_run.txt" 2>&1 || exit /b 1
schtasks /Run /I /TN "%TASK1%" > "%LAUNCH_LOG_DIR%\%SEED1_ID%_schedule_run.txt" 2>&1 || exit /b 1
schtasks /Query /TN "%TASK0%" /V /FO LIST > "%LAUNCH_LOG_DIR%\%SEED0_ID%_schedule_query.txt" 2>&1
schtasks /Query /TN "%TASK1%" /V /FO LIST > "%LAUNCH_LOG_DIR%\%SEED1_ID%_schedule_query.txt" 2>&1
echo launched>"%LAUNCH_LOG_DIR%\i1_gift64_cross_spn_typed_transfer_r3_65536_launched.marker"
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
