@echo off
setlocal EnableExtensions DisableDelayedExpansion

set RECOVERY_COMMIT=%~1
if "%RECOVERY_COMMIT%"=="" exit /b 2

set REPO_URL=git@github.com:swordfate3/blockcipher-structure-adaptive-nd-v1.git
set RUNS_ROOT=G:\lxy\blockcipher-structure-adaptive-nd-runs
set RECOVERY_ID=i1_gift64_mainstream_performance_1m_postprocess_repair_20260715
set RECOVERY_ROOT=%RUNS_ROOT%\%RECOVERY_ID%
set SOURCE_ROOT=%RECOVERY_ROOT%\source
set LOG_DIR=%RECOVERY_ROOT%\logs
set LAUNCH_LOG_DIR=%RUNS_ROOT%\launcher_logs
set TASK=I1_GIFT64_MAINSTREAM_1M_POSTPROCESS_REPAIR
set GITHUB_SSH_KEY=C:/Users/1304Lijinlin/.ssh/github_blockcipher_20260612_result_pusher_ed25519
set GIT_SSH_COMMAND=ssh -i %GITHUB_SSH_KEY% -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new

if not exist "%RUNS_ROOT%" mkdir "%RUNS_ROOT%"
if not exist "%RECOVERY_ROOT%" mkdir "%RECOVERY_ROOT%"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
if not exist "%LAUNCH_LOG_DIR%" mkdir "%LAUNCH_LOG_DIR%"

if exist "%SOURCE_ROOT%\.git" (
  cd /d "%SOURCE_ROOT%" || exit /b 1
  for /f "delims=" %%S in ('git status --porcelain') do exit /b 1
  git fetch origin || exit /b 1
) else (
  if exist "%SOURCE_ROOT%" rmdir /S /Q "%SOURCE_ROOT%"
  git clone --no-checkout "%REPO_URL%" "%SOURCE_ROOT%" || exit /b 1
)
cd /d "%SOURCE_ROOT%" || exit /b 1
git checkout --detach "%RECOVERY_COMMIT%" || exit /b 1
for /f "delims=" %%S in ('git status --porcelain') do exit /b 1
for /f "delims=" %%H in ('git rev-parse HEAD') do set ACTUAL_COMMIT=%%H
if /I not "%ACTUAL_COMMIT%"=="%RECOVERY_COMMIT%" exit /b 1
git rev-parse HEAD > "%RECOVERY_ROOT%\source_expected_commit.txt" || exit /b 1

set RECOVERY_CMD=%SOURCE_ROOT%\configs\remote\generated\recover_i1_gift64_mainstream_performance_1m_20260715.cmd
schtasks /Create /TN "%TASK%" /SC ONCE /ST 23:59 /RU SYSTEM /RL HIGHEST /TR "cmd.exe /c %RECOVERY_CMD% %RECOVERY_COMMIT%" /F > "%LOG_DIR%\schedule_create.txt" 2>&1 || exit /b 1
schtasks /Run /I /TN "%TASK%" > "%LOG_DIR%\schedule_run.txt" 2>&1 || exit /b 1
schtasks /Query /TN "%TASK%" /V /FO LIST > "%LOG_DIR%\schedule_query.txt" 2>&1
echo launched>"%RECOVERY_ROOT%\recovery_launched.marker"
exit /b 0
