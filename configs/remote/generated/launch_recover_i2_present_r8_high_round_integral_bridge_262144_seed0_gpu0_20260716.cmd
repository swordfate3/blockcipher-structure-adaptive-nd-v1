@echo off
setlocal EnableExtensions DisableDelayedExpansion

set REPO_URL=git@github.com:swordfate3/blockcipher-structure-adaptive-nd-v1.git
set RUN_ID=i2_present_r8_high_round_integral_bridge_262144_seed0_gpu0_20260716
set RUNS_ROOT=G:\lxy\blockcipher-structure-adaptive-nd-runs
set RUN_ROOT=%RUNS_ROOT%\%RUN_ID%
set RECOVERY_SOURCE=%RUN_ROOT%\recovery_source
set RECOVERY_COMMIT=%~1
if "%RECOVERY_COMMIT%"=="" exit /b 2

if not exist "%RUNS_ROOT%" mkdir "%RUNS_ROOT%"
if exist "%RECOVERY_SOURCE%\.git" (
  cd /d "%RECOVERY_SOURCE%" || exit /b 1
  for /f "delims=" %%S in ('git status --porcelain') do exit /b 1
  git fetch origin || exit /b 1
) else (
  if exist "%RECOVERY_SOURCE%" exit /b 3
  git clone --no-checkout "%REPO_URL%" "%RECOVERY_SOURCE%" || exit /b 1
)

cd /d "%RECOVERY_SOURCE%" || exit /b 1
git config --global --add safe.directory G:/lxy/blockcipher-structure-adaptive-nd-runs/%RUN_ID%/recovery_source
git checkout --detach "%RECOVERY_COMMIT%" || exit /b 1
for /f "delims=" %%S in ('git status --porcelain') do exit /b 1
for /f "delims=" %%H in ('git rev-parse HEAD') do set ACTUAL_COMMIT=%%H
if /I not "%ACTUAL_COMMIT%"=="%RECOVERY_COMMIT%" exit /b 1

call "%RECOVERY_SOURCE%\configs\remote\generated\recover_i2_present_r8_high_round_integral_bridge_262144_seed0_gpu0_20260716.cmd" "%ACTUAL_COMMIT%"
exit /b %ERRORLEVEL%
