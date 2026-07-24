@echo off
setlocal EnableExtensions

set SOURCE_COMMIT=%~1
set PHYSICAL_GPU=%~2
if "%SOURCE_COMMIT%"=="" exit /b 2
if "%PHYSICAL_GPU%"=="" exit /b 2

set REPO_URL=git@github.com:swordfate3/blockcipher-structure-adaptive-nd-v1.git
set RUN_ID=i1_rtg3a_skinny64_general_gf2_formal_1000000_seed0_20260725
set RUNS_ROOT=G:\lxy\blockcipher-structure-adaptive-nd-runs
set RUN_ROOT=%RUNS_ROOT%\%RUN_ID%
set SOURCE_ROOT=%RUN_ROOT%\source
set SCHEDULE_ROOT=G:\lxy\scheduled-runs
set SCHEDULE_CMD=%SCHEDULE_ROOT%\i1_rtg3a_s0.cmd
set LAUNCH_LOG_DIR=%RUNS_ROOT%\launcher_logs
set TASK_NAME=I1_RTG3A_SKINNY64_S0_GPU%PHYSICAL_GPU%
set PY=F:\Anaconda\envs\DWT\torch310\python.exe
set RTG2B_SEED0_ID=i1_rtg2b_skinny64_general_gf2_scale_262144_seed0_20260724
set RTG2B_SEED1_ID=i1_rtg2b_skinny64_general_gf2_scale_262144_seed1_20260724
set RTG2B_SEED0_GATE=%RUNS_ROOT%\%RTG2B_SEED0_ID%\source\results_archive\%RTG2B_SEED0_ID%\gate.json
set RTG2B_SEED1_GATE=%RUNS_ROOT%\%RTG2B_SEED1_ID%\source\results_archive\%RTG2B_SEED1_ID%\gate.json
set GITHUB_SSH_KEY=C:/Users/1304Lijinlin/.ssh/github_blockcipher_20260612_result_pusher_ed25519
set GIT_SSH_COMMAND=ssh -i %GITHUB_SSH_KEY% -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new

if not exist "%RUNS_ROOT%" mkdir "%RUNS_ROOT%"
if not exist "%SCHEDULE_ROOT%" mkdir "%SCHEDULE_ROOT%"
if not exist "%LAUNCH_LOG_DIR%" mkdir "%LAUNCH_LOG_DIR%"
if not exist "%RUN_ROOT%" mkdir "%RUN_ROOT%"

"%PY%" -c "import json,pathlib,sys; paths=[pathlib.Path(r'%RTG2B_SEED0_GATE%'),pathlib.Path(r'%RTG2B_SEED1_GATE%')]; gates=[json.loads(p.read_text(encoding='utf-8')) if p.is_file() else {} for p in paths]; expected=['innovation1_rtg2b_skinny_scale_seed0_supported','innovation1_rtg2b_skinny_scale_seed1_supported']; ok=all(g.get('phase') == 'rtg2b' and g.get('seed') == i and g.get('status') == 'pass' and g.get('decision') == expected[i] and all(g.get('protocol_checks', {}).values()) and all(g.get('research_checks', {}).values()) for i,g in enumerate(gates)); sys.exit(0 if ok else 7)" || exit /b 7

if exist "%SOURCE_ROOT%\.git" (
  cd /d "%SOURCE_ROOT%" || exit /b 1
  for /f "delims=" %%S in ('git status --porcelain') do exit /b 1
  git fetch origin || exit /b 1
) else (
  if exist "%SOURCE_ROOT%" rmdir /s /q "%SOURCE_ROOT%"
  git clone --no-checkout "%REPO_URL%" "%SOURCE_ROOT%" || exit /b 1
)
cd /d "%SOURCE_ROOT%" || exit /b 1
git checkout --detach "%SOURCE_COMMIT%" || exit /b 1
for /f "delims=" %%S in ('git status --porcelain') do exit /b 1
for /f "delims=" %%H in ('git rev-parse HEAD') do set ACTUAL_COMMIT=%%H
if /I not "%ACTUAL_COMMIT%"=="%SOURCE_COMMIT%" exit /b 1
git rev-parse HEAD > "%RUN_ROOT%\source_expected_commit.txt" || exit /b 1
git rev-parse HEAD > "%RUN_ROOT%\source_revision_before_schedule.txt" || exit /b 1

set RUN_CMD=%SOURCE_ROOT%\configs\remote\generated\run_i1_rtg3a_skinny64_general_gf2_formal_1000000_seed0_20260725.cmd
>"%SCHEDULE_CMD%" echo @echo off
>>"%SCHEDULE_CMD%" echo call "%RUN_CMD%" %PHYSICAL_GPU%

schtasks /Create /TN "%TASK_NAME%" /SC ONCE /ST 23:59 /RU SYSTEM /RL HIGHEST /TR "cmd.exe /c %SCHEDULE_CMD%" /F > "%LAUNCH_LOG_DIR%\%RUN_ID%_schedule_create.txt" 2>&1 || exit /b 1
schtasks /Run /I /TN "%TASK_NAME%" > "%LAUNCH_LOG_DIR%\%RUN_ID%_schedule_run.txt" 2>&1 || exit /b 1
schtasks /Query /TN "%TASK_NAME%" /V /FO LIST > "%LAUNCH_LOG_DIR%\%RUN_ID%_schedule_query.txt" 2>&1
echo launched>"%LAUNCH_LOG_DIR%\%RUN_ID%_launched.marker"
exit /b 0
