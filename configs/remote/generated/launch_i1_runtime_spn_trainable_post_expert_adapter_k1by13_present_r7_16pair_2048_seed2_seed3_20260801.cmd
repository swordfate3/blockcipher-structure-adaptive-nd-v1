@echo off
setlocal EnableExtensions

set SOURCE_COMMIT=%~1
set PHYSICAL_GPU=%~2
if "%SOURCE_COMMIT%"=="" exit /b 2
if "%PHYSICAL_GPU%"=="" exit /b 2
if not "%PHYSICAL_GPU%"=="0" if not "%PHYSICAL_GPU%"=="1" exit /b 3

set REPO_URL=git@github.com:swordfate3/blockcipher-structure-adaptive-nd-v1.git
set RUN_ID=i1_runtime_spn_trainable_post_expert_adapter_k1by13_present_r7_16pair_2048_seed2_seed3_20260801
set RUNS_ROOT=G:\lxy\blockcipher-structure-adaptive-nd-runs
set SOURCE_ROOT=G:\lxy\bcnd-k1by13-src
set SCHEDULE_ROOT=G:\lxy\scheduled-runs
set LAUNCH_LOG_DIR=%RUNS_ROOT%\launcher_logs
set GITHUB_SSH_KEY=C:/Users/1304Lijinlin/.ssh/github_blockcipher_20260612_result_pusher_ed25519
set GIT_SSH_COMMAND=ssh -i %GITHUB_SSH_KEY% -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o Hostname=ssh.github.com -p 443
set TASK_NAME=I1_K1BY13_PRESENT_R7_GPU%PHYSICAL_GPU%
set SCHEDULE_CMD=%SCHEDULE_ROOT%\i1_k1by13_present_r7_gpu%PHYSICAL_GPU%.cmd

if not exist "%RUNS_ROOT%" mkdir "%RUNS_ROOT%"
if not exist "%SCHEDULE_ROOT%" mkdir "%SCHEDULE_ROOT%"
if not exist "%LAUNCH_LOG_DIR%" mkdir "%LAUNCH_LOG_DIR%"
call :prepare_source "%SOURCE_COMMIT%" || exit /b 1
call :stage_source_evidence || exit /b 1
set RUN_CMD=%SOURCE_ROOT%\configs\remote\generated\run_%RUN_ID%.cmd
>"%SCHEDULE_CMD%" echo @echo off
>>"%SCHEDULE_CMD%" echo call "%RUN_CMD%" %PHYSICAL_GPU%
schtasks /Create /TN "%TASK_NAME%" /SC ONCE /ST 23:59 /RU SYSTEM /RL HIGHEST /TR "cmd.exe /c %SCHEDULE_CMD%" /F > "%LAUNCH_LOG_DIR%\%RUN_ID%_schedule_create.txt" 2>&1 || exit /b 1
schtasks /Run /I /TN "%TASK_NAME%" > "%LAUNCH_LOG_DIR%\%RUN_ID%_schedule_run.txt" 2>&1 || exit /b 1
schtasks /Query /TN "%TASK_NAME%" /V /FO LIST > "%LAUNCH_LOG_DIR%\%RUN_ID%_schedule_query.txt" 2>&1
echo launched>"%LAUNCH_LOG_DIR%\%RUN_ID%_launched.marker"
exit /b 0

:prepare_source
set EXPECTED_COMMIT=%~1
set RUN_ROOT=%RUNS_ROOT%\%RUN_ID%
if not exist "%RUN_ROOT%" mkdir "%RUN_ROOT%"
if exist "%SOURCE_ROOT%\.git" (
  cd /d "%SOURCE_ROOT%" || exit /b 1
  for /f "delims=" %%S in ('git status --porcelain') do exit /b 1
  git fetch origin || exit /b 1
) else (
  if exist "%SOURCE_ROOT%" exit /b 1
  git clone --no-checkout "%REPO_URL%" "%SOURCE_ROOT%" || exit /b 1
)
cd /d "%SOURCE_ROOT%" || exit /b 1
git checkout --detach "%EXPECTED_COMMIT%" || exit /b 1
for /f "delims=" %%S in ('git status --porcelain') do exit /b 1
for /f "delims=" %%H in ('git rev-parse HEAD') do set ACTUAL_COMMIT=%%H
if /I not "%ACTUAL_COMMIT%"=="%EXPECTED_COMMIT%" exit /b 1
git rev-parse HEAD > "%RUN_ROOT%\source_expected_commit.txt" || exit /b 1
exit /b 0

:stage_source_evidence
set EVIDENCE_ROOT=%RUN_ROOT%\source_evidence
set K1BY3_TARGET=%SOURCE_ROOT%\outputs\local_diagnostic\i1_runtime_spn_permutation_expert_k1by3_present_r7_16pair_2048_seed2_seed3_20260801
set K1BY12_TARGET=%SOURCE_ROOT%\outputs\local_audit\i1_runtime_spn_post_expert_edge_residual_k1by12_present_r7_seed2_seed3_20260801
if not exist "%EVIDENCE_ROOT%\k1by3\results.jsonl" exit /b 1
if not exist "%EVIDENCE_ROOT%\k1by3\gate.json" exit /b 1
if not exist "%EVIDENCE_ROOT%\k1by3\validation.json" exit /b 1
if not exist "%EVIDENCE_ROOT%\k1by12\results.jsonl" exit /b 1
if not exist "%EVIDENCE_ROOT%\k1by12\gate.json" exit /b 1
if not exist "%EVIDENCE_ROOT%\k1by12\validation.json" exit /b 1
if not exist "%K1BY3_TARGET%" mkdir "%K1BY3_TARGET%"
if not exist "%K1BY12_TARGET%" mkdir "%K1BY12_TARGET%"
copy /y "%EVIDENCE_ROOT%\k1by3\results.jsonl" "%K1BY3_TARGET%\results.jsonl" > nul || exit /b 1
copy /y "%EVIDENCE_ROOT%\k1by3\gate.json" "%K1BY3_TARGET%\gate.json" > nul || exit /b 1
copy /y "%EVIDENCE_ROOT%\k1by3\validation.json" "%K1BY3_TARGET%\validation.json" > nul || exit /b 1
copy /y "%EVIDENCE_ROOT%\k1by12\results.jsonl" "%K1BY12_TARGET%\results.jsonl" > nul || exit /b 1
copy /y "%EVIDENCE_ROOT%\k1by12\gate.json" "%K1BY12_TARGET%\gate.json" > nul || exit /b 1
copy /y "%EVIDENCE_ROOT%\k1by12\validation.json" "%K1BY12_TARGET%\validation.json" > nul || exit /b 1
cd /d "%SOURCE_ROOT%" || exit /b 1
for /f "delims=" %%S in ('git status --porcelain') do exit /b 1
exit /b 0
