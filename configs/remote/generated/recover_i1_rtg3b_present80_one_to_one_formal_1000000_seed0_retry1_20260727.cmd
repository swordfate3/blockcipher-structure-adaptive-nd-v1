@echo off
setlocal EnableExtensions DisableDelayedExpansion

set RECOVERY_COMMIT=%~1
if "%RECOVERY_COMMIT%"=="" exit /b 2

for %%I in ("%~dp0..\..\..") do set RECOVERY_SOURCE=%%~fI
set RUN_ID=i1_rtg3b_present80_one_to_one_formal_1000000_seed0_retry1_20260727
set RUNS_ROOT=G:\lxy\blockcipher-structure-adaptive-nd-runs
set RUN_ROOT=%RUNS_ROOT%\%RUN_ID%
set TRAIN_SOURCE=%RUN_ROOT%\source
set LOG_DIR=%RUN_ROOT%\logs
set RESULTS_DIR=%RUN_ROOT%\results
set CHECKPOINT_DIR=%RUN_ROOT%\checkpoints
set ARCHIVE_DIR=%TRAIN_SOURCE%\results_archive\%RUN_ID%
set PLAN=%RECOVERY_SOURCE%\configs\experiment\innovation1\innovation1_spn_present80_runtime_e4_formal_rtg3b_1000000_seed0.csv
set REMOTE_CONFIG=%RECOVERY_SOURCE%\configs\remote\innovation1_rtg3b_present80_one_to_one_formal_1000000_seed0_retry1_gpu0_20260727.json
set PY=F:\Anaconda\envs\DWT\torch310\python.exe
set PYTHONPATH=%RECOVERY_SOURCE%\src
set GITHUB_SSH_KEY=C:/Users/1304Lijinlin/.ssh/github_blockcipher_20260612_result_pusher_ed25519
set GIT_SSH_COMMAND=ssh -i %GITHUB_SSH_KEY% -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new

cd /d "%RECOVERY_SOURCE%" || goto recovery_failed
for /f "delims=" %%S in ('git status --porcelain') do goto recovery_failed
for /f "delims=" %%H in ('git rev-parse HEAD') do set ACTUAL_RECOVERY_COMMIT=%%H
if /I not "%ACTUAL_RECOVERY_COMMIT%"=="%RECOVERY_COMMIT%" goto recovery_failed

if not exist "%RUN_ROOT%\source_expected_commit.txt" goto recovery_failed
set TRAINING_COMMIT=
set /p TRAINING_COMMIT=<"%RUN_ROOT%\source_expected_commit.txt"
if "%TRAINING_COMMIT%"=="" goto recovery_failed
for /f "delims=" %%H in ('git -C "%TRAIN_SOURCE%" rev-parse HEAD') do set ACTUAL_TRAINING_COMMIT=%%H
if /I not "%ACTUAL_TRAINING_COMMIT%"=="%TRAINING_COMMIT%" goto recovery_failed
for /f "delims=" %%S in ('git -C "%TRAIN_SOURCE%" status --porcelain') do goto recovery_failed

if not exist "%RESULTS_DIR%\results.jsonl" goto recovery_failed
if not exist "%CHECKPOINT_DIR%\row0001_present_runtime_e4_equivariant_true_seed0.pt" goto recovery_failed
if not exist "%CHECKPOINT_DIR%\row0002_present_runtime_e4_equivariant_corrupted_seed0.pt" goto recovery_failed
if not exist "%CHECKPOINT_DIR%\row0003_present_runtime_e4_equivariant_independent_seed0.pt" goto recovery_failed
findstr /C:"No module named 'matplotlib'" "%LOG_DIR%\%RUN_ID%_gate_stderr.txt" > nul || goto recovery_failed

echo started>"%LOG_DIR%\%RUN_ID%_postprocess_recovery_started.marker"
"%PY%" "%RECOVERY_SOURCE%\scripts\validate-results" ^
  --plan "%PLAN%" ^
  --results "%RESULTS_DIR%\results.jsonl" ^
  --expected-rows 3 ^
  --output "%RESULTS_DIR%\validation-plan.recovery.json" ^
  > "%LOG_DIR%\%RUN_ID%_recovery_validation_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_recovery_validation_stderr.txt" || goto recovery_failed

"%PY%" "%RECOVERY_SOURCE%\scripts\gate-runtime-spn-present-transfer" ^
  --run-id "%RUN_ID%" ^
  --run-root "%RESULTS_DIR%" ^
  --seed 0 ^
  --samples-per-class 1000000 ^
  --phase rtg3b ^
  --no-plot ^
  > "%LOG_DIR%\%RUN_ID%_recovery_gate_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_recovery_gate_stderr.txt" || goto recovery_failed

"%PY%" -c "import json; from pathlib import Path; from blockcipher_nd.cli.gate_runtime_spn_skinny_medium import _read_jsonl, _verify_checkpoint_payloads; root=Path(r'%RUN_ROOT%'); report=_verify_checkpoint_payloads(_read_jsonl(root/'results'/'results.jsonl'), root/'checkpoints'); assert report['status']=='pass'; (root/'results'/'checkpoint-verification.recovery.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8'); print(json.dumps(report,ensure_ascii=False,sort_keys=True))" > "%LOG_DIR%\%RUN_ID%_recovery_checkpoint_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_recovery_checkpoint_stderr.txt" || goto recovery_failed

set RESULT_LINES=0
for /f "tokens=3" %%L in ('find /c /v "" "%RESULTS_DIR%\results.jsonl"') do set RESULT_LINES=%%L
echo result_lines=%RESULT_LINES%>"%LOG_DIR%\%RUN_ID%_result_gate.recovery.txt"
echo expected_rows=3>>"%LOG_DIR%\%RUN_ID%_result_gate.recovery.txt"
if not "%RESULT_LINES%"=="3" goto recovery_failed

if not exist "%ARCHIVE_DIR%" mkdir "%ARCHIVE_DIR%"
copy /Y "%RESULTS_DIR%\results.jsonl" "%ARCHIVE_DIR%\results.jsonl" > nul || goto recovery_failed
copy /Y "%RESULTS_DIR%\validation-plan.json" "%ARCHIVE_DIR%\validation-plan.json" > nul || goto recovery_failed
copy /Y "%RESULTS_DIR%\validation-plan.recovery.json" "%ARCHIVE_DIR%\validation-plan.recovery.json" > nul || goto recovery_failed
copy /Y "%RESULTS_DIR%\validation.json" "%ARCHIVE_DIR%\validation.json" > nul || goto recovery_failed
copy /Y "%RESULTS_DIR%\gate.json" "%ARCHIVE_DIR%\gate.json" > nul || goto recovery_failed
copy /Y "%RESULTS_DIR%\summary.json" "%ARCHIVE_DIR%\summary.json" > nul || goto recovery_failed
copy /Y "%RESULTS_DIR%\history.csv" "%ARCHIVE_DIR%\history.csv" > nul || goto recovery_failed
copy /Y "%RESULTS_DIR%\checkpoint-verification.recovery.json" "%ARCHIVE_DIR%\checkpoint-verification.json" > nul || goto recovery_failed
copy /Y "%RESULTS_DIR%\progress.jsonl" "%ARCHIVE_DIR%\postprocess-progress.jsonl" > nul || goto recovery_failed
copy /Y "%LOG_DIR%\progress.jsonl" "%ARCHIVE_DIR%\progress.jsonl" > nul || goto recovery_failed
copy /Y "%LOG_DIR%\%RUN_ID%_git_revision.txt" "%ARCHIVE_DIR%\git_revision.txt" > nul || goto recovery_failed
copy /Y "%LOG_DIR%\%RUN_ID%_git_status_before_run.txt" "%ARCHIVE_DIR%\git_status_before_run.txt" > nul || goto recovery_failed
copy /Y "%LOG_DIR%\%RUN_ID%_gpu_info.txt" "%ARCHIVE_DIR%\gpu_info.txt" > nul || goto recovery_failed
copy /Y "%LOG_DIR%\%RUN_ID%_torch_info.txt" "%ARCHIVE_DIR%\torch_info.txt" > nul || goto recovery_failed
copy /Y "%LOG_DIR%\%RUN_ID%_readiness.txt" "%ARCHIVE_DIR%\readiness.txt" > nul || goto recovery_failed
copy /Y "%LOG_DIR%\%RUN_ID%_cache_reuse_audit.txt" "%ARCHIVE_DIR%\cache_reuse_audit.txt" > nul || goto recovery_failed
copy /Y "%LOG_DIR%\%RUN_ID%_result_gate.recovery.txt" "%ARCHIVE_DIR%\result_gate.txt" > nul || goto recovery_failed
copy /Y "%LOG_DIR%\%RUN_ID%_train_stdout.txt" "%ARCHIVE_DIR%\train_stdout.txt" > nul || goto recovery_failed
copy /Y "%LOG_DIR%\%RUN_ID%_train_stderr.txt" "%ARCHIVE_DIR%\train_stderr.txt" > nul || goto recovery_failed
copy /Y "%LOG_DIR%\%RUN_ID%_validation_stdout.txt" "%ARCHIVE_DIR%\validation_stdout.txt" > nul || goto recovery_failed
copy /Y "%LOG_DIR%\%RUN_ID%_validation_stderr.txt" "%ARCHIVE_DIR%\validation_stderr.txt" > nul || goto recovery_failed
copy /Y "%LOG_DIR%\%RUN_ID%_gate_stderr.txt" "%ARCHIVE_DIR%\original_gate_stderr.txt" > nul || goto recovery_failed
copy /Y "%REMOTE_CONFIG%" "%ARCHIVE_DIR%\remote_config.json" > nul || goto recovery_failed
copy /Y "%PLAN%" "%ARCHIVE_DIR%\plan.csv" > nul || goto recovery_failed
copy /Y "%LOG_DIR%\%RUN_ID%_failed.marker" "%ARCHIVE_DIR%\original_failed.marker" > nul || goto recovery_failed
echo %TRAINING_COMMIT%>"%ARCHIVE_DIR%\training_commit.txt"
echo %RECOVERY_COMMIT%>"%ARCHIVE_DIR%\recovery_commit.txt"
echo recovered_existing_results_without_retraining>"%ARCHIVE_DIR%\recovered_without_retraining.marker"
echo plot_deferred_to_verified_local_retrieval>"%ARCHIVE_DIR%\plot_deferred.marker"
echo visual_qa_pending>"%ARCHIVE_DIR%\visual_qa_pending.marker"
echo * -text>"%ARCHIVE_DIR%\.gitattributes"
"%PY%" -c "import hashlib,pathlib; root=pathlib.Path(r'%ARCHIVE_DIR%'); files=sorted(p for p in root.rglob('*') if p.is_file() and not p.name == 'SHA256SUMS'); (root/'SHA256SUMS').write_text('\n'.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.relative_to(root).as_posix() for p in files)+'\n',encoding='utf-8')" > "%LOG_DIR%\%RUN_ID%_recovery_hash_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_recovery_hash_stderr.txt" || goto recovery_failed

cd /d "%TRAIN_SOURCE%" || goto recovery_failed
git config user.name "remote-experiment"
git config user.email "remote-experiment@local.invalid"
git checkout -B results/%RUN_ID% "%TRAINING_COMMIT%" > "%LOG_DIR%\%RUN_ID%_recovery_branch_checkout.txt" 2>&1 || goto recovery_failed
git add "results_archive\%RUN_ID%" || goto recovery_failed
git commit -m "results: %RUN_ID% recovered postprocess" > "%LOG_DIR%\%RUN_ID%_recovery_branch_commit.txt" 2>&1 || goto recovery_failed
git push origin HEAD:refs/heads/results/%RUN_ID% > "%LOG_DIR%\%RUN_ID%_recovery_branch_push.txt" 2>&1 || goto recovery_failed
git rev-parse HEAD > "%LOG_DIR%\%RUN_ID%_result_branch_revision.txt" 2>&1 || goto recovery_failed
echo pushed>"%LOG_DIR%\%RUN_ID%_result_branch_pushed.marker"
echo recovered>"%LOG_DIR%\%RUN_ID%_done.marker"
exit /b 0

:recovery_failed
echo failed>"%LOG_DIR%\%RUN_ID%_postprocess_recovery_failed.marker"
exit /b 1
