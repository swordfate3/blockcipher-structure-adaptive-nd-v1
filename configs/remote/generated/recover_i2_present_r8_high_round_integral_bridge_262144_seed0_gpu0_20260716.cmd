@echo off
setlocal EnableExtensions DisableDelayedExpansion

set RECOVERY_COMMIT=%~1
if "%RECOVERY_COMMIT%"=="" exit /b 2

for %%I in ("%~dp0..\..\..") do set RECOVERY_SOURCE=%%~fI
set RUN_ID=i2_present_r8_high_round_integral_bridge_262144_seed0_gpu0_20260716
set EXPECTED_SOURCE_COMMIT=4b3a2c33cc323b5586533f0fffb78edbe70e0adf
set RUNS_ROOT=G:\lxy\blockcipher-structure-adaptive-nd-runs
set RUN_ROOT=%RUNS_ROOT%\%RUN_ID%
set TRAIN_SOURCE=%RUN_ROOT%\source
set CACHE_ROOT=%RUN_ROOT%\dataset_cache
set LOG_DIR=%RUN_ROOT%\logs
set RESULTS_DIR=%RUN_ROOT%\results
set ARCHIVE_DIR=%TRAIN_SOURCE%\results_archive\%RUN_ID%
set REMOTE_CONFIG=%RECOVERY_SOURCE%\configs\remote\innovation2_present_r8_high_round_integral_bridge_262144_seed0_gpu0_20260716.json
set PY=F:\Anaconda\envs\DWT\torch310\python.exe
set GITHUB_SSH_KEY=C:/Users/1304Lijinlin/.ssh/github_blockcipher_20260612_result_pusher_ed25519
set GIT_SSH_COMMAND=ssh -i %GITHUB_SSH_KEY% -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new

cd /d "%RECOVERY_SOURCE%" || goto recovery_failed
for /f "delims=" %%S in ('git status --porcelain') do goto recovery_failed
for /f "delims=" %%H in ('git rev-parse HEAD') do set ACTUAL_RECOVERY_COMMIT=%%H
if /I not "%ACTUAL_RECOVERY_COMMIT%"=="%RECOVERY_COMMIT%" goto recovery_failed
if exist "%LOG_DIR%\%RUN_ID%_done.marker" exit /b 7

set ACTUAL_SOURCE_COMMIT=
set /p ACTUAL_SOURCE_COMMIT=<"%RUN_ROOT%\source_expected_commit.txt"
if /I not "%ACTUAL_SOURCE_COMMIT%"=="%EXPECTED_SOURCE_COMMIT%" goto recovery_failed
for %%F in (results.jsonl progress.jsonl dataset_summary.json fixed_baselines.json gate.json validation.json curves.svg history.csv) do if not exist "%RESULTS_DIR%\%%F" goto recovery_failed
if not exist "%ARCHIVE_DIR%\git_revision.txt" goto recovery_failed
echo started>"%LOG_DIR%\%RUN_ID%_recovery_started.marker"

"%PY%" -c "import json,pathlib,sys; sys.path.insert(0,r'%RECOVERY_SOURCE%\src'); from blockcipher_nd.cli.run_innovation2_high_round_integral import validate_artifacts; root=pathlib.Path(r'%RESULTS_DIR%'); report=validate_artifacts(root,expected_rows=4); (root/'validation.recovery.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n',encoding='utf-8'); assert report['status']=='pass'" > "%LOG_DIR%\%RUN_ID%_recovery_validation_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_recovery_validation_stderr.txt" || goto recovery_failed

"%PY%" "%RECOVERY_SOURCE%\scripts\readjudicate-innovation2-high-round-integral" ^
  --artifacts "%ARCHIVE_DIR%" ^
  --remote-config "%REMOTE_CONFIG%" ^
  --invalidate-anchor-layout ^
  --expected-source-commit "%EXPECTED_SOURCE_COMMIT%" ^
  --output "%RESULTS_DIR%\gate.recovery.json" ^
  > "%LOG_DIR%\%RUN_ID%_recovery_gate_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_recovery_gate_stderr.txt" || goto recovery_failed

"%PY%" -c "import json,pathlib; root=pathlib.Path(r'%RESULTS_DIR%'); gate=json.loads((root/'gate.recovery.json').read_text()); validation=json.loads((root/'validation.recovery.json').read_text()); cache=json.loads(pathlib.Path(r'%ARCHIVE_DIR%\cache_metadata.json').read_text()); assert gate['status']=='pass'; assert gate['readjudication']['source_revision_matches_expected']; assert gate['readjudication']['anchor_layout_invalidated']; assert validation['status']=='pass'; assert len(cache)==3 and all(v['status']=='complete' for v in cache.values())" > "%LOG_DIR%\%RUN_ID%_recovery_checks_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_recovery_checks_stderr.txt" || goto recovery_failed

for %%F in (results.jsonl progress.jsonl dataset_summary.json fixed_baselines.json gate.json validation.json curves.svg history.csv) do copy /Y "%RESULTS_DIR%\%%F" "%ARCHIVE_DIR%\%%F" > nul || goto recovery_failed
if exist "%RESULTS_DIR%\plot_deferred.marker" copy /Y "%RESULTS_DIR%\plot_deferred.marker" "%ARCHIVE_DIR%\plot_deferred.marker" > nul || goto recovery_failed
copy /Y "%RESULTS_DIR%\validation.recovery.json" "%ARCHIVE_DIR%\validation.recovery.json" > nul || goto recovery_failed
copy /Y "%RESULTS_DIR%\gate.recovery.json" "%ARCHIVE_DIR%\gate.recovery.json" > nul || goto recovery_failed
copy /Y "%LOG_DIR%\%RUN_ID%_recovery_validation_stderr.txt" "%ARCHIVE_DIR%\recovery_validation_stderr.txt" > nul || goto recovery_failed
copy /Y "%LOG_DIR%\%RUN_ID%_recovery_gate_stderr.txt" "%ARCHIVE_DIR%\recovery_gate_stderr.txt" > nul || goto recovery_failed
copy /Y "%LOG_DIR%\%RUN_ID%_recovery_checks_stderr.txt" "%ARCHIVE_DIR%\recovery_checks_stderr.txt" > nul || goto recovery_failed
echo %RECOVERY_COMMIT%>"%ARCHIVE_DIR%\recovery_commit.txt"
echo * -text>"%ARCHIVE_DIR%\.gitattributes"
"%PY%" -c "import hashlib,pathlib; root=pathlib.Path(r'%ARCHIVE_DIR%'); files=sorted(p for p in root.rglob('*') if p.is_file() and not p.name == 'SHA256SUMS'); (root/'SHA256SUMS').write_text('\n'.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.relative_to(root).as_posix() for p in files)+'\n',encoding='utf-8')" > "%LOG_DIR%\%RUN_ID%_recovery_hash_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_recovery_hash_stderr.txt" || goto recovery_failed

cd /d "%TRAIN_SOURCE%" || goto recovery_failed
git config user.name "remote-experiment"
git config user.email "remote-experiment@local.invalid"
git checkout -B results/%RUN_ID% "%EXPECTED_SOURCE_COMMIT%" > "%LOG_DIR%\%RUN_ID%_recovery_branch_checkout.txt" 2>&1 || goto recovery_failed
git add "results_archive\%RUN_ID%" || goto recovery_failed
git commit -m "results: %RUN_ID% recovered bridge" > "%LOG_DIR%\%RUN_ID%_recovery_branch_commit.txt" 2>&1 || goto recovery_failed
git push origin HEAD:refs/heads/results/%RUN_ID% > "%LOG_DIR%\%RUN_ID%_recovery_branch_push.txt" 2>&1 || goto recovery_failed
git rev-parse HEAD > "%LOG_DIR%\%RUN_ID%_result_branch_revision.txt" 2>&1 || goto recovery_failed
if exist "%LOG_DIR%\%RUN_ID%_failed.marker" del /Q "%LOG_DIR%\%RUN_ID%_failed.marker"
echo pushed>"%LOG_DIR%\%RUN_ID%_result_branch_pushed.marker"
echo done>"%LOG_DIR%\%RUN_ID%_done.marker"
echo recovered>"%LOG_DIR%\%RUN_ID%_recovery_done.marker"
exit /b 0

:recovery_failed
echo failed>"%LOG_DIR%\%RUN_ID%_recovery_failed.marker"
exit /b 1
