@echo off
setlocal EnableExtensions DisableDelayedExpansion

set RECOVERY_COMMIT=%~1
if "%RECOVERY_COMMIT%"=="" exit /b 2

set RUNS_ROOT=G:\lxy\blockcipher-structure-adaptive-nd-runs
set SEED2_ID=i1_gift64_cross_spn_target_adaptation_r4_65536_seed2
set SEED3_ID=i1_gift64_cross_spn_target_adaptation_r4_65536_seed3
set JOINT_ID=i1_gift64_cross_spn_target_adaptation_r4_65536_joint_seed2_seed3
set PY=F:\Anaconda\envs\DWT\torch310\python.exe
set GITHUB_SSH_KEY=C:/Users/1304Lijinlin/.ssh/github_blockcipher_20260612_result_pusher_ed25519
set GIT_SSH_COMMAND=ssh -i %GITHUB_SSH_KEY% -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new

call :publish_seed %SEED2_ID% || exit /b 1
call :publish_seed %SEED3_ID% || exit /b 1
call :publish_joint || exit /b 1
exit /b 0

:publish_seed
set RUN_ID=%~1
set RUN_ROOT=%RUNS_ROOT%\%RUN_ID%
set SOURCE_ROOT=%RUN_ROOT%\source
set LOG_DIR=%RUN_ROOT%\logs
set ARCHIVE_DIR=%SOURCE_ROOT%\results_archive\%RUN_ID%
set SOURCE_COMMIT=
set /p SOURCE_COMMIT=<"%RUN_ROOT%\source_expected_commit.txt"
if "%SOURCE_COMMIT%"=="" exit /b 1
if not exist "%ARCHIVE_DIR%\gate.json" exit /b 1
if not exist "%ARCHIVE_DIR%\results.jsonl" exit /b 1
if not exist "%ARCHIVE_DIR%\validation.json" exit /b 1
if not exist "%ARCHIVE_DIR%\paired_scores.csv.gz" exit /b 1
if not exist "%ARCHIVE_DIR%\scores" exit /b 1

echo %RECOVERY_COMMIT% > "%ARCHIVE_DIR%\recovery_commit.txt"
"%PY%" -c "import hashlib,pathlib; root=pathlib.Path(r'%ARCHIVE_DIR%'); files=sorted(p for p in root.rglob('*') if p.is_file() and not p.name == 'SHA256SUMS'); lines=[]; [(lines.append(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.relative_to(root).as_posix())) for p in files]; (root/'SHA256SUMS').write_text('\n'.join(lines)+'\n', encoding='utf-8')" > "%LOG_DIR%\%RUN_ID%_recovery_hash_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_recovery_hash_stderr.txt" || exit /b 1

cd /d "%SOURCE_ROOT%" || exit /b 1
git config user.name "remote-experiment"
git config user.email "remote-experiment@local.invalid"
git checkout -B results/%RUN_ID% "%SOURCE_COMMIT%" > "%LOG_DIR%\%RUN_ID%_recovery_branch_checkout.txt" 2>&1 || exit /b 1
git add "results_archive\%RUN_ID%" || exit /b 1
git commit -m "results: %RUN_ID% recovered target adaptation" > "%LOG_DIR%\%RUN_ID%_recovery_branch_commit.txt" 2>&1 || exit /b 1
git push origin HEAD:refs/heads/results/%RUN_ID% > "%LOG_DIR%\%RUN_ID%_recovery_branch_push.txt" 2>&1 || exit /b 1
git rev-parse HEAD > "%LOG_DIR%\%RUN_ID%_result_branch_revision.txt" 2>&1 || exit /b 1
if exist "%LOG_DIR%\%RUN_ID%_failed.marker" move /Y "%LOG_DIR%\%RUN_ID%_failed.marker" "%LOG_DIR%\%RUN_ID%_postprocess_failed_recovered.marker" > nul
echo pushed > "%LOG_DIR%\%RUN_ID%_result_branch_pushed.marker"
echo done > "%LOG_DIR%\%RUN_ID%_done.marker"
exit /b 0

:publish_joint
set SEED2_ROOT=%RUNS_ROOT%\%SEED2_ID%
set SEED3_ROOT=%RUNS_ROOT%\%SEED3_ID%
set JOINT_ROOT=%RUNS_ROOT%\%JOINT_ID%
set SOURCE_ROOT=%SEED2_ROOT%\source
set SOURCE_COMMIT=
set /p SOURCE_COMMIT=<"%SEED2_ROOT%\source_expected_commit.txt"
if "%SOURCE_COMMIT%"=="" exit /b 1
if not exist "%JOINT_ROOT%" mkdir "%JOINT_ROOT%"

cd /d "%SOURCE_ROOT%" || exit /b 1
"%PY%" scripts\gate-cross-spn-target-adaptation-joint ^
  --seed2-gate "%SEED2_ROOT%\results\gate.json" ^
  --seed3-gate "%SEED3_ROOT%\results\gate.json" ^
  --output "%JOINT_ROOT%\gate.json" ^
  > "%JOINT_ROOT%\joint_gate_stdout.txt" 2> "%JOINT_ROOT%\joint_gate_stderr.txt" || exit /b 1

set JOINT_ARCHIVE=%SOURCE_ROOT%\results_archive\%JOINT_ID%
if not exist "%JOINT_ARCHIVE%" mkdir "%JOINT_ARCHIVE%"
copy /Y "%JOINT_ROOT%\gate.json" "%JOINT_ARCHIVE%\gate.json" > nul || exit /b 1
copy /Y "%SEED2_ROOT%\results\gate.json" "%JOINT_ARCHIVE%\seed2_gate.json" > nul || exit /b 1
copy /Y "%SEED3_ROOT%\results\gate.json" "%JOINT_ARCHIVE%\seed3_gate.json" > nul || exit /b 1
copy /Y "%SEED2_ROOT%\results\results.jsonl" "%JOINT_ARCHIVE%\seed2_results.jsonl" > nul || exit /b 1
copy /Y "%SEED3_ROOT%\results\results.jsonl" "%JOINT_ARCHIVE%\seed3_results.jsonl" > nul || exit /b 1
echo %RECOVERY_COMMIT% > "%JOINT_ARCHIVE%\recovery_commit.txt"
"%PY%" -c "import hashlib,pathlib; root=pathlib.Path(r'%JOINT_ARCHIVE%'); files=sorted(p for p in root.rglob('*') if p.is_file() and not p.name == 'SHA256SUMS'); lines=[]; [(lines.append(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.relative_to(root).as_posix())) for p in files]; (root/'SHA256SUMS').write_text('\n'.join(lines)+'\n', encoding='utf-8')" > "%JOINT_ROOT%\joint_recovery_hash_stdout.txt" 2> "%JOINT_ROOT%\joint_recovery_hash_stderr.txt" || exit /b 1

git checkout -B results/%JOINT_ID% "%SOURCE_COMMIT%" > "%JOINT_ROOT%\joint_branch_checkout.txt" 2>&1 || exit /b 1
git add "results_archive\%JOINT_ID%" || exit /b 1
git commit -m "results: %JOINT_ID% recovered joint gate" > "%JOINT_ROOT%\joint_branch_commit.txt" 2>&1 || exit /b 1
git push origin HEAD:refs/heads/results/%JOINT_ID% > "%JOINT_ROOT%\joint_branch_push.txt" 2>&1 || exit /b 1
git rev-parse HEAD > "%JOINT_ROOT%\joint_branch_revision.txt" 2>&1 || exit /b 1
if exist "%JOINT_ROOT%\results_archive" rmdir /S /Q "%JOINT_ROOT%\results_archive"
xcopy /E /I /Y "%JOINT_ARCHIVE%" "%JOINT_ROOT%\results_archive\%JOINT_ID%" > nul || exit /b 1
echo pushed > "%JOINT_ROOT%\result_branch_pushed.marker"
echo done > "%JOINT_ROOT%\joint_done.marker"
exit /b 0
