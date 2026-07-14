@echo off
setlocal EnableExtensions DisableDelayedExpansion

set REPAIR_COMMIT=%~1
if "%REPAIR_COMMIT%"=="" exit /b 2

set RUNS_ROOT=G:\lxy\blockcipher-structure-adaptive-nd-runs
set SEED2_ID=i1_gift64_cross_spn_target_adaptation_r4_65536_seed2
set SEED3_ID=i1_gift64_cross_spn_target_adaptation_r4_65536_seed3
set PY=F:\Anaconda\envs\DWT\torch310\python.exe
set GITHUB_SSH_KEY=C:/Users/1304Lijinlin/.ssh/github_blockcipher_20260612_result_pusher_ed25519
set GIT_SSH_COMMAND=ssh -i %GITHUB_SSH_KEY% -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new

call :repair_seed %SEED2_ID% || exit /b 1
call :repair_seed %SEED3_ID% || exit /b 1
exit /b 0

:repair_seed
set RUN_ID=%~1
set RUN_ROOT=%RUNS_ROOT%\%RUN_ID%
set SOURCE_ROOT=%RUN_ROOT%\source
set LOG_DIR=%RUN_ROOT%\logs
set ARCHIVE_DIR=%SOURCE_ROOT%\results_archive\%RUN_ID%
if not exist "%ARCHIVE_DIR%\gate.json" exit /b 1

cd /d "%SOURCE_ROOT%" || exit /b 1
git checkout results/%RUN_ID% > "%LOG_DIR%\%RUN_ID%_hash_repair_checkout.txt" 2>&1 || exit /b 1
echo * -text>"%ARCHIVE_DIR%\.gitattributes"
echo %REPAIR_COMMIT% > "%ARCHIVE_DIR%\hash_repair_commit.txt"
"%PY%" -c "import hashlib,pathlib; root=pathlib.Path(r'%ARCHIVE_DIR%'); files=sorted(p for p in root.rglob('*') if p.is_file() and not p.name == 'SHA256SUMS'); lines=[]; [(lines.append(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.relative_to(root).as_posix())) for p in files]; (root/'SHA256SUMS').write_text('\n'.join(lines)+'\n', encoding='utf-8')" > "%LOG_DIR%\%RUN_ID%_hash_repair_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_hash_repair_stderr.txt" || exit /b 1
git add "results_archive\%RUN_ID%" || exit /b 1
git commit -m "results: repair %RUN_ID% archive hashes" > "%LOG_DIR%\%RUN_ID%_hash_repair_commit.txt" 2>&1 || exit /b 1
git push origin HEAD:refs/heads/results/%RUN_ID% > "%LOG_DIR%\%RUN_ID%_hash_repair_push.txt" 2>&1 || exit /b 1
git rev-parse HEAD > "%LOG_DIR%\%RUN_ID%_result_branch_revision.txt" 2>&1 || exit /b 1
exit /b 0
