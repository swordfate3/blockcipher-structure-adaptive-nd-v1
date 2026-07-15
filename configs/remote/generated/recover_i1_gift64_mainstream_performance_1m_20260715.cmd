@echo off
setlocal EnableExtensions DisableDelayedExpansion

set RECOVERY_COMMIT=%~1
if "%RECOVERY_COMMIT%"=="" exit /b 2

for %%I in ("%~dp0..\..\..") do set RECOVERY_SOURCE=%%~fI
for %%I in ("%RECOVERY_SOURCE%\..") do set RECOVERY_ROOT=%%~fI
set RUNS_ROOT=G:\lxy\blockcipher-structure-adaptive-nd-runs
set SEED6_ID=i1_gift64_mainstream_performance_1m_seed6
set SEED7_ID=i1_gift64_mainstream_performance_1m_seed7
set JOINT_ID=i1_gift64_mainstream_performance_1m_joint_seed6_seed7
set PY=F:\Anaconda\envs\DWT\torch310\python.exe
set GITHUB_SSH_KEY=C:/Users/1304Lijinlin/.ssh/github_blockcipher_20260612_result_pusher_ed25519
set GIT_SSH_COMMAND=ssh -i %GITHUB_SSH_KEY% -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new
set CURRENT_RUN_ID=
set CURRENT_LOG_DIR=

cd /d "%RECOVERY_SOURCE%" || exit /b 1
for /f "delims=" %%S in ('git status --porcelain') do exit /b 1
for /f "delims=" %%H in ('git rev-parse HEAD') do set ACTUAL_RECOVERY_COMMIT=%%H
if /I not "%ACTUAL_RECOVERY_COMMIT%"=="%RECOVERY_COMMIT%" exit /b 1

echo started>"%RECOVERY_ROOT%\recovery_started.marker"
call :publish_seed 6 %SEED6_ID% || goto recovery_failed
call :publish_seed 7 %SEED7_ID% || goto recovery_failed
call :publish_joint || goto recovery_failed
echo done>"%RECOVERY_ROOT%\recovery_done.marker"
exit /b 0

:publish_seed
set SEED=%~1
set RUN_ID=%~2
set RUN_ROOT=%RUNS_ROOT%\%RUN_ID%
set TRAIN_SOURCE=%RUN_ROOT%\source
set LOG_DIR=%RUN_ROOT%\logs
set RESULTS_DIR=%RUN_ROOT%\results
set SCORES_DIR=%RUN_ROOT%\scores
set ARCHIVE_DIR=%TRAIN_SOURCE%\results_archive\%RUN_ID%
set PLAN=%TRAIN_SOURCE%\configs\experiment\innovation1\innovation1_spn_gift64_mainstream_performance_1m_seed%SEED%.csv
if "%SEED%"=="6" set REMOTE_CONFIG=%TRAIN_SOURCE%\configs\remote\innovation1_gift64_mainstream_performance_1m_seed6_gpu0_20260715.json
if "%SEED%"=="7" set REMOTE_CONFIG=%TRAIN_SOURCE%\configs\remote\innovation1_gift64_mainstream_performance_1m_seed7_gpu1_20260715.json
set SOURCE_MANIFEST=%TRAIN_SOURCE%\configs\experiment\innovation1\innovation1_spn_gift64_mainstream_performance_sources.json
set CURRENT_RUN_ID=%RUN_ID%
set CURRENT_LOG_DIR=%LOG_DIR%

echo started>"%LOG_DIR%\%RUN_ID%_recovery_started.marker"
if not exist "%RESULTS_DIR%\results.jsonl" exit /b 1
if not exist "%RESULTS_DIR%\validation.json" exit /b 1
for %%R in (typed_scratch typed_source0 typed_source1 lstm resnet) do if not exist "%SCORES_DIR%\%%R\models.json" exit /b 1

"%PY%" "%RECOVERY_SOURCE%\scripts\validate-results" ^
  --plan "%PLAN%" ^
  --results "%RESULTS_DIR%\results.jsonl" ^
  --expected-rows 5 ^
  --output "%RESULTS_DIR%\validation.recovery.json" ^
  > "%LOG_DIR%\%RUN_ID%_recovery_validation_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_recovery_validation_stderr.txt" || exit /b 1

"%PY%" "%RECOVERY_SOURCE%\scripts\gate-cross-spn-mainstream-performance" ^
  --plan "%PLAN%" ^
  --results "%RESULTS_DIR%\results.jsonl" ^
  --expected-seed %SEED% ^
  --typed-scratch-scores "%SCORES_DIR%\typed_scratch" ^
  --typed-source0-scores "%SCORES_DIR%\typed_source0" ^
  --typed-source1-scores "%SCORES_DIR%\typed_source1" ^
  --lstm-scores "%SCORES_DIR%\lstm" ^
  --resnet-scores "%SCORES_DIR%\resnet" ^
  --output "%RESULTS_DIR%\gate.json" ^
  > "%LOG_DIR%\%RUN_ID%_recovery_gate_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_recovery_gate_stderr.txt" || exit /b 1

"%PY%" -c "import numpy as np,pathlib; root=pathlib.Path(r'%SCORES_DIR%'); roles=['typed_scratch','typed_source0','typed_source1','lstm','resnet']; first=root/roles[0]; payload={'labels':np.load(first/'labels.npy'),'sample_ids':np.load(first/'sample_ids.npy')}; payload.update({r+'_probabilities':np.load(root/r/'probabilities.npy') for r in roles}); payload.update({r+'_logits':np.load(root/r/'logits.npy') for r in roles}); np.savez_compressed(r'%RESULTS_DIR%\primary_scores.npz',**payload)" > "%LOG_DIR%\%RUN_ID%_recovery_score_pack_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_recovery_score_pack_stderr.txt" || exit /b 1

set RESULT_LINES=0
for /f "tokens=3" %%L in ('find /c /v "" "%RESULTS_DIR%\results.jsonl"') do set RESULT_LINES=%%L
echo result_lines=%RESULT_LINES%>"%LOG_DIR%\%RUN_ID%_result_gate.txt"
echo expected_rows=5>>"%LOG_DIR%\%RUN_ID%_result_gate.txt"
if not "%RESULT_LINES%"=="5" exit /b 1

"%PY%" "%RECOVERY_SOURCE%\scripts\plot-results" ^
  --results "%RESULTS_DIR%\results.jsonl" ^
  --output "%RESULTS_DIR%\curves.svg" ^
  --history-csv "%RESULTS_DIR%\history.csv" ^
  --title "%RUN_ID%" ^
  > "%LOG_DIR%\%RUN_ID%_recovery_plot_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_recovery_plot_stderr.txt"
if errorlevel 1 echo plot_deferred_to_local>"%LOG_DIR%\%RUN_ID%_plot_deferred.marker"

if not exist "%ARCHIVE_DIR%" mkdir "%ARCHIVE_DIR%"
if not exist "%ARCHIVE_DIR%\score_metadata" mkdir "%ARCHIVE_DIR%\score_metadata"
copy /Y "%RESULTS_DIR%\results.jsonl" "%ARCHIVE_DIR%\results.jsonl" > nul || exit /b 1
copy /Y "%RESULTS_DIR%\validation.json" "%ARCHIVE_DIR%\validation.json" > nul || exit /b 1
copy /Y "%RESULTS_DIR%\validation.recovery.json" "%ARCHIVE_DIR%\validation.recovery.json" > nul || exit /b 1
copy /Y "%RESULTS_DIR%\gate.json" "%ARCHIVE_DIR%\gate.json" > nul || exit /b 1
copy /Y "%LOG_DIR%\progress.jsonl" "%ARCHIVE_DIR%\progress.jsonl" > nul || exit /b 1
copy /Y "%LOG_DIR%\score_progress.jsonl" "%ARCHIVE_DIR%\score_progress.jsonl" > nul || exit /b 1
copy /Y "%LOG_DIR%\%RUN_ID%_git_revision.txt" "%ARCHIVE_DIR%\git_revision.txt" > nul || exit /b 1
copy /Y "%LOG_DIR%\%RUN_ID%_git_status_before_run.txt" "%ARCHIVE_DIR%\git_status_before_run.txt" > nul || exit /b 1
copy /Y "%LOG_DIR%\%RUN_ID%_gpu_info.txt" "%ARCHIVE_DIR%\gpu_info.txt" > nul || exit /b 1
copy /Y "%LOG_DIR%\%RUN_ID%_torch_info.txt" "%ARCHIVE_DIR%\torch_info.txt" > nul || exit /b 1
copy /Y "%LOG_DIR%\%RUN_ID%_result_gate.txt" "%ARCHIVE_DIR%\result_gate.txt" > nul || exit /b 1
copy /Y "%LOG_DIR%\%RUN_ID%_gate_stderr.txt" "%ARCHIVE_DIR%\original_gate_stderr.txt" > nul || exit /b 1
copy /Y "%LOG_DIR%\%RUN_ID%_recovery_gate_stderr.txt" "%ARCHIVE_DIR%\recovery_gate_stderr.txt" > nul || exit /b 1
copy /Y "%REMOTE_CONFIG%" "%ARCHIVE_DIR%\remote_config.json" > nul || exit /b 1
copy /Y "%PLAN%" "%ARCHIVE_DIR%\plan.csv" > nul || exit /b 1
copy /Y "%SOURCE_MANIFEST%" "%ARCHIVE_DIR%\source_manifest.json" > nul || exit /b 1
for %%R in (typed_scratch typed_source0 typed_source1 lstm resnet) do copy /Y "%SCORES_DIR%\%%R\models.json" "%ARCHIVE_DIR%\score_metadata\%%R.json" > nul || exit /b 1
echo %RECOVERY_COMMIT%>"%ARCHIVE_DIR%\recovery_commit.txt"
"%PY%" -c "import hashlib,pathlib; p=pathlib.Path(r'%RESULTS_DIR%\primary_scores.npz'); pathlib.Path(r'%ARCHIVE_DIR%\primary_scores.sha256').write_text(hashlib.sha256(p.read_bytes()).hexdigest()+'  primary_scores.npz\n',encoding='utf-8')" || exit /b 1
if exist "%RESULTS_DIR%\history.csv" copy /Y "%RESULTS_DIR%\history.csv" "%ARCHIVE_DIR%\history.csv" > nul
if exist "%RESULTS_DIR%\curves.svg" copy /Y "%RESULTS_DIR%\curves.svg" "%ARCHIVE_DIR%\curves.svg" > nul
if exist "%LOG_DIR%\%RUN_ID%_plot_deferred.marker" copy /Y "%LOG_DIR%\%RUN_ID%_plot_deferred.marker" "%ARCHIVE_DIR%\plot_deferred.marker" > nul
echo * -text>"%ARCHIVE_DIR%\.gitattributes"
"%PY%" -c "import hashlib,pathlib; root=pathlib.Path(r'%ARCHIVE_DIR%'); files=sorted(p for p in root.rglob('*') if p.is_file() and p.name!='SHA256SUMS'); (root/'SHA256SUMS').write_text('\n'.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.relative_to(root).as_posix() for p in files)+'\n',encoding='utf-8')" || exit /b 1

set SOURCE_COMMIT=
set /p SOURCE_COMMIT=<"%RUN_ROOT%\source_expected_commit.txt"
if "%SOURCE_COMMIT%"=="" exit /b 1
cd /d "%TRAIN_SOURCE%" || exit /b 1
git config user.name "remote-experiment"
git config user.email "remote-experiment@local.invalid"
git checkout -B results/%RUN_ID% "%SOURCE_COMMIT%" > "%LOG_DIR%\%RUN_ID%_recovery_branch_checkout.txt" 2>&1 || exit /b 1
git add "results_archive\%RUN_ID%" || exit /b 1
git commit -m "results: %RUN_ID% recovered performance gate" > "%LOG_DIR%\%RUN_ID%_recovery_branch_commit.txt" 2>&1 || exit /b 1
git push origin HEAD:refs/heads/results/%RUN_ID% > "%LOG_DIR%\%RUN_ID%_recovery_branch_push.txt" 2>&1 || exit /b 1
git rev-parse HEAD > "%LOG_DIR%\%RUN_ID%_result_branch_revision.txt" 2>&1 || exit /b 1
echo pushed>"%LOG_DIR%\%RUN_ID%_result_branch_pushed.marker"
echo done>"%LOG_DIR%\%RUN_ID%_done.marker"
exit /b 0

:publish_joint
set SEED6_ROOT=%RUNS_ROOT%\%SEED6_ID%
set SEED7_ROOT=%RUNS_ROOT%\%SEED7_ID%
set JOINT_ROOT=%RUNS_ROOT%\%JOINT_ID%
set TRAIN_SOURCE=%SEED6_ROOT%\source
set CURRENT_RUN_ID=%JOINT_ID%
set CURRENT_LOG_DIR=%JOINT_ROOT%
if not exist "%JOINT_ROOT%" mkdir "%JOINT_ROOT%"
echo started>"%JOINT_ROOT%\joint_recovery_started.marker"

"%PY%" "%RECOVERY_SOURCE%\scripts\gate-cross-spn-mainstream-performance-joint" ^
  --seed6-gate "%SEED6_ROOT%\results\gate.json" ^
  --seed7-gate "%SEED7_ROOT%\results\gate.json" ^
  --output "%JOINT_ROOT%\gate.json" ^
  > "%JOINT_ROOT%\joint_recovery_gate_stdout.txt" 2> "%JOINT_ROOT%\joint_recovery_gate_stderr.txt" || exit /b 1

set JOINT_ARCHIVE=%TRAIN_SOURCE%\results_archive\%JOINT_ID%
if not exist "%JOINT_ARCHIVE%" mkdir "%JOINT_ARCHIVE%"
copy /Y "%JOINT_ROOT%\gate.json" "%JOINT_ARCHIVE%\gate.json" > nul || exit /b 1
copy /Y "%SEED6_ROOT%\results\gate.json" "%JOINT_ARCHIVE%\seed6_gate.json" > nul || exit /b 1
copy /Y "%SEED7_ROOT%\results\gate.json" "%JOINT_ARCHIVE%\seed7_gate.json" > nul || exit /b 1
copy /Y "%SEED6_ROOT%\results\results.jsonl" "%JOINT_ARCHIVE%\seed6_results.jsonl" > nul || exit /b 1
copy /Y "%SEED7_ROOT%\results\results.jsonl" "%JOINT_ARCHIVE%\seed7_results.jsonl" > nul || exit /b 1
echo %RECOVERY_COMMIT%>"%JOINT_ARCHIVE%\recovery_commit.txt"
echo * -text>"%JOINT_ARCHIVE%\.gitattributes"
"%PY%" -c "import hashlib,pathlib; root=pathlib.Path(r'%JOINT_ARCHIVE%'); files=sorted(p for p in root.rglob('*') if p.is_file() and p.name!='SHA256SUMS'); (root/'SHA256SUMS').write_text('\n'.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.relative_to(root).as_posix() for p in files)+'\n',encoding='utf-8')" || exit /b 1

set SOURCE_COMMIT=
set /p SOURCE_COMMIT=<"%SEED6_ROOT%\source_expected_commit.txt"
if "%SOURCE_COMMIT%"=="" exit /b 1
cd /d "%TRAIN_SOURCE%" || exit /b 1
git checkout -B results/%JOINT_ID% "%SOURCE_COMMIT%" > "%JOINT_ROOT%\joint_recovery_branch_checkout.txt" 2>&1 || exit /b 1
git add "results_archive\%JOINT_ID%" || exit /b 1
git commit -m "results: %JOINT_ID% recovered joint gate" > "%JOINT_ROOT%\joint_recovery_branch_commit.txt" 2>&1 || exit /b 1
git push origin HEAD:refs/heads/results/%JOINT_ID% > "%JOINT_ROOT%\joint_recovery_branch_push.txt" 2>&1 || exit /b 1
git rev-parse HEAD > "%JOINT_ROOT%\joint_result_branch_revision.txt" 2>&1 || exit /b 1
if exist "%JOINT_ROOT%\results_archive" rmdir /S /Q "%JOINT_ROOT%\results_archive"
xcopy /E /I /Y "%JOINT_ARCHIVE%" "%JOINT_ROOT%\results_archive\%JOINT_ID%" > nul || exit /b 1
git checkout results/%SEED6_ID% > "%JOINT_ROOT%\joint_source_restore.txt" 2>&1 || exit /b 1
echo pushed>"%JOINT_ROOT%\result_branch_pushed.marker"
echo done>"%JOINT_ROOT%\joint_done.marker"
exit /b 0

:recovery_failed
if defined CURRENT_LOG_DIR echo failed>"%CURRENT_LOG_DIR%\%CURRENT_RUN_ID%_recovery_failed.marker"
echo failed>"%RECOVERY_ROOT%\recovery_failed.marker"
exit /b 1
