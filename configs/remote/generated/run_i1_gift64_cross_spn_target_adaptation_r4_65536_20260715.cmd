@echo off
setlocal EnableExtensions EnableDelayedExpansion

set SEED=%~1
set PHYSICAL_GPU=%~2
if "%SEED%"=="" goto invalid_arguments
if "%PHYSICAL_GPU%"=="" goto invalid_arguments
if not "%SEED%"=="2" if not "%SEED%"=="3" goto invalid_arguments

set RUN_ID=i1_gift64_cross_spn_target_adaptation_r4_65536_seed%SEED%
set RUNS_ROOT=G:\lxy\blockcipher-structure-adaptive-nd-runs
set RUN_ROOT=%RUNS_ROOT%\%RUN_ID%
set SOURCE_ROOT=%RUN_ROOT%\source
set LOG_DIR=%RUN_ROOT%\logs
set RESULTS_DIR=%RUN_ROOT%\results
set CHECKPOINT_DIR=%RUN_ROOT%\checkpoints
set CACHE_ROOT=%RUN_ROOT%\cache
set SCORES_DIR=%RUN_ROOT%\scores
set ARCHIVE_DIR=%SOURCE_ROOT%\results_archive\%RUN_ID%
set PLAN=configs\experiment\innovation1\innovation1_spn_gift64_cross_spn_target_adaptation_65536_seed%SEED%.csv
set REMOTE_CONFIG=configs\remote\innovation1_gift64_cross_spn_target_adaptation_r4_65536_seed%SEED%_gpu%PHYSICAL_GPU%_20260715.json
set SOURCE_MANIFEST=configs\experiment\innovation1\innovation1_spn_cross_spn_typed_transfer_seed0_sources.json
set PY=F:\Anaconda\envs\DWT\torch310\python.exe
set GITHUB_SSH_KEY=C:/Users/1304Lijinlin/.ssh/github_blockcipher_20260612_result_pusher_ed25519
set GIT_SSH_COMMAND=ssh -i %GITHUB_SSH_KEY% -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new
set CUDA_VISIBLE_DEVICES=%PHYSICAL_GPU%
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if not exist "%RUN_ROOT%" mkdir "%RUN_ROOT%"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
if not exist "%RESULTS_DIR%" mkdir "%RESULTS_DIR%"
if not exist "%CHECKPOINT_DIR%" mkdir "%CHECKPOINT_DIR%"
if not exist "%CACHE_ROOT%" mkdir "%CACHE_ROOT%"
if not exist "%SCORES_DIR%" mkdir "%SCORES_DIR%"

cd /d "%SOURCE_ROOT%" || goto failed
for /f "delims=" %%S in ('git status --porcelain') do goto dirty_source
git rev-parse HEAD > "%LOG_DIR%\%RUN_ID%_git_revision.txt" 2>&1 || goto failed
fc /b "%LOG_DIR%\%RUN_ID%_git_revision.txt" "%RUN_ROOT%\source_expected_commit.txt" > nul || goto source_revision_mismatch
git status --short --branch > "%LOG_DIR%\%RUN_ID%_git_status_before_run.txt" 2>&1 || goto failed
nvidia-smi > "%LOG_DIR%\%RUN_ID%_gpu_info.txt" 2>&1 || goto failed
"%PY%" -c "import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 1; print('torch', torch.__version__); print('cuda', torch.version.cuda); print('available', torch.cuda.is_available()); print('visible_count', torch.cuda.device_count()); print('device0', torch.cuda.get_device_name(0))" > "%LOG_DIR%\%RUN_ID%_torch_info.txt" 2> "%LOG_DIR%\%RUN_ID%_torch_info_stderr.txt" || goto failed
"%PY%" scripts\check-remote-readiness --config "%REMOTE_CONFIG%" > "%LOG_DIR%\%RUN_ID%_readiness.txt" 2> "%LOG_DIR%\%RUN_ID%_readiness_stderr.txt" || goto failed
if not exist "outputs\local_smoke\i1_present_cross_spn_typed_cell_r1_seed0\results.jsonl" goto missing_source_assets
if not exist "outputs\local_smoke\i1_present_cross_spn_typed_cell_r1_seed0\checkpoints\row0002_present_cross_spn_typed_cell_true_seed0.pt" goto missing_source_assets
if not exist "outputs\local_smoke\i1_present_cross_spn_typed_cell_r1_seed0\checkpoints\row0003_present_cross_spn_typed_cell_shuffled_seed0.pt" goto missing_source_assets

echo started>"%LOG_DIR%\%RUN_ID%_started.marker"
"%PY%" scripts\train ^
  --plan "%PLAN%" ^
  --epochs 1 ^
  --batch-size 256 ^
  --hidden-bits 32 ^
  --device cuda ^
  --initialization-manifest "%SOURCE_MANIFEST%" ^
  --checkpoint-output-dir "%CHECKPOINT_DIR%" ^
  --dataset-cache-root "%CACHE_ROOT%" ^
  --dataset-cache-chunk-size 512 ^
  --dataset-cache-workers 4 ^
  --progress-output "%LOG_DIR%\progress.jsonl" ^
  --output "%RESULTS_DIR%\results.jsonl" ^
  > "%LOG_DIR%\%RUN_ID%_train_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_train_stderr.txt"
if errorlevel 1 goto failed

"%PY%" scripts\validate-results ^
  --plan "%PLAN%" ^
  --results "%RESULTS_DIR%\results.jsonl" ^
  --expected-rows 4 ^
  --output "%RESULTS_DIR%\validation.json" ^
  > "%LOG_DIR%\%RUN_ID%_validation_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_validation_stderr.txt"
if errorlevel 1 goto failed

call :export_score 0 gift_cross_spn_typed_cell_true row0001_gift_cross_spn_typed_cell_true_seed%SEED%.pt typed_scratch || goto failed
call :export_score 1 gift_cross_spn_typed_cell_true_from_present_true row0002_gift_cross_spn_typed_cell_true_from_present_true_seed%SEED%.pt true_to_true || goto failed
call :export_score 2 gift_cross_spn_typed_cell_true_from_present_shuffled row0003_gift_cross_spn_typed_cell_true_from_present_shuffled_seed%SEED%.pt shuffled_to_true || goto failed
call :export_score 3 gift_cross_spn_typed_cell_shuffled_from_present_true row0004_gift_cross_spn_typed_cell_shuffled_from_present_true_seed%SEED%.pt true_to_shuffled || goto failed

"%PY%" scripts\gate-cross-spn-target-adaptation ^
  --plan "%PLAN%" ^
  --results "%RESULTS_DIR%\results.jsonl" ^
  --progress "%LOG_DIR%\progress.jsonl" ^
  --typed-scratch-scores "%SCORES_DIR%\typed_scratch" ^
  --true-to-true-scores "%SCORES_DIR%\true_to_true" ^
  --shuffled-to-true-scores "%SCORES_DIR%\shuffled_to_true" ^
  --true-to-shuffled-scores "%SCORES_DIR%\true_to_shuffled" ^
  --expected-seed %SEED% ^
  --samples-per-class 65536 ^
  --epochs 1 ^
  --bootstrap-replicates 10000 ^
  --bootstrap-seed 20260715 ^
  --paired-scores-output "%RESULTS_DIR%\paired_scores.csv.gz" ^
  --output "%RESULTS_DIR%\gate.json" ^
  > "%LOG_DIR%\%RUN_ID%_gate_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_gate_stderr.txt"
if errorlevel 1 goto failed

set RESULT_LINES=0
for /f "tokens=3" %%L in ('find /c /v "" "%RESULTS_DIR%\results.jsonl"') do set RESULT_LINES=%%L
echo result_lines=%RESULT_LINES% > "%LOG_DIR%\%RUN_ID%_result_gate.txt"
echo expected_rows=4 >> "%LOG_DIR%\%RUN_ID%_result_gate.txt"
if not "%RESULT_LINES%"=="4" goto incomplete_results

"%PY%" scripts\plot-results ^
  --results "%RESULTS_DIR%\results.jsonl" ^
  --output "%RESULTS_DIR%\curves.svg" ^
  --history-csv "%RESULTS_DIR%\history.csv" ^
  --title "%RUN_ID%" ^
  > "%LOG_DIR%\%RUN_ID%_plot_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_plot_stderr.txt"
if errorlevel 1 goto plot_deferred
"%PY%" -c "import xml.etree.ElementTree as ET; ET.parse(r'%RESULTS_DIR%\curves.svg'); print('svg_parse=pass')" > "%LOG_DIR%\%RUN_ID%_svg_parse.txt" 2> "%LOG_DIR%\%RUN_ID%_svg_parse_stderr.txt" || goto plot_deferred
goto plot_done

:plot_deferred
echo plot_deferred_to_local>"%LOG_DIR%\%RUN_ID%_plot_deferred.marker"
if exist "%RESULTS_DIR%\curves.svg" del /Q "%RESULTS_DIR%\curves.svg"

:plot_done
if not exist "%ARCHIVE_DIR%" mkdir "%ARCHIVE_DIR%"
copy /Y "%RESULTS_DIR%\results.jsonl" "%ARCHIVE_DIR%\results.jsonl" > nul || goto failed
copy /Y "%RESULTS_DIR%\validation.json" "%ARCHIVE_DIR%\validation.json" > nul || goto failed
copy /Y "%RESULTS_DIR%\gate.json" "%ARCHIVE_DIR%\gate.json" > nul || goto failed
copy /Y "%RESULTS_DIR%\paired_scores.csv.gz" "%ARCHIVE_DIR%\paired_scores.csv.gz" > nul || goto failed
copy /Y "%LOG_DIR%\progress.jsonl" "%ARCHIVE_DIR%\progress.jsonl" > nul || goto failed
copy /Y "%LOG_DIR%\score_progress.jsonl" "%ARCHIVE_DIR%\score_progress.jsonl" > nul || goto failed
copy /Y "%LOG_DIR%\%RUN_ID%_git_revision.txt" "%ARCHIVE_DIR%\git_revision.txt" > nul || goto failed
copy /Y "%LOG_DIR%\%RUN_ID%_git_status_before_run.txt" "%ARCHIVE_DIR%\git_status_before_run.txt" > nul || goto failed
copy /Y "%LOG_DIR%\%RUN_ID%_gpu_info.txt" "%ARCHIVE_DIR%\gpu_info.txt" > nul || goto failed
copy /Y "%LOG_DIR%\%RUN_ID%_torch_info.txt" "%ARCHIVE_DIR%\torch_info.txt" > nul || goto failed
copy /Y "%LOG_DIR%\%RUN_ID%_result_gate.txt" "%ARCHIVE_DIR%\result_gate.txt" > nul || goto failed
copy /Y "%REMOTE_CONFIG%" "%ARCHIVE_DIR%\remote_config.json" > nul || goto failed
copy /Y "%PLAN%" "%ARCHIVE_DIR%\plan.csv" > nul || goto failed
copy /Y "%SOURCE_MANIFEST%" "%ARCHIVE_DIR%\source_manifest.json" > nul || goto failed
xcopy /E /I /Y "%SCORES_DIR%" "%ARCHIVE_DIR%\scores" > nul || goto failed
if exist "%RESULTS_DIR%\history.csv" (
  copy /Y "%RESULTS_DIR%\history.csv" "%ARCHIVE_DIR%\history.csv" > nul || goto failed
)
if exist "%RESULTS_DIR%\curves.svg" (
  copy /Y "%RESULTS_DIR%\curves.svg" "%ARCHIVE_DIR%\curves.svg" > nul || goto failed
)
if exist "%LOG_DIR%\%RUN_ID%_plot_deferred.marker" (
  copy /Y "%LOG_DIR%\%RUN_ID%_plot_deferred.marker" "%ARCHIVE_DIR%\plot_deferred.marker" > nul || goto failed
)
echo * -text>"%ARCHIVE_DIR%\.gitattributes"
"%PY%" -c "import hashlib,pathlib; root=pathlib.Path(r'%ARCHIVE_DIR%'); files=sorted(p for p in root.rglob('*') if p.is_file() and not p.name == 'SHA256SUMS'); lines=[]; [(lines.append(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.relative_to(root).as_posix())) for p in files]; (root/'SHA256SUMS').write_text('\n'.join(lines)+'\n', encoding='utf-8')" || goto failed

git config user.name "remote-experiment"
git config user.email "remote-experiment@local.invalid"
git checkout -B results/%RUN_ID% > "%LOG_DIR%\%RUN_ID%_result_branch_checkout.txt" 2>&1 || goto failed
git add "results_archive\%RUN_ID%" || goto failed
git commit -m "results: %RUN_ID% remote target adaptation" > "%LOG_DIR%\%RUN_ID%_result_branch_commit.txt" 2>&1 || goto failed
git push origin HEAD:refs/heads/results/%RUN_ID% > "%LOG_DIR%\%RUN_ID%_result_branch_push.txt" 2>&1 || goto failed
git rev-parse HEAD > "%LOG_DIR%\%RUN_ID%_result_branch_revision.txt" 2>&1 || goto failed
echo pushed>"%LOG_DIR%\%RUN_ID%_result_branch_pushed.marker"

call :maybe_joint
if errorlevel 1 goto failed
echo done>"%LOG_DIR%\%RUN_ID%_done.marker"
exit /b 0

:export_score
set ROW_INDEX=%~1
set MODEL_KEY=%~2
set CHECKPOINT_NAME=%~3
set ARTIFACT_NAME=%~4
"%PY%" scripts\export-checkpoint-scores ^
  --checkpoint "%CHECKPOINT_DIR%\%CHECKPOINT_NAME%" ^
  --eval-plan "%PLAN%" ^
  --eval-row-index %ROW_INDEX% ^
  --split validation ^
  --samples-per-class 32768 ^
  --model-key %MODEL_KEY% ^
  --hidden-bits 32 ^
  --batch-size 256 ^
  --device cuda ^
  --dataset-cache-root "%CACHE_ROOT%" ^
  --dataset-cache-chunk-size 512 ^
  --dataset-cache-workers 4 ^
  --progress-output "%LOG_DIR%\score_progress.jsonl" ^
  --output-dir "%SCORES_DIR%\%ARTIFACT_NAME%" ^
  >> "%LOG_DIR%\%RUN_ID%_score_stdout.txt" 2>> "%LOG_DIR%\%RUN_ID%_score_stderr.txt"
exit /b %ERRORLEVEL%

:maybe_joint
set SEED2_ID=i1_gift64_cross_spn_target_adaptation_r4_65536_seed2
set SEED3_ID=i1_gift64_cross_spn_target_adaptation_r4_65536_seed3
set SEED2_ROOT=%RUNS_ROOT%\%SEED2_ID%
set SEED3_ROOT=%RUNS_ROOT%\%SEED3_ID%
set JOINT_ID=i1_gift64_cross_spn_target_adaptation_r4_65536_joint_seed2_seed3
set JOINT_ROOT=%RUNS_ROOT%\%JOINT_ID%
set JOINT_LOCK=%JOINT_ROOT%\joint.lock
if not exist "%SEED2_ROOT%\results\gate.json" exit /b 0
if not exist "%SEED3_ROOT%\results\gate.json" exit /b 0
if exist "%JOINT_ROOT%\joint_done.marker" exit /b 0
if not exist "%JOINT_ROOT%" mkdir "%JOINT_ROOT%"
mkdir "%JOINT_LOCK%" > nul 2>&1 || exit /b 0
echo started>"%JOINT_ROOT%\joint_started.marker"

"%PY%" scripts\gate-cross-spn-target-adaptation-joint ^
  --seed2-gate "%SEED2_ROOT%\results\gate.json" ^
  --seed3-gate "%SEED3_ROOT%\results\gate.json" ^
  --output "%JOINT_ROOT%\gate.json" ^
  > "%JOINT_ROOT%\joint_gate_stdout.txt" 2> "%JOINT_ROOT%\joint_gate_stderr.txt"
if errorlevel 1 exit /b 1

git checkout -B results/%JOINT_ID% > "%JOINT_ROOT%\joint_branch_checkout.txt" 2>&1 || exit /b 1
set JOINT_ARCHIVE=%SOURCE_ROOT%\results_archive\%JOINT_ID%
if not exist "%JOINT_ARCHIVE%" mkdir "%JOINT_ARCHIVE%"
copy /Y "%JOINT_ROOT%\gate.json" "%JOINT_ARCHIVE%\gate.json" > nul || exit /b 1
copy /Y "%SEED2_ROOT%\results\gate.json" "%JOINT_ARCHIVE%\seed2_gate.json" > nul || exit /b 1
copy /Y "%SEED3_ROOT%\results\gate.json" "%JOINT_ARCHIVE%\seed3_gate.json" > nul || exit /b 1
copy /Y "%SEED2_ROOT%\results\results.jsonl" "%JOINT_ARCHIVE%\seed2_results.jsonl" > nul || exit /b 1
copy /Y "%SEED3_ROOT%\results\results.jsonl" "%JOINT_ARCHIVE%\seed3_results.jsonl" > nul || exit /b 1
echo * -text>"%JOINT_ARCHIVE%\.gitattributes"
"%PY%" -c "import hashlib,pathlib; root=pathlib.Path(r'%JOINT_ARCHIVE%'); files=sorted(p for p in root.rglob('*') if p.is_file() and not p.name == 'SHA256SUMS'); lines=[]; [(lines.append(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.relative_to(root).as_posix())) for p in files]; (root/'SHA256SUMS').write_text('\n'.join(lines)+'\n', encoding='utf-8')" || exit /b 1
git add "results_archive\%JOINT_ID%" || exit /b 1
git commit -m "results: %JOINT_ID% remote joint gate" > "%JOINT_ROOT%\joint_branch_commit.txt" 2>&1 || exit /b 1
git push origin HEAD:refs/heads/results/%JOINT_ID% > "%JOINT_ROOT%\joint_branch_push.txt" 2>&1 || exit /b 1
git rev-parse HEAD > "%JOINT_ROOT%\joint_branch_revision.txt" 2>&1 || exit /b 1
xcopy /E /I /Y "%JOINT_ARCHIVE%" "%JOINT_ROOT%\results_archive\%JOINT_ID%" > nul || exit /b 1
echo pushed>"%JOINT_ROOT%\result_branch_pushed.marker"
echo done>"%JOINT_ROOT%\joint_done.marker"
git checkout results/%RUN_ID% > "%JOINT_ROOT%\joint_source_restore.txt" 2>&1 || exit /b 1
rmdir "%JOINT_LOCK%"
exit /b 0

:dirty_source
echo dirty_source>"%LOG_DIR%\%RUN_ID%_failed.marker"
echo Remote run-owned source clone is dirty.>"%LOG_DIR%\%RUN_ID%_failure_reason.txt"
exit /b 2

:missing_source_assets
echo missing_source_assets>"%LOG_DIR%\%RUN_ID%_failed.marker"
echo Frozen PRESENT source assets are missing from the pushed clone.>"%LOG_DIR%\%RUN_ID%_failure_reason.txt"
exit /b 3

:source_revision_mismatch
echo source_revision_mismatch>"%LOG_DIR%\%RUN_ID%_failed.marker"
echo Remote source HEAD does not match the launch-pinned commit.>"%LOG_DIR%\%RUN_ID%_failure_reason.txt"
exit /b 6

:incomplete_results
echo incomplete_results>"%LOG_DIR%\%RUN_ID%_failed.marker"
exit /b 4

:invalid_arguments
exit /b 5

:failed
echo failed>"%LOG_DIR%\%RUN_ID%_failed.marker"
exit /b 1
