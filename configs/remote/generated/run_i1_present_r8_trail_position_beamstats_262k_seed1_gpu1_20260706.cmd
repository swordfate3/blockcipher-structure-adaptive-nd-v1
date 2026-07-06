@echo off
setlocal EnableExtensions EnableDelayedExpansion

set RUN_ID=i1_present_r8_trail_position_beamstats_262k_seed1_gpu1_20260706
set REPO_URL=git@github.com:swordfate3/blockcipher-structure-adaptive-nd-v1.git
set BRANCH=main
set PROJECT_ROOT=G:\lxy\blockcipher-structure-adaptive-nd-runs\%RUN_ID%\source
set RUN_ROOT=G:\lxy\blockcipher-structure-adaptive-nd-runs\%RUN_ID%
set LOG_DIR=%RUN_ROOT%\logs
set RESULTS_DIR=%RUN_ROOT%\results
set CHECKPOINT_DIR=%RUN_ROOT%\checkpoints
set SCORE_ARTIFACT_DIR=%RUN_ROOT%\score_artifacts
set DATASET_CACHE_ROOT=G:\lxy\blockcipher-structure-adaptive-nd-runs\trail_position_beamstats_262k_cache
set PYTHON_EXE=F:\Anaconda\envs\DWT\torch310\python.exe
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set GITHUB_SSH_KEY=C:/Users/1304Lijinlin/.ssh/github_blockcipher_20260612_result_pusher_ed25519
set GIT_SSH_COMMAND=ssh -i %GITHUB_SSH_KEY% -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new

if not exist "%RUN_ROOT%" mkdir "%RUN_ROOT%"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
if not exist "%RESULTS_DIR%" mkdir "%RESULTS_DIR%"
if not exist "%CHECKPOINT_DIR%" mkdir "%CHECKPOINT_DIR%"
if not exist "%SCORE_ARTIFACT_DIR%" mkdir "%SCORE_ARTIFACT_DIR%"
if not exist "%DATASET_CACHE_ROOT%" mkdir "%DATASET_CACHE_ROOT%"

echo TRAIL_POSITION_BEAMSTATS_MEDIUM_PREPARED_ONLY>"%LOG_DIR%\%RUN_ID%_prepared_only.marker"
echo run_id=%RUN_ID%>"%LOG_DIR%\%RUN_ID%_launch_env.txt"
echo project_root=%PROJECT_ROOT%>>"%LOG_DIR%\%RUN_ID%_launch_env.txt"
echo run_root=%RUN_ROOT%>>"%LOG_DIR%\%RUN_ID%_launch_env.txt"
echo checkpoint_dir=%CHECKPOINT_DIR%>>"%LOG_DIR%\%RUN_ID%_launch_env.txt"
echo score_artifact_dir=%SCORE_ARTIFACT_DIR%>>"%LOG_DIR%\%RUN_ID%_launch_env.txt"
echo dataset_cache_root=%DATASET_CACHE_ROOT%>>"%LOG_DIR%\%RUN_ID%_launch_env.txt"
echo python_exe=%PYTHON_EXE%>>"%LOG_DIR%\%RUN_ID%_launch_env.txt"
echo branch=%BRANCH%>>"%LOG_DIR%\%RUN_ID%_launch_env.txt"
echo pytorch_cuda_alloc_conf=%PYTORCH_CUDA_ALLOC_CONF%>>"%LOG_DIR%\%RUN_ID%_launch_env.txt"

nvidia-smi > "%LOG_DIR%\%RUN_ID%_gpu_info.txt" 2>&1
"%PYTHON_EXE%" -c "import torch; print('torch', torch.__version__); print('cuda', torch.version.cuda); print('available', torch.cuda.is_available()); print('count', torch.cuda.device_count())" > "%LOG_DIR%\%RUN_ID%_torch_info.txt" 2> "%LOG_DIR%\%RUN_ID%_torch_info_stderr.txt"

if exist "%PROJECT_ROOT%\.git" (
  cd /d "%PROJECT_ROOT%" || goto failed
  git status --short --branch > "%LOG_DIR%\%RUN_ID%_git_status_before_run.txt" 2>&1
  git fetch origin %BRANCH% > "%LOG_DIR%\%RUN_ID%_git_fetch_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_git_fetch_stderr.txt" || goto failed
  git checkout %BRANCH% > "%LOG_DIR%\%RUN_ID%_git_checkout_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_git_checkout_stderr.txt" || goto failed
  git pull --ff-only origin %BRANCH% >> "%LOG_DIR%\%RUN_ID%_git_fetch_stdout.txt" 2>> "%LOG_DIR%\%RUN_ID%_git_fetch_stderr.txt" || goto failed
) else (
  if exist "%PROJECT_ROOT%" rmdir /s /q "%PROJECT_ROOT%"
  git clone --branch %BRANCH% "%REPO_URL%" "%PROJECT_ROOT%" > "%LOG_DIR%\%RUN_ID%_git_clone_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_git_clone_stderr.txt" || goto failed
  cd /d "%PROJECT_ROOT%" || goto failed
  git status --short --branch > "%LOG_DIR%\%RUN_ID%_git_status_before_run.txt" 2>&1
)

git rev-parse HEAD > "%LOG_DIR%\%RUN_ID%_git_revision.txt" 2>&1
"%PYTHON_EXE%" scripts\check-remote-readiness --config configs\remote\innovation1_spn_present_r8_trail_position_beamstats_262k_seed1_gpu1_20260706.json > "%LOG_DIR%\%RUN_ID%_readiness.txt" 2> "%LOG_DIR%\%RUN_ID%_readiness_stderr.txt" || goto failed
echo started>"%LOG_DIR%\%RUN_ID%_started.marker"

"%PYTHON_EXE%" scripts\train ^
  --plan configs\experiment\innovation1\innovation1_spn_present_r8_trail_position_beamstats_262k_seed1.csv ^
  --epochs 20 ^
  --batch-size 512 ^
  --hidden-bits 16 ^
  --device cuda:1 ^
  --learning-rate 0.0001 ^
  --optimizer adam ^
  --weight-decay 0.00001 ^
  --loss mse ^
  --lr-scheduler none ^
  --max-learning-rate 0 ^
  --checkpoint-metric val_auc ^
  --restore-best-checkpoint ^
  --checkpoint-output-dir "%CHECKPOINT_DIR%" ^
  --early-stopping-patience 0 ^
  --early-stopping-min-delta 0.0 ^
  --train-eval-interval 0 ^
  --sample-structure plaintext_integral_nibble_difference_matched_negative ^
  --negative-mode encrypted_random_plaintexts ^
  --key-rotation-interval 0 ^
  --dataset-cache-root "%DATASET_CACHE_ROOT%" ^
  --dataset-cache-chunk-size 8192 ^
  --dataset-cache-workers 4 ^
  --output "%RESULTS_DIR%\train_matrix.jsonl" ^
  --progress-output "%LOG_DIR%\trail_position_beamstats_progress.jsonl" ^
  > "%LOG_DIR%\%RUN_ID%_train_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_train_stderr.txt"

if errorlevel 1 goto failed
if not exist "%CHECKPOINT_DIR%\row0001_present_pairset_global_stats_seed1.pt" goto failed
if not exist "%CHECKPOINT_DIR%\row0002_present_trail_position_stats_pairset_seed1.pt" goto failed
echo train_done>"%LOG_DIR%\%RUN_ID%_train_done.marker"

"%PYTHON_EXE%" scripts\export-checkpoint-scores ^
  --checkpoint "%CHECKPOINT_DIR%\row0001_present_pairset_global_stats_seed1.pt" ^
  --eval-plan configs\experiment\innovation1\innovation1_spn_present_r8_trail_position_beamstats_262k_seed1.csv ^
  --eval-row-index 0 ^
  --model-key present_pairset_global_stats ^
  --hidden-bits 16 ^
  --batch-size 512 ^
  --device cuda:1 ^
  --dataset-cache-root "%DATASET_CACHE_ROOT%" ^
  --dataset-cache-chunk-size 8192 ^
  --dataset-cache-workers 4 ^
  --progress-output "%LOG_DIR%\trail_position_beamstats_score_export_progress.jsonl" ^
  --expert-family trail_position_global_control ^
  --candidate-status near_neighbor_control ^
  --output-dir "%SCORE_ARTIFACT_DIR%\global_stats_control" ^
  > "%LOG_DIR%\%RUN_ID%_export_global_stats_control_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_export_global_stats_control_stderr.txt"
if errorlevel 1 goto failed

"%PYTHON_EXE%" scripts\export-checkpoint-scores ^
  --checkpoint "%CHECKPOINT_DIR%\row0002_present_trail_position_stats_pairset_seed1.pt" ^
  --eval-plan configs\experiment\innovation1\innovation1_spn_present_r8_trail_position_beamstats_262k_seed1.csv ^
  --eval-row-index 1 ^
  --model-key present_trail_position_stats_pairset ^
  --hidden-bits 16 ^
  --batch-size 512 ^
  --device cuda:1 ^
  --dataset-cache-root "%DATASET_CACHE_ROOT%" ^
  --dataset-cache-chunk-size 8192 ^
  --dataset-cache-workers 4 ^
  --progress-output "%LOG_DIR%\trail_position_beamstats_score_export_progress.jsonl" ^
  --expert-family trail_position ^
  --candidate-status weak_positive ^
  --output-dir "%SCORE_ARTIFACT_DIR%\trail_position" ^
  > "%LOG_DIR%\%RUN_ID%_export_trail_position_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_export_trail_position_stderr.txt"
if errorlevel 1 goto failed

"%PYTHON_EXE%" scripts\verify-score-artifacts ^
  --artifacts "%SCORE_ARTIFACT_DIR%\global_stats_control" "%SCORE_ARTIFACT_DIR%\trail_position" ^
  --expected-rows 262144 ^
  --require-model present_pairset_global_stats:trail_position_global_control:near_neighbor_control ^
  --require-model present_trail_position_stats_pairset:trail_position:weak_positive ^
  --output "%SCORE_ARTIFACT_DIR%\verification_summary.json" ^
  > "%LOG_DIR%\%RUN_ID%_verify_score_artifacts_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_verify_score_artifacts_stderr.txt"
if errorlevel 1 goto failed

echo score_export_done>"%LOG_DIR%\%RUN_ID%_score_export_done.marker"
echo done>"%LOG_DIR%\%RUN_ID%_done.marker"
exit /b 0

:failed
echo failed>"%LOG_DIR%\%RUN_ID%_failed.marker"
exit /b 1
