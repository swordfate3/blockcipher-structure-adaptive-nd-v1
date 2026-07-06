@echo off
setlocal EnableExtensions EnableDelayedExpansion

set RUN_ID=i1_present_r8_trail_position_beamstats_65k_seed0_gpu0_20260706
set REPO_URL=git@github.com:swordfate3/blockcipher-structure-adaptive-nd-v1.git
set BRANCH=main
set PROJECT_ROOT=G:\lxy\blockcipher-structure-adaptive-nd-runs\%RUN_ID%\source
set RUN_ROOT=G:\lxy\blockcipher-structure-adaptive-nd-runs\%RUN_ID%
set LOG_DIR=%RUN_ROOT%\logs
set RESULTS_DIR=%RUN_ROOT%\results
set CHECKPOINT_DIR=%RUN_ROOT%\checkpoints
set SCORE_ARTIFACT_DIR=%RUN_ROOT%\score_artifacts
set DATASET_CACHE_ROOT=G:\lxy\blockcipher-structure-adaptive-nd-runs\trail_position_beamstats_cache
set PYTHON_EXE=F:\Anaconda\envs\DWT\torch310\python.exe
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set GITHUB_SSH_KEY=C:/Users/1304Lijinlin/.ssh/github_blockcipher_20260612_result_pusher_ed25519
set GIT_SSH_COMMAND=ssh -i %GITHUB_SSH_KEY% -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new

if not exist "%RUN_ROOT%" goto failed
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
if not exist "%RESULTS_DIR%\train_matrix.jsonl" goto failed
if not exist "%CHECKPOINT_DIR%\row0001_present_pairset_global_stats_seed0.pt" goto failed
if not exist "%CHECKPOINT_DIR%\row0002_present_trail_position_stats_pairset_seed0.pt" goto failed
if not exist "%SCORE_ARTIFACT_DIR%" mkdir "%SCORE_ARTIFACT_DIR%"
if not exist "%DATASET_CACHE_ROOT%" mkdir "%DATASET_CACHE_ROOT%"

echo repair_started>"%LOG_DIR%\%RUN_ID%_score_export_repair_started.marker"
echo run_id=%RUN_ID%>"%LOG_DIR%\%RUN_ID%_score_export_repair_env.txt"
echo project_root=%PROJECT_ROOT%>>"%LOG_DIR%\%RUN_ID%_score_export_repair_env.txt"
echo run_root=%RUN_ROOT%>>"%LOG_DIR%\%RUN_ID%_score_export_repair_env.txt"
echo checkpoint_dir=%CHECKPOINT_DIR%>>"%LOG_DIR%\%RUN_ID%_score_export_repair_env.txt"
echo score_artifact_dir=%SCORE_ARTIFACT_DIR%>>"%LOG_DIR%\%RUN_ID%_score_export_repair_env.txt"
echo dataset_cache_root=%DATASET_CACHE_ROOT%>>"%LOG_DIR%\%RUN_ID%_score_export_repair_env.txt"
echo branch=%BRANCH%>>"%LOG_DIR%\%RUN_ID%_score_export_repair_env.txt"

if exist "%PROJECT_ROOT%\.git" (
  cd /d "%PROJECT_ROOT%" || goto failed
  git status --short --branch > "%LOG_DIR%\%RUN_ID%_score_export_repair_git_status_before.txt" 2>&1
  git fetch origin %BRANCH% > "%LOG_DIR%\%RUN_ID%_score_export_repair_git_fetch_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_score_export_repair_git_fetch_stderr.txt" || goto failed
  git checkout %BRANCH% > "%LOG_DIR%\%RUN_ID%_score_export_repair_git_checkout_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_score_export_repair_git_checkout_stderr.txt" || goto failed
  git pull --ff-only origin %BRANCH% >> "%LOG_DIR%\%RUN_ID%_score_export_repair_git_fetch_stdout.txt" 2>> "%LOG_DIR%\%RUN_ID%_score_export_repair_git_fetch_stderr.txt" || goto failed
) else (
  git clone --branch %BRANCH% "%REPO_URL%" "%PROJECT_ROOT%" > "%LOG_DIR%\%RUN_ID%_score_export_repair_git_clone_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_score_export_repair_git_clone_stderr.txt" || goto failed
  cd /d "%PROJECT_ROOT%" || goto failed
)

git rev-parse HEAD > "%LOG_DIR%\%RUN_ID%_score_export_repair_git_revision.txt" 2>&1

"%PYTHON_EXE%" scripts\export-checkpoint-scores ^
  --checkpoint "%CHECKPOINT_DIR%\row0001_present_pairset_global_stats_seed0.pt" ^
  --eval-plan configs\experiment\innovation1\innovation1_spn_present_r8_trail_position_beamstats_65k_seed0.csv ^
  --eval-row-index 0 ^
  --model-key present_pairset_global_stats ^
  --hidden-bits 16 ^
  --batch-size 512 ^
  --device cuda:0 ^
  --dataset-cache-root "%DATASET_CACHE_ROOT%" ^
  --dataset-cache-chunk-size 8192 ^
  --dataset-cache-workers 4 ^
  --progress-output "%LOG_DIR%\trail_position_beamstats_score_export_repair_progress.jsonl" ^
  --expert-family trail_position_global_control ^
  --candidate-status near_neighbor_control ^
  --output-dir "%SCORE_ARTIFACT_DIR%\global_stats_control" ^
  > "%LOG_DIR%\%RUN_ID%_repair_export_global_stats_control_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_repair_export_global_stats_control_stderr.txt"
if errorlevel 1 goto failed

"%PYTHON_EXE%" scripts\export-checkpoint-scores ^
  --checkpoint "%CHECKPOINT_DIR%\row0002_present_trail_position_stats_pairset_seed0.pt" ^
  --eval-plan configs\experiment\innovation1\innovation1_spn_present_r8_trail_position_beamstats_65k_seed0.csv ^
  --eval-row-index 1 ^
  --model-key present_trail_position_stats_pairset ^
  --hidden-bits 16 ^
  --batch-size 512 ^
  --device cuda:0 ^
  --dataset-cache-root "%DATASET_CACHE_ROOT%" ^
  --dataset-cache-chunk-size 8192 ^
  --dataset-cache-workers 4 ^
  --progress-output "%LOG_DIR%\trail_position_beamstats_score_export_repair_progress.jsonl" ^
  --expert-family trail_position ^
  --candidate-status weak_positive ^
  --output-dir "%SCORE_ARTIFACT_DIR%\trail_position" ^
  > "%LOG_DIR%\%RUN_ID%_repair_export_trail_position_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_repair_export_trail_position_stderr.txt"
if errorlevel 1 goto failed

if not exist "%SCORE_ARTIFACT_DIR%\global_stats_control\models.json" goto failed
if not exist "%SCORE_ARTIFACT_DIR%\trail_position\models.json" goto failed
echo score_export_repair_done>"%LOG_DIR%\%RUN_ID%_score_export_repair_done.marker"
echo score_export_done>"%LOG_DIR%\%RUN_ID%_score_export_done.marker"
echo done>"%LOG_DIR%\%RUN_ID%_done.marker"
exit /b 0

:failed
echo score_export_repair_failed>"%LOG_DIR%\%RUN_ID%_score_export_repair_failed.marker"
exit /b 1
