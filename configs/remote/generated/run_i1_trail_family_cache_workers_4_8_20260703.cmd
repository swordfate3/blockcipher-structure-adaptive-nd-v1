@echo off
setlocal EnableExtensions

set RUN_ID=i1_trail_family_cache_workers_4_8_20260703
set RUN_ROOT=G:\lxy\blockcipher-structure-adaptive-nd-runs\%RUN_ID%
set SOURCE_DIR=%RUN_ROOT%\source
set LOG_DIR=%RUN_ROOT%\logs
set RESULT_DIR=%RUN_ROOT%\results
set CACHE_OUTPUT=%RESULT_DIR%\trail_family_cache_bench
set PYTHON=F:\Anaconda\envs\DWT\torch310\python.exe
set REPO_URL=git@github.com:swordfate3/blockcipher-structure-adaptive-nd-v1.git
set GITHUB_SSH_KEY=C:/Users/1304Lijinlin/.ssh/github_blockcipher_20260612_result_pusher_ed25519
set GIT_SSH_COMMAND=ssh -i %GITHUB_SSH_KEY% -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new

if not exist G:\lxy\blockcipher-structure-adaptive-nd-runs mkdir G:\lxy\blockcipher-structure-adaptive-nd-runs
if not exist "%RUN_ROOT%" mkdir "%RUN_ROOT%"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
if not exist "%RESULT_DIR%" mkdir "%RESULT_DIR%"

echo started>"%LOG_DIR%\%RUN_ID%_started.marker"
echo RUN_ID=%RUN_ID%>"%LOG_DIR%\%RUN_ID%_launch_env.txt"
echo RUN_ROOT=%RUN_ROOT%>>"%LOG_DIR%\%RUN_ID%_launch_env.txt"
echo SOURCE_DIR=%SOURCE_DIR%>>"%LOG_DIR%\%RUN_ID%_launch_env.txt"

if exist "%SOURCE_DIR%\.git" (
  cd /d "%SOURCE_DIR%"
  git fetch origin main >"%LOG_DIR%\%RUN_ID%_git_fetch_stdout.txt" 2>"%LOG_DIR%\%RUN_ID%_git_fetch_stderr.txt"
  if errorlevel 1 goto git_failed
  git checkout main >"%LOG_DIR%\%RUN_ID%_git_checkout_stdout.txt" 2>"%LOG_DIR%\%RUN_ID%_git_checkout_stderr.txt"
  if errorlevel 1 goto git_failed
  git pull --ff-only origin main >"%LOG_DIR%\%RUN_ID%_git_pull_stdout.txt" 2>"%LOG_DIR%\%RUN_ID%_git_pull_stderr.txt"
  if errorlevel 1 goto git_failed
) else (
  git clone "%REPO_URL%" "%SOURCE_DIR%" >"%LOG_DIR%\%RUN_ID%_git_clone_stdout.txt" 2>"%LOG_DIR%\%RUN_ID%_git_clone_stderr.txt"
  if errorlevel 1 goto git_failed
  cd /d "%SOURCE_DIR%"
)

git rev-parse HEAD >"%LOG_DIR%\%RUN_ID%_git_revision.txt" 2>"%LOG_DIR%\%RUN_ID%_git_revision_stderr.txt"
git status --short --branch >"%LOG_DIR%\%RUN_ID%_git_status_before_run.txt" 2>"%LOG_DIR%\%RUN_ID%_git_status_before_run_stderr.txt"
nvidia-smi >"%LOG_DIR%\%RUN_ID%_gpu_info.txt" 2>"%LOG_DIR%\%RUN_ID%_gpu_info_stderr.txt"
"%PYTHON%" -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('device_count', torch.cuda.device_count())" >"%LOG_DIR%\%RUN_ID%_torch_info.txt" 2>"%LOG_DIR%\%RUN_ID%_torch_info_stderr.txt"

set PYTHONPATH=%SOURCE_DIR%\plugins\blockcipher-training-accelerator\src;%SOURCE_DIR%\src
"%PYTHON%" -m blockcipher_training_accelerator bench-trail-family-cache ^
  --rounds 7 ^
  --difference-profile present_zhang_wang2022_mcnd ^
  --samples-per-class 262144 ^
  --pairs-per-sample 16 ^
  --sample-structure zhang_wang_case2_official_mcnd ^
  --negative-mode encrypted_random_plaintexts ^
  --key-rotation-interval 0 ^
  --beam-width 4 ^
  --depth 3 ^
  --seed 20260703 ^
  --chunk-size 8192 ^
  --workers 4 8 ^
  --output-root "%CACHE_OUTPUT%" ^
  >"%LOG_DIR%\%RUN_ID%_stdout.txt" 2>"%LOG_DIR%\%RUN_ID%_stderr.txt"

if errorlevel 1 goto run_failed
echo done>"%LOG_DIR%\%RUN_ID%_done.marker"
exit /b 0

:git_failed
echo git_failed>"%LOG_DIR%\%RUN_ID%_failed.marker"
exit /b 2

:run_failed
echo run_failed>"%LOG_DIR%\%RUN_ID%_failed.marker"
exit /b 1
