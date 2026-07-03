@echo off
setlocal EnableExtensions EnableDelayedExpansion

set RUN_ID=i1_active_auxiliary_r7_262k_seed0_gpu1_20260703
set REPO_URL=git@github.com:swordfate3/blockcipher-structure-adaptive-nd-v1.git
set BRANCH=main
set PROJECT_ROOT=G:\lxy\blockcipher-structure-adaptive-nd-runs\%RUN_ID%\source
set RUN_ROOT=G:\lxy\blockcipher-structure-adaptive-nd-runs\%RUN_ID%
set LOG_DIR=%RUN_ROOT%\logs
set RESULTS_DIR=%RUN_ROOT%\results
set ACTIVE_AUXILIARY_CACHE_ROOT=G:\lxy\blockcipher-structure-adaptive-nd-runs\active_auxiliary_cache
set PYTHON_EXE=F:\Anaconda\envs\DWT\torch310\python.exe
set GITHUB_SSH_KEY=C:/Users/1304Lijinlin/.ssh/github_blockcipher_20260612_result_pusher_ed25519
set GIT_SSH_COMMAND=ssh -i %GITHUB_SSH_KEY% -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new

if not exist "%RUN_ROOT%" mkdir "%RUN_ROOT%"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
if not exist "%RESULTS_DIR%" mkdir "%RESULTS_DIR%"
if not exist "%ACTIVE_AUXILIARY_CACHE_ROOT%" mkdir "%ACTIVE_AUXILIARY_CACHE_ROOT%"

echo run_id=%RUN_ID%>"%LOG_DIR%\%RUN_ID%_launch_env.txt"
echo project_root=%PROJECT_ROOT%>>"%LOG_DIR%\%RUN_ID%_launch_env.txt"
echo run_root=%RUN_ROOT%>>"%LOG_DIR%\%RUN_ID%_launch_env.txt"
echo active_auxiliary_cache_root=%ACTIVE_AUXILIARY_CACHE_ROOT%>>"%LOG_DIR%\%RUN_ID%_launch_env.txt"
echo python_exe=%PYTHON_EXE%>>"%LOG_DIR%\%RUN_ID%_launch_env.txt"
echo branch=%BRANCH%>>"%LOG_DIR%\%RUN_ID%_launch_env.txt"

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
"%PYTHON_EXE%" scripts\check-remote-readiness --config configs\remote\innovation1_spn_present_active_auxiliary_r7_262k_seed0_gpu1_20260703.json > "%LOG_DIR%\%RUN_ID%_readiness.txt" 2> "%LOG_DIR%\%RUN_ID%_readiness_stderr.txt" || goto failed
echo started>"%LOG_DIR%\%RUN_ID%_started.marker"

"%PYTHON_EXE%" scripts\spn-active-auxiliary-matrix ^
  --config configs\experiment\innovation1\innovation1_spn_present_active_auxiliary_r7_262k_seed0.json ^
  --output "%RESULTS_DIR%\%RUN_ID%.jsonl" ^
  > "%LOG_DIR%\%RUN_ID%_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_stderr.txt"

if errorlevel 1 goto failed
echo done>"%LOG_DIR%\%RUN_ID%_done.marker"
exit /b 0

:failed
echo failed>"%LOG_DIR%\%RUN_ID%_failed.marker"
exit /b 1
