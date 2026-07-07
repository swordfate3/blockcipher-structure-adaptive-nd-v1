@echo off
setlocal EnableExtensions EnableDelayedExpansion

set RUN_ID=i1_present_r8_residual_focus_262k
set REPO_URL=git@github.com:swordfate3/blockcipher-structure-adaptive-nd-v1.git
set BRANCH=main
set RUN_ROOT=G:\lxy\blockcipher-structure-adaptive-nd-runs\%RUN_ID%
set SOURCE_ROOT=%RUN_ROOT%\source
set ARTIFACT_ROOT=%RUN_ROOT%\artifacts
set LOG_DIR=%RUN_ROOT%\logs
set PYTHON_EXE=F:\Anaconda\envs\DWT\torch310\python.exe
set GITHUB_SSH_KEY=C:/Users/1304Lijinlin/.ssh/github_blockcipher_20260612_result_pusher_ed25519
set GIT_SSH_COMMAND=ssh -i %GITHUB_SSH_KEY% -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new

if not exist "%RUN_ROOT%" mkdir "%RUN_ROOT%"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
if not exist "%ARTIFACT_ROOT%" mkdir "%ARTIFACT_ROOT%"
echo run_id=%RUN_ID%>"%LOG_DIR%\%RUN_ID%_launch_env.txt"
echo source_root=%SOURCE_ROOT%>>"%LOG_DIR%\%RUN_ID%_launch_env.txt"
echo artifact_root=%ARTIFACT_ROOT%>>"%LOG_DIR%\%RUN_ID%_launch_env.txt"

if exist "%SOURCE_ROOT%\.git" (
  cd /d "%SOURCE_ROOT%" || goto failed
  git status --short --branch > "%LOG_DIR%\%RUN_ID%_git_status_before_run.txt" 2>&1
  git fetch origin %BRANCH% > "%LOG_DIR%\%RUN_ID%_git_fetch_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_git_fetch_stderr.txt" || goto failed
  git checkout %BRANCH% > "%LOG_DIR%\%RUN_ID%_git_checkout_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_git_checkout_stderr.txt" || goto failed
  git pull --ff-only origin %BRANCH% >> "%LOG_DIR%\%RUN_ID%_git_fetch_stdout.txt" 2>> "%LOG_DIR%\%RUN_ID%_git_fetch_stderr.txt" || goto failed
) else (
  if exist "%SOURCE_ROOT%" rmdir /s /q "%SOURCE_ROOT%"
  git clone --branch %BRANCH% "%REPO_URL%" "%SOURCE_ROOT%" > "%LOG_DIR%\%RUN_ID%_git_clone_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_git_clone_stderr.txt" || goto failed
  cd /d "%SOURCE_ROOT%" || goto failed
)

git rev-parse HEAD > "%LOG_DIR%\%RUN_ID%_git_revision.txt" 2>&1
echo started>"%LOG_DIR%\%RUN_ID%_started.marker"

echo done>"%LOG_DIR%\%RUN_ID%_done.marker"
exit /b 0

:failed
echo failed>"%LOG_DIR%\%RUN_ID%_failed.marker"
exit /b 1
