@echo off
setlocal EnableExtensions

set RUN_ID=i2_speck32_hwang_topology_pairs_gpu0_20260717
set PHASE_C_RUN_ID=i2_speck32_hwang_phase_c_32plus32_gpu0_20260717
set RUNS_ROOT=G:\lxy\blockcipher-structure-adaptive-nd-runs
set RUN_ROOT=%RUNS_ROOT%\%RUN_ID%
set SOURCE_ROOT=%RUN_ROOT%\source
set PHASE_C_ROOT=%RUNS_ROOT%\%PHASE_C_RUN_ID%\source\results_archive\%PHASE_C_RUN_ID%
set LOG_DIR=%RUN_ROOT%\logs
set ARTIFACT_DIR=%RUN_ROOT%\artifacts
set ARCHIVE_DIR=%SOURCE_ROOT%\results_archive\%RUN_ID%
set PY=F:\Anaconda\envs\DWT\torch310\python.exe
set GITHUB_SSH_KEY=C:/Users/1304Lijinlin/.ssh/github_blockcipher_20260612_result_pusher_ed25519
set GIT_SSH_COMMAND=ssh -i %GITHUB_SSH_KEY% -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new
set CUDA_VISIBLE_DEVICES=0

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
if not exist "%ARTIFACT_DIR%" mkdir "%ARTIFACT_DIR%"
if not exist "%PHASE_C_ROOT%\cache\anchor\parity_rows.npy" goto missing_phase_c
if not exist "%PHASE_C_ROOT%\cache\anchor\completed.npy" goto missing_phase_c
if not exist "%PHASE_C_ROOT%\cache\control\parity_rows.npy" goto missing_phase_c
if not exist "%PHASE_C_ROOT%\cache\control\completed.npy" goto missing_phase_c
cd /d "%SOURCE_ROOT%" || goto failed
for /f "delims=" %%S in ('git status --porcelain') do goto dirty_source
git rev-parse HEAD > "%LOG_DIR%\%RUN_ID%_git_revision.txt" 2>&1 || goto failed
fc /b "%LOG_DIR%\%RUN_ID%_git_revision.txt" "%RUN_ROOT%\source_expected_commit.txt" > nul || goto source_revision_mismatch
git status --short --branch > "%LOG_DIR%\%RUN_ID%_git_status_before_run.txt" 2>&1 || goto failed
nvidia-smi > "%LOG_DIR%\%RUN_ID%_gpu_info.txt" 2>&1 || goto failed
"%PY%" -c "import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 1; print('torch',torch.__version__); print('cuda',torch.version.cuda); print('device',torch.cuda.get_device_name(0))" > "%LOG_DIR%\%RUN_ID%_torch_info.txt" 2> "%LOG_DIR%\%RUN_ID%_torch_info_stderr.txt" || goto failed

echo started>"%LOG_DIR%\%RUN_ID%_started.marker"
set PYTHONPATH=%SOURCE_ROOT%\src
"%PY%" scripts\audit-innovation2-speck-hwang-topology-pairs ^
  --run-id "%RUN_ID%" ^
  --output-root "%ARTIFACT_DIR%" ^
  --phase-c-root "%PHASE_C_ROOT%" ^
  --chunk-size 16777216 ^
  --device cuda ^
  > "%LOG_DIR%\%RUN_ID%_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_stderr.txt"
if errorlevel 1 goto failed

"%PY%" -c "import json,pathlib; g=json.loads(pathlib.Path(r'%ARTIFACT_DIR%\gate.json').read_text()); assert g['status'] in ('pass','hold'); assert all(g['readiness_checks'].values())" > "%LOG_DIR%\%RUN_ID%_gate_check.txt" 2> "%LOG_DIR%\%RUN_ID%_gate_check_stderr.txt" || goto failed

if not exist "%ARCHIVE_DIR%" mkdir "%ARCHIVE_DIR%"
xcopy /E /I /Y "%ARTIFACT_DIR%\cache" "%ARCHIVE_DIR%\cache" > nul || goto failed
xcopy /E /I /Y "%ARTIFACT_DIR%\baseline_phase_c" "%ARCHIVE_DIR%\baseline_phase_c" > nul || goto failed
copy /Y "%ARTIFACT_DIR%\results.jsonl" "%ARCHIVE_DIR%\results.jsonl" > nul || goto failed
copy /Y "%ARTIFACT_DIR%\summary.json" "%ARCHIVE_DIR%\summary.json" > nul || goto failed
copy /Y "%ARTIFACT_DIR%\gate.json" "%ARCHIVE_DIR%\gate.json" > nul || goto failed
copy /Y "%ARTIFACT_DIR%\metadata.json" "%ARCHIVE_DIR%\metadata.json" > nul || goto failed
copy /Y "%ARTIFACT_DIR%\progress.jsonl" "%ARCHIVE_DIR%\progress.jsonl" > nul || goto failed
copy /Y "%ARTIFACT_DIR%\kernel_basis.csv" "%ARCHIVE_DIR%\kernel_basis.csv" > nul || goto failed
copy /Y "%ARTIFACT_DIR%\keys.csv" "%ARCHIVE_DIR%\keys.csv" > nul || goto failed
copy /Y "%ARTIFACT_DIR%\screen_parity_rows.npy" "%ARCHIVE_DIR%\screen_parity_rows.npy" > nul || goto failed
copy /Y "%ARTIFACT_DIR%\validation_parity_rows.npy" "%ARCHIVE_DIR%\validation_parity_rows.npy" > nul || goto failed
copy /Y "%ARTIFACT_DIR%\selected_candidates.json" "%ARCHIVE_DIR%\selected_candidates.json" > nul || goto failed
copy /Y "%RUN_ROOT%\source_expected_commit.txt" "%ARCHIVE_DIR%\source_expected_commit.txt" > nul || goto failed
copy /Y "%LOG_DIR%\%RUN_ID%_git_revision.txt" "%ARCHIVE_DIR%\git_revision.txt" > nul || goto failed
copy /Y "%LOG_DIR%\%RUN_ID%_git_status_before_run.txt" "%ARCHIVE_DIR%\git_status_before_run.txt" > nul || goto failed
copy /Y "%LOG_DIR%\%RUN_ID%_gpu_info.txt" "%ARCHIVE_DIR%\gpu_info.txt" > nul || goto failed
copy /Y "%LOG_DIR%\%RUN_ID%_torch_info.txt" "%ARCHIVE_DIR%\torch_info.txt" > nul || goto failed
copy /Y "%LOG_DIR%\%RUN_ID%_stdout.txt" "%ARCHIVE_DIR%\stdout.txt" > nul || goto failed
copy /Y "%LOG_DIR%\%RUN_ID%_stderr.txt" "%ARCHIVE_DIR%\stderr.txt" > nul || goto failed
echo * -text>"%ARCHIVE_DIR%\.gitattributes"
"%PY%" -c "import hashlib,pathlib; root=pathlib.Path(r'%ARCHIVE_DIR%'); fs=sorted(p for p in root.rglob('*') if p.is_file() and not p.name == 'SHA256SUMS'); (root/'SHA256SUMS').write_text('\n'.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.relative_to(root).as_posix() for p in fs)+'\n')" || goto failed

git config user.name "remote-experiment"
git config user.email "remote-experiment@local.invalid"
git checkout -B results/%RUN_ID% > "%LOG_DIR%\%RUN_ID%_branch_checkout.txt" 2>&1 || goto failed
git add "results_archive\%RUN_ID%" || goto failed
git commit -m "results: %RUN_ID% topology pairs" > "%LOG_DIR%\%RUN_ID%_branch_commit.txt" 2>&1 || goto failed
git push origin HEAD:refs/heads/results/%RUN_ID% > "%LOG_DIR%\%RUN_ID%_branch_push.txt" 2>&1 || goto failed
echo pushed>"%LOG_DIR%\%RUN_ID%_result_branch_pushed.marker"
echo done>"%LOG_DIR%\%RUN_ID%_done.marker"
exit /b 0

:missing_phase_c
echo missing_phase_c_baseline>"%LOG_DIR%\%RUN_ID%_failed.marker"
exit /b 4

:dirty_source
echo dirty_source>"%LOG_DIR%\%RUN_ID%_failed.marker"
exit /b 2

:source_revision_mismatch
echo source_revision_mismatch>"%LOG_DIR%\%RUN_ID%_failed.marker"
exit /b 3

:failed
echo failed>"%LOG_DIR%\%RUN_ID%_failed.marker"
exit /b 1
