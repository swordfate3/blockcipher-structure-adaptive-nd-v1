@echo off
setlocal EnableExtensions EnableDelayedExpansion

set PHYSICAL_GPU=%~1
if "%PHYSICAL_GPU%"=="" goto invalid_arguments

set RUN_ID=i2_present_r8_high_round_integral_bridge_262144_seed0_gpu0_20260716
set RUNS_ROOT=G:\lxy\blockcipher-structure-adaptive-nd-runs
set RUN_ROOT=%RUNS_ROOT%\%RUN_ID%
set SOURCE_ROOT=%RUN_ROOT%\source
set CACHE_ROOT=%RUN_ROOT%\dataset_cache
set LOG_DIR=%RUN_ROOT%\logs
set RESULTS_DIR=%RUN_ROOT%\results
set ARCHIVE_DIR=%SOURCE_ROOT%\results_archive\%RUN_ID%
set PLAN=configs\experiment\innovation2\innovation2_present_r8_high_round_integral_bridge_262144_seed0.json
set REMOTE_CONFIG=configs\remote\innovation2_present_r8_high_round_integral_bridge_262144_seed0_gpu0_20260716.json
set PY=F:\Anaconda\envs\DWT\torch310\python.exe
set GITHUB_SSH_KEY=C:/Users/1304Lijinlin/.ssh/github_blockcipher_20260612_result_pusher_ed25519
set GIT_SSH_COMMAND=ssh -i %GITHUB_SSH_KEY% -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new
set CUDA_VISIBLE_DEVICES=%PHYSICAL_GPU%
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if not exist "%RUN_ROOT%" mkdir "%RUN_ROOT%"
if not exist "%CACHE_ROOT%" mkdir "%CACHE_ROOT%"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
if not exist "%RESULTS_DIR%" mkdir "%RESULTS_DIR%"

cd /d "%SOURCE_ROOT%" || goto failed
for /f "delims=" %%S in ('git status --porcelain') do goto dirty_source
git rev-parse HEAD > "%LOG_DIR%\%RUN_ID%_git_revision.txt" 2>&1 || goto failed
fc /b "%LOG_DIR%\%RUN_ID%_git_revision.txt" "%RUN_ROOT%\source_expected_commit.txt" > nul || goto source_revision_mismatch
git status --short --branch > "%LOG_DIR%\%RUN_ID%_git_status_before_run.txt" 2>&1 || goto failed
nvidia-smi > "%LOG_DIR%\%RUN_ID%_gpu_info.txt" 2>&1 || goto failed
"%PY%" -c "import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 1; print('torch', torch.__version__); print('cuda', torch.version.cuda); print('available', torch.cuda.is_available()); print('visible_count', torch.cuda.device_count()); print('device0', torch.cuda.get_device_name(0))" > "%LOG_DIR%\%RUN_ID%_torch_info.txt" 2> "%LOG_DIR%\%RUN_ID%_torch_info_stderr.txt" || goto failed
"%PY%" scripts\check-remote-readiness --config "%REMOTE_CONFIG%" > "%LOG_DIR%\%RUN_ID%_readiness.txt" 2> "%LOG_DIR%\%RUN_ID%_readiness_stderr.txt" || goto failed

echo started>"%LOG_DIR%\%RUN_ID%_started.marker"
"%PY%" scripts\run-innovation2-high-round-integral ^
  --run-id "%RUN_ID%" ^
  --output-root "%RESULTS_DIR%" ^
  --cache-root "%CACHE_ROOT%" ^
  --rounds 8 ^
  --train-rows 262144 ^
  --validation-rows 32768 ^
  --test-rows 65536 ^
  --multiset-count 2 ^
  --epochs 5 ^
  --batch-size 128 ^
  --base-channels 16 ^
  --head-bits 256 ^
  --block-count 2 ^
  --dropout 0.1 ^
  --learning-rate 0.001 ^
  --weight-decay 0.00001 ^
  --cache-chunk-size 256 ^
  --seed 0 ^
  --device cuda ^
  --gate-mode bridge ^
  > "%LOG_DIR%\%RUN_ID%_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_stderr.txt"
if errorlevel 1 goto failed

for %%F in (results.jsonl progress.jsonl dataset_summary.json fixed_baselines.json gate.json validation.json curves.svg history.csv) do if not exist "%RESULTS_DIR%\%%F" goto incomplete_results
set RESULT_LINES=0
for /f "tokens=3" %%L in ('find /c /v "" "%RESULTS_DIR%\results.jsonl"') do set RESULT_LINES=%%L
echo result_lines=%RESULT_LINES% > "%LOG_DIR%\%RUN_ID%_result_gate.txt"
echo expected_rows=4 >> "%LOG_DIR%\%RUN_ID%_result_gate.txt"
if not "%RESULT_LINES%"=="4" goto incomplete_results
"%PY%" -c "import json,pathlib; root=pathlib.Path(r'%RESULTS_DIR%'); gate=json.loads((root/'gate.json').read_text()); validation=json.loads((root/'validation.json').read_text()); assert all(gate['bridge_plan_checks'].values()); assert validation['status']=='pass'; assert gate['artifact_validation']['status']=='pass'" || goto incomplete_results

if not exist "%ARCHIVE_DIR%" mkdir "%ARCHIVE_DIR%"
for %%F in (results.jsonl progress.jsonl dataset_summary.json fixed_baselines.json gate.json validation.json curves.svg history.csv) do copy /Y "%RESULTS_DIR%\%%F" "%ARCHIVE_DIR%\%%F" > nul || goto failed
copy /Y "%LOG_DIR%\%RUN_ID%_git_revision.txt" "%ARCHIVE_DIR%\git_revision.txt" > nul || goto failed
copy /Y "%LOG_DIR%\%RUN_ID%_git_status_before_run.txt" "%ARCHIVE_DIR%\git_status_before_run.txt" > nul || goto failed
copy /Y "%LOG_DIR%\%RUN_ID%_gpu_info.txt" "%ARCHIVE_DIR%\gpu_info.txt" > nul || goto failed
copy /Y "%LOG_DIR%\%RUN_ID%_torch_info.txt" "%ARCHIVE_DIR%\torch_info.txt" > nul || goto failed
copy /Y "%LOG_DIR%\%RUN_ID%_stdout.txt" "%ARCHIVE_DIR%\run_stdout.txt" > nul || goto failed
copy /Y "%LOG_DIR%\%RUN_ID%_stderr.txt" "%ARCHIVE_DIR%\run_stderr.txt" > nul || goto failed
copy /Y "%LOG_DIR%\%RUN_ID%_result_gate.txt" "%ARCHIVE_DIR%\result_gate.txt" > nul || goto failed
copy /Y "%REMOTE_CONFIG%" "%ARCHIVE_DIR%\remote_config.json" > nul || goto failed
copy /Y "%PLAN%" "%ARCHIVE_DIR%\plan.json" > nul || goto failed
"%PY%" -c "import json,pathlib; root=pathlib.Path(r'%CACHE_ROOT%'); out={p.parent.name:json.loads(p.read_text(encoding='utf-8')) for p in sorted(root.glob('*/metadata.json'))}; assert len(out)==3 and all(v.get('status')=='complete' for v in out.values()); pathlib.Path(r'%ARCHIVE_DIR%\cache_metadata.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n',encoding='utf-8')" || goto failed
echo * -text>"%ARCHIVE_DIR%\.gitattributes"
"%PY%" -c "import hashlib,pathlib; root=pathlib.Path(r'%ARCHIVE_DIR%'); files=sorted(p for p in root.rglob('*') if p.is_file() and p.name!='SHA256SUMS'); (root/'SHA256SUMS').write_text('\n'.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.relative_to(root).as_posix() for p in files)+'\n',encoding='utf-8')" || goto failed

git config user.name "remote-experiment"
git config user.email "remote-experiment@local.invalid"
git checkout -B results/%RUN_ID% > "%LOG_DIR%\%RUN_ID%_result_branch_checkout.txt" 2>&1 || goto failed
git add "results_archive\%RUN_ID%" || goto failed
git commit -m "results: %RUN_ID% bridge" > "%LOG_DIR%\%RUN_ID%_result_branch_commit.txt" 2>&1 || goto failed
git push origin HEAD:refs/heads/results/%RUN_ID% > "%LOG_DIR%\%RUN_ID%_result_branch_push.txt" 2>&1 || goto failed
git rev-parse HEAD > "%LOG_DIR%\%RUN_ID%_result_branch_revision.txt" 2>&1 || goto failed
echo pushed>"%LOG_DIR%\%RUN_ID%_result_branch_pushed.marker"
echo done>"%LOG_DIR%\%RUN_ID%_done.marker"
exit /b 0

:dirty_source
echo dirty_source>"%LOG_DIR%\%RUN_ID%_failed.marker"
echo Remote run-owned source clone is dirty.>"%LOG_DIR%\%RUN_ID%_failure_reason.txt"
exit /b 2

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
