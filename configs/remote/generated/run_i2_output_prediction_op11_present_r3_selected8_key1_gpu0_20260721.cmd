@echo off
setlocal EnableExtensions

set PHYSICAL_GPU=%~1
if "%PHYSICAL_GPU%"=="" goto invalid_arguments

set RUN_ID=i2_output_prediction_op11_present_r3_selected8_key1_gpu0_20260721
set RUNS_ROOT=G:\lxy\blockcipher-structure-adaptive-nd-runs
set RUN_ROOT=%RUNS_ROOT%\%RUN_ID%
set SOURCE_ROOT=%RUN_ROOT%\source
set LOG_DIR=%RUN_ROOT%\logs
set RESULTS_DIR=%RUN_ROOT%\results
set ARCHIVE_DIR=%SOURCE_ROOT%\results_archive\%RUN_ID%
set PLAN=configs\experiment\innovation2\innovation2_output_prediction_op11_present_r3_selected8_key1.json
set DOC_PLAN=docs\experiments\innovation2-output-prediction-op11-present-r3-selected8-independent-key-plan.md
set REMOTE_CONFIG=configs\remote\innovation2_output_prediction_op11_present_r3_selected8_key1_gpu0_20260721.json
set PY=F:\Anaconda\envs\DWT\torch310\python.exe
set PYTHONPATH=%SOURCE_ROOT%\src
set GITHUB_SSH_KEY=C:/Users/1304Lijinlin/.ssh/github_blockcipher_20260612_result_pusher_ed25519
set GIT_SSH_COMMAND=ssh -i %GITHUB_SSH_KEY% -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new
set CUDA_VISIBLE_DEVICES=%PHYSICAL_GPU%
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if not exist "%RUN_ROOT%" mkdir "%RUN_ROOT%"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
if not exist "%RESULTS_DIR%" mkdir "%RESULTS_DIR%"
if exist "%LOG_DIR%\%RUN_ID%_done.marker" goto already_complete
if exist "%LOG_DIR%\%RUN_ID%_failed.marker" del /Q "%LOG_DIR%\%RUN_ID%_failed.marker"
if exist "%LOG_DIR%\%RUN_ID%_started.marker" del /Q "%LOG_DIR%\%RUN_ID%_started.marker"
if exist "%LOG_DIR%\%RUN_ID%_failure_reason.txt" del /Q "%LOG_DIR%\%RUN_ID%_failure_reason.txt"

cd /d "%SOURCE_ROOT%" || goto failed
for /f "delims=" %%S in ('git status --porcelain') do goto dirty_source
git rev-parse HEAD > "%LOG_DIR%\%RUN_ID%_git_revision.txt" 2>&1 || goto failed
fc /b "%LOG_DIR%\%RUN_ID%_git_revision.txt" "%RUN_ROOT%\source_expected_commit.txt" > nul || goto source_revision_mismatch
git status --short --branch > "%LOG_DIR%\%RUN_ID%_git_status_before_run.txt" 2>&1 || goto failed
nvidia-smi > "%LOG_DIR%\%RUN_ID%_gpu_info.txt" 2>&1 || goto failed
"%PY%" -c "import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 1; print('torch', torch.__version__); print('cuda', torch.version.cuda); print('available', torch.cuda.is_available()); print('visible_count', torch.cuda.device_count()); print('device0', torch.cuda.get_device_name(0))" > "%LOG_DIR%\%RUN_ID%_torch_info.txt" 2> "%LOG_DIR%\%RUN_ID%_torch_info_stderr.txt" || goto failed
"%PY%" -c "import json,pathlib; cfg=json.loads(pathlib.Path(r'%REMOTE_CONFIG%').read_text(encoding='utf-8')); plan=json.loads(pathlib.Path(r'%PLAN%').read_text(encoding='utf-8')); common=plan['common']; assert cfg['train_total_rows']==131072 and cfg['test_total_rows']==65536 and cfg['expected_result_rows']==24 and cfg['epochs']==100 and cfg['batch_size']==250 and cfg['seed']==1 and cfg['sample_classification'] is False and cfg['selected_msb_indices']==[0,2,8,10,32,34,40,42] and common['sample_classification'] is False and common['target']=='eight_preregistered_msb_first_true_ciphertext_bits' and len(plan['rows'])==3; print('status=pass')" > "%LOG_DIR%\%RUN_ID%_readiness.txt" 2> "%LOG_DIR%\%RUN_ID%_readiness_stderr.txt" || goto failed

echo started>"%LOG_DIR%\%RUN_ID%_started.marker"
"%PY%" scripts\run-innovation2-selected-output-bit-head ^
  --mode independent_key_confirmation ^
  --run-id "%RUN_ID%" ^
  --device cuda ^
  --output-root "%RESULTS_DIR%" ^
  > "%LOG_DIR%\%RUN_ID%_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_stderr.txt"
if errorlevel 1 goto failed

echo deferred_to_local_verified_retrieval>"%LOG_DIR%\%RUN_ID%_plot_deferred.marker"
for %%F in (results.jsonl progress.jsonl history.csv metadata.json summary.json gate.json checkpoint_manifest.json) do if not exist "%RESULTS_DIR%\%%F" goto incomplete_results
for %%F in (plaintexts.npy features.npy full_targets.npy cache_metadata.json) do if not exist "%RESULTS_DIR%\data\%%F" goto incomplete_results
for %%F in (full64_mlp_true_output_final.pt selected8_mlp_true_output_final.pt selected8_mlp_label_shuffle_final.pt) do if not exist "%RESULTS_DIR%\models\%%F" goto incomplete_results
set RESULT_LINES=0
for /f "tokens=3" %%L in ('find /c /v "" "%RESULTS_DIR%\results.jsonl"') do set RESULT_LINES=%%L
echo result_lines=%RESULT_LINES% > "%LOG_DIR%\%RUN_ID%_result_gate.txt"
echo expected_rows=24 >> "%LOG_DIR%\%RUN_ID%_result_gate.txt"
if not "%RESULT_LINES%"=="24" goto incomplete_results
"%PY%" -c "import json,pathlib; root=pathlib.Path(r'%RESULTS_DIR%'); gate=json.loads((root/'gate.json').read_text(encoding='utf-8')); meta=json.loads((root/'metadata.json').read_text(encoding='utf-8')); cache=json.loads((root/'data'/'cache_metadata.json').read_text(encoding='utf-8')); checkpoints=json.loads((root/'checkpoint_manifest.json').read_text(encoding='utf-8')); assert gate['status'] in {'pass','hold'} and all(gate['protocol_checks'].values()) and all(gate['execution_checks'].values()) and meta['sample_classification'] is False and meta['config']['seed']==1 and meta['selected_msb_indices']==[0,2,8,10,32,34,40,42] and cache['status']=='complete' and cache['completed_rows']==196608 and len(checkpoints)==3 and all(row['sha256'] for row in checkpoints)" || goto incomplete_results

if not exist "%ARCHIVE_DIR%" mkdir "%ARCHIVE_DIR%"
for %%F in (results.jsonl progress.jsonl history.csv metadata.json summary.json gate.json checkpoint_manifest.json) do copy /Y "%RESULTS_DIR%\%%F" "%ARCHIVE_DIR%\%%F" > nul || goto failed
copy /Y "%RESULTS_DIR%\data\cache_metadata.json" "%ARCHIVE_DIR%\cache_metadata.json" > nul || goto failed
copy /Y "%LOG_DIR%\%RUN_ID%_git_revision.txt" "%ARCHIVE_DIR%\git_revision.txt" > nul || goto failed
copy /Y "%LOG_DIR%\%RUN_ID%_git_status_before_run.txt" "%ARCHIVE_DIR%\git_status_before_run.txt" > nul || goto failed
copy /Y "%LOG_DIR%\%RUN_ID%_gpu_info.txt" "%ARCHIVE_DIR%\gpu_info.txt" > nul || goto failed
copy /Y "%LOG_DIR%\%RUN_ID%_torch_info.txt" "%ARCHIVE_DIR%\torch_info.txt" > nul || goto failed
copy /Y "%LOG_DIR%\%RUN_ID%_stdout.txt" "%ARCHIVE_DIR%\run_stdout.txt" > nul || goto failed
copy /Y "%LOG_DIR%\%RUN_ID%_stderr.txt" "%ARCHIVE_DIR%\run_stderr.txt" > nul || goto failed
copy /Y "%LOG_DIR%\%RUN_ID%_result_gate.txt" "%ARCHIVE_DIR%\result_gate.txt" > nul || goto failed
copy /Y "%REMOTE_CONFIG%" "%ARCHIVE_DIR%\remote_config.json" > nul || goto failed
copy /Y "%PLAN%" "%ARCHIVE_DIR%\plan.json" > nul || goto failed
copy /Y "%DOC_PLAN%" "%ARCHIVE_DIR%\experiment_plan.md" > nul || goto failed
echo * -text>"%ARCHIVE_DIR%\.gitattributes"
"%PY%" -c "import hashlib,pathlib; root=pathlib.Path(r'%ARCHIVE_DIR%'); files=sorted(p for p in root.rglob('*') if p.is_file() and p.name != 'SHA256SUMS'); (root/'SHA256SUMS').write_text('\n'.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.relative_to(root).as_posix() for p in files)+'\n',encoding='utf-8')" || goto failed

git config user.name "remote-experiment"
git config user.email "remote-experiment@local.invalid"
git checkout -B results/%RUN_ID% > "%LOG_DIR%\%RUN_ID%_result_branch_checkout.txt" 2>&1 || goto failed
git add "results_archive\%RUN_ID%" || goto failed
git commit -m "results: %RUN_ID%" > "%LOG_DIR%\%RUN_ID%_result_branch_commit.txt" 2>&1 || goto failed
git push origin HEAD:refs/heads/results/%RUN_ID% > "%LOG_DIR%\%RUN_ID%_result_branch_push.txt" 2>&1 || goto failed
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

:already_complete
exit /b 7

:failed
echo failed>"%LOG_DIR%\%RUN_ID%_failed.marker"
exit /b 1
