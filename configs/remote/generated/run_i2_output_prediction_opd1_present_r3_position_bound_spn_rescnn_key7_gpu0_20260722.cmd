@echo off
setlocal EnableExtensions

set PHYSICAL_GPU=%~1
if "%PHYSICAL_GPU%"=="" goto invalid_arguments

set RUN_ID=i2_output_prediction_opd1_present_r3_position_bound_spn_rescnn_key7_gpu0_20260722
set REMOTE_DIR=i2_opd1_poshead_k7_20260722
set ARCHIVE_NAME=i2_opd1_poshead_k7_20260722
set RUNS_ROOT=G:\lxy\blockcipher-structure-adaptive-nd-runs
set RUN_ROOT=%RUNS_ROOT%\%REMOTE_DIR%
set SOURCE_ROOT=%RUN_ROOT%\source
set LOG_DIR=%RUN_ROOT%\logs
set RESULTS_DIR=%RUN_ROOT%\results
set ARCHIVE_DIR=%SOURCE_ROOT%\results_archive\%ARCHIVE_NAME%
set OPC1_GATE=%RUNS_ROOT%\i2_opc1_hybrid_k6_20260722\results\gate.json
set OPN1_GATE=%SOURCE_ROOT%\configs\experiment\innovation2\authorities\innovation2_output_prediction_opn1_gate_20260722.json
set PLAN=configs\experiment\innovation2\innovation2_output_prediction_opd1_present_r3_position_bound_spn_rescnn_key7.json
set DOC_PLAN=docs\experiments\innovation2-output-prediction-opd1-position-bound-spn-rescnn-plan.md
set REMOTE_CONFIG=configs\remote\innovation2_output_prediction_opd1_present_r3_position_bound_spn_rescnn_key7_gpu0_20260722.json
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
"%PY%" -c "import hashlib,json,pathlib; cfg=json.loads(pathlib.Path(r'%REMOTE_CONFIG%').read_text(encoding='utf-8')); plan=json.loads(pathlib.Path(r'%PLAN%').read_text(encoding='utf-8')); opc1_path=pathlib.Path(r'%OPC1_GATE%'); opn1_path=pathlib.Path(r'%OPN1_GATE%'); opc1=json.loads(opc1_path.read_text(encoding='utf-8')); opn1=json.loads(opn1_path.read_text(encoding='utf-8')); assert hashlib.sha256(opc1_path.read_bytes()).hexdigest()==cfg['opc1_gate_sha256']==plan['source_authorities']['opc1']['gate_sha256']; assert hashlib.sha256(opn1_path.read_bytes()).hexdigest()==cfg['opn1_gate_sha256']==plan['source_authorities']['opn1']['gate_sha256']; assert opc1['run_id']==plan['source_authorities']['opc1']['run_id'] and opc1['status']=='hold' and opc1['decision']=='innovation2_spn_rescnn_hybrid_not_supported' and all(opc1['protocol_checks'].values()) and all(opc1['execution_checks'].values()); assert opn1['run_id']==plan['source_authorities']['opn1']['run_id'] and opn1['status']=='pass' and opn1['decision']=='innovation2_spn_rescnn_final_routing_absorbable_by_global_head' and all(opn1['protocol_checks'].values()) and all(opn1['execution_checks'].values()); common=plan['common']; assert cfg['train_total_rows']==131072 and cfg['test_total_rows']==65536 and cfg['expected_result_rows']==40 and cfg['expected_history_rows']==500 and cfg['expected_checkpoints']==5 and cfg['expected_cache_rows']==196608 and cfg['seed']==7 and cfg['sample_classification'] is False and common['sample_classification'] is False and len(plan['rows'])==5 and len(set(row['parameters'] for row in plan['rows'][1:]))==1; print('status=pass')" > "%LOG_DIR%\%RUN_ID%_readiness.txt" 2> "%LOG_DIR%\%RUN_ID%_readiness_stderr.txt" || goto readiness_failed

echo started>"%LOG_DIR%\%RUN_ID%_started.marker"
"%PY%" scripts\run-innovation2-selected-output-position-bound-spn-rescnn ^
  --mode position_bound_head ^
  --opc1-gate "%OPC1_GATE%" ^
  --opn1-gate "%OPN1_GATE%" ^
  --run-id "%RUN_ID%" ^
  --device cuda ^
  --output-root "%RESULTS_DIR%" ^
  > "%LOG_DIR%\%RUN_ID%_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_stderr.txt"
if errorlevel 1 goto failed

echo deferred_to_local_verified_retrieval>"%LOG_DIR%\%RUN_ID%_plot_deferred.marker"
for %%F in (results.jsonl progress.jsonl history.csv metadata.json opc1_gate.json opn1_gate.json summary.json gate.json checkpoint_manifest.json) do if not exist "%RESULTS_DIR%\%%F" goto incomplete_results
for %%F in (plaintexts.npy features.npy full_targets.npy cache_metadata.json) do if not exist "%RESULTS_DIR%\data\%%F" goto incomplete_results
for %%F in (selected8_global_head_rescnn_anchor_true_output_final.pt selected8_position_head_rescnn_no_p_true_output_final.pt selected8_position_head_spn_rescnn_exact_p_true_output_final.pt selected8_position_head_spn_rescnn_wrong_p_true_output_final.pt selected8_position_head_spn_rescnn_exact_p_label_shuffle_final.pt) do if not exist "%RESULTS_DIR%\models\%%F" goto incomplete_results
set RESULT_LINES=0
for /f "tokens=3" %%L in ('find /c /v "" "%RESULTS_DIR%\results.jsonl"') do set RESULT_LINES=%%L
echo result_lines=%RESULT_LINES% > "%LOG_DIR%\%RUN_ID%_result_gate.txt"
echo expected_rows=40 >> "%LOG_DIR%\%RUN_ID%_result_gate.txt"
if not "%RESULT_LINES%"=="40" goto incomplete_results
"%PY%" -c "import csv,hashlib,json,pathlib; root=pathlib.Path(r'%RESULTS_DIR%'); gate=json.loads((root/'gate.json').read_text(encoding='utf-8')); meta=json.loads((root/'metadata.json').read_text(encoding='utf-8')); cache=json.loads((root/'data'/'cache_metadata.json').read_text(encoding='utf-8')); checkpoints=json.loads((root/'checkpoint_manifest.json').read_text(encoding='utf-8')); history=list(csv.DictReader((root/'history.csv').open(encoding='utf-8'))); opc1=root/'opc1_gate.json'; opn1=root/'opn1_gate.json'; assert hashlib.sha256(opc1.read_bytes()).hexdigest()=='ebb86a9feab6d2d9993937f5c0a7f4afe1bfe3597c8c1dff083956381e0310b4' and hashlib.sha256(opn1.read_bytes()).hexdigest()=='887a7db3643e73bdda67958bcaae470881a09db25ab0ba5ff6c3d6bb0a2503d7' and gate['status'] in {'pass','hold'} and all(gate['protocol_checks'].values()) and all(gate['execution_checks'].values()) and meta['sample_classification'] is False and meta['config']['seed']==7 and meta['config']['mode']=='position_bound_head' and cache['status']=='complete' and cache['completed_rows']==196608 and len(history)==500 and len(checkpoints)==5 and all(row['sha256'] for row in checkpoints)" || goto incomplete_results

if not exist "%ARCHIVE_DIR%" mkdir "%ARCHIVE_DIR%" || goto failed
for %%F in (results.jsonl progress.jsonl history.csv metadata.json opc1_gate.json opn1_gate.json summary.json gate.json checkpoint_manifest.json) do copy /Y "%RESULTS_DIR%\%%F" "%ARCHIVE_DIR%\%%F" > nul || goto failed
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
"%PY%" -c "import hashlib,pathlib; root=pathlib.Path(r'%ARCHIVE_DIR%'); files=sorted(p for p in root.rglob('*') if p.is_file() and not p.name == 'SHA256SUMS'); (root/'SHA256SUMS').write_text('\n'.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.relative_to(root).as_posix() for p in files)+'\n',encoding='utf-8')" || goto failed

git config user.name "remote-experiment"
git config user.email "remote-experiment@local.invalid"
git checkout -B results/%RUN_ID% > "%LOG_DIR%\%RUN_ID%_result_branch_checkout.txt" 2>&1 || goto failed
git add "results_archive\%ARCHIVE_NAME%" || goto failed
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

:readiness_failed
echo readiness_failed>"%LOG_DIR%\%RUN_ID%_failed.marker"
echo Frozen OPC1 or OPN1 readiness authority did not match the remote package.>"%LOG_DIR%\%RUN_ID%_failure_reason.txt"
exit /b 8

:already_complete
exit /b 7

:failed
echo failed>"%LOG_DIR%\%RUN_ID%_failed.marker"
exit /b 1
