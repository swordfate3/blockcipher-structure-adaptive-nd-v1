@echo off
setlocal EnableExtensions

set PHYSICAL_GPU=%~1
if "%PHYSICAL_GPU%"=="" goto invalid_arguments

set RUN_ID=i2_output_prediction_opf2_present_r4_position_bound_spn_rescnn_2p20_key7_gpu0_20260722
set REMOTE_DIR=i2_opf2_r4_poshead_2p20_k7_20260722
set ARCHIVE_NAME=i2_opf2_r4_poshead_2p20_k7_20260722
set RUNS_ROOT=G:\lxy\blockcipher-structure-adaptive-nd-runs
set RUN_ROOT=%RUNS_ROOT%\%REMOTE_DIR%
set SOURCE_ROOT=%RUN_ROOT%\source
set LOG_DIR=%RUN_ROOT%\logs
set RESULTS_DIR=%RUN_ROOT%\results
set ARCHIVE_DIR=%SOURCE_ROOT%\results_archive\%ARCHIVE_NAME%
set OPC1_GATE=%RUNS_ROOT%\i2_opc1_hybrid_k6_20260722\results\gate.json
set OPN1_GATE=%SOURCE_ROOT%\configs\experiment\innovation2\authorities\innovation2_output_prediction_opn1_gate_20260722.json
set OPD1_GATE=%RUNS_ROOT%\i2_opd1_poshead_k7_retry1_20260722\results\gate.json
set OPF1_GATE=%RUNS_ROOT%\i2_opf1_r4_poshead_k7_20260722\results\gate.json
set PLAN=configs\experiment\innovation2\innovation2_output_prediction_opf2_present_r4_position_bound_spn_rescnn_2p20_key7.json
set DOC_PLAN=docs\experiments\innovation2-output-prediction-opf2-present-r4-2p20-scale-adjudication-plan.md
set REMOTE_CONFIG=configs\remote\innovation2_output_prediction_opf2_present_r4_position_bound_spn_rescnn_2p20_key7_gpu0_20260722.json
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
"%PY%" -c "import hashlib,json,pathlib; cfg=json.loads(pathlib.Path(r'%REMOTE_CONFIG%').read_text(encoding='utf-8')); plan=json.loads(pathlib.Path(r'%PLAN%').read_text(encoding='utf-8')); paths=[pathlib.Path(r'%OPC1_GATE%'),pathlib.Path(r'%OPN1_GATE%'),pathlib.Path(r'%OPD1_GATE%'),pathlib.Path(r'%OPF1_GATE%')]; gates=[json.loads(p.read_text(encoding='utf-8')) for p in paths]; hashes=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]; assert hashes==[cfg['opc1_gate_sha256'],cfg['opn1_gate_sha256'],cfg['opd1_gate_sha256'],cfg['opf1_gate_sha256']]; assert gates[0]['status']=='hold' and gates[1]['status']=='pass' and gates[2]['status']=='hold' and gates[3]['status']=='hold'; assert gates[3]['decision']=='innovation2_position_bound_r4_boundary_observed'; assert all(all(g[group].values()) for g in gates for group in ('protocol_checks','execution_checks')); common=plan['common']; assert cfg['rounds']==4 and common['rounds']==4 and plan['only_changed_variable']=={'field':'train_total_rows','opf1_value':131072,'opf2_value':1048576} and cfg['seed']==7 and cfg['train_total_rows']==1048576 and cfg['test_total_rows']==65536 and cfg['epochs']==100 and cfg['sample_classification'] is False and common['sample_classification'] is False and len(plan['rows'])==5; print('status=pass')" > "%LOG_DIR%\%RUN_ID%_readiness.txt" 2> "%LOG_DIR%\%RUN_ID%_readiness_stderr.txt" || goto readiness_failed

echo started>"%LOG_DIR%\%RUN_ID%_started.marker"
"%PY%" scripts\run-innovation2-selected-output-position-bound-spn-rescnn ^
  --mode scale_extension ^
  --opc1-gate "%OPC1_GATE%" ^
  --opn1-gate "%OPN1_GATE%" ^
  --opd1-gate "%OPD1_GATE%" ^
  --opf1-gate "%OPF1_GATE%" ^
  --run-id "%RUN_ID%" ^
  --device cuda ^
  --output-root "%RESULTS_DIR%" ^
  > "%LOG_DIR%\%RUN_ID%_stdout.txt" 2> "%LOG_DIR%\%RUN_ID%_stderr.txt"
if errorlevel 1 goto failed

echo deferred_to_local_verified_retrieval>"%LOG_DIR%\%RUN_ID%_plot_deferred.marker"
for %%F in (results.jsonl progress.jsonl history.csv metadata.json opc1_gate.json opn1_gate.json opd1_gate.json opf1_gate.json summary.json gate.json checkpoint_manifest.json) do if not exist "%RESULTS_DIR%\%%F" goto incomplete_results
for %%F in (plaintexts.npy features.npy full_targets.npy cache_metadata.json) do if not exist "%RESULTS_DIR%\data\%%F" goto incomplete_results
set RESULT_LINES=0
for /f "tokens=3" %%L in ('find /c /v "" "%RESULTS_DIR%\results.jsonl"') do set RESULT_LINES=%%L
echo result_lines=%RESULT_LINES% > "%LOG_DIR%\%RUN_ID%_result_gate.txt"
echo expected_rows=40 >> "%LOG_DIR%\%RUN_ID%_result_gate.txt"
if not "%RESULT_LINES%"=="40" goto incomplete_results
"%PY%" -c "import csv,hashlib,json,pathlib,numpy as np; root=pathlib.Path(r'%RESULTS_DIR%'); gate=json.loads((root/'gate.json').read_text(encoding='utf-8')); meta=json.loads((root/'metadata.json').read_text(encoding='utf-8')); cache=json.loads((root/'data'/'cache_metadata.json').read_text(encoding='utf-8')); checkpoints=json.loads((root/'checkpoint_manifest.json').read_text(encoding='utf-8')); history=list(csv.DictReader((root/'history.csv').open(encoding='utf-8'))); source=[root/'opc1_gate.json',root/'opn1_gate.json',root/'opd1_gate.json',root/'opf1_gate.json']; expected=['ebb86a9feab6d2d9993937f5c0a7f4afe1bfe3597c8c1dff083956381e0310b4','887a7db3643e73bdda67958bcaae470881a09db25ab0ba5ff6c3d6bb0a2503d7','3d63163ab94e95b6c8c859be0867cc0a6b1f91382bd842e32dd3adbe04863579','dad638c7180682074f134233e807ed0b58bb40d7d671f7a5d01540721e399354']; p=np.load(root/'data'/'plaintexts.npy',mmap_mode='r'); layout=cache['split_layout']; assert [hashlib.sha256(x.read_bytes()).hexdigest() for x in source]==expected and gate['status'] in {'pass','hold'} and all(gate['protocol_checks'].values()) and all(gate['execution_checks'].values()) and meta['sample_classification'] is False and meta['config']['seed']==7 and meta['config']['rounds']==4 and meta['config']['mode']=='scale_extension' and cache['status']=='complete' and cache['completed_rows']==1114112 and layout['train_index_segments']==[[0,131072],[196608,1114112]] and layout['test_index_segment']==[131072,196608] and hashlib.sha256(np.asarray(p[:131072]).tobytes()).hexdigest()=='eca0f5705c2d9a6b4f0475bfb90e55d2bfa2d5e4d7b8c380b10ab55778a4555a' and hashlib.sha256(np.asarray(p[131072:196608]).tobytes()).hexdigest()=='5c5410d4c0761f729f5f705d43a7392bf90f6ae0bee65a57321760d515b82fec' and len(history)==500 and len(checkpoints)==5 and all(row['sha256'] for row in checkpoints)" || goto incomplete_results

if not exist "%ARCHIVE_DIR%" mkdir "%ARCHIVE_DIR%" || goto failed
for %%F in (results.jsonl progress.jsonl history.csv metadata.json opc1_gate.json opn1_gate.json opd1_gate.json opf1_gate.json summary.json gate.json checkpoint_manifest.json) do copy /Y "%RESULTS_DIR%\%%F" "%ARCHIVE_DIR%\%%F" > nul || goto failed
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
exit /b 6

:incomplete_results
echo incomplete_results>"%LOG_DIR%\%RUN_ID%_failed.marker"
exit /b 4

:invalid_arguments
exit /b 5

:readiness_failed
echo readiness_failed>"%LOG_DIR%\%RUN_ID%_failed.marker"
exit /b 8

:already_complete
exit /b 7

:failed
echo failed>"%LOG_DIR%\%RUN_ID%_failed.marker"
exit /b 1
