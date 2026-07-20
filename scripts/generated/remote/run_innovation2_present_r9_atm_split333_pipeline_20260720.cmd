@echo off
setlocal
set RUN_ID=i2_present_r9_atm_split333_resumable_generation_20260720
set RUN_ROOT=G:\lxy\blockcipher-structure-adaptive-nd-runs\%RUN_ID%
set SOURCE=%RUN_ROOT%\source
set ATM_ROOT=%RUN_ROOT%\atm-source
set OUTPUT=%RUN_ROOT%\results
set LOGS=%RUN_ROOT%\logs
set PY=%RUN_ROOT%\venv\Scripts\python.exe
set PYTHONUTF8=1

call G:\lxy\blockcipher-structure-adaptive-nd-v1-clean\scripts\generated\remote\setup_innovation2_present_r9_atm_split333_20260720.cmd
if errorlevel 1 goto pipeline_failed

%PY% %SOURCE%\scripts\supervise-innovation2-atm-stage --timeout-seconds 600 --stage-id probe_001 --marker-root %LOGS% --stdout %LOGS%\probe_001_stdout.txt --stderr %LOGS%\probe_001_stderr.txt -- %PY% %SOURCE%\scripts\run-innovation2-present-r9-atm-split333-generation --mode probe --atm-root %ATM_ROOT% --e103-anchor %SOURCE%\configs\experiment\innovation2\innovation2_present_sbox4_r3_real_atm_compatibility_gate_anchor.json --output-root %OUTPUT%
if errorlevel 1 goto probe_failed

%PY% %SOURCE%\scripts\supervise-innovation2-atm-stage --timeout-seconds 600 --stage-id probe_002 --marker-root %LOGS% --stdout %LOGS%\probe_002_stdout.txt --stderr %LOGS%\probe_002_stderr.txt -- %PY% %SOURCE%\scripts\run-innovation2-present-r9-atm-split333-generation --mode probe --atm-root %ATM_ROOT% --e103-anchor %SOURCE%\configs\experiment\innovation2\innovation2_present_sbox4_r3_real_atm_compatibility_gate_anchor.json --output-root %OUTPUT%
if errorlevel 1 goto probe_failed
if not exist %OUTPUT%\probe_002_passed.marker goto probe_failed

:stage_001
%PY% %SOURCE%\scripts\supervise-innovation2-atm-stage --timeout-seconds 43200 --stage-id stage_001 --marker-root %LOGS% --stdout %LOGS%\stage_001_stdout.txt --stderr %LOGS%\stage_001_stderr.txt -- %PY% %SOURCE%\scripts\run-innovation2-present-r9-atm-split333-generation --mode search --atm-root %ATM_ROOT% --e103-anchor %SOURCE%\configs\experiment\innovation2\innovation2_present_sbox4_r3_real_atm_compatibility_gate_anchor.json --output-root %OUTPUT%
set STAGE_RC=%ERRORLEVEL%
if exist %OUTPUT%\generation_passed.marker goto pipeline_passed
if "%STAGE_RC%"=="124" goto stage_002
goto pipeline_failed

:stage_002
%PY% %SOURCE%\scripts\supervise-innovation2-atm-stage --timeout-seconds 43200 --stage-id stage_002 --marker-root %LOGS% --stdout %LOGS%\stage_002_stdout.txt --stderr %LOGS%\stage_002_stderr.txt -- %PY% %SOURCE%\scripts\run-innovation2-present-r9-atm-split333-generation --mode search --atm-root %ATM_ROOT% --e103-anchor %SOURCE%\configs\experiment\innovation2\innovation2_present_sbox4_r3_real_atm_compatibility_gate_anchor.json --output-root %OUTPUT%
set STAGE_RC=%ERRORLEVEL%
if exist %OUTPUT%\generation_passed.marker goto pipeline_passed
if "%STAGE_RC%"=="124" goto stage_003
goto pipeline_failed

:stage_003
%PY% %SOURCE%\scripts\supervise-innovation2-atm-stage --timeout-seconds 43200 --stage-id stage_003 --marker-root %LOGS% --stdout %LOGS%\stage_003_stdout.txt --stderr %LOGS%\stage_003_stderr.txt -- %PY% %SOURCE%\scripts\run-innovation2-present-r9-atm-split333-generation --mode search --atm-root %ATM_ROOT% --e103-anchor %SOURCE%\configs\experiment\innovation2\innovation2_present_sbox4_r3_real_atm_compatibility_gate_anchor.json --output-root %OUTPUT%
set STAGE_RC=%ERRORLEVEL%
if exist %OUTPUT%\generation_passed.marker goto pipeline_passed
if "%STAGE_RC%"=="124" goto resource_cap
goto pipeline_failed

:pipeline_passed
echo pipeline_passed> %LOGS%\pipeline_passed.marker
exit /b 0

:resource_cap
echo resource_cap_hit> %LOGS%\resource_cap_hit.marker
exit /b 2

:probe_failed
echo probe_failed> %LOGS%\probe_failed.marker
exit /b 3

:pipeline_failed
echo pipeline_failed> %LOGS%\pipeline_failed.marker
exit /b 1
