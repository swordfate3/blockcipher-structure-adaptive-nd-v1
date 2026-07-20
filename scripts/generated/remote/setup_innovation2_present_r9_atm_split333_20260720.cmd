@echo off
setlocal
set HTTP_PROXY=
set HTTPS_PROXY=
set http_proxy=
set https_proxy=
set RUN_ID=i2_present_r9_atm_split333_resumable_generation_20260720
set RUN_ROOT=G:\lxy\blockcipher-structure-adaptive-nd-runs\%RUN_ID%
set SOURCE=%RUN_ROOT%\source
set ATM_ROOT=%RUN_ROOT%\atm-source
set VENV=%RUN_ROOT%\venv
set OUTPUT=%RUN_ROOT%\results
set LOGS=%RUN_ROOT%\logs
set PIP_CACHE_DIR=%RUN_ROOT%\pip-cache
set PYTHONUTF8=1
set BASE_PY=F:\Anaconda\envs\DWT\torch310\python.exe
set PY=%VENV%\Scripts\python.exe
set PYTHONPATH=%SOURCE%\src
set REPO=git@github.com:swordfate3/blockcipher-structure-adaptive-nd-v1.git
set ATM_REPO=https://github.com/michielverbauwhede/AlgebraicTransitionMatrices.git
set ATM_COMMIT=b2ffbb2bf0ef8f2ffabe3203896006874aa1c40b
set GITHUB_SSH_KEY=C:/Users/1304Lijinlin/.ssh/github_blockcipher_20260612_result_pusher_ed25519
set GIT_SSH_COMMAND=ssh -i %GITHUB_SSH_KEY% -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new

if not exist G:\lxy mkdir G:\lxy
if not exist G:\lxy\blockcipher-structure-adaptive-nd-runs mkdir G:\lxy\blockcipher-structure-adaptive-nd-runs
if not exist %RUN_ROOT% mkdir %RUN_ROOT%
if not exist %LOGS% mkdir %LOGS%
if not exist %OUTPUT% mkdir %OUTPUT%
if exist %LOGS%\setup_passed.marker del %LOGS%\setup_passed.marker
if exist %LOGS%\setup_failed.marker del %LOGS%\setup_failed.marker
if exist %LOGS%\source_dirty.marker del %LOGS%\source_dirty.marker
if exist %LOGS%\atm_dirty.marker del %LOGS%\atm_dirty.marker
if exist %LOGS%\readiness_started.marker del %LOGS%\readiness_started.marker
if exist %LOGS%\readiness_done.marker del %LOGS%\readiness_done.marker
if exist %LOGS%\readiness_timeout.marker del %LOGS%\readiness_timeout.marker
if exist %LOGS%\readiness_failed.marker del %LOGS%\readiness_failed.marker

if not exist %SOURCE%\.git git clone %REPO% %SOURCE%
if errorlevel 1 goto setup_failed
git config --global --add safe.directory %SOURCE%
git -C %SOURCE% status --short --branch > %LOGS%\source_status_before_sync.txt
for /f "delims=" %%A in ('git -C %SOURCE% status --porcelain') do goto source_dirty
git -C %SOURCE% fetch origin
if errorlevel 1 goto setup_failed
git -C %SOURCE% checkout main
if errorlevel 1 goto setup_failed
git -C %SOURCE% pull --ff-only origin main
if errorlevel 1 goto setup_failed
git -C %SOURCE% rev-parse HEAD > %LOGS%\source_revision.txt
git -C %SOURCE% status --short --branch > %LOGS%\source_status_after_sync.txt

if not exist %ATM_ROOT%\.git git clone %ATM_REPO% %ATM_ROOT%
if errorlevel 1 goto setup_failed
git config --global --add safe.directory %ATM_ROOT%
echo bitarrays/bitset*.pyd>> %ATM_ROOT%\.git\info\exclude
echo bitarrays/.build/>> %ATM_ROOT%\.git\info\exclude
for /f "delims=" %%A in ('git -C %ATM_ROOT% status --porcelain') do goto atm_dirty
git -C %ATM_ROOT% cat-file -e %ATM_COMMIT%
if errorlevel 1 git -C %ATM_ROOT% fetch origin
if errorlevel 1 goto setup_failed
git -C %ATM_ROOT% checkout %ATM_COMMIT%
if errorlevel 1 goto setup_failed
git -C %ATM_ROOT% rev-parse HEAD > %LOGS%\atm_revision.txt
git -C %ATM_ROOT% status --short --branch > %LOGS%\atm_status.txt

if not exist %PY% %BASE_PY% -m venv --system-site-packages %VENV%
if errorlevel 1 goto setup_failed
%PY% -c "import torch; print(torch.__version__)" > %LOGS%\venv_torch_smoke.txt 2>&1
if errorlevel 1 rmdir /s /q %VENV%
if not exist %PY% %BASE_PY% -m venv --system-site-packages %VENV%
if errorlevel 1 goto setup_failed
%PY% -m pip install --disable-pip-version-check --cache-dir %PIP_CACHE_DIR% --requirement %SOURCE%\configs\runtime\innovation2_atm_windows_py310_requirements.txt > %LOGS%\pip_install_stdout.txt 2> %LOGS%\pip_install_stderr.txt
if errorlevel 1 goto setup_failed
%PY% -m pip install --disable-pip-version-check --no-deps --editable %SOURCE% > %LOGS%\project_install_stdout.txt 2> %LOGS%\project_install_stderr.txt
if errorlevel 1 goto setup_failed
%PY% -m pip freeze > %LOGS%\pip_freeze.txt
%PY% --version > %LOGS%\python_version.txt 2>&1
cl.exe > %LOGS%\compiler_version.txt 2>&1

%PY% %SOURCE%\scripts\supervise-innovation2-atm-stage --timeout-seconds 600 --stage-id readiness --marker-root %LOGS% --stdout %LOGS%\readiness_stdout.txt --stderr %LOGS%\readiness_stderr.txt -- %PY% %SOURCE%\scripts\run-innovation2-present-r9-atm-split333-generation --mode readiness --atm-root %ATM_ROOT% --e103-anchor %SOURCE%\configs\experiment\innovation2\innovation2_present_sbox4_r3_real_atm_compatibility_gate_anchor.json --output-root %OUTPUT%
if errorlevel 1 goto setup_failed
echo setup_passed> %LOGS%\setup_passed.marker
exit /b 0

:setup_failed
echo setup_failed> %LOGS%\setup_failed.marker
exit /b 1

:source_dirty
echo source_clone_dirty> %LOGS%\source_dirty.marker
exit /b 5

:atm_dirty
echo atm_clone_dirty> %LOGS%\atm_dirty.marker
exit /b 6
