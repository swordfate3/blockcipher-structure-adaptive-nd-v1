@echo off
setlocal
set TASK_NAME=i2_e104_present_r9_split333_20260720
set PIPELINE=G:\lxy\blockcipher-structure-adaptive-nd-v1-clean\scripts\generated\remote\run_innovation2_present_r9_atm_split333_pipeline_20260720.cmd
schtasks /Create /TN %TASK_NAME% /SC ONCE /ST 23:59 /TR "cmd.exe /c %PIPELINE%" /RU SYSTEM /F
if errorlevel 1 exit /b 1
schtasks /Run /TN %TASK_NAME%
if errorlevel 1 exit /b 1
schtasks /Query /TN %TASK_NAME% /V /FO LIST
exit /b 0
