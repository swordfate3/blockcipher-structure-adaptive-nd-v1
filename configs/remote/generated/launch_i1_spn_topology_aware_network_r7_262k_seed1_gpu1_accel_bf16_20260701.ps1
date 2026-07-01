$root = 'G:\lxy\blockcipher-structure-adaptive-nd-runs'
$run = 'i1_spn_topology_aware_network_r7_262k_seed1_gpu1_accel_bf16_20260701'
$script = Join-Path $root ('run_' + $run + '.cmd')
$logdir = Join-Path $root 'launcher_logs'
New-Item -ItemType Directory -Force -Path $logdir | Out-Null
$out = Join-Path $logdir ($run + '_launcher_stdout.txt')
$err = Join-Path $logdir ($run + '_launcher_stderr.txt')
$arg = '/c "' + $script + '" > "' + $out + '" 2> "' + $err + '"'
Start-Process -FilePath 'cmd.exe' -ArgumentList $arg -WindowStyle Hidden
Write-Output ('STARTED ' + $script)
