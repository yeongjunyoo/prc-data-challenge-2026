# SPDX-License-Identifier: GPL-3.0-only
# R17-A: start one bash command line in research/r17_a as a hidden process outside the calling process tree (copy of research/r16_b/detach.ps1
# with this lane's folder).
#     powershell -NoProfile -ExecutionPolicy Bypass -File research/r17_a/detach.ps1 "bash pc_dl_r17a.sh ..."
param([Parameter(Mandatory = $true)][string]$c)
$dir = Join-Path $env:PC_REPO 'research\r17_a'
$bash = 'C:\Program Files\Git\bin\bash.exe'
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0 }
$cmd = '"' + $bash + '" -c "cd $PC_REPO/research/r17_a && ' + $c + '"'
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmd; CurrentDirectory = $dir; ProcessStartupInformation = $si }
"Win32_Process Create: return $($r.ReturnValue), pid $($r.ProcessId)"
