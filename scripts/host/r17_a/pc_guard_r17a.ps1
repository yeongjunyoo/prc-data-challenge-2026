# SPDX-License-Identifier: GPL-3.0-only
# R17-A PC watchdog for the PC download side (coordinator 10-08 22:3x: "stop the PC side at once if commit goes above 85% or free memory
# drops below 4 GB"). Every 20 s it reads commit % and free GB. On the first failure it stops the PC side at once and for good:
#   touches $env:ADSB_PC_WORK\STOP, kills only this lane's processes (bash running pc_dl_r17a.sh, python running adsb_day_r17a.py incl. its memcap
#   wrapper, and their curl children), releases the Mac claims of the interrupted days (ssh rmdir ${ADSB_WORK:-.}/claim/<day>) so the Mac runner
#   takes them, and moves their partial outputs to failed\<day>.guardstop. It exits after acting, or when the runner is gone.
#     powershell -NoProfile -ExecutionPolicy Bypass -File pc_guard_r17a.ps1      (started hidden with Win32_Process Create)
$wd = $env:ADSB_PC_WORK  # PC ADS-B work folder
$log = Join-Path $wd 'guard.log'
function Say($m) { "$(Get-Date -Format 'MM-dd HH:mm:ss') $m" | Out-File -Append -Encoding utf8 $log }
Say "watchdog start pid $PID"
while ($true) {
  $all = Get-CimInstance Win32_Process
  $runner = @($all | Where-Object { $_.CommandLine -like '*pc_dl_r17a.sh*' })
  if ($runner.Count -eq 0) { Say 'runner gone, watchdog exits'; break }
  $o = Get-CimInstance Win32_OperatingSystem
  $c = (Get-Counter '\Memory\% Committed Bytes In Use').CounterSamples[0].CookedValue
  $f = $o.FreePhysicalMemory / 1MB
  if ($c -ge 85 -or $f -lt 4) {
    Say ("guard failed: commit {0:N1}%, free {1:N2} GB -> stopping the PC side" -f $c, $f)
    New-Item -ItemType File -Force (Join-Path $wd 'STOP') | Out-Null
    $py = @($all | Where-Object { $_.CommandLine -like '*adsb_day_r17a.py*' })
    $days = @($py | ForEach-Object { if ($_.CommandLine -match '--day (\d{4}-\d{2}-\d{2})') { $Matches[1] } } | Sort-Object -Unique)
    $pyIds = @($py | ForEach-Object { $_.ProcessId })
    $curl = @($all | Where-Object { $_.Name -like 'curl*' -and $pyIds -contains $_.ParentProcessId })
    foreach ($p in ($runner + $py + $curl)) {
      try { Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop; Say "killed $($p.ProcessId) $($p.Name)" } catch { Say "kill $($p.ProcessId) failed: $_" }
    }
    Start-Sleep 3
    foreach ($d in $days) {
      & 'C:\Program Files\Git\usr\bin\ssh.exe' -n -o BatchMode=yes -o ConnectTimeout=20 $env:MAC_HOST "test -s `$ADSB_WORK/out/$d/meta.json || rmdir `$ADSB_WORK/claim/$d" 2>&1 | Out-Null
      $od = Join-Path $wd "out\$d"
      if (Test-Path $od) { Move-Item $od (Join-Path $wd "failed\$d.guardstop") -Force }
      Say "released day $d"
    }
    break
  }
  Start-Sleep 20
}
Say 'watchdog end'
