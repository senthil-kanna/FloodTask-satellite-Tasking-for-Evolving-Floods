# Waits for the current Stage B pass to finish, then runs Stage B once more (finished runs are skipped,
# failed/missing runs are retrained). Used after the recurrent_iou KeyError fix.
Set-Location $PSScriptRoot\..
while (-not (Test-Path results\logs\stage_b_done.flag)) { Start-Sleep 60 }
Remove-Item results\logs\stage_b_done.flag
"=== retry pass $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 results\logs\train_all.log
& powershell -NoProfile -ExecutionPolicy Bypass -File scripts\stage_b_train.ps1
