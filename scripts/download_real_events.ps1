# Detached real-event acquisition (ERA5 window rainfall + Sentinel-1 RTC time series) for both Assam replays.
Set-Location $PSScriptRoot\..
foreach ($e in @("assam2016", "assam2022")) {
    & .\.venv\Scripts\python.exe -u -m floodtask.data.real_event --event $e --step rain *>&1 | Out-File -Append -Encoding utf8 results\logs\real_event.log
    & .\.venv\Scripts\python.exe -u -m floodtask.data.real_event --event $e --step download *>&1 | Out-File -Append -Encoding utf8 results\logs\real_event.log
}
"done" | Out-File results\logs\real_download_done.flag
