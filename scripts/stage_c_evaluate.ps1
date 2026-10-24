# STAGE C: frozen policies -> test experiments, Assam real-event replays, analysis, manifest.
# Run after Stage B:  powershell -ExecutionPolicy Bypass -File scripts\stage_c_evaluate.ps1
Set-Location $PSScriptRoot\..
$py = ".\.venv\Scripts\python.exe"
$log = "results\logs\stage_c.log"
function Step($name, $argsList) {
    "=== $name $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 $log
    & $py -u @argsList *>&1 | Out-File -Append -Encoding utf8 $log
}
Step "unit tests"            @("-m", "pytest", "-q")
Step "test experiments"      @("scripts\run_experiments.py", "--suite", "all")
foreach ($e in @("assam2016", "assam2022")) {
    Step "predict $e"        @("-m", "floodtask.data.real_event", "--event", $e, "--step", "predict")
    Step "replay $e"         @("scripts\run_real_event.py", "--event", $e)
    Step "export RL $e"      @("scripts\export_maps.py", "--event", $e, "--policy", "ppo:recurrent:best")
    Step "export greedy $e"  @("scripts\export_maps.py", "--event", $e, "--policy", "greedy_unc")
}
Step "analysis"              @("scripts\analyze.py")
Step "manifest"              @("scripts\make_manifest.py")
"done" | Out-File results\logs\stage_c_done.flag
