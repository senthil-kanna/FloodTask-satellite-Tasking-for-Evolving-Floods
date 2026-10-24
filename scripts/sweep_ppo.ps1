# Validation hyperparameter sweep for PPO (feed-forward, 400k steps), 2 GPU runs at a time.
# Outputs: results/sweep/<tag>/ ; logs results/logs/sweep_<tag>.log ; chosen config by scripts/pick_sweep.py
Set-Location $PSScriptRoot\..
$runs = @(
  @("base",         "gamma=0.99"),
  @("g90",          "gamma=0.9"),
  @("g90_lr1e3",    "gamma=0.9 lr=0.001"),
  @("g90_ent003",   "gamma=0.9 ent_coef=0.003"),
  @("g90_lr1e3_e8", "gamma=0.9 lr=0.001 ent_coef=0.003 epochs=8")
)
$running = @()
foreach ($r in $runs) {
  while (($running | Where-Object { -not $_.HasExited }).Count -ge 2) { Start-Sleep 10 }
  $tag = $r[0]; $sets = $r[1].Split(" ")
  $running += Start-Process -PassThru -WindowStyle Hidden -FilePath .\.venv\Scripts\python.exe `
    -ArgumentList (@("-u","-m","floodtask.rl.train_ppo","--seed","0","--variant","feedforward","--steps","400000","--tag",$tag,"--set") + $sets) `
    -RedirectStandardOutput "results\logs\sweep_$tag.log" -RedirectStandardError "results\logs\sweep_$tag.err"
}
$running | Wait-Process
"done" | Out-File results\logs\sweep_done.flag
