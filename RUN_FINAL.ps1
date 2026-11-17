$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"

function Run-Step {
    param([string]$Name, [string[]]$Command)
    Write-Host "`n=== $Name ===" -ForegroundColor Cyan
    & $Command[0] $Command[1..($Command.Length-1)]
    if ($LASTEXITCODE -ne 0) { throw "$Name FAILED with exit code $LASTEXITCODE" }
}

Run-Step "Install requirements" @("python", "-m", "pip", "install", "-r", "requirements.txt")
Run-Step "Regression tests" @("python", "-m", "pytest", "-q")
Run-Step "Profiler" @("python", "-m", "src.performance", "profile")
Run-Step "Complexity" @("python", "-m", "src.performance", "complexity")
Run-Step "Benchmark" @("python", "-m", "src.performance", "benchmark")
Run-Step "Stability" @("python", "-m", "src.performance", "stability-check")
Run-Step "Parallel" @("python", "-m", "src.simulation", "--workers", "2")
Run-Step "Optimized full simulation" @("python", "-m", "src.simulation")
Run-Step "Analyze" @("python", "-m", "src.analyze")
Run-Step "Figures" @("python", "-m", "src.figures")
Run-Step "Full Unit 2 baseline timing" @("python", "-m", "src.baseline_timing")
Run-Step "Update report and check submission" @("python", "-m", "src.finalize_submission")
Write-Host "`nALL CHECKS PASSED. You may zip this folder for submission." -ForegroundColor Green
