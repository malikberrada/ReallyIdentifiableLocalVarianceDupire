param(
    [string]$Python = "",
    [ValidateSet("auto", "cuda", "cpp", "numpy")][string]$Backend = "auto",
    [int]$Threads = 18,
    [string]$Out = "",
    [switch]$Resume
)
$ErrorActionPreference = "Stop"
$Source = Split-Path -Parent $PSScriptRoot
$ProjectParent = Split-Path -Parent $Source
if (-not $Python) { $Python = Join-Path $ProjectParent ".venv\Scripts\python.exe" }
if (-not $Out) { $Out = Join-Path $ProjectParent "runs\targeted_v051" }
$Arguments = @(
    "-u", "-m", "dupire_identify", "target-general",
    "--quotes", (Join-Path $Source "examples\exact_initial\normalized.csv"),
    "--model-run", (Join-Path $Source "examples\exact_initial"),
    "--targets", (Join-Path $Source "examples\target_domains.json"),
    "--backend", $Backend, "--threads", "$Threads", "--out", $Out
)
if ($Resume) { $Arguments += "--resume" }
& $Python @Arguments
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python -u -m dupire_identify verify-targets --run $Out
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host ("Report: " + (Join-Path $Out "report.html"))
