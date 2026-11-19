param([string]$PythonPath = "", [string]$OutputRoot = "")
$ErrorActionPreference = "Stop"
$project = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$parent = (Resolve-Path (Join-Path $project "..")).Path
if (-not $PythonPath) { $PythonPath = Join-Path $parent ".venv\Scripts\python.exe" }
if (-not $OutputRoot) { $OutputRoot = Join-Path $parent "runs" }
if (-not (Test-Path $PythonPath)) { throw "Python venv absent: $PythonPath" }
$prior = Join-Path $OutputRoot "certified_v030"
$quotes = Join-Path $prior "normalized.csv"
$exact = Join-Path $OutputRoot "exact_v040"
$universal = Join-Path $OutputRoot "universal_v040"
$argsExact = @("-m", "dupire_identify", "saturate")
if (Test-Path $quotes) {
  $argsExact += @("--quotes", $quotes, "--initial-run", $prior)
} else {
  $quotes = Join-Path $project "examples\seed_v020\normalized.csv"
  $argsExact += @("--quotes", $quotes)
}
$argsExact += @("--out", $exact)
if (Test-Path (Join-Path $exact "checkpoint.json")) { $argsExact += "--resume" }
Write-Host "[1/4] Constructing the exact endpoint certificate"
& $PythonPath @argsExact
if ($LASTEXITCODE -ne 0) { throw "saturate failed" }
Write-Host "[2/4] Recomputing the endpoint proof"
& $PythonPath -m dupire_identify verify-exact --run $exact
if ($LASTEXITCODE -ne 0) { throw "verify-exact failed" }
if (-not (Test-Path (Join-Path $universal "universal_manifest.json"))) {
  Write-Host "[3/4] Covering the continuum of windows"
  & $PythonPath -m dupire_identify universal-frontier --certificate $exact --domain (Join-Path $project "examples\universal_domain.json") --out $universal
  if ($LASTEXITCODE -ne 0) { throw "universal-frontier failed" }
}
Write-Host "[4/4] Recomputing the continuum proof"
& $PythonPath -m dupire_identify verify-universal --run $universal
if ($LASTEXITCODE -ne 0) { throw "verify-universal failed" }
Write-Host "Reports: $exact\report.html ; $universal\frontier.html"
