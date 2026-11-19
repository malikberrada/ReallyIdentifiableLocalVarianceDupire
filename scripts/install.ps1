param(
    [ValidateSet("cpu", "cuda12", "cuda13")][string]$Backend = "cuda13"
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    py -3.12 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "Python 3.12 requis." }
}
$taskPython = Join-Path (Get-Location) ".venv\Scripts\python.exe"
& $taskPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Mise a jour pip echouee." }
$extras = if ($Backend -eq "cpu") { ".[yahoo]" } else { ".[yahoo,$Backend]" }
& $taskPython -m pip install $extras
if ($LASTEXITCODE -ne 0) { throw "Installation echouee." }
$engine = if ($Backend -eq "cpu") { "numpy" } else { "cuda" }
& $taskPython -m dupire_identify doctor --backend $engine
if ($LASTEXITCODE -ne 0) { throw "Verification du backend echouee." }
Write-Output "Installation terminee. Utiliser .\.venv\Scripts\python.exe -m dupire_identify demo --backend $engine"
