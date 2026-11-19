param(
    [string]$Python = "",
    [ValidateSet("cuda", "auto", "numpy", "cpp")][string]$Backend = "cuda",
    [switch]$Native,
    [switch]$SkipDoctor
)
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
if (-not $Python) {
    $candidates = @(
        (Join-Path $projectRoot ".venv\Scripts\python.exe"),
        (Join-Path (Split-Path -Parent $projectRoot) ".venv\Scripts\python.exe")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            $Python = $candidate
            break
        }
    }
}
if (-not $Python -or -not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Specify the existing venv: -Python .\.venv\Scripts\python.exe"
}
$taskPython = (Resolve-Path -LiteralPath $Python).Path
$builder = Join-Path $PSScriptRoot "build_install.py"
$builderArguments = @($builder, "--backend", $Backend)
if ($Native) { $builderArguments += "--native" }
if ($SkipDoctor) { $builderArguments += "--skip-doctor" }
& $taskPython @builderArguments
if ($LASTEXITCODE -ne 0) {
    throw "Build/install/doctor failed (exit $LASTEXITCODE)."
}
