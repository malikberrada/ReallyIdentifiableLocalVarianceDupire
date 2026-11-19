param(
    [string]$Python = '.\.venv\Scripts\python.exe',
    [string]$Quotes = '.\runs\exact_v040\normalized.csv',
    [string]$ModelRun = '.\runs\exact_v040',
    [string]$Output = '.\runs\general_v050',
    [int]$Threads = 18,
    [int]$MaxLeaves = 8,
    [switch]$Resume
)
$ErrorActionPreference = 'Stop'
$Project = Split-Path -Parent $PSScriptRoot
$Arguments = @('-m','dupire_identify','solve-general','--quotes',$Quotes,
    '--domain',(Join-Path $Project 'examples\general_domain.json'),
    '--threads',"$Threads",'--max-leaves',"$MaxLeaves",'--out',$Output)
if ($ModelRun) {
    $Arguments += @('--model-run',$ModelRun,'--fit-models','0')
} else {
    $Arguments += @('--fit-models','3')
}
if ($Resume) { $Arguments += '--resume' }
& $Python @Arguments
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python -m dupire_identify verify-general --run $Output
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "Rapport : $(Join-Path $Output 'frontier.html')"
