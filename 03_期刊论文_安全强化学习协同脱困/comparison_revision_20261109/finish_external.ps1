param([Parameter(Mandatory=$true)][int]$SeedIndex)
$ErrorActionPreference = "Stop"
$revisionRoot = $PSScriptRoot
$pythonBase = 'D:/miniconda/python.exe'
$pythonOfficial = Join-Path $revisionRoot '.venv_dgppo/Scripts/python.exe'
$trainingRoot = Join-Path $revisionRoot "official_dgppo_h500_margin_s$SeedIndex"
$outputRoot = Join-Path $revisionRoot "external_dgppo_s$SeedIndex"
while (-not (Test-Path -LiteralPath (Join-Path $trainingRoot 'done.json'))) {
    Start-Sleep -Seconds 5
}
& $pythonOfficial -B -X utf8 (Join-Path $revisionRoot 'evaluate_official_dgppo.py') --training $trainingRoot --output $outputRoot
if ($LASTEXITCODE -ne 0) { throw "DGPPO inference failed: $LASTEXITCODE" }
& $pythonBase -B -X utf8 (Join-Path $revisionRoot 'evaluate_external_common.py') --audit-only --output $outputRoot
if ($LASTEXITCODE -ne 0) { throw "DGPPO independent audit failed: $LASTEXITCODE" }
