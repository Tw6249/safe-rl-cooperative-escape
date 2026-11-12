param(
    [switch]$Figures
)

$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot
try {

if ($Figures) {
    Write-Host "==> Generating Chinese and English publication figures..."
    python make_journal_figures.py
}

Write-Host "==> Compiling Chinese Top Journal Edition (main_zh.tex -> main_zh.pdf)..."
latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=build_zh main_zh.tex
Copy-Item -LiteralPath "build_zh/main_zh.pdf" -Destination "main_zh.pdf" -Force
Copy-Item -LiteralPath "build_zh/main_zh.pdf" -Destination "main.pdf" -Force

Write-Host "==> Compiling English IEEE Transactions Edition (main_en.tex -> main_en.pdf)..."
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build_en main_en.tex
Copy-Item -LiteralPath "build_en/main_en.pdf" -Destination "main_en.pdf" -Force

$zhOverfull = @(Select-String -Path "build_zh/main_zh.log" -Pattern "Overfull \\hbox").Count
$zhWarn     = @(Select-String -Path "build_zh/main_zh.log" -Pattern "LaTeX Warning:|Package .* Warning:").Count
$enOverfull = @(Select-String -Path "build_en/main_en.log" -Pattern "Overfull \\hbox").Count
$enWarn     = @(Select-String -Path "build_en/main_en.log" -Pattern "LaTeX Warning:|Package .* Warning:").Count

Write-Host "============================================================"
Write-Host ("ZH Journal PDF : main_zh.pdf (Overfull \hbox = {0}, Warnings = {1})" -f $zhOverfull, $zhWarn)
Write-Host ("EN IEEE PDF    : main_en.pdf (Overfull \hbox = {0}, Warnings = {1})" -f $enOverfull, $enWarn)
Write-Host "============================================================"

} finally {
    Pop-Location
}
