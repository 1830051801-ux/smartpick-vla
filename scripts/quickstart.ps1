param(
    [string]$Python = ".\.venv\Scripts\python.exe"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-Python {
    & $Python @args
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code ${LASTEXITCODE}: $($args -join ' ')"
    }
}

if (-not (Test-Path -LiteralPath $Python)) {
    & py -3.13 -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create .venv with Python 3.13 (exit code ${LASTEXITCODE})"
    }
    Invoke-Python -m pip install -e ".[dev]"
}
Invoke-Python -m smartpick_vla doctor
Invoke-Python -m smartpick_vla demo-expert --seed 11 --task-class scratch --output results/quickstart/expert.gif
