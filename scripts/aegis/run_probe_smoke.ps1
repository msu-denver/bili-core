# =============================================================================
# PROBE real-LLM smoke test
# =============================================================================
#
# Runs the PROBE suite against three real provider families, all via their
# first-party (direct) APIs -- no AWS Bedrock or Google Vertex markup:
#   attacker:  DeepSeek         (remote_deepseek;      ~$0.14/M input)
#   victim:    Anthropic Claude (remote_anthropic;     model under test)
#   judge:     Google Gemini    (remote_google_genai;  cross-provider)
#
# Pre-flight:
#   1. Activate the bili-core conda env:
#        conda activate bili-core
#   2. Set the three direct-API keys in your shell:
#        $env:DEEPSEEK_API_KEY  = "..."
#        $env:ANTHROPIC_API_KEY = "..."
#        $env:GEMINI_API_KEY    = "..."   # or GOOGLE_API_KEY
#   3. (Optional) Generate a baseline first:
#        python -m bili.aegis.suites.baseline.run_baseline `
#            --configs bili/aether/config/examples/simple_chain.yaml `
#            --seeds 0
#
# Cost ceiling: --budget-cost-usd 0.50 per session caps individual runs.
# Expected actual cost: < $0.10 per PAIR session at 8 turns. Total smoke
# spend (3 policies × 1 config × 1 seed) should be < $1.
#
# Usage:
#   .\scripts\aegis\run_probe_smoke.ps1
#   .\scripts\aegis\run_probe_smoke.ps1 -Policy crescendo
#   .\scripts\aegis\run_probe_smoke.ps1 -Objective pr_safety_bypass_001 -BudgetTurns 6
# =============================================================================

param(
    [string]$BaselineDir = "bili/aegis/suites/baseline/results",
    [string]$AttackerModel = "deepseek-v4-pro",
    [string]$VictimModel = "claude-sonnet-4-6",
    [string]$JudgeModel = "gemini-2.5-flash",
    [string]$AttackerModelType = "remote_deepseek",
    [string]$VictimModelType = "remote_anthropic",
    [string]$JudgeModelType = "remote_google_genai",
    [string]$Policy = "pair",
    [string]$Objective = "pr_safety_bypass_001",
    [string]$Config = "bili/aether/config/examples/probe_victim_claude.yaml",
    [int]$BudgetTurns = 8,
    [int]$BudgetTokens = 200000,
    [double]$BudgetCostUsd = 0.50,
    [int]$Seed = 0,
    [string]$ResultsDir = "bili/aegis/suites/probe/results"
)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "============================================================"
Write-Host "PROBE real-LLM smoke test"
Write-Host "============================================================"
Write-Host "  attacker:   $AttackerModel ($AttackerModelType)"
Write-Host "  victim:     $VictimModel ($VictimModelType)"
Write-Host "  judge:      $JudgeModel ($JudgeModelType)"
Write-Host "  policy:     $Policy"
Write-Host "  objective:  $Objective"
Write-Host "  config:     $Config"
Write-Host "  budget:     $BudgetTurns turns / `$$BudgetCostUsd cost cap"
Write-Host "  seed:       $Seed"
Write-Host "  baseline:   $BaselineDir"
Write-Host "  results:    $ResultsDir"
Write-Host "============================================================"
Write-Host ""

# Resolve Python. Defaults to `python` on PATH, so activate the bili-core conda
# env first. Override $env:PYTHON to point at a specific interpreter.
if (-not $env:PYTHON) {
    $env:PYTHON = "python"
}

if (-not (Test-Path $env:PYTHON)) {
    Write-Error "Python not found at $env:PYTHON. Activate the bili-core conda env or set `$env:PYTHON to a valid interpreter."
    exit 1
}

# Run the suite
& $env:PYTHON -m bili.aegis.suites.probe.run_probe_suite `
    --smoke `
    --policies $Policy `
    --objectives $Objective `
    --configs $Config `
    --seeds $Seed `
    --baseline-results $BaselineDir `
    --budget-turns $BudgetTurns `
    --budget-tokens $BudgetTokens `
    --budget-cost-usd $BudgetCostUsd `
    --attacker-model $AttackerModel `
    --victim-model $VictimModel `
    --judge-model $JudgeModel `
    --attacker-model-type $AttackerModelType `
    --victim-model-type $VictimModelType `
    --judge-model-type $JudgeModelType `
    --results-dir $ResultsDir

$exitCode = $LASTEXITCODE
Write-Host ""
if ($exitCode -eq 0) {
    Write-Host "Smoke run completed. Inspect:"
    Write-Host "  $ResultsDir/probe_results_matrix.csv"
    Write-Host "  $ResultsDir/<mas_id>/sessions/*.json"
} else {
    Write-Error "Smoke run exited with code $exitCode"
}
exit $exitCode
