#!/usr/bin/env bash
# Month 10: Sequential cohort-pipeline runner (M7 -> M8 -> M9 -> M10).
#
# Chains the four PDE cohort scripts so that Month 10's ingest step always
# finds fresh metric JSONs:
#
#   42_anisotropic_pde.py            (Month 7)  -> anisotropic_geometry_metrics.json
#   43_stromal_feedback.py           (Month 8)  -> stromal_feedback_metrics.json
#   44_adaptive_therapy.py           (Month 9)  -> adaptive_geometry_metrics.json
#   46_sensitivity_analysis.py       (Phase 2b) -> sobol_sensitivity_results.json +
#                                                 sobol_tornado_plot.png
#   47_optimal_control.py            (Phase 3)  -> dual_drug_comparison.json
#   45_validation_synthesis.py       (Month 10) -> master_cohort_summary.json +
#                                                  master_cohort_synthesis.png +
#                                                  POSTER_KEY_FINDINGS.md +
#                                                  MONTH10_AUDIT.md
#
# Platform note: This is a bash script. On Windows run it under Git Bash or
# WSL. A PowerShell equivalent would be:
#
#   foreach ($s in 42,43,44,45) { & venv\Scripts\python.exe "src\${s}_*.py" }
#
# Usage:
#   bash run_all.sh            # run the full chain
#   bash run_all.sh --month10  # only run script 45 (assumes 42-44 already done)
#
# Idempotency: script 45 caches the spherical baseline to
# output/isotropic_baseline_metrics.json automatically (D5).
set -euo pipefail

# Resolve project root (directory containing this script). This works whether
# the user invokes the script from the repo root or from elsewhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$SCRIPT_DIR"

PY="${PYTHON:-venv/Scripts/python.exe}"   # override with PYTHON env var if needed
if [[ ! -x "$PY" ]]; then
    echo "[run_all] ERROR: interpreter not found at '$PY'." >&2
    echo "          Set PYTHON=/path/to/python or activate a venv before running." >&2
    exit 1
fi

declare -a SCRIPTS=(42_anisotropic_pde 43_stromal_feedback 44_adaptive_therapy 46_sensitivity_analysis 47_optimal_control 45_validation_synthesis)
declare -A MONTHS=(
    [42_anisotropic_pde]="7"
    [43_stromal_feedback]="8"
    [44_adaptive_therapy]="9"
    [46_sensitivity_analysis]="Phase 2b"
    [47_optimal_control]="Phase 3"
    [45_validation_synthesis]="10"
)

# --month10 => only run script 45
if [[ "${1:-}" == "--month10" ]]; then
    SCRIPTS=(45_validation_synthesis)
fi

echo "=================================================================="
echo " RUNNING COHORT PIPELINE  (Month 7 -> Month 10)"
echo "=================================================================="
echo "  Interpreter : $PY"
echo "  Project root: $SCRIPT_DIR"
echo "  Scripts     : ${SCRIPTS[*]}"
echo "=================================================================="

for s in "${SCRIPTS[@]}"; do
    src="src/${s}.py"
    if [[ ! -f "$src" ]]; then
        echo "[run_all] ERROR: missing $src" >&2
        exit 1
    fi
    echo
    echo "########## MONTH ${MONTHS[$s]} — $s ##########"
    "$PY" "$src"
    if [[ $? -ne 0 ]]; then
        echo "[run_all] FAILED at $src" >&2
        exit 1
    fi
done

echo
echo "=================================================================="
echo " ALL SCRIPTS COMPLETED SUCCESSFULLY"
echo "=================================================================="
echo " Month 10 deliverables:"
echo "   output/master_cohort_summary.json"
echo "   output/master_cohort_synthesis.png"
echo "   output/POSTER_KEY_FINDINGS.md"
echo "   output/MONTH10_AUDIT.md"
echo "   output/isotropic_baseline_metrics.json"
echo "=================================================================="
