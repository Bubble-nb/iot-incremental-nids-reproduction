#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
test -x .venv_controls/bin/python
mkdir -p fixed_target_logs
for seed in 1 2 3; do
  .venv_controls/bin/python -u run_fixed_target_controls.py --source iot_nidd --target ton_iot --seed "$seed" 2>&1 | tee "fixed_target_logs/full_forward_${seed}.log"
  .venv_controls/bin/python -u run_fixed_target_controls.py --source ton_iot --target iot_nidd --seed "$seed" 2>&1 | tee "fixed_target_logs/full_reverse_${seed}.log"
done
.venv_controls/bin/python verify_fixed_target_results.py
.venv_controls/bin/python collect_fixed_target_results.py
echo ALL_FIXED_TARGET_CONTROLS_COMPLETE
