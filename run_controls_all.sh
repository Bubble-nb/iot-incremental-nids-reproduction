#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
test -x .venv_controls/bin/python || { printf 'Run bash prepare_controls.sh first.\n'; exit 1; }
mkdir -p control_logs
for task_seed in 1 2 3; do
  .venv_controls/bin/python -u run_sampling_controls.py --source iot_nidd --target ton_iot --seed "$task_seed" 2>&1 | tee "control_logs/full_forward_seed${task_seed}.log"
  .venv_controls/bin/python -u run_sampling_controls.py --source ton_iot --target iot_nidd --seed "$task_seed" 2>&1 | tee "control_logs/full_reverse_seed${task_seed}.log"
done
.venv_controls/bin/python collect_control_results.py
printf 'ALL_SAMPLING_CONTROLS_COMPLETE\n'
