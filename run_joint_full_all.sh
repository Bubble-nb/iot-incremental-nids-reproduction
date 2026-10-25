#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p joint_full_logs
for seed in 1 2 3; do
 .venv_controls/bin/python -u run_joint_full_reference.py --source iot_nidd --target ton_iot --seed "$seed" 2>&1 | tee "joint_full_logs/full_forward_${seed}.log"
 .venv_controls/bin/python -u run_joint_full_reference.py --source ton_iot --target iot_nidd --seed "$seed" 2>&1 | tee "joint_full_logs/full_reverse_${seed}.log"
done
.venv_controls/bin/python collect_joint_full_results.py
echo ALL_JOINT_FULL_COMPLETE
