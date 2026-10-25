#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p edge_reference_logs
for seed in 1 2 3; do
 .venv_controls/bin/python -u run_edge_reference.py --seed "$seed" 2>&1 | tee "edge_reference_logs/full_seed${seed}.log"
done
.venv_controls/bin/python collect_edge_reference.py
echo ALL_EDGE_REFERENCE_COMPLETE
