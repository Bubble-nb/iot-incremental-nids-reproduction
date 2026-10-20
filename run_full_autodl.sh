#!/usr/bin/env bash
# Execute from the extracted code directory; preserve the original research results.
set -euo pipefail
cd -- "$(dirname -- "$0")"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
stamp="$(date +%Y%m%d_%H%M%S)"
record_dir="recorded_runs/full_seed1_${stamp}"
mkdir -p -- recorded_runs
mkdir -- "$record_dir"
python -c 'import json,platform,torch; print(json.dumps(dict(platform=platform.platform(),python=platform.python_version(),torch=torch.__version__,cuda=torch.cuda.is_available(),gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),ensure_ascii=False,indent=2))' | tee "$record_dir/environment.json"
printf '\nFull experiment: source=iot_nidd target=ton_iot seed=1 epochs=200\n'
printf 'Results: %s\n' "$record_dir"
python -u run_research.py --source iot_nidd --target ton_iot --seed 1 --epochs 200 --methods ft ft_mem replay weighted combined --output-root "$record_dir" 2>&1 | tee "$record_dir/full_training.log"
python check_recorded_run.py "$record_dir/iot_nidd_to_ton_iot_seed1_row_index_fixed" | tee "$record_dir/result_check.txt"
printf '\nTRAINING_AND_CHECK_COMPLETE %s\n' "$record_dir"
