from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED,ZIP_STORED
import csv,json
root=Path(__file__).resolve().parent
results=root/'fixed_target_results'
runs=sorted(results.glob('*_seed[123]_row_index_fixed'))
assert len(runs)==6,'Expected six direction/seed runs; preserve incomplete logs before asking for help.'
rows=[]
for run in runs:
    record=json.loads((run/'complete.json').read_text(encoding='utf-8'))
    assert record['branches']==4 and record['test_prediction_files']==8
    for row in record['results']:rows.append(dict(run=run.name,**row))
assert len(rows)==48
with (results/'all_control_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
archive=root/'fixed_target_results.zip'
with ZipFile(archive,'w') as z:
    for folder in (results,root/'fixed_target_logs',root/'fixed_target_review'):
        if not folder.is_dir():continue
        for path in sorted(folder.rglob('*')):
            if path.is_file():
                mode=ZIP_STORED if path.suffix in ('.npz','.pt') else ZIP_DEFLATED
                z.write(path,path.relative_to(root).as_posix(),compress_type=mode)
    for name in ('run_fixed_target_controls.py','fixed_target_sampler.py','verify_fixed_target_results.py','control_environment.json'):
        z.write(root/name,name,compress_type=ZIP_DEFLATED)
print('RETURN_THIS_FILE',archive,'bytes',archive.stat().st_size)
