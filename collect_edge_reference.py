from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED,ZIP_STORED
import json
root=Path(__file__).resolve().parent
runs=sorted((root/'edge_reference_results').glob('*_row_index_fixed'))
assert len(runs)==3
for run in runs:
    d=json.loads((run/'complete.json').read_text('utf-8'))
    assert set(d['methods'])=={'ft','ft_mem','replay','weighted','combined'}
    for name in d['methods']:
        assert all((run/name/(domain+'_predictions.npz')).is_file() for domain in ['source','target','source_validation','target_validation'])
out=root/'edge_reference_results.zip'
with ZipFile(out,'w') as z:
    for folder in [root/'edge_reference_results',root/'edge_reference_logs']:
        for p in sorted(folder.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(root).as_posix(),compress_type=ZIP_STORED if p.suffix in ['.pt','.npz'] else ZIP_DEFLATED)
    for name in ['run_edge_reference.py','control_environment.json']:z.write(root/name,name,compress_type=ZIP_DEFLATED)
print('RETURN_THIS_FILE',out,out.stat().st_size)
