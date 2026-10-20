"""核对完整录屏训练的结构与正式实验差异；不同设备不强求逐位一致。"""
from pathlib import Path
import sys,json,csv
folder=Path(sys.argv[1]);root=Path(__file__).resolve().parent
reference=list(csv.DictReader((root/'full_run_reference_seed1.csv').open(encoding='utf-8-sig')))
protocol=json.loads((folder/'protocol.json').read_text('utf-8'))
assert protocol['training_seed']==1 and protocol['max_epochs']==200
assert protocol['author_index'] is False and protocol['memory_per_class']==25
rows=[]
for method in ['ft','ft_mem','replay','weighted','combined']:
    result=json.loads((folder/method/'complete.json').read_text('utf-8'))
    logs=[json.loads(x) for x in (folder/method/'training.jsonl').read_text('utf-8').splitlines()]
    iterations=[x['iter'] for x in logs if x.get('group')=='valid' and x.get('name')=='loss']
    assert max(iterations)>1,'Detected a one-epoch demonstration, not the requested full run.'
    for domain in ['source','target']:
        expected=next(r for r in reference if r['method']==method and r['domain']==domain)
        observed=result[domain]['macro_f1'];delta=100*(observed-float(expected['macro_f1']))
        row=dict(method=method,domain=domain,reference_macro_f1=float(expected['macro_f1']),observed_macro_f1=observed,difference_percentage_points=delta,epochs=max(iterations))
        rows.append(row)
        print(f'{method:9s} {domain:6s} Macro-F1={100*observed:.4f}% reference={100*float(expected["macro_f1"]):.4f}% difference={delta:+.4f}pp epochs={max(iterations)}')
with (folder/'recording_comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print('Structural check passed. Preserve actual scores and environment; investigate numerical differences instead of replacing recorded results.')
