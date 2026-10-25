"""Recompute Edge-IIoT saved predictions; no training or GPU required."""
from pathlib import Path
from collections import defaultdict
import argparse,hashlib,json,csv
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True,help='Directory containing edge_reference_results')
a=p.parse_args();O=a.root.resolve()
rows=[];checks=0;hashes=0
folders=sorted((O/'edge_reference_results').glob('*_row_index_fixed'));assert len(folders)==3
for folder in folders:
 complete=json.loads((folder/'complete.json').read_text());assert set(complete['methods'])=={'ft','ft_mem','replay','weighted','combined'}
 mapping=json.loads((folder/'label_mapping.json').read_text())['global_labels'];assert len(mapping)==18
 benign=next(int(k) for k,v in mapping.items() if v=='benign')
 ref={};seed=int(folder.name.split('_seed')[1].split('_')[0])
 for branch in sorted(p for p in folder.iterdir() if p.is_dir() and (p/'source_predictions.npz').exists()):
  if (branch/'checkpoint_hash.json').exists():
   assert hashlib.sha256((branch/'checkpoint.pt').read_bytes()).hexdigest()==json.loads((branch/'checkpoint_hash.json').read_text())['sha256'];hashes+=1
  for npz in sorted(branch.glob('*_predictions.npz')):
   domain=npz.name.removesuffix('_predictions.npz')
   with np.load(npz,allow_pickle=False) as a:y=a['y_true'];pred=a['y_pred'];logits=a['logits']
   assert np.isfinite(logits).all() and len(y)==len(pred)==len(logits) and np.array_equal(pred,logits.argmax(1))
   assert logits.shape[1]==(10 if branch.name=='source' else 18),(branch,logits.shape)
   if domain in ref:assert np.array_equal(y,ref[domain])
   else:ref[domain]=y.copy()
   precision,recall,f1,support=precision_recall_fscore_support(y,pred,labels=np.unique(y),zero_division=0)
   normal=y==benign;attack=~normal
   values=dict(macro_f1=float(f1.mean()),fpr=float((pred[normal]!=benign).mean()),attack_recall=float((pred[attack]!=benign).mean()))
   recorded=json.loads(npz.with_name(domain+'_metrics.json').read_text())
   assert recorded['n']==len(y)
   for k,v in values.items():assert np.isclose(v,recorded[k],rtol=0,atol=1e-12),(branch,domain,k)
   assert np.allclose(recorded['per_class_f1'],f1,atol=1e-12,rtol=0)
   assert np.array_equal(recorded['support'],support)
   checks+=1
   if domain in ['source','target']:rows.append(dict(seed=seed,method=branch.name,domain=domain,n=len(y),**{k+'_percent':v*100 for k,v in values.items()}))
 print('EDGE_VERIFIED',folder.name,flush=True)
assert hashes==15 and checks==66 and len(rows)==36,(hashes,checks,len(rows))
def save(name,data):
 with (O/name).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
save('recomputed_metrics.csv',rows)
groups=defaultdict(list)
for row in rows:groups[row['method'],row['domain']].append(row)
means=[]
for (method,domain),data in sorted(groups.items()):
 assert len(data)==3
 means.append(dict(method=method,domain=domain,**{k:float(np.mean([r[k] for r in data])) for k in ['macro_f1_percent','fpr_percent','attack_recall_percent']}))
save('three_seed_means.csv',means)
print('EDGE_RESULT_VERIFICATION_PASS',checks,hashes,len(rows))
