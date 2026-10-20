"""固定检查点字段置换；不训练模型。输出新复算记录，保留历史置换记录。"""
from pathlib import Path
import argparse,os,sys,csv,json,hashlib
import numpy as np
import torch
from sklearn.metrics import f1_score
ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--output',default=str(ROOT/'research_results/field_permutation_recomputed_20261020'))
args=parser.parse_args();OUT=Path(args.output).resolve();OUT.mkdir(parents=True,exist_ok=True)
src=ROOT/'research_incremental_nids/src';os.chdir(src);sys.path.insert(0,str(src))
from datasets.data_loader import get_loaders
from networks.lopez17cnn import Lopez17CNN
from networks.network import LLL_Net
from utils import seed_everything
torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu'
seed_everything(1)
_,_,tests,taskcla=get_loaders(['iot_nidd','ton_iot'],1,0,0,64,0,validation=.1,num_pkts=10,fields=['PL','IAT','DIR','WIN'],seed=1)
run=ROOT/'research_results/iot_nidd_to_ton_iot_seed1_row_index_fixed'
def predict(model,x):
 out=[]
 with torch.inference_mode():
  for start in range(0,len(x),64):out.append(torch.cat(model(torch.from_numpy(x[start:start+64]).to(device)),1).cpu().numpy())
 return np.concatenate(out)
rows=[];hashes={}
for method in ['ft_mem','combined']:
 model=LLL_Net(Lopez17CNN(num_pkts=10,num_fields=4))
 for _,n in taskcla:model.add_head(n)
 checkpoint=run/method/'checkpoint.pt';hashes[str(checkpoint.relative_to(ROOT))]=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
 model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True));model.to(device).eval()
 for di,(domain,loader) in enumerate(zip(['source','target'],tests)):
  x=np.asarray(loader.dataset.images,dtype=np.float32);y=np.asarray(loader.dataset.labels)
  assert x.shape[1:]==(1,10,4),x.shape
  saved=np.load(run/method/(domain+'_predictions.npz'));assert np.array_equal(y,saved['y_true'])
  z=predict(model,x);assert np.allclose(z,saved['logits'],atol=1e-5,rtol=1e-5)
  base=f1_score(y,z.argmax(1),labels=np.unique(y),average='macro',zero_division=0)
  for field,fi in zip(['PL','IAT','DIR','WIN'],range(4)):
   for repeat in range(5):
    # Same permutation across fields and methods within each domain/repeat.
    perm=np.random.default_rng(20261003+repeat).permutation(len(x))
    changed=x.copy();changed[:,:,:,fi]=x[perm,:,:,fi]
    val=f1_score(y,predict(model,changed).argmax(1),labels=np.unique(y),average='macro',zero_division=0)
    rows.append(dict(method=method,domain=domain,field=field,repeat=repeat,baseline_macro_f1=base,permuted_macro_f1=val,delta_macro_f1=base-val))
  print(method,domain,'complete',flush=True)
with (OUT/'grouped_field_permutation.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(OUT/'protocol.json').write_text(json.dumps(dict(repeats=5,seed_base=20261003,rng='NumPy default_rng(seed_base + repeat); identical seed rule for each field and method',delta='baseline minus permuted Macro-F1; positive means drop',checkpoints=hashes,scope='Deterministic checkpoint re-evaluation; historical records preserved separately for comparison.',training='none'),ensure_ascii=False,indent=2),'utf-8')
print('Saved',len(rows),'records to',OUT,flush=True)
