from pathlib import Path
import argparse, json, os, sys, subprocess
import numpy as np
root=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--seed',type=int,default=1)
parser.add_argument('--source',default='iot_nidd')
parser.add_argument('--target',default='ton_iot')
args=parser.parse_args()
src=root/'research_incremental_nids/src'
os.chdir(src)
sys.path.insert(0,str(src))
from utils import seed_everything
from datasets.data_loader import get_loaders
seed_everything(args.seed)
train,valid,test,taskcla=get_loaders([args.source,args.target],1,0,0,64,0,
 validation=.1,num_pkts=10,fields=['PL','IAT','DIR','WIN'],seed=1)
names=[]
files={'iot_nidd':'classes_iot-nidd_dwn10p.txt','ton_iot':'classes_ton-iot_dwn10p.txt'}
for loader,file in zip(train,[files[args.source],files[args.target]]):
    original=np.loadtxt(src.parent/'data/uniform_label'/file,dtype=str,ndmin=1)
    for original_id in loader.dataset.transform:
        name=str(original[original_id])
        if name not in names:
            names.append(name)
assert len(names)==sum(n for _,n in taskcla)
folder=root/'research_results'/f'{args.source}_to_{args.target}_seed{args.seed}_row_index_fixed'
folder.mkdir(parents=True,exist_ok=True)
record=dict(global_labels={str(i):name for i,name in enumerate(names)},
 source_labels=sorted(map(int,np.unique(test[0].dataset.labels))),
 target_labels=sorted(map(int,np.unique(test[1].dataset.labels))),
 source_only_unseen_target_labels=sorted(set(map(int,test[1].dataset.labels))-set(map(int,test[0].dataset.labels))))
(folder/'label_mapping.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(record))
subprocess.run([sys.executable,str(root/'summarize_repeat_experiments.py')],check=True)
