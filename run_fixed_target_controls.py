"""Prospectively specified sampling controls; preserves the main experiment entry.

Reuses saved source checkpoints and memories, verifies split fingerprints, trains
all branches before exporting test metrics. No main-experiment file is modified.
"""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import argparse, copy, csv, hashlib, json, math, os, sys, time
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset, Sampler
from sklearn.metrics import precision_recall_fscore_support

ROOT=Path(__file__).resolve().parent
P=argparse.ArgumentParser(description=__doc__)
P.add_argument('--source',choices=['iot_nidd','ton_iot'],default='iot_nidd')
P.add_argument('--target',choices=['iot_nidd','ton_iot'],default='ton_iot')
P.add_argument('--seed',type=int,default=1)
P.add_argument('--epochs',type=int,default=200)
P.add_argument('--baseline-root',type=Path,default=ROOT/'control_baselines')
P.add_argument('--output-root',type=Path,default=ROOT/'fixed_target_results')
P.add_argument('--dry-run',action='store_true',help='Verify data, memory and plans; do not train or evaluate models.')
A=P.parse_args()
assert A.source!=A.target
if not A.dry_run and not torch.cuda.is_available():
    raise SystemExit('CUDA is unavailable; refusing a long CPU training run. Run --dry-run to inspect the environment.')
SRC=ROOT/'research_incremental_nids'/'src'
if not SRC.is_dir():raise SystemExit('Initialize the fixed author code first: python bootstrap_author_code.py')
os.chdir(SRC);sys.path.insert(0,str(SRC))
from datasets.data_loader import get_loaders
from datasets.exemplars_dataset import ExemplarsDataset
from approach.jointft import Appr as AuthorFT
from approach.incremental_learning import Inc_Learning_Appr
from networks.lopez17cnn import Lopez17CNN
from networks.network import LLL_Net
from utils import seed_everything
torch.set_num_threads(4)
DEVICE='cpu' if A.dry_run else 'cuda'
RUN=f'{A.source}_to_{A.target}_seed{A.seed}_row_index_fixed'
BASE=A.baseline_root.resolve()/RUN
OUT=A.output_root.resolve()/RUN
OUT.mkdir(parents=True,exist_ok=True)

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()

def write_json(path,value):
    def clean(x):
        if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
        if isinstance(x,(tuple,list)):return [clean(v) for v in x]
        if isinstance(x,np.generic):return x.item()
        return x
    path.write_text(json.dumps(clean(value),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def split_fingerprint(loader):
    h=hashlib.sha256()
    for x,y in zip(loader.dataset.images,loader.dataset.labels):
        arr=np.asarray(x,dtype='<f4').copy();arr[arr==0]=0
        assert np.isfinite(arr).all()
        h.update(np.asarray(arr.shape,dtype='<i8').tobytes())
        h.update(arr.tobytes());h.update(np.asarray([y],dtype='<i8').tobytes())
    return dict(n=len(loader.dataset),sha256=h.hexdigest())

seed_everything(A.seed)
TRAIN,VALID,TEST,TASKCLA=get_loaders([A.source,A.target],1,0,0,64,0,
    validation=.1,num_pkts=10,fields=['PL','IAT','DIR','WIN'],seed=1)
fingerprints={f'{split}_{domain}':split_fingerprint(loaders[d])
    for split,loaders in [('train',TRAIN),('valid',VALID),('test',TEST)]
    for d,domain in enumerate(['source','target'])}
expected=json.loads((BASE/'split_fingerprints.json').read_text(encoding='utf-8'))
assert fingerprints==expected, 'Data split fingerprint mismatch; refuse an incomparable run.'
MAPPING=json.loads((BASE/'label_mapping.json').read_text(encoding='utf-8'))
LABELS=MAPPING['global_labels']
BENIGN=next(int(k) for k,v in LABELS.items() if v=='benign')
K=sum(int(n) for _,n in TASKCLA)
assert len(LABELS)==K
source_checkpoint=BASE/'source'/'checkpoint.pt'
saved=torch.load(source_checkpoint,map_location='cpu',weights_only=False)
memory=ExemplarsDataset(TRAIN[0].dataset.class_indices,num_exemplars_per_class=25)
memory.images,memory.labels=saved['memory_images'],saved['memory_labels']
N,M=len(TRAIN[1].dataset),len(memory)
assert N>0 and M>0
steps=math.ceil((N+M)/64)

BRANCHES=[dict(name='fixed_target_r000',ratio=0.,weight_mode='none'),
dict(name='fixed_target_r010',ratio=.1,weight_mode='none'),
dict(name='fixed_target_r025',ratio=.25,weight_mode='none'),
dict(name='fixed_target_r050',ratio=.5,weight_mode='none')]
from fixed_target_sampler import FixedTargetBatches, batch_counts

def old_counts(ratio):
    # Cumulative rounding gives 6/7 old samples at ratio=.1, rather than rounding
    # every batch to 6. At .25/.5 it exactly preserves 16/32 old samples per batch.
    return [math.floor((i+1)*64*ratio+1e-9)-math.floor(i*64*ratio+1e-9) for i in range(steps)]

new_hist=np.bincount(TRAIN[1].dataset.labels,minlength=K).astype(float)
old_hist=np.bincount(memory.labels,minlength=K).astype(float)
plans=[]
for branch in BRANCHES:
    counts=batch_counts(N,branch['ratio'])
    plan=dict(branch,updates_per_epoch=len(counts),target_draws_per_epoch=N,
        old_draws_per_epoch=sum(o for _,o in counts),
        effective_old_ratio=sum(o for _,o in counts)/(N+sum(o for _,o in counts)),
        exposure_scope='Same target draw stream per epoch; total exposure differs if early stopping selects different run lengths.')
    plans.append(plan)
# Check three prospective epochs without training or test inference.
for epoch in (1,2,3):
    reference=None
    for branch in BRANCHES:
        sampler=FixedTargetBatches(N,M,branch['ratio'],A.seed)
        stream,_=sampler.streams(epoch)
        if reference is None:reference=stream
        else:assert np.array_equal(reference,stream)
        assert sum(n for n,_ in sampler.counts)==N
        assert all(n+o<=64 and n>0 for n,o in sampler.counts)
print('TARGET_EXPOSURE_STREAM_CHECK_PASS',flush=True)

identity=dict(protocol='fixed-target-exposure-per-epoch-v1',source=A.source,target=A.target,seed=A.seed,
    author_commit='842737730852c7c7c0955257c48ca72a22770343',
    script_sha256=digest(Path(__file__)),source_checkpoint_sha256=digest(source_checkpoint),
    splits=fingerprints,branches=plans,max_epochs=A.epochs,lr=.1,lr_factor=3,
    lr_min=.0001,patience=20,batch_size=64,weight_clip=[.2,5.],
    weight_definition='For sampler_expected: (N+M)*((1-rho)*target_hist/N+rho*memory_hist/M); same smoothing and clipping as static.',
    model_selection='Target validation unweighted cross-entropy; unchanged to isolate sampling/weight changes.',
    evaluation='Train all four branches before test prediction; all branches and both domains must be retained.',
    interpretation='New controls on previously examined fixed test samples; not independent confirmatory network data.')
plan_path=OUT/'frozen_plan.json'
if plan_path.exists():
    previous=json.loads(plan_path.read_text(encoding='utf-8'))
    assert previous['identity']==identity,'Frozen control configuration changed; use a fresh output directory.'
else:write_json(plan_path,dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),identity=identity))
write_json(OUT/'label_mapping.json',MAPPING)
write_json(OUT/'environment.json',dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,
    cuda_runtime=torch.version.cuda,cuda_available=torch.cuda.is_available(),
    gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,execution_device=DEVICE,threads=4))

seed_everything(A.seed)
source_model=LLL_Net(Lopez17CNN(num_pkts=10,num_fields=4))
source_model.add_head(TASKCLA[0][1]);source_model.load_state_dict(saved['model'])
seed_everything(A.seed+1000)
source_model.add_head(TASKCLA[1][1]);source_model.to(DEVICE)
expanded=source_model.get_copy()
print('PREFLIGHT_OK',RUN,'device',DEVICE,'target_train',N,'memory',M,'classes',K,flush=True)
for plan in plans:
    print('BRANCH_PLAN',json.dumps(plan),flush=True)
if A.dry_run:
    print('DRY_RUN_COMPLETE: no training or test predictions were performed.',flush=True)
    raise SystemExit(0)

class Logger:
    def __init__(self,folder):self.folder=folder
    def log_scalar(self,**record):
        with (self.folder/'training.jsonl').open('a',encoding='utf-8') as handle:
            handle.write(json.dumps(record)+'\n')

class RatioBatches(Sampler):
    def __init__(self,ratio):self.counts=old_counts(ratio)
    def __len__(self):return steps
    def __iter__(self):
        for count in self.counts:
            new=torch.randint(N,(64-count,)).tolist()
            old=(torch.randint(M,(count,))+N).tolist()
            batch=new+old
            order=torch.randperm(64).tolist()
            yield [batch[i] for i in order]

class ControlledFT(AuthorFT):
    def __init__(self,*args,plan,**kwargs):
        super().__init__(*args,**kwargs);self.plan=plan;self.class_weights=None
    def eval(self,t,loader):
        self.model.eval();loss_sum=hits=count=0;outputs_all=[];targets_all=[]
        with torch.no_grad():
            for x,y in loader:
                x,y=x.to(self.device),y.to(self.device)
                logits=torch.cat(self.model(x),dim=1)
                loss_sum+=torch.nn.functional.cross_entropy(logits,y,reduction='sum').item()
                hits+=(logits.argmax(1)==y).sum().item();count+=len(y)
                outputs_all.extend(logits.cpu().tolist());targets_all.extend(y.cpu().tolist())
        return loss_sum/count,hits/count,hits/count,outputs_all,targets_all,[]
    def criterion(self,t,outputs,targets,features=None,epoch=1):
        return torch.nn.functional.cross_entropy(torch.cat(outputs,dim=1),targets,
            weight=self.class_weights if self.model.training else None)
    def train_loop(self,t,train,valid):
        if 'weights' in self.plan:
            self.class_weights=torch.tensor(self.plan['weights'],dtype=torch.float32,device=self.device)
            self.logger.log_scalar(group='protocol',name='class_weights',value=self.plan['weights'])
        dataset=ConcatDataset([train.dataset,self.exemplars_dataset])
        if self.plan['ratio'] is None:
            loader=DataLoader(dataset,batch_size=64,shuffle=True)
        else:loader=DataLoader(dataset,batch_sampler=FixedTargetBatches(N,M,self.plan['ratio'],A.seed,self.logger.folder))
        return Inc_Learning_Appr.train_loop(self,t,loader,valid)
    def post_train_process(self,t,train,valid,correction=False):return None

common=dict(nepochs=A.epochs,lr=.1,lr_min=.0001,lr_factor=3,lr_patience=20)
for plan in plans:
    folder=OUT/plan['name'];folder.mkdir(exist_ok=True)
    if (folder/'trained.json').is_file():
        record=json.loads((folder/'trained.json').read_text(encoding='utf-8'))
        assert digest(folder/'checkpoint.pt')==record['checkpoint_sha256']
        print('RESUME_TRAINED',plan['name'],flush=True);continue
    if (folder/'training.jsonl').exists():
        raise RuntimeError(f'Incomplete branch {folder}; preserve logs and use a fresh output root before retrying.')
    seed_everything(A.seed+2000)
    model=copy.deepcopy(source_model);model.set_state_dict(expanded)
    approach=ControlledFT(model,DEVICE,plan=plan,logger=Logger(folder),
        exemplars_dataset=copy.deepcopy(memory),**common)
    print('START_BRANCH',plan['name'],flush=True);started=time.time()
    approach.train(1,TRAIN[1],VALID[1])
    torch.save(model.state_dict(),folder/'checkpoint.pt')
    write_json(folder/'trained.json',dict(elapsed_seconds=time.time()-started,
        checkpoint_sha256=digest(folder/'checkpoint.pt'),plan=plan))

def evaluate(model,loader,folder,domain):
    model.eval();ys=[];raw=[]
    with torch.no_grad():
        for x,y in loader:
            ys.append(y.numpy());raw.append(torch.cat(model(x.to(DEVICE)),dim=1).cpu().numpy())
    y=np.concatenate(ys);logits=np.concatenate(raw);pred=logits.argmax(1)
    labels=np.unique(y)
    precision,recall,f1,support=precision_recall_fscore_support(y,pred,labels=labels,zero_division=0)
    normal=y==BENIGN;attack=~normal
    result=dict(domain=domain,n=len(y),macro_f1_percent=100*float(f1.mean()),
        fpr_percent=100*float((pred[normal]!=BENIGN).mean()),
        attack_recall_percent=100*float((pred[attack]!=BENIGN).mean()),
        labels=labels.tolist(),support=support.tolist(),per_class_f1=f1.tolist(),
        per_class_precision=precision.tolist(),per_class_recall=recall.tolist())
    np.savez_compressed(folder/f'{domain}_predictions.npz',y_true=y,y_pred=pred,logits=logits)
    write_json(folder/f'{domain}_metrics.json',result)
    return result

rows=[]
for plan in plans:
    folder=OUT/plan['name']
    model=copy.deepcopy(source_model)
    model.load_state_dict(torch.load(folder/'checkpoint.pt',map_location=DEVICE,weights_only=True))
    for domain,loader in zip(['source_validation','target_validation'],VALID):
        evaluate(model,loader,folder,domain)
    for domain,loader in zip(['source','target'],TEST):
        result=evaluate(model,loader,folder,domain)
        rows.append(dict(method=plan['name'],domain=domain,n=result['n'],
            macro_f1_percent=result['macro_f1_percent'],fpr_percent=result['fpr_percent'],
            attack_recall_percent=result['attack_recall_percent']))
        print('CONTROL_METRICS',plan['name'],json.dumps(result),flush=True)
with (OUT/'summary.csv').open('w',encoding='utf-8-sig',newline='') as handle:
    writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
write_json(OUT/'complete.json',dict(run=RUN,branches=len(plans),test_prediction_files=8,
    validation_prediction_files=8,
    completed_at_utc=datetime.now(timezone.utc).isoformat(),results=rows))
print('FIXED_TARGET_COMPLETE',OUT,flush=True)
