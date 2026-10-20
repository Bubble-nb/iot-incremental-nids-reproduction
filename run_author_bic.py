"""作者 BiC 实现的独立补充对照；保留作者阶段划分及损失表达式。

它重新训练自己的源模型，不属于共享 FT 源模型的五分支消融。
作者 criterion 将蒸馏损失转为 float，导致该项不传梯度；本脚本保留并披露，
因此结果仅指该仓库实现，不充当修复后的标准 BiC 算法结论。
"""
from pathlib import Path
import argparse, os, sys, json, time
from collections import Counter
import numpy as np
import torch
from sklearn.metrics import f1_score

ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=1)
p.add_argument('--source',default='iot_nidd');p.add_argument('--target',default='ton_iot')
p.add_argument('--epochs',type=int,default=200);a=p.parse_args()
src=ROOT/'research_incremental_nids/src';os.chdir(src);sys.path.insert(0,str(src))
from approach.bic import Appr
from datasets.data_loader import get_loaders
from datasets.exemplars_dataset import ExemplarsDataset
from networks.lopez17cnn import Lopez17CNN
from networks.network import LLL_Net
from utils import seed_everything
torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu'
out=ROOT/'research_results'/f'{a.source}_to_{a.target}_seed{a.seed}_author_bic'
out.mkdir(exist_ok=True)
if (out/'complete.json').exists():
    print('ALREADY COMPLETE',out);sys.exit(0)
protocol=dict(source=a.source,target=a.target,training_seed=a.seed,dataset_sample_seed=1,
    author_commit='842737730852c7c7c0955257c48ca72a22770343',max_epochs=a.epochs,
    bias_epochs=200,val_exemplar_percentage=.1,temperature=2,memory_per_class=25,
    lr=.1,lr_min=.0001,lr_factor=3,patience=20,batch_size=64,momentum=0,weight_decay=0,
    scope='Independent BiC source training with author-held-out exemplars; not the shared-source five-way ablation.',
    implementation_note='Preserve author criterion float(gkd): distillation term is detached. Only evaluates this implementation.',
    compatibility='GPU-safe vectorized eval and class metrics, NumPy float64 independent reporting; no change to training/bias correction code.')
(out/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
class Logger:
    def log_scalar(self,**r):
        with (out/'training.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(r)+'\n')
class CompatibleBiC(Appr):
    def eval(self,t,loader):
        self.model.eval();total=hits=count=0.;zs=[];ys=[]
        with torch.no_grad():
            for x,y in loader:
                x,y=x.to(self.device),y.to(self.device)
                previous=self.bias_forward(self.model_old(x)) if t>0 else None
                outputs=self.bias_forward(self.model(x));z=torch.cat(outputs,dim=1)
                loss=self.criterion(t,outputs,y,previous,train=False)
                total+=loss.item()*len(y);hits+=(z.argmax(1)==y).sum().item();count+=len(y)
                zs.extend(z.cpu().tolist());ys.extend(y.cpu().tolist())
        return total/count,hits/count,hits/count,zs,ys,[]
seed_everything(a.seed)
train,valid,test,taskcla=get_loaders([a.source,a.target],1,0,0,64,0,validation=.1,num_pkts=10,fields=['PL','IAT','DIR','WIN'],seed=1)
mapping=json.loads((ROOT/'research_results'/f'{a.source}_to_{a.target}_seed{a.seed}_row_index_fixed/label_mapping.json').read_text('utf-8'))
(out/'label_mapping.json').write_text(json.dumps(mapping,ensure_ascii=False,indent=2),encoding='utf-8')
benign=next(int(k) for k,v in mapping['global_labels'].items() if v=='benign')
seed_everything(a.seed)
model=LLL_Net(Lopez17CNN(num_pkts=10,num_fields=4));model.add_head(taskcla[0][1]);model.to(device)
memory=ExemplarsDataset(train[0].dataset.class_indices,num_exemplars_per_class=25)
appr=CompatibleBiC(model,device,logger=Logger(),exemplars_dataset=memory,
    nepochs=a.epochs,lr=.1,lr_min=.0001,lr_factor=3,lr_patience=20,momentum=0,wd=0,
    val_exemplar_percentage=.1,num_bias_epochs=200,T=2)
start=time.time()
appr.train(0,train[0],valid[0])
source_seconds=time.time()-start
seed_everything(a.seed+1000);model.add_head(taskcla[1][1]);model.to(device)
seed_everything(a.seed+2000);start=time.time();appr.train(1,train[1],valid[1]);seconds=time.time()-start
torch.save(dict(model=model.state_dict(),bias=[x.state_dict() for x in appr.bias_layers]),out/'checkpoint.pt')
report=dict(source_training_seconds=source_seconds,adaptation_seconds=seconds,domains={})
for i,domain in enumerate(['source','target']):
    _,_,_,z,y,_=appr.eval(1,test[i]);z=np.asarray(z);y=np.asarray(y);pred=z.argmax(1)
    np.savez_compressed(out/f'{domain}_predictions.npz',logits=z,y_true=y,y_pred=pred)
    report['domains'][domain]=dict(macro_f1=float(f1_score(y,pred,labels=np.unique(y),average='macro',zero_division=0)),
        fpr=float(np.mean(pred[y==benign]!=benign)),attack_recall=float(np.mean(pred[y!=benign]!=benign)),n=len(y))
report['bias_parameters']=[dict(alpha=float(x.alpha.item()),beta=float(x.beta.item())) for x in appr.bias_layers]
report['post_update_memory_counts']={str(k):v for k,v in Counter(memory.labels).items()}
(out/'complete.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('COMPLETE',out,json.dumps(report),flush=True)
