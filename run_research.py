"""Fixed author sample experiment, not a full-data reproduction.

Reuses author data loader, model, SGD schedule and FT implementation.
One source checkpoint and identical memory are shared by all adaptations.
Test predictions never select hyperparameters or checkpoints.
"""
from pathlib import Path
import argparse, copy, hashlib, json, math, os, random, sys, time
from collections import Counter
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset, Sampler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--source', default='iot_nidd')
parser.add_argument('--target', default='ton_iot')
parser.add_argument('--methods', nargs='+', default=['ft', 'ft_mem', 'replay', 'weighted', 'combined'])
parser.add_argument('--author-index', action='store_true')
args = parser.parse_args()
src = ROOT / ('compatibility_incremental_nids' if args.author_index else 'research_incremental_nids') / 'src'
os.chdir(src)
sys.path.insert(0, str(src))
from datasets.data_loader import get_loaders
from datasets.exemplars_dataset import ExemplarsDataset
from approach.jointft import Appr as AuthorFT
from approach.incremental_learning import Inc_Learning_Appr
from networks.lopez17cnn import Lopez17CNN
from networks.network import LLL_Net
from utils import seed_everything

torch.set_num_threads(4)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
suffix = 'author_index' if args.author_index else 'row_index_fixed'
OUT = ROOT / 'research_results' / f'{args.source}_to_{args.target}_seed{args.seed}_{suffix}'
OUT.mkdir(parents=True, exist_ok=True)
run_identity = dict(source=args.source, target=args.target, training_seed=args.seed,
                    dataset_sample_seed=1, max_epochs=args.epochs,
                    author_index=args.author_index, lr=.1, lr_factor=3,
                    lr_min=.0001, patience=20, batch_size=64, memory_per_class=25,
                    replay_fraction=.5, weight_exponent=.5, weight_clip=[.2, 5.])
identity_path = OUT / 'run_identity.json'
if identity_path.exists():
    previous = json.loads(identity_path.read_text(encoding='utf-8'))
    if previous != run_identity:
        raise RuntimeError('Existing experiment configuration differs; use a separate output directory.')
elif (OUT / 'protocol.json').exists():
    previous = json.loads((OUT / 'protocol.json').read_text(encoding='utf-8'))
    for key, value in run_identity.items():
        if key in previous and previous[key] != value:
            raise RuntimeError(f'Existing protocol differs at {key}; refuse checkpoint reuse.')
identity_path.write_text(json.dumps(run_identity, indent=2), encoding='utf-8')

class Logger:
    def __init__(self, folder):
        self.folder = folder
        folder.mkdir(parents=True, exist_ok=True)
    def log_scalar(self, **record):
        with (self.folder / 'training.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(record) + '\n')

class ExactDomainBatches(Sampler):
    """Equal old/new counts, sampling with replacement, matched update budget.
    Number of optimizer updates is exactly the baseline ceil((N+M)/B).
    This changes exposure per domain; that difference is explicitly reported.
    """
    def __init__(self, n_new, n_old, batch_size=64):
        self.n_new, self.n_old, self.batch_size = n_new, n_old, batch_size
        self.steps = math.ceil((n_new + n_old) / batch_size)
    def __len__(self):
        return self.steps
    def __iter__(self):
        for _ in range(self.steps):
            new = torch.randint(self.n_new, (self.batch_size // 2,)).tolist()
            old = (torch.randint(self.n_old, (self.batch_size // 2,)) + self.n_new).tolist()
            order = torch.randperm(self.batch_size).tolist()
            batch = new + old
            yield [batch[i] for i in order]

class ResearchFT(AuthorFT):
    def __init__(self, *a, replay=False, weighted=False, **kw):
        super().__init__(*a, **kw)
        self.replay, self.weighted, self.class_weights = replay, weighted, None
    def eval(self, t, loader):
        # GPU-compatible unweighted CE, identical target validation criterion.
        self.model.eval()
        loss_sum, hits, count = 0., 0, 0
        outputs_all, targets_all = [], []
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = torch.cat(self.model(x), dim=1)
                loss_sum += torch.nn.functional.cross_entropy(logits, y, reduction='sum').item()
                hits += (logits.argmax(1) == y).sum().item()
                count += len(y)
                outputs_all.extend(logits.cpu().tolist())
                targets_all.extend(y.cpu().tolist())
        return loss_sum/count, hits/count, hits/count, outputs_all, targets_all, []
    def criterion(self, t, outputs, targets, features=None, epoch=1):
        weights = self.class_weights if self.model.training else None
        return torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets, weight=weights)
    def train_loop(self, t, train, valid):
        if t == 0:
            return super().train_loop(t, train, valid)
        memory = self.exemplars_dataset
        combined = ConcatDataset([train.dataset, memory]) if len(memory) else train.dataset
        labels = list(train.dataset.labels) + list(memory.labels)
        if self.weighted:
            counts = np.bincount(labels, minlength=int(self.model.task_cls.sum()))
            # Predeclared sqrt inverse counts with additive smoothing and clipping.
            weights = (counts.mean() + 1) ** .5 / (counts + 1) ** .5
            weights = np.clip(weights, .2, 5.)
            self.class_weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
            self.logger.log_scalar(group='protocol', name='class_weights', value=weights.tolist())
        if self.replay:
            assert len(memory) > 0
            train = DataLoader(combined, batch_sampler=ExactDomainBatches(len(train.dataset), len(memory)))
        else:
            train = DataLoader(combined, batch_size=64, shuffle=True)
        # Bypass parent concatenation; preserve author's epoch and LR schedule.
        return Inc_Learning_Appr.train_loop(self, t, train, valid)
    def post_train_process(self, t, train, valid, correction=False):
        if t == 0:
            return super().post_train_process(t, train, valid)
        # Two domains only: no later task needs refreshed memory.
        return None

def write_json(path, value):
    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, np.generic):
            return x.item()
        return x
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2), encoding='utf-8')

def measure(model, loader, domain, folder):
    model.eval()
    ys, ps, logits_list = [], [], []
    with torch.no_grad():
        for x, y in loader:
            logits = torch.cat(model(x.to(device)), dim=1).cpu()
            ys.extend(y.tolist())
            ps.extend(logits.argmax(1).tolist())
            logits_list.extend(logits.numpy())
    labels = np.unique(ys)
    precision, recall, f1, support = precision_recall_fscore_support(ys, ps, labels=labels, zero_division=0)
    report = dict(domain=domain, n=len(ys), accuracy=float(accuracy_score(ys, ps)),
                  macro_f1=float(f1.mean()), macro_recall=float(recall.mean()),
                  macro_precision=float(precision.mean()), labels=labels.tolist(),
                  per_class_f1=f1.tolist(), per_class_recall=recall.tolist(), support=support.tolist(),
                  note='Macro averages over ground-truth domain classes; outside-domain predictions remain errors.')
    np.savez_compressed(folder / f'{domain}_predictions.npz', y_true=np.array(ys),
                        y_pred=np.array(ps), logits=np.array(logits_list))
    write_json(folder / f'{domain}_metrics.json', report)
    print('METRICS', domain, json.dumps(report), flush=True)
    return report

seed_everything(args.seed)
train, valid, test, taskcla = get_loaders([args.source, args.target], 1, 0, 0, 64, 0,
                                         validation=.1, num_pkts=10, fields=['PL','IAT','DIR','WIN'], seed=1)
# prep1 fixes the author train/test split. args.seed also affects the author's
# random class order and training-internal validation selection before training.
write_json(OUT / 'protocol.json', dict(date='2026-10-03', author_commit='842737730852c7c7c0955257c48ca72a22770343',
    dataset_sample_seed=1, training_seed=args.seed, author_index=args.author_index, device=device,
    torch=torch.__version__, max_epochs=args.epochs, lr=.1, lr_factor=3, lr_min=.0001, patience=20,
    batch_size=64, memory_per_class=25, replay_fraction=.5, weight_exponent=.5, weight_clip=[.2,5.],
    validation='Author 10% per-class from fixed training split; target-only unweighted CE checkpoint selection.',
    train_counts=[dict(Counter(d.dataset.labels)) for d in train],
    validation_counts=[dict(Counter(d.dataset.labels)) for d in valid],
    test_counts=[dict(Counter(d.dataset.labels)) for d in test], taskcla=taskcla,
    limitations=['Public downsampled preprocessed sample, not full data.',
                 'Changing run seed preserves prep1 train/test, but changes class order, training-internal validation and training randomness.',
                 'Engineering index correction and modern dependency/GPU changes disclosed.',
                 'Predeclared improvement parameters; no test-guided tuning.']))

seed_everything(args.seed)
model = LLL_Net(Lopez17CNN(num_pkts=10, num_fields=4))
model.add_head(taskcla[0][1])
model.to(device)
common = dict(nepochs=args.epochs, lr=.1, lr_min=.0001, lr_factor=3, lr_patience=20)
source_folder = OUT / 'source'
source_folder.mkdir(exist_ok=True)
memory = ExemplarsDataset(train[0].dataset.class_indices, num_exemplars_per_class=25)
source = ResearchFT(model, device, logger=Logger(source_folder), exemplars_dataset=memory, **common)
checkpoint = source_folder / 'checkpoint.pt'
if checkpoint.exists():
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(saved['model'])
    memory.images, memory.labels = saved['memory_images'], saved['memory_labels']
    print('Loaded existing source checkpoint', flush=True)
else:
    source.train(0, train[0], valid[0])
    torch.save(dict(model=model.state_dict(), memory_images=memory.images, memory_labels=memory.labels), checkpoint)
write_json(source_folder / 'memory.json', dict(count=len(memory), per_class=dict(Counter(memory.labels))))
measure(model, test[0], 'source', source_folder)
measure(model, test[1], 'target', source_folder)

source_model = copy.deepcopy(model)
shared_memory = copy.deepcopy(memory)
seed_everything(args.seed + 1000)
model.add_head(taskcla[1][1])
model.to(device)
expanded = model.get_copy()
results = {}
for method in args.methods:
    folder = OUT / method
    folder.mkdir(exist_ok=True)
    if (folder / 'complete.json').exists():
        results[method] = json.loads((folder / 'complete.json').read_text(encoding='utf-8'))
        continue
    seed_everything(args.seed + 2000)
    current = copy.deepcopy(model)
    current.set_state_dict(expanded)
    mem = copy.deepcopy(shared_memory) if method != 'ft' else ExemplarsDataset(None)
    approach = ResearchFT(current, device, logger=Logger(folder), exemplars_dataset=mem,
                          replay=method in ('replay', 'combined'), weighted=method in ('weighted', 'combined'), **common)
    print('START METHOD', method, flush=True)
    start = time.time()
    approach.train(1, train[1], valid[1])
    torch.save(current.state_dict(), folder / 'checkpoint.pt')
    result = dict(method=method, elapsed_seconds=time.time()-start, memory_size=len(mem),
                  source=measure(current, test[0], 'source', folder),
                  target=measure(current, test[1], 'target', folder))
    write_json(folder / 'complete.json', result)
    results[method] = result
    write_json(OUT / 'summary.json', results)
print('COMPLETE', OUT, flush=True)
