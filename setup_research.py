from pathlib import Path
import shutil, difflib

root = Path(__file__).resolve().parent
source = root / 'compatibility_incremental_nids'
target = root / 'research_incremental_nids'
if target.exists():
    raise SystemExit('Research copy already exists; do not overwrite.')
shutil.copytree(source, target, ignore=shutil.ignore_patterns('__pycache__'))
p = target / 'src/datasets/networking_dataset.py'
before = p.read_text(encoding='utf-8')
after = before.replace('df = pd.read_parquet(prep_df_path)\n',
                       'df = pd.read_parquet(prep_df_path).reset_index(drop=True)\n')
assert before != after
p.write_text(after, encoding='utf-8')
(root / 'research_index_fix.diff').write_text(''.join(difflib.unified_diff(
    before.splitlines(True), after.splitlines(True),
    fromfile='compatibility/src/datasets/networking_dataset.py',
    tofile='research/src/datasets/networking_dataset.py')), encoding='utf-8')
print(target)
