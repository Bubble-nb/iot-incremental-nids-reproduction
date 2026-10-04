from pathlib import Path
import shutil
import difflib

root = Path(__file__).resolve().parent
original = root / 'original_incremental_nids'
working = root / 'compatibility_incremental_nids'
shutil.copytree(original, working, dirs_exist_ok=True)
path = working / 'src/datasets/networking_dataset.py'
before = path.read_text(encoding='utf-8')
after = before.replace('    data = {}\n    taskcla = []', "    full_path = full_path.replace('\\\\', '/')\n    data = {}\n    taskcla = []")
after = after.replace("if not os.path.exists(full_path):\n        full_path = '../' + full_path",
    "if not os.path.exists(full_path) and not os.path.exists(full_path.replace('.parquet', f'_prep{seed}.parquet')):\n        full_path = '../' + full_path")
after = after.replace("labels=np.sort(np.unique(pd.read_parquet(full_path, columns=['LABEL'])['LABEL'].values))",
    "if os.path.exists(full_path):\n        labels = np.sort(np.unique(pd.read_parquet(full_path, columns=['LABEL'])['LABEL'].values))\n    else:\n        labels = np.loadtxt(os.path.join(path, f'classes_{dataset_filename[:-8]}.txt'), dtype=str, ndmin=1)\n        assert len(labels) == num_classes, 'Class metadata does not match encoded labels'")
assert before != after
path.write_text(after, encoding='utf-8')
(root / 'compatibility_changes.diff').write_text(''.join(difflib.unified_diff(
    before.splitlines(True), after.splitlines(True), fromfile='original/networking_dataset.py',
    tofile='compatibility/networking_dataset.py')), encoding='utf-8')
print('Prepared compatibility copy; only prepared-data path and class-name fallback changed.')
