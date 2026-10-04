# IoT incremental NIDS: reproduction and ablations

This project reproduces selected methods on the author's public downsampled prep1 features. It does not reproduce the complete paper or rebuild all raw features.

## Attribution

Author repository: https://github.com/jmpr0/incremental-nids

Fixed commit: 842737730852c7c7c0955257c48ca72a22770343

Paper DOI: 10.1016/j.engappai.2025.110143. Preserve the author's MIT license and attribution (AUTHOR_LICENSE). The author code is downloaded rather than republished in this package. Added experiment scripts are provided under the MIT license in LICENSE. See NOTICE.md for the separation of original, compatibility, index correction and research changes.

## Preparation

Use Python 3.11. The observed package versions are recorded in requirements-observed.txt. Choose the matching official PyTorch CPU or CUDA distribution for your hardware; installing the recorded versions does not by itself guarantee identical hardware results.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-observed.txt
.venv\Scripts\python.exe bootstrap_author_code.py
```

Bootstrap downloads the pinned public author repository, verifies the fixed source hashes, and creates separate compatibility and index-corrected research copies. Existing directories are preserved. Download and one-epoch execution of all methods were exercised in an empty directory using the existing local Python environment; a separate fresh dependency installation has not been tested.

## Experiments

```powershell
.venv\Scripts\python.exe -u run_research.py --source iot_nidd --target ton_iot --seed 1 --epochs 200
.venv\Scripts\python.exe save_label_mapping.py --source iot_nidd --target ton_iot --seed 1
.venv\Scripts\python.exe -u run_research.py --source ton_iot --target iot_nidd --seed 1 --epochs 200
.venv\Scripts\python.exe save_label_mapping.py --source ton_iot --target iot_nidd --seed 1
```

Repeat the forward commands with seeds 2 and 3 to match the completed run set. The run seed also changes class ordering and training-internal validation sampling; prep1 train/test membership remains fixed. Different seeds are not independent raw test splits.

FT and FT-Mem reuse the author training framework. Replay changes batches to 32 target and 32 memory samples, with the same number of updates as FT-Mem but a different exposure distribution. Weighted uses smoothed square-root class weights clipped to [0.2, 5]. Combined applies both. These are established techniques, not claimed as globally novel methods.

## Results and limitations

Each run stores checkpoints, per-sample predictions, logits, per-class metrics, validation logs and actual memory size. Adaptation uses target-validation unweighted CE for model selection. Existing configurations are checked before checkpoint reuse.

The initial combined-method gains did not consistently repeat. Report every run, false-positive rates and retention costs. Do not tune parameters on the test set. The later threshold exploration was exploratory, designed after inspecting the first run, and is outside this minimal training package.

This package excludes raw traffic, weights, private credentials, personal student information and downloaded third-party papers. The README describes the completed local research; it does not claim that the reduced demonstration reproduces published performance or that every dependency installation has been verified.
