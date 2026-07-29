# HY Scoring

This folder contains the HY scoring model training and inference code.

Public release status:
- `HY_score_without_real_data.csv` is the public synthetic HY-turn training file currently included in this repository. It contains 12,193 synthetic HY assessment-turn records.
- In the full experimental setting described in the paper, these 12,193 synthetic records were combined with 507 non-public real clinical records to form a 12,700-record balanced HY scoring resource.
- Real clinical data and 5-fold private evaluation CSV files are not included.
- Model checkpoints are not committed. Publish an inference-only checkpoint separately, for example through Hugging Face or GitHub Releases, if reviewers should run inference without retraining.

Environment:
- Install dependencies with `pip install -r requirements.txt` from this folder, or `pip install -r HY_scoring/requirements.txt` from the repository root.
- For GPU training, install a CUDA-compatible PyTorch build according to your local CUDA version.

Files:
- `5_fold_valid_train_regression_divideHY.py`: training script. The default `crossval_dir` points to `private_real_data/`, which should be supplied locally only if authorized.
- `inf_regression.py`: batch inference script.
- `HY_Scoring/HY_Scoring.py`: inference engine used by the dialogue system.
- `HY_Scoring/label_mappings.json`: valid score labels for each public text-based HY item.
- `inf_regression.csv`: small synthetic inference example.
