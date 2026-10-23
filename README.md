# SeasLeaF

SeasLeaF is a dual-branch model for hierarchical time series forecasting.

## Project Structure

```text
SeasLeaF/
├── train_fusion.py
├── models/
│   └── seasleaf.py
└── data/
    ├── *_sfd.csv
    └── *_sfd_hier.csv
```

`train_fusion.py` handles training and evaluation, while `models/seasleaf.py` implements the model. Each data file is paired with a hierarchy file sharing the same prefix and ending in `_hier.csv`.

## Usage

Run from the project directory:

```bash
python train_fusion.py
```

By default, training uses `mixedhumid_sfd` with a lookback window of 120, a prediction horizon of 24, 100 epochs, and 3 runs. CUDA is used automatically when available.

Specify a dataset and training parameters:

```bash
python train_fusion.py --path data/cold_sfd --seq 120 --pred 24 --epochs 100 --runs 3
```

Run a quick smoke test:

```bash
python train_fusion.py --smoke
```

## Ablation Experiments

Ablation options include fusion strategies, calendar features, the gated template module, and loss configurations. Examples:

```bash
python train_fusion.py --fusion mean
python train_fusion.py --leaf-no-sfm
python train_fusion.py --no-calendar
python train_fusion.py --loss-scales all --loss-levels node
```

Model checkpoints, configurations, and evaluation results are saved under `outputs/seasleaf/`. Evaluation summaries are appended to `temp.txt`, or `smoke_temp.txt` in smoke-test mode.
