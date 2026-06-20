# BHHOSCA Feature Selection Script

This project now includes a standalone Python script, `algorithm.py`, that runs the notebook logic for the single `bHHOSCA` algorithm without opening Jupyter.

## Files

- `algorithm.py`: standalone CLI script for BHHOSCA.
- `requirements.txt`: Python dependencies.

## Setup

1. Create and activate a virtual environment.
2. Install the dependencies:

```bash
python3 -m pip install -r requirements.txt
```

## Important note

`algorithm.py` downloads datasets from the UCI Machine Learning Repository through `ucimlrepo`, so you need an internet connection when you run it.

## Supported datasets

Run this to see the built-in dataset ids:

```bash
python3 algorithm.py --list-datasets
```

Current dataset ids:

- `17`: Breast Cancer
- `52`: Ionosphere
- `78`: Page Blocks
- `81`: Digits
- `94`: Spambase
- `101`: Tic-tac-toe
- `109`: Wine
- `336`: Kidney
- `728`: Toxicity
- `732`: Darwin

## Run examples

Run the default dataset (`17`, Breast Cancer):

```bash
python3 algorithm.py
```

Run a specific dataset:

```bash
python3 algorithm.py --dataset-id 109
```

Run all built-in datasets:

```bash
python3 algorithm.py --all-datasets
```

Use custom optimization settings:

```bash
python3 algorithm.py --dataset-id 17 --runs 10 --epoch 100 --pop-size 10
```

Reduce CPU usage if needed:

```bash
python3 algorithm.py --dataset-id 17 --cv-jobs 1 --run-jobs 1
```

## CLI options

- `--dataset-id`: UCI dataset id to run. Default is `17`.
- `--all-datasets`: run every built-in dataset.
- `--list-datasets`: print supported dataset ids and exit.
- `--runs`: number of repeated BHHOSCA runs.
- `--epoch`: number of optimizer epochs.
- `--pop-size`: optimizer population size.
- `--neighbors`: KNN neighbors for evaluation.
- `--cv-splits`: number of cross-validation folds.
- `--cv-jobs`: parallel jobs used by `cross_val_score`.
- `--run-jobs`: parallel jobs used across repeated BHHOSCA runs.
- `--seed`: random seed.

## Output

The script prints:

- average accuracy
- accuracy standard deviation
- average number of selected features
- average fitness
- fitness standard deviation
- best fitness
- worst fitness

## Quick reminder

Some datasets can take a long time to finish, especially with:

- high `--runs`
- high `--epoch`
- `--all-datasets`
- full CPU parallelism from `--cv-jobs -1`
