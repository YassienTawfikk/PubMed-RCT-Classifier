# PubMed-RCT-Classifier

## Overview

This project implements a custom Transformer model to classify sentences from medical abstracts (PubMed 200k RCT dataset) into the following categories:

- `BACKGROUND`
- `OBJECTIVE`
- `METHODS`
- `RESULTS`
- `CONCLUSIONS`

## Project Structure

- `classifier_core/`: Modularized Python package containing the core logic (modeling, data, evaluation).
- `run/`: Shell scripts for execution (`run_experiment.sh`).
- `script/`: Entry point scripts (`training.py`, `evaluate.py`).
- `notebook/`: Demonstration notebook (`PubMed_RCT_Demo.ipynb`).
- `pubmed_rct/`: Data directory (downloaded automatically).

## Installation

1. Clone the repository.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   (Requires `tensorflow`, `jsonargparse`, `matplotlib`, `seaborn`, etc.)

## Usage

### Run Entire Experiment

To run the full pipeline (Training -> Evaluation):

```bash
./run/run_experiment.sh
```

### Run Scripts Individually

You can also run the python scripts directly with CLI arguments (powered by `jsonargparse`):

**Training:**

```bash
python script/training.py --train.epochs 5 --data.batch_size 32
```

**Evaluation:**

```bash
python script/evaluate.py --model_path path/to/model.keras
```

## Visualizations

The pipeline automatically generates and saves the following plots to the model directory:

- `loss_curve.png`: Training vs Validation loss.
- `accuracy_curve.png`: Training vs Validation accuracy.
- `confusion_matrix.png`: Model performance across classes.

## Notebook

The `notebook/PubMed_RCT_Demo.ipynb` provides an interactive walkthrough of the pipeline using the `classifier_core` package.
