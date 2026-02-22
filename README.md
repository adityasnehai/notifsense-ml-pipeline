# NotifSense

NotifSense is an Android-focused AI system for multi-label context understanding from mobile sensor streams.
This repository provides the initial training pipeline scaffold for a DeepConvLSTM model using the ExtraSensory data format in later phases.

## Project Purpose

- Build a robust multi-label activity/context classifier for sensor time series.
- Validate the full offline training pipeline before integrating real dataset loaders.
- Prepare the model for ONNX export, quantization, and Android deployment.

## Repository Structure

```text
notifsense/
  data/
    raw/
    processed/
    splits/
  models/
    deepconvlstm.py
    inference.py
    export_onnx.py
  configs/
    config.yaml
  utils/
    windowing.py
    dataset.py
    preprocessing.py
    metrics.py
  scripts/
  notebooks/
  requirements.txt
  train.py
  README.md
```

## Architecture Overview

Current baseline model (`models/deepconvlstm.py`):

1. Conv1d (`input_channels -> 64`, `kernel_size=5`)
2. ReLU
3. Conv1d (`64 -> 128`, `kernel_size=5`)
4. ReLU
5. LSTM (`input_size=128`, `hidden_size=128`, `batch_first=True`)
6. Linear (`128 -> num_labels`)

The model outputs logits. Apply sigmoid externally for probabilities.

## Phase Roadmap

1. Phase 1: Scaffold and sanity checks with dummy data (current).
2. Phase 2: Add ExtraSensory ingestion, cleaning, and train/val/test splits.
3. Phase 3: Full training and evaluation with real labels and metrics tracking.
4. Phase 4: Export to ONNX, quantize, benchmark latency on Android hardware.
5. Phase 5: Integrate with Android app inference runtime and notification intelligence logic.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Training Sanity Check

```bash
python train.py --config configs/config.yaml
```

This runs:

- device detection (`cuda` when available, otherwise `cpu`)
- model initialization and summary print
- dummy forward pass with input shape `(8, 6, 100)`
- `BCEWithLogitsLoss` computation against dummy multi-label targets
- placeholder epoch loop to validate optimization flow

## Inference

Use `models/inference.py` to load weights and return sigmoid probabilities from logits.

## ONNX Export

```bash
python models/export_onnx.py \
  --config configs/config.yaml \
  --checkpoint path/to/checkpoint.pt \
  --output models/deepconvlstm.onnx
```

## Future Android Deployment Plan

- Convert trained checkpoint to ONNX and validate parity with PyTorch outputs.
- Apply post-training quantization (or QAT if needed).
- Evaluate on-device latency and memory on representative Android devices.
- Integrate ONNX Runtime Mobile (or equivalent runtime) into the Android app.
- Add confidence gating and user-feedback loops for continuous model improvements.

## DeepConvLSTM Evaluation (Added)

Use the dedicated evaluator for the DeepConvLSTM model:

```bash
/home/aditya/projects/notifsense/notifsense_env/bin/python scripts/evaluate_deepconvlstm.py \
  --config configs/config.yaml \
  --checkpoint models/deepconvlstm_best.pt \
  --metadata data/processed/metadata.json \
  --split-csv data/processed/test.csv \
  --batch-size 256 \
  --threshold 0.5 \
  --output data/processed/deepconvlstm_eval.json
```

What it prints/saves:

- Macro F1
- Micro F1
- Subset accuracy
- Per-label precision/recall/F1/support (sorted by F1)
- Per-label confusion matrix (`tn/fp/fn/tp`)
- Full sklearn classification report
- JSON output file at `data/processed/deepconvlstm_eval.json`

Notes:

- This evaluator is strictly for `DeepConvLSTM` checkpoints.
- If you pass an MLP checkpoint (for example files with `net.*` keys), it will stop with a clear error message.
- Your currently existing `models/notifsense_best.pt` and `models/notifsense_final.pt` are MLP checkpoints, so use a DeepConv checkpoint path here.

## Re-run Existing Model Evaluation Results

If you already have earlier DeepConv evaluation results, keep them in:

- `data/processed/deepconvlstm_eval.json`

This keeps the metric artifact versioned and reproducible.
