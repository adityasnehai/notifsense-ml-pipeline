# NotifSense ML Pipeline

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU%2FGPU-EE4C2C?logo=pytorch&logoColor=white)
![Model](https://img.shields.io/badge/Model-Multi--label%20MLP-0A7EA4)
![Sequence Model](https://img.shields.io/badge/DeepConvLSTM-Supported-4B8BBE)
![Export](https://img.shields.io/badge/Android-TorchScript%20%2F%20PTL-3DDC84)
![Dataset](https://img.shields.io/badge/Dataset-ExtraSensory-6A5ACD)

Production-oriented ML pipeline for **mobile context recognition** in NotifSense.

This repository covers:
- ExtraSensory preprocessing for sensor-compatible features.
- Multi-label MLP training with imbalance handling, label pruning, and threshold tuning.
- DeepConvLSTM reference model + strict evaluator.
- PyTorch-only mobile export path (dynamic INT8 + TorchScript/PTL).

## Why This Repo Exists

NotifSense needs reliable, low-latency on-device context understanding before Android policy decisions.
This pipeline isolates and validates the full ML lifecycle outside the app codebase.

## Project Structure

```text
notifsense/
  configs/
    config.yaml
  data/
    raw/
    processed/
    splits/
  models/
    deepconvlstm.py
    export_onnx.py
    inference.py
    mlp_model.py
    quantize_and_export.py
  scripts/
    prepare_data.py
    prepare_extrasensory.py
    evaluate_deepconvlstm.py
  utils/
  train.py
  requirements.txt
```

## Data + Label Design

Sensor feature strategy is constrained to practical phone signals (no GPS/audio/WiFi/Bluetooth/watch-only features).

Current processed metadata:
- Selected input features: **79**
- Initial label taxonomy in processed CSV: **24 labels**
- Final trained checkpoint label set (after pruning): **9 labels**

Final labels in `models/notifsense_final.pt`:
- `Stationary`
- `Sleeping`
- `LyingDown`
- `Indoors`
- `Sitting`
- `PhoneOnTable`
- `Walking`
- `ComputerWork`
- `Moving`

## Model Stack

### 1) Primary model for deployment: `NotifSenseMLP`

Defined in `models/mlp_model.py`:
- `Linear(input_dim -> 256) + BatchNorm + ReLU + Dropout(0.3)`
- `Linear(256 -> 128) + BatchNorm + ReLU + Dropout(0.3)`
- `Linear(128 -> 64) + ReLU`
- `Linear(64 -> num_labels)`

Training choices in `train.py`:
- `BCEWithLogitsLoss` with `pos_weight` for class imbalance
- Optional focal loss (`use_focal_loss` in config)
- Early stopping + LR scheduling
- Automatic label pruning by validation F1/support
- Per-label threshold tuning (`data/processed/label_thresholds.json`)

### 2) Sequence reference model: `DeepConvLSTM`

Defined in `models/deepconvlstm.py`:
- `Conv1d -> ReLU -> Conv1d -> ReLU -> LSTM -> Linear`
- Outputs logits (sigmoid applied only at inference/eval)

## Quick Start

### 1) Environment

```bash
cd /home/aditya/projects/notifsense
python3 -m venv notifsense_env
source notifsense_env/bin/activate
pip install -r requirements.txt
```

### 2) Prepare ExtraSensory data

Expected ZIP location:
- `dataset/ExtraSensory.per_uuid_features_labels.zip`

Run:

```bash
python scripts/prepare_data.py
```

Outputs:
- `data/processed/train.csv`
- `data/processed/val.csv`
- `data/processed/test.csv`
- `data/processed/metadata.json`

### 3) Train optimized MLP

```bash
python train.py --config configs/config.yaml
```

Primary artifact:
- `models/notifsense_final.pt`

### 4) Evaluate trained MLP checkpoint

```bash
python - <<'PY'
import numpy as np, pandas as pd, torch
from sklearn.metrics import f1_score
from models.mlp_model import NotifSenseMLP

ckpt = torch.load('models/notifsense_final.pt', map_location='cpu', weights_only=False)
features = ckpt['feature_columns']
labels = ckpt['label_columns']
mean = np.asarray(ckpt['feature_mean'], dtype=np.float32)
std = np.asarray(ckpt['feature_std'], dtype=np.float32)
std = np.where(std < 1e-6, 1.0, std)


df = pd.read_csv('data/processed/test.csv', usecols=features + labels)
X = df[features].to_numpy(dtype=np.float32)
Y = (df[labels].to_numpy(dtype=np.float32) > 0).astype(np.int32)
X = ((X - mean) / std).astype(np.float32)

model = NotifSenseMLP(int(ckpt['input_dim']), int(ckpt['num_labels']))
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
with torch.no_grad():
    probs = torch.sigmoid(model(torch.from_numpy(X))).numpy()

thresholds = np.array([float(ckpt['thresholds'].get(lbl, 0.5)) for lbl in labels], dtype=np.float32)
pred = (probs >= thresholds.reshape(1, -1)).astype(np.int32)

print('Macro F1:', f1_score(Y, pred, average='macro', zero_division=0))
print('Micro F1:', f1_score(Y, pred, average='micro', zero_division=0))
PY
```

### 5) Quantize and export for Android (PyTorch-only)

```bash
python models/quantize_and_export.py \
  --checkpoint models/notifsense_final.pt \
  --eval-csv data/processed/test.csv
```

Generated mobile artifacts:
- `models/notifsense_quantized.pt`
- `models/notifsense_quantized_scripted.pt`
- `models/notifsense_android.ptl`

### 6) DeepConvLSTM evaluation (optional path)

Use this only with a real DeepConvLSTM checkpoint (not MLP checkpoints):

```bash
python scripts/evaluate_deepconvlstm.py \
  --config configs/config.yaml \
  --checkpoint models/deepconvlstm_best.pt \
  --metadata data/processed/metadata.json \
  --split-csv data/processed/test.csv \
  --batch-size 256 \
  --threshold 0.5 \
  --output data/processed/deepconvlstm_eval.json
```

The evaluator prints and saves:
- Macro F1, Micro F1, subset accuracy
- Per-label precision/recall/F1/support (sorted by F1)
- Per-label confusion matrices
- Full classification report

## Latest Local Evaluation Snapshot

Checkpoint: `models/notifsense_final.pt` on `data/processed/test.csv`

- Macro F1: **0.5578**
- Micro F1: **0.6367**
- Subset accuracy: **0.0880**

Selected per-label F1:
- `Stationary`: **0.8227**
- `LyingDown`: **0.7199**
- `Sitting`: **0.6686**
- `Indoors`: **0.6255**
- `Walking`: **0.4513**
- `ComputerWork`: **0.2378**

## Artifacts Reference

- Final model: `models/notifsense_final.pt`
- Label thresholds: `data/processed/label_thresholds.json`
- Dataset metadata: `data/processed/metadata.json`
- UUID split info: `data/splits/uuid_splits.json`
- Android PTL model: `models/notifsense_android.ptl`

## Notes

- This repo intentionally does **not** track large CSV/model binaries in git.
- `models/notifsense_best.pt` and `models/notifsense_final.pt` are MLP checkpoints.
- DeepConv scripts now fail fast if an MLP checkpoint is passed by mistake.

## Dataset Reference

- ExtraSensory: http://extrasensory.ucsd.edu/
- Download used: `ExtraSensory.per_uuid_features_labels.zip`
