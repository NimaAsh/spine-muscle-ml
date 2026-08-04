# Standalone Inference

The `examples/infer_folder.py` script runs ML inference on a folder of `.osim`
files without requiring the full dataset build pipeline. It produces
subject-specific revised OpenSim models using the bundled ML checkpoints.

By default it uses the **paper's optimal hybrid approach**:
- **Random Forest** for force prediction (Native-Data conditioning)
- **Latent/Autoencoder** for geometry prediction (Anthropometrically-Normalized)

If RF checkpoints are not available, it falls back to Latent for both.

## Prerequisites

- The generic template models must be accessible (set `OPENSIM_MUSCLE_NN_DATA_ROOT`
  or pass `--data-root`).
- The bundled normalization stats (`datasets/stats.json`) and template muscle
  properties (`datasets/generic_muscle_properties.json`) must be present.
- At least one model checkpoint must exist in `models/` (`latent.pt`, or
  `mlp_force.pt` + `mlp_via.pt`).

## Usage

```bash
python examples/infer_folder.py \
    --input-dir /path/to/osims \
    --sex male \
    --age 55
```

## Arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--input-dir` | Yes | — | Directory containing `.osim` files to process |
| `--demographics-csv` | No | — | Path to a CSV file containing `ID`, `Sex`, and `Age` columns. The script reads the OSIM filename prefix (e.g. `123` from `123_model.osim`) to dynamically look up and assign the sex and age. If this is not provided, you must specify `--sex`. |
| `--sex` | No | — | Subject sex: `male` or `female`. Required unless `--demographics-csv` is used. |
| `--age` | No | `65.0` | Subject age in years. Overridden if `--demographics-csv` is used. |
| `--model` | No | `hybrid` | Model strategy: `hybrid` (RF force + Latent via), `latent`, or `mlp` |
| `--output-dir` | No | `<input-dir>/revised/` | Output directory for revised OSIMs |
| `--data-root` | No | `$OPENSIM_MUSCLE_NN_DATA_ROOT` | Path to dataverse files root |
| `--height` | No | auto | Subject height in meters |
| `--weight` | No | auto | Subject weight in kg |

## How Inputs Are Determined

### With explicit arguments

When `--age`, `--height`, or `--weight` are provided, they override the
auto-estimated values in the feature vector before normalization and inference.

### Without explicit arguments

| Input | Auto-estimation method |
|---|---|
| **Sex** | Always required via `--sex` (determines template) |
| **Age** | Defaults to 65.0 (the reference age) |
| **Height** | Estimated from head_neck Y-translation relative to sacrum |
| **Weight** | Estimated from sacrum mass ratio × nominal (78 kg male / 61 kg female) |

## Input File Requirements

- Files must be valid OpenSim `.osim` XML files.
- They do **not** need to follow any specific folder naming convention (no
  `Male/AgeXXXX/` structure required).
- The models should be thoracolumbar spine models with the standard body
  hierarchy (sacrum, lumbar1–5, thoracic1–12).

## Examples

```bash
# Default hybrid: RF for force + Latent for geometry (paper's best)
python examples/infer_folder.py \
    --input-dir ./my_osims \
    --sex male

# Batch processing with dynamic demographics:
python examples/infer_folder.py \
    --input-dir ./my_osims \
    --demographics-csv ./subject_data.csv

# Use only latent model for both force and geometry
python examples/infer_folder.py \
    --input-dir ./my_osims \
    --sex male \
    --model latent

# Full control: specify everything
python examples/infer_folder.py \
    --input-dir ./my_osims \
    --output-dir ./results \
    --sex female \
    --age 70 \
    --height 1.60 \
    --weight 55

# Use MLP model only
python examples/infer_folder.py \
    --input-dir ./my_osims \
    --sex male \
    --age 55 \
    --model mlp
```

## Output

Each input file `subject.osim` produces one output:

- `subject_revised_hybrid.osim` (default: RF force + Latent geometry)
- `subject_revised_latent.osim` (when `--model latent`)
- `subject_revised_mlp.osim` (when `--model mlp`)

Each output file includes an XML comment with provenance metadata (model used,
sex, age, height/weight source, original file path).

## Model Strategy

| Strategy | Force model | Geometry model | Notes |
|---|---|---|---|
| `hybrid` (default) | Random Forest | Latent/Autoencoder | Paper's best. Falls back to Latent if RF not found |
| `latent` | Latent | Latent | Best bundled neural-only option |
| `mlp` | MLP | MLP | Simpler baseline |

### Data Conditioning (Native vs. Anthropometrically-Normalized)

The paper found that different data conditioning works best for different tasks:

- **Force**: models trained on **Native-Data** targets (raw delta from template,
  `FORCE_BASELINE_SCALING=False`) — RF excels here
- **Geometry**: models trained on **Anthropometrically-Normalized** targets
  (residuals after height-scaling baseline) — Latent excels here

The current dataset is built with exactly these settings, so all models
already use the optimal conditioning for their respective tasks.

The Random Forest checkpoints are not bundled in the public repo. To enable
the full hybrid approach, train them locally:

```bash
export OPENSIM_MUSCLE_NN_DATA_ROOT=/path/to/dataverse_files
python examples/build_dataset.py
python examples/train_models.py  # produces models/rf_force.pkl + rf_via.pkl
```

See [models/README.md](../models/README.md) for metrics.
