# Spine Muscle ML

Code accompanying the paper:

**Machine Learning Outperforms Anthropometric Scaling in Predicting Muscle Parameters and Spinal Loading: A Subject-Specific Musculoskeletal Modeling Study**

Status: accepted in *European Spine Journal*.

This repository packages the core code used to:

- parse and edit OpenSim `.osim` models
- build supervised learning datasets from subject-specific models
- train muscle force and via-point predictors
- write revised OpenSim models from predicted muscle parameters
- evaluate predictions against ground-truth models

## Source Data

The 250 subject-specific spine models used in this work come from:

Anderson, D., Mokhtarzadeh, H., Allaire, B., Burkhart, K., Bouxsein, M., 2020.
*Subject-specific spine models for 250 individuals from the Framingham Heart Study*.
[https://doi.org/10.7910/DVN/SJ5MVM](https://doi.org/10.7910/DVN/SJ5MVM)

## Included

- `osim_parser/`: lightweight OpenSim XML parser/editor
- `examples/`: curated training, inference, and evaluation scripts (including standalone `infer_folder.py`)
- `models/`: bundled neural checkpoints and evaluation metrics available in the current workspace
- `datasets/`: normalization metadata and cached generic-template muscle properties
- `docs/`: usage, reproducibility notes, and [standalone inference guide](docs/standalone-inference.md)

## Not Included

- raw CT-derived subject data
- generated scratch outputs and exploratory analysis files
- duplicate model folders and cached bytecode
- bundled Random Forest checkpoints for the accepted-paper hybrid model

The writer script is configured to use the bundled neural checkpoints by default. If you retrain and generate Random Forest checkpoints locally, you can re-enable that path.

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[torch]"
```

Set the data root to the directory that contains your `Male/`, `Female/`, and `generic/` folders:

```bash
export OPENSIM_MUSCLE_NN_DATA_ROOT=/path/to/dataverse_files
```

The expected layout is documented in [docs/data-layout.md](docs/data-layout.md).

## Quick Start

Build cached template properties:

```bash
python examples/cache_template_muscle_properties.py
```

Build the training dataset:

```bash
python examples/build_dataset.py
```

Train the bundled learning baselines:

```bash
python examples/train_models.py
python examples/train_latent.py
```

Write revised OpenSim models for the test split:

```bash
python examples/write_revised_models.py --split test
```

Run the end-to-end pipeline:

```bash
python run_all.py
```

## Standalone Inference (New OSIM Files)

To run inference on your own `.osim` files without rebuilding the dataset:

```bash
# Example 1: explicit age and sex
python examples/infer_folder.py \
    --input-dir /path/to/your/osims \
    --sex male \
    --age 55

# Example 2: read dynamic age and sex from a demographic CSV file
python examples/infer_folder.py \
    --input-dir /path/to/your/osims \
    --demographics-csv /path/to/demographics.csv
```

By default this uses the paper's best hybrid approach (RF for force + Latent
for geometry). It accepts explicit arguments or dynamic lookups from a CSV and
does not require the `Male/AgeXXXX/` folder convention. See
[docs/standalone-inference.md](docs/standalone-inference.md) for full details.

## Bundled Artifacts

The repository includes:

- `models/latent.pt`
- `models/mlp_force.pt`
- `models/mlp_via.pt`
- `models/metrics.json`
- `datasets/stats.json`
- `datasets/generic_muscle_properties.json`

From the bundled metrics snapshot, the latent model is the best included checkpoint in this repo:

- baseline absolute test MSE: force `781.57`, via `14.49`
- latent absolute test MSE: force `413.52`, via `6.26`

See [models/README.md](models/README.md) for details.

## Reproducibility

Reproduction notes and script order are in [docs/reproducibility.md](docs/reproducibility.md).

## Citation

Citation and funding information are in [CITATION.md](CITATION.md).

## License

MIT. See [LICENSE](LICENSE).
