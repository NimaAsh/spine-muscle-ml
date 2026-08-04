# Reproducibility

This repository keeps the publication-facing pipeline intentionally small.

## Typical Order

1. Set `OPENSIM_MUSCLE_NN_DATA_ROOT`.
2. Generate cached template muscle properties.
3. Build dataset JSON files.
4. Train MLP and latent models.
5. Write revised `.osim` outputs.
6. Export tables or compute prediction errors.

## Commands

```bash
python examples/cache_template_muscle_properties.py
python examples/build_dataset.py
python examples/train_models.py
python examples/train_latent.py
python examples/write_revised_models.py --split test
```

## Standalone Inference (No Dataset Build Required)

To run inference on new `.osim` files without rebuilding the dataset:

```bash
python examples/infer_folder.py \
    --input-dir /path/to/your/osims \
    --sex male \
    --age 55 \
    --model latent
```

See [standalone-inference.md](standalone-inference.md) for the full argument reference.

## Output Locations

- dataset metadata: `datasets/`
- trained checkpoints: `models/`
- generated OpenSim outputs: `outputs/` (dataset pipeline) or user-specified directory (standalone)

## Notes

- `datasets/train.json`, `datasets/test.json`, and `datasets/val.json` are not bundled in this public snapshot.
- The bundled checkpoints are neural models that were already present in the workspace.
- Random Forest inference is disabled by default in the public writer because the corresponding checkpoint files are not bundled here.
- The standalone inference script (`infer_folder.py`) requires `datasets/stats.json` and `datasets/generic_muscle_properties.json` but does not require the dataset JSON files.
