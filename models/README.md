# Model Artifacts

Bundled files:

- `latent.pt`: latent autoencoder + regressor checkpoint
- `mlp_force.pt`: MLP checkpoint for force prediction
- `mlp_via.pt`: MLP checkpoint for via-point prediction
- `metrics.json`: evaluation summary from the working repository snapshot

Selected metrics from `metrics.json`:

- baseline absolute test MSE: force `781.57`, via `14.49`
- MLP absolute test MSE: force `415.52`, via `6.48`
- latent absolute test MSE: force `413.52`, via `6.26`

Notes:

- Random Forest checkpoint files are not bundled in this public snapshot.
- The writer script therefore defaults to the bundled neural checkpoints.
