from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_step(title: str, script: Path, *args: str) -> None:
    print(f"\n=== {title} ===")
    cmd = [sys.executable, str(script), *args]
    print("$", " ".join(cmd))
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise SystemExit(f"Step failed ({title}) with exit code {proc.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run dataset build and model trainings in sequence")
    # By default we DO write revised OSIM files. Use --no-write to skip the writer step.
    parser.add_argument("--no-write", action="store_true", help="Skip writing revised OSIM files (default is to write)")
    parser.add_argument("--writer-split", choices=["train", "test", "all"], default="test", help="Split to export in writer")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent
    examples = repo / "examples"

    # Required steps
    run_step("Build dataset", examples / "build_dataset.py")
    run_step("Train MLP models", examples / "train_models.py")
    run_step("Train Transformer", examples / "train_transformer.py")
    run_step("Train Latent (AE+Regressor)", examples / "train_latent.py")
    run_step("Train Spine GNN", examples / "train_gnn.py")

    # Optional: write revised models (default ON; skip with --no-write)
    if not args.no_write:
        run_step("Write revised OSIMs", examples / "write_revised_models.py", "--split", args.writer_split)

    print("\nAll steps completed successfully.")


if __name__ == "__main__":
    main()

