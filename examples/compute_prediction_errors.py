#!/usr/bin/env python3
"""Compute errors between predicted and ground truth OSIM files.

This script:
1. Loads ground truth OSIM files from dataverse
2. Loads predicted OSIM files (e.g., from outputs)
3. Matches by patient ID
4. Computes force errors (predicted - ground_truth) for each muscle
5. Computes via point errors (average Euclidean distance) for each muscle
6. Exports error CSV files
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
if str(REPO_ROOT / "examples") not in sys.path:
    sys.path.append(str(REPO_ROOT / "examples"))

from osim_parser import OSIMModel
from export_osim_muscle_table import process_directory, _get_patient_id_from_path
from repo_config import female_template_path, get_data_root, male_template_path


DATAVERSE_ROOT = get_data_root()


def find_ground_truth_files(dataverse_root: Path | None = DATAVERSE_ROOT) -> Dict[str, Path]:
    """Find all ground truth OSIM files and map by patient ID.

    Returns:
        Dict mapping patient_id (e.g., "110") to file path
    """
    ground_truth_map: Dict[str, Path] = {}

    if dataverse_root is None:
        raise RuntimeError("Set OPENSIM_MUSCLE_NN_DATA_ROOT before computing prediction errors.")

    for osim_file in dataverse_root.rglob("*.osim"):
        # Skip generic templates
        if "generic" in osim_file.as_posix().lower():
            continue

        patient_id = _get_patient_id_from_path(osim_file)
        if patient_id and patient_id not in ground_truth_map:
            ground_truth_map[patient_id] = osim_file

    return ground_truth_map


def process_ground_truth_directory(
    dataverse_root: Path | None = DATAVERSE_ROOT,
    template_male: Optional[Path] = None,
    template_female: Optional[Path] = None,
) -> pd.DataFrame:
    """Process ground truth OSIM files from dataverse into a DataFrame.

    Similar to process_directory but specifically for ground truth files.
    """
    # Find all ground truth files
    ground_truth_map = find_ground_truth_files(dataverse_root)
    paths = sorted(ground_truth_map.values())

    if not paths:
        raise FileNotFoundError(f"No ground truth .osim files found in {dataverse_root}")

    print(f"Found {len(paths)} ground truth OSIM files")

    # Use the existing process_directory by creating a temporary directory structure
    # or just process directly
    from export_osim_muscle_table import collect_muscle_specs, build_row

    muscles, via_len = collect_muscle_specs(paths)

    rows = []
    template_cache: Dict[str, OSIMModel] = {}

    for p in paths:
        try:
            model = OSIMModel.from_file(p)
        except Exception as e:
            print(f"Skip {p} due to {e}")
            continue

        # Determine template
        template = None
        sex = "M" if "/Male/" in p.as_posix() else ("F" if "/Female/" in p.as_posix() else "")

        if template_male and template_female:
            key = "male" if sex == "M" else ("female" if sex == "F" else None)
            tpl_path = template_male if key == "male" else (template_female if key == "female" else None)

            if tpl_path and tpl_path.exists() and key:
                if key not in template_cache:
                    template_cache[key] = OSIMModel.from_file(tpl_path)
                template = template_cache.get(key)

        rows.append(build_row(model, p, muscles, via_len, template))

    df = pd.DataFrame(rows)

    # Order columns same way as process_directory
    column_order = ["name", "filepath", "height", "weight", "sex"]
    for mus in muscles:
        column_order.append(f"force_{mus}")
        for i in range(via_len[mus]):
            column_order.extend(
                [f"via_{mus}_pt{i}_x", f"via_{mus}_pt{i}_y", f"via_{mus}_pt{i}_z"]
            )

    df = df.reindex(columns=column_order)
    return df


def compute_force_errors(df_pred: pd.DataFrame, df_gt: pd.DataFrame, muscles: List[str]) -> pd.DataFrame:
    """Compute force errors (predicted - ground_truth) for each muscle.

    Returns:
        DataFrame with columns: name, filepath, muscle, force_pred, force_gt, force_error
    """
    rows = []

    # Match by patient ID (name column)
    for idx_pred, row_pred in df_pred.iterrows():
        patient_id = row_pred["name"]

        # Find matching ground truth
        gt_match = df_gt[df_gt["name"] == patient_id]

        if len(gt_match) == 0:
            print(f"Warning: No ground truth found for patient {patient_id}")
            continue

        row_gt = gt_match.iloc[0]

        for muscle in muscles:
            force_col = f"force_{muscle}"

            if force_col not in df_pred.columns or force_col not in df_gt.columns:
                continue

            force_pred = row_pred.get(force_col, np.nan)
            force_gt = row_gt.get(force_col, np.nan)

            # Skip if either is NaN
            if pd.isna(force_pred) or pd.isna(force_gt):
                continue

            force_error = float(force_pred) - float(force_gt)
            force_error_pct = (force_error / force_gt * 100) if force_gt != 0 else np.nan

            rows.append({
                "patient_id": patient_id,
                "filepath_pred": row_pred["filepath"],
                "filepath_gt": row_gt["filepath"],
                "muscle": muscle,
                "force_pred": float(force_pred),
                "force_gt": float(force_gt),
                "force_error": force_error,
                "force_error_abs": abs(force_error),
                "force_error_pct": force_error_pct,
                "force_error_pct_abs": abs(force_error_pct) if not pd.isna(force_error_pct) else np.nan,
            })

    return pd.DataFrame(rows)


def compute_via_errors(df_pred: pd.DataFrame, df_gt: pd.DataFrame, muscles: List[str], via_len: Dict[str, int]) -> pd.DataFrame:
    """Compute via point position errors (average Euclidean distance) for each muscle.

    Returns:
        DataFrame with columns: name, filepath, muscle, num_points, avg_distance_error, ...
    """
    rows = []

    for idx_pred, row_pred in df_pred.iterrows():
        patient_id = row_pred["name"]

        # Find matching ground truth
        gt_match = df_gt[df_gt["name"] == patient_id]

        if len(gt_match) == 0:
            continue

        row_gt = gt_match.iloc[0]

        for muscle in muscles:
            num_points = via_len.get(muscle, 0)

            if num_points == 0:
                continue

            distances = []

            for i in range(num_points):
                x_col = f"via_{muscle}_pt{i}_x"
                y_col = f"via_{muscle}_pt{i}_y"
                z_col = f"via_{muscle}_pt{i}_z"

                # Check if columns exist
                if not all(col in df_pred.columns and col in df_gt.columns for col in [x_col, y_col, z_col]):
                    continue

                x_pred = row_pred.get(x_col, np.nan)
                y_pred = row_pred.get(y_col, np.nan)
                z_pred = row_pred.get(z_col, np.nan)

                x_gt = row_gt.get(x_col, np.nan)
                y_gt = row_gt.get(y_col, np.nan)
                z_gt = row_gt.get(z_col, np.nan)

                # Skip if any coordinate is NaN
                if any(pd.isna(v) for v in [x_pred, y_pred, z_pred, x_gt, y_gt, z_gt]):
                    continue

                # Compute Euclidean distance
                dx = float(x_pred) - float(x_gt)
                dy = float(y_pred) - float(y_gt)
                dz = float(z_pred) - float(z_gt)
                distance = np.sqrt(dx**2 + dy**2 + dz**2)

                distances.append(distance)

            if not distances:
                continue

            # Average distance across all points for this muscle
            avg_distance = float(np.mean(distances))
            max_distance = float(np.max(distances))
            min_distance = float(np.min(distances))

            rows.append({
                "patient_id": patient_id,
                "filepath_pred": row_pred["filepath"],
                "filepath_gt": row_gt["filepath"],
                "muscle": muscle,
                "num_points": len(distances),
                "avg_distance_error_m": avg_distance,
                "avg_distance_error_mm": avg_distance * 1000.0,
                "max_distance_error_m": max_distance,
                "max_distance_error_mm": max_distance * 1000.0,
                "min_distance_error_m": min_distance,
                "min_distance_error_mm": min_distance * 1000.0,
            })

    return pd.DataFrame(rows)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compute prediction errors against ground truth")
    parser.add_argument("--pred-dir", type=Path, required=True, help="Directory with predicted OSIM files")
    parser.add_argument("--gt-dir", type=Path, default=DATAVERSE_ROOT, help="Ground truth directory")
    parser.add_argument("--output-prefix", type=str, default="error", help="Prefix for output CSV files")
    parser.add_argument("--output-dir", type=Path, default=Path("."), help="Output directory for CSV files")

    args = parser.parse_args()

    # Template paths
    data_root = args.gt_dir.resolve()
    template_male = male_template_path(data_root)
    template_female = female_template_path(data_root)

    print("=" * 80)
    print("Loading ground truth OSIM files...")
    print("=" * 80)
    df_gt = process_ground_truth_directory(args.gt_dir, template_male, template_female)
    print(f"Loaded {len(df_gt)} ground truth samples")
    print(f"Ground truth patient IDs: {sorted(df_gt['name'].unique())[:10]}...")

    print("\n" + "=" * 80)
    print("Loading predicted OSIM files...")
    print("=" * 80)
    df_pred = process_directory(args.pred_dir, template_male, template_female)
    print(f"Loaded {len(df_pred)} predicted samples")
    print(f"Predicted patient IDs: {sorted(df_pred['name'].unique())[:10]}...")

    # Find common patient IDs
    common_ids = set(df_pred["name"].unique()) & set(df_gt["name"].unique())
    print(f"\nFound {len(common_ids)} common patient IDs between predicted and ground truth")

    if len(common_ids) == 0:
        print("ERROR: No matching patient IDs found!")
        print("Predicted IDs:", sorted(df_pred["name"].unique())[:20])
        print("Ground truth IDs:", sorted(df_gt["name"].unique())[:20])
        return

    # Filter to only common IDs
    df_pred_filtered = df_pred[df_pred["name"].isin(common_ids)].copy()
    df_gt_filtered = df_gt[df_gt["name"].isin(common_ids)].copy()

    # Extract muscle list and via_len
    force_cols = [col for col in df_pred.columns if col.startswith("force_")]
    muscles = sorted([col.replace("force_", "") for col in force_cols])
    print(f"\nFound {len(muscles)} muscles")

    # Determine via_len
    via_len: Dict[str, int] = {}
    for muscle in muscles:
        i = 0
        while f"via_{muscle}_pt{i}_x" in df_pred.columns:
            i += 1
        via_len[muscle] = i

    print("\n" + "=" * 80)
    print("Computing force errors...")
    print("=" * 80)
    df_force_error = compute_force_errors(df_pred_filtered, df_gt_filtered, muscles)
    print(f"Computed force errors for {len(df_force_error)} muscle-patient pairs")

    if len(df_force_error) > 0:
        print("\nForce Error Summary:")
        print(f"  Mean absolute error: {df_force_error['force_error_abs'].mean():.4f} N")
        print(f"  Median absolute error: {df_force_error['force_error_abs'].median():.4f} N")
        print(f"  Max absolute error: {df_force_error['force_error_abs'].max():.4f} N")
    print(f"  Mean absolute percentage error: {df_force_error['force_error_pct_abs'].mean():.2f}%")

    print("\n" + "=" * 80)
    print("Computing via point errors...")
    print("=" * 80)
    df_via_error = compute_via_errors(df_pred_filtered, df_gt_filtered, muscles, via_len)
    print(f"Computed via errors for {len(df_via_error)} muscle-patient pairs")

    if len(df_via_error) > 0:
        print("\nVia Point Error Summary:")
        print(f"  Mean avg distance: {df_via_error['avg_distance_error_mm'].mean():.4f} mm")
        print(f"  Median avg distance: {df_via_error['avg_distance_error_mm'].median():.4f} mm")
        print(f"  Max avg distance: {df_via_error['avg_distance_error_mm'].max():.4f} mm")

    # Save to CSV
    args.output_dir.mkdir(parents=True, exist_ok=True)

    force_error_path = args.output_dir / f"{args.output_prefix}_force_errors.csv"
    force_error_abs_avg_path = args.output_dir / f"{args.output_prefix}_force_errors_by_muscle_abs.csv"
    force_error_pct_avg_path = args.output_dir / f"{args.output_prefix}_force_errors_by_muscle_pct.csv"
    via_error_path = args.output_dir / f"{args.output_prefix}_via_errors.csv"
    via_error_avg_path = args.output_dir / f"{args.output_prefix}_via_errors_by_muscle_mm.csv"
    gt_path = args.output_dir / f"{args.output_prefix}_ground_truth.csv"

    df_force_error.to_csv(force_error_path, index=False)
    print(f"\n✓ Saved force errors to: {force_error_path}")

    if not df_force_error.empty:
        force_abs_avg = (
            df_force_error.groupby("muscle", as_index=False)["force_error_abs"].mean()
        )
        force_abs_avg.to_csv(force_error_abs_avg_path, index=False)
        print(f"✓ Saved force abs error means to: {force_error_abs_avg_path}")

        if "force_error_pct_abs" in df_force_error.columns:
            force_pct_avg = (
                df_force_error.groupby("muscle", as_index=False)["force_error_pct_abs"].mean()
            )
            force_pct_avg.to_csv(force_error_pct_avg_path, index=False)
            print(f"✓ Saved force % error means to: {force_error_pct_avg_path}")

    df_via_error.to_csv(via_error_path, index=False)
    print(f"✓ Saved via errors to: {via_error_path}")

    if not df_via_error.empty:
        via_avg = (
            df_via_error.groupby("muscle", as_index=False)["avg_distance_error_mm"].mean()
        )
        via_avg.to_csv(via_error_avg_path, index=False)
        print(f"✓ Saved via error means to: {via_error_avg_path}")

    df_gt_filtered.to_csv(gt_path, index=False)
    print(f"✓ Saved ground truth data to: {gt_path}")

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
