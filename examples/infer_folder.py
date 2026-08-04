#!/usr/bin/env python3
"""Standalone inference script for converting OSIM files using trained ML models.

Accepts a folder of .osim files and produces subject-specific revised models
using the bundled ML checkpoints. Supports explicit age, sex, height, and
weight arguments.

By default uses the paper's optimal hybrid approach:
  - Random Forest for force prediction (Native-Data conditioning)
  - Latent/Autoencoder for geometry prediction (Anthropometrically-Normalized)

If Random Forest checkpoints are not available, falls back to the Latent model
for both force and geometry.

Usage:
    python examples/infer_folder.py \\
        --input-dir /path/to/osims \\
        --sex male \\
        --age 55
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Setup repo imports
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from osim_parser import OSIMModel
from repo_config import (
    DATASETS_DIR,
    MODELS_DIR as REPO_MODELS_DIR,
    male_template_path,
    female_template_path,
)

# Import feature extraction and baseline computation
from examples.build_dataset import (
    features_from_osim,
    baseline_targets,
    INCLUDE_AGE_FEATURE,
    _sacrum_mass_ratio,
    _corb_height_ratio_via_x,
)

# Import model inference functions
from examples.write_revised_models import (
    predict_latent,
    predict,
    predict_random_forest,
    load_checkpoint,
    scale_muscle_fiber_properties,
    scale_body_masses,
    _height_estimate,
)

DATASETS_PATH = DATASETS_DIR.resolve()
MODELS_PATH = REPO_MODELS_DIR.resolve()


def load_stats() -> Dict:
    with open(DATASETS_PATH / 'stats.json', 'r') as f:
        return json.load(f)


def load_template_props() -> Dict:
    path = DATASETS_PATH / 'generic_muscle_properties.json'
    if path.exists():
        with open(path, 'r') as f:
            return json.load(f)
    return {}


def compute_muscles_and_via(data_root: Path) -> Tuple[List[str], Dict[str, int]]:
    template_male = OSIMModel.from_file(male_template_path(data_root))
    template_female = OSIMModel.from_file(female_template_path(data_root))
    m_male = set(template_male.data.get('forces', {}).keys())
    m_fem = set(template_female.data.get('forces', {}).keys())
    muscles_order = sorted(m_male & m_fem)
    via_len: Dict[str, int] = {}
    for m in muscles_order:
        Lm = len((template_male.data['forces'].get(m) or {}).get('path_points', []))
        Lf = len((template_female.data['forces'].get(m) or {}).get('path_points', []))
        via_len[m] = min(Lm, Lf)
    return muscles_order, via_len


def estimate_height_weight(model: OSIMModel, sex_flag: float, template: OSIMModel) -> Tuple[float, float]:
    """Estimate height and weight from the OSIM model."""
    root_body = 'sacrum'
    q: Dict[str, float] = {}
    # Height
    for cand in ('head_neck', 'thoracic12', 'thoracic5'):
        if cand in model.data['bodies']:
            X = model.transform_relative_to(cand, root_body, q)
            h_est = float(X[1][3])
            break
    else:
        h_est = 1.75
    # Weight
    generic_w = 78.0 if sex_flag > 0.5 else 61.0
    w_ratio = _sacrum_mass_ratio(model, template)
    w_est = float(generic_w * w_ratio)
    return h_est, w_est

def assemble_chunks_if_needed(ckpt_path: Path):
    """Reassembles .part files into the full checkpoint if it doesn't exist."""
    if ckpt_path.exists():
        return
    part_files = sorted(glob.glob(f"{ckpt_path}.part*"))
    if not part_files:
        return
    print(f"Auto-assembling chunked model: {ckpt_path.name}...")
    with open(ckpt_path, 'wb') as out_f:
        for p in part_files:
            with open(p, 'rb') as in_f:
                out_f.write(in_f.read())
    print(f"Successfully assembled {ckpt_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description='Run ML inference on a folder of OSIM files to produce subject-specific revised models.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Paper's best hybrid: RF for force + Latent for geometry (default)
  python examples/infer_folder.py --input-dir ./my_osims --sex male --age 55

  # Use only the latent model for both force and geometry
  python examples/infer_folder.py --input-dir ./my_osims --sex male --model latent

  # Use only MLP for both force and geometry
  python examples/infer_folder.py --input-dir ./my_osims --sex female --model mlp

  # Explicit height/weight, custom output dir
  python examples/infer_folder.py --input-dir ./my_osims --output-dir ./results \\
      --sex female --age 70 --height 1.60 --weight 55
"""
    )
    parser.add_argument('--input-dir', required=True, type=str,
                        help='Directory containing .osim files to process')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for revised OSIMs (default: <input-dir>/revised/)')
    parser.add_argument('--sex', choices=['male', 'female'],
                        help='Subject sex (determines template selection). Required unless --demographics-csv is used.')
    parser.add_argument('--age', type=float, default=65.0,
                        help='Subject age in years (default: 65.0). Overridden if --demographics-csv is used.')
    parser.add_argument('--demographics-csv', type=str, default=None,
                        help='Path to CSV with ID,Sex,Age columns. The ID is the number before the first _ in the OSIM filename.')
    parser.add_argument('--model', choices=['hybrid', 'latent', 'mlp'], default='hybrid',
                        help='Model strategy: hybrid (RF force + Latent via, paper best), '
                             'latent (Latent for both), mlp (MLP for both). Default: hybrid')
    parser.add_argument('--data-root', type=str, default=None,
                        help='Override OPENSIM_MUSCLE_NN_DATA_ROOT env var')
    parser.add_argument('--weight', type=float, default=None,
                        help='Subject weight in kg (estimated from OSIM if not provided)')
    parser.add_argument('--height', type=float, default=None,
                        help='Subject height in meters (estimated from OSIM if not provided)')
    args = parser.parse_args()

    if not args.sex and not args.demographics_csv:
        parser.error("Either --sex or --demographics-csv must be provided.")

    # Resolve data root
    if args.data_root:
        os.environ['OPENSIM_MUSCLE_NN_DATA_ROOT'] = args.data_root
    data_root_raw = os.environ.get('OPENSIM_MUSCLE_NN_DATA_ROOT', '').strip()
    if not data_root_raw:
        print('ERROR: Set OPENSIM_MUSCLE_NN_DATA_ROOT or pass --data-root', file=sys.stderr)
        sys.exit(1)
    data_root = Path(data_root_raw).expanduser().resolve()

    # Resolve paths
    input_dir = Path(args.input_dir).resolve()
    if not input_dir.is_dir():
        print(f'ERROR: Input directory does not exist: {input_dir}', file=sys.stderr)
        sys.exit(1)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else input_dir / 'revised'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load demographics if provided
    demographics = {}
    if args.demographics_csv:
        import csv
        with open(args.demographics_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                uid = str(row.get('ID', '')).strip()
                if uid:
                    demographics[uid] = {
                        'sex': str(row.get('Sex', '')).strip().lower(),
                        'age': float(row.get('Age', 65.0))
                    }
        print(f"Loaded demographics for {len(demographics)} subjects from CSV.")

    # Discover OSIM files
    osim_files = sorted(glob.glob(str(input_dir / '*.osim')))
    if not osim_files:
        print(f'No .osim files found in {input_dir}', file=sys.stderr)
        sys.exit(1)
    print(f'Found {len(osim_files)} OSIM file(s) in {input_dir}')

    # Load shared resources
    print('Loading stats and model metadata...')
    stats = load_stats()
    build_flags = stats.get('_build_flags', {})
    VIA_MILLIMETERS = build_flags.get('VIA_MILLIMETERS', True)
    FORCE_LOG = build_flags.get('FORCE_LOG', False)
    FAIR = build_flags.get('TRAIN_ONLY_STATS', True)  # FAIR mode = standardized targets

    def via_to_meters(delta):
        if VIA_MILLIMETERS:
            return float(delta) / 1000.0
        return float(delta)

    template_props = load_template_props()
    muscles_order, via_len = compute_muscles_and_via(data_root)
    print(f'Processing {len(muscles_order)} muscles')

    # Load templates for feature extraction (since we might need both now)
    template_male = OSIMModel.from_file(male_template_path(data_root))
    template_female = OSIMModel.from_file(female_template_path(data_root))

    # Load FORCE_PREDICT_MASK and VIA_PREDICT_MASK_EXTRA from stats
    force_mask = np.array(stats.get('FORCE_PREDICT_MASK', [1] * len(muscles_order)), dtype=float)
    via_mask_extra = stats.get('VIA_PREDICT_MASK_EXTRA')

    # Build via-level mask
    def build_via_mask():
        mask_list = []
        for mi, m in enumerate(muscles_order):
            L = int(via_len[m])
            val = float(force_mask[mi])
            if via_mask_extra is not None:
                val *= float(via_mask_extra[mi])
            mask_list.extend([val] * (3 * L))
        return np.array(mask_list, dtype=float)

    via_mask_arr = build_via_mask()

    # ---------- Resolve which models to use for force vs. geometry ----------
    latent_ckpt_path = MODELS_PATH / 'latent.pt'
    mlp_force_ckpt_path = MODELS_PATH / 'mlp_force.pt'
    mlp_via_ckpt_path = MODELS_PATH / 'mlp_via.pt'
    rf_force_ckpt_path = MODELS_PATH / 'rf_force.pkl'
    rf_via_ckpt_path = MODELS_PATH / 'rf_via.pkl'

    # Auto-assemble RF chunks if the user just cloned the repo
    assemble_chunks_if_needed(rf_force_ckpt_path)
    assemble_chunks_if_needed(rf_via_ckpt_path)

    rf_available = rf_force_ckpt_path.exists()
    latent_available = latent_ckpt_path.exists()
    mlp_available = mlp_force_ckpt_path.exists() and mlp_via_ckpt_path.exists()

    # Determine force_model and via_model
    if args.model == 'hybrid':
        if rf_available and latent_available:
            force_model = 'rf'
            via_model = 'latent'
            print('Model strategy: HYBRID (paper best) — RF for force, Latent for geometry')
        elif latent_available:
            force_model = 'latent'
            via_model = 'latent'
            print('Model strategy: HYBRID requested but RF checkpoints not found.')
            print('  Falling back to Latent for both force and geometry.')
            print(f'  To enable hybrid: train RF with `python examples/train_models.py`')
            print(f'  Expected checkpoint: {rf_force_ckpt_path}')
        elif mlp_available:
            force_model = 'mlp'
            via_model = 'mlp'
            print('Model strategy: HYBRID requested but RF and Latent not found. Using MLP.')
        else:
            print('ERROR: No model checkpoints available.', file=sys.stderr)
            sys.exit(1)
    elif args.model == 'latent':
        if not latent_available:
            print(f'ERROR: Latent checkpoint not found at {latent_ckpt_path}', file=sys.stderr)
            sys.exit(1)
        force_model = 'latent'
        via_model = 'latent'
        print('Model strategy: Latent for both force and geometry')
    elif args.model == 'mlp':
        if not mlp_available:
            print(f'ERROR: MLP checkpoints not found at {MODELS_PATH}', file=sys.stderr)
            sys.exit(1)
        force_model = 'mlp'
        via_model = 'mlp'
        print('Model strategy: MLP for both force and geometry')

    # Preload checkpoints
    mlp_force_ckpt = load_checkpoint(mlp_force_ckpt_path) if force_model == 'mlp' else None
    mlp_via_ckpt = load_checkpoint(mlp_via_ckpt_path) if via_model == 'mlp' else None

    # RF de-standardization stats (RF outputs are in standardized space)
    force_stats_key = 'Y_force_log' if (FORCE_LOG and ('Y_force_log' in stats)) else 'Y_force_res'
    rf_yf_mu = np.array(stats[force_stats_key]['mean'], dtype=float)
    rf_yf_sd = np.array(stats[force_stats_key]['std'], dtype=float)
    rf_yv_mu = np.array(stats['Y_via_res']['mean'], dtype=float)
    rf_yv_sd = np.array(stats['Y_via_res']['std'], dtype=float)

    # Process each OSIM file
    suffix = f'_revised_{args.model}'

    for idx, osim_path in enumerate(osim_files, 1):
        stem = Path(osim_path).stem
        print(f'\n[{idx}/{len(osim_files)}] Processing {Path(osim_path).name}...')

        # Determine sex and age for this file
        current_sex_str = args.sex
        current_age = args.age
        
        if args.demographics_csv:
            uid = stem.split('_')[0]
            if uid not in demographics:
                print(f"  SKIPPED: ID '{uid}' not found in demographics CSV.")
                continue
            
            current_sex_str = demographics[uid]['sex']
            current_age = demographics[uid]['age']
        
        if current_sex_str not in ('male', 'female'):
            print(f"  SKIPPED: Invalid sex '{current_sex_str}' for file.")
            continue

        sex_flag = 1.0 if current_sex_str == 'male' else 0.0
        tmpl_path = male_template_path(data_root) if current_sex_str == 'male' else female_template_path(data_root)
        template = template_male if current_sex_str == 'male' else template_female

        try:
            model = OSIMModel.from_file(osim_path)

            # Build features
            feats, aux = features_from_osim(model, sex_flag, template, osim_path)

            # Override age, height, weight in feature vector
            # Feature layout (INCLUDE_AGE_FEATURE=True): [...vertebral_positions..., h_est, w_est, sex, age]
            if INCLUDE_AGE_FEATURE:
                feats[-1] = current_age  # Override age
                if args.height is not None:
                    feats[-4] = args.height
                if args.weight is not None:
                    feats[-3] = args.weight
            else:
                # No age feature: [..., h_est, w_est, sex]
                if args.height is not None:
                    feats[-3] = args.height
                if args.weight is not None:
                    feats[-2] = args.weight

            # Get sacrum mass ratio for body mass scaling
            w_ratio = _sacrum_mass_ratio(model, template)

            # Compute baseline deltas
            dF_bl, dVia_bl = baseline_targets(model, template, sex_flag, muscles_order, via_len)

            # Normalize features
            x_mu = np.array(stats['X']['mean'], dtype=float)
            x_sd = np.array(stats['X']['std'], dtype=float)
            Xn = ((np.array(feats, dtype=float) - x_mu) / x_sd).tolist()

            # ---------- FORCE prediction ----------
            if force_model == 'rf':
                # Random Forest: input is standardized features, output is standardized residuals
                dF_pred_std = predict_random_forest(rf_force_ckpt_path, [Xn])[0]
                # De-standardize
                dF_pred_res = (np.array(dF_pred_std, dtype=float) * rf_yf_sd) + rf_yf_mu
                # Apply force mask
                dF_pred_res = dF_pred_res * force_mask
                # Absolute delta = baseline + predicted residual
                dF_abs = np.array(dF_bl, dtype=float) + dF_pred_res

            elif force_model == 'latent':
                Yf_res, _ = predict_latent(latent_ckpt_path, [Xn], stats)
                Yf_res = Yf_res[0]
                # Handle log-space force if applicable
                if 'Y_force_log' in stats:
                    tmpl_forces = []
                    for m in muscles_order:
                        ft = (template.data['forces'].get(m) or {}).get('max_isometric_force', 0.0)
                        tmpl_forces.append(float(ft))
                    f_t_arr = np.array(tmpl_forces, dtype=float)
                    Yf_add = f_t_arr * (np.exp(Yf_res) - 1.0)
                else:
                    Yf_add = Yf_res
                Yf_add = np.array(Yf_add, dtype=float) * force_mask
                dF_abs = np.array(dF_bl, dtype=float) + Yf_add

            elif force_model == 'mlp':
                dF_pred = predict(mlp_force_ckpt, [Xn])[0]
                dF_pred = np.array(dF_pred, dtype=float) * force_mask
                dF_abs = np.array(dF_bl, dtype=float) + dF_pred

            # ---------- GEOMETRY (via-point) prediction ----------
            if via_model == 'latent':
                _, Yv_res = predict_latent(latent_ckpt_path, [Xn], stats)
                Yv_res = Yv_res[0]
                Yv_res = np.array(Yv_res, dtype=float) * via_mask_arr
                dVia_abs = np.array(dVia_bl, dtype=float) + Yv_res

            elif via_model == 'mlp':
                dVia_pred = predict(mlp_via_ckpt, [Xn])[0]
                dVia_pred = np.array(dVia_pred, dtype=float) * via_mask_arr
                dVia_abs = np.array(dVia_bl, dtype=float) + dVia_pred

            # ---------- Build revised model ----------
            rev = OSIMModel.from_file(tmpl_path)

            model_desc = f'force={force_model.upper()}, via={via_model.upper()}'
            comment = ET.Comment(
                f'Revised by: {model_desc}, sex={current_sex_str}, age={current_age}, '
                f'height={args.height or "auto"}, weight={args.weight or "auto"}, '
                f'source={osim_path}'
            )
            rev.root.insert(0, comment)

            # Scale body masses
            scale_body_masses(rev, w_ratio)

            # Apply force deltas
            for mi, m in enumerate(muscles_order):
                mt = rev.data['forces'].get(m) or {}
                f_t = mt.get('max_isometric_force') or 0.0
                rev.set_muscle_max_isometric_force(m, float(f_t + dF_abs[mi]))

            # Apply via deltas
            off = 0
            for m in muscles_order:
                L = via_len[m]
                for k in range(L):
                    dx, dy, dz = dVia_abs[off], dVia_abs[off + 1], dVia_abs[off + 2]
                    off += 3
                    pt = (rev.data['forces'][m].get('path_points') or [])[k]
                    loc0 = pt.get('location') or [0.0, 0.0, 0.0]
                    new_loc = [
                        loc0[0] + via_to_meters(dx),
                        loc0[1] + via_to_meters(dy),
                        loc0[2] + via_to_meters(dz),
                    ]
                    rev.set_muscle_path_point_location(m, k, new_loc)

            # Scale fiber properties
            if template_props:
                scale_muscle_fiber_properties(rev, muscles_order, template_props, current_sex_str)

            out_path = output_dir / f'{stem}{suffix}.osim'
            rev.save(out_path)
            print(f'  Wrote ({model_desc}): {out_path}')

        except Exception as e:
            print(f'  SKIPPED (error): {e}')
            import traceback
            traceback.print_exc()

    print(f'\nDone. Revised models written to: {output_dir}')


if __name__ == '__main__':
    main()
