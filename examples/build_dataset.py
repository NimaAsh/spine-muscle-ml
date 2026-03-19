from __future__ import annotations

import glob
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from osim_parser import OSIMModel  # noqa: E402
from osim_parser.parser import _R_to_euler_xyz_body_fixed  # noqa: E402
from repo_config import DATASETS_DIR, female_template_path, get_data_root, male_template_path, require_data_root  # noqa: E402


DATA_ROOT = get_data_root()
GENERIC_MALE = male_template_path(DATA_ROOT) if DATA_ROOT else None
GENERIC_FEMALE = female_template_path(DATA_ROOT) if DATA_ROOT else None
MAX_FILES = 250

# ========== LEGACY TOGGLES (revert to pre-Oct 1st behavior) ==========
# Set any of these to False to revert to the original behavior before October 1st changes
FORCE_LOG = False                    # Use log-ratio force targets instead of additive residuals
CORB_HEIGHT_SCALING = True          # Use CORB PathPoint x-coordinate for height scaling (vs legacy Y-translation)
APPLY_FORCE_PREDICT_MASK = True     # Mask out always-zero force muscles
APPLY_VIA_PREDICT_MASK = False       # Toggle whether force mask also affects via point predictions (masked muscles excluded from training)
APPLY_VIA_PREDICT_MASK_EXTRA = True # Exclude additional muscles that are purely height-scaled (threshold=0.01mm, criterion=0.95)
VIA_MILLIMETERS = True              # Scale via point deltas to millimeters (vs meters)
VALIDATION_SPLIT = True             # Create separate validation set (vs train/test only)
FIXED_LENGTH_FEATURES = False        # Zero-pad missing bodies for fixed feature length
TRAIN_ONLY_STATS = True             # Compute normalization stats from train set only (vs all data)
POS_ONLY = True                     # Use only position features (no orientation) for vertebrae
FORCE_BASELINE_SCALING = False       # Apply weight-based scaling to force baseline (if False, baseline=template, model learns full delta)

# ========== BASELINE SUMMARY FEATURES ==========
# When True: Models will append 2 features (bf, bv) = mean absolute baseline force/via errors per sample
# When False: Models use only the original input features (currently 94)
# This flag is imported by train_models.py, train_latent.py, and write_revised_models.py
USE_BASELINE_SUMMARY_FEATURES = False

# ========== AGE FEATURE ==========
# When True: Include age as a feature in the input (estimated from file path)
# When False: Exclude age feature from input
# This affects feature dimension and should match training configuration
INCLUDE_AGE_FEATURE = True


# ========== VERTEBRAE CONFIGURATION ==========
# When True: Use all vertebrae (5 lumbar + 12 thoracic = 17 bodies)
# When False: Use original subset (5 lumbar + 5 thoracic = 10 bodies)
USE_ALL_VERTEBRAE = True

LUMBAR = [f"lumbar{i}" for i in range(1, 6)]  # L1-L5 (always 5)
if USE_ALL_VERTEBRAE:
    THORACIC = [f"thoracic{i}" for i in range(1, 13)]  # T1-T12 (12 bodies)
else:
    THORACIC = [f"thoracic{i}" for i in range(1, 6)]   # T1-T5 (5 bodies, original)

# TARGET_BODIES: 17 bodies when USE_ALL_VERTEBRAE=True, 10 bodies otherwise
TARGET_BODIES = LUMBAR + THORACIC
SEQ_TOKENS = LUMBAR + THORACIC + ["head_neck"]  # ordered token list for sequence models


def mat4_to_Rp(X):
    R = [row[:3] for row in X[:3]]
    p = [X[0][3], X[1][3], X[2][3]]
    return R, p


def _sincos_angles(eul: List[float]) -> List[float]:
    rx, ry, rz = (eul + [0.0, 0.0, 0.0])[:3]
    return [math.sin(rx), math.cos(rx), math.sin(ry), math.cos(ry), math.sin(rz), math.cos(rz)]


def _age_from_path(path: str) -> float:
    """Extract age estimate from path containing Age4049, Age5059, etc."""
    age_map = {
        "Age4049": 45.0,
        "Age5059": 55.0,
        "Age6069": 65.0,
        "Age7079": 75.0,
        "Age80plus": 85.0,
    }
    for age_str, age_val in age_map.items():
        if age_str in path:
            return age_val
    # Default if not found
    return 65.0


def _subject_mass(model: OSIMModel) -> float:
    total = 0.0
    for b in model.data.get("bodies", {}).values():
        m = b.get("mass")
        if isinstance(m, (int, float)) and m is not None:
            total += float(m)
    return float(total)


def _sacrum_mass_ratio(model: OSIMModel, template: OSIMModel) -> float:
    try:
        ms = float(model.data["bodies"]["sacrum"]["mass"])  # type: ignore[index]
        mt = float(template.data["bodies"]["sacrum"]["mass"])  # type: ignore[index]
        return ms / mt if mt else 1.0
    except Exception:
        return 1.0


def _get_pathpoint_x(model: OSIMModel, muscle_name: str, point_name: str) -> float | None:
    try:
        m = model.data.get("forces", {}).get(muscle_name) or {}
        pts = m.get("path_points") or []
        for pt in pts:
            nm = pt.get("name")
            if isinstance(nm, str) and (nm == point_name):
                loc = pt.get("location") or [0.0, 0.0, 0.0]
                return float(loc[0])
    except Exception:
        pass
    return None


def _corb_height_ratio_via_x(model: OSIMModel, template: OSIMModel) -> float | None:
    """Estimate a scale factor using CORB.PathPoint_scapula_R x-coordinate ratio.

    Returns None if unavailable; caller should fall back to legacy estimate.
    """
    subj_x = _get_pathpoint_x(model, "CORB", "PathPoint_scapula_R")
    tmpl_x = _get_pathpoint_x(template, "CORB", "PathPoint_scapula_R")
    if (subj_x is not None) and (tmpl_x is not None):
        eps = 1e-8
        if abs(tmpl_x) > eps:
            return float(subj_x / tmpl_x)
    return None


def features_from_osim(model: OSIMModel, sex_flag: float, template: OSIMModel, osim_path: str = "") -> Tuple[List[float], Dict[str, float]]:
    # 6-DoF pose (Euler XYZ body-fixed + translation) relative to sacrum for selected bodies
    # If POS_ONLY=True, only use 3-DoF translation (no orientation)
    feats: List[float] = []
    root = "sacrum"
    q = {}

    if FIXED_LENGTH_FEATURES:
        # Build fixed-length feature vector in TARGET_BODIES order; zero-pad missing bodies
        for n in TARGET_BODIES:
            if n in model.data["bodies"]:
                X = model.transform_relative_to(n, root, q)
                R, p = mat4_to_Rp(X)
                if POS_ONLY:
                    # Only position: 3 features per body
                    feats.extend([p[0], p[1], p[2]])
                else:
                    # Full 6-DoF: orientation (6 sin/cos) + position (3)
                    eul = _R_to_euler_xyz_body_fixed(R)
                    feats.extend(_sincos_angles(eul) + [p[0], p[1], p[2]])
            else:
                # Zero-pad missing bodies
                if POS_ONLY:
                    feats.extend([0.0, 0.0, 0.0])
                else:
                    feats.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    else:
        # Original: only include bodies that exist (variable-length features)
        names = [n for n in TARGET_BODIES if n in model.data["bodies"]]
        for n in names:
            X = model.transform_relative_to(n, root, q)
            R, p = mat4_to_Rp(X)
            if POS_ONLY:
                # Only position: 3 features per body
                feats.extend([p[0], p[1], p[2]])
            else:
                # Full 6-DoF: orientation (6 sin/cos) + position (3)
                eul = _R_to_euler_xyz_body_fixed(R)
                feats.extend(_sincos_angles(eul) + [p[0], p[1], p[2]])

    # Height/weight/sex
    sex = float(sex_flag)  # 1.0 male, 0.0 female
    # Height estimate from Y translation relative to sacrum: prefer head_neck, else thoracic12, else thoracic5
    if "head_neck" in model.data["bodies"]:
        Xhn = model.transform_relative_to("head_neck", root, q)
        h_est = mat4_to_Rp(Xhn)[1][1]
    elif "thoracic12" in model.data["bodies"]:
        Xt12 = model.transform_relative_to("thoracic12", root, q)
        h_est = mat4_to_Rp(Xt12)[1][1]
    elif "thoracic5" in model.data["bodies"]:
        Xt5 = model.transform_relative_to("thoracic5", root, q)
        h_est = mat4_to_Rp(Xt5)[1][1]
    else:
        h_est = 1.75
    # Weight from sacrum mass ratio times generic nominal weight
    generic_w = 78.0 if sex > 0.5 else 61.0
    w_ratio = _sacrum_mass_ratio(model, template)
    w_est = float(generic_w * w_ratio)
    # Age from path
    age = _age_from_path(osim_path)

    # Conditionally include age feature
    if INCLUDE_AGE_FEATURE:
        feats.extend([h_est, w_est, sex, age])
    else:
        feats.extend([h_est, w_est, sex])

    # Return features and aux data (aux always includes age for metadata)
    aux: Dict[str, float] = {"num_bodies": float(len(names)), "h_est": float(h_est), "w_est": float(w_est), "sex": sex, "age": age}
    return feats, aux


def sequence_from_osim(model: OSIMModel, sex_flag: float, template: OSIMModel, osim_path: str = "") -> Tuple[List[List[float]], List[int]]:
    """Build per-token sequence features and a mask (1=present, 0=missing).

    Per-token feature vector (depends on INCLUDE_AGE_FEATURE flag):
    - If POS_ONLY=False and INCLUDE_AGE_FEATURE=True: [sin(rx),cos(rx),sin(ry),cos(ry),sin(rz),cos(rz), tx, ty, tz, height, weight, sex, age]
    - If POS_ONLY=True and INCLUDE_AGE_FEATURE=True: [tx, ty, tz, height, weight, sex, age]
    - If INCLUDE_AGE_FEATURE=False: same but without age feature
    Tokens ordered by SEQ_TOKENS.
    """
    root = "sacrum"
    q = {}
    sex = float(sex_flag)
    # Height
    if "head_neck" in model.data["bodies"]:
        Xhn = model.transform_relative_to("head_neck", root, q)
        h_est = mat4_to_Rp(Xhn)[1][1]
    elif "thoracic12" in model.data["bodies"]:
        Xt12 = model.transform_relative_to("thoracic12", root, q)
        h_est = mat4_to_Rp(Xt12)[1][1]
    elif "thoracic5" in model.data["bodies"]:
        Xt5 = model.transform_relative_to("thoracic5", root, q)
        h_est = mat4_to_Rp(Xt5)[1][1]
    else:
        h_est = 1.75
    # Weight
    generic_w = 78.0 if sex > 0.5 else 61.0
    w_ratio = _sacrum_mass_ratio(model, template)
    w_est = float(generic_w * w_ratio)
    # Age
    age = _age_from_path(osim_path)

    seq: List[List[float]] = []
    mask: List[int] = []
    for name in SEQ_TOKENS:
        if name in model.data["bodies"]:
            X = model.transform_relative_to(name, root, q)
            R, p = mat4_to_Rp(X)
            if POS_ONLY:
                # Only position + global features
                if INCLUDE_AGE_FEATURE:
                    seq.append([p[0], p[1], p[2], h_est, w_est, sex, age])
                else:
                    seq.append([p[0], p[1], p[2], h_est, w_est, sex])
            else:
                # Full 6-DoF orientation + position + global features
                eul = _R_to_euler_xyz_body_fixed(R)
                if INCLUDE_AGE_FEATURE:
                    seq.append(_sincos_angles(eul) + [p[0], p[1], p[2], h_est, w_est, sex, age])
                else:
                    seq.append(_sincos_angles(eul) + [p[0], p[1], p[2], h_est, w_est, sex])
            mask.append(1)
        else:
            # Zero-pad missing tokens
            if POS_ONLY:
                # 3 (position) + 2 (h,w) + 1 (sex) + [0 or 1 for age]
                pad_size = 6 + (1 if INCLUDE_AGE_FEATURE else 0)
                seq.append([0.0] * pad_size)
            else:
                # 6 (orientation sincos) + 3 (position) + 2 (h,w) + 1 (sex) + [0 or 1 for age]
                pad_size = 12 + (1 if INCLUDE_AGE_FEATURE else 0)
                seq.append([0.0] * pad_size)
            mask.append(0)
    return seq, mask


def _estimate_height_weight(model: OSIMModel, sex_flag: float, template: OSIMModel) -> Tuple[float, float]:
    root = "sacrum"
    q = {}
    if "head_neck" in model.data["bodies"]:
        Xhn = model.transform_relative_to("head_neck", root, q)
        h_est = mat4_to_Rp(Xhn)[1][1]
    elif "thoracic12" in model.data["bodies"]:
        Xt12 = model.transform_relative_to("thoracic12", root, q)
        h_est = mat4_to_Rp(Xt12)[1][1]
    elif "thoracic5" in model.data["bodies"]:
        Xt5 = model.transform_relative_to("thoracic5", root, q)
        h_est = mat4_to_Rp(Xt5)[1][1]
    else:
        h_est = 1.75
    generic_w = 78.0 if sex_flag > 0.5 else 61.0
    w_est = float(generic_w * _sacrum_mass_ratio(model, template))
    return float(h_est), float(w_est)


def baseline_targets(model: OSIMModel, template: OSIMModel, sex_flag: float, muscles_order: List[str], via_len: Dict[str, int]) -> Tuple[List[float], List[float]]:
    # Force baseline via sacrum mass ratio; via baseline via height ratio
    h_subj, _ = _estimate_height_weight(model, sex_flag, template)
    tmpl_sex = 1.0 if any(x in template.path.as_posix() for x in ["Male_Thoracolumbar_Spine_V1"]) else 0.0
    h_tmpl, _ = _estimate_height_weight(template, tmpl_sex, template)
    scale_w = _sacrum_mass_ratio(model, template)

    if CORB_HEIGHT_SCALING:
        # Prefer CORB-based x-ratio; fall back to legacy vertical estimate
        _corb_ratio = _corb_height_ratio_via_x(model, template)
        scale_h = float(_corb_ratio) if (_corb_ratio is not None) else ((h_subj / h_tmpl) if h_tmpl else 1.0)
    else:
        # Original: use legacy vertical height estimate only
        scale_h = (h_subj / h_tmpl) if h_tmpl else 1.0

    m_tmpl = template.data.get("forces", {})
    y_force_bl: List[float] = []
    y_via_bl: List[float] = []
    for m in muscles_order:
        mt = m_tmpl.get(m) or {}
        f_t = mt.get("max_isometric_force") or 0.0
        # Force baseline: apply weight scaling if enabled, otherwise 0 (baseline = template)
        if FORCE_BASELINE_SCALING:
            y_force_bl.append(float(f_t * (scale_w - 1.0)))
        else:
            y_force_bl.append(0.0)
        # Via baselines (always scaled by height)
        L = via_len.get(m, 0)
        pt = mt.get("path_points") or []
        for i in range(L):
            lt = (pt[i].get("location") if i < len(pt) else [0.0, 0.0, 0.0])
            y_via_bl.extend([lt[0] * (scale_h - 1.0), lt[1] * (scale_h - 1.0), lt[2] * (scale_h - 1.0)])
    return y_force_bl, y_via_bl


def targets_from_osim(model: OSIMModel, template: OSIMModel, muscles_order: List[str], via_len: Dict[str, int]) -> Tuple[List[float], List[float]]:
    # Return fixed-length vectors for force deltas and via-point deltas
    m_curr = model.data.get("forces", {})
    m_tmpl = template.data.get("forces", {})

    y_force: List[float] = []
    y_via: List[float] = []
    for m in muscles_order:
        mc = m_curr.get(m)
        mt = m_tmpl.get(m)
        # force delta
        f_c = (mc or {}).get("max_isometric_force") or 0.0
        f_t = (mt or {}).get("max_isometric_force") or 0.0
        y_force.append(float(f_c - f_t))
        # via deltas up to via_len[m] points
        L = via_len.get(m, 0)
        pc = (mc or {}).get("path_points") or []
        pt = (mt or {}).get("path_points") or []
        for i in range(L):
            lc = (pc[i].get("location") if i < len(pc) else [0.0, 0.0, 0.0])
            lt = (pt[i].get("location") if i < len(pt) else [0.0, 0.0, 0.0])
            y_via.extend([lc[0] - lt[0], lc[1] - lt[1], lc[2] - lt[2]])
    return y_force, y_via


def targets_from_osim_force_log(model: OSIMModel, template: OSIMModel, muscles_order: List[str]) -> Tuple[List[float], List[float]]:
    """Return per-muscle log ratio targets and template forces.

    y_force_log[i] = log((f_c+eps)/(f_t+eps))
    Also return f_t per-muscle to enable converting ratios to additive deltas downstream.
    """
    import math
    m_curr = model.data.get("forces", {})
    m_tmpl = template.data.get("forces", {})
    y_force_log: List[float] = []
    y_force_template: List[float] = []
    eps = 1e-6
    for m in muscles_order:
        mc = m_curr.get(m)
        mt = m_tmpl.get(m)
        f_c = float((mc or {}).get("max_isometric_force") or 0.0)
        f_t = float((mt or {}).get("max_isometric_force") or 0.0)
        y_force_log.append(float(math.log((f_c + eps) / (f_t + eps))))
        y_force_template.append(f_t)
    return y_force_log, y_force_template


SEED = 42


def main():
    data_root = require_data_root()
    # Deterministic build
    print(f"Building dataset with POS_ONLY={POS_ONLY}, USE_ALL_VERTEBRAE={USE_ALL_VERTEBRAE}")
    global_feat_count = 3 + (1 if INCLUDE_AGE_FEATURE else 0)  # h, w, sex, [age]
    global_feat_names = "h/w/sex" + ("/age" if INCLUDE_AGE_FEATURE else "")
    vertebrae_desc = f"{len(LUMBAR)} lumbar + {len(THORACIC)} thoracic = {len(TARGET_BODIES)} bodies"
    if POS_ONLY:
        print(f"  Feature dimensions per vertebra: 3 (position only)")
        print(f"  Vertebrae: {vertebrae_desc}")
        print(f"  Expected total features: {len(TARGET_BODIES) * 3} (vertebrae) + {global_feat_count} ({global_feat_names}) = {len(TARGET_BODIES) * 3 + global_feat_count}")
    else:
        print(f"  Feature dimensions per vertebra: 9 (6 orientation + 3 position)")
        print(f"  Vertebrae: {vertebrae_desc}")
        print(f"  Expected total features: {len(TARGET_BODIES) * 9} (vertebrae) + {global_feat_count} ({global_feat_names}) = {len(TARGET_BODIES) * 9 + global_feat_count}")
    t0 = time.time()
    # Gather OSIM files by sex/age, then select up to 25 per age group per sex
    male_paths = sorted(glob.glob(os.path.join(data_root, "Male", "**", "*.osim"), recursive=True))
    female_paths = sorted(glob.glob(os.path.join(data_root, "Female", "**", "*.osim"), recursive=True))
    assert male_paths or female_paths, "No OSIM files found"

    def bucket_by_age(paths, sex_label):
        from collections import defaultdict
        buckets = defaultdict(list)
        for p in paths:
            parts = Path(p).parts
            if sex_label in parts:
                i = parts.index(sex_label)
                age = parts[i+1] if i+1 < len(parts) else "Unknown"
            else:
                age = "Unknown"
            buckets[age].append(p)
        return buckets

    male_buckets = bucket_by_age(male_paths, "Male")
    female_buckets = bucket_by_age(female_paths, "Female")

    # Select up to 25 from each age bucket per sex
    def select_from_buckets(buckets, k=25):
        out = []
        for age, lst in sorted(buckets.items()):
            out.extend(sorted(lst)[:k])
        return out

    male_sel = select_from_buckets(male_buckets, 25)
    female_sel = select_from_buckets(female_buckets, 25)
    paths = (male_sel + female_sel)[:MAX_FILES]

    # Templates
    template_male = OSIMModel.from_file(male_template_path(data_root))
    template_female = OSIMModel.from_file(female_template_path(data_root))

    # Build muscle order and via length spec from intersection of both templates
    m_male = set(template_male.data.get("forces", {}).keys())
    m_fem = set(template_female.data.get("forces", {}).keys())
    muscles_order = sorted(m_male & m_fem)
    via_len: Dict[str, int] = {}
    for m in muscles_order:
        Lm = len((template_male.data["forces"].get(m) or {}).get("path_points", []))
        Lf = len((template_female.data["forces"].get(m) or {}).get("path_points", []))
        via_len[m] = min(Lm, Lf)

    X: List[List[float]] = []
    X_seq: List[List[List[float]]] = []
    X_seq_mask: List[List[int]] = []
    AUX: List[Dict[str, float]] = []
    Y_force: List[List[float]] = []
    Y_via: List[List[float]] = []
    Y_force_bl: List[List[float]] = []
    Y_via_bl: List[List[float]] = []
    Y_force_res: List[List[float]] = []
    Y_via_res: List[List[float]] = []
    Y_force_log: List[List[float]] = []
    Y_force_template: List[List[float]] = []
    IDS: List[str] = []
    STRATA: List[str] = []

    total = len(paths)
    for idx_p, p in enumerate(paths, 1):
        try:
            m = OSIMModel.from_file(p)
            sex_flag = 1.0 if "/Male/" in p else 0.0
            # age group folder after sex directory
            parts = Path(p).parts
            age_group = ""
            if "Male" in parts:
                i = parts.index("Male")
                age_group = parts[i+1] if i+1 < len(parts) else "Unknown"
            elif "Female" in parts:
                i = parts.index("Female")
                age_group = parts[i+1] if i+1 < len(parts) else "Unknown"
            tmpl = template_male if sex_flag > 0.5 else template_female
            feats, aux = features_from_osim(m, sex_flag, tmpl, p)
            seq, seq_mask = sequence_from_osim(m, sex_flag, tmpl, p)
            dF, dVia = targets_from_osim(m, tmpl, muscles_order, via_len)

            # Compute log-ratio targets if enabled or for downstream optional use
            if FORCE_LOG or True:  # Always compute for flexibility
                dF_log, f_t_vec = targets_from_osim_force_log(m, tmpl, muscles_order)
            else:
                dF_log, f_t_vec = None, None

            dF_bl, dVia_bl = baseline_targets(m, tmpl, sex_flag, muscles_order, via_len)

            # Scale via deltas to millimeters if enabled
            if VIA_MILLIMETERS:
                dVia = [float(v) * 1000.0 for v in dVia]
                dVia_bl = [float(v) * 1000.0 for v in dVia_bl]

            X.append(feats)
            X_seq.append(seq)
            X_seq_mask.append(seq_mask)
            AUX.append(aux)
            Y_force.append(dF)
            Y_via.append(dVia)
            if dF_log is not None:
                Y_force_log.append(dF_log)
                Y_force_template.append(f_t_vec)
            Y_force_bl.append(dF_bl)
            Y_via_bl.append(dVia_bl)
            Y_force_res.append([df - bl for df, bl in zip(dF, dF_bl)])
            Y_via_res.append([dv - bl for dv, bl in zip(dVia, dVia_bl)])
            IDS.append(p)
            STRATA.append(("M" if sex_flag > 0.5 else "F") + ":" + age_group)
            if idx_p % 10 == 0:
                print(f"Processed {idx_p}/{total} ...")
        except Exception as e:
            print("Skip", p, "due to", e)

    # Split train/val/test
    # Stratified by sex and age group
    from collections import defaultdict
    groups = defaultdict(list)
    for i, g in enumerate(STRATA):
        groups[g].append(i)
    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []
    random.seed(SEED)
    try:
        import numpy as np
        np.random.seed(SEED)
    except Exception:
        pass
    for g, idxs in groups.items():
        idxs = idxs[:]
        random.shuffle(idxs)
        n = len(idxs)
        n_train = int(round(0.8 * n))
        group_train = idxs[:n_train]
        group_test = idxs[n_train:]

        if VALIDATION_SPLIT:
            # Carve out validation from group_train (10% of group_train)
            n_val = max(1, int(round(0.1 * len(group_train)))) if len(group_train) > 0 else 0
            group_val = group_train[:n_val]
            group_train_final = group_train[n_val:]
            train_idx.extend(group_train_final)
            val_idx.extend(group_val)
        else:
            # Original: no validation split, all training data in train
            train_idx.extend(group_train)

        test_idx.extend(group_test)

    def _compute_force_predict_mask(indices: List[int]) -> List[int]:
        if not APPLY_FORCE_PREDICT_MASK:
            return [1] * len(muscles_order)
        try:
            import numpy as _np
            Yf_train = _np.array([Y_force[i] for i in indices], dtype=float) if len(indices) > 0 else _np.zeros((0, len(muscles_order)), dtype=float)
            return [int(bool(_np.any(_np.abs(Yf_train[:, i]) > 0.0))) for i in range(len(muscles_order))]
        except Exception:
            return [1] * len(muscles_order)

    def _compute_via_predict_mask_extra(indices: List[int], force_mask_train: List[int]) -> List[int]:
        via_mask = [1] * len(muscles_order)
        if not (APPLY_VIA_PREDICT_MASK_EXTRA and APPLY_FORCE_PREDICT_MASK):
            return via_mask
        try:
            import numpy as _np
            tolerance = 0.01  # mm
            criterion = 0.95
            Yv_res_arr = _np.array([Y_via_res[i] for i in indices], dtype=float) if len(indices) > 0 else _np.zeros((0, sum(via_len.values()) * 3), dtype=float)
            via_start_idx: Dict[str, int] = {}
            off = 0
            for muscle in muscles_order:
                via_start_idx[muscle] = off
                off += int(via_len.get(muscle, 0)) * 3
            for mi, muscle in enumerate(muscles_order):
                if int(force_mask_train[mi]) == 0:
                    via_mask[mi] = 0
                    continue
                if muscle not in via_len or via_len[muscle] == 0:
                    continue
                start_idx = via_start_idx[muscle]
                end_idx = start_idx + (int(via_len[muscle]) * 3)
                via_residuals = Yv_res_arr[:, start_idx:end_idx]
                if via_residuals.size == 0:
                    continue
                fraction_near_zero = float(_np.mean(_np.abs(via_residuals) < tolerance))
                if fraction_near_zero > criterion:
                    via_mask[mi] = 0
            print(f"Via mask extra (train-derived): excluded {sum(1 for m in via_mask if m == 0)} height-scaled muscles")
        except Exception as e:
            print(f"Warning: Could not compute via mask extra from train split: {e}")
        return via_mask

    force_predict_mask = _compute_force_predict_mask(train_idx)
    via_predict_mask_extra = _compute_via_predict_mask_extra(train_idx, force_predict_mask)

    # Apply train-derived masks to all splits after the split is fixed.
    try:
        for r in Y_force_bl:
            for i, mbit in enumerate(force_predict_mask):
                if int(mbit) == 0:
                    r[i] = 0.0
        for r in Y_force_res:
            for i, mbit in enumerate(force_predict_mask):
                if int(mbit) == 0:
                    r[i] = 0.0
    except Exception:
        pass

    out_dir = DATASETS_DIR.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    def sel(lst, ids):
        return [lst[i] for i in ids]

    # Compute stats for normalization
    import numpy as np

    if TRAIN_ONLY_STATS:
        # New behavior: compute stats from train set only (avoid data leakage)
        def _sel_np(lst):
            return np.array([lst[i] for i in train_idx], dtype=float)
        X_tr_only = _sel_np(X)
        X_mu = X_tr_only.mean(axis=0).tolist()
        X_sd = (X_tr_only.std(axis=0) + 1e-8).tolist()
        Xs_tr_only = _sel_np(X_seq)  # [N,T,F]
        Xs_mu = Xs_tr_only.reshape(len(Xs_tr_only), -1, Xs_tr_only.shape[-1]).mean(axis=(0, 1)).tolist()
        Xs_sd = (Xs_tr_only.reshape(len(Xs_tr_only), -1, Xs_tr_only.shape[-1]).std(axis=(0, 1)) + 1e-8).tolist()
        Yf_res_tr_only = _sel_np(Y_force_res)
        Yf_mu = Yf_res_tr_only.mean(axis=0).tolist()
        Yf_sd = (Yf_res_tr_only.std(axis=0) + 1e-8).tolist()
        if len(Y_force_log) > 0:
            Yf_log_tr_only = _sel_np(Y_force_log)
            Yf_log_mu = Yf_log_tr_only.mean(axis=0).tolist()
            Yf_log_sd = (Yf_log_tr_only.std(axis=0) + 1e-8).tolist()
        else:
            Yf_log_mu = [0.0] * len(muscles_order)
            Yf_log_sd = [1.0] * len(muscles_order)
        Yv_res_tr_only = _sel_np(Y_via_res)
        Yv_mu = Yv_res_tr_only.mean(axis=0).tolist()
        Yv_sd = (Yv_res_tr_only.std(axis=0) + 1e-8).tolist()
        # Baseline stats for using as inputs (train-only)
        Yf_bl_tr_only = _sel_np(Y_force_bl)
        Yf_bl_mu = Yf_bl_tr_only.mean(axis=0).tolist()
        Yf_bl_sd = (Yf_bl_tr_only.std(axis=0) + 1e-8).tolist()
        Yv_bl_tr_only = _sel_np(Y_via_bl)
        Yv_bl_mu = Yv_bl_tr_only.mean(axis=0).tolist()
        Yv_bl_sd = (Yv_bl_tr_only.std(axis=0) + 1e-8).tolist()
    else:
        # Original: compute stats from all data
        X_all = np.array(X, dtype=float)
        X_mu = X_all.mean(axis=0).tolist()
        X_sd = (X_all.std(axis=0) + 1e-8).tolist()
        Xs_all = np.array(X_seq, dtype=float)  # [N,T,F]
        Xs_mu = Xs_all.reshape(len(X_seq), -1, Xs_all.shape[-1]).mean(axis=(0, 1)).tolist()
        Xs_sd = (Xs_all.reshape(len(X_seq), -1, Xs_all.shape[-1]).std(axis=(0, 1)) + 1e-8).tolist()
        Yf_res_all = np.array(Y_force_res, dtype=float)
        Yf_mu = Yf_res_all.mean(axis=0).tolist()
        Yf_sd = (Yf_res_all.std(axis=0) + 1e-8).tolist()
        if len(Y_force_log) > 0:
            Yf_log_all = np.array(Y_force_log, dtype=float)
            Yf_log_mu = Yf_log_all.mean(axis=0).tolist()
            Yf_log_sd = (Yf_log_all.std(axis=0) + 1e-8).tolist()
        else:
            Yf_log_mu = [0.0] * len(muscles_order)
            Yf_log_sd = [1.0] * len(muscles_order)
        Yv_res_all = np.array(Y_via_res, dtype=float)
        Yv_mu = Yv_res_all.mean(axis=0).tolist()
        Yv_sd = (Yv_res_all.std(axis=0) + 1e-8).tolist()
        # Baseline stats
        Yf_bl_all = np.array(Y_force_bl, dtype=float)
        Yf_bl_mu = Yf_bl_all.mean(axis=0).tolist()
        Yf_bl_sd = (Yf_bl_all.std(axis=0) + 1e-8).tolist()
        Yv_bl_all = np.array(Y_via_bl, dtype=float)
        Yv_bl_mu = Yv_bl_all.mean(axis=0).tolist()
        Yv_bl_sd = (Yv_bl_all.std(axis=0) + 1e-8).tolist()

    stats = {
        "X": {"mean": X_mu, "std": X_sd},
        "X_seq": {"mean": Xs_mu, "std": Xs_sd},
        "Y_force_res": {"mean": Yf_mu, "std": Yf_sd},
        "Y_via_res": {"mean": Yv_mu, "std": Yv_sd},
        "Y_force_baseline": {"mean": Yf_bl_mu, "std": Yf_bl_sd},
        "Y_via_baseline": {"mean": Yv_bl_mu, "std": Yv_bl_sd},
    }
    if APPLY_FORCE_PREDICT_MASK:
        stats["FORCE_PREDICT_MASK"] = force_predict_mask
    if APPLY_VIA_PREDICT_MASK_EXTRA:
        stats["VIA_PREDICT_MASK_EXTRA"] = via_predict_mask_extra
    # Include log-ratio stats if computed
    if len(Y_force_log) > 0:
        stats["Y_force_log"] = {"mean": Yf_log_mu, "std": Yf_log_sd}

    # Store flags used to generate this dataset for reproducibility
    stats["_build_flags"] = {
        "FORCE_LOG": FORCE_LOG,
        "CORB_HEIGHT_SCALING": CORB_HEIGHT_SCALING,
        "APPLY_FORCE_PREDICT_MASK": APPLY_FORCE_PREDICT_MASK,
        "APPLY_VIA_PREDICT_MASK": APPLY_VIA_PREDICT_MASK,
        "APPLY_VIA_PREDICT_MASK_EXTRA": APPLY_VIA_PREDICT_MASK_EXTRA,
        "VIA_MILLIMETERS": VIA_MILLIMETERS,
        "VALIDATION_SPLIT": VALIDATION_SPLIT,
        "FIXED_LENGTH_FEATURES": FIXED_LENGTH_FEATURES,
        "TRAIN_ONLY_STATS": TRAIN_ONLY_STATS,
        "POS_ONLY": POS_ONLY,
        "FORCE_BASELINE_SCALING": FORCE_BASELINE_SCALING,
        "USE_BASELINE_SUMMARY_FEATURES": USE_BASELINE_SUMMARY_FEATURES,
        "INCLUDE_AGE_FEATURE": INCLUDE_AGE_FEATURE,
        "USE_ALL_VERTEBRAE": USE_ALL_VERTEBRAE,
        "NUM_LUMBAR": len(LUMBAR),
        "NUM_THORACIC": len(THORACIC),
        "NUM_TARGET_BODIES": len(TARGET_BODIES),
    }

    with open(out_dir / "stats.json", "w") as f:
        json.dump(stats, f)

    train_obj = {
        "X": sel(X, train_idx),
        "X_seq": sel(X_seq, train_idx),
        "X_seq_mask": sel(X_seq_mask, train_idx),
        "AUX": sel(AUX, train_idx),
        "Y_force": sel(Y_force, train_idx),
        "Y_via": sel(Y_via, train_idx),
        "Y_force_baseline": sel(Y_force_bl, train_idx),
        "Y_via_baseline": sel(Y_via_bl, train_idx),
        "Y_force_res": sel(Y_force_res, train_idx),
        "Y_via_res": sel(Y_via_res, train_idx),
        "IDS": sel(IDS, train_idx),
        "MUSCLES": muscles_order,
        "VIA_LEN": via_len,
        "SEQ_TOKENS": SEQ_TOKENS,
    }
    if len(Y_force_log) > 0:
        train_obj["Y_force_log"] = sel(Y_force_log, train_idx)
        train_obj["Y_force_template"] = sel(Y_force_template, train_idx)
    if APPLY_FORCE_PREDICT_MASK:
        train_obj["FORCE_PREDICT_MASK"] = force_predict_mask
    if APPLY_VIA_PREDICT_MASK_EXTRA:
        train_obj["VIA_PREDICT_MASK_EXTRA"] = via_predict_mask_extra
    with open(out_dir / "train.json", "w") as f:
        json.dump(train_obj, f)

    # Validation object (only if VALIDATION_SPLIT is enabled)
    if VALIDATION_SPLIT and len(val_idx) > 0:
        val_obj = {
            "X": sel(X, val_idx),
            "X_seq": sel(X_seq, val_idx),
            "X_seq_mask": sel(X_seq_mask, val_idx),
            "AUX": sel(AUX, val_idx),
            "Y_force": sel(Y_force, val_idx),
            "Y_via": sel(Y_via, val_idx),
            "Y_force_baseline": sel(Y_force_bl, val_idx),
            "Y_via_baseline": sel(Y_via_bl, val_idx),
            "Y_force_res": sel(Y_force_res, val_idx),
            "Y_via_res": sel(Y_via_res, val_idx),
            "IDS": sel(IDS, val_idx),
            "MUSCLES": muscles_order,
            "VIA_LEN": via_len,
            "SEQ_TOKENS": SEQ_TOKENS,
        }
        if len(Y_force_log) > 0:
            val_obj["Y_force_log"] = sel(Y_force_log, val_idx)
            val_obj["Y_force_template"] = sel(Y_force_template, val_idx)
        if APPLY_FORCE_PREDICT_MASK:
            val_obj["FORCE_PREDICT_MASK"] = force_predict_mask
        if APPLY_VIA_PREDICT_MASK_EXTRA:
            val_obj["VIA_PREDICT_MASK_EXTRA"] = via_predict_mask_extra
        with open(out_dir / "val.json", "w") as f:
            json.dump(val_obj, f)

    test_obj = {
        "X": sel(X, test_idx),
        "X_seq": sel(X_seq, test_idx),
        "X_seq_mask": sel(X_seq_mask, test_idx),
        "AUX": sel(AUX, test_idx),
        "Y_force": sel(Y_force, test_idx),
        "Y_via": sel(Y_via, test_idx),
        "Y_force_baseline": sel(Y_force_bl, test_idx),
        "Y_via_baseline": sel(Y_via_bl, test_idx),
        "Y_force_res": sel(Y_force_res, test_idx),
        "Y_via_res": sel(Y_via_res, test_idx),
        "IDS": sel(IDS, test_idx),
        "MUSCLES": muscles_order,
        "VIA_LEN": via_len,
        "SEQ_TOKENS": SEQ_TOKENS,
    }
    if len(Y_force_log) > 0:
        test_obj["Y_force_log"] = sel(Y_force_log, test_idx)
        test_obj["Y_force_template"] = sel(Y_force_template, test_idx)
    if APPLY_FORCE_PREDICT_MASK:
        test_obj["FORCE_PREDICT_MASK"] = force_predict_mask
    if APPLY_VIA_PREDICT_MASK_EXTRA:
        test_obj["VIA_PREDICT_MASK_EXTRA"] = via_predict_mask_extra
    with open(out_dir / "test.json", "w") as f:
        json.dump(test_obj, f)

    print("Wrote dataset to", out_dir, f"in {time.time()-t0:.1f}s (seed={SEED})")


if __name__ == "__main__":
    main()
