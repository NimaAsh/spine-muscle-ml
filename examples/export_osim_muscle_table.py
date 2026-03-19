from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from osim_parser import OSIMModel  # noqa: E402
from repo_config import OUTPUTS_DIR, female_template_path, get_data_root, male_template_path  # noqa: E402

DEFAULT_INPUT_DIR = OUTPUTS_DIR
DEFAULT_TEMPLATE_MALE = male_template_path(get_data_root()) if get_data_root() else None
DEFAULT_TEMPLATE_FEMALE = female_template_path(get_data_root()) if get_data_root() else None
DEFAULT_FORCE_CSV = None
DATAVERSE_ROOT = get_data_root()
USE_CORB_HEIGHT_RATIO = True


def mat4_to_Rp(X):
    R = [row[:3] for row in X[:3]]
    p = [X[0][3], X[1][3], X[2][3]]
    return R, p
def _get_pathpoint_x(model: OSIMModel, muscle_name: str, point_name: str) -> float | None:
    try:
        m = model.data.get("forces", {}).get(muscle_name) or {}
        pts = m.get("path_points") or []
        for pt in pts:
            nm = pt.get("name")
            if isinstance(nm, str) and nm == point_name:
                loc = pt.get("location") or [0.0, 0.0, 0.0]
                return float(loc[0])
    except Exception:
        pass
    return None


def _corb_height_ratio(model: OSIMModel, template: OSIMModel) -> float | None:
    subj_x = _get_pathpoint_x(model, "CORB", "PathPoint_scapula_R")
    tmpl_x = _get_pathpoint_x(template, "CORB", "PathPoint_scapula_R")
    if subj_x is None or tmpl_x is None:
        return None
    if abs(tmpl_x) < 1e-8:
        return None
    return float(subj_x / tmpl_x)


def _estimate_height(model: OSIMModel, template: OSIMModel | None = None) -> float:
    if USE_CORB_HEIGHT_RATIO and template is not None:
        ratio = _corb_height_ratio(model, template)
        if ratio is not None:
            h_tmpl = _estimate_height(template, None)
            if np.isfinite(h_tmpl) and h_tmpl > 0:
                return float(ratio * h_tmpl)
    root = "sacrum"
    q = {}
    for name in ("head_neck", "thoracic12", "thoracic5"):
        if name in model.data.get("bodies", {}):
            X = model.transform_relative_to(name, root, q)
            return float(mat4_to_Rp(X)[1][1])
    return float("nan")


def _estimate_weight(model: OSIMModel, template: OSIMModel | None = None, sex: str = "") -> float:
    if template is not None:
        try:
            ms = float(model.data["bodies"]["sacrum"]["mass"])  # type: ignore[index]
            mt = float(template.data["bodies"]["sacrum"]["mass"])  # type: ignore[index]
            if mt != 0:
                ratio = ms / mt
                if not sex:
                    sex = "M" if "male" in template.path.as_posix().lower() else "F"
                generic = 78.0 if sex == "M" else 61.0
                return float(generic * ratio)
        except Exception:
            pass
    total = 0.0
    for b in model.data.get("bodies", {}).values():
        m = b.get("mass")
        if isinstance(m, (int, float)):
            total += float(m)
    return float(total) if total > 0 else float("nan")


def _get_patient_id_from_path(path: Path) -> Optional[str]:
    """Extract 3-digit patient ID from filename (before first '_')."""
    stem = path.stem
    parts = stem.split("_", 1)
    if len(parts) > 0:
        candidate = parts[0]
        if candidate.isdigit() and len(candidate) == 3:
            return candidate
    return None


def _lookup_sex_in_dataverse(patient_id: str, dataverse_root: Path | None = DATAVERSE_ROOT) -> str:
    """Search for patient ID in Male/Female folders and return 'M' (male) or 'F' (female) or ''."""
    if dataverse_root is None:
        return ""
    male_dir = dataverse_root / "Male"
    female_dir = dataverse_root / "Female"
    
    # Search in Male folder
    if male_dir.exists():
        for osim_file in male_dir.rglob(f"{patient_id}_*.osim"):
            return "M"
    
    # Search in Female folder
    if female_dir.exists():
        for osim_file in female_dir.rglob(f"{patient_id}_*.osim"):
            return "F"
    
    return ""


def _infer_sex(path: Path) -> str:
    """Infer sex from patient ID lookup or path string. Returns 'M', 'F', or ''."""
    # Try patient ID lookup first
    patient_id = _get_patient_id_from_path(path)
    if patient_id is not None:
        sex = _lookup_sex_in_dataverse(patient_id)
        if sex:
            return sex
    
    # Fallback to path string search
    pstr = path.as_posix().lower()
    if "male" in pstr and "female" not in pstr:
        return "M"
    if "female" in pstr:
        return "F"
    return ""


def collect_muscle_specs(paths: List[Path]) -> Tuple[List[str], Dict[str, int]]:
    muscles: set[str] = set()
    via_len: Dict[str, int] = {}
    for p in paths:
        try:
            model = OSIMModel.from_file(p)
        except Exception:
            continue
        forces = model.data.get("forces", {})
        for name, info in forces.items():
            muscles.add(name)
            pts = info.get("path_points") or []
            via_len[name] = max(via_len.get(name, 0), len(pts))
    return sorted(muscles), via_len


def build_row(model: OSIMModel, path: Path, muscles: List[str], via_len: Dict[str, int], template: OSIMModel | None) -> Dict[str, float | str]:
    row: Dict[str, float | str] = {}
    stem = Path(path).stem
    prefix = stem.split("_", 1)[0]
    row["name"] = prefix
    row["filepath"] = str(path)

    sex = _infer_sex(path)
    h_est = _estimate_height(model, template)
    w_est = _estimate_weight(model, template, sex)

    row["height"] = h_est
    row["weight"] = w_est
    row["sex"] = sex

    forces = model.data.get("forces", {})

    for mus in muscles:
        info = forces.get(mus) or {}
        f = info.get("max_isometric_force")
        row[f"force_{mus}"] = float(f) if isinstance(f, (int, float)) else np.nan

        pts = info.get("path_points") or []
        max_pts = via_len.get(mus, 0)
        for i in range(max_pts):
            col_x = f"via_{mus}_pt{i}_x"
            col_y = f"via_{mus}_pt{i}_y"
            col_z = f"via_{mus}_pt{i}_z"
            if i < len(pts):
                loc = pts[i].get("location") or [np.nan, np.nan, np.nan]
                try:
                    x, y, z = float(loc[0]), float(loc[1]), float(loc[2])
                except Exception:
                    x = y = z = np.nan
            else:
                x = y = z = np.nan
            row[col_x] = x
            row[col_y] = y
            row[col_z] = z

    return row


def process_directory(
    input_dir: Path,
    template_male: Path | None = None,
    template_female: Path | None = None,
) -> pd.DataFrame:
    paths = sorted(input_dir.rglob("*.osim"))
    if not paths:
        raise FileNotFoundError(f"No .osim files found in {input_dir}")

    muscles, via_len = collect_muscle_specs(paths)

    rows = []
    template_cache: Dict[str, OSIMModel] = {}
    for p in paths:
        try:
            model = OSIMModel.from_file(p)
        except Exception as e:
            print(f"Skip {p} due to {e}")
            continue
        template = None
        if USE_CORB_HEIGHT_RATIO:
            sex = _infer_sex(p)
            key: str | None = "male" if sex == "M" else ("female" if sex == "F" else None)
            tpl_path: Path | None = None
            if key == "male":
                tpl_path = template_male
            elif key == "female":
                tpl_path = template_female
            if tpl_path is not None and tpl_path.exists():
                if key not in template_cache:
                    template_cache[key] = OSIMModel.from_file(tpl_path)
                template = template_cache.get(key)
        rows.append(build_row(model, p, muscles, via_len, template))

    df = pd.DataFrame(rows)
    column_order = ["name", "filepath", "height", "weight", "sex"]
    for mus in muscles:
        column_order.append(f"force_{mus}")
        for i in range(via_len[mus]):
            column_order.extend(
                [f"via_{mus}_pt{i}_x", f"via_{mus}_pt{i}_y", f"via_{mus}_pt{i}_z"]
            )
    df = df.reindex(columns=column_order)
    return df


def _parse_force_index(df: pd.DataFrame, column: str) -> pd.DataFrame:
    pattern = re.compile(r"^(?P<sample_id>[^_]+).*_motion_(?P<motion_id>\d+)$")

    sample_ids: List[str] = []
    motion_ids: List[int | float] = []
    for rec in df[column].astype(str):
        match = pattern.match(rec)
        if match:
            sample_ids.append(match.group("sample_id"))
            motion_ids.append(int(match.group("motion_id")))
        else:
            sample_ids.append(rec.split("_", 1)[0] if "_" in rec else rec)
            try:
                motion_ids.append(int(rec.rsplit("_", 1)[-1]))
            except ValueError:
                motion_ids.append(float("nan"))
    df["sample_id"] = sample_ids
    df["motion_id"] = motion_ids
    cols = [column, "sample_id", "motion_id"] + [c for c in df.columns if c not in {column, "sample_id", "motion_id"}]
    return df.loc[:, cols]


def load_force_dataframe(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, index_col=0)
    df = df.reset_index().rename(columns={"index": "record"})
    return _parse_force_index(df, "record")


def load_force_dataframes(csv_paths: Iterable[Path]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in csv_paths:
        try:
            frames.append(load_force_dataframe(path))
        except FileNotFoundError:
            print(f"Missing force CSV: {path}")
        except Exception as e:
            print(f"Failed to parse {path}: {e}")
    if not frames:
        raise ValueError("No force CSVs were loaded")
    return pd.concat(frames, ignore_index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export OSIM muscle table to Pandas DataFrame/CSV")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(DEFAULT_INPUT_DIR),
        help=f"Directory containing .osim files (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save CSV (if omitted, prints DataFrame info)",
    )
    parser.add_argument(
        "--template-male",
        type=Path,
        default=DEFAULT_TEMPLATE_MALE,
        help="Path to male template OSIM for CORB height estimation",
    )
    parser.add_argument(
        "--template-female",
        type=Path,
        default=DEFAULT_TEMPLATE_FEMALE,
        help="Path to female template OSIM for CORB height estimation",
    )
    parser.add_argument(
        "--force-csv",
        type=Path,
        default=DEFAULT_FORCE_CSV,
        help="Optional CSV of predicted forces to parse",
    )
    parser.add_argument(
        "--force-output",
        type=Path,
        default=None,
        help="Optional path to save processed force CSV",
    )
    parser.add_argument(
        "--force-csv-extra",
        type=Path,
        nargs="*",
        default=None,
        help="Optional additional force CSVs to merge (useful in notebooks)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tpl_male = args.template_male.resolve() if args.template_male else None
    tpl_female = args.template_female.resolve() if args.template_female else None
    df = process_directory(args.input_dir.resolve(), tpl_male, tpl_female)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"Wrote {len(df)} rows to {args.output}")
    else:
        print(df.head())
        print(f"Total rows: {len(df)}  Total columns: {df.shape[1]}")

    force_paths: List[Path] = []
    if args.force_csv:
        force_paths.append(args.force_csv.resolve())
    if args.force_csv_extra:
        force_paths.extend([p.resolve() for p in args.force_csv_extra])

    if force_paths:
        try:
            force_df = load_force_dataframes(force_paths)
        except Exception as e:
            print(f"Failed to load force CSVs: {e}")
            force_df = None
        if args.force_output:
            args.force_output.parent.mkdir(parents=True, exist_ok=True)
        if force_df is not None:
            force_df.to_csv(args.force_output, index=False)
            print(f"Wrote processed force CSV with {len(force_df)} rows to {args.force_output}")
    elif force_paths and force_df is not None:
        print(force_df.head())
        print(f"Force rows: {len(force_df)}  Force columns: {force_df.shape[1]}")


if __name__ == "__main__":
    main()
