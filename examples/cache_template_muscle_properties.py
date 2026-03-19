#!/usr/bin/env python3
"""Generate cached muscle length properties from generic template models.

This script computes and stores:
- Total muscle length (sum of via point distances)
- optimal_fiber_length
- tendon_slack_length

for both male and female generic templates.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from osim_parser import OSIMModel
from repo_config import DATASETS_DIR, female_template_path, male_template_path, require_data_root


DATA_ROOT = require_data_root()
GENERIC_MALE = male_template_path(DATA_ROOT)
GENERIC_FEMALE = female_template_path(DATA_ROOT)
OUTPUT_PATH = DATASETS_DIR / "generic_muscle_properties.json"


def extract_muscle_properties(model: OSIMModel, muscles: list[str]) -> dict[str, dict]:
    """Extract length, optimal_fiber_length, and tendon_slack_length for each muscle."""
    properties = {}
    
    for muscle_name in muscles:
        try:
            # Calculate total muscle length
            length = model.calculate_muscle_length(muscle_name)
            
            # Get other properties
            mus_data = model.data.get("forces", {}).get(muscle_name, {})
            ofl = mus_data.get("optimal_fiber_length")
            tsl = mus_data.get("tendon_slack_length")
            
            properties[muscle_name] = {
                "length": float(length),
                "optimal_fiber_length": float(ofl) if ofl is not None else None,
                "tendon_slack_length": float(tsl) if tsl is not None else None,
            }
        except Exception as e:
            print(f"Warning: Failed to process muscle {muscle_name}: {e}")
            properties[muscle_name] = {
                "length": None,
                "optimal_fiber_length": None,
                "tendon_slack_length": None,
            }
    
    return properties


def main():
    print("Loading generic models...")
    male_model = OSIMModel.from_file(GENERIC_MALE)
    female_model = OSIMModel.from_file(GENERIC_FEMALE)
    
    # Get common muscles between both templates
    m_male = set(male_model.data.get("forces", {}).keys())
    m_fem = set(female_model.data.get("forces", {}).keys())
    muscles_order = sorted(m_male & m_fem)
    
    print(f"Processing {len(muscles_order)} common muscles...")
    
    # Extract properties for both
    male_props = extract_muscle_properties(male_model, muscles_order)
    female_props = extract_muscle_properties(female_model, muscles_order)
    
    # Combine into output structure
    output = {
        "male": male_props,
        "female": female_props,
        "muscles": muscles_order,
        "source_male": GENERIC_MALE,
        "source_female": GENERIC_FEMALE,
    }
    
    # Save to JSON
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"Saved template muscle properties to: {OUTPUT_PATH}")
    
    # Print summary statistics
    male_lengths = [p["length"] for p in male_props.values() if p["length"] is not None]
    female_lengths = [p["length"] for p in female_props.values() if p["length"] is not None]
    
    if male_lengths:
        print(f"\nMale template muscle lengths: min={min(male_lengths):.4f}m, max={max(male_lengths):.4f}m, mean={sum(male_lengths)/len(male_lengths):.4f}m")
    if female_lengths:
        print(f"Female template muscle lengths: min={min(female_lengths):.4f}m, max={max(female_lengths):.4f}m, mean={sum(female_lengths)/len(female_lengths):.4f}m")


if __name__ == "__main__":
    main()
