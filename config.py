"""
Configuration module for the Psychometric & Epistemic BI Dashboard.
Defines the Ocean Beach Palette, item-to-aspect mappings, and TGcodes.csv label loader.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

# ==============================================================================
# 1. FILE PATH RESOLUTION (READ-ONLY INPUTS)
# ==============================================================================
SEARCH_DIRS: List[Path] = [
    Path(__file__).resolve().parent,
    Path.cwd(),
    Path("/working_dir/c_bfcab6d988141710/user_uploaded_files/ttl=63d/user_uploaded"),
    Path(__file__).resolve().parent.parent / "user_uploaded_files" / "ttl=63d" / "user_uploaded",
]


def resolve_input_file(filename: str) -> Path:
    """Locate a read-only input CSV file across standard repository and upload paths."""
    for directory in SEARCH_DIRS:
        candidate = directory / filename
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Unable to locate required read-only input file: {filename}")


# ==============================================================================
# 2. OCEAN BEACH PALETTE & UI THEME CONSTANTS
# Source: https://pikka.app/palettes/ocean-beach-palette
# ==============================================================================
OCEAN_BEACH_PALETTE: Dict[str, str] = {
    "deep_ocean_blue": "#003f5c",
    "twilight_indigo": "#2f4b7c",
    "dusk_purple": "#665191",
    "berry_violet": "#a05195",
    "coral_rose": "#d45087",
    "sunset_crimson": "#f95d6a",
    "warm_tangerine": "#ff7c43",
    "golden_sun": "#ffa600",
}

OCEAN_BEACH_SEQUENCE: List[str] = [
    "#003f5c",
    "#2f4b7c",
    "#665191",
    "#a05195",
    "#d45087",
    "#f95d6a",
    "#ff7c43",
    "#ffa600",
]

UI_THEME: Dict[str, str] = {
    "bg_main": "#f8f9fa",          # Main Coastal Foam
    "bg_card": "#eef2f5",          # Cool Coastal Card Surface
    "bg_sidebar": "#fff9f0",       # Warm Sand Sidebar/Card Surface
    "text_primary": "#003f5c",     # Primary Headers / Text
    "text_secondary": "#2f4b7c",   # Secondary / Muted Text
    "accent_fct": "#003f5c",       # Deep Ocean for FCT / Cognitive Constructs
    "accent_fct_alt": "#2f4b7c",   # Twilight Indigo secondary cognitive accent
    "accent_crd": "#f95d6a",       # Sunset Crimson for CRD / Affective Constructs
    "accent_crd_alt": "#ff7c43",   # Warm Tangerine secondary affective accent
    "accent_gold": "#ffa600",      # Golden Sun highlight
    "border_color": "#d6e0e8",
}

# Likert 1-5 Diverging Gradient (Strongly Disagree -> Strongly Agree)
LIKERT_COLORS: Dict[int, str] = {
    1: "#003f5c",  # 1 - Strongly Disagree (Deep Ocean Blue)
    2: "#665191",  # 2 - Disagree (Dusk Purple)
    3: "#d45087",  # 3 - Neutral / Undecided (Coral Rose)
    4: "#ff7c43",  # 4 - Agree (Warm Tangerine)
    5: "#ffa600",  # 5 - Strongly Agree (Golden Sun)
}

LIKERT_LABELS: Dict[int, str] = {
    1: "1 - Strongly Disagree",
    2: "2 - Disagree",
    3: "3 - Neutral",
    4: "4 - Agree",
    5: "5 - Strongly Agree",
}

# ==============================================================================
# 3. THEORETICAL CONSTRUCT & ASPECT ITEM MAPPINGS
# ==============================================================================
# Survey 1: 28 items (S1..S15, S17..S29; S16 does not exist)
SCIENCE_COGNITIVE_ITEMS: List[str] = [
    "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10",
    "S11", "S12", "S13", "S14", "S15", "S17", "S18", "S19", "S20", "S21",
    "S22", "S23", "S24", "S25", "S26", "S27", "S28",
]
SCIENCE_AFFECTIVE_ITEMS: List[str] = ["S29"]
ALL_SCIENCE_ITEMS: List[str] = SCIENCE_COGNITIVE_ITEMS + SCIENCE_AFFECTIVE_ITEMS

# Survey 2: 20 items (E1..E20)
ALL_EVOLUTION_ITEMS: List[str] = [f"E{i}" for i in range(1, 21)]

EVO_FCT_ITEMS: List[str] = ["E1", "E3", "E8", "E11", "E12", "E13", "E15", "E16", "E18", "E20"]
EVO_CRD_ITEMS: List[str] = ["E2", "E4", "E5", "E6", "E7", "E9", "E10", "E14", "E17", "E19"]

# 14 Science Aspects (13 Cognitive under SCI + 1 Affective under ATS)
SCIENCE_ASPECT_ITEMS: Dict[str, List[str]] = {
    "cns": ["S1", "S2"],
    "prv": ["S3", "S4"],
    "tsp": ["S5", "S6"],
    "tlm": ["S7", "S8", "S28"],
    "soc": ["S9", "S10"],
    "com": ["S11", "S12"],
    "app": ["S13", "S14"],
    "cre": ["S15", "S17"],
    "lin": ["S18", "S19"],
    "cnt": ["S20", "S21"],
    "tes": ["S22", "S23"],
    "cmp": ["S24", "S25"],
    "lmt": ["S26", "S27"],
    "ats": ["S29"],
}

SCIENCE_COGNITIVE_ASPECTS: List[str] = [k for k in SCIENCE_ASPECT_ITEMS if k != "ats"]
SCIENCE_ALL_ASPECTS: List[str] = list(SCIENCE_ASPECT_ITEMS.keys())

# 6 Evolution Theoretical Aspects
EVO_ASPECT_ITEMS: Dict[str, List[str]] = {
    "val": ["E2", "E10", "E12", "E13", "E20"],
    "pro": ["E1", "E9", "E18", "E19"],
    "evi": ["E4", "E6", "E8", "E16"],
    "bel": ["E7", "E11", "E14"],
    "hum": ["E3", "E15"],
    "scc": ["E5", "E17"],
}

EVO_ALL_ASPECTS: List[str] = list(EVO_ASPECT_ITEMS.keys())

# 10 Evolution Epistemic Sub-Aspects (Split by FCT vs CRD)
EVO_SPLIT_SUBASPECT_ITEMS: Dict[str, List[str]] = {
    "val_FCT": ["E12", "E13", "E20"],
    "val_CRD": ["E2", "E10"],
    "pro_FCT": ["E1", "E18"],
    "pro_CRD": ["E9", "E19"],
    "evi_FCT": ["E8", "E16"],
    "evi_CRD": ["E4", "E6"],
    "bel_FCT": ["E11"],
    "bel_CRD": ["E7", "E14"],
    "hum_FCT": ["E3", "E15"],
    "scc_CRD": ["E5", "E17"],
}

EVO_OVERLAPPING_ASPECTS: List[str] = ["val", "pro", "evi", "bel"]

# Allowed Demographic Moderators (Religion is strictly excluded)
ACTIVE_DEMOGRAPHICS: List[str] = ["Age", "Gender", "Religiosity"]

# Primary Composites
SCIENCE_PREDICTORS: List[str] = ["SCI", "SKSI", "ATS"]
EVOLUTION_CRITERIA: List[str] = ["EVO", "FCT", "CRD"]


# ==============================================================================
# 4. TGCODES.CSV LABEL LOADER & FORMATTERS
# ==============================================================================
def load_tg_codes(filepath: Path | None = None) -> Tuple[pd.DataFrame, Dict[str, str], Dict[str, str]]:
    """
    Dynamically parse TGcodes.csv (dropping empty trailing rows/columns) to map
    shorthand codes to human-readable meanings and formatted 'Meaning (code)' labels.
    """
    path = filepath or resolve_input_file("TGcodes.csv")
    raw_df = pd.read_csv(path)

    # Keep only valid columns (code, Type, Meaning) and drop empty rows
    valid_cols = [c for c in ["code", "Type", "Meaning"] if c in raw_df.columns]
    clean_df = raw_df[valid_cols].dropna(subset=["code", "Meaning"]).copy()
    clean_df["code"] = clean_df["code"].astype(str).str.strip()
    clean_df["Type"] = clean_df["Type"].astype(str).str.strip()
    clean_df["Meaning"] = clean_df["Meaning"].astype(str).str.strip()

    code_to_meaning: Dict[str, str] = dict(zip(clean_df["code"], clean_df["Meaning"]))
    # Add SKSI composite explicitly per specification
    code_to_meaning["SKSI"] = "Scientific Knowledge & Inquiry"

    code_to_label: Dict[str, str] = {
        code: f"{meaning} ({code})" for code, meaning in code_to_meaning.items()
    }

    # Add readable labels for the 10 Epistemic Split EVO Sub-Aspects
    for sub_code in EVO_SPLIT_SUBASPECT_ITEMS:
        base_aspect, ep_type = sub_code.split("_")
        base_meaning = code_to_meaning.get(base_aspect, base_aspect)
        ep_meaning = code_to_meaning.get(ep_type, ep_type)
        code_to_meaning[sub_code] = f"{base_meaning} - {ep_meaning}"
        code_to_label[sub_code] = f"{base_meaning}: {ep_meaning} ({sub_code})"

    return clean_df, code_to_meaning, code_to_label
