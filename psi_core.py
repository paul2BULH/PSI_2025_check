
import argparse
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, time as dtime
from typing import Any, Dict, List, Optional, Tuple, Set

import pandas as pd
import numpy as np

PSI_LIST = [3,4,5,6,7,8,9,10,11,12,13,14,15,17,18,19]  # PSI 16 retired

# -------------------------
# Utilities
# -------------------------

def _safe_upper(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().upper()

def norm_icd10(code: Any) -> str:
    """Upper + strip + remove dot. PCS usually 7 char; we don't enforce length here."""
    c = _safe_upper(code).replace(".", "")
    return c

def norm_drg(val: Any) -> str:
    """Return 3-digit DRG string, left-padded if needed; drop decimals if present."""
    if pd.isna(val):
        return ""
    s = str(val).strip()
    if "." in s:
        s = s.split(".", 1)[0]
    # remove non-digits
    s = "".join(ch for ch in s if ch.isdigit())
    if not s:
        return ""
    return s.zfill(3)[:3]

def parse_date_time(date_val: Any, time_val: Any) -> Optional[pd.Timestamp]:
    """Best-effort parse of date & time (separately provided columns)."""
    if pd.isna(date_val) and pd.isna(time_val):
        return None
    # Try parse date
    dt_part: Optional[pd.Timestamp] = None
    if not pd.isna(date_val):
        try:
            dt_part = pd.to_datetime(date_val, errors="coerce", dayfirst=False)
        except Exception:
            dt_part = None
    # Try parse time
    tm_part: Optional[dtime] = None
    if not pd.isna(time_val):
        if isinstance(time_val, dtime):
            tm_part = time_val
        else:
            try:
                tt = pd.to_datetime(str(time_val), errors="coerce")
                if tt is not None and not pd.isna(tt):
                    tm_part = dtime(hour=int(tt.hour), minute=int(tt.minute), second=int(tt.second))
            except Exception:
                tm_part = None
    if dt_part is None and tm_part is None:
        return None
    if dt_part is None:
        dt_part = pd.Timestamp(datetime.combine(datetime.today().date(), tm_part or dtime(0,0)))
    if tm_part is None:
        return pd.Timestamp(year=dt_part.year, month=dt_part.month, day=dt_part.day)
    return pd.Timestamp(datetime.combine(dt_part.date(), tm_part))

def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

# -------------------------
# Data classes
# -------------------------

@dataclass
class ChecklistItem:
    criterion: str
    expected: Any
    found: Any
    passed: bool
    note: str = ""

@dataclass
class PSIEvalResult:
    encounter_id: str
    psi: str
    result: str  # "INCLUSION" | "EXCLUSION" | "NOT_APPLICABLE"
    denominator_met: bool
    numerator_met: bool
    exclusions_applied: List[str] = field(default_factory=list)
    rationale_short: str = ""
    checklist: List[ChecklistItem] = field(default_factory=list)
    debug: Dict[str, Any] = field(default_factory=dict)
    debug_path: Optional[str] = None

    def to_row(self) -> Dict[str, Any]:
        return {
            "EncounterID": self.encounter_id,
            "PSI": self.psi,
            "Result": self.result,
            "Rationale_Short": self.rationale_short,
            "Denominator_Met": self.denominator_met,
            "Numerator_Met": self.numerator_met,
            "Exclusions_Applied": "; ".join(self.exclusions_applied) if self.exclusions_applied else "",
            "Checklist_Pass_Count": sum(1 for c in self.checklist if c.passed),
            "Checklist_Total": len(self.checklist),
            "Debug_File": self.debug_path or "",

    def __post_init__(self):
        # Fill a sensible default rationale when missing or placeholder
        placeholder_vals = {"", "awaiting rule text", "awaiting rule", "awaiting rationale", "todo"}
        val = (self.rationale_short or "").strip().lower()
        if val in placeholder_vals:
            base = (str(self.result or "")).strip().upper() or "RESULT"
            denom = "denominator met" if bool(self.denominator_met) else "denominator not met"
            numer = "numerator met" if bool(self.numerator_met) else "numerator not met"
            extras = []
            if self.exclusions_applied:
                extras.append("exclusions: " + ", ".join(self.exclusions_applied))
            summary = ", ".join([denom, numer] + extras)
            self.rationale_short = f"{base} — {summary}"

# ---------- Auto rationale templating ----------
_PSI_RATIONALE_TEMPLATES = {
    "PSI_03": {
        "INCLUSION": "Pressure ulcer present (meets numerator); verify stage/timing per spec.",
        "EXCLUSION": "Excluded by denominator rules or missing qualifying conditions for PSI-03."
    },
    "PSI_04": {
        "INCLUSION": "Iatrogenic pneumothorax met per criteria (post-procedural/iatrogenic).",
        "EXCLUSION": "Principal DX/POA or denominator exclusions prevent PSI-04 inclusion."
    },
    "PSI_05": {
        "INCLUSION": "Retained surgical item documented; meets numerator.",
        "EXCLUSION": "No retained item or excluded by qualifying conditions."
    },
    "PSI_06": {
        "INCLUSION": "Postprocedural respiratory failure criteria met.",
        "EXCLUSION": "Does not meet respiratory failure time/ventilation thresholds or excluded."
    },
    "PSI_07": {
        "INCLUSION": "CLABSI indicators met within eligible admission window.",
        "EXCLUSION": "Central line infection not qualifying or excluded by conditions."
    },
    "PSI_08": {
        "INCLUSION": "In-hospital fall with fracture documented; meets numerator.",
        "EXCLUSION": "No qualifying fall-associated fracture or denominator exclusions present."
    },
    "PSI_09": {
        "INCLUSION": "Postop hemorrhage/hematoma w/ transfusion or reoperation met.",
        "EXCLUSION": "No qualifying postop bleed/re-intervention or excluded conditions present."
    },
    "PSI_10": {
        "INCLUSION": "Postop acute kidney injury requiring dialysis met.",
        "EXCLUSION": "Dialysis not required post-op or excluded per timing/POA rules."
    },
    "PSI_11": {
        "INCLUSION": "Postop respiratory failure met (intubation/vent/ICU criteria).",
        "EXCLUSION": "Does not meet duration/timing thresholds or excluded."
    },
    "PSI_12": {
        "INCLUSION": "Perioperative PE/DVT met based on DX/PROC/timing.",
        "EXCLUSION": "No qualifying VTE event or excluded by prior conditions."
    },
    "PSI_13": {
        "INCLUSION": "Postop sepsis present; numerator criteria satisfied.",
        "EXCLUSION": "Sepsis not qualifying post-op or POA/exclusions apply."
    },
    "PSI_14": {
        "INCLUSION": "Wound dehiscence after abdominal surgery met.",
        "EXCLUSION": "No qualifying dehiscence or excluded per index/follow-up pairing."
    },
    "PSI_15": {
        "INCLUSION": "Accidental puncture/laceration occurred during procedure.",
        "EXCLUSION": "No qualifying accidental puncture/laceration or excluded by rules."
    },
    "PSI_17": {
        "INCLUSION": "Birth trauma in neonate/newborn per definition met.",
        "EXCLUSION": "Does not meet neonate/newborn criteria or excluded."
    },
    "PSI_18": {
        "INCLUSION": "Obstetric trauma – vaginal delivery (with instrument) met.",
        "EXCLUSION": "No qualifying obstetric trauma event or excluded."
    },
    "PSI_19": {
        "INCLUSION": "Obstetric trauma – vaginal delivery (without instrument) met.",
        "EXCLUSION": "No qualifying obstetric trauma event or excluded."
    },
}

def _format_auto_rationale(psi_code: str, result: str, denom: bool, numer: bool, exclusions, debug) -> str:
    # Prefer a PSI-specific template; then append quick flags & appendix matches
    psi_templates = _PSI_RATIONALE_TEMPLATES.get(psi_code, {})
    base = psi_templates.get(result.upper(), f"{result.upper()} based on evaluated rules.")
    parts = [base]

    # Quick flags
    parts.append("denominator met" if denom else "denominator not met")
    parts.append("numerator met" if numer else "numerator not met")

    # Exclusions list (if any)
    if exclusions:
        parts.append("exclusions: " + ", ".join(exclusions))

    # Try to surface appendix/code-set keys from debug if present
    keys = []
    if isinstance(debug, dict):
        for k in ("appendix_keys", "matched_appendix", "matched_code_sets", "code_sets_used"):
            v = debug.get(k)
            if isinstance(v, (list, tuple, set)):
                keys.extend([str(x) for x in v])
    if keys:
        parts.append("appendix keys: " + ", ".join(sorted(set(keys))))

    return " — ".join(parts)
        
        # If we still have a generic line, upgrade with PSI-specific template and appendix keys
        try:
            self.rationale_short = _format_auto_rationale(self.psi, self.result, self.denominator_met, self.numerator_met, self.exclusions_applied, self.debug)
        except Exception:
            pass
        }

# -------------------------
# Registry for PSI functions
# -------------------------

_PSI_REGISTRY: Dict[int, Any] = {}

def register_psi(n: int):
    def deco(fn):
        _PSI_REGISTRY[n] = fn
        return fn
    return deco

# -------------------------
# Code sets loading
# -------------------------

def load_code_sets(path: str) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    normed: Dict[str, List[str]] = {}
    for k, v in data.items():
        if isinstance(v, list):
            normed[k] = [norm_icd10(x) for x in v]
        else:
            normed[k] = v
    return normed

# -------------------------
# Input normalization
# -------------------------

def normalize_input(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # Normalize common header variants (case-insensitive mapping)
    col_map = {
        'AGE': 'Age',
        'SEX': 'Sex',
        'YEAR': 'Year',
    }
    df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)


    if "MS-DRG" in df.columns:
        df["MS-DRG_3digit"] = df["MS-DRG"].apply(norm_drg)

    for col in [c for c in df.columns if c.upper().startswith("DX")]:
        df[col] = df[col].apply(norm_icd10)
    if "Pdx" in df.columns:
        df["Pdx"] = df["Pdx"].apply(norm_icd10)

    for col in [c for c in df.columns if c.upper().startswith("POA")]:
        df[col] = df[col].astype(str).str.strip().str.upper().replace({"NAN": ""})

    for n in range(1, 21):
        pcol = f"Proc{n}"
        dcol = f"Proc{n}_Date"
        tcol = f"Proc{n}_Time"
        if pcol in df.columns:
            df[pcol] = df[pcol].apply(norm_icd10)
        if dcol in df.columns and tcol in df.columns:
            df[f"Proc{n}_DT"] = df.apply(lambda r: parse_date_time(r.get(dcol, np.nan), r.get(tcol, np.nan)), axis=1)

    for col in ["Admission_Date", "Discharge_Date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if "Length_of_stay" in df.columns:
        df["Length_of_stay"] = pd.to_numeric(df["Length_of_stay"], errors="coerce").fillna(0).astype(int)

    # Age numeric (if present)
    if "Age" in df.columns:
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

    return df

def extract_dx_poa(row: pd.Series) -> List[Tuple[str, str, str]]:
    out: List[Tuple[str, str, str]] = []
    pdx = row.get("Pdx", "")
    poa1 = row.get("POA1", "")
    if pdx:
        out.append((pdx, poa1, "PRINCIPAL"))
    for i in range(1, 40):
        dx_col = f"DX{i}"
        poa_col = f"POA{i+1}"
        if dx_col in row and str(row.get(dx_col, "")).strip():
            code = norm_icd10(row[dx_col])
            poa = _safe_upper(row.get(poa_col, ""))
            out.append((code, poa, "SECONDARY"))
    return out

def extract_procedures(row: pd.Series) -> List[Tuple[str, Optional[pd.Timestamp], str]]:
    procs: List[Tuple[str, Optional[pd.Timestamp], str]] = []
    for n in range(1, 21):
        pcol = f"Proc{n}"
        dtcol = f"Proc{n}_DT"
        if pcol in row and str(row.get(pcol, "")).strip():
            code = norm_icd10(row[pcol])
            pdt = row.get(dtcol, None)
            procs.append((code, pdt if pd.notna(pdt) else None, pcol))
    return procs

def code_in_set(code: str, codes: Dict[str, List[str]], set_name: str) -> bool:
    arr = codes.get(set_name, [])
    return code in arr

def any_proc_in_set(procs: List[Tuple[str, Optional[pd.Timestamp], str]], codes: Dict[str, List[str]], set_name: str) -> bool:
    return any(code_in_set(p[0], codes, set_name) for p in procs)

# -------------------------
# PSI-03 helpers: build dynamic unions for PI/DTI site sets
# -------------------------

def build_pressure_ulcer_sets(codes: Dict[str, List[str]]):
    """
    From site-specific keys in PSI_Code_Sets_2025.json, build unions and code->site maps.

    Returns:
      pi_d_union: set of all Stage3/4/Unstageable PI diagnosis codes (secondary) (PI~D)
      pi_exd_union: set of all PI/DTI principal DX exclusions at site (PI~EXD)
      dti_exd_union: set of all DTI exclusions (DTI~EXD)
      code_to_site_pi_d: map code -> site_key (e.g., 'PILHIPD')
      code_to_site_dti_exd: map code -> site_key (e.g., 'DTILHIPEXD')
      unspecified_pi_d_site_keys: set of site-keys representing unspecified site groups (PIN*, PIUNSPECD)
    """
    pi_d_union: Set[str] = set()
    pi_exd_union: Set[str] = set()
    dti_exd_union: Set[str] = set()

    code_to_site_pi_d: Dict[str, str] = {}
    code_to_site_dti_exd: Dict[str, str] = {}

    unspecified_site_keys: Set[str] = set()

    for key, arr in codes.items():
        if not isinstance(arr, list):
            continue
        # Normalize list
        norm_arr = [str(x).upper().replace(".", "").strip() for x in arr if str(x).strip()]
        if key.startswith("PI") and key.endswith("D") and not key.endswith("EXD"):
            # PI~D site lists (numerator)
            for c in norm_arr:
                pi_d_union.add(c)
                code_to_site_pi_d[c] = key
            # track unspecified-site keys
            if key.startswith("PIN") or key == "PIUNSPECD":
                unspecified_site_keys.add(key)
        elif key.startswith("PI") and key.endswith("EXD"):
            # PI~EXD (denominator exclusion: principal DX at same site)
            for c in norm_arr:
                pi_exd_union.add(c)
        elif key.startswith("DTI") and key.endswith("EXD"):
            # DTI~EXD (numerator exclusion at same site if POA=Y)
            for c in norm_arr:
                dti_exd_union.add(c)
                code_to_site_dti_exd[c] = key

    return {
        "pi_d_union": pi_d_union,
        "pi_exd_union": pi_exd_union,
        "dti_exd_union": dti_exd_union,
        "code_to_site_pi_d": code_to_site_pi_d,
        "code_to_site_dti_exd": code_to_site_dti_exd,
        "unspecified_pi_d_site_keys": unspecified_site_keys,
    }

def same_site_match(pi_site_key: str, dti_site_key: str) -> bool:
    """Heuristic: strip prefixes and suffixes and compare mid tokens for site equality."""
    def site_core(k: str) -> str:
        k = k.upper()
        if k.startswith("PI"):
            k = k[2:]
        if k.startswith("DTI"):
            k = k[3:]
        # remove trailing D or EXD
        if k.endswith("EXD"):
            k = k[:-3]
        elif k.endswith("D"):
            k = k[:-1]
        return k
    return site_core(pi_site_key) == site_core(dti_site_key)

# -------------------------
# PSI Implementations
# -------------------------

def _not_evaluated(enc_id: str, psi_n: int, why: str) -> PSIEvalResult:
    return PSIEvalResult(
        encounter_id=enc_id,
        psi=f"PSI_{psi_n:02d}",
        result="EXCLUSION",
        denominator_met=False,
        numerator_met=False,
        exclusions_applied=[],
        rationale_short=f"Rule not yet implemented: {why}",
        checklist=[],
        debug={"status": "stub"},
    )

@register_psi(3)
def evaluate_psi03(row: pd.Series, codes: Dict[str, List[str]]) -> PSIEvalResult:
    """PSI-03 Pressure Ulcer Rate (2025) per uploaded spec."""
    enc_id = str(row.get("EncounterID", ""))
    dx_list = extract_dx_poa(row)  # list of (code, poa, role)
    drg3 = row.get("MS-DRG_3digit", "")
    age = row.get("Age", np.nan)
    los = row.get("Length_of_stay", np.nan)

    sets = build_pressure_ulcer_sets(codes)
    pi_d_union = sets["pi_d_union"]
    pi_exd_union = sets["pi_exd_union"]
    dti_exd_union = sets["dti_exd_union"]
    code_to_site_pi_d = sets["code_to_site_pi_d"]
    code_to_site_dti_exd = sets["code_to_site_dti_exd"]
    unspecified_keys = sets["unspecified_pi_d_site_keys"]

    checklist: List[ChecklistItem] = []
    exclusions: List[str] = []

    # ---- Denominator: age >= 18 AND DRG in SURGI2R or MEDIC2R
    age_ok = False
    if pd.notna(age):
        try:
            age_ok = float(age) >= 18
        except Exception:
            age_ok = False
    drg_ok = drg3 in set(str(x).zfill(3) for x in codes.get("SURGI2R", [])) or \
             drg3 in set(str(x).zfill(3) for x in codes.get("MEDIC2R", []))

    checklist.append(ChecklistItem("Age >= 18", ">=18", age, age_ok))
    checklist.append(ChecklistItem("DRG in SURGI2R or MEDIC2R (3-digit)", "Yes", drg3, drg_ok))

    denominator_met = age_ok and drg_ok

    # ---- Denominator exclusions (precedence)
    # LOS < 3
    los_excl = False
    if pd.notna(los):
        try:
            los_excl = int(los) < 3
        except Exception:
            los_excl = False
    else:
        los_excl = False
    checklist.append(ChecklistItem("Length of stay >= 3 days", ">=3", los, not los_excl))
    if los_excl:
        exclusions.append("LOS<3")

    # Principal DX is site-specific PI or DTI at same site (PI~EXD)
    principal_dx = dx_list[0][0] if dx_list else ""
    principal_is_pi_exd = principal_dx in pi_exd_union
    checklist.append(ChecklistItem("Principal DX is PI/DTI site exclusion (PI~EXD)", "No", principal_dx, not principal_is_pi_exd))
    if principal_is_pi_exd:
        exclusions.append("PrincipalDX in PI~EXD")

    # Severe burns
    has_burn = any(code in set(codes.get("BURNDX", [])) for code, _, _ in dx_list)
    checklist.append(ChecklistItem("Has severe burns (BURNDX)", "No", "Yes" if has_burn else "No", not has_burn))
    if has_burn:
        exclusions.append("BURNDX")

    # Exfoliative skin disorders (if provided)
    has_exfol = any(code in set(codes.get("EXFOLIATXD", [])) for code, _, _ in dx_list)
    checklist.append(ChecklistItem("Has exfoliative disorders (EXFOLIATXD)", "No", "Yes" if has_exfol else "No", not has_exfol))
    if has_exfol:
        exclusions.append("EXFOLIATXD")

    # Obstetric (MDC 14) / Newborn (MDC 15) based on principal DX codes
    mdc14_hit = principal_dx in set(codes.get("MDC14PRINDX", []))
    mdc15_hit = principal_dx in set(codes.get("MDC15PRINDX", []))
    checklist.append(ChecklistItem("MDC14 obstetric (principal) excluded", "No", "Yes" if mdc14_hit else "No", not mdc14_hit))
    checklist.append(ChecklistItem("MDC15 newborn (principal) excluded", "No", "Yes" if mdc15_hit else "No", not mdc15_hit))
    if mdc14_hit: exclusions.append("MDC14PRINDX")
    if mdc15_hit: exclusions.append("MDC15PRINDX")

    # DRG 999
    drg999 = (drg3 == "999")
    checklist.append(ChecklistItem("DRG not 999 (ungroupable)", "Not 999", drg3, not drg999))
    if drg999: exclusions.append("DRG=999")

    # Missing required fields: Sex, Age, Quarter, Year, Principal DX
    def missing(col: str) -> bool:
        return (col not in row) or (pd.isna(row[col])) or (str(row[col]).strip() == "")
    miss_sex = missing("Sex")
    miss_age = missing("Age")
    miss_qtr = missing("DQTR") if "DQTR" in row else False  # only if present
    miss_year = missing("Year") if "Year" in row else False # only if present
    miss_pdx = principal_dx == ""

    checklist.append(ChecklistItem("Missing Sex", "Present", "Missing" if miss_sex else "Present", not miss_sex))
    checklist.append(ChecklistItem("Missing Age", "Present", "Missing" if miss_age else "Present", not miss_age))
    checklist.append(ChecklistItem("Missing DQTR (if provided)", "Present/NA", "Missing" if miss_qtr else "Present/NA", not miss_qtr))
    checklist.append(ChecklistItem("Missing Year (if provided)", "Present/NA", "Missing" if miss_year else "Present/NA", not miss_year))
    checklist.append(ChecklistItem("Missing Principal DX", "Present", "Missing" if miss_pdx else "Present", not miss_pdx))

    if miss_sex: exclusions.append("Missing Sex")
    if miss_age: exclusions.append("Missing Age")
    if miss_qtr: exclusions.append("Missing DQTR")
    if miss_year: exclusions.append("Missing Year")
    if miss_pdx: exclusions.append("Missing Pdx")

    # Optional: Missing MDC (only if user supplies MDC column)
    if "MDC" in row:
        miss_mdc = missing("MDC")
        checklist.append(ChecklistItem("Missing MDC (if provided)", "Present/NA", "Missing" if miss_mdc else "Present/NA", not miss_mdc))
        if miss_mdc: exclusions.append("Missing MDC")

    # If denominator not met at all, we can early exit as EXCLUSION (not in population)
    if not denominator_met:
        return PSIEvalResult(
            encounter_id=enc_id,
            psi="PSI_03",
            result="EXCLUSION",
            denominator_met=False,
            numerator_met=False,
            exclusions_applied=exclusions,
            rationale_short="Denominator not met for PSI-03",
            checklist=checklist,
            debug={
                "age": age, "drg3": drg3, "los": los,
                "principal_dx": principal_dx,
            },
        )

    # If any exclusion triggered, exclude
    if exclusions:
        return PSIEvalResult(
            encounter_id=enc_id,
            psi="PSI_03",
            result="EXCLUSION",
            denominator_met=True,
            numerator_met=False,
            exclusions_applied=exclusions,
            rationale_short="Denominator exclusion(s) applied",
            checklist=checklist,
            debug={
                "exclusions": exclusions,
                "principal_dx": principal_dx,
            },
        )

    # ---- Numerator
    # Find any SECONDARY DX in PI~D with POA != 'Y'
    # And ensure no POA=Y DTI~EXD at same site (except unspecified PI sites auto-qualify)
    numerator_hit = False
    details = []
    # Prebuild reverse maps for DTI to site keys for fast lookup of POA=Y
    dti_code_to_site = code_to_site_dti_exd  # code -> DTI site key
    # Build quick lookup of secondary DX by code and POA
    secondary_dx = [(c, p) for (c, p, role) in dx_list if role == "SECONDARY"]

    # Index DTI POA=Y by site
    dti_poaY_sites: Set[str] = set()
    for code, poa in secondary_dx:
        if code in dti_exd_union and poa == "Y":
            dti_site = dti_code_to_site.get(code, "")
            if dti_site:
                dti_poaY_sites.add(dti_site)

    # Evaluate PI~D candidates
    for code, poa in secondary_dx:
        if code in pi_d_union and poa != "Y":
            pi_site = code_to_site_pi_d.get(code, "")
            # Unspecified-site keys auto-qualify
            if any(pi_site.startswith(u) for u in list({"PIN", "PIUNSPECD"})):
                numerator_hit = True
                details.append(("PI~D unspecified site", code, poa, pi_site, None))
                break
            # Otherwise require no DTI POA=Y at same site
            same_site_block = False
            for dti_site in dti_poaY_sites:
                if same_site_match(pi_site, dti_site):
                    same_site_block = True
                    details.append(("Blocked by DTI POA=Y same site", code, poa, pi_site, dti_site))
                    break
            if not same_site_block:
                numerator_hit = True
                details.append(("PI~D no DTI POA=Y same site", code, poa, pi_site, None))
                break

    checklist.append(ChecklistItem("Numerator met (PI~D secondary not POA=Y, and no DTI POA=Y at same site)", "Yes", "Yes" if numerator_hit else "No", numerator_hit))

    result = "INCLUSION" if numerator_hit else "EXCLUSION"

    return PSIEvalResult(
        encounter_id=enc_id,
        psi="PSI_03",
        result=result,
        denominator_met=True,
        numerator_met=numerator_hit,
        exclusions_applied=[],
        rationale_short=("Triggered numerator (pressure ulcer not POA)" if numerator_hit else "No qualifying pressure ulcer found"),
        checklist=checklist,
        debug={
            "numerator_details": details,
            "dti_poaY_sites": list(dti_poaY_sites),
        },
    )

@register_psi(4)
def evaluate_psi04(row: pd.Series, codes: Dict[str, List[str]]) -> PSIEvalResult:
    """
    PSI-04 Death Rate among Surgical Inpatients with Serious Treatable Complications (2025).
    Implements denominator (incl. ATYPE/ORPROC timing), overall & stratum-specific exclusions,
    mutually-exclusive stratum assignment, and numerator (DISP=20).
    """
    enc_id = str(row.get("EncounterID", ""))
    checklist: List[ChecklistItem] = []
    exclusions: List[str] = []
    debug: Dict[str, Any] = {}

    # -------- Core fields
    age = row.get("Age", None)
    drg3 = row.get("MS-DRG_3digit", "")
    atype = str(row.get("ATYPE", "")).strip()
    disp = str(row.get("DISP", "")).strip()
    pt_origin = str(row.get("POINTOFORIGINUB04", "")).strip().upper()  # hospice 'F'
    principal_dx = str(row.get("Pdx", "")).strip().upper()
    mdc = str(row.get("MDC", "")).strip() if "MDC" in row else ""

    # Parse dx/poa and procedures (already normalized by scaffold)
    dx_list = extract_dx_poa(row)  # [(code, poa, role)]
    procs = extract_procedures(row)  # [(code, dt, col)]

    # Sets
    SURGI2R = set(str(x).zfill(3) for x in codes.get("SURGI2R", []))
    ORPROC = set(codes.get("ORPROC", []))

    # Strata sets
    FTR5DX = set(codes.get("FTR5DX", []))
    FTR5PR = set(codes.get("FTR5PR", []))
    FTR4DX = set(codes.get("FTR4DX", []))
    FTR3DX = set(codes.get("FTR3DX", []))
    FTR6DX = set(codes.get("FTR6DX", []))
    FTR2DXB = set(codes.get("FTR2DXB", []))

    # Overall exclusion sets
    MDC14PRINDX = set(codes.get("MDC14PRINDX", []))
    MDC15PRINDX = set(codes.get("MDC15PRINDX", []))

    # Stratum-specific exclusion sets
    TRAUMID = set(codes.get("TRAUMID", []))
    HEMORID = set(codes.get("HEMORID", []))
    GASTRID = set(codes.get("GASTRID", []))
    FTR5EX = set(codes.get("FTR5EX", []))
    FTR6GV = set(codes.get("FTR6GV", []))
    FTR6QD = set(codes.get("FTR6QD", []))
    INFECID = set(codes.get("INFECID", []))
    FTR3EXA = set(codes.get("FTR3EXA", []))
    FTR3EXB = set(codes.get("FTR3EXB", []))
    LUNGCIP = set(codes.get("LUNGCIP", []))
    ALCHLSM = set(codes.get("ALCHLSM", []))
    FTR6EX = set(codes.get("FTR6EX", []))
    OBEMBOL = set(codes.get("OBEMBOL", []))

    # Helpers for quick lookups
    all_dx = [c for (c, _, _) in dx_list]
    principal_in = lambda S: principal_dx in S
    any_dx_in = lambda S: any(c in S for c in all_dx)

    # -------- Denominator base: surgical discharge + age window (or obstetric principal), ORPROC present, ATYPE/ORPROC timing
    age_ok = False
    if age is not None and str(age).strip() != "":
        try:
            av = float(age)
            age_ok = (18 <= av <= 89)
        except Exception:
            age_ok = False
    obstetric_principal = principal_in(MDC14PRINDX)

    drg_ok = (drg3 in SURGI2R)
    has_orproc = any(p[0] in ORPROC for p in procs)

    # Admission timing rule: ATYPE=3 OR earliest ORPROC <= 2 days from admission
    adm_dt = row.get("Admission_Date", None)
    earliest_or_dt = None
    for code, pdt, _ in procs:
        if code in ORPROC and pdt is not None:
            if earliest_or_dt is None or pdt < earliest_or_dt:
                earliest_or_dt = pdt

    timing_ok = False
    if atype == "3":
        timing_ok = True
    elif adm_dt is not None and earliest_or_dt is not None:
        try:
            delta_days = (earliest_or_dt.normalize() - pd.to_datetime(adm_dt).normalize()).days
            timing_ok = (delta_days <= 2 and delta_days >= 0)
        except Exception:
            timing_ok = False

    checklist.append(ChecklistItem("DRG in SURGI2R", "Yes", drg3, drg_ok))
    checklist.append(ChecklistItem("Age 18–89 OR obstetric principal (MDC14PRINDX)", "Yes", f"Age={age}, MDC14_Principal={obstetric_principal}", age_ok or obstetric_principal))
    checklist.append(ChecklistItem("Any ORPROC present", "Yes", "Yes" if has_orproc else "No", has_orproc))
    checklist.append(ChecklistItem("ATYPE=3 OR earliest ORPROC ≤ 2 days from admission", "Yes", f"ATYPE={atype}, earliest_OR={earliest_or_dt}, admit={adm_dt}", timing_ok))

    denominator_base = drg_ok and (age_ok or obstetric_principal) and has_orproc and timing_ok

    # -------- Overall denominator exclusions (apply precedence)
    # Transfer to acute care (DISP=2)
    excl_transfer = (disp == "2")
    checklist.append(ChecklistItem("Not transferred to acute care (DISP=2)", "No", disp, not excl_transfer))
    if excl_transfer: exclusions.append("Transfer to acute (DISP=2)")

    # Admitted from hospice (POINTOFORIGINUB04 = F)
    excl_hospice = (pt_origin == "F")
    checklist.append(ChecklistItem("Not admitted from hospice (POINTOFORIGINUB04=F)", "No", pt_origin, not excl_hospice))
    if excl_hospice: exclusions.append("Admit from Hospice (F)")

    # Newborn MDC15
    excl_mdc15 = principal_in(MDC15PRINDX)
    checklist.append(ChecklistItem("Not MDC15 principal (newborn)", "No", "Yes" if excl_mdc15 else "No", not excl_mdc15))
    if excl_mdc15: exclusions.append("MDC15PRINDX")

    # DRG 999
    excl_drg999 = (drg3 == "999")
    checklist.append(ChecklistItem("DRG not 999", "Not 999", drg3, not excl_drg999))
    if excl_drg999: exclusions.append("DRG=999")

    # Missing required fields
    def missing(col: str) -> bool:
        return (col not in row) or (pd.isna(row[col])) or (str(row[col]).strip() == "")
    miss_disp = missing("DISP")
    miss_sex = missing("Sex")
    miss_age = missing("Age")
    miss_qtr = missing("DQTR") if "DQTR" in row else False
    miss_year = missing("Year") if "Year" in row else False
    miss_pdx = missing("Pdx")

    checklist.append(ChecklistItem("Missing DISP", "Present", "Missing" if miss_disp else "Present", not miss_disp))
    checklist.append(ChecklistItem("Missing Sex", "Present", "Missing" if miss_sex else "Present", not miss_sex))
    checklist.append(ChecklistItem("Missing Age", "Present", "Missing" if miss_age else "Present", not miss_age))
    checklist.append(ChecklistItem("Missing DQTR (if provided)", "Present/NA", "Missing" if miss_qtr else "Present/NA", not miss_qtr))
    checklist.append(ChecklistItem("Missing Year (if provided)", "Present/NA", "Missing" if miss_year else "Present/NA", not miss_year))
    checklist.append(ChecklistItem("Missing Principal DX", "Present", "Missing" if miss_pdx else "Present", not miss_pdx))

    if miss_disp: exclusions.append("Missing DISP")
    if miss_sex: exclusions.append("Missing Sex")
    if miss_age: exclusions.append("Missing Age")
    if miss_qtr: exclusions.append("Missing DQTR")
    if miss_year: exclusions.append("Missing Year")
    if miss_pdx: exclusions.append("Missing Pdx")

    # Missing MDC if provided
    if "MDC" in row:
        miss_mdc = missing("MDC")
        checklist.append(ChecklistItem("Missing MDC (if provided)", "Present/NA", "Missing" if miss_mdc else "Present/NA", not miss_mdc))
        if miss_mdc: exclusions.append("Missing MDC")

    # Early exits
    if not denominator_base:
        return PSIEvalResult(
            encounter_id=enc_id, psi="PSI_04",
            result="EXCLUSION", denominator_met=False, numerator_met=False,
            exclusions_applied=exclusions, rationale_short="Denominator not met",
            checklist=checklist, debug={"drg3": drg3, "age": age, "atype": atype, "earliest_or": str(earliest_or_dt)}
        )
    if exclusions:
        return PSIEvalResult(
            encounter_id=enc_id, psi="PSI_04",
            result="EXCLUSION", denominator_met=True, numerator_met=False,
            exclusions_applied=exclusions, rationale_short="Overall exclusion(s) applied",
            checklist=checklist, debug={"exclusions": exclusions}
        )

    # -------- Determine candidate strata
    # Build helpers
    secondary_dx = [c for (c, _, role) in dx_list if role == "SECONDARY"]
    # Earliest ORPROC date used for shock timing
    first_or_date = earliest_or_dt.normalize() if earliest_or_dt is not None else None

    # Inclusion flags
    cand_shock = False
    cand_sepsis = False
    cand_pneum = False
    cand_gi = False
    cand_dvtpe = False

    # SHOCK/CA: secondary FTR5DX OR FTR5PR same day/after first ORPROC
    # Note: We generally don't have DX timestamps; we accept secondary FTR5DX as satisfying timing.
    if any(c in FTR5DX for c in secondary_dx):
        cand_shock = True
    else:
        # look for FTR5PR at or after first OR date
        if first_or_date is not None:
            for (code, pdt, col) in procs:
                if code in FTR5PR and pdt is not None and pdt.normalize() >= first_or_date:
                    cand_shock = True
                    break

    # SEPSIS: any secondary FTR4DX
    if any(c in FTR4DX for c in secondary_dx):
        cand_sepsis = True

    # PNEUMONIA: any secondary FTR3DX
    if any(c in FTR3DX for c in secondary_dx):
        cand_pneum = True

    # GI HEMORRHAGE: any secondary FTR6DX
    if any(c in FTR6DX for c in secondary_dx):
        cand_gi = True

    # DVT/PE: any secondary FTR2DXB
    if any(c in FTR2DXB for c in secondary_dx):
        cand_dvtpe = True

    debug["candidate_strata"] = {
        "SHOCK": cand_shock, "SEPSIS": cand_sepsis, "PNEUMONIA": cand_pneum,
        "GI_HEMORRHAGE": cand_gi, "DVT_PE": cand_dvtpe
    }

    # Assign stratum per priority
    assigned = None
    if cand_shock: assigned = "STRATUM_SHOCK"
    elif cand_sepsis: assigned = "STRATUM_SEPSIS"
    elif cand_pneum: assigned = "STRATUM_PNEUMONIA"
    elif cand_gi: assigned = "STRATUM_GI_HEMORRHAGE"
    elif cand_dvtpe: assigned = "STRATUM_DVT_PE"

    checklist.append(ChecklistItem("Assigned a complication stratum", "One of SHOCK/SEPSIS/PNEUM/GI/DVT-PE", assigned or "None", assigned is not None))

    if assigned is None:
        # No stratum → not in denominator
        return PSIEvalResult(
            encounter_id=enc_id, psi="PSI_04",
            result="EXCLUSION", denominator_met=False, numerator_met=False,
            exclusions_applied=[], rationale_short="No qualifying complication stratum",
            checklist=checklist, debug=debug
        )

    # -------- Stratum-specific exclusions
    def add_excl(flag: bool, label: str, expect="No", found="Yes"):
        checklist.append(ChecklistItem(f"Stratum exclusion: {label}", expect, found if flag else "No", not flag))
        if flag: exclusions.append(label)

    if assigned == "STRATUM_SHOCK":
        add_excl(principal_in(FTR5DX), "Principal shock/cardiac arrest (FTR5DX)")
        add_excl(principal_in(TRAUMID), "Principal trauma (TRAUMID)")
        add_excl(principal_in(HEMORID), "Principal hemorrhage (HEMORID)")
        add_excl(principal_in(GASTRID), "Principal GI hemorrhage (GASTRID)")
        add_excl(principal_in(FTR5EX), "Principal abortion-related shock (FTR5EX)")
        # Secondary esophageal varices w/ bleed + qualifying principal
        add_excl(any_dx_in(FTR6GV) and principal_in(FTR6QD), "Esophageal varices w/ bleed w/ qualifying principal (FTR6GV + FTR6QD)")
        # MDC 4 or 5 if MDC provided
        if mdc:
            add_excl(mdc == "4", "MDC 4 (Respiratory System)")
            add_excl(mdc == "5", "MDC 5 (Circulatory System)")

    elif assigned == "STRATUM_SEPSIS":
        add_excl(principal_in(FTR4DX), "Principal sepsis (FTR4DX)")
        add_excl(principal_in(INFECID), "Principal infection (INFECID)")

    elif assigned == "STRATUM_PNEUMONIA":
        add_excl(principal_in(FTR3DX), "Principal pneumonia (FTR3DX)")
        add_excl(principal_in(FTR3EXA), "Principal respiratory complications (FTR3EXA)")
        add_excl(any_dx_in(FTR3EXB), "Any viral pneumonia/influenza (FTR3EXB)", expect="Absent", found="Present")
        # Any LUNGCIP procedure
        has_lung_proc = any(code in LUNGCIP for (code, _, _) in procs)
        add_excl(has_lung_proc, "Lung cancer procedure (LUNGCIP)")
        if mdc:
            add_excl(mdc == "4", "MDC 4 (Respiratory System)")

    elif assigned == "STRATUM_GI_HEMORRHAGE":
        add_excl(principal_in(FTR6DX), "Principal GI hemorrhage/acute ulcer (FTR6DX)")
        add_excl(any_dx_in(FTR6GV) and principal_in(FTR6QD), "Esophageal varices w/ bleed with qualifying principal (FTR6GV + FTR6QD)")
        add_excl(principal_in(TRAUMID), "Principal trauma (TRAUMID)")
        add_excl(principal_in(ALCHLSM), "Principal alcoholism (ALCHLSM)")
        add_excl(principal_in(FTR6EX), "Principal anemia (FTR6EX)")
        if mdc:
            add_excl(mdc == "6", "MDC 6 (Digestive System)")
            add_excl(mdc == "7", "MDC 7 (Hepatobiliary/Pancreas)")

    elif assigned == "STRATUM_DVT_PE":
        add_excl(principal_in(FTR2DXB), "Principal DVT/PE (FTR2DXB)")
        add_excl(principal_in(OBEMBOL), "Principal obstetric PE (OBEMBOL)")

    # If any stratum exclusion hit → overall exclusion
    if exclusions:
        return PSIEvalResult(
            encounter_id=enc_id, psi="PSI_04",
            result="EXCLUSION", denominator_met=True, numerator_met=False,
            exclusions_applied=exclusions, rationale_short=f"{assigned} exclusion(s) applied",
            checklist=checklist, debug={"assigned_stratum": assigned, "exclusions": exclusions}
        )

    # -------- Numerator: in-hospital death (DISP=20)
    numerator_met = (disp == "20")
    checklist.append(ChecklistItem("Numerator: death in-hospital (DISP=20)", "Yes", disp, numerator_met))

    result = "INCLUSION" if numerator_met else "EXCLUSION"
    rationale = "Death among surgical inpatient with treatable complication" if numerator_met else "No in-hospital death"

    return PSIEvalResult(
        encounter_id=enc_id, psi="PSI_04",
        result=result, denominator_met=True, numerator_met=numerator_met,
        exclusions_applied=[], rationale_short=rationale,
        checklist=checklist, debug={"assigned_stratum": assigned}
    )

@register_psi(5)
def evaluate_psi05(row: pd.Series, codes: Dict[str, List[str]]) -> PSIEvalResult:
    """
    PSI-05 Retained Surgical Item or Unretrieved Device Fragment (2025).
    Count indicator; here we classify encounters as INCLUSION when numerator criteria met and no exclusions.
    Spec source: 2025 PSI-05 markdown uploaded by user.
    """
    enc_id = str(row.get("EncounterID", ""))
    checklist: List[ChecklistItem] = []
    exclusions: List[str] = []
    debug: Dict[str, Any] = {}

    # Core fields
    age = row.get("Age", None)
    drg3 = row.get("MS-DRG_3digit", "")
    principal_dx = str(row.get("Pdx", "")).strip().upper()

    # Parse dx/poa
    dx_list = extract_dx_poa(row)  # [(code, poa, role)]
    secondary_dx = [(c,p) for (c,p,role) in dx_list if role == "SECONDARY"]

    # Sets
    SURGI2R = set(str(x).zfill(3) for x in codes.get("SURGI2R", []))
    MEDIC2R = set(str(x).zfill(3) for x in codes.get("MEDIC2R", []))
    MDC14PRINDX = set(codes.get("MDC14PRINDX", []))
    MDC15PRINDX = set(codes.get("MDC15PRINDX", []))
    FOREIID = set(codes.get("FOREIID", []))

    # Denominator / cohort: Surgical or medical discharge AND (Age>=18 OR obstetric principal)
    age_ok = False
    if age is not None and str(age).strip() != "":
        try:
            age_ok = float(age) >= 18
        except Exception:
            age_ok = False
    obstetric_principal = (principal_dx in MDC14PRINDX)
    drg_ok = (drg3 in SURGI2R) or (drg3 in MEDIC2R)

    checklist.append(ChecklistItem("DRG in SURGI2R or MEDIC2R", "Yes", drg3, drg_ok))
    checklist.append(ChecklistItem("Age >= 18 OR obstetric principal (MDC14PRINDX)", "Yes", f"Age={age}, MDC14_Principal={obstetric_principal}", age_ok or obstetric_principal))

    denominator_met = drg_ok and (age_ok or obstetric_principal)

    # Exclusions (precedence)
    # Principal in FOREIID
    prin_is_foreiid = (principal_dx in FOREIID)
    checklist.append(ChecklistItem("Principal DX is retained item/device fragment (FOREIID)", "No", principal_dx, not prin_is_foreiid))
    if prin_is_foreiid: exclusions.append("Principal in FOREIID")

    # Secondary FOREIID with POA=Y
    sec_foreiid_poaY = any((c in FOREIID) and (p == "Y") for (c,p) in secondary_dx)
    checklist.append(ChecklistItem("Secondary FOREIID present-on-admission", "No", "Yes" if sec_foreiid_poaY else "No", not sec_foreiid_poaY))
    if sec_foreiid_poaY: exclusions.append("Secondary FOREIID POA=Y")

    # Newborn principal MDC15
    mdc15 = (principal_dx in MDC15PRINDX)
    checklist.append(ChecklistItem("MDC15 principal (newborn) excluded", "No", "Yes" if mdc15 else "No", not mdc15))
    if mdc15: exclusions.append("MDC15PRINDX")

    # DRG 999
    drg999 = (drg3 == "999")
    checklist.append(ChecklistItem("DRG not 999 (ungroupable)", "Not 999", drg3, not drg999))
    if drg999: exclusions.append("DRG=999")

    # Missing fields
    def missing(col: str) -> bool:
        return (col not in row) or (pd.isna(row[col])) or (str(row[col]).strip() == "")
    miss_sex = missing("Sex")
    miss_age = missing("Age")
    miss_qtr = missing("DQTR") if "DQTR" in row else False
    miss_year = missing("Year") if "Year" in row else False
    miss_pdx = missing("Pdx")

    checklist.append(ChecklistItem("Missing Sex", "Present", "Missing" if miss_sex else "Present", not miss_sex))
    checklist.append(ChecklistItem("Missing Age", "Present", "Missing" if miss_age else "Present", not miss_age))
    checklist.append(ChecklistItem("Missing DQTR (if provided)", "Present/NA", "Missing" if miss_qtr else "Present/NA", not miss_qtr))
    checklist.append(ChecklistItem("Missing Year (if provided)", "Present/NA", "Missing" if miss_year else "Present/NA", not miss_year))
    checklist.append(ChecklistItem("Missing Principal DX", "Present", "Missing" if miss_pdx else "Present", not miss_pdx))

    if miss_sex: exclusions.append("Missing Sex")
    if miss_age: exclusions.append("Missing Age")
    if miss_qtr: exclusions.append("Missing DQTR")
    if miss_year: exclusions.append("Missing Year")
    if miss_pdx: exclusions.append("Missing Pdx")

    # Missing MDC if provided
    if "MDC" in row:
        miss_mdc = missing("MDC")
        checklist.append(ChecklistItem("Missing MDC (if provided)", "Present/NA", "Missing" if miss_mdc else "Present/NA", not miss_mdc))
        if miss_mdc: exclusions.append("Missing MDC")

    if not denominator_met:
        return PSIEvalResult(
            encounter_id=enc_id, psi="PSI_05",
            result="EXCLUSION", denominator_met=False, numerator_met=False,
            exclusions_applied=exclusions, rationale_short="Denominator not met",
            checklist=checklist, debug={"drg3": drg3, "age": age}
        )

    if exclusions:
        return PSIEvalResult(
            encounter_id=enc_id, psi="PSI_05",
            result="EXCLUSION", denominator_met=True, numerator_met=False,
            exclusions_applied=exclusions, rationale_short="Exclusion(s) applied",
            checklist=checklist, debug={"exclusions": exclusions}
        )

    # Numerator: any secondary FOREIID with POA != Y
    numerator_hit = any((c in FOREIID) and (p != "Y") for (c,p) in secondary_dx)
    checklist.append(ChecklistItem("Numerator: secondary FOREIID not POA=Y", "Yes", "Yes" if numerator_hit else "No", numerator_hit))

    result = "INCLUSION" if numerator_hit else "EXCLUSION"
    rationale = "Retained item/device fragment coded (secondary, not POA)" if numerator_hit else "No qualifying retained item/device fragment"

    return PSIEvalResult(
        encounter_id=enc_id, psi="PSI_05",
        result=result, denominator_met=True, numerator_met=numerator_hit,
        exclusions_applied=[], rationale_short=rationale,
        checklist=checklist, debug={"FOREIID_detected": numerator_hit}
    )

@register_psi(6)
def evaluate_psi06(row: pd.Series, codes: Dict[str, List[str]]) -> PSIEvalResult:
    """
    PSI-06 Iatrogenic Pneumothorax Rate (2025).
    Denominator: SURGI2R or MEDIC2R AND Age >=18.
    Exclusions: IATPTXD principal or POA=Y secondary; CTRAUMD; PLEURAD; THORAIP; CARDSIP; MDC14PRINDX; MDC15PRINDX; DRG=999; missing key fields; missing MDC (if provided).
    Numerator: secondary IATROID.
    Spec source: user-uploaded 2025 PSI-06 markdown.
    """
    enc_id = str(row.get("EncounterID", ""))
    checklist: List[ChecklistItem] = []
    exclusions: List[str] = []
    debug: Dict[str, Any] = {}

    # Core fields
    age = row.get("Age", None)
    drg3 = row.get("MS-DRG_3digit", "")
    principal_dx = str(row.get("Pdx", "")).strip().upper()

    # Parse DX/POA and procedures
    dx_list = extract_dx_poa(row)  # [(code, poa, role)]
    secondary_dx = [(c,p) for (c,p,role) in dx_list if role == "SECONDARY"]
    all_dx = [c for (c,_,_) in dx_list]
    procs = extract_procedures(row)

    # Sets
    SURGI2R = set(str(x).zfill(3) for x in codes.get("SURGI2R", []))
    MEDIC2R = set(str(x).zfill(3) for x in codes.get("MEDIC2R", []))
    IATROID = set(codes.get("IATROID", []))
    IATPTXD = set(codes.get("IATPTXD", []))
    CTRAUMD = set(codes.get("CTRAUMD", []))
    PLEURAD = set(codes.get("PLEURAD", []))
    THORAIP = set(codes.get("THORAIP", []))
    CARDSIP = set(codes.get("CARDSIP", []))
    MDC14PRINDX = set(codes.get("MDC14PRINDX", []))
    MDC15PRINDX = set(codes.get("MDC15PRINDX", []))

    # Denominator
    age_ok = False
    if age is not None and str(age).strip() != "":
        try:
            age_ok = float(age) >= 18
        except Exception:
            age_ok = False
    drg_ok = (drg3 in SURGI2R) or (drg3 in MEDIC2R)

    checklist.append(ChecklistItem("DRG in SURGI2R or MEDIC2R", "Yes", drg3, drg_ok))
    checklist.append(ChecklistItem("Age >= 18", ">=18", age, age_ok))

    denominator_met = drg_ok and age_ok

    # Exclusions (precedence)
    principal_in = lambda S: principal_dx in S
    any_dx_in = lambda S: any(c in S for c in all_dx)

    # Principal or secondary POA=Y non-traumatic pneumothorax (IATPTXD)
    excl_iatpt_principal = principal_in(IATPTXD)
    excl_iatpt_sec_poaY = any((c in IATPTXD) and (p == "Y") for (c,p) in secondary_dx)
    checklist.append(ChecklistItem("Principal non-traumatic pneumothorax (IATPTXD)", "No", "Yes" if excl_iatpt_principal else "No", not excl_iatpt_principal))
    checklist.append(ChecklistItem("Secondary IATPTXD present-on-admission", "No", "Yes" if excl_iatpt_sec_poaY else "No", not excl_iatpt_sec_poaY))
    if excl_iatpt_principal: exclusions.append("Principal IATPTXD")
    if excl_iatpt_sec_poaY: exclusions.append("Secondary IATPTXD POA=Y")

    # Chest trauma
    has_ctrauma = any_dx_in(CTRAUMD)
    checklist.append(ChecklistItem("Chest trauma codes (CTRAUMD)", "No", "Yes" if has_ctrauma else "No", not has_ctrauma))
    if has_ctrauma: exclusions.append("CTRAUMD")

    # Pleural effusion
    has_pleura = any_dx_in(PLEURAD)
    checklist.append(ChecklistItem("Pleural effusion (PLEURAD)", "No", "Yes" if has_pleura else "No", not has_pleura))
    if has_pleura: exclusions.append("PLEURAD")

    # Thoracic surgery / lung or pleural biopsy / diaphragmatic repair
    has_thorax_proc = any(code in THORAIP for (code,_,_) in procs)
    checklist.append(ChecklistItem("Thoracic surgery (THORAIP)", "No", "Yes" if has_thorax_proc else "No", not has_thorax_proc))
    if has_thorax_proc: exclusions.append("THORAIP")

    # Trans-pleural cardiac procedure
    has_cardsp = any(code in CARDSIP for (code,_,_) in procs)
    checklist.append(ChecklistItem("Trans-pleural cardiac procedure (CARDSIP)", "No", "Yes" if has_cardsp else "No", not has_cardsp))
    if has_cardsp: exclusions.append("CARDSIP")

    # Obstetric / Newborn
    obstetric_principal = principal_in(MDC14PRINDX)
    newborn_principal = principal_in(MDC15PRINDX)
    checklist.append(ChecklistItem("Obstetric principal (MDC14PRINDX)", "No", "Yes" if obstetric_principal else "No", not obstetric_principal))
    checklist.append(ChecklistItem("Newborn principal (MDC15PRINDX)", "No", "Yes" if newborn_principal else "No", not newborn_principal))
    if obstetric_principal: exclusions.append("MDC14PRINDX")
    if newborn_principal: exclusions.append("MDC15PRINDX")

    # DRG 999
    drg999 = (drg3 == "999")
    checklist.append(ChecklistItem("DRG not 999 (ungroupable)", "Not 999", drg3, not drg999))
    if drg999: exclusions.append("DRG=999")

    # Missing fields
    def missing(col: str) -> bool:
        return (col not in row) or (pd.isna(row[col])) or (str(row[col]).strip() == "")
    miss_sex = missing("Sex")
    miss_age = missing("Age")
    miss_qtr = missing("DQTR") if "DQTR" in row else False
    miss_year = missing("Year") if "Year" in row else False
    miss_pdx = missing("Pdx")

    checklist.append(ChecklistItem("Missing Sex", "Present", "Missing" if miss_sex else "Present", not miss_sex))
    checklist.append(ChecklistItem("Missing Age", "Present", "Missing" if miss_age else "Present", not miss_age))
    checklist.append(ChecklistItem("Missing DQTR (if provided)", "Present/NA", "Missing" if miss_qtr else "Present/NA", not miss_qtr))
    checklist.append(ChecklistItem("Missing Year (if provided)", "Present/NA", "Missing" if miss_year else "Present/NA", not miss_year))
    checklist.append(ChecklistItem("Missing Principal DX", "Present", "Missing" if miss_pdx else "Present", not miss_pdx))

    if miss_sex: exclusions.append("Missing Sex")
    if miss_age: exclusions.append("Missing Age")
    if miss_qtr: exclusions.append("Missing DQTR")
    if miss_year: exclusions.append("Missing Year")
    if miss_pdx: exclusions.append("Missing Pdx")

    # Missing MDC if provided
    if "MDC" in row:
        miss_mdc = missing("MDC")
        checklist.append(ChecklistItem("Missing MDC (if provided)", "Present/NA", "Missing" if miss_mdc else "Present/NA", not miss_mdc))
        if miss_mdc: exclusions.append("Missing MDC")

    # Early exits
    if not denominator_met:
        return PSIEvalResult(
            encounter_id=enc_id, psi="PSI_06",
            result="EXCLUSION", denominator_met=False, numerator_met=False,
            exclusions_applied=exclusions, rationale_short="Denominator not met",
            checklist=checklist, debug={"drg3": drg3, "age": age}
        )
    if exclusions:
        return PSIEvalResult(
            encounter_id=enc_id, psi="PSI_06",
            result="EXCLUSION", denominator_met=True, numerator_met=False,
            exclusions_applied=exclusions, rationale_short="Denominator exclusion(s) applied",
            checklist=checklist, debug={"exclusions": exclusions}
        )

    # Numerator: secondary IATROID
    numerator_hit = any((c in IATROID) for (c,_) in secondary_dx)
    checklist.append(ChecklistItem("Numerator: secondary IATROID present", "Yes", "Yes" if numerator_hit else "No", numerator_hit))

    result = "INCLUSION" if numerator_hit else "EXCLUSION"
    rationale = "Iatrogenic pneumothorax coded (secondary)" if numerator_hit else "No qualifying iatrogenic pneumothorax"

    return PSIEvalResult(
        encounter_id=enc_id, psi="PSI_06",
        result=result, denominator_met=True, numerator_met=numerator_hit,
        exclusions_applied=[], rationale_short=rationale,
        checklist=checklist, debug={}
    )

@register_psi(7)
def evaluate_psi07(row: pd.Series, codes: Dict[str, List[str]]) -> PSIEvalResult:
    """
    PSI-07 Central Venous Catheter-Related Bloodstream Infection Rate (2025).
    Denominator: SURGI2R or MEDIC2R AND (Age>=18 OR obstetric principal MDC14PRINDX).
    Exclusions: principal/secondary POA=Y IDTMC3D, LOS<2, cancer, immunocompromised dx/proc, MDC15PRINDX, DRG=999, missing fields.
    Numerator: secondary IDTMC3D not POA=Y.
    """
    enc_id = str(row.get("EncounterID", ""))
    checklist: List[ChecklistItem] = []
    exclusions: List[str] = []
    debug: Dict[str, Any] = {}

    # Core fields
    age = row.get("Age", None)
    drg3 = row.get("MS-DRG_3digit", "")
    principal_dx = str(row.get("Pdx", "")).strip().upper()
    los = row.get("Length_of_stay", None)

    dx_list = extract_dx_poa(row)
    secondary_dx = [(c,p) for (c,p,role) in dx_list if role=="SECONDARY"]
    all_dx = [c for (c,_,_) in dx_list]
    procs = extract_procedures(row)

    # Sets
    SURGI2R = set(str(x).zfill(3) for x in codes.get("SURGI2R", []))
    MEDIC2R = set(str(x).zfill(3) for x in codes.get("MEDIC2R", []))
    MDC14PRINDX = set(codes.get("MDC14PRINDX", []))
    MDC15PRINDX = set(codes.get("MDC15PRINDX", []))
    IDTMC3D = set(codes.get("IDTMC3D", []))
    CANCEID = set(codes.get("CANCEID", []))
    IMMUNID = set(codes.get("IMMUNID", []))
    IMMUNIP = set(codes.get("IMMUNIP", []))

    # Denominator
    age_ok = False
    if age is not None and str(age).strip()!="":
        try:
            age_ok = float(age) >= 18
        except Exception:
            age_ok = False
    obstetric_principal = (principal_dx in MDC14PRINDX)
    drg_ok = (drg3 in SURGI2R) or (drg3 in MEDIC2R)

    checklist.append(ChecklistItem("DRG in SURGI2R or MEDIC2R", "Yes", drg3, drg_ok))
    checklist.append(ChecklistItem("Age>=18 or obstetric principal", "Yes", f"Age={age},Obstetric={obstetric_principal}", age_ok or obstetric_principal))

    denominator_met = drg_ok and (age_ok or obstetric_principal)

    # Exclusions
    principal_in = lambda S: principal_dx in S
    any_dx_in = lambda S: any(c in S for c in all_dx)

    # Principal or secondary POA=Y IDTMC3D
    excl_idtmc_prin = principal_in(IDTMC3D)
    excl_idtmc_sec_poaY = any((c in IDTMC3D) and (p=="Y") for (c,p) in secondary_dx)
    checklist.append(ChecklistItem("Principal IDTMC3D", "No", "Yes" if excl_idtmc_prin else "No", not excl_idtmc_prin))
    checklist.append(ChecklistItem("Secondary IDTMC3D POA=Y", "No", "Yes" if excl_idtmc_sec_poaY else "No", not excl_idtmc_sec_poaY))
    if excl_idtmc_prin: exclusions.append("Principal IDTMC3D")
    if excl_idtmc_sec_poaY: exclusions.append("Secondary IDTMC3D POA=Y")

    # LOS <2
    los_excl = False
    if los is not None:
        try:
            los_excl = int(los) < 2
        except Exception:
            los_excl = False
    checklist.append(ChecklistItem("Length of stay >=2 days", ">=2", los, not los_excl))
    if los_excl: exclusions.append("LOS<2")

    # Cancer
    has_cancer = any_dx_in(CANCEID)
    checklist.append(ChecklistItem("Cancer DX present (CANCEID)", "No", "Yes" if has_cancer else "No", not has_cancer))
    if has_cancer: exclusions.append("CANCEID")

    # Immunocompromised DX/PROC
    has_immun_dx = any_dx_in(IMMUNID)
    has_immun_proc = any(code in IMMUNIP for (code,_,_) in procs)
    checklist.append(ChecklistItem("Immunocompromised DX present (IMMUNID)", "No", "Yes" if has_immun_dx else "No", not has_immun_dx))
    checklist.append(ChecklistItem("Immunocompromised procedure (IMMUNIP)", "No", "Yes" if has_immun_proc else "No", not has_immun_proc))
    if has_immun_dx: exclusions.append("IMMUNID")
    if has_immun_proc: exclusions.append("IMMUNIP")

    # Newborn
    newborn_principal = principal_in(MDC15PRINDX)
    checklist.append(ChecklistItem("Newborn principal (MDC15PRINDX)", "No", "Yes" if newborn_principal else "No", not newborn_principal))
    if newborn_principal: exclusions.append("MDC15PRINDX")

    # DRG 999
    drg999 = (drg3=="999")
    checklist.append(ChecklistItem("DRG not 999", "Not 999", drg3, not drg999))
    if drg999: exclusions.append("DRG=999")

    # Missing fields
    def missing(col: str)->bool:
        return (col not in row) or (pd.isna(row[col])) or (str(row[col]).strip()=="")
    miss_sex = missing("Sex")
    miss_age = missing("Age")
    miss_qtr = missing("DQTR") if "DQTR" in row else False
    miss_year = missing("Year") if "Year" in row else False
    miss_pdx = missing("Pdx")

    checklist.append(ChecklistItem("Missing Sex","Present","Missing" if miss_sex else "Present", not miss_sex))
    checklist.append(ChecklistItem("Missing Age","Present","Missing" if miss_age else "Present", not miss_age))
    checklist.append(ChecklistItem("Missing DQTR","Present","Missing" if miss_qtr else "Present", not miss_qtr))
    checklist.append(ChecklistItem("Missing Year","Present","Missing" if miss_year else "Present", not miss_year))
    checklist.append(ChecklistItem("Missing Principal DX","Present","Missing" if miss_pdx else "Present", not miss_pdx))

    if miss_sex: exclusions.append("Missing Sex")
    if miss_age: exclusions.append("Missing Age")
    if miss_qtr: exclusions.append("Missing DQTR")
    if miss_year: exclusions.append("Missing Year")
    if miss_pdx: exclusions.append("Missing Pdx")

    # Missing MDC if provided
    if "MDC" in row:
        miss_mdc = missing("MDC")
        checklist.append(ChecklistItem("Missing MDC (if provided)", "Present/NA","Missing" if miss_mdc else "Present/NA", not miss_mdc))
        if miss_mdc: exclusions.append("Missing MDC")

    # Early exits
    if not denominator_met:
        return PSIEvalResult(enc_id,"PSI_07","EXCLUSION",False,False,exclusions,"Denominator not met",checklist,debug)
    if exclusions:
        return PSIEvalResult(enc_id,"PSI_07","EXCLUSION",True,False,exclusions,"Denominator exclusion(s) applied",checklist,debug)

    # Numerator: secondary IDTMC3D not POA=Y
    numerator_hit = any((c in IDTMC3D) and (p!="Y") for (c,p) in secondary_dx)
    checklist.append(ChecklistItem("Numerator: secondary IDTMC3D not POA=Y","Yes","Yes" if numerator_hit else "No", numerator_hit))

    result = "INCLUSION" if numerator_hit else "EXCLUSION"
    rationale = "Central line infection coded (secondary, not POA)" if numerator_hit else "No qualifying central line infection"

    return PSIEvalResult(enc_id,"PSI_07",result,True,numerator_hit,[],rationale,checklist,debug)

@register_psi(7)
def evaluate_psi07(row: pd.Series, codes: Dict[str, List[str]]) -> PSIEvalResult:
    """
    PSI-07 Central Venous Catheter-Related Bloodstream Infection Rate (2025).
    Denominator: SURGI2R or MEDIC2R AND (Age >=18 OR obstetric principal via MDC14PRINDX).
    Exclusions: principal/secondary POA=Y IDTMC3D; LOS<2; CANCEID; IMMUNID; IMMUNIP; MDC15PRINDX; DRG=999; missing key fields.
    Numerator: secondary IDTMC3D (not POA=Y).
    Spec source: user-uploaded 2025 PSI-07 markdown.
    """
    enc_id = str(row.get("EncounterID", ""))
    checklist: List[ChecklistItem] = []
    exclusions: List[str] = []
    debug: Dict[str, Any] = {}

    # Core fields
    age = row.get("Age", None)
    drg3 = row.get("MS-DRG_3digit", "")
    principal_dx = str(row.get("Pdx", "")).strip().upper()
    los = row.get("Length_of_stay", None)

    # Parse DX/POA and procedures
    dx_list = extract_dx_poa(row)  # [(code, poa, role)]
    secondary_dx = [(c,p) for (c,p,role) in dx_list if role == "SECONDARY"]
    all_dx = [c for (c,_,_) in dx_list]
    procs = extract_procedures(row)

    # Sets
    SURGI2R = set(str(x).zfill(3) for x in codes.get("SURGI2R", []))
    MEDIC2R = set(str(x).zfill(3) for x in codes.get("MEDIC2R", []))
    MDC14PRINDX = set(codes.get("MDC14PRINDX", []))
    MDC15PRINDX = set(codes.get("MDC15PRINDX", []))
    IDTMC3D = set(codes.get("IDTMC3D", []))
    CANCEID = set(codes.get("CANCEID", []))
    IMMUNID = set(codes.get("IMMUNID", []))
    IMMUNIP = set(codes.get("IMMUNIP", []))

    # Denominator
    age_ok = False
    if age is not None and str(age).strip() != "":
        try:
            age_ok = float(age) >= 18
        except Exception:
            age_ok = False
    obstetric_principal = (principal_dx in MDC14PRINDX)
    drg_ok = (drg3 in SURGI2R) or (drg3 in MEDIC2R)

    checklist.append(ChecklistItem("DRG in SURGI2R or MEDIC2R", "Yes", drg3, drg_ok))
    checklist.append(ChecklistItem("Age >=18 OR obstetric principal (MDC14PRINDX)", "Yes", f"Age={age}, MDC14_Principal={obstetric_principal}", age_ok or obstetric_principal))

    denominator_met = drg_ok and (age_ok or obstetric_principal)

    # Exclusions (precedence)
    principal_in = lambda S: principal_dx in S
    any_dx_in = lambda S: any(c in S for c in all_dx)

    # Principal or secondary POA=Y IDTMC3D
    excl_prin_idtmc3d = principal_in(IDTMC3D)
    excl_sec_poaY_idtmc3d = any((c in IDTMC3D) and (p == "Y") for (c,p) in secondary_dx)
    checklist.append(ChecklistItem("Principal CVC-related BSI (IDTMC3D)", "No", "Yes" if excl_prin_idtmc3d else "No", not excl_prin_idtmc3d))
    checklist.append(ChecklistItem("Secondary IDTMC3D present-on-admission", "No", "Yes" if excl_sec_poaY_idtmc3d else "No", not excl_sec_poaY_idtmc3d))
    if excl_prin_idtmc3d: exclusions.append("Principal IDTMC3D")
    if excl_sec_poaY_idtmc3d: exclusions.append("Secondary IDTMC3D POA=Y")

    # LOS < 2
    los_excl = False
    if los is not None and str(los).strip() != "":
        try:
            los_excl = int(los) < 2
        except Exception:
            los_excl = False
    checklist.append(ChecklistItem("Length of stay >= 2 days", ">=2", los, not los_excl))
    if los_excl: exclusions.append("LOS<2")

    # Cancer / Immunocompromised DX or PROC
    has_cancer = any_dx_in(CANCEID)
    has_immun_dx = any_dx_in(IMMUNID)
    has_immun_proc = any(code in IMMUNIP for (code,_,_) in procs)
    checklist.append(ChecklistItem("Cancer diagnosis (CANCEID)", "No", "Yes" if has_cancer else "No", not has_cancer))
    checklist.append(ChecklistItem("Immunocompromised diagnosis (IMMUNID)", "No", "Yes" if has_immun_dx else "No", not has_immun_dx))
    checklist.append(ChecklistItem("Immunocompromised procedure (IMMUNIP)", "No", "Yes" if has_immun_proc else "No", not has_immun_proc))
    if has_cancer: exclusions.append("CANCEID")
    if has_immun_dx: exclusions.append("IMMUNID")
    if has_immun_proc: exclusions.append("IMMUNIP")

    # Newborn principal
    newborn_principal = principal_in(MDC15PRINDX)
    checklist.append(ChecklistItem("Newborn principal (MDC15PRINDX)", "No", "Yes" if newborn_principal else "No", not newborn_principal))
    if newborn_principal: exclusions.append("MDC15PRINDX")

    # DRG 999
    drg999 = (drg3 == "999")
    checklist.append(ChecklistItem("DRG not 999 (ungroupable)", "Not 999", drg3, not drg999))
    if drg999: exclusions.append("DRG=999")

    # Missing fields
    def missing(col: str) -> bool:
        return (col not in row) or (pd.isna(row[col])) or (str(row[col]).strip() == "")
    miss_sex = missing("Sex")
    miss_age = missing("Age")
    miss_qtr = missing("DQTR") if "DQTR" in row else False
    miss_year = missing("Year") if "Year" in row else False
    miss_pdx = missing("Pdx")

    checklist.append(ChecklistItem("Missing Sex", "Present", "Missing" if miss_sex else "Present", not miss_sex))
    checklist.append(ChecklistItem("Missing Age", "Present", "Missing" if miss_age else "Present", not miss_age))
    checklist.append(ChecklistItem("Missing DQTR (if provided)", "Present/NA", "Missing" if miss_qtr else "Present/NA", not miss_qtr))
    checklist.append(ChecklistItem("Missing Year (if provided)", "Present/NA", "Missing" if miss_year else "Present/NA", not miss_year))
    checklist.append(ChecklistItem("Missing Principal DX", "Present", "Missing" if miss_pdx else "Present", not miss_pdx))

    if miss_sex: exclusions.append("Missing Sex")
    if miss_age: exclusions.append("Missing Age")
    if miss_qtr: exclusions.append("Missing DQTR")
    if miss_year: exclusions.append("Missing Year")
    if miss_pdx: exclusions.append("Missing Pdx")

    # Early exits
    if not denominator_met:
        return PSIEvalResult(
            encounter_id=enc_id, psi="PSI_07",
            result="EXCLUSION", denominator_met=False, numerator_met=False,
            exclusions_applied=exclusions, rationale_short="Denominator not met",
            checklist=checklist, debug={"drg3": drg3, "age": age}
        )
    if exclusions:
        return PSIEvalResult(
            encounter_id=enc_id, psi="PSI_07",
            result="EXCLUSION", denominator_met=True, numerator_met=False,
            exclusions_applied=exclusions, rationale_short="Denominator exclusion(s) applied",
            checklist=checklist, debug={"exclusions": exclusions}
        )

    # Numerator: secondary IDTMC3D not POA=Y
    numerator_hit = any((c in IDTMC3D) and (p != "Y") for (c,p) in secondary_dx)
    checklist.append(ChecklistItem("Numerator: secondary IDTMC3D not POA=Y", "Yes", "Yes" if numerator_hit else "No", numerator_hit))

    result = "INCLUSION" if numerator_hit else "EXCLUSION"
    rationale = "Central line-associated BSI coded (secondary, not POA)" if numerator_hit else "No qualifying CVC-related BSI"

    return PSIEvalResult(
        encounter_id=enc_id, psi="PSI_07",
        result=result, denominator_met=True, numerator_met=numerator_hit,
        exclusions_applied=[], rationale_short=rationale,
        checklist=checklist, debug={}
    )

@register_psi(8)
def evaluate_psi08(row: pd.Series, codes: Dict[str, List[str]]) -> PSIEvalResult:
    """
    PSI-08 In-Hospital Fall-Associated Fracture Rate (2025).
    Denominator: SURGI2R or MEDIC2R AND Age >=18.
    Exclusions: principal FXID; secondary FXID with POA=Y; PROSFXID; MDC14PRINDX; MDC15PRINDX; DRG=999; missing key fields; missing MDC (if provided).
    Numerator: secondary FXID; stratify hip (HIPFXID) first, else other fracture.
    Spec source: user-uploaded 2025 PSI-08 markdown.
    """
    enc_id = str(row.get("EncounterID", ""))
    checklist: List[ChecklistItem] = []
    exclusions: List[str] = []
    debug: Dict[str, Any] = {}

    # Core fields
    age = row.get("Age", None)
    drg3 = row.get("MS-DRG_3digit", "")
    principal_dx = str(row.get("Pdx", "")).strip().upper()

    # Parse DX/POA
    dx_list = extract_dx_poa(row)  # [(code, poa, role)]
    secondary_dx = [(c,p) for (c,p,role) in dx_list if role == "SECONDARY"]
    all_dx = [c for (c,_,_) in dx_list]

    # Sets
    SURGI2R = set(str(x).zfill(3) for x in codes.get("SURGI2R", []))
    MEDIC2R = set(str(x).zfill(3) for x in codes.get("MEDIC2R", []))
    FXID = set(codes.get("FXID", []))
    HIPFXID = set(codes.get("HIPFXID", []))
    PROSFXID = set(codes.get("PROSFXID", []))
    MDC14PRINDX = set(codes.get("MDC14PRINDX", []))
    MDC15PRINDX = set(codes.get("MDC15PRINDX", []))

    # Denominator
    age_ok = False
    if age is not None and str(age).strip() != "":
        try:
            age_ok = float(age) >= 18
        except Exception:
            age_ok = False
    drg_ok = (drg3 in SURGI2R) or (drg3 in MEDIC2R)

    checklist.append(ChecklistItem("DRG in SURGI2R or MEDIC2R", "Yes", drg3, drg_ok))
    checklist.append(ChecklistItem("Age >= 18", ">=18", age, age_ok))

    denominator_met = drg_ok and age_ok

    # Exclusions (precedence)
    principal_in = lambda S: principal_dx in S
    any_dx_in = lambda S: any(c in S for c in all_dx)

    # Principal fracture
    excl_prin_fx = principal_in(FXID)
    checklist.append(ChecklistItem("Principal fracture diagnosis (FXID)", "No", "Yes" if excl_prin_fx else "No", not excl_prin_fx))
    if excl_prin_fx: exclusions.append("Principal FXID")

    # Secondary fracture POA=Y
    excl_sec_fx_poaY = any((c in FXID) and (p == "Y") for (c,p) in secondary_dx)
    checklist.append(ChecklistItem("Secondary FXID present-on-admission", "No", "Yes" if excl_sec_fx_poaY else "No", not excl_sec_fx_poaY))
    if excl_sec_fx_poaY: exclusions.append("Secondary FXID POA=Y")

    # Prosthesis-associated fracture
    excl_prosfx = any_dx_in(PROSFXID)
    checklist.append(ChecklistItem("Prosthesis-associated fracture (PROSFXID)", "No", "Yes" if excl_prosfx else "No", not excl_prosfx))
    if excl_prosfx: exclusions.append("PROSFXID")

    # Obstetric/Newborn principals
    excl_ob = principal_in(MDC14PRINDX)
    excl_nb = principal_in(MDC15PRINDX)
    checklist.append(ChecklistItem("Obstetric principal (MDC14PRINDX)", "No", "Yes" if excl_ob else "No", not excl_ob))
    checklist.append(ChecklistItem("Newborn principal (MDC15PRINDX)", "No", "Yes" if excl_nb else "No", not excl_nb))
    if excl_ob: exclusions.append("MDC14PRINDX")
    if excl_nb: exclusions.append("MDC15PRINDX")

    # DRG 999
    drg999 = (drg3 == "999")
    checklist.append(ChecklistItem("DRG not 999 (ungroupable)", "Not 999", drg3, not drg999))
    if drg999: exclusions.append("DRG=999")

    # Missing fields
    def missing(col: str) -> bool:
        return (col not in row) or (pd.isna(row[col])) or (str(row[col]).strip() == "")
    miss_sex = missing("Sex")
    miss_age = missing("Age")
    miss_qtr = missing("DQTR") if "DQTR" in row else False
    miss_year = missing("Year") if "Year" in row else False
    miss_pdx = missing("Pdx")

    checklist.append(ChecklistItem("Missing Sex", "Present", "Missing" if miss_sex else "Present", not miss_sex))
    checklist.append(ChecklistItem("Missing Age", "Present", "Missing" if miss_age else "Present", not miss_age))
    checklist.append(ChecklistItem("Missing DQTR (if provided)", "Present/NA", "Missing" if miss_qtr else "Present/NA", not miss_qtr))
    checklist.append(ChecklistItem("Missing Year (if provided)", "Present/NA", "Missing" if miss_year else "Present/NA", not miss_year))
    checklist.append(ChecklistItem("Missing Principal DX", "Present", "Missing" if miss_pdx else "Present", not miss_pdx))

    if miss_sex: exclusions.append("Missing Sex")
    if miss_age: exclusions.append("Missing Age")
    if miss_qtr: exclusions.append("Missing DQTR")
    if miss_year: exclusions.append("Missing Year")
    if miss_pdx: exclusions.append("Missing Pdx")

    # Missing MDC if provided
    if "MDC" in row:
        miss_mdc = missing("MDC")
        checklist.append(ChecklistItem("Missing MDC (if provided)", "Present/NA", "Missing" if miss_mdc else "Present/NA", not miss_mdc))
        if miss_mdc: exclusions.append("Missing MDC")

    # Early exits
    if not denominator_met:
        return PSIEvalResult(
            encounter_id=enc_id, psi="PSI_08",
            result="EXCLUSION", denominator_met=False, numerator_met=False,
            exclusions_applied=exclusions, rationale_short="Denominator not met",
            checklist=checklist, debug={"drg3": drg3, "age": age}
        )
    if exclusions:
        return PSIEvalResult(
            encounter_id=enc_id, psi="PSI_08",
            result="EXCLUSION", denominator_met=True, numerator_met=False,
            exclusions_applied=exclusions, rationale_short="Denominator exclusion(s) applied",
            checklist=checklist, debug={"exclusions": exclusions}
        )

    # Numerator with hip-vs-other hierarchy
    # Any secondary FXID? If yes, check if any in HIPFXID; else classify as OTHER.
    secondary_fx = [(c,p) for (c,p) in secondary_dx if c in FXID and p != "Y"]
    has_any_fx = len(secondary_fx) > 0
    has_hip_fx = any((c in HIPFXID) for (c,_) in secondary_fx)

    numerator_hit = has_any_fx
    stratum = "HIP_FRACTURE" if has_hip_fx else ("OTHER_FRACTURE" if has_any_fx else None)

    checklist.append(ChecklistItem("Numerator: secondary FXID (not POA=Y)", "Yes", "Yes" if has_any_fx else "No", has_any_fx))
    if has_any_fx:
        checklist.append(ChecklistItem("Hierarchy: hip fracture takes priority", "HIP over OTHER", "HIP" if has_hip_fx else "OTHER", True))

    result = "INCLUSION" if numerator_hit else "EXCLUSION"
    rationale = ("Fall-associated hip fracture" if has_hip_fx else
                 "Fall-associated non-hip fracture" if has_any_fx else
                 "No qualifying fracture")

    return PSIEvalResult(
        encounter_id=enc_id, psi="PSI_08",
        result=result, denominator_met=True, numerator_met=numerator_hit,
        exclusions_applied=[], rationale_short=rationale,
        checklist=checklist, debug={"stratum": stratum}
    )

@register_psi(9)
def evaluate_psi09(row: pd.Series, codes: Dict[str, List[str]]) -> PSIEvalResult:
    """
    PSI-09 Postoperative Hemorrhage or Hematoma Rate (2025).
    Denominator: SURGI2R AND Age >=18 AND at least one ORPROC.
    Numerator: secondary POHMRI2D AND any HEMOTH2P.
    Exclusions (precedence):
      - Principal or secondary (POA=Y) in POHMRI2D
      - Only ORPROC performed is a treatment procedure (HEMOTH2P)
      - Treatment procedure (HEMOTH2P) occurs BEFORE first ORPROC (if dates available)
      - Any DX in COAGDID
      - Principal or secondary (POA=Y) in MEDBLEEDD
      - Any THROMBOLYTICP before/on same day as first HEMOTH2P (if dates available)
      - Principal in MDC14PRINDX or MDC15PRINDX
      - DRG=999
      - Missing Sex/Age/DQTR/Year/Principal DX (and MDC if provided)
    """
    enc_id = str(row.get("EncounterID", ""))
    checklist: List[ChecklistItem] = []
    exclusions: List[str] = []
    debug: Dict[str, Any] = {}

    # Core fields
    age = row.get("Age", None)
    drg3 = row.get("MS-DRG_3digit", "")
    principal_dx = str(row.get("Pdx", "")).strip().upper()

    # Parse DX/POA and procedures
    dx_list = extract_dx_poa(row)  # [(code, poa, role)]
    secondary_dx = [(c,p) for (c,p,role) in dx_list if role == "SECONDARY"]
    all_dx = [c for (c,_,_) in dx_list]
    procs = extract_procedures(row)  # [(code, dt, col)]

    # Sets
    SURGI2R = set(str(x).zfill(3) for x in codes.get("SURGI2R", []))
    ORPROC = set(codes.get("ORPROC", []))
    POHMRI2D = set(codes.get("POHMRI2D", []))
    HEMOTH2P = set(codes.get("HEMOTH2P", []))
    COAGDID = set(codes.get("COAGDID", []))
    MEDBLEEDD = set(codes.get("MEDBLEEDD", []))
    THROMBOLYTICP = set(codes.get("THROMBOLYTICP", []))
    MDC14PRINDX = set(codes.get("MDC14PRINDX", []))
    MDC15PRINDX = set(codes.get("MDC15PRINDX", []))

    # Denominator
    age_ok = False
    if age is not None and str(age).strip() != "":
        try:
            age_ok = float(age) >= 18
        except Exception:
            age_ok = False
    drg_ok = (drg3 in SURGI2R)
    has_orproc = any(code in ORPROC for (code,_,_) in procs)

    checklist.append(ChecklistItem("DRG in SURGI2R (surgical)", "Yes", drg3, drg_ok))
    checklist.append(ChecklistItem("Age >= 18", ">=18", age, age_ok))
    checklist.append(ChecklistItem("At least one ORPROC", "Yes", "Yes" if has_orproc else "No", has_orproc))

    denominator_met = drg_ok and age_ok and has_orproc

    # Helpers
    principal_in = lambda S: principal_dx in S
    any_dx_in = lambda S: any(c in S for c in all_dx)

    # Exclusions (precedence)
    # 1) Principal or secondary POA=Y in POHMRI2D
    excl_prin_pohm = principal_in(POHMRI2D)
    excl_sec_poaY_pohm = any((c in POHMRI2D) and (p == "Y") for (c,p) in secondary_dx)
    checklist.append(ChecklistItem("Principal hemorrhage/hematoma (POHMRI2D)", "No", "Yes" if excl_prin_pohm else "No", not excl_prin_pohm))
    checklist.append(ChecklistItem("Secondary POHMRI2D present-on-admission", "No", "Yes" if excl_sec_poaY_pohm else "No", not excl_sec_poaY_pohm))
    if excl_prin_pohm: exclusions.append("Principal POHMRI2D")
    if excl_sec_poaY_pohm: exclusions.append("Secondary POHMRI2D POA=Y")

    # 2) Only ORPROC performed is a treatment procedure (HEMOTH2P)
    orproc_codes = [code for (code,_,_) in procs if code in ORPROC]
    only_treatment_or = False
    if orproc_codes:
        only_treatment_or = all(code in HEMOTH2P for code in orproc_codes)
    checklist.append(ChecklistItem("Only ORPROC codes are treatment (HEMOTH2P)", "No", "Yes" if only_treatment_or else "No", not only_treatment_or))
    if only_treatment_or: exclusions.append("Only ORPROC = HEMOTH2P")

    # 3) Treatment before first ORPROC (if dates available)
    # Find earliest ORPROC datetime and earliest HEMOTH2P datetime
    earliest_or_dt = None
    earliest_treat_dt = None
    for code, pdt, col in procs:
        if code in ORPROC and pdt is not None:
            if earliest_or_dt is None or pdt < earliest_or_dt:
                earliest_or_dt = pdt
        if code in HEMOTH2P and pdt is not None:
            if earliest_treat_dt is None or pdt < earliest_treat_dt:
                earliest_treat_dt = pdt
    treat_before_or = False
    if earliest_or_dt is not None and earliest_treat_dt is not None:
        treat_before_or = earliest_treat_dt.normalize() < earliest_or_dt.normalize()
    checklist.append(ChecklistItem("Treatment occurs before first ORPROC (if dated)", "No", f"treat={earliest_treat_dt}, or={earliest_or_dt}", not treat_before_or))
    if treat_before_or: exclusions.append("Treatment before first ORPROC")

    # 4) Any coagulation disorder
    has_coag = any_dx_in(COAGDID)
    checklist.append(ChecklistItem("Coagulation disorder (COAGDID)", "No", "Yes" if has_coag else "No", not has_coag))
    if has_coag: exclusions.append("COAGDID")

    # 5) Principal or secondary POA=Y medication-related coagulopathy
    excl_prin_medbleed = principal_in(MEDBLEEDD)
    excl_sec_medbleed_poaY = any((c in MEDBLEEDD) and (p == "Y") for (c,p) in secondary_dx)
    checklist.append(ChecklistItem("Principal med-related coagulopathy (MEDBLEEDD)", "No", "Yes" if excl_prin_medbleed else "No", not excl_prin_medbleed))
    checklist.append(ChecklistItem("Secondary MEDBLEEDD present-on-admission", "No", "Yes" if excl_sec_medbleed_poaY else "No", not excl_sec_medbleed_poaY))
    if excl_prin_medbleed: exclusions.append("Principal MEDBLEEDD")
    if excl_sec_medbleed_poaY: exclusions.append("Secondary MEDBLEEDD POA=Y")

    # 6) Any THROMBOLYTICP before/on same day as first HEMOTH2P (if dated)
    earliest_throm_dt = None
    for code, pdt, col in procs:
        if code in THROMBOLYTICP and pdt is not None:
            if earliest_throm_dt is None or pdt < earliest_throm_dt:
                earliest_throm_dt = pdt
    throm_before_or_same = False
    if earliest_treat_dt is not None and earliest_throm_dt is not None:
        try:
            throm_before_or_same = earliest_throm_dt.normalize() <= earliest_treat_dt.normalize()
        except Exception:
            throm_before_or_same = False
    checklist.append(ChecklistItem("Thrombolytic before/on same day as first treatment (if dated)", "No", f"throm={earliest_throm_dt}, treat={earliest_treat_dt}", not throm_before_or_same))
    if throm_before_or_same: exclusions.append("THROMBOLYTICP <= first HEMOTH2P date")

    # 7) Obstetric/Newborn principal
    obstetric_principal = principal_in(MDC14PRINDX)
    newborn_principal = principal_in(MDC15PRINDX)
    checklist.append(ChecklistItem("Obstetric principal (MDC14PRINDX)", "No", "Yes" if obstetric_principal else "No", not obstetric_principal))
    checklist.append(ChecklistItem("Newborn principal (MDC15PRINDX)", "No", "Yes" if newborn_principal else "No", not newborn_principal))
    if obstetric_principal: exclusions.append("MDC14PRINDX")
    if newborn_principal: exclusions.append("MDC15PRINDX")

    # 8) DRG 999
    drg999 = (drg3 == "999")
    checklist.append(ChecklistItem("DRG not 999 (ungroupable)", "Not 999", drg3, not drg999))
    if drg999: exclusions.append("DRG=999")

    # 9) Missing fields
    def missing(col: str) -> bool:
        return (col not in row) or (pd.isna(row[col])) or (str(row[col]).strip() == "")
    miss_sex = missing("Sex")
    miss_age = missing("Age")
    miss_qtr = missing("DQTR") if "DQTR" in row else False
    miss_year = missing("Year") if "Year" in row else False
    miss_pdx = missing("Pdx")

    checklist.append(ChecklistItem("Missing Sex", "Present", "Missing" if miss_sex else "Present", not miss_sex))
    checklist.append(ChecklistItem("Missing Age", "Present", "Missing" if miss_age else "Present", not miss_age))
    checklist.append(ChecklistItem("Missing DQTR (if provided)", "Present/NA", "Missing" if miss_qtr else "Present/NA", not miss_qtr))
    checklist.append(ChecklistItem("Missing Year (if provided)", "Present/NA", "Missing" if miss_year else "Present/NA", not miss_year))
    checklist.append(ChecklistItem("Missing Principal DX", "Present", "Missing" if miss_pdx else "Present", not miss_pdx))

    if miss_sex: exclusions.append("Missing Sex")
    if miss_age: exclusions.append("Missing Age")
    if miss_qtr: exclusions.append("Missing DQTR")
    if miss_year: exclusions.append("Missing Year")
    if miss_pdx: exclusions.append("Missing Pdx")

    # Missing MDC if provided
    if "MDC" in row:
        miss_mdc = missing("MDC")
        checklist.append(ChecklistItem("Missing MDC (if provided)", "Present/NA", "Missing" if miss_mdc else "Present/NA", not miss_mdc))
        if miss_mdc: exclusions.append("Missing MDC")

    # Early exits
    if not denominator_met:
        return PSIEvalResult(
            encounter_id=enc_id, psi="PSI_09",
            result="EXCLUSION", denominator_met=False, numerator_met=False,
            exclusions_applied=exclusions, rationale_short="Denominator not met",
            checklist=checklist, debug={"drg3": drg3, "age": age}
        )
    if exclusions:
        return PSIEvalResult(
            encounter_id=enc_id, psi="PSI_09",
            result="EXCLUSION", denominator_met=True, numerator_met=False,
            exclusions_applied=exclusions, rationale_short="Denominator exclusion(s) applied",
            checklist=checklist, debug={"exclusions": exclusions, "earliest_or": str(earliest_or_dt), "earliest_treat": str(earliest_treat_dt), "earliest_throm": str(earliest_throm_dt)}
        )

    # Numerator: any secondary POHMRI2D AND any HEMOTH2P
    has_pohm_sec = any((c in POHMRI2D) for (c,_) in secondary_dx)
    has_treatment = any(code in HEMOTH2P for (code,_,_) in procs)
    numerator_hit = has_pohm_sec and has_treatment

    checklist.append(ChecklistItem("Numerator: secondary POHMRI2D", "Yes", "Yes" if has_pohm_sec else "No", has_pohm_sec))
    checklist.append(ChecklistItem("Numerator: any HEMOTH2P", "Yes", "Yes" if has_treatment else "No", has_treatment))

    result = "INCLUSION" if numerator_hit else "EXCLUSION"
    rationale = "Postop hemorrhage/hematoma treated (secondary DX + procedure)" if numerator_hit else "No qualifying postop hemorrhage/hematoma"

    return PSIEvalResult(
        encounter_id=enc_id, psi="PSI_09",
        result=result, denominator_met=True, numerator_met=numerator_hit,
        exclusions_applied=[], rationale_short=rationale,
        checklist=checklist, debug={"earliest_or": str(earliest_or_dt), "earliest_treat": str(earliest_treat_dt), "earliest_throm": str(earliest_throm_dt)}
    )

@register_psi(10)
def evaluate_psi10(row: pd.Series, codes: Dict[str, List[str]]) -> PSIEvalResult:
    """
    PSI-10 Postoperative Acute Kidney Injury Requiring Dialysis Rate (2025).
    Denominator: elective surgical (SURGI2R) AND Age>=18 AND >=1 ORPROC AND ATYPE=3.
    Exclusions (precedence):
      - Principal or secondary POA=Y in PHYSIDB (AKI)
      - Any DIALYIP <= first ORPROC (if dated)
      - Any DIALY2P <= first ORPROC (if dated)
      - Principal or secondary POA=Y in CARDIID, CARDRID, SHOCKID, CRENLFD
      - Principal in URINARYOBSID
      - POA SOLKIDD AND any PNEPHREP (nephrectomy)
      - Obstetric/Newborn principal (MDC14PRINDX/MDC15PRINDX)
      - DRG=999
      - Missing Sex/Age/DQTR/Year/Pdx (and MDC if provided)
    Numerator: secondary PHYSIDB AND any DIALYIP (dialysis).
    """
    enc_id = str(row.get("EncounterID", ""))
    checklist: List[ChecklistItem] = []
    exclusions: List[str] = []
    debug: Dict[str, Any] = {}

    # Core fields
    age = row.get("Age", None)
    drg3 = row.get("MS-DRG_3digit", "")
    atype = str(row.get("ATYPE", "")).strip()
    principal_dx = str(row.get("Pdx", "")).strip().upper()

    # Parse DX/POA and procedures
    dx_list = extract_dx_poa(row)  # [(code, poa, role)]
    secondary_dx = [(c,p) for (c,p,role) in dx_list if role == "SECONDARY"]
    all_dx = [c for (c,_,_) in dx_list]
    procs = extract_procedures(row)  # [(code, dt, col)]

    # Sets
    SURGI2R = set(str(x).zfill(3) for x in codes.get("SURGI2R", []))
    ORPROC = set(codes.get("ORPROC", []))
    PHYSIDB = set(codes.get("PHYSIDB", []))
    DIALYIP = set(codes.get("DIALYIP", []))
    DIALY2P = set(codes.get("DIALY2P", []))
    CARDIID = set(codes.get("CARDIID", []))
    CARDRID = set(codes.get("CARDRID", []))
    SHOCKID = set(codes.get("SHOCKID", []))
    CRENLFD = set(codes.get("CRENLFD", []))
    URINARYOBSID = set(codes.get("URINARYOBSID", []))
    SOLKIDD = set(codes.get("SOLKIDD", []))
    PNEPHREP = set(codes.get("PNEPHREP", []))
    MDC14PRINDX = set(codes.get("MDC14PRINDX", []))
    MDC15PRINDX = set(codes.get("MDC15PRINDX", []))

    # Denominator
    age_ok = False
    if age is not None and str(age).strip() != "":
        try:
            age_ok = float(age) >= 18
        except Exception:
            age_ok = False
    drg_ok = (drg3 in SURGI2R)
    has_orproc = any(code in ORPROC for (code,_,_) in procs)
    elective_ok = (atype == "3")

    checklist.append(ChecklistItem("DRG in SURGI2R (surgical)", "Yes", drg3, drg_ok))
    checklist.append(ChecklistItem("Age >= 18", ">=18", age, age_ok))
    checklist.append(ChecklistItem("At least one ORPROC", "Yes", "Yes" if has_orproc else "No", has_orproc))
    checklist.append(ChecklistItem("Admission type elective (ATYPE=3)", "Yes", atype, elective_ok))

    denominator_met = drg_ok and age_ok and has_orproc and elective_ok

    # Helpers
    principal_in = lambda S: principal_dx in S
    any_dx_in = lambda S: any(c in S for c in all_dx)

    # Exclusions (precedence)
    # 1) Principal or secondary POA=Y in PHYSIDB
    excl_prin_aki = principal_in(PHYSIDB)
    excl_sec_aki_poaY = any((c in PHYSIDB) and (p == "Y") for (c,p) in secondary_dx)
    checklist.append(ChecklistItem("Principal AKI (PHYSIDB)", "No", "Yes" if excl_prin_aki else "No", not excl_prin_aki))
    checklist.append(ChecklistItem("Secondary AKI present-on-admission (PHYSIDB)", "No", "Yes" if excl_sec_aki_poaY else "No", not excl_sec_aki_poaY))
    if excl_prin_aki: exclusions.append("Principal PHYSIDB")
    if excl_sec_aki_poaY: exclusions.append("Secondary PHYSIDB POA=Y")

    # Find earliest ORPROC, earliest DIALYIP, earliest DIALY2P
    earliest_or_dt = None
    earliest_dialyip_dt = None
    earliest_dialy2p_dt = None
    for code, pdt, col in procs:
        if code in ORPROC and pdt is not None:
            if earliest_or_dt is None or pdt < earliest_or_dt:
                earliest_or_dt = pdt
        if code in DIALYIP and pdt is not None:
            if earliest_dialyip_dt is None or pdt < earliest_dialyip_dt:
                earliest_dialyip_dt = pdt
        if code in DIALY2P and pdt is not None:
            if earliest_dialy2p_dt is None or pdt < earliest_dialy2p_dt:
                earliest_dialy2p_dt = pdt

    # 2) Any DIALYIP <= first ORPROC (if dated)
    excl_dialyip_before_or = False
    if earliest_or_dt is not None and earliest_dialyip_dt is not None:
        excl_dialyip_before_or = earliest_dialyip_dt.normalize() <= earliest_or_dt.normalize()
    checklist.append(ChecklistItem("Dialysis (DIALYIP) on/before first ORPROC (if dated)", "No", f"dialyip={earliest_dialyip_dt}, or={earliest_or_dt}", not excl_dialyip_before_or))
    if excl_dialyip_before_or: exclusions.append("DIALYIP <= first ORPROC")

    # 3) Any DIALY2P <= first ORPROC (if dated)
    excl_dialy2p_before_or = False
    if earliest_or_dt is not None and earliest_dialy2p_dt is not None:
        excl_dialy2p_before_or = earliest_dialy2p_dt.normalize() <= earliest_or_dt.normalize()
    checklist.append(ChecklistItem("Dialysis other (DIALY2P) on/before first ORPROC (if dated)", "No", f"dialy2p={earliest_dialy2p_dt}, or={earliest_or_dt}", not excl_dialy2p_before_or))
    if excl_dialy2p_before_or: exclusions.append("DIALY2P <= first ORPROC")

    # 4) Principal or secondary POA=Y in cardiac arrest/dysrhythmia/shock/ESRD (CRENLFD)
    for label, S in [
        ("CARDIID", CARDIID),
        ("CARDRID", CARDRID),
        ("SHOCKID", SHOCKID),
        ("CRENLFD", CRENLFD),
    ]:
        prin = principal_in(S)
        poaY = any((c in S) and (p == "Y") for (c,p) in secondary_dx)
        checklist.append(ChecklistItem(f"Principal {label} or secondary POA=Y", "No", f"Prin={prin}, POA-Y={poaY}", not (prin or poaY)))
        if prin or poaY:
            exclusions.append(f"{label} principal/POA=Y")

    # 5) Principal urinary tract obstruction
    excl_urinary_obs = principal_in(URINARYOBSID)
    checklist.append(ChecklistItem("Principal urinary tract obstruction (URINARYOBSID)", "No", "Yes" if excl_urinary_obs else "No", not excl_urinary_obs))
    if excl_urinary_obs: exclusions.append("URINARYOBSID")

    # 6) POA solitary kidney AND any nephrectomy (PNEPHREP)
    has_poa_solkid = any((c in SOLKIDD) and (p == "Y") for (c,p) in secondary_dx) or (principal_dx in SOLKIDD and any(p == "Y" for (c,p) in [(principal_dx, row.get('POA1',''))]))
    has_nephrectomy = any(code in PNEPHREP for (code,_,_) in procs)
    excl_solkid_neph = has_poa_solkid and has_nephrectomy
    checklist.append(ChecklistItem("POA solitary kidney (SOLKIDD)", "No", "Yes" if has_poa_solkid else "No", not has_poa_solkid))
    checklist.append(ChecklistItem("Any nephrectomy (PNEPHREP)", "No", "Yes" if has_nephrectomy else "No", not has_nephrectomy))
    checklist.append(ChecklistItem("Exclude if both POA SOLKIDD and PNEPHREP", "No", f"{excl_solkid_neph}", not excl_solkid_neph))
    if excl_solkid_neph: exclusions.append("SOLKIDD POA + PNEPHREP")

    # 7) Obstetric/Newborn principal
    obstetric_principal = principal_in(MDC14PRINDX)
    newborn_principal = principal_in(MDC15PRINDX)
    checklist.append(ChecklistItem("Obstetric principal (MDC14PRINDX)", "No", "Yes" if obstetric_principal else "No", not obstetric_principal))
    checklist.append(ChecklistItem("Newborn principal (MDC15PRINDX)", "No", "Yes" if newborn_principal else "No", not newborn_principal))
    if obstetric_principal: exclusions.append("MDC14PRINDX")
    if newborn_principal: exclusions.append("MDC15PRINDX")

    # 8) DRG 999
    drg999 = (drg3 == "999")
    checklist.append(ChecklistItem("DRG not 999 (ungroupable)", "Not 999", drg3, not drg999))
    if drg999: exclusions.append("DRG=999")

    # 9) Missing fields
    def missing(col: str) -> bool:
        return (col not in row) or (pd.isna(row[col])) or (str(row[col]).strip() == "")
    miss_sex = missing("Sex")
    miss_age = missing("Age")
    miss_qtr = missing("DQTR") if "DQTR" in row else False
    miss_year = missing("Year") if "Year" in row else False
    miss_pdx = missing("Pdx")

    checklist.append(ChecklistItem("Missing Sex", "Present", "Missing" if miss_sex else "Present", not miss_sex))
    checklist.append(ChecklistItem("Missing Age", "Present", "Missing" if miss_age else "Present", not miss_age))
    checklist.append(ChecklistItem("Missing DQTR (if provided)", "Present/NA", "Missing" if miss_qtr else "Present/NA", not miss_qtr))
    checklist.append(ChecklistItem("Missing Year (if provided)", "Present/NA", "Missing" if miss_year else "Present/NA", not miss_year))
    checklist.append(ChecklistItem("Missing Principal DX", "Present", "Missing" if miss_pdx else "Present", not miss_pdx))

    if miss_sex: exclusions.append("Missing Sex")
    if miss_age: exclusions.append("Missing Age")
    if miss_qtr: exclusions.append("Missing DQTR")
    if miss_year: exclusions.append("Missing Year")
    if miss_pdx: exclusions.append("Missing Pdx")

    # Missing MDC if provided
    if "MDC" in row:
        miss_mdc = missing("MDC")
        checklist.append(ChecklistItem("Missing MDC (if provided)", "Present/NA", "Missing" if miss_mdc else "Present/NA", not miss_mdc))
        if miss_mdc: exclusions.append("Missing MDC")

    # Early exits
    if not denominator_met:
        return PSIEvalResult(
            encounter_id=enc_id, psi="PSI_10",
            result="EXCLUSION", denominator_met=False, numerator_met=False,
            exclusions_applied=exclusions, rationale_short="Denominator not met",
            checklist=checklist, debug={"drg3": drg3, "age": age, "atype": atype}
        )
    if exclusions:
        return PSIEvalResult(
            encounter_id=enc_id, psi="PSI_10",
            result="EXCLUSION", denominator_met=True, numerator_met=False,
            exclusions_applied=exclusions, rationale_short="Denominator exclusion(s) applied",
            checklist=checklist, debug={"exclusions": exclusions, "earliest_or": str(earliest_or_dt), "earliest_dialyip": str(earliest_dialyip_dt), "earliest_dialy2p": str(earliest_dialy2p_dt)}
        )

    # Numerator: secondary PHYSIDB AND any DIALYIP
    has_sec_aki = any((c in PHYSIDB) for (c,_) in secondary_dx)
    has_dialysis = any(code in DIALYIP for (code,_,_) in procs)

    checklist.append(ChecklistItem("Numerator: secondary AKI (PHYSIDB)", "Yes", "Yes" if has_sec_aki else "No", has_sec_aki))
    checklist.append(ChecklistItem("Numerator: any dialysis procedure (DIALYIP)", "Yes", "Yes" if has_dialysis else "No", has_dialysis))

    numerator_hit = has_sec_aki and has_dialysis

    result = "INCLUSION" if numerator_hit else "EXCLUSION"
    rationale = "Postop AKI requiring dialysis (secondary AKI + dialysis)" if numerator_hit else "No qualifying AKI+dialysis"

    return PSIEvalResult(
        encounter_id=enc_id, psi="PSI_10",
        result=result, denominator_met=True, numerator_met=numerator_hit,
        exclusions_applied=[], rationale_short=rationale,
        checklist=checklist, debug={"earliest_or": str(earliest_or_dt), "earliest_dialyip": str(earliest_dialyip_dt), "earliest_dialy2p": str(earliest_dialy2p_dt)}
    )

@register_psi(11)
def evaluate_psi11(row: pd.Series, codes: Dict[str, List[str]]) -> PSIEvalResult:
    """
    PSI-11 Postoperative Respiratory Failure Rate (2025).
    Denominator: elective surgical (SURGI2R) AND Age>=18 AND >=1 ORPROC AND ATYPE=3.
    Exclusions (precedence):
      - Principal or secondary POA=Y in ACURF3D (acute resp failure)
      - Principal or secondary POA=Y in ESHFD (end stage heart failure)
      - Any POA DX in TRACHID (tracheostomy status)
      - Only ORPROC = TRACHIP
      - TRACHIP occurs BEFORE first ORPROC (if dates available)
      - Any DX in MALHYPD (malignant HTN) (any position)
      - Any POA DX in NEUROMD or DGNEUID (neuromuscular/degenerative neuro disease)
      - Any procedure in NUCRANP, PRESOPP, LUNGCIP, LUNGHEARTTXP
      - MDC = 4 (if MDC provided)
      - Principal in MDC14PRINDX or MDC15PRINDX
      - DRG=999
      - Missing Sex/Age/DQTR/Year/Pdx (and MDC if provided)
    Numerator (any of the following):
      1) Secondary ACURF2D
      2) Last PR9672P (mech vent >96h) >= 0 days after first ORPROC
      3) Last PR9671P (mech vent 24–96h) >= 2 days after first ORPROC
      4) Last PR9604P (intubation) >= 1 day after first ORPROC
    """
    enc_id = str(row.get("EncounterID", ""))
    checklist: List[ChecklistItem] = []
    exclusions: List[str] = []
    debug: Dict[str, Any] = {}

    # Core fields
    age = row.get("Age", None)
    drg3 = row.get("MS-DRG_3digit", "")
    atype = str(row.get("ATYPE", "")).strip()
    principal_dx = str(row.get("Pdx", "")).strip().upper()
    mdc_val = str(row.get("MDC", "")).strip() if "MDC" in row else ""

    # Parse DX/POA and procedures
    dx_list = extract_dx_poa(row)  # [(code, poa, role)]
    secondary_dx = [(c,p) for (c,p,role) in dx_list if role == "SECONDARY"]
    all_dx = [c for (c,_,_) in dx_list]
    procs = extract_procedures(row)  # [(code, dt, col)]

    # Sets
    SURGI2R = set(str(x).zfill(3) for x in codes.get("SURGI2R", []))
    ORPROC = set(codes.get("ORPROC", []))
    ACURF2D = set(codes.get("ACURF2D", []))
    ACURF3D = set(codes.get("ACURF3D", []))
    ESHFD = set(codes.get("ESHFD", []))
    TRACHID = set(codes.get("TRACHID", []))
    TRACHIP = set(codes.get("TRACHIP", []))
    MALHYPD = set(codes.get("MALHYPD", []))
    NEUROMD = set(codes.get("NEUROMD", []))
    DGNEUID = set(codes.get("DGNEUID", []))
    NUCRANP = set(codes.get("NUCRANP", []))
    PRESOPP = set(codes.get("PRESOPP", []))
    LUNGCIP = set(codes.get("LUNGCIP", []))
    LUNGHEARTTXP = set(codes.get("LUNGHEARTTXP", []))
    MDC14PRINDX = set(codes.get("MDC14PRINDX", []))
    MDC15PRINDX = set(codes.get("MDC15PRINDX", []))
    PR9672P = set(codes.get("PR9672P", []))
    PR9671P = set(codes.get("PR9671P", []))
    PR9604P = set(codes.get("PR9604P", []))

    # Denominator
    age_ok = False
    if age is not None and str(age).strip() != "":
        try:
            age_ok = float(age) >= 18
        except Exception:
            age_ok = False
    drg_ok = (drg3 in SURGI2R)
    has_orproc = any(code in ORPROC for (code,_,_) in procs)
    elective_ok = (atype == "3")

    checklist.append(ChecklistItem("DRG in SURGI2R (surgical)", "Yes", drg3, drg_ok))
    checklist.append(ChecklistItem("Age >= 18", ">=18", age, age_ok))
    checklist.append(ChecklistItem("At least one ORPROC", "Yes", "Yes" if has_orproc else "No", has_orproc))
    checklist.append(ChecklistItem("Admission type elective (ATYPE=3)", "Yes", atype, elective_ok))

    denominator_met = drg_ok and age_ok and has_orproc and elective_ok

    # Helpers
    principal_in = lambda S: principal_dx in S
    any_dx_in = lambda S: any(c in S for c in all_dx)
    any_poa_dx_in = lambda S: any((c in S) and (p == "Y") for (c,p) in secondary_dx) or (principal_dx in S and row.get("POA1","") == "Y")

    # Exclusions (precedence)
    # 1) Principal or secondary POA=Y in ACURF3D; 2) Principal/POA=Y in ESHFD
    for label, S in [("ACURF3D", ACURF3D), ("ESHFD", ESHFD)]:
        prin = principal_in(S)
        poaY = any((c in S) and (p == "Y") for (c,p) in secondary_dx)
        checklist.append(ChecklistItem(f"Principal or secondary POA=Y in {label}", "No", f"Prin={prin}, POA-Y={poaY}", not (prin or poaY)))
        if prin or poaY:
            exclusions.append(f"{label} principal/POA=Y")

    # 3) Any POA tracheostomy diagnosis
    trachid_poa = any_poa_dx_in(TRACHID)
    checklist.append(ChecklistItem("POA tracheostomy diagnosis (TRACHID)", "No", "Yes" if trachid_poa else "No", not trachid_poa))
    if trachid_poa: exclusions.append("TRACHID POA")

    # 4) Only ORPROC = TRACHIP
    orproc_codes = [code for (code,_,_) in procs if code in ORPROC]
    only_trachip = (len(orproc_codes) > 0 and all(code in TRACHIP for code in orproc_codes))
    checklist.append(ChecklistItem("Only ORPROC are tracheostomy (TRACHIP)", "No", "Yes" if only_trachip else "No", not only_trachip))
    if only_trachip: exclusions.append("Only TRACHIP ORPROC")

    # 5) TRACHIP before first ORPROC (if dated)
    earliest_or_dt = None
    earliest_trach_dt = None
    for code, pdt, col in procs:
        if code in ORPROC and pdt is not None:
            if earliest_or_dt is None or pdt < earliest_or_dt:
                earliest_or_dt = pdt
        if code in TRACHIP and pdt is not None:
            if earliest_trach_dt is None or pdt < earliest_trach_dt:
                earliest_trach_dt = pdt
    trach_before_or = False
    if earliest_or_dt is not None and earliest_trach_dt is not None:
        trach_before_or = earliest_trach_dt.normalize() < earliest_or_dt.normalize()
    checklist.append(ChecklistItem("TRACHIP before first ORPROC (if dated)", "No", f"trach={earliest_trach_dt}, or={earliest_or_dt}", not trach_before_or))
    if trach_before_or: exclusions.append("TRACHIP before first ORPROC")

    # 6) Any MALHYPD (any position)
    has_malhyp = any_dx_in(MALHYPD)
    checklist.append(ChecklistItem("Malignant hypertension (MALHYPD)", "No", "Yes" if has_malhyp else "No", not has_malhyp))
    if has_malhyp: exclusions.append("MALHYPD")

    # 7) Any POA in NEUROMD or DGNEUID
    poa_neuro = any_poa_dx_in(NEUROMD) or any_poa_dx_in(DGNEUID)
    checklist.append(ChecklistItem("POA neuromuscular/degenerative neuro disease (NEUROMD/DGNEUID)", "No", "Yes" if poa_neuro else "No", not poa_neuro))
    if poa_neuro: exclusions.append("NEUROMD/DGNEUID POA")

    # 8) Any procedure in NUCRANP, PRESOPP, LUNGCIP, LUNGHEARTTXP
    proc_hits = {
        "NUCRANP": any(code in NUCRANP for (code,_,_) in procs),
        "PRESOPP": any(code in PRESOPP for (code,_,_) in procs),
        "LUNGCIP": any(code in LUNGCIP for (code,_,_) in procs),
        "LUNGHEARTTXP": any(code in LUNGHEARTTXP for (code,_,_) in procs),
    }
    for label, hit in proc_hits.items():
        checklist.append(ChecklistItem(f"Procedure exclusion: {label}", "No", "Yes" if hit else "No", not hit))
        if hit: exclusions.append(label)

    # 9) MDC = 4 (Respiratory System), if MDC provided
    if mdc_val:
        is_mdc4 = (mdc_val == "4")
        checklist.append(ChecklistItem("MDC 4 (if provided)", "Not 4", mdc_val, not is_mdc4))
        if is_mdc4: exclusions.append("MDC=4")

    # 10) Obstetric/Newborn principal
    obstetric_principal = principal_in(MDC14PRINDX)
    newborn_principal = principal_in(MDC15PRINDX)
    checklist.append(ChecklistItem("Obstetric principal (MDC14PRINDX)", "No", "Yes" if obstetric_principal else "No", not obstetric_principal))
    checklist.append(ChecklistItem("Newborn principal (MDC15PRINDX)", "No", "Yes" if newborn_principal else "No", not newborn_principal))
    if obstetric_principal: exclusions.append("MDC14PRINDX")
    if newborn_principal: exclusions.append("MDC15PRINDX")

    # 11) DRG 999
    drg999 = (drg3 == "999")
    checklist.append(ChecklistItem("DRG not 999 (ungroupable)", "Not 999", drg3, not drg999))
    if drg999: exclusions.append("DRG=999")

    # 12) Missing fields
    def missing(col: str) -> bool:
        return (col not in row) or (pd.isna(row[col])) or (str(row[col]).strip() == "")
    miss_sex = missing("Sex")
    miss_age = missing("Age")
    miss_qtr = missing("DQTR") if "DQTR" in row else False
    miss_year = missing("Year") if "Year" in row else False
    miss_pdx = missing("Pdx")

    checklist.append(ChecklistItem("Missing Sex", "Present", "Missing" if miss_sex else "Present", not miss_sex))
    checklist.append(ChecklistItem("Missing Age", "Present", "Missing" if miss_age else "Present", not miss_age))
    checklist.append(ChecklistItem("Missing DQTR (if provided)", "Present/NA", "Missing" if miss_qtr else "Present/NA", not miss_qtr))
    checklist.append(ChecklistItem("Missing Year (if provided)", "Present/NA", "Missing" if miss_year else "Present/NA", not miss_year))
    checklist.append(ChecklistItem("Missing Principal DX", "Present", "Missing" if miss_pdx else "Present", not miss_pdx))

    if miss_sex: exclusions.append("Missing Sex")
    if miss_age: exclusions.append("Missing Age")
    if miss_qtr: exclusions.append("Missing DQTR")
    if miss_year: exclusions.append("Missing Year")
    if miss_pdx: exclusions.append("Missing Pdx")

    # Missing MDC if provided
    if "MDC" in row:
        miss_mdc = missing("MDC")
        checklist.append(ChecklistItem("Missing MDC (if provided)", "Present/NA", "Missing" if miss_mdc else "Present/NA", not miss_mdc))
        if miss_mdc: exclusions.append("Missing MDC")

    # Early exits
    if not denominator_met:
        return PSIEvalResult(
            encounter_id=enc_id, psi="PSI_11",
            result="EXCLUSION", denominator_met=False, numerator_met=False,
            exclusions_applied=exclusions, rationale_short="Denominator not met",
            checklist=checklist, debug={"drg3": drg3, "age": age, "atype": atype}
        )
    if exclusions:
        return PSIEvalResult(
            encounter_id=enc_id, psi="PSI_11",
            result="EXCLUSION", denominator_met=True, numerator_met=False,
            exclusions_applied=exclusions, rationale_short="Denominator exclusion(s) applied",
            checklist=checklist, debug={"exclusions": exclusions}
        )

    # Numerator paths
    # Path 1: secondary ACURF2D
    path1 = any((c in ACURF2D) for (c,_) in secondary_dx)

    # Prepare earliest OR date and last proc dates for PR9672P/PR9671P/PR9604P
    earliest_or_dt = None
    last_9672_dt = None
    last_9671_dt = None
    last_9604_dt = None
    for code, pdt, col in procs:
        if code in ORPROC and pdt is not None:
            if earliest_or_dt is None or pdt < earliest_or_dt:
                earliest_or_dt = pdt
        if code in PR9672P and pdt is not None:
            if last_9672_dt is None or pdt > last_9672_dt:
                last_9672_dt = pdt
        if code in PR9671P and pdt is not None:
            if last_9671_dt is None or pdt > last_9671_dt:
                last_9671_dt = pdt
        if code in PR9604P and pdt is not None:
            if last_9604_dt is None or pdt > last_9604_dt:
                last_9604_dt = pdt

    # Path 2: last PR9672P >= 0 days after first OR
    path2 = False
    # Path 3: last PR9671P >= 2 days after first OR
    path3 = False
    # Path 4: last PR9604P >= 1 day after first OR
    path4 = False

    if earliest_or_dt is not None:
        if last_9672_dt is not None:
            delta2 = (last_9672_dt.normalize() - earliest_or_dt.normalize()).days
            path2 = (delta2 >= 0)
            debug["delta_PR9672P_days"] = delta2
        if last_9671_dt is not None:
            delta3 = (last_9671_dt.normalize() - earliest_or_dt.normalize()).days
            path3 = (delta3 >= 2)
            debug["delta_PR9671P_days"] = delta3
        if last_9604_dt is not None:
            delta4 = (last_9604_dt.normalize() - earliest_or_dt.normalize()).days
            path4 = (delta4 >= 1)
            debug["delta_PR9604P_days"] = delta4

    numerator_met = path1 or path2 or path3 or path4

    checklist.append(ChecklistItem("Numerator path 1: secondary ACURF2D", "Yes (any path)", "Yes" if path1 else "No", numerator_met))
    checklist.append(ChecklistItem("Numerator path 2: PR9672P last >= 0 days after first OR", "Yes (any path)", f"{last_9672_dt} vs OR {earliest_or_dt}", numerator_met if path2 else False))
    checklist.append(ChecklistItem("Numerator path 3: PR9671P last >= 2 days after first OR", "Yes (any path)", f"{last_9671_dt} vs OR {earliest_or_dt}", numerator_met if path3 else False))
    checklist.append(ChecklistItem("Numerator path 4: PR9604P last >= 1 day after first OR", "Yes (any path)", f"{last_9604_dt} vs OR {earliest_or_dt}", numerator_met if path4 else False))

    result = "INCLUSION" if numerator_met else "EXCLUSION"
    rationale = "Postop respiratory failure criteria met" if numerator_met else "No qualifying postoperative respiratory failure"

    debug.update({
        "earliest_or": str(earliest_or_dt),
        "last_PR9672P": str(last_9672_dt),
        "last_PR9671P": str(last_9671_dt),
        "last_PR9604P": str(last_9604_dt),
        "paths": {"p1": path1, "p2": path2, "p3": path3, "p4": path4}
    })

    return PSIEvalResult(
        encounter_id=enc_id, psi="PSI_11",
        result=result, denominator_met=True, numerator_met=numerator_met,
        exclusions_applied=[], rationale_short=rationale,
        checklist=checklist, debug=debug
    )

@register_psi(12)
def evaluate_psi12(row: pd.Series, codes: Dict[str, List[str]]) -> PSIEvalResult:
    """
    PSI-12 Perioperative Pulmonary Embolism or Deep Vein Thrombosis Rate (2025).
    Denominator: SURGI2R AND Age>=18 AND >=1 ORPROC.
    Numerator: secondary DEEPVIB (proximal DVT) OR secondary PULMOID (PE), not POA=Y.
    Exclusions (precedence):
      - Principal or secondary POA=Y in DEEPVIB or PULMOID
      - Any secondary HITD (heparin-induced thrombocytopenia)
      - Any VENACIP or THROMP on/before first ORPROC (if dated)
      - Only ORPROCs are VENACIP/THROMP
      - First ORPROC >= 10 days after admission (if dated)
      - POA NEURTRAD (acute brain/spinal injury)
      - Any ECMOP procedure
      - Obstetric/Newborn principal (MDC14PRINDX/MDC15PRINDX)
      - DRG=999
      - Missing Sex/Age/DQTR/Year/Pdx (and MDC if provided)
    """
    enc_id = str(row.get("EncounterID", ""))
    checklist: List[ChecklistItem] = []
    exclusions: List[str] = []
    debug: Dict[str, Any] = {}

    # Core fields
    age = row.get("Age", None)
    drg3 = row.get("MS-DRG_3digit", "")
    principal_dx = str(row.get("Pdx", "")).strip().upper()
    adm_dt = row.get("Admission_Date", None)
    mdc_val = str(row.get("MDC", "")).strip() if "MDC" in row else ""

    # Parse DX/POA and procedures
    dx_list = extract_dx_poa(row)  # [(code, poa, role)]
    secondary_dx = [(c,p) for (c,p,role) in dx_list if role == "SECONDARY"]
    all_dx = [c for (c,_,_) in dx_list]
    procs = extract_procedures(row)  # [(code, dt, col)]

    # Sets
    SURGI2R = set(str(x).zfill(3) for x in codes.get("SURGI2R", []))
    ORPROC = set(codes.get("ORPROC", []))
    DEEPVIB = set(codes.get("DEEPVIB", []))
    PULMOID = set(codes.get("PULMOID", []))
    HITD = set(codes.get("HITD", []))
    VENACIP = set(codes.get("VENACIP", []))
    THROMP = set(codes.get("THROMP", []))
    NEURTRAD = set(codes.get("NEURTRAD", []))
    ECMOP = set(codes.get("ECMOP", []))
    MDC14PRINDX = set(codes.get("MDC14PRINDX", []))
    MDC15PRINDX = set(codes.get("MDC15PRINDX", []))

    # Denominator
    age_ok = False
    if age is not None and str(age).strip() != "":
        try:
            age_ok = float(age) >= 18
        except Exception:
            age_ok = False
    drg_ok = (drg3 in SURGI2R)
    has_orproc = any(code in ORPROC for (code,_,_) in procs)

    checklist.append(ChecklistItem("DRG in SURGI2R (surgical)", "Yes", drg3, drg_ok))
    checklist.append(ChecklistItem("Age >= 18", ">=18", age, age_ok))
    checklist.append(ChecklistItem("At least one ORPROC", "Yes", "Yes" if has_orproc else "No", has_orproc))

    denominator_met = drg_ok and age_ok and has_orproc

    # Helpers
    principal_in = lambda S: principal_dx in S
    any_dx_in = lambda S: any(c in S for c in all_dx)
    any_poa_dx_in = lambda S: any((c in S) and (p == "Y") for (c,p) in secondary_dx) or (principal_dx in S and row.get("POA1","") == "Y")

    # Find earliest ORPROC and earliest VENACIP/THROMP
    earliest_or_dt = None
    earliest_vt_dt = None
    for code, pdt, col in procs:
        if code in ORPROC and pdt is not None:
            if earliest_or_dt is None or pdt < earliest_or_dt:
                earliest_or_dt = pdt
        if (code in VENACIP or code in THROMP) and pdt is not None:
            if earliest_vt_dt is None or pdt < earliest_vt_dt:
                earliest_vt_dt = pdt

    # Exclusions (precedence)
    # 1) Principal or secondary POA=Y in DEEPVIB or PULMOID
    prin_dvtpe = principal_in(DEEPVIB) or principal_in(PULMOID)
    poaY_dvtpe = any((c in DEEPVIB or c in PULMOID) and (p == "Y") for (c,p) in secondary_dx)
    checklist.append(ChecklistItem("Principal DVT/PE (DEEPVIB/PULMOID)", "No", "Yes" if prin_dvtpe else "No", not prin_dvtpe))
    checklist.append(ChecklistItem("Secondary DVT/PE present-on-admission", "No", "Yes" if poaY_dvtpe else "No", not poaY_dvtpe))
    if prin_dvtpe: exclusions.append("Principal DVT/PE")
    if poaY_dvtpe: exclusions.append("Secondary DVT/PE POA=Y")

    # 2) Any secondary HITD
    has_hitd = any((c in HITD) for (c,_) in secondary_dx)
    checklist.append(ChecklistItem("Secondary heparin-induced thrombocytopenia (HITD)", "No", "Yes" if has_hitd else "No", not has_hitd))
    if has_hitd: exclusions.append("HITD secondary")

    # 3) Any VENACIP/THROMP on/before first ORPROC (if dated)
    vt_before_or = False
    if earliest_or_dt is not None and earliest_vt_dt is not None:
        vt_before_or = earliest_vt_dt.normalize() <= earliest_or_dt.normalize()
    checklist.append(ChecklistItem("VENACIP/THROMP on/before first ORPROC (if dated)", "No", f"vt={earliest_vt_dt}, or={earliest_or_dt}", not vt_before_or))
    if vt_before_or: exclusions.append("VENACIP/THROMP <= first ORPROC")

    # 4) Only ORPROCs are VENACIP/THROMP
    orproc_codes = [code for (code,_,_) in procs if code in ORPROC]
    vt_set = VENACIP.union(THROMP)
    only_vt_or = (len(orproc_codes) > 0 and all(code in vt_set for code in orproc_codes))
    checklist.append(ChecklistItem("Only ORPROCs are VENACIP/THROMP", "No", "Yes" if only_vt_or else "No", not only_vt_or))
    if only_vt_or: exclusions.append("Only VENACIP/THROMP ORPROC")

    # 5) First ORPROC >= 10 days after admission (if dated)
    first_or_10plus = False
    if earliest_or_dt is not None and adm_dt is not None:
        try:
            delta_days = (earliest_or_dt.normalize() - pd.to_datetime(adm_dt).normalize()).days
            first_or_10plus = (delta_days >= 10)
            debug["days_first_OR_after_admit"] = delta_days
        except Exception:
            first_or_10plus = False
    checklist.append(ChecklistItem("First ORPROC < 10 days from admission", "<10 days", debug.get("days_first_OR_after_admit","NA"), not first_or_10plus))
    if first_or_10plus: exclusions.append("First ORPROC >=10 days from admission")

    # 6) POA NEURTRAD
    poa_neurtrad = any_poa_dx_in(NEURTRAD)
    checklist.append(ChecklistItem("POA acute brain/spinal injury (NEURTRAD)", "No", "Yes" if poa_neurtrad else "No", not poa_neurtrad))
    if poa_neurtrad: exclusions.append("NEURTRAD POA")

    # 7) Any ECMOP
    has_ecmo = any(code in ECMOP for (code,_,_) in procs)
    checklist.append(ChecklistItem("ECMO procedure (ECMOP)", "No", "Yes" if has_ecmo else "No", not has_ecmo))
    if has_ecmo: exclusions.append("ECMOP")

    # 8) Obstetric/Newborn principal
    obstetric_principal = principal_in(MDC14PRINDX)
    newborn_principal = principal_in(MDC15PRINDX)
    checklist.append(ChecklistItem("Obstetric principal (MDC14PRINDX)", "No", "Yes" if obstetric_principal else "No", not obstetric_principal))
    checklist.append(ChecklistItem("Newborn principal (MDC15PRINDX)", "No", "Yes" if newborn_principal else "No", not newborn_principal))
    if obstetric_principal: exclusions.append("MDC14PRINDX")
    if newborn_principal: exclusions.append("MDC15PRINDX")

    # 9) DRG 999
    drg999 = (drg3 == "999")
    checklist.append(ChecklistItem("DRG not 999 (ungroupable)", "Not 999", drg3, not drg999))
    if drg999: exclusions.append("DRG=999")

    # 10) Missing fields
    def missing(col: str) -> bool:
        return (col not in row) or (pd.isna(row[col])) or (str(row[col]).strip() == "")
    miss_sex = missing("Sex")
    miss_age = missing("Age")
    miss_qtr = missing("DQTR") if "DQTR" in row else False
    miss_year = missing("Year") if "Year" in row else False
    miss_pdx = missing("Pdx")

    checklist.append(ChecklistItem("Missing Sex", "Present", "Missing" if miss_sex else "Present", not miss_sex))
    checklist.append(ChecklistItem("Missing Age", "Present", "Missing" if miss_age else "Present", not miss_age))
    checklist.append(ChecklistItem("Missing DQTR (if provided)", "Present/NA", "Missing" if miss_qtr else "Present/NA", not miss_qtr))
    checklist.append(ChecklistItem("Missing Year (if provided)", "Present/NA", "Missing" if miss_year else "Present/NA", not miss_year))
    checklist.append(ChecklistItem("Missing Principal DX", "Present", "Missing" if miss_pdx else "Present", not miss_pdx))

    if miss_sex: exclusions.append("Missing Sex")
    if miss_age: exclusions.append("Missing Age")
    if miss_qtr: exclusions.append("Missing DQTR")
    if miss_year: exclusions.append("Missing Year")
    if miss_pdx: exclusions.append("Missing Pdx")

    # Missing MDC if provided
    if "MDC" in row:
        miss_mdc = missing("MDC")
        checklist.append(ChecklistItem("Missing MDC (if provided)", "Present/NA", "Missing" if miss_mdc else "Present/NA", not miss_mdc))
        if miss_mdc: exclusions.append("Missing MDC")

    # Early exits
    if not denominator_met:
        return PSIEvalResult(
            encounter_id=enc_id, psi="PSI_12",
            result="EXCLUSION", denominator_met=False, numerator_met=False,
            exclusions_applied=exclusions, rationale_short="Denominator not met",
            checklist=checklist, debug={"drg3": drg3, "age": age}
        )
    if exclusions:
        return PSIEvalResult(
            encounter_id=enc_id, psi="PSI_12",
            result="EXCLUSION", denominator_met=True, numerator_met=False,
            exclusions_applied=exclusions, rationale_short="Denominator exclusion(s) applied",
            checklist=checklist, debug={"exclusions": exclusions, "earliest_or": str(earliest_or_dt), "earliest_vt": str(earliest_vt_dt)}
        )

    # Numerator: secondary DEEPVIB or PULMOID, not POA=Y
    numerator_hit = any(((c in DEEPVIB) or (c in PULMOID)) and (p != "Y") for (c,p) in secondary_dx)
    checklist.append(ChecklistItem("Numerator: secondary DVT/PE not POA=Y", "Yes", "Yes" if numerator_hit else "No", numerator_hit))

    result = "INCLUSION" if numerator_hit else "EXCLUSION"
    rationale = "Perioperative PE/DVT (secondary, not POA)" if numerator_hit else "No qualifying perioperative PE/DVT"

    return PSIEvalResult(
        encounter_id=enc_id, psi="PSI_12",
        result=result, denominator_met=True, numerator_met=numerator_hit,
        exclusions_applied=[], rationale_short=rationale,
        checklist=checklist, debug={"earliest_or": str(earliest_or_dt), "earliest_vt": str(earliest_vt_dt)}
    )

@register_psi(13)
def evaluate_psi13(row: pd.Series, codes: Dict[str, List[str]]) -> PSIEvalResult:
    """
    PSI-13 Postoperative Sepsis Rate (2025).
    Denominator: elective surgical (SURGI2R) AND Age>=18 AND >=1 ORPROC AND ATYPE=3.
    Numerator: secondary SEPTI2D (sepsis), not POA=Y.
    Exclusions (precedence):
      - Principal SEPTI2D
      - Secondary SEPTI2D POA=Y
      - Principal INFECID
      - Secondary INFECID POA=Y
      - First ORPROC >= 10 days after admission (if dated)
      - Obstetric/Newborn principal (MDC14PRINDX/MDC15PRINDX)
      - DRG=999
      - Missing Sex/Age/DQTR/Year/Pdx (and MDC if provided)
    """
    enc_id = str(row.get("EncounterID", ""))
    checklist: List[ChecklistItem] = []
    exclusions: List[str] = []
    debug: Dict[str, Any] = {}

    # Core fields
    age = row.get("Age", None)
    drg3 = row.get("MS-DRG_3digit", "")
    atype = str(row.get("ATYPE", "")).strip()
    principal_dx = str(row.get("Pdx", "")).strip().upper()
    adm_dt = row.get("Admission_Date", None)

    # Parse DX/POA and procedures
    dx_list = extract_dx_poa(row)  # [(code, poa, role)]
    secondary_dx = [(c,p) for (c,p,role) in dx_list if role == "SECONDARY"]
    all_dx = [c for (c,_,_) in dx_list]
    procs = extract_procedures(row)  # [(code, dt, col)]

    # Sets
    SURGI2R = set(str(x).zfill(3) for x in codes.get("SURGI2R", []))
    ORPROC = set(codes.get("ORPROC", []))
    SEPTI2D = set(codes.get("SEPTI2D", []))
    INFECID = set(codes.get("INFECID", []))
    MDC14PRINDX = set(codes.get("MDC14PRINDX", []))
    MDC15PRINDX = set(codes.get("MDC15PRINDX", []))

    # Denominator
    age_ok = False
    if age is not None and str(age).strip() != "":
        try:
            age_ok = float(age) >= 18
        except Exception:
            age_ok = False
    drg_ok = (drg3 in SURGI2R)
    has_orproc = any(code in ORPROC for (code,_,_) in procs)
    elective_ok = (atype == "3")

    checklist.append(ChecklistItem("DRG in SURGI2R (surgical)", "Yes", drg3, drg_ok))
    checklist.append(ChecklistItem("Age >= 18", ">=18", age, age_ok))
    checklist.append(ChecklistItem("At least one ORPROC", "Yes", "Yes" if has_orproc else "No", has_orproc))
    checklist.append(ChecklistItem("Admission type elective (ATYPE=3)", "Yes", atype, elective_ok))

    denominator_met = drg_ok and age_ok and has_orproc and elective_ok

    # Helpers
    principal_in = lambda S: principal_dx in S
    any_dx_in = lambda S: any(c in S for c in all_dx)

    # Earliest OR
    earliest_or_dt = None
    for code, pdt, col in procs:
        if code in ORPROC and pdt is not None:
            if earliest_or_dt is None or pdt < earliest_or_dt:
                earliest_or_dt = pdt

    # Exclusions (precedence)
    # 1) Principal or secondary POA=Y in SEPTI2D
    excl_prin_sepsis = principal_in(SEPTI2D)
    excl_sec_sepsis_poaY = any((c in SEPTI2D) and (p == "Y") for (c,p) in secondary_dx)
    checklist.append(ChecklistItem("Principal sepsis (SEPTI2D)", "No", "Yes" if excl_prin_sepsis else "No", not excl_prin_sepsis))
    checklist.append(ChecklistItem("Secondary sepsis present-on-admission (SEPTI2D)", "No", "Yes" if excl_sec_sepsis_poaY else "No", not excl_sec_sepsis_poaY))
    if excl_prin_sepsis: exclusions.append("Principal SEPTI2D")
    if excl_sec_sepsis_poaY: exclusions.append("Secondary SEPTI2D POA=Y")

    # 2) Principal or secondary POA=Y in INFECID
    excl_prin_infec = principal_in(INFECID)
    excl_sec_infec_poaY = any((c in INFECID) and (p == "Y") for (c,p) in secondary_dx)
    checklist.append(ChecklistItem("Principal infection (INFECID)", "No", "Yes" if excl_prin_infec else "No", not excl_prin_infec))
    checklist.append(ChecklistItem("Secondary infection present-on-admission (INFECID)", "No", "Yes" if excl_sec_infec_poaY else "No", not excl_sec_infec_poaY))
    if excl_prin_infec: exclusions.append("Principal INFECID")
    if excl_sec_infec_poaY: exclusions.append("Secondary INFECID POA=Y")

    # 3) First OR >= 10 days after admission (if dated)
    first_or_10plus = False
    if earliest_or_dt is not None and adm_dt is not None:
        try:
            delta_days = (earliest_or_dt.normalize() - pd.to_datetime(adm_dt).normalize()).days
            first_or_10plus = (delta_days >= 10)
            debug["days_first_OR_after_admit"] = delta_days
        except Exception:
            first_or_10plus = False
    checklist.append(ChecklistItem("First ORPROC < 10 days from admission", "<10 days", debug.get("days_first_OR_after_admit","NA"), not first_or_10plus))
    if first_or_10plus: exclusions.append("First ORPROC >=10 days from admission")

    # 4) Obstetric/Newborn principal
    obstetric_principal = principal_in(MDC14PRINDX)
    newborn_principal = principal_in(MDC15PRINDX)
    checklist.append(ChecklistItem("Obstetric principal (MDC14PRINDX)", "No", "Yes" if obstetric_principal else "No", not obstetric_principal))
    checklist.append(ChecklistItem("Newborn principal (MDC15PRINDX)", "No", "Yes" if newborn_principal else "No", not newborn_principal))
    if obstetric_principal: exclusions.append("MDC14PRINDX")
    if newborn_principal: exclusions.append("MDC15PRINDX")

    # 5) DRG 999
    drg999 = (drg3 == "999")
    checklist.append(ChecklistItem("DRG not 999 (ungroupable)", "Not 999", drg3, not drg999))
    if drg999: exclusions.append("DRG=999")

    # 6) Missing fields
    def missing(col: str) -> bool:
        return (col not in row) or (pd.isna(row[col])) or (str(row[col]).strip() == "")
    miss_sex = missing("Sex")
    miss_age = missing("Age")
    miss_qtr = missing("DQTR") if "DQTR" in row else False
    miss_year = missing("Year") if "Year" in row else False
    miss_pdx = missing("Pdx")

    checklist.append(ChecklistItem("Missing Sex", "Present", "Missing" if miss_sex else "Present", not miss_sex))
    checklist.append(ChecklistItem("Missing Age", "Present", "Missing" if miss_age else "Present", not miss_age))
    checklist.append(ChecklistItem("Missing DQTR (if provided)", "Present/NA", "Missing" if miss_qtr else "Present/NA", not miss_qtr))
    checklist.append(ChecklistItem("Missing Year (if provided)", "Present/NA", "Missing" if miss_year else "Present/NA", not miss_year))
    checklist.append(ChecklistItem("Missing Principal DX", "Present", "Missing" if miss_pdx else "Present", not miss_pdx))

    if miss_sex: exclusions.append("Missing Sex")
    if miss_age: exclusions.append("Missing Age")
    if miss_qtr: exclusions.append("Missing DQTR")
    if miss_year: exclusions.append("Missing Year")
    if miss_pdx: exclusions.append("Missing Pdx")

    # Missing MDC if provided
    if "MDC" in row:
        miss_mdc = missing("MDC")
        checklist.append(ChecklistItem("Missing MDC (if provided)", "Present/NA", "Missing" if miss_mdc else "Present/NA", not miss_mdc))
        if miss_mdc: exclusions.append("Missing MDC")

    # Early exits
    if not denominator_met:
        return PSIEvalResult(
            encounter_id=enc_id, psi="PSI_13",
            result="EXCLUSION", denominator_met=False, numerator_met=False,
            exclusions_applied=exclusions, rationale_short="Denominator not met",
            checklist=checklist, debug={"drg3": drg3, "age": age, "atype": atype}
        )
    if exclusions:
        return PSIEvalResult(
            encounter_id=enc_id, psi="PSI_13",
            result="EXCLUSION", denominator_met=True, numerator_met=False,
            exclusions_applied=exclusions, rationale_short="Denominator exclusion(s) applied",
            checklist=checklist, debug={"exclusions": exclusions, "days_first_OR_after_admit": debug.get("days_first_OR_after_admit", None)}
        )

    # Numerator: secondary SEPTI2D not POA=Y
    numerator_hit = any((c in SEPTI2D) and (p != "Y") for (c,p) in secondary_dx)
    checklist.append(ChecklistItem("Numerator: secondary sepsis (SEPTI2D) not POA=Y", "Yes", "Yes" if numerator_hit else "No", numerator_hit))

    result = "INCLUSION" if numerator_hit else "EXCLUSION"
    rationale = "Postoperative sepsis coded (secondary, not POA)" if numerator_hit else "No qualifying postoperative sepsis"

    return PSIEvalResult(
        encounter_id=enc_id, psi="PSI_13",
        result=result, denominator_met=True, numerator_met=numerator_hit,
        exclusions_applied=[], rationale_short=rationale,
        checklist=checklist, debug={"days_first_OR_after_admit": debug.get("days_first_OR_after_admit", None)}
    )

@register_psi(14)
def evaluate_psi14(row: pd.Series, codes: Dict[str, List[str]]) -> PSIEvalResult:
    """
    PSI-14 Postoperative Wound Dehiscence Rate (2025).
    Denominator: Age >= 18 AND (any ABDOMIPOPEN OR ABDOMIPOTHER).
    Strata (mutually exclusive; priority: OPEN > NON_OPEN):
      - OPEN: at least one ABDOMIPOPEN (index surgery set = ABDOMIPOPEN)
      - NON_OPEN: at least one ABDOMIPOTHER (index surgery set = ABDOMIPOTHER)
    Exclusions:
      - Last repair (RECLOIP) occurs on/before first index surgery date
      - Principal or POA=Y ABWALLCD
      - LOS < 2 days
      - Obstetric/Newborn principal (MDC14PRINDX/MDC15PRINDX)
      - DRG=999
      - Missing Sex/Age/DQTR/Year/Pdx (and MDC if provided)
    Numerator: any RECLOIP AND any ABWALLCD (secondary, not POA=Y).
    Notes: When dates are missing we keep the case unless an exclusion requires dates to prove; comparisons use normalized date.
    """
    enc_id = str(row.get("EncounterID", ""))
    checklist: List[ChecklistItem] = []
    exclusions: List[str] = []
    debug: Dict[str, Any] = {}

    # Core fields
    age = row.get("Age", None)
    drg3 = row.get("MS-DRG_3digit", "")
    principal_dx = str(row.get("Pdx", "")).strip().upper()
    los = row.get("Length_of_stay", None)

    # Parse DX/POA and procedures
    dx_list = extract_dx_poa(row)  # [(code, poa, role)]
    secondary_dx = [(c,p) for (c,p,role) in dx_list if role == "SECONDARY"]
    all_dx = [c for (c,_,_) in dx_list]
    procs = extract_procedures(row)  # [(code, dt, col)]

    # Sets
    ABDOMIPOPEN = set(codes.get("ABDOMIPOPEN", []))
    ABDOMIPOTHER = set(codes.get("ABDOMIPOTHER", []))
    RECLOIP = set(codes.get("RECLOIP", []))
    ABWALLCD = set(codes.get("ABWALLCD", []))
    MDC14PRINDX = set(codes.get("MDC14PRINDX", []))
    MDC15PRINDX = set(codes.get("MDC15PRINDX", []))

    # Age
    age_ok = False
    if age is not None and str(age).strip() != "":
        try:
            age_ok = float(age) >= 18
        except Exception:
            age_ok = False
    checklist.append(ChecklistItem("Age >= 18", ">=18", age, age_ok))

    # Determine stratum by presence of index surgery
    has_open = any(code in ABDOMIPOPEN for (code,_,_) in procs)
    has_other = any(code in ABDOMIPOTHER for (code,_,_) in procs)
    stratum = None
    index_set = None
    if has_open:
        stratum = "OPEN"
        index_set = ABDOMIPOPEN
    elif has_other:
        stratum = "NON_OPEN"
        index_set = ABDOMIPOTHER

    checklist.append(ChecklistItem("Abdominopelvic surgery present", "Yes", f"OPEN={has_open}, OTHER={has_other}", bool(stratum)))

    denominator_met = age_ok and bool(stratum)

    # Exclusions (precedence)
    # Principal or POA=Y ABWALLCD
    prin_abw = principal_dx in ABWALLCD
    poa_abw = any((c in ABWALLCD) and (p == "Y") for (c,p) in secondary_dx)
    checklist.append(ChecklistItem("Principal ABWALLCD", "No", "Yes" if prin_abw else "No", not prin_abw))
    checklist.append(ChecklistItem("Secondary ABWALLCD present-on-admission", "No", "Yes" if poa_abw else "No", not poa_abw))
    if prin_abw: exclusions.append("Principal ABWALLCD")
    if poa_abw: exclusions.append("Secondary ABWALLCD POA=Y")

    # LOS < 2
    los_excl = False
    if los is not None and str(los).strip() != "":
        try:
            los_excl = int(los) < 2
        except Exception:
            los_excl = False
    checklist.append(ChecklistItem("Length of stay >= 2 days", ">=2", los, not los_excl))
    if los_excl: exclusions.append("LOS<2")

    # Obstetric/Newborn principal
    obstetric_principal = principal_dx in MDC14PRINDX
    newborn_principal = principal_dx in MDC15PRINDX
    checklist.append(ChecklistItem("Obstetric principal (MDC14PRINDX)", "No", "Yes" if obstetric_principal else "No", not obstetric_principal))
    checklist.append(ChecklistItem("Newborn principal (MDC15PRINDX)", "No", "Yes" if newborn_principal else "No", not newborn_principal))
    if obstetric_principal: exclusions.append("MDC14PRINDX")
    if newborn_principal: exclusions.append("MDC15PRINDX")

    # DRG 999
    drg999 = (drg3 == "999")
    checklist.append(ChecklistItem("DRG not 999 (ungroupable)", "Not 999", drg3, not drg999))
    if drg999: exclusions.append("DRG=999")

    # Missing fields
    def missing(col: str) -> bool:
        return (col not in row) or (pd.isna(row[col])) or (str(row[col]).strip() == "")
    miss_sex = missing("Sex")
    miss_age = missing("Age")
    miss_qtr = missing("DQTR") if "DQTR" in row else False
    miss_year = missing("Year") if "Year" in row else False
    miss_pdx = missing("Pdx")

    checklist.append(ChecklistItem("Missing Sex", "Present", "Missing" if miss_sex else "Present", not miss_sex))
    checklist.append(ChecklistItem("Missing Age", "Present", "Missing" if miss_age else "Present", not miss_age))
    checklist.append(ChecklistItem("Missing DQTR (if provided)", "Present/NA", "Missing" if miss_qtr else "Present/NA", not miss_qtr))
    checklist.append(ChecklistItem("Missing Year (if provided)", "Present/NA", "Missing" if miss_year else "Present/NA", not miss_year))
    checklist.append(ChecklistItem("Missing Principal DX", "Present", "Missing" if miss_pdx else "Present", not miss_pdx))

    if miss_sex: exclusions.append("Missing Sex")
    if miss_age: exclusions.append("Missing Age")
    if miss_qtr: exclusions.append("Missing DQTR")
    if miss_year: exclusions.append("Missing Year")
    if miss_pdx: exclusions.append("Missing Pdx")

    # Missing MDC if provided
    if "MDC" in row:
        miss_mdc = missing("MDC")
        checklist.append(ChecklistItem("Missing MDC (if provided)", "Present/NA", "Missing" if miss_mdc else "Present/NA", not miss_mdc))
        if miss_mdc: exclusions.append("Missing MDC")

    # If denominator not met or exclusions already triggered, early exit
    if not denominator_met:
        return PSIEvalResult(
            encounter_id=enc_id, psi="PSI_14",
            result="EXCLUSION", denominator_met=False, numerator_met=False,
            exclusions_applied=exclusions, rationale_short="Denominator not met",
            checklist=checklist, debug={"stratum": stratum}
        )
    if exclusions:
        return PSIEvalResult(
            encounter_id=enc_id, psi="PSI_14",
            result="EXCLUSION", denominator_met=True, numerator_met=False,
            exclusions_applied=exclusions, rationale_short="Denominator exclusion(s) applied",
            checklist=checklist, debug={"stratum": stratum}
        )

    # Timing exclusion: Last RECLOIP on/before first index surgery date
    # Compute earliest index surgery date (from chosen stratum) and last repair date
    earliest_index_dt = None
    last_repair_dt = None
    for code, pdt, col in procs:
        if code in index_set and pdt is not None:
            if earliest_index_dt is None or pdt < earliest_index_dt:
                earliest_index_dt = pdt
        if code in RECLOIP and pdt is not None:
            if last_repair_dt is None or pdt > last_repair_dt:
                last_repair_dt = pdt

    timing_excl = False
    if earliest_index_dt is not None and last_repair_dt is not None:
        timing_excl = last_repair_dt.normalize() <= earliest_index_dt.normalize()
    checklist.append(ChecklistItem("Last repair occurs AFTER index surgery date", "After", f"repair={last_repair_dt}, index={earliest_index_dt}", not timing_excl))
    if timing_excl:
        return PSIEvalResult(
            encounter_id=enc_id, psi="PSI_14",
            result="EXCLUSION", denominator_met=True, numerator_met=False,
            exclusions_applied=["Repair on/before index date"], rationale_short="Timing exclusion",
            checklist=checklist, debug={"stratum": stratum, "earliest_index": str(earliest_index_dt), "last_repair": str(last_repair_dt)}
        )

    # Numerator: any RECLOIP and any ABWALLCD (secondary not POA=Y)
    has_repair = any(code in RECLOIP for (code,_,_) in procs)
    has_abw_dx = any((c in ABWALLCD) and (p != "Y") for (c,p) in secondary_dx)

    checklist.append(ChecklistItem("Numerator: any RECLOIP", "Yes", "Yes" if has_repair else "No", has_repair))
    checklist.append(ChecklistItem("Numerator: secondary ABWALLCD not POA=Y", "Yes", "Yes" if has_abw_dx else "No", has_abw_dx))

    numerator_hit = has_repair and has_abw_dx

    result = "INCLUSION" if numerator_hit else "EXCLUSION"
    rationale = "Postop wound dehiscence (repair + disruption DX not POA)" if numerator_hit else "No qualifying repair+disruption diagnosis"

    return PSIEvalResult(
        encounter_id=enc_id, psi="PSI_14",
        result=result, denominator_met=True, numerator_met=numerator_hit,
        exclusions_applied=[], rationale_short=rationale,
        checklist=checklist, debug={"stratum": stratum, "earliest_index": str(earliest_index_dt), "last_repair": str(last_repair_dt)}
    )

@register_psi(15)
def evaluate_psi15(row: pd.Series, codes: Dict[str, List[str]]) -> PSIEvalResult:
    """
    PSI-15 Abdominopelvic Accidental Puncture or Laceration Rate (2025).
    Denominator: SURGI2R or MEDIC2R AND Age>=18 AND any ABDOMI15P (index procedure).
    Numerator (organ-matched; all must hold):
      - Secondary DX in organ-specific set (SPLEEN15D/ADRENAL15D/VESSEL15D/DIAPHR15D/GI15D/GU15D) with POA != Y
      - Subsequent evaluation/treatment procedure in corresponding set (SPLEEN15P/.../GU15P)
      - Related procedure occurs 1–30 days after the earliest index ABDOMI15P
      - No principal or POA=Y accidental puncture/laceration in that organ
    Exclusions:
      - Principal or secondary POA=Y organ-specific ~15D (any organ)
      - Missing index ABDOMI15P date
      - Missing all related organ-treatment procedure dates
      - Obstetric/Newborn principal (MDC14PRINDX/MDC15PRINDX)
      - DRG=999
      - Missing Sex/Age/DQTR/Year/Pdx (and MDC if provided)
    """
    enc_id = str(row.get("EncounterID", ""))
    checklist: List[ChecklistItem] = []
    exclusions: List[str] = []
    debug: Dict[str, Any] = {}

    # Core fields
    age = row.get("Age", None)
    drg3 = row.get("MS-DRG_3digit", "")
    principal_dx = str(row.get("Pdx", "")).strip().upper()

    # Parse DX/POA and procedures
    dx_list = extract_dx_poa(row)  # [(code, poa, role)]
    secondary_dx = [(c,p) for (c,p,role) in dx_list if role == "SECONDARY"]
    all_dx = [c for (c,_,_) in dx_list]
    procs = extract_procedures(row)  # [(code, dt, col)]

    # Sets
    SURGI2R = set(str(x).zfill(3) for x in codes.get("SURGI2R", []))
    MEDIC2R = set(str(x).zfill(3) for x in codes.get("MEDIC2R", []))
    ABDOMI15P = set(codes.get("ABDOMI15P", []))

    organ_D_sets = {
        "SPLEEN": set(codes.get("SPLEEN15D", [])),
        "ADRENAL": set(codes.get("ADRENAL15D", [])),
        "VESSEL": set(codes.get("VESSEL15D", [])),
        "DIAPHR": set(codes.get("DIAPHR15D", [])),
        "GI": set(codes.get("GI15D", [])),
        "GU": set(codes.get("GU15D", [])),
    }
    organ_P_sets = {
        "SPLEEN": set(codes.get("SPLEEN15P", [])),
        "ADRENAL": set(codes.get("ADRENAL15P", [])),
        "VESSEL": set(codes.get("VESSEL15P", [])),
        "DIAPHR": set(codes.get("DIAPHR15P", [])),
        "GI": set(codes.get("GI15P", [])),
        "GU": set(codes.get("GU15P", [])),
    }
    MDC14PRINDX = set(codes.get("MDC14PRINDX", []))
    MDC15PRINDX = set(codes.get("MDC15PRINDX", []))

    # Denominator
    age_ok = False
    if age is not None and str(age).strip() != "":
        try:
            age_ok = float(age) >= 18
        except Exception:
            age_ok = False
    drg_ok = (drg3 in SURGI2R) or (drg3 in MEDIC2R)
    has_index_proc = any(code in ABDOMI15P for (code,_,_) in procs)

    checklist.append(ChecklistItem("DRG in SURGI2R or MEDIC2R", "Yes", drg3, drg_ok))
    checklist.append(ChecklistItem("Age >= 18", ">=18", age, age_ok))
    checklist.append(ChecklistItem("Any abdominopelvic procedure (ABDOMI15P)", "Yes", "Yes" if has_index_proc else "No", has_index_proc))

    denominator_met = drg_ok and age_ok and has_index_proc

    # Early exit if denominator not met
    if not denominator_met:
        return PSIEvalResult(
            encounter_id=enc_id, psi="PSI_15",
            result="EXCLUSION", denominator_met=False, numerator_met=False,
            exclusions_applied=[], rationale_short="Denominator not met",
            checklist=checklist, debug={}
        )

    # Principal/POA exclusions across any organ
    any_principal_15D = any(principal_dx in S for S in organ_D_sets.values())
    any_poa_15D = any((c in S) and (p == "Y") for (c,p) in secondary_dx for S in organ_D_sets.values())

    checklist.append(ChecklistItem("Principal accidental puncture/laceration (any 15D)", "No", "Yes" if any_principal_15D else "No", not any_principal_15D))
    checklist.append(ChecklistItem("Secondary 15D present-on-admission (any organ)", "No", "Yes" if any_poa_15D else "No", not any_poa_15D))
    if any_principal_15D: exclusions.append("Principal ~15D")
    if any_poa_15D: exclusions.append("Secondary ~15D POA=Y")

    # Obstetric/Newborn
    obstetric_principal = principal_dx in MDC14PRINDX
    newborn_principal = principal_dx in MDC15PRINDX
    checklist.append(ChecklistItem("Obstetric principal (MDC14PRINDX)", "No", "Yes" if obstetric_principal else "No", not obstetric_principal))
    checklist.append(ChecklistItem("Newborn principal (MDC15PRINDX)", "No", "Yes" if newborn_principal else "No", not newborn_principal))
    if obstetric_principal: exclusions.append("MDC14PRINDX")
    if newborn_principal: exclusions.append("MDC15PRINDX")

    # DRG 999
    drg999 = (drg3 == "999")
    checklist.append(ChecklistItem("DRG not 999 (ungroupable)", "Not 999", drg3, not drg999))
    if drg999: exclusions.append("DRG=999")

    # Missing fields
    def missing(col: str) -> bool:
        return (col not in row) or (pd.isna(row[col])) or (str(row[col]).strip() == "")
    miss_sex = missing("Sex")
    miss_age = missing("Age")
    miss_qtr = missing("DQTR") if "DQTR" in row else False
    miss_year = missing("Year") if "Year" in row else False
    miss_pdx = missing("Pdx")

    checklist.append(ChecklistItem("Missing Sex", "Present", "Missing" if miss_sex else "Present", not miss_sex))
    checklist.append(ChecklistItem("Missing Age", "Present", "Missing" if miss_age else "Present", not miss_age))
    checklist.append(ChecklistItem("Missing DQTR (if provided)", "Present/NA", "Missing" if miss_qtr else "Present/NA", not miss_qtr))
    checklist.append(ChecklistItem("Missing Year (if provided)", "Present/NA", "Missing" if miss_year else "Present/NA", not miss_year))
    checklist.append(ChecklistItem("Missing Principal DX", "Present", "Missing" if miss_pdx else "Present", not miss_pdx))

    if miss_sex: exclusions.append("Missing Sex")
    if miss_age: exclusions.append("Missing Age")
    if miss_qtr: exclusions.append("Missing DQTR")
    if miss_year: exclusions.append("Missing Year")
    if miss_pdx: exclusions.append("Missing Pdx")

    # Missing MDC if provided
    if "MDC" in row:
        miss_mdc = missing("MDC")
        checklist.append(ChecklistItem("Missing MDC (if provided)", "Present/NA", "Missing" if miss_mdc else "Present/NA", not miss_mdc))
        if miss_mdc: exclusions.append("Missing MDC")

    if exclusions:
        return PSIEvalResult(
            encounter_id=enc_id, psi="PSI_15",
            result="EXCLUSION", denominator_met=True, numerator_met=False,
            exclusions_applied=exclusions, rationale_short="Denominator exclusion(s) applied",
            checklist=checklist, debug={}
        )

    # Find earliest index date across ABDOMI15P
    earliest_index_dt = None
    for code, pdt, col in procs:
        if code in ABDOMI15P and pdt is not None:
            if earliest_index_dt is None or pdt < earliest_index_dt:
                earliest_index_dt = pdt

    if earliest_index_dt is None:
        checklist.append(ChecklistItem("Index ABDOMI15P has a date", "Yes", "Missing", False))
        return PSIEvalResult(
            encounter_id=enc_id, psi="PSI_15",
            result="EXCLUSION", denominator_met=True, numerator_met=False,
            exclusions_applied=["Missing index ABDOMI15P date"], rationale_short="Missing dates",
            checklist=checklist, debug={}
        )
    else:
        checklist.append(ChecklistItem("Index ABDOMI15P has a date", "Yes", str(earliest_index_dt), True))

    # Evaluate organ-specific numerator (and date availability for related procedures)
    numerator_hit = False
    organ_matched = None
    window_days_used = None

    for organ, Dset in organ_D_sets.items():
        # secondary DX not POA=Y in this organ
        has_sec_dx = any((c in Dset) and (p != "Y") for (c,p) in secondary_dx)
        # exclude this organ if principal/POA in same organ (already handled globally, but ensure organ-specific)
        prin_in_organ = (principal_dx in Dset)
        poa_in_organ = any((c in Dset) and (p == "Y") for (c,p) in secondary_dx)

        if not has_sec_dx or prin_in_organ or poa_in_organ:
            continue

        # find any related procedure date in matching P set
        Pset = organ_P_sets[organ]
        # record the earliest related procedure date AFTER index
        earliest_rel_dt = None
        for code, pdt, col in procs:
            if code in Pset and pdt is not None:
                # compute day difference relative to index
                d = (pdt.normalize() - earliest_index_dt.normalize()).days
                if d >= 0:  # consider post-index only for evaluation
                    if earliest_rel_dt is None or pdt < earliest_rel_dt:
                        earliest_rel_dt = pdt
                        window_days_used = d

        if earliest_rel_dt is None:
            # missing all related procedure dates for this organ
            continue

        # Must be 1–30 days after index
        if 1 <= window_days_used <= 30:
            numerator_hit = True
            organ_matched = organ
            break

    if organ_matched is None and not numerator_hit:
        # If no organ had any related procedure date, check if indeed all related procedure dates missing
        # for all organs that had a qualifying secondary DX.
        had_any_candidate = any(any((c in Dset) and (p != "Y") for (c,p) in secondary_dx) for Dset in organ_D_sets.values())
        if had_any_candidate:
            checklist.append(ChecklistItem("Related organ procedure has a date", "Yes", "Missing/No match", False))
        else:
            checklist.append(ChecklistItem("Has qualifying secondary organ DX", "Yes", "No", False))

    checklist.append(ChecklistItem("Numerator: organ-matched within 1–30 days", "Yes", f"{organ_matched} ({window_days_used}d)" if numerator_hit else "No", numerator_hit))

    result = "INCLUSION" if numerator_hit else "EXCLUSION"
    rationale = ("Accidental puncture/laceration with subsequent matched treatment 1–30d" if numerator_hit
                 else "No qualifying organ-matched DX+procedure in 1–30d window")

    return PSIEvalResult(
        encounter_id=enc_id, psi="PSI_15",
        result=result, denominator_met=True, numerator_met=numerator_hit,
        exclusions_applied=[], rationale_short=rationale,
        checklist=checklist, debug={"earliest_index": str(earliest_index_dt), "organ": organ_matched, "days_after_index": window_days_used}
    )


@register_psi(17)
def evaluate_psi17(row: pd.Series, codes: Dict[str, List[str]]) -> PSIEvalResult:
    """
    PSI-17 Birth Trauma Rate – Injury to Neonate (2025) — reworked per user definition.

    Cohort definitions:
      Neonate: (age in days 0–28) OR (age_days missing AND AGE(years)=0 AND (ATYPE=4 OR any LIVEBND))
      Newborn: neonate AND ((any LIVEBND AND age_days in {0, missing}) OR
                            (ATYPE=4 AND age_days=0 AND no LIVEB2D) OR
                            (ATYPE=4 AND POINTOFORIGINUB04='5'))
      Normal newborn: newborn AND MS-DRG == 795
      Outborn: neonate AND NOT newborn AND (age_days <2 and not missing OR
                                            (ATYPE=4 and age_days missing) OR
                                            (ATYPE=4 and POINTOFORIGINUB04='6'))

    Denominator: **Neonates** (per above).
    Numerator: any-listed DX in BIRTHID.
    Exclusions: PRETEID, OSTEOID, principal MDC14PRINDX, DRG=999, missing required fields (and MDC if provided).
    Debug: emits neonate/newborn/normal_newborn/outborn flags.
    """
    enc_id = str(row.get("EncounterID", ""))
    checklist: List[ChecklistItem] = []
    exclusions: List[str] = []
    debug: Dict[str, Any] = {}

    # Field accessors & normalization
    def sval(x): 
        return ("" if x is None or (isinstance(x, float) and pd.isna(x)) else str(x).strip())

    age_years = row.get("Age", row.get("AGE", None))
    atype = sval(row.get("ATYPE", ""))
    point_origin = sval(row.get("POINTOFORIGINUB04", ""))
    msdrg3 = sval(row.get("MS-DRG_3digit", ""))
    msdrg_full = sval(row.get("MS-DRG", ""))

    # Try to get age in days if present (common names), else None
    age_days = None
    for cand in ["Age_in_days", "AGE_DAYS", "AGE_IN_DAYS", "AgeDays", "AgeDaysAtAdmit", "AGE_DAYS_AT_ADMISSION"]:
        if cand in row and str(row[cand]).strip() != "" and not pd.isna(row[cand]):
            try:
                age_days = int(float(row[cand]))
                break
            except Exception:
                pass

    # Parse DX/POA lists
    dx_list = extract_dx_poa(row)   # [(code, poa, role)]
    all_dx = [c for (c,_,_) in dx_list]
    secondary_dx = [(c,p) for (c,p,role) in dx_list if role == "SECONDARY"]

    # Code sets
    BIRTHID = set(codes.get("BIRTHID", []))
    PRETEID = set(codes.get("PRETEID", []))
    OSTEOID = set(codes.get("OSTEOID", []))
    MDC14PRINDX = set(codes.get("MDC14PRINDX", []))
    LIVEBND = set(codes.get("LIVEBND", []))
    LIVEB2D = set(codes.get("LIVEB2D", []))

    principal_dx = sval(row.get("Pdx", "")).upper()
    drg999 = (sval(row.get("MS-DRG_3digit","")) == "999" or sval(row.get("MS-DRG","")) == "999")

    # Helper: any-listed LIVEBND / LIVEB2D
    has_livebnd = any(c in LIVEBND for c in all_dx)
    has_liveb2d = any(c in LIVEB2D for c in all_dx)

    # Neonate definition
    neonate_by_days = (age_days is not None and 0 <= age_days <= 28)
    neonate_by_missingdays = (age_days is None and (age_years == 0 or sval(age_years) == "0") and (atype == "4" or has_livebnd))
    is_neonate = neonate_by_days or neonate_by_missingdays

    # Newborn definition
    newborn_case1 = (is_neonate and has_livebnd and (age_days in (0, None)))
    newborn_case2 = (is_neonate and atype == "4" and (age_days == 0) and (not has_liveb2d))
    newborn_case3 = (is_neonate and atype == "4" and point_origin == "5")
    is_newborn = newborn_case1 or newborn_case2 or newborn_case3

    # Normal newborn
    # Normalize MS-DRG to integer 3 digits if possible
    normal_newborn = False
    for val in (msdrg3, msdrg_full):
        try:
            if sval(val) != "":
                if int(float(val)) == 795:
                    normal_newborn = True
                    break
        except Exception:
            continue

    # Outborn definition
    outborn_case1 = (is_neonate and age_days is not None and age_days < 2)
    outborn_case2 = (is_neonate and atype == "4" and age_days is None)
    outborn_case3 = (is_neonate and atype == "4" and point_origin == "6")
    is_outborn = (not is_newborn) and (outborn_case1 or outborn_case2 or outborn_case3)

    # Record in checklist
    checklist.append(ChecklistItem("Neonate cohort per definition", "Yes", f"days={age_days}, AGE={age_years}, ATYPE={atype}, LIVEBND={has_livebnd}", is_neonate))
    checklist.append(ChecklistItem("Newborn (subset of neonates)", "Info", f"{is_newborn}", True))
    checklist.append(ChecklistItem("Normal newborn (MS-DRG=795)", "Info", f"{normal_newborn}", True))
    checklist.append(ChecklistItem("Outborn (subset of neonates not newborn)", "Info", f"{is_outborn}", True))

    # Required field checks
    def missing(col: str) -> bool:
        return (col not in row) or (pd.isna(row[col])) or (str(row[col]).strip() == "")
    miss_sex = missing("Sex") and missing("SEX")
    miss_age = (age_years is None) and (age_days is None)
    miss_qtr = missing("DQTR") and missing("DQTR")
    miss_year = missing("Year") and missing("YEAR")
    miss_pdx = missing("Pdx")

    checklist.append(ChecklistItem("Missing Sex", "Present", "Missing" if miss_sex else "Present", not miss_sex))
    checklist.append(ChecklistItem("Missing Age (years or days)", "Present", "Missing" if miss_age else "Present", not miss_age))
    checklist.append(ChecklistItem("Missing DQTR (if provided)", "Present/NA", "Missing" if miss_qtr else "Present/NA", not miss_qtr))
    checklist.append(ChecklistItem("Missing Year (if provided)", "Present/NA", "Missing" if miss_year else "Present/NA", not miss_year))
    checklist.append(ChecklistItem("Missing Principal DX", "Present", "Missing" if miss_pdx else "Present", not miss_pdx))

    if drg999: 
        exclusions.append("DRG=999")
    if miss_sex: exclusions.append("Missing Sex")
    if miss_age: exclusions.append("Missing Age")
    if miss_qtr: exclusions.append("Missing DQTR")
    if miss_year: exclusions.append("Missing Year")
    if miss_pdx: exclusions.append("Missing Pdx")

    # Denominator: must be neonate
    if not is_neonate:
        return PSIEvalResult(
            encounter_id=enc_id, psi="PSI_17",
            result="EXCLUSION", denominator_met=False, numerator_met=False,
            exclusions_applied=exclusions, rationale_short="Not neonate per definition",
            checklist=checklist, debug={"age_days": age_days, "ATYPE": atype, "LIVEBND": has_livebnd}
        )

    # Clinical exclusions
    has_preterm = any(c in PRETEID for c in all_dx)
    has_osteo = any(c in OSTEOID for c in all_dx)
    obstetric_principal = (principal_dx in MDC14PRINDX)

    checklist.append(ChecklistItem("Preterm <2000g (PRETEID)", "No", "Yes" if has_preterm else "No", not has_preterm))
    checklist.append(ChecklistItem("Osteogenesis imperfecta (OSTEOID)", "No", "Yes" if has_osteo else "No", not has_osteo))
    checklist.append(ChecklistItem("Obstetric principal (MDC14PRINDX)", "No", "Yes" if obstetric_principal else "No", not obstetric_principal))

    if has_preterm: exclusions.append("PRETEID")
    if has_osteo: exclusions.append("OSTEOID")
    if obstetric_principal: exclusions.append("MDC14PRINDX")

    if exclusions:
        return PSIEvalResult(
            encounter_id=enc_id, psi="PSI_17",
            result="EXCLUSION", denominator_met=True, numerator_met=False,
            exclusions_applied=exclusions, rationale_short="Denominator exclusion(s) applied",
            checklist=checklist, debug={"is_newborn": is_newborn, "is_outborn": is_outborn, "normal_newborn": normal_newborn}
        )

    # Numerator
    numerator_hit = any(c in BIRTHID for c in all_dx)
    checklist.append(ChecklistItem("Numerator: any birth trauma DX (BIRTHID)", "Yes", "Yes" if numerator_hit else "No", numerator_hit))

    result = "INCLUSION" if numerator_hit else "EXCLUSION"
    rationale = "Birth trauma diagnosis present" if numerator_hit else "No qualifying birth trauma diagnosis"

    return PSIEvalResult(
        encounter_id=enc_id, psi="PSI_17",
        result=result, denominator_met=True, numerator_met=numerator_hit,
        exclusions_applied=[], rationale_short=rationale,
        checklist=checklist, debug={"is_neonate": is_neonate, "is_newborn": is_newborn, "normal_newborn": normal_newborn, "is_outborn": is_outborn, "age_days": age_days, "ATYPE": atype, "PO": point_origin, "LIVEBND": has_livebnd, "LIVEB2D": has_liveb2d}
    )


@register_psi(18)
def evaluate_psi18(row: pd.Series, codes: Dict[str, List[str]]) -> PSIEvalResult:
    """
    PSI-18 Obstetric Trauma Rate – Vaginal Delivery With Instrument (2025).
    Denominator: any-listed DELOCMD diagnosis AND any VAGDELP procedure AND any INSTRIP procedure.
    Exclusions:
      - Principal DX in MDC15PRINDX (newborn)
      - DRG=999
      - Missing Sex/Age/DQTR/Year/Pdx (and MDC if provided)
    Numerator: any-listed OBTRAID diagnosis (3rd/4th degree obstetric injury).
    """
    enc_id = str(row.get("EncounterID", ""))
    checklist: List[ChecklistItem] = []
    exclusions: List[str] = []
    debug: Dict[str, Any] = {}

    # Core fields
    drg3 = str(row.get("MS-DRG_3digit", "")).strip()
    principal_dx = str(row.get("Pdx", "")).strip().upper()

    # Parse DX/POA and procedures
    dx_list = extract_dx_poa(row)  # [(code, poa, role)]
    all_dx = [c for (c,_,_) in dx_list]
    procs = extract_procedures(row)  # [(code, dt, col)]

    # Sets
    DELOCMD = set(codes.get("DELOCMD", []))
    VAGDELP = set(codes.get("VAGDELP", []))
    INSTRIP = set(codes.get("INSTRIP", []))
    OBTRAID = set(codes.get("OBTRAID", []))
    MDC15PRINDX = set(codes.get("MDC15PRINDX", []))

    # Denominator checks
    has_del_outcome = any(c in DELOCMD for c in all_dx)
    has_vag_del = any(code in VAGDELP for (code,_,_) in procs)
    has_instrument = any(code in INSTRIP for (code,_,_) in procs)

    checklist.append(ChecklistItem("Delivery outcome DX (DELOCMD)", "Yes", "Yes" if has_del_outcome else "No", has_del_outcome))
    checklist.append(ChecklistItem("Vaginal delivery procedure (VAGDELP)", "Yes", "Yes" if has_vag_del else "No", has_vag_del))
    checklist.append(ChecklistItem("Instrument-assisted procedure (INSTRIP)", "Yes", "Yes" if has_instrument else "No", has_instrument))

    denominator_met = has_del_outcome and has_vag_del and has_instrument

    # Exclusions
    obstetric_newborn_principal = (principal_dx in MDC15PRINDX)
    checklist.append(ChecklistItem("Principal DX in MDC15PRINDX (newborn)", "No", "Yes" if obstetric_newborn_principal else "No", not obstetric_newborn_principal))
    if obstetric_newborn_principal: exclusions.append("MDC15PRINDX principal")

    drg999 = (drg3 == "999")
    checklist.append(ChecklistItem("DRG not 999 (ungroupable)", "Not 999", drg3, not drg999))
    if drg999: exclusions.append("DRG=999")

    # Missing fields
    def missing(col: str) -> bool:
        return (col not in row) or (pd.isna(row[col])) or (str(row[col]).strip() == "")
    miss_sex = missing("Sex")
    miss_age = missing("Age")
    miss_qtr = missing("DQTR") if "DQTR" in row else False
    miss_year = missing("Year") if "Year" in row else False
    miss_pdx = missing("Pdx")

    checklist.append(ChecklistItem("Missing Sex", "Present", "Missing" if miss_sex else "Present", not miss_sex))
    checklist.append(ChecklistItem("Missing Age", "Present", "Missing" if miss_age else "Present", not miss_age))
    checklist.append(ChecklistItem("Missing DQTR (if provided)", "Present/NA", "Missing" if miss_qtr else "Present/NA", not miss_qtr))
    checklist.append(ChecklistItem("Missing Year (if provided)", "Present/NA", "Missing" if miss_year else "Present/NA", not miss_year))
    checklist.append(ChecklistItem("Missing Principal DX", "Present", "Missing" if miss_pdx else "Present", not miss_pdx))

    if miss_sex: exclusions.append("Missing Sex")
    if miss_age: exclusions.append("Missing Age")
    if miss_qtr: exclusions.append("Missing DQTR")
    if miss_year: exclusions.append("Missing Year")
    if miss_pdx: exclusions.append("Missing Pdx")

    if not denominator_met:
        return PSIEvalResult(
            encounter_id=enc_id, psi="PSI_18",
            result="EXCLUSION", denominator_met=False, numerator_met=False,
            exclusions_applied=exclusions, rationale_short="Denominator not met",
            checklist=checklist, debug={}
        )
    if exclusions:
        return PSIEvalResult(
            encounter_id=enc_id, psi="PSI_18",
            result="EXCLUSION", denominator_met=True, numerator_met=False,
            exclusions_applied=exclusions, rationale_short="Denominator exclusion(s) applied",
            checklist=checklist, debug={}
        )

    # Numerator
    numerator_hit = any(c in OBTRAID for c in all_dx)
    checklist.append(ChecklistItem("Numerator: obstetric injury DX (OBTRAID)", "Yes", "Yes" if numerator_hit else "No", numerator_hit))

    result = "INCLUSION" if numerator_hit else "EXCLUSION"
    rationale = "Obstetric trauma (3rd/4th degree) present" if numerator_hit else "No qualifying obstetric trauma DX"

    return PSIEvalResult(
        encounter_id=enc_id, psi="PSI_18",
        result=result, denominator_met=True, numerator_met=numerator_hit,
        exclusions_applied=[], rationale_short=rationale,
        checklist=checklist, debug={}
    )

@register_psi(19)
def evaluate_psi19(row: pd.Series, codes: Dict[str, List[str]]) -> PSIEvalResult:
    """
    PSI-19 Obstetric Trauma Rate – Vaginal Delivery Without Instrument (2025).
    Denominator: any-listed DELOCMD diagnosis AND any VAGDELP procedure.
    Exclusions:
      - Any-listed INSTRIP procedure (instrument-assisted delivery)
      - Principal DX in MDC15PRINDX (newborn)
      - DRG=999
      - Missing Sex/Age/DQTR/Year/Pdx (and MDC if provided)
    Numerator: any-listed OBTRAID diagnosis (3rd/4th degree obstetric injury).
    """
    enc_id = str(row.get("EncounterID", ""))
    checklist: List[ChecklistItem] = []
    exclusions: List[str] = []
    debug: Dict[str, Any] = {}

    # Core fields
    drg3 = str(row.get("MS-DRG_3digit", "")).strip()
    principal_dx = str(row.get("Pdx", "")).strip().upper()

    # Parse DX/POA and procedures
    dx_list = extract_dx_poa(row)  # [(code, poa, role)]
    all_dx = [c for (c,_,_) in dx_list]
    procs = extract_procedures(row)  # [(code, dt, col)]

    # Sets
    DELOCMD = set(codes.get("DELOCMD", []))
    VAGDELP = set(codes.get("VAGDELP", []))
    OBTRAID = set(codes.get("OBTRAID", []))
    INSTRIP = set(codes.get("INSTRIP", []))
    MDC15PRINDX = set(codes.get("MDC15PRINDX", []))

    # Denominator checks
    has_del_outcome = any(c in DELOCMD for c in all_dx)
    has_vag_del = any(code in VAGDELP for (code,_,_) in procs)

    checklist.append(ChecklistItem("Delivery outcome DX (DELOCMD)", "Yes", "Yes" if has_del_outcome else "No", has_del_outcome))
    checklist.append(ChecklistItem("Vaginal delivery procedure (VAGDELP)", "Yes", "Yes" if has_vag_del else "No", has_vag_del))

    denominator_met = has_del_outcome and has_vag_del

    # Exclusions
    has_instrument = any(code in INSTRIP for (code,_,_) in procs)
    checklist.append(ChecklistItem("Any instrument-assisted procedure (INSTRIP)", "No", "Yes" if has_instrument else "No", not has_instrument))
    if has_instrument: exclusions.append("INSTRIP present")

    obstetric_newborn_principal = (principal_dx in MDC15PRINDX)
    checklist.append(ChecklistItem("Principal DX in MDC15PRINDX (newborn)", "No", "Yes" if obstetric_newborn_principal else "No", not obstetric_newborn_principal))
    if obstetric_newborn_principal: exclusions.append("MDC15PRINDX principal")

    drg999 = (drg3 == "999")
    checklist.append(ChecklistItem("DRG not 999 (ungroupable)", "Not 999", drg3, not drg999))
    if drg999: exclusions.append("DRG=999")

    # Missing fields
    def missing(col: str) -> bool:
        return (col not in row) or (pd.isna(row[col])) or (str(row[col]).strip() == "")
    miss_sex = missing("Sex")
    miss_age = missing("Age")
    miss_qtr = missing("DQTR") if "DQTR" in row else False
    miss_year = missing("Year") if "Year" in row else False
    miss_pdx = missing("Pdx")

    checklist.append(ChecklistItem("Missing Sex", "Present", "Missing" if miss_sex else "Present", not miss_sex))
    checklist.append(ChecklistItem("Missing Age", "Present", "Missing" if miss_age else "Present", not miss_age))
    checklist.append(ChecklistItem("Missing DQTR (if provided)", "Present/NA", "Missing" if miss_qtr else "Present/NA", not miss_qtr))
    checklist.append(ChecklistItem("Missing Year (if provided)", "Present/NA", "Missing" if miss_year else "Present/NA", not miss_year))
    checklist.append(ChecklistItem("Missing Principal DX", "Present", "Missing" if miss_pdx else "Present", not miss_pdx))

    if miss_sex: exclusions.append("Missing Sex")
    if miss_age: exclusions.append("Missing Age")
    if miss_qtr: exclusions.append("Missing DQTR")
    if miss_year: exclusions.append("Missing Year")
    if miss_pdx: exclusions.append("Missing Pdx")

    if not denominator_met:
        return PSIEvalResult(
            encounter_id=enc_id, psi="PSI_19",
            result="EXCLUSION", denominator_met=False, numerator_met=False,
            exclusions_applied=exclusions, rationale_short="Denominator not met",
            checklist=checklist, debug={}
        )
    if exclusions:
        return PSIEvalResult(
            encounter_id=enc_id, psi="PSI_19",
            result="EXCLUSION", denominator_met=True, numerator_met=False,
            exclusions_applied=exclusions, rationale_short="Denominator exclusion(s) applied",
            checklist=checklist, debug={}
        )

    # Numerator
    numerator_hit = any(c in OBTRAID for c in all_dx)
    checklist.append(ChecklistItem("Numerator: obstetric injury DX (OBTRAID)", "Yes", "Yes" if numerator_hit else "No", numerator_hit))

    result = "INCLUSION" if numerator_hit else "EXCLUSION"
    rationale = "Obstetric trauma (3rd/4th degree) present" if numerator_hit else "No qualifying obstetric trauma DX"

    return PSIEvalResult(
        encounter_id=enc_id, psi="PSI_19",
        result=result, denominator_met=True, numerator_met=numerator_hit,
        exclusions_applied=[], rationale_short=rationale,
        checklist=checklist, debug={}
    )

@register_psi(19)
def evaluate_psi19(row: pd.Series, codes: Dict[str, List[str]]) -> PSIEvalResult:
    """
    PSI-19 Obstetric Trauma Rate – Vaginal Delivery Without Instrument (2025).
    Denominator: any-listed DELOCMD diagnosis AND any VAGDELP procedure.
    Exclusions:
      - Any INSTRIP procedure (instrument-assisted delivery)
      - Principal DX in MDC15PRINDX (newborn)
      - DRG=999
      - Missing Sex/Age/DQTR/Year/Pdx (and MDC if provided)
    Numerator: any-listed OBTRAID diagnosis (3rd/4th degree obstetric injury).
    """
    enc_id = str(row.get("EncounterID", ""))
    checklist: List[ChecklistItem] = []
    exclusions: List[str] = []
    debug: Dict[str, Any] = {}

    # Core fields
    drg3 = str(row.get("MS-DRG_3digit", "")).strip()
    principal_dx = str(row.get("Pdx", "")).strip().upper()

    # Parse DX/POA and procedures
    dx_list = extract_dx_poa(row)  # [(code, poa, role)]
    all_dx = [c for (c,_,_) in dx_list]
    procs = extract_procedures(row)  # [(code, dt, col)]

    # Sets
    DELOCMD = set(codes.get("DELOCMD", []))
    VAGDELP = set(codes.get("VAGDELP", []))
    INSTRIP = set(codes.get("INSTRIP", []))
    OBTRAID = set(codes.get("OBTRAID", []))
    MDC15PRINDX = set(codes.get("MDC15PRINDX", []))

    # Denominator checks
    has_del_outcome = any(c in DELOCMD for c in all_dx)
    has_vag_del = any(code in VAGDELP for (code,_,_) in procs)

    checklist.append(ChecklistItem("Delivery outcome DX (DELOCMD)", "Yes", "Yes" if has_del_outcome else "No", has_del_outcome))
    checklist.append(ChecklistItem("Vaginal delivery procedure (VAGDELP)", "Yes", "Yes" if has_vag_del else "No", has_vag_del))

    denominator_met = has_del_outcome and has_vag_del

    # Exclusions
    has_instrument = any(code in INSTRIP for (code,_,_) in procs)
    checklist.append(ChecklistItem("Instrument-assisted delivery (INSTRIP)", "No", "Yes" if has_instrument else "No", not has_instrument))
    if has_instrument: exclusions.append("INSTRIP present")

    obstetric_newborn_principal = (principal_dx in MDC15PRINDX)
    checklist.append(ChecklistItem("Principal DX in MDC15PRINDX (newborn)", "No", "Yes" if obstetric_newborn_principal else "No", not obstetric_newborn_principal))
    if obstetric_newborn_principal: exclusions.append("MDC15PRINDX principal")

    drg999 = (drg3 == "999")
    checklist.append(ChecklistItem("DRG not 999 (ungroupable)", "Not 999", drg3, not drg999))
    if drg999: exclusions.append("DRG=999")

    # Missing fields
    def missing(col: str) -> bool:
        return (col not in row) or (pd.isna(row[col])) or (str(row[col]).strip() == "")
    miss_sex = missing("Sex")
    miss_age = missing("Age")
    miss_qtr = missing("DQTR") if "DQTR" in row else False
    miss_year = missing("Year") if "Year" in row else False
    miss_pdx = missing("Pdx")

    checklist.append(ChecklistItem("Missing Sex", "Present", "Missing" if miss_sex else "Present", not miss_sex))
    checklist.append(ChecklistItem("Missing Age", "Present", "Missing" if miss_age else "Present", not miss_age))
    checklist.append(ChecklistItem("Missing DQTR (if provided)", "Present/NA", "Missing" if miss_qtr else "Present/NA", not miss_qtr))
    checklist.append(ChecklistItem("Missing Year (if provided)", "Present/NA", "Missing" if miss_year else "Present/NA", not miss_year))
    checklist.append(ChecklistItem("Missing Principal DX", "Present", "Missing" if miss_pdx else "Present", not miss_pdx))

    if miss_sex: exclusions.append("Missing Sex")
    if miss_age: exclusions.append("Missing Age")
    if miss_qtr: exclusions.append("Missing DQTR")
    if miss_year: exclusions.append("Missing Year")
    if miss_pdx: exclusions.append("Missing Pdx")

    if not denominator_met:
        return PSIEvalResult(
            encounter_id=enc_id, psi="PSI_19",
            result="EXCLUSION", denominator_met=False, numerator_met=False,
            exclusions_applied=exclusions, rationale_short="Denominator not met",
            checklist=checklist, debug={}
        )
    if exclusions:
        return PSIEvalResult(
            encounter_id=enc_id, psi="PSI_19",
            result="EXCLUSION", denominator_met=True, numerator_met=False,
            exclusions_applied=exclusions, rationale_short="Denominator exclusion(s) applied",
            checklist=checklist, debug={}
        )

    # Numerator
    numerator_hit = any(c in OBTRAID for c in all_dx)
    checklist.append(ChecklistItem("Numerator: obstetric injury DX (OBTRAID)", "Yes", "Yes" if numerator_hit else "No", numerator_hit))

    result = "INCLUSION" if numerator_hit else "EXCLUSION"
    rationale = "Obstetric trauma (3rd/4th degree) present" if numerator_hit else "No qualifying obstetric trauma DX"

    return PSIEvalResult(
        encounter_id=enc_id, psi="PSI_19",
        result=result, denominator_met=True, numerator_met=numerator_hit,
        exclusions_applied=[], rationale_short=rationale,
        checklist=checklist, debug={}
    )

# Stubs for others; will be replaced as rules are provided.
for n in [4,5,6,7,8,9,10,11,12,13,14,15,17,18,19]:
    def _stub(row: pd.Series, codes: Dict[str, List[str]], _n=n):
        enc_id = str(row.get("EncounterID", ""))
        return PSIEvalResult(
            encounter_id=enc_id,
            psi=f"PSI_{_n:02d}",
            result="EXCLUSION",
            denominator_met=False,
            numerator_met=False,
            exclusions_applied=[],
            rationale_short="Awaiting rule text",
            checklist=[],
            debug={"status": "stub"},
        )
    _PSI_REGISTRY[n] = _stub


@register_psi(5)
def evaluate_psi05(row: pd.Series, codes: Dict[str, List[str]]) -> PSIEvalResult:
    """
    PSI-05 Retained Surgical Item or Unretrieved Device Fragment Count (2025).
    """
    enc_id = str(row.get("EncounterID", ""))
    checklist: List[ChecklistItem] = []
    exclusions: List[str] = []

    # Core fields
    age = row.get("Age", None)
    drg3 = row.get("MS-DRG_3digit", "")
    principal_dx = str(row.get("Pdx", "")).strip().upper()

    # DRG sets
    SURGI2R = set(str(x).zfill(3) for x in codes.get("SURGI2R", []))
    MEDIC2R = set(str(x).zfill(3) for x in codes.get("MEDIC2R", []))

    # Other sets
    MDC14PRINDX = set(codes.get("MDC14PRINDX", []))
    MDC15PRINDX = set(codes.get("MDC15PRINDX", []))
    FOREIID = set(codes.get("FOREIID", []))

    # Extract dx+poa
    dx_list = extract_dx_poa(row)  # [(code, poa, role)]

    # --- Denominator check
    drg_ok = drg3 in SURGI2R or drg3 in MEDIC2R
    age_ok = False
    if age is not None and str(age).strip() != "":
        try:
            av = float(age)
            age_ok = av >= 18
        except Exception:
            age_ok = False
    obstetric_principal = principal_dx in MDC14PRINDX

    checklist.append(ChecklistItem("DRG in SURGI2R or MEDIC2R", "Yes", drg3, drg_ok))
    checklist.append(ChecklistItem("Age >=18 or obstetric principal", "Yes", f"Age={age}, obstetric={obstetric_principal}", age_ok or obstetric_principal))

    denominator_met = drg_ok and (age_ok or obstetric_principal)

    # --- Exclusions
    # Principal in FOREIID
    excl_pdx_forei = principal_dx in FOREIID
    checklist.append(ChecklistItem("Principal not in FOREIID", "No", "Yes" if excl_pdx_forei else "No", not excl_pdx_forei))
    if excl_pdx_forei: exclusions.append("PrincipalDX FOREIID")

    # Secondary FOREIID with POA=Y
    excl_sec_poaY = any(code in FOREIID and poa == "Y" for (code, poa, role) in dx_list if role == "SECONDARY")
    checklist.append(ChecklistItem("No secondary FOREIID with POA=Y", "No", "Yes" if excl_sec_poaY else "No", not excl_sec_poaY))
    if excl_sec_poaY: exclusions.append("Secondary FOREIID POA=Y")

    # MDC15
    excl_mdc15 = principal_dx in MDC15PRINDX
    checklist.append(ChecklistItem("Principal not MDC15 (newborn)", "No", "Yes" if excl_mdc15 else "No", not excl_mdc15))
    if excl_mdc15: exclusions.append("MDC15PRINDX")

    # DRG=999
    excl_drg999 = (drg3 == "999")
    checklist.append(ChecklistItem("DRG not 999", "Not 999", drg3, not excl_drg999))
    if excl_drg999: exclusions.append("DRG=999")

    # Missing required fields
    def missing(col: str) -> bool:
        return (col not in row) or (pd.isna(row[col])) or (str(row[col]).strip() == "")
    miss_sex = missing("Sex")
    miss_age = missing("Age")
    miss_qtr = missing("DQTR") if "DQTR" in row else False
    miss_year = missing("Year") if "Year" in row else False
    miss_pdx = missing("Pdx")

    checklist.append(ChecklistItem("Missing Sex", "Present", "Missing" if miss_sex else "Present", not miss_sex))
    checklist.append(ChecklistItem("Missing Age", "Present", "Missing" if miss_age else "Present", not miss_age))
    checklist.append(ChecklistItem("Missing DQTR (if provided)", "Present/NA", "Missing" if miss_qtr else "Present/NA", not miss_qtr))
    checklist.append(ChecklistItem("Missing Year (if provided)", "Present/NA", "Missing" if miss_year else "Present/NA", not miss_year))
    checklist.append(ChecklistItem("Missing Principal DX", "Present", "Missing" if miss_pdx else "Present", not miss_pdx))

    if miss_sex: exclusions.append("Missing Sex")
    if miss_age: exclusions.append("Missing Age")
    if miss_qtr: exclusions.append("Missing DQTR")
    if miss_year: exclusions.append("Missing Year")
    if miss_pdx: exclusions.append("Missing Pdx")

    if "MDC" in row:
        miss_mdc = missing("MDC")
        checklist.append(ChecklistItem("Missing MDC (if provided)", "Present/NA", "Missing" if miss_mdc else "Present/NA", not miss_mdc))
        if miss_mdc: exclusions.append("Missing MDC")

    # --- Early exits
    if not denominator_met:
        return PSIEvalResult(enc_id, "PSI_05", "EXCLUSION", False, False, exclusions,
            "Denominator not met", checklist, debug={"age": age, "drg3": drg3, "pdx": principal_dx})
    if exclusions:
        return PSIEvalResult(enc_id, "PSI_05", "EXCLUSION", True, False, exclusions,
            "Exclusion(s) applied", checklist, debug={"exclusions": exclusions})

    # --- Numerator: secondary FOREIID not POA=Y
    numerator_hit = any(code in FOREIID and poa != "Y" for (code, poa, role) in dx_list if role == "SECONDARY")
    checklist.append(ChecklistItem("Secondary FOREIID not POA=Y (numerator)", "Yes", "Yes" if numerator_hit else "No", numerator_hit))

    result = "INCLUSION" if numerator_hit else "EXCLUSION"
    rationale = "Triggered retained surgical item/device fragment" if numerator_hit else "No qualifying secondary FOREIID"

    return PSIEvalResult(enc_id, "PSI_05", result, True, numerator_hit, [], rationale, checklist, debug={"numerator_hit": numerator_hit})

# -------------------------
# Orchestrator
# -------------------------

def evaluate_dataframe(df: pd.DataFrame, codes: Dict[str, List[str]], debug_dir: Optional[str]) -> pd.DataFrame:
    results_rows: List[Dict[str, Any]] = []
    ensure_dir(debug_dir or "")

    for idx, row in df.iterrows():
        enc_id = str(row.get("EncounterID", ""))
        for psi_n in PSI_LIST:
            fn = _PSI_REGISTRY.get(psi_n)
            res: PSIEvalResult = fn(row, codes)

            # Save debug file
            if debug_dir:
                debug_path = os.path.join(debug_dir, f"{enc_id}_PSI_{psi_n:02d}.json")
                payload = asdict(res)
                payload["checklist"] = [asdict(c) for c in res.checklist]
                with open(debug_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2, default=str)
                res.debug_path = debug_path

            results_rows.append(res.to_row())

    return pd.DataFrame(results_rows)


def main():
    ap = argparse.ArgumentParser(description="PSI 03–19 Calculator (2025)")
    ap.add_argument("--excel", required=True, help="Path to input Excel with patient data")
    ap.add_argument("--sheet", default=0, help="Excel sheet name or index (default: first sheet)")
    ap.add_argument("--codes", default="/mnt/data/PSI_Code_Sets_2025.json", help="Path to PSI code sets JSON (2025)")
    ap.add_argument("--out", default="/mnt/data/psi_results.csv", help="Output CSV path")
    ap.add_argument("--debug_dir", default="/mnt/data/psi_debug", help="Directory to save per-PSI debug JSON files")
    args = ap.parse_args()

    codes = load_code_sets(args.codes)
    df_raw = pd.read_excel(args.excel, sheet_name=args.sheet, dtype=str)
    df = normalize_input(df_raw)
    df_out = evaluate_dataframe(df, codes, debug_dir=args.debug_dir)
    df_out.to_csv(args.out, index=False)
    print(f"Saved results to: {args.out}")
    if args.debug_dir:
        print(f"Saved debug JSON to: {args.debug_dir}")

if __name__ == "__main__":
    main()