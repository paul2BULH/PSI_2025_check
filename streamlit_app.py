
import io
import json
from typing import Dict, List, Any
import pandas as pd
import numpy as np
import streamlit as st

# Import user's evaluation engine
import psi_core as psi

st.set_page_config(page_title="PSI 03–19 Evaluator (2025)", layout="wide")

st.title("PSI 03–19 Evaluator (2025) — Bulwark/ARC+")
st.caption("Upload your compatible Excel. App uses your PSI core to compute results and show checklist per encounter.")

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    code_sets_path = st.text_input("Code Sets JSON", "PSI_Code_Sets_2025.json")
    show_debug = st.checkbox("Show debug blocks", value=False)
    multi_select_psis = st.multiselect(
        "Filter PSIs",
        options=[f"PSI_{n:02d}" for n in psi.PSI_LIST],
        default=[f"PSI_{n:02d}" for n in psi.PSI_LIST],
    )

uploaded = st.file_uploader("Upload compatible Excel file", type=["xlsx","xls"])

@st.cache_data(show_spinner=False)
def load_codes(path: str) -> Dict[str, List[str]]:
    return psi.load_code_sets(path)

@st.cache_data(show_spinner=False)
def normalize(df: pd.DataFrame) -> pd.DataFrame:
    return psi.normalize_input(df)

def safe_result_label(label: str) -> str:
    # Mr. Paul's instruction: no NOT_APPLICABLE; collapse to EXCLUSION
    if str(label).strip().upper() == "NOT_APPLICABLE":
        return "EXCLUSION"
    return str(label).strip().upper()

def evaluate_encounter_row(row: pd.Series, codes: Dict[str, List[str]]) -> List[psi.PSIEvalResult]:
    out = []
    for n in psi.PSI_LIST:
        if f"PSI_{n:02d}" not in multi_select_psis:
            continue
        fn = psi._PSI_REGISTRY.get(n)
        if fn is None:
            # Should not happen: safeguard
            out.append(psi.PSIEvalResult(
                encounter_id=str(row.get("EncounterID","")),
                psi=f"PSI_{n:02d}",
                result="EXCLUSION",
                denominator_met=False,
                numerator_met=False,
                rationale_short="PSI function not registered",
            ))
        else:
            try:
                res = fn(row, codes)
            except Exception as e:
                res = psi.PSIEvalResult(
                    encounter_id=str(row.get("EncounterID","")),
                    psi=f"PSI_{n:02d}",
                    result="EXCLUSION",
                    denominator_met=False,
                    numerator_met=False,
                    rationale_short=f"Runtime error: {e}",
                )
            # Enforce Inclusion/Exclusion only
            res.result = safe_result_label(res.result)
            out.append(res)
    return out

if uploaded:
    try:
        codes = load_codes(code_sets_path)
    except Exception as e:
        st.error(f"Failed to load code sets from {code_sets_path}: {e}")
        st.stop()

    try:
        xls = pd.ExcelFile(uploaded)
        sheet = st.selectbox("Select sheet", options=xls.sheet_names)
        raw_df = xls.parse(sheet)
    except Exception as e:
        st.error(f"Failed to read Excel: {e}")
        st.stop()

    df = normalize(raw_df)
    if "EncounterID" not in df.columns:
        st.warning("Expected column 'EncounterID' is missing — please include it to group rows.")
        # Create a surrogate ID to avoid total failure
        df["EncounterID"] = np.arange(1, len(df)+1).astype(str)

    # Evaluate all
    all_results: List[psi.PSIEvalResult] = []
    prog = st.progress(0.0, text="Evaluating...")
    for idx, row in df.iterrows():
        all_results.extend(evaluate_encounter_row(row, codes))
        if len(df) > 0:
            prog.progress((idx+1)/len(df), text=f"Evaluating... {idx+1}/{len(df)}")
    prog.empty()

    # Build consolidated table
    rows = [r.to_row() for r in all_results]
    out_df = pd.DataFrame(rows)
    out_df["Result"] = out_df["Result"].str.upper().replace({"NOT_APPLICABLE":"EXCLUSION"})

    # KPI summary
    # ---- Results table (EncounterID, PSI, Result) ----
    st.subheader("Results (EncounterID · PSI · Result)")
    simple_df = out_df[["EncounterID","PSI","Result"]].copy()
    st.dataframe(simple_df, use_container_width=True, hide_index=True)
    csv_simple = simple_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download results (EncounterID-PSI-Result).csv", csv_simple, file_name="psi_results_simple.csv", mime="text/csv")
    
    st.subheader("KPI Summary")
    kpi = (
        out_df
        .groupby(["PSI","Result"])
        .size()
        .reset_index(name="Count")
        .pivot_table(index="PSI", columns="Result", values="Count", fill_value=0)
        .reset_index()
    )
    st.dataframe(kpi, use_container_width=True)
    csv = out_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download consolidated CSV", csv, file_name="psi_results_consolidated.csv", mime="text/csv")

    # Encounter-wise dropdown UI
    st.subheader("Encounter-wise Review")
    enc_ids = list(out_df["EncounterID"].unique())
    st.write(f"Total Encounters: **{len(enc_ids)}**")
    for enc in enc_ids:
        with st.expander(f"Encounter {enc}"):
            enc_rows = [r for r in all_results if r.encounter_id == str(enc) and r.psi in multi_select_psis]
            # Render per-PSI blocks
            for r in enc_rows:
                # Compact header line with quick decision dropdown-like feel
                col1, col2, col3, col4 = st.columns([1.2, 1, 2, 3])
                with col1:
                    st.markdown(f"**{r.psi}**")
                with col2:
                    st.markdown(f"**{safe_result_label(r.result)}**")
                with col3:
                    st.markdown(r.rationale_short or "")
                with col4:
                    # A real dropdown component for details
                    with st.expander("Checklist & Details"):
                        # Checklist table
                        if r.checklist:
                            checklist_df = pd.DataFrame([{
                                "Criterion": c.criterion,
                                "Expected": c.expected,
                                "Found": c.found,
                                "Passed": bool(c.passed),
                                "Note": c.note or ""
                            } for c in r.checklist])
                            st.dataframe(checklist_df, use_container_width=True, hide_index=True)
                        else:
                            st.info("No checklist captured for this PSI.")
                        # Exclusions, denom/numer flags, debug
                        st.markdown(f"- **Denominator Met:** {bool(r.denominator_met)}")
                        st.markdown(f"- **Numerator Met:** {bool(r.numerator_met)}")
                        if r.exclusions_applied:
                            st.markdown(f"- **Exclusions Applied:** {', '.join(r.exclusions_applied)}")
                        # Surface appendix/code-set keys if present
                        keys = []
                        if isinstance(r.debug, dict):
                            for k in ("appendix_keys", "matched_appendix", "matched_code_sets", "code_sets_used"):
                                v = r.debug.get(k)
                                if isinstance(v, (list, tuple, set)):
                                    keys.extend([str(x) for x in v])
                        if keys:
                            st.markdown("**Matched Appendix Keys:** " + ", ".join(sorted(set(keys))))

                        if show_debug and r.debug:
                            st.json(r.debug)

    st.success("Evaluation complete.")
else:
    st.info("Please upload a compatible Excel to begin.")    
