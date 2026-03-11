"""
ClinicalGuard — LangGraph Pipeline
AI-Powered Clinical Coding with Evidence Validation

Graph: parse_note → extract_codes → validate_labs → validate_prescriptions → generate_report
"""

import json
import os
from typing import TypedDict, Annotated

import pandas as pd
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

load_dotenv()

# ---------------------------------------------------------------------------
# Data loaders (cached at module level)
# Local CSV fallback → Hugging Face raw URLs for cloud
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
HF_BASE = "https://huggingface.co/datasets/bavehackathon/2026-healthcare-ai/resolve/main/"

DATA_FILES = [
    "clinical_cases.csv",
    "diagnoses_subset.csv",
    "diagnosis_dictionary.csv",
    "labs_subset.csv",
    "lab_dictionary.csv",
    "prescriptions_subset.csv",
]

def _load_csv(name: str) -> pd.DataFrame:
    local_path = os.path.join(DATA_DIR, name)
    if os.path.exists(local_path):
        return pd.read_csv(local_path)
    # Fall back to Hugging Face (cloud deployment)
    gz_name = name + ".gz"
    url = HF_BASE + gz_name
    return pd.read_csv(url, compression="gzip")

_cases = None
_diag = None
_diag_dict = None
_labs = None
_lab_dict = None
_rx = None

def get_data():
    global _cases, _diag, _diag_dict, _labs, _lab_dict, _rx
    if _cases is None:
        _cases = _load_csv("clinical_cases.csv")
        _diag = _load_csv("diagnoses_subset.csv")
        _diag_dict = _load_csv("diagnosis_dictionary.csv")
        _labs = _load_csv("labs_subset.csv")
        _lab_dict = _load_csv("lab_dictionary.csv")
        _rx = _load_csv("prescriptions_subset.csv")
        # Coerce join keys to int-then-string — Hugging Face loads floats, local loads ints
        for df in [_cases, _diag, _labs, _rx]:
            df["hadm_id"] = pd.to_numeric(df["hadm_id"], errors="coerce").astype("Int64").astype(str)
        _diag["icd9_code"] = _diag["icd9_code"].astype(str).str.strip()
        _diag_dict["icd9_code"] = _diag_dict["icd9_code"].astype(str).str.strip()
        for df in [_labs, _lab_dict]:
            df["itemid"] = pd.to_numeric(df["itemid"], errors="coerce").astype("Int64").astype(str)
    return _cases, _diag, _diag_dict, _labs, _lab_dict, _rx


def get_patient_case(hadm_id: float) -> dict:
    """Load all data for a single patient admission."""
    cases, diag, diag_dict, labs, lab_dict, rx = get_data()
    hadm_str = str(int(hadm_id)) if hadm_id == int(hadm_id) else str(hadm_id)
    case = cases[cases["hadm_id"] == hadm_str].iloc[0]
    patient_diag = diag[diag["hadm_id"] == hadm_str].merge(diag_dict, on="icd9_code", how="left")
    patient_labs = labs[labs["hadm_id"] == hadm_str].merge(lab_dict, on="itemid", how="left")
    patient_rx = rx[rx["hadm_id"] == hadm_str]
    return {
        "case_id": case["case_id"],
        "hadm_id": hadm_id,
        "subject_id": case["subject_id"],
        "age": case["age"],
        "gender": case["gender"],
        "admission_diagnosis": case["admission_diagnosis"],
        "discharge_summary": case["discharge_summary"],
        "actual_diagnoses": patient_diag[["icd9_code", "short_title", "long_title", "seq_num"]].to_dict("records"),
        "labs": patient_labs[["lab_name", "value", "unit", "charttime", "fluid", "category"]].to_dict("records"),
        "prescriptions": patient_rx[["drug", "dose_value", "dose_unit", "route", "startdate", "enddate"]].to_dict("records"),
    }


def list_cases() -> list[dict]:
    """Return a summary list of all cases for the UI selector."""
    cases, diag, _, labs, _, rx = get_data()
    diag_counts = diag.groupby("hadm_id").size().rename("n_diag")
    lab_counts = labs.groupby("hadm_id").size().rename("n_labs")
    rx_counts = rx.groupby("hadm_id").size().rename("n_rx")
    enriched = cases.set_index("hadm_id").join([diag_counts, lab_counts, rx_counts]).reset_index()
    enriched = enriched.fillna(0)
    result = []
    for _, row in enriched.iterrows():
        result.append({
            "case_id": row["case_id"],
            "hadm_id": row["hadm_id"],
            "age": int(row["age"]),
            "gender": row["gender"],
            "admission_diagnosis": row["admission_diagnosis"],
            "n_diag": int(row["n_diag"]),
            "n_labs": int(row["n_labs"]),
            "n_rx": int(row["n_rx"]),
        })
    return result


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
def get_llm():
    return ChatAnthropic(
        model="claude-sonnet-4-20250514",
        temperature=0,
        max_tokens=4096,
    )


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class ClinicalState(TypedDict):
    # Input
    discharge_summary: str
    hadm_id: float
    patient_info: dict  # age, gender, admission_diagnosis
    # Loaded data
    lab_records: list[dict]
    prescription_records: list[dict]
    actual_icd_codes: list[dict]
    # Pipeline outputs (each node appends its result)
    extracted_concepts: list[dict]  # from parse step
    predicted_codes: list[dict]     # from ICD matching step
    lab_validation: list[dict]      # from lab cross-reference
    rx_validation: list[dict]       # from prescription cross-reference
    final_report: dict              # from report generation
    # Step logs for UI
    step_logs: list[dict]


# ---------------------------------------------------------------------------
# Node 1: Parse Note & Extract Medical Concepts
# ---------------------------------------------------------------------------
def parse_note(state: ClinicalState) -> dict:
    llm = get_llm()
    note = state["discharge_summary"]
    # Truncate very long notes to stay within context
    if len(note) > 12000:
        note = note[:12000] + "\n\n[Note truncated for processing]"

    response = llm.invoke([
        SystemMessage(content="""You are a clinical NLP system. Extract all medical concepts from this discharge summary.

For each concept, provide:
- concept: the medical term/condition as stated in the note
- type: one of [diagnosis, symptom, procedure, medication, lab_finding]
- confidence: high, medium, or low
- evidence: the exact quote from the note that supports this extraction

Return ONLY valid JSON — an array of objects. No markdown, no explanation."""),
        HumanMessage(content=f"Extract medical concepts from this discharge note:\n\n{note}")
    ])

    try:
        concepts = json.loads(response.content)
    except json.JSONDecodeError:
        # Try to extract JSON from response
        text = response.content
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            concepts = json.loads(text[start:end])
        else:
            concepts = []

    return {
        "extracted_concepts": concepts,
        "step_logs": state.get("step_logs", []) + [{
            "step": "1. Parse Note",
            "title": "Extract Medical Concepts",
            "summary": f"Extracted {len(concepts)} medical concepts from the discharge note",
            "detail": concepts,
        }]
    }


# ---------------------------------------------------------------------------
# Node 2: Match to ICD-9 Codes
# ---------------------------------------------------------------------------
def match_icd_codes(state: ClinicalState) -> dict:
    llm = get_llm()
    concepts = state["extracted_concepts"]
    _, _, diag_dict, _, _, _ = get_data()

    # Get only diagnosis-type concepts
    diagnoses = [c for c in concepts if c.get("type") in ("diagnosis", "symptom", "lab_finding")]
    if not diagnoses:
        diagnoses = concepts[:10]  # fallback

    # Build a lookup string from the ICD dictionary (top relevant codes)
    icd_reference = diag_dict[["icd9_code", "short_title", "long_title"]].to_string(index=False, max_rows=200)

    response = llm.invoke([
        SystemMessage(content="""You are a medical coding specialist. Given extracted medical concepts and an ICD-9 code dictionary, match each diagnosis to the most appropriate ICD-9 code.

For each match, provide:
- concept: the original extracted concept
- icd9_code: the matched ICD-9 code
- code_title: the title of the matched code
- confidence: 0-100 (percentage)
- reasoning: brief explanation of why this code was selected
- needs_review: true if confidence < 70 or the match is ambiguous

Return ONLY valid JSON — an array of objects. No markdown."""),
        HumanMessage(content=f"""Extracted diagnoses to code:
{json.dumps(diagnoses[:15], indent=2)}

ICD-9 Code Dictionary (reference):
{icd_reference}""")
    ])

    try:
        codes = json.loads(response.content)
    except json.JSONDecodeError:
        text = response.content
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            codes = json.loads(text[start:end])
        else:
            codes = []

    return {
        "predicted_codes": codes,
        "step_logs": state.get("step_logs", []) + [{
            "step": "2. Match ICD Codes",
            "title": "ICD-9 Code Assignment",
            "summary": f"Assigned {len(codes)} ICD-9 codes — {sum(1 for c in codes if c.get('needs_review'))} flagged for review",
            "detail": codes,
        }]
    }


# ---------------------------------------------------------------------------
# Node 3: Validate Against Lab Results
# ---------------------------------------------------------------------------
def validate_labs(state: ClinicalState) -> dict:
    llm = get_llm()
    codes = state["predicted_codes"]
    lab_records = state["lab_records"]

    if not lab_records:
        return {
            "lab_validation": [{"status": "no_labs", "message": "No lab data available for this admission"}],
            "step_logs": state.get("step_logs", []) + [{
                "step": "3. Lab Validation",
                "title": "Cross-Reference Lab Results",
                "summary": "No lab data available for this admission",
                "detail": [],
            }]
        }

    # Summarize labs (too many to send all — send unique tests with latest values)
    lab_df = pd.DataFrame(lab_records)
    lab_summary = lab_df.groupby("lab_name").agg(
        latest_value=("value", "last"),
        unit=("unit", "first"),
        count=("value", "count")
    ).reset_index().head(50).to_string(index=False)

    response = llm.invoke([
        SystemMessage(content="""You are a clinical validation system. Given a set of ICD-9 diagnosis codes and the patient's lab results, determine whether the lab evidence supports each diagnosis.

For each diagnosis code, assess:
- icd9_code: the code being validated
- code_title: the diagnosis name
- lab_support: "supported", "contradicted", "insufficient_data", or "not_applicable"
- key_labs: array of objects with {"lab_name": "...", "patient_value": "...", "unit": "...", "normal_range": "...", "interpretation": "normal/abnormal/critical"} for the specific lab values relevant to this diagnosis
- reasoning: clinical explanation of your assessment
- flag: true if the lab data contradicts the diagnosis

Return ONLY valid JSON — an array of objects. No markdown."""),
        HumanMessage(content=f"""ICD-9 Codes to validate:
{json.dumps(codes[:15], indent=2)}

Patient Lab Results (summary of {len(lab_records)} results):
{lab_summary}""")
    ])

    try:
        validation = json.loads(response.content)
    except json.JSONDecodeError:
        text = response.content
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            validation = json.loads(text[start:end])
        else:
            validation = []

    flagged = sum(1 for v in validation if v.get("flag"))
    return {
        "lab_validation": validation,
        "step_logs": state.get("step_logs", []) + [{
            "step": "3. Lab Validation",
            "title": "Cross-Reference Lab Results",
            "summary": f"Validated {len(validation)} codes against labs — {flagged} inconsistencies flagged",
            "detail": validation,
        }]
    }


# ---------------------------------------------------------------------------
# Node 4: Validate Against Prescriptions
# ---------------------------------------------------------------------------
def validate_prescriptions(state: ClinicalState) -> dict:
    llm = get_llm()
    codes = state["predicted_codes"]
    rx_records = state["prescription_records"]

    if not rx_records:
        return {
            "rx_validation": [{"status": "no_rx", "message": "No prescription data available"}],
            "step_logs": state.get("step_logs", []) + [{
                "step": "4. Prescription Validation",
                "title": "Cross-Reference Prescriptions",
                "summary": "No prescription data available for this admission",
                "detail": [],
            }]
        }

    # Summarize prescriptions
    rx_df = pd.DataFrame(rx_records)
    rx_summary = rx_df.groupby("drug").agg(
        dose=("dose_value", "first"),
        unit=("dose_unit", "first"),
        route=("route", "first"),
    ).reset_index().head(40).to_string(index=False)

    response = llm.invoke([
        SystemMessage(content="""You are a clinical validation system. Given ICD-9 diagnosis codes and the patient's prescriptions, determine whether the medications align with the coded diagnoses.

For each diagnosis code, assess:
- icd9_code: the code being validated
- code_title: the diagnosis name
- rx_support: "supported", "contradicted", "insufficient_data", or "not_applicable"
- relevant_drugs: which medications relate to this diagnosis
- reasoning: clinical explanation
- flag: true if expected medications are missing for a coded condition, or unexpected medications suggest an uncoded condition

Return ONLY valid JSON — an array of objects. No markdown."""),
        HumanMessage(content=f"""ICD-9 Codes to validate:
{json.dumps(codes[:15], indent=2)}

Patient Prescriptions (summary of {len(rx_records)} records):
{rx_summary}""")
    ])

    try:
        validation = json.loads(response.content)
    except json.JSONDecodeError:
        text = response.content
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            validation = json.loads(text[start:end])
        else:
            validation = []

    flagged = sum(1 for v in validation if v.get("flag"))
    return {
        "rx_validation": validation,
        "step_logs": state.get("step_logs", []) + [{
            "step": "4. Prescription Validation",
            "title": "Cross-Reference Prescriptions",
            "summary": f"Validated {len(validation)} codes against prescriptions — {flagged} inconsistencies flagged",
            "detail": validation,
        }]
    }


# ---------------------------------------------------------------------------
# Node 5: Generate Final Report
# ---------------------------------------------------------------------------
def generate_report(state: ClinicalState) -> dict:
    codes = state["predicted_codes"]
    lab_val = state["lab_validation"]
    rx_val = state["rx_validation"]
    patient = state["patient_info"]

    # Build consolidated report
    report_entries = []
    for code in codes:
        icd = code.get("icd9_code", "")
        entry = {
            "icd9_code": icd,
            "code_title": code.get("code_title", code.get("concept", "")),
            "confidence": code.get("confidence", 0),
            "reasoning": code.get("reasoning", ""),
            "needs_review": code.get("needs_review", False),
            "lab_status": "not_checked",
            "lab_detail": "",
            "rx_status": "not_checked",
            "rx_detail": "",
            "overall_status": "confirmed",
        }

        # Find matching lab validation
        for lv in lab_val:
            if lv.get("icd9_code") == icd:
                entry["lab_status"] = lv.get("lab_support", "not_checked")
                entry["lab_detail"] = lv.get("reasoning", "")
                if lv.get("flag"):
                    entry["overall_status"] = "inconsistency_flagged"
                break

        # Find matching rx validation
        for rv in rx_val:
            if rv.get("icd9_code") == icd:
                entry["rx_status"] = rv.get("rx_support", "not_checked")
                entry["rx_detail"] = rv.get("reasoning", "")
                if rv.get("flag"):
                    entry["overall_status"] = "inconsistency_flagged"
                break

        # Determine overall status
        # Priority: flagged > confirmed > needs_review > unvalidated
        if entry["overall_status"] != "inconsistency_flagged":
            has_lab_support = entry["lab_status"] == "supported"
            has_rx_support = entry["rx_status"] == "supported"
            has_contradiction = entry["lab_status"] == "contradicted" or entry["rx_status"] == "contradicted"

            if has_contradiction:
                entry["overall_status"] = "inconsistency_flagged"
            elif has_lab_support or has_rx_support:
                entry["overall_status"] = "confirmed"
            elif code.get("confidence", 100) < 50:
                entry["overall_status"] = "needs_review"
            elif entry["needs_review"] and not (has_lab_support or has_rx_support):
                entry["overall_status"] = "needs_review"
            else:
                entry["overall_status"] = "unvalidated"

        report_entries.append(entry)

    # Summary stats
    confirmed = sum(1 for e in report_entries if e["overall_status"] == "confirmed")
    flagged = sum(1 for e in report_entries if e["overall_status"] == "inconsistency_flagged")
    review = sum(1 for e in report_entries if e["overall_status"] == "needs_review")
    unvalidated = sum(1 for e in report_entries if e["overall_status"] == "unvalidated")

    report = {
        "patient": patient,
        "total_codes": len(report_entries),
        "confirmed": confirmed,
        "flagged": flagged,
        "needs_review": review,
        "unvalidated": unvalidated,
        "entries": report_entries,
    }

    return {
        "final_report": report,
        "step_logs": state.get("step_logs", []) + [{
            "step": "5. Final Report",
            "title": "Consolidated Validation Report",
            "summary": f"{len(report_entries)} codes: {confirmed} confirmed, {flagged} flagged, {review} need review, {unvalidated} unvalidated",
            "detail": report,
        }]
    }


# ---------------------------------------------------------------------------
# Build the Graph
# ---------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(ClinicalState)

    graph.add_node("parse_note", parse_note)
    graph.add_node("match_icd_codes", match_icd_codes)
    graph.add_node("validate_labs", validate_labs)
    graph.add_node("validate_prescriptions", validate_prescriptions)
    graph.add_node("generate_report", generate_report)

    graph.set_entry_point("parse_note")
    graph.add_edge("parse_note", "match_icd_codes")
    graph.add_edge("match_icd_codes", "validate_labs")
    graph.add_edge("validate_labs", "validate_prescriptions")
    graph.add_edge("validate_prescriptions", "generate_report")
    graph.add_edge("generate_report", END)

    return graph.compile()


def run_pipeline(hadm_id: float) -> ClinicalState:
    """Run the full ClinicalGuard pipeline for a patient admission."""
    patient = get_patient_case(hadm_id)
    app = build_graph()

    initial_state: ClinicalState = {
        "discharge_summary": patient["discharge_summary"],
        "hadm_id": hadm_id,
        "patient_info": {
            "case_id": patient["case_id"],
            "age": patient["age"],
            "gender": patient["gender"],
            "admission_diagnosis": patient["admission_diagnosis"],
        },
        "lab_records": patient["labs"],
        "prescription_records": patient["prescriptions"],
        "actual_icd_codes": patient["actual_diagnoses"],
        "extracted_concepts": [],
        "predicted_codes": [],
        "lab_validation": [],
        "rx_validation": [],
        "final_report": {},
        "step_logs": [],
    }

    result = app.invoke(initial_state)
    return result


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    hadm_id = float(sys.argv[1]) if len(sys.argv) > 1 else 171106.0  # SEPSIS case
    print(f"Running ClinicalGuard pipeline for hadm_id={hadm_id}...")
    result = run_pipeline(hadm_id)
    print(f"\n=== PIPELINE COMPLETE ===")
    for log in result["step_logs"]:
        print(f"\n--- {log['step']}: {log['title']} ---")
        print(log["summary"])
    report = result["final_report"]
    print(f"\n=== FINAL REPORT ===")
    print(f"Total codes: {report['total_codes']}")
    print(f"Confirmed: {report['confirmed']}")
    print(f"Flagged: {report['flagged']}")
    print(f"Needs review: {report['needs_review']}")
    for entry in report["entries"]:
        status_icon = {"confirmed": "[OK]", "inconsistency_flagged": "[FLAG]", "needs_review": "[REVIEW]", "unvalidated": "[--]"}.get(entry["overall_status"], "?")
        print(f"  {status_icon} {entry['icd9_code']} - {entry['code_title']} (confidence: {entry['confidence']}%)")
        if entry["overall_status"] == "inconsistency_flagged":
            print(f"     LAB: {entry['lab_detail']}")
            print(f"     RX:  {entry['rx_detail']}")
