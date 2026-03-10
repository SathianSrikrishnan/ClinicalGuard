"""
ClinicalGuard — Streamlit UI
AI-Powered Clinical Coding with Evidence Validation
"""

import json
import streamlit as st
import pandas as pd
from pipeline import list_cases, get_patient_case, run_pipeline

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ClinicalGuard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — dark theme consistent
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1B3A5C;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-top: 0;
    }
    .note-preview {
        background: #1a1a2e;
        color: #a8b2c1;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
        padding: 12px 16px;
        border-radius: 6px;
        border-left: 3px solid #e74c3c;
        margin: 10px 0 15px 0;
        line-height: 1.5;
        white-space: pre-wrap;
    }
    .note-label {
        font-size: 0.75rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 4px;
    }
    .alert-banner {
        background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
        color: white;
        padding: 18px 24px;
        border-radius: 8px;
        margin: 15px 0;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 700;
        box-shadow: 0 4px 12px rgba(220, 53, 69, 0.4);
    }
    .alert-banner-safe {
        background: linear-gradient(135deg, #28a745 0%, #218838 100%);
        color: white;
        padding: 18px 24px;
        border-radius: 8px;
        margin: 15px 0;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 700;
        box-shadow: 0 4px 12px rgba(40, 167, 69, 0.4);
    }
    .status-confirmed {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 10px 15px;
        margin: 5px 0;
        border-radius: 4px;
        color: #155724;
    }
    .status-flagged {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 12px 15px;
        margin: 5px 0;
        border-radius: 4px;
        color: #721c24;
        font-weight: 500;
    }
    .status-review {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 10px 15px;
        margin: 5px 0;
        border-radius: 4px;
        color: #856404;
    }
    .status-unvalidated {
        background-color: #e2e3e5;
        border-left: 4px solid #6c757d;
        padding: 10px 15px;
        margin: 5px 0;
        border-radius: 4px;
        color: #383d41;
    }
    .lab-value-table {
        width: 100%;
        border-collapse: collapse;
        margin: 8px 0;
        font-size: 0.9rem;
    }
    .lab-value-table th {
        background: #1B3A5C;
        color: white;
        padding: 8px 12px;
        text-align: left;
        font-weight: 600;
    }
    .lab-value-table td {
        padding: 6px 12px;
        border-bottom: 1px solid #dee2e6;
    }
    .lab-abnormal {
        color: #dc3545;
        font-weight: 700;
    }
    .lab-normal {
        color: #28a745;
        font-weight: 600;
    }
    .lab-critical {
        color: #dc3545;
        font-weight: 700;
        background: #fff5f5;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<p class="main-header">🏥 ClinicalGuard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Clinical Coding with Evidence Validation</p>', unsafe_allow_html=True)
st.markdown("---")

# ---------------------------------------------------------------------------
# Sidebar — Patient Selection
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Patient Selection")

    demo_cases = {
        "CASE_00210 — Sepsis (66F, 36 diagnoses)": 171106.0,
        "CASE_00796 — Pneumonia (84M, 20 diagnoses)": 109697.0,
        "CASE_01469 — CAD + Diabetes (64F, 5 diagnoses)": 164107.0,
    }

    st.markdown("**Recommended Demo Cases:**")
    selected_demo = st.selectbox(
        "Choose a case",
        options=list(demo_cases.keys()),
        index=0,
    )
    hadm_id = demo_cases[selected_demo]

    st.markdown("---")
    st.markdown("**Or browse all 2,000 cases:**")

    if st.checkbox("Show full case list"):
        cases = list_cases()
        case_df = pd.DataFrame(cases)
        case_df = case_df.sort_values("n_diag", ascending=False)
        st.dataframe(
            case_df[["case_id", "age", "gender", "admission_diagnosis", "n_diag", "n_labs", "n_rx"]],
            height=300,
            use_container_width=True,
        )
        custom_hadm = st.number_input("Enter hadm_id:", value=int(hadm_id), step=1)
        hadm_id = float(custom_hadm)

    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    **ClinicalGuard** reads hospital discharge notes, extracts ICD-9 diagnosis codes with reasoning, then validates those codes against the patient's lab results and prescriptions.

    Built with **LangGraph** + **Claude** for the UofT Healthcare AI Hackathon 2026.

    *By Sathian S.*
    """)

# ---------------------------------------------------------------------------
# Main area — Patient info + note preview
# ---------------------------------------------------------------------------
patient = get_patient_case(hadm_id)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Patient", patient["case_id"])
with col2:
    st.metric("Age / Gender", f"{patient['age']} / {patient['gender']}")
with col3:
    st.metric("Admission", patient["admission_diagnosis"][:30])
with col4:
    st.metric("Known Diagnoses", len(patient["actual_diagnoses"]))

# --- IMPROVEMENT 4: Note snippet before the button ---
note_text = patient["discharge_summary"]
# Extract a messy, interesting chunk (skip the header, grab the HPI)
snippet_lines = note_text[:800]
# Try to find a meaty section
for marker in ["History of Present Illness", "Chief Complaint", "HPI:", "HISTORY"]:
    idx = note_text.find(marker)
    if idx >= 0:
        snippet_lines = note_text[idx:idx+400]
        break

st.markdown('<p class="note-label">Raw Clinical Note Preview</p>', unsafe_allow_html=True)
st.markdown(f'<div class="note-preview">{snippet_lines.strip()}...</div>', unsafe_allow_html=True)

# Full note in expander
with st.expander("📋 View Full Discharge Summary", expanded=False):
    st.text_area(
        "Note",
        value=note_text[:8000] + ("\n\n[...truncated for display]" if len(note_text) > 8000 else ""),
        height=300,
        disabled=True,
        label_visibility="collapsed",
    )

st.markdown("---")

# Run pipeline button
if st.button("🚀 Run ClinicalGuard Analysis", type="primary", use_container_width=True):
    with st.status("Running ClinicalGuard pipeline...", expanded=True) as status:
        st.write("**Step 1:** Parsing discharge note and extracting medical concepts...")
        result = run_pipeline(hadm_id)
        status.update(label="Pipeline complete!", state="complete", expanded=True)

    st.session_state["result"] = result
    st.session_state["hadm_id"] = hadm_id

# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------
if "result" in st.session_state and st.session_state.get("hadm_id") == hadm_id:
    result = st.session_state["result"]
    report = result.get("final_report", {})

    # --- IMPROVEMENT 1: Prominent alert banner ---
    if report and report.get("entries"):
        flagged_count = report.get("flagged", 0)
        review_count = report.get("needs_review", 0)
        total_issues = flagged_count + review_count

        if flagged_count > 0:
            st.markdown(
                f'<div class="alert-banner">⚠️ {flagged_count} CODE{"S" if flagged_count != 1 else ""} '
                f'REQUIRE{"S" if flagged_count == 1 else ""} REVIEW BEFORE SUBMISSION'
                f'{"  |  " + str(review_count) + " additional codes need review" if review_count > 0 else ""}'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="alert-banner-safe">✅ ALL {report.get("total_codes", 0)} CODES VALIDATED — '
                f'NO INCONSISTENCIES DETECTED</div>',
                unsafe_allow_html=True
            )

    st.markdown("## Pipeline Results")

    # Step-by-step logs
    for log in result["step_logs"]:
        with st.expander(f"**{log['step']}:** {log['title']} — {log['summary']}", expanded=(log['step'].startswith("5"))):
            if isinstance(log["detail"], list):
                if log["detail"]:
                    st.json(log["detail"][:20])
            elif isinstance(log["detail"], dict):
                detail_report = log["detail"]
                if "entries" in detail_report:
                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.metric("Total Codes", detail_report.get("total_codes", 0))
                    with m2:
                        st.metric("✅ Confirmed", detail_report.get("confirmed", 0))
                    with m3:
                        st.metric("🔴 Flagged", detail_report.get("flagged", 0))
                    with m4:
                        st.metric("🟡 Review", detail_report.get("needs_review", 0))
                else:
                    st.json(detail_report)

    # --- Validation Report ---
    if report and report.get("entries"):
        st.markdown("---")
        st.markdown("## Validation Report")

        # Get lab validation data for inline display
        lab_validation_data = result.get("lab_validation", [])

        for entry in report["entries"]:
            entry_status = entry["overall_status"]
            icon = {
                "confirmed": "✅",
                "inconsistency_flagged": "🔴",
                "needs_review": "🟡",
                "unvalidated": "⚪"
            }.get(entry_status, "❓")

            css_class = {
                "confirmed": "status-confirmed",
                "inconsistency_flagged": "status-flagged",
                "needs_review": "status-review",
                "unvalidated": "status-unvalidated",
            }.get(entry_status, "status-unvalidated")

            st.markdown(f"""<div class="{css_class}">
                <strong>{icon} {entry['icd9_code']}</strong> — {entry['code_title']}
                &nbsp;&nbsp;|&nbsp;&nbsp;Confidence: <strong>{entry['confidence']}%</strong>
                &nbsp;&nbsp;|&nbsp;&nbsp;Status: <strong>{entry_status.replace('_', ' ').title()}</strong>
            </div>""", unsafe_allow_html=True)

            # --- IMPROVEMENT 2: Auto-expand flagged codes ---
            is_flagged = entry_status in ("inconsistency_flagged", "needs_review")
            if is_flagged:
                # Render details directly — no expander needed for flagged items
                st.markdown(f"**Coding Reasoning:** {entry['reasoning']}")

                # --- IMPROVEMENT 3: Inline lab values with normal ranges ---
                if entry.get("lab_detail"):
                    lab_icon = "✅" if entry["lab_status"] == "supported" else "⚠️"
                    st.markdown(f"**Lab Validation {lab_icon}:** {entry['lab_detail']}")

                    # Find detailed lab values from the validation data
                    matching_lab = next(
                        (lv for lv in lab_validation_data if lv.get("icd9_code") == entry["icd9_code"]),
                        None
                    )
                    if matching_lab and matching_lab.get("key_labs") and isinstance(matching_lab["key_labs"], list):
                        lab_html = '<table class="lab-value-table">'
                        lab_html += '<tr><th>Lab Test</th><th>Patient Value</th><th>Normal Range</th><th>Status</th></tr>'
                        for lab in matching_lab["key_labs"]:
                            if isinstance(lab, dict):
                                interp = lab.get("interpretation", "unknown")
                                css = "lab-critical" if interp == "critical" else ("lab-abnormal" if interp == "abnormal" else "lab-normal")
                                lab_html += (
                                    f'<tr>'
                                    f'<td>{lab.get("lab_name", "—")}</td>'
                                    f'<td class="{css}">{lab.get("patient_value", "—")} {lab.get("unit", "")}</td>'
                                    f'<td>{lab.get("normal_range", "—")}</td>'
                                    f'<td class="{css}">{interp.upper()}</td>'
                                    f'</tr>'
                                )
                        lab_html += '</table>'
                        st.markdown(lab_html, unsafe_allow_html=True)

                if entry.get("rx_detail"):
                    rx_icon = "✅" if entry["rx_status"] == "supported" else "⚠️"
                    st.markdown(f"**Rx Validation {rx_icon}:** {entry['rx_detail']}")

                st.markdown("---")

            else:
                # Confirmed/unvalidated — keep in expander
                with st.expander(f"Details for {entry['icd9_code']}"):
                    st.markdown(f"**Coding Reasoning:** {entry['reasoning']}")
                    if entry.get("lab_detail"):
                        lab_icon = "✅" if entry["lab_status"] == "supported" else "⚠️"
                        st.markdown(f"**Lab Validation {lab_icon}:** {entry['lab_detail']}")
                    if entry.get("rx_detail"):
                        rx_icon = "✅" if entry["rx_status"] == "supported" else "⚠️"
                        st.markdown(f"**Rx Validation {rx_icon}:** {entry['rx_detail']}")

        # --- Comparison with actual codes ---
        st.markdown("---")
        st.markdown("### Comparison: Predicted vs Actual ICD Codes")
        actual = patient["actual_diagnoses"]
        actual_codes = set(d["icd9_code"] for d in actual)
        predicted_codes = set(e["icd9_code"] for e in report["entries"])

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Correctly Predicted**")
            matched = predicted_codes & actual_codes
            for code in sorted(matched):
                title = next((d["short_title"] for d in actual if d["icd9_code"] == code), "")
                st.markdown(f"✅ `{code}` {title}")
            if not matched:
                st.markdown("*None matched exactly*")

        with col2:
            st.markdown("**Predicted but Not in Actual**")
            extra = predicted_codes - actual_codes
            for code in sorted(extra)[:10]:
                e = next((e for e in report["entries"] if e["icd9_code"] == code), {})
                st.markdown(f"➕ `{code}` {e.get('code_title', '')}")
            if not extra:
                st.markdown("*None*")

        with col3:
            st.markdown("**In Actual but Not Predicted**")
            missed = actual_codes - predicted_codes
            for code in sorted(missed)[:10]:
                title = next((d["short_title"] for d in actual if d["icd9_code"] == code), "")
                st.markdown(f"❌ `{code}` {title}")
            if not missed:
                st.markdown("*All covered*")

else:
    # Show placeholder
    st.info("👆 Select a patient case and click **Run ClinicalGuard Analysis** to begin.")

    st.markdown("### How ClinicalGuard Works")
    st.markdown("""
    ```
    ┌─────────────┐    ┌──────────────┐    ┌─────────────────┐
    │  Discharge   │───▶│  Extract     │───▶│  Match ICD-9    │
    │  Summary     │    │  Concepts    │    │  Codes          │
    └─────────────┘    └──────────────┘    └────────┬────────┘
                                                     │
                                    ┌────────────────┼────────────────┐
                                    ▼                                 ▼
                           ┌────────────────┐              ┌─────────────────┐
                           │  Cross-Check   │              │  Cross-Check    │
                           │  Lab Results   │              │  Prescriptions  │
                           └───────┬────────┘              └────────┬────────┘
                                   │                                │
                                   └──────────┬─────────────────────┘
                                              ▼
                                   ┌─────────────────────┐
                                   │  Validation Report   │
                                   │  ✅ 🟡 🔴            │
                                   └─────────────────────┘
    ```
    """)
