# ClinicalGuard

**AI-Powered Clinical Coding with Evidence Validation**

## The Problem

Every day in hospitals across North America, trained medical coders read thousands of discharge notes and manually assign ICD billing codes — a slow, error-prone process where a single wrong code costs ~$25 to rework and ~30% of insurance claim denials stem from coding errors. Worse, nobody routinely cross-validates whether the assigned codes actually match the patient's own lab results and prescriptions, meaning clinically inconsistent codes slip through to billing unchecked.

## The Solution

ClinicalGuard is a 5-step LangGraph pipeline that goes beyond simple code extraction. It reads hospital discharge notes, extracts ICD-9 diagnosis codes with full reasoning, and then validates each code against the patient's actual clinical evidence — lab results and prescription records — flagging inconsistencies for human review before they ever reach billing. The system doesn't replace coders; it gives them a first draft with evidence validation, turning a 15-minute manual task into a 2-minute review.

## How It Works

The pipeline has five steps. Each one is a separate node in a LangGraph graph, so you can inspect what happened at every stage.

1. **Parse Note** — I feed the raw discharge summary into Claude and ask it to pull out every medical concept it can find. Diagnoses, symptoms, procedures, meds mentioned — each one comes back with a confidence level and the exact quote from the note.
2. **Match ICD Codes** — Those extracted concepts get matched against a 2,390-code ICD-9 dictionary. For each diagnosis the system returns the best-fit code, a confidence score, and a short explanation of why it picked that code.
3. **Lab Validation** — This is where it gets interesting. Each predicted code gets cross-referenced against the patient's actual lab results. Coded for diabetes? Let me check the glucose. Coded for renal disease? What does the creatinine say? The actual values and normal ranges are shown inline.
4. **Prescription Validation** — Same idea but for meds. If you're coded for diabetes, are you on metformin or insulin? If you're coded for hypertension, where are the antihypertensives? Missing expected medications get flagged.
5. **Final Report** — Everything rolls up into a validation report. Each code gets one of three labels: Confirmed (evidence supports it), Needs Review (not enough data or low confidence), or Inconsistency Flagged (the clinical evidence actually contradicts the code).

## Demo Cases

- **CASE_01469** — 64F, Coronary Artery Disease + Diabetes. Clean validation: all 7 codes confirmed by lab and prescription evidence. Shows the system working correctly on a straightforward case.
- **CASE_00210** — 66F, Sepsis with 36 diagnoses. 3 codes flagged as inconsistent, including an ESRD flag based on creatinine mismatch. Demonstrates the consistency checker catching real clinical discrepancies in a complex patient.
- **CASE_00796** — 84M, Pneumonia with 20 diagnoses. 1 flag: hyperlipidemia coded but unsupported by prescriptions (no statins found). Shows the prescription cross-reference catching a gap.

## Tech Stack

LangGraph · Claude API (Anthropic) · Streamlit · Python · pandas · MIMIC-III clinical data (2,000 de-identified hospital admissions)

## How to Run

```bash
git clone <repo-url>
cd ClinicalGuard
pip install -r requirements.txt
echo "ANTHROPIC_API_KEY=your-key-here" > .env
streamlit run app.py
```

## Dataset

| Table | Records | Description |
|-------|---------|-------------|
| clinical_cases | 2,000 | Discharge summaries with patient demographics |
| diagnoses_subset | 23,428 | ICD-9 codes assigned to each admission |
| diagnosis_dictionary | 2,390 | ICD-9 code definitions |
| labs_subset | 841,507 | Laboratory test results |
| lab_dictionary | 554 | Lab test definitions |
| prescriptions_subset | 153,433 | Medication records |

---

Built by Sathian Srikrishnan for the UofT Healthcare AI Hackathon 2026 (BRAVE Career).
