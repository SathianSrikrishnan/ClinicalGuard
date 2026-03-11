# Why ClinicalGuard

The problem with existing clinical AI tools like Nuance, Suki, and Aidéo is that they suggest ICD codes from discharge notes but never verify those codes against the patient's own clinical evidence. ClinicalGuard's core differentiator is the consistency validation layer — each predicted code is cross-referenced against lab results and prescription records before it reaches billing. This cross-system validation does not exist in any current coding workflow.

# Design Decisions

The pipeline is intentionally human-in-the-loop — ClinicalGuard flags codes for review and never submits autonomously. Confidence scoring is conservative by design; the system flags when evidence is absent, not only when evidence actively contradicts. The comparison table showing predicted vs actual human-assigned codes is built in so the system's reasoning is auditable.

# Why This Dataset

MIMIC-III was chosen because it contains real discharge notes, real lab values, and real prescription records for the same admissions — making genuine cross-validation possible rather than simulated.
