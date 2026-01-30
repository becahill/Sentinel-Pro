# Threat Model (Lightweight)

## What it detects

- Toxic or abusive language (classifier-based)
- PII leaks (email/phone regex)
- Refusals / safety deflections
- Self-harm references (keyword heuristics)
- Jailbreak attempts (keyword heuristics)
- Bias indicators (keyword + protected group co-occurrence)

## What it does NOT guarantee

- It is not a content policy enforcement system.
- It cannot reliably detect nuanced, adversarial, or multi-lingual harm.
- It does not mitigate harm; it only flags for review.

## Known failure modes

- False positives on benign mentions (e.g., “suicide prevention hotline”).
- False negatives on obfuscated or coded content.
- Model-dependent toxicity scores (drift between versions).

## Mitigations

- Keep a human-in-the-loop for final decisions.
- Calibrate thresholds per project.
- Add specialized detectors for your domain.
- Redact PII before persistence to reduce data exposure.
