"""Semantic validation: compare OCR-extracted fields against KYC form input.

Training-free checks:
 • Name fuzzy match (form vs OCR)
 • DOB match (form vs OCR)
 • ID number format check (Aadhaar = 12 digits, PAN = AAAAA9999A)
 • ID number match (form vs OCR)

**Excluded by design** (user instruction):
 • Verhoeff checksum for Aadhaar
 • PAN 5th-letter must equal first letter of surname
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional

from .config import OCR_FUZZY_THRESHOLD
from .schemas import SemanticCheck, SemanticResult


def _fuzzy_ratio(a: str, b: str) -> float:
    """Case-insensitive similarity ratio in [0, 1].

    OCR on Aadhaar/PAN cards frequently appends noise after the real
    name (e.g. "Kaira" becomes "kaira TA TR").  Pure SequenceMatcher
    penalises the extra text too heavily, so we boost the score when
    the form name appears inside the OCR text — especially as a prefix.
    """
    a_low, b_low = a.lower().strip(), b.lower().strip()
    if not a_low or not b_low:
        return 0.0
    full_ratio = SequenceMatcher(None, a_low, b_low).ratio()
    shorter, longer = (a_low, b_low) if len(a_low) <= len(b_low) else (b_low, a_low)
    if shorter in longer:
        if longer.startswith(shorter):
            # Prefix match — very strong signal (OCR noise is after the name)
            full_ratio = max(full_ratio, 0.90)
        else:
            containment = len(shorter) / len(longer)
            full_ratio = max(full_ratio, 0.5 + 0.5 * containment)
    return full_ratio


def _normalise_date(raw: str) -> str:
    """Convert DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY → DD/MM/YYYY."""
    return re.sub(r"[\-\.]", "/", raw.strip())


def validate(
    ocr_fields: Dict[str, str],
    form_name: str = "",
    form_dob: str = "",
    form_id_number: str = "",
    doc_type: str = "aadhaar",
    fuzzy_threshold: float = OCR_FUZZY_THRESHOLD,
) -> SemanticResult:
    """Run all semantic checks and return an aggregate result.

    Parameters
    ----------
    ocr_fields : dict
        Output of ``ocr.extract_fields()`` — keys ``"name"``, ``"dob"``,
        ``"id_number"``.
    form_name, form_dob, form_id_number : str
        Values typed by the applicant in the KYC form.
    doc_type : str
        ``"aadhaar"`` or ``"pan"``.
    fuzzy_threshold : float
        Minimum SequenceMatcher ratio to count a name as "matching".

    Returns
    -------
    SemanticResult
    """
    checks: List[SemanticCheck] = []
    has_critical = False

    ocr_name = ocr_fields.get("name", "")
    ocr_dob = ocr_fields.get("dob", "")
    ocr_id = ocr_fields.get("id_number", "")

    # ── 1. Name match ────────────────────────────────────────────────
    if form_name and ocr_name:
        ratio = _fuzzy_ratio(form_name, ocr_name)
        if ratio >= fuzzy_threshold:
            checks.append(SemanticCheck(
                "name", "pass",
                f"Name match {ratio:.0%} (form='{form_name}', ocr='{ocr_name}')",
            ))
        else:
            checks.append(SemanticCheck(
                "name", "fail",
                f"Name mismatch {ratio:.0%} (form='{form_name}', ocr='{ocr_name}')",
            ))
            has_critical = True
    elif form_name and not ocr_name:
        checks.append(SemanticCheck("name", "unknown", "OCR could not extract name"))
    elif not form_name:
        checks.append(SemanticCheck("name", "unknown", "No name provided in form"))

    # ── 2. DOB match ─────────────────────────────────────────────────
    if form_dob and ocr_dob:
        fd = _normalise_date(form_dob)
        od = _normalise_date(ocr_dob)
        if fd == od:
            checks.append(SemanticCheck("dob", "pass", f"DOB matches ({fd})"))
        else:
            checks.append(SemanticCheck(
                "dob", "fail",
                f"DOB mismatch (form='{fd}', ocr='{od}')",
            ))
            has_critical = True
    elif form_dob and not ocr_dob:
        checks.append(SemanticCheck("dob", "unknown", "OCR could not extract DOB"))
    elif not form_dob:
        checks.append(SemanticCheck("dob", "unknown", "No DOB provided in form"))

    # ── 3. ID number format ──────────────────────────────────────────
    is_masked = ocr_id.upper().startswith("XXXX") if ocr_id else False
    if ocr_id:
        if doc_type.lower() == "pan":
            if re.fullmatch(r"[A-Z]{5}\d{4}[A-Z]", ocr_id):
                checks.append(SemanticCheck("id_format", "pass", f"PAN format valid ({ocr_id})"))
            else:
                checks.append(SemanticCheck(
                    "id_format", "warn",
                    f"PAN format invalid ({ocr_id}); expected AAAAA9999A",
                ))
        else:  # aadhaar
            digits = re.sub(r"\s", "", ocr_id)
            if re.fullmatch(r"\d{12}", digits):
                checks.append(SemanticCheck("id_format", "pass", "Aadhaar 12-digit format valid"))
            elif is_masked:
                checks.append(SemanticCheck("id_format", "pass", f"Aadhaar masked format (DigiLocker): {ocr_id}"))
            else:
                checks.append(SemanticCheck(
                    "id_format", "warn",
                    f"Aadhaar format issue: '{ocr_id}' is not 12 digits",
                ))
    else:
        checks.append(SemanticCheck("id_format", "unknown", "OCR could not extract ID number"))

    # ── 4. ID number match (form vs OCR) ─────────────────────────────
    if form_id_number and ocr_id:
        form_clean = re.sub(r"\s", "", form_id_number).upper()
        ocr_clean = re.sub(r"\s", "", ocr_id).upper()
        if form_clean == ocr_clean:
            checks.append(SemanticCheck(
                "id_number", "pass",
                f"ID number matches ({form_clean})",
            ))
        elif is_masked and form_clean[-4:] == ocr_clean[-4:]:
            checks.append(SemanticCheck(
                "id_number", "pass",
                f"Last 4 digits match ({form_clean[-4:]}) — masked Aadhaar",
            ))
        else:
            checks.append(SemanticCheck(
                "id_number", "fail",
                f"ID number mismatch (form='{form_clean}', ocr='{ocr_clean}')",
            ))
            has_critical = True
    elif form_id_number and not ocr_id:
        checks.append(SemanticCheck("id_number", "unknown", "OCR could not extract ID to compare"))
    elif not form_id_number:
        checks.append(SemanticCheck("id_number", "unknown", "No ID number provided in form"))

    return SemanticResult(
        checks=checks,
        ocr_text="",  # caller fills if wanted
        ocr_fields=ocr_fields,
        has_critical_fail=has_critical,
    )
