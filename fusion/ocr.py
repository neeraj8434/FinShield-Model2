"""OCR wrapper using EasyOCR (lazy-loaded, pretrained).

Extracts raw text from a document image, then uses regex patterns to
pull out Name, DOB, and ID number fields for Aadhaar and PAN cards.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Module-level singleton — created on first call
_reader = None


def _get_reader():
    """Lazy-init the EasyOCR reader (downloads models on first run)."""
    global _reader
    if _reader is None:
        try:
            import easyocr  # noqa: delayed import
            _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            logger.info("EasyOCR reader initialised (English, CPU)")
        except ImportError:
            logger.warning(
                "easyocr is not installed — OCR will return empty text. "
                "Install with: pip install easyocr"
            )
            return None
    return _reader


def extract_text(image) -> str:
    """Run OCR on a PIL Image or numpy array and return joined text.

    Parameters
    ----------
    image : PIL.Image.Image | numpy.ndarray | str (file path)

    Returns
    -------
    str  — full OCR text, lines joined by ``\\n``.
    """
    import numpy as np
    from PIL import Image

    reader = _get_reader()
    if reader is None:
        return ""

    # Convert PIL → numpy if needed
    if isinstance(image, Image.Image):
        image = np.array(image)

    try:
        results = reader.readtext(image, detail=0, paragraph=True)
        return "\n".join(results)
    except Exception as exc:
        logger.error("OCR failed: %s", exc)
        return ""


# ── Field extraction patterns ────────────────────────────────────────

# Aadhaar: 12 digits, usually formatted as XXXX XXXX XXXX (same line only)
_AADHAAR_PAT = re.compile(r"\b(\d{4}[ \t]?\d{4}[ \t]?\d{4})\b")

# Aadhaar masked: xxxxxxxx1234 or XXXX XXXX 1234 (DigiLocker format)
_AADHAAR_MASKED_PAT = re.compile(
    r"[xX]{4,8}\s?[xX]*\s?(\d{4})\b"
)

# PAN: AAAAA9999A  (5 letters, 4 digits, 1 letter)
_PAN_PAT = re.compile(r"\b([A-Z]{5}\d{4}[A-Z])\b")

# Date patterns: DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY, YYYY-MM-DD (DigiLocker)
_DATE_PAT = re.compile(r"\b(\d{2}[/\-\.]\d{2}[/\-\.]\d{4})\b")
_DATE_ISO_PAT = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")

# Name heuristic: line after "Name" label (case-insensitive)
_NAME_PAT = re.compile(r"(?:name|naam)\s*[:\-]?\s*(.+)", re.IGNORECASE)

# DigiLocker-style: name is typically the first line of Latin-alphabet text
# that looks like a person name (2-4 capitalized words, no digits)
_PERSON_NAME_PAT = re.compile(r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s*$", re.MULTILINE)

# DOB label
_DOB_LABEL_PAT = re.compile(
    r"(?:date\s*of\s*birth|dob|d\.o\.b|birth|जन्म\s*तिथि)\s*[:\-]?\s*(.+)",
    re.IGNORECASE,
)


def extract_fields(ocr_text: str, doc_type: str = "aadhaar") -> Dict[str, str]:
    """Parse OCR text into structured fields.

    Parameters
    ----------
    ocr_text : str
        Raw OCR output.
    doc_type : str
        ``"aadhaar"`` or ``"pan"``.

    Returns
    -------
    dict  — keys: ``"name"``, ``"dob"``, ``"id_number"``; values may be
    empty strings when not found.
    """
    fields: Dict[str, str] = {"name": "", "dob": "", "id_number": ""}

    # ── ID number ────────────────────────────────────────────────────
    if doc_type.lower() == "pan":
        m = _PAN_PAT.search(ocr_text.upper())
        if m:
            fields["id_number"] = m.group(1)
    else:
        m = _AADHAAR_PAT.search(ocr_text)
        if m:
            fields["id_number"] = m.group(1).replace(" ", "")
        else:
            # DigiLocker masked format: xxxxxxxx1234
            m2 = _AADHAAR_MASKED_PAT.search(ocr_text)
            if m2:
                fields["id_number"] = "XXXXXXXX" + m2.group(1)

    # ── DOB ──────────────────────────────────────────────────────────
    dob_match = _DOB_LABEL_PAT.search(ocr_text)
    if dob_match:
        date_in_label = _DATE_PAT.search(dob_match.group(1))
        if date_in_label:
            fields["dob"] = date_in_label.group(1)
        if not fields["dob"]:
            # Try ISO format (YYYY-MM-DD) from DigiLocker
            iso = _DATE_ISO_PAT.search(dob_match.group(1))
            if iso:
                fields["dob"] = f"{iso.group(3)}/{iso.group(2)}/{iso.group(1)}"
    if not fields["dob"]:
        # Fallback: first DD/MM/YYYY date found anywhere
        date_match = _DATE_PAT.search(ocr_text)
        if date_match:
            fields["dob"] = date_match.group(1)
    if not fields["dob"]:
        # Fallback: first YYYY-MM-DD date found anywhere (DigiLocker)
        iso_match = _DATE_ISO_PAT.search(ocr_text)
        if iso_match:
            fields["dob"] = f"{iso_match.group(3)}/{iso_match.group(2)}/{iso_match.group(1)}"

    # ── Name ─────────────────────────────────────────────────────────
    name_match = _NAME_PAT.search(ocr_text)
    if not name_match:
        # DigiLocker cards may not have "Name:" label — look for person-name
        # pattern (capitalized words like "Shloka Jaywant Pol")
        pn = _PERSON_NAME_PAT.search(ocr_text)
        if pn:
            name_match = pn
    if name_match:
        raw = name_match.group(1).strip()
        # Remove trailing labels (DOB, Date of Birth, etc.) before splitting
        raw = re.sub(
            r"\b(DOB|D\.?O\.?B|Date\s*of\s*Birth|Date|Birth|Year|YOB)\b.*",
            "", raw, flags=re.IGNORECASE,
        ).strip()
        # Preserve S/O, D/O, W/O, C/O (son/daughter/wife/care of) then
        # split on remaining "/" or "|"
        raw = re.sub(r"\b([SDWC])\s*/\s*([Oo])\b", r"\1_SLASH_\2", raw)
        raw = re.split(r"[/|]", raw)[0].strip()
        raw = raw.replace("_SLASH_", "/")
        # Remove trailing date/number if accidentally captured
        raw = re.sub(r"\d{2}[/\-\.]\d{2}[/\-\.]\d{4}.*", "", raw).strip()
        raw = re.sub(r"\d{4}\s?\d{4}\s?\d{4}.*", "", raw).strip()
        # Remove junk characters (keep letters, spaces, dots, hyphens)
        raw = re.sub(r"[^A-Za-z\s.\-'/]", "", raw).strip()
        # Collapse multiple spaces
        raw = re.sub(r"\s{2,}", " ", raw).strip()
        fields["name"] = raw

    return fields


def run_ocr(image, doc_type: str = "aadhaar") -> Tuple[str, Dict[str, str]]:
    """Convenience: OCR + field extraction in one call.

    Returns (ocr_text, fields_dict).
    """
    text = extract_text(image)
    fields = extract_fields(text, doc_type)
    return text, fields
