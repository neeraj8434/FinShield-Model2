"""Lightweight file-based history store for KYC verification results.

Each submission is saved as a JSON file in the history directory.
A small thumbnail of the document image is stored as base64.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from .config import MODEL2_ROOT

logger = logging.getLogger(__name__)

HISTORY_DIR = Path(os.getenv(
    "FINSHIELD_HISTORY_DIR",
    str(MODEL2_ROOT / "fusion" / "history_data"),
))
HISTORY_DIR.mkdir(parents=True, exist_ok=True)

THUMB_SIZE = (120, 120)


def _make_thumbnail(pil_image: Image.Image) -> str:
    """Create a small base64 JPEG thumbnail."""
    img = pil_image.copy()
    img.thumbnail(THUMB_SIZE, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def save(
    result_dict: Dict[str, Any],
    form_name: str,
    form_dob: str,
    form_id_number: str,
    doc_type: str,
    doc_filename: str,
    vid_filename: str,
    doc_image: Optional[Image.Image] = None,
) -> str:
    """Save a verification result to history. Returns the entry ID."""
    entry_id = uuid.uuid4().hex[:12]
    entry = {
        "id": entry_id,
        "timestamp": time.time(),
        "form_name": form_name,
        "form_dob": form_dob,
        "form_id_number": form_id_number,
        "doc_type": doc_type,
        "doc_filename": doc_filename,
        "vid_filename": vid_filename,
        "decision": result_dict.get("decision", "UNKNOWN"),
        "fused_risk": result_dict.get("fused_risk", 0),
        "thumbnail": _make_thumbnail(doc_image) if doc_image else None,
        "result": result_dict,
    }

    path = HISTORY_DIR / f"{entry_id}.json"
    path.write_text(json.dumps(entry, default=str), encoding="utf-8")
    logger.info("History saved: %s (%s)", entry_id, form_name)
    return entry_id


def list_all() -> List[Dict[str, Any]]:
    """Return all history entries, newest first (without full result)."""
    entries = []
    for f in HISTORY_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text("utf-8"))
            entries.append({
                "id": data["id"],
                "timestamp": data["timestamp"],
                "form_name": data.get("form_name", ""),
                "doc_type": data.get("doc_type", ""),
                "doc_filename": data.get("doc_filename", ""),
                "vid_filename": data.get("vid_filename", ""),
                "decision": data.get("decision", ""),
                "fused_risk": data.get("fused_risk", 0),
                "thumbnail": data.get("thumbnail"),
            })
        except Exception as exc:
            logger.warning("Skipping corrupt history file %s: %s", f.name, exc)
    entries.sort(key=lambda e: e["timestamp"], reverse=True)
    return entries


def get(entry_id: str) -> Optional[Dict[str, Any]]:
    """Return a full history entry including the result."""
    path = HISTORY_DIR / f"{entry_id}.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text("utf-8"))
    except Exception:
        return None


def delete(entry_id: str) -> bool:
    """Delete a history entry. Returns True if deleted."""
    path = HISTORY_DIR / f"{entry_id}.json"
    if path.is_file():
        path.unlink()
        logger.info("History deleted: %s", entry_id)
        return True
    return False


def clear_all() -> int:
    """Delete all history entries. Returns count deleted."""
    count = 0
    for f in HISTORY_DIR.glob("*.json"):
        f.unlink()
        count += 1
    return count
