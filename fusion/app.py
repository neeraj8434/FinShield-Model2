"""Flask web app — KYC-style form for the FinShield multimodal demo.

Run:
    cd FinShield-Model2
    python -m fusion.app            # real mode (needs models + deps)
    FINSHIELD_DEMO_MODE=1 python -m fusion.app   # mock mode

Then open http://localhost:5000
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import asdict
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from PIL import Image

from .config import MAX_CONTENT_LENGTH, UPLOAD_FOLDER
from . import history
from .pipeline import run_kyc

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")

app = Flask(
    __name__,
    template_folder=os.path.join(_FRONTEND_DIR, "templates"),
    static_folder=os.path.join(_FRONTEND_DIR, "static"),
)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
CORS(app)

UPLOAD_DIR = Path(UPLOAD_FOLDER)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _save_upload(file_storage, suffix: str) -> str:
    """Save an uploaded file and return the path."""
    fname = f"{uuid.uuid4().hex}{suffix}"
    path = UPLOAD_DIR / fname
    file_storage.save(str(path))
    return str(path)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    """Accept multipart form data with:
        - doc_image (file)          — required
        - video (file)              — required
        - doc_type (str)            — "aadhaar" or "pan"
        - form_name (str)           — applicant's name
        - form_dob (str)            — DD/MM/YYYY
        - form_id_number (str)      — Aadhaar or PAN number
    """
    try:
        # ── Validate uploads ─────────────────────────────────────────
        if "doc_image" not in request.files:
            return jsonify({"error": "No document image uploaded"}), 400
        if "video" not in request.files:
            return jsonify({"error": "No video uploaded"}), 400

        doc_file = request.files["doc_image"]
        vid_file = request.files["video"]

        if doc_file.filename == "" or vid_file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        # Save files
        doc_path = _save_upload(doc_file, Path(doc_file.filename).suffix or ".jpg")
        vid_path = _save_upload(vid_file, Path(vid_file.filename).suffix or ".mp4")

        # Open image
        pil_image = Image.open(doc_path).convert("RGB")

        # Form fields
        doc_type = request.form.get("doc_type", "aadhaar").strip().lower()
        form_name = request.form.get("form_name", "").strip()
        form_dob = request.form.get("form_dob", "").strip()
        form_id_number = request.form.get("form_id_number", "").strip()

        # ── Run pipeline ─────────────────────────────────────────────
        result = run_kyc(
            doc_image=pil_image,
            video_path=vid_path,
            doc_type=doc_type,
            form_name=form_name,
            form_dob=form_dob,
            form_id_number=form_id_number,
        )

        # ── Serialise ────────────────────────────────────────────────
        payload = asdict(result)

        # ── Save to history ──────────────────────────────────────────
        try:
            entry_id = history.save(
                result_dict=payload,
                form_name=form_name,
                form_dob=form_dob,
                form_id_number=form_id_number,
                doc_type=doc_type,
                doc_filename=doc_file.filename,
                vid_filename=vid_file.filename,
                doc_image=pil_image,
            )
            payload["history_id"] = entry_id
        except Exception as exc:
            logger.warning("Failed to save history: %s", exc)

        return jsonify(payload)

    except Exception as exc:
        logger.exception("Prediction failed")
        return jsonify({"error": str(exc)}), 500

    finally:
        # Cleanup uploaded files (best-effort)
        for p in [locals().get("doc_path"), locals().get("vid_path")]:
            if p and os.path.isfile(p):
                try:
                    os.remove(p)
                except OSError:
                    pass


@app.route("/api/history", methods=["GET"])
def get_history():
    """Return all history entries (summary only, newest first)."""
    return jsonify(history.list_all())


@app.route("/api/history/<entry_id>", methods=["GET"])
def get_history_entry(entry_id):
    """Return a single history entry with full result."""
    entry = history.get(entry_id)
    if entry is None:
        return jsonify({"error": "Not found"}), 404
    return jsonify(entry)


@app.route("/api/history/<entry_id>", methods=["DELETE"])
def delete_history_entry(entry_id):
    """Delete a history entry."""
    if history.delete(entry_id):
        return jsonify({"status": "deleted"})
    return jsonify({"error": "Not found"}), 404


@app.route("/api/history", methods=["DELETE"])
def clear_history():
    """Delete all history entries."""
    count = history.clear_all()
    return jsonify({"status": "cleared", "count": count})


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    logger.info("Starting FinShield Fusion demo on http://localhost:%d", port)
    app.run(host="0.0.0.0", port=port, debug=debug)
