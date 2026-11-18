"""Allow ``python -m fusion`` to start the Flask demo."""
from fusion.app import app

if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
