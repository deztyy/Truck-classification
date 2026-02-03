# Superset configuration overrides for embedding dashboards

FEATURE_FLAGS = {
    "EMBEDDED_SUPERSET": True,
}

# Allow embedding in iframes for local Streamlit app
X_FRAME_OPTIONS = "ALLOWALL"

# Relax CSP for local development
TALISMAN_ENABLED = True
TALISMAN_CONFIG = {
    "content_security_policy": {
        "default-src": ["'self'", "'unsafe-inline'", "'unsafe-eval'"],
        "img-src": ["'self'", "data:", "blob:", "http:", "https:"],
        "frame-ancestors": [
            "'self'",
            "http://localhost:8501",
            "http://127.0.0.1:8501",
        ],
        "frame-src": ["'self'", "http:", "https:"],
        "connect-src": ["'self'", "http:", "https:"],
        "style-src": ["'self'", "'unsafe-inline'", "http:", "https:"],
        "script-src": ["'self'", "'unsafe-inline'", "'unsafe-eval'", "http:", "https:"],
        "font-src": ["'self'", "data:", "http:", "https:"],
    },
    "force_https": False,
}
