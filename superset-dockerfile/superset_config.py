import os

# Superset configuration overrides for embedding dashboards

SQLALCHEMY_DATABASE_URI = os.getenv(
    "SQLALCHEMY_DATABASE_URI",
    os.getenv("SUPERSET_SQLALCHEMY_DATABASE_URI", "sqlite:////var/lib/superset/superset.db"),
)

FEATURE_FLAGS = {
    "EMBEDDED_SUPERSET": True,
}

# Allow embedding in iframes
X_FRAME_OPTIONS = "ALLOWALL"

# Enable public role for embedded dashboards
PUBLIC_ROLE_LIKE = "Gamma"

# Keep CSRF enabled for security, but with exemptions
WTF_CSRF_ENABLED = True
WTF_CSRF_EXEMPT_LIST = [
    "superset.views.core.log",
    "superset.security.api.guest_token",
]

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
            "*",
        ],
        "frame-src": ["'self'", "http:", "https:"],
        "connect-src": ["'self'", "http:", "https:"],
        "style-src": ["'self'", "'unsafe-inline'", "http:", "https:"],
        "script-src": ["'self'", "'unsafe-inline'", "'unsafe-eval'", "http:", "https:"],
        "font-src": ["'self'", "data:", "http:", "https:"],
    },
    "force_https": False,
}
