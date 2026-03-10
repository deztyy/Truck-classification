#!/bin/sh
set -eu

DATASET_EXPORT_PATH="${SUPERSET_DATASET_EXPORT_PATH:-/imports/dataset_export_20260205T094311.zip}"
DASHBOARD_EXPORT_PATH="${SUPERSET_DASHBOARD_EXPORT_PATH:-/imports/dashboard_export_20260310T045801.zip}"
IMPORT_USERNAME="${SUPERSET_IMPORT_USERNAME:-admin}"
SUPERSET_ANALYTICS_DB_URI="${SUPERSET_ANALYTICS_DB_URI:-postgresql://postgres:postgres1234@db:5432/vehicle_db}"
SUPERSET_DASHBOARD_SLUG="${SUPERSET_DASHBOARD_SLUG:-}"
SUPERSET_ALLOWED_EMBED_ORIGINS="${SUPERSET_ALLOWED_EMBED_ORIGINS:-http://localhost:8501,http://127.0.0.1:8501}"

patch_dataset_zip() {
  src_zip="$1"
  out_zip="$2"
  export SRC_ZIP="$src_zip"
  export OUT_ZIP="$out_zip"
  export PATCH_DB_URI="$SUPERSET_ANALYTICS_DB_URI"

  python - <<'PY'
import os
import re
import zipfile

src_zip = os.environ["SRC_ZIP"]
out_zip = os.environ["OUT_ZIP"]
db_uri = os.environ["PATCH_DB_URI"]

with zipfile.ZipFile(src_zip, "r") as zin, zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zout:
    for info in zin.infolist():
        data = zin.read(info.filename)
        normalized = info.filename.replace("\\", "/")

        if normalized.endswith(".yaml") and "/databases/" in normalized:
            text = data.decode("utf-8")
            text = re.sub(r"(?m)^sqlalchemy_uri:\s*.*$", f"sqlalchemy_uri: {db_uri}", text)
            data = text.encode("utf-8")

        zout.writestr(info, data)
PY
}

echo "Starting Superset asset import..."

if [ ! -f "$DATASET_EXPORT_PATH" ] && [ ! -f "$DASHBOARD_EXPORT_PATH" ]; then
  echo "No export files found. Skipping import."
  exit 0
fi

if [ -f "$DATASET_EXPORT_PATH" ]; then
  echo "Importing datasources from: $DATASET_EXPORT_PATH"
  PATCHED_DATASET_EXPORT_PATH="/tmp/dataset_export_patched.zip"
  patch_dataset_zip "$DATASET_EXPORT_PATH" "$PATCHED_DATASET_EXPORT_PATH"
  superset import-datasources -p "$PATCHED_DATASET_EXPORT_PATH" -u "$IMPORT_USERNAME"
else
  echo "Dataset export file not found: $DATASET_EXPORT_PATH"
fi

if [ -f "$DASHBOARD_EXPORT_PATH" ]; then
  echo "Importing dashboards from: $DASHBOARD_EXPORT_PATH"
  superset import-dashboards -p "$DASHBOARD_EXPORT_PATH" -u "$IMPORT_USERNAME"
else
  echo "Dashboard export file not found: $DASHBOARD_EXPORT_PATH"
fi

echo "Normalizing imported dashboard metadata (published/slug)..."
export NORMALIZE_DB_URI="${SQLALCHEMY_DATABASE_URI:-}"
export NORMALIZE_DASHBOARD_SLUG="$SUPERSET_DASHBOARD_SLUG"
export NORMALIZE_ALLOWED_ORIGINS="$SUPERSET_ALLOWED_EMBED_ORIGINS"
python - <<'PY'
import os
import uuid
from sqlalchemy import create_engine, text

db_uri = os.environ.get("NORMALIZE_DB_URI", "").strip()
slug = os.environ.get("NORMALIZE_DASHBOARD_SLUG", "").strip()
allowed_origins = os.environ.get("NORMALIZE_ALLOWED_ORIGINS", "").strip()

if not db_uri:
  raise SystemExit("Missing SQLALCHEMY_DATABASE_URI for dashboard normalization")

engine = create_engine(db_uri)
with engine.begin() as conn:
  conn.execute(text("UPDATE dashboards SET published = true WHERE published = false"))

  target_dashboard_id = None

  if slug:
    slug_exists = conn.execute(
      text("SELECT COUNT(*) FROM dashboards WHERE slug = :slug"),
      {"slug": slug},
    ).scalar_one()
    if not slug_exists:
      empty_slug_dashboard = conn.execute(
        text("SELECT id FROM dashboards WHERE slug IS NULL OR slug = '' ORDER BY id LIMIT 1")
      ).scalar()
      if empty_slug_dashboard is not None:
        conn.execute(
          text("UPDATE dashboards SET slug = :slug WHERE id = :dashboard_id"),
          {"slug": slug, "dashboard_id": int(empty_slug_dashboard)},
        )

    target_dashboard_id = conn.execute(
      text("SELECT id FROM dashboards WHERE slug = :slug ORDER BY id LIMIT 1"),
      {"slug": slug},
    ).scalar()

  if target_dashboard_id is None:
    target_dashboard_id = conn.execute(
      text("SELECT id FROM dashboards ORDER BY id LIMIT 1")
    ).scalar()

  if target_dashboard_id is not None:
    embedded_uuid = conn.execute(
      text("SELECT uuid FROM embedded_dashboards WHERE dashboard_id = :dashboard_id"),
      {"dashboard_id": int(target_dashboard_id)},
    ).scalar()

    if embedded_uuid is None:
      conn.execute(
        text(
          "INSERT INTO embedded_dashboards (uuid, dashboard_id, allow_domain_list) "
          "VALUES (:uuid, :dashboard_id, :allow_domain_list)"
        ),
        {
          "uuid": str(uuid.uuid4()),
          "dashboard_id": int(target_dashboard_id),
          "allow_domain_list": allowed_origins,
        },
      )
    else:
      conn.execute(
        text(
          "UPDATE embedded_dashboards "
          "SET allow_domain_list = :allow_domain_list "
          "WHERE dashboard_id = :dashboard_id"
        ),
        {
          "allow_domain_list": allowed_origins,
          "dashboard_id": int(target_dashboard_id),
        },
      )
PY

echo "Superset asset import completed."
