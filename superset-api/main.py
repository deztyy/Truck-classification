import os
from typing import Tuple

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPERSET_URL = os.getenv("SUPERSET_URL", "http://superset:8088")
SUPERSET_USER = os.getenv("SUPERSET_USER", "admin")
SUPERSET_PASS = os.getenv("SUPERSET_PASS", "admin123")
SUPERSET_EMBEDDED_DASHBOARD_ID = os.getenv(
    "SUPERSET_EMBEDDED_DASHBOARD_ID",
    os.getenv("SUPERSET_DASHBOARD_UUID", ""),
).strip()
SUPERSET_DASHBOARD_UUID = os.getenv("SUPERSET_DASHBOARD_UUID", "").strip()
SUPERSET_DASHBOARD_SLUG = os.getenv("SUPERSET_DASHBOARD_SLUG", "").strip()
SUPERSET_ALLOWED_EMBED_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "SUPERSET_ALLOWED_EMBED_ORIGINS",
        "http://localhost:8501,http://127.0.0.1:8501",
    ).split(",")
    if origin.strip()
]


async def get_access_token(client: httpx.AsyncClient) -> str:
    login = await client.post(f"{SUPERSET_URL}/api/v1/security/login", json={
        "username": SUPERSET_USER,
        "password": SUPERSET_PASS,
        "provider": "db"
    })
    if login.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Login failed: {login.text}")
    return login.json()["access_token"]


async def get_csrf_token(client: httpx.AsyncClient, access_token: str) -> Tuple[str, httpx.Cookies]:
    csrf_res = await client.get(
        f"{SUPERSET_URL}/api/v1/security/csrf_token/",
        headers={"Authorization": f"Bearer {access_token}"}
    )
    if csrf_res.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Get CSRF token failed: {csrf_res.text}")

    csrf_token = csrf_res.json().get("result", "")
    if not csrf_token:
        raise HTTPException(status_code=500, detail="Get CSRF token failed: empty token")

    return csrf_token, csrf_res.cookies


async def get_or_create_embedded_uuid(
    client: httpx.AsyncClient,
    access_token: str,
    csrf_token: str,
    cookies: httpx.Cookies,
    dashboard_id: int,
) -> str:
    embed_res = await client.get(
        f"{SUPERSET_URL}/api/v1/dashboard/{dashboard_id}/embedded",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    if embed_res.status_code == 200:
        uuid = embed_res.json().get("result", {}).get("uuid", "")
        if uuid:
            return uuid

    create_embed_res = await client.post(
        f"{SUPERSET_URL}/api/v1/dashboard/{dashboard_id}/embedded",
        headers={
            "Authorization": f"Bearer {access_token}",
            "X-CSRFToken": csrf_token,
            "Referer": SUPERSET_URL,
            "Content-Type": "application/json",
        },
        cookies=cookies,
        json={"allowed_domains": SUPERSET_ALLOWED_EMBED_ORIGINS},
    )
    if create_embed_res.status_code in (200, 201):
        uuid = create_embed_res.json().get("result", {}).get("uuid", "")
        if uuid:
            return uuid

    raise HTTPException(
        status_code=500,
        detail=(
            "Fetch embedded UUID failed. "
            f"POST /embedded -> {create_embed_res.status_code}: {create_embed_res.text}; "
            f"GET /embedded -> {embed_res.status_code}: {embed_res.text}"
        ),
    )


async def get_dashboard_uuid(
    client: httpx.AsyncClient,
    access_token: str,
    csrf_token: str,
    cookies: httpx.Cookies,
) -> str:
    """Resolve embedded dashboard UUID using env and auto-enable embedding if required."""
    if SUPERSET_EMBEDDED_DASHBOARD_ID:
        return SUPERSET_EMBEDDED_DASHBOARD_ID

    res = await client.get(
        f"{SUPERSET_URL}/api/v1/dashboard/",
        params={"page_size": 1000},
        headers={"Authorization": f"Bearer {access_token}"}
    )
    if res.status_code != 200:
        raise HTTPException(status_code=500, detail=f"List dashboards failed: {res.text}")

    dashboards = res.json().get("result", [])
    if not dashboards:
        raise HTTPException(
            status_code=500,
            detail="No dashboards found. Set SUPERSET_EMBEDDED_DASHBOARD_ID from Superset Embed dialog.",
        )

    selected_dashboard = None

    if SUPERSET_DASHBOARD_UUID:
        selected_dashboard = next(
            (d for d in dashboards if str(d.get("uuid", "")).strip() == SUPERSET_DASHBOARD_UUID),
            None,
        )
        if not selected_dashboard:
            raise HTTPException(
                status_code=500,
                detail=f"Configured SUPERSET_DASHBOARD_UUID '{SUPERSET_DASHBOARD_UUID}' not found",
            )

    if not selected_dashboard and SUPERSET_DASHBOARD_SLUG:
        selected_dashboard = next(
            (d for d in dashboards if str(d.get("slug", "")).strip() == SUPERSET_DASHBOARD_SLUG),
            None,
        )
        if not selected_dashboard:
            raise HTTPException(
                status_code=500,
                detail=f"Configured SUPERSET_DASHBOARD_SLUG '{SUPERSET_DASHBOARD_SLUG}' not found",
            )

    if not selected_dashboard:
        selected_dashboard = dashboards[0]

    dashboard_id = selected_dashboard["id"]

    return await get_or_create_embedded_uuid(
        client=client,
        access_token=access_token,
        csrf_token=csrf_token,
        cookies=cookies,
        dashboard_id=dashboard_id,
    )

@app.get("/guest-token")
async def get_guest_token():
    async with httpx.AsyncClient() as client:
        # 1. Login
        access_token = await get_access_token(client)

        # 2. ขอ CSRF token
        csrf_token, cookies = await get_csrf_token(client, access_token)

        # 3. ดึง UUID อัตโนมัติ
        dashboard_uuid = await get_dashboard_uuid(client, access_token, csrf_token, cookies)

        # 4. ขอ guest token
        guest = await client.post(
            f"{SUPERSET_URL}/api/v1/security/guest_token/",
            headers={
                "Authorization": f"Bearer {access_token}",
                "X-CSRFToken": csrf_token,
                "Referer": SUPERSET_URL,
                "Content-Type": "application/json",
            },
            cookies=cookies,
            json={
                "resources": [{"type": "dashboard", "id": dashboard_uuid}],
                "rls": [],
                "user": {"username": "guest", "first_name": "Guest", "last_name": "User"}
            }
        )

        if guest.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Guest token failed: {guest.text}")

        return {"token": guest.json()["token"], "uuid": dashboard_uuid}