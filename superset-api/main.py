import os

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

async def get_access_token(client: httpx.AsyncClient) -> str:
    login = await client.post(f"{SUPERSET_URL}/api/v1/security/login", json={
        "username": SUPERSET_USER,
        "password": SUPERSET_PASS,
        "provider": "db"
    })
    if login.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Login failed: {login.text}")
    return login.json()["access_token"]

async def get_dashboard_uuid(client: httpx.AsyncClient, access_token: str) -> str:
    """ดึง embedded UUID โดยใช้ embedded id จาก env ก่อน จากนั้นค่อย fallback ด้วย slug/uuid"""
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
    
    # ดึง embedded UUID
    embed_res = await client.get(
        f"{SUPERSET_URL}/api/v1/dashboard/{dashboard_id}/embedded",
        headers={"Authorization": f"Bearer {access_token}"}
    )
    if embed_res.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Fetch embedded UUID failed: {embed_res.text}")

    uuid = embed_res.json().get("result", {}).get("uuid")
    if not uuid:
        raise HTTPException(status_code=500, detail="Dashboard embedding not enabled")
    return uuid

@app.get("/guest-token")
async def get_guest_token():
    async with httpx.AsyncClient() as client:
        # 1. Login
        access_token = await get_access_token(client)

        # 2. ดึง UUID อัตโนมัติ
        dashboard_uuid = await get_dashboard_uuid(client, access_token)

        # 3. ขอ CSRF token
        csrf_res = await client.get(
            f"{SUPERSET_URL}/api/v1/security/csrf_token/",
            headers={"Authorization": f"Bearer {access_token}"}
        )
        csrf_token = csrf_res.json()["result"]
        cookies = csrf_res.cookies

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