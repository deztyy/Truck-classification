#!/usr/bin/env python3
"""
Superset Database Configuration Script
Automatically configures the analytics database connection in Apache Superset
"""

import requests
import time
import sys
from typing import Optional

# Configuration
SUPERSET_URL = "http://localhost:8088"
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"

DATABASE_CONFIG = {
    "database_name": "vehicle_analytics",
    "sqlalchemy_uri": "postgresql://postgres:postgres1234@vid_analytics_db:5432/vehicle_db",
    "driver": "postgresql",
    "extra": {
        "metadata_params": {
            "bind": True
        }
    }
}


class SupersetConfigurator:
    def __init__(self, base_url: str, username: str, password: str):
        self.base_url = base_url
        self.username = username
        self.password = password
        self.session = requests.Session()
        self.csrf_token = None
        self.access_token = None

    def wait_for_superset(self, max_retries: int = 30) -> bool:
        """Wait for Superset to be ready"""
        print("Waiting for Superset to be ready...")
        for attempt in range(max_retries):
            try:
                response = self.session.get(f"{self.base_url}/health")
                if response.status_code == 200:
                    print("✓ Superset is ready!")
                    return True
            except requests.exceptions.ConnectionError:
                pass

            if attempt < max_retries - 1:
                print(f"  Attempt {attempt + 1}/{max_retries}... retrying in 2 seconds")
                time.sleep(2)

        return False

    def login(self) -> bool:
        """Login to Superset and get access token"""
        print(f"\nLogging in as {self.username}...")

        try:
            # Login โดยตรง ไม่ต้องขอ CSRF ก่อน
            login_response = self.session.post(
                f"{self.base_url}/api/v1/security/login",
                json={
                    "username": self.username,
                    "password": self.password,
                    "provider": "db"
                }
            )

            if login_response.status_code == 200:
                self.access_token = login_response.json().get("access_token")
                self.session.headers.update({
                    "Authorization": f"Bearer {self.access_token}"
                })

                # ขอ CSRF token หลัง login
                csrf_response = self.session.get(
                    f"{self.base_url}/api/v1/security/csrf_token/",
                )
                if csrf_response.status_code == 200:
                    self.csrf_token = csrf_response.json().get("result")
                    self.session.headers.update({
                        "X-CSRFToken": self.csrf_token,
                        "Referer": self.base_url,
                    })
                    print("✓ Login successful!")
                    return True
                else:
                    print(f"⚠ Login ok but CSRF failed: {csrf_response.text}")
                    return True  # ยังใช้งานได้โดยไม่มี CSRF สำหรับ read operations

            else:
                print(f"✗ Login failed: {login_response.text}")
                return False

        except Exception as e:
            print(f"✗ Login error: {e}")
            return False

    def check_database_exists(self, db_name: str) -> Optional[int]:
        """Check if database already exists"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/databases?q=(filters:!((col:database_name,opr:eq,value:{db_name})))"
            )
            if response.status_code == 200:
                databases = response.json().get("result", [])
                if databases:
                    return databases[0].get("id")
        except Exception as e:
            print(f"Error checking database: {e}")

        return None

    def add_database(self) -> bool:
        """Add analytics database to Superset"""
        db_name = DATABASE_CONFIG["database_name"]

        print(f"\nAdding database '{db_name}'...")

        # Check if already exists
        existing_id = self.check_database_exists(db_name)
        if existing_id:
            print(f"✓ Database '{db_name}' already exists (ID: {existing_id})")
            return True

        try:
            payload = {
                "database_name": db_name,
                "sqlalchemy_uri": DATABASE_CONFIG["sqlalchemy_uri"],
                "driver": DATABASE_CONFIG["driver"],
                "extra": DATABASE_CONFIG["extra"]
            }

            response = self.session.post(
                f"{self.base_url}/api/v1/databases/",
                json=payload,
                headers={"X-CSRFToken": self.csrf_token} if self.csrf_token else {}
            )

            if response.status_code == 201:
                db_id = response.json().get("id")
                print(f"✓ Database added successfully! (ID: {db_id})")

                # Test connection
                if self._test_database_connection(db_id):
                    return True
                else:
                    print("⚠ Database added but connection test failed")
                    return True  # Database was created

            else:
                print(f"✗ Failed to add database: {response.text}")
                return False

        except Exception as e:
            print(f"✗ Error adding database: {e}")
            return False

    def _test_database_connection(self, db_id: int) -> bool:
        """Test the database connection"""
        print(f"Testing database connection...")
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/databases/{db_id}/test_connection/"
            )
            if response.status_code == 200:
                print("✓ Connection test passed!")
                return True
            else:
                print(f"✗ Connection test failed: {response.text}")
                return False
        except Exception as e:
            print(f"⚠ Could not test connection: {e}")
            return False

    def enable_dashboard_embedding(self) -> Optional[str]:
        """Enable embedding for all dashboards and return UUID of the first one"""
        print(f"\nEnabling embedding for dashboards...")

        try:
            response = self.session.get(f"{self.base_url}/api/v1/dashboard/")
            if response.status_code != 200:
                print(f"✗ Failed to get dashboards: {response.text}")
                return None

            dashboards = response.json().get("result", [])
            if not dashboards:
                print("⚠ No dashboards found, skipping embedding setup")
                return None

            first_uuid = None
            for d in dashboards:
                dashboard_id = d.get("id")
                title = d.get("title", "Unknown")
                print(f"  Enabling embedding for: [{dashboard_id}] {title}")

                embed_response = self.session.post(
                    f"{self.base_url}/api/v1/dashboard/{dashboard_id}/embedded",
                    json={"allowed_domains": []},
                    headers={"X-CSRFToken": self.csrf_token} if self.csrf_token else {}
                )

                if embed_response.status_code in [200, 201]:
                    uuid = embed_response.json().get("result", {}).get("uuid", "")
                    print(f"  ✓ Enabled! UUID: {uuid}")
                    if first_uuid is None:
                        first_uuid = uuid
                else:
                    print(f"  ✗ Failed: {embed_response.text}")

            return first_uuid

        except Exception as e:
            print(f"✗ Error enabling embedding: {e}")
            return None

    def run(self) -> bool:
        """Run the full configuration"""
        print("=" * 60)
        print("Apache Superset Database Configuration")
        print("=" * 60)

        # Wait for Superset
        if not self.wait_for_superset():
            print("✗ Superset did not start within timeout")
            return False

        # Login
        if not self.login():
            print("✗ Failed to login to Superset")
            return False

        # Add database
        if not self.add_database():
            print("✗ Failed to add database")
            return False

        # Enable dashboard embedding
        embedded_uuid = self.enable_dashboard_embedding()
        if embedded_uuid:
            print(f"\n⚠ Update DASHBOARD_UUID in main.py and superset-api/main.py to: {embedded_uuid}")

        print("\n" + "=" * 60)
        print("✓ Configuration complete!")
        print("=" * 60)
        print(f"\nAccess Superset at: {SUPERSET_URL}")
        print(f"Username: {ADMIN_USERNAME}")
        print(f"Password: {ADMIN_PASSWORD}")
        print(f"\nDatabase connected: {DATABASE_CONFIG['database_name']}")
        if embedded_uuid:
            print(f"Dashboard UUID: {embedded_uuid}")

        return True


if __name__ == "__main__":
    configurator = SupersetConfigurator(SUPERSET_URL, ADMIN_USERNAME, ADMIN_PASSWORD)
    success = configurator.run()
    sys.exit(0 if success else 1)