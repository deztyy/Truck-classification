#!/bin/bash
# Apache Superset Integration - Start Script
# This script starts all necessary services for Superset and configures it

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

echo "═══════════════════════════════════════════════════════════════"
echo "   Apache Superset Integration - Automated Setup"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Start Docker containers
echo -e "${BLUE}Step 1: Starting Docker containers...${NC}"
echo "This may take a minute or two..."
docker-compose up -d superset superset-postgres db redis minio
echo -e "${GREEN}✓ Containers started${NC}"
echo ""

# Step 2: Wait for services
echo -e "${BLUE}Step 2: Waiting for services to be ready...${NC}"
echo "Checking Superset..."
for i in {1..60}; do
    if docker exec superset curl -f http://localhost:8088/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Superset is ready${NC}"
        break
    fi
    if [ $i -eq 60 ]; then
        echo -e "${YELLOW}⚠ Superset took longer than expected. Check logs with: docker logs superset${NC}"
    fi
    echo -n "."
    sleep 2
done
echo ""

# Step 3: Configure database
echo -e "${BLUE}Step 3: Configuring Superset database connection...${NC}"
if command -v python &> /dev/null; then
    python configure_superset.py
else
    echo -e "${YELLOW}⚠ Python not found. Running configure_superset.py manually:${NC}"
    echo "  python configure_superset.py"
fi
echo ""

# Step 4: Final information
echo "═══════════════════════════════════════════════════════════════"
echo -e "${GREEN}✓ Apache Superset Setup Complete!${NC}"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo -e "${BLUE}Access your dashboard:${NC}"
echo "  URL: http://localhost:8088"
echo "  Username: admin"
echo "  Password: admin123"
echo ""
echo -e "${BLUE}Other Services:${NC}"
echo "  Streamlit Dashboard: http://localhost:8501"
echo "  Analytics Database: localhost:5428"
echo "  MinIO Console: http://localhost:9001"
echo ""
echo -e "${BLUE}Documentation:${NC}"
echo "  Quick Start: SUPERSET_QUICK_REF.md"
echo "  Full Guide: SUPERSET_GUIDE.md"
echo "  Setup Details: SUPERSET_SETUP.md"
echo "  Integration Info: SUPERSET_INTEGRATION.md"
echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo "  1. Login to http://localhost:8088"
echo "  2. Go to Settings → Database Connections"
echo "  3. Verify 'vehicle_analytics' database is connected"
echo "  4. Click + Create → Dataset"
echo "  5. Select 'vehicle_analytics' and your desired table"
echo "  6. Start building dashboards!"
echo ""
echo "═══════════════════════════════════════════════════════════════"
