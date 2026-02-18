@echo off
REM Apache Superset Integration - Start Script for Windows
REM This script starts all necessary services for Superset and configures it

setlocal enabledelayedexpansion

pushd "%~dp0.."

cls
echo ===============================================================
echo    Apache Superset Integration - Automated Setup (Windows)
echo ===============================================================
echo.

REM Step 1: Start Docker containers
echo [Step 1] Starting Docker containers...
echo This may take a minute or two...
docker-compose up -d superset superset-postgres db redis minio
if errorlevel 1 (
    echo Error starting containers. Check Docker is running.
    popd
    pause
    exit /b 1
)
echo [✓] Containers started
echo.

REM Step 2: Wait for services
echo [Step 2] Waiting for services to be ready...
echo Checking Superset...
set "count=0"
:wait_loop
if %count% geq 60 (
    echo [⚠] Superset took longer than expected. Check logs with: docker logs superset
    goto skip_wait
)
docker exec superset curl -f http://localhost:8088/health >nul 2>&1
if errorlevel 0 (
    echo [✓] Superset is ready
    goto skip_wait
)
echo Waiting... attempt %count%/60
timeout /t 2 /nobreak >nul
set /a count+=1
goto wait_loop

:skip_wait
echo.

REM Step 3: Configure database
echo [Step 3] Configuring Superset database connection...
if exist "superset-dockerfile\configure_superset.py" (
    python superset-dockerfile\configure_superset.py
    if errorlevel 1 (
        echo.
        echo [⚠] Database configuration script failed.
        echo You can run it manually: python superset-dockerfile\configure_superset.py
    )
) else (
    echo [⚠] superset-dockerfile\configure_superset.py not found
)
echo.

REM Step 4: Final information
echo ===============================================================
echo [✓] Apache Superset Setup Complete!
echo ===============================================================
echo.
echo [Access your dashboard:]
echo   URL: http://localhost:8088
echo   Username: admin
echo   Password: admin123
echo.
echo [Other Services:]
echo   Streamlit Dashboard: http://localhost:8501
echo   Analytics Database: localhost:5428
echo   MinIO Console: http://localhost:9001
echo.
echo [Documentation:]
echo   Quick Start: SUPERSET_QUICK_REF.md
echo   Full Guide: SUPERSET_GUIDE.md
echo   Setup Details: SUPERSET_SETUP.md
echo   Integration Info: SUPERSET_INTEGRATION.md
echo.
echo [Next Steps:]
echo   1. Login to http://localhost:8088
echo   2. Go to Settings ^> Database Connections
echo   3. Verify 'vehicle_analytics' database is connected
echo   4. Click + Create ^> Dataset
echo   5. Select 'vehicle_analytics' and your desired table
echo   6. Start building dashboards!
echo.
echo ===============================================================
echo.

popd
pause
