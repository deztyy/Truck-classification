-- Apache Superset initialization script
-- This file sets up the database connection from Superset to the analytics database

-- Note: Run these commands manually in Superset UI or via API
-- URL: http://localhost:8088
-- Default credentials: admin / admin123

-- SQL to register the vehicle_db database in Superset:
-- Database URI: postgresql://postgres:postgres1234@db:5432/vehicle_db
-- Database name: vehicle_analytics
-- Driver: postgresql
-- Host: db
-- Port: 5432
-- Database: vehicle_db
-- Username: postgres
-- Password: postgres1234
