# 🔐 Security Configuration Guide

## Environment Variables Setup

This application requires all credentials to be provided via environment variables. **No default credentials are hardcoded** for security reasons.

### Required Environment Variables

All of the following variables MUST be set before running the application:

```bash
# Redis Configuration
REDIS_HOST=redis
REDIS_PORT=6379

# MinIO Configuration
MINIO_ENDPOINT=minio:9000
MINIO_ACCESS_KEY=your_minio_access_key_here
MINIO_SECRET_KEY=your_minio_secret_key_here
MINIO_SECURE=false

# PostgreSQL Database Configuration
DB_HOST=db
DB_PORT=5432
POSTGRES_DB=vehicle_db
POSTGRES_USER=your_postgres_username_here
POSTGRES_PASSWORD=your_secure_password_here
```

## Setup Instructions

### 1. Create `.env` File

Copy the example file and fill in your credentials:

```bash
cp .env.example .env
```

Then edit `.env` with your actual credentials.

### 2. Security Best Practices

- ✅ **NEVER** commit `.env` file to git
- ✅ **ALWAYS** use `.env.example` as template (without real credentials)
- ✅ Use strong passwords (minimum 16 characters)
- ✅ Rotate credentials regularly
- ✅ Use different credentials for development and production
- ✅ Limit access to `.env` file (chmod 600)

### 3. Verify `.env` is Ignored

Check that `.env` is in your `.gitignore`:

```bash
git check-ignore .env
```

Should return: `.env`

### 4. Production Deployment

For production, use:
- **Docker secrets** (for Docker Swarm)
- **Kubernetes secrets** (for Kubernetes)
- **Environment variables** (for cloud platforms)
- **Secret management services** (AWS Secrets Manager, Azure Key Vault, HashiCorp Vault)

## Credential Generation

### PostgreSQL Password
```bash
# Generate strong password (Linux/Mac)
openssl rand -base64 32

# Or use pwgen
pwgen -s 32 1
```

### MinIO Credentials
```bash
# Access Key (20 characters)
openssl rand -hex 10

# Secret Key (40 characters)
openssl rand -hex 20
```

## Troubleshooting

### Missing Environment Variables Error

If you see:
```
Missing required environment variables: REDIS_HOST, MINIO_ACCESS_KEY, ...
```

**Solution**: Ensure all required variables are set in your `.env` file.

### Docker Compose

Make sure your `docker-compose.yml` includes:

```yaml
services:
  processing-worker:
    env_file:
      - .env
```

## Emergency Response

### If Credentials Are Exposed

1. **Immediately** rotate all exposed credentials
2. Check git history: `git log --all --full-history -- .env`
3. If committed, use `git-filter-repo` or BFG Repo-Cleaner to remove
4. Invalidate old credentials in all services
5. Update `.env` with new credentials
6. Review access logs for unauthorized access

## Security Checklist

- [ ] `.env` file created with actual credentials
- [ ] `.env` is in `.gitignore`
- [ ] Strong passwords used (16+ characters)
- [ ] File permissions set: `chmod 600 .env`
- [ ] Credentials not hardcoded in source code
- [ ] Different credentials for dev/prod environments
- [ ] Team members briefed on security practices
- [ ] Credentials stored securely (password manager)
- [ ] Regular credential rotation scheduled

---
**Last Updated**: January 2026
