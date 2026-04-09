# AutoVision — Docker & CI/CD Quick Start

## ✅ What's Included

This setup adds **production-ready** containerization and CI/CD to AutoVision:

### Files Created/Updated
```
✓ backend/Dockerfile          — Python 3.11 + FastAPI + PyTorch (CPU)
✓ frontend/Dockerfile         — Node 18 + React + Nginx (multi-stage)
✓ docker-compose.yml          — Orchestrate all services
✓ Jenkinsfile                 — 9-stage CI/CD pipeline
✓ .env.example                — Environment configuration template
✓ DOCKER_SETUP.md             — Complete setup & troubleshooting guide
✓ PROJECT_INFO.md             — Updated with Docker info
```

### Key Features
- ✅ **Non-root containers** for security
- ✅ **Multi-stage builds** for size optimization  
- ✅ **Health checks** on all services
- ✅ **Persistent volumes** for data
- ✅ **Docker layer caching** for faster rebuilds
- ✅ **Jenkins pipeline** with test + lint + build + deploy
- ✅ **CPU-only PyTorch** (no GPU required)
- ✅ **Local Ollama support** (runs on host or in container)

---

## 🚀 Quick Start (5 minutes)

### 1. Start Everything Locally

```bash
cd autovision

# Copy environment template
cp .env.example .env

# Build and run with Docker Compose
docker-compose build   # ~3 minutes first time
docker-compose up -d   # Start services

# Wait 10 seconds for services to start
sleep 10

# Check status
docker-compose ps
```

### 2. Verify Services

```bash
# Backend API (should return system info)
curl http://localhost:8000/api/v1/system/info

# Frontend (should return HTML)
curl http://localhost:5173 

# Ollama (should return model tags)
curl http://localhost:11434/api/tags
```

### 3. Access UI

- **Frontend**: http://localhost:5173
- **Backend Swagger Docs**: http://localhost:8000/docs
- **Ollama API**: http://localhost:11434

### 4. Stop Services

```bash
docker-compose down
```

---

## ⚙️ Jenkins Pipeline Setup (10 minutes)

### Prerequisites
- Jenkins ≥ 2.300 installed
- Docker installed on Jenkins machine
- Git access configured

### Setup Steps

#### 1. Install Plugins
In Jenkins:
1. **Manage Jenkins** → **Plugins** → **Available Plugins**
2. Install: `Pipeline`, `Git`, `Docker Pipeline`, `JUnit`, `Cobertura`

#### 2. Create Pipeline Job
1. **New Item** → Name: `autovision` → **Pipeline** → OK
2. Configure:
   - **Definition**: Pipeline script from SCM
   - **SCM**: Git
   - **Repository URL**: `https://github.com/your-org/autovision.git`
   - **Credentials**: Configure Git credentials
   - **Branch specifier**: `*/main`
   - **Script path**: `Jenkinsfile`
3. **Save**

#### 3. Run Pipeline
1. Click **Build Now**
2. Wait for all 9 stages to complete:
   ```
   ✓ Checkout
   ✓ Setup
   ✓ Install Dependencies
   ✓ Test (runs pytest)
   ✓ Lint
   ✓ Build Backend Image
   ✓ Build Frontend Image
   ✓ Deploy (docker-compose up)
   ✓ Health Check
   ```
3. View logs: **Console Output**
4. View tests: **Test Results**
5. View coverage: **Coverage Report**

#### 4. Configure Auto-Trigger (Optional)
Add to **Build Triggers**:
- **Poll SCM**: `H/15 * * * *` (check every 15 min)
- **GitHub webhook**: (if using GitHub)

---

## 📖 Complete Guide

For detailed setup, troubleshooting, and production deployment:
→ **See [DOCKER_SETUP.md](./DOCKER_SETUP.md)**

---

## 🔧 Common Commands

### Docker Compose
```bash
docker-compose up -d          # Start services
docker-compose down           # Stop services
docker-compose ps             # List services
docker-compose logs -f        # View logs
docker-compose build          # Rebuild images
docker-compose down -v        # Stop + remove volumes
```

### In Containers
```bash
# Run tests
docker-compose exec backend pytest tests/ -v

# Access backend shell
docker-compose exec backend /bin/bash

# Access frontend shell
docker-compose exec frontend /bin/sh

# View backend logs
docker-compose logs -f backend
```

### Docker Images
```bash
docker images | grep autovision  # List images
docker rmi image-name            # Remove image
docker build -t autovision:backend ./backend
```

---

## 📋 Architecture

```
User Request
    ↓
Frontend (http://localhost:5173)
    ↓ (API calls)
Backend (http://localhost:8000)
    ↓ (LLM queries)
Ollama (http://localhost:11434)

All services run in Docker containers with persistent volumes.
```

---

## 🔐 Default Ports

| Service | Port | URL |
|---------|------|-----|
| Frontend | 5173 | http://localhost:5173 |
| Backend | 8000 | http://localhost:8000 |
| Ollama | 11434 | http://localhost:11434 |
| Jenkins | 8080 | http://localhost:8080 |

---

## ✨ What Changed (for existing users)

### ✅ No Breaking Changes
- AutoVision logic **unchanged**
- Project structure **preserved**
- All existing features **work identically**

### ➕ What Was Added
1. Docker containers for isolated deployment
2. docker-compose.yml for easy orchestration
3. Jenkins pipeline for automated testing & deployment
4. Environment configuration via .env
5. Comprehensive setup & troubleshooting guide

---

## 🚀 Next Steps

1. **Local Testing**: `docker-compose up -d` on your machine
2. **Jenkins Setup**: Configure pipeline job
3. **First Pipeline Run**: Verify all stages pass
4. **Production Deployment**: Use Jenkins to trigger builds
5. **Monitoring**: (Optional) Add health monitoring

---

## 💡 Tips

- **First build is slow** (~5-10 min) due to Python packages. Subsequent builds use cache (~2 min).
- **If ports conflict**: Change in `.env` or `docker-compose.yml`
- **To rebuild without cache**: `docker-compose build --no-cache`
- **To see live logs**: `docker-compose logs -f backend`
- **To clear everything**: `docker-compose down -v && docker system prune`

---

## 🆘 Need Help?

1. Check **[DOCKER_SETUP.md](./DOCKER_SETUP.md)** for detailed troubleshooting
2. Review **container logs**: `docker-compose logs backend`
3. Check **Jenkins console output**: Click build → Console Output
4. Verify **ports aren't in use**: `lsof -i :8000`
5. Ensure **Docker is running**: `docker info`

---

**Ready to deploy AutoVision? Start with:** `docker-compose up -d` 🚀
