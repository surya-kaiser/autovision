# AutoVision — Docker & Jenkins Setup Guide

## Overview

This guide covers:
1. **Containerization** — Dockerize backend, frontend, and Ollama
2. **Local Development** — Run with Docker Compose
3. **CI/CD Pipeline** — Jenkins automation for build, test, and deployment

---

## 📋 Prerequisites

### System Requirements
- **Docker** ≥ 20.10 (with Docker Compose ≥ 1.29)
- **Jenkins** ≥ 2.300 (for CI/CD)
- **Git** (for repository cloning)
- **Disk space** ≥ 30 GB (for images, models, and data)
- **Memory** ≥ 8 GB RAM (16 GB recommended)

### Installation

**Docker (Mac/Windows/Linux)**
```bash
# Install Docker Desktop from: https://www.docker.com/products/docker-desktop
# Or for Linux:
curl -fsSL https://get.docker.com -o get-docker.sh && sh get-docker.sh
```

**Jenkins (Mac)**
```bash
brew install jenkins-lts
brew services start jenkins-lts
# Access at: http://localhost:8080
```

**Jenkins (Linux - Docker)**
```bash
docker run -d \
  --name jenkins \
  -p 8080:8080 \
  -p 50000:50000 \
  -v jenkins_home:/var/jenkins_home \
  jenkins/jenkins:lts
```

---

## 🚀 Quick Start — Docker Compose (Local Development)

### 1. Setup Environment File

```bash
cp .env.example .env
# Edit .env if needed (defaults work for local development)
```

### 2. Build and Start Services

```bash
# Build images
docker-compose build

# Start all services (-d for background)
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f backend
docker-compose logs -f frontend
docker-compose logs -f ollama
```

### 3. Verify Services

**Backend**: http://localhost:8000
- Health check: `curl http://localhost:8000/api/v1/system/info`
- API docs: http://localhost:8000/docs (Swagger UI)

**Frontend**: http://localhost:5173
- Direct access: http://localhost:5173

**Ollama**: http://localhost:11434
- API endpoint: `curl http://localhost:11434/api/tags`

### 4. Test the System

```bash
# Upload a sample image and train a model
# Use the web UI at http://localhost:5173

# Or test with curl
curl -X POST http://localhost:8000/api/v1/dataset/upload \
  -F "file=@sample_image.jpg"
```

### 5. Stop Services

```bash
# Stop containers (keep volumes)
docker-compose down

# Stop and remove volumes
docker-compose down -v

# Stop specific service
docker-compose stop backend
docker-compose start backend
```

---

## 🔧 Docker & Image Management

### Build Images Manually

```bash
# Backend
docker build -t autovision:backend-latest ./backend

# Frontend
docker build -t autovision:frontend-latest ./frontend

# Verify images
docker images | grep autovision
```

### View Container Logs

```bash
# All services
docker-compose logs

# Specific service
docker-compose logs -f backend

# Show last 100 lines
docker-compose logs --tail=100 backend
```

### Execute Commands in Containers

```bash
# Run pytest in backend
docker-compose exec backend pytest tests/ -v

# Run npm in frontend
docker-compose exec frontend npm run build

# Get shell access
docker-compose exec backend /bin/bash
docker-compose exec frontend /bin/sh
```

### Health Checks

```bash
# Check container health
docker ps --format "table {{.Names}}\t{{.Status}}"

# Detailed health status
docker inspect autovision-backend | grep -A10 "HealthStatus"
```

---

## 🔄 Volume & Data Persistence

### Default Volumes

| Volume | Purpose | Mount Point |
|--------|---------|------------|
| `autovision_data` | Models & uploads | `/tmp/autovision` |
| `ollama_data` | LLM cache & models | `/root/.ollama` |

### Backup Data

```bash
# Backup volumes
docker run --rm \
  -v autovision_data:/data \
  -v $(pwd)/backups:/backups \
  alpine tar czf /backups/autovision_data.tar.gz -C /data .

# Restore volumes
docker run --rm \
  -v autovision_data:/data \
  -v $(pwd)/backups:/backups \
  alpine tar xzf /backups/autovision_data.tar.gz -C /data
```

### Clear Data

```bash
# Clear only application data (keep volumes)
docker-compose down

# Clear everything including volumes
docker-compose down -v

# Remove all autovision images
docker rmi autovision:*
```

---

## 🔧 Jenkins CI/CD Pipeline Setup

### 1. Setup Jenkins

#### Initial Configuration (First Time)

```bash
# Get initial admin password
docker logs jenkins | grep "Initial Admin Password"

# Access Jenkins
open http://localhost:8080

# Follow setup wizard:
# 1. Enter admin password
# 2. Install suggested plugins
# 3. Create admin user
# 4. Configure Jenkins URL
```

#### Install Required Plugins

1. **Manage Jenkins** → **Plugins** → **Available Plugins**
2. Search and install:
   - Pipeline
   - Git
   - Docker Pipeline
   - JUnit Plugin
   - Cobertura Plugin (for coverage)

### 2. Create Pipeline Job

#### Method A: From Jenkinsfile (Recommended)

1. **New Item** → Name: `autovision` → **Pipeline** → OK
2. **Definition**: Pipeline script from SCM
3. **SCM**: Git
4. **Repository URL**: `https://github.com/your-org/autovision.git`
5. **Branch**: `*/main`
6. **Script Path**: `Jenkinsfile`
7. **Save** → **Build Now**

#### Method B: Declarative Pipeline

1. **New Item** → Name: `autovision` → **Pipeline** → OK
2. **Definition**: Pipeline script
3. Copy-paste content of `Jenkinsfile` into the text area
4. **Save** → **Build Now**

### 3. Configure Build Triggers

1. **Pipeline** → **Build Triggers**
2. Options:
   - **Poll SCM**: `H/15 * * * *` (check every 15 minutes)
   - **GitHub hook trigger**: Requires GitHub webhook setup

### 4. Monitor Pipeline Execution

```
Jenkins UI → autovision → Build #1
├── Checkout ✓
├── Setup ✓
├── Install Dependencies ✓
├── Test ✓
├── Lint ✓
├── Build Backend Image ✓
├── Build Frontend Image ✓
├── Deploy ✓
└── Health Check ✓
```

View logs:
- **Console Output**: Current/previous build logs
- **Testing**: Test results and coverage report
- **Artifacts**: Build artifacts and test logs

---

## 📊 Pipeline Stages Explained

### 1. **Checkout**
- Clones the Git repository
- Prepares workspace

### 2. **Setup**
- Installs Python 3.11 and Node.js 18
- Creates virtual environments

### 3. **Install Dependencies**
- Backend: pip install -r requirements.txt + PyTorch
- Frontend: npm ci (clean install)

### 4. **Test**
- Runs pytest with coverage
- Generates JUnit XML report
- Publishes coverage report

### 5. **Lint** (Optional)
- Runs flake8/pylint
- Continues even if warnings exist

### 6. **Build Backend Image**
- Docker build with caching
- Tags: `autovision:backend-{BUILD_NUMBER}-{GIT_HASH}`

### 7. **Build Frontend Image**
- Multi-stage: Node → Nginx
- Tags: `autovision:frontend-{BUILD_NUMBER}-{GIT_HASH}`

### 8. **Deploy**
- Pulls latest compose file
- Stops previous containers
- Starts new containers with docker-compose

### 9. **Health Check**
- Verifies backend endpoint: `/api/v1/system/info`
- Verifies frontend responds
- Verifies Ollama is accessible
- Retries 5 times with 2-second delays

---

## 🐛 Troubleshooting

### Issue: Containers won't start
```bash
# Check logs
docker-compose logs backend

# Common causes:
# 1. Port already in use
lsof -i :8000
lsof -i :5173

# 2. Insufficient disk space
docker system df

# 3. Image build failed - rebuild
docker-compose build --no-cache
```

### Issue: Backend can't reach Ollama
```bash
# Check if Ollama is running
docker-compose ps | grep ollama

# Test connectivity from backend container
docker-compose exec backend curl http://ollama:11434/api/tags

# If on host: update OLLAMA_BASE_URL in docker-compose.yml
# Mac/Windows: http://host.docker.internal:11434
# Linux: http://<host-ip>:11434
```

### Issue: Tests fail in Jenkins
```bash
# Run tests locally first
cd backend
python -m pytest tests/ -v

# Check test logs in Jenkins
# Jenkins → Build → Console Output

# Common causes:
# 1. Missing dependencies
pip install pytest pytest-cov pytest-asyncio

# 2. Port already in use (from previous unsuccessful run)
docker ps -a | grep autovision
docker rm autovision-backend autovision-frontend
```

### Issue: Frontend doesn't connect to backend
```bash
# Check VITE_API_URL in docker-compose.yml
# Should be: http://backend:8000/api/v1

# Test from frontend container
docker-compose exec frontend curl http://backend:8000/api/v1/system/info

# Check frontend build
docker-compose exec frontend ls -la /usr/share/nginx/html
```

### Issue: Jenkins job can't find Docker
```bash
# Jenkins needs Docker access
# Add Jenkins user to docker group (Linux):
sudo usermod -aG docker jenkins

# Restart Jenkins
sudo systemctl restart jenkins

# Or use Docker socket volume (Jenkins in container):
docker run -v /var/run/docker.sock:/var/run/docker.sock ...
```

---

## 📈 Production Deployment Checklist

Before deploying to production:

- [ ] All tests passing
- [ ] Code lint checks passing
- [ ] Docker images built successfully
- [ ] Health checks verify all services
- [ ] Volumes mounted for data persistence
- [ ] Environment variables properly configured
- [ ] Container restart policies set
- [ ] Resource limits defined (CPU, memory)
- [ ] Network policies configured
- [ ] Security: Run containers as non-root
- [ ] Logging aggregated and monitored
- [ ] Backups configured for volumes

---

## 🔐 Security Best Practices

### In Dockerfile
- ✅ Run as non-root user
- ✅ Use slim base images
- ✅ Don't run with `--privileged`
- ✅ Use specific package versions

### In docker-compose
- ✅ Don't use `latest` tags in production
- ✅ Limit resource usage
- ✅ Use read-only volumes where appropriate
- ✅ Isolate networks

### In Jenkins
- ✅ Use credentials plugin for secrets
- ✅ Don't commit `.env` files
- ✅ Access Jenkins on secure network
- ✅ Use RBAC for multi-user setup

---

## 📚 File Structure

```
autovision/
├── backend/
│   ├── Dockerfile           ← Backend container definition
│   ├── requirements.txt
│   └── app/
├── frontend/
│   ├── Dockerfile           ← Frontend container definition
│   ├── nginx.conf           ← Nginx reverse proxy config
│   └── src/
├── docker-compose.yml       ← Orchestration config
├── .env.example             ← Environment template
├── Jenkinsfile              ← CI/CD pipeline
└── DOCKER_SETUP.md          ← This file
```

---

## 🚀 Next Steps

1. **Start locally**: `docker-compose up -d`
2. **Setup Jenkins**: Configure pipeline job
3. **Run first pipeline**: Verify all stages pass
4. **Configure backups**: Schedule volume backups
5. **Setup monitoring**: (Optional) Add health monitoring
6. **Deploy**: Push images to registry and deploy to production

---

## 📞 Support

For issues:
1. Check troubleshooting section above
2. Review container logs: `docker-compose logs`
3. Check Jenkins console output
4. Verify environment variables in `.env`

---

## 📖 Additional Resources

- **Docker Docs**: https://docs.docker.com/
- **Docker Compose**: https://docs.docker.com/compose/
- **Jenkins**: https://www.jenkins.io/doc/
- **PyTorch CPU**: https://pytorch.org/
- **Ollama**: https://ollama.ai/

---

## 📝 Useful Commands Quick Reference

```bash
# Docker Compose
docker-compose build               # Build images
docker-compose up -d              # Start services
docker-compose down               # Stop services
docker-compose down -v            # Stop and remove volumes
docker-compose ps                 # List services
docker-compose logs -f backend    # Follow logs

# Docker
docker images                     # List images
docker ps                        # List running containers
docker exec -it container bash  # Enter container
docker logs container            # View logs
docker rm container              # Remove container

# Jenkins
curl http://localhost:8080       # Access Jenkins
curl http://localhost:8000       # Access backend
curl http://localhost:5173       # Access frontend

# Cleanup
docker system prune              # Remove unused images/containers
docker volume prune              # Remove unused volumes
```

---

Generated: 2024 | AutoVision MLOps — Production-Ready Docker & CI/CD
