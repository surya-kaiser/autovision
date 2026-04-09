# AutoVision — Complete Running Guide & Commands

## 🎯 Quick Decision: Which Path to Run?

| Path | Use Case | Time to Start | Complexity | Best For |
|------|----------|---------------|-----------|----------|
| **Docker Compose** | Production, CI/CD ready | 5 min | Low | Deployment, testing |
| **Local Dev** | Development, debugging | 10 min | Medium | Feature development |
| **Jenkins** | Automated builds & tests | 20 min | High | Team deployment |

---

## 🐳 METHOD 1: Docker Compose (Fastest)

### Start Everything in 3 Commands

```bash
cd autovision
cp .env.example .env
```

> On Windows, make sure Docker Desktop is started first and set to Linux containers or WSL2 mode.

```bash
docker-compose up -d
```

**Access Services Immediately:**
```
Frontend:  http://localhost:5173
Backend:   http://localhost:8000
Docs:      http://localhost:8000/docs
Ollama:    http://localhost:11434
```

### Verify Everything Works

```bash
# Check all containers running
docker-compose ps

# Test backend endpoint
curl http://localhost:8000/api/v1/system/info
# Expected: {"status": "success", "data": {"gpu_available": false, ...}}

# Test frontend
curl http://localhost:5173
# Expected: HTML response (200)

# Test Ollama
curl http://localhost:11434/api/tags
# Expected: {"models": [...]}
```

### Run & Monitor

```bash
# Live logs from backend
docker-compose logs -f backend

# Live logs from frontend
docker-compose logs -f frontend

# All logs
docker-compose logs -f

# Last 50 lines
docker-compose logs --tail=50

# Specific service with timestamp
docker-compose logs -t backend
```

### Common Operations

```bash
# Run pytest inside backend container
docker-compose exec backend pytest tests/ -v

# Enter backend shell
docker-compose exec backend /bin/bash

# Check disk usage
docker system df

# Stop (keep containers)
docker-compose stop

# Restart backend
docker-compose restart backend

# Full cleanup (deletes volumes!)
docker-compose down -v
```

### Troubleshooting Docker

```bash
# Port is already in use?
lsof -i :8000
lsof -i :5173

# Kill container
docker kill autovision-backend

# Remove old containers
docker-compose rm -f

# Rebuild everything fresh
docker-compose down -v
docker-compose build --no-cache
docker-compose up -d

# Check container health
docker inspect autovision-backend | grep -A 5 HealthStatus

# View detailed container info
docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}"
```

---

## 💻 METHOD 2: Local Development (Full Control)

### Setup Backend

```bash
# Navigate
cd autovision/backend

# Create environment
python3.11 -m venv venv
source venv/bin/activate          # macOS/Linux
venv\Scripts\activate             # Windows

# Install packages
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Run tests
pytest tests/ -v

# Start server (must have Ollama running separately)
uvicorn app.main:app --reload --port 8000
```

### Setup Frontend (new terminal)

```bash
# Navigate
cd autovision/frontend

# Install
npm install

# Dev server (auto-hot-reload on file changes)
npm run dev

# Or build for production
npm run build
npm run preview   # Test production build locally
```

### Start Ollama (new terminal)

```bash
# Terminal 1: Start Ollama server
ollama serve

# Terminal 2 (after seeing "Listening on..."):
ollama pull llama3.2
```

### Now You Have 3 Services Running

```
Terminal 1: Backend at http://localhost:8000
Terminal 2: Frontend at http://localhost:5173
Terminal 3: Ollama at http://localhost:11434
```

### Development Workflow

```bash
# Mode 1: Watch for changes and auto-reload
# Backend: uvicorn has --reload flag
# Frontend: npm run dev auto-reloads

# Mode 2: Run tests continuously
cd backend
pytest tests/ -v --watch

# Mode 3: Debug specific test
pytest tests/test_preprocessor.py::test_csv_pipeline -v -s

# Mode 4: Debug with breakpoint
# Add to Python code:
import pdb; pdb.set_trace()
# Run pytest normally, it will pause at breakpoint
```

---

## 🔄 METHOD 3: Jenkins CI/CD Pipeline

### Step 1: Start Jenkins

```bash
# Option A: Docker (recommended)
docker run -d \
  --name jenkins \
  -p 8080:8080 \
  -p 50000:50000 \
  -v jenkins_home:/var/jenkins_home \
  -v /var/run/docker.sock:/var/run/docker.sock \
  jenkins/jenkins:lts

# Option B: Local installation (macOS)
brew install jenkins-lts
brew services start jenkins-lts

# Option C: Local installation (Linux)
curl -fsSL https://pkg.jenkins.io/debian-stable/jenkins.io.key | sudo tee /usr/share/keyrings/jenkins-keyring.asc
sudo sh -c 'echo deb [signed-by=/usr/share/keyrings/jenkins-keyring.asc] https://pkg.jenkins.io/debian-stable binary/ | tee /etc/apt/sources.list.d/jenkins.list'
sudo apt-get update
sudo apt-get install jenkins
sudo systemctl start jenkins
```

### Step 2: Get Initial Password

```bash
# Docker
docker logs jenkins | grep "Initial Admin Password"

# Local
sudo cat /var/lib/jenkins/secrets/initialAdminPassword

# Then: Open http://localhost:8080
```

### Step 3: Install Plugins

1. Follow setup wizard
2. Click **Install suggested plugins**
3. Create admin user
4. Go to **Manage Jenkins** → **Plugins** → **Available**
5. Search and install:
   - Pipeline
   - Git
   - Docker Pipeline
   - JUnit Plugin
   - Cobertura Plugin

### Step 4: Create Pipeline Job

**Option A: From Git (Recommended)**
```
1. Jenkins home → New Item
2. Name: autovision
3. Type: Pipeline
4. OK
5. Configuration:
   Definition: Pipeline script from SCM
   SCM: Git
   Repository URL: https://github.com/YOUR_ORG/autovision
   Credentials: (add GitHub creds)
   Branch: */main
   Script Path: Jenkinsfile
6. Save
7. Build Now
```

**Option B: From Text**
```
1. Jenkins home → New Item
2. Name: autovision
3. Type: Pipeline
4. OK
5. Configuration:
   Definition: Pipeline script
   Script: (paste entire Jenkinsfile content)
6. Save
7. Build Now
```

### Step 5: Monitor Pipeline

```
Jenkins home → autovision → Build #1

View Output:
├── Console Output      (Full logs, real-time)
├── Test Results        (Pass/fail breakdown)
├── Code Coverage       (Coverage graph)
├── Artifacts           (Build logs, reports)
```

**Via CLI:**
```bash
# Current build status
curl http://jenkins:8080/job/autovision/lastBuild/api/json | jq '.result'

# Test results
curl http://jenkins:8080/job/autovision/lastBuild/testReport/api/json | jq '.passCount, .failCount'

# Full console output
curl http://jenkins:8080/job/autovision/lastBuild/consoleText

# Trigger build remotely
curl -X POST http://jenkins:8080/job/autovision/build
```

### Step 6: Setup Auto-Trigger

**GitHub Webhook (Recommended):**
```
1. GitHub Repo → Settings → Webhooks → Add webhook
2. Payload URL: http://your-jenkins-server:8080/github-webhook/
3. Events: Push events, Pull requests
4. Active: ✓
5. Save
```

**Poll SCM (Check every 15 min):**
```
Jenkins Job → Configure → Build Triggers
  ✓ Poll SCM
  Schedule: H/15 * * * *
```

### Jenkins Pipeline Stages Explained

| Stage | What It Does | Command | Time |
|-------|-------------|---------|------|
| **Checkout** | Clone repo | git clone | 10s |
| **Setup** | Install Python/Node | python3.11, node18 | 20s |
| **Install Deps** | pip + npm install | pip install ... | 2 min |
| **Test** | Run pytest | pytest tests/ -v | 1 min |
| **Lint** | Code quality | flake8 app/ | 30s |
| **Build Backend** | Docker build | docker build | 3 min |
| **Build Frontend** | Docker build | docker build | 2 min |
| **Deploy** | docker-compose | docker-compose up | 1 min |
| **Health Check** | Verify endpoints | curl /api/v1/system/info | 30s |

**Total Time:** ~10 minutes first run, ~5 minutes subsequent (cached layers)

---

## 🔄 Workflow Examples

### Example 1: Local Development + Testing

```bash
# Terminal 1: Backend
cd autovision/backend
source venv/bin/activate
uvicorn app.main:app --reload --port 8000

# Terminal 2: Frontend
cd autovision/frontend
npm run dev

# Terminal 3: Ollama
ollama serve

# Terminal 4: Run tests
cd autovision/backend
pytest tests/ -v --watch

# Edit code → auto-reload → test
```

### Example 2: Prepare for Production

```bash
# Build and test locally
docker-compose build
docker-compose up -d

# Run full test suite
docker-compose exec backend pytest tests/ -v

# View coverage
docker-compose exec backend pytest --cov=app --cov-report=html tests/

# Check images
docker images | grep autovision

# Push to registry (if configured)
docker tag autovision:backend-latest registry.example.com/autovision:backend
docker push registry.example.com/autovision:backend
```

### Example 3: CI/CD with Jenkins

```bash
# Push to main branch
git add .
git commit -m "Add new feature"
git push origin main

# GitHub webhook triggers Jenkins
# Jenkins auto-runs pipeline:
#   ✓ Tests pass
#   ✓ Images build
#   ✓ Deployed with docker-compose
#   ✓ Health checks pass

# Services live at:
# http://jenkins-server:5173 (frontend)
# http://jenkins-server:8000 (backend)

# View results
open http://localhost:8080/job/autovision/lastBuild/
```

---

## 📊 Service Architecture

### Docker Compose Setup

```
┌─────────────────────────────────────────────────────────────┐
│                      Docker Network                         │
│                    (autovision bridge)                      │
│                                                             │
│  ┌──────────────────┐   ┌──────────────────┐               │
│  │  autovision-     │   │  autovision-     │               │
│  │  backend:8000    │◄─►│  frontend:80     │               │
│  │                  │   │  (Nginx)         │               │
│  │  FastAPI         │   │  Vite React      │               │
│  │  Python 3.11     │   │  Node 18         │               │
│  │  Port: 8000      │   │  Port: 5173→80   │               │
│  │  Inside: 8000    │   │  Inside: 80      │               │
│  └────────┬─────────┘   └──────────────────┘               │
│           │                                                │
│           │ http://ollama:11434                           │
│           ▼                                                │
│  ┌──────────────────┐                                      │
│  │  autovision-     │                                      │
│  │  ollama:11434    │                                      │
│  │                  │                                      │
│  │  Ollama LLM      │                                      │
│  │  Port: 11434     │                                      │
│  │  Model: llama3.2 │                                      │
│  └──────────────────┘                                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
         ▲                              ▲
         │                              │
         │ http://localhost:8000        │ http://localhost:5173
         │                              │
    User Browser (Host)             User Browser (Host)
```

### Volumes (Data Persistence)

```
Docker Volumes:
├── autovision_data
│   ├── uploads/      (User-uploaded datasets)
│   ├── models/       (Trained models)
│   └── logs/         (Training logs)
│
└── ollama_data
    └── .ollama/      (LLM cache)
```

---

## 🆘 Quick Troubleshooting

### "Port already in use"
```bash
# Find what's using port 8000
lsof -i :8000

# Kill it
kill -9 <PID>

# Or change port in .env
BACKEND_PORT=8001
```

### "Backend can't reach Ollama"
```bash
# Inside backend container
docker-compose exec backend curl http://ollama:11434/api/tags

# If fails:
# On Mac/Windows Docker Desktop:
docker-compose down
# Edit docker-compose.yml
# Change: http://ollama:11434
# To: http://host.docker.internal:11434

# On Linux, use host IP:
docker-compose down
# Change: http://ollama:11434
# To: http://192.168.1.100:11434
```

### "Tests failing in Jenkins"
```bash
# Run tests locally first
docker-compose exec backend pytest tests/ -v

# If pass locally but fail in Jenkins:
# Check Jenkins workspace for file permissions
# Run with verbose output:
docker-compose exec backend pytest tests/ -vv --tb=long

# Check environment variables:
docker-compose exec backend env | grep OLLAMA
```

### "Build fails with RPC error or EOF"

**Problem:** Build takes 45+ minutes then fails with "failed to receive status: rpc error: code = Unavailable desc = error reading from server: EOF"

**Solutions:**

```bash
# Option 1: Rebuild with no cache (clears failed layers)
docker-compose build --no-cache

# Option 2: Build with more memory (Docker Desktop → Settings → Resources)
# Increase memory to 8GB+ for ML builds

# Option 3: Build step by step to isolate the issue
docker build ./backend --no-cache -t autovision:backend-test
docker build ./frontend --no-cache -t autovision:frontend-test

# Option 4: Use pre-built base image (faster)
# Edit backend/Dockerfile to use a pre-built image with PyTorch
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime  # If you have GPU
# or
FROM pytorch/pytorch:2.1.0-cpu  # CPU only

# Option 5: Add build timeout and retry
docker-compose build --parallel --memory=4g
```

**Prevention:**
- Use Docker Desktop with 8GB+ RAM allocated
- Build during off-peak network hours
- Consider using a pre-built base image with PyTorch already installed

---

## 📋 Useful Commands Reference

```bash
# ─── Docker ────────────────────────────────────────────────
docker ps                               # List running containers
docker ps -a                            # All containers
docker images                           # List images
docker logs container                   # View logs
docker exec -it container /bin/bash    # Enter container
docker rm container                     # Remove container
docker rmi image                        # Remove image

# ─── Docker Compose ────────────────────────────────────────
docker-compose up -d                    # Start services hidden
docker-compose up                       # Start with visible output
docker-compose down                     # Stop services
docker-compose down -v                  # Stop + delete volumes
docker-compose ps                       # List services
docker-compose logs -f                  # Watch logs
docker-compose exec service cmd         # Run command in container
docker-compose build                    # Build images
docker-compose restart service          # Restart service

# ─── Backend (Local) ──────────────────────────────────────
python -m venv venv                     # Create venv
source venv/bin/activate                # Activate venv (Unix)
venv\Scripts\activate                   # Activate venv (Windows)
pip install -r requirements.txt         # Install dependencies
pytest tests/ -v                        # Run tests
pytest tests/ -v --cov=app              # With coverage
uvicorn app.main:app --reload           # Start server

# ─── Frontend (Local) ──────────────────────────────────────
npm install                             # Install dependencies
npm run dev                             # Dev server (hot-reload)
npm run build                           # Production build
npm run preview                         # Preview production build

# ─── Jenkins ──────────────────────────────────────────────
curl http://jenkins:8080/job/autovision/build           # Trigger
curl http://jenkins:8080/job/autovision/lastBuild/      # Last build
curl http://jenkins:8080/job/autovision/lastBuild/consoleText  # Logs

# ─── Testing ───────────────────────────────────────────────
curl http://localhost:8000/api/v1/system/info           # Backend health
curl http://localhost:5173                              # Frontend
curl http://localhost:11434/api/tags                    # Ollama
```

---

## Production Checklist Before Deploying

- [ ] All tests pass: `pytest tests/ -v`
- [ ] Docker images build: `docker-compose build`
- [ ] Services start: `docker-compose up -d`
- [ ] Health checks pass: `curl http://localhost:8000/api/v1/system/info`
- [ ] Frontend loads: `curl http://localhost:5173`
- [ ] Ollama responds: `curl http://localhost:11434/api/tags`
- [ ] Jenkins pipeline succeeds
- [ ] No hardcoded secrets in `.env` or Dockerfile
- [ ] Environment variables documented in `.env.example`
- [ ] Data volumes are mapped correctly
- [ ] Backups scheduled for `autovision_data` volume
- [ ] Monitoring/logging configured

---

**Everything Ready? Start with:** 
```bash
docker-compose up -d
```

Then visit: http://localhost:5173
