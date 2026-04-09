# AutoVision MLOps — Project Reference

## Overview

AutoVision is a **full-stack Autonomous ML Engineer** that accepts an arbitrary dataset (CSV,
ZIP of images, or image folder), detects the task type, selects the right model, preprocesses the
data, trains it, and returns metrics — all without user configuration.

**Stack:** FastAPI (Python 3.11) + React 18 (Vite + Tailwind)  
**LLM:** Ollama running locally at `http://localhost:11434` with model `llama3.2`  
**Deep learning:** PyTorch (CPU wheel) + Ultralytics YOLO (ultralytics package)  
**Tabular ML:** scikit-learn, XGBoost, LightGBM  
**Tests:** pytest — 46 existing tests + 14 new task-router tests = 60 total  
**UI Tabs:** Train (upload→preprocess→train), History (past sessions), Predict (model selection + inference)  
**Deployment:** Docker + Docker Compose + Jenkins CI/CD (See [DOCKER_SETUP.md](DOCKER_SETUP.md))

---

## 🚀 Quick Start (3 Commands)

**See [RUNNING_GUIDE.md](RUNNING_GUIDE.md) for comprehensive commands and troubleshooting!**

The fastest way to get AutoVision running:

```bash
cd autovision
cp .env.example .env
docker-compose up -d
```

Then:
- 🌐 Frontend: http://localhost:5173
- 📡 Backend API: http://localhost:8000
- 📚 API Docs: http://localhost:8000/docs

**Choose your path:**
| Path | Command | Time | For |
|------|---------|------|-----|
| Docker (Production) | `docker-compose up -d` | 5 min | Deployment, testing |
| Local Dev | `source venv/bin/activate && uvicorn app.main:app --reload` | 10 min | Feature development |
| Jenkins CI/CD | See [RUNNING_GUIDE.md](RUNNING_GUIDE.md#method-3-jenkins-cicd-pipeline) | 20 min | Team automation |

[📖 Full Commands & Troubleshooting](RUNNING_GUIDE.md)

---

## Repository Layout

```
autovision/
├── backend/
│   ├── app/
│   │   ├── api/routes/
│   │   │   ├── dataset.py        # Upload, preview, preprocess, session management
│   │   │   ├── training.py       # /training/start, /auto-start, /status, /results, WS
│   │   │   ├── inference.py      # /predict, /models, /chat
│   │   │   └── pilot.py          # /pilot/run, pilot WS
│   │   ├── core/
│   │   │   ├── config.py         # Settings (UPLOAD_DIR, MODEL_DIR, LLM settings)
│   │   │   ├── llm_engine.py     # Ollama integration + ALLOWED_MODELS_BY_TASK validation
│   │   │   └── task_router.py    # NEW: strict dataset-structure task detection
│   │   ├── models/
│   │   │   └── schemas.py        # Pydantic models: TaskType, ModelType, TrainingConfig, etc.
│   │   ├── services/
│   │   │   ├── preprocessor.py   # CSV + image preprocessing, detect_task_type, augmentation
│   │   │   ├── trainer.py        # sklearn + YOLO training entry point
│   │   │   ├── dl_trainer.py     # PyTorch CNN/ResNet/UNet/DeepLabV3 training
│   │   │   ├── pilot_runner.py   # Autonomous 5-step pipeline (analyze→preprocess→research→plan→train)
│   │   │   ├── metadata_store.py # JSON-based session persistence, training run history
│   │   │   └── trainers/         # NEW thin wrapper package
│   │   │       ├── __init__.py
│   │   │       ├── segmentation_trainer.py      # Validates + delegates to dl_trainer
│   │   │       ├── image_classification_trainer.py
│   │   │       ├── detection_trainer.py
│   │   │       └── tabular_trainer.py
│   │   └── utils/
│   │       ├── file_handler.py   # save_upload, extract_zip
│   │       └── logger.py
│   ├── tests/
│   │   ├── test_dataset.py
│   │   ├── test_training.py
│   │   ├── test_inference.py
│   │   ├── test_llm_engine.py
│   │   ├── test_pilot.py
│   │   ├── test_preprocessor.py
│   │   └── test_task_router.py   # NEW: 14 tests for routing + trainer guards
│   ├── requirements.txt
│   └── main.py
└── frontend/
    ├── src/
    │   ├── App.jsx               # Thin wrapper: <ToastProvider><MainPage /></ToastProvider>
    │   ├── pages/
    │   │   └── MainPage.jsx      # Single-page UI: Upload → Train → Results + History + Predict tabs
    │   └── api/
    │       └── client.js         # Axios wrappers for all backend endpoints + WS helpers
    ├── vite.config.js            # Proxy: /api → localhost:8000
    └── package.json
```

---

## Data Flow

### Upload → Preprocess → Train

```
User drops file
  │
  ▼
POST /dataset/upload
  • Saves to UPLOAD_DIR/{session_id}/
  • Detects format (CSV / image_folder / YOLO / COCO / zip)
  • Calls get_dataset_summary() → returns num_samples, num_classes, class_names immediately
  │
  ▼
[User selects task type or accepts auto-detected]
  │
  ▼
POST /dataset/preprocess
  • PreprocessConfig.task_type_hint overrides detection for non-CSV data
  • Runs CSV pipeline (scale, encode, split) OR image pipeline (resize, augment, split)
  • Writes preprocessed/ folder: train.csv/val.csv OR split_manifest.json OR seg_manifest.json
  • Stores dataset metadata to UPLOAD_DIR/{session_id}/metadata.json
  │
  ▼
POST /training/auto-start?session_id=...&task_type=...
  • Launches pilot_runner.PilotRunner in a background thread
  • Opens WebSocket at /training/ws/{session_id} for live log streaming
  │
  ▼
PilotRunner (5 steps):
  1. _analyze_dataset()     — calls task_router.detect_task() for canonical task
  2. _preprocess_dataset()  — (skipped if already preprocessed)
  3. _research_models()     — asks Ollama LLM for recommendation
  4. _plan_training()       — validates model via ALLOWED_MODELS_BY_TASK
  5. _execute_training()    — calls trainer.train_model() → correct sub-trainer
  │
  ▼
trainer.train_model()
  • Routes by task_type + model_type to:
    - trainer._train_sklearn_model()   (XGBoost, LightGBM, RandomForest, Linear, Ridge)
    - trainer._train_yolo()            (YOLOv8n/s/m, YOLOv8n-seg/s-seg)
    - dl_trainer.train_dl_classification()  (CNN, ResNet)
    - dl_trainer.train_dl_segmentation()    (UNet, DeepLabV3)
  │
  ▼
Results pushed over WebSocket:
  {status: "completed", metrics: {...}, best_model: "..."}
  Metrics stored in metadata.json training_runs[]
  Logs written to MODEL_DIR/{session_id}/{model_type}/train.log
```

---

## Task Type → Model Mapping

| Task | Allowed Models | Metrics |
|------|---------------|---------|
| classification | CNN, ResNet, XGBoost, LightGBM, RandomForest | accuracy, f1 |
| segmentation | UNet, DeepLabV3, YOLOv8n-seg, YOLOv8s-seg | iou, dice, pixel_accuracy |
| object_detection | YOLOv8n, YOLOv8s, YOLOv8m | map50, map50_95 |
| regression | XGBoost, LightGBM, RandomForest, Linear, Ridge | rmse, r2 |

---

## Key Files & Their Roles

### `backend/app/core/task_router.py` (NEW)
```python
TASK_IMAGE_SEGMENTATION = "image_segmentation"
TASK_IMAGE_CLASSIFICATION = "image_classification"
TASK_OBJECT_DETECTION = "object_detection"
TASK_TABULAR = "tabular"

def detect_task(dataset_path: Path) -> str:
    # Fail-fast: CSV → "tabular"
    # Fail-fast: images/ without masks/ → raises ValueError
    # images/ + masks/ → "image_segmentation"
    # class subdirs → "image_classification"
    # YOLO/COCO layout → "object_detection"
    # Else → raises ValueError
```

### `backend/app/core/llm_engine.py`
- `get_recommendation(task_type, summary)` → calls Ollama, validates JSON output
- `ALLOWED_MODELS_BY_TASK` dict gates LLM output per task
- `_validate_model_for_task(model_type_str, task_type)` → bool
- Falls back to `_rule_based_recommendation()` if LLM returns wrong model

### `backend/app/services/pilot_runner.py`
- `PilotRunner(session_id, task_type_override)` — core orchestrator
- `_map_model_name(model_name, task_type)` — hard-rejects tabular models for segmentation
- WS queue: asyncio.Queue populated by background thread via `loop.call_soon_threadsafe(queue.put_nowait, msg)`

### `backend/app/services/metadata_store.py`
- JSON files at `UPLOAD_DIR/{session_id}/metadata.json`
- `get_all_sessions()` → list of all sessions for History tab
- `get_session_summary(session_id)` → full detail including log_tail per run
- `record_training_run(...)` → appends/updates a run entry
- Training logs at `MODEL_DIR/{session_id}/{model_type}/train.log`

### `frontend/src/pages/MainPage.jsx`
- Three sequential sections: Upload → Train → Results
- History tab: `<HistoryPanel>` lists all sessions, expandable to show run metrics + logs
- `TASK_OPTIONS` + `selectedTask` state — user can override auto-detection before preprocessing
- WebSocket: `ws.onclose` handles final state, NOT `ws.onmessage`
- `autoTrain(sessionId, taskTypeOverride)` sends user-selected task to backend

---

## Environment Setup

### 🚀 Option 1: Docker Compose (Recommended for Production)

#### Quick Start (5 minutes)
```bash
# Clone and navigate
cd autovision

# Setup environment file
cp .env.example .env
# Edit .env if needed (defaults work for local development)

# Build images (first time: ~5 min)
docker-compose build

# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f backend    # Watch backend logs
docker-compose logs -f frontend   # Watch frontend logs
```

#### Access Services
| Service | URL | Purpose |
|---------|-----|---------|
| Frontend | http://localhost:5173 | Main web UI |
| Backend API | http://localhost:8000 | REST API |
| API Docs | http://localhost:8000/docs | Swagger/OpenAPI |
| Ollama | http://localhost:11434 | LLM service |

#### Verify Services Running
```bash
# Backend health check
curl http://localhost:8000/api/v1/system/info

# Frontend response
curl http://localhost:5173

# Ollama API
curl http://localhost:11434/api/tags

# View individual service logs
docker-compose logs --tail=50 backend
docker-compose logs --tail=50 frontend
docker-compose logs --tail=50 ollama
```

#### Common Docker Compose Commands
```bash
# Stop all services (keep volumes)
docker-compose down

# Stop and remove volumes (clear data)
docker-compose down -v

# Restart a specific service
docker-compose restart backend

# Execute command in container
docker-compose exec backend pytest tests/ -v
docker-compose exec backend /bin/bash           # Get shell access
docker-compose exec frontend /bin/sh            # Get frontend shell

# View container resource usage
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Rebuild images without cache
docker-compose build --no-cache

# View Docker images
docker images | grep autovision
```

#### Data Persistence
```bash
# Volumes are stored by Docker (automatic persistence)
docker volume ls | grep autovision

# Backup volumes
docker run --rm -v autovision_data:/data -v $(pwd):/backup \
  alpine tar czf /backup/autovision_data.tar.gz -C /data .

# Clear only data (keep containers)
docker-compose down

# Clean up everything (hard reset)
docker-compose down -v
docker system prune -a
```

---

### 🛠️ Option 2: Local Development (Manual Setup)

#### Backend
```bash
cd backend

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate          # macOS/Linux
venv\Scripts\activate             # Windows

# Install dependencies
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Run tests
pytest tests/ -v

# Start server
uvicorn app.main:app --reload --port 8000
```

#### Frontend
```bash
cd frontend

# Install dependencies
npm install

# Start dev server
npm run dev    # → http://localhost:5173

# Build for production
npm run build
```

#### Ollama (in separate terminal)
```bash
ollama serve
# In another terminal:
ollama pull llama3.2
```

---

### 🔄 Option 3: Jenkins CI/CD Pipeline

#### Jenkins Setup (One-time)

**Prerequisites:**
- Jenkins ≥ 2.300 installed
- Docker accessible from Jenkins
- Git plugin installed

**Initial Configuration:**
```bash
# Start Jenkins (if using Docker)
docker run -d \
  --name jenkins \
  -p 8080:8080 \
  -v jenkins_home:/var/jenkins_home \
  -v /var/run/docker.sock:/var/run/docker.sock \
  jenkins/jenkins:lts

# Get initial password
docker logs jenkins | grep "Initial Admin Password"

# Access at http://localhost:8080
```

**Install Plugins:**
1. Go to **Manage Jenkins** → **Plugins** → **Available**
2. Install these plugins:
   - Pipeline
   - Git
   - Docker Pipeline
   - JUnit Plugin
   - Cobertura Plugin

#### Create Jenkins Job

**Method 1: From Git Repository (Recommended)**
1. Jenkins Dashboard → **New Item**
2. Name: `autovision` → **Pipeline** → **OK**
3. **Configuration:**
   ```
   Definition: Pipeline script from SCM
   SCM: Git
   Repository URL: https://github.com/your-org/autovision.git
   Credentials: (setup Github credentials)
   Branch: */main
   Script Path: Jenkinsfile
   ```
4. **Save** → **Build Now**

**Method 2: Declarative Pipeline**
1. Jenkins Dashboard → **New Item**
2. Name: `autovision` → **Pipeline** → **OK**
3. Copy entire content of `Jenkinsfile` into **Pipeline script**
4. **Save** → **Build Now**

#### Monitor Pipeline

```
Pipeline Stages (9 total):
├── ✓ Checkout         (Clone repo)
├── ✓ Setup            (Python + Node)
├── ✓ Install Dependencies  (pip + npm)
├── ✓ Test             (pytest with coverage)
├── ✓ Lint             (flake8, optional)
├── ✓ Build Backend    (Docker image)
├── ✓ Build Frontend   (Multi-stage Nginx)
├── ✓ Deploy           (docker-compose up)
└── ✓ Health Check     (Verify endpoints)
```

**View Build Results:**
```
Jenkins → autovision → Build #1
├── Console Output    (Full logs)
├── Test Results      (Pass/fail tests)
├── Code Coverage     (Coverage report)
├── Artifacts         (Build logs)
```

#### Jenkins Commands (via CLI)

```bash
# Trigger build remotely
curl -X POST http://localhost:8080/job/autovision/build

# Get build status
curl http://localhost:8080/job/autovision/lastBuild/api/json | jq '.result'

# Get console output
curl http://localhost:8080/job/autovision/lastBuild/consoleText

# List all builds
curl http://localhost:8080/job/autovision/api/json | jq '.builds[].number'
```

#### Auto-Trigger Pipeline

**Option 1: Poll SCM** (Check every 15 minutes)
```
Build Triggers → Poll SCM
Schedule: H/15 * * * *
```

**Option 2: GitHub Webhook** (Trigger on push)
1. GitHub Repo → Settings → Webhooks → Add Webhook
2. Payload URL: `http://jenkins-server:8080/github-webhook/`
3. Events: Push events
4. Jenkins auto-detects this

---

## Component Roles

### 🐳 Docker Components

| Component | Role | Port | Volume |
|-----------|------|------|--------|
| **Backend** | FastAPI MLOps engine | 8000 | `/tmp/autovision/uploads`, `/tmp/autovision/models` |
| **Frontend** | React web UI | 5173 | None |
| **Ollama** | LLM service (llama3.2) | 11434 | `/root/.ollama` |

### 📚 Service Communication (Docker Networking)

```
User Browser
    ↓ (http://localhost:5173)
Frontend Container
    ↓ (http://backend:8000/api)
Backend Container
    ↓ (http://ollama:11434)
Ollama Container
```

**Note:** Inside Docker, services use service names (not localhost):
- Backend → Ollama: `http://ollama:11434`
- Frontend → Backend: `http://backend:8000`
- Host → Backend: `http://localhost:8000`

### 🔄 Jenkins Pipeline Roles

| Stage | Role | Command | Output |
|-------|------|---------|--------|
| **Checkout** | Clone Git repo | `git clone` | Source code in workspace |
| **Setup** | Install runtimes | `python3.11`, `node18` | Environments ready |
| **Install Deps** | Pip + npm install | `pip install`, `npm ci` | Dependencies cached |
| **Test** | Run pytest | `pytest tests/ -v` | JUnit XML + coverage |
| **Lint** | Code quality | `flake8 app/` | Lint report (optional) |
| **Build Backend** | Docker build | `docker build` | `autovision:backend-{TAG}` |
| **Build Frontend** | Docker build | `docker build` | `autovision:frontend-{TAG}` |
| **Deploy** | Start services | `docker-compose up` | Services running |
| **Health Check** | Verify endpoints | `curl /api/v1/system/info` | All green ✓ |

---

### Backend
```bash
cd frontend
npm install
npm run dev    # → http://localhost:5173
```

### Ollama
```bash
ollama serve                  # must be running on port 11434
ollama pull llama3.2          # model used by llm_engine.py
```

### Tests
```bash
cd backend
pytest tests/ -v
# Expect: ~60 tests passing (46 original + 14 new task-router tests)
```

---

## Known Architecture Decisions

1. **No database** — metadata is stored in flat JSON files per session. Fast enough for single-user local MLOps.

2. **CPU-only PyTorch** — installed via the `whl/cpu` index. Works without NVIDIA GPU.

3. **Thin trainer wrappers** — `services/trainers/*.py` add validation layers but delegate all logic to existing `trainer.py` and `dl_trainer.py`. No logic duplication.

4. **WS drain pattern** — the backend WS handler sends all log messages, then closes the connection. The frontend NEVER closes the WS from `onmessage` — only `onclose` triggers final state.

5. **Task override** — `PreprocessConfig.task_type_hint` lets the user force a task on the image dataset before the manifest is written. This determines which manifest format is created (`split_manifest.json` vs `seg_manifest.json` vs `data.yaml`).

6. **LLM validation** — if Ollama suggests a tabular model for a segmentation task, `_validate_model_for_task()` catches it and falls back to rule-based selection (returns UNet for segmentation).

---

## Common Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| Training fails with `ModuleNotFoundError: torch` | PyTorch not installed | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu` |
| Samples/Classes show "—" after upload | Upload response missing summary | `get_dataset_summary()` is called in upload endpoint; check backend logs |
| Classes not detected for nested ZIP | Double-nesting e.g. `animals/animals/cats/` | `_find_dataset_root()` in `preprocessor.py` descends single-subdir chains automatically |
| LLM picks XGBoost for segmentation task | Model validation missing | Fixed: `_validate_model_for_task()` + `_map_model_name()` task-aware guard |
| `images/` folder without `masks/` passes as classification | task_router didn't check | Fixed: explicit `images/` detection raises ValueError if no `masks/` sibling |
| Test suite breaks with "event loop closed" | `asyncio.run()` destroys global loop | Use `asyncio.new_event_loop()` + `loop.close()` in async tests |

---

## API Endpoints Summary

```
POST /api/v1/dataset/upload                     — Upload dataset file
GET  /api/v1/dataset/preview/{session_id}       — Preview dataset structure
POST /api/v1/dataset/preprocess                 — Preprocess (with task_type_hint)
GET  /api/v1/dataset/recommend/{session_id}     — LLM model recommendation
GET  /api/v1/dataset/sessions                   — List all sessions
GET  /api/v1/dataset/session/{session_id}/summary — Full session detail
DELETE /api/v1/dataset/{session_id}             — Delete session

POST /api/v1/training/start                     — Manual training start
POST /api/v1/training/auto-start                — Autonomous pipeline
GET  /api/v1/training/status/{session_id}       — Training status
GET  /api/v1/training/results/{session_id}      — Training results
GET  /api/v1/training/log/{session_id}/{model}  — Training log file
WS   /api/v1/training/ws/{session_id}           — Live log stream

POST /api/v1/inference/predict                  — Run prediction
GET  /api/v1/inference/models/{session_id}      — List trained models
GET  /api/v1/inference/chat                     — Chat with LLM about dataset

GET  /api/v1/system/info                        — System/GPU info
GET  /api/v1/llm/status                         — Ollama status
```

---

## 🐳 Docker & CI/CD (NEW)

### Docker Files & Build Process

| File | Purpose | Tech Stack | Size |
|------|---------|-----------|------|
| **backend/Dockerfile** | ML engine container | Python 3.11 + FastAPI + PyTorch (CPU) | ~2 GB |
| **frontend/Dockerfile** | Web UI container | Node 18 → Nginx (multi-stage) | ~100 MB |
| **docker-compose.yml** | Orchestration | Docker Compose v3.8 | - |

### Build & Run with Docker

```bash
# Build images
docker-compose build                    # Full build with layers cached
docker-compose build --no-cache         # Rebuild without cache

# Start services
docker-compose up -d                    # Run in background
docker-compose up                       # Run with visible logs

# Check services
docker-compose ps                       # List containers & status
docker-compose images                   # List images

# View logs
docker-compose logs                     # All services
docker-compose logs -f backend          # Follow backend logs
docker-compose logs -f frontend         # Follow frontend logs
docker-compose logs -f ollama           # Follow ollama logs
docker-compose logs --tail=100 backend  # Last 100 lines

# Stop services
docker-compose stop                     # Stop (keep containers)
docker-compose down                     # Stop & remove containers
docker-compose down -v                  # Stop & remove volumes (DELETE DATA)

# Restart
docker-compose restart                  # Restart all services
docker-compose restart backend          # Restart specific service

# Execute commands in containers
docker-compose exec backend pytest tests/ -v          # Run tests
docker-compose exec backend /bin/bash                 # Shell access
docker-compose exec backend python -c "print('hi')"  # Run Python
docker-compose exec frontend /bin/sh                  # Frontend shell
docker-compose exec frontend npm run build            # Build frontend
```

### Key Docker Features & Configuration

**Security:**
- ✅ Non-root user (`mlops`) in backend container
- ✅ Read-only volumes where appropriate
- ✅ Network isolation via bridge network
- ✅ No exposed credentials

**Performance:**
- ✅ Layer caching (requirements before source code)
- ✅ Multi-stage frontend build (removes node_modules from final image)
- ✅ Alpine-based images where possible
- ✅ `.dockerignore` optimization

**Networking:**
```
Docker Bridge Network: autovision
├── backend:8000   (http://backend:8000 from inside containers)
├── frontend:80    (served via nginx on :5173 externally)
└── ollama:11434   (http://ollama:11434 from inside containers)

Host Access:
├── http://localhost:5173      → frontend
├── http://localhost:8000      → backend
└── http://localhost:11434     → ollama
```

**Volumes (Data Persistence):**
```
autovision_data:
├── /tmp/autovision/uploads/    (user datasets)
├── /tmp/autovision/models/     (trained models)
└── /tmp/autovision/logs/       (training logs)

ollama_data:
└── /root/.ollama/              (LLM models & cache)
```

### Jenkins Pipeline (Jenkinsfile) — 9 Stages

```
╔════════════════════════════════════════════════════════════════════════════╗
║                       AutoVision Jenkins Pipeline                          ║
╠════════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║  Stage 1: CHECKOUT                                                        ║
║  ├─ git clone repository                                                 ║
║  └─ Display commit hash & message                                        ║
║                                                                            ║
║  Stage 2: SETUP (Parallel)                                               ║
║  ├─ Setup Python 3.11 environment                                        ║
║  └─ Setup Node 18 environment                                            ║
║                                                                            ║
║  Stage 3: INSTALL DEPENDENCIES (Parallel)                                ║
║  ├─ Backend: pip install -r requirements.txt + PyTorch (CPU)             ║
║  └─ Frontend: npm ci (clean install, uses package-lock.json)             ║
║                                                                            ║
║  Stage 4: TEST                                                            ║
║  ├─ pytest tests/ -v                                                      ║
║  ├─ Generate coverage report                                             ║
║  └─ Publish JUnit XML results                                            ║
║                                                                            ║
║  Stage 5: LINT (Optional, continues on failure)                          ║
║  ├─ flake8 app/ --max-line-length=120                                    ║
║  └─ (warnings don't fail pipeline)                                       ║
║                                                                            ║
║  Stage 6: BUILD BACKEND IMAGE                                            ║
║  ├─ docker build --tag autovision:backend-{BUILD_NUMBER}-{COMMIT}        ║
║  ├─ Layer caching enabled                                                ║
║  └─ Size: ~2 GB (includes PyTorch CPU)                                   ║
║                                                                            ║
║  Stage 7: BUILD FRONTEND IMAGE                                           ║
║  ├─ docker build (multi-stage: Node → Nginx)                             ║
║  ├─ Final image size: ~100 MB                                            ║
║  └─ Served via Nginx reverse proxy                                       ║
║                                                                            ║
║  Stage 8: DEPLOY                                                          ║
║  ├─ docker-compose down (stop old containers)                            ║
║  ├─ docker-compose up -d (start new containers)                          ║
║  ├─ Wait for services to initialize (10s)                                ║
║  └─ Display service logs                                                 ║
║                                                                            ║
║  Stage 9: HEALTH CHECK                                                    ║
║  ├─ curl http://localhost:8000/api/v1/system/info (backend)              ║
║  ├─ curl http://localhost:5173 (frontend)                                ║
║  ├─ curl http://localhost:11434/api/tags (ollama)                        ║
║  └─ Retry up to 5 times with 2s delays                                   ║
║                                                                            ║
║  ✅ SUCCESS: All services healthy, pipeline complete                     ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
```

### Jenkins Commands

```bash
# Trigger build remotely
curl -X POST http://jenkins-server:8080/job/autovision/build

# Trigger with parameters
curl -X POST http://jenkins-server:8080/job/autovision/build \
  -u user:token \
  -H "Content-Type: application/json" \
  -d '{"parameter": [{"name": "BRANCH", "value": "main"}]}'

# Get last build status
curl http://jenkins-server:8080/job/autovision/lastBuild/api/json | jq '.result'

# Get full build info
curl http://jenkins-server:8080/job/autovision/lastBuild/api/json | jq '.{result, duration, timestamp}'

# View console output
curl http://jenkins-server:8080/job/autovision/lastBuild/consoleText

# Get build log (first 1000 lines)
curl http://jenkins-server:8080/job/autovision/lastBuild/consoleText | head -1000

# List all builds
curl http://jenkins-server:8080/job/autovision/api/json | jq '.builds[].number'

# Get test results
curl http://jenkins-server:8080/job/autovision/lastBuild/testReport/api/json

# Get coverage report
curl http://jenkins-server:8080/job/autovision/lastBuild/Coverage_Report/api/json
```

### Jenkins Configuration: Auto-Trigger Options

**Option 1: Poll SCM (Check every N minutes)**
```
Build Triggers:
  ✓ Poll SCM
  Schedule: H/15 * * * *      # Check every 15 minutes
```

**Option 2: GitHub Webhook (Trigger on push)**
```
1. GitHub Repo → Settings → Webhooks → Add Webhook
2. Payload URL: http://jenkins-server:8080/github-webhook/
3. Events: Let me select individual events
   ✓ Push events
   ✓ Pull requests
4. Active: ✓
5. Jenkins auto-detects when repo changes
```

**Option 3: Scheduled (Cron)**
```
Build Triggers:
  ✓ Build periodically
  Schedule: H 2 * * *         # 2 AM daily
```

### Jenkins Best Practices

```groovy
// In Jenkinsfile

// 1. Use environment variables
environment {
    BUILD_TAG = ""${BUILD_NUMBER}-${GIT_COMMIT.take(7)}"
    DOCKER_IMAGE = "autovision:backend-${BUILD_TAG}"
}

// 2. Parallel stages for speed
parallel {
    stage('Setup Python') { ... }
    stage('Setup Node') { ... }
}

// 3. Catch errors without failing
catchError(buildResult: 'SUCCESS', stageResult: 'UNSTABLE') {
    sh 'flake8 app/ || true'  # Lint won't fail pipeline
}

// 4. Archive artifacts
archiveArtifacts artifacts: '**/test-results.xml'
publishHTML([reportDir: 'coverage', reportFiles: 'index.html'])

// 5. Post actions
post {
    always {
        // Cleanup
        sh 'docker-compose logs > docker.log || true'
    }
    failure {
        // Debug on failure
        sh 'docker ps -a'
        sh 'docker-compose logs'
    }
}
```

### Integration: How Docker & Jenkins Work Together

**Development Loop:**
```
Developer pushes code to Git
    ↓
GitHub sends webhook to Jenkins
    ↓
Jenkins clones repo
    ↓
Jenkinsfile stages execute:
  1. Setup environments
  2. Install dependencies
  3. Run tests
  4. Build Docker images
  5. Deploy with docker-compose
  6. Health check
    ↓
Services running at:
  - http://localhost:5173 (frontend)
  - http://localhost:8000 (backend)
  - http://localhost:11434 (ollama)
```

**Image Versioning:**
```
Build #1:  autovision:backend-1-a1b2c3d
Build #2:  autovision:backend-2-e4f5g6h
...
Latest:    autovision:backend-latest

Tags persisted via docker-compose.yml
```

---

### Setup & Deployment
- See **[DOCKER_SETUP.md](DOCKER_SETUP.md)** for complete setup guide
- See **[DOCKER_CI_CD_README.md](DOCKER_CI_CD_README.md)** for quick start
- See **[RUNNING_GUIDE.md](RUNNING_GUIDE.md)** for all commands & troubleshooting
- Quick start: `docker-compose up -d`
- Jenkins setup: Configure Jenkinsfile from Git repo

### Configuration Files
- **`.env.example`** — Environment variables template
- **`docker-compose.yml`** — Service orchestration
- **`Jenkinsfile`** — CI/CD pipeline definition
- **`backend/Dockerfile`** — Backend container
- **`frontend/Dockerfile`** — Frontend container

---

## Troubleshooting Guide

### Container Won't Start

**Problem:** `docker-compose up -d` fails silently, or container exits

```bash
# Solution 1: Check logs
docker-compose logs backend
docker-compose logs frontend
docker-compose logs ollama

# Solution 2: Run without -d to see output
docker-compose up

# Solution 3: Check container status
docker-compose ps

# Solution 4: Restart from clean state
docker-compose down -v
docker-compose build --no-cache
docker-compose up -d
```

### Port Already in Use

**Problem:** "Address already in use" or port conflict

```bash
# Find what's using port 8000
lsof -i :8000
lsof -i :5173
lsof -i :11434

# Kill the process
kill -9 <PID>

# Or change port in .env
BACKEND_PORT=8001
FRONTEND_PORT=5174
OLLAMA_PORT=11435

# Then restart
docker-compose down
docker-compose up -d
```

### Backend Can't Reach Ollama

**Problem:** Backend errors when requesting Ollama (timeouts, connection refused)

```bash
# Inside running container
docker-compose exec backend curl http://ollama:11434/api/tags

# If fails on Docker Desktop for Mac/Windows
# Edit docker-compose.yml:
# Change: http://ollama:11434
# To:     http://host.docker.internal:11434

# If fails on Linux:
# Find your host IP
hostname -I

# Edit docker-compose.yml:
# Change: http://ollama:11434
# To:     http://192.168.1.100:11434  # Your host IP
```

### Tests Failing in Docker

**Problem:** Tests pass locally but fail in Docker/Jenkins

```bash
# Run tests with verbose output
docker-compose exec backend pytest tests/ -vv --tb=long

# Check environment variables
docker-compose exec backend env | grep OLLAMA
docker-compose exec backend env | grep PYTHONPATH

# Verify data directories exist
docker-compose exec backend ls -la /tmp/autovision/

# Check permissions
docker-compose exec backend ls -la /tmp/autovision/models/
```

### Jenkins Can't Access Docker

**Problem:** Jenkins pipeline fails at "docker build" stage

```bash
# On Linux: Add Jenkins user to docker group
sudo usermod -aG docker jenkins
sudo systemctl restart jenkins

# On Linux: Via socket (if Jenkins in container)
docker run ... \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /usr/bin/docker:/usr/bin/docker \
  ...

# Check Docker daemon running
docker ps

# Check Jenkins Docker permissions
docker exec jenkins docker ps
```

### Frontend Not Loading

**Problem:** http://localhost:5173 times out or gives 404

```bash
# Check if frontend container is running
docker-compose ps frontend

# Check frontend logs
docker-compose logs -f frontend

# Verify build succeeded
docker-compose logs frontend | grep -i "error\|build"

# Test directly from container
docker-compose exec frontend curl http://localhost:80

# Check nginx config
docker-compose exec frontend cat /etc/nginx/conf.d/default.conf
```

### Models Not Training

**Problem:** Training starts but hangs or uses no GPU

```bash
# Check PyTorch installation in container
docker-compose exec backend python -c "import torch; print(torch.cuda.is_available())"
# Expected: False (CPU-only, which is expected)

# Check available disk space
docker-compose exec backend df -h /tmp/autovision/

# Check permissions on data directories
docker-compose exec backend ls -la /tmp/autovision/models/

# Check logs
docker-compose logs -f backend | grep -i "train\|error"

# Run specific test
docker-compose exec backend pytest tests/test_trainer.py -v
```

### Ollama Model Not Loading

**Problem:** "Model not found" or "Could not load model"

```bash
# List available models
docker-compose exec ollama ollama list
# Expected: llama3.2 in list

# If not present, pull it
docker-compose exec ollama ollama pull llama3.2

# Test Ollama health
curl http://localhost:11434/api/tags

# Check Ollama logs
docker-compose logs ollama
```

### Health Check Failing

**Problem:** Jenkins stage "HEALTH CHECK" fails, endpoints returning 500

```bash
# Check each endpoint from Docker Desktop/Linux
curl http://localhost:8000/api/v1/system/info
curl http://localhost:5173
curl http://localhost:11434/api/tags

# If 500 errors, check logs
docker-compose logs backend | tail -100
docker-compose logs frontend | tail -100

# Restart failed service
docker-compose restart backend

# Wait and retry (sometimes services need time)
sleep 10
curl http://localhost:8000/api/v1/system/info
```

### Volumes Not Persisting

**Problem:** Models disappear after `docker-compose down`

```bash
# Verify volumes are defined in docker-compose.yml
grep "autovision_data\|ollama_data" docker-compose.yml

# List all volumes
docker volume ls

# Check volume contents
docker volume inspect autovision_data
# Look at "Mountpoint" path

# If volume is missing
docker-compose down -v        # Remove all volumes
docker-compose up -d          # Recreate volumes
```

### Build Fails on Windows

**Problem:** Docker build fails with path or line-ending issues

```bash
# Ensure .gitattributes is committed
git add .gitattributes
git commit

# Convert line endings
dos2unix backend/Dockerfile
dos2unix frontend/Dockerfile

# Or rebuild with --no-cache
docker-compose build --no-cache

# Check Docker Desktop settings (Settings → Resources → File Sharing)
# Ensure C:\Users is shared
```

### Memory Issues (Container Killed)

**Problem:** Containers get killed, OOMKilled messages

```bash
# Check Docker memory limit
docker stats

# Increase Docker Desktop memory (Settings → Resources)
# Min: 4GB, Recommended: 8GB for PyTorch

# Or reduce in docker-compose.yml
services:
  backend:
    mem_limit: 4g       # Add this line
```

### Jenkins Pipeline Takes 30+ Minutes

**Problem:** First run is slow (expected), but even cached builds are slow

```bash
# Check layer caching
docker history autovision:backend-latest | head -20

# Rebuild with cache
docker-compose build

# Monitor build stages
docker-compose build --progress=plain

# Check network issues
docker-compose down
docker-compose up -d
docker-compose logs

# Use BuildKit for faster builds
export DOCKER_BUILDKIT=1
docker-compose build
```

### Git Credentials Error in Jenkins

**Problem:** "Permission denied" or "Could not authenticate" during `git clone`

```bash
# In Jenkins GUI:
# 1. Manage Jenkins → Credentials → Add Credentials
# 2. SSH or GitHub token
# 3. Copy credential ID

# In Jenkinsfile:
withCredentials([usernamePassword(credentialsId: 'github-creds', ...)]) {
    sh 'git clone ...'
}

# Or set Git URL with token
git clone https://token@github.com/org/autovision.git
```

---

## Performance Optimization Tips

### Speed Up Docker Builds

```bash
# Use BuildKit (faster parallel builds)
export DOCKER_BUILDKIT=1
docker-compose build

# Build in parallel on multi-core
docker-compose build --parallel

# Use pre-built base images
docker pull python:3.11-slim beforehand
docker pull node:18-alpine beforehand
```

### Speed Up Tests

```bash
# Run tests in parallel (pytest-xdist)
pip install pytest-xdist
pytest tests/ -n auto

# Run only new/changed tests
pytest --lf              # Last failed
pytest --ff              # Failed first
```

### Reduce Docker Image Size

```bash
# Current sizes
docker images | grep autovision

# Backend: ~2GB (includes PyTorch CPU)
# Frontend: ~100MB (Nginx + React build)

# For production, consider:
# - Multi-stage builds (already done)
# - Alpine base images (already done for frontend)
# - Layer caching (already optimized)
```

---

## Getting Help

**Quick Reference:** See [RUNNING_GUIDE.md](RUNNING_GUIDE.md) for:
- All 50+ commands organized by category
- Quick decision matrix (Docker vs Local vs Jenkins)
- Common operations with examples
- Workflow examples (dev, production, CI/CD)

**Documentation:**
- [DOCKER_SETUP.md](DOCKER_SETUP.md) — Comprehensive Docker guide
- [DOCKER_CI_CD_README.md](DOCKER_CI_CD_README.md) — Quick CI/CD start
- [PROJECT_INFO.md](PROJECT_INFO.md) — Complete project reference (this file)
- [RUNNING_GUIDE.md](RUNNING_GUIDE.md) — Commands & troubleshooting

**Debugging Checklist:**
- [ ] Check logs: `docker-compose logs -f service`
- [ ] Verify containers: `docker-compose ps`
- [ ] Test connectivity: `docker-compose exec service curl <endpoint>`
- [ ] Check volumes: `docker volume inspect volumename`
- [ ] Verify environment: `docker-compose exec service env`
- [ ] Test health: `curl http://localhost:8000/api/v1/system/info`

**Next Steps:**
1. Start with [RUNNING_GUIDE.md](RUNNING_GUIDE.md)
2. Run `docker-compose up -d`
3. Visit http://localhost:5173
4. Upload a dataset and train a model
5. Check logs if any issues: `docker-compose logs -f`
