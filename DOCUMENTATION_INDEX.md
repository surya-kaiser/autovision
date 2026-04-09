# AutoVision Documentation Index

Complete guide to all AutoVision documentation. Start here to find what you need.

---

## 🚀 Getting Started (Choose Your Path)

### I Want to Run AutoVision Right Now
→ **[RUNNING_GUIDE.md](RUNNING_GUIDE.md)** — 3-minute quick start with Docker
- Copy/paste commands to start everything
- 3 execution methods: Docker, Local Dev, Jenkins
- Complete troubleshooting section
- 50+ useful commands organized by category

### I Want to Understand the Project
→ **[PROJECT_INFO.md](PROJECT_INFO.md)** — Complete project reference
- What is AutoVision and why it exists
- Full architecture and data flow
- Component roles and responsibilities
- Environment setup options
- Integration with Docker & Jenkins

### I Want to Set Up Docker/Jenkins
→ **[DOCKER_SETUP.md](DOCKER_SETUP.md)** — Comprehensive Docker guide
- Step-by-step Docker setup
- Jenkins installation and configuration
- Production deployment checklist
- Volume management
- Common issues and solutions

### I Want Quick CI/CD Start
→ **[DOCKER_CI_CD_README.md](DOCKER_CI_CD_README.md)** — 5-minute CI/CD quick start
- Fastest way to get Jenkins running
- Copy/paste job configuration
- Common commands
- Quick troubleshooting

---

## 📚 Documentation Overview

| Document | Purpose | Best For | Read Time |
|----------|---------|----------|-----------|
| [RUNNING_GUIDE.md](RUNNING_GUIDE.md) | Commands, troubleshooting, workflows | Users who want to execute | 15 min |
| [PROJECT_INFO.md](PROJECT_INFO.md) | Architecture, design, setup paths | Developers & architects | 20 min |
| [DOCKER_SETUP.md](DOCKER_SETUP.md) | Complete Docker/Jenkins setup | DevOps & deployment | 30 min |
| [DOCKER_CI_CD_README.md](DOCKER_CI_CD_README.md) | Quick CI/CD setup | Teams wanting fast start | 5 min |
| [SETUP_COMPLETE.md](SETUP_COMPLETE.md) | Initial setup confirmation | First-time users | 2 min |
| [USER_GUIDE.md](USER_GUIDE.md) | How to use the UI | End users | 10 min |
| [QUICKSTART.md](QUICKSTART.md) | Original quick start | Reference | - |

---

## 📖 Read These Documents

### For Non-Technical Users
1. **[USER_GUIDE.md](USER_GUIDE.md)** — How to use AutoVision's web interface
2. **[RUNNING_GUIDE.md](RUNNING_GUIDE.md#method-1-docker-compose-fastest)** — How to start it (just 3 commands)

### For Developers (Local Development)
1. **[PROJECT_INFO.md](PROJECT_INFO.md)** — Overview & architecture
2. **[RUNNING_GUIDE.md](RUNNING_GUIDE.md#method-2-local-development-full-control)** — Local dev setup
3. **[PROJECT_INFO.md](PROJECT_INFO.md#key-files--their-roles)** — Understanding the codebase

### For DevOps/Platform Engineers
1. **[DOCKER_SETUP.md](DOCKER_SETUP.md)** — Complete setup
2. **[DOCKER_CI_CD_README.md](DOCKER_CI_CD_README.md)** — Quick start
3. **[PROJECT_INFO.md](PROJECT_INFO.md#docker--cicd)** — Architecture & integration
4. **[RUNNING_GUIDE.md](RUNNING_GUIDE.md#method-3-jenkins-cicd-pipeline)** — Jenkins commands

### For Teams
1. **[DOCKER_CI_CD_README.md](DOCKER_CI_CD_README.md)** — Get Jenkins running (5 min)
2. **[RUNNING_GUIDE.md](RUNNING_GUIDE.md#method-3-jenkins-cicd-pipeline)** — Pipeline details
3. **[PROJECT_INFO.md](PROJECT_INFO.md#jenkins-pipeline-jenkinsfile--9-stages)** — Deep dive

---

## 🎯 Quick Links by Task

### "How do I run AutoVision?"
→ **[RUNNING_GUIDE.md — Method 1](RUNNING_GUIDE.md#-method-1-docker-compose-fastest)**
```bash
cd autovision
cp .env.example .env
docker-compose up -d
```

### "I want to develop locally"
→ **[RUNNING_GUIDE.md — Method 2](RUNNING_GUIDE.md#-method-2-local-development-full-control)**
```bash
cd backend && python -m venv venv && source venv/bin/activate
pip install -r requirements.txt && uvicorn app.main:app --reload
```

### "How do I set up Jenkins?"
→ **[DOCKER_CI_CD_README.md](DOCKER_CI_CD_README.md)** or **[RUNNING_GUIDE.md — Method 3](RUNNING_GUIDE.md#-method-3-jenkins-cicd-pipeline)**

### "Something isn't working"
→ **[RUNNING_GUIDE.md — Troubleshooting](RUNNING_GUIDE.md#-quick-troubleshooting)**

### "I want to understand the architecture"
→ **[PROJECT_INFO.md — Overview](PROJECT_INFO.md#overview)** and **[PROJECT_INFO.md — Repository Layout](PROJECT_INFO.md#repository-layout)**

### "How do I use the web UI?"
→ **[USER_GUIDE.md](USER_GUIDE.md)**

### "Show me all the Docker commands"
→ **[RUNNING_GUIDE.md — Commands Reference](RUNNING_GUIDE.md#-useful-commands-reference)**

### "Show me all the Jenkins commands"
→ **[PROJECT_INFO.md — Jenkins Commands](PROJECT_INFO.md#jenkins-commands)**

---

## 💡 Common Scenarios

### Scenario 1: New Developer Starting Today
1. Read: [RUNNING_GUIDE.md](RUNNING_GUIDE.md) (15 min)
2. Run: `docker-compose up -d` (5 min)
3. Test: Visit http://localhost:5173
4. Read: [PROJECT_INFO.md](PROJECT_INFO.md#repository-layout) (10 min)
5. Develop: Make changes, test locally

### Scenario 2: Team CI/CD Setup
1. Read: [DOCKER_CI_CD_README.md](DOCKER_CI_CD_README.md) (5 min)
2. Read: [RUNNING_GUIDE.md — Jenkins](RUNNING_GUIDE.md#-method-3-jenkins-cicd-pipeline) (10 min)
3. Follow: Setup steps in [DOCKER_SETUP.md](DOCKER_SETUP.md#jenkins-setup) (20 min)
4. Configure: Jenkins job from [Jenkinsfile](Jenkinsfile)
5. Test: Push code, watch pipeline run

### Scenario 3: Production Deployment
1. Read: [DOCKER_SETUP.md](DOCKER_SETUP.md) (30 min)
2. Follow: Production Checklist in [DOCKER_SETUP.md](DOCKER_SETUP.md#production-deployment-checklist)
3. Configure: `.env` for production
4. Deploy: Via Jenkins or docker-compose
5. Monitor: Health checks and logs

### Scenario 4: Troubleshooting Issues
1. Check: [RUNNING_GUIDE.md — Troubleshooting](RUNNING_GUIDE.md#-quick-troubleshooting)
2. If not found: [PROJECT_INFO.md — Troubleshooting](PROJECT_INFO.md#troubleshooting-guide)
3. Logs: `docker-compose logs -f service`
4. Debug: [RUNNING_GUIDE.md — Debugging Tips](RUNNING_GUIDE.md#-quick-troubleshooting)

---

## 📄 File Descriptions

### Configuration Files
- **`.env.example`** — Template for all environment variables (60+ settings)
- **`docker-compose.yml`** — Docker Compose configuration (3 services)
- **`Jenkinsfile`** — CI/CD pipeline definition (9 stages)
- **`backend/Dockerfile`** — Backend container with Python 3.11 + PyTorch
- **`frontend/Dockerfile`** — Frontend container with Node 18 + Nginx

### Documentation Files
- **`PROJECT_INFO.md`** — Complete project reference (890+ lines)
- **`RUNNING_GUIDE.md`** — Commands & quick start (600+ lines)
- **`DOCKER_SETUP.md`** — Docker/Jenkins setup (500+ lines)
- **`DOCKER_CI_CD_README.md`** — Quick CI/CD start (200+ lines)
- **`USER_GUIDE.md`** — Web UI usage guide
- **`SETUP_COMPLETE.md`** — Initial setup confirmation
- **`QUICKSTART.md`** — Original quick start
- **`DOCUMENTATION_INDEX.md`** — This file

---

## 🔍 Search by Topic

### Docker & Containers
- [DOCKER_SETUP.md](DOCKER_SETUP.md) — Full guide
- [RUNNING_GUIDE.md #Method 1](RUNNING_GUIDE.md#-method-1-docker-compose-fastest) — Quick start
- [PROJECT_INFO.md #Docker](PROJECT_INFO.md#-option-1-docker-compose-recommended-for-production) — Architecture

### Jenkins & CI/CD
- [DOCKER_CI_CD_README.md](DOCKER_CI_CD_README.md) — Quick start
- [RUNNING_GUIDE.md #Method 3](RUNNING_GUIDE.md#-method-3-jenkins-cicd-pipeline) — Commands
- [PROJECT_INFO.md #Jenkins](PROJECT_INFO.md#jenkins-pipeline-jenkinsfile--9-stages) — Pipeline details

### Local Development
- [RUNNING_GUIDE.md #Method 2](RUNNING_GUIDE.md#-method-2-local-development-full-control) — Setup & workflow
- [PROJECT_INFO.md #Repository Layout](PROJECT_INFO.md#repository-layout) — Code structure

### Troubleshooting
- [RUNNING_GUIDE.md #Troubleshooting](RUNNING_GUIDE.md#-quick-troubleshooting) — Quick fixes
- [PROJECT_INFO.md #Troubleshooting](PROJECT_INFO.md#troubleshooting-guide) — Comprehensive guide

### Commands & References
- [RUNNING_GUIDE.md #Commands](RUNNING_GUIDE.md#-useful-commands-reference) — All docker-compose, pytest, etc.
- [PROJECT_INFO.md #Jenkins Commands](PROJECT_INFO.md#jenkins-commands) — Jenkins CLI

### Architecture & Design
- [PROJECT_INFO.md #Overview](PROJECT_INFO.md#overview) — What AutoVision is
- [PROJECT_INFO.md #Data Flow](PROJECT_INFO.md#data-flow) — How data moves
- [PROJECT_INFO.md #Task Mapping](PROJECT_INFO.md#task-type--model-mapping) — Model selection

### UI & Usage
- [USER_GUIDE.md](USER_GUIDE.md) — How to use the web interface
- [PROJECT_INFO.md #Key Files](PROJECT_INFO.md#key-files--their-roles) — Frontend components

---

## ✅ Documentation Checklist

- ✅ **Getting Started** — 3 execution methods documented
- ✅ **Commands** — 50+ commands with examples
- ✅ **Architecture** — Data flow & component roles explained
- ✅ **Troubleshooting** — 15+ common issues with solutions
- ✅ **Docker** — Multi-stage builds, security hardening, volumes
- ✅ **Jenkins** — 9-stage pipeline, auto-triggers, CLI commands
- ✅ **Local Dev** — Setup venv, run tests, debug workflow
- ✅ **UI Guide** — Web interface documentation
- ✅ **Performance** — Tips for speeding up builds & tests
- ✅ **Production** — Deployment checklist & best practices

---

## 🎓 Learning Path

### For Everyone (Start Here)
1. **[RUNNING_GUIDE.md](RUNNING_GUIDE.md)** — 3-method overview (15 min)
2. **[USER_GUIDE.md](USER_GUIDE.md)** — How to use the app (10 min)

### For Developers
3. **[PROJECT_INFO.md](PROJECT_INFO.md)** — Architecture & design (20 min)
4. **[RUNNING_GUIDE.md #Method 2](RUNNING_GUIDE.md#-method-2-local-development-full-control)** — Local dev setup (10 min)

### For DevOps Engineers
3. **[DOCKER_CI_CD_README.md](DOCKER_CI_CD_README.md)** — Quick start (5 min)
4. **[DOCKER_SETUP.md](DOCKER_SETUP.md)** — Complete setup (30 min)
5. **[RUNNING_GUIDE.md #Method 3](RUNNING_GUIDE.md#-method-3-jenkins-cicd-pipeline)** — Jenkins details (15 min)

### For Troubleshooting
- **[RUNNING_GUIDE.md #Troubleshooting](RUNNING_GUIDE.md#-quick-troubleshooting)** — Quick fixes
- **[PROJECT_INFO.md #Troubleshooting](PROJECT_INFO.md#troubleshooting-guide)** — Deep dive

---

## 📞 Getting Help

### Quick Questions
- **"How do I start it?"** → [RUNNING_GUIDE.md #Method 1](RUNNING_GUIDE.md#-method-1-docker-compose-fastest)
- **"How do I develop?"** → [RUNNING_GUIDE.md #Method 2](RUNNING_GUIDE.md#-method-2-local-development-full-control)
- **"How do I set up Jenkins?"** → [DOCKER_CI_CD_README.md](DOCKER_CI_CD_README.md)
- **"Something broke"** → [RUNNING_GUIDE.md #Troubleshooting](RUNNING_GUIDE.md#-quick-troubleshooting)

### Detailed Information
- **Architecture** → [PROJECT_INFO.md](PROJECT_INFO.md)
- **Commands** → [RUNNING_GUIDE.md](RUNNING_GUIDE.md)
- **Docker/Jenkins Setup** → [DOCKER_SETUP.md](DOCKER_SETUP.md)

### All Commands
- **Docker Compose** → [RUNNING_GUIDE.md #Commands](RUNNING_GUIDE.md#-useful-commands-reference)
- **Backend** → [RUNNING_GUIDE.md #Commands](RUNNING_GUIDE.md#-useful-commands-reference)
- **Frontend** → [RUNNING_GUIDE.md #Commands](RUNNING_GUIDE.md#-useful-commands-reference)
- **Jenkins** → [PROJECT_INFO.md #Jenkins Commands](PROJECT_INFO.md#jenkins-commands)

---

## 📊 Documentation Statistics

- **Total Lines:** 2,500+ across all docs
- **Code Examples:** 100+
- **Commands:** 50+
- **Architecture Diagrams:** 5+
- **Troubleshooting Issues:** 15+
- **Time to First Success:** 5-10 minutes (Docker) or 15-20 minutes (Local)

---

## ⚙️ Technical Stack

**verified Working Versions:**
- Python: 3.11
- Node.js: 18
- Docker: 20.10+
- Docker Compose: 2.0+
- FastAPI: 0.100+
- React: 18.2+
- Ollama: Latest (llama3.2)
- Jenkins: 2.300+

---

**Ready to start?** → **[RUNNING_GUIDE.md](RUNNING_GUIDE.md)** or **[DOCKER_CI_CD_README.md](DOCKER_CI_CD_README.md)**
