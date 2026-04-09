// ============================================================================
// AutoVision MLOps — Jenkins Declarative Pipeline
// ============================================================================
// Stages:
//   1. Checkout         — Clone/update repository
//   2. Setup            — Python venv + Node check (parallel)
//   3. Install Deps     — Backend pip + frontend npm (parallel)
//   4. Test             — pytest with XML + coverage reports
//   5. Lint             — flake8 (non-blocking)
//   6. Build Images     — docker build backend + frontend (parallel)
//   7. Deploy           — docker-compose up -d
//   8. Health Check     — Poll services until healthy (60-second window)
// ============================================================================

pipeline {
    agent any

    // ────────────────────────────────────────────────────────────────────────
    // Environment Variables
    // ────────────────────────────────────────────────────────────────────────
    environment {
        PROJECT_NAME        = "autovision"
        COMPOSE_PROJECT_NAME = "autovision"

        // Dynamic image tag is set in the Checkout stage (GIT_COMMIT not yet
        // available here), so we start with BUILD_NUMBER and patch it later.
        BACKEND_IMAGE       = "${PROJECT_NAME}:backend-latest"
        FRONTEND_IMAGE      = "${PROJECT_NAME}:frontend-latest"
    }

    // ────────────────────────────────────────────────────────────────────────
    // Build Options
    // ────────────────────────────────────────────────────────────────────────
    options {
        buildDiscarder(logRotator(numToKeepStr: '10', artifactNumToKeepStr: '5'))
        timestamps()
        timeout(time: 90, unit: 'MINUTES')
        disableConcurrentBuilds()
    }

    // ────────────────────────────────────────────────────────────────────────
    // Stages
    // ────────────────────────────────────────────────────────────────────────
    stages {

        // ────────────────────────────────────────────────────────────────────
        // Stage 1: Checkout
        // ────────────────────────────────────────────────────────────────────
        stage('Checkout') {
            steps {
                checkout scm
                script {
                    // Build a safe image tag: <build>-<short-sha>
                    // GIT_COMMIT is now populated after checkout scm.
                    def shortSha = (env.GIT_COMMIT ?: 'unknown').take(7)
                    env.IMAGE_TAG      = "${BUILD_NUMBER}-${shortSha}"
                    env.BACKEND_IMAGE  = "${PROJECT_NAME}:backend-${env.IMAGE_TAG}"
                    env.FRONTEND_IMAGE = "${PROJECT_NAME}:frontend-${env.IMAGE_TAG}"
                    echo "Building image tag: ${env.IMAGE_TAG}"
                    sh 'git log -1 --oneline'
                }
            }
        }

        // ────────────────────────────────────────────────────────────────────
        // Stage 2: Setup Environments (parallel)
        // ────────────────────────────────────────────────────────────────────
        stage('Setup') {
            parallel {
                stage('Python venv') {
                    steps {
                        sh '''
                            python3 --version
                            python3 -m venv backend/.venv
                            . backend/.venv/bin/activate
                            pip install --quiet --upgrade pip setuptools wheel
                        '''
                    }
                }
                stage('Node check') {
                    steps {
                        sh '''
                            node --version
                            npm  --version
                        '''
                    }
                }
            }
        }

        // ────────────────────────────────────────────────────────────────────
        // Stage 3: Install Dependencies (parallel)
        // ────────────────────────────────────────────────────────────────────
        stage('Install Dependencies') {
            parallel {
                stage('Backend deps') {
                    steps {
                        sh '''
                            . backend/.venv/bin/activate
                            pip install --quiet -r backend/requirements.txt
                            pip install --quiet \
                                torch torchvision \
                                --index-url https://download.pytorch.org/whl/cpu
                            pip install --quiet pytest pytest-cov pytest-asyncio
                        '''
                    }
                }
                stage('Frontend deps') {
                    steps {
                        sh '''
                            cd frontend
                            npm ci --prefer-offline --no-audit --no-fund
                        '''
                    }
                }
            }
        }

        // ────────────────────────────────────────────────────────────────────
        // Stage 4: Test
        // ────────────────────────────────────────────────────────────────────
        stage('Test') {
            steps {
                sh '''
                    . backend/.venv/bin/activate
                    cd backend
                    pytest tests/ -v --tb=short \
                        --junit-xml=test-results.xml \
                        --cov=app --cov-report=html:coverage \
                        --cov-report=xml:coverage.xml
                '''
            }
            post {
                always {
                    junit 'backend/test-results.xml'
                    publishHTML([
                        allowMissing: true,
                        reportDir:   'backend/coverage',
                        reportFiles: 'index.html',
                        reportName:  'Coverage Report'
                    ])
                }
            }
        }

        // ────────────────────────────────────────────────────────────────────
        // Stage 5: Lint (non-blocking — marks UNSTABLE if it fails)
        // ────────────────────────────────────────────────────────────────────
        stage('Lint') {
            steps {
                catchError(buildResult: 'SUCCESS', stageResult: 'UNSTABLE') {
                    sh '''
                        . backend/.venv/bin/activate
                        pip install --quiet flake8
                        flake8 backend/app/ \
                            --max-line-length=120 \
                            --extend-ignore=E203,W503,E501 \
                            --exclude=__pycache__,migrations
                    '''
                }
            }
        }

        // ────────────────────────────────────────────────────────────────────
        // Stage 6: Build Docker Images (parallel)
        // ────────────────────────────────────────────────────────────────────
        stage('Build Images') {
            parallel {
                stage('Build Backend') {
                    steps {
                        sh '''
                            docker build \
                                --tag "${BACKEND_IMAGE}" \
                                --tag "${PROJECT_NAME}:backend-latest" \
                                --cache-from "${PROJECT_NAME}:backend-latest" \
                                --build-arg BUILDKIT_INLINE_CACHE=1 \
                                backend/
                        '''
                    }
                }
                stage('Build Frontend') {
                    steps {
                        sh '''
                            docker build \
                                --tag "${FRONTEND_IMAGE}" \
                                --tag "${PROJECT_NAME}:frontend-latest" \
                                --cache-from "${PROJECT_NAME}:frontend-latest" \
                                --build-arg BUILDKIT_INLINE_CACHE=1 \
                                frontend/
                        '''
                    }
                }
            }
        }

        // ────────────────────────────────────────────────────────────────────
        // Stage 7: Deploy
        // ────────────────────────────────────────────────────────────────────
        stage('Deploy') {
            steps {
                sh '''
                    # Tear down any previous run (ignore errors if nothing running)
                    docker-compose -p "${COMPOSE_PROJECT_NAME}" down --remove-orphans || true

                    # Start all services in background
                    docker-compose -p "${COMPOSE_PROJECT_NAME}" up -d

                    # Give containers a moment to initialise
                    sleep 5
                    docker-compose -p "${COMPOSE_PROJECT_NAME}" ps
                '''
            }
        }

        // ────────────────────────────────────────────────────────────────────
        // Stage 8: Health Check
        // ────────────────────────────────────────────────────────────────────
        stage('Health Check') {
            steps {
                sh '''
                    # ── Backend ────────────────────────────────────────────────
                    echo "Waiting for backend..."
                    for i in $(seq 1 12); do
                        if curl -sf http://localhost:8000/api/v1/system/info > /dev/null; then
                            echo "Backend healthy after attempt $i"
                            break
                        fi
                        echo "  attempt $i/12 failed — sleeping 5s"
                        sleep 5
                    done
                    curl -sf http://localhost:8000/api/v1/system/info || \
                        { echo "ERROR: backend not healthy"; exit 1; }

                    # ── Frontend ───────────────────────────────────────────────
                    echo "Waiting for frontend..."
                    for i in $(seq 1 6); do
                        if curl -sf http://localhost:5173 > /dev/null; then
                            echo "Frontend healthy after attempt $i"
                            break
                        fi
                        echo "  attempt $i/6 failed — sleeping 5s"
                        sleep 5
                    done
                    curl -sf http://localhost:5173 || \
                        { echo "ERROR: frontend not healthy"; exit 1; }

                    # ── Ollama (non-fatal — may take longer to pull model) ─────
                    echo "Checking Ollama..."
                    curl -sf http://localhost:11434/api/tags > /dev/null \
                        && echo "Ollama healthy" \
                        || echo "WARNING: Ollama not yet ready (model may still be pulling)"
                '''
            }
        }
    }

    // ────────────────────────────────────────────────────────────────────────
    // Post Actions
    // ────────────────────────────────────────────────────────────────────────
    post {
        always {
            // Capture compose logs regardless of result
            sh 'docker-compose -p "${COMPOSE_PROJECT_NAME}" logs --no-color > docker-compose.log 2>&1 || true'
            archiveArtifacts artifacts: 'docker-compose.log, backend/test-results.xml', allowEmptyArchive: true
        }
        success {
            echo "Pipeline PASSED — AutoVision is live at http://localhost:5173"
        }
        failure {
            sh '''
                echo "=== Backend logs ==="
                docker-compose -p "${COMPOSE_PROJECT_NAME}" logs --tail=50 backend  || true
                echo "=== Frontend logs ==="
                docker-compose -p "${COMPOSE_PROJECT_NAME}" logs --tail=50 frontend || true
                echo "=== Ollama logs ==="
                docker-compose -p "${COMPOSE_PROJECT_NAME}" logs --tail=20 ollama   || true
            '''
        }
    }
}
