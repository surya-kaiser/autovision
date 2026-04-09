// ============================================================================
// AutoVision MLOps — Jenkins Declarative Pipeline (Docker-based)
// ============================================================================
// All build/test steps run inside Docker containers.
// Jenkins only needs: git, docker CLI, curl (standard in jenkins:lts)
// Docker socket must be mounted: -v /var/run/docker.sock:/var/run/docker.sock
// ============================================================================

pipeline {
    agent any

    environment {
        PROJECT_NAME         = "autovision"
        COMPOSE_PROJECT_NAME = "autovision"
        BACKEND_IMAGE        = "${PROJECT_NAME}:backend-latest"
        FRONTEND_IMAGE       = "${PROJECT_NAME}:frontend-latest"
    }

    options {
        buildDiscarder(logRotator(numToKeepStr: '10', artifactNumToKeepStr: '5'))
        timestamps()
        timeout(time: 90, unit: 'MINUTES')
        disableConcurrentBuilds()
    }

    stages {

        // ── Stage 1: Checkout ────────────────────────────────────────────────
        stage('Checkout') {
            steps {
                checkout scm
                script {
                    def shortSha = (env.GIT_COMMIT ?: 'unknown').take(7)
                    env.IMAGE_TAG      = "${BUILD_NUMBER}-${shortSha}"
                    env.BACKEND_IMAGE  = "${PROJECT_NAME}:backend-${env.IMAGE_TAG}"
                    env.FRONTEND_IMAGE = "${PROJECT_NAME}:frontend-${env.IMAGE_TAG}"
                    echo "Image tag: ${env.IMAGE_TAG}"
                    sh 'git log -1 --oneline'
                }
            }
        }

        // ── Stage 2: Verify Docker available ────────────────────────────────
        stage('Verify Tools') {
            steps {
                sh '''
                    echo "=== Docker version ==="
                    docker --version || { echo "ERROR: Docker not found. Restart Jenkins with: -v //var/run/docker.sock:/var/run/docker.sock"; exit 1; }
                    echo "=== Docker Compose ==="
                    docker compose version 2>/dev/null || docker-compose --version 2>/dev/null || echo "WARNING: docker compose plugin not found"
                    echo "=== Workspace ==="
                    pwd && ls -la
                '''
            }
        }

        // ── Stage 3: Build Docker Images (parallel) ──────────────────────────
        stage('Build Images') {
            parallel {
                stage('Build Backend') {
                    steps {
                        sh '''
                            docker build \
                                --tag "${BACKEND_IMAGE}" \
                                --tag "${PROJECT_NAME}:backend-latest" \
                                --cache-from "${PROJECT_NAME}:backend-latest" \
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
                                frontend/
                        '''
                    }
                }
            }
        }

        // ── Stage 4: Test (runs inside the backend image) ────────────────────
        stage('Test') {
            steps {
                sh '''
                    # Run WITHOUT --rm so we can docker cp results after
                    docker run \
                        --name autovision-test-${BUILD_NUMBER} \
                        -e PYTHONPATH=/app \
                        "${PROJECT_NAME}:backend-latest" \
                        sh -c "cd /app && \
                               pip install --quiet pytest pytest-cov pytest-asyncio httpx && \
                               pytest tests/ -v --tb=short \
                                   --junit-xml=test-results.xml \
                                   --cov=app --cov-report=xml:coverage.xml"
                    TEST_EXIT=$?

                    # Copy results out before removing container
                    docker cp autovision-test-${BUILD_NUMBER}:/app/test-results.xml backend/test-results.xml 2>/dev/null || true
                    docker rm autovision-test-${BUILD_NUMBER} || true

                    exit $TEST_EXIT
                '''
            }
            post {
                always {
                    junit allowEmptyResults: true, testResults: 'backend/test-results.xml'
                }
            }
        }

        // ── Stage 5: Lint (non-blocking) ─────────────────────────────────────
        stage('Lint') {
            steps {
                catchError(buildResult: 'SUCCESS', stageResult: 'UNSTABLE') {
                    sh '''
                        docker run --rm \
                            -v "$(pwd)/backend/app:/app/app:ro" \
                            -w /app \
                            "${PROJECT_NAME}:backend-latest" \
                            sh -c "pip install --quiet flake8 && \
                                   flake8 app/ \
                                       --max-line-length=120 \
                                       --extend-ignore=E203,W503,E501 \
                                       --exclude=__pycache__"
                    '''
                }
            }
        }

        // ── Stage 6: Deploy ───────────────────────────────────────────────────
        stage('Deploy') {
            steps {
                sh '''
                    DC="docker compose"
                    $DC -p "${COMPOSE_PROJECT_NAME}" down --remove-orphans || true
                    $DC -p "${COMPOSE_PROJECT_NAME}" up -d
                    sleep 5
                    $DC -p "${COMPOSE_PROJECT_NAME}" ps
                '''
            }
        }

        // ── Stage 7: Health Check ─────────────────────────────────────────────
        stage('Health Check') {
            steps {
                sh '''
                    echo "Waiting for backend..."
                    for i in $(seq 1 12); do
                        if curl -sf http://localhost:8000/api/v1/system/info > /dev/null 2>&1; then
                            echo "Backend healthy after attempt $i"
                            break
                        fi
                        echo "  attempt $i/12 — sleeping 5s"
                        sleep 5
                    done
                    curl -sf http://localhost:8000/api/v1/system/info \
                        || { echo "ERROR: backend not healthy"; exit 1; }

                    echo "Waiting for frontend..."
                    for i in $(seq 1 6); do
                        if curl -sf http://localhost:5173 > /dev/null 2>&1; then
                            echo "Frontend healthy after attempt $i"
                            break
                        fi
                        echo "  attempt $i/6 — sleeping 5s"
                        sleep 5
                    done
                    curl -sf http://localhost:5173 \
                        || { echo "ERROR: frontend not healthy"; exit 1; }

                    echo "Checking Ollama..."
                    curl -sf http://localhost:11434/api/tags > /dev/null \
                        && echo "Ollama healthy" \
                        || echo "WARNING: Ollama not yet ready"
                '''
            }
        }
    }

    post {
        always {
            sh 'docker compose -p autovision logs --no-color > docker-compose.log 2>&1 || true'
            archiveArtifacts artifacts: 'docker-compose.log', allowEmptyArchive: true
        }
        success {
            echo "Pipeline PASSED — AutoVision is live at http://localhost:5173"
        }
        failure {
            sh '''
                echo "=== Backend logs ==="
                docker compose -p autovision logs --tail=50 backend  || true
                echo "=== Frontend logs ==="
                docker compose -p autovision logs --tail=50 frontend || true
            '''
        }
    }
}
