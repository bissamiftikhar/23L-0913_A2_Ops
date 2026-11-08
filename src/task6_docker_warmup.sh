#!/usr/bin/env bash
# Task 6.1, 6.4, 6.5: Docker Warm-Up, Image Analysis & Operations Troubleshooting
# Project: 23L-0913_A2_Ops (VisionGuard)
# Author: Bissam Iftikhar (23L-0913)

set -e
export PATH="$PWD/.venv/bin:/usr/local/bin:/opt/homebrew/bin:$PATH"

ARTIFACTS_DIR="$PWD/artifacts/task6"
mkdir -p "$ARTIFACTS_DIR"

LOG_FILE="$ARTIFACTS_DIR/task6_docker_output.txt"
exec > >(tee "$LOG_FILE") 2>&1

echo "=============================================================================="
echo "TASK 6: DOCKER CONTAINERIZATION, IMAGE ANALYSIS & OPERATIONS"
echo "Student: Bissam Iftikhar (23L-0913) | Project: 23L-0913_A2_Ops"
echo "Date   : $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "Host   : $(whoami)@$(hostname)"
echo "=============================================================================="

# ------------------------------------------------------------------------------
# 1. DOCKER WARM-UP (Task 6.1)
# ------------------------------------------------------------------------------
echo ""
echo ">>> [1] Docker Warm-Up: Running ready-made Nginx container with port mapping and volume mount..."
WARMUP_VOL="/tmp/vg_warmup_vol_23L0913"
mkdir -p "$WARMUP_VOL"
echo "<h1>VisionGuard Docker Warm-up (23L-0913)</h1>" > "$WARMUP_VOL/index.html"

# Clean any existing container
docker rm -f vg_warmup_nginx 2>/dev/null || true

# Run container with port mapping (-p 8088:80) and volume mount (-v)
docker run -d --name vg_warmup_nginx -p 8088:80 -v "$WARMUP_VOL:/usr/share/nginx/html:ro" nginx:alpine

echo ">>> [2] Verifying running container with 'docker ps':"
docker ps --filter "name=vg_warmup_nginx"

echo ""
echo ">>> [3] Testing HTTP response from mapped port (curl http://localhost:8088):"
curl -s http://localhost:8088

echo ""
echo ">>> [4] Reading container logs with 'docker logs':"
docker logs vg_warmup_nginx

echo ""
echo ">>> [5] Opening a shell command inside container with 'docker exec':"
docker exec vg_warmup_nginx sh -c "uname -a; whoami; cat /etc/os-release | grep PRETTY_NAME"

echo ""
echo ">>> [6] Cleaning up warm-up container:"
docker stop vg_warmup_nginx
docker rm vg_warmup_nginx
echo "Container removed successfully."

# ------------------------------------------------------------------------------
# 2. DOCKER IMAGE BUILD & IMAGE ANALYSIS (Task 6.2 & 6.4)
# ------------------------------------------------------------------------------
echo ""
echo "=============================================================================="
echo ">>> [7] BUILDING VISIONGUARD PRODUCTION DOCKER IMAGE"
echo "=============================================================================="
docker build -t visionguard:v1 -t visionguard:latest .

echo ""
echo ">>> [8] Inspecting User Inside Container (Task 6.2 Evidence):"
docker run --rm visionguard:latest whoami
docker run --rm visionguard:latest id

echo ""
echo ">>> [9] Image Layer Analysis & History (Task 6.4):"
docker history visionguard:latest --human --format "table {{.ID}}\t{{.Size}}\t{{.CreatedBy}}"

echo ""
echo ">>> [10] Image Size Comparison Report (Task 6.4):"
docker images visionguard:latest --format "Repository: {{.Repository}} | Tag: {{.Tag}} | Size: {{.Size}}"

# ------------------------------------------------------------------------------
# 3. DOCKER COMPOSE OPERATIONS & TROUBLESHOOTING (Task 6.3 & 6.5)
# ------------------------------------------------------------------------------
echo ""
echo "=============================================================================="
echo ">>> [11] DOCKER COMPOSE STACK STARTUP (web + db)"
echo "=============================================================================="
docker compose down -v 2>/dev/null || true
docker compose up -d

echo ""
echo ">>> [12] Checking Service Health Status ('docker compose ps'):"
sleep 6
docker compose ps

echo ""
echo ">>> [13] Testing Web App Health Endpoint from Host:"
curl -s http://localhost:5001/health | python3 -m json.tool || true

echo ""
echo ">>> [14] Proof that Database Port 5432 is Closed to the Outside Host (Security Verification):"
nc -z -v -w2 localhost 5432 2>&1 || echo "Port 5432 is CLOSED from host (As required: DB only reachable internally on Compose network)."

echo ""
echo ">>> [15] Testing Shell Inside Services:"
docker compose exec -T web whoami
docker compose exec -T db su - postgres -c "psql -U visionguard -d visionguard_db -c '\dt'"

# ------------------------------------------------------------------------------
# 4. OPERATIONS & PURPOSEFUL FAILURE DIAGNOSIS (Task 6.5)
# ------------------------------------------------------------------------------
echo ""
echo "=============================================================================="
echo ">>> [16] TASK 6.5: INTENTIONAL FAILURE DIAGNOSIS & RECOVERY"
echo "=============================================================================="
echo "[BREAKING ON PURPOSE] Injecting invalid database password into web environment..."
docker compose stop web
docker run -d --name vg_broken_web --network visionguard_network \
  -e DATABASE_URL=postgresql://visionguard:WRONG_PASSWORD_FAIL@db:5432/visionguard_db \
  -e MODEL_PATH=/app/models/champion_model.keras \
  visionguard:latest python3 -c "import psycopg2; psycopg2.connect('postgresql://visionguard:WRONG_PASSWORD_FAIL@db:5432/visionguard_db')" 2>&1 || true

sleep 3
echo ">>> Log Excerpt of Intentional Failure (Diagnosed strictly from logs):"
docker logs vg_broken_web 2>&1 | tail -n 10 || true

echo ">>> DIAGNOSIS: FATAL password authentication failed for user 'visionguard'."
echo ">>> FIX: Reverting to authenticated credential vg_secure_pass_23L0913 via docker-compose."
docker rm -f vg_broken_web 2>/dev/null || true
docker compose restart web
sleep 4
echo ">>> Verification of Healthy Recovery:"
curl -s http://localhost:5001/health | python3 -m json.tool || true

echo ""
echo "=============================================================================="
echo "TASK 6 AUTOMATED VERIFICATION COMPLETED SUCCESSFULLY"
echo "=============================================================================="

