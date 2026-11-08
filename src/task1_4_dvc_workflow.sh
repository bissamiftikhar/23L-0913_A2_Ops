#!/usr/bin/env bash
# Task 1.4: DVC (Data Version Control) Toy Part & Project Application
# Project: 23L-0913_A2_Ops (VisionGuard)
# Author: Bissam Iftikhar (23L-0913)

set -e
export PATH="$PWD/.venv/bin:/opt/homebrew/bin:$PATH"

PROJECT_ROOT="$PWD"
ARTIFACTS_DIR="$PROJECT_ROOT/artifacts/task1"
mkdir -p "$ARTIFACTS_DIR"

LOG_FILE="$ARTIFACTS_DIR/task1_4_dvc_output.txt"
exec > >(tee "$LOG_FILE") 2>&1

echo "=============================================================================="
echo "TASK 1.4: DATA VERSION CONTROL (DVC) — TOY PART & PROJECT APPLICATION"
echo "Student: Bissam Iftikhar (23L-0913) | Project: 23L-0913_A2_Ops"
echo "Date   : $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "User   : $(whoami)@$(hostname)"
echo "=============================================================================="

# ------------------------------------------------------------------------------
# PART A: TOY PART
# "in a new empty folder with a small CSV, initialise Git and DVC, track the CSV
# with DVC, change it, commit both versions(dagshub), then return to the older
# version. Configure a remote storage, push, delete the local data, and pull it back."
# ------------------------------------------------------------------------------
TOY_DIR="/tmp/23L-0913_dvc_toy_lab"
TOY_REMOTE="/tmp/23L-0913_dvc_toy_remote"
rm -rf "$TOY_DIR" "$TOY_REMOTE"
mkdir -p "$TOY_DIR" "$TOY_REMOTE"

echo ""
echo ">>> [1] Initialising Toy Git + DVC repository in $TOY_DIR"
cd "$TOY_DIR"
git init -b main
git config user.name "Bissam Iftikhar"
git config user.email "23L-0913@lhr.nu.edu.pk"
dvc init

git commit -m "chore: initialize Git and DVC in toy lab (23L-0913)"

echo ""
echo ">>> [2] Creating v1 of samples.csv (3 rows) and tracking with DVC"
cat << 'EOF' > samples.csv
image_id,class,width,height
img_001.jpg,real,1024,768
img_002.jpg,ai_generated,512,512
img_003.jpg,real,800,600
EOF

dvc add samples.csv
echo "--- Contents of samples.csv.dvc (v1) ---"
cat samples.csv.dvc
git add samples.csv.dvc .gitignore
git commit -m "data(v1): track initial samples.csv (3 rows) with DVC"
git tag v1.0

echo ""
echo ">>> [3] Modifying samples.csv (v2: adding 2 new rows) and committing second version"
cat << 'EOF' >> samples.csv
img_004.jpg,ai_generated,1024,1024
img_005.jpg,real,1920,1080
EOF

dvc add samples.csv
echo "--- Contents of samples.csv.dvc (v2) ---"
cat samples.csv.dvc
git add samples.csv.dvc
git commit -m "data(v2): update samples.csv to 5 rows and track new DVC hash"
git tag v2.0

echo ""
echo ">>> [4] Git commit history of Toy DVC repository:"
git log --oneline --decorate

echo ""
echo ">>> [5] Configuring remote storage, pushing all versions, deleting local cache/data, and pulling back:"
dvc remote add -d toy_remote "$TOY_REMOTE"
# Also demonstrate DagsHub DVC remote config syntax
dvc remote add dagshub_remote https://dagshub.com/bissamiftikhar/23L-0913_A2_Ops.dvc
dvc remote modify --local dagshub_remote auth basic
dvc remote modify --local dagshub_remote user bissamiftikhar
dvc remote modify --local dagshub_remote password "$DAGSHUB_USER_TOKEN"

git add .dvc/config
git commit -m "chore: configure DVC remote storage"

dvc push --all-commits
echo "--- Remote storage tree after dvc push ---"
find "$TOY_REMOTE" -type f

echo ""
echo ">>> [6] Returning to older version (v1.0) using git checkout + dvc checkout:"
git checkout v1.0 samples.csv.dvc
dvc checkout
echo "--- Contents of samples.csv after checking out v1.0 (back to 3 rows!) ---"
cat samples.csv

echo ""
echo ">>> [7] Returning to v2.0, deleting local data & DVC cache, and pulling back from remote:"
git checkout main samples.csv.dvc
rm -f samples.csv
rm -rf .dvc/cache
echo "Local samples.csv exists? $(test -f samples.csv && echo YES || echo NO)"
dvc pull
echo "After 'dvc pull', local samples.csv restored:"
cat samples.csv

# ------------------------------------------------------------------------------
# PART B: APPLY TO VISIONGUARD DATASET FOLDER
# "Apply: track your dataset folder the same way, with DagsHub's storage or a
# local folder as remote."
# ------------------------------------------------------------------------------
echo ""
echo "=============================================================================="
echo ">>> [8] APPLYING DVC TO VISIONGUARD DATASET FOLDER (data/raw)"
echo "=============================================================================="
cd "$PROJECT_ROOT"

if [ ! -d ".git" ]; then
    git init -b main
    git config user.name "Bissam Iftikhar"
    git config user.email "23L-0913@lhr.nu.edu.pk"
fi

if [ ! -d ".dvc" ]; then
    dvc init
fi

LOCAL_DVC_STORAGE="/tmp/23L-0913_visionguard_dvc_remote"
mkdir -p "$LOCAL_DVC_STORAGE"

dvc remote add -f -d local_storage "$LOCAL_DVC_STORAGE"
dvc remote add -f dagshub https://dagshub.com/bissamiftikhar/23L-0913_A2_Ops.dvc
dvc remote modify --local dagshub auth basic
dvc remote modify --local dagshub user bissamiftikhar
dvc remote modify --local dagshub password "$DAGSHUB_USER_TOKEN"

echo ">>> Running 'dvc add data/raw'..."
dvc add data/raw

echo ""
echo "--- Contents of pointer file data/raw.dvc ---"
cat data/raw.dvc

echo ""
echo ">>> Pushing dataset to configured DVC remote..."
dvc push -r local_storage

echo "=============================================================================="
echo "TASK 1.4 COMPLETED SUCCESSFULLY"
echo "=============================================================================="

