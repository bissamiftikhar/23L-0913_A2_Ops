#!/usr/bin/env bash
# Task 6.6: GitHub Actions CI Toy Lab
# Student: Bissam Iftikhar (23L-0913) | Project: 23L-0913_A2_Ops

set -e
export PATH="$PWD/.venv/bin:/opt/homebrew/bin:$PATH"

TOY_REPO_DIR="/tmp/23L-0913_toy_ci_repo"
rm -rf "$TOY_REPO_DIR"
mkdir -p "$TOY_REPO_DIR"
cd "$TOY_REPO_DIR"

echo "=============================================================================="
echo "TASK 6.6: GITHUB ACTIONS CI TOY LAB (RED RUN -> GREEN RUN MATRIX)"
echo "Student: Bissam Iftikhar (23L-0913)"
echo "=============================================================================="

# 1. Create remote throwaway repo using gh CLI
TOY_REPO_NAME="toy-ci-23L0913"
gh repo delete "bissamiftikhar/$TOY_REPO_NAME" --yes 2>/dev/null || true
gh repo create "$TOY_REPO_NAME" --public --description "Toy CI Matrix Lab for MLOps Assignment 2 (23L-0913)"

# 2. Initialize local Git
git init -b main
git config user.name "Bissam Iftikhar"
git config user.email "23L-0913@lhr.nu.edu.pk"
git remote add origin "https://github.com/bissamiftikhar/$TOY_REPO_NAME.git"

# 3. Create workflow file with matrix of two Python versions (3.11, 3.12)
mkdir -p .github/workflows
cat << 'EOF' > .github/workflows/toy_ci.yml
name: Toy CI Matrix

on: [push]

jobs:
  test-matrix:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - name: Run Test
        run: python test_sample.py
EOF

# 4. STEP 1: Introduce deliberate FAILING test to produce RED run
cat << 'EOF' > test_sample.py
import sys
print(f"Running toy test on Python {sys.version.split()[0]}...")
# Deliberate failure for Red run demonstration
assert 1 + 1 == 3, "Intentional failure for Task 6.6 Red Run demonstration!"
EOF

git add .
git commit -m "ci: add matrix workflow and deliberate failing test (red run)"
git push -u origin main --force

echo ""
echo ">>> [PUSHED RED RUN] Waiting for GitHub Actions runner to trigger and record failure..."
sleep 15
gh run list --repo "bissamiftikhar/$TOY_REPO_NAME" --limit 2 || true

# 5. STEP 2: FIX the test to produce GREEN run
cat << 'EOF' > test_sample.py
import sys
print(f"Running toy test on Python {sys.version.split()[0]}...")
# Fixed assertion for Green run
assert 1 + 1 == 2, "Test passed successfully!"
print("All matrix tests PASSED!")
EOF

git add test_sample.py
git commit -m "fix(ci): correct assertion to produce passing green run"
git push origin main

echo ""
echo ">>> [PUSHED GREEN FIX] Waiting for GitHub Actions runner to execute green matrix run..."
sleep 20
gh run list --repo "bissamiftikhar/$TOY_REPO_NAME" --limit 3 || true

echo "=============================================================================="
echo "TASK 6.6 TOY LAB COMPLETE: RED AND GREEN MATRIX RUNS RECORDED"
echo "Repository: https://github.com/bissamiftikhar/$TOY_REPO_NAME"
echo "=============================================================================="

