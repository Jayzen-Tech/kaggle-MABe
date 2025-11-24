#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

echo "[1/2] 运行验证脚本：baseline/social-action-recognition-in-mice-xgboost-validate.py"
python baseline/social-action-recognition-in-mice-xgboost-validate.py

echo "[2/2] 运行推理脚本：baseline/social-action-recognition-in-mice-xgboost-submit.py"
python baseline/social-action-recognition-in-mice-xgboost-submit.py

echo "Kaggle baseline 流程完成。"
