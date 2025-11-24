#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

echo "[1/4] 运行验证流程 baseline/ensemble-validate.py ..."
python baseline/ensemble-validate.py

THRESH_PATH="models/ensemble_xgb_lgb_cat/thresholds.pkl"
if [[ -f "${THRESH_PATH}" ]]; then
  echo "[2/4] 阈值文件生成成功：${THRESH_PATH}"
else
  echo "[2/4] 未找到阈值文件：${THRESH_PATH}" >&2
  exit 1
fi

echo "[3/4] 验证阈值文件可读取..."
python - <<'PY'
import sys
from pathlib import Path
import joblib

path = Path("models/ensemble_xgb_lgb_cat/thresholds.pkl")
try:
    data = joblib.load(path)
except Exception as exc:
    print(f"[失败] 无法读取 {path}: {exc}")
    sys.exit(1)

single_sections = sorted(data.get("single", {}).keys())
pair_sections = sorted(data.get("pair", {}).keys())
print(f"[成功] 已加载阈值：single sections={single_sections}, pair sections={pair_sections}")
PY

echo "[4/4] 运行提交脚本 baseline/ensemble-submit-new.py ..."
python baseline/ensemble-submit-new.py

echo "流程完成。"
