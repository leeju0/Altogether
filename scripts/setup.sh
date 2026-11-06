#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
printf '설치 완료. .venv/bin/python scripts/smoke_test.py 로 확인하세요.\n'
