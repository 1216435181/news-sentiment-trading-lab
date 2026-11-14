#!/usr/bin/env bash
# 一键复现: 环境 -> 流水线 -> 报告 PDF -> 截图 -> 启动服务
set -e
cd "$(dirname "$0")"

PY=./.venv/Scripts/python.exe
[ -f "$PY" ] || PY=./.venv/bin/python

echo "[1/5] 安装依赖"
$PY -m pip install -q -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "[2/5] 运行实验流水线"
$PY -m scripts.run_pipeline "$@"

echo "[3/5] 生成报告 PDF"
$PY -m scripts.make_report_pdf

echo "[4/5] 截图(需 playwright)"
$PY -m scripts.screenshots || echo "截图失败可忽略(需 pip install playwright)"

echo "[5/5] 启动服务"
echo "  uvicorn src.api.main:app --port 8000"
echo "  streamlit run src/app/streamlit_app.py"
