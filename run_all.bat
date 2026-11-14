@echo off
chcp 65001 >nul
cd /d %~dp0
echo [1/4] 安装依赖
.venv\Scripts\python.exe -m pip install -q -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
echo [2/4] 运行实验流水线
.venv\Scripts\python.exe -m scripts.run_pipeline %*
echo [3/4] 生成报告 PDF
.venv\Scripts\python.exe -m scripts.make_report_pdf
echo [4/4] 启动后端与前端
start "" .venv\Scripts\python.exe -m uvicorn src.api.main:app --port 8000
start "" .venv\Scripts\python.exe -m streamlit run src/app/streamlit_app.py
