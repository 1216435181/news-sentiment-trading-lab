"""自动截图: 启动 FastAPI + Streamlit, 用 Playwright 驱动本机 Edge 截取页面。

不下载浏览器内核(使用系统已安装的 msedge), 需要 `pip install playwright`。
    python -m scripts.screenshots
产出: docs/screenshots/*.png
"""
from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "docs" / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
if not Path(PY).exists():
    PY = sys.executable


def _trim_bottom(path: Path, pad: int = 24) -> None:
    """裁掉截图底部的纯空白区域, 避免超长白边。"""
    from PIL import Image
    im = Image.open(path).convert("RGB")
    w, h = im.size
    px = im.load()
    last = h - 1
    while last > 0:
        row = [px[x, last] for x in range(0, w, 17)]
        if any(sum(c) < 740 for c in row):   # 出现非白像素即认为有内容
            break
        last -= 1
    if last < h - 1:
        im.crop((0, 0, w, min(h, last + pad))).save(path)


def wait_port(port: int, timeout: float = 90) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        with socket.socket() as s:
            s.settimeout(1)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.5)
    return False


def main() -> None:
    from playwright.sync_api import sync_playwright

    procs = []
    try:
        api = subprocess.Popen(
            [PY, "-m", "uvicorn", "src.api.main:app", "--host", "127.0.0.1", "--port", "8000"],
            cwd=ROOT, stdout=open(ROOT / "logs" / "api.log", "w", encoding="utf-8"),
            stderr=subprocess.STDOUT)
        procs.append(api)
        st = subprocess.Popen(
            [PY, "-m", "streamlit", "run", "src/app/streamlit_app.py",
             "--server.headless", "true", "--server.port", "8501"],
            cwd=ROOT, stdout=open(ROOT / "logs" / "streamlit.log", "w", encoding="utf-8"),
            stderr=subprocess.STDOUT)
        procs.append(st)
        print("FastAPI ready:", wait_port(8000), "| Streamlit ready:", wait_port(8501))
        time.sleep(4)

        with sync_playwright() as p:
            browser = p.chromium.launch(channel="msedge", headless=True)
            ctx = browser.new_context(viewport={"width": 1560, "height": 2900},
                                      device_scale_factor=1.2)

            def snap(url: str, name: str, expect: str | None = None,
                     settle: int = 3500, chars: int = 400) -> None:
                page = ctx.new_page()
                page.goto(url, wait_until="load", timeout=60000)
                if expect:
                    try:
                        page.wait_for_selector(f"text={expect}", timeout=45000)
                    except Exception:  # noqa: BLE001
                        print(f"  [warn] 未等到 {expect}")
                page.wait_for_timeout(settle)
                out = OUT / f"{name}.png"
                page.screenshot(path=str(out), full_page=False)
                _trim_bottom(out)
                txt = page.inner_text("body")
                print(f"  OK  {name}.png  ({len(txt)} chars)  {url}")
                if len(txt) < chars:
                    print("      页面内容过少, 可能未渲染完成:", txt[:200].replace("\n", " "))
                page.close()

            snap("http://127.0.0.1:8501/?page=1", "01_frontend_overview", "数据概览")
            snap("http://127.0.0.1:8501/?page=2", "02_frontend_sentiment", "模型对比表")
            snap("http://127.0.0.1:8501/?page=3", "04_frontend_backtest", "净值曲线")
            snap("http://127.0.0.1:8501/?page=4", "06_frontend_db_logs", "数据库表记录数")
            snap("http://127.0.0.1:8000/docs", "03_api_docs", "新闻情感交易信号")
            snap("http://127.0.0.1:8000/monitor", "05_backend_monitor", "pipeline_log")
            browser.close()
    finally:
        for p in procs:
            p.terminate()
        time.sleep(2)
        for p in procs:
            if p.poll() is None:
                p.kill()
        print("服务已关闭")


if __name__ == "__main__":
    main()
