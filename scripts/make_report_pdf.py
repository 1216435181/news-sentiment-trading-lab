"""把 reports/REPORT.md 渲染成排版精良的 HTML, 再用本机 Edge 打印成 PDF。

    python -m scripts.make_report_pdf
产出: reports/REPORT.html, reports/REPORT.pdf
"""
from __future__ import annotations

import base64
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import markdown  # noqa: E402

from src.config import P  # noqa: E402

EDGE = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]

CSS = """
@page { size: A4; margin: 16mm 14mm; }
body { font-family: "Microsoft YaHei", "PingFang SC", sans-serif; font-size: 10.5pt;
       line-height: 1.75; color: #1f2933; }
h1 { font-size: 20pt; text-align: center; margin: 0 0 6mm; letter-spacing: 1px; }
h2 { font-size: 13pt; margin: 8mm 0 3mm; padding-left: 3mm; border-left: 4px solid #d64545; }
h3 { font-size: 11.5pt; margin: 6mm 0 2mm; color: #374151; }
p, li { text-align: justify; }
table { border-collapse: collapse; width: 100%; margin: 3mm 0 4mm; font-size: 9pt;
        page-break-inside: avoid; }
th { background: #f2f4f8; font-weight: 600; }
th, td { border: 1px solid #dfe3ea; padding: 4px 6px; }
tr:nth-child(even) td { background: #fafbfd; }
img { max-width: 100%; margin: 3mm 0; border: 1px solid #eceff3; border-radius: 3px; }
blockquote { background: #f7f8fa; border-left: 4px solid #cbd5e1; margin: 3mm 0;
             padding: 2mm 4mm; color: #4b5563; font-size: 9.5pt; }
code { background: #f1f3f6; padding: 1px 4px; border-radius: 3px; font-size: 9pt; }
pre { background: #f6f8fa; border: 1px solid #e3e7ee; border-radius: 4px; padding: 3mm;
      font-size: 9pt; overflow-x: auto; }
pre code { background: none; }
hr { border: none; border-top: 1px solid #e5e8ee; margin: 6mm 0; }
"""


def _inline_images(html: str) -> str:
    """把本地图片路径替换成 base64, 保证 PDF 里图片一定显示。"""
    def repl(m):
        src = m.group(1)
        p = (ROOT / "reports" / src).resolve() if not src.startswith("/") else Path(src)
        if not p.exists():
            p = (ROOT / src).resolve()
        if not p.exists():
            return m.group(0)
        b64 = base64.b64encode(p.read_bytes()).decode()
        return f'src="data:image/png;base64,{b64}"'

    return re.sub(r'src="([^"]+)"', repl, html)


def main() -> None:
    md = P.report_md.read_text(encoding="utf-8")
    html_body = markdown.markdown(md, extensions=["tables", "fenced_code", "toc"])
    html_body = _inline_images(html_body)
    html = (f"<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>"
            f"<title>新闻情感交易信号实验报告</title><style>{CSS}</style></head>"
            f"<body>{html_body}</body></html>")
    html_path = ROOT / "reports" / "REPORT.html"
    html_path.write_text(html, encoding="utf-8")
    print("HTML:", html_path)

    exe = next((e for e in EDGE if Path(e).exists()), None)
    if not exe:
        print("未找到 Edge, 跳过 PDF 生成(可手动打印 HTML)")
        return
    pdf_path = ROOT / "reports" / "REPORT.pdf"
    subprocess.run([exe, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf_path}", html_path.as_uri()],
                   check=False, capture_output=True, timeout=180)
    print("PDF:", pdf_path, pdf_path.exists(), pdf_path.stat().st_size if pdf_path.exists() else 0)


if __name__ == "__main__":
    main()
