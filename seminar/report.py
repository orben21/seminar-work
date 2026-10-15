"""Write the results: CSV tables named by plan section, and one self-contained
HTML page that walks through them in the plan's order.

Table files are prefixed t<section>_ (t3_5_exclusion_log.csv is the §3.5
exclusion table) or app<letter>_ for the appendices, so every number in the
seminar can be traced to the file it came from.
"""

from __future__ import annotations

import base64
import datetime as dt
import html
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from . import paths

CSS = """
body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
       margin: 0; background: #f4f6f8; color: #22282e; line-height: 1.5; }
.wrap { max-width: 1120px; margin: 0 auto; padding: 32px 24px 80px; }
h1 { font-size: 27px; margin: 0 0 4px; }
h2 { font-size: 20px; margin: 40px 0 6px; padding-top: 18px;
     border-top: 2px solid #d7dde3; }
h3 { font-size: 14px; margin: 22px 0 8px; color: #46505a;
     text-transform: uppercase; letter-spacing: .04em; }
.sub { color: #667180; margin: 0 0 22px; }
.card { background: #fff; border: 1px solid #e2e7ec; border-radius: 8px;
        padding: 16px 20px; margin: 14px 0; overflow-x: auto; }
.reading { background: #1f4e79; color: #fff; border-radius: 8px;
           padding: 14px 20px; margin: 14px 0; }
table { border-collapse: collapse; width: 100%; font-size: 12.5px;
        margin: 6px 0 4px; }
th, td { text-align: right; padding: 5px 9px; border-bottom: 1px solid #e8ecf0;
         vertical-align: top; }
th:first-child, td:first-child { text-align: left; }
th { background: #eef2f6; font-weight: 600; }
td.text, th.text { text-align: left; }
img { max-width: 100%; height: auto; display: block; margin: 6px 0; }
.note { background: #fff8e6; border-left: 4px solid #e0a800;
        padding: 10px 14px; margin: 10px 0; font-size: 13px; }
.caption { color: #667180; font-size: 12.5px; margin: 2px 0 0; }
.toc a { display: inline-block; margin-right: 14px; color: #1f4e79; }
"""


def write_csv(table: pd.DataFrame, name: str,
              directory: Path = paths.TABLES_DIR) -> Path:
    path = Path(directory) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, encoding="utf-8-sig")
    return path


# A block is one of:
#   ("text", str)            a paragraph (HTML allowed)
#   ("note", str)            a highlighted note
#   ("reading", str)         the reading instruction, highlighted
#   ("table", DataFrame, caption)
#   ("figure", Path, caption)
Block = Tuple


@dataclass
class Section:
    anchor: str
    title: str
    blocks: List[Block] = field(default_factory=list)

    def text(self, value: str):
        self.blocks.append(("text", value))

    def note(self, value: str):
        self.blocks.append(("note", value))

    def reading(self, value: str):
        self.blocks.append(("reading", value))

    def table(self, frame: Optional[pd.DataFrame], caption: str = ""):
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            self.blocks.append(("table", frame, caption))

    def figure(self, path: Optional[Union[str, Path]], caption: str = ""):
        if path and Path(path).exists():
            self.blocks.append(("figure", Path(path), caption))


def _embed(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return (f'<img src="data:image/png;base64,{encoded}" '
            f'alt="{html.escape(path.name)}">')


def _format(value) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (float, np.floating)):
        if value != 0 and (abs(value) >= 1e5 or abs(value) < 1e-3):
            return f"{value:.3g}"
        return f"{value:,.3f}" if abs(value) < 1000 else f"{value:,.0f}"
    return html.escape(str(value))


def _table_html(frame: pd.DataFrame, max_rows: int = 400) -> str:
    frame = frame.head(max_rows)
    text_columns = {c for c in frame.columns
                    if not pd.api.types.is_numeric_dtype(frame[c])}
    head = "".join(f"<th class='{'text' if c in text_columns else ''}'>"
                   f"{html.escape(str(c))}</th>" for c in frame.columns)
    body = []
    for _, row in frame.iterrows():
        cells = "".join(
            f"<td class='{'text' if c in text_columns else ''}'>"
            f"{_format(row[c])}</td>" for c in frame.columns)
        body.append(f"<tr>{cells}</tr>")
    return f"<table><tr>{head}</tr>{''.join(body)}</table>"


def build_report(title: str, subtitle: str, run_info: List[Tuple[str, str]],
                 sections: List[Section],
                 path: Path = paths.REPORT_HTML) -> Path:
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    parts = ["<!doctype html><html><head><meta charset='utf-8'>",
             f"<title>{html.escape(title)}</title>",
             f"<style>{CSS}</style></head><body><div class='wrap'>",
             f"<h1>{html.escape(title)}</h1>",
             f"<p class='sub'>{html.escape(subtitle)} &middot; generated "
             f"{now}</p>"]
    parts.append("<div class='card'><h3>This run</h3><table>")
    for key, value in run_info:
        parts.append(f"<tr><td class='text'>{html.escape(key)}</td>"
                     f"<td class='text'>{html.escape(str(value))}</td></tr>")
    parts.append("</table></div>")
    links = " ".join(f"<a href='#{s.anchor}'>{html.escape(s.title)}</a>"
                     for s in sections)
    parts.append(f"<p class='toc'>{links}</p>")

    for section in sections:
        parts.append(f"<h2 id='{section.anchor}'>{html.escape(section.title)}"
                     f"</h2>")
        for block in section.blocks:
            kind = block[0]
            if kind == "text":
                parts.append(f"<p>{block[1]}</p>")
            elif kind == "note":
                parts.append(f"<div class='note'>{block[1]}</div>")
            elif kind == "reading":
                parts.append(f"<div class='reading'>{block[1]}</div>")
            elif kind == "table":
                caption = (f"<p class='caption'>{html.escape(block[2])}</p>"
                           if block[2] else "")
                parts.append(f"<div class='card'>{_table_html(block[1])}"
                             f"{caption}</div>")
            elif kind == "figure":
                caption = (f"<p class='caption'>{html.escape(block[2])}</p>"
                           if block[2] else "")
                parts.append(f"<div class='card'>{_embed(block[1])}{caption}"
                             f"</div>")
    parts.append("</div></body></html>")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")
    return path
