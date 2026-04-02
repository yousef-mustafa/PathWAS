"""
PathWAS Rich HTML Report Generator
------------------------------------
Creates self-contained, single-file HTML experiment reports with embedded
figures, data previews, pipeline timing, and full configuration details.

Public API
----------
StageTimer            — records wall-clock time for named pipeline stages
DataSnapshot          — lightweight DataFrame preview for embedding
snapshot_dataframe()  — create a DataSnapshot from a DataFrame
generate_experiment_report() — write the HTML (and stub MD) report
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# StageTimer
# ---------------------------------------------------------------------------

class StageTimer:
    """Records wall-clock start/stop times for named pipeline stages."""

    def __init__(self) -> None:
        self._starts: Dict[str, datetime] = {}
        self._stops: Dict[str, datetime] = {}
        self._order: List[str] = []
        self.experiment_start: Optional[datetime] = None
        self.experiment_end: Optional[datetime] = None

    def start(self, stage: str) -> None:
        now = datetime.now()
        self._starts[stage] = now
        if stage not in self._order:
            self._order.append(stage)
        if self.experiment_start is None:
            self.experiment_start = now

    def stop(self, stage: str) -> None:
        now = datetime.now()
        self._stops[stage] = now
        self.experiment_end = now

    def elapsed(self, stage: str) -> float:
        if stage not in self._starts or stage not in self._stops:
            return 0.0
        return (self._stops[stage] - self._starts[stage]).total_seconds()

    def total_elapsed(self) -> float:
        if self.experiment_start is None or self.experiment_end is None:
            return 0.0
        return (self.experiment_end - self.experiment_start).total_seconds()

    def to_records(self) -> List[Dict]:
        return [
            {
                "stage": s,
                "start": self._starts.get(s),
                "stop": self._stops.get(s),
                "seconds": self.elapsed(s),
            }
            for s in self._order
        ]


# ---------------------------------------------------------------------------
# DataSnapshot
# ---------------------------------------------------------------------------

@dataclass
class DataSnapshot:
    """Lightweight preview of a DataFrame for embedding in the report."""
    name: str
    shape: tuple
    columns: list
    head_html: str
    dtypes: dict
    describe_html: str
    file_path: str
    memory_mb: float


def snapshot_dataframe(
    df: pd.DataFrame,
    name: str,
    file_path: str = "",
    head_n: int = 5,
    max_cols: int = 12,
) -> DataSnapshot:
    """Create a DataSnapshot from a DataFrame."""
    truncated = df
    if df.shape[1] > max_cols:
        truncated = df.iloc[:, :max_cols].copy()
        truncated[f"... +{df.shape[1] - max_cols} more"] = "..."

    fmt = lambda x: f"{x:.4g}" if isinstance(x, float) else x  # noqa: E731

    head_html = truncated.head(head_n).to_html(
        classes="data-table", border=0,
        float_format=lambda x: f"{x:.4g}",
    )
    try:
        describe_html = df.describe().to_html(
            classes="data-table", border=0,
            float_format=lambda x: f"{x:.4g}",
        )
    except Exception:
        describe_html = "<em>Could not compute summary statistics.</em>"

    memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

    return DataSnapshot(
        name=name,
        shape=df.shape,
        columns=list(df.columns),
        head_html=head_html,
        dtypes={str(k): str(v) for k, v in df.dtypes.items()},
        describe_html=describe_html,
        file_path=file_path,
        memory_mb=memory_mb,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_seconds(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s}s"
    else:
        h, rem = divmod(int(seconds), 3600)
        m, s = divmod(rem, 60)
        return f"{h}h {m}m {s}s"


def _embed_image(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{data}"


def _config_to_html(config: Any, indent: int = 0) -> str:
    if isinstance(config, dict):
        items = []
        for k, v in config.items():
            if isinstance(v, dict):
                nested = _config_to_html(v, indent + 1)
                items.append(
                    f'<div><dt>{k}</dt><dd>'
                    f'<div class="config-nested">{nested}</div>'
                    f'</dd></div>'
                )
            elif isinstance(v, list):
                val = "[" + ", ".join(str(i) for i in v) + "]"
                items.append(f'<div><dt>{k}</dt><dd><code>{val}</code></dd></div>')
            elif v is None:
                items.append(f'<div><dt>{k}</dt><dd><em>not set</em></dd></div>')
            else:
                items.append(f'<div><dt>{k}</dt><dd><code>{v}</code></dd></div>')
        return '<dl class="config-dl">' + "".join(items) + "</dl>"
    return str(config)


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _build_timing_section(timer: Optional[StageTimer]) -> str:
    if timer is None:
        return "<p class='muted'>No timing data available.</p>"

    records = timer.to_records()
    if not records:
        return "<p class='muted'>No stages recorded.</p>"

    total = timer.total_elapsed()
    max_elapsed = max(r["seconds"] for r in records) if records else 1.0

    rows = []
    for r in records:
        label = r["stage"].replace("_", " ").title()
        secs = r["seconds"]
        width_pct = (secs / max_elapsed * 100) if max_elapsed > 0 else 0
        pct_of_total = (secs / total * 100) if total > 0 else 0
        rows.append(
            f'<tr>'
            f'<td class="timing-label">{label}</td>'
            f'<td class="timing-bar-cell">'
            f'<div class="timing-bar-bg">'
            f'<div class="timing-bar-fill" style="width:{width_pct:.1f}%"></div>'
            f'</div></td>'
            f'<td class="timing-value">{_fmt_seconds(secs)}</td>'
            f'<td class="timing-pct">{pct_of_total:.1f}%</td>'
            f'</tr>'
        )

    return (
        '<table class="timing-table">'
        + "".join(rows)
        + "</table>"
        + f'<p class="timing-total">Total wall-clock time: <strong>{_fmt_seconds(total)}</strong></p>'
    )


def _build_data_section(snapshots: Optional[List[DataSnapshot]]) -> str:
    if not snapshots:
        return "<p class='muted'>No data snapshots available.</p>"

    cards = []
    for s in snapshots:
        cards.append(
            f'<div class="snapshot-card">'
            f'<div class="snapshot-header">'
            f'<span class="snapshot-name">{s.name}</span>'
            f'<span class="badge">{s.shape[0]:,} rows \u00d7 {s.shape[1]:,} cols</span>'
            f'<span class="badge">{s.memory_mb:.2f} MB</span>'
            f'<span class="snapshot-path">{s.file_path}</span>'
            f'</div>'
            f'<div class="snapshot-body">'
            f'<details open><summary>Preview (first rows)</summary>'
            f'<div class="table-wrap">{s.head_html}</div></details>'
            f'<details><summary>Summary Statistics</summary>'
            f'<div class="table-wrap">{s.describe_html}</div></details>'
            f'</div></div>'
        )
    return "\n".join(cards)


def _build_results_section(results: Dict, config: Dict) -> str:
    html: List[str] = []

    pas_summary = results.get("pas_summary", {})
    num_pathways = pas_summary.get("num_pathways", 0)
    num_samples = pas_summary.get("num_samples", 0)

    kpis = [
        (str(num_pathways), "Pathways"),
        (str(num_samples), "Samples"),
    ]

    pas_analysis = results.get("pas_analysis", {})
    if pas_analysis and "error" not in pas_analysis:
        kpis += [
            (str(pas_analysis.get("num_tested", "\u2013")), "Tested"),
            (str(pas_analysis.get("num_significant", "\u2013")), "Significant"),
            (str(pas_analysis.get("method", "\u2013")).upper(), "Method"),
            (str(pas_analysis.get("alpha", "\u2013")), "Alpha"),
            (str(pas_analysis.get("correction", "\u2013")), "Correction"),
        ]

    kpi_html = "".join(
        f'<div class="kpi-card">'
        f'<div class="kpi-value">{val}</div>'
        f'<div class="kpi-label">{label}</div>'
        f'</div>'
        for val, label in kpis
    )
    html.append(f'<div class="kpi-row">{kpi_html}</div>')

    # Differential analysis table
    top_diff = pas_analysis.get("top_differential", [])
    if top_diff:
        html.append("<h3 class='subsection-title'>Top Differential Pathways</h3>")
        rows = []
        ndash = "\u2013"
        for i, row in enumerate(top_diff, 1):
            sig = row.get("significant", False)
            sig_class = " class='row-sig'" if sig else ""
            sig_icon = "\u2713" if sig else ""
            pathway = row.get("pathway") or ndash
            rows.append(
                f"<tr{sig_class}>"
                f"<td>{i}</td>"
                f"<td>{pathway}</td>"
                f"<td>{row.get('pvalue', 0):.3e}</td>"
                f"<td>{row.get('pvalue_adj', 0):.3e}</td>"
                f"<td>{row.get('effect_size', 0):.4f}</td>"
                f"<td>{sig_icon}</td>"
                f"</tr>"
            )
        html.append(
            '<div class="table-wrap">'
            '<table class="results-table"><thead>'
            "<tr><th>#</th><th>Pathway</th><th>P-value</th>"
            "<th>P-adj</th><th>Effect Size</th><th>Sig</th></tr>"
            "</thead><tbody>"
            + "".join(rows)
            + "</tbody></table></div>"
        )

    # Top pathways by mean PAS
    top_pathways = results.get("top_pathways", [])
    if top_pathways:
        html.append("<h3 class='subsection-title' style='margin-top:24px;'>Top Pathways by Mean PAS</h3>")
        rows = [
            f"<tr><td>{i}</td><td>{pw['name']}</td>"
            f"<td>{pw['mean']:.4f}</td><td>{pw['std']:.4f}</td></tr>"
            for i, pw in enumerate(top_pathways[:15], 1)
        ]
        html.append(
            '<div class="table-wrap">'
            '<table class="results-table"><thead>'
            "<tr><th>#</th><th>Pathway</th><th>Mean PAS</th><th>Std PAS</th></tr>"
            "</thead><tbody>"
            + "".join(rows)
            + "</tbody></table></div>"
        )

    # Association results
    top_assoc = results.get("association", {}).get("top_pathways", [])
    if top_assoc:
        html.append("<h3 class='subsection-title' style='margin-top:24px;'>Top Association Results</h3>")
        rows = []
        ndash = "\u2013"
        for i, assoc in enumerate(top_assoc[:15], 1):
            min_p = assoc.get("pvalue", assoc.get("min_pvalue", ndash))
            pval_str = f"{min_p:.3e}" if isinstance(min_p, float) else str(min_p)
            pathway = assoc.get("pathway") or ndash
            rows.append(f"<tr><td>{i}</td><td>{pathway}</td><td>{pval_str}</td></tr>")
        html.append(
            '<div class="table-wrap">'
            '<table class="results-table"><thead>'
            "<tr><th>#</th><th>Pathway</th><th>Min P-value</th></tr>"
            "</thead><tbody>"
            + "".join(rows)
            + "</tbody></table></div>"
        )

    if not html:
        return "<p class='muted'>No results available yet.</p>"
    return "\n".join(html)


def _build_figures_section(figures: List[Dict]) -> str:
    if not figures:
        return "<p class='muted'>No figures generated.</p>"

    cards = []
    for fig in figures:
        title = Path(fig["path"]).stem.replace("_", " ").title()
        cards.append(
            f'<div class="figure-card">'
            f'<div class="figure-title">{title}</div>'
            f'<img src="{fig["data_uri"]}" alt="{title}" loading="lazy">'
            f'</div>'
        )
    return f'<div class="figures-grid">{"".join(cards)}</div>'


def _build_manifest_section(manifest: List[Dict]) -> str:
    if not manifest:
        return "<p class='muted'>No files found.</p>"

    ext_icons = {
        ".csv": "\U0001f4ca", ".tsv": "\U0001f4ca",
        ".json": "\U0001f4cb", ".yaml": "\u2699\ufe0f", ".yml": "\u2699\ufe0f",
        ".png": "\U0001f5bc\ufe0f", ".jpg": "\U0001f5bc\ufe0f", ".jpeg": "\U0001f5bc\ufe0f",
        ".html": "\U0001f310", ".md": "\U0001f4dd",
    }

    rows = []
    for f in manifest:
        ext = Path(f["path"]).suffix.lower()
        icon = ext_icons.get(ext, "\U0001f4c4")
        rows.append(
            f"<tr>"
            f"<td>{icon}</td>"
            f'<td class="file-path">{f["path"]}</td>'
            f"<td>{f['size']}</td>"
            f"</tr>"
        )

    return (
        '<div class="table-wrap">'
        '<table class="manifest-table"><thead>'
        "<tr><th></th><th>File</th><th>Size</th></tr>"
        "</thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

_REPORT_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta name="generated" content="{timestamp_iso}">
<title>PathWAS \u2014 {experiment_name}</title>
<style>
:root {{
  --bg: #f6f8fa;
  --surface: #ffffff;
  --border: #d0d7de;
  --text: #1f2328;
  --text-muted: #656d76;
  --accent: #0969da;
  --accent-light: #ddf4ff;
  --success: #1a7f37;
  --success-bg: #dafbe1;
  --danger: #cf222e;
  --danger-bg: #ffebe9;
  --code-bg: #f6f8fa;
  --nav-bg: #24292f;
  --nav-text: #ffffff;
  --timing-bar: #0969da;
  --kpi-bg: #f0f6ff;
  --sig-row-bg: #dafbe1;
  --font-mono: "JetBrains Mono","Fira Code","Cascadia Code",Consolas,monospace;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --bg: #0d1117;
    --surface: #161b22;
    --border: #30363d;
    --text: #e6edf3;
    --text-muted: #8b949e;
    --accent: #58a6ff;
    --accent-light: #1c2c3e;
    --success: #3fb950;
    --success-bg: #1a3a2a;
    --danger: #f85149;
    --danger-bg: #3a1a1e;
    --code-bg: #161b22;
    --nav-bg: #161b22;
    --nav-text: #c9d1d9;
    --timing-bar: #58a6ff;
    --kpi-bg: #1c2c3e;
    --sig-row-bg: #1a3a2a;
  }}
}}
*,*::before,*::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.6;
  font-size: 15px;
}}
/* ---- Nav ---- */
nav {{
  position: sticky; top: 0; z-index: 100;
  background: var(--nav-bg);
  display: flex; align-items: center; gap: 2px;
  padding: 0 20px; height: 48px;
  box-shadow: 0 1px 4px rgba(0,0,0,.4);
}}
.nav-brand {{
  color: var(--nav-text); font-weight: 700; font-size: 14px;
  margin-right: 14px; letter-spacing: .4px; opacity: .95;
}}
nav a {{
  color: var(--nav-text); text-decoration: none; font-size: 13px;
  padding: 5px 11px; border-radius: 6px; opacity: .7;
  transition: opacity .15s, background .15s; white-space: nowrap;
}}
nav a:hover {{ opacity: 1; background: rgba(255,255,255,.12); }}
/* ---- Container ---- */
.container {{ max-width: 1100px; margin: 0 auto; padding: 32px 24px 72px; }}
/* ---- Header ---- */
.exp-header {{
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 10px; padding: 28px 32px; margin-bottom: 36px;
}}
.exp-header h1 {{ font-size: 26px; font-weight: 700; margin-bottom: 10px; }}
.header-meta {{
  display: flex; flex-wrap: wrap; gap: 18px;
  font-size: 13px; color: var(--text-muted); margin-top: 12px;
}}
.header-meta span {{ display: flex; align-items: center; gap: 5px; }}
.header-meta code {{
  font-family: var(--font-mono); font-size: 12px;
  background: var(--code-bg); padding: 1px 5px; border-radius: 4px;
}}
.status-badge {{
  display: inline-block; padding: 3px 11px; border-radius: 12px;
  font-size: 12px; font-weight: 600; letter-spacing: .3px;
}}
.status-success {{ background: var(--success-bg); color: var(--success); }}
.status-failed  {{ background: var(--danger-bg);  color: var(--danger); }}
.description-box {{
  margin-top: 18px; padding: 12px 16px;
  border-left: 4px solid var(--accent);
  background: var(--accent-light); border-radius: 0 6px 6px 0;
  font-size: 14px;
}}
/* ---- Sections ---- */
section {{ margin-bottom: 44px; }}
.section-title {{
  font-size: 18px; font-weight: 600;
  padding-bottom: 10px; border-bottom: 2px solid var(--border); margin-bottom: 22px;
}}
.subsection-title {{ font-size: 15px; font-weight: 600; margin-bottom: 12px; }}
.muted {{ color: var(--text-muted); font-style: italic; font-size: 14px; }}
/* ---- Timing ---- */
.timing-table {{ width: 100%; border-collapse: collapse; }}
.timing-table tr {{ border-bottom: 1px solid var(--border); }}
.timing-table td {{ padding: 8px 0; vertical-align: middle; }}
.timing-label {{
  width: 200px; font-size: 13px; color: var(--text-muted);
  padding-right: 16px; white-space: nowrap;
}}
.timing-bar-cell {{ width: 100%; }}
.timing-bar-bg {{
  background: var(--border); border-radius: 4px; height: 20px;
  position: relative; overflow: hidden;
}}
.timing-bar-fill {{
  height: 100%; background: var(--timing-bar); border-radius: 4px;
}}
.timing-value {{
  width: 80px; text-align: right; font-size: 12px;
  font-family: var(--font-mono); padding-left: 12px;
}}
.timing-pct {{
  width: 50px; text-align: right; font-size: 12px;
  color: var(--text-muted); padding-left: 8px;
}}
.timing-total {{
  margin-top: 12px; font-size: 13px;
  color: var(--text-muted); text-align: right;
}}
/* ---- KPI cards ---- */
.kpi-row {{ display: flex; flex-wrap: wrap; gap: 14px; margin-bottom: 24px; }}
.kpi-card {{
  background: var(--kpi-bg); border: 1px solid var(--border);
  border-radius: 8px; padding: 16px 20px; min-width: 120px;
}}
.kpi-value {{ font-size: 28px; font-weight: 700; color: var(--accent); line-height: 1; }}
.kpi-label {{
  font-size: 11px; text-transform: uppercase;
  letter-spacing: .6px; color: var(--text-muted); margin-top: 4px;
}}
/* ---- Snapshot cards ---- */
.snapshot-card {{
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; margin-bottom: 20px; overflow: hidden;
}}
.snapshot-header {{
  padding: 12px 20px; background: var(--code-bg);
  border-bottom: 1px solid var(--border);
  display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
}}
.snapshot-name {{ font-weight: 600; font-size: 15px; }}
.badge {{
  display: inline-block; padding: 2px 8px; border-radius: 10px;
  font-size: 11px; font-weight: 600;
  background: var(--accent-light); color: var(--accent);
}}
.snapshot-path {{
  font-family: var(--font-mono); font-size: 11px;
  color: var(--text-muted); margin-left: auto;
}}
.snapshot-body {{ padding: 14px 20px; }}
details summary {{
  cursor: pointer; font-size: 13px; font-weight: 600;
  color: var(--accent); padding: 6px 0; user-select: none;
}}
details summary:hover {{ text-decoration: underline; }}
/* ---- Tables ---- */
.table-wrap {{ overflow-x: auto; margin-top: 8px; }}
table.data-table {{
  width: 100%; border-collapse: collapse;
  font-size: 12px; font-family: var(--font-mono);
}}
table.data-table thead tr {{
  background: var(--code-bg); position: sticky; top: 48px;
}}
table.data-table th {{
  padding: 8px 12px; text-align: left;
  border-bottom: 2px solid var(--border); font-weight: 600;
  white-space: nowrap;
  font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  font-size: 12px;
}}
table.data-table td {{
  padding: 5px 12px; border-bottom: 1px solid var(--border); white-space: nowrap;
}}
table.data-table tr:hover td {{ background: var(--accent-light); }}
table.results-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
table.results-table th {{
  padding: 10px 14px; text-align: left; background: var(--code-bg);
  border-bottom: 2px solid var(--border); font-weight: 600;
  font-size: 12px; text-transform: uppercase; letter-spacing: .4px;
}}
table.results-table td {{ padding: 9px 14px; border-bottom: 1px solid var(--border); }}
table.results-table tr:hover td {{ background: var(--accent-light); }}
table.results-table tr.row-sig td {{ background: var(--sig-row-bg); }}
/* ---- Figures ---- */
.figures-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
  gap: 20px;
}}
.figure-card {{
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; overflow: hidden;
}}
.figure-title {{
  padding: 11px 16px; font-size: 13px; font-weight: 600;
  border-bottom: 1px solid var(--border); background: var(--code-bg);
}}
.figure-card img {{ width: 100%; height: auto; display: block; }}
/* ---- Config ---- */
dl.config-dl {{ margin: 0; }}
dl.config-dl > div {{
  display: flex; gap: 16px; padding: 6px 0;
  border-bottom: 1px solid var(--border);
}}
dl.config-dl dt {{ min-width: 200px; font-weight: 600; font-size: 13px; }}
dl.config-dl dd {{ font-size: 13px; color: var(--text-muted); }}
dl.config-dl dd code {{
  background: var(--code-bg); padding: 1px 6px; border-radius: 4px;
  font-family: var(--font-mono); font-size: 12px;
}}
dl.config-dl dd em {{ color: var(--text-muted); font-style: italic; }}
.config-nested {{
  margin-left: 20px; margin-top: 8px;
  border-left: 2px solid var(--border); padding-left: 16px;
}}
/* ---- Manifest ---- */
table.manifest-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
table.manifest-table th {{
  padding: 8px 12px; text-align: left; background: var(--code-bg);
  border-bottom: 2px solid var(--border); font-weight: 600; font-size: 12px;
}}
table.manifest-table td {{ padding: 7px 12px; border-bottom: 1px solid var(--border); }}
table.manifest-table td:last-child {{
  text-align: right; font-family: var(--font-mono); font-size: 12px;
}}
table.manifest-table tr:hover td {{ background: var(--accent-light); }}
.file-path {{ font-family: var(--font-mono); font-size: 12px; }}
/* ---- Print ---- */
@media print {{
  nav {{ display: none; }}
  details {{ display: block; }}
  .container {{ max-width: none; padding: 0; }}
  .figures-grid {{ grid-template-columns: 1fr 1fr; }}
}}
</style>
</head>
<body>

<nav>
  <span class="nav-brand">PathWAS</span>
  <a href="#timing">\u23f1 Timing</a>
  <a href="#data">\U0001f4ca Data ({n_snapshots})</a>
  <a href="#results">\U0001f4c8 Results</a>
  <a href="#figures">\U0001f5bc Figures ({n_figures})</a>
  <a href="#config">\u2699\ufe0f Config</a>
  <a href="#files">\U0001f4c1 Files</a>
</nav>

<div class="container">

  <div class="exp-header">
    <h1>{experiment_name}</h1>
    <div><span class="status-badge {status_class}">{status}</span></div>
    <div class="header-meta">
      <span>\U0001f550 Generated: <strong>{timestamp_display}</strong></span>
      <span>\u23f1 Runtime: <strong>{total_time}</strong></span>
      <span>\U0001f516 Commit: <code>{git_commit}</code></span>
      <span>\U0001f4c2 <code>{exp_dir}</code></span>
    </div>
    <div class="description-box">{description}</div>
  </div>

  <section id="timing">
    <h2 class="section-title">\u23f1 Pipeline Timing</h2>
    {timing_section}
  </section>

  <section id="data">
    <h2 class="section-title">\U0001f4ca Data Snapshots ({n_snapshots})</h2>
    {data_section}
  </section>

  <section id="results">
    <h2 class="section-title">\U0001f4c8 Results</h2>
    {results_section}
  </section>

  <section id="figures">
    <h2 class="section-title">\U0001f5bc Figures ({n_figures})</h2>
    {figures_section}
  </section>

  <section id="config">
    <h2 class="section-title">\u2699\ufe0f Configuration</h2>
    {config_section}
  </section>

  <section id="files">
    <h2 class="section-title">\U0001f4c1 Output Files</h2>
    {manifest_section}
  </section>

</div>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_experiment_report(
    config: Dict[str, Any],
    results: Dict[str, Any],
    exp_dir: Path,
    timer: Optional[StageTimer] = None,
    snapshots: Optional[List[DataSnapshot]] = None,
) -> str:
    """Generate a self-contained HTML experiment report.

    Parameters
    ----------
    config : dict
        Full merged experiment configuration.
    results : dict
        Results dict returned by run_experiment().
    exp_dir : Path
        Experiment output directory.
    timer : StageTimer, optional
        Pipeline stage timer populated during the run.
    snapshots : list of DataSnapshot, optional
        DataFrame snapshots collected during the run.

    Returns
    -------
    str
        Absolute path to the written HTML file.
    """
    exp_dir = Path(exp_dir)

    # Collect and embed figures
    figures: List[Dict] = []
    figures_dir = exp_dir / "figures"
    if figures_dir.exists():
        for p in sorted(figures_dir.iterdir()):
            if p.suffix.lower() in (".png", ".jpg", ".jpeg"):
                try:
                    figures.append({"path": p.name, "data_uri": _embed_image(p)})
                except Exception as exc:
                    logging.warning("Could not embed figure %s: %s", p.name, exc)

    # Build file manifest
    manifest: List[Dict] = []
    for p in sorted(exp_dir.rglob("*")):
        if p.is_file():
            size = p.stat().st_size
            if size < 1024:
                size_str = f"{size} B"
            elif size < 1024 * 1024:
                size_str = f"{size / 1024:.1f} KB"
            else:
                size_str = f"{size / 1024 / 1024:.1f} MB"
            manifest.append({"path": str(p.relative_to(exp_dir)), "size": size_str})

    # Extract metadata from results
    meta = results.get("metadata", {})
    git_commit = meta.get("git_commit") or "unknown"
    if len(git_commit) > 12:
        git_commit = git_commit[:12]

    exp_name = config.get("experiment", {}).get("name", "unknown")
    description = config.get("experiment", {}).get("description", "No description provided.")

    qc = results.get("qc", {})
    if qc.get("status") == "failed":
        status, status_class = "failed", "status-failed"
    else:
        status, status_class = "completed", "status-success"

    now = datetime.now()
    timestamp_display = now.strftime("%Y-%m-%d %H:%M:%S")
    timestamp_iso = now.isoformat()
    start_time = timer.experiment_start.strftime("%H:%M:%S") if timer and timer.experiment_start else "\u2013"
    end_time = timer.experiment_end.strftime("%H:%M:%S") if timer and timer.experiment_end else "\u2013"
    total_time = _fmt_seconds(timer.total_elapsed()) if timer else "\u2013"

    html = _REPORT_TEMPLATE.format(
        experiment_name=exp_name,
        description=description,
        timestamp_display=timestamp_display,
        timestamp_iso=timestamp_iso,
        start_time=start_time,
        end_time=end_time,
        total_time=total_time,
        git_commit=git_commit,
        status=status,
        status_class=status_class,
        exp_dir=str(exp_dir),
        timing_section=_build_timing_section(timer),
        data_section=_build_data_section(snapshots),
        results_section=_build_results_section(results, config),
        figures_section=_build_figures_section(figures),
        config_section=_config_to_html(config),
        manifest_section=_build_manifest_section(manifest),
        n_figures=len(figures),
        n_snapshots=len(snapshots) if snapshots else 0,
    )

    html_path = exp_dir / "experiment_report.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    logging.info("Generated HTML report: %s", html_path)

    md_path = exp_dir / "experiment_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# {exp_name}\n\nOpen `experiment_report.html` for the full interactive report.\n")

    return str(html_path)
