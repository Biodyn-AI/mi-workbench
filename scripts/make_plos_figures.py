#!/usr/bin/env python
"""Build the PLOS ONE revision figures (Fig2-Fig5, S1 Fig) from the experiment outputs.

Run through ``scripts/make_plos_figures.sh`` (sets the interpreter and the
matplotlib / temp directories), or directly:

    PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR="/Volumes/Crucial X6/tmp_miw/mpl" \\
        /Users/ihorkendiukhov/anaconda3/bin/python scripts/make_plos_figures.py [--only Fig2,Fig5]

Inputs (read only; macOS ``._*`` files are never read):

- Fig2  experiments/benchmark/results/{benchmark_results.json, recall_by_condition.csv,
        recall_by_family.csv, false_alarms_clean.csv, cost_by_condition.csv}
- Fig3  experiments/benchmark/results/merge_eval.json
- Fig4  experiments/stopping/results/{stopping_results.json, stopping_rules.csv}
- Fig5  experiments/case_study/results/independent_validation.json
- S1    experiments/orchestration/{overhead.json, real_concurrency.json}

Outputs, in paper/plos/figures/ (PLOS ONE figure requirements):

- ``<name>.tif``: RGB, 8 bit per channel, no alpha, LZW, 600 dpi, 7.5 in wide,
  height <= 8.75 in. Each TIFF is re-opened with PIL and checked.
- ``<name>_preview.png`` (200 dpi) and ``<name>.pdf`` (TrueType-embedded Arial).
- ``FIGURE_NOTES.md``: per figure and panel, the exact JSON keys / CSV rows
  plotted and the key numbers shown (generated from the same values that are
  drawn, so it cannot drift from the figures).

A figure whose inputs are missing or incomplete is skipped with a message; the
others are still built. Style: Arial 7-9 pt; one y-axis per panel; light grid on
the value axis only; text in #0b0b0b / #52514e; the three model configurations
always use the same colour and marker (gpt-5.6-sol medium: blue circle,
gpt-5.5 medium: orange square, gpt-5.6-luna low: aqua triangle; pooled: black
diamond); colours are used for nothing else.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

REPO = Path(__file__).resolve().parent.parent
BENCH = REPO / "experiments" / "benchmark" / "results"
STOP = REPO / "experiments" / "stopping" / "results"
CASE = REPO / "experiments" / "case_study" / "results"
ORCH = REPO / "experiments" / "orchestration"
DEFAULT_OUT = REPO / "paper" / "plos" / "figures"

# ── style tokens ─────────────────────────────────────────────────────────

INK = "#0b0b0b"        # primary text, pooled series
INK_2 = "#52514e"      # secondary text
NEUTRAL = "#8a8984"    # neutral (non-config) marks, axis lines
NEUTRAL_LIGHT = "#bdbcb7"
BAND = "#e2e1dd"       # light interval band
GRID = "#e7e6e2"
WHITE = "#ffffff"

# The three model configurations: fixed colour + marker, same in every figure.
CONFIGS: dict[str, dict[str, str]] = {
    "gpt-5.6-sol": {"label": "gpt-5.6-sol medium", "color": "#2a78d6", "marker": "o"},
    "gpt-5.5": {"label": "gpt-5.5 medium", "color": "#eb6834", "marker": "s"},
    "gpt-5.6-luna": {"label": "gpt-5.6-luna low", "color": "#1baf7a", "marker": "^"},
}
POOLED = {"label": "pooled", "color": INK, "marker": "D"}
SEQ_LOW, SEQ_HIGH = "#cde2fb", "#104281"   # sequential (heatmap) ramp

PRIMARY_JUDGE = "gpt-5.6-sol"
CONDITIONS = ["C1", "C2", "C3", "C4", "C5"]
COND_LABEL = {"C1": "1 rigor", "C2": "1 combined", "C3": "rigor ×3",
              "C4": "combined ×3", "C5": "3-lens panel"}
COND_TICK = {"C1": "1\nrigor", "C2": "1\ncombined", "C3": "rigor\n×3",
             "C4": "combined\n×3", "C5": "3-lens\npanel"}
FAMILIES = ["statistics", "confounding", "causal_logic", "technical", "biological"]

FIG_WIDTH = 7.5        # in (PLOS maximum; used for every multi-panel figure)
MAX_HEIGHT = 8.75      # in
TIFF_DPI = 600
PREVIEW_DPI = 200

ARIAL_DIR = Path("/System/Library/Fonts/Supplemental")
ARIAL_FILES = ["Arial.ttf", "Arial Bold.ttf", "Arial Italic.ttf", "Arial Bold Italic.ttf"]
FONT_MIN_PT, FONT_MAX_PT = 7.0, 9.0

# FIGURE_NOTES.md content, filled by the plot functions.
NOTES: dict[str, list[str]] = {}


def note(fig: str, line: str) -> None:
    NOTES.setdefault(fig, []).append(line)


# ── small helpers ────────────────────────────────────────────────────────


def load_json(path: Path) -> Optional[dict]:
    if not path.is_file() or path.name.startswith("._"):
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        print(f"  ! {path.name} does not parse ({exc})")
        return None


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def rel(path: Path) -> str:
    return str(path.relative_to(REPO))


def f3(x: Optional[float], nd: int = 3) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def nan_if_none(x: Optional[float]) -> float:
    return float("nan") if x is None else float(x)


def setup_matplotlib():
    """Register Arial from the macOS system fonts and set the shared rcParams."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager as fm
    for name in ARIAL_FILES:
        path = ARIAL_DIR / name
        if not path.is_file():
            sys.exit(f"Arial font file missing: {path}")
        fm.fontManager.addfont(str(path))
    matplotlib.rcParams.update({
        "font.family": "Arial",
        "font.sans-serif": ["Arial"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "mathtext.sf": "Arial",
        "mathtext.default": "regular",
        "pdf.fonttype": 42,          # embed TrueType (Arial) in the PDF
        "ps.fonttype": 42,
        "font.size": 7.5,
        "axes.titlesize": 8,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlecolor": INK,
        "axes.titlepad": 5,
        "axes.labelsize": 7.5,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": NEUTRAL,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.facecolor": WHITE,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "xtick.color": NEUTRAL,
        "ytick.color": NEUTRAL,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.borderaxespad": 0.3,
        "legend.labelcolor": INK,
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        "errorbar.capsize": 0,
        "figure.facecolor": WHITE,
        "savefig.facecolor": WHITE,
        "figure.constrained_layout.h_pad": 0.04,
        "figure.constrained_layout.w_pad": 0.04,
        "figure.constrained_layout.hspace": 0.06,
        "figure.constrained_layout.wspace": 0.04,
    })
    return matplotlib


def value_grid(ax, axis: str = "y") -> None:
    """Recessive grid on the value axis only."""
    ax.grid(False)
    (ax.yaxis if axis == "y" else ax.xaxis).grid(True, color=GRID, linewidth=0.5, zorder=0)


def cfg_style(model: str) -> dict[str, str]:
    return POOLED if model == "pooled" else CONFIGS[model]


def config_handles(models: list[str], include_pooled: bool = True):
    from matplotlib.lines import Line2D
    hs = []
    for m in models:
        s = CONFIGS[m]
        hs.append(Line2D([], [], color=s["color"], marker=s["marker"], ls="none", ms=4.5,
                         mec=WHITE, mew=0.5, label=s["label"]))
    if include_pooled:
        hs.append(Line2D([], [], color=INK, marker="D", ls="none", ms=4, mec=WHITE, mew=0.5,
                         label="pooled (3 configurations)"))
    return hs


def add_panel_letters(fig, axes: list) -> None:
    """Freeze the constrained layout, then put a bold letter at the top-left of
    each panel (left edge of the panel's tight bbox, level with its title)."""
    fig.canvas.draw()
    fig.set_layout_engine("none")
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    for ax, letter in zip(axes, "ABCDEFGH"):
        bb = ax.get_tightbbox(renderer).transformed(inv)
        title_bb = ax.title.get_window_extent(renderer).transformed(inv)
        top = max(title_bb.y1, ax.get_position().y1) if ax.get_title(loc="left") else bb.y1
        x = max(bb.x0 - 0.004, 0.002)
        fig.text(x, top, letter, fontsize=9, fontweight="bold", color=INK,
                 ha="left", va="top")


def contrast_text_color(rgb) -> str:
    """Black or white annotation, whichever has higher WCAG contrast on rgb."""
    def lin(c):
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (lin(c) for c in rgb[:3])
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    c_black = (lum + 0.05) / 0.05
    c_white = 1.05 / (lum + 0.05)
    return INK if c_black >= c_white else WHITE


# ── output + checks ──────────────────────────────────────────────────────


def verify_fonts(fig) -> list[str]:
    """Every visible text artist must resolve to an Arial file and be 7-9 pt;
    no glyph may fall back to another font. Returns the Arial files used."""
    import matplotlib.text as mtext
    from matplotlib import font_manager as fm
    used: set[str] = set()
    problems: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig.canvas.draw()
    glyph = [str(w.message) for w in caught if "Glyph" in str(w.message) or "missing" in str(w.message)]
    problems += [f"glyph warning: {g}" for g in glyph]
    for t in fig.findobj(mtext.Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        try:
            path = fm.findfont(t.get_fontproperties(), fallback_to_default=False)
        except ValueError as exc:
            problems.append(f"font not found for {t.get_text()!r}: {exc}")
            continue
        name = Path(path).name
        used.add(name)
        if name not in ARIAL_FILES:
            problems.append(f"{t.get_text()!r} resolves to {name}")
        size = t.get_fontsize()
        if not (FONT_MIN_PT - 1e-6 <= size <= FONT_MAX_PT + 1e-6):
            problems.append(f"{t.get_text()!r} is {size:g} pt")
    if problems:
        raise AssertionError("font check failed:\n  " + "\n  ".join(problems))
    return sorted(used)


def pdf_font_names(pdf_path: Path) -> list[str]:
    data = pdf_path.read_bytes()
    return sorted({m.decode() for m in re.findall(rb"/BaseFont\s*/([A-Za-z0-9+_\-]+)", data)})


def check_tiff(path: Path) -> str:
    from PIL import Image
    with Image.open(path) as im:
        mode, (w, h) = im.mode, im.size
        comp = im.info.get("compression")
        dpi = tuple(float(v) for v in im.info.get("dpi", (0, 0)))
        bands = im.getbands()
    w_in, h_in = w / float(dpi[0]), h / float(dpi[1])
    assert mode == "RGB", f"{path.name}: mode {mode} (want RGB, no alpha)"
    assert bands == ("R", "G", "B"), f"{path.name}: bands {bands}"
    assert comp == "tiff_lzw", f"{path.name}: compression {comp}"
    assert abs(dpi[0] - TIFF_DPI) < 1 and abs(dpi[1] - TIFF_DPI) < 1, f"{path.name}: dpi {dpi}"
    assert 2.63 - 1e-3 <= w_in <= 7.5 + 1e-3, f"{path.name}: width {w_in:.3f} in"
    assert h_in <= MAX_HEIGHT + 1e-3, f"{path.name}: height {h_in:.3f} in"
    return (f"CHECK {path.name}: mode={mode} (8-bit/channel, no alpha) compression={comp} "
            f"dpi={dpi[0]:.0f}x{dpi[1]:.0f} size={w}x{h}px = {w_in:.3f} x {h_in:.3f} in  OK")


def save_figure(fig, base: str, out_dir: Path, letter_axes: list) -> None:
    """Letters, font check, then PDF + PNG preview + LZW TIFF (re-opened and checked)."""
    import matplotlib.pyplot as plt
    from PIL import Image
    add_panel_letters(fig, letter_axes)
    w, h = fig.get_size_inches()
    assert abs(w - FIG_WIDTH) < 1e-6 and h <= MAX_HEIGHT, (w, h)
    fonts = verify_fonts(fig)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{base}.pdf"
    fig.savefig(pdf, format="pdf")
    fig.savefig(out_dir / f"{base}_preview.png", dpi=PREVIEW_DPI, format="png")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=TIFF_DPI)
    plt.close(fig)
    buf.seek(0)
    with Image.open(buf) as im:
        rgba = im.convert("RGBA")
    rgb = Image.new("RGB", rgba.size, (255, 255, 255))
    rgb.paste(rgba, mask=rgba.split()[3])
    tif = out_dir / f"{base}.tif"
    rgb.save(tif, format="TIFF", compression="tiff_lzw", dpi=(TIFF_DPI, TIFF_DPI))
    pdf_fonts = pdf_font_names(pdf)
    assert pdf_fonts and all("Arial" in f for f in pdf_fonts), f"PDF fonts: {pdf_fonts}"
    print(f"  fonts used (matplotlib): {', '.join(fonts)}; embedded in PDF: {', '.join(pdf_fonts)}")
    print("  " + check_tiff(tif))


# ── Fig 2: planted-flaw benchmark ───────────────────────────────────────


def fig2_benchmark(plt, out_dir: Path) -> bool:
    """(A) recall by condition, (B) recall by family x condition (pooled),
    (C) false alarms on clean items, (D) recall vs output tokens."""
    paths = {k: BENCH / f for k, f in {
        "json": "benchmark_results.json", "recall": "recall_by_condition.csv",
        "family": "recall_by_family.csv", "fa": "false_alarms_clean.csv",
        "cost": "cost_by_condition.csv"}.items()}
    missing = [p.name for p in paths.values() if not p.is_file()]
    if missing:
        print(f"Fig2: skipped (missing {', '.join(missing)})")
        return False
    bj = load_json(paths["json"])
    assert bj["settings"]["primary_judge"] == PRIMARY_JUDGE, bj["settings"]["primary_judge"]
    view = f"judge:{PRIMARY_JUDGE}"
    recall = {(r["model"], r["condition"]): r for r in read_csv(paths["recall"]) if r["view"] == view}
    family = {(r["condition"], r["family"]): r for r in read_csv(paths["family"])
              if r["view"] == view and r["model"] == "pooled"}
    fa = {(r["model"], r["condition"]): r for r in read_csv(paths["fa"]) if r["view"] == view}
    cost = {(r["model"], r["condition"]): r for r in read_csv(paths["cost"])}
    models = [m for m in CONFIGS if (m, "C1") in recall]
    # Cross-check the CSVs against the JSON 'primary' section (same judge).
    prim = bj["primary"]
    for m in models:
        for c in CONDITIONS:
            assert abs(float(recall[(m, c)]["estimate"]) - prim["recall"][m][c]["estimate"]) < 1e-3
            assert abs(float(fa[(m, c)]["mean_high_critical_merged"])
                       - prim["false_alarms_clean"][m][c]["mean_high_critical_merged"]) < 1e-3

    fig = plt.figure(figsize=(FIG_WIDTH, 5.75), layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.0])
    axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    axC, axD = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    x = np.arange(len(CONDITIONS))
    F = "Fig2"
    note(F, f"Primary judge view `view == '{view}'` throughout (benchmark_results.json "
            f"`settings.primary_judge`); CSV values were cross-checked against "
            f"`primary.recall` / `primary.false_alarms_clean` in benchmark_results.json.")
    note(F, "Conditions: " + "; ".join(f"{c} = {COND_LABEL[c]} "
                                         f"({'+'.join(bj['settings']['conditions'][c])})" for c in CONDITIONS)
         + f". Units: {prim['n_flawed_units']} flawed + {prim['n_clean_units']} clean "
           f"(model x item x repeat); recall = {bj['settings']['recall_definition']}.")

    # (A) recall by condition: configs dodged, pooled diamond, 95% CI.
    series = models + ["pooled"]
    offsets = np.linspace(-0.27, 0.27, len(series))
    for off, m in zip(offsets, series):
        st = cfg_style(m)
        est = np.array([float(recall[(m, c)]["estimate"]) for c in CONDITIONS])
        lo = np.array([float(recall[(m, c)]["ci_low"]) for c in CONDITIONS])
        hi = np.array([float(recall[(m, c)]["ci_high"]) for c in CONDITIONS])
        axA.errorbar(x + off, est, yerr=[est - lo, hi - est], fmt="none", ecolor=st["color"],
                     elinewidth=0.8, zorder=2)
        axA.plot(x + off, est, ls="none", marker=st["marker"], color=st["color"],
                 ms=4.6 if m != "pooled" else 4.2, mec=WHITE, mew=0.5, zorder=3)
    axA.set_xticks(x, [COND_TICK[c] for c in CONDITIONS])
    axA.set_xlim(-0.6, len(CONDITIONS) - 0.4)
    axA.set_ylim(0.5, 1.012)
    axA.set_yticks(np.arange(0.5, 1.01, 0.1))
    axA.set_ylabel("Flaw recall (95% CI)")
    axA.set_title("Planted-flaw recall by condition")
    value_grid(axA)
    axA.tick_params(axis="x", length=0)
    rows = []
    for m in series:
        rows.append(f"  - {m}: " + ", ".join(
            f"{c} {float(recall[(m, c)]['estimate']):.3f} [{float(recall[(m, c)]['ci_low']):.3f}, "
            f"{float(recall[(m, c)]['ci_high']):.3f}]" for c in CONDITIONS))
    note(F, "**A** recall_by_condition.csv, rows `view == judge:gpt-5.6-sol`, columns "
            "`estimate, ci_low, ci_high` (item-cluster bootstrap 95% CI), model in "
            "{gpt-5.6-sol, gpt-5.5, gpt-5.6-luna, pooled}; n_units 36 per config / 108 pooled, 12 items:")
    NOTES[F] += rows

    # (B) recall by family x condition, pooled, annotated heatmap.
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    cmap = LinearSegmentedColormap.from_list("seq_blue", [SEQ_LOW, SEQ_HIGH])
    norm = Normalize(vmin=0.5, vmax=1.0)
    M = np.array([[float(family[(c, f)]["estimate"]) for c in CONDITIONS] for f in FAMILIES])
    n_items = {f: int(family[("C1", f)]["n_items"]) for f in FAMILIES}
    mesh = axB.pcolormesh(np.arange(len(CONDITIONS) + 1) - 0.5, np.arange(len(FAMILIES) + 1) - 0.5, M,
                          cmap=cmap, norm=norm, edgecolors=WHITE, linewidth=1.5)
    for i in range(len(FAMILIES)):
        for j in range(len(CONDITIONS)):
            axB.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                     color=contrast_text_color(cmap(norm(M[i, j]))))
    axB.set_xticks(x, [COND_TICK[c] for c in CONDITIONS])
    axB.set_yticks(np.arange(len(FAMILIES)),
                   [f"{f.replace('_', ' ')} ({n_items[f]})" for f in FAMILIES])
    axB.set_ylim(len(FAMILIES) - 0.5, -0.5)
    axB.tick_params(length=0)
    for s in axB.spines.values():
        s.set_visible(False)
    axB.set_title("Recall by flaw family, pooled")
    cb = fig.colorbar(mesh, ax=axB, fraction=0.05, pad=0.02, aspect=18)
    cb.set_label("Flaw recall", color=INK_2, fontsize=7)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, length=2, color=NEUTRAL, labelcolor=INK_2)
    cb.set_ticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    note(F, "**B** recall_by_family.csv, rows `view == judge:gpt-5.6-sol` and `model == pooled`, "
            "column `estimate`; colour scale fixed 0.5-1.0; row label = family (number of flawed "
            "items carrying that family; n = 3 configs x 3 repeats x items):")
    for f in FAMILIES:
        NOTES[F].append(f"  - {f} (n_items {n_items[f]}, n {family[('C1', f)]['n']}): "
                        + ", ".join(f"{c} {float(family[(c, f)]['estimate']):.2f}" for c in CONDITIONS))

    # (C) false alarms on clean items: pooled outline bar + per-config points.
    pooled_fa = np.array([float(fa[("pooled", c)]["mean_high_critical_merged"]) for c in CONDITIONS])
    axC.bar(x, pooled_fa, width=0.56, facecolor="none", edgecolor=INK, linewidth=0.9, zorder=2)
    offs = np.linspace(-0.16, 0.16, len(models))
    for off, m in zip(offs, models):
        st = cfg_style(m)
        vals = [float(fa[(m, c)]["mean_high_critical_merged"]) for c in CONDITIONS]
        axC.plot(x + off, vals, ls="none", marker=st["marker"], color=st["color"], ms=4.6,
                 mec=WHITE, mew=0.5, zorder=3)
    axC.set_xticks(x, [COND_TICK[c] for c in CONDITIONS])
    axC.set_xlim(-0.6, len(CONDITIONS) - 0.4)
    axC.set_ylim(0, max(float(fa[(m, c)]["mean_high_critical_merged"]) for m in models
                        for c in CONDITIONS) * 1.12)
    axC.set_ylabel("High/critical critiques per clean item")
    axC.set_title("High/critical critiques on clean items")
    value_grid(axC)
    axC.tick_params(axis="x", length=0)
    any_all = all(float(fa[(m, c)]["frac_units_any_high_critical"]) == 1.0
                  for m in series for c in CONDITIONS)
    if any_all:
        axC.text(0.02, 0.97, "Every clean item drew ≥1 high/critical\ncritique in every condition",
                 transform=axC.transAxes, ha="left", va="top", fontsize=7, color=INK_2)
    from matplotlib.patches import Patch
    axC.legend(handles=[Patch(facecolor="none", edgecolor=INK, linewidth=0.9, label="pooled mean")],
               loc="upper right", bbox_to_anchor=(1.0, 1.0))
    note(F, "**C** false_alarms_clean.csv, rows `view == judge:gpt-5.6-sol`, column "
            "`mean_high_critical_merged` (mean number of merged critiques with severity high or "
            "critical per clean unit; 18 clean units per config, 54 pooled). Bar = pooled, points = configs:")
    for m in series:
        NOTES[F].append(f"  - {m}: " + ", ".join(
            f"{c} {float(fa[(m, c)]['mean_high_critical_merged']):.2f}" for c in CONDITIONS))
    note(F, f"  - `frac_units_any_high_critical` = 1.0 for every model x condition: {any_all} "
            "(annotated in the panel).")

    # (D) pooled recall vs mean output tokens per item, one point per condition.
    tok = np.array([float(cost[("pooled", c)]["mean_output"]) for c in CONDITIONS]) / 1000.0
    rp = np.array([float(recall[("pooled", c)]["estimate"]) for c in CONDITIONS])
    rlo = np.array([float(recall[("pooled", c)]["ci_low"]) for c in CONDITIONS])
    rhi = np.array([float(recall[("pooled", c)]["ci_high"]) for c in CONDITIONS])
    axD.errorbar(tok, rp, yerr=[rp - rlo, rhi - rp], fmt="none", ecolor=INK, elinewidth=0.8, zorder=2)
    axD.plot(tok, rp, ls="none", marker="D", color=INK, ms=4.2, mec=WHITE, mew=0.5, zorder=3)
    # Direct labels, centred on each point just above its upper CI end, or just below
    # its lower CI end where the label above would collide with a neighbour.
    below = {"C1", "C3"}
    for c, xt, ylo_, yhi_ in zip(CONDITIONS, tok, rlo, rhi):
        if c in below:
            axD.annotate(COND_LABEL[c], (xt, ylo_), xytext=(0, -3), textcoords="offset points",
                         ha="center", va="top", fontsize=7, color=INK)
        else:
            axD.annotate(COND_LABEL[c], (xt, yhi_), xytext=(0, 3), textcoords="offset points",
                         ha="center", va="bottom", fontsize=7, color=INK)
    axD.set_xlim(0, max(tok) * 1.15)
    axD.set_ylim(0.5, 1.012)
    axD.set_yticks(np.arange(0.5, 1.01, 0.1))
    axD.set_xlabel("Mean output tokens per item (thousands)")
    axD.set_ylabel("Flaw recall, pooled (95% CI)")
    axD.set_title("Recall vs output cost")
    value_grid(axD)
    note(F, "**D** x = cost_by_condition.csv rows `model == pooled`, column `mean_output` "
            "(output tokens of the successful call of each record, summed over the condition's "
            "calls, mean over all 162 units = flawed + clean; retries excluded); y = pooled recall "
            "as in A:")
    for c in CONDITIONS:
        NOTES[F].append(f"  - {c} {COND_LABEL[c]}: {float(cost[('pooled', c)]['mean_output']):.0f} "
                        f"output tokens, recall {float(recall[('pooled', c)]['estimate']):.3f}")

    fig.legend(handles=config_handles(models), loc="outside upper center", ncol=4,
               handletextpad=0.3, columnspacing=1.6)
    save_figure(fig, "Fig2", out_dir, [axA, axB, axC, axD])
    return True


# ── Fig 3: merge evaluation ─────────────────────────────────────────────

MERGE_METHODS = ["jaccard", "tfidf", "llm"]   # embedding is not evaluated: never plotted
MERGE_LABEL = {"jaccard": "Jaccard\n(lexical)", "tfidf": "TF-IDF\ncosine", "llm": "LLM\nadjudication"}
METRIC_TONES = {"precision": NEUTRAL_LIGHT, "recall": NEUTRAL, "f1": INK}
METRIC_LABEL = {"precision": "precision", "recall": "recall", "f1": "F1"}


def merge_eval_ready(path: Path) -> tuple[bool, str]:
    import subprocess
    running = subprocess.run(["pgrep", "-f", "merge_eval.py"], capture_output=True, text=True).stdout.strip()
    if running:
        return False, f"merge_eval.py still running (pid {running.split()[0]})"
    d = load_json(path)
    if d is None:
        return False, f"{path.name} missing or not parseable"
    if not d.get("methods"):
        return False, f"{path.name} has no methods"
    return True, ""


def fig3_merge(plt, out_dir: Path) -> bool:
    path = BENCH / "merge_eval.json"
    ok, why = merge_eval_ready(path)
    if not ok:
        print(f"Fig3: skipped ({why})")
        return False
    d = load_json(path)
    methods = {k: v for k, v in d["methods"].items()
               if k in MERGE_METHODS and not v.get("skipped") and v.get("held_out")}
    if not methods:
        print("Fig3: skipped (no evaluated method with held-out metrics)")
        return False
    order = [m for m in MERGE_METHODS if m in methods]
    F = "Fig3"
    note(F, f"merge_eval.json generated {d.get('generated_utc')}, judge {d.get('judge')}, panel calls "
            f"{d.get('panel_calls')}; {d.get('n_instances')} panel instances "
            f"({d.get('n_instances_flawed')} flawed, {d.get('n_scenarios_flawed')} flawed scenarios), "
            f"{d.get('n_critiques')} critiques, labelled cross-lens pairs {d.get('n_labelled_pairs')}; "
            f"same_lens_merge = {d.get('same_lens_merge')}.")
    skipped = {k: v.get("reason") for k, v in d["methods"].items() if v.get("skipped")}
    note(F, f"Methods plotted: {order}. Not plotted: embedding (not evaluated"
            + (f"; JSON reason: {skipped['embedding']!r}" if "embedding" in skipped else "") + ").")
    rec = d.get("recommended_default", {})
    note(F, f"`recommended_default`: {rec}.")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(FIG_WIDTH, 2.85), layout="constrained",
                                   gridspec_kw={"width_ratios": [1.0, 1.15]})
    # (A) held-out pairwise precision / recall / F1 per method (grouped thin bars).
    x = np.arange(len(order))
    width = 0.22
    for k, metric in enumerate(["precision", "recall", "f1"]):
        vals = [nan_if_none(methods[m]["held_out"].get(metric)) for m in order]
        xs = x + (k - 1) * width
        axA.bar(xs, np.nan_to_num(vals), width=width * 0.86, color=METRIC_TONES[metric],
                label=METRIC_LABEL[metric], zorder=2)
        for xi, v in zip(xs, vals):
            if np.isnan(v):
                axA.text(xi, 0.02, "n/a", ha="center", va="bottom", fontsize=7, color=INK_2, rotation=90)
            elif metric == "f1":
                axA.text(xi, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=7, color=INK)
    axA.set_xticks(x, [MERGE_LABEL[m] for m in order])
    axA.tick_params(axis="x", length=0)
    axA.set_ylim(0, 1.12)
    axA.set_yticks(np.arange(0, 1.01, 0.2))
    axA.set_ylabel("Pairwise score, held out")
    axA.set_title("Duplicate detection, leave-one-scenario-out")
    value_grid(axA)
    axA.legend(loc="upper left", ncol=3, handlelength=1.0, columnspacing=1.0,
               bbox_to_anchor=(0.0, 1.0))
    note(F, "**A** `methods.<m>.held_out.{precision,recall,f1}` (pooled pair counts over the "
            "held-out scenarios; jaccard/tfidf thresholds chosen on the other scenarios in each fold, "
            "LLM has no threshold). F1 printed above its bar.")
    for m in order:
        h = methods[m]["held_out"]
        extra = ""
        if m == "llm":
            hx = methods[m].get("held_out_excluding_repaired") or {}
            extra = (f"; excluding repaired instances: P {f3(hx.get('precision'))}, R {f3(hx.get('recall'))}, "
                     f"F1 {f3(hx.get('f1'))}; n_instances {methods[m].get('n_instances')}, failed "
                     f"{methods[m].get('failed_instances')}, repaired {methods[m].get('repaired_instances')}")
        folds = methods[m].get("folds") or []
        thr = sorted({f.get("threshold") for f in folds if f.get("threshold") is not None})
        if thr:
            extra += f"; fold thresholds {thr}"
        NOTES[F].append(f"  - {m}: precision {f3(h.get('precision'))}, recall {f3(h.get('recall'))}, "
                        f"F1 {f3(h.get('f1'))} (tp {h.get('tp')}, fp {h.get('fp')}, fn {h.get('fn')}, "
                        f"tn {h.get('tn')}); ARI held out {f3(methods[m].get('ari_mean_held_out'))}{extra}")

    # (B) threshold sensitivity (in-sample curve) if present, else escalation validity.
    curve_methods = [m for m in ("jaccard", "tfidf") if methods.get(m, {}).get("curve")]
    if curve_methods:
        tone = {"jaccard": NEUTRAL, "tfidf": INK}
        name = {"jaccard": "Jaccard", "tfidf": "TF-IDF"}
        note(F, "**B** `methods.{jaccard,tfidf}.curve[].{threshold,precision,recall}` over "
                "`threshold_grid` on all instances (in-sample, not held out). Open circle = "
                "`methods.<m>.default_threshold` (uncalibrated threshold) with `at_default_threshold`; dotted vertical line = calibrated default 0.1; filled "
                "circle = the threshold selected most often in the leave-one-scenario-out folds "
                "(`methods.<m>.folds[].threshold`), drawn at its all-data recall. Curve points with recall > 0:")
        max_t = 0.0
        for m in curve_methods:
            meth = methods[m]
            cv = meth["curve"]
            t = np.array([p["threshold"] for p in cv])
            pr = np.array([nan_if_none(p.get("precision")) for p in cv])
            rc = np.array([nan_if_none(p.get("recall")) for p in cv])
            max_t = max(max_t, max((p["threshold"] for p in cv if (p.get("recall") or 0) > 0), default=0))
            axB.plot(t, pr, color=tone[m], lw=1.0, ls=(0, (1.2, 1.2)), zorder=3)
            axB.plot(t, rc, color=tone[m], lw=1.3, ls="-", label=f"{name[m]} recall", zorder=3)
            # Fold-selected threshold (modal) and the uncalibrated threshold, marked on the recall curve.
            thr = [f["threshold"] for f in meth.get("folds", []) if f.get("threshold") is not None]
            if thr:
                vals, counts = np.unique(np.round(thr, 4), return_counts=True)
                t_sel = float(vals[np.argmax(counts)])
                r_sel = rc[np.argmin(np.abs(t - t_sel))]
                axB.plot([t_sel], [r_sel], ls="none", marker="o", ms=4.2, color=tone[m], mec=WHITE,
                         mew=0.5, zorder=5)
            # (CV-selected thresholds are labelled 'modal CV threshold' in the legend.)
            dft = meth.get("default_threshold")
            at_d = meth.get("at_default_threshold") or {}
            if dft is not None and at_d.get("recall") is not None:
                axB.plot([dft], [at_d["recall"]], ls="none", marker="o", ms=4.6, mfc=WHITE, mec=tone[m],
                         mew=1.0, zorder=5)
                low = at_d["recall"] < 0.2
                axB.annotate(f"{name[m]} uncalibrated {dft:g}:\nrecall {at_d['recall']:.2f}",
                             (dft, at_d["recall"]), xytext=(34, 30) if low else (-2, 4),
                             textcoords="offset points", ha="left", va="bottom", fontsize=7, color=INK,
                             arrowprops=dict(arrowstyle="-", color=NEUTRAL, lw=0.6, shrinkA=1, shrinkB=3)
                             if low else None)
            NOTES[F].append(f"  - {m}: default threshold {dft} -> precision {f3(at_d.get('precision'))}, "
                            f"recall {f3(at_d.get('recall'))}, F1 {f3(at_d.get('f1'))}; fold-selected thresholds "
                            f"{dict(zip(*np.unique(np.round(thr, 4), return_counts=True)))} ; "
                            f"all-data best threshold {meth.get('all_data_best_threshold')} (F1 "
                            f"{f3((meth.get('all_data_best') or {}).get('f1'))})")
            NOTES[F].append(f"    curve: " + "; ".join(
                f"t={p['threshold']:.3f} P={f3(p.get('precision'), 3)} R={f3(p.get('recall'), 3)}"
                for p in cv if (p.get("recall") or 0) > 0))
        pmin = min(nan_if_none(p.get("precision")) for m in curve_methods for p in methods[m]["curve"]
                   if p.get("precision") is not None)
        axB.text(0.98, pmin - 0.03, f"precision ≥ {np.floor(pmin * 1000) / 1000:.3f} at every threshold",
                 transform=axB.get_yaxis_transform(), ha="right", va="top", fontsize=7, color=INK_2)
        axB.axvline(0.1, color=NEUTRAL, lw=0.8, ls=(0, (1, 2)), zorder=1)
        axB.set_xlim(0, min(1.0, round(max_t + 0.075, 2)))
        axB.set_ylim(0, 1.06)
        axB.set_yticks(np.arange(0, 1.01, 0.2))
        axB.set_xlabel("Similarity threshold")
        axB.set_ylabel("Pairwise score (all instances)")
        axB.set_title("Threshold sensitivity, in-sample")
        value_grid(axB)
        from matplotlib.lines import Line2D
        mk = [Line2D([], [], color=INK_2, lw=1.0, ls=(0, (1.2, 1.2)), label="precision (both)"),
              Line2D([], [], ls="none", marker="o", ms=4.2, color=INK_2, mec=WHITE, mew=0.5,
                     label="modal CV threshold"),
              Line2D([], [], ls="none", marker="o", ms=4.6, mfc=WHITE, mec=INK_2, mew=1.0,
                     label="uncalibrated threshold"),
              Line2D([], [], color=NEUTRAL, lw=0.8, ls=(0, (1, 2)), label="platform default 0.1")]
        h, l = axB.get_legend_handles_labels()
        axB.legend(handles=h + mk, loc="upper right", bbox_to_anchor=(1.0, 0.89), ncol=1)
        # Escalation validity is not plotted (one panel only), but recorded for the text.
        for m in order:
            e = methods[m].get("escalation_held_out_flawed") or {}
            mu, si = e.get("multi_lens_groups") or {}, e.get("single_lens_groups") or {}
            NOTES[F].append(f"  - (not plotted) escalation, held out, flawed items, {m}: multi-lens groups "
                            f"n {mu.get('n')} share matching a flaw {f3(mu.get('share_any_flaw'))}; single-lens "
                            f"n {si.get('n')} share {f3(si.get('share_any_flaw'))}; flaws raised by >=2 lenses "
                            f"{e.get('flaws_raised_by_2plus_lenses')}, escalated {e.get('of_which_escalated')} "
                            f"(escalation recall {f3(e.get('escalation_recall'))})")
    else:
        # Fallback: escalation validity (share of multi- vs single-lens groups that match a flaw).
        cats = []
        for m in order:
            e = methods[m].get("escalation_held_out_flawed") or {}
            cats.append((m, e))
        xb = np.arange(len(cats))
        for k, (grp, tone) in enumerate([("multi_lens_groups", INK), ("single_lens_groups", NEUTRAL)]):
            vals = [nan_if_none((e.get(grp) or {}).get("share_any_flaw")) for _, e in cats]
            axB.bar(xb + (k - 0.5) * 0.3, np.nan_to_num(vals), width=0.26, color=tone,
                    label=grp.replace("_", " "), zorder=2)
        axB.set_xticks(xb, [MERGE_LABEL[m] for m, _ in cats])
        axB.set_ylim(0, 1.05)
        axB.set_ylabel("Share of groups matching a planted flaw")
        axB.set_title("Escalation validity, held out")
        value_grid(axB)
        axB.legend(loc="upper right")
        note(F, "**B** `methods.<m>.escalation_held_out_flawed.{multi,single}_lens_groups.share_any_flaw`.")
    save_figure(fig, "Fig3", out_dir, [axA, axB])
    return True


# ── Fig 4: stopping calibration ─────────────────────────────────────────

RULE_SHORT = {
    "default_any_of_3": "any of 3 signals",
    "grade_stable_only": "grade stable only",
    "no_critical_high_only": "no critical/high only",
    "output_similar_only": "output similar only",
    "all_of_3": "all 3 signals",
    "noCH_and_(grade_stable_or_similar)": "no crit/high AND (grade stable OR similar)",
    "noCH_and_(grade_stable_or_similar)+gate": "same + consensus gate",
}


def stopping_complete_configs(sj: dict) -> tuple[list[str], dict[str, str]]:
    """Configs whose trajectories are all judged and whose replay covers k = 0..horizon."""
    done, why = [], {}
    horizon = sj.get("horizon", 5)
    for name, c in (sj.get("configs") or {}).items():
        n = c.get("n_trajectories") or 0
        ks = sorted(r["k"] for r in c.get("per_revision", []))
        if n == 0 or n != len(c.get("items", [])) or c.get("n_fully_judged") != n:
            why[name] = f"{c.get('n_fully_judged')}/{n} trajectories judged"
        elif ks != list(range(horizon + 1)) or any(r.get("n_judged") != r.get("n")
                                                   for r in c["per_revision"]):
            why[name] = "per-revision replay incomplete"
        elif not c.get("rules"):
            why[name] = "no rule replay"
        else:
            done.append(name)
    return done, why


def fig4_stopping(plt, out_dir: Path) -> bool:
    """Stopping calibration: per-revision trajectories for each configuration and the
    candidate rules pooled over configurations (premature stops summed over runs)."""
    sj = load_json(STOP / "stopping_results.json")
    if sj is None:
        print("Fig4: skipped (stopping_results.json missing)")
        return False
    configs, why = stopping_complete_configs(sj)
    for name, reason in why.items():
        print(f"Fig4: config {name} not plotted ({reason})")
    if not configs:
        print("Fig4: skipped (no complete config)")
        return False
    F = "Fig4"
    horizon = sj.get("horizon", 5)
    note(F, f"stopping_results.json generated {sj.get('generated_at')}, judge {sj.get('judge_model')}, "
            f"horizon {horizon} executor revisions. Configs plotted: {configs}"
            + (f"; not plotted: {why}" if why else "") + ".")
    note(F, f"Definitions (JSON `definitions`): premature stop = {sj['definitions']['premature_stop']}; "
            f"cycles = {sj['definitions']['cycles']}.")
    fig, axes = plt.subplots(1, 4, figsize=(FIG_WIDTH, 2.6), layout="constrained",
                             gridspec_kw={"width_ratios": [1.0, 1.0, 1.0, 1.25]})
    axA, axB, axC, axD = axes
    models = []
    for name in configs:
        c = sj["configs"][name]
        m = c["model"]
        models.append(m)
        st = CONFIGS[m]
        pr = sorted(c["per_revision"], key=lambda r: r["k"])
        k = np.array([r["k"] for r in pr])
        series = {
            axA: np.array([r["mean_unresolved"] for r in pr]),
            axB: np.array([r["mean_new_errors"] for r in pr]),
            axC: np.array([r["mean_critical_high"] for r in pr]),
        }
        for ax, y in series.items():
            ax.plot(k, y, color=st["color"], lw=1.2, marker=st["marker"], ms=4.4, mec=WHITE,
                    mew=0.5, label=st["label"], zorder=3)
        note(F, f"**A** `configs.{name}.per_revision[k].mean_unresolved`: "
                + ", ".join(f"k{int(a)} {b:.3f}" for a, b in zip(k, series[axA]))
                + "; frac_all_resolved: " + ", ".join(f"k{r['k']} {r['frac_all_resolved']:.3f}" for r in pr))
        note(F, f"**B** `configs.{name}.per_revision[k].mean_new_errors`: "
                + ", ".join(f"k{int(a)} {b:.2f}" for a, b in zip(k, series[axB])))
        note(F, f"**C** `configs.{name}.per_revision[k].mean_critical_high`: "
                + ", ".join(f"k{int(a)} {b:.2f}" for a, b in zip(k, series[axC])))
    for ax in (axA, axB, axC):
        ax.set_xticks(range(horizon + 1))
        ax.set_xlim(-0.3, horizon + 0.3)
        ax.set_xlabel("Revision k")
        value_grid(ax)
    axA.set_ylim(0, 4.3)
    axA.set_yticks(range(5))
    axA.set_ylabel("Unresolved planted flaws per item")
    axA.set_title("Planted flaws left")
    axB.set_ylim(0, None)
    axB.set_ylabel("New errors per item")
    axB.set_title("Errors introduced")
    axC.set_ylim(0, None)
    axC.set_ylabel("Critical + high per review")
    axC.set_title("Panel severity")

    # (D) rules pooled over configurations.
    pooled: dict[str, dict] = {}
    for name in configs:
        for r in sj["configs"][name]["rules"]:
            if r["family"] not in ("primary", "fixed"):
                continue
            d = pooled.setdefault(r["name"], {"family": r["family"], "prem": 0, "n": 0, "cyc": 0.0,
                                              "early": 0.0})
            d["prem"] += r["n_premature"]
            d["n"] += r["n_judged"]
            d["cyc"] += r["mean_cycles"] * r["n_judged"]
            d["early"] += r["stop_before_horizon_rate"] * r["n_judged"]
    for d in pooled.values():
        d["rate"] = d["prem"] / d["n"]
        d["mcyc"] = d["cyc"] / d["n"]
        d["early_rate"] = d["early"] / d["n"]
    fixed = sorted([(n, d) for n, d in pooled.items() if d["family"] == "fixed"], key=lambda x: x[1]["mcyc"])
    adaptive = [(n, d) for n, d in pooled.items() if d["family"] == "primary"]
    axD.plot([d["mcyc"] for _, d in fixed], [d["rate"] for _, d in fixed], color=INK, lw=0.8,
             ls=(0, (2, 1.5)), zorder=2)
    for n, d in fixed:
        kk = int(n.rsplit("k", 1)[-1])
        sel = kk == 4
        axD.plot([d["mcyc"]], [d["rate"]], ls="none", marker="o", ms=6.0 if sel else 4.8,
                 mfc=INK if sel else WHITE, mec=INK, mew=1.0, zorder=3)
        lab = f"budget {kk}" + (" (default)" if sel else "")
        axD.annotate(lab, (d["mcyc"], d["rate"]), xytext=(7, 3) if kk != 5 else (-6, 6),
                     textcoords="offset points", ha="left" if kk != 5 else "right", va="bottom",
                     fontsize=7, color=INK)
    never = [(n, d) for n, d in adaptive if d["early_rate"] == 0]
    fired = [(n, d) for n, d in adaptive if d["early_rate"] > 0]
    for n, d in fired:
        axD.plot([d["mcyc"]], [d["rate"]], ls="none", marker="D", ms=4.8, color=NEUTRAL, mec=WHITE,
                 mew=0.5, zorder=4)
    if fired:
        d0 = fired[0][1]
        axD.annotate("grade stable\n(= any signal)", (d0["mcyc"], d0["rate"]), xytext=(-28, -34),
                     textcoords="offset points", ha="right", va="top", fontsize=7, color=INK,
                     arrowprops=dict(arrowstyle="-", color=NEUTRAL, lw=0.6, shrinkA=0, shrinkB=3))
    if never:
        axD.text(0.97, 0.97, f"{len(never)} rules needing no critical/high\nor stable output never stopped\nwithin {horizon} revisions",
                 transform=axD.transAxes, ha="right", va="top", fontsize=7, color=INK_2)
    axD.set_xlim(0.5, horizon + 0.5)
    axD.set_xticks(range(1, horizon + 1))
    axD.set_ylim(-0.01, 0.33)
    axD.set_xlabel("Mean revisions used")
    axD.set_ylabel("Premature-stop rate (pooled)")
    axD.set_title("Stopping rules")
    value_grid(axD)
    note(F, "**D** rules pooled over configurations: premature = sum of `n_premature` / sum of `n_judged`; "
            "mean revisions = run-weighted mean of `mean_cycles`. Filled circle = selected fixed budget (k=4).")
    for n, d in sorted(pooled.items(), key=lambda x: (x[1]["family"], x[1]["mcyc"])):
        NOTES[F].append(f"  - {n} ({d['family']}): premature {d['prem']}/{d['n']} = {d['rate']:.3f}; mean "
                        f"revisions {d['mcyc']:.3f}; stopped before horizon {d['early_rate']:.3f}")
    fig.legend(handles=config_handles(models, include_pooled=False), loc="outside upper center",
               ncol=len(models))
    save_figure(fig, "Fig4", out_dir, [axA, axB, axC, axD])
    return True


# ── Fig 5: case-study independent validation ────────────────────────────


def fig5_case(plt, out_dir: Path) -> bool:
    d = load_json(CASE / "independent_validation.json")
    if d is None:
        print("Fig5: skipped (independent_validation.json missing)")
        return False
    a = d["a_asymmetry"]
    t = d["c_edge_recovery"]["trrust"]
    F = "Fig5"
    meta = d["meta"]
    note(F, f"independent_validation.json: {meta['n_cells']} cells, {meta['n_genes']} genes, "
            f"{meta['n_layers']} layers x {meta['n_heads']} heads; {meta['orientation']}; primary score: "
            f"{meta['primary_score']}. JSON `layer` is 0-based; the figure labels layers 1-12 "
            f"(layer + 1), matching SUMMARY.md and the earlier case-study figure (best sym layer = JSON "
            f"layer 2 = 'L3').")

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(FIG_WIDTH, 2.8), layout="constrained")
    # (A) asymmetry: head-mean Spearman per layer + per-head strip.
    lay = np.array([r["layer"] for r in a["per_layer_head_mean"]]) + 1
    sp = np.array([r["spearman"] for r in a["per_layer_head_mean"]])
    ph = np.array(a["per_head"]["spearman"])            # [layer][head]
    n_heads = ph.shape[1]
    spread = np.linspace(-0.3, 0.3, n_heads)
    for li in range(ph.shape[0]):
        axA.plot(li + 1 + spread, ph[li], ls="none", marker="o", ms=1.8, color=NEUTRAL_LIGHT,
                 mec="none", zorder=2)
    axA.plot(lay, sp, color=INK, lw=1.2, marker="o", ms=3.8, mec=WHITE, mew=0.5, zorder=3)
    axA.axhline(1.0, color=INK_2, lw=0.7, ls=(0, (4, 2)), zorder=1)
    axA.text(12.4, 0.985, "perfect symmetry", ha="right", va="top", fontsize=7, color=INK_2)
    axA.set_ylim(-0.3, 1.06)
    axA.set_ylabel(r"Spearman($A_{ij}$, $A_{ji}$)")
    axA.set_title("Attention asymmetry by layer")
    from matplotlib.lines import Line2D
    axA.legend(handles=[Line2D([], [], color=INK, marker="o", ms=3.8, mec=WHITE, mew=0.5,
                               label="head-mean attention"),
                        Line2D([], [], ls="none", marker="o", ms=2.5, color=NEUTRAL_LIGHT, mec="none",
                               label=f"individual heads ({n_heads}/layer)")],
               loc="upper left", bbox_to_anchor=(0.0, 0.9))
    note(F, f"**A** line = `a_asymmetry.per_layer_head_mean[l].spearman` (Spearman between A_ij and A_ji "
            f"over {a['n_pairs']} gene pairs, {a['pair_filter']}, on head-mean attention); grey dots = "
            f"`a_asymmetry.per_head.spearman[l][h]` (spread horizontally by head index, not jittered). "
            f"Head-mean per layer 1-12: " + ", ".join(f"{v:.3f}" for v in sp)
         + f". Per-head range {ph.min():.3f} to {ph.max():.3f}, median "
           f"{a['per_head_summary']['spearman']['median']:.3f}. Dashed line at 1 = perfect symmetry. "
           f"Verdict 'attention is approximately symmetric': {d['verdicts']['attention is approximately symmetric']['verdict']}.")

    # (B) TRRUST edge recovery: sym AUROC with TF-bootstrap CI, degree-null mean + 95% band, co-expr line.
    sym = sorted(t["per_layer"]["sym"], key=lambda r: r["layer"])
    null = {r["layer"]: r for r in t["permutation_null"]["layers"]["sym"]}
    L = np.array([r["layer"] for r in sym]) + 1
    au = np.array([r["auroc"] for r in sym])
    lo = np.array([r["auroc_ci95"][0] for r in sym])
    hi = np.array([r["auroc_ci95"][1] for r in sym])
    nm = np.array([null[r["layer"]]["null_mean"] for r in sym])
    nlo = np.array([null[r["layer"]]["null_ci95"][0] for r in sym])
    nhi = np.array([null[r["layer"]]["null_ci95"][1] for r in sym])
    coexpr_key = t.get("best_coexpression_baseline", "abs_spearman")
    coexpr = t["baselines"]["abs_spearman"]["auroc"]
    axB.fill_between(L, nlo, nhi, color=BAND, lw=0, zorder=1)
    axB.plot(L, nm, color=NEUTRAL, lw=1.0, zorder=2)
    axB.axhline(coexpr, color=INK_2, lw=0.8, ls=(0, (4, 2)), zorder=2)
    axB.errorbar(L, au, yerr=[au - lo, hi - au], fmt="none", ecolor=INK, elinewidth=0.7, zorder=3)
    axB.plot(L, au, color=INK, lw=1.2, marker="o", ms=3.8, mec=WHITE, mew=0.5, zorder=4)
    axB.set_ylabel("TRRUST AUROC")
    axB.set_title("TRRUST edge recovery")
    from matplotlib.patches import Patch
    null_handle = (Patch(facecolor=BAND, edgecolor="none"), Line2D([], [], color=NEUTRAL, lw=1.0))
    hB = [Line2D([], [], color=INK, marker="o", ms=3.8, mec=WHITE, mew=0.5),
          null_handle, Line2D([], [], color=INK_2, lw=0.8, ls=(0, (4, 2)))]
    axB.legend(hB, ["symmetrized attention (95% CI)", "degree-preserving null",
                    "|Spearman| co-expression"], loc="upper right", handlelength=1.8)
    note(F, f"**B** black = `c_edge_recovery.trrust.per_layer.sym[l].auroc` with `auroc_ci95` (TF-level "
            f"bootstrap, {meta['n_boot_tf']} resamples); grey line/band = "
            f"`c_edge_recovery.trrust.permutation_null.layers.sym[l].null_mean` / `null_ci95` "
            f"(curveball rewiring preserving TF out- and target in-degree, {meta['n_perm']} permutations); "
            f"dashed = `c_edge_recovery.trrust.baselines.abs_spearman.auroc` = {coexpr:.4f} "
            f"(`best_coexpression_baseline` = {coexpr_key}; |Pearson| = "
            f"{t['baselines']['abs_pearson']['auroc']:.4f}). Candidates: "
            f"{t['candidates']['n_candidate_pairs']} TF-gene pairs, {t['candidates']['n_positive']} positives.")
    for i, r in enumerate(sym):
        n = null[r["layer"]]
        NOTES[F].append(f"  - layer {r['layer'] + 1}: AUROC {r['auroc']:.4f} [{lo[i]:.4f}, {hi[i]:.4f}]; "
                        f"null mean {n['null_mean']:.4f} [{n['null_ci95'][0]:.4f}, {n['null_ci95'][1]:.4f}]; "
                        f"excess {n['excess_over_null']:.4f}; p_BH(layers) {n['p_bh_layers']:.5f}")
    bl = t["permutation_null"]["best_layer"]
    note(F, f"  - best layer (`permutation_null.best_layer`): {bl['variant']} layer {bl['layer'] + 1}, AUROC "
            f"{bl['auroc']:.4f}, null mean {bl['null_mean']:.4f}, max-over-layers p "
            f"{bl['p_empirical_max_over_layers']:.4f}.")

    # (C) post hoc: pair-specific part residualized on covariates, vs its degree-null mean.
    mp = t["marginal_vs_pair_specific_post_hoc"]["interaction_residualised"]["sym"]
    mp = sorted(mp, key=lambda r: r["layer"])
    Lc = np.array([r["layer"] for r in mp]) + 1
    ac = np.array([r["auroc"] for r in mp])
    clo = np.array([r["auroc_ci95"][0] for r in mp])
    chi = np.array([r["auroc_ci95"][1] for r in mp])
    cnm = np.array([r["null_mean"] for r in mp])
    has_band = all("null_ci95" in r for r in mp)
    if has_band:
        axC.fill_between(Lc, [r["null_ci95"][0] for r in mp], [r["null_ci95"][1] for r in mp],
                         color=BAND, lw=0, zorder=1)
    axC.plot(Lc, cnm, color=NEUTRAL, lw=1.0, zorder=2)
    axC.axhline(0.5, color=INK_2, lw=0.6, ls=(0, (1, 1.5)), zorder=1)
    axC.errorbar(Lc, ac, yerr=[ac - clo, chi - ac], fmt="none", ecolor=INK, elinewidth=0.7, zorder=3)
    axC.plot(Lc, ac, color=INK, lw=1.2, marker="o", ms=3.8, mec=WHITE, mew=0.5, zorder=4)
    n_bh = sum(1 for r in mp if r["p_bh_36"] < 0.05)
    axC.text(0.97, 0.68, f"BH p < 0.05 in {n_bh}/{len(mp)} layers\n(36 layer × variant tests)",
             transform=axC.transAxes, ha="right", va="top", fontsize=7, color=INK_2)
    axC.text(0.6, 0.5, "chance", ha="left", va="bottom", fontsize=7, color=INK_2)
    axC.set_ylabel("TRRUST AUROC, residual")
    axC.set_title("Pair-specific, residualized (post hoc)")
    axC.legend([Line2D([], [], color=INK, marker="o", ms=3.8, mec=WHITE, mew=0.5), null_handle],
               ["pair-specific residual (95% CI)", "degree-preserving null"], loc="upper left",
               handlelength=1.8)
    covs = t["residualised"]["covariates"]
    note(F, "**C** (post hoc; added after inspecting the main results) black = "
            "`c_edge_recovery.trrust.marginal_vs_pair_specific_post_hoc.interaction_residualised.sym[l].auroc` "
            "with `auroc_ci95` (pair-specific interaction part of log10 head-mean attention, row and column "
            f"effects removed, then OLS-residualized on {covs}); grey line/band = `null_mean` / `null_ci95` "
            f"(degree-preserving null); dotted line = 0.5. BH over 36 layer x variant tests (`p_bh_36`): "
            f"{n_bh}/{len(mp)} sym layers < 0.05.")
    for r in mp:
        NOTES[F].append(f"  - layer {r['layer'] + 1}: AUROC {r['auroc']:.4f} [{r['auroc_ci95'][0]:.4f}, "
                        f"{r['auroc_ci95'][1]:.4f}]; null mean {r['null_mean']:.4f}; p_emp {r['p_empirical']:.5f}; "
                        f"p_BH36 {r['p_bh_36']:.5f}")

    # Shared x styling; B and C share a y range so the two AUROC scales compare directly.
    ylo = min(lo.min(), nlo.min(), clo.min(), coexpr) - 0.02
    yhi = max(hi.max(), nhi.max(), chi.max()) + 0.065   # headroom for the legends
    for ax in (axB, axC):
        ax.set_ylim(np.floor(ylo * 50) / 50, np.ceil(yhi * 50) / 50)
    for ax in (axA, axB, axC):
        ax.set_xticks(range(1, 13))
        ax.set_xlim(0.4, 12.6)
        ax.set_xlabel("Layer")
        value_grid(ax)
    v = d["verdicts"]["attention recovers TRRUST edges above co-expression"]
    note(F, "Caution when quoting layer labels from `verdicts`: `variant_layers_above_coexpression_bh`, "
            "`variant_layers_residual_ci_above_0.5` and `variant_layers_beating_degree_null_bh` use 0-based "
            f"labels (e.g. {v['variant_layers_beating_degree_null_bh'][:3]}), whereas "
            "`post_hoc_interaction_only_beats_degree_null_bh`, `residual_beats_degree_null_bh`, "
            "`post_hoc_degree_matched_variant_layers_bh`, `post_hoc_combined_beyond_coexpr_rank_degree_bh` and "
            "the `post_hoc` texts / SUMMARY.md use 1-based labels (layer + 1), as does this figure.")
    save_figure(fig, "Fig5", out_dir, [axA, axB, axC])
    return True


# ── S1 Fig: orchestration overhead ──────────────────────────────────────

STORAGE_LABEL = {"internal_apfs": "internal SSD (APFS)", "external_exfat": "external SSD (exFAT)"}


def s1_orchestration(plt, out_dir: Path) -> bool:
    ov = load_json(ORCH / "overhead.json")
    rc = load_json(ORCH / "real_concurrency.json")
    if ov is None and rc is None:
        print("S1_Fig: skipped (overhead.json and real_concurrency.json missing)")
        return False
    F = "S1_Fig"
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(FIG_WIDTH, 2.9), layout="constrained",
                                   gridspec_kw={"width_ratios": [1.25, 1.0]})
    from matplotlib.lines import Line2D
    letter_axes = []
    if ov is not None:
        results = ov.get("results") or [ov]
        storages = [r.get("storage", {}).get("label", "default") for r in results]
        tone = {s: tc for s, tc in zip(storages, [INK, NEUTRAL, INK_2])}
        marker = {s: mk for s, mk in zip(storages, ["o", "v", "p"])}
        note(F, "**A** overhead.json `results[storage].sweeps[delay].rows[c].overhead_ms_per_iteration` "
                f"({results[0]['definitions']['overhead_ms_per_iteration']}); line/marker = p50, whisker = p95 "
                f"(nearest rank over all runs of {results[0]['repeats']} repeats). Mock adapter, "
                f"{results[0]['iterations_per_run']} iterations per run, preset {results[0]['preset']}; "
                f"integrity_ok = {ov.get('integrity_ok')}.")
        for res, s in zip(results, storages):
            for sw in res["sweeps"]:
                delay = sw["adapter_delay_s"]
                rows = sw["rows"]
                cx = np.array([r["concurrency"] for r in rows])
                p50 = np.array([r["overhead_ms_per_iteration"]["p50"] for r in rows])
                p95 = np.array([r["overhead_ms_per_iteration"]["p95"] for r in rows])
                dash = "-" if delay == 0 else (0, (3.5, 2))
                filled = delay == 0
                lab = f"{STORAGE_LABEL.get(s, s)}, mock delay {delay:g} s"
                axA.errorbar(cx, p50, yerr=[np.zeros_like(p50), p95 - p50], fmt="none",
                             ecolor=tone[s], elinewidth=0.7, zorder=2)
                axA.plot(cx, p50, color=tone[s], ls=dash, lw=1.1, marker=marker[s], ms=4,
                         mfc=tone[s] if filled else WHITE, mec=tone[s], mew=0.9, label=lab, zorder=3)
                load = [r.get("loadavg_start", [None])[0] for r in rows]
                NOTES[F].append(f"  - {s}, delay {delay:g} s: " + ", ".join(
                    f"c{c} p50 {a:.1f} / p95 {b:.1f} ms" for c, a, b in zip(cx, p50, p95))
                    + f" (1-min load average at level start: {load})")
        lv = sorted({r["concurrency"] for res in results for sw in res["sweeps"] for r in sw["rows"]})
        axA.set_xscale("log")
        axA.set_xticks(lv, [str(v) for v in lv])
        axA.minorticks_off()
        axA.set_xlim(lv[0] / 1.25, lv[-1] * 1.25)
        axA.set_yscale("log")
        allv = [r["overhead_ms_per_iteration"][q] for res in results for sw in res["sweeps"]
                for r in sw["rows"] for q in ("p50", "p95")]
        ticks = [t_ for t_ in (10, 20, 50, 100, 200, 500, 1000, 2000, 5000) if min(allv) / 1.6 <= t_ <= max(allv) * 3]
        axA.set_yticks(ticks, [f"{t_:,}" for t_ in ticks])
        axA.yaxis.set_minor_locator(__import__("matplotlib").ticker.NullLocator())
        axA.set_ylim(min(allv) / 1.6, max(allv) * 3.2)
        axA.set_xlabel("Concurrent runs")
        axA.set_ylabel("Control-plane overhead (ms/iteration, log)")
        axA.set_title("Orchestration overhead (mock backend)")
        value_grid(axA)
        axA.legend(loc="upper left", handlelength=2.4)
        axA.text(0.98, 0.04, "marker: median; whisker: p95", transform=axA.transAxes, ha="right",
                 va="bottom", fontsize=7, color=INK_2)
        letter_axes.append(axA)
    else:
        axA.set_visible(False)
    if rc is not None:
        summ = sorted(rc.get("summary_by_level", []), key=lambda s: s["k"])
        k = np.array([s["k"] for s in summ])
        thr = np.array([s["throughput_successful_calls_per_min"] for s in summ])
        model = rc.get("config", {}).get("model", "gpt-5.6-luna")
        effort = rc.get("config", {}).get("effort", "")
        st = CONFIGS.get(model, {"label": f"{model} {effort}", "color": INK, "marker": "o"})
        base = thr[k == 1][0] if (k == 1).any() else thr[0] / k[0]
        kk = np.array([0, k.max() * 1.06])
        axB.plot(kk, base * kk, color=NEUTRAL, lw=0.9, ls=(0, (4, 2)), zorder=1,
                 label=f"linear scaling ({base:.2f} × panels)")
        axB.plot(k, thr, color=st["color"], lw=1.2, marker=st["marker"], ms=4.8, mec=WHITE, mew=0.5,
                 label=f"{st['label']} (Codex CLI)", zorder=3)
        for kv, tv in zip(k, thr):
            axB.annotate(f"{tv:.1f}", (kv, tv), xytext=(5, -3), textcoords="offset points",
                         ha="left", va="top", fontsize=7, color=INK_2)
        axB.set_xticks(k)
        axB.set_xlim(0, k.max() * 1.1)
        axB.set_ylim(0, max(base * k.max(), thr.max()) * 1.12)
        axB.set_xlabel("Concurrent reviewer panels")
        axB.set_ylabel("Successful reviewer calls per minute")
        axB.set_title("Real-backend throughput")
        value_grid(axB)
        axB.legend(loc="upper left")
        note(F, f"**B** real_concurrency.json `summary_by_level[k].throughput_successful_calls_per_min` "
                f"({model}, effort {effort}, provider {rc.get('config', {}).get('provider')}, panel of "
                f"{len(rc.get('config', {}).get('panel', []))} lenses; levels {list(k)}); dashed = linear "
                f"scaling from k = 1 ({base:.3f} calls/min x k). Values labelled on the points.")
        for s in summ:
            NOTES[F].append(
                f"  - k={s['k']}: {s['throughput_successful_calls_per_min']:.3f} calls/min "
                f"(linear {base * s['k']:.3f}; {100 * s['throughput_successful_calls_per_min'] / (base * s['k']):.0f}% "
                f"of linear); panels/min {s['throughput_panels_per_min']:.3f}; wall {s['wall_seconds']:.1f} s; "
                f"call latency p50 {s['call_latency_s_successful']['p50']:.1f} s; retries {s['retries']}, "
                f"failed attempts {s['failed_attempts']}, panels complete {s['panels_complete']}/{s['panels']}")
        env = rc.get("environment", {})
        note(F, f"  - host 1-min load average start/end: {env.get('loadavg_start', [None])[0]} / "
                f"{env.get('loadavg_end', [None])[0]}.")
        letter_axes.append(axB)
    else:
        axB.set_visible(False)
    save_figure(fig, "S1_Fig", out_dir, letter_axes)
    return True


# ── notes + main ────────────────────────────────────────────────────────

FIGURES: dict[str, Callable] = {
    "Fig2": fig2_benchmark,
    "Fig3": fig3_merge,
    "Fig4": fig4_stopping,
    "Fig5": fig5_case,
    "S1_Fig": s1_orchestration,
}
TITLES = {
    "Fig2": "Fig 2. Planted-flaw benchmark",
    "Fig3": "Fig 3. Critique-merge evaluation",
    "Fig4": "Fig 4. Stopping-rule calibration",
    "Fig5": "Fig 5. Case study: independent validation",
    "S1_Fig": "S1 Fig. Orchestration overhead and real-backend throughput",
}


def write_notes(out_dir: Path, built: dict[str, bool]) -> None:
    from datetime import datetime, timezone
    lines = ["# Figure notes (PLOS ONE revision)", "",
             f"Generated by `scripts/make_plos_figures.py` on "
             f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}. Every number below is the value "
             "drawn in the figure, read from the listed JSON key / CSV rows. Re-running the script "
             "regenerates this file.", "",
             "Common encoding: gpt-5.6-sol medium = blue circle, gpt-5.5 medium = orange square, "
             "gpt-5.6-luna low = aqua triangle, pooled = black diamond. Files per figure: `<name>.tif` "
             "(RGB, LZW, 600 dpi, 7.5 in wide), `<name>.pdf`, `<name>_preview.png` (200 dpi).", ""]
    for key, title in TITLES.items():
        lines.append(f"## {title} (`{key}.tif`)")
        lines.append("")
        if not built.get(key):
            lines.append("Not built in this run (inputs missing or incomplete; see the script output).")
            lines.append("")
            continue
        lines += NOTES.get(key, [])
        lines.append("")
    (out_dir / "FIGURE_NOTES.md").write_text("\n".join(lines))
    print(f"wrote {rel(out_dir / 'FIGURE_NOTES.md')}")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", default="", help="comma-separated subset, e.g. Fig2,Fig5")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    want = [s.strip() for s in args.only.split(",") if s.strip()] or list(FIGURES)
    unknown = [w for w in want if w not in FIGURES]
    if unknown:
        ap.error(f"unknown figure(s) {unknown}; choose from {list(FIGURES)}")
    setup_matplotlib()
    import matplotlib.pyplot as plt
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    built: dict[str, bool] = {}
    for name in want:
        print(f"{name}:")
        built[name] = bool(FIGURES[name](plt, out_dir))
    if set(want) == set(FIGURES):
        write_notes(out_dir, built)
    else:
        print("(FIGURE_NOTES.md is only rewritten when all figures are requested)")
    print("built: " + ", ".join(f"{k}={'yes' if v else 'no'}" for k, v in built.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
