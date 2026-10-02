"""Assemble paper/plos/mi_workbench_plos.tex from the section files.

Figure captions are inserted after the first paragraph that cites each figure (PLOS
requirement); tables after the first paragraph that cites them. British spellings are
normalised to American English. Usage: python assemble.py [--bbl]  (with --bbl the
bibliography is inlined from mi_workbench_plos.bbl, as PLOS requests for LaTeX files).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLOS = HERE.parent


def read(name: str) -> str:
    return (HERE / name).read_text().strip() + "\n"


def captions() -> dict[str, str]:
    text = read("captions.tex")
    parts = re.split(r"^%(FIG\d|END)%\s*$", text, flags=re.M)
    out = {}
    for i in range(1, len(parts) - 1, 2):
        if parts[i] != "END":
            out[parts[i].lower()] = parts[i + 1].strip() + "\n"
    return out


def insert_after_first_citation(body: str, label: str, block: str, kind: str = "Fig") -> str:
    """Insert block after the paragraph that first cites \\ref{label}."""
    paras = body.split("\n\n")
    for i, p in enumerate(paras):
        if f"\\ref{{{label}}}" in p:
            paras.insert(i + 1, block.strip())
            return "\n\n".join(paras)
    raise SystemExit(f"no citation of {label} found")


SPELLING = [
    (r"\brigour\b", "rigor"), (r"\bRigour\b", "Rigor"),
    (r"\bbehaviour", "behavior"), (r"\bfavour", "favor"), (r"\bcolour", "color"),
    (r"\blabelled\b", "labeled"), (r"\blabelling\b", "labeling"), (r"\bmodelled\b", "modeled"),
    (r"\blevelled\b", "leveled"), (r"\bjudgement", "judgment"), (r"\bartefact", "artifact"),
    (r"\banalyse\b", "analyze"), (r"\banalysed\b", "analyzed"), (r"\banalysing\b", "analyzing"),
    (r"\bemphasise", "emphasize"), (r"\bemphasising\b", "emphasizing"),
    (r"\bcatalogue\b", "catalog"), (r"\bprogramme", "program"),
]
ISE_STEMS = ["organis", "normalis", "tokenis", "recognis", "specialis", "centralis", "summaris", "optimis",
             "minimis", "maximis", "randomis", "characteris", "stabilis", "utilis", "residualis", "symmetris",
             "prioritis", "parameteris", "generalis", "visualis", "realis", "initialis", "standardis",
             "categoris", "penalis", "operationalis", "serialis", "decentralis", "apologis"]


def americanize(text: str) -> str:
    for pat, rep in SPELLING:
        text = re.sub(pat, rep, text)
    for stem in ISE_STEMS:
        z = stem[:-1] + "z"
        text = re.sub(rf"\b{stem}(e|ed|es|ing|ation|ations|er|ers)\b", lambda m, z=z: z + m.group(1), text)
        text = re.sub(rf"\b{stem.capitalize()}(e|ed|es|ing|ation|ations|er|ers)\b",
                      lambda m, z=z: z.capitalize() + m.group(1), text)
    return text


def main() -> None:
    cap = captions()
    results = []
    x1 = read("results_x1.tex")
    x1 = insert_after_first_citation(x1, "table_benchmark", read("table_benchmark.tex"))
    x1 = insert_after_first_citation(x1, "fig2", cap["fig2"])
    x2 = insert_after_first_citation(read("results_x2.tex"), "fig3", cap["fig3"])
    x3 = insert_after_first_citation(read("results_x3.tex"), "fig4", cap["fig4"])
    x5 = insert_after_first_citation(read("results_x5.tex"), "fig5", cap["fig5"])
    x4 = insert_after_first_citation(read("results_x4.tex"), "table_case", read("table_case.tex"))
    results = "\n".join([x1, x2, x3, "\\subsection*{Case study: does Geneformer attention encode regulation?}\n",
                         x5, x4, read("results_x6.tex")])
    methods = read("methods.tex") + "\n" + read("implementation.tex")
    tpl = read("_template.tex")
    out = (tpl.replace("%ABSTRACT%", read("abstract.tex"))
              .replace("%INTRO%", read("intro.tex"))
              .replace("%RELATED%", read("related.tex"))
              .replace("%METHODS%", methods)
              .replace("%RESULTS%", results)
              .replace("%DISCUSSION%", read("discussion.tex"))
              .replace("%CONCLUSION%", read("conclusion.tex"))
              .replace("%SI%", read("si.tex")))
    out = americanize(out)
    if "--bbl" in sys.argv:
        bbl = (PLOS / "mi_workbench_plos.bbl").read_text()
        # plos2015.bst quirks: doubled period after "et al." and a missing space before
        # "Available from" when a note precedes the URL.
        bbl = bbl.replace("et~al..", "et~al.")
        bbl = re.sub(r"(\S)\.Available from:", r"\1. Available from:", bbl)
        out = out.replace("\\bibliography{references}", bbl.strip())
    left = re.findall(r"%[A-Z]+%", out)
    if left:
        raise SystemExit(f"unfilled placeholders: {left}")
    (PLOS / "mi_workbench_plos.tex").write_text(out)
    print(f"wrote {PLOS / 'mi_workbench_plos.tex'} ({len(out.split())} words incl. markup)")


if __name__ == "__main__":
    main()
