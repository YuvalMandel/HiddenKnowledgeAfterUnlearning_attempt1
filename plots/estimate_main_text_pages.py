#!/usr/bin/env python3
"""Estimate main-text length against ICLR's 9-page limit.

Main text = everything the venue counts: title block through the END OF
DISCUSSION. Limitations sits in the appendix in this paper, and the AI-use,
Ethics and Reproducibility statements are explicitly exempt, so nothing after
Discussion is counted here.

The estimate is a model, not a compile -- there is no LaTeX toolchain locally.
Calibration: ~620 words/page of body prose at ICLR's 10pt on a 5.5x9in block;
figures ~0.36pg and tables ~0.18pg each including caption. LaTeX comment lines
are excluded (they do not render); inline reviewer notes ARE counted, because
they do render.

Usage: python plots/estimate_main_text_pages.py [path.tex]
"""
import re
import sys
from pathlib import Path

WPP, FIG, TAB = 620.0, 0.36, 0.18
LIMIT = 9.0

path = Path(sys.argv[1] if len(sys.argv) > 1
            else "overleaf_claims/iclr2027_conference.tex")
lines = path.read_text(encoding="utf-8").split("\n")

start = next(i for i, l in enumerate(lines) if l.startswith(r"\section{Introduction}"))
disc = next(i for i, l in enumerate(lines) if l.startswith(r"\section{Discussion}"))
# Discussion ends where the exempt statements begin. ICLR states explicitly that
# the AI-use, Ethics and Reproducibility statements do NOT count toward the page
# limit, and they are marked up as \subsection*/\subsubsection*, not \section.
end = next(i for i, l in enumerate(lines)
           if i > disc and (l.startswith(r"\subsection*")
                            or l.startswith(r"\subsubsection*")
                            or l.startswith(r"\section")))

body = [l for l in lines[start:end] if not l.strip().startswith("%")]
txt = "\n".join(body)

prose = re.sub(r"\\begin\{(figure|table)\}.*?\\end\{\1\}", "", txt, flags=re.S)
prose = re.sub(r"\\[a-zA-Z]+\*?", " ", prose)
prose = re.sub(r"[{}$\\&~]", " ", prose)
words = len(prose.split())
nf, nt = txt.count(r"\begin{figure}"), txt.count(r"\begin{table}")

# inline reviewer notes still render and therefore still bill
notes = re.findall(r"\((?:YM|Z|z)\s*:[^)]*\)", txt)
note_words = sum(len(n.split()) for n in notes)

est = words / WPP + nf * FIG + nt * TAB
clean = (words - note_words) / WPP + nf * FIG + nt * TAB

print(f"main text: source lines {start+1}..{end}  (Introduction .. end of Discussion)")
print(f"  prose words        : {words}   -> {words/WPP:.2f} pg")
print(f"  figures / tables   : {nf} / {nt}   -> {nf*FIG + nt*TAB:.2f} pg")
print(f"  abstract+title etc : ~0.55 pg (not in the span above)")
print()
print(f"  ESTIMATE incl. reviewer notes : {est + 0.55:.2f} pg")
print(f"  ESTIMATE once notes removed   : {clean + 0.55:.2f} pg")
print(f"  LIMIT                         : {LIMIT:.1f} pg")
over = clean + 0.55 - LIMIT
print(f"  -> {'OVER by %.2f pg' % over if over > 0 else 'within limit (%.2f pg spare)' % -over}")
if notes:
    print(f"\n  {len(notes)} inline reviewer notes still render, costing "
          f"~{note_words/WPP:.2f} pg")
print("\nper-section prose words:")
cur, cnt = "(front)", 0
for l in body:
    m = re.match(r"\\(?:sub)?section\*?\{(.*?)\}", l)
    if m:
        if cnt:
            print(f"   {cnt:>5}w  {cur}")
        cur = re.sub(r"\\[a-zA-Z]+|[{}$\\]", "", m.group(1))[:52]
        cnt = 0
    p = re.sub(r"\\[a-zA-Z]+\*?", " ", l)
    cnt += len(re.sub(r"[{}$\\&~]", " ", p).split())
print(f"   {cnt:>5}w  {cur}")
