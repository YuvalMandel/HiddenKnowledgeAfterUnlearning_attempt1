"""Offline structural checks on the ICLR tex (no LaTeX toolchain available)."""
import os
import re
import sys

path = os.path.abspath(sys.argv[1])
s = open(path, encoding="utf-8").read()
os.chdir(os.path.dirname(path))

print("braces      : %d open / %d close" % (s.count("{"), s.count("}")))

for env in ("figure", "table", "itemize", "tabular"):
    b = len(re.findall(r"\\begin\{%s\}" % env, s))
    e = len(re.findall(r"\\end\{%s\}" % env, s))
    flag = "" if b == e else "   <-- MISMATCH"
    print("%-12s: %d begin / %d end%s" % (env, b, e, flag))

imgs = sorted(set(re.findall(r"\{(imgs/[^}]+)\}", s)))
missing = [i for i in imgs if not os.path.exists(i)]
print("images      : %d referenced, %d missing" % (len(imgs), len(missing)))
for m in missing:
    print("   MISSING", m)

labels = set(re.findall(r"\\label\{([^}]+)\}", s))
refs = set(re.findall(r"\\ref\{([^}]+)\}", s))
dangling = sorted(refs - labels)
print("dangling refs:", dangling or "none")

# Two \label{x} with the same name compile with only a warning, and every
# \ref{x} silently resolves to whichever came last -- easy to introduce when
# moving a block to the appendix while reusing its label in the new text.
seen = {}
for i, line in enumerate(s.split("\n"), 1):
    for lab in re.findall(r"\\label\{([^}]+)\}", line):
        seen.setdefault(lab, []).append(i)
dupes = {k: v for k, v in seen.items() if len(v) > 1}
print("duplicate labels:", "none" if not dupes else "")
for k, v in sorted(dupes.items()):
    print("   DUPLICATE %-28s lines %s" % (k, v))

# A \label right after a float binds to that float's counter, not the section,
# so \ref would render a table/figure number.
orphan = re.findall(r"\\end\{(?:table|figure)\}\s*\n\s*\\label\{([^}]+)\}", s)
print("orphaned labels after floats:", orphan or "none")

# \section/\paragraph/floats inside \textcolor{}{...} break the compile.
wrapped = re.findall(r"\\textcolor\{[a-z]+\}\{\\(section|subsection|paragraph|begin)", s)
print("sectioning/float in textcolor:", wrapped or "none")

keys = set()
for grp in re.findall(r"\\cite[tp]?\{([^}]+)\}", s):
    keys.update(k.strip() for k in grp.split(","))
bib = open(path.replace(".tex", ".bib"), encoding="utf-8").read()
bibkeys = set(re.findall(r"@\w+\{([^,]+),", bib))
print("bad cite keys:", sorted(keys - bibkeys) or "none")
