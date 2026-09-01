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

# Paragraphs in this file are single reflowed lines, so a mid-line '%' comments
# out the REST OF THE PARAGRAPH. Putting a "% was: ..." note after a corrected
# number silently deletes the rest of the sentence.
swallowed = []
for n, line in enumerate(s.split("\n"), 1):
    j = None
    for m in re.finditer(r"(?<!\\)%", line):
        j = m.start()
        break
    if j is None or j == 0 or not line[:j].strip():
        continue
    before, after = line[:j].rstrip(), line[j + 1:].strip()
    # legitimate: comment trails a finished row/environment, or is short
    if before.endswith(("\\\\", "}", "{")) or len(after.split()) <= 6:
        continue
    swallowed.append((n, after[:60]))
print("mid-line %% swallowing prose:", "none" if not swallowed else "")
for n, a in swallowed:
    print("   LINE %d swallows: %s..." % (n, a))

# The above only catches a '%' with prose BEFORE it on the same line. The worse
# case is a line that STARTS with '%' but has real LaTeX appended to its end --
# what happens when a multi-line replacement ends in a comment and the original
# line's remainder is dragged along. That text vanishes from the PDF silently.
# Signature: a comment line carrying body-text markup far from the '%'.
MARKUP = (r"\text{", r"\emph{", r"\textcolor{", r"\citep{", r"\citet{",
          r"\ref{", r"\textbf{", r"\textsc{")
buried = []
for n, line in enumerate(s.split("\n"), 1):
    st = line.lstrip()
    if not st.startswith("%"):
        continue
    hits = [m for m in MARKUP if m in line]
    # ignore short notes that merely mention a command; require markup late in a long line
    if hits and len(line) > 260 and max(line.rfind(m) for m in hits) > 200:
        buried.append((n, hits, line[-90:]))
print("prose buried in comment lines:", "none" if not buried else "")
for n, hits, tail in buried:
    print("   LINE %d ends with markup %s" % (n, hits))
    print("      ...%s" % tail)

# An abstract is typically one long line; a comment inside the environment is
# how the above happens in practice. Flag comments inside \begin{abstract}.
ab = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", s, re.S)
if ab:
    inner = [l for l in ab.group(1).split("\n") if l.lstrip().startswith("%")]
    print("comments inside abstract:", len(inner) or "none",
          "  <-- MOVE THEM OUTSIDE" if inner else "")

# \section/\paragraph/floats inside \textcolor{}{...} break the compile.
wrapped = re.findall(r"\\textcolor\{[a-z]+\}\{\\(section|subsection|paragraph|begin)", s)
print("sectioning/float in textcolor:", wrapped or "none")

keys = set()
for grp in re.findall(r"\\cite[tp]?\{([^}]+)\}", s):
    keys.update(k.strip() for k in grp.split(","))
bib = open(path.replace(".tex", ".bib"), encoding="utf-8").read()
bibkeys = set(re.findall(r"@\w+\{([^,]+),", bib))
print("bad cite keys:", sorted(keys - bibkeys) or "none")

# Escape mangling from a shell heredoc. `python - <<EOF` interprets backslash
# escapes inside string literals, so \textcolor becomes <TAB>extcolor and \ref
# becomes <CR>ef. Braces stay balanced and no \ref is left dangling, so every
# other check here passes clean -- this class of damage was invisible until it
# reached the PDF (2026-09-01, one paragraph in sec:progression).
# NOTE: read the raw bytes, not read_text(); universal-newline translation
# silently converts a stray CR to \n and hides the evidence.
_raw = open(path, "rb").read().decode("utf-8")
_stray_cr = _raw.count("\r") - _raw.count("\r\n")
_tabs = _raw.count("\t")
_other = sum(1 for ch in _raw if ord(ch) < 32 and ch not in "\t\n\r")
_orphans = []
# The lookbehinds must see a LITERAL backslash followed by the eaten letter, so
# they are written \\t / \\r / \\b. Writing \t here would mean the TAB character
# and the guard would never fire -- which is exactly what happened when this
# block was first appended through a heredoc.
for _pat, _cmd in ((r"(?<!\\t)extcolor\{", r"\textcolor"),
                   (r"(?<!\\r)ef\{(?:fig|tab|sec|eq|app):", r"\ref"),
                   (r"(?<!\\t)extbf\{", r"\textbf"),
                   (r"(?<!f)rac\{", r"\frac/\tfrac"),
                   (r"(?<!\\b)egin\{", r"\begin")):
    _n = len(re.findall(_pat, _raw))
    if _n:
        _orphans.append(f"{_cmd} x{_n}")
_bad = _stray_cr or _tabs or _other or _orphans
print("mangled escapes:",
      "none" if not _bad else
      f"stray CR={_stray_cr} TAB={_tabs} other-ctrl={_other} "
      f"orphans={_orphans or 'none'}  <-- HEREDOC DAMAGE, write a real .py file")
