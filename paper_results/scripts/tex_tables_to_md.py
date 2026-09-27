"""Extract every table of main.tex / supplementary.tex into Markdown, with its printed number and caption."""
import re, sys
def grab(t, i):
    """t[i] == '{' -> (content, index after matching '}')"""
    d = 0
    for j in range(i, len(t)):
        if t[j] == '{' and t[j-1] != '\\': d += 1
        elif t[j] == '}' and t[j-1] != '\\':
            d -= 1
            if d == 0: return t[i+1:j], j+1
    raise ValueError
GREEK = dict(varphi="φ", vartheta="ϑ", varepsilon="ε", epsilon="ε", Pi="Π", pi="π", omega="ω", Omega="Ω", phi="φ", Phi="Φ",
    beta="β", gamma="γ", Gamma="Γ", delta="δ", kappa="κ", nu="ν", xi="ξ", rho="ρ", Sigma="Σ", Psi="Ψ", Lambda="Λ", upsilon="υ",
    varrho="ϱ", iota="ι", ell="ℓ", nabla="∇", partial="∂", sum="Σ", prod="Π", in_="∈", neq="≠", ne="≠", star="*", prime="′",
    lVert="‖", rVert="‖", lvert="|", rvert="|", mid="|", vert="|", langle="⟨", rangle="⟩", log="log", exp="exp", min="min", max="max", arg="arg")
SUBS = [(r"\times", "×"), (r"\pm", "±"), (r"\leq", "≤"), (r"\geq", "≥"), (r"\le", "≤"), (r"\ge", "≥"), (r"\alpha", "α"),
        (r"\tau", "τ"), (r"\mu", "μ"), (r"\sigma", "σ"), (r"\chi", "χ"), (r"\approx", "≈"), (r"\infty", "∞"), (r"\rightarrow", "→"),
        (r"\to", "→"), (r"\cdot", "·"), (r"\ldots", "…"), (r"\dots", "…"), (r"\Delta", "Δ"), (r"\eta", "η"), (r"\psi", "ψ"), (r"\zeta", "ζ"),
        (r"\theta", "θ"), (r"\lambda", "λ"), (r"\sim", "~"), (r"\%", "%"), (r"\&", "&"), (r"\_", "_"), (r"\#", "#"), (r"\textbullet", "•"),
        (r"\checkmark", "✓"), (r"\,", " "), (r"\;", " "), (r"\!", ""), (r"\quad", " "), (r"\textdegree", "°"), (r"^\circ", "°"),(r"\circ", "°"),
        (r"\-", ""), (r"\newline", " "), (r"\linebreak", " "), (r"\bfseries", ""), (r"\centering", ""), (r"\raggedright", ""),
        (r"\hfill", ""), (r"\scriptsize", ""), (r"\footnotesize", ""), (r"\small", ""), (r"\tiny", ""), (r"\normalsize", "")]
LAB = {}
def clean(s):
    s = re.sub(r"(?<!\\)%.*", "", s)
    s = re.sub(r"\\(cellcolor|rowcolor|columncolor)(\[[^\]]*\])?\{[^}]*\}", "", s)
    s = re.sub(r"\\color\{[^}]*\}", "", s)
    for _ in range(4):
        s = re.sub(r"\\multicolumn\{[^}]*\}\{[^}]*\}\{((?:[^{}]|\{[^{}]*\})*)\}", r"\1", s)
        s = re.sub(r"\\multirow\{[^}]*\}(\[[^\]]*\])?\{[^}]*\}\{((?:[^{}]|\{[^{}]*\})*)\}", r"\2", s)
        s = re.sub(r"\\textsuperscript\{([^{}]*)\}", r"^\1", s)
        s = re.sub(r"\\textsubscript\{([^{}]*)\}", r"_\1", s)
        s = re.sub(r"\\(textbf|textit|emph|mathrm|mathbf|text|textrm|texttt|mbox|makecell|shortstack|underline|hbox|mathit|operatorname|boldsymbol)(\[[^\]]*\])?\{((?:[^{}]|\{[^{}]*\})*)\}", r"\3", s)
    s = re.sub(r"\\(mathcal|mathbb|hat|tilde|bar|widehat|widetilde|overline|vec|dot)\{([^{}]*)\}", r"\2", s)
    s = re.sub(r"\\(ref|SUPP)\{([^}]*)\}", lambda m: LAB.get(m.group(2), "[" + m.group(2) + "]"), s)
    s = re.sub(r"\\(cite|label)\{[^}]*\}", "", s)
    s = re.sub(r"\\(hspace|vspace|rule)\*?(\[[^\]]*\])?\{[^}]*\}(\{[^}]*\})?", "", s)
    s = s.replace("---", "—").replace("--", "–").replace("~", " ")
    s = re.sub(r"\\left|\\right(?![a-z])|\\big[lr]?|\\Big[lr]?", "", s)
    for a, b in SUBS:
        s = re.sub(re.escape(a) + r"(?![A-Za-z])", lambda m: b, s) if re.fullmatch(r"\\[A-Za-z]+", a) else s.replace(a, b)
    s = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\\(in)(?![A-Za-z])", "∈", s)
    s = re.sub(r"\\([A-Za-z]+)(?![A-Za-z])", lambda m: GREEK.get(m.group(1), m.group(0)), s)
    s = s.replace("$", "").replace("{", "").replace("}", "").replace("\\\\", " ")
    s = re.sub(r"\\[A-Za-z]+\*?", "", s)
    return re.sub(r"\s+", " ", s).strip().replace("|", "\\|")
def tab_to_md(body):
    body = re.sub(r"(?<!\\)%.*", "", body)
    body = re.sub(r"\\(hline|toprule|midrule|bottomrule|endhead|endfirsthead|endfoot|endlastfoot)", "", body)
    body = re.sub(r"\\(cline|cmidrule)(\([^)]*\))?\{[^}]*\}", "", body)
    rows = [r for r in re.split(r"\\\\(?:\[[^\]]*\])?", body) if r.strip()]
    rows = [[clean(c) for c in re.split(r"(?<!\\)&", r)] for r in rows]
    rows = [r for r in rows if any(r)]
    if not rows: return ""
    w = max(len(r) for r in rows); rows = [r + [""] * (w - len(r)) for r in rows]
    out = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * w]
    out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(out)
def tabulars(t):
    res = []
    for m in re.finditer(r"\\begin\{(tabular\*?|tabularx|longtable)\}", t):
        env = m.group(1); i = m.end()
        if env in ("tabularx", "tabular*"): _, i = grab(t, t.index("{", i))
        while i < len(t) and t[i] in " \n": i += 1
        if t[i] == "[": i = t.index("]", i) + 1
        _, i = grab(t, i)                                   # column spec
        j = t.index("\\end{%s}" % env, i)
        res.append((m.start(), j, t[i:j]))
    return res
def extract(path, aux=None, supp=False):
    t = open(path).read()
    labnum = {}
    if aux:
        for m in re.finditer(r"\\newlabel\{([^}]*)\}\{\{([^}]*)\}", open(aux).read()): labnum[m.group(1)] = m.group(2)
        LAB.update(labnum)
    # caption sources: \caption{...} (with optional \label), or "\textbf{Table Sx.} text\par"
    caps = []
    for m in re.finditer(r"\\caption(\[[^\]]*\])?\{", t):
        c, e = grab(t, m.end() - 1); lab = re.search(r"\\label\{([^}]*)\}", t[e:e+80]) or re.search(r"\\label\{([^}]*)\}", c)
        caps.append((m.start(), "caption", c, lab.group(1) if lab else None))
    for m in re.finditer(r"\\textbf\{Table (S?\d+)\.\}(.*?)\\par", t, flags=re.S):
        caps.append((m.start(), "manual", m.group(2), m.group(1)))
    caps.sort()
    envs = [(m.start(), t.index("\\end{table", m.end())) for m in re.finditer(r"\\begin\{table\*?\}", t)]
    groups = {}; order = []
    counter = 0; supp_counter = 19
    for s, e, body in tabulars(t):
        env = next(((a, b) for a, b in envs if a < s < b), None)
        if env:
            cand = [c for c in caps if env[0] < c[0] < env[1] and c[1] == "caption"] or [c for c in caps if c[0] < s][-1:]
        else:
            cand = [c for c in caps if c[0] < s][-1:]
        key = cand[0][0] if cand else ("nocap", s)
        if key not in groups:
            groups[key] = dict(cap=cand[0] if cand else None, bodies=[]); order.append(key)
        groups[key]["bodies"].append(body)
    out = []
    for k in order:
        g = groups[k]; c = g["cap"]
        if c is None: num, text = "?", ""
        elif c[1] == "manual": num, text = c[3], c[2]
        else:
            num = labnum.get(c[3]) if c[3] else None
            if num is None:
                supp_counter += 1; num = f"S{supp_counter}" if supp else "?"
            text = c[2]
        if supp and c is not None and c[1] == "caption" and c[3] in labnum: pass
        md = "\n\n".join(tab_to_md(b) for b in g["bodies"])
        out.append((num, clean(text), md))
    return out
if __name__ == "__main__":
    path, aux, supp, dest = sys.argv[1], sys.argv[2], sys.argv[3] == "supp", sys.argv[4]
    tabs = extract(path, aux if aux != "-" else None, supp)
    with open(dest, "w") as f:
        for num, cap, md in tabs:
            f.write(f"### Table {num}\n\n{cap}\n\n{md}\n\n")
    print(len(tabs), "tables ->", dest, [n for n, _, _ in tabs])
