"""Where an app's code lives, written out of the code itself.

    # the app's own map.py, at its root
    from labkit import codemap
    codemap.main(os.path.dirname(os.path.abspath(__file__)), HEAD)

Six thousand lines of ``app.js`` and two thousand of ``app.py`` are not readable
in one sitting, and nobody should have to read them to find out where one pane
keeps its handlers.  Everything in the map is read back out of the code -- the
route table, the module one-liners, the section ranges, the id prefix each pane
actually uses -- so it cannot drift from what is true the way a hand-kept one
does.  What it deliberately does not do is explain anything: the prose about
*why* belongs in README.md, and this only says *where*.

These apps all have the one shape -- a Flask ``app.py`` whose routes lean on a
package of thinking modules, one big ``static/app.js`` divided by banner
comments, and an ``index.html`` of panes -- so the reading is shared and only
the words at the top of the document are the app's own.  ``head`` is that
prose; ``pkg`` names the package, ``extra`` lists modules outside it.
"""

import ast
import os
import re
import sys
from html.parser import HTMLParser


def first_line(doc):
    return (doc or "").strip().split("\n")[0].strip() if doc else ""


def alias_map(src, pkg="seqlib"):
    """`{'GIB': 'gibson'}` from the server's own imports."""
    out = {}
    for m in re.finditer(r"^from %s import (\w+) as (\w+)" % pkg, src, re.M):
        out[m.group(2)] = "%s/%s.py" % (pkg, m.group(1))
    for m in re.finditer(r"^import (\w+) as (\w+)$", src, re.M):
        out[m.group(2)] = "%s.py" % m.group(1)
    return out


def routes(root, pkg="seqlib", server="app.py"):
    """Every route, with the handler's one-line summary and what it leans on."""
    src = open(os.path.join(root, server), encoding="utf-8").read()
    lines = src.split("\n")
    aliases = alias_map(src, pkg)
    tree = ast.parse(src)
    body = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    out = []
    for fn in body.values():
        verbs = []
        for d in fn.decorator_list:
            # `@app.get("/api/...")` and friends.  Not every `app.something(...)`
            # decorator is a route: `@app.errorhandler(Exception)` takes a class,
            # and reading that as a path is a crash, not a line in the map.
            if (isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute)
                    and getattr(d.func.value, "id", "") == "app" and d.args
                    and isinstance(d.args[0], ast.Constant)
                    and isinstance(d.args[0].value, str)):
                verbs.append((d.func.attr.upper(), ast.literal_eval(d.args[0])))
        if not verbs:
            continue
        span = "\n".join(lines[fn.lineno - 1:fn.end_lineno])
        leans = sorted({aliases[a] for a in aliases if re.search(r"\b%s\." % a, span)})
        for verb, path in verbs:
            out.append((path, verb, fn.lineno, fn.name,
                        first_line(ast.get_docstring(fn)), leans))
    return sorted(out)


def modules(root, pkg="seqlib", extra=()):
    """Each module in the package, by what its docstring says it is.

    `extra` are modules outside the package that think as well, by their path
    from the app root, and they come first.
    """
    names = sorted(f for f in os.listdir(os.path.join(root, pkg))
                   if f.endswith(".py") and f != "__init__.py")
    out = []
    for rel in list(extra) + ["%s/%s" % (pkg, f) for f in names]:
        src = open(os.path.join(root, rel), encoding="utf-8").read()
        out.append((rel, len(src.split("\n")),
                    first_line(ast.get_docstring(ast.parse(src)))))
    return out


BANNER = re.compile(r"^\s*(?://|/\*)\s*-{3,}\s*(.+?)\s*-{3,}", re.M)
JSFN = re.compile(r"^  (?:async )?function (\w+)", re.M)


def js_sections(root, js="static/app.js"):
    """The client's own banner comments, where each starts, and what is in it."""
    src = open(os.path.join(root, js), encoding="utf-8").read()
    lines = src.split("\n")
    marks = [(src[:m.start()].count("\n") + 1, m.group(1)) for m in BANNER.finditer(src)]
    marks.append((len(lines) + 1, None))
    out = []
    for (at, name), (nxt, _) in zip(marks, marks[1:]):
        if name is None:
            continue
        fns = JSFN.findall("\n".join(lines[at:nxt - 1]))
        out.append((at, nxt - 1, name, fns))
    return out


class Panes(HTMLParser):
    """Which ids sit inside each pane and each tab.

    A regex cannot do this: a pane is several nested blocks, and the last tab
    has nothing after it to stop at, so a lookahead swallows the rest of the
    page.  The stdlib parser knows when an element closes, which is the whole
    question.
    """

    VOID = {"input", "img", "br", "hr", "meta", "link", "option"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack, self.open, self.ids = [], [], {}

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        name = a.get("data-pane") or (
            a.get("id", "")[4:] if a.get("id", "").startswith("tab-") else None)
        if tag not in self.VOID:
            self.stack.append((tag, name))
        if name:
            self.ids.setdefault(name, [])
        if a.get("id"):
            for _, held in self.stack:
                if held:
                    self.ids.setdefault(held, []).append(a["id"])

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                del self.stack[i:]
                return


def panes(root, html="static/index.html"):
    """Each pane and tab, and the id prefix its controls actually carry.

    Derived, not declared: a pane whose ids do not agree on one prefix is a pane
    sharing its names with something else, which is worth seeing.
    """
    p = Panes()
    p.feed(open(os.path.join(root, html), encoding="utf-8").read())
    out = []
    for name, ids in p.ids.items():
        heads = {}
        for i in ids:
            bare = re.sub(r"^btn", "", i)
            m = re.match(r"[a-z]+", bare[0].lower() + bare[1:])
            if m:
                heads[m.group(0)] = heads.get(m.group(0), 0) + 1
        if not heads:
            continue
        top = sorted(heads.items(), key=lambda kv: (-kv[1], kv[0]))
        out.append((name, top[0][0], len(ids), [k for k, v in top[1:] if v > 1]))
    return sorted(out)


def tests(root):
    out = []
    for f in sorted(f for f in os.listdir(root) if re.match(r"test_.*\.py$", f)):
        src = open(os.path.join(root, f), encoding="utf-8").read()
        n = len(re.findall(r"^def (?:test_\w+|demo)\(", src, re.M))
        out.append((f, n, first_line(ast.get_docstring(ast.parse(src)))))
    return out


def build(root, head, pkg="seqlib", extra=(), server="app.py",
          js="static/app.js", html="static/index.html",
          science="each takes data and gives back answers"):
    """The whole of MAP.md, as text."""
    o = [head, "\n## Routes\n",
         "| route | %s | leans on | what it answers |" % server,
         "|---|---|---|---|"]
    for path, verb, at, fn, doc, leans in routes(root, pkg, server):
        o.append("| `%s %s` | [%d](%s#L%d) | %s | %s |"
                 % (verb, path, at, server, at,
                    ", ".join("`%s`" % x for x in leans) or "--", doc))

    o += ["\n## The science: one module a subject\n",
          "No Flask, no page, no state -- %s.\n" % science,
          "| module | lines | what it is |", "|---|---|---|"]
    for name, n, doc in modules(root, pkg, extra):
        o.append("| [`%s`](%s) | %d | %s |" % (name, name, n, doc))

    o += ["\n## %s, by its own section banners\n" % js,
          "One IIFE. The line numbers move; re-run `map.py` rather than "
          "trusting them.\n"]
    for at, end, name, fns in js_sections(root, js):
        o.append("- **%s** -- [%d-%d](%s#L%d)  \n  %s"
                 % (name, at, end, js, at,
                    ", ".join("`%s`" % f for f in fns)
                    or "_(no top-level functions)_"))

    o += ["\n## Panes, and the id prefix each one's controls carry\n",
          "Derived from the page, so a pane that shares a prefix shows up here.\n",
          "| pane | prefix | ids | also present |", "|---|---|---|---|"]
    for name, pre, n, others in panes(root, html):
        o.append("| `%s` | `%s` | %d | %s |"
                 % (name, pre, n, ", ".join("`%s`" % x for x in others) or "--"))

    o += ["\n## What the tests pin\n", "| file | tests | what it is for |",
          "|---|---|---|"]
    for f, n, doc in tests(root):
        o.append("| [`%s`](%s) | %d | %s |" % (f, f, n, doc))

    o.append("\n---\n\n`python map.py` rewrites this file; `--check` says "
             "whether it is stale.\n")
    return "\n".join(o)


def main(root, head, out="MAP.md", argv=None, **kw):
    """Write the map, or with `--check` say whether the one on disk is stale."""
    text = build(root, head, **kw)
    path = os.path.join(root, out)
    if "--check" in (sys.argv if argv is None else argv):
        old = open(path, encoding="utf-8").read() if os.path.exists(path) else ""
        print("%s is up to date" % out if old == text
              else "%s is stale -- run map.py" % out)
        sys.exit(0 if old == text else 1)
    open(path, "w", encoding="utf-8").write(text)
    print("%s: %d lines" % (out, len(text.split("\n"))))


def demo():
    """A whole little app in a temp folder, and the map it comes out as."""
    import tempfile

    with tempfile.TemporaryDirectory() as root:
        w = lambda rel, s: (os.makedirs(os.path.join(root, os.path.dirname(rel)),
                                        exist_ok=True)
                            if os.path.dirname(rel) else None,
                            open(os.path.join(root, rel), "w").write(s))
        w("app.py", '"""The server."""\n'
                    "from bits import cut as CUT\n"
                    "import verify as VF\n"
                    "\n"
                    '@app.get("/api/cut")\n'
                    "def api_cut():\n"
                    '    """What an enzyme leaves."""\n'
                    "    return CUT.run(), VF.grade()\n"
                    "\n"
                    "@app.errorhandler(Exception)\n"       # not a route
                    "def oops(e):\n"
                    "    return 500\n")
        w("verify.py", '"""Reads against a reference."""\n')
        w("bits/__init__.py", "")
        w("bits/cut.py", '"""Where an enzyme cuts."""\n')
        w("bits/zz.py", '"""Last by name."""\n')
        w("static/app.js", "(function () {\n"
                           "  // --- chrome ---\n"
                           "  function toast() {}\n"
                           "  async function api() {}\n"
                           "  // --- the map ---\n"
                           "  function draw() {}\n"
                           "}());\n")
        w("static/index.html", '<div id="tab-seq"><input id="seqStart">'
                               '<button id="btnSeqGo"></button></div>'
                               '<div data-pane="gel"><b id="gelRow"></b></div>')
        w("test_cut.py", '"""Pins the cut."""\ndef test_a():\n    pass\n'
                         "def demo():\n    pass\n")

        text = build(root, "# head", pkg="bits", extra=["verify.py"])

    assert "| `GET /api/cut` | [6](app.py#L6) | `bits/cut.py`, `verify.py` | " \
           "What an enzyme leaves. |" in text, "a route, where it is, what it leans on"
    assert "errorhandler" not in text, "a decorator that is not a route stays out"
    assert text.index("| [`verify.py`](verify.py) | 2 |") \
        < text.index("| [`bits/cut.py`](bits/cut.py) | 2 |") < text.index("bits/zz.py"), \
        "the outsiders first, then the package by name"
    assert "__init__" not in text, "the package's own __init__ is not a subject"
    assert "- **chrome** -- [2-4](static/app.js#L2)  \n  `toast`, `api`" in text, \
        "a section runs to the next banner, async counted"
    assert "- **the map** -- [5-8](static/app.js#L5)" in text, "the last one runs to the end"
    assert "| `seq` | `seq` | 3 |" in text, "a tab's ids, its own among them"
    assert "| `gel` | `gel` | 1 |" in text, "and a pane names itself"
    assert "| [`test_cut.py`](test_cut.py) | 2 | Pins the cut. |" in text, \
        "a demo counts as a test"

    # The map is a function of the code, so building it twice is the same text.
    print("labkit.codemap: all checks passed.")


if __name__ == "__main__":
    demo()
