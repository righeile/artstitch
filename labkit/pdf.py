"""An SVG as one vector PDF page, printed by the browser already on the machine.

No SVG->PDF converter is installed (cairosvg, svglib, reportlab and fitz are
all absent), `sips` answers an SVG with a JPEG inside a PDF wrapper, and PIL's
PDF is a bitmap.  Headless Chrome keeps text as embedded TrueType and data-URI
images as lossless /FlateDecode.  First written for purifig (2026-10-14).
"""

from __future__ import annotations

import io
import os
import re
import shutil
import subprocess
import tempfile
import urllib.request

CHROME_PATHS = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
]


def find_chrome():
    for path in CHROME_PATHS:
        if os.path.exists(path):
            return path
    for name in ("google-chrome", "chromium", "chromium-browser", "chrome", "msedge"):
        found = shutil.which(name)
        if found:
            return found
    return None


#: `size` in millimeters and no margin, so the paper *is* the figure.  Chrome
#: still snaps the page to its own 1/300 inch grid -- 183 mm comes out 183.13
#: and 240 mm comes out 239.86, which leaves the bottom edge of the figure
#: outside the page box -- so the box is written again exactly afterwards.
PDF_PAGE = """<!doctype html><meta charset="utf-8">
<style>@page{{size:{w:.4f}mm {h:.4f}mm;margin:0}}
html,body{{margin:0;padding:0;background:transparent}}svg{{display:block}}</style>
{svg}
"""


def exact_page(path, w_mm, h_mm):
    """Chrome's page box, rounded to its device grid, made exact again.

    The drawing is placed from the top-left and is exact already, so the box is
    anchored there too."""
    import pypdf

    reader = pypdf.PdfReader(path)
    writer = pypdf.PdfWriter()
    w, h = w_mm / 25.4 * 72, h_mm / 25.4 * 72
    for page in reader.pages:
        top = float(page.mediabox.top)
        page.mediabox = pypdf.generic.RectangleObject([0, top - h, w, top])
        writer.add_page(page)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def svg_to_pdf(svg, w_mm, h_mm):
    """One page, exactly w x h mm, the text still text and the images embedded.
    RuntimeError when there is no browser to print with."""
    exe = find_chrome()
    if not exe:
        raise RuntimeError(
            "No Chrome, Chromium or Edge on this machine to render the PDF with. "
            "Export SVG instead, or open the figure in a browser and print it.")
    svg = re.sub(r"^\s*<\?xml[^>]*\?>\s*", "", svg)
    with tempfile.TemporaryDirectory() as tmp:
        page = os.path.join(tmp, "figure.html")
        out = os.path.join(tmp, "figure.pdf")
        with open(page, "w", encoding="utf-8") as fh:
            fh.write(PDF_PAGE.format(w=w_mm, h=h_mm, svg=svg))
        # Its own profile directory: the user's browser is open with their own
        # tabs in it, and headless must not touch that profile.
        done = subprocess.run(
            [exe, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
             "--user-data-dir=" + os.path.join(tmp, "profile"),
             "--print-to-pdf=" + out,
             "file://" + urllib.request.pathname2url(page)],
            capture_output=True, timeout=240)
        if not os.path.exists(out):
            tail = done.stderr.decode("utf-8", "replace").strip()[-400:]
            raise RuntimeError("the browser wrote no PDF: " + (tail or "no output"))
        return exact_page(out, w_mm, h_mm)
