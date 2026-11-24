"""Magic LLM – erzeugt die Logo-Dateien in assets/.   python tools/make_logo.py assets

Final Magic LLM logo (concept 'Token M prompt'): mark, wordmark outlined from Inter Display SemiBold (OFL)."""
from fontTools.ttLib import TTFont
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
import sys
from pathlib import Path
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
xs=[2,6.6,11.2,15.8,20.4]; hs=[25,18,11,18,25]
MARK = "".join(f'<rect x="{x}" y="{29-h}" width="3.4" height="{h}" rx="0.6"/>' for x,h in zip(xs,hs)) + \
       '<rect x="25.5" y="24.5" width="4.5" height="4.5" rx="0.5"/>'

def wordmark(text="Magic LLM", size=18.0, x0=40.0, baseline=23.0, tracking=-0.02):
    f = TTFont("/usr/share/fonts/opentype/inter/InterDisplay-SemiBold.otf")
    gs = f.getGlyphSet(); cmap = f.getBestCmap(); upm = f["head"].unitsPerEm; s = size / upm
    hmtx = f["hmtx"]; d = []; x = x0
    for ch in text:
        g = cmap[ord(ch)]
        pen = SVGPathPen(gs, ntos=lambda v: f"{v:.2f}".rstrip("0").rstrip("."))
        gs[g].draw(TransformPen(pen, (s, 0, 0, -s, x, baseline)))
        d.append(pen.getCommands())
        x += hmtx[g][0] * s + tracking * size
    return " ".join(p for p in d if p), x

def _r(v): return f"{round(v, 2):g}"
K, O = 0.725, 4.4          # Marke im Tile: 72,5 % gross, 4,4 Einheiten Rand
TILE = "".join(f'<rect x="{_r(O + K * x)}" y="{_r(O + K * (29 - h))}" width="{_r(K * 3.4)}" height="{_r(K * h)}" '
               f'rx="{_r(K * 0.6)}"/>' for x, h in zip(xs, hs)) + \
       f'<rect x="{_r(O + K * 25.5)}" y="{_r(O + K * 24.5)}" width="{_r(K * 4.5)}" height="{_r(K * 4.5)}" rx="{_r(K * 0.5)}"/>'

path, end = wordmark()
W = round(end + 2, 2)
def svg(w, body, fill, title=True):
    t = "<title>Magic LLM</title>" if title else ""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} 32" width="{w}" height="32" fill="{fill}" '
            f'role="img">{t}{body}</svg>\n')
files = {
 # mark only, follows the surrounding text colour (for inline use)
 "magic-llm-mark.svg": svg(32, MARK, "currentColor"),
 # full logo, fixed colours for <img> use (light and dark background)
 "magic-llm-logo.svg": svg(W, f'<g id="icon">{MARK}</g><g id="wordmark"><path d="{path}"/></g>', "#0A0A0A"),
 "magic-llm-logo-dark.svg": svg(W, f'<g id="icon">{MARK}</g><g id="wordmark"><path d="{path}"/></g>', "#FAFAFA"),
 "magic-llm-mark-black.svg": svg(32, MARK, "#0A0A0A"),
 "magic-llm-mark-white.svg": svg(32, MARK, "#FAFAFA"),
 # favicon / app icon: mark on a tile so it works on light and dark browser tabs (coordinates flattened)
 "magic-llm-icon.svg": ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" width="32" height="32" role="img">'
   '<title>Magic LLM</title><rect width="32" height="32" rx="7" fill="#0A0A0A"/><g fill="#FAFAFA">' + TILE + '</g></svg>\n'),
}
for n, s in files.items():
    (OUT / n).write_text(s, encoding="utf-8")
print("width", W)
