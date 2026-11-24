"""Themes: Dunkel, Hell, Terminal – und eigene, frei bearbeitbare Themes.

Ein Theme besteht aus wenigen Grundfarben. Alles andere (Hover-Farben, Gitterlinien, Text auf Akzent-Knöpfen …)
wird daraus abgeleitet, damit auch selbst gemischte Themes stimmig aussehen.

Angewendet wird ein Theme auf zwei Ebenen:
  1. Streamlits eigenes Theme (alle Eingabefelder, Tabellen, Schieberegler …) – über die Config-Optionen
     ``theme.*``, die sich zur Laufzeit setzen lassen (danach einmal neu laden).
  2. Die CSS-Variablen ``--ml-*`` der App und die Farben aller Diagramme.

Gespeichert wird in ``settings/theme.json`` (oder ``$MERIDIAN_SETTINGS_DIR``).
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SETTINGS_DIR = Path(os.environ.get("MERIDIAN_SETTINGS_DIR") or ROOT / "settings")
THEME_FILE = SETTINGS_DIR / "theme.json"
HEX = re.compile(r"^#[0-9A-Fa-f]{6}$")


@dataclass
class Theme:
    name: str
    base: str                      # "dark" | "light"
    bg: str                        # Hintergrund
    surface: str                   # Karten, Eingabefelder
    border: str                    # Rahmen
    fg: str                        # Text
    muted: str                     # gedämpfter Text
    primary: str                   # Akzent: Hauptknöpfe, Fortschritt, wichtigste Diagramm-Linie
    sidebar: str                   # Seitenleiste
    chart: list[str] = field(default_factory=list)   # Diagramm-Farben (Reihenfolge = Serien)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Theme":
        base = PRESETS["Dunkel"].to_dict()
        clean = {k: d[k] for k in base if k in d}
        t = cls(**{**base, **clean})
        t.validate()
        return t

    def validate(self) -> None:
        if self.base not in ("dark", "light"):
            raise ValueError("base muss 'dark' oder 'light' sein")
        for k in ("bg", "surface", "border", "fg", "muted", "primary", "sidebar"):
            if not HEX.match(getattr(self, k)):
                raise ValueError(f"Farbe '{k}' ist kein Hex-Wert wie #1A2B3C")
        if not self.chart or not all(HEX.match(c) for c in self.chart):
            raise ValueError("Diagramm-Farben müssen Hex-Werte sein")
        self.name = (self.name or "Eigenes Theme").strip()[:40]


PRESETS: dict[str, Theme] = {
    "Dunkel": Theme("Dunkel", "dark", "#0A0A0A", "#171717", "#262626", "#FAFAFA", "#A3A3A3", "#FAFAFA", "#0F0F0F",
                    ["#E5E5E5", "#60A5FA", "#F59E0B", "#34D399", "#F472B6", "#A78BFA", "#737373"]),
    "Hell": Theme("Hell", "light", "#FFFFFF", "#F5F5F5", "#E5E5E5", "#0A0A0A", "#737373", "#171717", "#FAFAFA",
                  ["#262626", "#2563EB", "#D97706", "#059669", "#DB2777", "#7C3AED", "#A3A3A3"]),
    "Terminal": Theme("Terminal", "dark", "#0C0C0C", "#141414", "#1F2A1F", "#D4D4D4", "#8A8F8A", "#4ADE80", "#0A0A0A",
                      ["#4ADE80", "#22D3EE", "#FACC15", "#F472B6", "#A78BFA", "#FB923C", "#6B7280"]),
}


# --------------------------------------------------------------------------- Farb-Helfer
def _rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _hex(r: float, g: float, b: float) -> str:
    return "#{:02X}{:02X}{:02X}".format(*(max(0, min(255, round(v))) for v in (r, g, b)))


def mix(a: str, b: str, t: float) -> str:
    """Farbe a mit t Anteil von Farbe b mischen."""
    (r1, g1, b1), (r2, g2, b2) = _rgb(a), _rgb(b)
    return _hex(r1 + (r2 - r1) * t, g1 + (g2 - g1) * t, b1 + (b2 - b1) * t)


def luminance(h: str) -> float:
    def ch(c):
        c /= 255
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = _rgb(h)
    return 0.2126 * ch(r) + 0.7152 * ch(g) + 0.0722 * ch(b)


def contrast(a: str, b: str) -> float:
    la, lb = sorted((luminance(a), luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def tokens(t: Theme) -> dict[str, str]:
    """Alle abgeleiteten Farben eines Themes."""
    on_primary = "#0A0A0A" if contrast("#0A0A0A", t.primary) >= contrast("#FFFFFF", t.primary) else "#FFFFFF"
    return {
        "bg": t.bg, "surface": t.surface, "surface2": mix(t.surface, t.fg, 0.05), "border": t.border,
        "border_strong": mix(t.border, t.fg, 0.10), "fg": t.fg, "muted": t.muted, "dim": mix(t.muted, t.bg, 0.30),
        "primary": t.primary, "on_primary": on_primary, "primary_hover": mix(t.primary, t.bg, 0.15),
        "sidebar": t.sidebar, "grid": mix(t.bg, t.fg, 0.08), "code": mix(t.bg, t.fg, 0.04),
        "gen_bg": mix(t.bg, t.chart[1] if len(t.chart) > 1 else t.primary, 0.14),
    }


# --------------------------------------------------------------------------- Speichern / Laden
def _read() -> dict:
    try:
        data = json.loads(THEME_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write(data: dict) -> None:
    SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = THEME_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, THEME_FILE)


def custom_themes() -> dict[str, Theme]:
    out = {}
    for name, d in (_read().get("custom") or {}).items():
        try:
            out[name] = Theme.from_dict({**d, "name": name})
        except (ValueError, TypeError):
            continue
    return out


def all_themes() -> dict[str, Theme]:
    return {**PRESETS, **custom_themes()}


def active_theme() -> Theme:
    name = _read().get("active", "Dunkel")
    return all_themes().get(name, PRESETS["Dunkel"])


def set_active(name: str) -> None:
    data = _read()
    data["active"] = name if name in all_themes() else "Dunkel"
    _write(data)


def save_custom(theme: Theme, activate: bool = True) -> str:
    theme.validate()
    if theme.name in PRESETS:
        theme.name = f"{theme.name} (eigenes)"
    data = _read()
    data.setdefault("custom", {})[theme.name] = theme.to_dict()
    if activate:
        data["active"] = theme.name
    _write(data)
    return theme.name


def delete_custom(name: str) -> None:
    data = _read()
    (data.get("custom") or {}).pop(name, None)
    if data.get("active") == name:
        data["active"] = "Dunkel"
    _write(data)


# --------------------------------------------------------------------------- Anwenden
def streamlit_options(t: Theme) -> dict:
    k = tokens(t)
    return {
        "theme.base": t.base, "theme.primaryColor": t.primary, "theme.backgroundColor": t.bg,
        "theme.secondaryBackgroundColor": t.surface, "theme.textColor": t.fg, "theme.linkColor": t.fg,
        "theme.borderColor": t.border, "theme.codeBackgroundColor": k["code"],
        "theme.dataframeHeaderBackgroundColor": t.surface, "theme.chartCategoricalColors": list(t.chart),
        "theme.sidebar.backgroundColor": t.sidebar, "theme.sidebar.secondaryBackgroundColor": t.surface,
    }


def apply_to_streamlit(t: Theme) -> bool:
    """Setzt Streamlits eigenes Theme. Gibt True zurück, wenn sich etwas geändert hat (dann neu laden)."""
    import streamlit as st
    try:
        from streamlit import config as st_config
    except ImportError:          # pragma: no cover
        return False
    changed = False
    for key, val in streamlit_options(t).items():
        try:
            if st.get_option(key) != val:
                st_config.set_option(key, val)
                changed = True
        except Exception:        # Option gibt es in dieser Streamlit-Version nicht – CSS übernimmt
            continue
    return changed
