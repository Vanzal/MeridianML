"""Hugging-Face-Zugangsschlüssel (Token) – nur mit Leserechten nötig.

Wofür? Manche Datensätze sind „gesperrt“ (gated): Man muss sie auf huggingface.co einmal akzeptieren und sich
beim Laden ausweisen. Ausserdem gelten mit Token höhere Abruf-Grenzen.

Wo wird er gespeichert? In ``settings/hf_token`` (Dateirechte 600, nur für dich lesbar; der Ordner ist in
.gitignore). Beim Start wird er als Umgebungsvariable ``HF_TOKEN`` gesetzt – so finden ihn ``datasets`` und
``huggingface_hub`` automatisch. Ist ``HF_TOKEN`` schon von aussen gesetzt, hat dieser Wert Vorrang.

Der Token wird nie vollständig angezeigt, geloggt oder irgendwohin ausser an huggingface.co geschickt.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOKEN_RE = re.compile(r"^hf_[A-Za-z0-9]{30,}$")
_ENV_FROM_FILE = "_MAGICLLM_HF_TOKEN_FROM_FILE"


def settings_dir() -> Path:
    return Path(os.environ.get("MAGICLLM_SETTINGS_DIR") or ROOT / "settings")


def token_file() -> Path:
    return settings_dir() / "hf_token"


def mask(token: str | None) -> str:
    if not token:
        return "–"
    return f"{token[:3]}…{token[-4:]}" if len(token) > 10 else "…"


def looks_valid(token: str) -> bool:
    return bool(TOKEN_RE.match(token.strip()))


def stored_token() -> str | None:
    try:
        t = token_file().read_text(encoding="utf-8").strip()
        return t or None
    except OSError:
        return None


def source() -> str:
    """Woher der aktive Token kommt: 'umgebung' (von aussen gesetzt), 'datei' (in der App gespeichert) oder ''."""
    if os.environ.get("HF_TOKEN") and not os.environ.get(_ENV_FROM_FILE):
        return "umgebung"
    return "datei" if stored_token() else ""


def active_token() -> str | None:
    return os.environ.get("HF_TOKEN") or stored_token()


def apply() -> None:
    """Gespeicherten Token als HF_TOKEN setzen (ausser er ist schon von aussen gesetzt)."""
    if os.environ.get("HF_TOKEN") and not os.environ.get(_ENV_FROM_FILE):
        return
    t = stored_token()
    if t:
        os.environ["HF_TOKEN"] = t
        os.environ[_ENV_FROM_FILE] = "1"
    elif os.environ.get(_ENV_FROM_FILE):
        os.environ.pop("HF_TOKEN", None)
        os.environ.pop(_ENV_FROM_FILE, None)


def save(token: str) -> None:
    token = token.strip()
    if not looks_valid(token):
        raise ValueError("Das sieht nicht wie ein Hugging-Face-Token aus (beginnt mit „hf_“).")
    d = settings_dir()
    d.mkdir(parents=True, exist_ok=True)
    f = token_file()
    tmp = f.with_name(".hf_token.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(token)
    os.replace(tmp, f)
    try:
        os.chmod(f, 0o600)
    except OSError:               # pragma: no cover - z. B. Windows
        pass
    apply()


def remove() -> None:
    try:
        token_file().unlink()
    except FileNotFoundError:
        pass
    apply()


@dataclass
class Check:
    ok: bool
    user: str = ""
    role: str = ""             # read | write | fineGrained | ""
    message: str = ""

    @property
    def read_only(self) -> bool:
        return self.role in ("read", "fineGrained")


def verify(token: str, timeout: float = 10.0) -> Check:
    """Fragt huggingface.co, wem der Token gehört und welche Rechte er hat."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return Check(False, message="Paket huggingface_hub fehlt (pip install huggingface_hub).")
    try:
        info = HfApi().whoami(token=token.strip())
    except Exception as e:     # Netzwerk, ungültig, abgelaufen …
        text = str(e)
        if "401" in text or "Invalid" in text or "Unauthorized" in text:
            return Check(False, message="Hugging Face kennt diesen Token nicht (ungültig oder gelöscht).")
        return Check(False, message=f"Konnte Hugging Face nicht erreichen: {type(e).__name__}.")
    auth = (info.get("auth") or {}).get("accessToken") or {}
    role = auth.get("role", "")
    return Check(True, user=info.get("name", ""), role=role)
