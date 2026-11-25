#!/usr/bin/env python3
"""Bereitet alles vor, was die Desktop-App (Tauri) mitbringt. Vor `tauri dev` und `tauri build` ausführen.

    python3 desktop/scripts/stage.py            # für das eigene System
    python3 desktop/scripts/stage.py --target x86_64-pc-windows-msvc

Legt an (alles in .gitignore):
    src-tauri/binaries/uv-<target>[.exe]   uv von Astral (Sidecar): lädt beim ersten Start Python + PyTorch
    src-tauri/resources/app/               der Code von Meridian (ohne Tests, Doku, Packaging)
    src-tauri/resources/wheels/            optional: Rust-Tokenizer (meridian_fastbpe-*.whl aus build/wheels)
"""
from __future__ import annotations

import argparse
import os
import shutil
import stat
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DESKTOP = HERE.parent
ROOT = DESKTOP.parent
TAURI = DESKTOP / "src-tauri"
UV_VERSION = os.environ.get("UV_VERSION", "0.13.9")

# Rust-Zielplattform -> Wheel-Plattform von uv auf PyPI
UV_PLATFORMS = {
    "x86_64-pc-windows-msvc": "win_amd64",
    "aarch64-pc-windows-msvc": "win_arm64",
    "x86_64-apple-darwin": "macosx_10_12_x86_64",
    "aarch64-apple-darwin": "macosx_11_0_arm64",
    "x86_64-unknown-linux-gnu": "manylinux_2_17_x86_64",
    "aarch64-unknown-linux-gnu": "manylinux_2_17_aarch64",
}
# Nicht mit in die App: Entwicklung, Tests, Doku, alte Installer
SKIP_PREFIXES = ("docs/", "tests/", "packaging/", "tools/", "rust/", ".github/", "desktop/")
SKIP_FILES = {".gitignore", "AGENTS.md"}


def host_target() -> str:
    out = subprocess.run(["rustc", "-vV"], capture_output=True, text=True, check=True).stdout
    return next(line.split(":", 1)[1].strip() for line in out.splitlines() if line.startswith("host:"))


def copy_app(dest: Path) -> int:
    if dest.exists():
        shutil.rmtree(dest)
    files = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout
    n = 0
    for name in files.decode("utf-8").split("\0"):
        if not name or name.startswith(SKIP_PREFIXES) or name in SKIP_FILES or not (ROOT / name).is_file():
            continue
        (dest / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest / name)
        n += 1
    return n


def fetch_uv(target: str) -> Path:
    plat = UV_PLATFORMS.get(target)
    if plat is None:
        sys.exit(f"Unbekannte Zielplattform {target!r}. Bekannt: {', '.join(UV_PLATFORMS)}")
    exe = ".exe" if "windows" in target else ""
    out = TAURI / "binaries" / f"uv-{target}{exe}"
    stamp = out.with_suffix(out.suffix + ".version")
    if out.exists() and stamp.exists() and stamp.read_text().strip() == UV_VERSION:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    cache = ROOT / "build" / "uv-cache" / plat
    cache.mkdir(parents=True, exist_ok=True)
    wheels = sorted(cache.glob(f"uv-{UV_VERSION}-*.whl"))
    if not wheels:
        subprocess.run([sys.executable, "-m", "pip", "download", "-q", f"uv=={UV_VERSION}", "--platform", plat,
                        "--only-binary=:all:", "--no-deps", "-d", str(cache)], check=True)
        wheels = sorted(cache.glob(f"uv-{UV_VERSION}-*.whl"))
    if not wheels:
        sys.exit(f"uv {UV_VERSION} für {plat} nicht gefunden")
    with zipfile.ZipFile(wheels[0]) as z:
        member = next(n for n in z.namelist() if n.endswith(f"/uv{exe}") and ".data/scripts/" in n)
        out.write_bytes(z.read(member))
        lic = "".join(f"\n\n===== {n.split('/')[-1]} =====\n\n" + z.read(n).decode("utf-8")
                      for n in z.namelist() if "/licenses/" in n)
    out.chmod(out.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    stamp.write_text(UV_VERSION)
    (TAURI / "resources" / "THIRD-PARTY-NOTICES.txt").parent.mkdir(parents=True, exist_ok=True)
    (TAURI / "resources" / "THIRD-PARTY-NOTICES.txt").write_text(
        "Meridian bündelt uv (https://github.com/astral-sh/uv), lizenziert unter MIT oder Apache-2.0.\n"
        "Python, PyTorch und die übrigen Pakete werden beim ersten Start heruntergeladen und stehen unter ihren\n"
        "eigenen Lizenzen." + lic, encoding="utf-8")
    return out


def copy_wheels(dest: Path) -> list[str]:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    src = Path(os.environ.get("MERIDIAN_WHEELS_DIR", ROOT / "build" / "wheels"))
    found = sorted(src.glob("meridian_fastbpe-*.whl")) if src.is_dir() else []
    for w in found:
        shutil.copy2(w, dest / w.name)
    # Platzhalter, damit der Ordner nie leer ist (Tauri verlangt existierende Ressourcen)
    (dest / "LIESMICH.txt").write_text("Optionale Python-Pakete, die beim Einrichten mitinstalliert werden.\n")
    return [w.name for w in found]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default=os.environ.get("TAURI_ENV_TARGET_TRIPLE") or None,
                    help="Rust-Zielplattform (Standard: dieses System)")
    args = ap.parse_args()
    target = args.target or host_target()
    print(f"» Meridian Desktop vorbereiten für {target}")
    n = copy_app(TAURI / "resources" / "app")
    print(f"  App-Dateien: {n}")
    uv = fetch_uv(target)
    print(f"  uv {UV_VERSION}: {uv.relative_to(DESKTOP)}")
    wheels = copy_wheels(TAURI / "resources" / "wheels")
    print("  Rust-Tokenizer: " + (", ".join(wheels) if wheels else
          "keiner (build/wheels leer) – die App nutzt den Python-Tokenizer"))


if __name__ == "__main__":
    main()
