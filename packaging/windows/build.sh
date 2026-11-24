#!/usr/bin/env bash
# Baut den Windows-Installer dist/Meridian-Setup-<version>.exe – auf Linux (oder in GitHub Actions).
#
# Braucht: go (>= 1.22), makensis (NSIS 3), x86_64-w64-mingw32-windres (binutils-mingw-w64), python3 + pip, git.
#   Ubuntu/Debian:  sudo apt-get install nsis binutils-mingw-w64-x86-64
#
# Was drin ist:
#   Meridian.exe   kleines Startprogramm (Go, Quelle in launcher/): eigenes App-Fenster (WebView2), richtet beim
#                  ersten Start Python + PyTorch ein
#   uv.exe         Paket- und Python-Manager von Astral (lädt Python und PyTorch beim ersten Start)
#   app\           der Code von Meridian (ohne Tests und Doku)
#   wheels\        optional: meridian_fastbpe (Rust-Tokenizer) als Windows-Wheel, falls in build/wheels vorhanden
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
VERSION="$(python3 -c "import re,sys;print(re.search(r'__version__ = \"(.+?)\"', open(sys.argv[1]).read()).group(1))" "$ROOT/meridianlm/__init__.py")"
UV_VERSION="${UV_VERSION:-0.13.9}"
BUILD="$ROOT/build/windows"
STAGE="$BUILD/stage"
DIST="$ROOT/dist"
OUT="$DIST/Meridian-Setup-$VERSION.exe"

echo "» Meridian $VERSION – Windows-Installer bauen"
rm -rf "$STAGE"
mkdir -p "$STAGE/app" "$DIST" "$BUILD/cache"

echo "» App-Dateien sammeln"
cd "$ROOT"
git ls-files -z | while IFS= read -r -d '' f; do
  case "$f" in
    docs/*|tests/*|packaging/*|tools/*|rust/*|.github/*|.gitignore|AGENTS.md) continue ;;
  esac
  mkdir -p "$STAGE/app/$(dirname "$f")"
  cp "$f" "$STAGE/app/$f"
done

echo "» Startprogramm bauen (Meridian.exe)"
cd "$HERE/launcher"
x86_64-w64-mingw32-windres --preprocessor=cpp --preprocessor-arg=-xc-header -O coff -i meridian.rc -o rsrc_windows_amd64.syso
# -H windowsgui: Fenster-Programm ohne Konsole (Meridian öffnet sich im eigenen App-Fenster, WebView2).
# Ohne go.work fehlende go.sum-Einträge (golang.org/x/sys) automatisch ergänzen und gegen sum.golang.org prüfen.
if [ -z "${GOWORK:-}" ]; then export GOFLAGS="${GOFLAGS:+$GOFLAGS }-mod=mod"; fi
GOOS=windows GOARCH=amd64 CGO_ENABLED=0 go build -trimpath -ldflags "-s -w -H windowsgui" -o "$STAGE/Meridian.exe" .
cp "$HERE/build-assets/meridian.ico" "$STAGE/meridian.ico"

echo "» uv $UV_VERSION für Windows holen"
WHEEL="$BUILD/cache/uv-$UV_VERSION-py3-none-win_amd64.whl"
if [ ! -f "$WHEEL" ]; then
  python3 -m pip download -q "uv==$UV_VERSION" --platform win_amd64 --only-binary=:all: --no-deps -d "$BUILD/cache"
fi
python3 - "$WHEEL" "$STAGE" <<'PY'
import sys, zipfile
whl, stage = sys.argv[1], sys.argv[2]
z = zipfile.ZipFile(whl)
exe = next(n for n in z.namelist() if n.endswith("/uv.exe"))
open(f"{stage}/uv.exe", "wb").write(z.read(exe))
lic = "".join(f"\n\n===== {n.split('/')[-1]} =====\n\n" + z.read(n).decode("utf-8")
              for n in z.namelist() if "/licenses/" in n)
open(f"{stage}/THIRD-PARTY-NOTICES.txt", "w", encoding="utf-8").write(
    "Meridian bündelt uv (https://github.com/astral-sh/uv), lizenziert unter MIT oder Apache-2.0.\n"
    "Python, PyTorch und die übrigen Pakete werden beim ersten Start heruntergeladen und stehen unter ihren\n"
    "eigenen Lizenzen." + lic)
PY

echo "» Schneller Tokenizer (Rust) für Windows"
WHEEL_DIR="${MERIDIAN_WHEELS_DIR:-$ROOT/build/wheels}"
NSIS_WHEELS=()
shopt -s nullglob
WIN_WHEELS=("$WHEEL_DIR"/meridian_fastbpe-*-win_amd64.whl)
shopt -u nullglob
if [ ${#WIN_WHEELS[@]} -gt 0 ]; then
  mkdir -p "$STAGE/wheels"
  cp "${WIN_WHEELS[@]}" "$STAGE/wheels/"
  NSIS_WHEELS=(-DHAVE_WHEELS=1)
  echo "   mit Rust-Kern: $(basename "${WIN_WHEELS[0]}")"
else
  echo "   kein Windows-Wheel gefunden ($WHEEL_DIR) – Installer nutzt den Python-Tokenizer."
  echo "   Wheel bauen (auf Windows oder in GitHub Actions): maturin build --release -m rust/meridian_fastbpe/Cargo.toml"
fi

echo "» Installer bauen"
makensis -V2 -DVERSION="$VERSION" -DSTAGE="$STAGE" -DOUTFILE="$OUT" "${NSIS_WHEELS[@]}" "$HERE/meridian.nsi"
ls -la "$OUT"
sha256sum "$OUT" | tee "$OUT.sha256"
echo "» Fertig: $OUT"
