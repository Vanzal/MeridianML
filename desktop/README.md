# Meridian Desktop (Tauri)

Die Desktop-App von Meridian: ein natives Fenster ([Tauri 2](https://tauri.app), Rust), das beim ersten Start
Python + PyTorch einrichtet (mit [uv](https://github.com/astral-sh/uv)), die Streamlit-App von Meridian im
Hintergrund startet und sie im Fenster zeigt. Fenster zu = alles beendet.

## Ablauf

1. Fenster öffnet sich mit dem Ladebildschirm (`ui/index.html`).
2. `src-tauri/src/launcher.rs` sucht die App-Dateien und uv, prüft den Fingerabdruck der Einrichtung
   (`setup-ok.txt`: requirements.txt, Version, CPU/GPU, mitgelieferte Wheels) und richtet bei Bedarf ein:
   Python 3.12 → PyTorch (NVIDIA: CUDA 13.0 → 12.8 → CPU; ohne NVIDIA: CPU; Mac: Standard mit MPS;
   pytorch.org gesperrt → PyPI) → `requirements.txt` → optional Rust-Tokenizer.
3. Startet Streamlit auf `127.0.0.1` (erster freier Port ab 8501), wartet auf `/_stcore/health` und lädt die
   Seite ins Fenster.
4. Überwacht Streamlit. Stürzt es ab, kommt der Ladebildschirm mit Meldung und den Knöpfen Neu starten,
   Neu einrichten und Protokoll öffnen zurück.
5. Beim Beenden wird Streamlit samt Unterprozessen gestoppt. Zusätzlich hält die App die Eingabe (stdin) von
   Streamlit offen; endet die App unerwartet (Absturz, Task-Manager), schliesst das Betriebssystem die Eingabe
   und Streamlit beendet sich selbst. Es bleibt also nie ein Server im Hintergrund hängen.

Externe Links (Hugging Face usw.) öffnet der normale Browser (`src/init.js` + `on_navigation` in `src/lib.rs`),
Downloads aus der App landen im Download-Ordner. Ein zweiter Start holt nur das offene Fenster nach vorne.

## Bauen

Braucht: Rust (stable), Node.js 18+, Python 3, unter Linux zusätzlich
`libwebkit2gtk-4.1-dev librsvg2-dev` (siehe [Tauri-Voraussetzungen](https://tauri.app/start/prerequisites/)).

```bash
cd desktop
npm install
npm run stage      # App-Dateien, uv (Sidecar) und Wheels nach src-tauri/ kopieren – vor dev/build nötig
npm run dev        # Entwicklung: nimmt den Code direkt aus dem Repo (Änderungen gelten nach Neustart)
npm run build      # Installer: src-tauri/target/release/bundle/ (Windows .exe, macOS .dmg, Linux .deb/.AppImage)
cd src-tauri && cargo test   # Tests des Startprogramms
```

Rust-Tokenizer mit einpacken: vorher `maturin build --release -m rust/meridian_fastbpe/Cargo.toml -o build/wheels`.
Fertige Installer für alle drei Systeme baut GitHub Actions (`.github/workflows/desktop.yml`), bei einem Tag `v*`
hängen sie am Release.

## Dateien

| Pfad | Inhalt |
|---|---|
| `ui/` | Ladebildschirm (HTML/CSS/JS, Farben wie die Themes Dunkel/Hell) |
| `src-tauri/src/lib.rs` | Fenster, externe Links, Downloads, Einzel-Instanz, Beenden |
| `src-tauri/src/launcher.rs` | Einrichten mit uv, Streamlit starten/überwachen/beenden, Befehle für den Ladebildschirm |
| `src-tauri/src/init.js` | Läuft in jeder Seite: leitet externe Links an den Browser weiter |
| `src-tauri/tauri.conf.json` | Name, Version, Icons, Installer (NSIS ohne Admin, WebView2 wird bei Bedarf nachgeladen) |
| `scripts/stage.py` | Sammelt App-Dateien, lädt uv für die Zielplattform, kopiert Wheels |

Die Version steht in `meridianlm/__init__.py`, `src-tauri/tauri.conf.json`, `src-tauri/Cargo.toml` und
`package.json` – ein Test (`tests/test_meridian.py`) prüft, dass alle gleich sind.
