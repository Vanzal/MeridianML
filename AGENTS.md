# AGENTS.md – Übergabe für Coding-Agents (OMP, Claude Code, …)

Lies diese Datei zuerst. Sie fasst zusammen, was das Projekt ist, welche Regeln gelten, was erledigt ist und
was als Nächstes kommt.

## Projekt in einem Satz

**Meridian** (Repo `Vanzal/MeridianML`, vorher `Vanzal/MagicAI-Studio`): ein kleines GPT-Sprachmodell, komplett selbst gebaut (BPE-Tokenizer,
Transformer, Training, Sampling) in PyTorch, mit einer Streamlit-App, die jeden Schritt live sichtbar macht, plus
Kommandozeilen-Tools im Stil von llama.cpp.

## Besitzer

Vanzal (Noah). Lernt gerade programmieren: Erklärungen kurz und verständlich halten, Fachbegriffe beim ersten
Mal in einem Halbsatz erklären. Hat eine **RTX 5080** – GPU-Tests auf seinem Rechner sind möglich.

## Regeln (nicht brechen)

1. **Keine Emojis** in App, README und Doku. Icons über Material-Symbole (`:material/name:`), Status über
   CSS-Punkte (`dot()` in `ui/common.py`). `tests/test_meridian.py::test_no_emojis_in_user_facing_files` prüft das.
2. **Oberfläche auf Deutsch** (Schweizer Schreibweise: „ss“ statt „ß“ in UI-Texten). Code-Kommentare ebenfalls
   Deutsch, wie im Rest des Projekts.
3. **Name ist „Meridian“.** Alte Projektnamen dürfen in README, UI und dieser Datei nicht vorkommen (Test
   `test_branding_is_meridian`). Der Copyright-Hinweis in `LICENSE` (MIT, Joel Weber) bleibt – das verlangt die Lizenz.
   Python-Paket heisst `meridianlm` (nicht `meridian`: das ist Googles Meridian-Bibliothek).
4. **Design:** neutrales Schwarz/Weiss wie die llama.cpp-WebUI, Terminal-Akzente wie omp. Farben kommen **nur**
   aus dem aktiven Theme (`ui/theme.py` → `T`, `PALETTE`, `COLORS` in `ui/common.py`). Keine festen Hex-Farben in
   Seiten; Graphviz-Diagramme über `themed_dot()`, Heatmaps über `heat_scale()`.
5. **Tempo-Änderungen nur mit Messung** nach `docs/perf/PLAN.md`: eine Änderung pro Experiment,
   Kontrolle – Kandidat – Kontrolle, Median aus ≥ 6, Urteil + Grund ins Log. Verlustfrei heisst: gieriger
   Ausgabe-Vergleich (`bench.greedy_equal`) muss bestehen.
6. **Secrets:** Der Hugging-Face-Token liegt in `settings/hf_token` (Rechte 600, in .gitignore). Nie ausgeben,
   loggen oder committen; nur maskiert anzeigen (`hf_auth.mask`).
7. **Vor jedem Commit:** `python -m pytest -q` muss grün sein. Bei UI-Änderungen die App starten und die
   betroffenen Seiten anschauen (hell **und** dunkel).
8. Commits gehen direkt auf `main` von `Vanzal/MeridianML`, wenn Vanzal es so will – sonst Branch.

## Starten

```bash
pip install -r requirements.txt          # PyTorch: für die GPU die CUDA-Variante von pytorch.org nehmen
streamlit run app.py                     # App
python -m pytest -q                      # Tests (~180, < 30 s auf CPU)
python train_cli.py --datasets tinystories=2000 --steps 500 --name demo
python generate_cli.py -m demo -p "Once upon a time" -n 120 --min-p 0.05
python bench_cli.py -m demo                                     # Tempo + Obergrenzen
python bench_cli.py -m demo --ab use_cache=false use_cache=true # A/B-Test
python tools/make_logo.py assets                                # Logo-Dateien neu erzeugen
```

Umgebungsvariablen: `MERIDIAN_DATA_DIR`, `MERIDIAN_RUNS_DIR`, `MERIDIAN_SETTINGS_DIR`, `HF_TOKEN`.
**Desktop-App (Tauri, Hauptweg):** `cd desktop && npm install && npm run stage && npm run dev` (Entwicklung) bzw.
`npm run build` (Installer für das eigene System). Alle drei Systeme baut `.github/workflows/desktop.yml`; bei einem
Tag `v*` hängen die Installer am Release. Details: `desktop/README.md`. Tests des Startprogramms:
`cd desktop/src-tauri && cargo test` (vorher `npm run stage`, weil Tauri die Ressourcen beim Kompilieren prüft).
Datenordner: Windows `%LOCALAPPDATA%\Meridian`, macOS `~/Library/Application Support/Meridian`, Linux
`~/.local/share/Meridian` (darin `venv/`, `data/`, `runs/`, `settings/`, `meridian.log`).
Alter Windows-Installer (Go, nur noch von Hand): `packaging/windows/build.sh` → `dist/Meridian-Setup-<version>.exe`. Der Launcher setzt die drei
`MERIDIAN_*`-Ordner auf `%LOCALAPPDATA%\Meridian\{data,runs,settings}`; die App darf nie in den Programmordner schreiben.
Version steht in `meridianlm/__init__.py`; beim Erhöhen mitziehen: `desktop/src-tauri/tauri.conf.json`,
`desktop/src-tauri/Cargo.toml`, `desktop/package.json` (Test `test_desktop_app_version_matches`), alter Launcher:
`version` in `main.go`, `meridian.rc`.

## Aufbau

| Pfad | Inhalt |
|---|---|
| `app.py` | Einstieg: Theme anwenden (+ einmal neu laden), Logo, Navigation, Seitenleiste |
| `meridianlm/tokenizer.py` | Byte-BPE, selbst geschrieben; `merge_counts` = Häufigkeit beim Verschmelzen |
| `meridianlm/model.py` | GPT (gelernte Positionen, LayerNorm, GELU, Weight Tying) + `KVCache` + `MoE`; `GPTConfig.validate()` prüft alle Einstellungen (ValueError mit deutscher Meldung) |
| `meridianlm/generate.py` | `stream_generate` (KV-Cache, `inference_mode`, inkrementeller UTF-8-Decoder), Temperatur/Top-k/Top-p/Min-p/Wiederholungs-Strafe |
| `meridianlm/trainer.py` | Training in einem Thread, atomare Checkpoints (`model.pt`, `run.json`), `list_runs`, `load_model`; `CHECKPOINT_FORMAT` (jetzt 2, ohne Feld = 1) + `check_checkpoint` → `CheckpointError` mit verständlicher Meldung |
| `meridianlm/data.py`, `catalog.py`, `custom.py` | 41 HF-Datensätze per Streaming, Mischen, Vorbereiten |
| `meridianlm/bench.py` | Messungen, Obergrenzen, A/B mit Urteil, Ledger |
| `meridianlm/hf_auth.py` | HF-Token speichern/prüfen (`whoami`), setzt `HF_TOKEN` beim Import von `catalog` |
| `ui/common.py` | Design-System (CSS mit `--ml-*`-Variablen), `hero()`, `style_fig()`, Theme-Helfer |
| `ui/theme.py` | Presets Dunkel/Hell/Terminal, eigene Themes, `apply_to_streamlit()` (setzt `theme.*` zur Laufzeit) |
| `ui/page_*.py` | Seiten: start, data, tokenizer, model, train, generate, runs, bench (Tempo), settings |
| `train_cli.py`, `generate_cli.py`, `bench_cli.py` | Kommandozeile (wie llama.cpp: train / llama-cli / llama-bench) |
| `docs/perf/` | Tempo-Log (`PLAN.md`), Theorie (`THEORY.md`), `ledger.jsonl`, `configs/`, `best-config.json` |
| `docs/logo/concepts/` | 8 Logo-Konzepte + Vorschau; gewählt wurde „Token M prompt“ (`M_`) |
| `assets/` | finales Logo (Marke, Logo hell/dunkel, App-Icon); Wortmarke als Pfade (Inter Display SemiBold, OFL) |
| `tests/test_core.py`, `tests/test_meridian.py` | Tests |
| `rust/meridian_fastbpe/` | Rust-Kern (PyO3, abi3) für BPE-Training + Encoding; `tokenizer.py` nutzt ihn, wenn installiert (`backend()`), sonst Python. `MERIDIAN_PURE_PYTHON=1` erzwingt Python. Muss **bit-genau** gleich bleiben: `tests/test_meridian.py` (Rust = Python) + `tools/check_tokenizer_equal.py <korpus>` |
| `desktop/` | **Desktop-App (Tauri 2, Rust)** für Windows/macOS/Linux. `src-tauri/src/launcher.rs`: Einrichten mit uv (Sidecar), Streamlit starten (Wächter über stdin: stirbt die App, beendet sich Streamlit selbst), Health-Check, ins Fenster laden, überwachen, beim Schliessen beenden. `src-tauri/src/lib.rs`: Fenster, externe Links → Browser, Downloads → Download-Ordner, Einzel-Instanz. `ui/`: Ladebildschirm. `scripts/stage.py`: App-Dateien, uv und Wheels nach `src-tauri/` kopieren (vor jedem Build) |
| `packaging/windows/` | Alter Windows-Installer (von `desktop/` abgelöst): `launcher/` (Go, `Meridian.exe`, Fenster-Programm `-H windowsgui`: eigenes App-Fenster über WebView2 (`github.com/jchv/go-webview2`, reines Go), Ladebildschirm mit Fortschritt (`loading.go`), richtet beim ersten Start mit uv Python 3.12 + PyTorch ein, CUDA 13.0 → 12.8 → CPU, startet Streamlit ohne Konsolenfenster und zeigt es im Fenster; Fenster zu = alles beenden; Protokoll `%LOCALAPPDATA%\Meridian\meridian.log`; ohne WebView2 oder mit `--browser` alter Konsolen-+-Browser-Modus), `meridian.nsi` (NSIS, ohne Admin, nach `%LOCALAPPDATA%\Programs\Meridian`), `build.sh` |

Gut zu wissen:
- Die App liest das Theme bei jedem Lauf aus `settings/theme.json`; `apply_to_streamlit` ändert Streamlits
  globale Config (`streamlit.config.set_option`) – privates API, getestet mit Streamlit 1.67.
- `stream_generate` liefert `(token_id, text, probs, raw_probs)`; das End-Token liefert `""` als Text.
- Kontext voll → das Fenster der letzten `block_size` Tokens wird neu gerechnet (identisch zu ohne Cache).
- Auf der Entwicklungs-Maschine war Hugging Face gesperrt: Screenshots stammen von einem offline erzeugten
  Korpus. Echte Downloads und die Token-Prüfung bei HF wurden noch **nicht** live getestet.

## Erledigt

- Rebrand zu Meridian, neues Design, keine Emojis, Logo (`M_`), README ohne alte Projektnamen.
- Fehler behoben: Umlaute beim Streamen, halb geschriebene Checkpoints, kaputte Lauf-Ordner, GPU-Absturz im
  Gewichts-Histogramm, unsichere Lauf-Namen, Löschen ohne Bestätigung, falsche Merge-Zahlen.
- Min-p, Wiederholungs-Strafe, Stopp am Text-Ende, tok/s-Statistik, Download von Text/Checkpoints.
- KV-Cache + `inference_mode`: Decode ≈ 430 → 1'317 tok/s (CPU, 2 Kerne), verlustfrei.
- Tempo-Seite, `bench_cli.py`, Ledger.
- Heller Modus, eigene Themes (Editor, Vorschau, Kontrast-Check, Import/Export).
- Hugging-Face-Token mit Leserechten (Einstellungen → Hugging Face).

- Rust-Tokenizer: Training ≈ 3,6× schneller, identische Ausgabe; Fehler „_ geht verloren“ im Regex behoben.
- Windows-Installer + Startprogramm (gebaut und auf Linux getestet; auf echtem Windows noch **nicht** getestet).
- Eigenes App-Fenster statt Browser (WebView2). Auf echtem Windows noch **nicht** getestet. Lokal bauen ohne
  Internet-Zugang zu golang.org: `go.work` mit `replace golang.org/x/sys => <Kopie von github.com/golang/sys>`;
  CI ergänzt go.sum mit `GOFLAGS=-mod=mod`.
- Prüfung der Modell-Einstellungen, Checkpoint-Format-Version, Prüfung beim Laden (Format, Konfiguration,
  Gewichte, Tokenizer-Vokabular). Alte Checkpoints (ohne `format`) laden weiterhin.
- **MoE:** `n_experts` (0 = aus, 2–64), `top_k_experts`, `moe_aux_weight` (Balance-Loss, Switch-Formel, 1.0 =
  gleichmässig). Kein Token wird verworfen. `n_experts=0` liefert bit-genau dasselbe wie vorher (getestet). Loss
  im Training = CE + Gewicht × Balance; angezeigt wird nur die CE (`forward(..., return_aux=True)`). Auslastung
  pro Experte: `GPT.expert_usage()` (nur im Training gezählt), landet in `evals[*]["experts"]`. `active_params()`
  zählt nur k Experten; `ceilings()` nutzt das. CPU-Messung in `docs/perf/PLAN.md` (≈ 0,4–0,5× von dicht).
- **Desktop-App mit Tauri** (`desktop/`): ersetzt das Go-Startprogramm als Hauptweg, läuft auf Windows, macOS
  und Linux. Auf Linux (Xvfb) komplett getestet: Erst-Einrichtung, App im Fenster, Fehler-Bildschirm bei Absturz,
  Beenden beim Schliessen, kein verwaister Server nach `kill -9`, `.deb` aus dem Release-Build. Auf Windows und
  macOS noch **nicht** getestet. Fällt download.pytorch.org aus, wird PyTorch von PyPI geholt.
- Rauchtest der Seiten Modell + Training (hell/dunkel, mit/ohne MoE) über `streamlit.testing` (`tests/test_pages.py`).

## Als Nächstes (mit Vanzal abgesprochen, Reihenfolge vorgeschlagen)

1. **Auf der RTX 5080 prüfen** – Desktop-App (`desktop.yml`-Installer) auf Windows durchspielen (Erst-Einrichtung, GPU-Erkennung), echte HF-Downloads, Token-Prüfung, Training mit „GPU-Maximum“,
   `bench_cli.py` auf CUDA. Ergebnisse in `docs/perf/PLAN.md` nachtragen (eigene Maschine, eigener Abschnitt).
2. **GPU-Tempo-Hebel** aus `docs/perf/PLAN.md` → „Nächste Hebel“: CUDA Graphs / `torch.compile(mode="reduce-overhead")`
   für den Decode-Schritt, Würfeln auf der GPU, Batch. Jeweils mit A/B und Gleichheits-Check.
3. **MoE auf der GPU schneller machen:** Experten als gestapelte Gewichte + grouped GEMM statt Python-Schleife,
   `torch.compile` des MoE-Blocks; Exp. 8 aus `docs/perf/PLAN.md` auf der GPU wiederholen. Immer mit A/B.
4. **GPT-2 von Hugging Face laden:** nur GPT-2-Familie passt zur Architektur (gpt2, gpt2-medium, -large, -xl).
   Gewichte umbenennen/transponieren (Conv1D → Linear), GPT-2-Byte-BPE-Tokenizer anbinden (anderes Byte-Mapping
   und Vortokenisierung als unser Tokenizer). Neue Seite oder Bereich „Modell laden“. Llama/Qwen/Mistral gehen
   **nicht** ohne zweite Architektur – Vanzal so erklären.
5. **LoRA-Training:** LoRA-Adapter (Rang r, alpha) für `qkv`/`proj`/`fc`, Basisgewichte eingefroren,
   Adapter separat speichern und wieder einmischen. Vor allem für geladene GPT-2-Modelle.
6. **Rust-Tokenizer weiter:** erledigt bis auf die Vor-Zerlegung (Regex, bleibt Python). Windows-Wheel baut nur
   GitHub Actions (`windows-installer.yml`, Job `wheel-windows`); lokal gebaute Installer haben ihn nur, wenn ein
   `meridian_fastbpe-*-win_amd64.whl` in `build/wheels/` liegt.

## Bekannte Grenzen

- Kontext ist eine gelernte Positions-Tabelle (`block_size`); längere Texte nur per Fenster.
- MoE verteilt die Tokens mit einer Python-Schleife über die Experten: auf der CPU spürbar langsamer als dicht.
- Streamlit-Theme wird global gesetzt – mehrere Nutzer gleichzeitig teilen sich ein Theme (lokale App, ok).
