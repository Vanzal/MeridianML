# AGENTS.md – Übergabe für Coding-Agents (OMP, Claude Code, …)

Lies diese Datei zuerst. Sie fasst zusammen, was das Projekt ist, welche Regeln gelten, was erledigt ist und
was als Nächstes kommt.

## Projekt in einem Satz

**Meridian** (Repo `Vanzal/Meridian`, vorher `Vanzal/MagicAI-Studio`): ein kleines GPT-Sprachmodell, komplett selbst gebaut (BPE-Tokenizer,
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
8. Commits gehen direkt auf `main` von `Vanzal/Meridian`, wenn Vanzal es so will – sonst Branch.

## Starten

```bash
pip install -r requirements.txt          # PyTorch: für die GPU die CUDA-Variante von pytorch.org nehmen
streamlit run app.py                     # App
python -m pytest -q                      # Tests (~90, < 30 s auf CPU)
python train_cli.py --datasets tinystories=2000 --steps 500 --name demo
python generate_cli.py -m demo -p "Once upon a time" -n 120 --min-p 0.05
python bench_cli.py -m demo                                     # Tempo + Obergrenzen
python bench_cli.py -m demo --ab use_cache=false use_cache=true # A/B-Test
python tools/make_logo.py assets                                # Logo-Dateien neu erzeugen
```

Umgebungsvariablen: `MERIDIAN_DATA_DIR`, `MERIDIAN_RUNS_DIR`, `MERIDIAN_SETTINGS_DIR`, `HF_TOKEN`.
Installer bauen: `packaging/windows/build.sh` → `dist/Meridian-Setup-<version>.exe`. Der Launcher setzt die drei
`MERIDIAN_*`-Ordner auf `%LOCALAPPDATA%\Meridian\{data,runs,settings}`; die App darf nie in den Programmordner schreiben.
Version steht nur in `meridianlm/__init__.py` (Launcher: `version` in `main.go`, `meridian.rc` – beim Erhöhen mitziehen).

## Aufbau

| Pfad | Inhalt |
|---|---|
| `app.py` | Einstieg: Theme anwenden (+ einmal neu laden), Logo, Navigation, Seitenleiste |
| `meridianlm/tokenizer.py` | Byte-BPE, selbst geschrieben; `merge_counts` = Häufigkeit beim Verschmelzen |
| `meridianlm/model.py` | GPT (gelernte Positionen, LayerNorm, GELU, Weight Tying) + `KVCache` |
| `meridianlm/generate.py` | `stream_generate` (KV-Cache, `inference_mode`, inkrementeller UTF-8-Decoder), Temperatur/Top-k/Top-p/Min-p/Wiederholungs-Strafe |
| `meridianlm/trainer.py` | Training in einem Thread, atomare Checkpoints (`model.pt`, `run.json`), `list_runs`, `load_model` |
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
| `packaging/windows/` | Windows-Installer: `launcher/` (Go, `Meridian.exe`: richtet beim ersten Start mit uv Python 3.12 + PyTorch ein, CUDA 13.0 → 12.8 → CPU, startet Streamlit, öffnet Browser), `meridian.nsi` (NSIS, ohne Admin, nach `%LOCALAPPDATA%\Programs\Meridian`), `build.sh` |

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

- Windows-Installer + Startprogramm (gebaut und auf Linux getestet; auf echtem Windows noch **nicht** getestet).

## Als Nächstes (mit Vanzal abgesprochen, Reihenfolge vorgeschlagen)

1. **Auf der RTX 5080 prüfen** – Installer auf Windows durchspielen (Erst-Einrichtung, GPU-Erkennung), echte HF-Downloads, Token-Prüfung, Training mit „GPU-Maximum“,
   `bench_cli.py` auf CUDA. Ergebnisse in `docs/perf/PLAN.md` nachtragen (eigene Maschine, eigener Abschnitt).
2. **GPU-Tempo-Hebel** aus `docs/perf/PLAN.md` → „Nächste Hebel“: CUDA Graphs / `torch.compile(mode="reduce-overhead")`
   für den Decode-Schritt, Würfeln auf der GPU, Batch. Jeweils mit A/B und Gleichheits-Check.
3. **MoE (Mixture of Experts):** `GPTConfig` um `n_experts` und `top_k_experts` erweitern; MLP durch Router +
   Experten ersetzen (Top-k-Gating, Load-Balancing-Loss), mit `n_experts=0` = heutiges Verhalten. Modell-Seite:
   Regler + Diagramm; Parameter „gesamt“ vs. „aktiv pro Token“ zeigen; `estimate_params` und `ceilings()`
   (aktive Experten zählen) anpassen. Tests: gleiche Ausgabe bei `n_experts=0`, Gradienten fliessen, Balance-Loss.
4. **GPT-2 von Hugging Face laden:** nur GPT-2-Familie passt zur Architektur (gpt2, gpt2-medium, -large, -xl).
   Gewichte umbenennen/transponieren (Conv1D → Linear), GPT-2-Byte-BPE-Tokenizer anbinden (anderes Byte-Mapping
   und Vortokenisierung als unser Tokenizer). Neue Seite oder Bereich „Modell laden“. Llama/Qwen/Mistral gehen
   **nicht** ohne zweite Architektur – Vanzal so erklären.
5. **LoRA-Training:** LoRA-Adapter (Rang r, alpha) für `qkv`/`proj`/`fc`, Basisgewichte eingefroren,
   Adapter separat speichern und wieder einmischen. Vor allem für geladene GPT-2-Modelle.
6. **Rust-Tokenizer (optional):** BPE-Training/Encoding in Rust (PyO3 + maturin) mit Python-Fallback. Nur dort
   bringt Rust etwas – Training und Generieren laufen schon in C++/CUDA. Gleichheit mit dem Python-Tokenizer
   testen; Installation darf ohne Rust nicht brechen.

## Bekannte Grenzen

- Kontext ist eine gelernte Positions-Tabelle (`block_size`); längere Texte nur per Fenster.
- `ceilings()` geht von dichten Gewichten aus (bei MoE anpassen).
- Streamlit-Theme wird global gesetzt – mehrere Nutzer gleichzeitig teilen sich ein Theme (lokale App, ok).
