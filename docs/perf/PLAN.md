# Tempo-Optimierung – Experiment-Log

Methode: „inference-optimisation“-Skill (sybil-solutions/inference-optimisation-skill). Eine Änderung pro
Experiment, Kontrolle – Kandidat – Kontrolle, Median aus ≥ 6 (meist 12) Messungen. Angenommen nur, wenn der
Kandidat beide Kontrollen beim Ziel schlägt und keine andere Kennzahl > 3 % schlechter ist als beide Kontrollen.
Rohdaten: `ledger.jsonl` (eine Zeile pro Messung). Konfigurationen: `configs/`. Beste: `best-config.json`.

Maschine: 2-Kern-Cloud-VM, nur CPU, torch 2.15, Python 3.13. Modell: `lauf-1` (876k Parameter, Kontext 128).
Messung: 96 erzeugte Tokens nach 16 Prompt-Tokens; Prefill bei 32/64/128 Tokens.

## Ergebnis

| Kennzahl | Grundlinie | Jetzt | Obergrenze | Wie gemessen |
|---|---|---|---|---|
| Decode (c1) | ≈ 430 tok/s (376–497) | **1'317 tok/s** | ≈ 4'600 tok/s | Median aus 12, warm |
| TTFT | ≈ 1,7 ms | 1,5 ms | – | Median aus 12, warm |
| Prefill 128 | ≈ 45'000 tok/s | ≈ 45'000 tok/s | ≈ 62'000 tok/s | Median aus 12, ≥ 30 ms pro Messung |

Decode ≈ **3× schneller**, Ausgabe verlustfrei gleich (gieriger Vergleich über 5 Prompts × 200 Tokens,
inklusive Fenster-Verschiebung über 128 Tokens hinaus).

## Experimente

| # | Änderung | Ziel | Ergebnis | Urteil | Grund |
|---|---|---|---|---|---|
| 1 | KV-Cache | decode | 2,9× | verworfen | Prefill 36–60 % langsamer: Prefill lief über die nicht zusammenhängende Cache-Ansicht |
| 2 | KV-Cache, Prefill direkt | decode | 3,0× | verworfen | Prefill 32 noch −27 % |
| – | Mikromessung Prefill 32 | – | Cache und ohne Cache innerhalb 3 % | – | „Verlust“ war Mess-Rauschen: eine Prefill-Messung dauerte nur ~1 ms → Harness misst jetzt ≥ 30 ms pro Messung |
| 3 | KV-Cache (neue Harness) | decode | 2,2× | verworfen | Prefill 128 −5,5 % gegenüber dem Mittel der Kontrollen – aber die zwei Kontrollen selbst lagen bis 33 % auseinander |
| – | Regel geschärft | – | – | – | Verschlechterung zählt nur, wenn schlechter als *beide* Kontrollen (Streuung der Kontrollen = Rauschen) |
| 4 | KV-Cache, 12 Messungen | decode | **2,41×** | **angenommen** | keine Kennzahl schlechter als beide Kontrollen |
| 5 | `torch.inference_mode` statt `no_grad` | decode | 1,15× | angenommen (knapp) | → wiederholen, weil nahe am Rauschen |
| 6 | 1 Thread statt 2 | decode | 1,00× | verworfen | innerhalb Rauschen |
| 7 | `inference_mode` wiederholt | decode | **1,43×** | **angenommen** | bestätigt |

## STATUS (Plateau auf dieser Maschine)

- Beste vs. Grundlinie vs. Obergrenze: 1'317 vs. ≈ 430 vs. ≈ 4'600 tok/s (≈ 29 % der Obergrenze).
- Was bremst: fester Aufwand pro Token (≈ 40 kleine PyTorch-Aufrufe + Würfeln auf der CPU + Python), nicht
  die Bandbreite. Belege: 0,76 ms pro Token gegenüber 0,22 ms reiner Lesezeit; ein Thread statt zwei ändert
  nichts (Exp. 6).
- Nicht verifiziert: alles auf GPU (keine Grafikkarte auf der Test-Maschine).

### Nächste Hebel (nach Kosten sortiert)

| Hebel | Erwartung | Aufwand | Hinweis |
|---|---|---|---|
| CUDA Graphs für den Decode-Schritt (GPU) | oft ~2× bei c1 | mittel | KV-Cache ist schon fest vorbelegt – Voraussetzung erfüllt |
| `torch.compile(mode="reduce-overhead")` für Decode (GPU) | 1,5–2× | gering | erster Aufruf langsam (Kompilieren); Cache persistent halten |
| Würfeln auf der GPU lassen | spart eine Synchronisation pro Token | gering | ändert, welche Zufallszahlen ein Seed liefert |
| Mehrere Texte gleichzeitig (Batch) | Gesamtdurchsatz × Batch | mittel | Cache unterstützt schon `batch > 1` |
| Fenster beim vollen Kontext um ½ verschieben statt jedes Token neu rechnen | grosser Gewinn bei langen Texten | gering | **verlustbehaftet** (andere Positionen) – nur als Option |
| bfloat16-Gewichte | ~2× bei grossen Modellen auf GPU | gering | **verlustbehaftet**, Gleichheits-Check entfällt |


---

# Tokenizer in Rust (`rust/meridian_fastbpe`)

Ziel: BPE-Training und -Encoding schneller, **bei exakt gleichem Ergebnis**. Die Vor-Zerlegung in Wörter (Regex)
bleibt in Python, weil Pythons `\w`/`\s` Unicode anders einteilt als Rusts Regex-Bibliothek – nur die Schleifen
wandern nach Rust (PyO3, eine abi3-Wheel für Python ≥ 3.10).

Korpus: 16 MB gemischt (Python-Standardbibliothek als Code, Lizenztexte als Prosa, deutsches README), 1'203 Dokumente.
Maschine: 2-Kern-Cloud-VM, Python 3.13. Skripte: `tools/bench_tokenizer.py`, `tools/check_tokenizer_equal.py`.
Rohdaten: `tokenizer-rust-ab.json`.

## Profil vorher (Python)

| Schritt | Zeit |
|---|---|
| Vor-Zerlegung + Zählen (Regex) | 0,9 s |
| Merge-Schleife, Vokabular 4'096 | 6,8 s |
| Merge-Schleife, Vokabular 16'384 | 9,3 s |
| Encoding 16 MB (davon Regex) | 1,9 s (1,2 s) |

→ Training ist der Hebel; beim Encoding kann Rust höchstens den Nicht-Regex-Teil (~⅓) beschleunigen.

## Gleichheits-Gate

Rust und Python liefern **identische Merges, Merge-Zahlen und Token-IDs**: Korpus bei Vokabular 300 / 4'096 /
16'384 / 32'768, 200 zufällige Unicode-Texte (Umlaute, Emoji mit Hautton, Flaggen, CJK, kombinierende Akzente,
Sonder-Leerzeichen) und Randfälle (leer, ein Zeichen, Wiederholungen, nur Leerraum). Gleichstände werden wie im
Python-Heap aufgelöst: grösste Häufigkeit, dann kleinstes Paar `(a, b)`.

**Nebenbei gefundener Fehler (auch in der Python-Version):** Der Vor-Zerlegungs-Regex hatte keine Gruppe für „_“ –
Unterstriche gingen beim Encoding **verloren** (`snake_case` → `snakecase`, `__init__` → `init`). Für die Code-Datensätze
heisst das: das Modell hat nie einen Unterstrich gesehen. Behoben; geprüft für alle 1,1 Mio. Unicode-Zeichen
(`decode(encode(text)) == text`). Bereits vorbereitete Datensätze enthalten weiterhin keine „_“ – neu vorbereiten.

## A/B (Kontrolle Python – Kandidat Rust – Kontrolle Python, Median aus 6)

| Kennzahl | Python | Rust | Faktor | Urteil |
|---|---|---|---|---|
| Training, Vokabular 4'096 | 6,9 s | **1,7 s** | **3,7×** | angenommen |
| Training, Vokabular 16'384 | 10,9 s | **3,0 s** | **3,6×** | angenommen |
| Encoding, neuer Text (kalter Cache) | 1,59 s | 1,35 s | 1,16× | angenommen (knapp) |
| Encoding, bekannter Text (warmer Cache) | 1,25 s | 1,36 s | 0,9× | **verworfen**: Übergabe jeder Wortliste an Rust kostet mehr als Pythons Dict-Treffer |

Daraufhin **Hybrid**: Python behält seinen Wort-Cache, nur unbekannte Wörter gehen an Rust.

| Kennzahl | Python | Rust-Hybrid | Faktor |
|---|---|---|---|
| Encoding, kalter Cache | 1,63 / 1,58 s | **1,19 s** | **1,33×** |
| Encoding, warmer Cache (abwechselnd gemessen, 12×) | 1,134 s | 1,122 s | gleich (gleicher Code-Pfad; erste Messung 12 % langsamer war Rauschen) |

## Stand

Training ≈ 3,6× schneller, Encoding neuer Texte ≈ 1,3× schneller, nichts langsamer, Ausgabe identisch. Was jetzt
bremst: die Regex-Vor-Zerlegung in Python (0,9 s beim Training, 1,2 s beim Encoding von 16 MB). Nächster Hebel wäre
eine eigene, zu Python **bit-genau** passende Vor-Zerlegung in Rust (Unicode-Klassen wie Pythons `str.isalnum`) –
nur mit Gleichheits-Test über alle Code-Points.

---

# Mixture of Experts (MoE)

Gleiche Maschine (2-Kern-CPU). Modell: 4 Schichten, Breite 128, Kontext 128, Vokabular 521. Dicht: 876k Parameter.
MoE 8 Experten, top-2: 4,57 Mio. Parameter, davon 1,41 Mio. aktiv pro Token (1,6× so viel Rechnen wie dicht).
Abwechselnd gemessen (dicht – MoE – dicht – MoE – dicht), je Median aus 6. Rohdaten: `moe-vs-dense.json`.

| Kennzahl | Dicht | MoE 8×2 | Faktor |
|---|---|---|---|
| Decode (tok/s) | 798 / 920 / 1'413 | 468 / 582 | ≈ 0,5× |
| Prefill 128 (tok/s) | 41'899 / 44'490 / 45'731 | 17'150 / 17'228 | ≈ 0,39× |
| Training (tok/s, Batch 16) | 21'953 / 26'411 / 28'476 | 9'635 / 10'764 | ≈ 0,39× |

Erwartet wären ≈ 0,62× (1,6× mehr aktive Rechnung). Der Rest ist das Verteilen der Tokens auf die Experten
(Python-Schleife über 8 Experten, `nonzero`, `index_add_`): auf einer kleinen CPU teurer als die Rechnung selbst.
Darum zeigt die App auf der CPU den Hinweis „MoE lohnt sich vor allem auf der GPU“. Auf der RTX 5080 **noch nicht gemessen**.

| # | Änderung | Ziel | Ergebnis | Urteil | Grund |
|---|---|---|---|---|---|
| 8 | MoE: Schleife nur über Experten, die Tokens bekommen (`chosen.unique()`) statt über alle 8 | decode | 571 / 661 gegen Kontrollen 677 / 650 / 530 | verworfen | erster Kandidat schlechter als beide Kontrollen, zweiter besser – innerhalb Rauschen (Kontrollen streuen ±30 %). Ausgabe identisch. Rohdaten: `moe-ab-unique.json`. Auf GPU nochmals prüfen: dort spart es Synchronisationen. |

Nächste MoE-Hebel (GPU): alle Experten als ein gestapelter Tensor + `torch.bmm` / grouped GEMM statt Python-Schleife;
`torch.compile` des MoE-Blocks; Tokens nach Experte sortieren statt `nonzero` pro Experte.
