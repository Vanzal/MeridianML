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
