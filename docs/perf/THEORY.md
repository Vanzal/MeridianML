# Obergrenzen (Ceilings)

Gerechnet für das Demo-Modell `lauf-1` (876'416 Parameter, 4 Schichten, Breite 128, Kontext 128, float32)
auf der Test-Maschine (2-Kern-Cloud-VM, nur CPU). `bench_cli.py` rechnet das für jedes Modell und jede
Maschine selbst aus.

| Grösse | Gemessen / gerechnet |
|---|---|
| Speicher-Bandbreite (Kopie von 256 MB) | ≈ 18 GB/s |
| Matrix-Multiplikation (1024², float32) | ≈ 245 GFLOP/s |
| Gewichte | 876'416 × 4 B = 3,5 MB |
| KV-Cache bei 112 Tokens | 2 × 4 Schichten × 112 × 128 × 4 B = 0,46 MB |

## Decode (ein Token nach dem anderen) – durch die Bandbreite begrenzt

Pro erzeugtem Token müssen alle Gewichte einmal gelesen werden (das Token-Embedding ist gleichzeitig die
Ausgabe-Schicht und wird dort voll gelesen) plus der KV-Cache bis zur aktuellen Länge:

    bytes/token ≈ 3,5 MB + 0,46 MB ≈ 3,96 MB
    Obergrenze  ≈ 18 GB/s ÷ 3,96 MB ≈ 4'500 tok/s

Einschränkung: 3,5 MB passen fast in den CPU-Cache. Die echte Grenze liegt bei diesem Mini-Modell also eher
*höher*; die Prozentzahl ist eine vorsichtige Schätzung.

## Prefill (Prompt einlesen) – durch die Rechenleistung begrenzt

    FLOPs/token ≈ 2 × Parameter + 4 × Schichten × Breite × Kontext ≈ 1,98 MFLOP
    praktisch   ≈ 50 % × 245 GFLOP/s ÷ 1,98 MFLOP ≈ 62'000 tok/s

Gemessen: ≈ 45'000 tok/s bei 128 Tokens (≈ 72 %). Hier ist wenig zu holen.

## TTFT

Ein Vorwärtsdurchlauf über den Prompt + Tokenisieren + erstes Würfeln. Gemessen ≈ 1,5 ms – das ist bereits
nahe am Boden; kein Hebel lohnt sich.

## Warum Decode bei ~29 % stehen bleibt

Bei einem so kleinen Modell dauert ein Token ≈ 0,76 ms, die reine Speicherzeit wäre ≈ 0,22 ms. Der Rest ist
Verwaltungsaufwand pro Token: ~40 kleine PyTorch-Aufrufe (je einige µs), das Würfeln auf der CPU und
Python selbst. Grössere Modelle sind näher an der Grenze, weil dieser feste Aufwand im Verhältnis kleiner wird.
