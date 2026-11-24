"""Startseite: Überblick über das Projekt und die Pipeline."""
from __future__ import annotations

import streamlit as st

from magicllm.catalog import CATALOG
from magicllm.data import list_prepared
from magicllm.trainer import list_runs
from ui.common import hero, next_step, themed_dot

PIPELINE_DOT = """
digraph G {
  rankdir=TB; bgcolor="transparent"; nodesep=0.4; ranksep=0.7;
  node [shape=box, style="rounded,filled", fillcolor="#141414", color="#333333", fontcolor="#E5E5E5",
        fontname="Helvetica", fontsize=12, margin="0.18,0.1", penwidth=1.2];
  edge [color="#525252", penwidth=1.2, arrowsize=0.7];

  hf   [label="Hugging Face\\nDatensätze"];
  raw  [label="Rohtexte\\n(Dokumente)"];
  bpe  [label="BPE-Tokenizer\\n(selbst gebaut)", color="#60A5FA"];
  ids  [label="Token-IDs\\n[17, 305, 8, ...]"];
  batch[label="Batches\\n(zufällige Fenster)"];
  gpt  [label="GPT-Transformer\\n(selbst gebaut)", color="#FAFAFA", penwidth=2];
  loss [label="Loss\\n(Cross-Entropy)"];
  opt  [label="Backprop +\\nAdamW", color="#34D399"];
  ckpt [label="Checkpoint\\nmodel.pt"];
  gen  [label="Text erzeugen\\n(Sampling)", color="#F59E0B"];

  // Obere Reihe: Daten vorbereiten (links -> rechts)
  { rank=same; hf -> raw -> bpe -> ids -> batch; }
  // Untere Reihe: lernen und nutzen (rechts -> links, deshalb dir=back)
  { rank=same; gen -> ckpt [dir=back]; ckpt -> opt [dir=back]; opt -> loss [dir=back]; loss -> gpt [dir=back]; }
  batch -> gpt;
  opt -> gpt [style=dashed, constraint=false, label="  Gewichte verbessern", fontcolor="#737373", fontsize=10];
}
"""

STEPS = [
    ("01", "Daten", f"Wähle aus {len(CATALOG)} Hugging-Face-Datensätzen (Geschichten, Wikipedia, Web, Nachrichten, "
                  "Code, Chat …) und mische sie nach Belieben."),
    ("02", "Tokenizer", "Der Text wird in Teilstücke (Tokens) zerlegt. Der eigene BPE-Tokenizer lernt selbst, welche "
                      "Buchstabenfolgen häufig sind."),
    ("03", "Modell", "Ein GPT-Transformer mit einstellbarer Grösse – von 0,3 bis über 200 Millionen Parameter."),
    ("04", "Training", "Das Modell rät das nächste Token, bekommt seinen Fehler (Loss) gezeigt und verbessert sich. "
                     "Du siehst live Kurven, Beispieltexte und Aufmerksamkeit."),
    ("05", "Generieren", "Gib einen Anfang vor – das Modell schreibt weiter. Regle Temperatur, Top-k, Top-p, Min-p und Wiederholungs-Strafe."),
]


def render() -> None:
    hero("Magic LLM",
         "Ein eigenes Sprachmodell – von Grund auf gebaut, mit Hugging-Face-Daten trainiert und live sichtbar gemacht.",
         crumb="start", logo=True)

    prepared, runs = list_prepared(), list_runs()
    best = min((r["best_val"] for r in runs if r.get("best_val")), default=None)
    c = st.columns(4)
    c[0].metric("Datensätze im Katalog", len(CATALOG))
    c[1].metric("Vorbereitete Daten", len(prepared))
    c[2].metric("Trainierte Modelle", len(runs))
    c[3].metric("Bester Val-Loss", f"{best:.3f}" if best is not None else "–")

    next_step("data", "Los geht's: Daten auswählen")

    st.markdown(
        '<div class="ml-term" style="margin-top:18px">'
        '<span class="p">$</span> <span class="c">streamlit run app.py</span><br>'
        '<span class="p">#</span> oder ohne Oberfläche, wie llama.cpp:<br>'
        '<span class="p">$</span> <span class="c">python train_cli.py --datasets tinystories=2000 --steps 500 --name demo</span><br>'
        '<span class="p">$</span> <span class="c">python generate_cli.py -m demo -p "Once upon a time" -n 120 --min-p 0.05</span>'
        '</div>', unsafe_allow_html=True)

    st.subheader("So fliesst ein Text durch das System")
    st.graphviz_chart(themed_dot(PIPELINE_DOT), width="stretch")

    st.subheader("Die Schritte in der App")
    cols = st.columns(len(STEPS))
    for col, (num, title, text) in zip(cols, STEPS):
        col.markdown(f'<div class="step-card"><span class="n">{num}</span><b>{title}</b>'
                     f'<span class="muted">{text}</span></div>', unsafe_allow_html=True)

    st.write("")
    with st.expander("Wie lernt ein Sprachmodell eigentlich?", expanded=False):
        st.markdown(
            """
**Die Aufgabe ist erstaunlich einfach:** Dem Modell wird ein Textstück gezeigt, und es muss das **nächste Token**
vorhersagen. Aus dem Satz *„Die Katze sitzt auf der“* soll z. B. *„Matte“* werden.

1. Das Modell gibt für **jedes mögliche Token** eine Wahrscheinlichkeit aus.
2. Der **Loss** misst, wie wenig Wahrscheinlichkeit es dem richtigen Token gegeben hat (Cross-Entropy).
3. Per **Backpropagation** wird berechnet, welches der Millionen Gewichte wie stark am Fehler schuld war.
4. **AdamW** verschiebt alle Gewichte ein kleines Stück in die bessere Richtung.
5. Nach Tausenden Wiederholungen lernt das Modell Buchstaben → Wörter → Grammatik → Zusammenhänge.

Der **Loss** startet bei ca. ln(Vokabulargrösse) (reines Raten) und sinkt dann. Die **Perplexity** `e^Loss`
sagt, zwischen wie vielen Tokens das Modell im Schnitt „schwankt“.
"""
        )
    with st.expander("Realistische Erwartungen"):
        st.markdown(
            "Dieses Projekt läuft auf einem normalen Laptop (CPU). Die Modelle haben Millionen statt Milliarden "
            "Parameter – sie lernen Wörter, Satzbau und kurze Geschichten, **aber keine Weltkenntnisse** wie ChatGPT. "
            "Am besten klappt es mit **TinyStories**: Dort entstehen schon nach wenigen Minuten lesbare Sätze."
        )
