"""Seite: Tempo messen und verbessern (Methode des „inference-optimisation“-Skills)."""
from __future__ import annotations

import html

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from meridianlm import bench
from meridianlm.trainer import RUNS_DIR, CheckpointError, list_runs, load_model
from ui.common import PALETTE, fmt_int, hero, next_step, store, style_fig

LEVERS = {
    "KV-Cache aus": ("use_cache", False),
    "KV-Cache an": ("use_cache", True),
    "no_grad statt inference_mode": ("grad_mode", "no_grad"),
    "1 Thread": ("threads", 1),
    "2 Threads": ("threads", 2),
    "4 Threads": ("threads", 4),
}
LABELS = {"decode_tok_s": "Decode (tok/s)", "ttft_ms": "TTFT (ms)"}


@st.cache_resource(show_spinner=False, max_entries=2)
def _load(run: str, mtime: float):
    return load_model(run)


@st.cache_data(show_spinner=False)
def _machine(device: str) -> dict:
    import torch
    d = torch.device(device)
    return {"bandwidth_gbs": bench.measure_bandwidth(d), "gflops": bench.measure_matmul(d)}


def _label(k: str) -> str:
    if k in LABELS:
        return LABELS[k]
    if k.startswith("prefill"):
        return f"Prefill {k[7:].split('_')[0]} (tok/s)"
    return k


def _metric_row(res: bench.BenchResult) -> None:
    m, c = res.metrics, res.ceilings
    cols = st.columns(4)
    cols[0].metric("Decode", f"{m['decode_tok_s']:,.0f} tok/s",
                   help="Tokens pro Sekunde beim Erzeugen, ein Token nach dem anderen (Median).")
    cols[1].metric("Anteil Obergrenze", f"{c['decode_pct']:.1f} %",
                   help=f"Obergrenze {c['decode_tok_s']:,.0f} tok/s = Bandbreite ÷ Bytes pro Token.")
    cols[2].metric("Zeit bis 1. Token", f"{m['ttft_ms']:.2f} ms")
    pre = [k for k in m if k.startswith("prefill")]
    cols[3].metric(f"Prefill {pre[-1][7:].split('_')[0]}", f"{m[pre[-1]]:,.0f} tok/s",
                   help=f"Obergrenze ≈ {c['prefill_tok_s']:,.0f} tok/s (50 % der gemessenen Rechenleistung).")


def render() -> None:
    hero("Tempo", "Wie schnell schreibt dein Modell – und wie nah ist das an der physikalischen Grenze? Messen, "
                  "eine Sache ändern, gegen Kontrollen prüfen, behalten oder verwerfen.")
    runs = list_runs()
    if not runs:
        st.info("Es gibt noch kein trainiertes Modell.", icon=":material/info:")
        next_step("train", "Zum Training")
        st.stop()

    trainer = store()["trainer"]
    busy = bool(trainer and trainer.is_running)
    names = [r["name"] for r in runs]
    run = st.selectbox("Modell", names, key="b_run")
    ckpt = RUNS_DIR / run / "model.pt"
    try:
        model, tok, _ = _load(run, ckpt.stat().st_mtime)
    except Exception as e:
        why = str(e) if isinstance(e, CheckpointError) else f"{type(e).__name__}: {e}"
        st.error(f"Modell konnte nicht geladen werden: {why}")
        st.stop()
    device = str(next(model.parameters()).device)
    if busy:
        st.warning("Gerade läuft ein Training. Messungen wären verfälscht (und würden das Training bremsen) – "
                   "warte, bis es fertig ist.", icon=":material/hourglass_top:")

    with st.expander("So wird gemessen", icon=":material/rule:"):
        st.markdown(
            "- **Decode** = Tokens pro Sekunde beim Erzeugen; **Prefill** = beim Einlesen eines Prompts; "
            "**TTFT** = Zeit bis zum ersten Token.\n"
            "- Jede Zahl ist der **Median aus mindestens 6 Messungen** – einzelne Läufe schwanken um ±5–10 %.\n"
            "- **Obergrenze Decode**: Pro Token müssen alle Gewichte einmal aus dem Speicher gelesen werden. "
            "Gemessene Bandbreite ÷ Bytes pro Token = mehr geht physikalisch nicht.\n"
            "- **A/B-Test**: Kontrolle – Kandidat – Kontrolle. Angenommen nur, wenn der Kandidat *beide* Kontrollen "
            "schlägt und nichts anderes um mehr als 3 % schlechter ist als beide Kontrollen.\n"
            "- Jede Messung landet im Ledger `runs/<modell>/bench.jsonl`. Ausführliches Beispiel: "
            "`docs/perf/PLAN.md`.")

    c1, c2, c3, c4 = st.columns(4)
    n_tok = c1.slider("Erzeugte Tokens", 16, 256, 96, step=16, key="b_n")
    samples = c2.slider("Messungen (Median)", 6, 20, 6, key="b_s")
    use_cache = c3.toggle("KV-Cache", value=True, key="b_cache",
                          help="Speichert Keys/Values, damit pro Token nur das neue Token gerechnet wird.")
    c4.markdown(f'<div class="ml-stats" style="margin-top:30px">gerät <b>{html.escape(device)}</b> · '
                f'{fmt_int(model.num_params())} param.</div>', unsafe_allow_html=True)
    kw = dict(n_tokens=n_tok, prompt_len=16, prefill_lengths=(32, 64, model.cfg.block_size), samples=samples)

    tab_measure, tab_ab, tab_ledger = st.tabs(["Messen", "A/B-Test", "Ledger"])
    ledger = RUNS_DIR / run / "bench.jsonl"

    with tab_measure:
        if st.button("Messen", type="primary", icon=":material/speed:", disabled=busy):
            with st.spinner("Messe Maschine und Modell …"):
                machine = _machine(device)
                cfg = bench.BenchConfig(name="KV-Cache an" if use_cache else "KV-Cache aus", use_cache=use_cache)
                res = bench.run(model, tok, cfg, machine=machine, **kw)
            bench.append_ledger(ledger, run, res)
            st.session_state["b_last"] = res
        res = st.session_state.get("b_last")
        if res is not None:
            if not res.ok:
                st.error(res.problem)
            _metric_row(res)
            c = res.ceilings
            fig = go.Figure()
            fig.add_bar(y=["Decode"], x=[res.metrics["decode_tok_s"]], orientation="h", name="gemessen",
                        marker_color=PALETTE["fg"])
            fig.add_bar(y=["Decode"], x=[max(0.0, c["decode_tok_s"] - res.metrics["decode_tok_s"])],
                        orientation="h", name="Luft bis zur Obergrenze", marker_color=PALETTE["grey"], opacity=0.35)
            fig.update_layout(barmode="stack", title="Decode im Verhältnis zur Obergrenze (tok/s)")
            st.plotly_chart(style_fig(fig, 200), width="stretch", key="b_ceiling")
            st.caption(f"Maschine: {c['bandwidth_gbs']} GB/s Speicher-Bandbreite · {c['gflops']} GFLOP/s · "
                       f"{c['bytes_per_token'] / 1e6:.2f} MB pro Token zu lesen. Bei kleinen Modellen bremst vor allem "
                       "der feste Aufwand pro Token (viele kleine Rechen-Aufrufe), nicht der Speicher.")

    with tab_ab:
        st.caption("Ändere genau **eine** Einstellung gegenüber der Kontrolle. Dauert etwa dreimal so lange wie eine "
                   "Messung.")
        a1, a2 = st.columns(2)
        lever = a1.selectbox("Kandidat", list(LEVERS), index=0, key="b_lever")
        target = a2.selectbox("Ziel-Kennzahl", ["decode_tok_s", "ttft_ms", f"prefill{model.cfg.block_size}_tok_s"],
                              format_func=_label, key="b_target")
        control = bench.BenchConfig(name="Kontrolle", use_cache=use_cache)
        key, val = LEVERS[lever]
        cand = bench.BenchConfig(**{**control.to_dict(), "name": lever, key: val})
        same = cand.to_dict() | {"name": ""} == control.to_dict() | {"name": ""}
        if same:
            st.info("Der Kandidat ist gleich wie die Kontrolle – wähle eine andere Änderung.", icon=":material/info:")
        if st.button("A/B-Test starten", type="primary", icon=":material/compare_arrows:", disabled=busy or same):
            with st.spinner("Kontrolle – Kandidat – Kontrolle …"):
                verdict, results = bench.ab_test(model, tok, control, cand, target=target, **kw)
            for r in (results[0], results[2]):
                bench.append_ledger(ledger, run, r)
            bench.append_ledger(ledger, run, results[1], verdict)
            st.session_state["b_ab"] = (verdict, results, lever)
        if st.session_state.get("b_ab"):
            verdict, results, lev = st.session_state["b_ab"]
            (st.success if verdict.accepted else st.warning)(
                f"**{'Angenommen' if verdict.accepted else 'Verworfen'}:** {lev} – {verdict.reason}",
                icon=":material/check_circle:" if verdict.accepted else ":material/block:")
            rows = []
            for label, r in zip(["Kontrolle", f"Kandidat: {lev}", "Kontrolle"], results):
                rows.append({"Lauf": label, **{_label(k): v for k, v in r.metrics.items()}})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    with tab_ledger:
        rows = bench.read_ledger(ledger)
        if not rows:
            st.caption("Noch keine Messungen für dieses Modell.")
        else:
            df = pd.DataFrame([{"Zeit": r["time"], "Konfiguration": r["config"].get("name", ""),
                                "Urteil": ("angenommen" if r["verdict"]["accepted"] else "verworfen")
                                if r.get("verdict") else "",
                                **{_label(k): v for k, v in r["metrics"].items()},
                                "% Obergrenze": r.get("ceilings", {}).get("decode_pct")} for r in rows])
            st.dataframe(df.iloc[::-1], hide_index=True, width="stretch")
            st.download_button("Ledger herunterladen (.jsonl)", ledger.read_bytes(), file_name=f"{run}-bench.jsonl",
                               icon=":material/download:")
