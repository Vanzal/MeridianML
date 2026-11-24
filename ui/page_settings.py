"""Seite: Einstellungen – Design (Themes) und Hugging-Face-Zugang."""
from __future__ import annotations

import html
import json

import streamlit as st

from magicllm import hf_auth
from ui import theme as th
from ui.common import hero

MARK = ('<rect x="2" y="4" width="3.4" height="25" rx="0.6"/><rect x="6.6" y="11" width="3.4" height="18" rx="0.6"/>'
        '<rect x="11.2" y="18" width="3.4" height="11" rx="0.6"/><rect x="15.8" y="11" width="3.4" height="18" rx="0.6"/>'
        '<rect x="20.4" y="4" width="3.4" height="25" rx="0.6"/><rect x="25.5" y="24.5" width="4.5" height="4.5" rx="0.5"/>')

FIELDS = [("bg", "Hintergrund"), ("surface", "Flächen"), ("border", "Rahmen"), ("sidebar", "Seitenleiste"),
          ("fg", "Text"), ("muted", "Text gedämpft"), ("primary", "Akzent (Hauptknöpfe)")]


def _switch(name: str) -> None:
    th.set_active(name)
    st.session_state.pop("_theme_sig", None)


def _preview(t: th.Theme) -> str:
    k = th.tokens(t)
    chips = "".join(f'<span style="display:inline-block;width:18px;height:18px;border-radius:5px;background:{c};'
                    f'margin-right:4px;border:1px solid {k["border"]}"></span>' for c in t.chart)
    bars = "".join(f'<div style="width:{w}%;height:8px;border-radius:3px;background:{c};margin:5px 0"></div>'
                   for w, c in zip((86, 64, 48, 30), t.chart))
    return f"""
<div style="background:{t.bg};border:1px solid {t.border};border-radius:12px;overflow:hidden;font-family:Inter,
     ui-sans-serif,system-ui,sans-serif;display:flex;min-height:230px">
  <div style="width:150px;background:{t.sidebar};border-right:1px solid {t.border};padding:14px">
    <div style="display:flex;gap:8px;align-items:center;margin-bottom:14px">
      <svg viewBox="0 0 32 32" width="20" height="20" fill="{t.fg}">{MARK}</svg>
      <b style="color:{t.fg};font-size:13px">Magic LLM</b></div>
    <div style="background:{k['surface2']};color:{t.fg};border-radius:6px;padding:4px 8px;font-size:12px">Start</div>
    <div style="color:{t.muted};padding:4px 8px;font-size:12px">Training</div>
    <div style="color:{t.muted};padding:4px 8px;font-size:12px">Generieren</div>
  </div>
  <div style="flex:1;padding:16px 18px">
    <div style="font:11px ui-monospace,monospace;color:{k['dim']}">magic-llm / vorschau</div>
    <div style="color:{t.fg};font-size:18px;font-weight:600;margin:4px 0 2px">Vorschau</div>
    <div style="color:{t.muted};font-size:12px;margin-bottom:12px">So sieht dein Theme aus.</div>
    <div style="display:flex;gap:10px;margin-bottom:12px">
      <div style="flex:1;background:{t.surface};border:1px solid {t.border};border-radius:8px;padding:8px 10px">
        <div style="font-size:9px;letter-spacing:.06em;color:{k['dim']}">VAL-LOSS</div>
        <div style="font:500 16px ui-monospace,monospace;color:{t.fg}">1.942</div></div>
      <div style="flex:2;background:{t.surface};border:1px solid {t.border};border-radius:8px;padding:6px 10px">
        {bars}</div>
    </div>
    <span style="background:{t.primary};color:{k['on_primary']};border-radius:7px;padding:6px 12px;font-size:12px;
          font-weight:500">Training starten</span>
    <span style="border:1px solid {k['border_strong']};color:{t.fg};border-radius:7px;padding:5px 12px;
          font-size:12px;margin-left:6px">Abbrechen</span>
    <div style="margin-top:12px">{chips}</div>
  </div>
</div>"""


def _design_tab() -> None:
    themes = th.all_themes()
    active = th.active_theme()
    st.markdown("#### Theme wählen")
    names = list(themes)
    pick = st.radio("Theme", names, index=names.index(active.name) if active.name in names else 0,
                    horizontal=True, label_visibility="collapsed", key="set_theme_pick")
    if pick != active.name:
        _switch(pick)
        st.rerun()

    base = themes[pick]
    st.markdown("#### Bearbeiten")
    st.caption("Ändere Farben und speichere das Ergebnis als eigenes Theme. Vorlagen bleiben unverändert.")
    left, right = st.columns([0.45, 0.55], gap="large")
    pre = f"ed_{pick}_"
    with left:
        is_custom = pick not in th.PRESETS
        name = st.text_input("Name", value=pick if is_custom else f"{pick} (eigenes)", key=pre + "name")
        mode = st.segmented_control("Grundton", ["Dunkel", "Hell"], default="Hell" if base.base == "light" else "Dunkel",
                                    key=pre + "base")
        cols = st.columns(2)
        vals = {}
        for i, (key, label) in enumerate(FIELDS):
            vals[key] = cols[i % 2].color_picker(label, value=getattr(base, key), key=pre + key)
        st.markdown("**Diagramm-Farben**")
        ccols = st.columns(len(base.chart))
        chart = [ccols[i].color_picker(f"{i + 1}", value=c, key=f"{pre}chart{i}", label_visibility="collapsed")
                 for i, c in enumerate(base.chart)]
    mode = mode or ("Hell" if base.base == "light" else "Dunkel")
    edited = th.Theme(name=name, base="light" if mode == "Hell" else "dark", chart=chart, **vals)
    with right:
        st.markdown(_preview(edited), unsafe_allow_html=True)
        k = th.tokens(edited)
        c_text, c_btn = th.contrast(edited.fg, edited.bg), th.contrast(k["on_primary"], edited.primary)
        c_muted = th.contrast(edited.muted, edited.bg)
        msg = (f"Kontrast Text/Hintergrund **{c_text:.1f}:1** · gedämpfter Text **{c_muted:.1f}:1** · "
               f"Knopf **{c_btn:.1f}:1**")
        if min(c_text, c_btn) < 4.5 or c_muted < 3:
            st.warning(msg + " – unter der Empfehlung (Text und Knöpfe ≥ 4,5:1, gedämpfter Text ≥ 3:1). "
                       "Schwer lesbar.", icon=":material/contrast:")
        else:
            st.caption(msg + " – gut lesbar.")

    b1, b2, b3 = st.columns(3)
    if b1.button("Speichern & anwenden", type="primary", icon=":material/check:", width="stretch"):
        try:
            saved = th.save_custom(edited)
        except ValueError as e:
            st.error(str(e))
        else:
            st.session_state.pop("_theme_sig", None)
            st.session_state.pop("set_theme_pick", None)
            st.toast(f"Theme „{saved}“ gespeichert.")
            st.rerun()
    b2.download_button("Exportieren (.json)", json.dumps(edited.to_dict(), ensure_ascii=False, indent=1),
                       file_name=f"magic-llm-theme-{edited.name.lower().replace(' ', '-')}.json",
                       mime="application/json", icon=":material/download:", width="stretch")
    if pick not in th.PRESETS:
        with b3.popover("Löschen", icon=":material/delete:", width="stretch"):
            st.write(f"Theme **{pick}** löschen?")
            if st.button("Ja, löschen", key="del_theme"):
                th.delete_custom(pick)
                st.session_state.pop("_theme_sig", None)
                st.session_state.pop("set_theme_pick", None)
                st.rerun()

    up = st.file_uploader("Theme importieren (.json)", type=["json"], key="theme_upload")
    if up is not None and st.button("Importieren", icon=":material/upload:"):
        try:
            t = th.Theme.from_dict(json.loads(up.getvalue().decode("utf-8")))
            saved = th.save_custom(t)
        except (ValueError, TypeError, json.JSONDecodeError, UnicodeDecodeError) as e:
            st.error(f"Ungültige Theme-Datei: {e}")
        else:
            st.session_state.pop("_theme_sig", None)
            st.session_state.pop("set_theme_pick", None)
            st.rerun()


def _hf_tab() -> None:
    st.markdown("#### Hugging-Face-Token")
    st.markdown(
        "Nötig für **gesperrte Datensätze** (z. B. manche Wikipedia- oder Web-Auszüge) und für höhere Abruf-Grenzen. "
        "Ein Token mit **Leserechten** reicht – erstelle ihn auf "
        "[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) mit Typ **Read**. Bei gesperrten "
        "Datensätzen musst du ausserdem einmal auf der Seite des Datensatzes die Bedingungen akzeptieren.")
    if msg := st.session_state.pop("hf_msg", None):
        kind, text = msg
        {"ok": st.success, "warn": st.warning, "err": st.error}[kind](text)
    src = hf_auth.source()
    if src == "umgebung":
        st.info(f"Aktiv: Token aus der Umgebungsvariable `HF_TOKEN` ({hf_auth.mask(hf_auth.active_token())}). "
                "Er hat Vorrang vor einem hier gespeicherten Token.", icon=":material/key:")
    elif src == "datei":
        st.success(f"Gespeichert: {hf_auth.mask(hf_auth.stored_token())} (in `settings/hf_token`, nur für dich "
                   "lesbar, nicht in Git).", icon=":material/key:")
    else:
        st.caption("Noch kein Token hinterlegt – öffentliche Datensätze funktionieren trotzdem.")

    with st.form("hf_form", clear_on_submit=True):
        token = st.text_input("Token", type="password", placeholder="hf_…",
                              help="Wird nur auf deinem Rechner gespeichert und nur an huggingface.co geschickt.")
        check = st.checkbox("Bei Hugging Face prüfen", value=True,
                            help="Fragt huggingface.co, wem der Token gehört und ob er nur Leserechte hat.")
        sent = st.form_submit_button("Speichern", type="primary", icon=":material/save:")
    if sent:
        token = token.strip()
        if not hf_auth.looks_valid(token):
            st.error("Das sieht nicht wie ein Hugging-Face-Token aus. Er beginnt mit „hf_“ und ist etwa 37 Zeichen lang.")
        else:
            result = hf_auth.verify(token) if check else None
            if result is not None and not result.ok and "kennt diesen Token nicht" in result.message:
                st.error(result.message + " Nicht gespeichert.")
            else:
                hf_auth.save(token)
                if result is None:
                    msg = ("ok", f"Gespeichert ({hf_auth.mask(token)}), ungeprüft.")
                elif not result.ok:
                    msg = ("warn", f"Gespeichert ({hf_auth.mask(token)}), aber nicht geprüft: {result.message}")
                elif result.read_only:
                    msg = ("ok", f"Gespeichert. Gehört **{html.escape(result.user)}**, Rechte: "
                                 f"{'nur lesen' if result.role == 'read' else 'fein eingestellt'}.")
                else:
                    msg = ("warn", f"Gespeichert. Gehört **{html.escape(result.user)}**, aber der Token darf auch "
                                   "**schreiben**. Für diese App reicht ein Lese-Token – sicherer wäre, einen Token vom "
                                   "Typ „Read“ zu erstellen und diesen hier zu verwenden.")
                st.session_state["hf_msg"] = msg
                st.rerun()
    if hf_auth.stored_token():
        c1, c2 = st.columns(2)
        if c1.button("Gespeicherten Token prüfen", icon=":material/verified_user:", width="stretch"):
            r = hf_auth.verify(hf_auth.stored_token() or "")
            if r.ok:
                rights = {"read": "nur lesen", "fineGrained": "fein eingestellt", "write": "lesen und schreiben"}
                st.success(f"Gültig. Gehört **{html.escape(r.user)}**, Rechte: {rights.get(r.role, r.role or '?')}.")
            else:
                st.error(r.message)
        if c2.button("Token entfernen", icon=":material/key_off:", width="stretch"):
            hf_auth.remove()
            st.rerun()


def render() -> None:
    hero("Einstellungen", "Aussehen der App und Zugang zu Hugging Face.")
    tab_design, tab_hf = st.tabs(["Design", "Hugging Face"])
    with tab_design:
        _design_tab()
    with tab_hf:
        _hf_tab()
