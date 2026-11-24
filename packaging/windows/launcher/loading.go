package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
)

// isLightTheme liest das in Meridian gewählte Theme, damit der Ladebildschirm dazu passt.
func isLightTheme(home string) bool {
	b, err := os.ReadFile(filepath.Join(home, "settings", "theme.json"))
	if err != nil {
		return false
	}
	var data struct {
		Active string                    `json:"active"`
		Custom map[string]map[string]any `json:"custom"`
	}
	if json.Unmarshal(b, &data) != nil {
		return false
	}
	if data.Active == "Hell" {
		return true
	}
	if c, ok := data.Custom[data.Active]; ok {
		base, _ := c["base"].(string)
		return base == "light"
	}
	return false
}

func loadingPage(light bool, logPath string) string {
	bg, fg, muted, border, surface := "#0a0a0a", "#e5e5e5", "#8a8a8a", "#262626", "#141414"
	if light {
		bg, fg, muted, border, surface = "#ffffff", "#171717", "#6b6b6b", "#e5e5e5", "#fafafa"
	}
	r := strings.NewReplacer("{bg}", bg, "{fg}", fg, "{muted}", muted, "{border}", border,
		"{surface}", surface, "{version}", version, "{log}", htmlEscape(logPath), "{mark}", markSVG)
	return r.Replace(loadingHTML)
}

func htmlEscape(s string) string {
	return strings.NewReplacer("&", "&amp;", "<", "&lt;", ">", "&gt;", `"`, "&quot;").Replace(s)
}

// Logo-Marke "M_" (assets/meridian-mark.svg), Farbe über currentColor.
const markSVG = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" width="44" height="44" fill="currentColor" aria-hidden="true"><rect x="2" y="4" width="3.4" height="25" rx="0.6"/><rect x="6.6" y="11" width="3.4" height="18" rx="0.6"/><rect x="11.2" y="18" width="3.4" height="11" rx="0.6"/><rect x="15.8" y="11" width="3.4" height="18" rx="0.6"/><rect x="20.4" y="4" width="3.4" height="25" rx="0.6"/><rect class="cur" x="25.5" y="24.5" width="4.5" height="4.5" rx="0.5"/></svg>`

const loadingHTML = `<!DOCTYPE html>
<html lang="de"><head><meta charset="utf-8"><title>Meridian</title>
<style>
  html,body{margin:0;height:100%;background:{bg};color:{fg};
    font-family:"Segoe UI Variable","Segoe UI",system-ui,sans-serif;-webkit-user-select:none;user-select:none}
  .wrap{height:100%;display:flex;align-items:center;justify-content:center}
  .card{width:min(560px,86vw)}
  .brand{display:flex;align-items:center;gap:14px;margin-bottom:28px}
  .brand b{font-size:26px;font-weight:600;letter-spacing:-.01em}
  .brand span{color:{muted};font:12px "Cascadia Mono",Consolas,monospace;border:1px solid {border};
    border-radius:4px;padding:1px 6px;margin-left:2px}
  .cur{animation:blink 1.1s steps(1) infinite}
  @keyframes blink{50%{opacity:0}}
  .step{font-size:15px;line-height:1.45;min-height:22px}
  .info{color:{muted};font-size:13px;margin-top:6px;min-height:18px}
  .bar{height:3px;background:{border};border-radius:2px;overflow:hidden;margin:18px 0 12px}
  .bar i{display:block;height:100%;width:30%;background:{fg};border-radius:2px;animation:slide 1.4s ease-in-out infinite}
  @keyframes slide{0%{transform:translateX(-100%)}100%{transform:translateX(340%)}}
  .detail{font:12px "Cascadia Mono",Consolas,monospace;color:{muted};white-space:nowrap;overflow:hidden;
    text-overflow:ellipsis;min-height:16px}
  .err{display:none;margin-top:8px;background:{surface};border:1px solid {border};border-radius:8px;padding:16px}
  .err p{margin:0 0 6px;font-size:14px;line-height:1.5}
  .err .msg{font:12px "Cascadia Mono",Consolas,monospace;color:{fg};white-space:pre-wrap;word-break:break-word}
  .err .hint{color:{muted};font-size:12px;margin-top:10px}
  .btns{display:flex;gap:8px;margin-top:14px;flex-wrap:wrap}
  button{font:13px "Segoe UI",system-ui,sans-serif;padding:7px 14px;border-radius:6px;cursor:pointer;
    border:1px solid {border};background:{bg};color:{fg}}
  button.primary{background:{fg};color:{bg};border-color:{fg}}
  button:hover{opacity:.85}
  .failed .bar,.failed .detail{display:none}
  .failed .err{display:block}
</style></head>
<body><div class="wrap"><div class="card" id="card">
  <div class="brand">{mark}<b>Meridian</b><span>v{version}</span></div>
  <div class="step" id="step">Starte …</div>
  <div class="info" id="info"></div>
  <div class="bar"><i></i></div>
  <div class="detail" id="detail"></div>
  <div class="err" id="err">
    <p><b>Meridian konnte nicht starten.</b></p>
    <div class="msg" id="msg"></div>
    <div class="hint">Protokoll: {log}</div>
    <div class="btns">
      <button class="primary" onclick="window.meridianReinstall()">Neu einrichten</button>
      <button onclick="window.meridianOpenLog()">Protokoll öffnen</button>
      <button onclick="window.meridianBrowser()">Im Browser öffnen</button>
    </div>
  </div>
</div></div>
<script>
  const $ = id => document.getElementById(id);
  window.setStep = t => { $("step").textContent = t; $("info").textContent = ""; $("detail").textContent = ""; };
  window.setInfo = t => { $("info").textContent = t; };
  window.setDetail = t => { $("detail").textContent = t; };
  window.setError = t => { $("card").classList.add("failed"); $("step").textContent = "Fehler"; $("msg").textContent = t; };
</script>
</body></html>`
