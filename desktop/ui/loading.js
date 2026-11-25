// Ladebildschirm: zeigt den Fortschritt, den die App (src-tauri/src/launcher.rs) als Ereignis "progress" schickt.
const $ = (id) => document.getElementById(id);
const tauri = window.__TAURI__;

function show(s) {
  if (s.step !== undefined) { $("step").textContent = s.step; }
  if (s.info !== undefined) { $("info").textContent = s.info; }
  if (s.detail !== undefined) { $("detail").textContent = s.detail; }
  if (s.error) {
    $("card").classList.add("failed");
    $("step").textContent = "Fehler";
    $("msg").textContent = s.error;
    for (const b of document.querySelectorAll("button")) { b.disabled = false; }
  }
}

async function call(cmd) {
  for (const b of document.querySelectorAll("button")) { b.disabled = true; }
  try {
    await tauri.core.invoke(cmd);
  } catch (e) {
    show({ error: String(e) });
  }
  if (cmd !== "open_log") {
    $("card").classList.remove("failed");
  } else {
    for (const b of document.querySelectorAll("button")) { b.disabled = false; }
  }
}

async function init() {
  $("btn-retry").addEventListener("click", () => call("restart"));
  $("btn-reinstall").addEventListener("click", () => call("reinstall"));
  $("btn-log").addEventListener("click", () => call("open_log"));
  if (!tauri) { show({ error: "Tauri-Schnittstelle fehlt." }); return; }
  await tauri.event.listen("progress", (e) => show(e.payload));
  // Stand abholen, falls schon Meldungen kamen, bevor die Seite geladen war
  const s = await tauri.core.invoke("loading_info");
  document.documentElement.classList.toggle("light", s.light);
  $("version").textContent = "v" + s.version;
  $("log").textContent = s.log;
  show({ step: s.step, info: s.info });
  if (s.error) { show({ error: s.error }); }
}
init();
