//! Startprogramm von Meridian: richtet beim ersten Start Python + PyTorch ein (mit uv), startet die Streamlit-App
//! auf einem freien Port und zeigt sie im App-Fenster. Fenster zu = alles beenden.
//!
//! Ordner (gleich wie beim alten Windows-Startprogramm, eine bestehende Einrichtung wird weiterbenutzt):
//!   Windows: %LOCALAPPDATA%\Meridian      macOS: ~/Library/Application Support/Meridian
//!   Linux:   ~/.local/share/Meridian
//!   darin:   venv/ (Python-Umgebung, ~3–5 GB mit PyTorch), data/, runs/, settings/, meridian.log

use std::fs::{self, File};
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, Instant};

use serde::Serialize;
use sha2::{Digest, Sha256};
use tauri::{AppHandle, Emitter, Manager, Url};
use tauri_plugin_opener::OpenerExt;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
const PYTHON_VERSION: &str = "3.12";
/// Reihenfolge, in der PyTorch mit GPU-Unterstützung versucht wird (RTX 50xx braucht CUDA 12.8 oder neuer).
const CUDA_INDEXES: &[&str] = &["https://download.pytorch.org/whl/cu130", "https://download.pytorch.org/whl/cu128"];
/// PyTorch nur für die CPU (unter Linux sonst mehrere GB CUDA-Pakete, die ohne NVIDIA-Karte nichts bringen).
const CPU_INDEX: &str = "https://download.pytorch.org/whl/cpu";
const START_TIMEOUT: Duration = Duration::from_secs(180);
/// Startet Streamlit mit einem Wächter: Endet die Eingabe (stdin), weil die App beendet wurde oder abgestürzt ist,
/// beendet sich Python sofort. So bleibt nie ein verwaister Streamlit-Server zurück, auf keinem System.
const STREAMLIT_BOOT: &str = r#"import os, sys, threading
def _watch():
    try:
        while os.read(0, 1024):
            pass
    except Exception:
        pass
    os._exit(0)
threading.Thread(target=_watch, name="meridian-watchdog", daemon=True).start()
sys.argv[0] = "streamlit"
from streamlit.web import cli
sys.exit(cli.main())
"#;

/// Startoptionen: `--reinstall` (Umgebung neu aufbauen), `--cpu` (PyTorch ohne Grafikkarte), `--port N`.
#[derive(Clone, Debug, PartialEq)]
pub struct Options {
    pub reinstall: bool,
    pub cpu: bool,
    pub port: u16,
}

impl Default for Options {
    fn default() -> Self {
        Options { reinstall: false, cpu: false, port: 8501 }
    }
}

impl Options {
    pub fn from_args(args: impl IntoIterator<Item = String>) -> Self {
        let mut o = Options::default();
        let mut it = args.into_iter();
        while let Some(a) = it.next() {
            match a.as_str() {
                "--reinstall" => o.reinstall = true,
                "--cpu" => o.cpu = true,
                "--port" => {
                    if let Some(p) = it.next().and_then(|v| v.parse().ok()) {
                        o.port = p;
                    }
                }
                _ => {
                    if let Some(p) = a.strip_prefix("--port=").and_then(|v| v.parse().ok()) {
                        o.port = p;
                    }
                }
            }
        }
        if std::env::var("MERIDIAN_CPU").is_ok_and(|v| v == "1") {
            o.cpu = true;
        }
        o
    }
}

/// Was der Ladebildschirm gerade anzeigt.
#[derive(Default, Clone)]
struct Status {
    step: String,
    info: String,
    error: Option<String>,
}

/// Nachricht an den Ladebildschirm (Ereignis "progress"). Leere Felder werden nicht geschickt.
#[derive(Serialize, Clone, Default)]
struct Progress {
    #[serde(skip_serializing_if = "Option::is_none")]
    step: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    info: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    detail: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

struct Paths {
    app: PathBuf,
    uv: PathBuf,
    wheels: Vec<PathBuf>,
    home: PathBuf,
    venv: PathBuf,
    python: PathBuf,
    marker: PathBuf,
}

/// Gemeinsamer Zustand (von Tauri verwaltet, `app.state::<Launcher>()`).
pub struct Launcher {
    opts: Mutex<Options>,
    home: PathBuf,
    status: Mutex<Status>,
    child: Mutex<Option<Child>>,
    /// Offene Eingabe von Streamlit (Wächter, siehe STREAMLIT_BOOT). Schliessen = Streamlit beendet sich.
    child_stdin: Mutex<Option<ChildStdin>>,
    /// Jeder Start bekommt eine neue Nummer. Ältere Abläufe merken daran, dass sie aufhören sollen.
    run_id: AtomicU64,
    quitting: AtomicBool,
    log: Mutex<Option<File>>,
    last_detail: Mutex<Option<Instant>>,
    loading_url: Mutex<Option<Url>>,
}

impl Launcher {
    pub fn new(opts: Options, home: PathBuf) -> Self {
        let _ = fs::create_dir_all(&home);
        let log = File::create(home.join("meridian.log")).ok();
        Launcher {
            opts: Mutex::new(opts),
            home,
            status: Mutex::new(Status::default()),
            child: Mutex::new(None),
            child_stdin: Mutex::new(None),
            run_id: AtomicU64::new(0),
            quitting: AtomicBool::new(false),
            log: Mutex::new(log),
            last_detail: Mutex::new(None),
            loading_url: Mutex::new(None),
        }
    }

    pub fn log_path(&self) -> PathBuf {
        self.home.join("meridian.log")
    }

    pub fn set_loading_url(&self, url: Url) {
        *self.loading_url.lock().unwrap() = Some(url);
    }

    fn current(&self, id: u64) -> bool {
        self.run_id.load(Ordering::SeqCst) == id && !self.quitting.load(Ordering::SeqCst)
    }

    fn log(&self, line: &str) {
        if let Some(f) = self.log.lock().unwrap().as_mut() {
            let _ = writeln!(f, "{line}");
            let _ = f.flush();
        }
    }

    fn step(&self, h: &AppHandle, msg: &str) {
        self.log(&format!("» {msg}"));
        {
            let mut s = self.status.lock().unwrap();
            s.step = msg.to_string();
            s.info.clear();
        }
        let _ = h.emit("progress", Progress { step: Some(msg.into()), info: Some(String::new()),
                                               detail: Some(String::new()), ..Default::default() });
    }

    fn info(&self, h: &AppHandle, msg: &str) {
        self.log(&format!("  {msg}"));
        self.status.lock().unwrap().info = msg.to_string();
        let _ = h.emit("progress", Progress { info: Some(msg.into()), ..Default::default() });
    }

    /// Letzte Ausgabezeile von uv/Python: immer ins Protokoll, höchstens ~8x pro Sekunde ins Fenster.
    fn detail(&self, h: &AppHandle, line: &str) {
        self.log(&format!("    {line}"));
        let mut last = self.last_detail.lock().unwrap();
        if last.is_some_and(|t| t.elapsed() < Duration::from_millis(120)) {
            return;
        }
        *last = Some(Instant::now());
        let _ = h.emit("progress", Progress { detail: Some(line.into()), ..Default::default() });
    }

    /// Fehler anzeigen: zurück zum Ladebildschirm (falls die App schon offen war) mit Meldung und Knöpfen.
    fn fail(&self, h: &AppHandle, msg: &str) {
        self.log(&format!("FEHLER: {msg}"));
        self.status.lock().unwrap().error = Some(msg.to_string());
        let loading = self.loading_url.lock().unwrap().clone();
        if let (Some(win), Some(loading)) = (h.get_webview_window("main"), loading) {
            let on_loading = win.url().is_ok_and(|u| u.scheme() == loading.scheme() && u.host() == loading.host());
            if !on_loading {
                let _ = win.navigate(loading); // die Seite holt den Fehler über loading_info ab
                return;
            }
        }
        let _ = h.emit("progress", Progress { error: Some(msg.into()), ..Default::default() });
    }

    /// Laufenden Kind-Prozess (uv oder Streamlit) samt Unterprozessen beenden.
    fn stop_child(&self) {
        self.child_stdin.lock().unwrap().take();
        let child = self.child.lock().unwrap().take();
        if let Some(mut c) = child {
            self.log(&format!("Beende Prozess {}", c.id()));
            kill_tree(&mut c);
        }
    }

    /// Beim Beenden der App: alles stoppen, was noch läuft.
    pub fn shutdown(&self) {
        self.quitting.store(true, Ordering::SeqCst);
        self.run_id.fetch_add(1, Ordering::SeqCst);
        self.stop_child();
    }

    fn find_paths(&self, h: &AppHandle) -> Result<Paths, String> {
        let res = h.path().resource_dir().ok();
        let bundled = res.as_ref().map(|r| r.join("app")).filter(|a| a.join("app.py").is_file());
        // Entwicklung (`tauri dev`): direkt den Code im Repository nehmen, damit Änderungen sofort gelten.
        let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("..");
        let app = if cfg!(debug_assertions) && repo.join("app.py").is_file() {
            repo.canonicalize().unwrap_or(repo)
        } else {
            bundled.ok_or_else(|| "Programmdateien von Meridian (app.py) nicht gefunden. Bitte neu installieren.".to_string())?
        };
        let uv = find_uv().ok_or("uv nicht gefunden. Bitte Meridian neu installieren.")?;
        let mut wheels: Vec<PathBuf> = res
            .map(|r| r.join("wheels"))
            .and_then(|d| fs::read_dir(d).ok())
            .map(|rd| rd.flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "whl")).collect())
            .unwrap_or_default();
        wheels.sort();
        let home = self.home.clone();
        let venv = home.join("venv");
        let python = if cfg!(windows) { venv.join("Scripts").join("python.exe") } else { venv.join("bin").join("python") };
        Ok(Paths { app, uv, wheels, marker: home.join("setup-ok.txt"), home, venv, python })
    }

    /// Ein kompletter Start: (Einrichten) → Streamlit starten → warten → im Fenster zeigen → überwachen.
    fn run(&self, h: &AppHandle, id: u64, reinstall: bool) -> Result<(), String> {
        let opts = {
            let mut o = self.opts.lock().unwrap();
            let snapshot = o.clone();
            o.reinstall = false; // --reinstall gilt nur für den ersten Start
            snapshot
        };
        self.step(h, "Starte …");
        let p = self.find_paths(h)?;
        self.log(&format!("Meridian {VERSION} · App: {} · uv: {} · Ordner: {}",
                          p.app.display(), p.uv.display(), p.home.display()));
        for d in ["data", "runs", "settings"] {
            let _ = fs::create_dir_all(p.home.join(d));
        }
        let want = fingerprint(&p, opts.cpu);
        let have = fs::read_to_string(&p.marker).unwrap_or_default();
        if reinstall || opts.reinstall || have.trim() != want || !p.python.is_file() {
            let _ = fs::remove_file(&p.marker);
            self.setup(h, id, &p, opts.cpu)?;
            fs::write(&p.marker, &want).map_err(|e| format!("Konnte {} nicht schreiben: {e}", p.marker.display()))?;
        }
        if !self.current(id) {
            return Ok(());
        }

        let port = free_port(opts.port);
        let url = format!("http://127.0.0.1:{port}");
        self.step(h, "Starte Meridian …");
        let mut cmd = Command::new(&p.python);
        cmd.args(["-c", STREAMLIT_BOOT, "run"])
            .arg(p.app.join("app.py"))
            .args(["--server.port", &port.to_string(), "--server.address", "127.0.0.1",
                   "--server.headless", "true", "--browser.gatherUsageStats", "false",
                   "--server.fileWatcherType", if cfg!(debug_assertions) { "auto" } else { "none" }])
            .current_dir(&p.app) // .streamlit/config.toml (Theme) liegt dort
            .env("MERIDIAN_DATA_DIR", p.home.join("data"))
            .env("MERIDIAN_RUNS_DIR", p.home.join("runs"))
            .env("MERIDIAN_SETTINGS_DIR", p.home.join("settings"))
            .env("PYTHONUTF8", "1")
            .env("PYTHONUNBUFFERED", "1")
            .env("HF_HUB_DISABLE_SYMLINKS_WARNING", "1");
        self.spawn(h, id, &mut cmd, true).map_err(|e| format!("App konnte nicht gestartet werden: {e}"))?;

        let deadline = Instant::now() + START_TIMEOUT;
        loop {
            if !self.current(id) {
                return Ok(());
            }
            if let Some(code) = self.child_exited() {
                return Err(format!("Meridian wurde beim Start beendet ({code}). Details im Protokoll."));
            }
            if healthy(port) {
                break;
            }
            if Instant::now() > deadline {
                return Err("Meridian antwortet nicht (nach 3 Minuten). Details im Protokoll.".into());
            }
            thread::sleep(Duration::from_millis(300));
        }
        self.log(&format!("Meridian läuft: {url}"));
        if let Some(win) = h.get_webview_window("main") {
            let target = Url::parse(&url).map_err(|e| e.to_string())?;
            win.navigate(target).map_err(|e| format!("Fenster konnte die App nicht laden: {e}"))?;
        }

        // Überwachen: stürzt Streamlit ab, zurück zum Ladebildschirm mit Meldung.
        loop {
            thread::sleep(Duration::from_millis(500));
            if !self.current(id) {
                return Ok(());
            }
            if let Some(code) = self.child_exited() {
                return Err(format!("Meridian wurde unerwartet beendet ({code}). Details im Protokoll."));
            }
        }
    }

    /// Ist der Kind-Prozess beendet? Liefert dann eine kurze Beschreibung des Exit-Codes.
    fn child_exited(&self) -> Option<String> {
        let mut guard = self.child.lock().unwrap();
        let c = guard.as_mut()?;
        match c.try_wait() {
            Ok(Some(st)) => {
                guard.take();
                Some(match st.code() {
                    Some(code) => format!("Exit-Code {code}"),
                    None => "durch ein Signal".into(),
                })
            }
            Ok(None) => None,
            Err(e) => Some(e.to_string()),
        }
    }

    /// Startet einen Kind-Prozess ohne Konsolenfenster; die Ausgabe geht ins Protokoll und als Detail ans Fenster.
    /// `watchdog`: stdin offen halten (für STREAMLIT_BOOT).
    fn spawn(&self, h: &AppHandle, id: u64, cmd: &mut Command, watchdog: bool) -> std::io::Result<()> {
        cmd.stdin(if watchdog { Stdio::piped() } else { Stdio::null() }).stdout(Stdio::piped()).stderr(Stdio::piped());
        background(cmd);
        let args: Vec<String> = cmd.get_args().map(|a| a.to_string_lossy().to_string())
            .map(|a| if a.contains('\n') { "<script>".into() } else { a }).collect();
        self.log(&format!("$ {} {}", cmd.get_program().to_string_lossy(), args.join(" ")));
        let mut child = cmd.spawn()?;
        for pipe in [child.stdout.take().map(|s| Box::new(s) as Box<dyn Read + Send>),
                     child.stderr.take().map(|s| Box::new(s) as Box<dyn Read + Send>)].into_iter().flatten() {
            let h = h.clone();
            thread::spawn(move || {
                let me = h.state::<Launcher>();
                pump_lines(pipe, |line| {
                    if me.current(id) {
                        me.detail(&h, line);
                    } else {
                        me.log(&format!("    {line}"));
                    }
                });
            });
        }
        *self.child_stdin.lock().unwrap() = child.stdin.take();
        if let Some(mut old) = self.child.lock().unwrap().replace(child) {
            kill_tree(&mut old);
        }
        Ok(())
    }

    /// uv ausführen und auf das Ende warten (abbrechbar über `stop_child`).
    fn uv(&self, h: &AppHandle, id: u64, p: &Paths, args: &[&str]) -> Result<(), String> {
        let mut cmd = Command::new(&p.uv);
        cmd.args(args)
            .env("UV_PYTHON_PREFERENCE", "only-managed")
            .env("UV_LINK_MODE", "copy")
            .env("NO_COLOR", "1");
        self.spawn(h, id, &mut cmd, false).map_err(|e| format!("uv konnte nicht gestartet werden: {e}"))?;
        loop {
            if !self.current(id) {
                return Err("abgebrochen".into());
            }
            if let Some(code) = self.child_exited() {
                return if code == "Exit-Code 0" { Ok(()) } else { Err(format!("uv {} ({code})", args[0])) };
            }
            thread::sleep(Duration::from_millis(100));
        }
    }

    fn setup(&self, h: &AppHandle, id: u64, p: &Paths, cpu: bool) -> Result<(), String> {
        self.step(h, "Erster Start: Meridian wird eingerichtet (einmalig, 5–15 Minuten, ~3 GB Download).");
        thread::sleep(Duration::from_millis(600)); // damit man die Meldung lesen kann
        let _ = fs::remove_dir_all(&p.venv);
        self.step(h, &format!("Python {PYTHON_VERSION} und eigene Umgebung anlegen …"));
        let venv = p.venv.to_string_lossy().to_string();
        let python = p.python.to_string_lossy().to_string();
        self.uv(h, id, p, &["venv", "--python", PYTHON_VERSION, &venv])
            .map_err(|e| format!("Python-Umgebung konnte nicht angelegt werden: {e}"))?;

        let mut installed = false;
        if !cpu && has_nvidia_gpu() {
            for idx in CUDA_INDEXES {
                let tag = idx.rsplit('/').next().unwrap_or(idx);
                self.step(h, &format!("PyTorch mit GPU-Unterstützung installieren ({tag}) …"));
                if self.uv(h, id, p, &["pip", "install", "--python", &python, "torch", "--index-url", idx]).is_ok() {
                    installed = true;
                    break;
                }
                if !self.current(id) {
                    return Err("abgebrochen".into());
                }
                self.info(h, "Hat nicht geklappt – nächste Variante.");
            }
        } else if !cpu && !cfg!(target_os = "macos") {
            self.info(h, "Keine NVIDIA-Grafikkarte gefunden – PyTorch läuft auf der CPU.");
        }
        if !installed {
            self.step(h, "PyTorch installieren …");
            let mut args = vec!["pip", "install", "--python", &python, "torch"];
            if !cfg!(target_os = "macos") {
                args.extend(["--index-url", CPU_INDEX]);
            }
            if let Err(e) = self.uv(h, id, p, &args) {
                if cfg!(target_os = "macos") || !self.current(id) {
                    return Err(format!("PyTorch konnte nicht installiert werden: {e}"));
                }
                // download.pytorch.org nicht erreichbar (z. B. Firmen- oder Schulnetz): über PyPI versuchen
                self.info(h, "pytorch.org nicht erreichbar – versuche es über PyPI.");
                self.uv(h, id, p, &["pip", "install", "--python", &python, "torch"])
                    .map_err(|e| format!("PyTorch konnte nicht installiert werden: {e}"))?;
            }
        }

        self.step(h, "Restliche Pakete installieren …");
        let req = p.app.join("requirements.txt").to_string_lossy().to_string();
        self.uv(h, id, p, &["pip", "install", "--python", &python, "-r", &req])
            .map_err(|e| format!("Pakete konnten nicht installiert werden: {e}"))?;

        // Optional: schneller Tokenizer-Kern in Rust (rust/meridian_fastbpe). Ohne ihn läuft alles in Python.
        if !p.wheels.is_empty() {
            self.step(h, "Schnellen Tokenizer (Rust) installieren …");
            let wheels: Vec<String> = p.wheels.iter().map(|w| w.to_string_lossy().to_string()).collect();
            let mut args = vec!["pip", "install", "--python", &python, "--force-reinstall", "--no-deps"];
            args.extend(wheels.iter().map(String::as_str));
            if self.uv(h, id, p, &args).is_err() {
                if !self.current(id) {
                    return Err("abgebrochen".into());
                }
                self.info(h, "Hat nicht geklappt – Meridian benutzt den (langsameren) Python-Tokenizer.");
            }
        }

        let mut check = Command::new(&p.python);
        check.args(["-c", "import torch\n\
            if torch.cuda.is_available(): print('GPU: ' + torch.cuda.get_device_name(0))\n\
            elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available(): print('Apple GPU (MPS)')\n\
            else: print('CPU')"]);
        background(&mut check);
        let out = check.output().map_err(|e| format!("PyTorch lässt sich nicht laden: {e}"))?;
        if !out.status.success() {
            self.log(&String::from_utf8_lossy(&out.stderr));
            return Err("PyTorch lässt sich nicht laden. Details im Protokoll.".into());
        }
        self.step(h, &format!("Fertig eingerichtet. Rechnet auf: {}", String::from_utf8_lossy(&out.stdout).trim()));
        thread::sleep(Duration::from_millis(800));
        Ok(())
    }
}

/// Neuen Start auslösen (beim Öffnen, „Neu starten“, „Neu einrichten“). Ein laufender Ablauf wird abgebrochen.
pub fn start(h: AppHandle, reinstall: bool) {
    let me = h.state::<Launcher>();
    let id = me.run_id.fetch_add(1, Ordering::SeqCst) + 1;
    me.stop_child();
    *me.status.lock().unwrap() = Status::default();
    thread::spawn(move || {
        let me = h.state::<Launcher>();
        if let Err(e) = me.run(&h, id, reinstall) {
            if me.current(id) {
                me.fail(&h, &e);
            }
        }
    });
}

// ------------------------------------------------------------------------------------------- Befehle für das Fenster

#[derive(Serialize)]
pub struct LoadingInfo {
    version: &'static str,
    light: bool,
    log: String,
    step: String,
    info: String,
    error: Option<String>,
}

#[tauri::command]
pub fn loading_info(state: tauri::State<'_, Launcher>) -> LoadingInfo {
    let s = state.status.lock().unwrap().clone();
    LoadingInfo {
        version: VERSION,
        light: is_light_theme(&state.home),
        log: state.log_path().display().to_string(),
        step: s.step,
        info: s.info,
        error: s.error,
    }
}

#[tauri::command]
pub async fn restart(h: AppHandle) {
    start(h, false);
}

#[tauri::command]
pub async fn reinstall(h: AppHandle) {
    start(h, true);
}

#[tauri::command]
pub async fn open_log(h: AppHandle) -> Result<(), String> {
    let path = h.state::<Launcher>().log_path();
    h.opener().open_path(path.to_string_lossy(), None::<&str>).map_err(|e| e.to_string())
}

// ------------------------------------------------------------------------------------------- Hilfsfunktionen

/// Ändern sich requirements.txt, die mitgelieferten Wheels oder die Wahl CPU/GPU, wird die Einrichtung wiederholt.
fn fingerprint(p: &Paths, cpu: bool) -> String {
    let mut h = Sha256::new();
    h.update(fs::read(p.app.join("requirements.txt")).unwrap_or_default());
    h.update(format!("tauri|{VERSION}|{cpu}|{PYTHON_VERSION}|{CUDA_INDEXES:?}|{CPU_INDEX}"));
    for w in &p.wheels {
        let size = fs::metadata(w).map(|m| m.len()).unwrap_or(0);
        h.update(format!("|{}:{size}", w.file_name().unwrap_or_default().to_string_lossy()));
    }
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

/// uv liegt als „Sidecar“ neben dem Programm (Tauri `externalBin`); sonst eines aus dem PATH nehmen.
fn find_uv() -> Option<PathBuf> {
    let name = format!("uv{}", std::env::consts::EXE_SUFFIX);
    if let Some(dir) = std::env::current_exe().ok().and_then(|e| e.parent().map(Path::to_path_buf)) {
        let p = dir.join(&name);
        if p.is_file() {
            return Some(p);
        }
    }
    std::env::var_os("PATH").and_then(|paths| std::env::split_paths(&paths).map(|d| d.join(&name)).find(|p| p.is_file()))
}

fn has_nvidia_gpu() -> bool {
    let mut cmd = Command::new("nvidia-smi");
    cmd.arg("-L");
    background(&mut cmd);
    cmd.output().is_ok_and(|o| o.status.success() && String::from_utf8_lossy(&o.stdout).contains("GPU"))
}

/// Theme aus den Meridian-Einstellungen lesen, damit der Ladebildschirm dazu passt.
fn is_light_theme(home: &Path) -> bool {
    let Ok(text) = fs::read_to_string(home.join("settings").join("theme.json")) else { return false };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) else { return false };
    let active = v.get("active").and_then(|a| a.as_str()).unwrap_or("");
    active == "Hell"
        || v.get("custom").and_then(|c| c.get(active)).and_then(|c| c.get("base")).and_then(|b| b.as_str()) == Some("light")
}

/// Erster freier Port ab `start` (höchstens 50 weiter), sonst irgendein freier Port.
pub fn free_port(start: u16) -> u16 {
    for port in start..start.saturating_add(50) {
        if TcpListener::bind(("127.0.0.1", port)).is_ok() {
            return port;
        }
    }
    TcpListener::bind(("127.0.0.1", 0)).and_then(|l| l.local_addr()).map(|a| a.port()).unwrap_or(start)
}

/// Antwortet Streamlit? (`/_stcore/health` liefert 200, sobald die App bereit ist)
fn healthy(port: u16) -> bool {
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    let Ok(mut s) = TcpStream::connect_timeout(&addr, Duration::from_millis(500)) else { return false };
    let _ = s.set_read_timeout(Some(Duration::from_secs(2)));
    if s.write_all(b"GET /_stcore/health HTTP/1.0\r\nHost: 127.0.0.1\r\n\r\n").is_err() {
        return false;
    }
    let mut buf = [0u8; 64];
    let n = s.read(&mut buf).unwrap_or(0);
    let head = String::from_utf8_lossy(&buf[..n]);
    head.starts_with("HTTP/") && head.split_whitespace().nth(1) == Some("200")
}

/// Liest eine Ausgabe und gibt jede vollständige Zeile (getrennt durch \n oder \r) weiter.
pub fn pump_lines(mut r: impl Read, mut emit: impl FnMut(&str)) {
    let mut buf = [0u8; 4096];
    let mut line: Vec<u8> = Vec::new();
    loop {
        let n = match r.read(&mut buf) {
            Ok(0) | Err(_) => break,
            Ok(n) => n,
        };
        for &c in &buf[..n] {
            if c == b'\n' || c == b'\r' {
                let s = String::from_utf8_lossy(&line);
                let t = s.trim();
                if !t.is_empty() {
                    emit(t);
                }
                line.clear();
            } else if line.len() < 4096 {
                line.push(c);
            }
        }
    }
    let s = String::from_utf8_lossy(&line);
    if !s.trim().is_empty() {
        emit(s.trim());
    }
}

/// Kind-Prozess im Hintergrund: unter Windows ohne Konsolenfenster, sonst in eigener Prozessgruppe
/// (damit beim Beenden auch alle Unterprozesse mitgehen).
fn background(cmd: &mut Command) {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
}

/// Prozess und alle seine Unterprozesse beenden.
fn kill_tree(c: &mut Child) {
    let pid = c.id().to_string();
    #[cfg(windows)]
    {
        let mut k = Command::new("taskkill");
        k.args(["/T", "/F", "/PID", &pid]);
        background(&mut k);
        let _ = k.output();
    }
    #[cfg(unix)]
    {
        // erst freundlich (SIGTERM an die ganze Gruppe), nach 3 s hart
        let _ = Command::new("kill").args(["-TERM", &format!("-{pid}")]).output();
        let until = Instant::now() + Duration::from_secs(3);
        while Instant::now() < until {
            if matches!(c.try_wait(), Ok(Some(_))) {
                return;
            }
            thread::sleep(Duration::from_millis(50));
        }
        let _ = Command::new("kill").args(["-KILL", &format!("-{pid}")]).output();
    }
    let _ = c.kill();
    let _ = c.wait();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn options_from_args() {
        let o = Options::from_args(["--cpu", "--port", "9000", "--reinstall"].map(String::from));
        assert_eq!(o, Options { reinstall: true, cpu: true, port: 9000 });
        assert_eq!(Options::from_args(["--port=8600".to_string()]).port, 8600);
        assert_eq!(Options::from_args(Vec::<String>::new()), Options::default());
    }

    #[test]
    fn free_port_skips_busy_port() {
        let first = free_port(18_601);
        let _busy = TcpListener::bind(("127.0.0.1", first)).unwrap();
        assert_ne!(free_port(first), first);
    }

    #[test]
    fn pump_lines_splits_on_cr_and_lf() {
        let mut got = Vec::new();
        pump_lines(&b"a\r\nb\rc\n\n  d  "[..], |l| got.push(l.to_string()));
        assert_eq!(got, ["a", "b", "c", "d"]);
    }

    #[test]
    fn fingerprint_changes_with_wheels_and_cpu() {
        let dir = std::env::temp_dir().join(format!("meridian-fp-{}", std::process::id()));
        let _ = fs::create_dir_all(&dir);
        fs::write(dir.join("requirements.txt"), "torch\n").unwrap();
        let mut p = Paths { app: dir.clone(), uv: dir.clone(), wheels: vec![], home: dir.clone(),
                            venv: dir.clone(), python: dir.clone(), marker: dir.clone() };
        let before = fingerprint(&p, false);
        assert_ne!(before, fingerprint(&p, true));
        let w = dir.join("meridian_fastbpe-2.1.0-cp310-abi3-win_amd64.whl");
        fs::write(&w, "x").unwrap();
        p.wheels.push(w);
        assert_ne!(before, fingerprint(&p, false));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn light_theme_detection() {
        let dir = std::env::temp_dir().join(format!("meridian-theme-{}", std::process::id()));
        fs::create_dir_all(dir.join("settings")).unwrap();
        assert!(!is_light_theme(&dir));
        fs::write(dir.join("settings/theme.json"), r#"{"active":"Hell"}"#).unwrap();
        assert!(is_light_theme(&dir));
        fs::write(dir.join("settings/theme.json"), r#"{"active":"Mein","custom":{"Mein":{"base":"light"}}}"#).unwrap();
        assert!(is_light_theme(&dir));
        fs::write(dir.join("settings/theme.json"), r#"{"active":"Dunkel"}"#).unwrap();
        assert!(!is_light_theme(&dir));
        let _ = fs::remove_dir_all(&dir);
    }
}
