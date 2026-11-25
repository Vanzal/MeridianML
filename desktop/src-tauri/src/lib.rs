//! Meridian als Desktop-App: ein Fenster (Tauri), darin zuerst der Ladebildschirm (ui/index.html), dann die
//! Streamlit-App von Meridian. Das Starten und Einrichten macht `launcher.rs`.

mod launcher;

use std::path::PathBuf;

use tauri::webview::DownloadEvent;
use tauri::{AppHandle, LogicalPosition, LogicalSize, Manager, RunEvent, Url, WebviewUrl, WebviewWindowBuilder};
use tauri_plugin_opener::OpenerExt;

const INIT_JS: &str = include_str!("init.js");

/// Seiten von Meridian selbst bleiben im Fenster, alles andere öffnet der normale Browser.
fn allow_navigation(h: &AppHandle, url: &Url) -> bool {
    let external = match url.scheme() {
        "http" | "https" => !matches!(url.host_str(), Some("127.0.0.1" | "localhost" | "tauri.localhost" | "[::1]")),
        "mailto" => true,
        _ => false,
    };
    if external {
        let _ = h.opener().open_url(url.as_str(), None::<&str>);
    }
    !external
}

/// Freier Dateiname im Download-Ordner („text.txt“, sonst „text (1).txt“ …).
fn download_target(dir: PathBuf, suggested: &std::path::Path, url: &Url) -> PathBuf {
    let name = suggested
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .filter(|n| !n.is_empty())
        .or_else(|| url.path_segments().and_then(|mut s| s.next_back()).map(str::to_string))
        .filter(|n| !n.is_empty())
        .unwrap_or_else(|| "meridian-download".into());
    let path = dir.join(&name);
    if !path.exists() {
        return path;
    }
    let p = std::path::Path::new(&name);
    let stem = p.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
    let ext = p.extension().map(|e| format!(".{}", e.to_string_lossy())).unwrap_or_default();
    (1..)
        .map(|i| dir.join(format!("{stem} ({i}){ext}")))
        .find(|c| !c.exists())
        .unwrap()
}

pub fn run() {
    let opts = launcher::Options::from_args(std::env::args().skip(1));
    #[cfg(target_os = "linux")]
    if std::env::var_os("WEBKIT_DISABLE_DMABUF_RENDERER").is_none() {
        // verhindert ein leeres Fenster mit manchen NVIDIA-Treibern unter Linux
        std::env::set_var("WEBKIT_DISABLE_DMABUF_RENDERER", "1");
    }

    let app = tauri::Builder::default()
        // Zweiter Start: kein zweites Meridian, sondern das offene Fenster nach vorne holen.
        .plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| {
            if let Some(w) = app.get_webview_window("main") {
                let _ = w.unminimize();
                let _ = w.show();
                let _ = w.set_focus();
            }
        }))
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![
            launcher::loading_info,
            launcher::restart,
            launcher::reinstall,
            launcher::open_log
        ])
        .setup(move |app| {
            let home = app.path().local_data_dir()?.join("Meridian");
            app.manage(launcher::Launcher::new(opts, home));

            let nav = app.handle().clone();
            let dl = app.handle().clone();
            let win = WebviewWindowBuilder::new(app, "main", WebviewUrl::App("index.html".into()))
                .title("Meridian")
                .inner_size(1440.0, 920.0)
                .min_inner_size(900.0, 600.0)
                .initialization_script(INIT_JS)
                .on_navigation(move |url| allow_navigation(&nav, url))
                .on_download(move |_webview, event| {
                    match event {
                        DownloadEvent::Requested { url, destination } => {
                            if let Ok(dir) = dl.path().download_dir() {
                                *destination = download_target(dir, destination, &url);
                            }
                        }
                        DownloadEvent::Finished { path: Some(path), success: true, .. } => {
                            let _ = dl.opener().reveal_item_in_dir(path);
                        }
                        _ => {}
                    }
                    true
                })
                .build()?;
            // nicht grösser als der Bildschirm, mittig darauf
            if let Ok(Some(m)) = win.current_monitor().map(|m| m.or_else(|| win.primary_monitor().ok().flatten())) {
                let scale = m.scale_factor();
                let screen = m.size().to_logical::<f64>(scale);
                let origin = m.position().to_logical::<f64>(scale);
                let (w, h) = (1440f64.min(screen.width * 0.9), 920f64.min(screen.height * 0.85));
                let _ = win.set_size(LogicalSize::new(w, h));
                let _ = win.set_position(LogicalPosition::new(origin.x + (screen.width - w) / 2.0,
                                                              origin.y + (screen.height - h) / 2.0));
            }
            if let Ok(url) = win.url() {
                app.state::<launcher::Launcher>().set_loading_url(url);
            }
            launcher::start(app.handle().clone(), false);
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("Meridian konnte nicht gestartet werden");

    app.run(|h, event| {
        if let RunEvent::Exit = event {
            if let Some(l) = h.try_state::<launcher::Launcher>() {
                l.shutdown();
            }
        }
    });
}
