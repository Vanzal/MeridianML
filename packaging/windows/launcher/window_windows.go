//go:build windows

package main

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sync"
	"syscall"
	"time"

	webview2 "github.com/jchv/go-webview2"
)

// windowUI zeigt den Fortschritt im App-Fenster (Ladebildschirm) und schreibt alles ins Protokoll.
type windowUI struct {
	w        webview2.WebView
	log      *os.File
	mu       sync.Mutex
	lastSent time.Time
}

func (u *windowUI) js(fn string, arg string) {
	b, _ := json.Marshal(arg)
	code := fmt.Sprintf("window.%s && window.%s(%s)", fn, fn, b)
	u.w.Dispatch(func() { u.w.Eval(code) })
}

func (u *windowUI) logf(format string, a ...any) {
	if u.log != nil {
		fmt.Fprintf(u.log, format, a...)
	}
}

func (u *windowUI) Step(msg string) { u.logf("» %s\n", msg); u.js("setStep", msg) }
func (u *windowUI) Info(msg string) { u.logf("  %s\n", msg); u.js("setInfo", msg) }

// Detail höchstens ~8x pro Sekunde ans Fenster schicken (uv gibt sehr viele Zeilen aus).
func (u *windowUI) Detail(msg string) {
	u.mu.Lock()
	due := time.Since(u.lastSent) > 120*time.Millisecond
	if due {
		u.lastSent = time.Now()
	}
	u.mu.Unlock()
	if due {
		u.js("setDetail", msg)
	}
}

// runWindowed öffnet Meridian im eigenen Fenster. Liefert false, wenn WebView2 nicht verfügbar ist –
// dann übernimmt der Konsolen-Modus (Browser).
func runWindowed(opt options, home string, logFile *os.File) bool {
	runtime.LockOSThread() // das Fenster muss auf dem Thread laufen, der es erzeugt hat

	width, height := windowSize()
	w := webview2.NewWithOptions(webview2.WebViewOptions{
		AutoFocus: true,
		DataPath:  filepath.Join(home, "webview"),
		WindowOptions: webview2.WindowOptions{
			Title: "Meridian", Width: uint(width), Height: uint(height), IconId: 1, Center: true,
		},
	})
	if w == nil {
		if logFile != nil {
			fmt.Fprintln(logFile, "WebView2 nicht verfügbar – öffne Meridian im Browser.")
		}
		return false
	}
	defer w.Destroy()
	w.SetSize(scaled(900), scaled(600), webview2.HintMin)

	ui := &windowUI{w: w, log: logFile}
	var logw = logWriter(logFile)
	l := &launcher{opt: opt, ui: ui, log: logw, hidden: true}
	logPath := filepath.Join(home, "meridian.log")

	_ = w.Bind("meridianReinstall", func() {
		l.stop()
		if exe, err := os.Executable(); err == nil {
			_ = exec.Command(exe, "--reinstall").Start()
		}
		w.Terminate()
	})
	_ = w.Bind("meridianOpenLog", func() {
		_ = exec.Command("notepad.exe", logPath).Start()
	})
	_ = w.Bind("meridianBrowser", func() {
		l.stop()
		if exe, err := os.Executable(); err == nil {
			_ = exec.Command(exe, "--browser").Start()
		}
		w.Terminate()
	})

	w.SetHtml(loadingPage(isLightTheme(home), logPath))

	go func() {
		showError := func(err error) {
			ui.logf("FEHLER: %v\n", err)
			ui.js("setError", err.Error())
		}
		if err := l.prepare(); err != nil {
			showError(err)
			return
		}
		cmd, url, err := l.startServer()
		if err != nil {
			showError(err)
			return
		}
		exited := make(chan struct{})
		go func() {
			err := cmd.Wait()
			close(exited)
			msg := "Meridian wurde beendet."
			if err != nil {
				msg = fmt.Sprintf("Meridian wurde mit einem Fehler beendet: %v", err)
			}
			ui.logf("%s\n", msg)
			// Nur anzeigen, wenn das Fenster noch offen ist (beim Schliessen beenden wir Streamlit selbst).
			w.Dispatch(func() {
				w.SetHtml(loadingPage(isLightTheme(home), logPath))
				w.Eval(fmt.Sprintf("window.setError && window.setError(%s)", mustJSON(msg)))
			})
		}()
		if waitHealthy(exited, url, 3*time.Minute) {
			ui.logf("Meridian läuft: %s\n", url)
			w.Dispatch(func() { w.Navigate(url) })
		} else {
			select {
			case <-exited: // Meldung kommt von oben
			default:
				showError(fmt.Errorf("Meridian antwortet nicht (nach 3 Minuten)"))
			}
		}
	}()

	w.Run()  // bis das Fenster geschlossen wird
	l.stop() // Streamlit (oder eine laufende Einrichtung) beenden
	return true
}

func mustJSON(s string) string { b, _ := json.Marshal(s); return string(b) }

// windowSize: 1440 x 920 bei 100 % Skalierung, angepasst an die Bildschirm-Skalierung, höchstens 90 % des Bildschirms.
func windowSize() (int, int) {
	u := syscall.NewLazyDLL("user32.dll")
	sw, _, _ := u.NewProc("GetSystemMetrics").Call(0) // SM_CXSCREEN
	sh, _, _ := u.NewProc("GetSystemMetrics").Call(1) // SM_CYSCREEN
	w, h := scaled(1440), scaled(920)
	if sw > 0 && w > int(sw)*9/10 {
		w = int(sw) * 9 / 10
	}
	if sh > 0 && h > int(sh)*85/100 {
		h = int(sh) * 85 / 100
	}
	return w, h
}

func scaled(px int) int {
	proc := syscall.NewLazyDLL("user32.dll").NewProc("GetDpiForSystem")
	if proc.Find() != nil {
		return px
	}
	dpi, _, _ := proc.Call()
	if dpi == 0 {
		return px
	}
	return px * int(dpi) / 96
}
