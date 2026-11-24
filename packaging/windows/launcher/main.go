// Meridian.exe – Startprogramm für Windows.
//
// Öffnet Meridian in einem eigenen App-Fenster (WebView2, in Windows 10/11 eingebaut – kein Browser nötig).
// Beim ersten Start richtet es alles ein (Python + PyTorch + Pakete, mit uv) und zeigt den Fortschritt im Fenster.
// Fenster schliessen = Meridian beenden. Fehlt WebView2, öffnet es Meridian wie früher im Browser.
//
// Ordner:
//
//	<Installationsordner>\Meridian.exe, uv.exe, app\...   (Programm, wird bei Updates ersetzt)
//	%LOCALAPPDATA%\Meridian\venv                          (Python-Umgebung, ~3–5 GB mit PyTorch)
//	%LOCALAPPDATA%\Meridian\data, runs, settings          (deine Daten, Modelle, Einstellungen)
//	%LOCALAPPDATA%\Meridian\meridian.log                  (Protokoll des letzten Starts)
//
// Optionen: --reinstall (Umgebung neu aufbauen), --cpu (PyTorch ohne Grafikkarte), --port N,
// --browser (im Browser statt im App-Fenster, mit Konsole), --no-browser (nur Server starten, mit Konsole)
package main

import (
	"bufio"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"flag"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"
)

const (
	version       = "2.1.0"
	pythonVersion = "3.12"
)

// Reihenfolge, in der PyTorch mit GPU-Unterstützung versucht wird (RTX 50xx braucht CUDA 12.8 oder neuer).
var cudaIndexes = []string{
	"https://download.pytorch.org/whl/cu130",
	"https://download.pytorch.org/whl/cu128",
}

type paths struct {
	install, app, uv, home, venv, python, marker string
}

type options struct {
	reinstall, cpu, browser, noBrowser bool
	port                               int
}

// reporter zeigt den Fortschritt an – in der Konsole oder im App-Fenster.
type reporter interface {
	Step(msg string)   // ein neuer Arbeitsschritt
	Info(msg string)   // Hinweis zum aktuellen Schritt
	Detail(msg string) // letzte Zeile der Ausgabe von uv/Python (nur im Fenster sichtbar, sonst im Protokoll)
}

// launcher hält alles, was Einrichten und Starten brauchen.
type launcher struct {
	opt    options
	p      paths
	ui     reporter
	log    io.Writer // Protokoll-Datei (+ Konsole im Konsolen-Modus)
	hidden bool      // Kind-Prozesse ohne eigenes Konsolenfenster starten (App-Fenster-Modus)

	mu    sync.Mutex
	child *exec.Cmd // gerade laufender Kind-Prozess (uv oder Streamlit), wird beim Schliessen beendet
}

func main() {
	var opt options
	flag.BoolVar(&opt.reinstall, "reinstall", false, "Python-Umgebung neu aufbauen")
	flag.BoolVar(&opt.cpu, "cpu", false, "PyTorch ohne GPU installieren")
	flag.IntVar(&opt.port, "port", 8501, "bevorzugter Port")
	flag.BoolVar(&opt.browser, "browser", false, "im Browser statt im App-Fenster öffnen")
	flag.BoolVar(&opt.noBrowser, "no-browser", false, "nur den Server starten (Konsole)")
	flag.Parse()

	home := homeDir()
	_ = os.MkdirAll(home, 0o755)
	logFile, _ := os.Create(filepath.Join(home, "meridian.log"))
	if logFile != nil {
		defer logFile.Close()
	}

	if !opt.browser && !opt.noBrowser && runWindowed(opt, home, logFile) {
		return
	}
	attachConsole()
	runConsole(opt, logFile)
}

// ---------------------------------------------------------------------------------------------- Konsolen-Modus

type consoleUI struct{ log io.Writer }

func (c consoleUI) Step(msg string)   { fmt.Printf("  » %s\n", msg); fmt.Fprintf(c.log, "» %s\n", msg) }
func (c consoleUI) Info(msg string)   { fmt.Printf("    %s\n", msg); fmt.Fprintf(c.log, "  %s\n", msg) }
func (c consoleUI) Detail(msg string) {}

// runConsole ist der alte Ablauf: Konsole zeigt alles, die App öffnet sich im Browser.
func runConsole(opt options, logFile *os.File) {
	var logw io.Writer = io.Discard
	if logFile != nil {
		logw = logFile
	}
	banner()
	l := &launcher{opt: opt, ui: consoleUI{logw}, log: io.MultiWriter(os.Stdout, logw)}
	if err := l.prepare(); err != nil {
		fail(err)
	}
	cmd, url, err := l.startServer()
	if err != nil {
		fail(err)
	}
	exited := make(chan struct{})
	go func() {
		if waitHealthy(exited, url, 3*time.Minute) {
			fmt.Printf("\n  Meridian läuft:  %s\n  Dieses Fenster offen lassen. Schliessen = Meridian beenden.\n\n", url)
			if !opt.noBrowser {
				openBrowser(url)
			}
		} else {
			fmt.Println("\n  Die App antwortet nicht. Siehe Meldungen oben.")
		}
	}()
	err = cmd.Wait()
	close(exited)
	if err != nil {
		fail(fmt.Errorf("Meridian wurde mit einem Fehler beendet: %w", err))
	}
}

func banner() {
	fmt.Println()
	fmt.Println("  ▌▌▌▌▌▖   Meridian " + version)
	fmt.Println("  ein eigenes Sprachmodell – von Grund auf gebaut")
	fmt.Println()
}

func fail(err error) {
	fmt.Printf("\n  FEHLER: %v\n\n  Tipp: Meridian mit --reinstall starten, um die Umgebung neu aufzubauen.\n", err)
	fmt.Print("  Enter drücken zum Schliessen …")
	_, _ = bufio.NewReader(os.Stdin).ReadString('\n')
	os.Exit(1)
}

// ---------------------------------------------------------------------------------------------- Einrichten + Starten

// prepare sucht die Programmteile und richtet bei Bedarf Python + PyTorch ein.
func (l *launcher) prepare() error {
	p, err := findPaths()
	if err != nil {
		return err
	}
	l.p = p
	for _, d := range []string{"data", "runs", "settings"} {
		_ = os.MkdirAll(filepath.Join(p.home, d), 0o755)
	}
	want := setupFingerprint(p, l.opt.cpu)
	if l.opt.reinstall || readFile(p.marker) != want || !exists(p.python) {
		_ = os.Remove(p.marker)
		if err := l.setup(); err != nil {
			return err
		}
		_ = os.WriteFile(p.marker, []byte(want), 0o644)
	}
	return nil
}

// startServer startet Streamlit auf einem freien Port und liefert den Prozess und die Adresse.
func (l *launcher) startServer() (*exec.Cmd, string, error) {
	p := l.p
	free := freePort(l.opt.port)
	url := fmt.Sprintf("http://localhost:%d", free)
	cmd := exec.Command(p.python, "-m", "streamlit", "run", filepath.Join(p.app, "app.py"),
		"--server.port", fmt.Sprint(free), "--server.address", "localhost", "--server.headless", "true",
		"--browser.gatherUsageStats", "false")
	cmd.Dir = p.app // .streamlit\config.toml (Theme) liegt dort
	cmd.Env = append(os.Environ(),
		"MERIDIAN_DATA_DIR="+filepath.Join(p.home, "data"),
		"MERIDIAN_RUNS_DIR="+filepath.Join(p.home, "runs"),
		"MERIDIAN_SETTINGS_DIR="+filepath.Join(p.home, "settings"),
		"PYTHONUTF8=1", "HF_HUB_DISABLE_SYMLINKS_WARNING=1",
	)
	cmd.Stdout, cmd.Stderr = l.log, l.log
	if l.hidden {
		hideWindow(cmd)
	}
	l.ui.Step("Starte Meridian …")
	if err := cmd.Start(); err != nil {
		return nil, "", fmt.Errorf("App konnte nicht gestartet werden: %w", err)
	}
	l.setChild(cmd)
	return cmd, url, nil
}

func (l *launcher) setChild(c *exec.Cmd) {
	l.mu.Lock()
	l.child = c
	l.mu.Unlock()
}

// stop beendet den laufenden Kind-Prozess samt Unterprozessen (beim Schliessen des Fensters).
func (l *launcher) stop() {
	l.mu.Lock()
	c := l.child
	l.mu.Unlock()
	if c != nil && c.Process != nil {
		killTree(c)
	}
}

func exists(p string) bool { _, err := os.Stat(p); return err == nil }

func readFile(p string) string {
	b, err := os.ReadFile(p)
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(b))
}

// homeDir: %LOCALAPPDATA%\Meridian – Python-Umgebung, Daten, Einstellungen, Protokoll.
func homeDir() string {
	base := os.Getenv("LOCALAPPDATA")
	if base == "" {
		base, _ = os.UserConfigDir()
	}
	return filepath.Join(base, "Meridian")
}

func findPaths() (paths, error) {
	exe, err := os.Executable()
	if err != nil {
		return paths{}, err
	}
	install := filepath.Dir(exe)
	app := filepath.Join(install, "app")
	if !exists(filepath.Join(app, "app.py")) { // Entwicklung: Meridian.exe neben app.py
		app = install
	}
	if !exists(filepath.Join(app, "app.py")) {
		return paths{}, fmt.Errorf("app.py nicht gefunden in %s", install)
	}
	uv := filepath.Join(install, "uv.exe")
	if runtime.GOOS != "windows" {
		uv = filepath.Join(install, "uv")
	}
	if !exists(uv) {
		if found, err := exec.LookPath("uv"); err == nil {
			uv = found
		} else {
			return paths{}, fmt.Errorf("uv nicht gefunden (%s)", uv)
		}
	}
	home := homeDir()
	venv := filepath.Join(home, "venv")
	python := filepath.Join(venv, "Scripts", "python.exe")
	if runtime.GOOS != "windows" {
		python = filepath.Join(venv, "bin", "python")
	}
	return paths{install, app, uv, home, venv, python, filepath.Join(home, "setup-ok.txt")}, nil
}

// Ändern sich requirements.txt, die mitgelieferten Wheels oder die Wahl CPU/GPU, wird die Einrichtung wiederholt.
func setupFingerprint(p paths, cpu bool) string {
	req, _ := os.ReadFile(filepath.Join(p.app, "requirements.txt"))
	extra := fmt.Sprint(version, cpu, cudaIndexes)
	for _, w := range bundledWheels(p) {
		if st, err := os.Stat(w); err == nil {
			extra += fmt.Sprintf("|%s:%d", filepath.Base(w), st.Size())
		}
	}
	h := sha256.Sum256(append(req, []byte(extra)...))
	return hex.EncodeToString(h[:])
}

// Mitgelieferte Python-Pakete (z. B. der Rust-Tokenizer) im Ordner <Installation>\wheels.
func bundledWheels(p paths) []string {
	found, _ := filepath.Glob(filepath.Join(p.install, "wheels", "*.whl"))
	return found
}

// run führt uv aus. Die Ausgabe geht ins Protokoll; die letzte Zeile erscheint im Fenster (Detail).
func (l *launcher) run(args ...string) error {
	cmd := exec.Command(l.p.uv, args...)
	lw := &lineWriter{emit: l.ui.Detail}
	cmd.Stdout = io.MultiWriter(l.log, lw)
	cmd.Stderr = io.MultiWriter(l.log, lw)
	cmd.Env = append(os.Environ(), "UV_PYTHON_PREFERENCE=only-managed", "UV_LINK_MODE=copy", "NO_COLOR=1")
	if l.hidden {
		hideWindow(cmd)
	}
	if err := cmd.Start(); err != nil {
		return err
	}
	l.setChild(cmd)
	defer l.setChild(nil)
	return cmd.Wait()
}

func (l *launcher) output(name string, args ...string) ([]byte, error) {
	cmd := exec.Command(name, args...)
	if l.hidden {
		hideWindow(cmd)
	}
	return cmd.Output()
}

func (l *launcher) hasNvidiaGPU() bool {
	out, err := l.output("nvidia-smi", "-L")
	return err == nil && strings.Contains(string(out), "GPU")
}

func (l *launcher) setup() error {
	p, ui := l.p, l.ui
	ui.Step("Erster Start: Meridian wird eingerichtet (einmalig, 5–15 Minuten, ~3 GB Download).")
	_ = os.RemoveAll(p.venv)
	ui.Step("Python " + pythonVersion + " und eigene Umgebung anlegen …")
	if err := l.run("venv", "--python", pythonVersion, p.venv); err != nil {
		return fmt.Errorf("Python-Umgebung konnte nicht angelegt werden: %w", err)
	}
	installed := false
	if !l.opt.cpu && l.hasNvidiaGPU() {
		for _, idx := range cudaIndexes {
			ui.Step("PyTorch mit GPU-Unterstützung installieren (" + idx[strings.LastIndex(idx, "/")+1:] + ") …")
			if err := l.run("pip", "install", "--python", p.python, "torch", "--index-url", idx); err == nil {
				installed = true
				break
			}
			ui.Info("Hat nicht geklappt – nächste Variante.")
		}
	} else if !l.opt.cpu {
		ui.Info("Keine NVIDIA-Grafikkarte gefunden – PyTorch läuft auf der CPU.")
	}
	if !installed {
		ui.Step("PyTorch (CPU) installieren …")
		if err := l.run("pip", "install", "--python", p.python, "torch"); err != nil {
			return fmt.Errorf("PyTorch konnte nicht installiert werden: %w", err)
		}
	}
	ui.Step("Restliche Pakete installieren …")
	if err := l.run("pip", "install", "--python", p.python, "-r", filepath.Join(p.app, "requirements.txt")); err != nil {
		return fmt.Errorf("Pakete konnten nicht installiert werden: %w", err)
	}
	// Optional: schneller Tokenizer-Kern in Rust (rust/meridian_fastbpe). Ohne ihn läuft alles in Python.
	if wheels := bundledWheels(p); len(wheels) > 0 {
		ui.Step("Schnellen Tokenizer (Rust) installieren …")
		args := append([]string{"pip", "install", "--python", p.python, "--force-reinstall", "--no-deps"}, wheels...)
		if err := l.run(args...); err != nil {
			ui.Info("Hat nicht geklappt – Meridian benutzt den (langsameren) Python-Tokenizer.")
		}
	}
	out, err := l.output(p.python, "-c",
		"import torch; print('GPU: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')")
	if err != nil {
		return errors.New("PyTorch lässt sich nicht laden")
	}
	ui.Step("Fertig eingerichtet. Rechnet auf: " + strings.TrimSpace(string(out)))
	return nil
}

// lineWriter gibt jede vollständige Ausgabezeile (getrennt durch \n oder \r) an emit weiter.
type lineWriter struct {
	buf  []byte
	emit func(string)
}

func (w *lineWriter) Write(b []byte) (int, error) {
	for _, c := range b {
		if c == '\n' || c == '\r' {
			if line := strings.TrimSpace(string(w.buf)); line != "" && w.emit != nil {
				w.emit(line)
			}
			w.buf = w.buf[:0]
			continue
		}
		if len(w.buf) < 4096 {
			w.buf = append(w.buf, c)
		}
	}
	return len(b), nil
}

func freePort(start int) int {
	for port := start; port < start+50; port++ {
		l, err := net.Listen("tcp", fmt.Sprintf("localhost:%d", port))
		if err == nil {
			_ = l.Close()
			return port
		}
	}
	return start
}

// waitHealthy wartet, bis Streamlit antwortet. Bricht sofort ab, wenn “exited“ geschlossen wird (Prozess beendet).
func waitHealthy(exited <-chan struct{}, url string, timeout time.Duration) bool {
	client := http.Client{Timeout: 2 * time.Second}
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		select {
		case <-exited:
			return false
		default:
		}
		resp, err := client.Get(url + "/_stcore/health")
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == 200 {
				return true
			}
		}
		time.Sleep(400 * time.Millisecond)
	}
	return false
}

func openBrowser(url string) {
	switch runtime.GOOS {
	case "windows":
		_ = exec.Command("rundll32", "url.dll,FileProtocolHandler", url).Start()
	case "darwin":
		_ = exec.Command("open", url).Start()
	default:
		_ = exec.Command("xdg-open", url).Start()
	}
}
