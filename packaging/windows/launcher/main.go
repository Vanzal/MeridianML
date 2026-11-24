// Meridian.exe – Startprogramm für Windows.
//
// Beim ersten Start richtet es alles ein (Python + PyTorch + Pakete, mit uv), danach startet es die App und
// öffnet sie im Browser. Das Konsolenfenster zeigt, was passiert – wie bei llama.cpp. Fenster schliessen = beenden.
//
// Ordner:
//
//	<Installationsordner>\Meridian.exe, uv.exe, app\...   (Programm, wird bei Updates ersetzt)
//	%LOCALAPPDATA%\Meridian\venv                          (Python-Umgebung, ~3–5 GB mit PyTorch)
//	%LOCALAPPDATA%\Meridian\data, runs, settings          (deine Daten, Modelle, Einstellungen)
//
// Optionen: --reinstall (Umgebung neu aufbauen), --cpu (PyTorch ohne Grafikkarte), --port N, --no-browser
package main

import (
	"bufio"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"flag"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
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

func main() {
	reinstall := flag.Bool("reinstall", false, "Python-Umgebung neu aufbauen")
	cpu := flag.Bool("cpu", false, "PyTorch ohne GPU installieren")
	port := flag.Int("port", 8501, "bevorzugter Port")
	noBrowser := flag.Bool("no-browser", false, "Browser nicht automatisch öffnen")
	flag.Parse()

	banner()
	p, err := findPaths()
	if err != nil {
		fail(err)
	}
	for _, d := range []string{"data", "runs", "settings"} {
		_ = os.MkdirAll(filepath.Join(p.home, d), 0o755)
	}

	want := setupFingerprint(p, *cpu)
	if *reinstall || readFile(p.marker) != want || !exists(p.python) {
		if err := setup(p, *cpu); err != nil {
			fail(err)
		}
		_ = os.WriteFile(p.marker, []byte(want), 0o644)
	}

	free := freePort(*port)
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
	cmd.Stdout, cmd.Stderr = os.Stdout, os.Stderr
	step("Starte Meridian …")
	if err := cmd.Start(); err != nil {
		fail(fmt.Errorf("App konnte nicht gestartet werden: %w", err))
	}
	go func() {
		if waitHealthy(free, 3*time.Minute) {
			fmt.Printf("\n  Meridian läuft:  %s\n  Dieses Fenster offen lassen. Schliessen = Meridian beenden.\n\n", url)
			if !*noBrowser {
				openBrowser(url)
			}
		} else {
			fmt.Println("\n  Die App antwortet nicht. Siehe Meldungen oben.")
		}
	}()
	if err := cmd.Wait(); err != nil {
		fail(fmt.Errorf("Meridian wurde mit einem Fehler beendet: %w", err))
	}
}

func banner() {
	fmt.Println()
	fmt.Println("  ▌▌▌▌▌▖   Meridian " + version)
	fmt.Println("  ein eigenes Sprachmodell – von Grund auf gebaut")
	fmt.Println()
}

func step(msg string)      { fmt.Printf("  » %s\n", msg) }
func info(msg string)      { fmt.Printf("    %s\n", msg) }
func exists(p string) bool { _, err := os.Stat(p); return err == nil }

func fail(err error) {
	fmt.Printf("\n  FEHLER: %v\n\n  Tipp: Meridian mit --reinstall starten, um die Umgebung neu aufzubauen.\n", err)
	fmt.Print("  Enter drücken zum Schliessen …")
	_, _ = bufio.NewReader(os.Stdin).ReadString('\n')
	os.Exit(1)
}

func readFile(p string) string {
	b, err := os.ReadFile(p)
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(b))
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
	base := os.Getenv("LOCALAPPDATA")
	if base == "" {
		base, _ = os.UserConfigDir()
	}
	home := filepath.Join(base, "Meridian")
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

func run(p paths, args ...string) error {
	cmd := exec.Command(p.uv, args...)
	cmd.Stdout, cmd.Stderr = os.Stdout, os.Stderr
	cmd.Env = append(os.Environ(), "UV_PYTHON_PREFERENCE=only-managed", "UV_LINK_MODE=copy")
	return cmd.Run()
}

func hasNvidiaGPU() bool {
	out, err := exec.Command("nvidia-smi", "-L").Output()
	return err == nil && strings.Contains(string(out), "GPU")
}

func setup(p paths, cpu bool) error {
	step("Erster Start: Meridian wird eingerichtet (einmalig, 5–15 Minuten, ~3 GB Download).")
	_ = os.RemoveAll(p.venv)
	step("Python " + pythonVersion + " und eigene Umgebung anlegen …")
	if err := run(p, "venv", "--python", pythonVersion, p.venv); err != nil {
		return fmt.Errorf("Python-Umgebung konnte nicht angelegt werden: %w", err)
	}
	installed := false
	if !cpu && hasNvidiaGPU() {
		for _, idx := range cudaIndexes {
			step("PyTorch mit GPU-Unterstützung installieren (" + idx[strings.LastIndex(idx, "/")+1:] + ") …")
			if err := run(p, "pip", "install", "--python", p.python, "torch", "--index-url", idx); err == nil {
				installed = true
				break
			}
			info("Hat nicht geklappt – nächste Variante.")
		}
	} else if !cpu {
		info("Keine NVIDIA-Grafikkarte gefunden – PyTorch läuft auf der CPU.")
	}
	if !installed {
		step("PyTorch (CPU) installieren …")
		if err := run(p, "pip", "install", "--python", p.python, "torch"); err != nil {
			return fmt.Errorf("PyTorch konnte nicht installiert werden: %w", err)
		}
	}
	step("Restliche Pakete installieren …")
	if err := run(p, "pip", "install", "--python", p.python, "-r", filepath.Join(p.app, "requirements.txt")); err != nil {
		return fmt.Errorf("Pakete konnten nicht installiert werden: %w", err)
	}
	// Optional: schneller Tokenizer-Kern in Rust (rust/meridian_fastbpe). Ohne ihn läuft alles in Python.
	if wheels := bundledWheels(p); len(wheels) > 0 {
		step("Schnellen Tokenizer (Rust) installieren …")
		args := append([]string{"pip", "install", "--python", p.python, "--force-reinstall", "--no-deps"}, wheels...)
		if err := run(p, args...); err != nil {
			info("Hat nicht geklappt – Meridian benutzt den (langsameren) Python-Tokenizer.")
		}
	}
	out, err := exec.Command(p.python, "-c",
		"import torch; print('GPU: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')").Output()
	if err != nil {
		return errors.New("PyTorch lässt sich nicht laden")
	}
	step("Fertig eingerichtet. Rechnet auf: " + strings.TrimSpace(string(out)))
	return nil
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

func waitHealthy(port int, timeout time.Duration) bool {
	client := http.Client{Timeout: 2 * time.Second}
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		resp, err := client.Get(fmt.Sprintf("http://localhost:%d/_stcore/health", port))
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == 200 {
				return true
			}
		}
		time.Sleep(500 * time.Millisecond)
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
