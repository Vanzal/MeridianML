package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestBundledWheelsChangeFingerprint(t *testing.T) {
	dir := t.TempDir()
	app := filepath.Join(dir, "app")
	if err := os.MkdirAll(app, 0o755); err != nil {
		t.Fatal(err)
	}
	_ = os.WriteFile(filepath.Join(app, "requirements.txt"), []byte("torch\n"), 0o644)
	p := paths{install: dir, app: app}
	before := setupFingerprint(p, false)
	if len(bundledWheels(p)) != 0 {
		t.Fatal("expected no wheels")
	}
	_ = os.MkdirAll(filepath.Join(dir, "wheels"), 0o755)
	_ = os.WriteFile(filepath.Join(dir, "wheels", "meridian_fastbpe-2.1.0-cp310-abi3-win_amd64.whl"), []byte("x"), 0o644)
	if got := bundledWheels(p); len(got) != 1 {
		t.Fatalf("expected 1 wheel, got %v", got)
	}
	if setupFingerprint(p, false) == before {
		t.Fatal("fingerprint must change when a wheel is added")
	}
	if setupFingerprint(p, true) == setupFingerprint(p, false) {
		t.Fatal("fingerprint must differ between CPU and GPU")
	}
}

func TestFreePortSkipsBusyPort(t *testing.T) {
	first := freePort(18501)
	if first < 18501 {
		t.Fatal("port below start")
	}
}

func TestLoadingPageHasNoPlaceholdersAndAllHooks(t *testing.T) {
	for _, light := range []bool{false, true} {
		html := loadingPage(light, `C:\Users\x\AppData\Local\Meridian\meridian.log`)
		if strings.Contains(html, "{bg}") || strings.Contains(html, "{mark}") || strings.Contains(html, "{log}") {
			t.Fatal("unreplaced placeholder in loading page")
		}
		for _, hook := range []string{"window.setStep", "window.setInfo", "window.setDetail", "window.setError",
			"meridianReinstall()", "meridianOpenLog()", "meridianBrowser()", "v" + version} {
			if !strings.Contains(html, hook) {
				t.Fatalf("loading page misses %q", hook)
			}
		}
	}
	if !strings.Contains(loadingPage(true, ""), "#ffffff") || !strings.Contains(loadingPage(false, ""), "#0a0a0a") {
		t.Fatal("light/dark colours not applied")
	}
	if strings.Contains(loadingPage(false, `<x>&"`), `<x>`) {
		t.Fatal("log path must be HTML-escaped")
	}
}

func TestIsLightThemeReadsMeridianSettings(t *testing.T) {
	home := t.TempDir()
	if isLightTheme(home) {
		t.Fatal("no settings -> dark")
	}
	_ = os.MkdirAll(filepath.Join(home, "settings"), 0o755)
	write := func(s string) { _ = os.WriteFile(filepath.Join(home, "settings", "theme.json"), []byte(s), 0o644) }
	write(`{"active": "Hell"}`)
	if !isLightTheme(home) {
		t.Fatal("Hell preset -> light")
	}
	write(`{"active": "Mein", "custom": {"Mein": {"base": "light"}}}`)
	if !isLightTheme(home) {
		t.Fatal("custom light theme -> light")
	}
	write(`{"active": "Terminal"}`)
	if isLightTheme(home) {
		t.Fatal("Terminal -> dark")
	}
	write(`kaputt`)
	if isLightTheme(home) {
		t.Fatal("broken file -> dark")
	}
}

func TestLineWriterSplitsOnNewlinesAndCarriageReturns(t *testing.T) {
	var got []string
	w := &lineWriter{emit: func(s string) { got = append(got, s) }}
	_, _ = w.Write([]byte("Resolved 3 packages\r\nDownloading torch (2.1GiB)\rPrepared"))
	_, _ = w.Write([]byte(" 1 package\n\n  \n"))
	want := []string{"Resolved 3 packages", "Downloading torch (2.1GiB)", "Prepared 1 package"}
	if strings.Join(got, "|") != strings.Join(want, "|") {
		t.Fatalf("got %q", got)
	}
}

func TestWaitHealthyStopsWhenProcessExits(t *testing.T) {
	exited := make(chan struct{})
	close(exited)
	start := time.Now()
	if waitHealthy(exited, "http://localhost:1", time.Minute) {
		t.Fatal("must report unhealthy")
	}
	if time.Since(start) > 5*time.Second {
		t.Fatal("must return quickly once the process has exited")
	}
}
