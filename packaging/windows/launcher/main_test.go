package main

import (
	"os"
	"path/filepath"
	"testing"
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
