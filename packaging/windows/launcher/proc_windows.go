//go:build windows

package main

import (
	"fmt"
	"io"
	"os"
	"os/exec"
	"syscall"
	"unsafe"
)

const createNoWindow = 0x08000000

// hideWindow: Kind-Prozess (uv, Python) ohne eigenes schwarzes Konsolenfenster starten.
func hideWindow(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{HideWindow: true, CreationFlags: createNoWindow}
}

// killTree beendet den Prozess und alle seine Unterprozesse.
func killTree(cmd *exec.Cmd) {
	k := exec.Command("taskkill", "/T", "/F", "/PID", fmt.Sprint(cmd.Process.Pid))
	hideWindow(k)
	if k.Run() != nil {
		_ = cmd.Process.Kill()
	}
}

// attachConsole: Meridian.exe ist ein Fenster-Programm ohne Konsole. Für den Browser-/Server-Modus wird eine
// Konsole geholt – die des aufrufenden Terminals, sonst eine neue – und auf UTF-8 gestellt.
func attachConsole() {
	k := syscall.NewLazyDLL("kernel32.dll")
	const attachParentProcess = ^uintptr(0) // (DWORD)-1
	if r, _, _ := k.NewProc("AttachConsole").Call(attachParentProcess); r == 0 {
		_, _, _ = k.NewProc("AllocConsole").Call()
	}
	if f, err := os.OpenFile("CONOUT$", os.O_RDWR, 0); err == nil {
		os.Stdout, os.Stderr = f, f
	}
	if f, err := os.OpenFile("CONIN$", os.O_RDWR, 0); err == nil {
		os.Stdin = f
	}
	_, _, _ = k.NewProc("SetConsoleOutputCP").Call(65001)
	_, _, _ = k.NewProc("SetConsoleCP").Call(65001)
	title, _ := syscall.UTF16PtrFromString("Meridian")
	_, _, _ = k.NewProc("SetConsoleTitleW").Call(uintptr(unsafe.Pointer(title)))
}

func logWriter(f *os.File) io.Writer {
	if f == nil {
		return io.Discard
	}
	return f
}
