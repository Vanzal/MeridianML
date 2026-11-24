//go:build windows

package main

import "syscall"

// Konsole auf UTF-8 stellen, damit Umlaute (ä, ö, ü) und Blockzeichen richtig erscheinen.
func init() {
	k := syscall.NewLazyDLL("kernel32.dll")
	_, _, _ = k.NewProc("SetConsoleOutputCP").Call(65001)
	_, _, _ = k.NewProc("SetConsoleCP").Call(65001)
	_, _, _ = k.NewProc("SetConsoleTitleW").Call(uintptr(unsafePtr("Meridian")))
}
