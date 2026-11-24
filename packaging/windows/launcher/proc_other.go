//go:build !windows

package main

import (
	"os"
	"os/exec"
)

// Auf anderen Systemen (nur für Tests und Entwicklung): kein App-Fenster, normale Prozesse.
func runWindowed(opt options, home string, logFile *os.File) bool { return false }
func hideWindow(cmd *exec.Cmd)                                    {}
func killTree(cmd *exec.Cmd)                                      { _ = cmd.Process.Kill() }
func attachConsole()                                              {}
