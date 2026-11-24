; Meridian – Windows-Installer (NSIS 3)
; Gebaut von packaging/windows/build.sh:  makensis -DVERSION=x.y.z -DSTAGE=<ordner> -DOUTFILE=<datei> meridian.nsi
Unicode true
SetCompressor /SOLID lzma
RequestExecutionLevel user          ; ohne Admin-Rechte, nur für den aktuellen Benutzer

!ifndef VERSION
  !define VERSION "0.0.0"
!endif
!define APPNAME "Meridian"
!define UNINSTKEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\Meridian"

Name "${APPNAME}"
OutFile "${OUTFILE}"
InstallDir "$LOCALAPPDATA\Programs\Meridian"
InstallDirRegKey HKCU "Software\Meridian" "InstallDir"
BrandingText "Meridian ${VERSION}"
VIProductVersion "${VERSION}.0"
VIAddVersionKey "ProductName" "Meridian"
VIAddVersionKey "FileDescription" "Meridian Setup"
VIAddVersionKey "FileVersion" "${VERSION}"
VIAddVersionKey "ProductVersion" "${VERSION}"
VIAddVersionKey "LegalCopyright" "MIT License"

!include "MUI2.nsh"
!define MUI_ICON "${STAGE}\meridian.ico"
!define MUI_UNICON "${STAGE}\meridian.ico"
!define MUI_ABORTWARNING
!define MUI_WELCOMEPAGE_TITLE "Meridian ${VERSION} installieren"
!define MUI_WELCOMEPAGE_TEXT "Meridian ist ein eigenes Sprachmodell – von Grund auf gebaut, mit Daten von Hugging Face trainiert und live sichtbar gemacht.$\r$\n$\r$\nDer Installer braucht keine Admin-Rechte. Beim ersten Start lädt Meridian einmalig Python und PyTorch herunter (etwa 3 GB, 5–15 Minuten). Mit einer NVIDIA-Grafikkarte wird automatisch die GPU-Version installiert.$\r$\n$\r$\nWeiter mit »Weiter«."
!define MUI_FINISHPAGE_RUN "$INSTDIR\Meridian.exe"
!define MUI_FINISHPAGE_RUN_TEXT "Meridian jetzt starten (richtet beim ersten Mal alles ein)"
!define MUI_FINISHPAGE_SHOWREADME ""
!define MUI_FINISHPAGE_SHOWREADME_TEXT "Verknüpfung auf dem Desktop anlegen"
!define MUI_FINISHPAGE_SHOWREADME_FUNCTION DesktopShortcut

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE "${STAGE}\app\LICENSE"
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_LANGUAGE "German"

Function DesktopShortcut
  CreateShortCut "$DESKTOP\Meridian.lnk" "$INSTDIR\Meridian.exe" "" "$INSTDIR\Meridian.exe" 0
FunctionEnd

Section "Meridian" SecMain
  SectionIn RO
  ; Alte Programmdateien ersetzen – deine Daten liegen woanders (%LOCALAPPDATA%\Meridian) und bleiben erhalten
  RMDir /r "$INSTDIR\app"
  SetOutPath "$INSTDIR"
  File "${STAGE}\Meridian.exe"
  File "${STAGE}\uv.exe"
  File "${STAGE}\meridian.ico"
  File "${STAGE}\THIRD-PARTY-NOTICES.txt"
  SetOutPath "$INSTDIR\app"
  File /r "${STAGE}\app\*.*"
  SetOutPath "$INSTDIR"

  CreateDirectory "$SMPROGRAMS\Meridian"
  CreateShortCut "$SMPROGRAMS\Meridian\Meridian.lnk" "$INSTDIR\Meridian.exe" "" "$INSTDIR\Meridian.exe" 0
  CreateShortCut "$SMPROGRAMS\Meridian\Meridian neu einrichten.lnk" "$INSTDIR\Meridian.exe" "--reinstall" "$INSTDIR\Meridian.exe" 0
  CreateShortCut "$SMPROGRAMS\Meridian\Meridian deinstallieren.lnk" "$INSTDIR\Uninstall.exe"

  WriteRegStr HKCU "Software\Meridian" "InstallDir" "$INSTDIR"
  WriteRegStr HKCU "${UNINSTKEY}" "DisplayName" "Meridian"
  WriteRegStr HKCU "${UNINSTKEY}" "DisplayVersion" "${VERSION}"
  WriteRegStr HKCU "${UNINSTKEY}" "Publisher" "Vanzal"
  WriteRegStr HKCU "${UNINSTKEY}" "DisplayIcon" "$INSTDIR\Meridian.exe"
  WriteRegStr HKCU "${UNINSTKEY}" "InstallLocation" "$INSTDIR"
  WriteRegStr HKCU "${UNINSTKEY}" "UninstallString" '"$INSTDIR\Uninstall.exe"'
  WriteRegDWORD HKCU "${UNINSTKEY}" "NoModify" 1
  WriteRegDWORD HKCU "${UNINSTKEY}" "NoRepair" 1
  WriteUninstaller "$INSTDIR\Uninstall.exe"
SectionEnd

Section "Uninstall"
  Delete "$DESKTOP\Meridian.lnk"
  RMDir /r "$SMPROGRAMS\Meridian"
  RMDir /r "$INSTDIR\app"
  Delete "$INSTDIR\Meridian.exe"
  Delete "$INSTDIR\uv.exe"
  Delete "$INSTDIR\meridian.ico"
  Delete "$INSTDIR\THIRD-PARTY-NOTICES.txt"
  Delete "$INSTDIR\Uninstall.exe"
  RMDir "$INSTDIR"
  DeleteRegKey HKCU "${UNINSTKEY}"
  DeleteRegKey HKCU "Software\Meridian"
  ; Daten nur auf ausdrücklichen Wunsch löschen (Standard: behalten)
  IfFileExists "$LOCALAPPDATA\Meridian\*.*" 0 done
  MessageBox MB_YESNO|MB_ICONQUESTION|MB_DEFBUTTON2 "Auch deine Meridian-Daten löschen?$\r$\n$\r$\nDas sind trainierte Modelle, vorbereitete Datensätze, Einstellungen und die Python-Umgebung in:$\r$\n$LOCALAPPDATA\Meridian$\r$\n$\r$\n»Nein« behält alles für eine spätere Neuinstallation." IDNO done
  RMDir /r "$LOCALAPPDATA\Meridian"
  done:
SectionEnd
