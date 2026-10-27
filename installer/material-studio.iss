; Generated build-config.iss and package-files.iss are supplied by build_installer.py.
; Never compile against a recursive wildcard or an unverified portable directory.
#include "build-config.iss"

[Setup]
AppId={{C0673919-301E-4B52-AD89-A3960836BB2C}
AppName=Material Studio
AppVersion={#AppVersion}
AppVerName=Material Studio {#AppVersion}
VersionInfoVersion={#AppVersion}.0
VersionInfoProductName=Material Studio
VersionInfoDescription=Material Studio offline installer
DefaultDirName={localappdata}\Programs\Material Studio
DefaultGroupName=Material Studio
PrivilegesRequired=lowest
ArchitecturesAllowed=x64os
ArchitecturesInstallIn64BitMode=x64os
; Windows 10 1903 is the upstream GPUI/Zed floor; see docs/INSTALLATION.md.
MinVersion=10.0.18362
DisableDirPage=no
DisableProgramGroupPage=no
AllowNoIcons=yes
WizardStyle=modern
SetupIconFile={#AppIcon}
UninstallDisplayIcon={app}\MaterialStudio.ico
UninstallDisplayName=Material Studio
OutputDir={#BuildOutput}
OutputBaseFilename=MaterialStudio-{#AppVersion}-windows-x64-Setup
Compression=lzma2/ultra64
SolidCompression=yes
LZMANumBlockThreads=2
DiskSpanning={#Spanning}
DiskSliceSize=2000000000
SlicesPerDisk=1
CloseApplications=yes
RestartApplications=no
ChangesEnvironment=no
Uninstallable=yes

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Shortcuts:"; Flags: unchecked

[Files]
#include "package-files.iss"

[Icons]
Name: "{group}\Material Studio"; Filename: "{app}\MaterialStudio.exe"; WorkingDir: "{app}"; IconFilename: "{app}\MaterialStudio.ico"
Name: "{group}\Uninstall Material Studio"; Filename: "{uninstallexe}"
Name: "{userdesktop}\Material Studio"; Filename: "{app}\MaterialStudio.exe"; WorkingDir: "{app}"; IconFilename: "{app}\MaterialStudio.ico"; Tasks: desktopicon

[Run]
Filename: "{app}\MaterialStudio.exe"; WorkingDir: "{app}"; Description: "Launch Material Studio"; Flags: nowait postinstall skipifsilent

; Intentionally no [InstallDelete], [UninstallDelete], or recursive deletion code.
; Inno removes only logged installed files/shortcuts and empty directories.
