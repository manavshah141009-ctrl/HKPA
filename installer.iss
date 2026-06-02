[Setup]
AppName=Personal Dictation Assistant
AppVersion=2.0
AppPublisher=Manav Shah
AppPublisherURL=https://github.com/manavshah141009-ctrl/HKPA
DefaultDirName={autopf}\PersonalDictationAssistant
DefaultGroupName=Personal Dictation Assistant
UninstallDisplayIcon={app}\PersonalDictationAssistant.exe
Compression=lzma2
SolidCompression=yes
OutputDir=dist\setup
OutputBaseFilename=PersonalDictationAssistant_Setup
PrivilegesRequired=lowest
; Optional: If you compile a custom icon, use it here too:
; SetupIconFile=app_icon.ico

[Tasks]
Name: "startupicon"; Description: "Launch on Windows Startup"; GroupDescription: "Startup options:"

[Files]
; Grab everything from the PyInstaller dist output directory
Source: "dist\PersonalDictationAssistant\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{userprograms}\Personal Dictation Assistant"; Filename: "{app}\PersonalDictationAssistant.exe"
Name: "{userprograms}\{cm:UninstallProgram,Personal Dictation Assistant}"; Filename: "{uninstallexe}"
Name: "{autostartup}\Personal Dictation Assistant"; Filename: "{app}\PersonalDictationAssistant.exe"; Tasks: startupicon

[Run]
Filename: "{app}\PersonalDictationAssistant.exe"; Description: "{cm:LaunchProgram,Personal Dictation Assistant}"; Flags: nowait postinstall skipifsilent
