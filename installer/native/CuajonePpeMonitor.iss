; SPDX-License-Identifier: AGPL-3.0-only

#ifndef StageDir
  #error StageDir must be provided by build-installer.ps1
#endif
#ifndef OutputDir
  #error OutputDir must be provided by build-installer.ps1
#endif
#ifndef AppVersion
  #define AppVersion "0.1.0-internal.3"
#endif
#ifndef FileVersion
  #define FileVersion "0.1.0.3"
#endif
#ifndef OutputBaseFilename
  #define OutputBaseFilename "CuajonePPEMonitor-Internal-Setup"
#endif
#ifndef SetupOriginalFilename
  #define SetupOriginalFilename "CuajonePPEMonitorSetup.exe"
#endif

[Setup]
AppId={{88A886C2-8F6D-4669-B6FB-7DFC1E7B0397}
AppName=Cuajone PPE Monitor
AppVersion={#AppVersion}
#ifdef PreviewBuild
AppVerName=Cuajone PPE Monitor {#AppVersion} (Internal Development)
#else
AppVerName=Cuajone PPE Monitor {#AppVersion}
#endif
AppPublisher=Cuajone PPE Monitor Project
AppPublisherURL=https://github.com/jeanpaulandradeleiva-crypto/CAMARAS-IP-CUAJONE
AppSupportURL=https://github.com/jeanpaulandradeleiva-crypto/CAMARAS-IP-CUAJONE/issues
AppUpdatesURL=https://github.com/jeanpaulandradeleiva-crypto/CAMARAS-IP-CUAJONE/releases
#ifdef PreviewBuild
AppComments=Open-source AGPL-3.0 internal preview. Pilot signing and real-engine validation limitations apply.
#else
AppComments=Open-source AGPL-3.0 release. Third-party components retain their own license terms.
#endif
DefaultDirName={autopf}\Cuajone PPE Monitor
DefaultGroupName=Cuajone PPE Monitor
DisableProgramGroupPage=auto
PrivilegesRequired=admin
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
SetupArchitecture=x64
MinVersion=10.0.17763
OutputDir={#OutputDir}
OutputBaseFilename={#OutputBaseFilename}
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
SetupIconFile={#StageDir}\CuajonePPEMonitor.ico
LicenseFile={#StageDir}\licenses\AGPL-3.0.txt
UninstallDisplayIcon={app}\CuajonePPEMonitor.ico
#ifdef PreviewBuild
UninstallDisplayName=Cuajone PPE Monitor {#AppVersion} (Internal Development)
#else
UninstallDisplayName=Cuajone PPE Monitor {#AppVersion}
#endif
VersionInfoVersion={#FileVersion}
VersionInfoProductVersion={#FileVersion}
VersionInfoCompany=Cuajone PPE Monitor Project
#ifdef PreviewBuild
VersionInfoDescription=Cuajone PPE Monitor open-source internal preview installer
#else
VersionInfoDescription=Cuajone PPE Monitor open-source installer
#endif
VersionInfoProductName=Cuajone PPE Monitor
VersionInfoOriginalFileName={#SetupOriginalFilename}
CloseApplications=yes
RestartApplications=no
SetupLogging=yes
UsePreviousAppDir=yes
DisableWelcomePage=no
#ifdef SignToolName
SignTool={#SignToolName}
SignedUninstaller=yes
#else
SignedUninstaller=no
#endif

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "spanish"; MessagesFile: "compiler:Languages\Spanish.isl"

[CustomMessages]
english.DesktopHelpShortcut=Create a desktop shortcut for command-line help
spanish.DesktopHelpShortcut=Crear un acceso directo al escritorio para la ayuda de linea de comandos
english.ReadmeDescription=View the deployment and license README
spanish.ReadmeDescription=Ver la guia README de despliegue y licencias

[Tasks]
Name: "desktophelp"; Description: "{cm:DesktopHelpShortcut}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Dirs]
Name: "{app}\runtime\models"; Flags: uninsneveruninstall
Name: "{app}\runtime\config"; Flags: uninsneveruninstall
Name: "{app}\runtime\output"; Flags: uninsneveruninstall
Name: "{app}\runtime\logs"; Flags: uninsneveruninstall

[Files]
Source: "{#StageDir}\bin\*"; DestDir: "{app}\bin"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#StageDir}\docs\*"; DestDir: "{app}\docs"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#StageDir}\licenses\*"; DestDir: "{app}\licenses"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#StageDir}\CuajonePPEMonitor.ico"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#StageDir}\build-metadata.json"; DestDir: "{app}\manifest"; Flags: ignoreversion
Source: "{#StageDir}\SHA256SUMS.txt"; DestDir: "{app}\manifest"; Flags: ignoreversion

[Icons]
Name: "{group}\Cuajone PPE Monitor - README"; Filename: "{app}\docs\README.md"
Name: "{group}\Cuajone PPE Monitor - Command Help"; Filename: "{cmd}"; Parameters: "/K """"{app}\bin\cuajone_native.exe"" --help"""; WorkingDir: "{app}\bin"
Name: "{autodesktop}\Cuajone PPE Monitor - Command Help"; Filename: "{cmd}"; Parameters: "/K """"{app}\bin\cuajone_native.exe"" --help"""; WorkingDir: "{app}\bin"; Tasks: desktophelp

[Run]
Filename: "{app}\docs\README.md"; Description: "{cm:ReadmeDescription}"; Flags: postinstall shellexec skipifsilent unchecked

[Code]
procedure RunIcacls(const Directory, Arguments: String);
var
  ExitCode: Integer;
  Parameters: String;
begin
  Parameters := '"' + Directory + '" ' + Arguments;
  Log('Applying hardened ACL: icacls ' + Parameters);
  if (not Exec(ExpandConstant('{sys}\icacls.exe'), Parameters, '', SW_HIDE,
      ewWaitUntilTerminated, ExitCode)) or (ExitCode <> 0) then
    RaiseException('ACL hardening failed for ' + Directory +
      ' (icacls exit code ' + IntToStr(ExitCode) + ').');
end;

procedure HardenDirectory(const Directory, UsersPermission: String);
begin
  if not DirExists(Directory) then
    RaiseException('ACL hardening target does not exist: ' + Directory);

  { Reset upgrades as well as fresh installs, remove inherited grants, then apply
    an exact localized-name-independent ACL using well-known SIDs. }
  RunIcacls(Directory, '/reset /T');
  RunIcacls(Directory, '/inheritance:r');
  RunIcacls(Directory,
    '/grant:r "*S-1-5-18:(OI)(CI)F" "*S-1-5-32-544:(OI)(CI)F" ' +
    '"*S-1-5-32-545:(OI)(CI)' + UsersPermission + '" /T');
end;

function HasNvidiaRuntime: Boolean;
var
  ExitCode: Integer;
  NvidiaSmi: String;
begin
  NvidiaSmi := ExpandConstant('{sys}\nvidia-smi.exe');
  Result := FileExists(ExpandConstant('{sys}\nvcuda.dll')) and
    FileExists(NvidiaSmi) and
    Exec(NvidiaSmi, '-L', '', SW_HIDE, ewWaitUntilTerminated, ExitCode) and
    (ExitCode = 0);
end;

function InitializeSetup: Boolean;
begin
  Result := False;
  if not IsWin64 then
  begin
    MsgBox('Cuajone PPE Monitor requires 64-bit Windows.' + #13#10 +
      'Cuajone PPE Monitor requiere Windows de 64 bits.', mbCriticalError, MB_OK);
    Exit;
  end;
  if not HasNvidiaRuntime then
  begin
    MsgBox('A working NVIDIA GPU and driver were not detected. No driver will be downloaded or installed.' + #13#10 +
      'No se detecto una GPU NVIDIA con controlador operativo. No se descargara ni instalara ningun controlador.',
      mbCriticalError, MB_OK);
    Exit;
  end;
  Result := True;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
  begin
    HardenDirectory(ExpandConstant('{app}\bin'), 'RX');
    HardenDirectory(ExpandConstant('{app}\runtime\models'), 'M');
    HardenDirectory(ExpandConstant('{app}\runtime\config'), 'M');
    HardenDirectory(ExpandConstant('{app}\runtime\output'), 'M');
    HardenDirectory(ExpandConstant('{app}\runtime\logs'), 'M');
  end;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if (CurUninstallStep = usUninstall) and (not UninstallSilent) then
    MsgBox('Runtime models, configuration, output, and logs are preserved and must be deleted manually if no longer needed.' + #13#10 +
      'Los modelos, la configuracion, las salidas y los registros se conservan y deben eliminarse manualmente si ya no se necesitan.',
      mbInformation, MB_OK);
end;
