<# :
@echo off
REM Double-click to start ArtStitch. A copy still running on the
REM port is stopped first, so the page never talks to an older server behind
REM the current files. Below this header the file is PowerShell, which can see
REM what each process was started with.
setlocal
cd /d "%~dp0"
set "SELF=%~f0"
powershell -NoProfile -Command "Invoke-Expression (Get-Content -Raw -LiteralPath $env:SELF)"
pause
exit /b
#>
$app = Split-Path -Parent $env:SELF
$script = 'app.py'
$server = Join-Path $app $script
$port = 8800
if ($env:PORT) { $port = [int]$env:PORT }

# A process is ours if it is a Python whose command line names this folder's
# server. Windows does not say which folder a process started in, so a bare
# name counts as this folder's.
function Test-Ours($id) {
    $p = Get-CimInstance Win32_Process -Filter "ProcessId = $id"
    if ($p.Name -notlike 'python*') { return $false }
    $names = $server, $script, (Split-Path -Leaf $script)
    foreach ($word in [regex]::Matches([string]$p.CommandLine, '"[^"]*"|\S+')) {
        if ($names -contains $word.Value.Trim('"').Replace('/', '\')) { return $true }
    }
    return $false
}

# Stop our old copy; step past a port some other program holds.
for ($n = 0; $n -lt 20; $n++) {
    $id = Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue |
        Select-Object -First 1 -ExpandProperty OwningProcess
    if ($null -eq $id) { break }
    if (-not (Test-Ours $id)) { $port++; continue }
    Write-Host "Stopping the copy already running on port $port (pid $id)."
    taskkill /T /F /PID $id *> $null
    Wait-Process -Id $id -Timeout 5 -ErrorAction SilentlyContinue
}

# Open the browser once the server answers.
$null = Start-Job -ArgumentList $port -ScriptBlock {
    param($port)
    for ($i = 0; $i -lt 120; $i++) {
        try { [Net.Sockets.TcpClient]::new('127.0.0.1', $port).Close(); Start-Process "http://127.0.0.1:$port"; return }
        catch { Start-Sleep -Milliseconds 500 }
    }
}
& python $server --port $port
