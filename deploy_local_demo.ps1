$ErrorActionPreference = "Stop"

$projectRoot = "C:\Users\Big Duck\proyectos\helix-proto"
$workspaceRoot = "C:\Users\Big Duck\proyectos\helix-proto\workspace-agentic"
$pythonPath = "C:\Users\Big Duck\AppData\Local\Programs\Python\Python311\python.exe"
$npxPath = "C:\Program Files\nodejs\npx.cmd"
$port = 8080

$env:PYTHONPATH = "$projectRoot\src"
$env:HELIX_WORKSPACE_ROOT = $workspaceRoot
$env:PORT = "$port"

Write-Host "Starting Helix Proto backend on port $port..."
$backend = Start-Process -FilePath $pythonPath -ArgumentList @(
    "-m",
    "helix_proto.server"
) -WorkingDirectory $projectRoot -PassThru -WindowStyle Hidden

Start-Sleep -Seconds 3

Write-Host "Starting LocalTunnel..."
$tunnel = Start-Process -FilePath $npxPath -ArgumentList @(
    "localtunnel",
    "--port",
    "$port"
) -WorkingDirectory $projectRoot -PassThru -RedirectStandardOutput "$projectRoot\tunnel.out" -RedirectStandardError "$projectRoot\tunnel.err" -WindowStyle Hidden

Start-Sleep -Seconds 8

if (Test-Path "$projectRoot\tunnel.out") {
    Get-Content "$projectRoot\tunnel.out"
}

Write-Host ""
Write-Host "Backend PID: $($backend.Id)"
Write-Host "Tunnel PID: $($tunnel.Id)"
Write-Host "Workspace Root: $workspaceRoot"
Write-Host "To stop them later:"
Write-Host "Stop-Process -Id $($backend.Id),$($tunnel.Id)"
