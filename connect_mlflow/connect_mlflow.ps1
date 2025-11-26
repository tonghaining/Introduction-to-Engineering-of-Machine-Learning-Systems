Write-Host "==============================="
Write-Host " MLflow Remote Connection Setup"
Write-Host "==============================="

$RemoteHost = Read-Host "Enter remote host IP"
$RemoteUser = Read-Host "Enter SSH username"
$KeyPath = Read-Host "Enter path to SSH private key (e.g. $HOME\.ssh\id_rsa)"

if (!(Test-Path $KeyPath)) {
    Write-Host "ERROR: SSH key not found at $KeyPath"
    exit
}

Write-Host ""
Write-Host "Connecting to $RemoteUser@$RemoteHost ..."
Write-Host "Forwarding ports:"
Write-Host "  MLflow → localhost:5000"
Write-Host "  MinIO UI → localhost:9001"
Write-Host ""

ssh -i $KeyPath `
    -L 5000:localhost:5000 `
    -L 9001:localhost:9001 `
    "$RemoteUser@$RemoteHost"
