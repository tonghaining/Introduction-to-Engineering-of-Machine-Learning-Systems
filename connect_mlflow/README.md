# Remote Host SSH Connectivity Test

This repository provides simple scripts that allow students to test SSH connectivity to a remote host on macOS, Linux, and Windows.  
The script asks the user for:
- Remote host IP
- SSH username
- Path to their private key

It then attempts to connect and prints whether the connection succeeded.

## Files

- `connect_test_mac_linux.sh`  
  Bash script for macOS and Linux.

- `connect_test_windows.ps1`  
  PowerShell script for Windows.

## Usage

### 1. macOS / Linux

1. Open a terminal.
2. Run the script:
    ```bash
    ./connect_test_mac_linux.sh
    ```
3. Follow the prompts to enter the remote host IP, SSH username, and path to your private key.

### 2. Windows
1. Right-click Start and choose "Windows PowerShell".
2. Run the script:
    ```powershell
    .\connect_test_windows.ps1
    ```
    If PowerShell blocks the script, enable script execution temporarily:
    ```powershell
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
    ```
3. Follow the prompts to enter the remote host IP, SSH username, and path to your private key.

## Notes
- Ensure your private key has the correct permissions (e.g., `chmod 600` on macOS/Linux).
- Windows requires SSH client support, available in recent versions of Windows 10 and later.