#!/usr/bin/env python3
import argparse
import csv
import shlex
import subprocess
import sys
import tempfile
import textwrap
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

REPO_PATH = "/home/student/Introduction-to-Engineering-of-Machine-Learning-Systems"
HTTPS_ORIGIN = "https://github.com/tonghaining/Introduction-to-Engineering-of-Machine-Learning-Systems.git"
CONNECT_TIMEOUT = 10
DEFAULT_IDENTITY_PATH = "~/.ssh/config"
DEFAULT_TEMPLATE_PATH = Path("ssh_config.template")

URLS_TO_CHECK = [
    "kserve-gateway.local",
    "ml-pipeline-ui.local",
    "mlflow-server.local",
    "mlflow-minio-ui.local",
    "mlflow-minio.local",
    "prometheus-server.local",
    "grafana-server.local",
    "evidently-monitor-ui.local",
]


def run(
    cmd: Sequence[str],
    *,
    check: bool = True,
    capture_output: bool = False,
    input_text: Optional[str] = None,
):
    result = subprocess.run(cmd, text=True, capture_output=capture_output, input=input_text)
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
    return result


@dataclass
class Client:
    row_index: int  # 1-based CSV row number
    email: str
    group_id: str
    student_id: str
    client_internal_ip: str
    client_public_ip: str
    remote_internal_ip: str
    remote_public_ip: str
    public_key: str
    private_key: str
    tested_flag: str

    @property
    def label(self) -> str:
        gid = self.group_id or "?"
        sid = self.student_id or "?"
        return f"group{gid}-student{sid}"

    @property
    def candidate_ips(self) -> List[str]:
        return [ip for ip in (self.client_public_ip, self.client_internal_ip) if ip]

    def matches_ip(self, ip: str) -> bool:
        return ip in self.candidate_ips

    def preferred_ip(self) -> Optional[str]:
        return self.candidate_ips[0] if self.candidate_ips else None

    def preferred_remote_ip(self) -> Optional[str]:
        if self.remote_public_ip:
            return self.remote_public_ip
        if self.remote_internal_ip:
            return self.remote_internal_ip
        return None


def read_and_fill(csv_path: Path):
    with csv_path.open(newline="") as f:
        rows = list(csv.reader(f))

    if len(rows) < 2:
        raise SystemExit(f"{csv_path} does not look like the expected sheet (needs at least 2 header rows)")

    headers = rows[1]
    data_rows = rows[2:]
    prev = ["" for _ in headers]
    filled_rows: List[List[str]] = []

    for row in data_rows:
        row = (row + [""] * len(headers))[: len(headers)]
        filled = []
        for idx, val in enumerate(row):
            if val != "":
                prev[idx] = val
            filled.append(prev[idx])
        if any(cell != "" for cell in filled):
            filled_rows.append(filled)

    return rows[:2], headers, filled_rows


def write_processed_csv(header_rows: List[List[str]], filled_rows: List[List[str]], output_path: Path):
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(header_rows)
        writer.writerows(filled_rows)
    print(f"Wrote filled CSV to {output_path}")


def build_clients(headers: List[str], filled_rows: List[List[str]]) -> List[Client]:
    clients: List[Client] = []
    for idx, row in enumerate(filled_rows, start=3):
        padded = (row + [""] * len(headers))[: len(headers)]
        clients.append(
            Client(
                row_index=idx,
                email=padded[0],
                group_id=padded[1],
                student_id=padded[2],
                client_internal_ip=padded[3],
                client_public_ip=padded[4],
                remote_internal_ip=padded[5],
                remote_public_ip=padded[6],
                public_key=padded[7],
                private_key=padded[8],
                tested_flag=padded[9] if len(padded) > 9 else "",
            )
        )
    return clients


def ssh_base(ip: str, key_path: Path) -> List[str]:
    return [
        "ssh",
        "-i",
        str(key_path),
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={CONNECT_TIMEOUT}",
        "-o",
        "StrictHostKeyChecking=accept-new",
        f"student@{ip}",
    ]


def ssh_run(ip: str, key_path: Path, remote_cmd: str, *, check: bool = True, capture_output: bool = False):
    cmd = ssh_base(ip, key_path) + [remote_cmd]
    return run(cmd, check=check, capture_output=capture_output)


def ssh_run_with_input(ip: str, key_path: Path, remote_cmd: str, input_text: str):
    cmd = ssh_base(ip, key_path) + [remote_cmd]
    return run(cmd, input_text=input_text)


def repo_exists(ip: str, key_path: Path, repo_path: str) -> bool:
    repo_q = shlex.quote(repo_path)
    cmd = f"test -d {repo_q} && test -d {repo_q}/.git"
    result = ssh_run(ip, key_path, cmd, check=False)
    return result.returncode == 0


def git_status(ip: str, key_path: Path, repo_path: str):
    repo_q = shlex.quote(repo_path)
    status = ssh_run(
        ip,
        key_path,
        f"cd {repo_q} && git status -sb",
        check=False,
        capture_output=True,
    )
    head = ssh_run(
        ip,
        key_path,
        f"cd {repo_q} && git log -1 --oneline",
        check=False,
        capture_output=True,
    )
    remote_url = ssh_run(
        ip,
        key_path,
        f"cd {repo_q} && git remote get-url origin",
        check=False,
        capture_output=True,
    )

    def _clean(res: subprocess.CompletedProcess[str]) -> str:
        if res.stdout:
            return res.stdout.strip()
        if res.stderr:
            return res.stderr.strip()
        return ""

    return _clean(status), _clean(head), _clean(remote_url)


def update_remote_and_reset(ip: str, key_path: Path, repo_path: str):
    repo_q = shlex.quote(repo_path)
    set_url = ssh_run(
        ip,
        key_path,
        f"cd {repo_q} && git remote set-url origin {shlex.quote(HTTPS_ORIGIN)}",
        check=False,
    )
    if set_url.returncode != 0:
        ssh_run(
            ip,
            key_path,
            f"cd {repo_q} && git remote add origin {shlex.quote(HTTPS_ORIGIN)}",
            check=True,
        )

    ssh_run(ip, key_path, f"cd {repo_q} && git fetch origin main", check=True)
    ssh_run(ip, key_path, f"cd {repo_q} && git reset --hard origin/main", check=True)


def run_connectivity_check(ip: str, key_path: Path):
    script = textwrap.dedent(
        """\
        import requests

        URLS = [
            "kserve-gateway.local",
            "ml-pipeline-ui.local",
            "mlflow-server.local",
            "mlflow-minio-ui.local",
            "mlflow-minio.local",
            "prometheus-server.local",
            "grafana-server.local",
            "evidently-monitor-ui.local",
        ]

        def test_connection():
            for url in URLS:
                try:
                    requests.get(f"http://{url}")
                except Exception as e:
                    print(f"Failed to connect to {url}: {e}")
                    raise

        test_connection()
        """
    )
    ssh_run_with_input(ip, key_path, "python3 -", input_text=script)


def prompt_confirm(message: str) -> bool:
    reply = input(f"{message} [y/N]: ").strip().lower()
    return reply in {"y", "yes"}


def pick_clients(all_clients: List[Client], ips: Optional[List[str]], operate_on_all: bool) -> List[Client]:
    if operate_on_all:
        return all_clients
    if not ips:
        raise SystemExit("Specify --all or at least one --client-ip")

    picked = []
    for client in all_clients:
        if any(client.matches_ip(ip) for ip in ips):
            picked.append(client)

    if not picked:
        raise SystemExit(f"No clients found matching IPs: {', '.join(ips)}")
    return picked


def write_key_to_temp(key_material: str, tmpdir: Path, label: str) -> Path:
    key_path = tmpdir / f"{label}_id_ed25519"
    key_text = key_material
    if not key_text.endswith("\n"):
        key_text += "\n"
    key_path.write_text(key_text, encoding="utf-8")
    key_path.chmod(0o600)
    return key_path


def process_client(client: Client, key_path: Path, args):
    target_ip = client.preferred_ip()
    if not target_ip:
        print(f"[{client.label}] No IP available, skipping.")
        return

    print(f"\n== {client.label} ({target_ip}) row {client.row_index} ==")

    repo_present = repo_exists(target_ip, key_path, args.repo_path)
    if not repo_present:
        print(f"[{client.label}] Repository missing at {args.repo_path} on host.")
    else:
        status, head, remote_url = git_status(target_ip, key_path, args.repo_path)
        print(f"[{client.label}] remote origin: {remote_url}")
        print(f"[{client.label}] HEAD: {head}")
        print(f"[{client.label}] git status -sb:\n{status}")

    if args.dry_run:
        return

    if repo_present:
        skip_prompt = args.all and args.force
        do_reset = skip_prompt or prompt_confirm(f"Apply remote URL + reset on {client.label} at {target_ip}?")
        if do_reset:
            update_remote_and_reset(target_ip, key_path, args.repo_path)
            print(f"[{client.label}] Updated origin and reset to origin/master.")
        else:
            print(f"[{client.label}] Skipped git changes by user.")

    run_connectivity_check(target_ip, key_path)
    print(f"[{client.label}] Connectivity checks completed.")


def render_config(template_text: str, client_ip: str, remote_ip: str, identity_path: str) -> str:
    out = template_text
    out = out.replace("<client-vm-ip>", client_ip)
    out = out.replace("<remote-vm-ip>", remote_ip)
    out = out.replace("<path-to-private-key>", identity_path)
    return out


def package_client(client: Client, output_dir: Path, template_path: Path, identity_path: str):
    target_ip = client.preferred_ip()
    remote_ip = client.preferred_remote_ip()
    if not target_ip:
        print(f"[{client.label}] No client IP available, skipping package.")
        return
    if not remote_ip:
        print(f"[{client.label}] No remote IP available, skipping package.")
        return
    if not client.private_key.strip():
        print(f"[{client.label}] Missing private key in CSV, skipping package.")
        return

    template_text = template_path.read_text(encoding="utf-8")
    config_text = render_config(template_text, target_ip, remote_ip, identity_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / f"{client.label}.zip"

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        key_file = td_path / "id_ed25519"
        key_text = client.private_key
        if not key_text.endswith("\n"):
            key_text += "\n"
        key_file.write_text(key_text, encoding="utf-8")
        key_file.chmod(0o600)

        config_file = td_path / "config"
        config_file.write_text(config_text, encoding="utf-8")

        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(key_file, arcname="id_ed25519")
            zf.write(config_file, arcname="config")

    print(f"[{client.label}] Wrote package {zip_path}")


def main():
    ap = argparse.ArgumentParser(description="Refresh student repo clones and verify connectivity on clients.")
    ap.add_argument("--csv", default="MLOp-keys - Sheet1.csv", help="Path to the raw CSV export.")
    ap.add_argument(
        "--processed-csv",
        default="processed.csv",
        help="Path to write the filled CSV when using --preprocess.",
    )
    ap.add_argument("--preprocess", action="store_true", help="Fill empty CSV cells and write processed CSV, then exit.")
    ap.add_argument("--client-ip", action="append", help="Operate on a specific client IP (public or csc-internal).")
    ap.add_argument("--all", action="store_true", help="Operate on all clients listed in the CSV.")
    ap.add_argument("--force", action="store_true", help="Skip confirmations (only honored together with --all).")
    ap.add_argument("--dry-run", action="store_true", help="Show git status/commit without changing anything.")
    ap.add_argument(
        "--repo-path",
        default=REPO_PATH,
        help="Path to the repository on the client hosts.",
    )
    ap.add_argument("--package", action="store_true", help="Create zip with private key and SSH config instead of SSHing.")
    ap.add_argument("--package-dir", default="packages", help="Where to write generated zip packages.")
    ap.add_argument(
        "--config-template",
        default=str(DEFAULT_TEMPLATE_PATH),
        help="Path to SSH config template used for --package.",
    )
    ap.add_argument(
        "--identity-file-path",
        default=DEFAULT_IDENTITY_PATH,
        help="IdentityFile path to substitute into the SSH config template.",
    )

    if len(sys.argv) == 1:
        ap.print_help()
        return

    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    header_rows, headers, filled_rows = read_and_fill(csv_path)

    if args.preprocess:
        write_processed_csv(header_rows, filled_rows, Path(args.processed_csv))
        return

    clients = build_clients(headers, filled_rows)
    targets = pick_clients(clients, args.client_ip, args.all)

    if args.package:
        template_path = Path(args.config_template)
        if not template_path.exists():
            raise SystemExit(f"Config template not found: {template_path}")
        output_dir = Path(args.package_dir)
        for client in targets:
            package_client(client, output_dir, template_path, args.identity_file_path)
        return

    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        for client in targets:
            if not client.private_key.strip():
                print(f"[{client.label}] Missing private key in CSV, skipping.")
                continue
            key_path = write_key_to_temp(client.private_key, tmpdir, client.label)
            process_client(client, key_path, args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(1)
