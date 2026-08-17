"""Helpers for serving the demo UI to other devices on the same network.

The demo is normally shown on one laptop and watched on others, so the server
binds every interface by default and the startup banner prints the URL that
those other devices should actually type.
"""
from __future__ import annotations

import socket

FIREWALL_HINT = (
    'New-NetFirewallRule -DisplayName "TNLA Video RAG" -Direction Inbound '
    '-Protocol TCP -LocalPort {port} -Action Allow -Profile Any'
)


def lan_ip() -> str | None:
    """This machine's LAN IPv4, or None if it can't be determined.

    Opens a UDP socket toward a public address. Nothing is sent — the OS just
    resolves which interface it *would* route through, which is the one other
    devices on the network can reach. Picking the default route this way avoids
    returning a virtual-adapter address (Hyper-V, WSL, VPN), which looks right
    but is unreachable from anywhere else.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
    except OSError:
        return None
    finally:
        sock.close()
    return None if ip.startswith("127.") else ip


def startup_banner(host: str, port: int) -> str:
    """The block printed when the UI starts, naming the URL to share."""
    lines = ["", "  TNLA Video RAG is running", ""]
    lines.append(f"    On this machine :  http://127.0.0.1:{port}")

    if host in ("127.0.0.1", "localhost"):
        lines += [
            "",
            "    Other devices   :  not reachable — bound to this machine only.",
            "                       Restart with --host 0.0.0.0 to share it.",
        ]
    else:
        ip = lan_ip()
        if ip:
            lines += [
                f"    Other devices   :  http://{ip}:{port}   <- share this one",
                "",
                "    If another device can't connect, allow the port through",
                "    Windows Firewall once, in an Administrator PowerShell:",
                "",
                f"      {FIREWALL_HINT.format(port=port)}",
            ]
        else:
            lines += [
                "    Other devices   :  no network address found — is Wi-Fi connected?",
            ]

    lines.append("")
    return "\n".join(lines)
