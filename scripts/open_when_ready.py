"""
Cho backend san sang (port 8000 LISTENING) roi mo trinh duyet vao Dashboard.

Chay detached tu run_demo.bat song song voi backend (chay foreground). Poll
port thay vi timeout cung, nen khong mo trinh duyet qua som (luc pipeline AI /
torch chua nap xong -> trang trang).

Chay:  python scripts/open_when_ready.py [port]
"""

from __future__ import annotations

import socket
import sys
import time
import webbrowser

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
URL = f"http://localhost:{PORT}"
TIMEOUT_S = 120  # pipeline AI co the mat vai chuc giay de nap model


def port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        return s.connect_ex(("127.0.0.1", port)) == 0


def main() -> int:
    deadline = time.monotonic() + TIMEOUT_S
    while time.monotonic() < deadline:
        if port_open(PORT):
            # Them 1s cho FastAPI dang ky xong route static
            time.sleep(1.0)
            print(f"[OPEN] {URL}")
            webbrowser.open(URL)
            return 0
        time.sleep(1.5)
    print(f"[TIMEOUT] Server chua len sau {TIMEOUT_S}s — mo thu cong {URL}",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
