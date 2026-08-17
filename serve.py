"""Start the demo UI.

    python serve.py                      # shared on the local network
    python serve.py --port 9000
    python serve.py --host 127.0.0.1     # this machine only
"""
from __future__ import annotations

import argparse


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0",
                    help="0.0.0.0 (default) serves other devices on the network; "
                         "127.0.0.1 restricts it to this machine")
    args = ap.parse_args()
    import uvicorn
    from vrag.net import startup_banner
    print(startup_banner(args.host, args.port))
    uvicorn.run("vrag.app:app", host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
