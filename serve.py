"""Start the demo UI.

    python serve.py            # http://127.0.0.1:8000
    python serve.py --port 9000
"""
from __future__ import annotations

import argparse


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    import uvicorn
    print(f"TNLA Video RAG  ->  http://{args.host}:{args.port}")
    uvicorn.run("vrag.app:app", host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
