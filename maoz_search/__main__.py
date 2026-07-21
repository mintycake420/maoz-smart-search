"""Launch with: python -m maoz_search"""

from __future__ import annotations

import argparse
import sys
import threading
import time
import webbrowser
from pathlib import Path

from .embeddings import EncoderUnavailableError
from .artifacts import ArtifactMismatchError
from .search import SearchEngine
from .web import create_app

WARMUP_QUERY = "חינוך בלתי פורמלי"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MAOZ Hebrew semantic-search POC (synthetic data only)")
    parser.add_argument("--host", default="127.0.0.1", help="bind address; defaults to local machine only")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true", help="do not open the local UI automatically")
    parser.add_argument("--model-dir", type=Path, default=None, help="explicit local BGE-M3 ONNX directory")
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="skip the startup query; the first search in the UI then pays the model load",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print("MAOZ POC · synthetic data only · local inference")
    try:
        engine = SearchEngine.from_default(model_dir=args.model_dir)
    except (EncoderUnavailableError, ArtifactMismatchError) as exc:
        # These are the two ways a checkout can be incomplete. Both messages are
        # actionable, so surface them plainly instead of a traceback.
        print(f"\nCannot start: {exc}", file=sys.stderr)
        raise SystemExit(1) from None

    # Load the encoder before the browser opens. ONNX session creation over the
    # 580 MB graph dominates the first query and has been measured at 146 seconds on
    # a genuinely cold file cache; paying it here means the demo does not open onto a
    # spinner.
    if not args.no_warmup:
        print("Loading the local encoder (one-time; tens of seconds warm, minutes on a cold cache)…", flush=True)
        started = time.perf_counter()
        try:
            engine.search(WARMUP_QUERY)
        except (EncoderUnavailableError, ArtifactMismatchError) as exc:
            print(f"\nCannot start: {exc}", file=sys.stderr)
            raise SystemExit(1) from None
        print(f"Encoder ready in {time.perf_counter() - started:.1f}s. Searches are now sub-second.")

    app = create_app(engine)
    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    print(f"\n{url}\nPress CTRL+C to stop.")
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False, threaded=True)


if __name__ == "__main__":
    main()
