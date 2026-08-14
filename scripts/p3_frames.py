"""P3 runner — scene detect, extract keyframes, batched vision description.

Usage:
    python scripts/p3_frames.py --video data/videos/TNLA.mp4
    python scripts/p3_frames.py                    # picks the only .mp4 in data/videos
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from vrag import config
from vrag.frames import process_frames
from vrag.gemini import GeminiClient
from vrag.logging_setup import preview, setup_logging

log = logging.getLogger("p3")


def _default_video() -> pathlib.Path:
    vids = sorted(config.VIDEO_DIR.glob("*.mp4"))
    if len(vids) != 1:
        raise SystemExit(
            f"Expected exactly one .mp4 in {config.VIDEO_DIR}, found "
            f"{[v.name for v in vids]} — pass --video explicitly.")
    return vids[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", type=pathlib.Path, default=None)
    ap.add_argument("--video-id", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    setup_logging()
    video = args.video or _default_video()

    with GeminiClient() as c:
        art = process_frames(c, video, video_id=args.video_id,
                             overwrite=args.overwrite)
        log.info(c.usage_summary())

    log.info("=" * 74)
    log.info("P3 SUMMARY — %s", art["video_id"])
    log.info("  frames          %d (%d scene cuts)", art["frame_count"],
             len(art["scene_cuts"]))
    log.info("  chyron frames   %d", art["chyron_frame_count"])
    log.info("  standing frames %d", art["standing_frame_count"])
    log.info("  chyron names seen:")
    seen = {}
    for f in art["frames"]:
        if f["chyron_name"]:
            seen.setdefault(f["chyron_name"], []).append(f["t"])
    for name, ts in sorted(seen.items(), key=lambda kv: kv[1][0]):
        log.info("    %-40s @ %s", preview(name, 40),
                 ", ".join(f"{t:.0f}s" for t in ts[:6]) + ("…" if len(ts) > 6 else ""))
    if not seen:
        log.info("    (none)")
    log.info("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
