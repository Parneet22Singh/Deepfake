"""Command-line JSON interface."""

import argparse
import json
import sys

from .analyzer import AnalysisConfig, analyze_video


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deterministic classical video forensics")
    parser.add_argument("input", help="video path (or WAV for audio-only provenance)")
    parser.add_argument("--samples", type=int, default=48, help="adaptive sample budget")
    parser.add_argument("--max-frames", type=int, default=32, help="frames per diagnostic branch")
    parser.add_argument("--jpeg-quality", type=int, default=75, help="controlled JPEG quality (1-100)")
    parser.add_argument("--stability", action="store_true",
                        help="evaluate 10/20/30/40-second windows at alternate sample budgets")
    parser.add_argument("--stability-windows", type=int, nargs="+", metavar="SECONDS",
                        help="override stability windows (defaults to 10 20 30 40)")
    parser.add_argument("--stability-budgets", type=int, nargs="+", metavar="SAMPLES",
                        help="override stability budgets (defaults to 24 48 96)")
    parser.add_argument("--transcode-check", action="store_true",
                        help="run an ephemeral ffmpeg half-resolution transcode stability check")
    parser.add_argument("--production-snapshot-root", default="",
                        help="absolute protected production snapshot root for optional routers")
    parser.add_argument("--specialist-checkpoint", default="",
                        help="absolute three-class router checkpoint inside the snapshot root")
    parser.add_argument("--binary-checkpoint", default="",
                        help="absolute binary authenticity checkpoint inside the snapshot root")
    parser.add_argument("--no-hash", action="store_true", help="omit SHA-256 hashing")
    parser.add_argument("--indent", type=int, default=2, help="JSON indentation; use 0 for compact")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        stability_windows = tuple(args.stability_windows) if args.stability_windows else (10, 20, 30, 40)
        stability_budgets = tuple(args.stability_budgets) if args.stability_budgets else (24, 48, 96)
        config = AnalysisConfig(samples=max(1, args.samples), max_frames_per_branch=max(1, args.max_frames),
                                jpeg_quality=min(100, max(1, args.jpeg_quality)), include_hash=not args.no_hash,
                                stability=args.stability, stability_windows=stability_windows,
                                stability_budgets=stability_budgets,
                                transcode_check=args.transcode_check,
                                production_snapshot_root=args.production_snapshot_root,
                                specialist_checkpoint=args.specialist_checkpoint,
                                binary_checkpoint=args.binary_checkpoint)
        result = analyze_video(args.input, config)
        print(json.dumps(result.to_dict(), indent=None if args.indent == 0 else args.indent,
                          sort_keys=True, allow_nan=False))
        return 0
    except Exception as exc:
        # Keep CLI machine-readable even for malformed inputs or optional decoder failures.
        print(json.dumps({"schema_version": "1.0", "error": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    sys.exit(main())
