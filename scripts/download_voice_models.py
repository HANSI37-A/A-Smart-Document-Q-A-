"""Download the local voice models once, so the first spoken question is not slow.

    python scripts/download_voice_models.py                # Whisper 'base' + Piper en_US-lessac-medium
    python scripts/download_voice_models.py --whisper small --piper en_GB-alan-medium

Models go to DATA_DIR/models/piper (Piper) and the Hugging Face cache (Whisper).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.config import load_settings  # noqa: E402


def main() -> None:
    settings = load_settings()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--whisper", default=settings.whisper_model,
                        help="faster-whisper model size (tiny, base, small, ...)")
    parser.add_argument("--piper", default="en_US-lessac-medium", help="Piper voice name")
    parser.add_argument("--skip-whisper", action="store_true")
    parser.add_argument("--skip-piper", action="store_true")
    args = parser.parse_args()

    if not args.skip_piper:
        from piper.download_voices import download_voice

        target = settings.piper_voice.parent
        target.mkdir(parents=True, exist_ok=True)
        print(f"Downloading Piper voice {args.piper} to {target} ...")
        download_voice(args.piper, target)
        print(f"Done. Set PIPER_VOICE={target / (args.piper + '.onnx')} if you changed the voice.")

    if not args.skip_whisper:
        from faster_whisper.utils import download_model

        print(f"Downloading Whisper model {args.whisper!r} ...")
        print(f"Done: {download_model(args.whisper)}")


if __name__ == "__main__":
    main()
