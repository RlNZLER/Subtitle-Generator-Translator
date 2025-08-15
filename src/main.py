#!/usr/bin/env python3
import sys, shutil, subprocess
from pathlib import Path

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 to_audio.py /path/to/video")
        sys.exit(1)

    if shutil.which("ffmpeg") is None:
        sys.exit("Error: ffmpeg not found. Install it (Ubuntu: sudo apt-get install ffmpeg, macOS: brew install ffmpeg).")

    inp = Path(sys.argv[1])
    if not inp.exists():
        sys.exit(f"Input not found: {inp}")

    outp = inp.with_suffix(".flac")  # same folder, same name, .flac extension

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(inp),
        "-vn", "-sn",          # ignore video/subtitles
        "-ac", "1",            # mono
        "-ar", "16000",        # 16 kHz
        "-c:a", "flac",        # FLAC (lossless)
        "-sample_fmt", "s16",  # 16-bit samples (widely supported)
        str(outp)
    ]

    subprocess.run(cmd, check=True)
    print(f"Saved: {outp}")

if __name__ == "__main__":
    main()
