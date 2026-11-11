"""Append SHA-256 hashes of files to results/futurology/FREEZE.txt with a UTC timestamp and a note
(PREREG_futurology.md §6). (2026-11-03)

    python3 freeze.py "plan frozen" PREREG_futurology.md src/futurology_collect.py ..."""
import datetime as dt
import hashlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get("FREEZE_FILE") or os.path.join(ROOT, "results", "futurology", "FREEZE.txt")  # sites: results/sites/FREEZE.txt


def main(note, files):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    lines = [f"## {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}  {note}"]
    for f in files:
        p = os.path.join(ROOT, f)
        lines.append(f"{hashlib.sha256(open(p, 'rb').read()).hexdigest()}  {f}  ({os.path.getsize(p)} bytes)")
    with open(OUT, "a") as fh:
        fh.write("\n".join(lines) + "\n\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
