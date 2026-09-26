"""Verify the small benchmark release independently of model and simulator dependencies."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1] / "data"
manifest = json.loads((root / "SHA256SUMS.json").read_text())
for relative, expected in manifest.items():
    actual = hashlib.sha256((root / relative).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"Checksum mismatch: {relative}")
print(f"Verified {len(manifest)} benchmark files")
