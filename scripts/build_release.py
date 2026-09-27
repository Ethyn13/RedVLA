"""Build a clean source bundle including benchmark data; red-libero is installed separately."""
from pathlib import Path
import tarfile

root = Path(__file__).resolve().parents[1]
out = root / "dist"
out.mkdir(exist_ok=True)
destination = out / "redvla-0.1.0-release.tar.gz"
include = ["src", "configs", "docs", "scripts", "tests", "licenses", "data", "examples", "requirements",
           ".github", "README.md", "LICENSE", "CHANGELOG.md", "CITATION.cff", "environment.yml", "pyproject.toml", "MANIFEST.in", ".gitignore"]
with tarfile.open(destination, "w:gz") as archive:
    for item in include:
        source = root / item
        files = sorted(source.rglob("*")) if source.is_dir() else [source]
        for path in files:
            relative = path.relative_to(root)
            if relative.parts[:2] == ("configs", "local") or path.name.endswith((".local.yml", ".local.yaml", ".local.json")):
                continue
            if not path.is_file() or any(p in {"__pycache__", "dist", "build"} or p.endswith(".egg-info") for p in path.parts):
                continue
            if path.suffix in {".pyc", ".log", ".mp4"}:
                continue
            archive.add(path, arcname=Path("redvla-0.1.0") / path.relative_to(root), recursive=False)
print(destination)
