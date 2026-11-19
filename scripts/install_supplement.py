"""Install the author's numerical evidence archive alongside a source checkout."""
import argparse
from hashlib import file_digest
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
from tempfile import TemporaryDirectory
from zipfile import BadZipFile, ZipFile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def install(archive, root):
    expected = json.loads((ROOT / "data/supplement.json").read_bytes())
    if archive.stat().st_size != expected["bytes"] or digest(archive) != expected["sha256"]:
        raise ValueError(f"Expected {expected['filename']} with SHA-256 {expected['sha256']}")
    root.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive) as source, TemporaryDirectory(dir=root) as temp:
        items = source.infolist()
        if len(items) != expected["members"]:
            raise ValueError("Unexpected number of supplement files.")
        missing = []
        for item in items:
            name = PurePosixPath(item.filename)
            if name.is_absolute() or ".." in name.parts or name.parts[0] not in {"examples", "rejeu_r3"}:
                raise ValueError(f"Unexpected supplement path: {name}")
            path = root.joinpath(*name.parts)
            if not path.resolve().is_relative_to(root.resolve()):
                raise ValueError(f"Supplement path points outside the checkout: {name}")
            if path.exists():
                with source.open(item) as stream:
                    if not path.is_file() or digest(path) != file_digest(stream, "sha256").hexdigest():
                        raise ValueError(f"Existing file differs from the archive: {path}")
                continue
            staged = Path(temp).joinpath(*name.parts)
            staged.parent.mkdir(parents=True, exist_ok=True)
            with source.open(item) as stream, staged.open("wb") as destination:
                shutil.copyfileobj(stream, destination)
            missing.append((staged, path))
        for staged, target in missing:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and digest(target) != digest(staged):
                raise ValueError(f"File changed during installation: {target}")
            staged.replace(target)
    print(f"Verified {len(items)} files, installed {len(missing)} in {root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        install(args.archive, args.root)
    except (BadZipFile, OSError, ValueError) as error:
        sys.exit(f"Supplement installation failed: {error}")
