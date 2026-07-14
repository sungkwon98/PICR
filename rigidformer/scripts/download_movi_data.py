from __future__ import annotations

import argparse
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path


ZENODO_RECORD = "https://zenodo.org/records/15800434/files"

DATASET_FILES = {
    "movi-spheres": "MOVi-spheres.tar.gz",
    "movi-a": "MOVi-A.tar.gz",
    "movi-b": "MOVi-B.tar.gz",
    "movi-c": "MOVi-C.tar.gz",
}


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents = True, exist_ok = True)

    if dest.is_file() and dest.stat().st_size > 0:
        print(f"exists: {dest}")
        return

    tmp_dest = dest.with_suffix(dest.suffix + ".part")
    if tmp_dest.exists():
        tmp_dest.unlink()

    print(f"downloading: {url}")
    with urllib.request.urlopen(url) as response, tmp_dest.open("wb") as f:
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0

        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break

            f.write(chunk)
            downloaded += len(chunk)

            if total:
                pct = 100. * downloaded / total
                print(f"\r  {downloaded / 1024 ** 3:.2f} / {total / 1024 ** 3:.2f} GiB ({pct:.1f}%)", end = "")
            else:
                print(f"\r  {downloaded / 1024 ** 3:.2f} GiB", end = "")

    print()
    tmp_dest.rename(dest)


def _safe_extract(archive: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents = True, exist_ok = True)
    resolved_out = out_dir.resolve()

    print(f"extracting: {archive}")
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            target = (out_dir / member.name).resolve()
            if not str(target).startswith(str(resolved_out)):
                raise RuntimeError(f"Unsafe tar path: {member.name}")
        tar.extractall(out_dir)


def _download_objects(objects_dir: Path, force: bool = False) -> None:
    marker = objects_dir / "cube" / "collision_geometry.obj"
    if marker.is_file() and not force:
        print(f"exists: {objects_dir}")
        return

    if objects_dir.exists() and force:
        shutil.rmtree(objects_dir)

    objects_dir.parent.mkdir(parents = True, exist_ok = True)

    with tempfile.TemporaryDirectory() as tmp:
        repo_dir = Path(tmp) / "hopnet"
        commands = [
            ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", "https://github.com/AmauryWEI/hopnet.git", str(repo_dir)],
            ["git", "-C", str(repo_dir), "sparse-checkout", "set", "data/objects"],
        ]

        for command in commands:
            subprocess.run(command, check = True)

        source = repo_dir / "data" / "objects"
        shutil.copytree(source, objects_dir, dirs_exist_ok = True)

    print(f"objects ready: {objects_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description = "Download MOVi rigid-body datasets and KuBasic collision meshes.")
    parser.add_argument(
        "--datasets",
        nargs = "+",
        choices = sorted(DATASET_FILES),
        default = ["movi-spheres"],
        help = "Datasets to download from Zenodo.",
    )
    parser.add_argument("--out-dir", type = Path, default = Path("data/movi"), help = "Directory for extracted datasets.")
    parser.add_argument("--archive-dir", type = Path, default = Path("data/archives"), help = "Directory for downloaded tar.gz files.")
    parser.add_argument("--objects-dir", type = Path, default = Path("data/movi_objects/objects"), help = "Directory for KuBasic object meshes.")
    parser.add_argument("--skip-extract", action = "store_true", help = "Only download archives; do not extract.")
    parser.add_argument("--force-objects", action = "store_true", help = "Re-download object meshes.")
    args = parser.parse_args()

    _download_objects(args.objects_dir, force = args.force_objects)

    for dataset in args.datasets:
        filename = DATASET_FILES[dataset]
        archive = args.archive_dir / filename
        url = f"{ZENODO_RECORD}/{filename}?download=1"

        _download(url, archive)

        if not args.skip_extract:
            _safe_extract(archive, args.out_dir)

    print("done")


if __name__ == "__main__":
    main()
