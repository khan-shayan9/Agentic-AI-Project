"""Back up and restore the local caches between sessions (Google Drive on Colab, the output folder on Kaggle).

The SQLite reply cache stays on local disk: SQLite's locking is not safe on network or cloud
filesystems and can corrupt the database (https://sqlite.org/useovernet.html), and the cache
commits after every model call. Instead it is copied out after each run and copied back at the
start of a session.

  python -m experiments.state save --to DIR
  python -m experiments.state restore --from DIR

save    DIR/llm_cache.sqlite (sqlite3 backup API), DIR/pubmed_cache.zip, DIR/final_config.json
restore the same files back: the reply cache only if there is none locally or the saved one is
        larger, the PubMed cache always (unzipped over the local one), final_config.json only
        if it is missing locally.
Every file is first written under a temporary name and then renamed, so a half-written copy
never replaces a good one.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

CACHE_DIR = Path("data/cache")
LLM_CACHE = "llm_cache.sqlite"
PUBMED_DIR = "pubmed"
PUBMED_ZIP = "pubmed_cache.zip"
FINAL_NAME = "final_config.json"


def _copy_then_rename(source: Path, target: Path) -> None:
    tmp = target.with_name(target.name + ".tmp")
    shutil.copyfile(source, tmp)
    os.replace(tmp, target)


def _snapshot_sqlite(db: Path, target: Path) -> None:
    """A consistent copy of a live SQLite file: backup API into a local temp file, then copy and rename."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        snapshot = Path(tmp_dir) / db.name
        source, copy = sqlite3.connect(str(db)), sqlite3.connect(str(snapshot))
        try:
            source.backup(copy)
        finally:
            source.close()
            copy.close()
        _copy_then_rename(snapshot, target)


def _zip_folder(folder: Path, target: Path) -> int:
    tmp = target.with_name(target.name + ".tmp")
    files = sorted(p for p in folder.rglob("*") if p.is_file())
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for path in files:
            z.write(path, path.relative_to(folder).as_posix())
    os.replace(tmp, target)
    return len(files)


def save(to: Path, cache_dir: Path = CACHE_DIR, final_config: Optional[Path] = None) -> list[str]:
    """Copy the reply cache, the PubMed cache and the chosen trigger into `to`. Returns what was saved."""
    to, cache_dir = Path(to), Path(cache_dir)
    to.mkdir(parents=True, exist_ok=True)
    saved = []
    db = cache_dir / LLM_CACHE
    if db.exists():
        _snapshot_sqlite(db, to / LLM_CACHE)
        saved.append(f"{LLM_CACHE} ({(to / LLM_CACHE).stat().st_size / 1e6:.1f} MB)")
    pubmed = cache_dir / PUBMED_DIR
    if pubmed.is_dir():
        saved.append(f"{PUBMED_ZIP} ({_zip_folder(pubmed, to / PUBMED_ZIP)} files)")
    if final_config is not None and Path(final_config).exists():
        _copy_then_rename(Path(final_config), to / FINAL_NAME)
        saved.append(FINAL_NAME)
    return saved


def restore(source: Path, cache_dir: Path = CACHE_DIR, final_config: Optional[Path] = None) -> list[str]:
    """Copy a saved state back; never replaces a larger local reply cache or an existing final_config.json."""
    source, cache_dir = Path(source), Path(cache_dir)
    restored = []
    saved_db, local_db = source / LLM_CACHE, cache_dir / LLM_CACHE
    if saved_db.exists():
        if not local_db.exists() or saved_db.stat().st_size > local_db.stat().st_size:
            cache_dir.mkdir(parents=True, exist_ok=True)
            _copy_then_rename(saved_db, local_db)
            restored.append(f"{LLM_CACHE} ({local_db.stat().st_size / 1e6:.1f} MB)")
        else:
            print(f"Kept the local {LLM_CACHE}: it is not smaller than the saved one.")
    saved_zip = source / PUBMED_ZIP
    if saved_zip.exists():
        with zipfile.ZipFile(saved_zip) as z:
            z.extractall(cache_dir / PUBMED_DIR)
            restored.append(f"{PUBMED_ZIP} ({len(z.namelist())} files)")
    saved_final = source / FINAL_NAME
    if final_config is not None and saved_final.exists() and not Path(final_config).exists():
        Path(final_config).parent.mkdir(parents=True, exist_ok=True)
        _copy_then_rename(saved_final, Path(final_config))
        restored.append(FINAL_NAME)
    return restored


def main(argv=None) -> int:
    from .run import FINAL_CONFIG

    p = argparse.ArgumentParser(prog="experiments.state", description="Back up or restore the local caches.")
    sub = p.add_subparsers(dest="action", required=True)
    sub.add_parser("save").add_argument("--to", required=True, help="backup folder")
    sub.add_parser("restore").add_argument("--from", dest="source", required=True, help="backup folder")
    args = p.parse_args(argv)
    if args.action == "save":
        done = save(Path(args.to), final_config=FINAL_CONFIG)
        print(f"Saved to {args.to}: {', '.join(done) or 'nothing yet'}")
    else:
        done = restore(Path(args.source), final_config=FINAL_CONFIG)
        print(f"Restored from {args.source}: {', '.join(done) or 'nothing'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
