#!/usr/bin/env python3
"""
Export AdaFace OpenVINO embeddings to JSONL.

Usage (example):

    python export_to_jsonl.py \
        --db photos-test.db \
        --embeddings_dir /Users/odms/Documents/aditya_ws/adaface_ov_embeddings \
        --out final_embeddings_adaface_ov.jsonl \
        --ext .bin            # change if your files use .ov, .dat, etc.
"""

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm    # `pip install tqdm` if you don’t have it


def find_embedding_file(img_path: Path, embeddings_dir: Path, ext: str) -> Path | None:
    """
    Build the expected embedding‑file path for a given image.

    Strategy:
    - Take the image’s *stem* (filename without suffix) → e.g. `CHNG_000016_1002`
    - Append the desired extension (`.bin` by default)
    - Look for that file directly inside `embeddings_dir`
      *OR* inside a mirrored sub‑directory structure if it exists.

    Adapt this logic if your own naming or directory layout is different.
    """
    candidate = embeddings_dir / f"{img_path.stem}{ext}"
    if candidate.is_file():
        return candidate

    # If your embeddings mirror the original photo folder hierarchy:
    mirror = embeddings_dir / img_path.with_suffix(ext).relative_to(img_path.anchor)
    if mirror.is_file():
        return mirror

    return None


def export_embeddings_to_jsonl(db_path: str,
                               embeddings_dir: str | os.PathLike,
                               output_jsonl_path: str,
                               embedding_ext: str = ".bin") -> None:
    db_path = Path(db_path).expanduser().resolve()
    embeddings_dir = Path(embeddings_dir).expanduser().resolve()
    output_jsonl_path = Path(output_jsonl_path).expanduser().resolve()

    if not db_path.is_file():
        sys.exit(f"❌ Database not found: {db_path}")
    if not embeddings_dir.is_dir():
        sys.exit(f"❌ Embeddings directory not found: {embeddings_dir}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT path FROM photos")
    rows = cursor.fetchall()

    written, skipped, missing = 0, 0, 0

    with output_jsonl_path.open("w") as jsonl_file:
        for (img_path_str,) in tqdm(rows, unit="img"):
            img_path = Path(img_path_str)

            emb_file = find_embedding_file(img_path, embeddings_dir, embedding_ext)
            if emb_file is None:
                missing += 1
                continue  # Could log if you need a report of missing embeddings

            try:
                embedding = np.fromfile(emb_file, dtype=np.float32).tolist()
                record = {"path": str(img_path), "feat": embedding}
                jsonl_file.write(json.dumps(record) + "\n")
                written += 1
            except Exception as exc:  # noqa: BLE001
                skipped += 1
                print(f"⚠️  Error reading {emb_file}: {exc}", file=sys.stderr)

    conn.close()
    print(f"\n✅ Export finished → {output_jsonl_path}")
    print(f"   Records written : {written}")
    print(f"   Embeddings missing : {missing}")
    print(f"   Errors skipped : {skipped}")


def main():
    parser = argparse.ArgumentParser(description="Export AdaFace OV embeddings to JSONL")
    parser.add_argument("--db", required=True, help="SQLite database containing the photos table")
    parser.add_argument("--embeddings_dir", required=True, help="Folder that holds *.bin / *.ov files")
    parser.add_argument("--out", required=True, help="Destination .jsonl file")
    parser.add_argument("--ext", default=".bin", help="Embedding file extension (default: .bin)")
    args = parser.parse_args()

    export_embeddings_to_jsonl(
        db_path=args.db,
        embeddings_dir=args.embeddings_dir,
        output_jsonl_path=args.out,
        embedding_ext=args.ext,
    )


if __name__ == "__main__":
    main()
