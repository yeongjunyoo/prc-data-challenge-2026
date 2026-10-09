# SPDX-License-Identifier: GPL-3.0-only
"""Read-only download of the final-phase files from prc-2026-datasets (get calls only)."""
import sys, hashlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.s3io import client, DATA_BUCKET  # noqa: E402
dest = ROOT / "data" / "final_phase"
dest.mkdir(parents=True, exist_ok=True)
s3 = client()
for key in ("final_ranking.parquet", "final_submitting.parquet"):
    head = s3.head_object(Bucket=DATA_BUCKET, Key=key)
    p = dest / key
    s3.download_file(DATA_BUCKET, key, str(p))
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    print(key, "size", p.stat().st_size, "remote", head["ContentLength"], "etag", head.get("ETag"), "lastmod", head["LastModified"].isoformat(), "sha256", h)
