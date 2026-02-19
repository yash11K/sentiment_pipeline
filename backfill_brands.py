"""
One-time script to backfill brand info on reviews that were ingested
before brand support was added. Re-processes branded S3 files with
enrich=False so only the brand column gets updated on existing rows.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from dotenv import load_dotenv
load_dotenv()

from ingestion.pipeline import IngestionPipeline
import os

bucket = os.getenv("REVIEWS_S3_BUCKET", "google-reviews-extract")
prefix = os.getenv("REVIEWS_S3_PREFIX", "")

pipeline = IngestionPipeline(bucket_name=bucket, prefix=prefix)

# These are the branded files from your ingestion history
branded_files = [
    "google/avis/ATL_2026-02-12_33.6407_-84.4277.json",
    "google/avis/JFK_2026-02-12_40.6413111_-73.7781391.json",
    "google/avis/LAX_2026-02-12_33.9499276_-118.3760274.json",
    "google/avis/MCO_2026-02-12_28.4312_-81.3081.json",
    "google/avis/DEN_2026-02-12_39.8561_-104.6737.json",
    "google/avis/MIA_2026-02-12_25.7959_-80.287.json",
    "google/avis/PHX_2026-02-12_33.4373_-112.0078.json",
    "google/hertz/ATL_2026-02-12_33.6407_-84.4277.json",
    "google/hertz/JFK_hertz_run.json",
    "google/hertz/LAX_hertz_run.json",
    "google/hertz/MCO_hertz_run.json",
    "google/hertz/DEN_hertz_run.json",
    "google/hertz/PHX_hertz_run.json",
]

print(f"Backfilling brand info for {len(branded_files)} files (enrich=False)...\n")

for s3_key in branded_files:
    print(f"Processing: {s3_key}")
    try:
        result = pipeline.process_file(s3_key, enrich=False)
        print(f"  -> status={result['status']}, reviews={result['reviews_count']}\n")
    except Exception as e:
        print(f"  -> ERROR: {e}\n")

print("Done! Brand backfill complete.")
