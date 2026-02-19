"""
Rename inconsistent S3 files to match the standard naming convention.

Renames files like:
  google/hertz/DEN_hertz_run.json
To:
  google/hertz/DEN_2026-02-12_39.8561_-104.6737.json

The brand context lives in the subfolder (google/hertz/), so the filename
should follow: LOCODE_YYYY-MM-DD_LAT_LON.json
"""
import boto3
import os
import sys
from dotenv import load_dotenv

load_dotenv()

BUCKET = os.getenv("REVIEWS_S3_BUCKET", "google-reviews-extract")
REGION = os.getenv("AWS_REGION", "us-east-1")
DRY_RUN = "--dry-run" in sys.argv

# Scrape date from S3 upload timestamps (Feb 12, 2026)
SCRAPE_DATE = "2026-02-12"

# Airport rental location coordinates
LOCATION_COORDS = {
    "DEN": (39.8561, -104.6737),
    "JFK": (40.6413, -73.7781),
    "LAX": (33.9425, -118.4081),
    "MCO": (28.4294, -81.3089),
    "PHX": (33.4373, -112.0078),
}

# Files to rename: old_key -> new_key
RENAMES = {}
for loc, (lat, lon) in LOCATION_COORDS.items():
    old_key = f"google/hertz/{loc}_hertz_run.json"
    new_key = f"google/hertz/{loc}_{SCRAPE_DATE}_{lat}_{lon}.json"
    RENAMES[old_key] = new_key


def main():
    s3 = boto3.client("s3", region_name=REGION)

    print(f"Bucket: {BUCKET}")
    print(f"Dry run: {DRY_RUN}\n")

    # Verify files exist first
    print("Checking existing files...")
    existing = set()
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix="google/hertz/"):
        for obj in page.get("Contents", []):
            existing.add(obj["Key"])

    print(f"Found {len(existing)} files in google/hertz/\n")

    success = 0
    skipped = 0
    errors = 0

    for old_key, new_key in RENAMES.items():
        if old_key not in existing:
            print(f"  SKIP  {old_key} (not found)")
            skipped += 1
            continue

        if new_key in existing:
            print(f"  SKIP  {old_key} -> {new_key} (target already exists)")
            skipped += 1
            continue

        if DRY_RUN:
            print(f"  WOULD RENAME  {old_key}")
            print(f"            ->  {new_key}")
            success += 1
            continue

        try:
            # Copy to new key
            s3.copy_object(
                Bucket=BUCKET,
                CopySource={"Bucket": BUCKET, "Key": old_key},
                Key=new_key,
            )
            # Delete old key
            s3.delete_object(Bucket=BUCKET, Key=old_key)
            print(f"  OK  {old_key}")
            print(f"  ->  {new_key}")
            success += 1
        except Exception as e:
            print(f"  ERROR  {old_key}: {e}")
            errors += 1

    print(f"\nDone: {success} renamed, {skipped} skipped, {errors} errors")


if __name__ == "__main__":
    main()
