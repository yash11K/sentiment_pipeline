"""
S3 Source - Handles S3 bucket operations for review file ingestion

Supports bucket structures:
1. Legacy: s3://{bucket}/{prefix}/LOCODE_SOURCE_DD_MM_YYYY.json
2. Coordinate: s3://{bucket}/{source}/LOCODE_YYYY-MM-DD_LAT_LON.json
3. Brand: s3://{bucket}/{source}/{brand}/LOCODE_YYYY-MM-DD_LAT_LON.json

Examples:
  - Legacy: reviews/LAX_google_10_01_2026.json
  - Coordinate: google/MCO_2026-01-15_28.4294_-81.3089.json
  - Brand: google/avis/ATL_2026-02-12_33.6407_-84.4277.json
  - Brand (competitor): google/hertz/ATL_hertz_2026-02-12_33.6407_-84.4277.json
"""
import boto3
import json
import re
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from storage.db import Database
from utils.logger import get_logger

logger = get_logger(__name__)

# Our brands - everything else is a competitor
OUR_BRANDS = {'avis', 'budget', 'payless', 'apex', 'maggiore'}


class S3ReviewSource:
    # Pattern 1: LOCODE_SOURCE_DD_MM_YYYY.json (legacy)
    FILENAME_PATTERN_LEGACY = re.compile(
        r'^([A-Z]{3})_([a-z_]+)_(\d{2})_(\d{2})_(\d{4})\.json$',
        re.IGNORECASE
    )

    # Pattern 2: LOCODE_YYYY-MM-DD_LAT_LON.json (coordinate format)
    FILENAME_PATTERN_COORDS = re.compile(
        r'^([A-Z]{3})_(\d{4}-\d{2}-\d{2})_(-?\d+\.?\d*)_(-?\d+\.?\d*)\.json$',
        re.IGNORECASE
    )

    # Pattern 3: LOCODE_BRAND_YYYY-MM-DD_LAT_LON.json (brand embedded in filename)
    FILENAME_PATTERN_BRAND_COORDS = re.compile(
        r'^([A-Z]{3})_([a-z]+)_(\d{4}-\d{2}-\d{2})_(-?\d+\.?\d*)_(-?\d+\.?\d*)\.json$',
        re.IGNORECASE
    )

    # Pattern 4: Any JSON file - extract location from filename
    FILENAME_PATTERN_FLEXIBLE = re.compile(
        r'^([A-Z]{3}).*\.json$',
        re.IGNORECASE
    )

    def __init__(self, bucket_name: str = None, prefix: str = None, region: str = None):
        from config import config
        self.bucket_name = bucket_name or config.REVIEWS_S3_BUCKET
        self.prefix = prefix if prefix is not None else config.REVIEWS_S3_PREFIX
        self.region = region or config.AWS_REGION
        self.s3_client = boto3.client('s3', region_name=self.region)
        self.db = Database()
        logger.s3(f"S3 source initialized (bucket={self.bucket_name}, prefix={self.prefix})")

    @staticmethod
    def classify_brand(brand_name: str) -> bool:
        """Returns True if the brand is a competitor (not one of ours)."""
        if not brand_name:
            return False
        return brand_name.lower() not in OUR_BRANDS

    def _extract_path_segments(self, s3_key: str) -> Dict:
        """
        Extract source and brand from S3 path segments.

        Path structures:
          source/brand/filename.json  -> source=google, brand=avis
          source/filename.json        -> source=google, brand=None
          filename.json               -> source=None, brand=None
        """
        path_parts = Path(s3_key).parts
        # Exclude the filename
        dirs = [p for p in path_parts[:-1] if p and p not in ('', '.')]

        source = dirs[0].lower() if len(dirs) >= 1 else None
        brand = dirs[1].lower() if len(dirs) >= 2 else None

        return {"source": source, "brand": brand}

    def parse_filename(self, s3_key: str) -> Optional[Dict]:
        """
        Parse S3 key to extract metadata including brand and competitor flag.

        Returns dict with location_id, source, brand, is_competitor,
        scrape_date, latitude, longitude or None if invalid.
        """
        path_parts = Path(s3_key).parts
        basename = Path(s3_key).name
        segments = self._extract_path_segments(s3_key)

        # Try brand-in-filename pattern: LOCODE_BRAND_YYYY-MM-DD_LAT_LON.json
        match = self.FILENAME_PATTERN_BRAND_COORDS.match(basename)
        if match:
            location_id, brand_in_file, date_str, lat, lon = match.groups()
            try:
                scrape_date = datetime.strptime(date_str, "%Y-%m-%d")
                # Brand from path takes precedence, fallback to filename
                brand = segments["brand"] or brand_in_file.lower()
                source = segments["source"] or "google"
                return {
                    "location_id": location_id.upper(),
                    "source": source,
                    "brand": brand,
                    "is_competitor": self.classify_brand(brand),
                    "scrape_date": scrape_date.isoformat(),
                    "scrape_date_display": scrape_date.strftime("%Y-%m-%d"),
                    "latitude": float(lat),
                    "longitude": float(lon),
                }
            except ValueError:
                pass

        # Try coordinate pattern: LOCODE_YYYY-MM-DD_LAT_LON.json
        match = self.FILENAME_PATTERN_COORDS.match(basename)
        if match:
            location_id, date_str, lat, lon = match.groups()
            try:
                scrape_date = datetime.strptime(date_str, "%Y-%m-%d")
                source = segments["source"] or "google"
                brand = segments["brand"]
                return {
                    "location_id": location_id.upper(),
                    "source": source,
                    "brand": brand,
                    "is_competitor": self.classify_brand(brand),
                    "scrape_date": scrape_date.isoformat(),
                    "scrape_date_display": scrape_date.strftime("%Y-%m-%d"),
                    "latitude": float(lat),
                    "longitude": float(lon),
                }
            except ValueError:
                pass

        # Try legacy pattern: LOCODE_SOURCE_DD_MM_YYYY.json
        match = self.FILENAME_PATTERN_LEGACY.match(basename)
        if match:
            location_id, source, day, month, year = match.groups()
            try:
                scrape_date = datetime(int(year), int(month), int(day))
                brand = segments["brand"]
                return {
                    "location_id": location_id.upper(),
                    "source": source.lower(),
                    "brand": brand,
                    "is_competitor": self.classify_brand(brand),
                    "scrape_date": scrape_date.isoformat(),
                    "scrape_date_display": scrape_date.strftime("%Y-%m-%d"),
                    "latitude": None,
                    "longitude": None,
                }
            except ValueError:
                pass

        # Flexible fallback: derive source from path
        source_from_path = segments["source"]
        if source_from_path:
            flexible_match = self.FILENAME_PATTERN_FLEXIBLE.match(basename)
            if flexible_match:
                location_id = flexible_match.group(1).upper()
                scrape_date = datetime.now()

                date_patterns = [
                    r'(\d{4})_(\d{2})_(\d{2})',
                    r'(\d{2})_(\d{2})_(\d{4})',
                    r'(\d{4})-(\d{2})-(\d{2})',
                ]
                for pattern in date_patterns:
                    date_match = re.search(pattern, basename)
                    if date_match:
                        groups = date_match.groups()
                        try:
                            if len(groups[0]) == 4:
                                scrape_date = datetime(int(groups[0]), int(groups[1]), int(groups[2]))
                            else:
                                scrape_date = datetime(int(groups[2]), int(groups[1]), int(groups[0]))
                            break
                        except ValueError:
                            continue

                brand = segments["brand"]
                return {
                    "location_id": location_id,
                    "source": source_from_path,
                    "brand": brand,
                    "is_competitor": self.classify_brand(brand),
                    "scrape_date": scrape_date.isoformat(),
                    "scrape_date_display": scrape_date.strftime("%Y-%m-%d"),
                    "latitude": None,
                    "longitude": None,
                }

        return None

    def list_s3_files(self, prefixes: List[str] = None) -> List[Dict]:
        """
        List all JSON files in the S3 bucket.

        Args:
            prefixes: Optional list of prefixes to scan (e.g., ['google/', 'google/avis/'])
                     If None, uses self.prefix
        """
        files = []
        paginator = self.s3_client.get_paginator('list_objects_v2')

        scan_prefixes = prefixes or [self.prefix]

        try:
            for prefix in scan_prefixes:
                for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
                    for obj in page.get('Contents', []):
                        key = obj['Key']
                        if key.endswith('.json'):
                            metadata = self.parse_filename(key)
                            files.append({
                                "s3_key": key,
                                "size_bytes": obj['Size'],
                                "last_modified": obj['LastModified'].isoformat(),
                                "metadata": metadata
                            })
        except Exception as e:
            logger.error(f"Error listing S3 files: {e}")

        return files

    def get_pending_files(self, prefixes: List[str] = None) -> List[Dict]:
        """
        Get list of S3 files that haven't been processed yet.
        Compares S3 bucket contents against ingestion_files table.

        Args:
            prefixes: Optional list of prefixes to scan
        """
        s3_files = self.list_s3_files(prefixes)
        processed_keys = set(self.db.get_processed_s3_keys())

        pending = []
        for file_info in s3_files:
            s3_key = file_info['s3_key']
            metadata = file_info.get('metadata')

            if s3_key not in processed_keys and metadata:
                pending.append({
                    "s3_key": s3_key,
                    "location_id": metadata['location_id'],
                    "source": metadata['source'],
                    "brand": metadata.get('brand'),
                    "is_competitor": metadata.get('is_competitor', False),
                    "scrape_date": metadata['scrape_date_display'],
                    "size_bytes": file_info['size_bytes'],
                    "last_modified": file_info['last_modified']
                })

        return pending

    def download_file(self, s3_key: str) -> Dict:
        """Download and parse a JSON file from S3"""
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            content = response['Body'].read()

            for encoding in ['utf-8', 'utf-16-le', 'utf-16', 'utf-8-sig', 'latin-1']:
                try:
                    text = content.decode(encoding)
                    if text.startswith('\ufeff'):
                        text = text[1:]
                    return json.loads(text)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue

            raise ValueError("Could not decode file with any supported encoding")

        except Exception as e:
            raise Exception(f"Error downloading {s3_key}: {e}")

    def validate_file_structure(self, data: Dict) -> Tuple[bool, str]:
        """
        Validate that the JSON file has the expected structure.
        Returns (is_valid, error_message)
        """
        if not isinstance(data, dict):
            return False, "Root element must be an object"

        reviews = data.get('reviews') or data.get('data', {}).get('reviews')

        if not reviews:
            return False, "No 'reviews' array found in file"

        if not isinstance(reviews, list):
            return False, "'reviews' must be an array"

        if len(reviews) == 0:
            return False, "Reviews array is empty"

        sample = reviews[0]
        if 'review_id' not in sample:
            return False, "Reviews missing required field: review_id"

        return True, ""
