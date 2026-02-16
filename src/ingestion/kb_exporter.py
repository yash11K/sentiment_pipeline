"""
KB Exporter - Exports enriched reviews to S3 for Bedrock Knowledge Base ingestion.

After enrichment, uploads a per-location JSON document to:
  s3://{KB_S3_BUCKET}/{KB_S3_PREFIX}{brand}/reviews_{LOCODE}.json

The KB auto-syncs from this bucket.
"""
import boto3
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional

sys.path.append(str(Path(__file__).parent.parent))

from storage.db import Database
from utils.logger import get_logger

logger = get_logger(__name__)


class KBExporter:
    def __init__(self, db: Database = None, bucket: str = None, prefix: str = None,
                 region: str = "us-east-1"):
        self.db = db or Database()
        self.bucket = bucket or os.getenv('KB_S3_BUCKET', 'google-structured-reviews')
        self.prefix = prefix or os.getenv('KB_S3_PREFIX', 'reviews/')
        self.s3_client = boto3.client('s3', region_name=region)
        logger.start(f"KB exporter initialized (bucket={self.bucket}, prefix={self.prefix})")

    def _build_s3_key(self, location_id: str, brand: str) -> str:
        """Build S3 key: {prefix}{brand}/reviews_{LOCODE}.json"""
        brand_folder = (brand or 'unknown').lower()
        return f"{self.prefix}{brand_folder}/reviews_{location_id.upper()}.json"

    def export_location(self, location_id: str, brand: str = None) -> Optional[Dict]:
        """
        Export enriched reviews for a location+brand to S3.

        Returns dict with s3_key, review_count or None on failure.
        """
        try:
            reviews = self.db.get_enriched_reviews_for_export(location_id, brand)
            if not reviews:
                logger.info(f"No enriched reviews to export for {location_id}/{brand}")
                return None

            s3_key = self._build_s3_key(location_id, brand)
            body = json.dumps(reviews, indent=2, default=str)

            self.s3_client.put_object(
                Bucket=self.bucket,
                Key=s3_key,
                Body=body.encode('utf-8'),
                ContentType='application/json'
            )

            logger.s3(f"Exported {len(reviews)} reviews to s3://{self.bucket}/{s3_key}")
            return {"s3_key": s3_key, "review_count": len(reviews)}

        except Exception as e:
            logger.error(f"KB export failed for {location_id}/{brand}: {e}")
            return None
