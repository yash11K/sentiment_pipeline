"""
KB Exporter - Exports enriched reviews to S3 for Bedrock Knowledge Base ingestion.

After enrichment, uploads a per-location Markdown document to:
  s3://{KB_S3_BUCKET}/{KB_S3_PREFIX}{brand}/reviews_{LOCODE}.md

The KB auto-syncs from this bucket. Markdown format enables Bedrock Knowledge Base
to chunk reviews at section boundaries for better retrieval.
"""
import boto3
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

sys.path.append(str(Path(__file__).parent.parent))

from storage.db import Database
from utils.logger import get_logger

logger = get_logger(__name__)


class KBExporter:
    def __init__(self, db: Database = None, bucket: str = None, prefix: str = None,
                 region: str = None):
        from config import config
        self.db = db or Database()
        self.bucket = bucket or config.KB_S3_BUCKET
        self.prefix = prefix or config.KB_S3_PREFIX
        self.s3_client = boto3.client('s3', region_name=region or config.AWS_REGION)
        logger.start(f"KB exporter initialized (bucket={self.bucket}, prefix={self.prefix})")

    def _build_s3_key(self, location_id: str, brand: str) -> str:
        """Build S3 key: {prefix}{brand}/reviews_{LOCODE}.md"""
        brand_folder = brand.lower()
        return f"{self.prefix}{brand_folder}/reviews_{location_id.upper()}.md"

    def _format_review_section(self, review: Dict) -> str:
        """Format a single review as a Markdown section."""
        topics = ", ".join(review.get("topics", []))
        text = review.get("review_text", "").replace("\n", "\n> ")
        return (
            f"## Review: {review['review_id']}\n\n"
            f"- Rating: {review.get('rating', 'N/A')}\n"
            f"- Reviewer: {review.get('reviewer_name', 'Anonymous')}\n"
            f"- Date: {review.get('review_date', 'N/A')}\n"
            f"- Sentiment: {review.get('sentiment', 'N/A')} ({review.get('sentiment_score', 'N/A')})\n"
            f"- Topics: {topics or 'N/A'}\n"
            f"- Urgency: {review.get('urgency_level', 'N/A')}\n\n"
            f"> {text}\n"
        )

    def _format_markdown(self, reviews: List[Dict], location_id: str, brand: str) -> str:
        """Format all reviews into a complete Markdown document."""
        header = (
            f"# Reviews: {location_id.upper()} — {brand}\n\n"
            f"Generated: {datetime.utcnow().isoformat()}Z\n\n"
            f"---\n\n"
        )
        sections = "\n\n---\n\n".join(
            self._format_review_section(r) for r in reviews
        )
        return header + sections

    def export_location(self, location_id: str, brand: str = None) -> Optional[Dict]:
        """
        Export enriched reviews for a location+brand to S3 as Markdown.

        Returns dict with s3_key, review_count or None on failure.
        """
        try:
            reviews = self.db.get_enriched_reviews_for_export(location_id, brand)
            if not reviews:
                logger.info(f"No enriched reviews to export for {location_id}/{brand}")
                return None

            s3_key = self._build_s3_key(location_id, brand)
            body = self._format_markdown(reviews, location_id, brand)

            self.s3_client.put_object(
                Bucket=self.bucket,
                Key=s3_key,
                Body=body.encode('utf-8'),
                ContentType='text/markdown',
            )

            logger.s3(f"Exported {len(reviews)} reviews to s3://{self.bucket}/{s3_key}")
            return {"s3_key": s3_key, "review_count": len(reviews)}

        except Exception as e:
            logger.error(f"KB export failed for {location_id}/{brand}: {e}")
            return None
