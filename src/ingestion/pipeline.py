"""
Ingestion Pipeline Service - Orchestrates S3-based review ingestion
"""
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

sys.path.append(str(Path(__file__).parent.parent))

from storage.db import Database
from ingestion.parser import ReviewParser
from ingestion.enricher import ReviewEnricher, EnrichmentError
from ingestion.s3_source import S3ReviewSource
from ingestion.kb_exporter import KBExporter
from utils.logger import get_logger

logger = get_logger(__name__)


class IngestionPipeline:
    def __init__(self, bucket_name: str = None, prefix: str = None,
                 region: str = None, batch_size: int = 20):
        from config import config
        self.db = Database()
        self.parser = ReviewParser(self.db)
        self.enricher = ReviewEnricher(self.db, batch_size=batch_size)
        self.s3_source = S3ReviewSource(
            bucket_name or config.REVIEWS_S3_BUCKET,
            prefix if prefix is not None else config.REVIEWS_S3_PREFIX,
            region or config.AWS_REGION
        )
        self.kb_exporter = KBExporter(self.db, region=region or config.AWS_REGION)
        self.batch_size = batch_size
        logger.start(f"Ingestion pipeline initialized (bucket={self.s3_source.bucket_name}, batch_size={batch_size})")

    def get_pending_files(self) -> List[Dict]:
        """Get list of S3 files pending ingestion"""
        return self.s3_source.get_pending_files()

    def get_ingestion_history(self, limit: int = 50) -> List[Dict]:
        """Get recent ingestion history"""
        return self.db.get_ingestion_history(limit)

    def process_file(self, s3_key: str) -> Dict:
        """
        Process a single S3 file through the ingestion pipeline.
        Enrichment is always performed — no option to skip.

        Returns dict with status, reviews_count, enriched_count, error
        """
        logger.s3(f"Processing file: {s3_key}")
        result = {
            "s3_key": s3_key,
            "status": "pending",
            "reviews_count": 0,
            "enriched_count": 0,
            "error": None
        }

        # Parse filename metadata
        metadata = self.s3_source.parse_filename(s3_key)
        if not metadata:
            result["status"] = "failed"
            result["error"] = f"Invalid filename format: {s3_key}"
            logger.error(f"Invalid filename format: {s3_key}")
            self.db.upsert_ingestion_file(
                s3_key=s3_key,
                location_id="UNKNOWN",
                source="unknown",
                status="failed",
                error_message=result["error"]
            )
            return result

        location_id = metadata['location_id']
        source = metadata['source']
        brand = metadata.get('brand')
        is_competitor = metadata.get('is_competitor', False)
        scrape_date = datetime.fromisoformat(metadata['scrape_date'])
        latitude = metadata.get('latitude')
        longitude = metadata.get('longitude')

        # Brand validation: fail file if brand is None
        if brand is None:
            result["status"] = "failed"
            result["error"] = f"No brand found in S3 path for file: {s3_key}"
            logger.error(result["error"])
            self.db.upsert_ingestion_file(
                s3_key=s3_key,
                location_id=location_id,
                source=source,
                status="failed",
                error_message=result["error"]
            )
            return result

        logger.info(f"📍 Location: {location_id}, Source: {source}, Brand: {brand}, Competitor: {is_competitor}")

        # Auto-populate locations table with brand info
        self.db.upsert_location(
            location_id=location_id,
            latitude=latitude,
            longitude=longitude,
            brand=brand,
            is_competitor=is_competitor
        )
        if latitude is not None and longitude is not None:
            logger.info(f"📍 Location coordinates saved: ({latitude}, {longitude})")

        # Mark as processing
        self.db.upsert_ingestion_file(
            s3_key=s3_key,
            location_id=location_id,
            source=source,
            brand=brand,
            is_competitor=is_competitor,
            scrape_date=metadata['scrape_date_display'],
            status="processing",
            started_at=datetime.now().isoformat()
        )

        new_review_ids = []

        try:
            # Download and validate file
            logger.progress("Downloading and validating file...")
            data = self.s3_source.download_file(s3_key)
            is_valid, error_msg = self.s3_source.validate_file_structure(data)

            if not is_valid:
                raise ValueError(f"Invalid file structure: {error_msg}")

            # Parse and insert reviews
            logger.parse(f"Parsing reviews from {s3_key}...")
            reviews_count, new_review_ids = self.parser.ingest_data(
                data, location_id, source, scrape_date,
                brand=brand, is_competitor=is_competitor
            )
            result["reviews_count"] = reviews_count
            logger.success(f"Inserted {reviews_count} reviews")

            # Enrich reviews (mandatory — enrich any unenriched reviews for this location)
            enriched_count = 0
            logger.enrich(f"Enriching unenriched reviews for {location_id}...")
            enriched_count = self.enricher.enrich_all_reviews(
                location_id=location_id
            )
            result["enriched_count"] = enriched_count

            # Mark as completed
            result["status"] = "completed"
            logger.complete(f"File processed: {reviews_count} reviews, {enriched_count} enriched")

            # Export enriched reviews to KB S3 bucket
            if enriched_count > 0:
                logger.s3(f"Exporting enriched reviews to Knowledge Base S3...")
                kb_result = self.kb_exporter.export_location(location_id, brand)
                if kb_result:
                    result["kb_s3_key"] = kb_result["s3_key"]
                    result["kb_review_count"] = kb_result["review_count"]

            self.db.upsert_ingestion_file(
                s3_key=s3_key,
                location_id=location_id,
                source=source,
                brand=brand,
                is_competitor=is_competitor,
                scrape_date=metadata['scrape_date_display'],
                status="completed",
                reviews_count=reviews_count,
                enriched_count=enriched_count,
                completed_at=datetime.now().isoformat()
            )

        except EnrichmentError as e:
            # Enrichment failed after retries — rollback all reviews for this file
            self._rollback_file(s3_key, new_review_ids, location_id, str(e), metadata, source, brand, is_competitor)
            result["status"] = "failed"
            result["error"] = str(e)

        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            logger.error(f"Failed to process {s3_key}: {e}")
            self.db.upsert_ingestion_file(
                s3_key=s3_key,
                location_id=location_id,
                source=source,
                brand=brand,
                is_competitor=is_competitor,
                scrape_date=metadata.get('scrape_date_display'),
                status="failed",
                error_message=str(e)
            )

        return result

    def _rollback_file(self, s3_key: str, review_ids: List[str],
                       location_id: str, error: str, metadata: Dict,
                       source: str, brand: str, is_competitor: bool):
        """Delete all reviews for a file and mark ingestion_file as failed."""
        logger.warning(f"Rolling back file {s3_key}: {error}")
        try:
            if review_ids:
                deleted = self.db.delete_reviews_by_ids(review_ids)
                logger.warning(f"Rolled back {deleted} reviews for {s3_key}")
        except Exception as rollback_err:
            logger.error(f"Rollback delete failed for {s3_key}: {rollback_err}")

        self.db.upsert_ingestion_file(
            s3_key=s3_key,
            location_id=location_id,
            source=source,
            brand=brand,
            is_competitor=is_competitor,
            scrape_date=metadata.get('scrape_date_display'),
            status="failed",
            error_message=error
        )

    def process_files(self, s3_keys: List[str]) -> List[Dict]:
        """Process multiple S3 files"""
        logger.batch(f"Processing {len(s3_keys)} files...")
        results = []
        for i, s3_key in enumerate(s3_keys, 1):
            logger.progress(f"File {i}/{len(s3_keys)}: {s3_key}")
            result = self.process_file(s3_key)
            results.append(result)
        logger.complete(f"Batch complete: {len(results)} files processed")
        return results

    def process_all_pending(self) -> List[Dict]:
        """Process all pending S3 files"""
        pending = self.get_pending_files()
        logger.info(f"📥 Found {len(pending)} pending files")
        s3_keys = [f['s3_key'] for f in pending]
        return self.process_files(s3_keys)
