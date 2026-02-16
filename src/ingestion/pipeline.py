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
from ingestion.enricher import ReviewEnricher
from ingestion.s3_source import S3ReviewSource
from ingestion.kb_exporter import KBExporter
from utils.logger import get_logger

logger = get_logger(__name__)


class IngestionPipeline:
    def __init__(self, bucket_name: str, prefix: str = "reviews/",
                 region: str = "us-east-1", batch_size: int = 20):
        self.db = Database()
        self.parser = ReviewParser(self.db)
        self.enricher = ReviewEnricher(self.db, batch_size=batch_size)
        self.s3_source = S3ReviewSource(bucket_name, prefix, region)
        self.kb_exporter = KBExporter(self.db, region=region)
        self.batch_size = batch_size
        logger.start(f"Ingestion pipeline initialized (bucket={bucket_name}, batch_size={batch_size})")

    def get_pending_files(self) -> List[Dict]:
        """Get list of S3 files pending ingestion"""
        return self.s3_source.get_pending_files()

    def get_ingestion_history(self, limit: int = 50) -> List[Dict]:
        """Get recent ingestion history"""
        return self.db.get_ingestion_history(limit)

    def process_file(self, s3_key: str, enrich: bool = True) -> Dict:
        """
        Process a single S3 file through the ingestion pipeline.

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

        logger.info(f"📍 Location: {location_id}, Source: {source}, Brand: {brand or 'unknown'}, Competitor: {is_competitor}")

        # Auto-populate locations table if coordinates are available
        if latitude is not None and longitude is not None:
            self.db.upsert_location(
                location_id=location_id,
                latitude=latitude,
                longitude=longitude
            )
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

        try:
            # Download and validate file
            logger.progress("Downloading and validating file...")
            data = self.s3_source.download_file(s3_key)
            is_valid, error_msg = self.s3_source.validate_file_structure(data)

            if not is_valid:
                raise ValueError(f"Invalid file structure: {error_msg}")

            # Parse and insert reviews
            logger.parse(f"Parsing reviews from {s3_key}...")
            reviews_count = self.parser.ingest_data(
                data, location_id, source, scrape_date,
                brand=brand, is_competitor=is_competitor
            )
            result["reviews_count"] = reviews_count
            logger.success(f"Inserted {reviews_count} reviews")

            # Enrich reviews if requested
            enriched_count = 0
            if enrich and reviews_count > 0:
                logger.enrich(f"Enriching {reviews_count} reviews...")
                enriched_count = self.enricher.enrich_all_reviews(
                    location_id=location_id,
                    limit=reviews_count
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

    def process_files(self, s3_keys: List[str], enrich: bool = True) -> List[Dict]:
        """Process multiple S3 files"""
        logger.batch(f"Processing {len(s3_keys)} files...")
        results = []
        for i, s3_key in enumerate(s3_keys, 1):
            logger.progress(f"File {i}/{len(s3_keys)}: {s3_key}")
            result = self.process_file(s3_key, enrich=enrich)
            results.append(result)
        logger.complete(f"Batch complete: {len(results)} files processed")
        return results

    def process_all_pending(self, enrich: bool = True) -> List[Dict]:
        """Process all pending S3 files"""
        pending = self.get_pending_files()
        logger.info(f"📥 Found {len(pending)} pending files")
        s3_keys = [f['s3_key'] for f in pending]
        return self.process_files(s3_keys, enrich=enrich)
