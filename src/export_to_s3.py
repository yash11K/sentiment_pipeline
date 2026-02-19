"""
Backfill tool - Export all enriched reviews to KB S3 bucket.

This is a standalone script for bulk re-exporting. During normal operation,
the pipeline exports automatically after enrichment.

Usage:
    python src/export_to_s3.py
    python src/export_to_s3.py --brand avis
    python src/export_to_s3.py --location ATL
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import config
from storage.db import Database
from ingestion.kb_exporter import KBExporter
from utils.logger import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Backfill KB S3 bucket with enriched reviews")
    parser.add_argument("--location", help="Export only this location (e.g. ATL)")
    parser.add_argument("--brand", help="Export only this brand (e.g. avis)")
    args = parser.parse_args()

    db = Database()
    exporter = KBExporter(db)

    # Get distinct location+brand combos from the DB
    with db.get_session() as session:
        from storage.models import Review
        query = session.query(Review.location_id, Review.brand).distinct()
        if args.location:
            query = query.filter(Review.location_id == args.location.upper())
        if args.brand:
            query = query.filter(Review.brand == args.brand.lower())
        combos = query.all()

    logger.start(f"Found {len(combos)} location/brand combinations to export")

    total_reviews = 0
    for location_id, brand in combos:
        result = exporter.export_location(location_id, brand)
        if result:
            total_reviews += result["review_count"]

    logger.complete(f"Backfill done. Exported {total_reviews} reviews across {len(combos)} files.")


if __name__ == "__main__":
    main()
