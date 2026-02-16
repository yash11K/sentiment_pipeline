"""
Ingestion Pipeline - Run this to process review files
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from storage.db import Database
from ingestion.parser import ReviewParser
from ingestion.enricher import ReviewEnricher
from utils.logger import get_logger

logger = get_logger(__name__)


def run_ingestion(file_path: str, location_id: str = None, batch_size: int = 20):
    """Run full ingestion pipeline with batch LLM processing"""
    logger.start(f"Starting ingestion for {file_path}...")
    
    # Step 1: Parse and insert reviews
    logger.parse("Step 1: Parsing reviews...")
    parser = ReviewParser()
    count = parser.ingest_file(file_path, location_id)
    logger.success(f"Inserted {count} reviews")
    
    # Step 2: Enrich reviews in batches
    logger.enrich(f"Step 2: Enriching reviews (batch size: {batch_size})...")
    enricher = ReviewEnricher(batch_size=batch_size)
    enriched = enricher.enrich_all_reviews(location_id=location_id)
    logger.success(f"Enriched {enriched} reviews")
    
    logger.complete(f"Ingestion complete! Processed {count} reviews.")

if __name__ == "__main__":
    # Process JFK reviews with batch LLM enrichment
    run_ingestion("data/raw/LAX_reviews_10_01_2026.json", "LAX", batch_size=20)
