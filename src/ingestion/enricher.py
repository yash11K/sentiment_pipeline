import json
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from storage.db import Database
from utils.bedrock import BedrockClient
from utils.logger import get_logger
from typing import List, Dict

logger = get_logger(__name__)


class EnrichmentError(Exception):
    """Raised when enrichment fails after all retries."""
    pass


class ReviewEnricher:
    MAX_RETRIES = 2

    def __init__(self, db: Database = None, batch_size: int = 20):
        self.db = db or Database()
        self.bedrock = BedrockClient()
        self.batch_size = batch_size
        logger.start(f"Review enricher initialized (batch_size={batch_size})")

    def _trim_review_for_llm(self, review: Dict) -> Dict:
        """Extract only review_id, rating, text, reviewer, relative_date from raw_json.

        Parses the raw_json field (a JSON string stored in the DB) and returns
        a dict with exactly the five allowed keys for LLM enrichment.
        """
        raw = json.loads(review.get("raw_json", "{}"))
        return {
            "review_id": raw.get("review_id"),
            "rating": raw.get("rating"),
            "text": raw.get("text"),
            "reviewer": raw.get("reviewer"),
            "relative_date": raw.get("relative_date"),
        }


    def enrich_batch(self, reviews: List[Dict]) -> List[Dict]:
        """Enrich a batch of reviews using LLM with retry logic.

        Trims each review to only the essential fields before sending to Bedrock.
        Retries up to MAX_RETRIES additional times on failure.
        Raises EnrichmentError if all attempts fail.

        After receiving enrichments, assigns review_id purely by position —
        we never trust the LLM to reproduce those long base64 IDs correctly.
        """
        trimmed = [self._trim_review_for_llm(r) for r in reviews]
        last_error = None
        for attempt in range(1, self.MAX_RETRIES + 2):  # 1 initial + MAX_RETRIES retries
            try:
                logger.batch(f"Sending batch of {len(trimmed)} reviews to LLM (attempt {attempt}/{self.MAX_RETRIES + 1})...")
                enrichments = self.bedrock.enrich_reviews_batch(trimmed)
                logger.success(f"Received {len(enrichments)} enrichments")
                # Stamp each enrichment with the known-good DB review_id by position
                for idx, enrichment in enumerate(enrichments):
                    if idx < len(reviews):
                        enrichment["review_id"] = reviews[idx]["review_id"]
                    else:
                        logger.warning(f"LLM returned more enrichments ({len(enrichments)}) than reviews sent ({len(reviews)}), dropping extra")
                        enrichments = enrichments[:len(reviews)]
                        break
                return enrichments
            except Exception as e:
                last_error = e
                if attempt <= self.MAX_RETRIES:
                    logger.warning(f"Batch enrichment attempt {attempt} failed: {e}. Retrying...")
                else:
                    logger.error(f"Batch enrichment failed after {self.MAX_RETRIES + 1} attempts: {e}")
        raise EnrichmentError(f"Enrichment failed after {self.MAX_RETRIES + 1} attempts: {last_error}")

    def enrich_all_reviews(self, location_id: str = None, limit: int = None):
        """Enrich all reviews in database using batch processing"""
        reviews = self.db.get_reviews(location_id=location_id, unenriched_only=True, limit=limit or 10000)
        total = len(reviews)
        count = 0

        logger.enrich(f"Processing {total} reviews in batches of {self.batch_size}...")

        # Process in batches
        for i in range(0, len(reviews), self.batch_size):
            batch_num = (i // self.batch_size) + 1
            total_batches = (total + self.batch_size - 1) // self.batch_size
            batch = reviews[i:i + self.batch_size]

            logger.progress(f"Batch {batch_num}/{total_batches}")
            try:
                enrichments = self.enrich_batch(batch)
                for enrichment in enrichments:
                    self.db.insert_enrichment(enrichment)
                    count += 1
                logger.database(f"Saved to database. Progress: {count}/{total} reviews ({int(count/total*100)}%)")
            except EnrichmentError:
                raise  # Propagate to pipeline for rollback
            except Exception as e:
                logger.error(f"Batch {batch_num} failed: {e}")

        logger.complete(f"Enrichment complete: {count}/{total} reviews processed")
        return count


