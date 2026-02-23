import json
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from storage.db import Database
from utils.bedrock import BedrockClient
from utils.logger import get_logger
from typing import List, Dict, Optional

logger = get_logger(__name__)

REQUIRED_FIELDS = {"sentiment", "sentiment_score", "topics"}
VALID_SENTIMENTS = {"positive", "negative", "neutral"}
OPTIONAL_DEFAULTS = {
    "entities": [],
    "key_phrases": [],
    "urgency_level": "low",
    "actionable": False,
    "suggested_action": None,
}


def validate_enrichment(enrichment: dict) -> Optional[dict]:
    """Validate and normalize a single enrichment dict.

    Returns the enrichment with defaults filled, or None if invalid.
    """
    # Check required fields exist
    for field in REQUIRED_FIELDS:
        if field not in enrichment or enrichment[field] is None:
            return None

    # Validate sentiment value
    if enrichment["sentiment"] not in VALID_SENTIMENTS:
        return None

    # Validate sentiment_score range
    try:
        score = float(enrichment["sentiment_score"])
        if score < -1.0 or score > 1.0:
            return None
        enrichment["sentiment_score"] = score
    except (TypeError, ValueError):
        return None

    # Validate topics is a list
    if not isinstance(enrichment["topics"], list):
        return None

    # Fill optional field defaults
    for field, default in OPTIONAL_DEFAULTS.items():
        if field not in enrichment or enrichment[field] is None:
            enrichment[field] = default if not isinstance(default, list) else list(default)

    return enrichment


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
        """Extract only the fields needed for LLM enrichment from parsed columns."""
        return {
            "review_id": review.get("review_id"),
            "rating": review.get("rating"),
            "text": review.get("review_text"),
            "reviewer": review.get("reviewer_name"),
            "relative_date": review.get("relative_date"),
        }

    def enrich_batch(self, reviews: List[Dict]) -> List[Dict]:
        """Enrich a batch with partial-result recovery and targeted retry.

        1. Call LLM for all reviews
        2. Validate each returned enrichment
        3. If some reviews lack valid enrichments, retry only those
        4. Accumulate across retries
        5. Raise EnrichmentError only if zero valid enrichments after all attempts
        """
        all_enrichments = [None] * len(reviews)  # positional slots
        remaining_indices = list(range(len(reviews)))

        for attempt in range(1, self.MAX_RETRIES + 2):
            batch_reviews = [reviews[i] for i in remaining_indices]
            trimmed = [self._trim_review_for_llm(r) for r in batch_reviews]

            logger.batch(f"Sending batch of {len(trimmed)} reviews to LLM (attempt {attempt}/{self.MAX_RETRIES + 1})...")
            raw_enrichments = self.bedrock.enrich_reviews_batch(trimmed)
            logger.success(f"Received {len(raw_enrichments)} enrichments")

            # Validate each enrichment, stamp review_id by position
            newly_filled = []
            for idx, enrichment in enumerate(raw_enrichments):
                if idx >= len(remaining_indices):
                    break
                validated = validate_enrichment(enrichment)
                if validated:
                    original_idx = remaining_indices[idx]
                    validated["review_id"] = reviews[original_idx]["review_id"]
                    all_enrichments[original_idx] = validated
                    newly_filled.append(original_idx)

            remaining_indices = [i for i in remaining_indices if all_enrichments[i] is None]

            if not remaining_indices:
                break  # All reviews enriched

            if not newly_filled and attempt > self.MAX_RETRIES:
                # Zero progress on this attempt and out of retries
                break

        valid = [e for e in all_enrichments if e is not None]

        if not valid:
            raise EnrichmentError(f"Zero valid enrichments after {self.MAX_RETRIES + 1} attempts")

        if remaining_indices:
            missing_ids = [reviews[i]["review_id"] for i in remaining_indices]
            logger.warning(f"Could not enrich {len(missing_ids)} reviews: {missing_ids}")

        return valid


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


