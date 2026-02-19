"""
Review Parser - Handles parsing and normalization of review JSON data
"""
import json
from typing import List, Dict, Tuple
from datetime import datetime
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from storage.db import Database
from ingestion.date_parser import parse_relative_date
from utils.logger import get_logger

logger = get_logger(__name__)


class ReviewParser:
    def __init__(self, db: Database = None):
        self.db = db or Database()
    
    def parse_json_data(self, data: Dict, location_id: str, source: str,
                        scrape_date: datetime, brand: str = None,
                        is_competitor: bool = False) -> List[Dict]:
        """Parse review data from a dict (used for S3 downloads)"""
        return self._normalize_reviews(data, location_id, source, scrape_date,
                                       brand=brand, is_competitor=is_competitor)
    
    def _normalize_reviews(self, data: Dict, location_id: str, source: str,
                           scrape_date: datetime, brand: str = None,
                           is_competitor: bool = False) -> List[Dict]:
        """Normalize review data into standard format, skipping reviews with empty text"""
        reviews = []

        # Handle nested structure - reviews can be at root or under 'data'
        review_list = data.get('data', {}).get('reviews', []) or data.get('reviews', [])

        for review in review_list:
            # Skip reviews with empty/None/whitespace-only text (Req 8.1, 8.2)
            text = review.get('text')
            if not text or not text.strip():
                review_id = review.get('review_id', 'unknown')
                reason = "missing" if text is None else "empty or whitespace-only"
                logger.warning(f"Skipping review {review_id}: text is {reason}")
                continue

            normalized = {
                'location_id': location_id,
                'source': source,
                'brand': brand,
                'is_competitor': is_competitor,
                'review_id': review.get('review_id'),
                'rating': review.get('rating'),
                'review_text': review.get('text', ''),
                'reviewer_name': review.get('reviewer'),
                'reviewer_type': self._extract_reviewer_type(review.get('reviewer', '')),
                'relative_date': review.get('relative_date'),
                'review_date': parse_relative_date(review.get('relative_date', ''), scrape_date),
                'language': 'en',
                'raw_json': json.dumps(review)  # Store original review as JSON string
            }
            reviews.append(normalized)

        return reviews
    
    def _extract_reviewer_type(self, reviewer_info: str) -> str:
        """Extract reviewer type from reviewer string"""
        if reviewer_info and 'Local Guide' in reviewer_info:
            return 'local_guide'
        return 'standard'
    
    def ingest_data(self, data: Dict, location_id: str, source: str,
                    scrape_date: datetime, brand: str = None,
                    is_competitor: bool = False) -> Tuple[int, List[str]]:
        """Parse and insert reviews from dict data (used for S3).

        Returns:
            Tuple of (count_of_new_reviews, list_of_new_review_ids)
        """
        reviews = self.parse_json_data(data, location_id, source, scrape_date,
                                       brand=brand, is_competitor=is_competitor)
        count = 0
        new_review_ids: List[str] = []

        for review in reviews:
            try:
                _id, is_new = self.db.insert_review(review)
                if is_new:
                    count += 1
                    new_review_ids.append(review.get('review_id'))
            except Exception as e:
                logger.error(f"Error inserting review {review.get('review_id')}: {e}")

        return count, new_review_ids



