"""
Review Parser - Handles parsing and normalization of review JSON files
Supports file naming convention: {LOCODE}_{SOURCE}_{DD}_{MM}_{YYYY}.json
Also supports legacy format: {LOCODE}_reviews_{DD}_{MM}_{YYYY}.json
"""
import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import sys
import re
sys.path.append(str(Path(__file__).parent.parent))

from storage.db import Database
from ingestion.date_parser import parse_relative_date
from utils.logger import get_logger

logger = get_logger(__name__)


class ReviewParser:
    # Pattern: LOCODE_SOURCE_DD_MM_YYYY.json (new format)
    FILENAME_PATTERN = re.compile(
        r'^([A-Z]{3})_([a-z_]+)_(\d{2})_(\d{2})_(\d{4})\.json$',
        re.IGNORECASE
    )
    
    def __init__(self, db: Database = None):
        self.db = db or Database()
    
    def parse_filename_metadata(self, filename: str) -> Dict:
        """
        Parse filename to extract location_id, source, and scrape_date.
        Supports both old format (LOCODE_reviews_DD_MM_YYYY) and new format (LOCODE_SOURCE_DD_MM_YYYY)
        """
        basename = Path(filename).name
        stem = Path(filename).stem
        parts = stem.split('_')
        
        # Try new format: LOCODE_SOURCE_DD_MM_YYYY
        match = self.FILENAME_PATTERN.match(basename)
        if match:
            location_id, source, day, month, year = match.groups()
            try:
                scrape_date = datetime(int(year), int(month), int(day))
                return {
                    "location_id": location_id.upper(),
                    "source": source.lower(),
                    "scrape_date": scrape_date
                }
            except ValueError:
                pass
        
        # Fallback to old format: LOCODE_reviews_DD_MM_YYYY (assumes google source)
        if len(parts) >= 4:
            try:
                location_id = parts[0]
                day, month, year = int(parts[-3]), int(parts[-2]), int(parts[-1])
                scrape_date = datetime(year, month, day)
                return {
                    "location_id": location_id.upper(),
                    "source": "google",
                    "scrape_date": scrape_date
                }
            except (ValueError, IndexError):
                pass
        
        # Default fallback
        return {
            "location_id": parts[0].upper() if parts else "UNKNOWN",
            "source": "google",
            "scrape_date": datetime.now()
        }
    
    def parse_json_file(self, file_path: str, location_id: str = None, 
                        source: str = None) -> List[Dict]:
        """Parse a review JSON file and normalize the data"""
        # Try multiple encodings
        for encoding in ['utf-16-le', 'utf-16', 'utf-8', 'utf-8-sig', 'latin-1']:
            try:
                with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                    data = json.load(f)
                    break
            except (UnicodeDecodeError, json.JSONDecodeError) as e:
                if encoding == 'latin-1':
                    raise Exception(f"Could not parse JSON file: {e}")
                continue
        
        # Extract metadata from filename
        file_metadata = self.parse_filename_metadata(file_path)
        
        if not location_id:
            location_id = file_metadata['location_id']
        if not source:
            source = file_metadata['source']
        scrape_date = file_metadata['scrape_date']
        
        return self._normalize_reviews(data, location_id, source, scrape_date)
    
    def parse_json_data(self, data: Dict, location_id: str, source: str,
                        scrape_date: datetime, brand: str = None,
                        is_competitor: bool = False) -> List[Dict]:
        """Parse review data from a dict (used for S3 downloads)"""
        return self._normalize_reviews(data, location_id, source, scrape_date,
                                       brand=brand, is_competitor=is_competitor)
    
    def _normalize_reviews(self, data: Dict, location_id: str, source: str,
                           scrape_date: datetime, brand: str = None,
                           is_competitor: bool = False) -> List[Dict]:
        """Normalize review data into standard format"""
        reviews = []
        
        # Handle nested structure - reviews can be at root or under 'data'
        review_list = data.get('data', {}).get('reviews', []) or data.get('reviews', [])
        
        for review in review_list:
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
    
    def ingest_file(self, file_path: str, location_id: str = None, 
                    source: str = None) -> int:
        """Parse and insert reviews into database"""
        logger.parse(f"Ingesting file: {file_path}")
        reviews = self.parse_json_file(file_path, location_id, source)
        count = 0
        
        for review in reviews:
            try:
                self.db.insert_review(review)
                count += 1
            except Exception as e:
                logger.error(f"Error inserting review {review.get('review_id')}: {e}")
        
        logger.success(f"Ingested {count} reviews from {file_path}")
        return count
    
    def ingest_data(self, data: Dict, location_id: str, source: str,
                    scrape_date: datetime, brand: str = None,
                    is_competitor: bool = False) -> int:
        """Parse and insert reviews from dict data (used for S3)"""
        reviews = self.parse_json_data(data, location_id, source, scrape_date,
                                       brand=brand, is_competitor=is_competitor)
        count = 0
        
        for review in reviews:
            try:
                self.db.insert_review(review)
                count += 1
            except Exception as e:
                logger.error(f"Error inserting review {review.get('review_id')}: {e}")
        
        return count


if __name__ == "__main__":
    # Test with sample data
    parser = ReviewParser()
    count = parser.ingest_file("data/raw/LAX_reviews_10_01_2026.json", "LAX")
    logger.complete(f"Ingested {count} reviews")
