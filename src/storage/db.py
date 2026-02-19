"""
Database access layer using SQLAlchemy ORM
Supports both SQLite (local dev) and PostgreSQL (production)
"""
from datetime import datetime
from typing import List, Dict, Optional
import json
import os

from sqlalchemy import create_engine, text, func, and_, or_
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager

from storage.models import (
    Base, Review, Enrichment, InsightsCache, Embedding, 
    Location, IngestionFile, create_db_engine, get_database_url
)
from utils.logger import get_logger

logger = get_logger(__name__)


class Database:
    def __init__(self, database_url: str = None):
        self.database_url = database_url or get_database_url()
        self.engine = create_db_engine(self.database_url)
        self.SessionLocal = sessionmaker(bind=self.engine, autocommit=False, autoflush=False)
        self.init_db()
        logger.database(f"Database initialized ({self.database_url.split('@')[-1] if '@' in self.database_url else 'sqlite'})")
    
    def init_db(self):
        """Create all tables if they don't exist"""
        Base.metadata.create_all(bind=self.engine)
    
    @contextmanager
    def get_session(self) -> Session:
        """Context manager for database sessions"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def get_connection(self):
        """Legacy method for raw SQL - returns a connection"""
        return self.engine.connect()
    
    # ============ REVIEW METHODS ============
    
    def insert_review(self, review: Dict) -> tuple:
        """Insert a new review, or skip entirely if review_id already exists.
        Returns (id, is_new) where is_new is True for new inserts."""
        with self.get_session() as session:
            existing = session.query(Review).filter_by(
                review_id=review.get('review_id')
            ).first()

            if existing:
                return (existing.id, False)

            db_review = Review(
                location_id=review.get('location_id'),
                source=review.get('source', 'google'),
                brand=review.get('brand'),
                is_competitor=review.get('is_competitor', False),
                review_id=review.get('review_id'),
                rating=review.get('rating'),
                review_text=review.get('review_text'),
                reviewer_name=review.get('reviewer_name'),
                reviewer_type=review.get('reviewer_type'),
                review_date=review.get('review_date'),
                relative_date=review.get('relative_date'),
                language=review.get('language', 'en'),
                raw_json=review.get('raw_json')
            )
            session.add(db_review)
            session.flush()
            return (db_review.id, True)
    
    def get_reviews(self, location_id: Optional[str] = None,
                    min_rating: Optional[float] = None,
                    max_rating: Optional[float] = None,
                    brand: Optional[str] = None,
                    is_competitor: Optional[bool] = None,
                    unenriched_only: bool = False,
                    limit: int = 100) -> List[Dict]:
        """Get reviews with optional filters"""
        with self.get_session() as session:
            query = session.query(Review)

            if location_id:
                query = query.filter(Review.location_id == location_id)
            if min_rating:
                query = query.filter(Review.rating >= min_rating)
            if max_rating:
                query = query.filter(Review.rating <= max_rating)
            if brand:
                query = query.filter(Review.brand == brand.lower())
            if is_competitor is not None:
                query = query.filter(Review.is_competitor == is_competitor)
            if unenriched_only:
                query = query.filter(~Review.review_id.in_(
                    session.query(Enrichment.review_id)
                ))

            query = query.order_by(Review.review_date.desc()).limit(limit)

            return [self._review_to_dict(r) for r in query.all()]
    
    def get_review_with_enrichment(self, review_id: str) -> Optional[Dict]:
        """Get a review with its enrichment data"""
        with self.get_session() as session:
            review = session.query(Review).filter(Review.review_id == review_id).first()
            if not review:
                return None
            
            result = self._review_to_dict(review)
            if review.enrichment:
                result.update({
                    'topics': review.enrichment.topics,
                    'sentiment': review.enrichment.sentiment,
                    'sentiment_score': review.enrichment.sentiment_score,
                    'entities': review.enrichment.entities
                })
            return result
    
    def _review_to_dict(self, review: Review) -> Dict:
        """Convert Review model to dictionary"""
        return {
            'id': review.id,
            'location_id': review.location_id,
            'source': review.source,
            'brand': review.brand,
            'is_competitor': review.is_competitor,
            'review_id': review.review_id,
            'rating': review.rating,
            'review_text': review.review_text,
            'reviewer_name': review.reviewer_name,
            'reviewer_type': review.reviewer_type,
            'review_date': review.review_date,
            'relative_date': review.relative_date,
            'language': review.language,
            'raw_json': review.raw_json,
            'ingested_at': review.ingested_at.isoformat() if review.ingested_at else None
        }
    
    # ============ ENRICHMENT METHODS ============
    
    def insert_enrichment(self, enrichment: Dict):
        """Insert or update enrichment data"""
        with self.get_session() as session:
            existing = session.query(Enrichment).filter_by(
                review_id=enrichment.get('review_id')
            ).first()

            if existing:
                existing.topics = json.dumps(enrichment.get('topics', []))
                existing.sentiment = enrichment.get('sentiment')
                existing.sentiment_score = enrichment.get('sentiment_score')
                existing.entities = json.dumps(enrichment.get('entities', []))
                existing.key_phrases = json.dumps(enrichment.get('key_phrases', []))
                existing.urgency_level = enrichment.get('urgency_level')
                existing.actionable = bool(enrichment.get('actionable'))
                existing.suggested_action = enrichment.get('suggested_action')
                existing.processed_at = datetime.utcnow()
            else:
                db_enrichment = Enrichment(
                    review_id=enrichment.get('review_id'),
                    topics=json.dumps(enrichment.get('topics', [])),
                    sentiment=enrichment.get('sentiment'),
                    sentiment_score=enrichment.get('sentiment_score'),
                    entities=json.dumps(enrichment.get('entities', [])),
                    key_phrases=json.dumps(enrichment.get('key_phrases', [])),
                    urgency_level=enrichment.get('urgency_level'),
                    actionable=bool(enrichment.get('actionable')),
                    suggested_action=enrichment.get('suggested_action')
                )
                session.add(db_enrichment)

    def get_enriched_reviews_for_export(self, location_id: str, brand: str = None) -> List[Dict]:
        """Get all reviews with enrichment data for KB export, filtered by location and optionally brand."""
        with self.get_session() as session:
            query = session.query(Review, Enrichment).join(
                Enrichment, Review.review_id == Enrichment.review_id
            ).filter(Review.location_id == location_id)

            if brand:
                query = query.filter(Review.brand == brand.lower())

            query = query.order_by(Review.review_date.desc())

            results = []
            for review, enrichment in query.all():
                results.append({
                    'review_id': review.review_id,
                    'location_id': review.location_id,
                    'brand': review.brand,
                    'is_competitor': review.is_competitor,
                    'rating': review.rating,
                    'review_text': review.review_text,
                    'reviewer_name': review.reviewer_name,
                    'review_date': review.review_date,
                    'sentiment': enrichment.sentiment,
                    'sentiment_score': enrichment.sentiment_score,
                    'topics': json.loads(enrichment.topics) if enrichment.topics else [],
                    'entities': json.loads(enrichment.entities) if enrichment.entities else [],
                    'key_phrases': json.loads(enrichment.key_phrases) if enrichment.key_phrases else [],
                    'urgency_level': enrichment.urgency_level,
                    'actionable': enrichment.actionable,
                    'suggested_action': enrichment.suggested_action,
                })
            return results

    
    # ============ LOCATION METHODS ============
    
    def upsert_location(self, location_id: str, name: str = None,
                        latitude: float = None, longitude: float = None,
                        address: str = None, brand: str = None,
                        is_competitor: bool = False):
        """Insert or update location metadata. Merges brand into brands list.
        Uses AI to get name/address if not provided on first insert."""
        with self.get_session() as session:
            location = session.query(Location).filter(
                Location.location_id == location_id
            ).first()
            
            if location:
                if name: location.name = name
                if latitude: location.latitude = latitude
                if longitude: location.longitude = longitude
                if address: location.address = address
                # Merge brand into existing brands list
                if brand:
                    existing_brands = json.loads(location.brands) if location.brands else []
                    brand_names = {b['brand'] for b in existing_brands}
                    if brand.lower() not in brand_names:
                        existing_brands.append({'brand': brand.lower(), 'is_competitor': is_competitor})
                        location.brands = json.dumps(existing_brands)
            else:
                # Use AI to get name and address if not provided
                if not name or not address:
                    from utils.bedrock import BedrockClient
                    bedrock = BedrockClient()
                    location_info = bedrock.get_location_info(location_id)
                    if not name:
                        name = location_info.get('name', location_id)
                    if not address:
                        address = location_info.get('address', location_id)
                
                brands_list = [{'brand': brand.lower(), 'is_competitor': is_competitor}] if brand else []
                location = Location(
                    location_id=location_id,
                    name=name,
                    latitude=latitude,
                    longitude=longitude,
                    address=address,
                    brands=json.dumps(brands_list)
                )
                session.add(location)
    
    def update_location(self, location_id: str, name: str = None,
                        latitude: float = None, longitude: float = None,
                        address: str = None) -> Optional[Dict]:
        """Update an existing location. Returns updated location or None if not found."""
        with self.get_session() as session:
            location = session.query(Location).filter(
                Location.location_id == location_id
            ).first()
            
            if not location:
                return None
            
            if name is not None:
                location.name = name
            if latitude is not None:
                location.latitude = latitude
            if longitude is not None:
                location.longitude = longitude
            if address is not None:
                location.address = address
            
            session.flush()
            return self._location_to_dict(location)
    
    def get_all_locations(self) -> List[Dict]:
        """Get all locations directly from the locations table — no joins needed."""
        with self.get_session() as session:
            locations = session.query(Location).all()
            return [self._location_to_dict(loc) for loc in locations]
    
    def _location_to_dict(self, location: Location) -> Dict:
        """Convert Location model to dictionary."""
        return {
            'location_id': location.location_id,
            'name': location.name,
            'latitude': location.latitude,
            'longitude': location.longitude,
            'address': location.address,
            'brands': json.loads(location.brands) if location.brands else []
        }

    
    # ============ INGESTION FILE METHODS ============
    
    def get_ingestion_file(self, s3_key: str) -> Optional[Dict]:
        """Get ingestion file record by S3 key"""
        with self.get_session() as session:
            record = session.query(IngestionFile).filter(
                IngestionFile.s3_key == s3_key
            ).first()
            return self._ingestion_file_to_dict(record) if record else None
    
    def upsert_ingestion_file(self, s3_key: str, location_id: str, source: str,
                              brand: str = None, is_competitor: bool = False,
                              scrape_date: str = None, status: str = 'pending',
                              reviews_count: int = 0, enriched_count: int = 0,
                              error_message: str = None, started_at: str = None,
                              completed_at: str = None):
        """Insert or update ingestion file tracking record"""
        with self.get_session() as session:
            record = session.query(IngestionFile).filter(
                IngestionFile.s3_key == s3_key
            ).first()
            
            if record:
                record.status = status
                record.reviews_count = reviews_count
                record.enriched_count = enriched_count
                record.error_message = error_message
                if brand is not None:
                    record.brand = brand
                    record.is_competitor = is_competitor
                if started_at:
                    record.started_at = datetime.fromisoformat(started_at)
                if completed_at:
                    record.completed_at = datetime.fromisoformat(completed_at)
            else:
                record = IngestionFile(
                    s3_key=s3_key,
                    location_id=location_id,
                    source=source,
                    brand=brand,
                    is_competitor=is_competitor,
                    scrape_date=scrape_date,
                    status=status,
                    reviews_count=reviews_count,
                    enriched_count=enriched_count,
                    error_message=error_message,
                    started_at=datetime.fromisoformat(started_at) if started_at else None,
                    completed_at=datetime.fromisoformat(completed_at) if completed_at else None
                )
                session.add(record)
    
    def get_processed_s3_keys(self) -> List[str]:
        """Get list of all S3 keys that have been processed or are in progress"""
        with self.get_session() as session:
            records = session.query(IngestionFile.s3_key).filter(
                IngestionFile.status.in_(['completed', 'processing'])
            ).all()
            return [r.s3_key for r in records]
    def delete_reviews_by_ids(self, review_ids: List[str]) -> int:
        """Delete reviews by list of review_ids. CASCADE handles enrichments/embeddings."""
        if not review_ids:
            return 0
        with self.get_session() as session:
            deleted = session.query(Review).filter(
                Review.review_id.in_(review_ids)
            ).delete(synchronize_session='fetch')
            return deleted


    
    def get_ingestion_history(self, limit: int = 50) -> List[Dict]:
        """Get recent ingestion history"""
        with self.get_session() as session:
            records = session.query(IngestionFile).order_by(
                IngestionFile.created_at.desc()
            ).limit(limit).all()
            return [self._ingestion_file_to_dict(r) for r in records]
    
    def _ingestion_file_to_dict(self, record: IngestionFile) -> Dict:
        """Convert IngestionFile model to dictionary"""
        return {
            'id': record.id,
            's3_key': record.s3_key,
            'location_id': record.location_id,
            'source': record.source,
            'brand': record.brand,
            'is_competitor': record.is_competitor,
            'scrape_date': record.scrape_date,
            'status': record.status,
            'reviews_count': record.reviews_count,
            'enriched_count': record.enriched_count,
            'error_message': record.error_message,
            'started_at': record.started_at.isoformat() if record.started_at else None,
            'completed_at': record.completed_at.isoformat() if record.completed_at else None,
            'created_at': record.created_at.isoformat() if record.created_at else None
        }
    
    # ============ INSIGHTS CACHE METHODS ============
    
    def get_cached_insights(self, location_id: str, time_window: str = 'all') -> Optional[Dict]:
        """Get cached insights for a location"""
        with self.get_session() as session:
            record = session.query(InsightsCache).filter(
                and_(
                    InsightsCache.location_id == location_id,
                    InsightsCache.time_window == time_window
                )
            ).order_by(InsightsCache.created_at.desc()).first()
            
            if not record:
                return None
            
            return {
                'location_id': record.location_id,
                'time_window': record.time_window,
                'top_topics': json.loads(record.top_topics) if record.top_topics else [],
                'key_drivers': json.loads(record.key_drivers) if record.key_drivers else [],
                'representative_quotes': json.loads(record.representative_quotes) if record.representative_quotes else [],
                'anomalies': json.loads(record.anomalies) if record.anomalies else [],
                'generated_summary': record.generated_summary,
                'created_at': record.created_at.isoformat() if record.created_at else None
            }
    
    def save_insights(self, location_id: str, time_window: str, insights: Dict):
        """Save generated insights to cache"""
        with self.get_session() as session:
            record = InsightsCache(
                location_id=location_id,
                time_window=time_window,
                top_topics=json.dumps(insights.get('top_topics', [])),
                key_drivers=json.dumps(insights.get('key_drivers', [])),
                representative_quotes=json.dumps(insights.get('representative_quotes', [])),
                anomalies=json.dumps(insights.get('anomalies', [])),
                generated_summary=insights.get('generated_summary')
            )
            session.add(record)
