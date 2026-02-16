"""
SQLAlchemy ORM Models for Review Intelligence
"""
from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, Text, Boolean, DateTime, 
    ForeignKey, create_engine, Index
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.pool import QueuePool
import os

Base = declarative_base()


class Review(Base):
    __tablename__ = 'reviews'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    location_id = Column(String(10), nullable=False, index=True)
    source = Column(String(50), default='google')
    brand = Column(String(100), nullable=True, index=True)
    is_competitor = Column(Boolean, default=False, index=True)
    review_id = Column(String(255), unique=True, nullable=False)
    rating = Column(Float, nullable=False)
    review_text = Column(Text)
    reviewer_name = Column(String(255))
    reviewer_type = Column(String(50))
    review_date = Column(String(50))
    relative_date = Column(String(100))
    language = Column(String(10))
    raw_json = Column(Text)
    ingested_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship to enrichment
    enrichment = relationship("Enrichment", back_populates="review", uselist=False)
    embedding = relationship("Embedding", back_populates="review", uselist=False)
    
    __table_args__ = (
        Index('idx_reviews_location_date', 'location_id', 'review_date'),
        Index('idx_reviews_brand_location', 'brand', 'location_id'),
        Index('idx_reviews_competitor', 'is_competitor', 'location_id'),
    )


class Enrichment(Base):
    __tablename__ = 'enrichments'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    review_id = Column(String(255), ForeignKey('reviews.review_id'), unique=True, nullable=False)
    topics = Column(Text)  # JSON array
    sentiment = Column(String(20))
    sentiment_score = Column(Float)
    entities = Column(Text)  # JSON array
    key_phrases = Column(Text)  # JSON array
    urgency_level = Column(String(20))
    actionable = Column(Boolean, default=False)
    suggested_action = Column(Text)
    processed_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship back to review
    review = relationship("Review", back_populates="enrichment")


class InsightsCache(Base):
    __tablename__ = 'insights_cache'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    location_id = Column(String(10), nullable=False, index=True)
    time_window = Column(String(50), nullable=False)
    top_topics = Column(Text)
    key_drivers = Column(Text)
    representative_quotes = Column(Text)
    anomalies = Column(Text)
    generated_summary = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


class Embedding(Base):
    __tablename__ = 'embeddings'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    review_id = Column(String(255), ForeignKey('reviews.review_id'), unique=True, nullable=False)
    embedding = Column(Text)  # Store as JSON string for PostgreSQL compatibility
    created_at = Column(DateTime, default=datetime.utcnow)
    
    review = relationship("Review", back_populates="embedding")


class Location(Base):
    __tablename__ = 'locations'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    location_id = Column(String(10), unique=True, nullable=False)
    name = Column(String(255))
    latitude = Column(Float)
    longitude = Column(Float)
    address = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


class IngestionFile(Base):
    __tablename__ = 'ingestion_files'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    s3_key = Column(String(500), unique=True, nullable=False)
    location_id = Column(String(10), nullable=False)
    source = Column(String(50), nullable=False)
    brand = Column(String(100), nullable=True)
    is_competitor = Column(Boolean, default=False)
    scrape_date = Column(String(50))
    status = Column(String(20), default='pending')
    reviews_count = Column(Integer, default=0)
    enriched_count = Column(Integer, default=0)
    error_message = Column(Text)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_ingestion_status', 'status'),
    )


# Database connection helper
def get_database_url():
    """
    Get database URL based on environment.

    Resolution order:
    1. DATABASE_URL (explicit override)
    2. APP_ENV -> DATABASE_URL_DEV or DATABASE_URL_PROD
    3. SQLite fallback
    """
    # Explicit override takes priority
    explicit = os.getenv('DATABASE_URL')
    if explicit:
        return explicit

    env = os.getenv('APP_ENV', 'dev').lower()
    if env == 'production':
        url = os.getenv('DATABASE_URL_PROD')
    else:
        url = os.getenv('DATABASE_URL_DEV')

    return url or 'sqlite:///data/reviews.db'


def create_db_engine(database_url: str = None):
    """Create SQLAlchemy engine with appropriate settings"""
    url = database_url or get_database_url()
    
    if url.startswith('postgresql'):
        # PostgreSQL with connection pooling
        return create_engine(
            url,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True
        )
    else:
        # SQLite
        return create_engine(url, connect_args={"check_same_thread": False})


def get_session_factory(engine=None):
    """Create session factory"""
    if engine is None:
        engine = create_db_engine()
    return sessionmaker(bind=engine, autocommit=False, autoflush=False)
