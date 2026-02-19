"""
Centralized configuration — resolves all settings based on APP_ENV.

Usage:
    from config import config

    config.DATABASE_URL      # resolved for current environment
    config.KB_S3_BUCKET      # dev or prod bucket
    config.BEDROCK_KB_ID     # dev or prod knowledge base
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Environment-aware configuration. Reads APP_ENV and resolves all values."""

    def __init__(self):
        self.APP_ENV = os.getenv("APP_ENV", "dev").lower()
        self.is_prod = self.APP_ENV == "production"

        # --- Database ---
        explicit_db = os.getenv("DATABASE_URL")
        if explicit_db:
            self.DATABASE_URL = explicit_db
        elif self.is_prod:
            self.DATABASE_URL = os.getenv("DATABASE_URL_PROD", "")
        else:
            self.DATABASE_URL = os.getenv("DATABASE_URL_DEV", "") or "sqlite:///data/reviews.db"

        # --- S3: Raw reviews source (shared across environments) ---
        self.REVIEWS_S3_BUCKET = os.getenv("REVIEWS_S3_BUCKET", "google-reviews-extract")
        self.REVIEWS_S3_PREFIX = os.getenv("REVIEWS_S3_PREFIX", "")

        # --- S3: KB structured reviews (per-environment) ---
        if self.is_prod:
            self.KB_S3_BUCKET = os.getenv("KB_S3_BUCKET_PROD", "abg-india-cdp-sandbox")
            self.KB_S3_PREFIX = os.getenv("KB_S3_PREFIX_PROD", "reviews/")
        else:
            self.KB_S3_BUCKET = os.getenv("KB_S3_BUCKET_DEV", "google-structured-reviews")
            self.KB_S3_PREFIX = os.getenv("KB_S3_PREFIX_DEV", "reviews/")

        # --- Bedrock ---
        self.AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
        self.BEDROCK_MODEL_ID = os.getenv(
            "BEDROCK_MODEL_ID", "anthropic.claude-sonnet-4-5-20250929-v1:0"
        )
        if self.is_prod:
            self.BEDROCK_KB_ID = os.getenv("BEDROCK_KB_ID_PROD", "JYLEB3IBMG")
        else:
            self.BEDROCK_KB_ID = os.getenv("BEDROCK_KB_ID_DEV", "4EJ0BSHUTO")

    def __repr__(self):
        return (
            f"Config(env={self.APP_ENV}, db={'postgres' if 'postgresql' in self.DATABASE_URL else 'sqlite'}, "
            f"kb_bucket={self.KB_S3_BUCKET}, kb_id={self.BEDROCK_KB_ID})"
        )


config = Config()
