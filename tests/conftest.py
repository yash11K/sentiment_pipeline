"""
Shared test fixtures for the ingestion pipeline overhaul.

Provides:
- In-memory SQLite database with fresh schema per test
- Database instance wired to the in-memory engine
- Mock Bedrock client that returns deterministic enrichment data
- Hypothesis strategies for generating sample review data
"""
import json
import pytest
from datetime import datetime, date
from unittest.mock import MagicMock, patch

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from storage.models import Base, Review, Enrichment, Embedding, IngestionFile, Location
from storage.db import Database

import hypothesis.strategies as st
from hypothesis import settings

# ---------------------------------------------------------------------------
# Hypothesis global settings — minimum 100 examples per property test
# ---------------------------------------------------------------------------
settings.register_profile("ci", max_examples=100)
settings.load_profile("ci")

# ---------------------------------------------------------------------------
# Our brands constant (mirrors src/ingestion/s3_source.py)
# ---------------------------------------------------------------------------
OUR_BRANDS = {"avis", "budget", "payless", "apex", "maggiore"}


# ===================================================================
# Database fixtures
# ===================================================================

@pytest.fixture()
def db_engine():
    """Create an in-memory SQLite engine with FK enforcement."""
    engine = create_engine("sqlite:///:memory:")

    # SQLite needs explicit FK pragma per connection
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_conn, _connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(bind=engine)
    yield engine
    engine.dispose()


@pytest.fixture()
def db_session_factory(db_engine):
    """Session factory bound to the in-memory engine."""
    return sessionmaker(bind=db_engine, autocommit=False, autoflush=False)


@pytest.fixture()
def db(db_engine, db_session_factory):
    """A Database instance wired to the in-memory SQLite engine.

    Bypasses the normal Database.__init__ so no real DB URL is needed.
    """
    instance = object.__new__(Database)
    instance.database_url = "sqlite:///:memory:"
    instance.engine = db_engine
    instance.SessionLocal = db_session_factory
    return instance


# ===================================================================
# Mock Bedrock client
# ===================================================================

def _make_mock_enrichment(review_id: str) -> dict:
    """Return a deterministic enrichment dict for a given review_id."""
    return {
        "review_id": review_id,
        "topics": ["staff_behavior", "vehicle_condition"],
        "sentiment": "positive",
        "sentiment_score": 0.85,
        "entities": ["Preferred"],
        "key_phrases": ["friendly staff"],
        "urgency_level": "low",
        "actionable": False,
        "suggested_action": None,
    }


@pytest.fixture()
def mock_bedrock():
    """A MagicMock standing in for BedrockClient.

    `enrich_reviews_batch` returns deterministic enrichments keyed on review_id.
    """
    client = MagicMock()
    client.enrich_reviews_batch.side_effect = lambda reviews: [
        _make_mock_enrichment(r["review_id"]) for r in reviews
    ]
    client.invoke.return_value = "[]"
    client.retrieve.return_value = []
    return client


# ===================================================================
# Sample data helpers
# ===================================================================

def make_review_dict(
    review_id: str = "rev-001",
    location_id: str = "JFK",
    brand: str = "avis",
    rating: float = 4.0,
    text: str = "Great service at the counter.",
    reviewer: str = "Jane D.",
    relative_date: str = "2 weeks ago",
    is_competitor: bool = False,
    source: str = "google",
) -> dict:
    """Build a normalised review dict matching what ReviewParser produces."""
    raw = {
        "review_id": review_id,
        "rating": rating,
        "text": text,
        "reviewer": reviewer,
        "relative_date": relative_date,
        "reviewer_profile_url": "https://example.com/profile",
        "rating_label": "Rated 4 out of 5",
    }
    return {
        "location_id": location_id,
        "source": source,
        "brand": brand,
        "is_competitor": is_competitor,
        "review_id": review_id,
        "rating": rating,
        "review_text": text,
        "reviewer_name": reviewer,
        "reviewer_type": "standard",
        "relative_date": relative_date,
        "review_date": date(2025, 1, 1),
        "language": "en",
        "raw_json": json.dumps(raw),
    }


def make_raw_review_json(
    review_id: str = "rev-001",
    rating: float = 4.0,
    text: str = "Great service at the counter.",
    reviewer: str = "Jane D.",
    relative_date: str = "2 weeks ago",
    **extra_fields,
) -> dict:
    """Build a raw review dict as it appears inside an S3 JSON file."""
    d = {
        "review_id": review_id,
        "rating": rating,
        "text": text,
        "reviewer": reviewer,
        "relative_date": relative_date,
        "reviewer_profile_url": "https://example.com/profile",
        "rating_label": f"Rated {rating} out of 5",
    }
    d.update(extra_fields)
    return d


@pytest.fixture()
def sample_review():
    """A single normalised review dict ready for db.insert_review."""
    return make_review_dict()


@pytest.fixture()
def sample_s3_file_data():
    """A minimal S3 file payload with two reviews."""
    return {
        "data": {
            "reviews": [
                make_raw_review_json(review_id="rev-001", text="Good car, clean."),
                make_raw_review_json(review_id="rev-002", text="Terrible wait time."),
            ]
        }
    }


# ===================================================================
# Hypothesis strategies for review data generation
# ===================================================================

# Strategy: non-empty review text (printable, at least 1 char)
valid_review_text = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
    min_size=1,
).filter(lambda t: t.strip())

# Strategy: empty / whitespace-only text (should be filtered out)
empty_review_text = st.one_of(
    st.just(""),
    st.just(None),
    st.text(alphabet=" \t\n\r", min_size=0, max_size=10),
)

# Strategy: brand names (mix of our brands and competitors)
brand_strategy = st.one_of(
    st.sampled_from(sorted(OUR_BRANDS)),
    st.sampled_from(["hertz", "enterprise", "sixt", "national", "dollar"]),
)

# Strategy: ratings (1-5 scale, one decimal)
rating_strategy = st.floats(min_value=1.0, max_value=5.0).map(lambda f: round(f, 1))

# Strategy: IATA-style location codes
location_strategy = st.from_regex(r"[A-Z]{3}", fullmatch=True)

# Strategy: unique review IDs
review_id_strategy = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789-",
    min_size=5,
    max_size=30,
).filter(lambda s: len(s.strip()) >= 5)

# Strategy: relative date strings
relative_date_strategy = st.one_of(
    st.just("2 weeks ago"),
    st.just("3 days ago"),
    st.just("a month ago"),
    st.just("yesterday"),
    st.just("just now"),
    st.builds(
        lambda n, u: f"{n} {u}s ago",
        st.integers(min_value=1, max_value=12),
        st.sampled_from(["day", "week", "month", "year"]),
    ),
)


def raw_review_strategy(
    review_id=None, text=None, include_extra_fields=True
):
    """Hypothesis strategy that produces raw review dicts (S3 JSON shape)."""
    rid = review_id or review_id_strategy
    txt = text or valid_review_text
    base = st.fixed_dictionaries({
        "review_id": rid,
        "rating": rating_strategy,
        "text": txt,
        "reviewer": st.text(min_size=1, max_size=40),
        "relative_date": relative_date_strategy,
    })
    if include_extra_fields:
        extras = st.fixed_dictionaries({
            "reviewer_profile_url": st.just("https://example.com/profile"),
            "rating_label": st.just("Rated 4 out of 5"),
        })
        return st.tuples(base, extras).map(lambda t: {**t[0], **t[1]})
    return base
