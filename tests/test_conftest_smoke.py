"""
Smoke tests to verify conftest fixtures work correctly.
These validate the test infrastructure before real tests are written.
"""
import json
import pytest
from datetime import datetime

from storage.models import Review, Enrichment, Embedding, IngestionFile, Location


class TestDatabaseFixture:
    """Verify the in-memory DB fixture creates tables and supports CRUD."""

    def test_db_engine_creates_tables(self, db_engine):
        """All ORM tables should exist in the in-memory engine."""
        table_names = db_engine.table_names() if hasattr(db_engine, 'table_names') else []
        from sqlalchemy import inspect
        inspector = inspect(db_engine)
        table_names = inspector.get_table_names()
        assert "reviews" in table_names
        assert "enrichments" in table_names
        assert "embeddings" in table_names
        assert "ingestion_files" in table_names
        assert "locations" in table_names

    def test_db_insert_and_read_review(self, db):
        """Database fixture should support insert_review and get_reviews."""
        from tests.conftest import make_review_dict
        review = make_review_dict(review_id="smoke-001", brand="avis")
        row_id, is_new = db.insert_review(review)
        assert row_id is not None
        assert isinstance(row_id, int)
        assert is_new is True

        reviews = db.get_reviews(location_id="JFK")
        assert len(reviews) == 1
        assert reviews[0]["review_id"] == "smoke-001"

    def test_db_session_isolation(self, db):
        """Each test should get a fresh database (no leftover data)."""
        reviews = db.get_reviews()
        assert len(reviews) == 0

    def test_foreign_key_enforcement(self, db_engine, db_session_factory):
        """SQLite FK pragma should be enabled — inserting orphan enrichment fails."""
        session = db_session_factory()
        try:
            enrichment = Enrichment(
                review_id="nonexistent-review",
                sentiment="positive",
            )
            session.add(enrichment)
            with pytest.raises(Exception):
                session.flush()
        finally:
            session.rollback()
            session.close()


class TestMockBedrock:
    """Verify the mock Bedrock client returns expected data."""

    def test_enrich_reviews_batch(self, mock_bedrock):
        reviews = [{"review_id": "r1"}, {"review_id": "r2"}]
        results = mock_bedrock.enrich_reviews_batch(reviews)
        assert len(results) == 2
        assert results[0]["review_id"] == "r1"
        assert results[1]["review_id"] == "r2"
        assert results[0]["sentiment"] == "positive"
        assert "topics" in results[0]

    def test_invoke_returns_string(self, mock_bedrock):
        assert mock_bedrock.invoke() == "[]"

    def test_retrieve_returns_list(self, mock_bedrock):
        assert mock_bedrock.retrieve() == []


class TestSampleDataHelpers:
    """Verify the helper functions produce valid data."""

    def test_make_review_dict_defaults(self):
        from tests.conftest import make_review_dict
        review = make_review_dict()
        assert review["review_id"] == "rev-001"
        assert review["brand"] == "avis"
        assert review["rating"] == 4.0
        assert review["review_text"] == "Great service at the counter."
        assert "raw_json" in review
        raw = json.loads(review["raw_json"])
        assert "review_id" in raw

    def test_make_review_dict_custom(self):
        from tests.conftest import make_review_dict
        review = make_review_dict(
            review_id="custom-99", brand="hertz", rating=2.0,
            text="Bad experience.", is_competitor=True
        )
        assert review["review_id"] == "custom-99"
        assert review["brand"] == "hertz"
        assert review["is_competitor"] is True

    def test_make_raw_review_json(self):
        from tests.conftest import make_raw_review_json
        raw = make_raw_review_json(review_id="raw-01", text="Nice car.")
        assert raw["review_id"] == "raw-01"
        assert raw["text"] == "Nice car."
        assert "reviewer_profile_url" in raw
        assert "rating_label" in raw

    def test_sample_review_fixture(self, sample_review):
        assert sample_review["review_id"] == "rev-001"
        assert sample_review["location_id"] == "JFK"

    def test_sample_s3_file_data_fixture(self, sample_s3_file_data):
        reviews = sample_s3_file_data["data"]["reviews"]
        assert len(reviews) == 2
        assert reviews[0]["review_id"] == "rev-001"
        assert reviews[1]["review_id"] == "rev-002"


class TestHypothesisStrategies:
    """Quick sanity checks that hypothesis strategies produce valid data."""

    def test_valid_review_text_strategy(self):
        from tests.conftest import valid_review_text
        from hypothesis import given, settings as h_settings
        examples = []

        @h_settings(max_examples=10)
        @given(valid_review_text)
        def _check(text):
            assert text is not None
            assert text.strip() != ""
            examples.append(text)

        _check()
        assert len(examples) > 0

    def test_raw_review_strategy(self):
        from tests.conftest import raw_review_strategy
        from hypothesis import given, settings as h_settings

        @h_settings(max_examples=10)
        @given(raw_review_strategy())
        def _check(review):
            assert "review_id" in review
            assert "rating" in review
            assert "text" in review
            assert "reviewer" in review
            assert "relative_date" in review
            # Extra fields should be present
            assert "reviewer_profile_url" in review

        _check()
