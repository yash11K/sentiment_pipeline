"""add brand and competitor columns

Revision ID: b3f7a2c8d901
Revises: a1806d1e6aca
Create Date: 2026-02-16 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b3f7a2c8d901'
down_revision: Union[str, Sequence[str], None] = 'a1806d1e6aca'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add brand and is_competitor to reviews
    op.add_column('reviews', sa.Column('brand', sa.String(length=100), nullable=True))
    op.add_column('reviews', sa.Column('is_competitor', sa.Boolean(), server_default='false', nullable=True))
    op.create_index('idx_reviews_brand', 'reviews', ['brand'], unique=False)
    op.create_index('idx_reviews_brand_location', 'reviews', ['brand', 'location_id'], unique=False)
    op.create_index('idx_reviews_competitor', 'reviews', ['is_competitor', 'location_id'], unique=False)

    # Add brand and is_competitor to ingestion_files
    op.add_column('ingestion_files', sa.Column('brand', sa.String(length=100), nullable=True))
    op.add_column('ingestion_files', sa.Column('is_competitor', sa.Boolean(), server_default='false', nullable=True))


def downgrade() -> None:
    op.drop_column('ingestion_files', 'is_competitor')
    op.drop_column('ingestion_files', 'brand')
    op.drop_index('idx_reviews_competitor', table_name='reviews')
    op.drop_index('idx_reviews_brand_location', table_name='reviews')
    op.drop_index('idx_reviews_brand', table_name='reviews')
    op.drop_column('reviews', 'is_competitor')
    op.drop_column('reviews', 'brand')
