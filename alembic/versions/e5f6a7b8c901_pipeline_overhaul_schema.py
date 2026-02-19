"""pipeline overhaul schema

Adds CASCADE DELETE on enrichments/embeddings FKs, sets brand NOT NULL
on reviews and ingestion_files, converts reviews.review_date from String to Date.

Revision ID: e5f6a7b8c901
Revises: c4d8e5f6a012
Create Date: 2026-02-20 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e5f6a7b8c901'
down_revision: str = 'c4d8e5f6a012'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # --- Reviews table: convert review_date from String(50) to Date ---
    # --- Reviews table: set brand to NOT NULL ---
    with op.batch_alter_table('reviews', schema=None) as batch_op:
        batch_op.alter_column(
            'review_date',
            existing_type=sa.String(50),
            type_=sa.Date(),
            existing_nullable=True,
            postgresql_using='review_date::date'
        )
        batch_op.alter_column(
            'brand',
            existing_type=sa.String(100),
            nullable=False
        )

    # --- Ingestion files table: set brand to NOT NULL ---
    with op.batch_alter_table('ingestion_files', schema=None) as batch_op:
        batch_op.alter_column(
            'brand',
            existing_type=sa.String(100),
            nullable=False
        )

    # --- Enrichments table: recreate FK with CASCADE DELETE ---
    # Look up actual FK constraint name from the database
    conn = op.get_bind()
    insp = sa.inspect(conn)

    enrichment_fks = insp.get_foreign_keys('enrichments')
    enrichment_fk_name = None
    for fk in enrichment_fks:
        if fk['referred_table'] == 'reviews' and 'review_id' in fk['constrained_columns']:
            enrichment_fk_name = fk['name']
            break

    if enrichment_fk_name:
        with op.batch_alter_table('enrichments', schema=None) as batch_op:
            batch_op.drop_constraint(enrichment_fk_name, type_='foreignkey')
            batch_op.create_foreign_key(
                'fk_enrichments_review_id',
                'reviews',
                ['review_id'],
                ['review_id'],
                ondelete='CASCADE'
            )
    else:
        # No existing FK — just create it (e.g. fresh table after truncate)
        with op.batch_alter_table('enrichments', schema=None) as batch_op:
            batch_op.create_foreign_key(
                'fk_enrichments_review_id',
                'reviews',
                ['review_id'],
                ['review_id'],
                ondelete='CASCADE'
            )

    # --- Embeddings table: recreate FK with CASCADE DELETE ---
    embedding_fks = insp.get_foreign_keys('embeddings')
    embedding_fk_name = None
    for fk in embedding_fks:
        if fk['referred_table'] == 'reviews' and 'review_id' in fk['constrained_columns']:
            embedding_fk_name = fk['name']
            break

    if embedding_fk_name:
        with op.batch_alter_table('embeddings', schema=None) as batch_op:
            batch_op.drop_constraint(embedding_fk_name, type_='foreignkey')
            batch_op.create_foreign_key(
                'fk_embeddings_review_id',
                'reviews',
                ['review_id'],
                ['review_id'],
                ondelete='CASCADE'
            )
    else:
        # No existing FK — just create it
        with op.batch_alter_table('embeddings', schema=None) as batch_op:
            batch_op.create_foreign_key(
                'fk_embeddings_review_id',
                'reviews',
                ['review_id'],
                ['review_id'],
                ondelete='CASCADE'
            )


def downgrade() -> None:
    # --- Embeddings: revert FK to no CASCADE ---
    with op.batch_alter_table('embeddings', schema=None) as batch_op:
        batch_op.drop_constraint(
            'fk_embeddings_review_id',
            type_='foreignkey'
        )
        batch_op.create_foreign_key(
            'fk_embeddings_review_id',
            'reviews',
            ['review_id'],
            ['review_id']
        )

    # --- Enrichments: revert FK to no CASCADE ---
    with op.batch_alter_table('enrichments', schema=None) as batch_op:
        batch_op.drop_constraint(
            'fk_enrichments_review_id',
            type_='foreignkey'
        )
        batch_op.create_foreign_key(
            'fk_enrichments_review_id',
            'reviews',
            ['review_id'],
            ['review_id']
        )

    # --- Ingestion files: revert brand to nullable ---
    with op.batch_alter_table('ingestion_files', schema=None) as batch_op:
        batch_op.alter_column(
            'brand',
            existing_type=sa.String(100),
            nullable=True
        )

    # --- Reviews: revert review_date to String(50), brand to nullable ---
    with op.batch_alter_table('reviews', schema=None) as batch_op:
        batch_op.alter_column(
            'brand',
            existing_type=sa.String(100),
            nullable=True
        )
        batch_op.alter_column(
            'review_date',
            existing_type=sa.Date(),
            type_=sa.String(50),
            existing_nullable=True
        )
