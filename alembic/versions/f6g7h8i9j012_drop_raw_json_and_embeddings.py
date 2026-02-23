"""drop raw_json column and embeddings table

Drops the raw_json column from reviews (no longer needed — enricher reads
from parsed columns) and drops the embeddings table entirely (unused).

Revision ID: f6g7h8i9j012
Revises: e5f6a7b8c901
Create Date: 2026-03-01 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f6g7h8i9j012'
down_revision: str = 'e5f6a7b8c901'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # --- Drop raw_json column from reviews ---
    with op.batch_alter_table('reviews', schema=None) as batch_op:
        batch_op.drop_column('raw_json')

    # --- Drop embeddings table ---
    op.drop_table('embeddings')


def downgrade() -> None:
    # --- Re-create embeddings table ---
    op.create_table(
        'embeddings',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('review_id', sa.String(255), nullable=False),
        sa.Column('embedding_vector', sa.Text(), nullable=True),
        sa.Column('model_id', sa.String(100), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(
            ['review_id'],
            ['reviews.review_id'],
            name='fk_embeddings_review_id',
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('review_id'),
    )

    # --- Add raw_json column back to reviews ---
    with op.batch_alter_table('reviews', schema=None) as batch_op:
        batch_op.add_column(sa.Column('raw_json', sa.Text(), nullable=True))
