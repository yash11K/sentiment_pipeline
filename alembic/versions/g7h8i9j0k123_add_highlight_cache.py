"""add highlight_cache table

Revision ID: g7h8i9j0k123
Revises: f6g7h8i9j012
Create Date: 2026-02-23 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'g7h8i9j0k123'
down_revision = 'f6g7h8i9j012'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'highlight_cache',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('location_id', sa.String(10), nullable=False, index=True),
        sa.Column('brand', sa.String(100), nullable=True, index=True),
        sa.Column('analysis', sa.Text(), nullable=False),
        sa.Column('severity', sa.String(20), nullable=False),
        sa.Column('followup_questions', sa.Text(), nullable=True),
        sa.Column('citations', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index('idx_highlight_location_brand', 'highlight_cache', ['location_id', 'brand'])


def downgrade() -> None:
    op.drop_index('idx_highlight_location_brand', table_name='highlight_cache')
    op.drop_table('highlight_cache')
