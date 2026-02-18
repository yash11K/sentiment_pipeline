"""add brands to locations

Revision ID: c4d8e5f6a012
Revises: b3f7a2c8d901
Create Date: 2026-02-18 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c4d8e5f6a012'
down_revision: str = 'b3f7a2c8d901'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('locations', sa.Column('brands', sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column('locations', 'brands')
