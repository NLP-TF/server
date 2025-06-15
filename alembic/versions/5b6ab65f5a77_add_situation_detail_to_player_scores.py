"""add_situation_detail_to_player_scores

Revision ID: 5b6ab65f5a77
Revises: 084a0ff11ebf
Create Date: 2025-06-15 18:25:48.485393

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '5b6ab65f5a77'
down_revision: Union[str, None] = '084a0ff11ebf'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
