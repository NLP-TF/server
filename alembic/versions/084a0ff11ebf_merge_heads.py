"""merge heads

Revision ID: 084a0ff11ebf
Revises: add_situation_to_player_scores, dd5015f6dd74
Create Date: 2025-06-15 18:16:13.737635

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '084a0ff11ebf'
down_revision: Union[str, None] = ('add_situation_to_player_scores', 'dd5015f6dd74')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
