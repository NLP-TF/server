"""Add situation_detail to player_scores

Revision ID: 123456789abc
Revises: add_situation_to_player_scores
Create Date: 2025-06-15 19:15:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '123456789abc'
down_revision = 'add_situation_to_player_scores'
branch_labels = None
depends_on = None

def upgrade():
    # Add situation_detail column to player_scores table
    op.add_column('player_scores', 
                 sa.Column('situation_detail', sa.String(), nullable=True, comment='Detailed description of the situation'))
    
    # If you want to set a default value for existing rows:
    # op.execute("UPDATE player_scores SET situation_detail = '' WHERE situation_detail IS NULL")
    # Then make the column non-nullable:
    # op.alter_column('player_scores', 'situation_detail', nullable=False)

def downgrade():
    # Drop the situation_detail column
    op.drop_column('player_scores', 'situation_detail')
