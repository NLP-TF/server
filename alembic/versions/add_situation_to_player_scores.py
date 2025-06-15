"""Add situation to player_scores

Revision ID: add_situation_to_player_scores
Revises: 
Create Date: 2025-06-15 18:10:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'add_situation_to_player_scores'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    # Add the situation column to player_scores table
    op.add_column('player_scores', 
                 sa.Column('situation', sa.String(), nullable=False, server_default='친구_갈등'))
    
    # Update existing rows to have a default situation
    # This is necessary because we set nullable=False
    op.execute("UPDATE player_scores SET situation = '친구_갈등' WHERE situation IS NULL")

def downgrade():
    # Drop the situation column
    op.drop_column('player_scores', 'situation')
