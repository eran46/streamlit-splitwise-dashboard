"""
Migration registry
Import all migrations here to make them available to the MigrationManager
"""

# Import migrations as they're created
from .v1_0_0_to_v1_1_0 import Migration_v1_0_0_to_v1_1_0
# from .v1_1_0_to_v1_2_0 import Migration_v1_1_0_to_v1_2_0

# Registry of all available migrations
MIGRATION_REGISTRY = {
    '1.0.0_to_1.1.0': Migration_v1_0_0_to_v1_1_0(),
    # '1.1.0_to_1.2.0': Migration_v1_1_0_to_v1_2_0(),
}

__all__ = ['MIGRATION_REGISTRY']
