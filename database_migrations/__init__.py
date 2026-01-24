"""
Database Migration System for ExpenseInfo

Provides automated schema migration capabilities to update existing data
structures without data loss.

Usage:
    from database_migrations import MigrationManager
    
    manager = MigrationManager()
    if manager.check_migrations_needed():
        manager.run_all_migrations()
"""

from .migration_manager import MigrationManager
from .migration_base import BaseMigration

__version__ = "1.0.0"
__all__ = ['MigrationManager', 'BaseMigration']
