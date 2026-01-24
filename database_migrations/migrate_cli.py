#!/usr/bin/env python3
"""
Command-line tool to manually run database migrations

Usage:
    python database_migrations/migrate_cli.py           # Check and run all migrations
    python database_migrations/migrate_cli.py --check   # Check only (dry run)
    python database_migrations/migrate_cli.py --history # Show migration history
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database_migrations.migration_manager import MigrationManager


def main():
    parser = argparse.ArgumentParser(
        description="Database Migration Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                 Run all pending migrations
  %(prog)s --check         Check what migrations are needed
  %(prog)s --history       View migration history
        """
    )
    
    parser.add_argument(
        '--check',
        action='store_true',
        help='Check what migrations are needed (dry run)'
    )
    
    parser.add_argument(
        '--history',
        action='store_true',
        help='Show migration history'
    )
    
    parser.add_argument(
        '--data-path',
        default='user_data',
        help='Path to user data directory (default: user_data)'
    )
    
    args = parser.parse_args()
    
    # Initialize migration manager
    print("🔍 Initializing migration manager...")
    migration_mgr = MigrationManager(user_data_path=args.data_path)
    print()
    
    # Show history
    if args.history:
        show_history(migration_mgr)
        return
    
    # Check migrations
    needs_migration = migration_mgr.check_migrations_needed()
    
    if not needs_migration:
        print("✅ All data files are up to date!")
        print(f"   Current schema version: {migration_mgr.CURRENT_SCHEMA_VERSION}")
        return
    
    # Show what needs migration
    total_files = len(needs_migration)
    print(f"📊 Found {total_files} file(s) that need migration:")
    print()
    
    for file_path, migrations in needs_migration.items():
        file_name = Path(file_path).parent.name
        print(f"  📁 {file_name}")
        for migration_name in migrations:
            migration_obj = migration_mgr.migrations.get(migration_name)
            if migration_obj:
                print(f"     → {migration_obj}")
        print()
    
    # Dry run?
    if args.check:
        print("ℹ️  This was a dry run. Use without --check to apply migrations.")
        return
    
    # Confirm
    print("⚠️  This will modify your data files (backups will be created)")
    response = input("Continue? [y/N]: ").strip().lower()
    
    if response != 'y':
        print("❌ Migration cancelled")
        return
    
    print()
    print("=" * 60)
    
    # Run migrations
    results = migration_mgr.run_all_migrations(show_progress=True)
    
    print("=" * 60)
    print()
    
    # Summary
    successful = sum(1 for v in results.values() if v)
    failed = total_files - successful
    
    if failed == 0:
        print(f"✅ Successfully migrated all {total_files} file(s)!")
        print(f"   New schema version: {migration_mgr.CURRENT_SCHEMA_VERSION}")
    else:
        print(f"⚠️  Migration completed with issues:")
        print(f"   ✅ Successful: {successful}/{total_files}")
        print(f"   ❌ Failed: {failed}/{total_files}")
        print()
        print("Failed files have been restored from backup.")


def show_history(migration_mgr):
    """Show migration history"""
    print("📜 Migration History")
    print("=" * 60)
    
    history = migration_mgr.get_migration_history()
    
    if not history:
        print("No migrations have been run yet.")
        return
    
    for idx, entry in enumerate(reversed(history), 1):
        print(f"\n{idx}. {entry['timestamp']}")
        print(f"   Target Version: {entry['target_version']}")
        print(f"   Success Rate: {entry['success_rate']}")
        
        if entry['files_successful']:
            print(f"   ✅ Successful: {len(entry['files_successful'])} files")
        
        if entry['files_failed']:
            print(f"   ❌ Failed: {len(entry['files_failed'])} files")
            for file in entry['files_failed']:
                print(f"      - {Path(file).parent.name}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Migration cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
