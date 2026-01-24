"""
Main migration manager that orchestrates database migrations.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import re


class MigrationManager:
    """Orchestrate database migrations across all files
    
    Handles:
    - Detecting which files need migration
    - Running migration chains
    - Creating backups
    - Validating results
    - Rollback on failure
    """
    
    CURRENT_SCHEMA_VERSION = "1.1.0"  # Update when adding migrations
    
    def __init__(self, user_data_path: str = "user_data"):
        """Initialize migration manager
        
        Args:
            user_data_path: Path to user data directory
        """
        self.user_data_path = Path(user_data_path)
        self.migrations = self._load_migrations()
        self.migration_log_file = self.user_data_path / "migration_history.json"
    
    def _load_migrations(self) -> Dict[str, Any]:
        """Load all available migrations from migrations/ directory
        
        Returns:
            Dictionary mapping migration names to migration instances
        """
        migrations = {}
        
        # Import migrations dynamically
        try:
            from .migrations.v1_0_0_to_v1_1_0 import Migration_v1_0_0_to_v1_1_0
            migrations['1.0.0_to_1.1.0'] = Migration_v1_0_0_to_v1_1_0()
        except ImportError:
            pass  # Migration not yet created
        
        # Add more migrations as they're created
        # try:
        #     from .migrations.v1_1_0_to_v1_2_0 import Migration_v1_1_0_to_v1_2_0
        #     migrations['1.1.0_to_1.2.0'] = Migration_v1_1_0_to_v1_2_0()
        # except ImportError:
        #     pass
        
        return migrations
    
    def check_migrations_needed(self) -> Dict[str, List[str]]:
        """Scan all data files and return which need migration
        
        Returns:
            Dictionary mapping file paths to list of migrations needed
            Example: {
                'user_data/groups/test/transactions.json': ['1.0.0_to_1.1.0'],
                'user_data/groups/wedding/transactions.json': ['1.0.0_to_1.1.0']
            }
        """
        needs_migration = {}
        
        # Check if groups directory exists
        groups_path = self.user_data_path / "groups"
        if not groups_path.exists():
            return needs_migration
        
        # Check transactions.json in each group
        for group_dir in groups_path.iterdir():
            if group_dir.is_dir():
                trans_file = group_dir / "transactions.json"
                if trans_file.exists():
                    try:
                        current_version = self._get_file_version(trans_file)
                        if current_version != self.CURRENT_SCHEMA_VERSION:
                            migration_path = self._get_migration_path(
                                current_version, self.CURRENT_SCHEMA_VERSION
                            )
                            if migration_path:
                                needs_migration[str(trans_file)] = migration_path
                    except Exception as e:
                        print(f"Warning: Could not check version for {trans_file}: {e}")
        
        return needs_migration
    
    def run_all_migrations(self, show_progress: bool = True) -> Dict[str, bool]:
        """Run all needed migrations
        
        Args:
            show_progress: Whether to print progress messages
            
        Returns:
            Dictionary mapping file paths to success status
        """
        needed = self.check_migrations_needed()
        
        if not needed:
            if show_progress:
                print("✅ All data files are up to date!")
            return {}
        
        results = {}
        total_files = len(needed)
        
        if show_progress:
            print(f"🔄 Starting migration of {total_files} file(s)...")
            print()
        
        for idx, (file_path, migration_chain) in enumerate(needed.items(), 1):
            try:
                if show_progress:
                    print(f"[{idx}/{total_files}] Migrating {Path(file_path).name}...")
                    for migration_name in migration_chain:
                        migration = self.migrations.get(migration_name)
                        if migration:
                            print(f"  → {migration}")
                
                success = self._run_migration_chain(file_path, migration_chain, show_progress)
                results[file_path] = success
                
                if show_progress:
                    if success:
                        print(f"  ✅ Success!")
                    else:
                        print(f"  ❌ Failed!")
                    print()
                
            except Exception as e:
                results[file_path] = False
                if show_progress:
                    print(f"  ❌ ERROR: {e}")
                    print()
        
        # Log migration
        self._log_migration(needed, results)
        
        if show_progress:
            successful = sum(1 for v in results.values() if v)
            print(f"📊 Migration complete: {successful}/{total_files} files updated successfully")
        
        return results
    
    def _run_migration_chain(self, file_path: str, migrations: List[str], 
                           show_progress: bool = False) -> bool:
        """Run sequence of migrations on a file
        
        Args:
            file_path: Path to file to migrate
            migrations: List of migration names to run in sequence
            show_progress: Whether to show progress
            
        Returns:
            True if successful, False otherwise
        """
        file_path_obj = Path(file_path)
        
        # 1. Create backup
        backup_path = self._create_backup(file_path_obj, "pre_migration")
        
        try:
            # 2. Load data
            with open(file_path_obj, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            original_version = data.get('schema_version', '1.0.0')
            
            # 3. Run each migration in sequence
            for migration_name in migrations:
                migration = self.migrations.get(migration_name)
                if not migration:
                    raise Exception(f"Migration {migration_name} not found!")
                
                # Validate input
                if not migration.validate_input(data):
                    raise Exception(f"Data does not match expected format for {migration_name}")
                
                # Run migration
                data = migration.up(data)
                
                # Validate output
                if not migration.validate_output(data):
                    raise Exception(f"Migration {migration_name} produced invalid output")
            
            # 4. Validate final result
            self._validate_data(data, self.CURRENT_SCHEMA_VERSION)
            
            # 5. Save migrated data
            with open(file_path_obj, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            if show_progress:
                print(f"  📝 Updated schema: {original_version} → {data['schema_version']}")
            
            return True
            
        except Exception as e:
            # Rollback on error
            print(f"  ⚠️ Migration failed, rolling back: {e}")
            self._restore_backup(backup_path, file_path_obj)
            return False
    
    def _get_file_version(self, file_path: Path) -> str:
        """Get schema version from file
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Schema version string (defaults to "1.0.0" if not present)
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get('schema_version', '1.0.0')
        except Exception:
            return '1.0.0'
    
    def _get_migration_path(self, from_version: str, to_version: str) -> List[str]:
        """Determine which migrations to run
        
        Args:
            from_version: Starting version
            to_version: Target version
            
        Returns:
            List of migration names to run in order
        """
        # For now, simple direct mapping
        # Later can implement graph-based pathfinding for skipping versions
        key = f"{from_version}_to_{to_version}"
        if key in self.migrations:
            return [key]
        
        # Could implement multi-hop later:
        # e.g., 1.0.0 → 1.2.0 via 1.0.0 → 1.1.0 → 1.2.0
        return []
    
    def _create_backup(self, file_path: Path, reason: str = "backup") -> Path:
        """Create timestamped backup of file
        
        Args:
            file_path: File to backup
            reason: Reason for backup (included in filename)
            
        Returns:
            Path to backup file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_dir = file_path.parent / "backups"
        backup_dir.mkdir(exist_ok=True)
        
        backup_name = f"{file_path.stem}_{reason}_{timestamp}{file_path.suffix}"
        backup_path = backup_dir / backup_name
        
        shutil.copy2(file_path, backup_path)
        return backup_path
    
    def _restore_backup(self, backup_path: Path, original_path: Path):
        """Restore file from backup
        
        Args:
            backup_path: Path to backup file
            original_path: Path to restore to
        """
        if backup_path.exists():
            shutil.copy2(backup_path, original_path)
    
    def _validate_data(self, data: Dict[str, Any], expected_version: str):
        """Validate migrated data structure
        
        Args:
            data: Data to validate
            expected_version: Expected schema version
            
        Raises:
            Exception: If validation fails
        """
        # Check schema version
        if data.get('schema_version') != expected_version:
            raise Exception(f"Schema version mismatch: expected {expected_version}, got {data.get('schema_version')}")
        
        # Check basic structure
        if 'metadata' not in data:
            raise Exception("Missing 'metadata' field")
        
        if 'transactions' not in data:
            raise Exception("Missing 'transactions' field")
        
        # Validate transactions is a list
        if not isinstance(data['transactions'], list):
            raise Exception("'transactions' must be a list")
    
    def _log_migration(self, needed: Dict[str, List[str]], results: Dict[str, bool]):
        """Log migration to history file
        
        Args:
            needed: Migrations that were attempted
            results: Results of migrations
        """
        # Load existing history
        if self.migration_log_file.exists():
            with open(self.migration_log_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
        else:
            history = {'migrations': []}
        
        # Add new entry
        entry = {
            'timestamp': datetime.now().isoformat(),
            'files_attempted': list(needed.keys()),
            'files_successful': [f for f, success in results.items() if success],
            'files_failed': [f for f, success in results.items() if not success],
            'success_rate': f"{sum(results.values())}/{len(results)}",
            'target_version': self.CURRENT_SCHEMA_VERSION
        }
        
        history['migrations'].append(entry)
        
        # Save history
        self.user_data_path.mkdir(exist_ok=True)
        with open(self.migration_log_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    
    def get_migration_history(self) -> List[Dict[str, Any]]:
        """Get migration history
        
        Returns:
            List of migration log entries
        """
        if not self.migration_log_file.exists():
            return []
        
        with open(self.migration_log_file, 'r', encoding='utf-8') as f:
            history = json.load(f)
        
        return history.get('migrations', [])
