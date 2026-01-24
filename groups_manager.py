import json
import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import uuid


class GroupsManager:
    """Manage expense groups with separate data directories per group"""
    
    def __init__(self):
        self.groups_base_path = Path("user_data/groups")
        self.groups_config_path = self.groups_base_path / "groups_config.json"
        self._ensure_groups_structure()
    
    def _ensure_groups_structure(self):
        """Create groups directory structure if not exists"""
        self.groups_base_path.mkdir(parents=True, exist_ok=True)
        
        # Create config if doesn't exist
        if not self.groups_config_path.exists():
            default_config = {
                "groups": [],
                "active_group_id": None,
                "settings": {
                    "show_group_selector_in_sidebar": True,
                    "remember_last_group": True,
                    "default_group_id": None
                }
            }
            self._save_config(default_config)
    
    def _load_config(self) -> Dict:
        """Load groups configuration"""
        try:
            with open(self.groups_config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # Return default config if file doesn't exist or is corrupted
            return {
                "groups": [],
                "active_group_id": None,
                "settings": {
                    "show_group_selector_in_sidebar": True,
                    "remember_last_group": True,
                    "default_group_id": None
                }
            }
    
    def _save_config(self, config: Dict):
        """Save groups configuration"""
        with open(self.groups_config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    
    def _sanitize_directory_name(self, name: str) -> str:
        """Sanitize group name for use as directory name"""
        # Replace problematic characters but keep emojis
        sanitized = name.replace('/', '_').replace('\\', '_').replace(':', '_')
        # Ensure it's not empty
        if not sanitized.strip():
            sanitized = f"group_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return sanitized
    
    def get_all_groups(self, include_archived: bool = False) -> List[Dict]:
        """Get list of all groups"""
        config = self._load_config()
        groups = config.get('groups', [])
        
        if not include_archived:
            groups = [g for g in groups if not g.get('is_archived', False)]
        
        return groups
    
    def has_groups(self) -> bool:
        """Check if any groups exist"""
        return len(self.get_all_groups()) > 0
    
    def get_group_by_id(self, group_id: str) -> Optional[Dict]:
        """Get specific group by ID"""
        groups = self.get_all_groups(include_archived=True)
        for group in groups:
            if group['id'] == group_id:
                return group
        return None
    
    def get_active_group(self) -> Optional[Dict]:
        """Get currently active group"""
        config = self._load_config()
        active_id = config.get('active_group_id')
        
        if active_id:
            group = self.get_group_by_id(active_id)
            if group and not group.get('is_archived', False):
                return group
        
        # Fallback to first non-archived group
        groups = self.get_all_groups()
        if groups:
            # Update active group in config
            self.set_active_group(groups[0]['id'])
            return groups[0]
        
        return None
    
    def set_active_group(self, group_id: str) -> bool:
        """Set active group"""
        group = self.get_group_by_id(group_id)
        if not group or group.get('is_archived', False):
            return False
        
        config = self._load_config()
        config['active_group_id'] = group_id
        
        # Update last accessed time
        for g in config['groups']:
            if g['id'] == group_id:
                g['last_accessed'] = datetime.now().isoformat()
                break
        
        self._save_config(config)
        return True
    
    def create_group(self, name: str, description: str = "", emoji: str = "📁", 
                    members: List[str] = None, color: str = "#4CAF50") -> Dict:
        """Create new expense group"""
        config = self._load_config()
        
        # Check for duplicate names
        existing_names = [g['name'] for g in config.get('groups', [])]
        if name in existing_names:
            raise ValueError(f"Group with name '{name}' already exists. Please choose a different name.")
        
        # Generate unique ID
        group_id = f"group_{uuid.uuid4().hex[:8]}"
        
        # Sanitize directory name
        dir_name = self._sanitize_directory_name(name)
        
        # Ensure directory name is unique
        base_dir_name = dir_name
        counter = 1
        while (self.groups_base_path / dir_name).exists():
            dir_name = f"{base_dir_name}_{counter}"
            counter += 1
        
        # Create group object
        group = {
            "id": group_id,
            "name": name,
            "emoji": emoji,
            "description": description,
            "members": members or [],
            "created_at": datetime.now().isoformat(),
            "last_accessed": datetime.now().isoformat(),
            "color": color,
            "is_archived": False,
            "data_directory": dir_name
        }
        
        # Create group directory structure
        group_path = self.groups_base_path / dir_name
        group_path.mkdir(parents=True, exist_ok=True)
        (group_path / "cache").mkdir(exist_ok=True)
        (group_path / "backups").mkdir(exist_ok=True)
        
        # Initialize empty transactions file
        transactions_file = group_path / "transactions.json"
        if not transactions_file.exists():
            empty_data = {
                "metadata": {
                    "version": "1.0",
                    "last_updated": datetime.now().isoformat(),
                    "total_transactions": 0,
                    "date_range": {
                        "earliest": None,
                        "latest": None
                    },
                    "currencies": [],
                    "categories": []
                },
                "transactions": []
            }
            with open(transactions_file, 'w', encoding='utf-8') as f:
                json.dump(empty_data, f, ensure_ascii=False, indent=2)
        
        # Add to config
        config['groups'].append(group)
        
        # Set as active if it's the first group
        if len(config['groups']) == 1:
            config['active_group_id'] = group_id
            config['settings']['default_group_id'] = group_id
        
        self._save_config(config)
        
        return group
    
    def update_group(self, group_id: str, **kwargs) -> bool:
        """Update group metadata"""
        config = self._load_config()
        
        for group in config['groups']:
            if group['id'] == group_id:
                # Update allowed fields
                allowed_fields = ['name', 'emoji', 'description', 'members', 'color', 
                                'last_accessed', 'is_archived']
                for key, value in kwargs.items():
                    if key in allowed_fields:
                        group[key] = value
                
                self._save_config(config)
                return True
        
        return False
    
    def archive_group(self, group_id: str) -> bool:
        """Archive group (hide from main list but keep data)"""
        return self.update_group(group_id, is_archived=True)
    
    def delete_group(self, group_id: str, delete_data: bool = False) -> bool:
        """Delete group (optionally delete data)"""
        config = self._load_config()
        group = self.get_group_by_id(group_id)
        
        if not group:
            return False
        
        # Remove from config
        config['groups'] = [g for g in config['groups'] if g['id'] != group_id]
        
        # Update active group if this was active
        if config['active_group_id'] == group_id:
            remaining_groups = [g for g in config['groups'] if not g.get('is_archived', False)]
            config['active_group_id'] = remaining_groups[0]['id'] if remaining_groups else None
        
        self._save_config(config)
        
        # Delete data directory if requested
        if delete_data:
            group_path = self.groups_base_path / group['data_directory']
            if group_path.exists():
                shutil.rmtree(group_path)
        
        return True
    
    def get_group_data_path(self, group_id: str) -> Optional[str]:
        """Get path to group's data directory"""
        group = self.get_group_by_id(group_id)
        if not group:
            return None
        
        return str(self.groups_base_path / group['data_directory'])
    
    def get_group_members(self, group_id: str) -> List[str]:
        """Get list of members in a group"""
        group = self.get_group_by_id(group_id)
        if not group:
            return []
        
        return group.get('members', [])
    
    def get_all_members_across_groups(self, group_ids: List[str]) -> List[str]:
        """Get unique members across multiple groups"""
        all_members = set()
        
        for group_id in group_ids:
            members = self.get_group_members(group_id)
            all_members.update(members)
        
        return sorted(list(all_members))
    
    def migrate_existing_data(self, default_group_name: str = "דירה 🏡", 
                            default_description: str = "Existing expenses") -> bool:
        """
        Migrate current transactions.json to first group
        Returns True if migration was performed, False if already migrated
        """
        # Check if already migrated
        if self.groups_config_path.exists():
            config = self._load_config()
            if config.get('groups'):
                return False  # Already migrated
        
        old_data_path = Path("user_data")
        old_transactions = old_data_path / "transactions.json"
        old_income = old_data_path / "income_data.json"
        old_cache = old_data_path / "cache"
        old_backups = old_data_path / "backups"
        
        # Create default group
        group = self.create_group(
            name=default_group_name,
            description=default_description,
            emoji="🏡",
            members=[]
        )
        
        group_path = self.groups_base_path / group['data_directory']
        
        # Move existing files to new group directory
        if old_transactions.exists():
            shutil.move(str(old_transactions), str(group_path / "transactions.json"))
        
        if old_income.exists():
            shutil.move(str(old_income), str(group_path / "income_data.json"))
        
        if old_cache.exists() and old_cache.is_dir():
            # Remove default cache that was created
            if (group_path / "cache").exists():
                shutil.rmtree(group_path / "cache")
            shutil.move(str(old_cache), str(group_path / "cache"))
        
        if old_backups.exists() and old_backups.is_dir():
            # Remove default backups that was created
            if (group_path / "backups").exists():
                shutil.rmtree(group_path / "backups")
            shutil.move(str(old_backups), str(group_path / "backups"))
        
        return True
    
    def get_group_statistics(self, group_id: str) -> Dict:
        """Get statistics for a specific group"""
        from data_manager import DataManager
        
        group = self.get_group_by_id(group_id)
        if not group:
            return {}
        
        group_path = self.get_group_data_path(group_id)
        dm = DataManager(group_path)
        
        stats = {
            'group_name': group['name'],
            'group_id': group_id,
            'transaction_count': 0,
            'total_spending': 0,
            'date_range': {
                'earliest': None,
                'latest': None
            },
            'categories': [],
            'members': group.get('members', [])
        }
        
        try:
            data = dm.load_data()
            stats['transaction_count'] = len(data.get('transactions', []))
            
            if data.get('transactions'):
                df = dm.get_dataframe()
                if not df.empty:
                    stats['total_spending'] = float(df['Cost'].sum())
                    stats['date_range']['earliest'] = df['Date'].min().isoformat()
                    stats['date_range']['latest'] = df['Date'].max().isoformat()
                    stats['categories'] = df['Category'].unique().tolist()
        except Exception:
            pass  # Return default stats if data loading fails
        
        return stats
