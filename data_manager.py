import json
import hashlib
import pickle
import fcntl
import pandas as pd
from pathlib import Path
from datetime import datetime
import shutil


class DataManager:
    """Handle all data persistence operations with concurrent access support"""
    
    def __init__(self, data_dir="user_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.data_file = self.data_dir / "transactions.json"
        self.cache_dir = self.data_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.backup_dir = self.data_dir / "backups"
        self.backup_dir.mkdir(exist_ok=True)
    
    def _create_empty_dataset(self):
        """Create an empty dataset structure"""
        return {
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
    
    def load_data(self):
        """Load persisted data with file locking"""
        if not self.data_file.exists():
            return self._create_empty_dataset()
        
        with open(self.data_file, 'r', encoding='utf-8') as f:
            fcntl.flock(f, fcntl.LOCK_SH)  # Shared lock for reading
            try:
                data = json.load(f)
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
        
        return data
    
    def save_data(self, data):
        """Save data with automatic backup and file locking"""
        # Create backup first
        if self.data_file.exists():
            self._create_backup()
        
        # Update metadata
        data['metadata']['last_updated'] = datetime.now().isoformat()
        
        # Save new data with exclusive lock
        with open(self.data_file, 'w', encoding='utf-8') as f:
            fcntl.flock(f, fcntl.LOCK_EX)  # Exclusive lock for writing
            try:
                json.dump(data, f, ensure_ascii=False, indent=2)
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
        
        # Clear cache
        self._clear_cache()
    
    def get_dataframe(self):
        """Get data as pandas DataFrame with caching"""
        cache_file = self.cache_dir / "dataframe.pkl"
        
        # Check if cache is valid
        if cache_file.exists() and self.data_file.exists():
            cache_mtime = cache_file.stat().st_mtime
            data_mtime = self.data_file.stat().st_mtime
            
            if cache_mtime > data_mtime:
                return pd.read_pickle(cache_file)
        
        # Load and cache
        data = self.load_data()
        
        if not data['transactions']:
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=['Date', 'Description', 'Category', 'Cost', 'Currency'])
        
        df = pd.DataFrame(data['transactions'])
        
        # Process dates
        df['Date'] = pd.to_datetime(df['date'])
        df['Description'] = df['description']
        df['Category'] = df['category']
        df['Cost'] = pd.to_numeric(df['cost'], errors='coerce')
        df['Currency'] = df['currency']
        
        # Cache it
        df.to_pickle(cache_file)
        
        return df
    
    def append_transactions(self, new_transactions):
        """Add new transactions with duplicate detection"""
        data = self.load_data()
        
        added = 0
        skipped = 0
        duplicates = []
        
        for txn in new_transactions:
            # Validate transaction first
            try:
                self._validate_transaction(txn)
            except ValueError as e:
                # Skip invalid transactions
                skipped += 1
                continue
            
            # Generate hash for duplicate detection
            txn_hash = self._generate_hash(txn)
            txn['hash'] = txn_hash
            
            # Check for duplicates
            if self._is_duplicate(txn_hash, data['transactions']):
                skipped += 1
                duplicates.append(txn)
                continue
            
            # Add unique transaction
            txn['id'] = self._generate_id()
            txn['created_at'] = datetime.now().isoformat()
            data['transactions'].append(txn)
            added += 1
        
        # Update metadata
        data['metadata']['total_transactions'] = len(data['transactions'])
        self._update_metadata(data)
        
        # Save
        self.save_data(data)
        
        return {
            'added': added,
            'skipped': skipped,
            'duplicates': duplicates
        }
    
    def _validate_transaction(self, txn):
        """Ensure data quality"""
        required = ['date', 'description', 'cost', 'category']
        
        for field in required:
            if field not in txn or txn[field] is None or txn[field] == '':
                raise ValueError(f"Missing required field: {field}")
        
        # Validate date format
        try:
            if isinstance(txn['date'], str):
                datetime.fromisoformat(txn['date'])
        except (ValueError, TypeError):
            raise ValueError(f"Invalid date format: {txn['date']}")
        
        # Validate amount
        try:
            cost = float(txn['cost'])
            if cost < 0:
                raise ValueError(f"Invalid cost: {txn['cost']}")
        except (ValueError, TypeError):
            raise ValueError(f"Invalid cost: {txn['cost']}")
        
        return True
    
    def _generate_hash(self, transaction):
        """Generate unique hash for transaction"""
        # Use date + description + amount + category
        date_str = str(transaction.get('date', ''))
        desc = str(transaction.get('description', ''))
        cost = str(transaction.get('cost', ''))
        category = str(transaction.get('category', ''))
        
        key = f"{date_str}_{desc}_{cost}_{category}"
        return hashlib.md5(key.encode()).hexdigest()
    
    def _is_duplicate(self, txn_hash, existing_transactions):
        """Check if transaction already exists"""
        return any(txn.get('hash') == txn_hash for txn in existing_transactions)
    
    def _generate_id(self):
        """Generate unique transaction ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        random_suffix = hashlib.md5(str(datetime.now().timestamp()).encode()).hexdigest()[:6]
        return f"txn_{timestamp}_{random_suffix}"
    
    def _update_metadata(self, data):
        """Update metadata after adding transactions"""
        if not data['transactions']:
            return
        
        # Update date range
        dates = [txn['date'] for txn in data['transactions'] if 'date' in txn]
        if dates:
            data['metadata']['date_range']['earliest'] = min(dates)
            data['metadata']['date_range']['latest'] = max(dates)
        
        # Update currencies
        currencies = set(txn.get('currency', 'ILS') for txn in data['transactions'])
        data['metadata']['currencies'] = sorted(list(currencies))
        
        # Update categories
        categories = set(txn.get('category', '') for txn in data['transactions'] if txn.get('category'))
        data['metadata']['categories'] = sorted(list(categories))
    
    def _create_backup(self):
        """Create backup of current data"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_file = self.backup_dir / f"transactions_backup_{timestamp}.json"
        
        # Copy current file
        shutil.copy(self.data_file, backup_file)
        
        # Maintain only N most recent backups
        self._cleanup_old_backups(keep=5)
    
    def _cleanup_old_backups(self, keep=5):
        """Keep only N most recent backups"""
        backups = sorted(self.backup_dir.glob("transactions_backup_*.json"))
        if len(backups) > keep:
            for old_backup in backups[:-keep]:
                old_backup.unlink()
    
    def _clear_cache(self):
        """Clear all cache files"""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
    
    def get_backups(self):
        """Get list of available backups"""
        backups = sorted(self.backup_dir.glob("transactions_backup_*.json"), reverse=True)
        backup_info = []
        
        for backup_file in backups:
            stat = backup_file.stat()
            backup_info.append({
                'file': backup_file.name,
                'path': backup_file,
                'date': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d'),
                'time': datetime.fromtimestamp(stat.st_mtime).strftime('%H:%M:%S'),
                'size': f"{stat.st_size / 1024:.1f} KB"
            })
        
        return backup_info
    
    def restore_backup(self, backup_path):
        """Restore data from a backup"""
        backup_path = Path(backup_path)
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_path}")
        
        # Create a backup of current state before restoring
        if self.data_file.exists():
            self._create_backup()
        
        # Copy backup to main file
        shutil.copy(backup_path, self.data_file)
        
        # Clear cache
        self._clear_cache()
    
    def clear_all_data(self):
        """Clear all data and reset to empty state"""
        # Create final backup
        if self.data_file.exists():
            self._create_backup()
        
        # Remove data file
        if self.data_file.exists():
            self.data_file.unlink()
        
        # Clear cache
        self._clear_cache()
    
    def export_to_dict(self):
        """Export all data as dictionary"""
        return self.load_data()
    
    def data_exists(self):
        """Check if data file exists"""
        return self.data_file.exists() and self.data_file.stat().st_size > 0
