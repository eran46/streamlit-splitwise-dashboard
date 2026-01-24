"""
Transaction Metadata Manager
Handles notes and tags for transactions
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Set
from datetime import datetime


class TransactionMetadataManager:
    """Manage notes and tags for transactions"""
    
    def __init__(self, data_dir="user_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.metadata_file = self.data_dir / "transaction_metadata.json"
        self._cache = None
    
    @staticmethod
    def create_transaction_id(date: str, description: str, cost: float) -> str:
        """Create unique identifier for a transaction
        
        Args:
            date: Transaction date (ISO format or datetime string)
            description: Transaction description
            cost: Transaction cost
            
        Returns:
            MD5 hash as transaction identifier
        """
        # Normalize date to just date part (ignore time)
        if 'T' in str(date):
            date = str(date).split('T')[0]
        
        identifier = f"{date}_{description}_{cost}"
        return hashlib.md5(identifier.encode()).hexdigest()
    
    def load_metadata(self) -> Dict:
        """Load all transaction metadata from file
        
        Returns:
            Dictionary mapping transaction IDs to metadata
        """
        if self._cache is not None:
            return self._cache
        
        if not self.metadata_file.exists():
            self._cache = {}
            return self._cache
        
        try:
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                self._cache = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            self._cache = {}
        
        return self._cache
    
    def save_metadata(self, metadata: Dict) -> None:
        """Save transaction metadata to file
        
        Args:
            metadata: Dictionary of transaction metadata
        """
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        # Update cache
        self._cache = metadata
    
    def get_transaction_metadata(self, transaction_id: str) -> Dict:
        """Get metadata for a specific transaction
        
        Args:
            transaction_id: Transaction identifier
            
        Returns:
            Dictionary with 'notes' and 'tags' keys
        """
        metadata = self.load_metadata()
        
        if transaction_id not in metadata:
            return {
                'notes': '',
                'tags': [],
                'date_modified': None
            }
        
        return metadata[transaction_id]
    
    def update_transaction_metadata(self, transaction_id: str, notes: Optional[str] = None, 
                                    tags: Optional[List[str]] = None) -> None:
        """Update metadata for a transaction
        
        Args:
            transaction_id: Transaction identifier
            notes: New notes (if provided)
            tags: New tags list (if provided)
        """
        metadata = self.load_metadata()
        
        if transaction_id not in metadata:
            metadata[transaction_id] = {
                'notes': '',
                'tags': [],
                'date_modified': None
            }
        
        if notes is not None:
            metadata[transaction_id]['notes'] = notes
        
        if tags is not None:
            metadata[transaction_id]['tags'] = tags
        
        metadata[transaction_id]['date_modified'] = datetime.now().isoformat()
        
        self.save_metadata(metadata)
    
    def add_note(self, transaction_id: str, note_text: str) -> None:
        """Add or update note for a transaction
        
        Args:
            transaction_id: Transaction identifier
            note_text: Note text
        """
        self.update_transaction_metadata(transaction_id, notes=note_text)
    
    def add_tag(self, transaction_id: str, tag: str) -> None:
        """Add a tag to a transaction
        
        Args:
            transaction_id: Transaction identifier
            tag: Tag to add
        """
        metadata = self.load_metadata()
        
        if transaction_id not in metadata:
            metadata[transaction_id] = {
                'notes': '',
                'tags': [],
                'date_modified': None
            }
        
        if tag not in metadata[transaction_id]['tags']:
            metadata[transaction_id]['tags'].append(tag)
            metadata[transaction_id]['date_modified'] = datetime.now().isoformat()
            self.save_metadata(metadata)
    
    def remove_tag(self, transaction_id: str, tag: str) -> None:
        """Remove a tag from a transaction
        
        Args:
            transaction_id: Transaction identifier
            tag: Tag to remove
        """
        metadata = self.load_metadata()
        
        if transaction_id in metadata:
            if tag in metadata[transaction_id]['tags']:
                metadata[transaction_id]['tags'].remove(tag)
                metadata[transaction_id]['date_modified'] = datetime.now().isoformat()
                self.save_metadata(metadata)
    
    def get_all_tags(self) -> Set[str]:
        """Get all unique tags across all transactions
        
        Returns:
            Set of all tags
        """
        metadata = self.load_metadata()
        all_tags = set()
        
        for txn_metadata in metadata.values():
            all_tags.update(txn_metadata.get('tags', []))
        
        return all_tags
    
    def get_tag_counts(self) -> Dict[str, int]:
        """Get count of transactions for each tag
        
        Returns:
            Dictionary mapping tags to transaction counts
        """
        metadata = self.load_metadata()
        tag_counts = {}
        
        for txn_metadata in metadata.values():
            for tag in txn_metadata.get('tags', []):
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        return tag_counts
    
    def get_transactions_by_tag(self, tag: str) -> List[str]:
        """Get all transaction IDs with a specific tag
        
        Args:
            tag: Tag to search for
            
        Returns:
            List of transaction IDs
        """
        metadata = self.load_metadata()
        transaction_ids = []
        
        for txn_id, txn_metadata in metadata.items():
            if tag in txn_metadata.get('tags', []):
                transaction_ids.append(txn_id)
        
        return transaction_ids
    
    def search_notes(self, query: str) -> List[str]:
        """Search notes for a query string
        
        Args:
            query: Search query
            
        Returns:
            List of transaction IDs with matching notes
        """
        metadata = self.load_metadata()
        transaction_ids = []
        query_lower = query.lower()
        
        for txn_id, txn_metadata in metadata.items():
            notes = txn_metadata.get('notes', '').lower()
            if query_lower in notes:
                transaction_ids.append(txn_id)
        
        return transaction_ids
    
    def bulk_add_tag(self, transaction_ids: List[str], tag: str) -> None:
        """Add a tag to multiple transactions
        
        Args:
            transaction_ids: List of transaction IDs
            tag: Tag to add
        """
        for txn_id in transaction_ids:
            self.add_tag(txn_id, tag)
    
    def bulk_remove_tag(self, transaction_ids: List[str], tag: str) -> None:
        """Remove a tag from multiple transactions
        
        Args:
            transaction_ids: List of transaction IDs
            tag: Tag to remove
        """
        for txn_id in transaction_ids:
            self.remove_tag(txn_id, tag)
    
    def clear_cache(self) -> None:
        """Clear the metadata cache"""
        self._cache = None
    
    def export_metadata(self) -> Dict:
        """Export all metadata
        
        Returns:
            Complete metadata dictionary
        """
        return self.load_metadata()
    
    def import_metadata(self, metadata: Dict) -> None:
        """Import metadata from dictionary
        
        Args:
            metadata: Metadata dictionary to import
        """
        self.save_metadata(metadata)
