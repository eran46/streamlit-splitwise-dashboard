"""
Migration v1.0.0 → v1.1.0: Add currency conversion fields to transactions
"""

from typing import Dict, Any
from datetime import datetime
from ..migration_base import BaseMigration


class Migration_v1_0_0_to_v1_1_0(BaseMigration):
    """Add currency conversion tracking fields to transactions
    
    Adds:
    - original_cost: Cost in original currency (for imported transactions)
    - original_currency: Currency code (e.g., "USD", "EUR")
    - exchange_rate: Exchange rate used for conversion
    - rate_date: Date when rate was retrieved
    - rate_source: Source of exchange rate (e.g., "ECB", "manual")
    - import_batch_id: Identifier for batch imports
    - conversion_date: When the conversion happened
    """
    
    from_version = "1.0.0"
    to_version = "1.1.0"
    description = "Add currency conversion fields to transactions for multi-currency support"
    
    def up(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Add currency conversion fields with safe defaults
        
        Args:
            data: Transaction data in v1.0.0 format
            
        Returns:
            Transaction data in v1.1.0 format
        """
        # Update schema version
        data['schema_version'] = self.to_version
        
        # Add migration timestamp to metadata
        if 'metadata' not in data:
            data['metadata'] = {}
        
        data['metadata']['last_migration'] = datetime.now().isoformat()
        data['metadata']['migrated_to_version'] = self.to_version
        
        # Add currency fields to each transaction
        for transaction in data.get('transactions', []):
            # Only add if not already present (idempotent)
            if 'original_cost' not in transaction:
                transaction['original_cost'] = None
            
            if 'original_currency' not in transaction:
                transaction['original_currency'] = None
            
            if 'exchange_rate' not in transaction:
                transaction['exchange_rate'] = None
            
            if 'rate_date' not in transaction:
                transaction['rate_date'] = None
            
            if 'rate_source' not in transaction:
                transaction['rate_source'] = None
            
            if 'import_batch_id' not in transaction:
                transaction['import_batch_id'] = None
            
            if 'conversion_date' not in transaction:
                transaction['conversion_date'] = None
        
        return data
    
    def down(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove currency conversion fields (rollback)
        
        Args:
            data: Transaction data in v1.1.0 format
            
        Returns:
            Transaction data in v1.0.0 format
        """
        # Update schema version
        data['schema_version'] = self.from_version
        
        # Remove migration metadata
        if 'metadata' in data:
            data['metadata'].pop('last_migration', None)
            data['metadata'].pop('migrated_to_version', None)
        
        # Remove currency fields from each transaction
        currency_fields = [
            'original_cost',
            'original_currency',
            'exchange_rate',
            'rate_date',
            'rate_source',
            'import_batch_id',
            'conversion_date'
        ]
        
        for transaction in data.get('transactions', []):
            for field in currency_fields:
                transaction.pop(field, None)
        
        return data
    
    def validate_input(self, data: Dict[str, Any]) -> bool:
        """Validate data is in v1.0.0 format
        
        Args:
            data: Data to validate
            
        Returns:
            True if valid v1.0.0 format
        """
        # Check version (allow None or 1.0.0)
        version = data.get('schema_version')
        if version is not None and version != self.from_version:
            return False
        
        # Check basic structure
        if 'transactions' not in data:
            return False
        
        if not isinstance(data['transactions'], list):
            return False
        
        # Check transactions have required fields
        # Note: Real data uses member names as keys, not payer/beneficiaries
        required_fields = ['date', 'description', 'cost']
        for transaction in data['transactions']:
            for field in required_fields:
                if field not in transaction:
                    return False
        
        return True
    
    def validate_output(self, data: Dict[str, Any]) -> bool:
        """Validate data is in v1.1.0 format
        
        Args:
            data: Data to validate
            
        Returns:
            True if valid v1.1.0 format
        """
        # Check version
        if data.get('schema_version') != self.to_version:
            return False
        
        # Check basic structure
        if 'transactions' not in data:
            return False
        
        if not isinstance(data['transactions'], list):
            return False
        
        # Check transactions have new currency fields
        currency_fields = [
            'original_cost',
            'original_currency',
            'exchange_rate',
            'rate_date',
            'rate_source',
            'import_batch_id',
            'conversion_date'
        ]
        
        for transaction in data['transactions']:
            for field in currency_fields:
                if field not in transaction:
                    return False
        
        return True
    
    def __str__(self) -> str:
        """String representation"""
        return f"Migration {self.from_version} → {self.to_version}: {self.description}"
