"""
Base class for all database migrations.

Each migration must inherit from BaseMigration and implement up() and down() methods.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
from datetime import datetime


class BaseMigration(ABC):
    """Base class for all migrations
    
    Attributes:
        from_version (str): Starting schema version (e.g., "1.0.0")
        to_version (str): Target schema version (e.g., "1.1.0")
        description (str): Human-readable description of migration
    """
    
    from_version: str = None
    to_version: str = None
    description: str = None
    
    @abstractmethod
    def up(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate data forward to new schema
        
        Args:
            data: Current data structure
            
        Returns:
            Migrated data structure with updated schema
            
        Raises:
            Exception: If migration fails
        """
        pass
    
    @abstractmethod
    def down(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Rollback migration (reverse transformation)
        
        Args:
            data: Current data structure
            
        Returns:
            Data structure with reverted schema
            
        Raises:
            Exception: If rollback fails
        """
        pass
    
    def validate_input(self, data: Dict[str, Any]) -> bool:
        """Validate data is in expected 'from' format
        
        Args:
            data: Data to validate
            
        Returns:
            True if valid, False otherwise
        """
        current_version = data.get('schema_version', '1.0.0')
        return current_version == self.from_version
    
    def validate_output(self, data: Dict[str, Any]) -> bool:
        """Validate data is in expected 'to' format
        
        Args:
            data: Data to validate
            
        Returns:
            True if valid, False otherwise
        """
        return data.get('schema_version') == self.to_version
    
    def get_info(self) -> Dict[str, str]:
        """Get migration information
        
        Returns:
            Dictionary with migration metadata
        """
        return {
            'from_version': self.from_version,
            'to_version': self.to_version,
            'description': self.description,
            'class_name': self.__class__.__name__
        }
    
    def __str__(self) -> str:
        return f"{self.from_version} → {self.to_version}: {self.description}"
