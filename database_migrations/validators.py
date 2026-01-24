"""
Data validation utilities for migrations
"""

from typing import Dict, Any, List, Optional


def validate_transaction_schema(transaction: Dict[str, Any], required_fields: List[str]) -> tuple[bool, Optional[str]]:
    """Validate a single transaction has required fields
    
    Args:
        transaction: Transaction dictionary
        required_fields: List of field names that must be present
        
    Returns:
        (is_valid, error_message) tuple
    """
    for field in required_fields:
        if field not in transaction:
            return False, f"Missing required field: {field}"
    
    return True, None


def validate_json_structure(data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """Validate basic JSON structure
    
    Args:
        data: Data to validate
        
    Returns:
        (is_valid, error_message) tuple
    """
    # Check it's a dictionary
    if not isinstance(data, dict):
        return False, "Data must be a dictionary"
    
    # Check for transactions
    if 'transactions' not in data:
        return False, "Missing 'transactions' field"
    
    # Check transactions is a list
    if not isinstance(data['transactions'], list):
        return False, "'transactions' must be a list"
    
    return True, None


def validate_data_types(transaction: Dict[str, Any], field_types: Dict[str, type]) -> tuple[bool, Optional[str]]:
    """Validate field data types
    
    Args:
        transaction: Transaction dictionary
        field_types: Dictionary mapping field names to expected types
        
    Returns:
        (is_valid, error_message) tuple
    """
    for field, expected_type in field_types.items():
        if field in transaction:
            value = transaction[field]
            # Allow None for optional fields
            if value is not None and not isinstance(value, expected_type):
                return False, f"Field '{field}' should be {expected_type.__name__}, got {type(value).__name__}"
    
    return True, None


def validate_schema_version(data: Dict[str, Any], expected_version: str) -> tuple[bool, Optional[str]]:
    """Validate schema version matches expected
    
    Args:
        data: Data to validate
        expected_version: Expected version string
        
    Returns:
        (is_valid, error_message) tuple
    """
    version = data.get('schema_version')
    
    if version != expected_version:
        return False, f"Expected schema version {expected_version}, got {version}"
    
    return True, None


def validate_required_fields(data: Dict[str, Any], transaction_fields: List[str]) -> tuple[bool, Optional[str]]:
    """Validate all transactions have required fields
    
    Args:
        data: Full data dictionary
        transaction_fields: Required fields for transactions
        
    Returns:
        (is_valid, error_message) tuple
    """
    transactions = data.get('transactions', [])
    
    for idx, transaction in enumerate(transactions):
        for field in transaction_fields:
            if field not in transaction:
                return False, f"Transaction {idx} missing required field: {field}"
    
    return True, None
