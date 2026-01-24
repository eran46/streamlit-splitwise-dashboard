"""
Currency Manager for ExpenseInfo Dashboard

Manages dashboard-wide currency settings including:
- Target currency configuration
- Exchange rate storage
- Currency detection from transactions
- Dashboard currency initialization
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Set
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CurrencyManager:
    """Manage dashboard-wide currency settings"""
    
    def __init__(self, user_data_dir="user_data"):
        self.user_data_dir = Path(user_data_dir)
        self.settings_file = self.user_data_dir / "dashboard_currency_settings.json"
        self.user_data_dir.mkdir(parents=True, exist_ok=True)
        
    def is_initialized(self) -> bool:
        """Check if dashboard currency settings exist"""
        return self.settings_file.exists()
    
    def get_target_currency(self) -> Optional[str]:
        """Get the current target currency"""
        if not self.is_initialized():
            return None
        
        settings = self.load_settings()
        return settings['dashboard_settings']['target_currency']
    
    def initialize_dashboard(self, target_currency: str, detected_currencies: List[str]) -> None:
        """
        Initialize dashboard with target currency
        Called during first import
        """
        settings = {
            "dashboard_settings": {
                "target_currency": target_currency,
                "created_at": datetime.now().isoformat(),
                "last_currency_change": datetime.now().isoformat(),
                "supported_currencies": sorted(set(detected_currencies + [target_currency])),
                "exchange_rates": {},
                "auto_update": False,
                "api_key": None
            },
            "historical_rates": {}
        }
        
        self.save_settings(settings)
        logger.info(f"Initialized dashboard with target currency: {target_currency}")
    
    def load_settings(self) -> Dict:
        """Load dashboard currency settings"""
        if not self.settings_file.exists():
            raise FileNotFoundError("Dashboard currency settings not found. Initialize dashboard first.")
        
        with open(self.settings_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def save_settings(self, settings: Dict) -> None:
        """Save dashboard currency settings"""
        with open(self.settings_file, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)
    
    def add_supported_currency(self, currency: str) -> None:
        """Add a currency to supported currencies list"""
        if not self.is_initialized():
            return
        
        settings = self.load_settings()
        if currency not in settings['dashboard_settings']['supported_currencies']:
            settings['dashboard_settings']['supported_currencies'].append(currency)
            settings['dashboard_settings']['supported_currencies'].sort()
            self.save_settings(settings)
            logger.info(f"Added currency to supported list: {currency}")
    
    def update_exchange_rate(self, currency: str, rate: float, source: str = "manual") -> None:
        """Update exchange rate for a currency"""
        if not self.is_initialized():
            return
        
        settings = self.load_settings()
        settings['dashboard_settings']['exchange_rates'][currency] = {
            "rate": rate,
            "last_updated": datetime.now().isoformat(),
            "source": source
        }
        self.save_settings(settings)
        logger.info(f"Updated exchange rate: 1 {settings['dashboard_settings']['target_currency']} = {rate} {currency}")
    
    def get_exchange_rate(self, currency: str) -> Optional[float]:
        """Get stored exchange rate for a currency"""
        if not self.is_initialized():
            return None
        
        settings = self.load_settings()
        target_currency = settings['dashboard_settings']['target_currency']
        
        if currency == target_currency:
            return 1.0
        
        rate_info = settings['dashboard_settings']['exchange_rates'].get(currency)
        return rate_info['rate'] if rate_info else None
    
    def detect_currencies_from_data(self, transactions: List[Dict]) -> Set[str]:
        """
        Detect all unique currencies from transaction data
        Looks for 'Currency' field in transactions
        """
        currencies = set()
        
        for txn in transactions:
            if 'Currency' in txn and txn['Currency']:
                currencies.add(txn['Currency'])
            # Also check original_currency if exists
            if 'original_currency' in txn and txn['original_currency']:
                currencies.add(txn['original_currency'])
        
        return currencies
    
    def get_currency_info(self) -> Dict:
        """Get comprehensive currency information"""
        if not self.is_initialized():
            return {
                'initialized': False,
                'target_currency': None,
                'supported_currencies': [],
                'exchange_rates': {}
            }
        
        settings = self.load_settings()
        dashboard_settings = settings['dashboard_settings']
        
        return {
            'initialized': True,
            'target_currency': dashboard_settings['target_currency'],
            'supported_currencies': dashboard_settings['supported_currencies'],
            'exchange_rates': dashboard_settings['exchange_rates'],
            'created_at': dashboard_settings['created_at'],
            'last_currency_change': dashboard_settings.get('last_currency_change'),
            'auto_update': dashboard_settings.get('auto_update', False)
        }
    
    def migrate_existing_dashboard(self, default_currency: str = "ILS") -> None:
        """
        Migrate existing dashboard without currency settings
        Sets default currency and prepares for future conversions
        """
        if self.is_initialized():
            logger.info("Dashboard already initialized, skipping migration")
            return
        
        settings = {
            "dashboard_settings": {
                "target_currency": default_currency,
                "created_at": datetime.now().isoformat(),
                "last_currency_change": datetime.now().isoformat(),
                "migrated_from_legacy": True,
                "supported_currencies": [default_currency],
                "exchange_rates": {},
                "auto_update": False,
                "api_key": None
            },
            "historical_rates": {}
        }
        
        self.save_settings(settings)
        logger.info(f"Migrated existing dashboard with default currency: {default_currency}")


# Currency symbols mapping
CURRENCY_SYMBOLS = {
    'ILS': '₪',
    'USD': '$',
    'EUR': '€',
    'GBP': '£',
    'JPY': '¥',
    'CNY': '¥',
    'BGN': 'лв',
    'CHF': 'Fr',
    'CAD': 'C$',
    'AUD': 'A$',
    'NZD': 'NZ$',
    'HKD': 'HK$',
    'SGD': 'S$',
    'THB': '฿',
    'INR': '₹',
    'KRW': '₩',
    'MXN': 'Mex$',
    'BRL': 'R$',
    'ZAR': 'R',
    'RUB': '₽',
    'TRY': '₺',
    'PLN': 'zł',
    'SEK': 'kr',
    'NOK': 'kr',
    'DKK': 'kr',
    'CZK': 'Kč',
    'HUF': 'Ft',
    'RON': 'lei',
    'VND': '₫',
}


def get_currency_symbol(currency: str) -> str:
    """Get currency symbol, fallback to currency code"""
    return CURRENCY_SYMBOLS.get(currency, currency)


def format_currency_amount(amount: float, currency: str) -> str:
    """Format amount with currency symbol"""
    symbol = get_currency_symbol(currency)
    
    # Symbol before for USD, EUR, GBP
    if currency in ['USD', 'EUR', 'GBP', 'CAD', 'AUD', 'NZD', 'HKD', 'SGD', 'MXN', 'BRL', 'ZAR']:
        return f"{symbol}{amount:,.2f}"
    else:
        return f"{amount:,.2f} {symbol}"


# ISO 4217 currency names
CURRENCY_NAMES = {
    'ILS': 'Israeli Shekel',
    'USD': 'US Dollar',
    'EUR': 'Euro',
    'GBP': 'British Pound',
    'JPY': 'Japanese Yen',
    'CNY': 'Chinese Yuan',
    'BGN': 'Bulgarian Lev',
    'CHF': 'Swiss Franc',
    'CAD': 'Canadian Dollar',
    'AUD': 'Australian Dollar',
    'NZD': 'New Zealand Dollar',
    'HKD': 'Hong Kong Dollar',
    'SGD': 'Singapore Dollar',
    'THB': 'Thai Baht',
    'INR': 'Indian Rupee',
    'KRW': 'South Korean Won',
    'MXN': 'Mexican Peso',
    'BRL': 'Brazilian Real',
    'ZAR': 'South African Rand',
    'RUB': 'Russian Ruble',
    'TRY': 'Turkish Lira',
    'PLN': 'Polish Złoty',
    'SEK': 'Swedish Krona',
    'NOK': 'Norwegian Krone',
    'DKK': 'Danish Krone',
    'CZK': 'Czech Koruna',
    'HUF': 'Hungarian Forint',
    'RON': 'Romanian Leu',
    'VND': 'Vietnamese Dong',
}


def get_currency_name(currency: str) -> str:
    """Get full currency name"""
    return CURRENCY_NAMES.get(currency, currency)


if __name__ == "__main__":
    # Test the currency manager
    print("Testing CurrencyManager...")
    print("=" * 80)
    
    # Use test directory
    import tempfile
    import shutil
    test_dir = Path(tempfile.mkdtemp())
    print(f"Test directory: {test_dir}")
    
    try:
        manager = CurrencyManager(user_data_dir=str(test_dir))
        
        # Test 1: Check initialization
        print("\n1. Testing initialization check:")
        print(f"   Is initialized: {manager.is_initialized()}")
        print(f"   Target currency: {manager.get_target_currency()}")
        
        # Test 2: Initialize dashboard
        print("\n2. Initializing dashboard with ILS:")
        manager.initialize_dashboard("ILS", ["ILS", "USD", "BGN", "EUR"])
        print(f"   ✅ Dashboard initialized")
        print(f"   Target currency: {manager.get_target_currency()}")
        
        # Test 3: Get currency info
        print("\n3. Getting currency info:")
        info = manager.get_currency_info()
        print(f"   Target: {info['target_currency']}")
        print(f"   Supported: {', '.join(info['supported_currencies'])}")
        
        # Test 4: Currency formatting
        print("\n4. Testing currency formatting:")
        print(f"   200 BGN: {format_currency_amount(200, 'BGN')}")
        print(f"   50 USD: {format_currency_amount(50, 'USD')}")
        print(f"   300 EUR: {format_currency_amount(300, 'EUR')}")
        print(f"   1000 ILS: {format_currency_amount(1000, 'ILS')}")
        
        print("\n" + "=" * 80)
        print("Tests complete!")
        
    finally:
        # Cleanup
        shutil.rmtree(test_dir)
        print(f"Cleaned up test directory")
