"""
Exchange Rate Manager for ExpenseInfo Dashboard

Manages exchange rate fetching and caching with dual API support:
- Primary API: exchangerate-api.com (latest rates for all currencies)
- Fallback API: frankfurter.app (FREE historical rates for ECB currencies)

Strategy:
1. Try historical rates first (frankfurter.app for ECB currencies)
2. Fall back to import-time rates (exchangerate-api.com for non-ECB)
"""

import requests
from datetime import datetime, timedelta
from pathlib import Path
import json
from typing import Dict, Optional, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExchangeRateManager:
    """Manage exchange rate fetching and caching with fallback support"""
    
    def __init__(self, cache_dir="user_data/exchange_rate_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.primary_url = "https://api.exchangerate-api.com/v4"
        self.fallback_url = "https://api.frankfurter.app"
        self.cache_ttl_days = 1  # Cache latest rates for 1 day
        
        # ECB currencies supported by frankfurter.app (verified 2026-01-23)
        # NOTE: BGN is NOT included despite being an EU country!
        # ILS is included in frankfurter.app but not in this set since it's our base
        self.ecb_currencies = {
            'AUD', 'BRL', 'CAD', 'CHF', 'CNY', 'CZK', 'DKK',
            'EUR', 'GBP', 'HKD', 'HUF', 'IDR', 'INR', 'ISK',
            'JPY', 'KRW', 'MXN', 'MYR', 'NOK', 'NZD', 'PHP', 'PLN',
            'RON', 'SEK', 'SGD', 'THB', 'TRY', 'USD', 'ZAR'
        }
        # Total: 29 currencies (excluding ILS which is supported as base)
        
    def get_latest_rates(self, base_currency: str = "ILS") -> Dict[str, float]:
        """
        Get latest exchange rates for base currency
        Uses cache if available and fresh (< 1 day old)
        
        IMPORTANT: To avoid inconsistent rates, always fetches EUR-based rates
        and calculates cross-rates via triangulation. This ensures mathematical
        consistency: rate(A->B) * rate(B->C) = rate(A->C)
        """
        cache_file = self.cache_dir / f"latest_{base_currency}.json"
        
        # Check cache first
        if cache_file.exists():
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
                cache_date = datetime.fromisoformat(cached_data['cached_at'])
                
                # Use cache if less than 1 day old
                if datetime.now() - cache_date < timedelta(days=self.cache_ttl_days):
                    logger.info(f"Using cached rates for {base_currency}")
                    return cached_data['rates']
        
        # Fetch EUR-based rates and triangulate to get base_currency rates
        # This avoids the inconsistent rates problem with exchangerate-api.com
        try:
            url = f"{self.primary_url}/latest/EUR"
            logger.info(f"Fetching EUR-based rates from primary API for triangulation")
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            eur_rates = data['rates']
            
            # Get the EUR -> base_currency rate
            base_rate_from_eur = eur_rates.get(base_currency)
            if base_rate_from_eur is None:
                raise ValueError(f"Base currency {base_currency} not found in EUR rates")
            
            # Calculate cross-rates via EUR triangulation
            # If 1 EUR = X base_currency and 1 EUR = Y target_currency
            # Then 1 base_currency = Y/X target_currency
            rates = {}
            for target, eur_to_target_rate in eur_rates.items():
                rates[target] = eur_to_target_rate / base_rate_from_eur
            
            # Add base currency itself
            rates[base_currency] = 1.0
            
            # Cache the result
            cache_data = {
                'cached_at': datetime.now().isoformat(),
                'api_date': data['date'],
                'base': base_currency,
                'rates': rates,
                'source': 'exchangerate-api.com (EUR triangulation)'
            }
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2)
            
            logger.info(f"Successfully calculated {len(rates)} triangulated rates for {base_currency}")
            return rates
            
        except requests.RequestException as e:
            logger.warning(f"Primary API failed: {e}. Trying fallback...")
            
            # Try fallback API (frankfurter.app) if base is ECB currency
            if base_currency in self.ecb_currencies or base_currency == 'ILS':
                try:
                    url = f"{self.fallback_url}/latest?from={base_currency}"
                    logger.info(f"Fetching latest rates from fallback API: {url}")
                    response = requests.get(url, timeout=10)
                    response.raise_for_status()
                    
                    data = response.json()
                    rates = data['rates']
                    rates[base_currency] = 1.0  # Add base currency itself
                    
                    # Cache the result
                    cache_data = {
                        'cached_at': datetime.now().isoformat(),
                        'api_date': data['date'],
                        'base': base_currency,
                        'rates': rates,
                        'source': 'frankfurter.app'
                    }
                    
                    with open(cache_file, 'w', encoding='utf-8') as f:
                        json.dump(cache_data, f, indent=2)
                    
                    logger.info(f"Successfully fetched {len(rates)} rates from fallback API")
                    return rates
                    
                except requests.RequestException as e2:
                    logger.error(f"Fallback API also failed: {e2}")
            
            # If both APIs fail, use cached data even if stale
            if cache_file.exists():
                logger.warning("Using stale cached data")
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)['rates']
            
            raise Exception(f"All APIs failed and no cache available: Primary={e}")
    
    def get_rate_for_date(self, base_currency: str, target_currency: str, 
                         date: datetime) -> Optional[float]:
        """
        Get exchange rate for specific date
        
        Strategy (priority order):
        1. Check cache first
        2. Try frankfurter.app for historical rates (FREE, ECB currencies only)
        3. Fall back to latest rate if not ECB currency
        4. Store rate at import time for consistency
        """
        date_str = date.strftime('%Y-%m-%d')
        cache_file = self.cache_dir / f"historical_{base_currency}_{target_currency}_{date_str}.json"
        
        # Check cache first
        if cache_file.exists():
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
                logger.info(f"Using cached historical rate for {target_currency} on {date_str}")
                return cached_data['rate']
        
        # Try frankfurter.app for historical rates (if both are ECB currencies or ILS)
        if ((base_currency in self.ecb_currencies or base_currency == 'ILS') and 
            (target_currency in self.ecb_currencies or target_currency == 'ILS')):
            try:
                url = f"{self.fallback_url}/{date_str}?from={base_currency}&to={target_currency}"
                logger.info(f"Fetching historical rate from frankfurter.app: {url}")
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                rate = data['rates'].get(target_currency)
                
                if rate:
                    # Cache this historical rate
                    cache_data = {
                        'cached_at': datetime.now().isoformat(),
                        'date': date_str,
                        'base': base_currency,
                        'target': target_currency,
                        'rate': rate,
                        'source': 'frankfurter.app (historical)',
                        'note': 'Historical rate from ECB data'
                    }
                    
                    with open(cache_file, 'w', encoding='utf-8') as f:
                        json.dump(cache_data, f, indent=2)
                    
                    logger.info(f"Successfully fetched historical rate: 1 {base_currency} = {rate} {target_currency}")
                    return rate
                    
            except requests.RequestException as e:
                logger.warning(f"Frankfurter historical rate failed: {e}. Using latest rate as fallback...")
        else:
            logger.info(f"Historical rates not available for {target_currency} (non-ECB currency)")
        
        # Fall back to latest rate (non-ECB currencies or if historical fetch failed)
        logger.warning(f"Using latest rate for {date_str} (historical data unavailable)")
        
        try:
            latest_rates = self.get_latest_rates(base_currency)
            rate = latest_rates.get(target_currency)
            
            if rate is None:
                raise ValueError(f"Rate for {target_currency} not found")
            
            # Cache this rate for the specific date as fallback
            cache_data = {
                'cached_at': datetime.now().isoformat(),
                'date': date_str,
                'base': base_currency,
                'target': target_currency,
                'rate': rate,
                'source': 'latest_rate_approximation (import-time fallback)',
                'note': 'Historical rate unavailable, using latest rate'
            }
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2)
            
            return rate
            
        except Exception as e:
            raise Exception(f"Failed to get rate for {target_currency}: {e}")
    
    def convert_amount(self, amount: float, from_currency: str, 
                      to_currency: str, date: Optional[datetime] = None) -> Tuple[float, Dict]:
        """
        Convert amount from one currency to another
        Returns: (converted_amount, rate_metadata)
        
        Rate metadata includes:
        - exchange_rate: The rate used
        - rate_date: When the rate was fetched
        - rate_source: Which API/strategy was used
        """
        if from_currency == to_currency:
            return amount, {
                'exchange_rate': 1.0,
                'rate_date': datetime.now().strftime('%Y-%m-%d'),
                'rate_source': 'same_currency'
            }
        
        # Get the rate
        if date:
            # Try historical rate first
            # Rate is: 1 from_currency = X to_currency
            rate = self.get_rate_for_date(from_currency, to_currency, date)
            rate_date = date.strftime('%Y-%m-%d')
        else:
            # Use latest rate
            # Rates are: 1 from_currency = X to_currency
            rates = self.get_latest_rates(from_currency)
            rate = rates.get(to_currency)
            rate_date = datetime.now().strftime('%Y-%m-%d')
        
        if rate is None:
            raise ValueError(f"Rate not available for {from_currency} -> {to_currency}")
        
        # Convert: AMOUNT * RATE = CONVERTED_AMOUNT
        # If rate is 1.8925 (1 BGN = 1.8925 ILS), and we have 200 BGN:
        # 200 BGN * 1.8925 = 378.5 ILS
        converted_amount = amount * rate
        
        # Get rate metadata from cache
        if date:
            cache_file = self.cache_dir / f"historical_{from_currency}_{to_currency}_{date.strftime('%Y-%m-%d')}.json"
        else:
            cache_file = self.cache_dir / f"latest_{from_currency}.json"
        
        rate_source = 'unknown'
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    rate_source = cache_data.get('source', 'unknown')
            except:
                pass
        
        metadata = {
            'exchange_rate': rate,
            'rate_date': rate_date,
            'rate_source': rate_source
        }
        
        return converted_amount, metadata
    
    def create_import_batch_cache(self, group_name: str, base_currency: str, 
                                   currencies_with_rates: Dict[str, Dict], 
                                   transaction_count: int) -> str:
        """
        Create an import batch cache file for non-ECB currencies
        Returns the import_batch_id
        """
        import_timestamp = datetime.now()
        import_batch_id = f"import_{import_timestamp.strftime('%Y-%m-%d_%H%M%S')}"
        
        # Separate historical and fallback rates
        fallback_rates = {}
        historical_rates = {}
        
        for currency, rate_info in currencies_with_rates.items():
            if 'import-time fallback' in rate_info.get('source', ''):
                fallback_rates[currency] = rate_info
            elif 'historical' in rate_info.get('source', ''):
                if currency not in historical_rates:
                    historical_rates[currency] = {
                        'source': rate_info['source'],
                        'dates': []
                    }
                if 'date' in rate_info:
                    historical_rates[currency]['dates'].append(rate_info['date'])
        
        batch_data = {
            'import_timestamp': import_timestamp.isoformat(),
            'group_name': group_name,
            'base_currency': base_currency,
            'fallback_rates': fallback_rates,
            'historical_rates_used': historical_rates,
            'transaction_count': transaction_count,
            'currencies_detected': list(currencies_with_rates.keys())
        }
        
        cache_file = self.cache_dir / f"{import_batch_id}.json"
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(batch_data, f, indent=2)
        
        logger.info(f"Created import batch cache: {import_batch_id}")
        return import_batch_id
    
    def clear_old_cache(self, days: int = 90):
        """Remove cached rates older than N days"""
        cutoff = datetime.now() - timedelta(days=days)
        cleared_count = 0
        
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    cached_at = datetime.fromisoformat(data['cached_at'])
                    
                    if cached_at < cutoff:
                        cache_file.unlink()
                        cleared_count += 1
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
        
        logger.info(f"Cleared {cleared_count} old cache files")
        return cleared_count


if __name__ == "__main__":
    # Test the exchange rate manager
    print("Testing ExchangeRateManager...")
    print("=" * 80)
    
    manager = ExchangeRateManager()
    
    # Test 1: Get latest rates
    print("\n1. Testing latest rates (ILS base):")
    try:
        rates = manager.get_latest_rates("ILS")
        print(f"   ✅ Got {len(rates)} rates")
        print(f"   USD: {rates.get('USD', 'N/A')}")
        print(f"   EUR: {rates.get('EUR', 'N/A')}")
        print(f"   BGN: {rates.get('BGN', 'N/A')}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 2: Convert amount (BGN to ILS - fallback)
    print("\n2. Testing conversion (200 BGN → ILS):")
    try:
        converted, metadata = manager.convert_amount(200, "BGN", "ILS")
        print(f"   ✅ 200 BGN = ₪{converted:.2f}")
        print(f"   Rate: {metadata['exchange_rate']}")
        print(f"   Source: {metadata['rate_source']}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 3: Historical rate (EUR - should work)
    print("\n3. Testing historical rate (EUR, 3 months ago):")
    try:
        three_months_ago = datetime.now() - timedelta(days=90)
        rate = manager.get_rate_for_date("ILS", "EUR", three_months_ago)
        print(f"   ✅ Rate on {three_months_ago.strftime('%Y-%m-%d')}: {rate}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 80)
    print("Tests complete!")
