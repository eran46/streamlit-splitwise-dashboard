import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import io
import json
import time
from data_manager import DataManager
from groups_manager import GroupsManager
from user_expense_calculator import UserExpenseCalculator
from currency_manager import CurrencyManager, get_currency_symbol, format_currency_amount, get_currency_name
from exchange_rate_manager import ExchangeRateManager
from database_migrations.migration_manager import MigrationManager

# Page configuration
st.set_page_config(
    page_title="ExpenseInfo - Splitwise Expense Analyzer",
    page_icon="💰",
    layout="wide"
)

# Initialize managers
@st.cache_resource
def get_groups_manager():
    """Singleton groups manager"""
    return GroupsManager()

@st.cache_resource
def get_currency_manager():
    """Singleton currency manager"""
    return CurrencyManager()

@st.cache_resource
def get_exchange_rate_manager():
    """Singleton exchange rate manager"""
    return ExchangeRateManager()

def get_data_manager(group_data_path=None):
    """Get data manager for specific group or active group"""
    if group_data_path is None:
        groups_mgr = get_groups_manager()
        if groups_mgr.has_groups():
            active_group = groups_mgr.get_active_group()
            group_data_path = groups_mgr.get_group_data_path(active_group['id'])
    return DataManager(group_data_path)

def show_import_rate_summary(df, transactions):
    """Show summary of exchange rates used during import"""
    currency_mgr = get_currency_manager()
    if not currency_mgr.is_initialized():
        return
    
    target_currency = currency_mgr.get_target_currency()
    target_symbol = get_currency_symbol(target_currency)
    
    # Collect rate information from transactions
    rate_info = {}
    for txn in transactions:
        if 'original_currency' in txn and txn.get('original_currency') != target_currency:
            orig_curr = txn['original_currency']
            if orig_curr not in rate_info:
                rate_info[orig_curr] = {
                    'count': 0,
                    'rate': txn.get('exchange_rate'),
                    'rate_date': txn.get('rate_date'),
                    'rate_source': txn.get('rate_source'),
                    'total_original': 0,
                    'total_converted': 0
                }
            rate_info[orig_curr]['count'] += 1
            rate_info[orig_curr]['total_original'] += txn.get('original_cost', 0)
            rate_info[orig_curr]['total_converted'] += txn.get('cost', 0)
    
    if not rate_info:
        return  # No foreign currencies
    
    st.markdown("---")
    st.subheader("💱 Currency Conversion Summary")
    
    # Create columns for each currency
    cols = st.columns(min(len(rate_info), 3))
    
    for idx, (currency, info) in enumerate(rate_info.items()):
        with cols[idx % len(cols)]:
            curr_symbol = get_currency_symbol(currency)
            curr_name = get_currency_name(currency)
            
            # Determine strategy used
            rate_source = info['rate_source'] or 'unknown'
            if 'historical' in rate_source.lower():
                strategy = "📊 Historical Rate"
                strategy_color = "🟢"
                accuracy = "Most accurate"
            else:
                strategy = "⏱️ Import-time Rate"
                strategy_color = "🟡"
                accuracy = "~1% accurate"
            
            st.markdown(f"#### {currency} {curr_symbol}")
            st.caption(f"{curr_name}")
            
            # Show rate
            if info['rate']:
                st.metric(
                    "Exchange Rate",
                    f"1 {currency} = {info['rate']:.4f} {target_currency}",
                    delta=None
                )
            
            # Show transaction count and total
            st.write(f"**{info['count']} transactions**")
            st.write(f"{curr_symbol}{info['total_original']:,.2f} → {target_symbol}{info['total_converted']:,.2f}")
            
            # Show strategy and accuracy
            st.caption(f"{strategy_color} {strategy}")
            st.caption(f"✓ {accuracy}")
            
            # Show rate date
            if info['rate_date']:
                st.caption(f"📅 Rate from: {info['rate_date']}")
    
    # Show overall summary
    st.info(
        f"💡 **Conversion Strategy**: "
        f"Historical rates used when available (ECB currencies), "
        f"import-time rates for others. All rates cached for consistency."
    )

def convert_df_to_transactions(df, apply_currency_conversion=False):
    """Convert DataFrame to transaction list, preserving all columns including member splits
    
    Args:
        df: DataFrame with transaction data
        apply_currency_conversion: If True, convert amounts to target currency
    """
    transactions = []
    
    # Check if currency conversion is needed
    currency_mgr = get_currency_manager()
    exchange_mgr = get_exchange_rate_manager()
    target_currency = currency_mgr.get_target_currency() if currency_mgr.is_initialized() else None
    
    # Standard columns
    standard_cols = ['Date', 'Description', 'Category', 'Cost', 'Currency']
    
    # Track currencies for batch cache
    currencies_used = {}
    
    for _, row in df.iterrows():
        original_cost = float(row['Cost']) if pd.notna(row['Cost']) else 0.0
        original_currency = str(row['Currency']) if pd.notna(row['Currency']) else 'ILS'
        converted_cost = original_cost
        rate_metadata = None
        
        # Apply currency conversion if enabled
        if apply_currency_conversion and target_currency and original_currency != target_currency:
            try:
                # Get transaction date for historical rate
                txn_date = pd.to_datetime(row['Date']) if pd.notna(row['Date']) else None
                
                # Convert amount
                converted_cost, rate_metadata = exchange_mgr.convert_amount(
                    original_cost,
                    original_currency,
                    target_currency,
                    date=txn_date
                )
                
                # Track currency usage for batch cache
                if original_currency not in currencies_used:
                    currencies_used[original_currency] = rate_metadata
                
            except Exception as e:
                st.warning(f"Could not convert {original_cost} {original_currency}: {e}")
                converted_cost = original_cost
        
        # Create transaction
        txn = {
            'date': row['Date'].isoformat() if pd.notna(row['Date']) else None,
            'description': str(row['Description']) if pd.notna(row['Description']) else '',
            'category': str(row['Category']) if pd.notna(row['Category']) else '',
            'cost': converted_cost,  # Converted amount
            'currency': target_currency if (apply_currency_conversion and target_currency) else original_currency,
            'source': 'splitwise_import' if apply_currency_conversion else 'import'
        }
        
        # Add original currency information if conversion was applied
        if apply_currency_conversion and target_currency and original_currency != target_currency:
            txn['original_cost'] = original_cost
            txn['original_currency'] = original_currency
            if rate_metadata:
                txn['exchange_rate'] = rate_metadata['exchange_rate']
                txn['rate_date'] = rate_metadata['rate_date']
                txn['rate_source'] = rate_metadata['rate_source']
        
        # Preserve all other columns (member splits, etc.)
        # Note: Member splits will also need to be converted if currency conversion is applied
        for col in df.columns:
            if col not in standard_cols and col not in txn:
                # Add any non-standard column (like member names)
                if pd.notna(row[col]):
                    value = float(row[col]) if isinstance(row[col], (int, float)) else str(row[col])
                    
                    # Convert member split amounts if this is a numeric column (likely a member split)
                    if isinstance(value, float) and apply_currency_conversion and target_currency and original_currency != target_currency:
                        try:
                            # Same conversion ratio as the main cost
                            if original_cost > 0:
                                conversion_ratio = converted_cost / original_cost
                                value = value * conversion_ratio
                        except:
                            pass
                    
                    txn[col] = value
        
        transactions.append(txn)
    
    # Create import batch cache if conversions were applied
    if apply_currency_conversion and currencies_used:
        groups_mgr = get_groups_manager()
        active_group = groups_mgr.get_active_group()
        group_name = active_group['name'] if active_group else "Unknown"
        
        import_batch_id = exchange_mgr.create_import_batch_cache(
            group_name,
            target_currency,
            currencies_used,
            len(transactions)
        )
        
        # Add import_batch_id to transactions that used fallback rates
        for txn in transactions:
            if txn.get('rate_source') and 'fallback' in txn.get('rate_source', ''):
                txn['import_batch_id'] = import_batch_id
    
    return transactions

@st.cache_data
def detect_currencies_in_df(df: pd.DataFrame) -> list:
    """Detect unique currencies in a dataframe"""
    currencies = []
    if 'Currency' in df.columns:
        unique_currencies = df['Currency'].dropna().unique()
        currencies = [c for c in unique_currencies if c]
    return sorted(set(currencies)) if currencies else ['ILS']  # Default to ILS if no currency column

def format_amount_with_original(transaction: dict, target_currency: str = None) -> str:
    """
    Format amount showing both converted and original currency
    
    Args:
        transaction: Transaction dict with cost, currency, original_cost, original_currency
        target_currency: Target currency for display (optional, uses transaction currency if not provided)
    
    Returns:
        Formatted string like "₪375.51 (200 BGN)" or "₪450.00" if no conversion
    """
    cost = transaction.get('cost', 0)
    currency = transaction.get('currency', target_currency or 'ILS')
    original_cost = transaction.get('original_cost')
    original_currency = transaction.get('original_currency')
    
    # Format the converted amount
    converted_str = format_currency_amount(cost, currency)
    
    # If there's an original amount and it's different from converted, show both
    if (original_cost is not None and original_currency and 
        original_currency != currency and abs(original_cost - cost) > 0.01):
        original_str = format_currency_amount(original_cost, original_currency)
        return f"{converted_str} ({original_str})"
    
    return converted_str

def load_data_from_file(file, file_type):
    """Load and preprocess the expense data from uploaded file"""
    try:
        if file_type == 'csv':
            # Try different encodings to handle Hebrew characters
            encodings = ['utf-8-sig', 'utf-8', 'cp1255', 'iso-8859-8']
            df = None
            for encoding in encodings:
                try:
                    file.seek(0)
                    df = pd.read_csv(file, encoding=encoding)
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue
            
            if df is None:
                st.error("Could not decode CSV file. Please export from Google Sheets as Excel.")
                return None
        else:
            # Excel file
            df = pd.read_excel(file)
        
        # Data cleaning
        # Convert Date column to datetime
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # Ensure Cost is numeric
        df['Cost'] = pd.to_numeric(df['Cost'], errors='coerce')
        
        # Remove rows with invalid dates or costs
        df = df.dropna(subset=['Date', 'Cost'])
        
        # Sort by date
        df = df.sort_values('Date')
        
        return df
    
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def update_group_members_from_data(group_id: str, df: pd.DataFrame):
    """Update group members list based on detected member columns in data
    
    Merges new members with existing members to preserve historical member data.
    This ensures that members from previous imports remain visible even when
    importing new data with different member combinations.
    """
    dm = get_data_manager()
    member_cols = dm.get_member_columns(df)
    
    if member_cols:
        groups_mgr = get_groups_manager()
        group = groups_mgr.get_group_by_id(group_id)
        
        if group:
            # MERGE members: combine existing with new ones
            current_members = set(group.get('members', []))
            new_members = set(member_cols)
            merged_members = current_members.union(new_members)
            
            # Only update if there are actually new members
            if merged_members != current_members:
                groups_mgr.update_group(group_id, members=list(merged_members))
                return list(merged_members)
    
    return []

def load_persisted_data():
    """Load data from persistence layer"""
    dm = get_data_manager()
    return dm.get_dataframe()

def get_member_columns(df):
    """Identify columns that represent member names (not standard columns)"""
    standard_cols = ['Date', 'Description', 'Category', 'Cost', 'Currency']
    member_cols = [col for col in df.columns if col not in standard_cols]
    return member_cols

def is_reimbursement_transaction(row, member_cols):
    """
    Check if a transaction is a reimbursement (one person owed full amount).
    In such transactions, one member has +Cost and others sum to -Cost.
    Member sum (absolute) ≈ 2×Cost for 2 members, higher for more members.
    """
    if not member_cols:
        return False
    
    cost = row['Cost']
    if cost == 0:
        return False
    
    # Calculate absolute sum of member values, handling non-numeric values
    member_sum_abs = 0
    for m in member_cols:
        if m in row and pd.notna(row[m]):
            try:
                val = float(row[m])
                member_sum_abs += abs(val)
            except (ValueError, TypeError):
                # Skip non-numeric values
                continue
    
    # Calculate ratio
    ratio = member_sum_abs / cost if cost > 0 else 0
    
    # For reimbursements, ratio should be approximately 2.0
    # (one person has +cost, others sum to -cost, so abs sum = 2×cost)
    return 1.9 < ratio < 2.1

def exclude_payments(df):
    """Exclude Payment and Settlement categories from calculations"""
    exclude_categories = ['Payment', 'Settlement', 'payment', 'settlement']
    return df[~df['Category'].isin(exclude_categories)]

def exclude_reimbursements(df):
    """
    Exclude reimbursement transactions where one person is owed the full amount.
    These are internal transfers, not real household expenses.
    """
    if df.empty:
        return df
    
    # Get member columns
    dm = get_data_manager()
    member_cols = dm.get_member_columns(df)
    
    if not member_cols:
        return df
    
    # Filter out reimbursement transactions
    mask = ~df.apply(lambda row: is_reimbursement_transaction(row, member_cols), axis=1)
    return df[mask]

def exclude_payments_and_reimbursements(df):
    """Exclude both Payment/Settlement categories and reimbursement transactions"""
    df = exclude_payments(df)
    df = exclude_reimbursements(df)
    return df

def calculate_total_spending(df):
    """
    Calculate Total Spending as the sum of all member expenses (what they actually paid).
    This matches the sum of all members' "Total Personal Expense" values.
    
    Returns the sum of positive member values (what members actually paid out).
    """
    if df.empty:
        return 0.0
    
    # Get member columns
    dm = get_data_manager()
    member_cols = dm.get_member_columns(df)
    
    if not member_cols:
        return 0.0
    
    # Sum all positive values (what members paid)
    total = 0.0
    for member_col in member_cols:
        if member_col in df.columns:
            # Sum only positive values (payments made) - convert to numeric first
            member_values = pd.to_numeric(df[member_col], errors='coerce').fillna(0)
            positive_values = member_values[member_values > 0].sum()
            total += positive_values
    
    return total

# Income calculation functions
def normalize_to_monthly(amount, frequency):
    """Convert any frequency to monthly amount"""
    multipliers = {
        'weekly': 4.33,
        'biweekly': 2.17,
        'monthly': 1,
        'annual': 1/12
    }
    return amount * multipliers.get(frequency, 1)

def calculate_net_income(amount, is_net, tax_rate):
    """Calculate net income if gross is provided"""
    if is_net:
        return amount
    return amount * (1 - tax_rate / 100)

def get_total_monthly_income(income_entries, as_of_date=None):
    """Calculate total monthly income as of specific date"""
    dm = get_data_manager()
    active_entries = dm.get_income_entries(active_only=True, as_of_date=as_of_date)
    
    total = 0
    for entry in active_entries:
        monthly = normalize_to_monthly(entry['amount'], entry['frequency'])
        net = calculate_net_income(monthly, entry['is_net'], entry.get('tax_rate', 0))
        # For now, assume all in ILS or handle currency later
        total += net
    
    return total

def render_group_selector():
    """Render group selector in sidebar"""
    groups_mgr = get_groups_manager()
    
    # Check if groups exist
    if not groups_mgr.has_groups():
        return
    
    groups = groups_mgr.get_all_groups()
    active_group = groups_mgr.get_active_group()
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📁 Expense Groups")
    
    # Group dropdown
    group_display = [f"{g['emoji']} {g['name']}" for g in groups]
    group_ids = [g['id'] for g in groups]
    
    try:
        current_index = group_ids.index(active_group['id'])
    except (ValueError, KeyError):
        current_index = 0
        groups_mgr.set_active_group(group_ids[0])
    
    selected_display = st.sidebar.selectbox(
        "Select Group",
        group_display,
        index=current_index,
        key="group_selector"
    )
    
    selected_index = group_display.index(selected_display)
    selected_id = group_ids[selected_index]
    
    # Handle group switching
    if selected_id != active_group['id']:
        # Store current page before switching
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 'Overview'
        
        groups_mgr.set_active_group(selected_id)
        st.cache_data.clear()
        st.rerun()
    
    # Quick actions
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("➕ New", use_container_width=True, key="new_group_btn"):
            st.session_state.show_new_group_modal = True
            st.rerun()
    with col2:
        if st.button("⚙️ Manage", use_container_width=True, key="manage_groups_btn"):
            st.session_state.navigate_to_manage_groups = True
            st.rerun()
    
    # Currency selector
    st.sidebar.markdown("---")
    st.sidebar.subheader("💱 Dashboard Currency")
    
    # Get dashboard-wide target currency from CurrencyManager
    currency_mgr = get_currency_manager()
    target_currency = currency_mgr.get_target_currency()
    
    if target_currency:
        # Show target currency with symbol
        symbol = get_currency_symbol(target_currency)
        currency_name = get_currency_name(target_currency)
        
        st.sidebar.info(f"""**{target_currency} {symbol}**  
{currency_name}

ℹ️ All amounts displayed in this currency  
💡 Change via Data Management → Database Import""")
    else:
        # Dashboard not initialized yet
        st.sidebar.warning("""⚠️ **Currency Not Set**

Import your first Splitwise file to set up your dashboard currency.""")

def get_available_currencies():
    """Get all currencies available in current group's data"""
    try:
        groups_mgr = get_groups_manager()
        if not groups_mgr.has_groups():
            return ['ILS']
        
        active_group = groups_mgr.get_active_group()
        group_path = groups_mgr.get_group_data_path(active_group['id'])
        dm = DataManager(group_path)
        
        data = dm.load_data()
        currencies = set()
        
        # Get currencies from transactions
        for txn in data.get('transactions', []):
            currency = txn.get('currency', 'ILS')
            if currency and pd.notna(currency):
                currencies.add(currency)
        
        # Always include ILS as an option
        currencies.add('ILS')
        
        return sorted(list(currencies))
    except:
        return ['ILS']

def render_member_filter(df):
    """Add member filter to sidebar for group data"""
    if df.empty:
        return df, None
    
    dm = get_data_manager()
    member_cols = dm.get_member_columns(df)
    
    if not member_cols:
        return df, None
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("👤 Member Filter")
    
    filter_mode = st.sidebar.radio(
        "View",
        options=["All Transactions", "Individual Member"],
        key="member_filter_mode"
    )
    
    if filter_mode == "Individual Member":
        selected_member = st.sidebar.selectbox(
            "Select Member",
            options=member_cols,
            key="selected_member"
        )
        
        # Filter to transactions where member is involved
        filtered_df = dm.get_member_transactions(df, selected_member, exclude_zero=True)
        
        # Show member's stats
        st.sidebar.markdown(f"**{selected_member}'s Stats:**")
        stats = dm.calculate_member_expenses(filtered_df, selected_member)
        
        if stats:
            st.sidebar.metric("Personal Expense", f"₪{stats['total_paid']:,.2f}")
            st.sidebar.metric("Owed to You", f"₪{stats['total_owed_to_them']:,.2f}")
            balance_label = "You Owe" if stats['net_balance'] > 0 else "Owed to You"
            st.sidebar.metric(balance_label, f"₪{abs(stats['net_balance']):,.2f}")
        
        return filtered_df, selected_member
    
    return df, None

def show_new_group_modal():
    """Modal for creating new group"""
    st.subheader("➕ Create New Group")
    
    with st.form("new_group_form"):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            group_name = st.text_input("Group Name *", placeholder="e.g., Wedding Expenses")
        
        with col2:
            group_emoji = st.text_input("Emoji", value="📁", max_chars=2)
        
        group_description = st.text_area(
            "Description",
            placeholder="Brief description of this expense group"
        )
        
        members_input = st.text_input(
            "Members (comma-separated)",
            placeholder="e.g., Person 1, Person 2"
        )
        
        submitted = st.form_submit_button("Create Group", type="primary")
        
        if submitted:
            if not group_name:
                st.error("Group name is required")
            else:
                try:
                    groups_mgr = get_groups_manager()
                    members = [m.strip() for m in members_input.split(',')] if members_input else []
                    
                    new_group = groups_mgr.create_group(
                        name=group_name,
                        description=group_description,
                        emoji=group_emoji,
                        members=members
                    )
                    
                    st.success(f"✅ Created group: {group_emoji} {group_name}")
                    st.session_state.show_new_group_modal = False
                    
                    # Switch to new group
                    groups_mgr.set_active_group(new_group['id'])
                    st.cache_data.clear()
                    st.rerun()
                except ValueError as e:
                    st.error(f"❌ {str(e)}")

def show_currency_selection(detected_currencies: list):
    """Show currency selection dialog for first import"""
    st.markdown("### 🎉 Welcome! Set Up Your Dashboard")
    
    st.info("📊 This is your first import! Please choose your target currency for display across all dashboard groups.")
    
    # Show detected currencies
    if detected_currencies:
        st.markdown("**Detected currencies in your import:**")
        for currency in detected_currencies:
            symbol = get_currency_symbol(currency)
            name = get_currency_name(currency)
            st.markdown(f"• **{currency}** ({name}) {symbol}")
    
    st.markdown("---")
    
    # Currency selection
    st.markdown("**Choose Target Currency:**")
    st.markdown("_All amounts across all groups will be displayed in this currency._")
    
    # Common currencies
    common_currencies = ['ILS', 'USD', 'EUR', 'GBP', 'JPY']
    # Add detected currencies that aren't in common list
    all_options = list(dict.fromkeys(detected_currencies + common_currencies))
    
    selected_currency = st.selectbox(
        "Select your target currency:",
        options=all_options,
        format_func=lambda c: f"{c} ({get_currency_name(c)}) {get_currency_symbol(c)}",
        index=0 if detected_currencies else all_options.index('ILS'),
        key="target_currency_selection"
    )
    
    st.markdown("---")
    st.markdown("**ℹ️ About Target Currency:**")
    st.markdown("""
    - All transactions will be converted to this currency for unified display
    - You can change it later by exporting and re-importing your database
    - **Recommended:** Choose the currency you use most often
    """)
    
    return selected_currency

def show_setup_wizard():
    """First-time setup wizard"""
    st.title("👋 Welcome to ExpenseInfo Dashboard!")
    st.markdown("Let's set up your expense tracking system.")
    
    # Check if we need currency initialization
    currency_mgr = get_currency_manager()
    needs_currency_init = not currency_mgr.is_initialized()
    
    # Initialize session state for upload flow
    if 'uploaded_df' not in st.session_state:
        st.session_state.uploaded_df = None
    if 'uploaded_currencies' not in st.session_state:
        st.session_state.uploaded_currencies = []
    if 'show_currency_selection' not in st.session_state:
        st.session_state.show_currency_selection = False
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Option 1: Import Existing Data")
        st.markdown("Upload your Splitwise export file to get started.")
        
        # Show currency selection if we have uploaded data
        if st.session_state.show_currency_selection and st.session_state.uploaded_df is not None:
            selected_currency = show_currency_selection(st.session_state.uploaded_currencies)
            
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Cancel", type="secondary"):
                    st.session_state.uploaded_df = None
                    st.session_state.show_currency_selection = False
                    st.rerun()
            
            with col_b:
                if st.button("Set Currency & Import", type="primary"):
                    with st.spinner("Initializing currency and importing data..."):
                        # Initialize currency
                        currency_mgr.initialize_dashboard(
                            selected_currency,
                            st.session_state.uploaded_currencies
                        )
                        
                        # Now import the data with currency conversion
                        dm = get_data_manager()
                        df = st.session_state.uploaded_df
                        transactions = convert_df_to_transactions(df, apply_currency_conversion=True)
                        result = dm.append_transactions(transactions)
                        
                        # Update group members if groups exist
                        groups_mgr = get_groups_manager()
                        if groups_mgr.has_groups():
                            active_group = groups_mgr.get_active_group()
                            update_group_members_from_data(active_group['id'], df)
                        
                        st.success(f"✅ Successfully imported {result['added']} transactions!")
                        if result['skipped'] > 0:
                            st.info(f"ℹ️ Skipped {result['skipped']} duplicate transactions")
                        
                        # Show currency conversion summary
                        show_import_rate_summary(df, transactions)
                        
                        # Clear session state
                        st.session_state.uploaded_df = None
                        st.session_state.show_currency_selection = False
                        
                        st.balloons()
                        time.sleep(2)  # Give user time to see the summary
                        st.rerun()
        else:
            # Regular upload form
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=['csv', 'xlsx', 'xls'],
                help="Upload CSV or Excel file. For Hebrew names, use Excel exported from Google Sheets.",
                key="setup_wizard_uploader"
            )
            
            if uploaded_file:
                file_type = 'csv' if uploaded_file.name.endswith('.csv') else 'excel'
                
                if st.button("Import Data", type="primary"):
                    with st.spinner("Processing your data..."):
                        df = load_data_from_file(uploaded_file, file_type)
                        
                        if df is not None and not df.empty:
                            # Detect currencies
                            detected_currencies = detect_currencies_in_df(df)
                            
                            if needs_currency_init:
                                # Store data and show currency selection
                                st.session_state.uploaded_df = df
                                st.session_state.uploaded_currencies = detected_currencies
                                st.session_state.show_currency_selection = True
                                st.rerun()
                            else:
                                # Currency already initialized, just import with conversion
                                dm = get_data_manager()
                                transactions = convert_df_to_transactions(df, apply_currency_conversion=True)
                                result = dm.append_transactions(transactions)
                                
                                # Update group members if groups exist
                                groups_mgr = get_groups_manager()
                                if groups_mgr.has_groups():
                                    active_group = groups_mgr.get_active_group()
                                    update_group_members_from_data(active_group['id'], df)
                                
                                st.success(f"✅ Successfully imported {result['added']} transactions!")
                                if result['skipped'] > 0:
                                    st.info(f"ℹ️ Skipped {result['skipped']} duplicate transactions")
                                
                                # Show currency conversion summary
                                show_import_rate_summary(df, transactions)
                                
                                st.balloons()
                                time.sleep(2)  # Give user time to see the summary
                                st.rerun()
    
    with col2:
        st.subheader("Option 2: Start Fresh")
        st.markdown("Begin with an empty dataset and add transactions manually later.")
        
        if st.button("Start Fresh"):
            dm = get_data_manager()
            # Create empty dataset
            empty_data = dm._create_empty_dataset()
            dm.save_data(empty_data)
            st.success("✅ Fresh dataset created!")
            st.rerun()
    
    st.markdown("---")
    st.info("💡 Your data is stored locally on your device. No internet connection needed!")

def show_data_management():
    """Data management page"""
    st.title("💾 Data Management")
    
    # Group selector at the top
    groups_mgr = get_groups_manager()
    groups = groups_mgr.get_all_groups()
    
    if not groups:
        st.warning("No groups available. Create a group in the Manage Groups page first.")
        return
    
    # Get current active group
    active_group = groups_mgr.get_active_group()
    active_group_id = active_group['id'] if active_group else groups[0]['id']
    
    # Find index of active group for default selection
    default_index = next((i for i, g in enumerate(groups) if g['id'] == active_group_id), 0)
    
    # Group selector
    st.markdown("### 📁 Select Group for Data Management")
    selected_group = st.selectbox(
        "Choose which group's data to manage:",
        options=groups,
        format_func=lambda g: f"{g['emoji']} {g['name']}",
        index=default_index,
        key="data_mgmt_group_selector"
    )
    
    # Get data manager for the selected group
    group_path = groups_mgr.get_group_data_path(selected_group['id'])
    dm = DataManager(group_path)
    data = dm.load_data()
    
    st.markdown("---")
    
    # Current dataset info
    st.subheader(f"📊 Current Dataset - {selected_group['emoji']} {selected_group['name']}")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Transactions", data['metadata']['total_transactions'])
    
    with col2:
        date_range = data['metadata']['date_range']
        if date_range['earliest'] and date_range['latest']:
            st.metric("Date Range", f"{date_range['earliest'][:10]} to {date_range['latest'][:10]}")
        else:
            st.metric("Date Range", "No data")
    
    with col3:
        last_updated = data['metadata']['last_updated']
        st.metric("Last Updated", last_updated[:10] if last_updated else "N/A")
    
    with col4:
        if dm.data_file.exists():
            size_kb = dm.data_file.stat().st_size / 1024
            st.metric("Storage Size", f"{size_kb:.1f} KB")
        else:
            st.metric("Storage Size", "0 KB")
    
    st.markdown("---")
    
    # Import section
    st.subheader("📥 Import New Data")
    
    uploaded_file = st.file_uploader(
        "Upload file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload CSV or Excel file with expense data"
    )
    
    if uploaded_file:
        file_type = 'csv' if uploaded_file.name.endswith('.csv') else 'excel'
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"📄 File: {uploaded_file.name}")
        
        with col2:
            duplicate_handling = st.radio(
                "Duplicate handling:",
                ["Skip duplicates", "Keep all"],
                horizontal=True
            )
        
        if st.button("Import and Process", type="primary"):
            with st.spinner("Processing file and converting currencies..."):
                df = load_data_from_file(uploaded_file, file_type)
                
                if df is not None and not df.empty:
                    transactions = convert_df_to_transactions(df, apply_currency_conversion=True)
                    result = dm.append_transactions(transactions)
                    
                    # Update group members from uploaded data
                    updated_members = update_group_members_from_data(selected_group['id'], df)
                    if updated_members:
                        st.info(f"📝 Updated group members: {', '.join(updated_members)}")
                    
                    st.success(f"✅ Added {result['added']} new transactions to {selected_group['emoji']} {selected_group['name']}")
                    
                    if result['skipped'] > 0:
                        st.warning(f"⚠️ Skipped {result['skipped']} duplicate transactions")
                        
                        with st.expander("View duplicate transactions"):
                            if result['duplicates']:
                                dup_df = pd.DataFrame(result['duplicates'])
                                st.dataframe(dup_df[['date', 'description', 'cost', 'category']])
                    
                    # Show currency conversion summary
                    show_import_rate_summary(df, transactions)
                    
                    # Clear cache and reload
                    st.cache_data.clear()
                    time.sleep(3)  # Give user time to see the summary
                    st.rerun()
    
    st.markdown("---")
    
    # Export section
    st.subheader("📤 Export Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Export to CSV"):
            df = dm.get_dataframe()
            if not df.empty:
                csv = df.to_csv(index=False)
                st.download_button(
                    "📥 Download CSV",
                    csv,
                    f"transactions_{datetime.now().strftime('%Y%m%d')}.csv",
                    "text/csv",
                    key="download_csv"
                )
            else:
                st.info("No data to export")
    
    with col2:
        if st.button("Export to Excel"):
            df = dm.get_dataframe()
            if not df.empty:
                excel_buffer = io.BytesIO()
                df.to_excel(excel_buffer, index=False, engine='openpyxl')
                st.download_button(
                    "📥 Download Excel",
                    excel_buffer.getvalue(),
                    f"transactions_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_excel"
                )
            else:
                st.info("No data to export")
    
    with col3:
        if st.button("Export to JSON"):
            data_json = json.dumps(data, ensure_ascii=False, indent=2)
            st.download_button(
                "📥 Download JSON",
                data_json,
                f"transactions_{datetime.now().strftime('%Y%m%d')}.json",
                "application/json",
                key="download_json"
            )
    
    st.markdown("---")
    
    # Dashboard Export/Import section
    st.subheader("💾 Full Dashboard Backup & Currency Change")
    
    st.info("📦 **Dashboard Export/Import** allows you to backup your entire dashboard or change your target currency. "
            "All groups, transactions, income data, and settings are included.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📥 Export Dashboard")
        st.write("Create a complete backup of your dashboard including all groups and settings.")
        
        if st.button("📦 Export Full Dashboard", type="primary"):
            try:
                import zipfile
                import tempfile
                from pathlib import Path
                
                # Create temporary zip file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
                    zip_path = tmp_file.name
                
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    # Add groups config
                    groups_config_path = Path("user_data/groups/groups_config.json")
                    if groups_config_path.exists():
                        zipf.write(groups_config_path, "groups_config.json")
                    
                    # Add currency settings
                    currency_settings_path = Path("user_data/dashboard_currency_settings.json")
                    if currency_settings_path.exists():
                        zipf.write(currency_settings_path, "dashboard_currency_settings.json")
                    
                    # Add all group data
                    for group in groups:
                        group_id = group['id']
                        group_dir = Path(f"user_data/groups/{group_id}")
                        
                        if group_dir.exists():
                            # Add transactions
                            trans_file = group_dir / "transactions.json"
                            if trans_file.exists():
                                zipf.write(trans_file, f"groups/{group_id}/transactions.json")
                            
                            # Add income data
                            income_file = group_dir / "income_data.json"
                            if income_file.exists():
                                zipf.write(income_file, f"groups/{group_id}/income_data.json")
                    
                    # Add export metadata
                    export_metadata = {
                        "export_timestamp": datetime.now().isoformat(),
                        "export_version": "1.0",
                        "total_groups": len(groups),
                        "currency_system_initialized": get_currency_manager().is_initialized()
                    }
                    
                    # Write metadata to temp file then add to zip
                    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as meta_file:
                        json.dump(export_metadata, meta_file, indent=2)
                        meta_path = meta_file.name
                    
                    zipf.write(meta_path, "export_metadata.json")
                    Path(meta_path).unlink()  # Clean up temp file
                
                # Read zip file for download
                with open(zip_path, 'rb') as f:
                    zip_data = f.read()
                
                # Clean up temp zip
                Path(zip_path).unlink()
                
                # Offer download
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                st.download_button(
                    "📥 Download Dashboard Backup",
                    zip_data,
                    f"dashboard_export_{timestamp}.zip",
                    "application/zip",
                    key="download_dashboard"
                )
                
                st.success(f"✅ Dashboard exported successfully! ({len(groups)} groups included)")
                
            except Exception as e:
                st.error(f"❌ Error exporting dashboard: {e}")
    
    with col2:
        st.markdown("#### 📤 Import Dashboard")
        st.write("Restore from backup or change target currency. **Warning:** This replaces your current dashboard!")
        
        uploaded_dashboard = st.file_uploader(
            "Upload Dashboard ZIP",
            type=['zip'],
            help="Upload a previously exported dashboard backup",
            key="dashboard_upload"
        )
        
        if uploaded_dashboard:
            try:
                import zipfile
                import tempfile
                
                # Extract and analyze ZIP
                with tempfile.TemporaryDirectory() as tmp_dir:
                    tmp_path = Path(tmp_dir)
                    
                    # Extract ZIP
                    with zipfile.ZipFile(uploaded_dashboard, 'r') as zipf:
                        zipf.extractall(tmp_path)
                    
                    # Read metadata
                    metadata_file = tmp_path / "export_metadata.json"
                    if metadata_file.exists():
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        
                        st.success(f"✅ Dashboard analyzed: {metadata.get('total_groups', 0)} groups found")
                        
                        # Check for currency settings
                        currency_file = tmp_path / "dashboard_currency_settings.json"
                        has_currency = currency_file.exists()
                        
                        if has_currency:
                            with open(currency_file, 'r') as f:
                                currency_data = json.load(f)
                                original_currency = currency_data.get('target_currency', 'Unknown')
                            
                            st.info(f"📋 Original Dashboard Currency: **{original_currency}**")
                            
                            # Currency change option
                            st.markdown("**⚙️ Import Options:**")
                            
                            change_currency = st.checkbox(
                                "Change target currency during import",
                                help="Re-convert all transactions to a new target currency"
                            )
                            
                            if change_currency:
                                st.warning("⚠️ **Currency Conversion:** All amounts will be re-converted from their original currencies.")
                                
                                # Currency selector
                                available_currencies = ['ILS', 'USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'BGN']
                                new_currency = st.selectbox(
                                    "Select NEW target currency:",
                                    options=available_currencies,
                                    index=0 if original_currency not in available_currencies else available_currencies.index(original_currency)
                                )
                                
                                if new_currency != original_currency:
                                    st.info(f"💱 Will convert from **{original_currency}** to **{new_currency}**")
                            else:
                                new_currency = original_currency
                        else:
                            st.warning("⚠️ No currency settings found in backup")
                            new_currency = None
                        
                        # Confirmation and import
                        st.markdown("---")
                        confirm_import = st.checkbox(
                            "⚠️ I understand this will REPLACE my current dashboard",
                            help="This action cannot be undone. Current dashboard will be backed up first."
                        )
                        
                        if confirm_import and st.button("🔄 Import Dashboard", type="primary"):
                            with st.spinner("Importing dashboard..."):
                                try:
                                    # Create backup of current dashboard first
                                    st.info("📦 Creating backup of current dashboard...")
                                    
                                    # Import groups config
                                    groups_config_src = tmp_path / "groups_config.json"
                                    groups_config_dst = Path("user_data/groups/groups_config.json")
                                    
                                    if groups_config_src.exists():
                                        import shutil
                                        groups_config_dst.parent.mkdir(parents=True, exist_ok=True)
                                        shutil.copy(groups_config_src, groups_config_dst)
                                    
                                    # Import currency settings
                                    if has_currency and new_currency:
                                        currency_src = tmp_path / "dashboard_currency_settings.json"
                                        currency_dst = Path("user_data/dashboard_currency_settings.json")
                                        
                                        with open(currency_src, 'r') as f:
                                            curr_settings = json.load(f)
                                        
                                        # Update target currency if changed
                                        curr_settings['target_currency'] = new_currency
                                        
                                        with open(currency_dst, 'w') as f:
                                            json.dump(curr_settings, f, indent=2)
                                    
                                    # Import group data
                                    groups_dir = tmp_path / "groups"
                                    if groups_dir.exists():
                                        for group_dir in groups_dir.iterdir():
                                            if group_dir.is_dir():
                                                group_id = group_dir.name
                                                dst_dir = Path(f"user_data/groups/{group_id}")
                                                dst_dir.mkdir(parents=True, exist_ok=True)
                                                
                                                # Copy transactions
                                                trans_src = group_dir / "transactions.json"
                                                if trans_src.exists():
                                                    import shutil
                                                    shutil.copy(trans_src, dst_dir / "transactions.json")
                                                
                                                # Copy income data
                                                income_src = group_dir / "income_data.json"
                                                if income_src.exists():
                                                    import shutil
                                                    shutil.copy(income_src, dst_dir / "income_data.json")
                                    
                                    st.success("✅ Dashboard imported successfully!")
                                    st.info("🔄 Reloading application...")
                                    
                                    # Clear all caches
                                    st.cache_data.clear()
                                    st.cache_resource.clear()
                                    
                                    # Rerun
                                    st.rerun()
                                    
                                except Exception as e:
                                    st.error(f"❌ Error importing dashboard: {e}")
                    else:
                        st.error("❌ Invalid dashboard backup: missing metadata")
                        
            except Exception as e:
                st.error(f"❌ Error reading dashboard backup: {e}")
    
    st.markdown("---")
    
    # Backups section
    st.subheader("💾 Group Backups")
    
    backups = dm.get_backups()
    
    if backups:
        backup_df = pd.DataFrame(backups)[['date', 'time', 'size', 'file']]
        st.dataframe(backup_df, use_container_width=True, hide_index=True)
        
        selected_backup = st.selectbox(
            "Select backup to restore:",
            options=[b['file'] for b in backups],
            key="backup_select"
        )
        
        if st.button("Restore Selected Backup", type="secondary"):
            backup_to_restore = next(b for b in backups if b['file'] == selected_backup)
            
            if st.checkbox(f"⚠️ Confirm: Restore backup from {backup_to_restore['date']} {backup_to_restore['time']}?"):
                dm.restore_backup(backup_to_restore['path'])
                st.success("✅ Backup restored successfully!")
                st.cache_data.clear()
                st.rerun()
    else:
        st.info("No backups available yet. Backups are created automatically when data is saved.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Create Manual Backup"):
            dm._create_backup()
            st.success("✅ Backup created!")
            st.rerun()
    
    st.markdown("---")
    
    # Danger zone
    st.subheader("⚠️ Danger Zone")
    
    with st.expander("Clear All Data"):
        st.warning("This action cannot be undone! All your data will be permanently deleted.")
        
        confirm_text = st.text_input("Type 'DELETE' to confirm:")
        
        if st.button("Clear All Data", type="primary", disabled=(confirm_text != "DELETE")):
            dm.clear_all_data()
            st.success("All data cleared!")
            st.cache_data.clear()
            st.rerun()

def show_overview(df):
    """Show overview page with summary and charts"""
    # Calculate Total Spending as sum of member expenses (NO exclusions - matches Combined Analytics)
    total_spending = calculate_total_spending(df)
    
    # Get expenses dataframe for plots (exclude Payment and reimbursements for display)
    df_expenses = exclude_payments_and_reimbursements(df)
    
    # Get member columns for calculations
    dm = get_data_manager()
    member_cols = dm.get_member_columns(df) if not df.empty else []
    
    # Historic monthly average - use member expenses (no exclusions)
    if not df.empty and member_cols:
        df_copy = df.copy()
        df_copy['YearMonth'] = df_copy['Date'].dt.to_period('M')
        
        # Calculate monthly totals as sum of member expenses
        monthly_data = []
        for year_month, group in df_copy.groupby('YearMonth'):
            month_total = 0.0
            for member_col in member_cols:
                if member_col in group.columns:
                    member_values = pd.to_numeric(group[member_col], errors='coerce').fillna(0)
                    month_total += member_values[member_values > 0].sum()
            monthly_data.append({'YearMonth': year_month, 'Total': month_total})
        
        if monthly_data:
            monthly_df = pd.DataFrame(monthly_data)
            historic_avg = monthly_df['Total'].mean()
        else:
            historic_avg = 0
    else:
        historic_avg = 0
    
    # Current month spending - use Cost from filtered data (matches Trends & History tab)
    current_month = datetime.now().replace(day=1).date()
    current_month_data = df_expenses[df_expenses['Date'].dt.date >= current_month]
    if not current_month_data.empty:
        current_month_spending = current_month_data['Cost'].sum()
    else:
        current_month_spending = 0
    
    # Get income data
    dm = get_data_manager()
    income_entries = dm.load_income_data().get('income_entries', [])
    total_monthly_income = get_total_monthly_income(income_entries)
    
    # Display metrics
    st.header("📊 Summary Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Spending",
            f"₪{total_spending:,.0f}"
        )
    
    with col2:
        st.metric(
            "Historic Monthly Average",
            f"₪{historic_avg:,.0f}"
        )
    
    with col3:
        delta = current_month_spending - historic_avg
        delta_pct = (delta / historic_avg * 100) if historic_avg > 0 else 0
        st.metric(
            "Current Month",
            f"₪{current_month_spending:,.0f}",
            f"{delta:+,.0f} ({delta_pct:+.1f}%)"
        )
    
    with col4:
        if total_monthly_income > 0:
            expense_ratio = (current_month_spending / total_monthly_income) * 100
            st.metric(
                "Expense Ratio",
                f"{expense_ratio:.1f}%",
                help="Current month expenses as % of monthly income"
            )
        else:
            st.metric(
                "Expense Ratio",
                "N/A",
                help="Add income sources in Income & Savings page"
            )
    
    # Show income vs expense comparison if income data exists
    if total_monthly_income > 0:
        st.markdown("---")
        st.subheader("💰 Income vs Expenses (Current Month)")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Monthly Income", f"₪{total_monthly_income:,.0f}")
        
        with col2:
            st.metric("Current Month Expenses", f"₪{current_month_spending:,.0f}")
        
        with col3:
            savings = total_monthly_income - current_month_spending
            savings_rate = (savings / total_monthly_income * 100) if total_monthly_income > 0 else 0
            st.metric(
                "Savings This Month",
                f"₪{savings:,.0f}",
                f"{savings_rate:.1f}% savings rate"
            )
    
    # Show multi-currency breakdown if currencies detected
    currency_mgr = get_currency_manager()
    if currency_mgr.is_initialized() and not df.empty:
        # Check if there are any transactions with original currency
        has_original_currency = 'original_currency' in df.columns and df['original_currency'].notna().any()
        
        if has_original_currency:
            st.markdown("---")
            st.subheader("💱 Multi-Currency Breakdown")
            
            # Calculate spending by original currency
            currency_spending = []
            target_currency = currency_mgr.get_target_currency()
            target_symbol = get_currency_symbol(target_currency)
            
            # Group by original currency
            for currency in df['original_currency'].dropna().unique():
                if currency and currency != target_currency:
                    # Get transactions in this currency
                    currency_txns = df[df['original_currency'] == currency]
                    
                    # Sum original amounts
                    original_total = currency_txns['original_cost'].sum()
                    
                    # Sum converted amounts (in target currency)
                    converted_total = currency_txns['Cost'].sum()
                    
                    # Count transactions
                    txn_count = len(currency_txns)
                    
                    currency_spending.append({
                        'currency': currency,
                        'symbol': get_currency_symbol(currency),
                        'original_total': original_total,
                        'converted_total': converted_total,
                        'count': txn_count
                    })
            
            # Also include target currency transactions
            target_txns = df[(df['original_currency'].isna()) | (df['original_currency'] == target_currency)]
            if not target_txns.empty:
                target_total = target_txns['Cost'].sum()
                currency_spending.append({
                    'currency': target_currency,
                    'symbol': target_symbol,
                    'original_total': target_total,
                    'converted_total': target_total,
                    'count': len(target_txns)
                })
            
            # Sort by converted total (descending)
            currency_spending = sorted(currency_spending, key=lambda x: x['converted_total'], reverse=True)
            
            if len(currency_spending) > 0:
                # Display cards in columns
                cols = st.columns(min(len(currency_spending), 4))
                
                grand_total = sum(c['converted_total'] for c in currency_spending)
                
                for idx, curr_data in enumerate(currency_spending):
                    col_idx = idx % 4
                    with cols[col_idx]:
                        percentage = (curr_data['converted_total'] / grand_total * 100) if grand_total > 0 else 0
                        
                        # Format original amount display
                        if curr_data['currency'] == target_currency:
                            amount_display = f"{target_symbol}{curr_data['converted_total']:,.0f}"
                        else:
                            amount_display = f"{target_symbol}{curr_data['converted_total']:,.0f}"
                            original_display = f"({curr_data['symbol']}{curr_data['original_total']:,.0f})"
                        
                        st.metric(
                            f"{curr_data['currency']} {curr_data['symbol']}",
                            amount_display,
                            f"{percentage:.1f}% • {curr_data['count']} txns"
                        )
                        
                        if curr_data['currency'] != target_currency:
                            st.caption(f"Original: {curr_data['symbol']}{curr_data['original_total']:,.0f} {curr_data['currency']}")
                
                # Foreign currency exposure summary
                foreign_total = sum(c['converted_total'] for c in currency_spending if c['currency'] != target_currency)
                if foreign_total > 0:
                    foreign_pct = (foreign_total / grand_total * 100) if grand_total > 0 else 0
                    st.info(f"💡 **Foreign Currency Exposure:** {target_symbol}{foreign_total:,.0f} ({foreign_pct:.1f}% of total spending)")
    
    st.markdown("---")
    
    # Tabs for different views
    tab1, tab2 = st.tabs(["📊 Category Breakdown", "📈 Trends & History"])
    
    with tab1:
        if not df_expenses.empty:
            # Month selector
            col1, col2 = st.columns([1, 3])
            
            with col1:
                # Get available months
                df_expenses_temp = df_expenses.copy()
                df_expenses_temp['YearMonth'] = df_expenses_temp['Date'].dt.to_period('M')
                available_months = sorted(df_expenses_temp['YearMonth'].unique(), reverse=True)
                
                # Create options: "All Time" + specific months
                month_options = ['All Time'] + [str(m) for m in available_months]
                
                selected_month = st.selectbox(
                    "Select Period",
                    options=month_options,
                    index=0,
                    key="category_breakdown_month"
                )
            
            # Filter data by selected month
            if selected_month == 'All Time':
                filtered_df = df_expenses
                title = 'Spending by Category (All Time)'
            else:
                filtered_df = df_expenses_temp[df_expenses_temp['YearMonth'] == selected_month].copy()
                title = f'Spending by Category ({selected_month})'
            
            if not filtered_df.empty:
                # Calculate category totals using Cost (for plots only)
                category_totals = filtered_df.groupby('Category')['Cost'].sum().reset_index()
                category_totals.columns = ['Category', 'Total']
                category_totals = category_totals.sort_values('Total', ascending=False)
                
                fig_pie = px.pie(
                    category_totals,
                    values='Total',
                    names='Category',
                    title=title,
                    hole=0.4
                )
                
                fig_pie.update_traces(
                    textposition='auto',
                    textinfo='percent+label'
                )
                
                fig_pie.update_layout(
                    showlegend=True,
                    legend=dict(
                        orientation="v",
                        yanchor="middle",
                        y=0.5,
                        xanchor="left",
                        x=1.02
                    )
                )
                
                st.plotly_chart(fig_pie, use_container_width=True)
                
                # Show total for selected period (this is for filtered expenses only)
                total = category_totals['Total'].sum()
                st.metric(f"Expenses Shown in Chart ({selected_month})", f"₪{total:,.0f}",
                         help="This shows expenses excluding Payment category and reimbursements")
            else:
                st.info(f"No expense data available for {selected_month}")
        else:
            st.info("No expense data available")
    
    with tab2:
        if not df_expenses.empty:
            # Calculate monthly spending using Cost (excluding Payment & reimbursements)
            df_timeline = df_expenses.copy()
            df_timeline['YearMonth'] = df_timeline['Date'].dt.to_period('M').astype(str)
            
            monthly_spending = df_timeline.groupby('YearMonth')['Cost'].sum().reset_index()
            monthly_spending.columns = ['Month', 'Total']
            
            fig_line = px.line(
                monthly_spending,
                x='Month',
                y='Total',
                title='Monthly Spending Trend',
                markers=True
            )
            
            fig_line.update_layout(
                xaxis_title="Month",
                yaxis_title="Amount (₪)",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("No expense data available")
    
    # Transaction details
    with st.expander("🔍 Transaction Details"):
        st.subheader("All Transactions")
        
        search = st.text_input("🔍 Search transactions", placeholder="Search by description, category...")
        
        if search:
            mask = (
                df['Description'].str.contains(search, case=False, na=False) |
                df['Category'].str.contains(search, case=False, na=False)
            )
            display_df = df[mask]
        else:
            display_df = df
        
        # Prepare display with original currency info
        display_data = display_df.copy()
        
        # Add formatted amount column that shows original currency if different
        if 'original_cost' in display_data.columns and 'original_currency' in display_data.columns:
            # Convert to dict records for easier processing
            dm = get_data_manager()
            data = dm.load_data()
            transactions = data.get('transactions', [])
            
            # Create a lookup dict by description and date for matching
            txn_lookup = {}
            for txn in transactions:
                key = (txn.get('date'), txn.get('description'))
                txn_lookup[key] = txn
            
            # Add formatted amount column
            def format_row_amount(row):
                key = (row['Date'].strftime('%Y-%m-%d') if hasattr(row['Date'], 'strftime') else str(row['Date']), 
                       row['Description'])
                txn = txn_lookup.get(key, row.to_dict())
                return format_amount_with_original(txn, row.get('Currency', 'ILS'))
            
            display_data['Amount'] = display_data.apply(format_row_amount, axis=1)
            columns_to_show = ['Date', 'Description', 'Category', 'Amount']
        else:
            # No original currency data, just format regular amount
            display_data['Amount'] = display_data.apply(
                lambda row: format_currency_amount(row['Cost'], row.get('Currency', 'ILS')), 
                axis=1
            )
            columns_to_show = ['Date', 'Description', 'Category', 'Amount']
        
        st.dataframe(
            display_data[columns_to_show].sort_values('Date', ascending=False),
            use_container_width=True,
            hide_index=True
        )

def show_income_tracking():
    """Income tracking and savings page"""
    st.title("💰 Income & Savings")
    
    dm = get_data_manager()
    income_data = dm.load_income_data()
    income_entries = income_data.get('income_entries', [])
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_monthly_income = get_total_monthly_income(income_entries)
        st.metric("Total Monthly Income", f"₪{total_monthly_income:,.0f}")
    
    with col2:
        active_sources = len(dm.get_income_entries(active_only=True))
        total_sources = len(income_entries)
        st.metric("Active Income Sources", f"{active_sources} / {total_sources}")
    
    with col3:
        # Calculate expense ratio if transaction data exists
        if dm.data_exists():
            df = dm.get_dataframe()
            if not df.empty:
                # Get current month expenses
                now = datetime.now()
                current_month_df = df[
                    (df['Date'].dt.year == now.year) & 
                    (df['Date'].dt.month == now.month)
                ]
                df_expenses = exclude_payments_and_reimbursements(current_month_df)
                monthly_expenses = df_expenses['Cost'].sum()
                
                if total_monthly_income > 0:
                    expense_ratio = (monthly_expenses / total_monthly_income) * 100
                    st.metric("Expense Ratio (This Month)", f"{expense_ratio:.1f}%")
                else:
                    st.metric("Expense Ratio (This Month)", "N/A")
            else:
                st.metric("Expense Ratio (This Month)", "N/A")
        else:
            st.metric("Expense Ratio (This Month)", "N/A")
    
    st.markdown("---")
    
    # Add income source button
    if st.button("➕ Add Income Source", type="primary"):
        st.session_state['show_income_modal'] = True
        st.session_state['edit_income_id'] = None
    
    # Income entry modal
    if st.session_state.get('show_income_modal', False):
        show_income_entry_modal(dm, st.session_state.get('edit_income_id'))
    
    # Display income sources
    st.subheader("Current Income Sources")
    
    if income_entries:
        # Group by member
        members = {}
        for entry in income_entries:
            member_name = entry.get('member_name', 'Unknown')
            if member_name not in members:
                members[member_name] = []
            members[member_name].append(entry)
        
        for member_name, member_entries in members.items():
            st.markdown(f"### {member_name}")
            
            for entry in member_entries:
                # Check if active
                is_active = entry in dm.get_income_entries(active_only=True)
                status_icon = "✅" if is_active else "⏸️"
                
                # Calculate monthly amount
                monthly = normalize_to_monthly(entry['amount'], entry['frequency'])
                net = calculate_net_income(monthly, entry['is_net'], entry.get('tax_rate', 0))
                
                # Display income source card
                with st.container():
                    col1, col2 = st.columns([4, 1])
                    
                    with col1:
                        st.markdown(f"**{status_icon} {entry['source']}** - {entry['category']}")
                        income_type = "Net" if entry['is_net'] else f"Gross (Tax: {entry.get('tax_rate', 0)}%)"
                        st.markdown(f"₪{net:,.0f}/month ({income_type})")
                        st.caption(f"Started: {entry['start_date']}" + 
                                 (f" | Ended: {entry['end_date']}" if entry.get('end_date') else " | Ongoing"))
                        if entry.get('notes'):
                            st.caption(f"📝 {entry['notes']}")
                    
                    with col2:
                        col_edit, col_delete = st.columns(2)
                        with col_edit:
                            if st.button("✏️", key=f"edit_{entry['id']}", help="Edit"):
                                st.session_state['show_income_modal'] = True
                                st.session_state['edit_income_id'] = entry['id']
                                st.rerun()
                        with col_delete:
                            if st.button("🗑️", key=f"delete_{entry['id']}", help="Delete"):
                                if dm.delete_income_entry(entry['id']):
                                    st.success("Income source deleted!")
                                    st.rerun()
                    
                    st.markdown("---")
    else:
        st.info("No income sources added yet. Click 'Add Income Source' to get started!")
    
    # Income breakdown visualization
    if income_entries:
        st.subheader("📊 Income Breakdown")
        
        tab1, tab2, tab3 = st.tabs(["By Source", "By Category", "By Member"])
        
        with tab1:
            # Income by source
            active_entries = dm.get_income_entries(active_only=True)
            if active_entries:
                income_by_source = []
                for entry in active_entries:
                    monthly = normalize_to_monthly(entry['amount'], entry['frequency'])
                    net = calculate_net_income(monthly, entry['is_net'], entry.get('tax_rate', 0))
                    income_by_source.append({
                        'Source': entry['source'],
                        'Amount': net
                    })
                
                df_source = pd.DataFrame(income_by_source)
                fig_source = px.pie(
                    df_source,
                    values='Amount',
                    names='Source',
                    title='Income Distribution by Source'
                )
                st.plotly_chart(fig_source, use_container_width=True)
        
        with tab2:
            # Income by category
            active_entries = dm.get_income_entries(active_only=True)
            if active_entries:
                income_by_category = {}
                for entry in active_entries:
                    monthly = normalize_to_monthly(entry['amount'], entry['frequency'])
                    net = calculate_net_income(monthly, entry['is_net'], entry.get('tax_rate', 0))
                    category = entry['category']
                    income_by_category[category] = income_by_category.get(category, 0) + net
                
                df_category = pd.DataFrame([
                    {'Category': k, 'Amount': v}
                    for k, v in income_by_category.items()
                ])
                
                fig_category = px.bar(
                    df_category,
                    x='Category',
                    y='Amount',
                    title='Income by Category'
                )
                st.plotly_chart(fig_category, use_container_width=True)
        
        with tab3:
            # Income by member
            active_entries = dm.get_income_entries(active_only=True)
            if active_entries:
                income_by_member = {}
                for entry in active_entries:
                    monthly = normalize_to_monthly(entry['amount'], entry['frequency'])
                    net = calculate_net_income(monthly, entry['is_net'], entry.get('tax_rate', 0))
                    member = entry.get('member_name', 'Unknown')
                    income_by_member[member] = income_by_member.get(member, 0) + net
                
                df_member = pd.DataFrame([
                    {'Member': k, 'Amount': v}
                    for k, v in income_by_member.items()
                ])
                
                fig_member = px.bar(
                    df_member,
                    x='Member',
                    y='Amount',
                    title='Income by Household Member'
                )
                st.plotly_chart(fig_member, use_container_width=True)

def show_income_entry_modal(dm, edit_id=None):
    """Modal for adding/editing income entry"""
    # Get existing entry if editing
    existing_entry = None
    if edit_id:
        income_data = dm.load_income_data()
        existing_entry = next((e for e in income_data['income_entries'] if e['id'] == edit_id), None)
    
    modal_title = "Edit Income Source" if existing_entry else "Add Income Source"
    
    with st.form("income_entry_form", clear_on_submit=True):
        st.subheader(modal_title)
        
        col1, col2 = st.columns(2)
        
        with col1:
            member_name = st.text_input(
                "Member Name",
                value=existing_entry['member_name'] if existing_entry else "",
                help="Name of the household member receiving this income"
            )
            
            source = st.text_input(
                "Income Source",
                value=existing_entry['source'] if existing_entry else "",
                placeholder="e.g., Salary - Company A",
                help="Description of the income source"
            )
            
            category = st.selectbox(
                "Category",
                options=["Salary", "Freelance", "Investment", "Rental", "Other"],
                index=["Salary", "Freelance", "Investment", "Rental", "Other"].index(existing_entry['category']) if existing_entry else 0
            )
        
        with col2:
            amount = st.number_input(
                "Amount",
                min_value=0.0,
                value=float(existing_entry['amount']) if existing_entry else 0.0,
                step=100.0
            )
            
            currency = st.selectbox(
                "Currency",
                options=["ILS", "USD", "EUR"],
                index=["ILS", "USD", "EUR"].index(existing_entry['currency']) if existing_entry else 0
            )
            
            frequency = st.selectbox(
                "Frequency",
                options=["monthly", "annual", "weekly", "biweekly"],
                index=["monthly", "annual", "weekly", "biweekly"].index(existing_entry['frequency']) if existing_entry else 0
            )
        
        col3, col4 = st.columns(2)
        
        with col3:
            is_net = st.radio(
                "Income Type",
                options=[True, False],
                format_func=lambda x: "Net (After Tax)" if x else "Gross (Before Tax)",
                index=0 if (existing_entry and existing_entry['is_net']) else 1 if existing_entry else 0
            )
        
        with col4:
            tax_rate = st.number_input(
                "Tax Rate (%)",
                min_value=0.0,
                max_value=100.0,
                value=float(existing_entry.get('tax_rate', 0)) if existing_entry else 0.0,
                step=0.5,
                disabled=is_net,
                help="Only needed for gross income"
            )
        
        col5, col6 = st.columns(2)
        
        with col5:
            start_date = st.date_input(
                "Start Date",
                value=datetime.fromisoformat(existing_entry['start_date']).date() if existing_entry else datetime.now().date()
            )
        
        with col6:
            has_end_date = st.checkbox(
                "Has End Date",
                value=bool(existing_entry and existing_entry.get('end_date'))
            )
            
            if has_end_date:
                end_date = st.date_input(
                    "End Date",
                    value=datetime.fromisoformat(existing_entry['end_date']).date() if existing_entry and existing_entry.get('end_date') else datetime.now().date()
                )
            else:
                end_date = None
        
        notes = st.text_area(
            "Notes",
            value=existing_entry.get('notes', '') if existing_entry else "",
            placeholder="Optional notes about this income source"
        )
        
        col_save, col_cancel = st.columns(2)
        
        with col_save:
            submitted = st.form_submit_button("💾 Save", type="primary", use_container_width=True)
        
        with col_cancel:
            cancelled = st.form_submit_button("❌ Cancel", use_container_width=True)
        
        if submitted:
            # Validate inputs
            if not member_name or not source or amount <= 0:
                st.error("Please fill in all required fields with valid values")
            else:
                # Create income entry
                income_entry = {
                    'member_name': member_name,
                    'source': source,
                    'amount': amount,
                    'currency': currency,
                    'frequency': frequency,
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat() if end_date else None,
                    'is_net': is_net,
                    'tax_rate': tax_rate if not is_net else 0.0,
                    'category': category,
                    'notes': notes
                }
                
                if existing_entry:
                    # Update existing
                    dm.update_income_entry(edit_id, income_entry)
                    st.success("Income source updated!")
                else:
                    # Add new
                    dm.add_income_entry(income_entry)
                    st.success("Income source added!")
                
                st.session_state['show_income_modal'] = False
                st.session_state['edit_income_id'] = None
                st.rerun()
        
        if cancelled:
            st.session_state['show_income_modal'] = False
            st.session_state['edit_income_id'] = None
            st.rerun()

def show_combined_analytics():
    """Combined analytics across multiple groups"""
    st.title("📊 Combined Groups Analytics")
    
    groups_mgr = get_groups_manager()
    groups = groups_mgr.get_all_groups()
    
    if not groups:
        st.warning("No groups available. Create a group first.")
        return
    
    # Multi-select for groups
    selected_group_ids = st.multiselect(
        "Select Groups to Analyze",
        options=[g['id'] for g in groups],
        format_func=lambda gid: f"{next((g['emoji'] for g in groups if g['id']==gid), '📁')} {next((g['name'] for g in groups if g['id']==gid), 'Unknown')}",
        default=[groups[0]['id']]
    )
    
    if not selected_group_ids:
        st.warning("Please select at least one group to view analytics")
        return
    
    # Load data from selected groups
    all_data = []
    for group_id in selected_group_ids:
        group = next((g for g in groups if g['id'] == group_id), None)
        if not group:
            continue
        
        group_path = groups_mgr.get_group_data_path(group_id)
        dm = DataManager(group_path)
        df = dm.get_dataframe()
        
        if not df.empty:
            df['Group'] = group['name']
            df['GroupEmoji'] = group['emoji']
            all_data.append(df)
    
    if not all_data:
        st.info("No data available in selected groups")
        return
    
    combined_df_original = pd.concat(all_data, ignore_index=True)
    
    # For plots, exclude Payment/Settlement and reimbursement transactions (same as Overview)
    combined_df = exclude_payments_and_reimbursements(combined_df_original)
    
    # Get all members across selected groups
    all_members = groups_mgr.get_all_members_across_groups(selected_group_ids)
    
    # Member view mode
    st.markdown("---")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        view_mode = st.radio(
            "View Mode",
            options=["All Group Expenses", "Individual Member View"],
            horizontal=True
        )
    
    selected_member = None
    if view_mode == "Individual Member View":
        with col2:
            if all_members:
                selected_member = st.selectbox("Select Member", options=all_members)
    
    st.markdown("---")
    
    # Calculate metrics based on view mode
    if view_mode == "All Group Expenses":
        # Date range selection
        st.subheader("📅 Date Range")
        col_date1, col_date2 = st.columns(2)
        
        # Get min/max dates from combined data
        min_date = combined_df['Date'].min().date()
        max_date = combined_df['Date'].max().date()
        
        with col_date1:
            start_date = st.date_input(
                "Start Date",
                value=min_date,
                min_value=min_date,
                max_value=max_date,
                key="combined_all_start_date"
            )
        
        with col_date2:
            end_date = st.date_input(
                "End Date",
                value=max_date,
                min_value=min_date,
                max_value=max_date,
                key="combined_all_end_date"
            )
        
        # Apply date range filter
        combined_df_filtered = combined_df[
            (combined_df['Date'].dt.date >= start_date) & 
            (combined_df['Date'].dt.date <= end_date)
        ].copy()
        
        combined_df_original_filtered = combined_df_original[
            (combined_df_original['Date'].dt.date >= start_date) & 
            (combined_df_original['Date'].dt.date <= end_date)
        ].copy()
        
        # Month filter for all plots (except timeline) - within selected date range
        combined_df_filtered['YearMonth'] = combined_df_filtered['Date'].dt.to_period('M')
        available_months_all = sorted(combined_df_filtered['YearMonth'].unique(), reverse=True)
        
        month_options_all = ['All Time'] + [str(m) for m in available_months_all]
        selected_month_all = st.selectbox(
            "Filter by Month (within selected date range)",
            options=month_options_all,
            key="combined_all_month_filter"
        )
        
        # Apply month filter if selected
        if selected_month_all != 'All Time':
            combined_df_filtered = combined_df_filtered[combined_df_filtered['YearMonth'] == pd.Period(selected_month_all)]
            combined_df_original_filtered = combined_df_original_filtered[
                combined_df_original_filtered['Date'].dt.to_period('M') == pd.Period(selected_month_all)
            ]
        
        # Show metrics for all expenses
        col1, col2, col3, col4 = st.columns(4)
        
        # Calculate Total Spent as sum of member expenses from filtered data
        dm = get_data_manager()
        member_cols = dm.get_member_columns(combined_df_original_filtered) if not combined_df_original_filtered.empty else []
        
        total_spent = 0.0
        if member_cols:
            for member_col in member_cols:
                if member_col in combined_df_original_filtered.columns:
                    # Convert to numeric to avoid string comparison errors
                    member_values = pd.to_numeric(combined_df_original_filtered[member_col], errors='coerce').fillna(0)
                    total_spent += member_values[member_values > 0].sum()
        
        with col1:
            st.metric("Total Groups", len(selected_group_ids))
        with col2:
            st.metric("Total Transactions", len(combined_df_filtered))
        with col3:
            st.metric("Total Spent", f"₪{total_spent:,.2f}")
        with col4:
            if not combined_df_filtered.empty:
                date_range_str = f"{combined_df_filtered['Date'].min().date()} to {combined_df_filtered['Date'].max().date()}"
            else:
                date_range_str = "No data"
            st.metric("Date Range", date_range_str)
        
        # Visualizations
        st.markdown("---")
        
        # For plots, exclude Payment and reimbursements
        combined_df_for_plots = exclude_payments_and_reimbursements(combined_df_filtered)
        
        # Spending by group
        st.subheader("💰 Spending by Group")
        
        # Calculate spending by group using Cost
        group_spending = combined_df_for_plots.groupby(['Group', 'GroupEmoji'])['Cost'].sum().reset_index()
        group_spending.columns = ['Group', 'GroupEmoji', 'Total']
        group_spending['Display'] = group_spending['GroupEmoji'] + ' ' + group_spending['Group']
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_pie = px.pie(
                group_spending,
                values='Total',
                names='Display',
                title=f'Spending Distribution by Group{" - " + selected_month_all if selected_month_all != "All Time" else ""}'
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            fig_bar = px.bar(
                group_spending,
                x='Display',
                y='Total',
                title=f'Total Spending by Group{" - " + selected_month_all if selected_month_all != "All Time" else ""}'
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Category breakdown
        st.subheader("📊 Category Breakdown Across Groups")
        
        # Calculate category totals using Cost
        category_data = combined_df_for_plots.groupby('Category')['Cost'].sum().reset_index()
        category_data.columns = ['Category', 'Total']
        category_data = category_data.sort_values('Total', ascending=False).head(10)
        
        fig_categories = px.bar(
            category_data,
            x='Category',
            y='Total',
            color='Category',
            title=f'Top 10 Categories Across All Groups{" - " + selected_month_all if selected_month_all != "All Time" else ""}'
        )
        st.plotly_chart(fig_categories, use_container_width=True)
        
        # Timeline
        st.subheader("📈 Timeline Across Groups")
        
        # Calculate timeline data using Cost (Payment/reimbursements already excluded in combined_df)
        combined_df_timeline = combined_df.copy()
        combined_df_timeline['YearMonth'] = combined_df_timeline['Date'].dt.to_period('M').astype(str)
        
        timeline_data = combined_df_timeline.groupby(['YearMonth', 'Group'])['Cost'].sum().reset_index()
        timeline_data.columns = ['YearMonth', 'Group', 'Total']
        
        fig_timeline = px.line(
            timeline_data,
            x='YearMonth',
            y='Total',
            color='Group',
            title='Monthly Spending by Group'
        )
        st.plotly_chart(fig_timeline, use_container_width=True)
    
    else:
        # Individual member view
        if not selected_member:
            st.warning("Please select a member")
            return
        
        user_calc = UserExpenseCalculator(groups_mgr)
        member_data = user_calc.calculate_user_total_expense(selected_group_ids, selected_member)
        
        # Store original all-time metrics (unaffected by filters)
        total_expense_alltime = member_data['total_expense']
        net_balance_alltime = member_data['net_balance']
        
        # Member metrics - All Time (unaffected by date range)
        st.subheader(f"📊 {selected_member}'s Overall Summary")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Personal Expense (All Time)", f"₪{total_expense_alltime:,.2f}", 
                     help="Total amount this person actually paid across all transactions")
        with col2:
            if net_balance_alltime > 0:
                # Positive net = paid more than owed = they are OWED money
                balance_label = "They Are Owed (All Time)"
                balance_help = "This person is owed money (paid more than their share)"
            elif net_balance_alltime < 0:
                # Negative net = owed more than paid = they OWE money
                balance_label = "They Owe (All Time)"
                balance_help = "This person owes money (paid less than their share)"
            else:
                balance_label = "Balanced (All Time)"
                balance_help = "This person is settled up"
            st.metric(balance_label, f"₪{abs(net_balance_alltime):,.2f}",
                     help=balance_help)
        
        st.markdown("---")
        
        # Date range selection for member view
        st.subheader("📅 Filter by Date Range")
        member_txns = member_data.get('transactions', [])
        selected_month_member = 'All Time'
        
        if member_txns and len(member_txns) > 0:
            member_txns_df_all = pd.DataFrame(member_txns)
            
            if 'date' in member_txns_df_all.columns:
                member_txns_df_all['date'] = pd.to_datetime(member_txns_df_all['date'])
                
                # Get min/max dates from member transactions
                min_date_member = member_txns_df_all['date'].min().date()
                max_date_member = member_txns_df_all['date'].max().date()
                
                col_date1, col_date2 = st.columns(2)
                
                with col_date1:
                    start_date_member = st.date_input(
                        "Start Date",
                        value=min_date_member,
                        min_value=min_date_member,
                        max_value=max_date_member,
                        key="combined_member_start_date"
                    )
                
                with col_date2:
                    end_date_member = st.date_input(
                        "End Date",
                        value=max_date_member,
                        min_value=min_date_member,
                        max_value=max_date_member,
                        key="combined_member_end_date"
                    )
                
                # Apply date range filter to member transactions
                member_txns_df_filter = member_txns_df_all[
                    (member_txns_df_all['date'].dt.date >= start_date_member) & 
                    (member_txns_df_all['date'].dt.date <= end_date_member)
                ].copy()
                
                # Month filter within selected date range
                member_txns_df_filter['YearMonth'] = member_txns_df_filter['date'].dt.to_period('M')
                available_months_member = sorted(member_txns_df_filter['YearMonth'].unique(), reverse=True)
                
                month_options_member = ['All Time'] + [str(m) for m in available_months_member]
                selected_month_member = st.selectbox(
                    "Filter by Month (within selected date range)",
                    options=month_options_member,
                    key="member_all_month_filter"
                )
        
        st.markdown("---")
        
        # Category breakdown for member
        st.subheader(f"📊 {selected_member}'s Expense Breakdown")
        
        # Get member transactions for month filtering (using date range filtered data)
        if member_txns and len(member_txns) > 0:
            # Use the already filtered data from date range selection
            if 'date' in member_txns_df_filter.columns:
                member_txns_df = member_txns_df_filter.copy()
                
                # Apply month filter if selected
                if selected_month_member != 'All Time':
                    member_txns_df = member_txns_df[member_txns_df['YearMonth'] == pd.Period(selected_month_member)]
                
                # Recalculate category data from filtered transactions using Cost
                category_data = {}
                for _, row in member_txns_df.iterrows():
                    total_cost = row.get('total_cost', 0)
                    try:
                        total_cost = float(total_cost)
                        if total_cost > 0:
                            category = row.get('category', 'Uncategorized')
                            category_data[category] = category_data.get(category, 0) + total_cost
                    except (ValueError, TypeError):
                        continue
            else:
                category_data = user_calc.get_member_expense_by_category(selected_group_ids, selected_member)
        else:
            category_data = user_calc.get_member_expense_by_category(selected_group_ids, selected_member)
        
        if category_data:
            df_categories = pd.DataFrame([
                {'Category': cat, 'Amount': amt}
                for cat, amt in category_data.items()
            ]).sort_values('Amount', ascending=False)
            
            fig_pie = px.pie(
                df_categories,
                values='Amount',
                names='Category',
                title=f'{selected_member} - Spending by Category{" - " + selected_month_member if selected_month_member != "All Time" else ""}'
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Spending by group for this member
        st.subheader(f"{selected_member}'s Spending by Group")
        
        # Apply month filter to group spending if available (using date range filtered data)
        if member_txns and len(member_txns) > 0:
            # Use the already filtered data from date range selection
            if 'date' in member_txns_df_filter.columns:
                member_txns_df_group = member_txns_df_filter.copy()
                
                if selected_month_member != 'All Time':
                    member_txns_df_group = member_txns_df_group[member_txns_df_group['YearMonth'] == pd.Period(selected_month_member)]
                
                # Recalculate group breakdown from filtered transactions using Cost
                group_breakdown = {}
                for _, row in member_txns_df_group.iterrows():
                    group_name = row.get('group_name', 'Unknown')
                    total_cost = row.get('total_cost', 0)
                    try:
                        total_cost = float(total_cost)
                        if total_cost > 0:
                            group_breakdown[group_name] = group_breakdown.get(group_name, 0) + total_cost
                    except (ValueError, TypeError):
                        continue
            else:
                # No date column, calculate from all groups
                group_breakdown = {}
                for group_id in selected_group_ids:
                    group = next((g for g in groups if g['id'] == group_id), None)
                    if group:
                        member_stats = user_calc.calculate_user_total_expense([group_id], selected_member)
                        group_breakdown[f"{group['emoji']} {group['name']}"] = member_stats['total_expense']
        else:
            group_breakdown = {}
            for group_id in selected_group_ids:
                group = next((g for g in groups if g['id'] == group_id), None)
                if group:
                    member_stats = user_calc.calculate_user_total_expense([group_id], selected_member)
                    group_breakdown[f"{group['emoji']} {group['name']}"] = member_stats['total_expense']
        
        df_groups = pd.DataFrame([
            {'Group': name, 'Amount': amt}
            for name, amt in group_breakdown.items()
        ])
        
        if not df_groups.empty:
            fig_group_bar = px.bar(
                df_groups,
                x='Group',
                y='Amount',
                title=f'{selected_member} - Spending by Group{" - " + selected_month_member if selected_month_member != "All Time" else ""}'
            )
            st.plotly_chart(fig_group_bar, use_container_width=True)
        
        # Transaction list - use filtered transactions
        st.subheader(f"{selected_member}'s Transactions")
        
        # Use filtered transactions based on date range and month selection
        if member_txns and len(member_txns) > 0 and 'date' in member_txns_df_filter.columns:
            member_txns_df_display = member_txns_df_filter.copy()
            
            # Apply month filter if selected
            if selected_month_member != 'All Time':
                member_txns_df_display = member_txns_df_display[
                    member_txns_df_display['YearMonth'] == pd.Period(selected_month_member)
                ]
            
            if not member_txns_df_display.empty:
                df_display = member_txns_df_display[['date', 'group_name', 'description', 'category', 
                                      'total_cost', 'member_share']].copy()
                df_display.columns = ['Date', 'Group', 'Description', 'Category', 
                                     'Total Cost', 'Your Share']
                df_display = df_display.sort_values('Date', ascending=False)
            else:
                df_display = pd.DataFrame()
        elif member_txns:
            df_txns = pd.DataFrame(member_txns)
            df_display = df_txns[['date', 'group_name', 'description', 'category', 
                                  'total_cost', 'member_share']].copy()
            df_display.columns = ['Date', 'Group', 'Description', 'Category', 
                                 'Total Cost', 'Your Share']
            df_display = df_display.sort_values('Date', ascending=False)
        else:
            df_display = pd.DataFrame()
        
        if not df_display.empty:
            
            st.dataframe(df_display, use_container_width=True, hide_index=True)
            
            # Export option
            csv = df_display.to_csv(index=False)
            st.download_button(
                label=f"📥 Download {selected_member}'s Transactions",
                data=csv,
                file_name=f"{selected_member}_expenses.csv",
                mime="text/csv"
            )
        else:
            st.info("No transactions found for the selected date range.")

def show_manage_groups_page():
    """Manage groups page"""
    st.title("⚙️ Manage Groups")
    
    groups_mgr = get_groups_manager()
    groups = groups_mgr.get_all_groups()
    
    if not groups:
        st.info("No groups yet. Create your first group!")
        show_new_group_modal()
        return
    
    # Handle group editing
    if 'editing_group_id' in st.session_state and st.session_state.editing_group_id:
        group_to_edit = next((g for g in groups if g['id'] == st.session_state.editing_group_id), None)
        if group_to_edit:
            st.subheader(f"✏️ Edit Group: {group_to_edit['name']}")
            
            with st.form(f"edit_group_{group_to_edit['id']}"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    new_name = st.text_input("Group Name", value=group_to_edit['name'])
                
                with col2:
                    new_emoji = st.text_input("Emoji", value=group_to_edit.get('emoji', '📁'), max_chars=2)
                
                new_description = st.text_area(
                    "Description",
                    value=group_to_edit.get('description', '')
                )
                
                current_members = ', '.join(group_to_edit.get('members', []))
                new_members_input = st.text_input(
                    "Members (comma-separated)",
                    value=current_members
                )
                
                col_save, col_cancel = st.columns(2)
                
                with col_save:
                    if st.form_submit_button("💾 Save Changes", type="primary"):
                        new_members = [m.strip() for m in new_members_input.split(',')] if new_members_input else []
                        
                        groups_mgr.update_group(
                            group_to_edit['id'],
                            name=new_name,
                            emoji=new_emoji,
                            description=new_description,
                            members=new_members
                        )
                        
                        st.success("✅ Group updated!")
                        st.session_state.editing_group_id = None
                        st.rerun()
                
                with col_cancel:
                    if st.form_submit_button("❌ Cancel"):
                        st.session_state.editing_group_id = None
                        st.rerun()
            
            st.markdown("---")
    
    # Display groups
    st.subheader("📁 Your Groups")
    
    for group in groups:
        with st.expander(f"{group['emoji']} {group['name']}", expanded=False):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**Description:** {group.get('description', 'No description')}")
                
                members = group.get('members', [])
                if members:
                    st.markdown(f"**Members:** {', '.join(members)}")
                else:
                    st.markdown("**Members:** None")
                
                # Load stats
                group_path = groups_mgr.get_group_data_path(group['id'])
                dm = DataManager(group_path)
                df = dm.get_dataframe()
                
                if not df.empty:
                    total_txns = len(df)
                    total_cost = exclude_payments_and_reimbursements(df)['Cost'].sum()
                    st.markdown(f"**Transactions:** {total_txns}")
                    st.markdown(f"**Total Spent:** ₪{total_cost:,.2f}")
                else:
                    st.markdown("**Transactions:** 0")
                    st.markdown("**Total Spent:** ₪0.00")
                
                created_at = group.get('created_at', '')
                if created_at:
                    st.markdown(f"**Created:** {created_at[:10]}")
            
            with col2:
                is_active = group['id'] == groups_mgr.get_active_group()['id']
                
                if st.button("👁️ View", key=f"view_{group['id']}", use_container_width=True, disabled=is_active):
                    groups_mgr.set_active_group(group['id'])
                    st.cache_data.clear()
                    st.session_state.navigate_to_overview = True
                    st.rerun()
                
                if st.button("✏️ Edit", key=f"edit_{group['id']}", use_container_width=True):
                    st.session_state.editing_group_id = group['id']
                    st.rerun()
                
                if is_active:
                    st.success("✓ Active")
                
                if len(groups) > 1 and not is_active:
                    if st.button("🗑️ Delete", key=f"delete_{group['id']}", use_container_width=True):
                        st.session_state.confirm_delete_group = group['id']
                        st.rerun()
    
    # Delete confirmation
    if 'confirm_delete_group' in st.session_state and st.session_state.confirm_delete_group:
        group_to_delete = next((g for g in groups if g['id'] == st.session_state.confirm_delete_group), None)
        if group_to_delete:
            st.warning(f"⚠️ Are you sure you want to delete '{group_to_delete['name']}'? This will delete all data in this group!")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Yes, Delete", type="primary", key="confirm_delete_yes"):
                    groups_mgr.delete_group(group_to_delete['id'], delete_data=True)
                    st.session_state.confirm_delete_group = None
                    st.success("Group deleted!")
                    st.rerun()
            with col2:
                if st.button("Cancel", key="confirm_delete_cancel"):
                    st.session_state.confirm_delete_group = None
                    st.rerun()

def show_currency_settings():
    """Show currency settings page with exchange rates and management
    
    Phase 4 Implementation (Currency Settings Page):
    ✅ Displays current dashboard target currency with symbol and name
    ✅ Shows exchange rates table for 15+ common currencies
    ✅ "Update All Rates Now" button with cache clearing
    ✅ Import batch history viewer with detailed rate information
    ✅ Cache management UI with statistics and cleanup
    ✅ Distinguishes historical vs fallback rates
    
    Features:
    - Target currency display (code, symbol, full name)
    - Exchange rates table (both direct and inverse rates)
    - Rate source and last updated timestamp
    - Import batch history with expandable details
    - Cache statistics (file count, size, historical rates)
    - Clear old cache button (>90 days)
    
    Missing (for future):
    - Manual rate editing UI
    - Automatic rate update configuration
    - API key management
    - Individual currency rate history
    """
    st.header("💱 Currency Settings")
    
    currency_mgr = get_currency_manager()
    exchange_mgr = get_exchange_rate_manager()
    
    if not currency_mgr.is_initialized():
        st.warning("⚠️ Currency system not initialized. Import your first Splitwise export to set up currencies.")
        return
    
    # Current target currency section
    st.subheader("🎯 Dashboard Target Currency")
    
    target_currency = currency_mgr.get_target_currency()
    symbol = get_currency_symbol(target_currency)
    name = get_currency_name(target_currency)
    
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        st.metric("Target Currency", f"{target_currency} {symbol}")
    with col2:
        st.metric("Currency Name", name)
    with col3:
        st.info("💡 **To change target currency:** Export your dashboard, then import it back with a new currency.")
    
    st.divider()
    
    # Exchange rates section
    st.subheader("💱 Exchange Rates")
    
    try:
        # Fetch latest rates
        with st.spinner("Fetching exchange rates..."):
            rates = exchange_mgr.get_latest_rates(target_currency)
        
        # Display rates in a table
        st.write(f"**Latest Rates** (1 {target_currency} = X currency)")
        
        # Filter to common currencies for display
        common_currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CNY', 'CHF', 'CAD', 'AUD', 
                            'BGN', 'THB', 'VND', 'INR', 'BRL', 'MXN', 'KRW']
        
        # Create DataFrame for display
        rate_data = []
        for curr in sorted(common_currencies):
            if curr in rates and curr != target_currency:
                rate = rates[curr]
                symbol_curr = get_currency_symbol(curr)
                name_curr = get_currency_name(curr)
                
                rate_data.append({
                    'Currency': f"{curr} {symbol_curr}",
                    'Name': name_curr,
                    'Rate': f"{rate:.4f}",
                    'Inverse Rate': f"{1/rate:.4f}" if rate > 0 else "N/A"
                })
        
        if rate_data:
            rates_df = pd.DataFrame(rate_data)
            st.dataframe(
                rates_df,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.warning("No exchange rates available.")
        
        # Manual rate editing section
        st.markdown("---")
        st.write("**✏️ Manually Edit Exchange Rate**")
        
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            # Currency selector
            available_currencies = [c for c in common_currencies if c != target_currency]
            selected_currency = st.selectbox(
                "Select currency to edit:",
                options=available_currencies,
                format_func=lambda x: f"{x} {get_currency_symbol(x)} - {get_currency_name(x)}"
            )
        
        with col2:
            # Current rate display and new rate input
            current_rate = rates.get(selected_currency, 0) if rates else 0
            
            st.caption(f"Current rate: 1 {target_currency} = {current_rate:.4f} {selected_currency}")
            
            new_rate = st.number_input(
                f"New rate (1 {target_currency} = ? {selected_currency}):",
                min_value=0.0001,
                max_value=1000000.0,
                value=float(current_rate) if current_rate > 0 else 1.0,
                step=0.0001,
                format="%.4f",
                help=f"Enter the exchange rate: how many {selected_currency} equals 1 {target_currency}"
            )
        
        with col3:
            st.write("")  # Spacing
            st.write("")  # Spacing
            if st.button("💾 Save Rate", type="secondary", use_container_width=True):
                if new_rate > 0:
                    try:
                        # Update the cached rates
                        cache_file = exchange_mgr.cache_dir / f"latest_{target_currency}.json"
                        
                        if cache_file.exists():
                            with open(cache_file, 'r', encoding='utf-8') as f:
                                cache_data = json.load(f)
                        else:
                            # Create new cache if doesn't exist
                            cache_data = {
                                'cached_at': datetime.now().isoformat(),
                                'base': target_currency,
                                'source': 'manual_edit',
                                'rates': {}
                            }
                        
                        # Update the specific rate
                        cache_data['rates'][selected_currency] = new_rate
                        cache_data['cached_at'] = datetime.now().isoformat()
                        cache_data['source'] = f"manual_edit (last: {selected_currency})"
                        
                        # Save updated cache
                        with open(cache_file, 'w', encoding='utf-8') as f:
                            json.dump(cache_data, f, indent=2)
                        
                        st.success(f"✅ Updated rate: 1 {target_currency} = {new_rate:.4f} {selected_currency}")
                        st.info(f"💡 Inverse rate: 1 {selected_currency} = {1/new_rate:.4f} {target_currency}")
                        time.sleep(1)
                        st.rerun()
                    
                    except Exception as e:
                        st.error(f"❌ Error saving rate: {e}")
                else:
                    st.error("❌ Rate must be greater than 0")
        
        st.caption("⚠️ **Note:** Manual edits affect future conversions. Existing transactions keep their original rates.")
        
        st.markdown("---")
        
        # Update button
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("🔄 Update All Rates Now", type="primary"):
                # Clear cache and fetch fresh rates
                cache_file = exchange_mgr.cache_dir / f"latest_{target_currency}.json"
                if cache_file.exists():
                    cache_file.unlink()
                st.rerun()
        
        # Show cache info
        cache_file = exchange_mgr.cache_dir / f"latest_{target_currency}.json"
        if cache_file.exists():
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
                cache_date = datetime.fromisoformat(cache_data['cached_at'])
                source = cache_data.get('source', 'Unknown')
                
                st.caption(f"📅 Last updated: {cache_date.strftime('%Y-%m-%d %H:%M:%S')} | 🔗 Source: {source}")
    
    except Exception as e:
        st.error(f"❌ Error fetching exchange rates: {e}")
        st.info("💡 Check your internet connection or try again later.")
    
    st.divider()
    
    # Import batch history section
    st.subheader("📦 Import Batch History")
    
    # Find all import batch cache files
    import_batches = sorted(
        exchange_mgr.cache_dir.glob("import_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    
    if import_batches:
        st.write(f"Found **{len(import_batches)}** import batches with stored exchange rates:")
        
        # Show each batch
        for batch_file in import_batches[:10]:  # Show last 10
            try:
                with open(batch_file, 'r', encoding='utf-8') as f:
                    batch_data = json.load(f)
                
                batch_id = batch_file.stem
                timestamp = batch_data.get('import_timestamp', 'Unknown')
                group_name = batch_data.get('group_name', 'Unknown')
                currencies = batch_data.get('currencies_detected', [])
                txn_count = batch_data.get('transaction_count', 0)
                base_currency = batch_data.get('base_currency', target_currency)
                
                # Format timestamp
                try:
                    ts = datetime.fromisoformat(timestamp)
                    timestamp_str = ts.strftime('%Y-%m-%d %H:%M:%S')
                except:
                    timestamp_str = timestamp
                
                with st.expander(f"📦 {batch_id} - {group_name} ({txn_count} transactions)"):
                    st.write(f"**Import Time:** {timestamp_str}")
                    st.write(f"**Group:** {group_name}")
                    st.write(f"**Base Currency:** {base_currency}")
                    st.write(f"**Currencies Detected:** {', '.join(currencies) if currencies else 'None'}")
                    
                    # Show fallback rates used
                    if 'fallback_rates' in batch_data and batch_data['fallback_rates']:
                        st.write("**📊 Import-Time Fallback Rates Used:**")
                        for curr, rate_info in batch_data['fallback_rates'].items():
                            rate = rate_info.get('exchange_rate', rate_info.get('rate', 'N/A'))
                            source = rate_info.get('rate_source', rate_info.get('source', 'Unknown'))
                            st.write(f"- `1 {curr} = {rate} {base_currency}` _(Source: {source})_")
                        st.caption("💡 These rates were fetched at import time (historical API not available)")
                    
                    # Show historical rates info
                    if 'historical_rates_used' in batch_data and batch_data['historical_rates_used']:
                        st.write("**📅 Historical Rates Used:**")
                        for curr, rate_info in batch_data['historical_rates_used'].items():
                            source = rate_info.get('source', 'Unknown')
                            dates = rate_info.get('dates', [])
                            if dates:
                                st.write(f"- `{curr}`: {len(dates)} unique dates _(Source: {source})_")
                            else:
                                st.write(f"- `{curr}` _(Source: {source})_")
                        st.caption("✅ These transactions used exact historical rates from their transaction dates")
            
            except Exception as e:
                st.warning(f"Could not read batch {batch_file.name}: {e}")
        
        if len(import_batches) > 10:
            st.caption(f"Showing 10 of {len(import_batches)} batches")
    else:
        st.info("📭 No import batches yet. Import Splitwise data to see batch history.")
    
    st.divider()
    
    # Cache management section
    st.subheader("🗄️ Cache Management")
    
    cache_files = list(exchange_mgr.cache_dir.glob("*.json"))
    total_size = sum(f.stat().st_size for f in cache_files)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Cached Files", len(cache_files))
    with col2:
        st.metric("Cache Size", f"{total_size / 1024:.1f} KB")
    with col3:
        st.metric("Historical Rates", len([f for f in cache_files if f.stem.startswith('historical_')]))
    
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🗑️ Clear Old Cache", help="Remove cached rates older than 90 days"):
            try:
                cutoff = datetime.now() - timedelta(days=90)
                deleted = 0
                for cache_file in cache_files:
                    if cache_file.stat().st_mtime < cutoff.timestamp():
                        cache_file.unlink()
                        deleted += 1
                st.success(f"✅ Deleted {deleted} old cache files")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error clearing cache: {e}")
    
    st.divider()
    
    # Automatic rate updates settings
    st.subheader("🔄 Automatic Rate Updates")
    st.markdown("Configure automatic exchange rate updates for your dashboard.")
    
    # Load current settings
    settings = currency_mgr.load_settings()
    auto_update_enabled = settings.get('dashboard_settings', {}).get('auto_update', False)
    api_service = settings.get('dashboard_settings', {}).get('api_service', 'exchangerate-api.com')
    api_key = settings.get('dashboard_settings', {}).get('api_key', '')
    update_frequency = settings.get('dashboard_settings', {}).get('update_frequency', 'daily')
    last_auto_update = settings.get('dashboard_settings', {}).get('last_auto_update', None)
    
    # Enable/disable toggle
    col1, col2 = st.columns([1, 3])
    
    with col1:
        new_auto_update = st.toggle(
            "Enable Auto-Updates",
            value=auto_update_enabled,
            key="auto_update_toggle",
            help="Automatically fetch latest exchange rates on schedule"
        )
    
    with col2:
        if last_auto_update:
            last_update_dt = datetime.fromisoformat(last_auto_update)
            st.caption(f"Last auto-update: {last_update_dt.strftime('%Y-%m-%d %H:%M')}")
        else:
            st.caption("Never run")
    
    if new_auto_update:
        # Show configuration options
        st.markdown("**📋 Configuration:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            new_api_service = st.selectbox(
                "API Service",
                options=['exchangerate-api.com', 'frankfurter.app', 'custom'],
                index=['exchangerate-api.com', 'frankfurter.app', 'custom'].index(api_service) if api_service in ['exchangerate-api.com', 'frankfurter.app', 'custom'] else 0,
                key="api_service_select",
                help="Choose which API to use for rate updates"
            )
            
            st.caption("**API Information:**")
            if new_api_service == 'exchangerate-api.com':
                st.caption("✅ Free: 1,500 requests/month")
                st.caption("✅ 160+ currencies")
                st.caption("ℹ️ No API key needed (free tier)")
            elif new_api_service == 'frankfurter.app':
                st.caption("✅ Free: Unlimited")
                st.caption("⚠️ 29 ECB currencies only")
                st.caption("✅ Historical rates available")
            else:
                st.caption("⚠️ Custom API requires configuration")
        
        with col2:
            new_update_frequency = st.selectbox(
                "Update Frequency",
                options=['daily', 'weekly', 'manual'],
                index=['daily', 'weekly', 'manual'].index(update_frequency) if update_frequency in ['daily', 'weekly', 'manual'] else 0,
                key="update_frequency_select",
                help="How often to automatically update rates"
            )
            
            st.caption("**Frequency Options:**")
            if new_update_frequency == 'daily':
                st.caption("Updates every 24 hours")
            elif new_update_frequency == 'weekly':
                st.caption("Updates every 7 days")
            else:
                st.caption("No automatic updates (manual only)")
        
        # API Key input (optional)
        if new_api_service == 'exchangerate-api.com':
            st.markdown("**🔑 API Key (Optional):**")
            new_api_key = st.text_input(
                "API Key",
                value=api_key,
                type="password",
                key="api_key_input",
                help="Optional: Provide API key for higher rate limits and premium features"
            )
            
            if not new_api_key:
                st.info("ℹ️ **Using Free Tier**: No API key provided. Using free tier with 1,500 requests/month. This is sufficient for personal use.")
        else:
            new_api_key = api_key
        
        # Save settings button
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("💾 Save Settings", use_container_width=True):
                try:
                    # Update settings
                    if 'dashboard_settings' not in settings:
                        settings['dashboard_settings'] = {}
                    
                    settings['dashboard_settings']['auto_update'] = new_auto_update
                    settings['dashboard_settings']['api_service'] = new_api_service
                    settings['dashboard_settings']['api_key'] = new_api_key
                    settings['dashboard_settings']['update_frequency'] = new_update_frequency
                    
                    # Save to file
                    currency_mgr.settings_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(currency_mgr.settings_file, 'w', encoding='utf-8') as f:
                        json.dump(settings, f, indent=2, ensure_ascii=False)
                    
                    st.success("✅ Auto-update settings saved!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error saving settings: {e}")
        
        with col2:
            if st.button("🔄 Update Now", use_container_width=True):
                try:
                    with st.spinner("Updating exchange rates..."):
                        # Update rates using the configured service
                        exchange_mgr = get_exchange_rate_manager()
                        target_currency = currency_mgr.get_target_currency()
                        
                        # Clear cache to force fresh fetch
                        cache_file = exchange_mgr.cache_dir / f"latest_{target_currency}.json"
                        if cache_file.exists():
                            cache_file.unlink()
                        
                        # Fetch new rates
                        rates = exchange_mgr.get_latest_rates(target_currency)
                        
                        # Update last_auto_update timestamp
                        if 'dashboard_settings' not in settings:
                            settings['dashboard_settings'] = {}
                        settings['dashboard_settings']['last_auto_update'] = datetime.now().isoformat()
                        
                        with open(currency_mgr.settings_file, 'w', encoding='utf-8') as f:
                            json.dump(settings, f, indent=2, ensure_ascii=False)
                        
                        st.success(f"✅ Updated rates for {len(rates)} currencies!")
                        st.rerun()
                except Exception as e:
                    st.error(f"❌ Error updating rates: {e}")
        
        # Status information
        st.markdown("---")
        st.markdown("**ℹ️ How Auto-Updates Work:**")
        st.markdown("""
        - **Scheduled Updates**: Rates are automatically refreshed based on your chosen frequency
        - **Import Priority**: When importing Splitwise data, the system always tries to fetch historical rates first
        - **Cache Efficiency**: Updated rates are cached to minimize API calls
        - **Fallback Support**: If the primary API fails, the system uses cached rates or fallback APIs
        - **Manual Override**: You can always manually edit rates in the table above
        """)
        
        if new_update_frequency != 'manual':
            st.info(f"💡 **Next scheduled update**: Approximately {new_update_frequency} from last update")
    
    else:
        # Auto-updates disabled
        st.info("🔕 **Automatic updates disabled**. Exchange rates will only be updated:")
        st.markdown("""
        - When you click **"Update All Rates Now"** button above
        - When importing Splitwise data (historical rates for transaction dates)
        - When you manually edit rates
        """)
        
        # Show save button to persist disabled state
        if auto_update_enabled != new_auto_update:
            if st.button("💾 Save Settings (Disable Auto-Updates)"):
                try:
                    if 'dashboard_settings' not in settings:
                        settings['dashboard_settings'] = {}
                    settings['dashboard_settings']['auto_update'] = False
                    
                    with open(currency_mgr.settings_file, 'w', encoding='utf-8') as f:
                        json.dump(settings, f, indent=2, ensure_ascii=False)
                    
                    st.success("✅ Auto-update settings saved!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error saving settings: {e}")

def show_multi_currency_analytics():
    """Show dedicated multi-currency analytics page with trends and insights"""
    st.header("💱 Multi-Currency Analytics")
    
    currency_mgr = get_currency_manager()
    
    if not currency_mgr.is_initialized():
        st.warning("⚠️ Currency system not initialized. Import your first Splitwise export to set up currencies.")
        return
    
    # Get data from all groups
    groups_mgr = get_groups_manager()
    all_groups_data = []
    
    for group in groups_mgr.get_all_groups():
        dm = get_data_manager(groups_mgr.get_group_data_path(group['id']))
        group_df = dm.get_dataframe()
        if not group_df.empty:
            group_df['group_name'] = group['name']
            all_groups_data.append(group_df)
    
    if not all_groups_data:
        st.info("📭 No transaction data available. Import data from Data Management page.")
        return
    
    # Combine all data
    df_all = pd.concat(all_groups_data, ignore_index=True)
    df_all = exclude_payments_and_reimbursements(df_all)
    
    # Check for multi-currency data
    has_multi_currency = 'original_currency' in df_all.columns and df_all['original_currency'].notna().any()
    
    if not has_multi_currency:
        st.info("💡 **Single Currency Dashboard**: All your transactions are in the target currency. Multi-currency analytics will appear when you import transactions in other currencies.")
        return
    
    target_currency = currency_mgr.get_target_currency()
    target_symbol = get_currency_symbol(target_currency)
    
    # Summary metrics
    st.subheader("📊 Currency Overview")
    
    # Calculate currency breakdown
    currency_totals = {}
    for _, row in df_all.iterrows():
        orig_curr = row.get('original_currency')
        if pd.notna(orig_curr):
            if orig_curr not in currency_totals:
                currency_totals[orig_curr] = {'original_sum': 0, 'converted_sum': 0, 'count': 0}
            currency_totals[orig_curr]['original_sum'] += row.get('original_cost', 0)
            currency_totals[orig_curr]['converted_sum'] += row.get('Cost', 0)
            currency_totals[orig_curr]['count'] += 1
        else:
            # Target currency transaction
            if target_currency not in currency_totals:
                currency_totals[target_currency] = {'original_sum': 0, 'converted_sum': 0, 'count': 0}
            currency_totals[target_currency]['original_sum'] += row.get('Cost', 0)
            currency_totals[target_currency]['converted_sum'] += row.get('Cost', 0)
            currency_totals[target_currency]['count'] += 1
    
    # Display currency cards
    cols = st.columns(min(len(currency_totals), 4))
    for idx, (currency, data) in enumerate(sorted(currency_totals.items(), key=lambda x: x[1]['converted_sum'], reverse=True)):
        with cols[idx % len(cols)]:
            curr_symbol = get_currency_symbol(currency)
            percentage = (data['converted_sum'] / df_all['Cost'].sum()) * 100
            
            st.metric(
                f"{currency} {curr_symbol}",
                f"{target_symbol}{data['converted_sum']:,.0f}",
                f"{percentage:.1f}% • {data['count']} txns"
            )
    
    st.markdown("---")
    
    # Monthly spending by currency
    st.subheader("📈 Monthly Spending Trends by Currency")
    
    # Prepare data for monthly chart
    df_all['YearMonth'] = pd.to_datetime(df_all['Date']).dt.to_period('M')
    monthly_currency_data = []
    
    for currency in currency_totals.keys():
        if currency == target_currency:
            # Target currency transactions
            currency_df = df_all[df_all['original_currency'].isna() | (df_all['original_currency'] == target_currency)]
        else:
            # Foreign currency transactions
            currency_df = df_all[df_all['original_currency'] == currency]
        
        monthly_spending = currency_df.groupby('YearMonth')['Cost'].sum().reset_index()
        monthly_spending['Currency'] = f"{currency} {get_currency_symbol(currency)}"
        monthly_spending['YearMonth'] = monthly_spending['YearMonth'].astype(str)
        monthly_currency_data.append(monthly_spending)
    
    if monthly_currency_data:
        monthly_df = pd.concat(monthly_currency_data, ignore_index=True)
        
        fig_monthly = px.line(
            monthly_df,
            x='YearMonth',
            y='Cost',
            color='Currency',
            title=f"Monthly Spending by Original Currency (in {target_currency})",
            labels={'Cost': f'Amount ({target_currency})', 'YearMonth': 'Month'},
            markers=True
        )
        
        fig_monthly.update_layout(
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_monthly, use_container_width=True)
    
    st.markdown("---")
    
    # Currency distribution over time
    st.subheader("📊 Currency Distribution Over Time")
    
    # Calculate percentage distribution by month
    monthly_totals = df_all.groupby('YearMonth')['Cost'].sum()
    distribution_data = []
    
    for currency in currency_totals.keys():
        if currency == target_currency:
            currency_df = df_all[df_all['original_currency'].isna() | (df_all['original_currency'] == target_currency)]
        else:
            currency_df = df_all[df_all['original_currency'] == currency]
        
        monthly_currency = currency_df.groupby('YearMonth')['Cost'].sum()
        
        for month in monthly_totals.index:
            amount = monthly_currency.get(month, 0)
            percentage = (amount / monthly_totals[month] * 100) if monthly_totals[month] > 0 else 0
            
            distribution_data.append({
                'Month': str(month),
                'Currency': f"{currency} {get_currency_symbol(currency)}",
                'Percentage': percentage
            })
    
    if distribution_data:
        dist_df = pd.DataFrame(distribution_data)
        
        fig_dist = px.area(
            dist_df,
            x='Month',
            y='Percentage',
            color='Currency',
            title="Currency Distribution Over Time (%)",
            labels={'Percentage': 'Percentage of Total Spending'},
        )
        
        fig_dist.update_layout(
            hovermode='x unified',
            yaxis=dict(range=[0, 100]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_dist, use_container_width=True)
    
    st.markdown("---")
    
    # Detailed currency breakdown table
    st.subheader("💰 Detailed Currency Breakdown")
    
    breakdown_data = []
    for currency, data in sorted(currency_totals.items(), key=lambda x: x[1]['converted_sum'], reverse=True):
        curr_symbol = get_currency_symbol(currency)
        curr_name = get_currency_name(currency)
        percentage = (data['converted_sum'] / df_all['Cost'].sum()) * 100
        
        # Calculate average transaction
        avg_original = data['original_sum'] / data['count'] if data['count'] > 0 else 0
        avg_converted = data['converted_sum'] / data['count'] if data['count'] > 0 else 0
        
        breakdown_data.append({
            'Currency': f"{currency} {curr_symbol}",
            'Name': curr_name,
            'Transactions': data['count'],
            'Total (Original)': f"{curr_symbol}{data['original_sum']:,.2f}" if currency != target_currency else f"{target_symbol}{data['original_sum']:,.2f}",
            f'Total ({target_currency})': f"{target_symbol}{data['converted_sum']:,.2f}",
            'Percentage': f"{percentage:.1f}%",
            'Avg/Transaction': f"{target_symbol}{avg_converted:,.2f}"
        })
    
    breakdown_df = pd.DataFrame(breakdown_data)
    st.dataframe(breakdown_df, use_container_width=True, hide_index=True)
    
    # Foreign currency exposure
    foreign_total = sum(data['converted_sum'] for curr, data in currency_totals.items() if curr != target_currency)
    total_spending = df_all['Cost'].sum()
    foreign_percentage = (foreign_total / total_spending * 100) if total_spending > 0 else 0
    
    if foreign_total > 0:
        st.info(f"💡 **Foreign Currency Exposure**: {target_symbol}{foreign_total:,.0f} ({foreign_percentage:.1f}% of total spending)")
    
    st.markdown("---")
    
    # Category breakdown by currency
    st.subheader("🏷️ Spending by Category and Currency")
    
    category_currency_data = []
    for _, row in df_all.iterrows():
        orig_curr = row.get('original_currency')
        currency_display = f"{orig_curr} {get_currency_symbol(orig_curr)}" if pd.notna(orig_curr) else f"{target_currency} {target_symbol}"
        
        category_currency_data.append({
            'Category': row.get('Category', 'Uncategorized'),
            'Currency': currency_display,
            'Amount': row.get('Cost', 0)
        })
    
    if category_currency_data:
        cat_curr_df = pd.DataFrame(category_currency_data)
        cat_summary = cat_curr_df.groupby(['Category', 'Currency'])['Amount'].sum().reset_index()
        
        # Top 10 categories
        top_categories = cat_curr_df.groupby('Category')['Amount'].sum().nlargest(10).index
        cat_summary_top = cat_summary[cat_summary['Category'].isin(top_categories)]
        
        fig_cat = px.bar(
            cat_summary_top,
            x='Category',
            y='Amount',
            color='Currency',
            title=f"Top 10 Categories - Spending by Currency (in {target_currency})",
            labels={'Amount': f'Amount ({target_currency})'},
            barmode='stack'
        )
        
        fig_cat.update_layout(
            xaxis_tickangle=-45,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_cat, use_container_width=True)
    
    # "What If" Rate Change Calculator
    st.markdown("---")
    st.subheader("🔮 What If Rate Change Calculator")
    st.markdown("See how changes in exchange rates would impact your spending.")
    
    # Get current rates for foreign currencies
    rate_mgr = get_exchange_rate_manager()
    current_rates = {}
    
    for currency in currency_totals.keys():
        if currency != target_currency:
            try:
                rate = rate_mgr.get_rate(currency, target_currency)
                current_rates[currency] = rate
            except:
                current_rates[currency] = None
    
    if current_rates:
        # Create rate adjustment interface
        st.markdown("**Adjust Exchange Rates:**")
        
        rate_changes = {}
        cols = st.columns(min(len(current_rates), 3))
        
        for idx, (currency, current_rate) in enumerate(current_rates.items()):
            if current_rate is None:
                continue
                
            col_idx = idx % 3
            with cols[col_idx]:
                curr_symbol = currency_mgr.get_currency_symbol(currency)
                
                # Show current rate
                st.caption(f"**{currency} {curr_symbol}**")
                st.caption(f"Current: 1 {currency} = {current_rate:.4f} {target_currency}")
                
                # Input for new rate
                new_rate = st.number_input(
                    f"New rate (1 {currency} → {target_currency})",
                    min_value=0.0001,
                    max_value=1000000.0,
                    value=float(current_rate),
                    step=float(current_rate * 0.01),  # 1% increments
                    format="%.4f",
                    key=f"whatif_rate_{currency}"
                )
                
                # Show percentage change
                if new_rate != current_rate:
                    pct_change = ((new_rate - current_rate) / current_rate) * 100
                    change_emoji = "📈" if pct_change > 0 else "📉"
                    st.caption(f"{change_emoji} {pct_change:+.1f}% change")
                
                rate_changes[currency] = new_rate
        
        # Calculate impact
        st.markdown("---")
        st.markdown("**💡 Impact Analysis:**")
        
        total_impact = 0
        impact_details = []
        
        for currency, data in currency_totals.items():
            if currency == target_currency:
                continue
            
            if currency not in rate_changes or rate_changes[currency] is None:
                continue
            
            original_amount = data['original_sum']
            current_converted = data['converted_sum']
            
            # Recalculate with new rate
            new_rate = rate_changes[currency]
            current_rate = current_rates[currency]
            
            new_converted = original_amount * new_rate
            difference = new_converted - current_converted
            
            total_impact += difference
            
            curr_symbol = currency_mgr.get_currency_symbol(currency)
            target_symbol = currency_mgr.get_currency_symbol(target_currency)
            
            impact_details.append({
                'Currency': f"{currency} {curr_symbol}",
                'Original Amount': f"{curr_symbol}{original_amount:,.2f}",
                'Current Converted': f"{target_symbol}{current_converted:,.2f}",
                'New Converted': f"{target_symbol}{new_converted:,.2f}",
                'Difference': f"{target_symbol}{difference:+,.2f}"
            })
        
        if impact_details:
            # Display impact table
            impact_df = pd.DataFrame(impact_details)
            st.dataframe(impact_df, use_container_width=True, hide_index=True)
            
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                original_total = sum(data['converted_sum'] for data in currency_totals.values())
                st.metric(
                    "Current Total",
                    f"{target_symbol}{original_total:,.0f}"
                )
            
            with col2:
                new_total = original_total + total_impact
                st.metric(
                    "New Total",
                    f"{target_symbol}{new_total:,.0f}",
                    delta=f"{total_impact:+,.0f}"
                )
            
            with col3:
                if original_total > 0:
                    pct_impact = (total_impact / original_total) * 100
                    impact_emoji = "📈" if total_impact > 0 else "📉"
                    st.metric(
                        "Total Impact",
                        f"{pct_impact:+.2f}%",
                        delta=f"{target_symbol}{total_impact:+,.0f}"
                    )
            
            # Interpretation
            if abs(total_impact) > 0.01:
                if total_impact > 0:
                    st.warning(f"⚠️ **Negative Impact**: If exchange rates change as specified, your total spending would INCREASE by {target_symbol}{total_impact:,.2f} ({abs(pct_impact):.2f}%). This means foreign currency expenses would cost you more.")
                else:
                    st.success(f"✅ **Positive Impact**: If exchange rates change as specified, your total spending would DECREASE by {target_symbol}{abs(total_impact):,.2f} ({abs(pct_impact):.2f}%). This means foreign currency expenses would cost you less.")
            else:
                st.info("💡 Exchange rates are at current levels. Adjust the rates above to see impact scenarios.")
        
        # Quick scenarios
        st.markdown("---")
        st.markdown("**⚡ Quick Scenarios:**")
        
        scenario_cols = st.columns(3)
        
        with scenario_cols[0]:
            if st.button("📈 +5% Rate Increase", use_container_width=True):
                for currency in rate_changes.keys():
                    if currency in current_rates and current_rates[currency]:
                        st.session_state[f"whatif_rate_{currency}"] = current_rates[currency] * 1.05
                st.rerun()
        
        with scenario_cols[1]:
            if st.button("📉 -5% Rate Decrease", use_container_width=True):
                for currency in rate_changes.keys():
                    if currency in current_rates and current_rates[currency]:
                        st.session_state[f"whatif_rate_{currency}"] = current_rates[currency] * 0.95
                st.rerun()
        
        with scenario_cols[2]:
            if st.button("🔄 Reset to Current", use_container_width=True):
                for currency in rate_changes.keys():
                    if currency in current_rates and current_rates[currency]:
                        st.session_state[f"whatif_rate_{currency}"] = current_rates[currency]
                st.rerun()
        
        st.caption("💡 **Tip**: Use this calculator to understand your currency exposure risk. Large impacts suggest you may want to consider the timing of future purchases in foreign currencies.")
    else:
        st.info("No foreign currency data available for rate change scenarios.")

def show_analytics(df):
    """Show analytics page with detailed charts"""
    st.header("📊 Analytics")
    
    df_all_expenses = exclude_payments_and_reimbursements(df)
    
    if df_all_expenses.empty:
        st.info("No expense data available for analysis")
        return
    
    # Currency filter section (if multi-currency data exists)
    currency_mgr = get_currency_manager()
    df_filtered = df_all_expenses.copy()
    
    if currency_mgr.is_initialized() and 'original_currency' in df_all_expenses.columns:
        has_multi_currency = df_all_expenses['original_currency'].notna().any()
        
        if has_multi_currency:
            st.subheader("💱 Currency Filter")
            
            # Get all currencies
            target_currency = currency_mgr.get_target_currency()
            all_currencies = [target_currency]
            
            foreign_currencies = df_all_expenses['original_currency'].dropna().unique()
            for curr in foreign_currencies:
                if curr and curr != target_currency and curr not in all_currencies:
                    all_currencies.append(curr)
            
            # Filter options
            col1, col2 = st.columns([1, 3])
            
            with col1:
                filter_mode = st.radio(
                    "Filter by Currency",
                    options=["All Currencies", f"{target_currency} Only", "Foreign Currencies Only", "Specific Currencies"],
                    key="currency_filter_mode"
                )
            
            with col2:
                if filter_mode == "Specific Currencies":
                    selected_currencies = st.multiselect(
                        "Select Currencies",
                        options=all_currencies,
                        default=all_currencies,
                        key="selected_currencies"
                    )
            
            # Apply filter
            if filter_mode == f"{target_currency} Only":
                df_filtered = df_all_expenses[
                    (df_all_expenses['original_currency'].isna()) | 
                    (df_all_expenses['original_currency'] == target_currency)
                ]
            elif filter_mode == "Foreign Currencies Only":
                df_filtered = df_all_expenses[
                    (df_all_expenses['original_currency'].notna()) & 
                    (df_all_expenses['original_currency'] != target_currency)
                ]
            elif filter_mode == "Specific Currencies":
                if selected_currencies:
                    mask = (
                        (df_all_expenses['original_currency'].isin(selected_currencies)) |
                        ((df_all_expenses['original_currency'].isna()) & (target_currency in selected_currencies))
                    )
                    df_filtered = df_all_expenses[mask]
                else:
                    df_filtered = pd.DataFrame()  # Empty if no currencies selected
            
            # Show filtered count
            st.caption(f"📊 Showing {len(df_filtered)} of {len(df_all_expenses)} transactions")
            
            st.divider()
    
    # Check if we have data after filtering
    if df_filtered.empty:
        st.warning("No data available with the selected currency filter")
        return
    
    # Currency breakdown chart (if multi-currency)
    if currency_mgr.is_initialized() and 'original_currency' in df_filtered.columns:
        has_multi_currency = df_filtered['original_currency'].notna().any()
        
        if has_multi_currency:
            st.subheader("💱 Spending by Currency")
            
            target_currency = currency_mgr.get_target_currency()
            target_symbol = get_currency_symbol(target_currency)
            
            # Calculate spending by currency
            currency_data = []
            
            for currency in df_filtered['original_currency'].dropna().unique():
                if currency and currency != target_currency:
                    currency_txns = df_filtered[df_filtered['original_currency'] == currency]
                    converted_total = currency_txns['Cost'].sum()
                    currency_data.append({
                        'Currency': f"{currency} {get_currency_symbol(currency)}",
                        'Amount': converted_total
                    })
            
            # Target currency transactions
            target_txns = df_filtered[
                (df_filtered['original_currency'].isna()) | 
                (df_filtered['original_currency'] == target_currency)
            ]
            if not target_txns.empty:
                target_total = target_txns['Cost'].sum()
                currency_data.append({
                    'Currency': f"{target_currency} {target_symbol}",
                    'Amount': target_total
                })
            
            if currency_data:
                currency_df = pd.DataFrame(currency_data)
                currency_df = currency_df.sort_values('Amount', ascending=False)
                
                fig_currency = px.pie(
                    currency_df,
                    values='Amount',
                    names='Currency',
                    title=f"Spending by Original Currency (in {target_currency})",
                    hole=0.4
                )
                
                fig_currency.update_traces(
                    textposition='auto',
                    textinfo='percent+label+value'
                )
                
                st.plotly_chart(fig_currency, use_container_width=True)
            
            st.divider()
    
    # Monthly spending by category
    st.subheader("Monthly Spending by Category")
    categories = ['All Categories'] + sorted(df_filtered['Category'].unique().tolist())
    selected_category_monthly = st.selectbox(
        "Select Category",
        options=categories,
        key="monthly_category"
    )
    
    df_monthly = df_filtered.copy()
    if selected_category_monthly != 'All Categories':
        df_monthly = df_monthly[df_monthly['Category'] == selected_category_monthly]
    
    df_monthly['YearMonth'] = df_monthly['Date'].dt.to_period('M').astype(str)
    
    # Calculate monthly totals using Cost
    monthly_totals = df_monthly.groupby(['YearMonth', 'Category'])['Cost'].sum().reset_index()
    monthly_totals.columns = ['YearMonth', 'Category', 'Total']
    
    fig_monthly = px.bar(
        monthly_totals,
        x='YearMonth',
        y='Total',
        color='Category',
        title=f"Monthly Spending - {selected_category_monthly}",
        barmode='stack'
    )
    
    fig_monthly.update_layout(
        xaxis_title="Month",
        yaxis_title="Amount (₪)",
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )
    
    st.plotly_chart(fig_monthly, use_container_width=True)
    
    st.markdown("---")
    
    # Yearly spending by category - Monthly Average Pie Chart
    st.subheader("Monthly Average Spending by Category")
    
    df_yearly = df_all_expenses.copy()
    df_yearly['Year'] = df_yearly['Date'].dt.year
    df_yearly['YearMonth'] = df_yearly['Date'].dt.to_period('M')
    
    # Get available years
    available_years = sorted(df_yearly['Year'].unique(), reverse=True)
    
    if len(available_years) > 0:
        selected_year = st.selectbox(
            "Select Year",
            options=available_years,
            index=0,
            key="yearly_pie_year"
        )
        
        # Filter by selected year
        df_year_filtered = df_yearly[df_yearly['Year'] == selected_year]
        
        # Calculate monthly totals per category
        monthly_by_cat = df_year_filtered.groupby(['YearMonth', 'Category'])['Cost'].sum().reset_index()
        
        # Calculate average monthly spending per category
        category_monthly_avg = monthly_by_cat.groupby('Category')['Cost'].mean().reset_index()
        category_monthly_avg.columns = ['Category', 'Monthly Average']
        category_monthly_avg = category_monthly_avg.sort_values('Monthly Average', ascending=False)
        
        fig_yearly_pie = px.pie(
            category_monthly_avg,
            values='Monthly Average',
            names='Category',
            title=f"Average Monthly Spending by Category - {selected_year}",
            hole=0.4
        )
        
        fig_yearly_pie.update_traces(
            textposition='auto',
            textinfo='percent+label'
        )
        
        st.plotly_chart(fig_yearly_pie, use_container_width=True)
    else:
        st.info("No data available for yearly analysis")
    
    st.markdown("---")
    
    # Year-over-Year analysis
    st.subheader("Year-over-Year Monthly Average")
    
    # Category selector
    selected_category_yoy = st.selectbox(
        "Select Category",
        options=categories,
        key="yoy_category"
    )
    
    df_yoy = df_all_expenses.copy()
    if selected_category_yoy != 'All Categories':
        df_yoy = df_yoy[df_yoy['Category'] == selected_category_yoy]
    
    df_yoy['Year'] = df_yoy['Date'].dt.year
    df_yoy['Month'] = df_yoy['Date'].dt.month
    df_yoy['YearMonth'] = df_yoy['Date'].dt.to_period('M').astype(str)
    
    # Calculate monthly totals using Cost
    monthly_totals_yoy = df_yoy.groupby(['YearMonth', 'Year', 'Month', 'Category'])['Cost'].sum().reset_index()
    monthly_totals_yoy.columns = ['YearMonth', 'Year', 'Month', 'Category', 'Total']
    
    # Date range selector for this plot
    if not monthly_totals_yoy.empty:
        col1, col2 = st.columns(2)
        
        available_months = sorted(monthly_totals_yoy['YearMonth'].unique())
        
        with col1:
            start_month = st.selectbox(
                "Start Month",
                options=available_months,
                index=0,
                key="yoy_start_month"
            )
        
        with col2:
            end_month = st.selectbox(
                "End Month",
                options=available_months,
                index=len(available_months) - 1,
                key="yoy_end_month"
            )
        
        # Filter data by selected range
        monthly_totals_yoy = monthly_totals_yoy[
            (monthly_totals_yoy['YearMonth'] >= start_month) & 
            (monthly_totals_yoy['YearMonth'] <= end_month)
        ]
        
        # Create plot showing monthly values for each category
        fig_yoy = px.line(
            monthly_totals_yoy,
            x='YearMonth',
            y='Total',
            color='Category',
            title=f"Monthly Spending by Category - {selected_category_yoy}",
            markers=True
        )
        
        fig_yoy.update_layout(
            xaxis_title="Month",
            yaxis_title="Amount (₪)",
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            )
        )
        
        st.plotly_chart(fig_yoy, use_container_width=True)
    else:
        st.info("No data available for the selected category")

def run_database_migrations():
    """Check and run database migrations on app startup"""
    # Check if we've already run migrations this session
    if 'migrations_checked' not in st.session_state:
        st.session_state.migrations_checked = False
    
    if st.session_state.migrations_checked:
        return  # Already checked this session
    
    try:
        # Initialize migration manager
        migration_mgr = MigrationManager(user_data_path="user_data")
        
        # Check if migrations are needed
        needs_migration = migration_mgr.check_migrations_needed()
        
        if not needs_migration:
            # No migrations needed
            st.session_state.migrations_checked = True
            return
        
        # Show migration dialog
        st.warning("🔄 Database migrations are required to add new features!")
        
        total_files = len(needs_migration)
        st.info(f"📊 {total_files} file(s) need to be updated")
        
        # Show what will be migrated
        with st.expander("📋 View migration details", expanded=True):
            for file_path, migrations in needs_migration.items():
                file_name = file_path.split('/')[-2]  # Get group name
                st.write(f"**{file_name}**")
                for migration_name in migrations:
                    migration_obj = migration_mgr.migrations.get(migration_name)
                    if migration_obj:
                        st.write(f"  • {migration_obj}")
        
        st.markdown("---")
        st.markdown("### 🛡️ Safety Features")
        st.write("✅ Automatic backups created before migration")
        st.write("✅ Automatic rollback if any errors occur")
        st.write("✅ All data validated before and after migration")
        
        st.markdown("---")
        
        # Migration button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Run Migrations Now", type="primary", use_container_width=True):
                # Create progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("🔄 Running migrations...")
                
                # Run migrations
                results = migration_mgr.run_all_migrations(show_progress=False)
                
                # Update progress
                successful = sum(1 for v in results.values() if v)
                progress_bar.progress(successful / total_files)
                
                # Show results
                if successful == total_files:
                    status_text.empty()
                    progress_bar.empty()
                    st.success(f"✅ Successfully migrated {successful}/{total_files} files!")
                    st.balloons()
                    
                    # Mark migrations as checked
                    st.session_state.migrations_checked = True
                    
                    # Force refresh after 2 seconds
                    time.sleep(2)
                    st.rerun()
                else:
                    failed = total_files - successful
                    st.error(f"❌ Migration failed for {failed} file(s)")
                    st.warning("Your data has been preserved. Please check the error messages above.")
        
        # Skip button (not recommended)
        st.markdown("---")
        with st.expander("⚠️ Skip migrations (not recommended)"):
            st.warning("Skipping migrations may cause features to not work correctly.")
            if st.button("Skip for now"):
                st.session_state.migrations_checked = True
                st.rerun()
        
        # Stop execution until migration is complete
        st.stop()
        
    except Exception as e:
        st.error(f"❌ Error checking migrations: {e}")
        # Allow continuing on error
        st.session_state.migrations_checked = True

def main():
    st.title("💰 streamlit-splitwise-dashboard")
    
    # Initialize session state for modals
    if 'show_new_group_modal' not in st.session_state:
        st.session_state.show_new_group_modal = False
    if 'show_manage_groups' not in st.session_state:
        st.session_state.show_manage_groups = False
    if 'navigate_to_manage_groups' not in st.session_state:
        st.session_state.navigate_to_manage_groups = False
    
    # Run database migrations if needed
    run_database_migrations()
    
    # Check if groups need to be migrated
    groups_mgr = get_groups_manager()
    dm = get_data_manager()
    
    # Handle migration from old structure to groups
    if not groups_mgr.has_groups() and dm.data_exists():
        with st.spinner("Migrating to groups structure..."):
            groups_mgr.migrate_existing_data()
        st.success("✅ Successfully migrated to groups structure!")
        st.rerun()
    
    if not dm.data_exists():
        # First-time setup
        show_setup_wizard()
        return
    
    # Render group selector
    render_group_selector()
    
    # Show new group modal if requested
    if st.session_state.show_new_group_modal:
        with st.container():
            show_new_group_modal()
            if st.button("Cancel"):
                st.session_state.show_new_group_modal = False
                st.rerun()
    
    # Load data
    df = load_persisted_data()
    
    # Navigation
    st.sidebar.header("📊 Navigation")
    
    # Handle navigation from Manage button or restore after group switch
    if st.session_state.navigate_to_manage_groups:
        default_page = "Manage Groups"
        # Clear the flag AFTER using it
    elif 'current_page' in st.session_state:
        default_page = st.session_state.current_page
    else:
        default_page = "Overview"
    
    page_options = ["Overview", "Analytics", "Income & Savings", "Data Management", "Currency Settings", "Multi-Currency Analytics", "Combined Analytics", "Manage Groups"]
    page = st.sidebar.radio("Select Page", page_options, 
                           index=page_options.index(default_page) if default_page in page_options else 0,
                           key="main_page_selector")
    
    # Clear navigation flag after page is selected
    if st.session_state.navigate_to_manage_groups:
        st.session_state.navigate_to_manage_groups = False
    
    # Store current page selection only if it changed
    if 'current_page' not in st.session_state or st.session_state.current_page != page:
        st.session_state.current_page = page
    
    # Filters (if not on data management or income page)
    if page not in ["Data Management", "Income & Savings", "Currency Settings", "Multi-Currency Analytics", "Combined Analytics", "Manage Groups"] and not df.empty:
        st.sidebar.header("🔍 Filters")
        
        # Date range filter
        min_date = df['Date'].min().date()
        max_date = df['Date'].max().date()
        
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        # Apply date filters
        if len(date_range) == 2:
            df = df[
                (df['Date'].dt.date >= date_range[0]) &
                (df['Date'].dt.date <= date_range[1])
            ]
    
    # Show selected page
    if page == "Overview":
        if not df.empty:
            show_overview(df)
        else:
            st.info("No data available. Go to Data Management to import data.")
    elif page == "Analytics":
        if not df.empty:
            show_analytics(df)
        else:
            st.info("No data available. Go to Data Management to import data.")
    elif page == "Income & Savings":
        show_income_tracking()
    elif page == "Data Management":
        show_data_management()
    elif page == "Currency Settings":
        show_currency_settings()
    elif page == "Multi-Currency Analytics":
        show_multi_currency_analytics()
    elif page == "Combined Analytics":
        show_combined_analytics()
    elif page == "Manage Groups":
        show_manage_groups_page()

if __name__ == "__main__":
    main()
