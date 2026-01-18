import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import io
import json
from data_manager import DataManager
from groups_manager import GroupsManager
from user_expense_calculator import UserExpenseCalculator

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

def get_data_manager(group_data_path=None):
    """Get data manager for specific group or active group"""
    if group_data_path is None:
        groups_mgr = get_groups_manager()
        if groups_mgr.has_groups():
            active_group = groups_mgr.get_active_group()
            group_data_path = groups_mgr.get_group_data_path(active_group['id'])
    return DataManager(group_data_path)

def convert_df_to_transactions(df):
    """Convert DataFrame to transaction list, preserving all columns including member splits"""
    transactions = []
    
    # Standard columns
    standard_cols = ['Date', 'Description', 'Category', 'Cost', 'Currency']
    
    for _, row in df.iterrows():
        txn = {
            'date': row['Date'].isoformat() if pd.notna(row['Date']) else None,
            'description': str(row['Description']) if pd.notna(row['Description']) else '',
            'category': str(row['Category']) if pd.notna(row['Category']) else '',
            'cost': float(row['Cost']) if pd.notna(row['Cost']) else 0.0,
            'currency': str(row['Currency']) if pd.notna(row['Currency']) else 'ILS',
            'source': 'import'
        }
        
        # Preserve all other columns (member splits, etc.)
        for col in df.columns:
            if col not in standard_cols and col not in txn:
                # Add any non-standard column (like member names)
                if pd.notna(row[col]):
                    txn[col] = float(row[col]) if isinstance(row[col], (int, float)) else str(row[col])
        
        transactions.append(txn)
    return transactions

@st.cache_data
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
    """Update group members list based on detected member columns in data"""
    dm = get_data_manager()
    member_cols = dm.get_member_columns(df)
    
    if member_cols:
        groups_mgr = get_groups_manager()
        group = groups_mgr.get_group_by_id(group_id)
        
        if group:
            # Update members if they're not already set or are different
            current_members = set(group.get('members', []))
            new_members = set(member_cols)
            
            if current_members != new_members:
                groups_mgr.update_group(group_id, members=list(new_members))
                return list(new_members)
    
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

def show_setup_wizard():
    """First-time setup wizard"""
    st.title("👋 Welcome to streamlit-splitwise-dashboard !")
    st.markdown("Let's set up your expense tracking system.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Option 1: Import Existing Data")
        st.markdown("Upload your Splitwise export file to get started.")
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'xlsx', 'xls'],
            help="Upload CSV or Excel file. For Hebrew names, use Excel exported from Google Sheets."
        )
        
        if uploaded_file:
            file_type = 'csv' if uploaded_file.name.endswith('.csv') else 'excel'
            
            if st.button("Import Data", type="primary"):
                with st.spinner("Processing your data..."):
                    df = load_data_from_file(uploaded_file, file_type)
                    
                    if df is not None and not df.empty:
                        # Convert to transactions and save
                        dm = get_data_manager()
                        transactions = convert_df_to_transactions(df)
                        result = dm.append_transactions(transactions)
                        
                        # Update group members if groups exist
                        groups_mgr = get_groups_manager()
                        if groups_mgr.has_groups():
                            active_group = groups_mgr.get_active_group()
                            update_group_members_from_data(active_group['id'], df)
                        
                        st.success(f"✅ Successfully imported {result['added']} transactions!")
                        if result['skipped'] > 0:
                            st.info(f"ℹ️ Skipped {result['skipped']} duplicate transactions")
                        
                        st.balloons()
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
            with st.spinner("Processing file..."):
                df = load_data_from_file(uploaded_file, file_type)
                
                if df is not None and not df.empty:
                    transactions = convert_df_to_transactions(df)
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
                    
                    # Clear cache and reload
                    st.cache_data.clear()
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
    
    # Backups section
    st.subheader("💾 Backups")
    
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
        
        st.dataframe(
            display_df[['Date', 'Description', 'Category', 'Cost', 'Currency']].sort_values('Date', ascending=False),
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
        # Month filter for all plots (except timeline)
        combined_df_filtered = combined_df.copy()
        combined_df_filtered['YearMonth'] = combined_df_filtered['Date'].dt.to_period('M')
        available_months_all = sorted(combined_df_filtered['YearMonth'].unique(), reverse=True)
        
        month_options_all = ['All Time'] + [str(m) for m in available_months_all]
        selected_month_all = st.selectbox(
            "Filter by Month",
            options=month_options_all,
            key="combined_all_month_filter"
        )
        
        # Apply month filter if selected
        if selected_month_all != 'All Time':
            combined_df_filtered = combined_df_filtered[combined_df_filtered['YearMonth'] == pd.Period(selected_month_all)]
        
        # Show metrics for all expenses
        col1, col2, col3, col4 = st.columns(4)
        
        # Calculate Total Spent as sum of member expenses from ORIGINAL unfiltered data (matches Overview ₪222k)
        dm = get_data_manager()
        member_cols = dm.get_member_columns(combined_df_original) if not combined_df_original.empty else []
        
        total_spent = 0.0
        if member_cols:
            for member_col in member_cols:
                if member_col in combined_df_original.columns:
                    # Convert to numeric to avoid string comparison errors
                    member_values = pd.to_numeric(combined_df_original[member_col], errors='coerce').fillna(0)
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
        
        # Month filter for individual member view - at the top
        member_txns = member_data.get('transactions', [])
        selected_month_member = 'All Time'
        
        if member_txns and len(member_txns) > 0:
            member_txns_df_filter = pd.DataFrame(member_txns)
            
            if 'date' in member_txns_df_filter.columns:
                member_txns_df_filter['date'] = pd.to_datetime(member_txns_df_filter['date'])
                member_txns_df_filter['YearMonth'] = member_txns_df_filter['date'].dt.to_period('M')
                available_months_member = sorted(member_txns_df_filter['YearMonth'].unique(), reverse=True)
                
                month_options_member = ['All Time'] + [str(m) for m in available_months_member]
                selected_month_member = st.selectbox(
                    "Filter by Month",
                    options=month_options_member,
                    key="member_all_month_filter"
                )
        
        # Calculate how much they're owed FROM or owe TO everyone else
        all_members_data = user_calc.calculate_all_members_expenses(selected_group_ids)
        
        # For this member, calculate total debt to/from ALL other members
        total_debt_to_member = 0  # How much others owe to this member
        total_debt_from_member = 0  # How much this member owes to others
        
        for other_member, other_data in all_members_data.items():
            if other_member != selected_member:
                # If other member has negative balance, they owe money
                # If this member should receive that money, add to debt_to_member
                # This is approximate - real debt tracking needs transaction-level analysis
                other_balance = other_data['net_balance']
                if other_balance < 0:  # Other person owes
                    # They might owe to our member
                    pass  # Skip for now, use simple net balance
        
        # Simplified: just show net balance from member's perspective
        # Member metrics - removed redundant Total Owed By Them
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Personal Expense", f"₪{member_data['total_expense']:,.2f}", 
                     help="Total amount this person actually paid")
        with col2:
            net_balance = member_data['net_balance']
            if net_balance > 0:
                # Positive net = paid more than owed = they are OWED money
                balance_label = "They Are Owed"
                balance_help = "This person is owed money (paid more than their share)"
            elif net_balance < 0:
                # Negative net = owed more than paid = they OWE money
                balance_label = "They Owe"
                balance_help = "This person owes money (paid less than their share)"
            else:
                balance_label = "Balanced"
                balance_help = "This person is settled up"
            st.metric(balance_label, f"₪{abs(net_balance):,.2f}",
                     help=balance_help)
        
        st.markdown("---")
        
        # Category breakdown for member
        st.subheader(f"📊 {selected_member}'s Expense Breakdown")
        
        # Get member transactions for month filtering
        if member_txns and len(member_txns) > 0:
            member_txns_df = pd.DataFrame(member_txns)
            
            # Check if date column exists and convert
            if 'date' in member_txns_df.columns:
                member_txns_df['date'] = pd.to_datetime(member_txns_df['date'])
                member_txns_df['YearMonth'] = member_txns_df['date'].dt.to_period('M')
                
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
        
        # Apply month filter to group spending if available
        if member_txns and len(member_txns) > 0:
            member_txns_df_group = pd.DataFrame(member_txns)
            
            if 'date' in member_txns_df_group.columns and selected_month_member != 'All Time':
                member_txns_df_group['date'] = pd.to_datetime(member_txns_df_group['date'])
                member_txns_df_group['YearMonth'] = member_txns_df_group['date'].dt.to_period('M')
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
                # No month filter, calculate from all groups
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
        
        # Transaction list
        st.subheader(f"{selected_member}'s Transactions")
        member_txns = member_data['transactions']
        
        if member_txns:
            df_txns = pd.DataFrame(member_txns)
            df_display = df_txns[['date', 'group_name', 'description', 'category', 
                                  'total_cost', 'member_share']].copy()
            df_display.columns = ['Date', 'Group', 'Description', 'Category', 
                                 'Total Cost', 'Your Share']
            df_display = df_display.sort_values('Date', ascending=False)
            
            st.dataframe(df_display, use_container_width=True, hide_index=True)
            
            # Export option
            csv = df_display.to_csv(index=False)
            st.download_button(
                label=f"📥 Download {selected_member}'s Transactions",
                data=csv,
                file_name=f"{selected_member}_expenses.csv",
                mime="text/csv"
            )

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
                    groups_mgr.delete_group(group_to_delete['id'])
                    st.session_state.confirm_delete_group = None
                    st.success("Group deleted!")
                    st.rerun()
            with col2:
                if st.button("Cancel", key="confirm_delete_cancel"):
                    st.session_state.confirm_delete_group = None
                    st.rerun()

def show_analytics(df):
    """Show analytics page with detailed charts"""
    st.header("📊 Analytics")
    
    df_all_expenses = exclude_payments_and_reimbursements(df)
    
    if df_all_expenses.empty:
        st.info("No expense data available for analysis")
        return
    
    # Monthly spending by category
    st.subheader("Monthly Spending by Category")
    categories = ['All Categories'] + sorted(df_all_expenses['Category'].unique().tolist())
    selected_category_monthly = st.selectbox(
        "Select Category",
        options=categories,
        key="monthly_category"
    )
    
    df_monthly = df_all_expenses.copy()
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

def main():
    st.title("💰 streamlit-splitwise-dashboard")
    
    # Initialize session state for modals
    if 'show_new_group_modal' not in st.session_state:
        st.session_state.show_new_group_modal = False
    if 'show_manage_groups' not in st.session_state:
        st.session_state.show_manage_groups = False
    if 'navigate_to_manage_groups' not in st.session_state:
        st.session_state.navigate_to_manage_groups = False
    
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
    
    page_options = ["Overview", "Analytics", "Income & Savings", "Data Management", "Combined Analytics", "Manage Groups"]
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
    if page not in ["Data Management", "Income & Savings", "Combined Analytics", "Manage Groups"] and not df.empty:
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
    elif page == "Combined Analytics":
        show_combined_analytics()
    elif page == "Manage Groups":
        show_manage_groups_page()

if __name__ == "__main__":
    main()
