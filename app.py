import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import io
import json
from data_manager import DataManager

# Page configuration
st.set_page_config(
    page_title="ExpenseInfo - Splitwise Expense Analyzer",
    page_icon="💰",
    layout="wide"
)

# Initialize data manager
@st.cache_resource
def get_data_manager():
    """Singleton data manager"""
    return DataManager()

def convert_df_to_transactions(df):
    """Convert DataFrame to transaction list"""
    transactions = []
    for _, row in df.iterrows():
        txn = {
            'date': row['Date'].isoformat() if pd.notna(row['Date']) else None,
            'description': str(row['Description']) if pd.notna(row['Description']) else '',
            'category': str(row['Category']) if pd.notna(row['Category']) else '',
            'cost': float(row['Cost']) if pd.notna(row['Cost']) else 0.0,
            'currency': str(row['Currency']) if pd.notna(row['Currency']) else 'ILS',
            'source': 'import'
        }
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

def load_persisted_data():
    """Load data from persistence layer"""
    dm = get_data_manager()
    return dm.get_dataframe()

def get_member_columns(df):
    """Identify columns that represent member names (not standard columns)"""
    standard_cols = ['Date', 'Description', 'Category', 'Cost', 'Currency']
    member_cols = [col for col in df.columns if col not in standard_cols]
    return member_cols

def exclude_payments(df):
    """Exclude Payment and Settlement categories from calculations"""
    exclude_categories = ['Payment', 'Settlement', 'payment', 'settlement']
    return df[~df['Category'].isin(exclude_categories)]

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
    
    dm = get_data_manager()
    data = dm.load_data()
    
    # Current dataset info
    st.subheader("📊 Current Dataset")
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
                    
                    st.success(f"✅ Added {result['added']} new transactions")
                    
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
    # Exclude payments for metrics
    df_expenses = exclude_payments(df)
    
    # Calculate metrics
    total_spending = df_expenses['Cost'].sum()
    
    # Historic monthly average
    df_expenses_copy = df_expenses.copy()
    df_expenses_copy['YearMonth'] = df_expenses_copy['Date'].dt.to_period('M')
    monthly_totals = df_expenses_copy.groupby('YearMonth')['Cost'].sum()
    historic_avg = monthly_totals.mean() if len(monthly_totals) > 0 else 0
    
    # Current month spending
    current_month = datetime.now().replace(day=1).date()
    current_month_data = df_expenses[df_expenses['Date'].dt.date >= current_month]
    current_month_spending = current_month_data['Cost'].sum()
    
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
            category_totals = df_expenses.groupby('Category')['Cost'].sum().reset_index()
            category_totals = category_totals.sort_values('Cost', ascending=False)
            
            fig_pie = px.pie(
                category_totals,
                values='Cost',
                names='Category',
                title='Spending by Category',
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
        else:
            st.info("No expense data available")
    
    with tab2:
        if not df_expenses.empty:
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
                df_expenses = exclude_payments(current_month_df)
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

def show_analytics(df):
    """Show analytics page with detailed charts"""
    st.header("📊 Analytics")
    
    df_all_expenses = exclude_payments(df)
    
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
    monthly_totals = df_monthly.groupby(['YearMonth', 'Category'])['Cost'].sum().reset_index()
    
    fig_monthly = px.bar(
        monthly_totals,
        x='YearMonth',
        y='Cost',
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
    
    # Yearly spending by category
    st.subheader("Yearly Spending by Category")
    selected_category_yearly = st.selectbox(
        "Select Category",
        options=categories,
        key="yearly_category"
    )
    
    df_yearly = df_all_expenses.copy()
    if selected_category_yearly != 'All Categories':
        df_yearly = df_yearly[df_yearly['Category'] == selected_category_yearly]
    
    df_yearly['Year'] = df_yearly['Date'].dt.year.astype(str)
    yearly_totals = df_yearly.groupby(['Year', 'Category'])['Cost'].sum().reset_index()
    
    fig_yearly = px.bar(
        yearly_totals,
        x='Year',
        y='Cost',
        color='Category',
        title=f"Yearly Spending - {selected_category_yearly}",
        barmode='stack'
    )
    
    fig_yearly.update_layout(
        xaxis_title="Year",
        yaxis_title="Amount (₪)",
        xaxis_type='category',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )
    
    st.plotly_chart(fig_yearly, use_container_width=True)
    
    st.markdown("---")
    
    # Year-over-Year analysis
    st.subheader("Year-over-Year Monthly Average")
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
    
    monthly_totals = df_yoy.groupby(['Year', 'Month', 'Category'])['Cost'].sum().reset_index()
    yoy_avg = monthly_totals.groupby(['Year', 'Category'])['Cost'].mean().reset_index()
    yoy_avg.columns = ['Year', 'Category', 'Monthly Average']
    yoy_avg['Year'] = yoy_avg['Year'].astype(str)
    
    fig_yoy = px.line(
        yoy_avg,
        x='Year',
        y='Monthly Average',
        color='Category',
        title=f"Monthly Average by Category (Year-over-Year) - {selected_category_yoy}",
        markers=True
    )
    
    fig_yoy.update_layout(
        xaxis_title="Year",
        yaxis_title="Monthly Average (₪)",
        xaxis_type='category',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )
    
    st.plotly_chart(fig_yoy, use_container_width=True)

def main():
    st.title("💰 streamlit-splitwise-dashboard")
    
    # Check if data exists
    dm = get_data_manager()
    
    if not dm.data_exists():
        # First-time setup
        show_setup_wizard()
        return
    
    # Load data
    df = load_persisted_data()
    
    # Navigation
    st.sidebar.header("📊 Navigation")
    page = st.sidebar.radio("Select Page", ["Overview", "Analytics", "Income & Savings", "Data Management"])
    
    # Filters (if not on data management or income page)
    if page not in ["Data Management", "Income & Savings"] and not df.empty:
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
        
        # Apply filters
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

if __name__ == "__main__":
    main()
