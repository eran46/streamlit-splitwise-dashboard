import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Splitwise Expense Analyzer",
    page_icon="💰",
    layout="wide"
)

st.title("💰 Splitwise Expense Analyzer")

# Page navigation
st.sidebar.header("📊 Navigation")
page = st.sidebar.radio("Select Page", ["Overview", "Analytics"])

# File upload section
st.sidebar.header("📁 Upload Data")
uploaded_file = st.sidebar.file_uploader(
    "Choose a file",
    type=['csv', 'xlsx', 'xls'],
    help="Upload CSV or Excel file. For Hebrew names, use Excel exported from Google Sheets."
)

@st.cache_data
def load_data(file, file_type):
    """Load and preprocess the expense data"""
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

def get_member_columns(df):
    """Identify columns that represent member names (not standard columns)"""
    standard_cols = ['Date', 'Description', 'Category', 'Cost', 'Currency']
    member_cols = [col for col in df.columns if col not in standard_cols]
    return member_cols

def exclude_payments(df):
    """Exclude Payment and Settlement categories from calculations"""
    exclude_categories = ['Payment', 'Settlement', 'payment', 'settlement']
    return df[~df['Category'].isin(exclude_categories)]

if uploaded_file is not None:
    # Determine file type
    file_type = 'csv' if uploaded_file.name.endswith('.csv') else 'excel'
    
    # Load data
    df = load_data(uploaded_file, file_type)
    
    if df is not None and not df.empty:
        # Get member columns
        member_cols = get_member_columns(df)
        
        # Sidebar filters
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
        
        # Month filter
        df['YearMonth'] = df['Date'].dt.to_period('M')
        available_months = sorted(df['YearMonth'].unique(), reverse=True)
        month_options = ['All Months'] + [str(m) for m in available_months]
        
        selected_month = st.sidebar.selectbox(
            "Filter by Month",
            options=month_options
        )
        
        # Apply filters
        if len(date_range) == 2:
            df_filtered = df[
                (df['Date'].dt.date >= date_range[0]) &
                (df['Date'].dt.date <= date_range[1])
            ]
        else:
            df_filtered = df.copy()
        
        if selected_month != 'All Months':
            df_filtered = df_filtered[df_filtered['YearMonth'] == pd.Period(selected_month)]
        
        # Exclude payments for metrics
        df_expenses = exclude_payments(df_filtered)
        df_all_expenses = exclude_payments(df)
        
        # Calculate metrics
        total_spending = df_expenses['Cost'].sum()
        
        # Historic monthly average (excluding current/selected month if specific month selected)
        if selected_month != 'All Months':
            df_historic = df_all_expenses[df_all_expenses['YearMonth'] != pd.Period(selected_month)]
        else:
            df_historic = df_all_expenses.copy()
        
        if not df_historic.empty:
            monthly_totals = df_historic.groupby('YearMonth')['Cost'].sum()
            historic_avg = monthly_totals.mean() if len(monthly_totals) > 0 else 0
        else:
            historic_avg = 0
        
        # Display metrics
        st.header("📊 Summary Metrics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Total Spending",
                f"₪{total_spending:,.2f}",
                help="Total spending excluding payments and settlements"
            )
        
        with col2:
            st.metric(
                "Historic Monthly Average",
                f"₪{historic_avg:,.2f}",
                help="Average monthly spending (excluding selected period)"
            )
        
        with col3:
            if selected_month != 'All Months' and historic_avg > 0:
                difference = total_spending - historic_avg
                delta_pct = (difference / historic_avg) * 100
                st.metric(
                    "vs. Historic Average",
                    f"₪{difference:,.2f}",
                    f"{delta_pct:+.1f}%",
                    delta_color="inverse"
                )
            else:
                st.metric("Number of Transactions", len(df_expenses))
        
        # Page routing
        if page == "Overview":
            # OVERVIEW PAGE - Tabs for different views
            st.header("📊 Overview")
            
            tab1, tab2 = st.tabs(["Category Breakdown", "Trends & History"])
            
            with tab1:
                # Pie chart
                if not df_expenses.empty:
                    category_totals = df_expenses.groupby('Category')['Cost'].sum().sort_values(ascending=False).reset_index()
                    category_totals.columns = ['Category', 'Cost']
                    
                    fig_pie = px.pie(
                        category_totals,
                        values='Cost',
                        names='Category',
                        hole=0.4,
                        color_discrete_sequence=px.colors.qualitative.Pastel
                    )
                    fig_pie.update_traces(textposition='auto', textinfo='percent+label')
                    st.plotly_chart(fig_pie, use_container_width=True, config={
                        'scrollZoom': False,
                        'displayModeBar': False,
                        'doubleClick': False
                    })
                else:
                    st.info("No expense data available for the selected filters")
            
            with tab2:
                # Monthly spending trend
                if not df_all_expenses.empty:
                    trend_data = df_all_expenses.groupby(df_all_expenses['Date'].dt.to_period('M'))['Cost'].sum().reset_index()
                    trend_data['Date'] = trend_data['Date'].dt.to_timestamp()
                    
                    fig_trend = go.Figure()
                    # Actual Spending
                    fig_trend.add_trace(go.Scatter(
                        x=trend_data['Date'],
                        y=trend_data['Cost'],
                        mode='lines+markers',
                        name='Actual Spending',
                        line=dict(color='#636EFA', width=3)
                    ))
                    # Average Line
                    if historic_avg > 0:
                        fig_trend.add_trace(go.Scatter(
                            x=trend_data['Date'],
                            y=[historic_avg]*len(trend_data),
                            mode='lines',
                            name='Historic Average',
                            line=dict(color='red', dash='dash')
                        ))
                    
                    fig_trend.update_layout(
                        xaxis_title="Month",
                        yaxis_title="Amount (₪)",
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig_trend, use_container_width=True, config={
                        'scrollZoom': False,
                        'displayModeBar': False,
                        'doubleClick': False
                    })
                else:
                    st.info("No expense data available")
            
            # Searchable data table in expander
            with st.expander("🔎 View Transaction Details"):
                # Search box
                search_term = st.text_input("Search transactions (by description or date)", "")
                
                # Filter data based on search
                if search_term:
                    mask = df_filtered.astype(str).apply(
                        lambda x: x.str.contains(search_term, case=False, na=False)
                    ).any(axis=1)
                    df_display = df_filtered[mask]
                else:
                    df_display = df_filtered
                
                # Display transaction count
                st.markdown(f"**Showing {len(df_display)} of {len(df_filtered)} transactions**")
                
                # Format the dataframe for display
                df_show = df_display.copy()
                df_show['Date'] = df_show['Date'].dt.strftime('%Y-%m-%d')
                df_show['Cost'] = df_show['Cost'].apply(lambda x: f"₪{x:,.2f}")
                
                # Remove the YearMonth column from display
                if 'YearMonth' in df_show.columns:
                    df_show = df_show.drop('YearMonth', axis=1)
                
                st.dataframe(
                    df_show,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Download filtered data
                csv = df_display.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label="📥 Download Filtered Data as CSV",
                    data=csv,
                    file_name=f"splitwise_expenses_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                )
        
        else:  # Analytics page
            # ANALYTICS PAGE - All trend charts
            st.header("📈 Analytics")
            
            # Row 1: Monthly and Yearly charts
            col1, col2 = st.columns(2)
            
            with col1:
                if not df_all_expenses.empty:
                    # Category filter dropdown
                    categories = ['All Categories'] + sorted(df_all_expenses['Category'].unique().tolist())
                    selected_category_monthly = st.selectbox(
                        "Select Category",
                        options=categories,
                        key="monthly_category"
                    )
                    
                    # Monthly category spending
                    df_monthly = df_all_expenses.copy()
                    if selected_category_monthly != 'All Categories':
                        df_monthly = df_monthly[df_monthly['Category'] == selected_category_monthly]
                    
                    df_monthly['YearMonth'] = df_monthly['Date'].dt.to_period('M')
                    monthly_category = df_monthly.groupby(['YearMonth', 'Category'])['Cost'].sum().reset_index()
                    monthly_category['YearMonth'] = monthly_category['YearMonth'].astype(str)
                    
                    fig_line = px.line(
                        monthly_category,
                        x='YearMonth',
                        y='Cost',
                        color='Category',
                        title="Monthly Spending by Category",
                        markers=True
                    )
                    
                    fig_line.update_layout(
                        xaxis_title="Month",
                        yaxis_title="Amount (₪)",
                        hovermode='x unified',
                        legend=dict(
                            orientation="v",
                            yanchor="top",
                            y=1,
                            xanchor="left",
                            x=1.02
                        )
                    )
                    
                    st.plotly_chart(fig_line, use_container_width=True, config={
                        'scrollZoom': False,
                        'displayModeBar': False,
                        'doubleClick': False
                    })
                else:
                    st.info("No expense data available for the selected filters")
            
            with col2:
                if not df_all_expenses.empty:
                    # Category filter dropdown
                    categories = ['All Categories'] + sorted(df_all_expenses['Category'].unique().tolist())
                    selected_category_yearly = st.selectbox(
                        "Select Category",
                        options=categories,
                        key="yearly_category"
                    )
                    
                    # Yearly category spending
                    df_yearly = df_all_expenses.copy()
                    if selected_category_yearly != 'All Categories':
                        df_yearly = df_yearly[df_yearly['Category'] == selected_category_yearly]
                    
                    df_yearly['Year'] = df_yearly['Date'].dt.year.astype(str)
                    yearly_category = df_yearly.groupby(['Year', 'Category'])['Cost'].sum().reset_index()
                    
                    fig_yearly = px.bar(
                        yearly_category,
                        x='Year',
                        y='Cost',
                        color='Category',
                        title="Yearly Spending by Category",
                        barmode='group'
                    )
                    
                    fig_yearly.update_layout(
                        xaxis_title="Year",
                        yaxis_title="Amount (₪)",
                        legend=dict(
                            orientation="v",
                            yanchor="top",
                            y=1,
                            xanchor="left",
                            x=1.02
                        )
                    )
                    
                    st.plotly_chart(fig_yearly, use_container_width=True, config={
                        'scrollZoom': False,
                        'displayModeBar': False,
                        'doubleClick': False
                    })
                else:
                    st.info("No expense data available")
            
            # Row 2: Year-over-Year monthly average
            if not df_all_expenses.empty:
                # Category filter dropdown
                categories = ['All Categories'] + sorted(df_all_expenses['Category'].unique().tolist())
                selected_category_yoy = st.selectbox(
                    "Select Category",
                    options=categories,
                    key="yoy_category"
                )
                
                # Calculate monthly averages per year per category
                df_yoy = df_all_expenses.copy()
                if selected_category_yoy != 'All Categories':
                    df_yoy = df_yoy[df_yoy['Category'] == selected_category_yoy]
                
                df_yoy['Year'] = df_yoy['Date'].dt.year
                df_yoy['Month'] = df_yoy['Date'].dt.month
                
                # Group by year, month, category and sum
                monthly_totals = df_yoy.groupby(['Year', 'Month', 'Category'])['Cost'].sum().reset_index()
                
                # Calculate average per month per category per year
                yoy_avg = monthly_totals.groupby(['Year', 'Category'])['Cost'].mean().reset_index()
                yoy_avg.columns = ['Year', 'Category', 'Monthly Average']
                yoy_avg['Year'] = yoy_avg['Year'].astype(str)
                
                fig_yoy = px.line(
                    yoy_avg,
                    x='Year',
                    y='Monthly Average',
                    color='Category',
                    title="Monthly Average by Category (Year-over-Year)",
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
                
                st.plotly_chart(fig_yoy, use_container_width=True, config={
                    'scrollZoom': False,
                    'displayModeBar': False,
                    'doubleClick': False
                })
            else:
                st.info("No expense data available")
    
    elif df is not None:
        st.warning("The uploaded file is empty or could not be processed.")

else:
    # Instructions when no file is uploaded
    st.info("👆 Please upload a CSV or Excel file to get started")
    
    st.markdown("""
    ### 📋 Expected File Format
    
    Your file should contain the following columns:
    - **Date**: Transaction date
    - **Description**: Description of the expense
    - **Category**: Expense category (e.g., Food, Transport)
    - **Cost**: Amount spent
    - **Currency**: Currency code
    - **Member Names**: Additional columns with individual member names (in Hebrew or any language)
    
    ### 💡 Tips for Hebrew Characters
    
    If you're seeing mangled Hebrew characters in your CSV:
    1. Open the CSV file in Google Sheets
    2. File → Download → Microsoft Excel (.xlsx)
    3. Upload the downloaded Excel file here
    
    ### 🎯 Features
    
    - **Automatic Payment Exclusion**: Payments and settlements are automatically excluded from spending calculations
    - **Date Range Filtering**: Filter transactions by custom date range
    - **Monthly Analysis**: Compare spending against historic averages
    - **Category Breakdown**: Interactive pie chart for category-wise spending
    - **Trend Analysis**: Visualize spending patterns over time
    - **Member Contributions**: See who owes whom with a clear bar chart
    - **Searchable Table**: Find specific transactions quickly
    """)
