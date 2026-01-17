# ExpenseInfo 💰

A Streamlit-based Python application to visualize and analyze Splitwise expense data in an interactive and informative way, with persistent data storage and comprehensive data management.

## Features

### 💾 Data Management
- **Persistent Storage**: Your data is saved locally - no need to re-upload files each session
- **Automatic Backups**: Every save creates a backup (keeps 5 most recent)
- **Duplicate Detection**: Automatically identifies and skips duplicate transactions
- **Multiple Import Formats**: Upload CSV or Excel files
- **Multiple Export Formats**: Export to CSV, Excel, or JSON
- **Data Integrity**: Validates all transactions before saving
- **Concurrent Access Protection**: File locking prevents data corruption

### 📊 Visualization & Analysis
- **Overview Page**:
  - Total spending metrics with historic monthly average comparison
  - Category breakdown (interactive pie chart)
  - Spending trends over time
  - Searchable transaction table
  
- **Analytics Page**:
  - Monthly spending by category
  - Yearly spending by category
  - Year-over-Year monthly averages
  - Interactive filtering by category

### 🌐 Multi-Language Support
- **Hebrew Character Support**: Handles Hebrew names correctly (use Excel export from Google Sheets for best results)
- **Smart Payment Filtering**: Automatically excludes "Payment" and "Settlement" categories from spending calculations

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd ExpenseInfo
```

2. Install dependencies:
```bash
python3 -m pip install -r requirements.txt
```

## Usage

### First Time Setup

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser to the URL shown (typically http://localhost:8501)

3. Choose your setup option:
   - **Option 1: Import Existing Data**
     - Upload your Splitwise export file
     - For Hebrew names: Export CSV from Splitwise → Open in Google Sheets → Download as Excel (.xlsx)
     - Data will be saved and automatically loaded in future sessions
   
   - **Option 2: Start Fresh**
     - Begin with an empty dataset
     - Add transactions manually or import files later

### Subsequent Usage

1. Run the app - your data loads automatically
2. Navigate between pages:
  **No re-uploads needed**: Your data persists between sessions
- **Safe to import**: Duplicate transactions are automatically skipped
- **Automatic backups**: Your data is backed up before every save
- **Data stays local**: All data stored on your device for privacy
- Use the sidebar filters to focus on specific time periods
- Search the transaction table to find specific expenses
- Export your data anytime in your preferred format
- Restore from backups if you need to undo changes

## Data Storage

All data is stored locally in the `user_data/` directory:
- `transactions.json` - Your main data file
- `backups/` - Automatic backups (5 most recent kept)
- `cache/` - Performance cache files

This directory is git-ignored for privacy and security.
### Data Management

- **Import**: Upload CSV/Excel files, duplicates automatically detected
- **Export**: Download your data in CSV, Excel, or JSON format
- **Backups**: View and restore from automatic backups
- **Clear Data**: Reset the app to start fresh (with confirmation)

## Expected File Format

Your file should contain these columns:
- `Date`: Transaction date
- `Description`: Expense description
- `Category`: Expense category (Food, Transport, etc.)
- `Cost`: Amount spent (numeric)
- `Currency`: Currency code
- Additional columns with member names (in any language including Hebrew)

## Tips

- The app automatically detects member columns (any column not in the standard set)
- Use the sidebar filters to focus on specific time periods
- Search the transaction table to find specific expenses
- Download filtered data as CSV for further analysis
- Negative net contributions = member owes money
- Positive net contributions = member is owed money

## Requirements

- Python 3.8+
- streamlit
- pandas
- plotly
- openpyxl (for Excel support)
- numpy
