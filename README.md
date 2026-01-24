# streamlit-splitwise-dashboard 💰

A powerful Streamlit-based Python application for visualizing and analyzing Splitwise expense data. Transform your Splitwise exports into beautiful, interactive dashboards with comprehensive analytics, multi-currency support, multi-group management, notes & tags, and smart expense tracking.

## 🎯 Who Is This For?

This app is perfect for **Splitwise power users** who:
- **Consistently track expenses** through Splitwise (households, roommates, travel groups, etc.)
- Want **deeper insights** beyond what Splitwise provides
- Need to **analyze spending patterns** across multiple groups and currencies
- Track both **personal expenses** and **household/group costs**
- Want to **monitor income vs expenses** and savings rates
- Need **historical analysis** and trend visualization
- Value **privacy** with local data storage
- Work with **multiple currencies** and need unified reporting

## 📋 Requirements

### What You Need:
1. **Active Splitwise Usage**: Regular expense tracking in Splitwise
2. **Python 3.8+** installed on your computer
3. **Splitwise Export Files**: CSV exports from your Splitwise groups
4. **Basic Terminal Knowledge**: To run Python commands

### How to Export from Splitwise:

#### Option 1: Splitwise Website (Recommended)
1. Go to [Splitwise.com](https://www.splitwise.com) and log in
2. Navigate to your group (or "All expenses" for everything)
3. Click on the **⚙️ Settings** icon (top right)
4. Select **"Export as spreadsheet"**
5. Choose **CSV** format and click **Download**
6. Save the file to your computer

#### Option 2: Via Google Sheets (Best for Hebrew/Non-English Names)
1. Export CSV from Splitwise (as above)
2. Open the CSV file in **Google Sheets**
3. Go to **File → Download → Microsoft Excel (.xlsx)**
4. This preserves character encoding for Hebrew and other languages

**Note**: For multiple groups, export each group separately and import them individually into the app.

## ✨ Features

### 📊 Overview Dashboard
- **Total Spending Metrics**: See aggregate personal expenses across all members
- **Historic Monthly Average**: Compare current month spending to historical patterns
- **Income vs Expenses (Current Month)**: Complete financial snapshot with:
  - Monthly income tracking
  - Current month expenses with delta comparison
  - Expense ratio (spending as % of income)
  - Savings amount and savings rate percentage
- **Multi-Currency Support**: Automatic display in your chosen target currency
- **Category Breakdown**: Visual pie chart of spending by category (excluding internal transfers)
- **Timeline Visualization**: Track spending trends over time with interactive charts
- **Smart Filtering**: Automatically excludes Payment/Settlement categories and reimbursement transactions
- **Transaction Management**: View all transactions with integrated notes and tags

### 🏷️ Notes & Tags (NEW!)
- **Transaction Notes**: Add custom notes to any transaction for context and tracking
- **Flexible Tagging**: Tag transactions with custom labels (vacation, medical, business, etc.)
- **Quick Filters**: Filter transactions by tags in sidebar with transaction counts
- **Tag Analytics**: 
  - Visual spending breakdown by tags (pie & bar charts)
  - Tag spending totals and averages
  - Track expenses across custom categories beyond Splitwise defaults
- **Search Functionality**: Search across transaction descriptions and notes
- **Popular Tags**: Quick-add buttons for frequently used tags
- **Persistent Metadata**: Notes and tags saved separately, never lost
- **Integrated UI**: Click ✏️ button on any transaction to add notes/tags via popup editor

### 💱 Multi-Currency Support (NEW!)
- **Unified Currency Display**: Convert and display all transactions in your chosen target currency
- **Automatic Conversion**: Real-time currency conversion using live exchange rates
- **Historical Accuracy**: Transactions converted using rates from their transaction dates
- **Original Amount Preservation**: View both converted and original amounts with currency symbols
- **Multiple Currency Tracking**: 
  - See spending breakdown by original currency
  - Foreign currency exposure metrics and percentages
  - Currency-specific transaction counts
- **Flexible Currency Management**: 
  - Set target currency for unified reporting
  - Switch target currency and auto-convert existing data
  - Support for ILS, USD, EUR, GBP, JPY, and more
- **Currency Analytics Page**: Dedicated page for:
  - Currency trends over time
  - Exchange rate visualizations
  - Spending patterns by currency
  - Conversion impact analysis

### 📈 Expense Breakdown (Analytics)
- **Monthly Spending Analysis**: Detailed breakdown by category with stacked visualizations
- **Yearly Spending Overview**: Interactive pie chart showing category distribution by year
- **Year-over-Year Comparison**: Track spending patterns with customizable date ranges
- **Cost-Based Calculations**: All plots show actual transaction costs (not just individual shares)
- **Interactive Charts**: Hover details, zoom, and filter capabilities

### 💰 Income & Savings Tracking
- **Income Sources Management**: Track multiple income streams with custom labels
- **Savings Rate Calculator**: Automatic calculation of monthly savings percentage
- **Income vs Expenses**: Visual comparison of monthly income and spending
- **Historical Income Tracking**: View income history and trends over time
- **Monthly Breakdown**: Add, edit, and delete income entries by month

### 👥 Multi-Group Support
- **Multiple Groups**: Manage data from different Splitwise groups separately
- **Group Switching**: Easily switch between groups with visual selectors (emojis + names)
- **Active Group Tracking**: Set which group you're currently viewing
- **Group Management Page**: Create, edit, delete, and organize groups
- **Per-Group Data**: Each group maintains separate transactions, notes, tags, and metadata
- **Cross-Group Analysis**: Analyze spending across multiple groups simultaneously
- **Individual Member Views**: Deep dive into any member's spending patterns

### 🔄 Cross-Group Analysis (Multi-Group)
- **All Group Expenses View**:
  - Total spending across all selected groups
  - Spending distribution by group (pie + bar charts)
  - Category breakdown across groups with color-coded visualization
  - Timeline tracking across multiple groups
  - Month-based filtering for all visualizations
  
- **Individual Member View**:
  - Track any member's expenses across all groups
  - Personal expense breakdown by category
  - Spending by group comparison
  - Transaction history with search and filtering
  - Net balance tracking (owed vs owes)

### 💾 Data & Backups Management
- **Group-Specific Data Management**: Select which group to import/export data for
- **Persistent Storage**: Data saved locally - no need to re-upload each session
- **Automatic Backups**: Every save creates a backup (keeps 5 most recent)
- **Backup Restoration**: Restore from any backup with one click
- **Duplicate Detection**: Smart detection prevents duplicate transactions
- **Multiple Import Formats**: Support for CSV and Excel files
- **Multiple Export Formats**: Export to CSV, Excel, or JSON
- **Data Integrity**: Validates all transactions before saving
- **Concurrent Access Protection**: File locking prevents data corruption
- **Member Auto-Discovery**: Automatically detects and updates group members from imported data
- **Data Caching**: Smart caching for improved performance
- **Database Migrations**: Automatic data schema updates when adding new features

### 🏢 Group Management
- **Create Multiple Groups**: Set up separate groups with custom names and emojis
- **Edit Group Details**: Change names, emojis, descriptions, and members anytime
- **Delete Groups**: Remove groups with confirmation (includes data cleanup)
- **Set Active Group**: Choose which group to view in Overview/Analytics via sidebar
- **Visual Organization**: Emoji-based identification for quick group recognition
- **Group Statistics**: View transaction counts and total spending per group

### ⚙️ Currency Settings
- **Set Target Currency**: Choose your primary reporting currency
- **Exchange Rate Configuration**: Manual rate entry or API-based rates
- **Rate History**: View and manage historical exchange rates
- **Bulk Conversion**: Convert all existing transactions to new target currency
- **Custom Rate Sources**: Configure preferred exchange rate providers

### 🎨 Smart Features
- **Payment Filtering**: Automatically excludes "Payment" and "Settlement" transactions from spending analysis
- **Reimbursement Detection**: Identifies and filters out "owed full amount" transactions (where member debt equals transaction cost)
- **Multi-Language Support**: Full support for Hebrew and other non-English characters
- **Responsive Design**: Clean, intuitive interface with Streamlit
- **Interactive Visualizations**: Plotly-powered charts with hover details and zoom
- **Real-time Filtering**: Date range selectors with automatic recalculation
- **Pagination**: Efficient display of large transaction lists (10/25/50/100 per page)
- **Search & Filter**: Powerful search across descriptions and notes, filter by tags
- **Performance Optimized**: Smart caching and lazy loading for fast response times

## 🚀 Installation

1. **Clone this repository**:
```bash
git clone <repository-url>
cd ExpenseInfo
```

2. **Install dependencies**:
```bash
python3 -m pip install -r requirements.txt
```

## 📖 Usage

### First Time Setup

1. **Run the Streamlit app**:
```bash
streamlit run app.py
```

2. **Open your browser** to the URL shown (typically http://localhost:8501)

3. **Create Your First Group**:
   - Go to **"Groups"** in the sidebar
   - Click **"Create New Group"**
   - Enter a name (e.g., "Home 🏡" or "Travel ✈️")
   - Choose an emoji for visual identification
   - Click **Create**

4. **Import Your Splitwise Data**:
   - Navigate to **"Data & Backups"** page
   - Select the group you want to import data for
   - Click **"Choose files"** and upload your Splitwise CSV/Excel export
   - The app will automatically:
     - Detect and add group members
     - Skip duplicate transactions
     - Save your data locally
     - Create an automatic backup

5. **Start Exploring**:
   - Switch to **"Overview"** to see your spending dashboard
   - Use **"Expense Breakdown"** for detailed analytics
   - Add notes and tags to transactions for better organization
   - Track income in **"Income & Savings"**
   - Compare groups in **"Cross-Group Analysis"**
   - View currency breakdown in **"Currency Analysis"**

### Subsequent Usage

Once set up, simply run `streamlit run app.py` and:
- ✅ Your data loads automatically
- ✅ All groups are preserved
- ✅ No need to re-upload files
- ✅ Continue where you left off

### Adding More Data

1. Export new data from Splitwise (follow export instructions above)
2. Go to **"Data & Backups"**
3. Select the appropriate group
4. Upload the new file
5. Duplicates are automatically skipped - safe to import overlapping data!

### Using Notes & Tags

1. **Add Notes to Transactions**:
   - In Overview, find "All Transactions" section
   - Click the ✏️ button on any transaction
   - Type your note in the text area
   - Click "💾 Save Notes"

2. **Tag Transactions**:
   - Click ✏️ on a transaction
   - Type a tag name (e.g., "vacation", "medical", "business")
   - Click "➕ Add" or use quick-add buttons for popular tags
   - Remove tags by clicking × next to them

3. **Filter & Search**:
   - Use sidebar "Filter by Tags" to show only tagged transactions
   - Use search box to find transactions by description or notes
   - View "Tag Analytics" for spending breakdown by tags

### Working with Multiple Currencies

1. **Set Target Currency**:
   - Go to **"Data & Backups"**
   - Upload your first Splitwise export
   - Choose target currency when prompted
   - All future imports will convert to this currency

2. **View Currency Analytics**:
   - Navigate to **"Currency Analysis"** page
   - See spending by original currency
   - View exchange rate trends
   - Check foreign currency exposure

3. **Change Target Currency**:
   - Go to **"Currency Settings"**
   - Select new target currency
   - Optionally re-convert all existing transactions

## 📁 Data Storage

All data is stored locally in the `user_data/` directory:

```
user_data/
├── groups/
│   ├── groups_config.json       # Group configuration
│   └── [group_id]/
│       ├── transactions.json    # Transaction data
│       ├── transaction_metadata.json  # Notes & tags (NEW!)
│       ├── income_data.json     # Income tracking
│       ├── backups/
│       │   ├── transactions_backup_*.json
│       │   └── income_backup_*.json
│       └── cache/               # Performance cache (NEW!)
└── currency_manager/
    ├── settings.json            # Currency settings (NEW!)
    └── exchange_rates.json      # Exchange rate history (NEW!)
```

**Privacy Note**: This directory is git-ignored. Your financial data stays private and local.

## 📊 Expected File Format

Your Splitwise export should contain these columns:

| Column | Description | Example |
|--------|-------------|---------|
| `Date` | Transaction date | 2024-01-15 |
| `Description` | Expense description | Groceries |
| `Category` | Expense category | Food |
| `Cost` | Total transaction amount | 8500.00 |
| `Currency` | Currency code | ILS, USD, EUR |
| `Member Names` | One column per member | Positive = paid, Negative = owes |

**Important Notes**:
- Member columns are **auto-detected** (any column not in the standard set)
- Positive values = Member paid this amount
- Negative values = Member owes this amount
- Hebrew and special characters are fully supported
- Payment/Settlement transactions are automatically filtered from analysis

## 💡 Tips & Tricks

### Understanding the Numbers
- **Total Spending**: Sum of all positive member values = what everyone paid out of pocket
- **Plot Amounts**: Show actual transaction costs (excluding internal transfers)
- **Net Balance**: Positive = owed money, Negative = owes money
- **Currency Display**: All amounts shown in target currency with original preserved

### Best Practices
1. **Export regularly** from Splitwise to keep data current
2. **Use meaningful group names** with emojis for quick identification
3. **Set up income sources** for accurate savings rate tracking
4. **Tag transactions** for custom categorization beyond Splitwise categories
5. **Add notes** to transactions for context (e.g., "Client dinner", "Birthday gift for Mom")
6. **Use Cross-Group Analysis** to see the big picture across all groups
7. **Check backups** before clearing data - they're there for a reason!
8. **Keep track of exchange rates** if dealing with multiple currencies

### Working with Multiple Groups
- Each group maintains **separate transaction data, notes, and tags**
- **Switch active group** in sidebar to change Overview view
- Use **Cross-Group Analysis** to analyze spending across all groups
- Import data to the **correct group** via Data & Backups page

### Data Safety
- ✅ **5 automatic backups** kept at all times
- ✅ **Create manual backups** before major imports
- ✅ **Restore from backup** if something goes wrong
- ✅ **Export data regularly** as an extra precaution
- ✅ **Database migrations** preserve data when adding new features

### Performance Tips
- Use **pagination controls** to display fewer rows per page for faster loading
- **10 rows per page** is fastest for large datasets
- **Clear cache** in Data & Backups if you notice slowdowns
- **Tag filtering** is faster than full-text search for large datasets

## 🔧 Requirements

**Python Packages** (auto-installed with requirements.txt):
- `streamlit` - Web interface framework
- `pandas` - Data processing and analysis
- `plotly` - Interactive visualizations
- `openpyxl` - Excel file support
- `numpy` - Numerical computations

**System Requirements**:
- Python 3.8 or higher
- ~50MB disk space for installation
- Modern web browser (Chrome, Firefox, Safari, Edge)

## 🤝 Contributing

Found a bug or have a feature request? Feel free to open an issue or submit a pull request!

## 📝 License

This project is open source and available under the MIT License.

---

**Made with ❤️ for Splitwise users who want deeper insights into their spending patterns.**
