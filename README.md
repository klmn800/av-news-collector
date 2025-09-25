# Alpha Vantage News Collector

A simple, standalone Python tool for collecting news and sentiment data from Alpha Vantage's NEWS_SENTIMENT API. Perfect for market research, sentiment analysis, and news aggregation.

## Features

- **Symbol-based news collection** - Get news for specific stocks, crypto, or forex
- **Topic-based filtering** - Collect news by categories (earnings, economy, technology, etc.)
- **Built-in rate limiting** - Respects Alpha Vantage API limits automatically
- **Configurable subscription tiers** - Supports free and premium accounts
- **Persistent caching** - Minimizes API usage with intelligent caching
- **SQLite database storage** - Clean, portable data storage
- **Data export** - Export to CSV or JSON formats
- **Automation-friendly** - No-interaction mode for scripts and CI/CD
- **No external dependencies** - Just Python standard library + requests

## Installation

1. **Download the script:**
   ```bash
   wget https://raw.githubusercontent.com/klmn800/av-news-collector/main/av_news_collector.py
   # or
   curl -O https://raw.githubusercontent.com/klmn800/av-news-collector/main/av_news_collector.py
   ```

2. **Install requirements:**
   ```bash
   pip install requests
   ```

3. **Get your Alpha Vantage API key:**
   - Visit https://www.alphavantage.co/support/#api-key
   - Sign up for a free account
   - Copy your API key

## Quick Start

### Interactive Mode (Recommended)

1. **Launch the interactive interface:**
   ```bash
   python av_news_collector.py
   ```
   This opens a user-friendly menu system that guides you through:
   - First-time API key setup
   - Symbol or topic-based news collection
   - Data export options
   - Settings configuration

2. **Follow the guided workflows:**
   - Choose collection type from the main menu
   - Enter symbols/topics with helpful prompts
   - Set optional parameters (dates default to today)
   - Preview and confirm your collection
   - Choose output format (database, CSV, JSON, or all)

### Command Line Mode (Automation)

1. **First-time setup:**
   ```bash
   python av_news_collector.py --setup
   ```

2. **Direct collection:**
   ```bash
   # Get news for specific symbols
   python av_news_collector.py --symbol AAPL
   python av_news_collector.py --symbols NVDA,TSLA,AMD

   # Get news by topic
   python av_news_collector.py --topic technology
   ```

3. **Export existing data:**
   ```bash
   python av_news_collector.py --export csv
   python av_news_collector.py --export json
   ```

## Usage Examples

### Interactive Mode Examples

**Launch and navigate:**
```bash
# Start interactive interface
python av_news_collector.py

# Follow the menu prompts:
# 1. Collect news by Symbol → Enter: NVDA → Defaults → Confirm → Choose format
# 2. Collect news by Topic → Select: earnings → Set parameters → Export
# 3. Export existing data → Choose CSV/JSON → Set filename
# 4. Settings → Configure API key and subscription tier
```

**Features:**
- Guided workflows with previews and confirmations
- Smart defaults (today's date, 1000 article limit)
- Organized output folders: `news_downloads/database/`, `csv/`, `json/`
- Back/cancel navigation at each step
- Masked API key display for security

### Command Line Examples

**Basic Collection:**
```bash
# Single symbol
python av_news_collector.py --symbol NVDA

# Multiple symbols
python av_news_collector.py --symbols AAPL,MSFT,GOOGL

# Topic-based collection
python av_news_collector.py --topic earnings
```

**Advanced Command Line Options:**
```bash
# With date range
python av_news_collector.py --symbol TSLA --from 2025-09-15 --to 2025-09-19

# Limit results
python av_news_collector.py --topic technology --limit 100

# Custom database location
python av_news_collector.py --symbol AAPL --db /path/to/my_news.db
```

**Automation & Scripting:**
```bash
# No-interaction mode (perfect for scripts)
python av_news_collector.py --symbol AAPL --no-interact

# One-time API key override
python av_news_collector.py --symbol NVDA --api-key YOUR_KEY
```

**Premium Accounts:**
```bash
# Set your subscription tier (one time)
python av_news_collector.py --set-tier premium

# Override tier for one collection
python av_news_collector.py --symbol AAPL --tier premium-aggressive
```

**Data Export:**
```bash
# Export to CSV (includes sentiment data)
python av_news_collector.py --export csv

# Export to JSON with custom filename
python av_news_collector.py --export json --export-file my_analysis.json

# Export articles only (no symbol sentiment)
python av_news_collector.py --export csv --export-no-sentiment
```

## Output Organization

The tool automatically creates an organized folder structure:

```
news_downloads/
├── database/        # SQLite database files
│   └── news_data.db
├── csv/            # CSV export files
│   └── news_export_20250925_143022.csv
└── json/           # JSON export files
    └── news_export_20250925_143022.json
```

**File Naming:**
- Auto-generated: `news_export_YYYYMMDD_HHMMSS.format`
- Custom names supported in both interactive and CLI modes
- Organized by format type for easy management

## Available Topics

**Interactive Mode:** Topics are displayed when you select "Collect news by Topic"

**Command Line:** Run `python av_news_collector.py --list-topics` for full list

- `blockchain` - Blockchain
- `earnings` - Earnings
- `ipo` - IPO
- `financial_markets` - Financial Markets
- `technology` - Technology
- `economy_macro` - Economy - Macro/Overall
- And 9 more...

## Subscription Tiers

Run `python av_news_collector.py --show-tiers` to see rate limits:

- **Free**: 25 calls/day, ~5 calls/minute
- **Premium**: No daily limit, ~60 calls/minute (conservative)
- **Premium Aggressive**: No daily limit, ~300 calls/minute

## Database Schema

The tool creates two main tables:

### news_articles
- `article_url` (Primary Key)
- `title`, `summary`, `source`, `authors`
- `time_published`, `overall_sentiment_score`, `overall_sentiment_label`
- `symbols_mentioned` (JSON array)

### news_symbol_sentiment
- `article_url`, `symbol` (Composite Primary Key)
- `relevance_score`, `symbol_sentiment_score`, `symbol_sentiment_label`

## Command Reference

### Interactive Mode
```bash
python av_news_collector.py          # Launch interactive interface
```

### Information Commands
```bash
--help                 # Show help message
--list-topics         # Show available news topics
--show-tiers          # Show subscription tier options
```

### Setup Commands
```bash
--setup               # Set up and save API key
--set-tier TIER       # Save subscription tier
```

### Collection Commands
```bash
--symbol SYMBOL       # Collect news for single symbol
--symbols SYM1,SYM2   # Collect news for multiple symbols
--topic TOPIC         # Collect news by topic
--limit N             # Maximum articles to collect
--from YYYY-MM-DD     # Start date
--to YYYY-MM-DD       # End date
```

### Export Commands
```bash
--export FORMAT       # Export data (csv or json)
--export-file FILE    # Custom export filename
--export-no-sentiment # Skip symbol sentiment data
```

### Control Options
```bash
--dry-run            # Test without API calls
--no-interact        # Automation mode (no prompts)
--tier TIER          # Override subscription tier
--api-key KEY        # Override stored API key
--db PATH            # Custom database path
--cache-dir PATH     # Custom cache directory
--debug              # Enable debug logging
```

## API Rate Limits

The tool automatically handles Alpha Vantage rate limits:

- **Free accounts**: 25 calls per day, rate-limited to ~5 per minute
- **Premium accounts**: No daily limits, minute-based rate limiting only
- **Smart caching**: Reduces duplicate API calls
- **Usage tracking**: Shows remaining quota

## Error Handling

Common issues and solutions:

- **"API key required"**: Run `--setup` to configure your key
- **"Daily limit exceeded"**: Wait until midnight or upgrade to premium
- **"No data found"**: Try different symbols/topics or check date range
- **Database errors**: Check file permissions and disk space

## Contributing

This is a standalone script designed for simplicity. For feature requests or bug reports, please create an issue on GitHub.

## Disclaimer

**This tool is not affiliated with Alpha Vantage.** It is an independent third-party application that uses Alpha Vantage's public API. Alpha Vantage is a trademark of Alpha Vantage Inc.

## License

MIT License - see LICENSE file for details.

## Author

Created for easy Alpha Vantage news collection and analysis.
