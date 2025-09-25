#!/usr/bin/env python3
"""
Alpha Vantage News Collector - Standalone Version
================================================

A simple, standalone tool for collecting news and sentiment data from Alpha Vantage's
NEWS_SENTIMENT API. No external dependencies except the Python standard library and requests.

Features:
- Symbol-based news collection (stocks, crypto, forex)
- Topic-based news filtering (earnings, IPO, economy, etc.)
- Time range filtering
- Built-in rate limiting (25 calls/day for free tier)
- Persistent caching
- SQLite database storage
- No configuration files required

Usage Examples:
  # Get news for a specific symbol
  python av_news_collector.py --symbol NVDA --api-key YOUR_KEY

  # Get news for multiple symbols
  python av_news_collector.py --symbols NVDA,AAPL,TSLA --api-key YOUR_KEY

  # Get news by topic
  python av_news_collector.py --topic technology --api-key YOUR_KEY

  # With date range (YYYY-MM-DD format)
  python av_news_collector.py --symbol NVDA --from 2025-09-15 --to 2025-09-19 --api-key YOUR_KEY

  # Using environment variable for API key
  export ALPHA_VANTAGE_API_KEY=your_key_here
  python av_news_collector.py --symbol NVDA

  # Dry run to see what would be collected
  python av_news_collector.py --symbol NVDA --dry-run --api-key YOUR_KEY

Alpha Vantage API Details:
- Function: NEWS_SENTIMENT
- Rate Limit: 25 calls per day (free tier)
- Covers: Stocks, cryptocurrencies, forex
- Sources: Premier news outlets worldwide
- Limit: Up to 1000 results per call

Requirements:
- Python 3.6+
- requests library: pip install requests

License: MIT

Author: Ben Keilman
"""

import os
import sys
import json
import time
import sqlite3
import requests
import argparse
import logging
import csv
from datetime import datetime, timedelta
from pathlib import Path


# Alpha Vantage API constants
API_BASE_URL = "https://www.alphavantage.co/query"

# Alpha Vantage subscription tier rate limits
# Note: Alpha Vantage doesn't publish specific minute limits for premium tiers
# These are conservative estimates to avoid hitting undocumented limits
AV_RATE_LIMITS = {
    'free': {
        'requests_per_minute': 5,
        'requests_per_day': 25,
        'description': 'Free (25 calls/day, ~5 calls/minute)'
    },
    'premium-75': {
        'requests_per_minute': 75,
        'requests_per_day': None,
        'description': 'Premium 75 ($49.99/month - 75 calls/minute)'
    },
    'premium-150': {
        'requests_per_minute': 150,
        'requests_per_day': None,
        'description': 'Premium 150 ($99.99/month - 150 calls/minute)'
    },
    'premium-300': {
        'requests_per_minute': 300,
        'requests_per_day': None,
        'description': 'Premium 300 ($149.99/month - 300 calls/minute)'
    },
    'premium-600': {
        'requests_per_minute': 600,
        'requests_per_day': None,
        'description': 'Premium 600 ($199.99/month - 600 calls/minute)'
    },
    'premium-1200': {
        'requests_per_minute': 1200,
        'requests_per_day': None,
        'description': 'Premium 1200 ($249.99/month - 1200 calls/minute)'
    },
    'custom': {
        'requests_per_minute': 60,  # Default, user configurable
        'requests_per_day': None,
        'description': 'Custom rate limit (user configured)'
    }
}

# Valid Alpha Vantage news topics
ALPHA_VANTAGE_TOPICS = {
    'blockchain': 'Blockchain',
    'earnings': 'Earnings',
    'ipo': 'IPO',
    'mergers_and_acquisitions': 'Mergers & Acquisitions',
    'financial_markets': 'Financial Markets',
    'economy_fiscal': 'Economy - Fiscal Policy (e.g., tax reform, government spending)',
    'economy_monetary': 'Economy - Monetary Policy (e.g., interest rates, inflation)',
    'economy_macro': 'Economy - Macro/Overall',
    'energy_transportation': 'Energy & Transportation',
    'finance': 'Finance',
    'life_sciences': 'Life Sciences',
    'manufacturing': 'Manufacturing',
    'real_estate': 'Real Estate & Construction',
    'retail_wholesale': 'Retail & Wholesale',
    'technology': 'Technology'
}



class AlphaVantageRateLimiter:
    """Manages Alpha Vantage API request timing to stay within rate limits"""

    def __init__(self, subscription_tier='free', cache_dir="./cache"):
        # Get rate limits for subscription tier
        if subscription_tier not in AV_RATE_LIMITS:
            raise ValueError("Invalid subscription tier. Must be one of: {}".format(list(AV_RATE_LIMITS.keys())))

        self.tier_config = AV_RATE_LIMITS[subscription_tier]
        self.subscription_tier = subscription_tier
        self.requests_per_minute = self.tier_config['requests_per_minute']
        self.requests_per_day = self.tier_config['requests_per_day']  # Can be None for premium tiers

        # Calculate interval between requests (in seconds)
        self.interval = 60 / self.requests_per_minute
        self.last_request_time = 0

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.usage_file = self.cache_dir / "alphavantage_daily_usage.json"

        # Load persisted daily counter (only if daily limit applies)
        self.daily_request_count = 0
        self.daily_reset_time = None
        if self.requests_per_day is not None:
            self.load_daily_counter()
            self.reset_daily_counter()

    def load_daily_counter(self):
        """Load daily counter from persistent storage"""
        try:
            if self.usage_file.exists():
                with open(self.usage_file, 'r') as f:
                    data = json.load(f)
                    self.daily_request_count = data.get('daily_request_count', 0)
                    reset_time_str = data.get('daily_reset_time', '2000-01-01T00:00:00')
                    self.daily_reset_time = datetime.fromisoformat(reset_time_str)
        except Exception as e:
            logging.debug("Could not load daily counter: {}".format(e))
            self.daily_request_count = 0
            self.daily_reset_time = None

    def save_daily_counter(self):
        """Save daily counter to persistent storage"""
        try:
            data = {
                'daily_request_count': self.daily_request_count,
                'daily_reset_time': self.daily_reset_time.isoformat() if self.daily_reset_time else None,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.usage_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logging.debug("Could not save daily counter: {}".format(e))

    def reset_daily_counter(self):
        """Reset daily counter at midnight"""
        current_time = datetime.now()
        if self.daily_reset_time is None or current_time.date() > self.daily_reset_time.date():
            self.daily_request_count = 0
            self.daily_reset_time = current_time
            self.save_daily_counter()
            print("[INFO] Alpha Vantage daily request counter reset")

    def can_make_request(self):
        """Check if we can make a request without exceeding limits"""
        if self.requests_per_day is not None:
            self.reset_daily_counter()
            return self.daily_request_count < self.requests_per_day
        return True  # No daily limit for premium tiers

    def wait_if_needed(self, quiet=False):
        """Wait if necessary to comply with rate limits"""
        # Check daily limit (only for free tier)
        if self.requests_per_day is not None:
            self.reset_daily_counter()
            if not self.can_make_request():
                raise Exception("Daily API request limit ({}) exceeded".format(self.requests_per_day))

        # Rate limiting for per-minute limits
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.interval:
            wait_time = self.interval - time_since_last
            if not quiet:
                print("Rate limiting: waiting {:.1f} seconds...".format(wait_time))
            time.sleep(wait_time)

        self.last_request_time = time.time()

        # Only increment daily counter if daily limits apply
        if self.requests_per_day is not None:
            self.daily_request_count += 1
            self.save_daily_counter()

    def get_usage_stats(self):
        """Get current usage statistics"""
        if self.requests_per_day is not None:
            self.reset_daily_counter()
            return {
                'subscription_tier': self.subscription_tier,
                'tier_description': self.tier_config['description'],
                'daily_requests_used': self.daily_request_count,
                'daily_requests_remaining': self.requests_per_day - self.daily_request_count,
                'has_daily_limit': True,
                'requests_per_minute': self.requests_per_minute
            }
        else:
            return {
                'subscription_tier': self.subscription_tier,
                'tier_description': self.tier_config['description'],
                'daily_requests_used': None,
                'daily_requests_remaining': None,
                'has_daily_limit': False,
                'requests_per_minute': self.requests_per_minute
            }


class AlphaVantageNewsClient:
    """Simplified Alpha Vantage NEWS_SENTIMENT API client"""

    def __init__(self, api_key, subscription_tier='free', cache_dir="./cache", quiet=False):
        self.api_key = api_key
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limiter = AlphaVantageRateLimiter(subscription_tier=subscription_tier, cache_dir=cache_dir)
        self.quiet = quiet

    def _make_request(self, params, cache_key=None, max_age_seconds=300):
        """Make API request with caching and rate limiting"""
        # Check cache first
        if cache_key and max_age_seconds > 0:
            cache_file = self.cache_dir / "{}.json".format(cache_key)
            if cache_file.exists():
                file_age = time.time() - cache_file.stat().st_mtime
                if file_age < max_age_seconds:
                    try:
                        with open(cache_file, 'r') as f:
                            cached_data = json.load(f)
                            if not self.quiet:
                                print("Using cached data (age: {:.1f}s)".format(file_age))
                            return cached_data
                    except Exception as e:
                        logging.debug("Cache read failed: {}".format(e))

        # Make API request with rate limiting
        self.rate_limiter.wait_if_needed(quiet=self.quiet)

        try:
            response = requests.get(API_BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            # Check for API errors
            if 'Error Message' in data:
                raise Exception("API Error: {}".format(data['Error Message']))
            if 'Note' in data:
                raise Exception("API Note: {}".format(data['Note']))

            # Cache successful response
            if cache_key and max_age_seconds > 0:
                try:
                    cache_file = self.cache_dir / "{}.json".format(cache_key)
                    with open(cache_file, 'w') as f:
                        json.dump(data, f, indent=2)
                except Exception as e:
                    logging.debug("Cache write failed: {}".format(e))

            return data

        except requests.exceptions.RequestException as e:
            raise Exception("Request failed: {}".format(e))
        except json.JSONDecodeError as e:
            raise Exception("Invalid JSON response: {}".format(e))

    def get_symbol_news(self, symbol, limit=1000, time_from=None, time_to=None, max_age_seconds=300):
        """Get news for a specific symbol"""
        params = {
            'function': 'NEWS_SENTIMENT',
            'tickers': symbol,
            'limit': min(limit, 1000),
            'apikey': self.api_key
        }

        if time_from:
            params['time_from'] = time_from
        if time_to:
            params['time_to'] = time_to

        cache_key = sanitize_cache_key("news_symbol_{}_{}_{}".format(
            symbol,
            time_from or "default",
            time_to or "default"
        ))

        return self._make_request(params, cache_key, max_age_seconds)

    def get_topic_news(self, topic, limit=1000, time_from=None, time_to=None, max_age_seconds=300):
        """Get news for specific topics"""
        params = {
            'function': 'NEWS_SENTIMENT',
            'topics': topic,
            'limit': min(limit, 1000),
            'apikey': self.api_key
        }

        if time_from:
            params['time_from'] = time_from
        if time_to:
            params['time_to'] = time_to

        cache_key = sanitize_cache_key("news_topic_{}_{}_{}".format(
            topic.replace(',', '_'),
            time_from or "default",
            time_to or "default"
        ))

        return self._make_request(params, cache_key, max_age_seconds)

    def get_usage_stats(self):
        """Get current API usage statistics"""
        return self.rate_limiter.get_usage_stats()


class NewsCollector:
    """Main news collector class"""

    def __init__(self, api_key, db_path="news_data.db", cache_dir="./cache", subscription_tier='free', quiet=False):
        self.news_client = AlphaVantageNewsClient(api_key, subscription_tier, cache_dir, quiet)
        self.db_path = db_path
        self.quiet = quiet
        self.init_database()

    @staticmethod
    def get_config_dir():
        """Get user's configuration directory"""
        home = Path.home()
        config_dir = home / ".av_news_collector"
        config_dir.mkdir(exist_ok=True)
        return config_dir

    @staticmethod
    def save_api_key(api_key):
        """Save API key to user's config directory"""
        config_dir = NewsCollector.get_config_dir()
        config_file = config_dir / "config.json"

        config = {}
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
            except json.JSONDecodeError:
                print("Warning: Config file {} contains invalid JSON - creating new config".format(config_file))
                config = {}
            except Exception:
                pass

        config['api_key'] = api_key

        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        print("API key saved to: {}".format(config_file))

    @staticmethod
    def save_subscription_tier(tier, custom_rate_per_minute=None):
        """Save subscription tier to user's config directory"""
        config_dir = NewsCollector.get_config_dir()
        config_file = config_dir / "config.json"

        config = {}
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
            except json.JSONDecodeError:
                print("Warning: Config file {} contains invalid JSON - creating new config".format(config_file))
                config = {}
            except Exception:
                pass

        config['subscription_tier'] = tier
        if tier == 'custom' and custom_rate_per_minute is not None:
            config['custom_rate_per_minute'] = custom_rate_per_minute
            AV_RATE_LIMITS['custom']['requests_per_minute'] = custom_rate_per_minute
            AV_RATE_LIMITS['custom']['description'] = "Custom ({} calls/minute)".format(custom_rate_per_minute)

        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        print("Subscription tier saved: {}".format(AV_RATE_LIMITS[tier]['description']))

    @staticmethod
    def load_api_key():
        """Load API key from user's config directory"""
        config_dir = NewsCollector.get_config_dir()
        config_file = config_dir / "config.json"

        if not config_file.exists():
            return None

        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                return config.get('api_key')
        except json.JSONDecodeError:
            print("Warning: Config file {} contains invalid JSON - using defaults".format(config_file))
            return None
        except Exception:
            return None

    @staticmethod
    def load_subscription_tier():
        """Load subscription tier from user's config directory"""
        config_dir = NewsCollector.get_config_dir()
        config_file = config_dir / "config.json"

        if not config_file.exists():
            return 'free'

        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                tier = config.get('subscription_tier', 'free')

                # Handle custom rate limits
                if tier == 'custom' and 'custom_rate_per_minute' in config:
                    custom_rate = config['custom_rate_per_minute']
                    AV_RATE_LIMITS['custom']['requests_per_minute'] = custom_rate
                    AV_RATE_LIMITS['custom']['description'] = "Custom ({} calls/minute)".format(custom_rate)

                # Validate tier
                if tier in AV_RATE_LIMITS:
                    return tier
                return 'free'
        except json.JSONDecodeError:
            print("Warning: Config file {} contains invalid JSON - using free tier".format(config_file))
            return 'free'
        except Exception:
            return 'free'

    def init_database(self):
        """Initialize database tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Create news_articles table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS news_articles (
                        article_url TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        summary TEXT,
                        source TEXT,
                        authors TEXT,  -- JSON array
                        time_published TEXT,
                        overall_sentiment_score REAL,
                        overall_sentiment_label TEXT,
                        symbols_mentioned TEXT,  -- JSON array
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Create news_symbol_sentiment table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS news_symbol_sentiment (
                        article_url TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        relevance_score REAL,
                        symbol_sentiment_score REAL,
                        symbol_sentiment_label TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (article_url, symbol),
                        FOREIGN KEY (article_url) REFERENCES news_articles (article_url)
                    )
                """)

                # Create index for efficient queries
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_symbol_sentiment
                    ON news_symbol_sentiment (symbol)
                """)

                conn.commit()
                logging.info("Database tables initialized successfully")

        except Exception as e:
            logging.error("Database initialization failed: {}".format(e))
            raise

    def collect_symbol_news(self, symbol, limit=1000, time_from=None, time_to=None, dry_run=False):
        """Collect news for a single symbol"""
        print("Collecting news for symbol: {}".format(symbol))
        print("Limit: {}, Time range: {} to {}".format(limit, time_from or "default", time_to or "default"))

        if dry_run:
            print("DRY RUN - No actual API calls or database writes")
            return True

        # Get news data
        news_data = self.news_client.get_symbol_news(
            symbol=symbol,
            limit=limit,
            time_from=time_from,
            time_to=time_to
        )

        if not news_data:
            print("No news data retrieved for {}".format(symbol))
            return False

        # Debug: Show what we got from Alpha Vantage
        if not self.quiet:
            print("Debug: Alpha Vantage response keys: {}".format(list(news_data.keys())))
            print("Debug: Response size: {} bytes".format(len(str(news_data))))
            if 'articles' in news_data:
                print("Debug: Articles array length: {}".format(len(news_data['articles'])))
            else:
                print("Debug: No 'articles' key in response")

        # Alpha Vantage uses 'feed' not 'articles' for news data
        articles = news_data.get('articles', []) or news_data.get('feed', [])
        if not articles:
            print("No articles found for {}".format(symbol))
            print("Debug: Full API response: {}".format(json.dumps(news_data, indent=2)[:500]))
            return False

        print("Found {} articles for {}".format(len(articles), symbol))

        # Store in database
        stored_count = self.store_articles(articles)
        print("Stored {} articles for {} in database".format(stored_count, symbol))

        return stored_count > 0

    def collect_topic_news(self, topic, limit=1000, time_from=None, time_to=None, dry_run=False):
        """Collect news for specific topics"""
        print("Collecting news for topic: {}".format(topic))
        print("Limit: {}, Time range: {} to {}".format(limit, time_from or "default", time_to or "default"))

        if dry_run:
            print("DRY RUN - No actual API calls or database writes")
            return True

        # Get news data
        news_data = self.news_client.get_topic_news(
            topic=topic,
            limit=limit,
            time_from=time_from,
            time_to=time_to
        )

        if not news_data:
            print("No news data retrieved for topic {}".format(topic))
            return False

        # Alpha Vantage uses 'feed' not 'articles' for news data
        articles = news_data.get('articles', []) or news_data.get('feed', [])
        if not articles:
            print("No articles found for topic {}".format(topic))
            return False

        print("Found {} articles for topic {}".format(len(articles), topic))

        # Store in database
        stored_count = self.store_articles(articles)
        print("Stored {} articles for topic '{}' in database".format(stored_count, topic))

        # Show symbol coverage
        self.show_symbol_coverage(articles)

        return stored_count > 0

    def store_articles(self, articles):
        """Store articles and sentiment data in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                articles_stored = 0
                symbols_stored = 0

                for article in articles:
                    try:
                        # Extract symbols mentioned
                        symbols_mentioned = []
                        ticker_sentiments = article.get('ticker_sentiment', [])

                        for ticker_sentiment in ticker_sentiments:
                            symbols_mentioned.append(ticker_sentiment['ticker'])

                        # Insert article (ignore duplicates)
                        try:
                            cursor.execute("""
                                INSERT OR IGNORE INTO news_articles (
                                    article_url, title, summary, source, authors,
                                    time_published, overall_sentiment_score,
                                    overall_sentiment_label, symbols_mentioned
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                article['url'],
                                article['title'],
                                article['summary'],
                                article['source'],
                                json.dumps(article['authors']),
                                article['time_published'],
                                article['overall_sentiment_score'],
                                article['overall_sentiment_label'],
                                json.dumps(symbols_mentioned)
                            ))
                        except Exception as insert_error:
                            logging.debug("Article insert failed: {}".format(insert_error))
                            continue

                        if cursor.rowcount > 0:
                            articles_stored += 1

                            # Store symbol sentiment data
                            for ticker_sentiment in ticker_sentiments:
                                symbol = ticker_sentiment['ticker']
                                try:
                                    cursor.execute("""
                                        INSERT OR REPLACE INTO news_symbol_sentiment (
                                            article_url, symbol, relevance_score,
                                            symbol_sentiment_score, symbol_sentiment_label
                                        ) VALUES (?, ?, ?, ?, ?)
                                    """, (
                                        article['url'],
                                        symbol,
                                        float(ticker_sentiment.get('relevance_score', 0.0)),
                                        float(ticker_sentiment.get('ticker_sentiment_score', 0.0)),
                                        ticker_sentiment.get('ticker_sentiment_label', 'Neutral')
                                    ))
                                    symbols_stored += 1
                                except Exception as sentiment_error:
                                    logging.debug("Symbol sentiment insert failed for {}: {}".format(symbol, sentiment_error))

                    except Exception as e:
                        logging.debug("Error storing article {}: {}".format(
                            article.get('url', 'unknown'), e))
                        continue

                conn.commit()
                print("Database summary: {} articles, {} symbol sentiment records".format(
                    articles_stored, symbols_stored))
                return articles_stored

        except Exception as e:
            logging.error("Database storage error: {}".format(e))
            return 0

    def show_symbol_coverage(self, articles):
        """Show symbol coverage statistics"""
        symbols_found = set()

        for article in articles:
            for ticker_sentiment in article.get('ticker_sentiment', []):
                symbols_found.add(ticker_sentiment['ticker'])

        print()
        print("Symbol Coverage: {} unique symbols mentioned".format(len(symbols_found)))
        if len(symbols_found) <= 20:  # Show symbols if not too many
            print("Symbols: {}".format(", ".join(sorted(symbols_found))))

    def export_data(self, format='csv', output_file=None, include_sentiment=True):
        """Export collected news data to CSV or JSON"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = "news_export_{}.{}".format(timestamp, format)

        try:
            with sqlite3.connect(self.db_path) as conn:
                if format.lower() == 'csv':
                    self._export_csv(conn, output_file, include_sentiment)
                elif format.lower() == 'json':
                    self._export_json(conn, output_file, include_sentiment)
                else:
                    raise ValueError("Format must be 'csv' or 'json'")

                if not self.quiet:
                    print("Data exported to: {}".format(output_file))
                return output_file

        except Exception as e:
            print("Export failed: {}".format(e))
            return None

    def _export_csv(self, conn, output_file, include_sentiment):
        """Export to CSV format"""
        cursor = conn.cursor()

        # Main articles query
        if include_sentiment:
            query = """
                SELECT
                    a.article_url,
                    a.title,
                    a.summary,
                    a.source,
                    a.time_published,
                    a.overall_sentiment_score,
                    a.overall_sentiment_label,
                    a.symbols_mentioned,
                    s.symbol,
                    s.relevance_score,
                    s.symbol_sentiment_score,
                    s.symbol_sentiment_label
                FROM news_articles a
                LEFT JOIN news_symbol_sentiment s ON a.article_url = s.article_url
                ORDER BY a.time_published DESC
            """
        else:
            query = """
                SELECT
                    article_url,
                    title,
                    summary,
                    source,
                    time_published,
                    overall_sentiment_score,
                    overall_sentiment_label,
                    symbols_mentioned
                FROM news_articles
                ORDER BY time_published DESC
            """

        cursor.execute(query)
        results = cursor.fetchall()

        if not results:
            raise Exception("No data found to export")

        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            if include_sentiment:
                fieldnames = [
                    'article_url', 'title', 'summary', 'source', 'time_published',
                    'overall_sentiment_score', 'overall_sentiment_label', 'symbols_mentioned',
                    'symbol', 'relevance_score', 'symbol_sentiment_score', 'symbol_sentiment_label'
                ]
            else:
                fieldnames = [
                    'article_url', 'title', 'summary', 'source', 'time_published',
                    'overall_sentiment_score', 'overall_sentiment_label', 'symbols_mentioned'
                ]

            writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
            writer.writerow(fieldnames)
            writer.writerows(results)

    def _export_json(self, conn, output_file, include_sentiment):
        """Export to JSON format"""
        cursor = conn.cursor()

        # Get articles
        cursor.execute("""
            SELECT article_url, title, summary, source, authors, time_published,
                   overall_sentiment_score, overall_sentiment_label, symbols_mentioned
            FROM news_articles ORDER BY time_published DESC
        """)

        articles = []
        for row in cursor.fetchall():
            article = {
                'article_url': row[0],
                'title': row[1],
                'summary': row[2],
                'source': row[3],
                'authors': json.loads(row[4]) if row[4] else [],
                'time_published': row[5],
                'overall_sentiment_score': row[6],
                'overall_sentiment_label': row[7],
                'symbols_mentioned': json.loads(row[8]) if row[8] else [],
                'symbol_sentiments': []
            }

            if include_sentiment:
                # Get sentiment data for this article
                cursor.execute("""
                    SELECT symbol, relevance_score, symbol_sentiment_score, symbol_sentiment_label
                    FROM news_symbol_sentiment WHERE article_url = ?
                """, (row[0],))

                for sent_row in cursor.fetchall():
                    article['symbol_sentiments'].append({
                        'symbol': sent_row[0],
                        'relevance_score': sent_row[1],
                        'sentiment_score': sent_row[2],
                        'sentiment_label': sent_row[3]
                    })

            articles.append(article)

        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'total_articles': len(articles),
            'articles': articles
        }

        with open(output_file, 'w', encoding='utf-8') as jsonfile:
            json.dump(export_data, jsonfile, indent=2, ensure_ascii=False)

    def show_usage_stats(self):
        """Show current API usage"""
        if self.quiet:
            return

        stats = self.news_client.get_usage_stats()
        print()
        print("API Usage Status:")
        print("  Subscription: {}".format(stats['tier_description']))
        print("  Rate limit: {} calls/minute".format(stats['requests_per_minute']))

        if stats['has_daily_limit']:
            print("  Calls used today: {}".format(stats['daily_requests_used']))
            print("  Calls remaining: {} (resets at midnight)".format(stats['daily_requests_remaining']))

            # Show usage warning if getting close to limit
            if stats['daily_requests_remaining'] <= 5:
                print("  WARNING: Only {} calls remaining today!".format(stats['daily_requests_remaining']))
            elif stats['daily_requests_remaining'] <= 10:
                print("  NOTE: {} calls remaining today".format(stats['daily_requests_remaining']))
        else:
            print("  Daily limit: No limit (premium subscription)")


def show_welcome_screen(no_interact=False):
    """Display welcome screen with program info"""
    print("=" * 65)
    print(" " * 18 + "Alpha Vantage News Collector")
    print("=" * 65)
    print()
    print("  Collect news and sentiment data from premium outlets")
    print("  * Symbol-based news (stocks, crypto, forex)")
    print("  * Topic-based filtering (earnings, economy, tech, etc.)")
    print("  * Built-in rate limiting & caching")
    print("  * SQLite database storage")
    print()
    print("  Free Alpha Vantage API: 25 calls per day")
    print("  Get your key: https://www.alphavantage.co/support/#api-key")
    print()
    print("=" * 65)
    print()

    if not no_interact:
        try:
            input("Press Enter to begin...")
            print()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            sys.exit(0)


def setup_api_key():
    """Interactive setup for API key storage"""
    print("Alpha Vantage News Collector - First Time Setup")
    print("=" * 50)
    print()
    print("To use this tool, you need an Alpha Vantage API key.")
    print("You can get a free API key at: https://www.alphavantage.co/support/#api-key")
    print()
    print("Your API key will be stored securely in: {}".format(NewsCollector.get_config_dir() / "config.json"))
    print()

    while True:
        api_key = input("Enter your Alpha Vantage API key: ").strip()
        if api_key:
            break
        print("API key cannot be empty. Please try again.")

    NewsCollector.save_api_key(api_key)
    print()
    print("Setup complete! You can now use the news collector without specifying --api-key")
    print()
    print("Try it out:")
    print("  python av_news_collector.py --symbol AAPL --dry-run")
    print("  python av_news_collector.py --topic technology --limit 50")


def show_subscription_tiers():
    """Display available subscription tiers and their rate limits"""
    print("Alpha Vantage Subscription Tiers:")
    print("=" * 50)
    print()

    for tier, config in AV_RATE_LIMITS.items():
        print("  {}: {}".format(tier.upper(), config['description']))

    print()
    print("Usage:")
    print("  python av_news_collector.py --set-tier premium")
    print("  python av_news_collector.py --tier premium --symbol AAPL  # One-time override")


def list_topics():
    """Display all available topics with descriptions"""
    print("Available Alpha Vantage News Topics:")
    print("=" * 40)
    print()

    # Calculate max width for alignment
    max_key_length = max(len(key) for key in ALPHA_VANTAGE_TOPICS.keys())

    for key, description in sorted(ALPHA_VANTAGE_TOPICS.items()):
        print("  {:<{}} - {}".format(key, max_key_length, description))

    print()
    print("Usage examples:")
    print("  python av_news_collector.py --topic technology")
    print("  python av_news_collector.py --topic earnings --limit 100")
    print("  python av_news_collector.py --topic financial_markets --from 2025-09-15")


def validate_date_format(date_string):
    """Validate and convert date string from YYYY-MM-DD to YYYYMMDDT0000/T2359"""
    if not date_string:
        return None

    try:
        # Validate the date format
        datetime.strptime(date_string, "%Y-%m-%d")
        return date_string.replace("-", "")
    except ValueError:
        raise ValueError("Invalid date format '{}'. Expected YYYY-MM-DD (e.g., 2025-09-15)".format(date_string))


def sanitize_cache_key(cache_key):
    """Sanitize cache key to be safe for filenames on all platforms"""
    # Replace problematic characters with underscores
    problematic_chars = [':', '/', '\\', '|', '?', '*', '<', '>', '"']
    for char in problematic_chars:
        cache_key = cache_key.replace(char, '_')
    return cache_key


def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')


def get_user_input(prompt, valid_options=None, allow_empty=False):
    """Get user input with validation"""
    while True:
        try:
            user_input = input(prompt).strip()

            if not user_input and not allow_empty:
                print("Please enter a value.")
                continue

            if valid_options and user_input not in valid_options:
                print("Invalid option. Please choose from: {}".format(', '.join(valid_options)))
                continue

            return user_input

        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            sys.exit(0)


def show_header():
    """Display application header"""
    print("=" * 65)
    print(" " * 18 + "Alpha Vantage News Collector")
    print("=" * 65)
    print()
    print("  Interactive News Collection Tool")
    print("  * Symbol-based news (stocks, crypto, forex)")
    print("  * Topic-based filtering (earnings, economy, tech, etc.)")
    print("  * Built-in rate limiting & caching")
    print("  * Multiple export formats")
    print()
    print("  ** Not affiliated with Alpha Vantage **")
    print()
    print("=" * 65)


def show_main_menu():
    """Display main menu options"""
    print("\n" + "=" * 30 + " MAIN MENU " + "=" * 30)
    print()
    print("  1. Collect news by Symbol")
    print("  2. Collect news by Topic")
    print("  3. Export existing data")
    print("  4. Settings")
    print("  5. Exit")
    print()
    print("=" * 71)
    return get_user_input("Select option (1-5): ", ['1', '2', '3', '4', '5'])


def ensure_downloads_folder():
    """Create organized download folder structure"""
    base_dir = Path("news_downloads")
    folders = ['database', 'csv', 'json']

    for folder in folders:
        (base_dir / folder).mkdir(parents=True, exist_ok=True)

    return base_dir


def interactive_symbol_collection():
    """Interactive workflow for symbol-based collection"""
    print("\n" + "=" * 20 + " SYMBOL COLLECTION " + "=" * 20)
    print()

    # Get symbol(s)
    symbols_input = get_user_input("Enter symbol(s) (comma-separated, or 'back' to return): ").upper()
    if symbols_input.lower() == 'back':
        return
    symbols = [s.strip() for s in symbols_input.split(',')]

    # Get optional parameters
    print("\nOptional parameters (press Enter for defaults):")
    limit = get_user_input("Limit (max articles, default 1000): ", allow_empty=True)
    limit = int(limit) if limit else 1000

    today_str = datetime.now().strftime("%Y-%m-%d")
    date_from = get_user_input("Start date (YYYY-MM-DD, default today {}): ".format(today_str), allow_empty=True)
    date_from = date_from or today_str
    date_to = get_user_input("End date (YYYY-MM-DD, default today {}): ".format(today_str), allow_empty=True)
    date_to = date_to or today_str

    # Show preview
    print("\n" + "=" * 50)
    print("COLLECTION PREVIEW:")
    print("  Symbols: {}".format(', '.join(symbols)))
    print("  Limit: {} articles per symbol".format(limit))
    if date_from or date_to:
        print("  Date range: {} to {}".format(date_from or 'earliest', date_to or 'latest'))
    print("=" * 50)

    # Get confirmation
    if get_user_input("\nProceed with collection? (y/n): ", ['y', 'n', 'Y', 'N']).lower() != 'y':
        return

    # Get output format
    print("\nOutput format:")
    print("  1. Database only")
    print("  2. CSV export")
    print("  3. JSON export")
    print("  4. All formats")
    format_choice = get_user_input("Choose format (1-4): ", ['1', '2', '3', '4'])

    # Execute collection
    return execute_collection(symbols, None, limit, date_from, date_to, format_choice)


def interactive_topic_collection():
    """Interactive workflow for topic-based collection"""
    print("\n" + "=" * 20 + " TOPIC COLLECTION " + "=" * 20)
    print()

    # Show available topics
    print("Available topics:")
    topics_list = list(ALPHA_VANTAGE_TOPICS.keys())
    for i, topic in enumerate(topics_list, 1):
        print("  {:<2}. {}".format(i, topic))

    print("\nEnter topic number or type topic name (or 'back' to return):")
    topic_input = get_user_input("Topic: ")

    if topic_input.lower() == 'back':
        return

    # Parse topic input
    if topic_input.isdigit():
        topic_num = int(topic_input)
        if 1 <= topic_num <= len(topics_list):
            topic = topics_list[topic_num - 1]
        else:
            print("Invalid topic number")
            return
    else:
        if topic_input in ALPHA_VANTAGE_TOPICS:
            topic = topic_input
        else:
            print("Invalid topic name")
            return

    # Get optional parameters
    print("\nOptional parameters (press Enter for defaults):")
    limit = get_user_input("Limit (max articles, default 1000): ", allow_empty=True)
    limit = int(limit) if limit else 1000

    today_str = datetime.now().strftime("%Y-%m-%d")
    date_from = get_user_input("Start date (YYYY-MM-DD, default today {}): ".format(today_str), allow_empty=True)
    date_from = date_from or today_str
    date_to = get_user_input("End date (YYYY-MM-DD, default today {}): ".format(today_str), allow_empty=True)
    date_to = date_to or today_str

    # Show preview
    print("\n" + "=" * 50)
    print("COLLECTION PREVIEW:")
    print("  Topic: {} - {}".format(topic, ALPHA_VANTAGE_TOPICS[topic]))
    print("  Limit: {} articles".format(limit))
    if date_from or date_to:
        print("  Date range: {} to {}".format(date_from or 'earliest', date_to or 'latest'))
    print("=" * 50)

    # Get confirmation
    if get_user_input("\nProceed with collection? (y/n): ", ['y', 'n', 'Y', 'N']).lower() != 'y':
        return

    # Get output format
    print("\nOutput format:")
    print("  1. Database only")
    print("  2. CSV export")
    print("  3. JSON export")
    print("  4. All formats")
    format_choice = get_user_input("Choose format (1-4): ", ['1', '2', '3', '4'])

    # Execute collection
    return execute_collection(None, topic, limit, date_from, date_to, format_choice)


def execute_collection(symbols, topic, limit, date_from, date_to, format_choice):
    """Execute the news collection with the specified parameters"""
    try:
        # Get API key and subscription tier
        api_key = NewsCollector.load_api_key()
        if not api_key:
            print("\nError: No API key found. Please configure in Settings.")
            input("Press Enter to continue...")
            return False

        subscription_tier = NewsCollector.load_subscription_tier()
        downloads_dir = ensure_downloads_folder()

        # Initialize collector
        db_path = downloads_dir / "database" / "news_data.db"
        collector = NewsCollector(str(api_key), str(db_path), "./cache", subscription_tier, quiet=False)

        # Convert and validate date format if provided
        try:
            time_from = None
            time_to = None
            if date_from:
                validated_from = validate_date_format(date_from)
                time_from = validated_from + "T0000" if validated_from else None
            if date_to:
                validated_to = validate_date_format(date_to)
                time_to = validated_to + "T2359" if validated_to else None
        except ValueError as e:
            print("\\nError: {}".format(e))
            input("Press Enter to continue...")
            return False

        # Show initial usage stats
        collector.show_usage_stats()
        print("\n" + "="*50)
        print("Starting collection...")
        print("="*50)

        success = False

        # Execute collection
        if symbols:
            if len(symbols) == 1:
                success = collector.collect_symbol_news(symbols[0], limit, time_from, time_to)
            else:
                success_count = 0
                for symbol in symbols:
                    print("\nProcessing {}...".format(symbol))
                    if collector.collect_symbol_news(symbol, limit, time_from, time_to):
                        success_count += 1
                success = success_count > 0
                print("\nSuccessfully collected news for {}/{} symbols".format(success_count, len(symbols)))

        elif topic:
            success = collector.collect_topic_news(topic, limit, time_from, time_to)

        if success:
            print("\n" + "="*50)
            print("Collection completed successfully!")
            print("Database: {}".format(db_path))

            # Handle export based on format choice
            if format_choice in ['2', '3', '4']:  # CSV, JSON, or All
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                collection_name = symbols[0] if symbols and len(symbols) == 1 else topic if topic else "multi_symbol"

                if format_choice in ['2', '4']:  # CSV
                    csv_path = downloads_dir / "csv" / "{}_{}.csv".format(collection_name, timestamp)
                    collector.export_data('csv', str(csv_path), True)

                if format_choice in ['3', '4']:  # JSON
                    json_path = downloads_dir / "json" / "{}_{}.json".format(collection_name, timestamp)
                    collector.export_data('json', str(json_path), True)

            collector.show_usage_stats()
            print("="*50)

        else:
            print("\nCollection failed or returned no data")

        input("\nPress Enter to continue...")
        return success

    except Exception as e:
        print("\nError during collection: {}".format(e))
        input("Press Enter to continue...")
        return False


def show_topics_list():
    """Display available topics in interactive format"""
    print("\n" + "=" * 20 + " AVAILABLE TOPICS " + "=" * 20)
    print()

    # Calculate max width for alignment
    max_key_length = max(len(key) for key in ALPHA_VANTAGE_TOPICS.keys())

    for i, (key, description) in enumerate(sorted(ALPHA_VANTAGE_TOPICS.items()), 1):
        print("  {:<2}. {:<{}} - {}".format(i, key, max_key_length, description))

    print("\n" + "=" * 57)
    input("Press Enter to continue...")


def interactive_export():
    """Interactive data export menu"""
    print("\n" + "=" * 25 + " EXPORT DATA " + "=" * 25)
    print()

    # Check if database exists
    db_path = Path("news_downloads/database/news_data.db")
    if not db_path.exists():
        print("No database found. Please collect some news first.")
        input("Press Enter to continue...")
        return

    print("Export format:")
    print("  1. CSV (spreadsheet format)")
    print("  2. JSON (structured data)")
    format_choice = get_user_input("Choose format (1-2): ", ['1', '2'])

    print("\nInclude sentiment data?")
    print("  1. Yes (full export)")
    print("  2. No (articles only)")
    sentiment_choice = get_user_input("Choice (1-2): ", ['1', '2'])

    # Get custom filename
    default_name = "news_export_{}".format(datetime.now().strftime("%Y%m%d_%H%M%S"))
    custom_name = get_user_input("Filename (press Enter for '{}'): ".format(default_name), allow_empty=True)
    filename = custom_name if custom_name else default_name

    try:
        collector = NewsCollector("dummy", str(db_path), "./cache", 'free', True)
        downloads_dir = ensure_downloads_folder()

        format_ext = 'csv' if format_choice == '1' else 'json'
        output_path = downloads_dir / format_ext / "{}.{}".format(filename, format_ext)
        include_sentiment = sentiment_choice == '1'

        result = collector.export_data(format_ext, str(output_path), include_sentiment)

        if result:
            print("\nExport completed: {}".format(output_path))
        else:
            print("\nExport failed")

    except Exception as e:
        print("\nExport error: {}".format(e))

    input("Press Enter to continue...")


def interactive_settings():
    """Interactive settings menu"""
    while True:
        print("\n" + "=" * 25 + " SETTINGS " + "=" * 25)
        print()
        print("  1. Change API Key")
        print("  2. Set Subscription Tier")
        print("  3. View Current Settings")
        print("  4. Back to Main Menu")
        print()
        print("=" * 59)

        choice = get_user_input("Select option (1-4): ", ['1', '2', '3', '4'])

        if choice == '1':
            # Change API Key
            print("\nCurrent API key: {}...{}".format(
                (NewsCollector.load_api_key() or "None")[:4],
                (NewsCollector.load_api_key() or "")[-4:] if NewsCollector.load_api_key() else ""
            ))
            new_key = get_user_input("Enter new API key (or press Enter to cancel): ", allow_empty=True)
            if new_key:
                NewsCollector.save_api_key(new_key)
                print("API key updated successfully!")
            else:
                print("API key change cancelled.")
            input("Press Enter to continue...")

        elif choice == '2':
            # Set subscription tier
            print("\nAvailable tiers:")
            tiers = list(AV_RATE_LIMITS.keys())
            for i, tier in enumerate(tiers, 1):
                print("  {}. {} - {}".format(i, tier.upper(), AV_RATE_LIMITS[tier]['description']))

            tier_choice = get_user_input("Select tier (1-{}): ".format(len(tiers)),
                                       [str(i) for i in range(1, len(tiers) + 1)])
            selected_tier = tiers[int(tier_choice) - 1]

            # Handle custom rate limit
            if selected_tier == 'custom':
                custom_rate = get_user_input("Enter your custom rate limit (calls per minute): ")
                try:
                    custom_rate = int(custom_rate)
                    if custom_rate <= 0:
                        print("Rate limit must be positive.")
                        input("Press Enter to continue...")
                        continue
                    NewsCollector.save_subscription_tier(selected_tier, custom_rate)
                except ValueError:
                    print("Invalid number. Please try again.")
                    input("Press Enter to continue...")
                    continue
            else:
                NewsCollector.save_subscription_tier(selected_tier)

            print("Subscription tier updated!")
            input("Press Enter to continue...")

        elif choice == '3':
            # View current settings
            api_key = NewsCollector.load_api_key()
            tier = NewsCollector.load_subscription_tier()

            print("\nCurrent Settings:")
            print("  API Key: {}...{}".format(
                (api_key or "None")[:4],
                (api_key or "")[-4:] if api_key else ""
            ))
            print("  Subscription: {} - {}".format(
                tier.upper(),
                AV_RATE_LIMITS[tier]['description']
            ))
            print("  Database: news_downloads/database/news_data.db")
            print("  Cache: ./cache")
            input("Press Enter to continue...")

        elif choice == '4':
            break


def interactive_main():
    """Main interactive application loop"""
    # Check for API key on startup
    if not NewsCollector.load_api_key():
        clear_screen()
        show_header()
        print("\nFirst time setup required!")
        print("You need an Alpha Vantage API key to collect news.")
        print("Get a free key at: https://www.alphavantage.co/support/#api-key")
        print()

        api_key = get_user_input("Enter your Alpha Vantage API key: ")
        NewsCollector.save_api_key(api_key)
        print("\nSetup complete! Starting application...")
        input("Press Enter to continue...")

    while True:
        clear_screen()
        show_header()

        try:
            choice = show_main_menu()

            if choice == '1':
                interactive_symbol_collection()
            elif choice == '2':
                interactive_topic_collection()
            elif choice == '3':
                interactive_export()
            elif choice == '4':
                interactive_settings()
            elif choice == '5':
                print("\nThank you for using Alpha Vantage News Collector!")
                break

        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break


def main():
    """Main entry point - detects CLI vs interactive mode"""
    # If command line arguments provided, use CLI mode
    if len(sys.argv) > 1:
        cli_main()
    else:
        interactive_main()


def cli_main():
    """Original CLI interface"""
    parser = argparse.ArgumentParser(
        description="Collect news and sentiment data from Alpha Vantage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # First time setup (save API key)
  python av_news_collector.py --setup

  # View available topics
  python av_news_collector.py --list-topics

  # After setup, no API key needed:
  python av_news_collector.py --symbol NVDA
  python av_news_collector.py --symbols NVDA,AAPL,TSLA
  python av_news_collector.py --topic technology

  # With date range (format: YYYY-MM-DD)
  python av_news_collector.py --symbol NVDA --from 2025-09-15 --to 2025-09-19

  # Export collected data
  python av_news_collector.py --export csv
  python av_news_collector.py --export json --export-file my_news.json

  # Automation mode (no prompts)
  python av_news_collector.py --symbol AAPL --no-interact

  # Override stored API key for one-time use
  python av_news_collector.py --symbol NVDA --api-key DIFFERENT_KEY
        """
    )

    # Setup and info options
    parser.add_argument("--setup", action="store_true", help="Set up and save your Alpha Vantage API key")
    parser.add_argument("--list-topics", action="store_true", help="Show all available news topics")
    parser.add_argument("--set-tier", choices=list(AV_RATE_LIMITS.keys()),
                       help="Set your Alpha Vantage subscription tier for rate limiting")
    parser.add_argument("--custom-rate", type=int,
                       help="Set custom rate limit (calls per minute) when using --set-tier custom")
    parser.add_argument("--show-tiers", action="store_true", help="Show available subscription tiers")

    # Source options (mutually exclusive with setup)
    source_group = parser.add_mutually_exclusive_group(required=False)
    source_group.add_argument("--symbol", help="Single symbol to collect news for")
    source_group.add_argument("--symbols", help="Comma-separated list of symbols")
    source_group.add_argument("--topic", help="Topic to collect news for (use --list-topics to see all options)")

    # Collection options
    parser.add_argument("--limit", type=int, default=1000, help="Max articles to collect (default: 1000)")
    parser.add_argument("--from", dest="time_from", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--to", dest="time_to", help="End date (YYYY-MM-DD)")

    # Configuration
    parser.add_argument("--api-key", help="Alpha Vantage API key (overrides stored key)")
    parser.add_argument("--db", default="news_data.db", help="Database file path")
    parser.add_argument("--cache-dir", default="./cache", help="Cache directory path")

    # Control options
    parser.add_argument("--dry-run", action="store_true", help="Show what would be collected without API calls")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--no-interact", action="store_true", help="Non-interactive mode for automation (no prompts)")
    parser.add_argument("--tier", choices=list(AV_RATE_LIMITS.keys()),
                       help="Override subscription tier for this run")

    # Export options
    parser.add_argument("--export", choices=['csv', 'json'], help="Export existing data to CSV or JSON format")
    parser.add_argument("--export-file", help="Output filename for export (auto-generated if not specified)")
    parser.add_argument("--export-no-sentiment", action="store_true", help="Export articles only, skip symbol sentiment data")

    args = parser.parse_args()

    # Handle setup mode
    if args.setup:
        setup_api_key()
        return

    # Handle list topics mode
    if args.list_topics:
        list_topics()
        return

    # Handle show tiers mode
    if args.show_tiers:
        show_subscription_tiers()
        return

    # Handle set tier mode
    if args.set_tier:
        if args.set_tier == 'custom':
            if not args.custom_rate:
                print("Error: --custom-rate required when using --set-tier custom")
                print("Example: python av_news_collector.py --set-tier custom --custom-rate 120")
                sys.exit(1)
            NewsCollector.save_subscription_tier(args.set_tier, args.custom_rate)
        else:
            NewsCollector.save_subscription_tier(args.set_tier)
        return

    # Handle export mode
    if args.export:
        # For export, we don't need an API key, just access to the database
        try:
            collector = NewsCollector("dummy_key", args.db, args.cache_dir, 'free', True)  # quiet mode
            include_sentiment = not args.export_no_sentiment
            result = collector.export_data(args.export, args.export_file, include_sentiment)
            if result:
                print("Export completed: {}".format(result))
            else:
                print("Export failed")
                sys.exit(1)
        except Exception as e:
            print("Export failed: {}".format(e))
            sys.exit(1)
        return

    # Require at least one source option for data collection
    if not any([args.symbol, args.symbols, args.topic]):
        print("Error: Must specify --symbol, --symbols, or --topic")
        print("       Use --setup for first-time configuration")
        print("       Use --list-topics to see available topics")
        print("       Use --show-tiers to see subscription options")
        print("       Use --export to export existing data")
        print("       Use --help for more information")
        sys.exit(1)

    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

    # Get API key (prioritize: command line > stored config > environment variable)
    api_key = args.api_key or NewsCollector.load_api_key() or os.environ.get('ALPHA_VANTAGE_API_KEY')

    if not api_key:
        print("Error: No Alpha Vantage API key found.")
        print()
        print("Options:")
        print("  1. Run setup: python av_news_collector.py --setup")
        print("  2. Use command line: python av_news_collector.py --api-key YOUR_KEY ...")
        print("  3. Set environment variable: export ALPHA_VANTAGE_API_KEY=your_key")
        sys.exit(1)

    # Get subscription tier (prioritize: command line > stored config > default free)
    subscription_tier = args.tier or NewsCollector.load_subscription_tier()

    try:
        # Show welcome screen for data collection (unless in no-interact mode)
        if not args.dry_run and not args.no_interact:
            show_welcome_screen()

        # Initialize collector with subscription tier and quiet mode
        collector = NewsCollector(api_key, args.db, args.cache_dir, subscription_tier, args.no_interact)

        # Convert and validate date format if provided
        time_from = None
        time_to = None
        try:
            if args.time_from:
                validated_from = validate_date_format(args.time_from)
                time_from = validated_from + "T0000" if validated_from else None
            if args.time_to:
                validated_to = validate_date_format(args.time_to)
                time_to = validated_to + "T2359" if validated_to else None
        except ValueError as e:
            print("Error: {}".format(e))
            sys.exit(1)

        # Show current API usage
        collector.show_usage_stats()
        print()

        # Execute based on arguments
        success = False

        if args.symbol:
            success = collector.collect_symbol_news(
                args.symbol, args.limit, time_from, time_to, args.dry_run
            )

        elif args.symbols:
            symbols_list = [s.strip().upper() for s in args.symbols.split(',')]
            print("Collecting news for {} symbols: {}".format(len(symbols_list), ", ".join(symbols_list)))

            success_count = 0
            for symbol in symbols_list:
                if collector.collect_symbol_news(symbol, args.limit, time_from, time_to, args.dry_run):
                    success_count += 1
                print()  # Spacing between symbols

            success = success_count > 0
            print("Successfully collected news for {}/{} symbols".format(success_count, len(symbols_list)))

        elif args.topic:
            if args.topic not in ALPHA_VANTAGE_TOPICS:
                print("Warning: '{}' is not a recognized topic. Available topics:".format(args.topic))
                for key, description in ALPHA_VANTAGE_TOPICS.items():
                    print("  {}: {}".format(key, description))
                print()

            success = collector.collect_topic_news(
                args.topic, args.limit, time_from, time_to, args.dry_run
            )

        # Show final API usage
        if not args.dry_run:
            collector.show_usage_stats()

        if success:
            print("\nNews collection completed successfully!")
            print("Database: {}".format(args.db))
            print("Cache: {}".format(args.cache_dir))

            if not args.no_interact:
                # Show final completion message
                print("\n" + "="*50)
                print("Collection Complete! Your news data is ready.")
                print("="*50)

                try:
                    input("\nPress Enter to finish...")
                except (KeyboardInterrupt, EOFError):
                    pass

                print("Thank you for using Alpha Vantage News Collector!")
        else:
            print("\nNews collection failed or returned no data")
            if not args.no_interact:
                print("Try using --dry-run to test your parameters")

    except Exception as e:
        logging.error("News collection failed: {}".format(e))
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()