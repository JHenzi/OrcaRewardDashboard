"""
News Sentiment Analysis Module

This module fetches news from multiple sources, generates local embeddings,
performs sentiment analysis, and provides features for the trading system.

Features:
- Multi-source RSS feed aggregation (crypto, finance, tech, general news)
- Local sentence-transformers embeddings (no cloud API calls)
- Trainable sentiment classifier (good/bad for trading)
- News topic classification
- Integration with bandit trading system
"""

import sqlite3
import json
import logging
import feedparser
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import numpy as np
from pathlib import Path
import pickle

# For local embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not installed. Install with: pip install sentence-transformers")

# For sentiment classification
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not installed. Install with: pip install scikit-learn")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database setup
NEWS_DB = "news_sentiment.db"
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
FEEDS_CONFIG_PATH = Path("news_feeds.json")

# Default news sources (fallback if JSON config not found)
DEFAULT_NEWS_SOURCES = {
    "crypto": [
        "https://cointelegraph.com/rss",
        "https://decrypt.co/feed",
    ],
    "finance": [
        "https://feeds.bloomberg.com/markets/news.rss",
    ],
    "tech": [
        "https://techcrunch.com/feed/",
        "https://www.theverge.com/rss/index.xml",
        "https://arstechnica.com/feed/",
    ],
    "general": [
        "https://feeds.bbci.co.uk/news/rss.xml",
        "http://feeds.abcnews.com/abcnews/topstories",
        "http://www.npr.org/rss/rss.php?id=1001",
        "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
        "http://feeds.nbcnews.com/feeds/topstories",
        "http://feeds.foxnews.com/foxnews/latest",
    ]
}

def load_news_feeds_config() -> Dict:
    """
    Load RSS feed configuration from JSON file.
    
    Returns:
        Dict with feeds and settings, or default if file not found
    """
    if FEEDS_CONFIG_PATH.exists():
        try:
            with open(FEEDS_CONFIG_PATH, 'r') as f:
                config = json.load(f)
                logger.info(f"Loaded news feeds config from {FEEDS_CONFIG_PATH}")
                return config
        except Exception as e:
            logger.warning(f"Failed to load news feeds config: {e}. Using defaults.")
    
    # Return default structure
    return {
        "feeds": {cat: [{"url": url, "enabled": True, "priority": 1} 
                       for url in urls] 
                 for cat, urls in DEFAULT_NEWS_SOURCES.items()},
        "settings": {
            "fetch_cooldown_minutes": 5,
            "max_headlines_per_fetch": 100,
            "hours_back": 24,
            "embedding_model": "all-MiniLM-L6-v2"
        }
    }

# Keywords to identify crypto/SOL-related news
CRYPTO_KEYWORDS = [
    "bitcoin", "btc", "ethereum", "eth", "solana", "sol", "crypto", "cryptocurrency",
    "blockchain", "defi", "nft", "web3", "altcoin", "stablecoin", "token", "coin",
    "exchange", "wallet", "mining", "staking", "liquidity", "pool", "trading"
]


class NewsSentimentAnalyzer:
    """
    Analyzes news sentiment for trading signals.
    
    Uses local embeddings and a trainable classifier to determine:
    1. What the news is about (topic classification)
    2. Whether it's good or bad for crypto/SOL trading
    """
    
    def __init__(self, embedding_model_name: str = None):
        """
        Initialize the news sentiment analyzer.
        
        Args:
            embedding_model_name: Name of sentence-transformers model to use
                                  (default: from config or "all-MiniLM-L6-v2")
        """
        # Load feed configuration
        self.config = load_news_feeds_config()
        self.feeds_config = self.config.get("feeds", {})
        self.settings = self.config.get("settings", {})
        
        # Use model from config or provided parameter
        self.embedding_model_name = embedding_model_name or self.settings.get("embedding_model", "all-MiniLM-L6-v2")
        self.fetch_cooldown_minutes = self.settings.get("fetch_cooldown_minutes", 5)
        self.max_headlines_per_fetch = self.settings.get("max_headlines_per_fetch", 100)
        
        self.embedding_model = None
        self.sentiment_classifier = None
        self.topic_classifier = None
        
        # Track last fetch time
        self.last_fetch_time = None
        
        # Initialize database
        self.init_database()
        
        # Load or initialize models
        self._load_models()
        
    def init_database(self):
        """Initialize the news sentiment database."""
        conn = sqlite3.connect(NEWS_DB)
        cursor = conn.cursor()
        
        # News articles table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS news_articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                description TEXT,
                link TEXT UNIQUE,
                source TEXT,
                category TEXT,
                published_date TIMESTAMP,
                fetched_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_crypto_related BOOLEAN DEFAULT 0,
                embedding BLOB,
                sentiment_score REAL,
                sentiment_label TEXT,
                topic_label TEXT,
                price_at_news REAL,
                price_1h_later REAL,
                price_24h_later REAL,
                actual_impact REAL,
                is_labeled BOOLEAN DEFAULT 0
            )
        """)
        
        # Create indexes for performance
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_published_date 
            ON news_articles(published_date DESC)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_is_crypto_related 
            ON news_articles(is_crypto_related)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_sentiment_label 
            ON news_articles(sentiment_label)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_is_labeled 
            ON news_articles(is_labeled)
        """)
        
        conn.commit()
        conn.close()
        logger.info("News sentiment database initialized")
    
    def _load_models(self):
        """Load or initialize embedding and classification models."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.error("sentence-transformers not available. Install with: pip install sentence-transformers")
            return
        
        # Load embedding model
        try:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info(f"Loaded embedding model: {self.embedding_model_name}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            return
        
        # Load or initialize sentiment classifier
        sentiment_model_path = MODEL_DIR / "sentiment_classifier.pkl"
        if sentiment_model_path.exists():
            try:
                with open(sentiment_model_path, "rb") as f:
                    self.sentiment_classifier = pickle.load(f)
                logger.info("Loaded trained sentiment classifier")
            except Exception as e:
                logger.warning(f"Failed to load sentiment classifier: {e}")
                self._init_default_classifier()
        else:
            self._init_default_classifier()
        
        # Load or initialize topic classifier
        topic_model_path = MODEL_DIR / "topic_classifier.pkl"
        if topic_model_path.exists():
            try:
                with open(topic_model_path, "rb") as f:
                    self.topic_classifier = pickle.load(f)
                logger.info("Loaded trained topic classifier")
            except Exception as e:
                logger.warning(f"Failed to load topic classifier: {e}")
                self._init_default_topic_classifier()
        else:
            self._init_default_topic_classifier()
    
    def _init_default_classifier(self):
        """Initialize a default sentiment classifier (untrained)."""
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available. Sentiment classification disabled.")
            return
        
        # Simple logistic regression classifier
        # Will be trained on labeled data
        self.sentiment_classifier = LogisticRegression(
            max_iter=1000,
            random_state=42
        )
        logger.info("Initialized default sentiment classifier (untrained)")
    
    def _init_default_topic_classifier(self):
        """Initialize a default topic classifier (untrained)."""
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available. Topic classification disabled.")
            return
        
        self.topic_classifier = LogisticRegression(
            max_iter=1000,
            random_state=42,
            multi_class='multinomial'
        )
        logger.info("Initialized default topic classifier (untrained)")
    
    def should_fetch(self) -> bool:
        """
        Check if enough time has passed since last fetch.
        
        Returns:
            True if should fetch, False if in cooldown period
        """
        if self.last_fetch_time is None:
            return True
        
        time_since_last = datetime.now() - self.last_fetch_time
        cooldown = timedelta(minutes=self.fetch_cooldown_minutes)
        
        if time_since_last < cooldown:
            remaining = (cooldown - time_since_last).total_seconds() / 60
            logger.info(f"News fetch in cooldown. {remaining:.1f} minutes remaining.")
            return False
        
        return True
    
    def get_latest_news_timestamp(self) -> Optional[datetime]:
        """
        Get the timestamp of the most recently fetched news article.
        
        Returns:
            Datetime of latest news, or None if no news exists
        """
        try:
            conn = sqlite3.connect(NEWS_DB)
            cursor = conn.cursor()
            cursor.execute("SELECT MAX(fetched_date) FROM news_articles")
            result = cursor.fetchone()
            conn.close()
            
            if result and result[0]:
                # Parse the timestamp string - handle various formats
                ts_str = str(result[0])
                try:
                    # Try ISO format first (with or without timezone)
                    if 'T' in ts_str:
                        # Handle ISO format: remove timezone if present
                        if ts_str.endswith('Z'):
                            ts_clean = ts_str[:-1]
                            return datetime.fromisoformat(ts_clean)
                        elif '+' in ts_str or ts_str.count('-') > 2:
                            # Has timezone info
                            return datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                        else:
                            return datetime.fromisoformat(ts_str)
                    else:
                        # Try SQLite datetime format: 'YYYY-MM-DD HH:MM:SS'
                        return datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S')
                except (ValueError, AttributeError) as e:
                    logger.warning(f"Failed to parse timestamp '{ts_str}': {e}")
                    return None
            return None
        except Exception as e:
            logger.error(f"Error getting latest news timestamp: {e}")
            return None
    
    def is_news_stale(self, stale_hours: int = 1) -> bool:
        """
        Check if news is stale (older than specified hours).
        
        Default is 1 hour to ensure fresh data for RL agent training.
        The 5-minute cooldown prevents rate limit issues while allowing
        frequent updates.
        
        Args:
            stale_hours: Number of hours to consider news stale (default: 1)
            
        Returns:
            True if news is stale or doesn't exist, False otherwise
        """
        latest = self.get_latest_news_timestamp()
        if latest is None:
            return True  # No news = stale
        
        # Ensure both datetimes are timezone-aware for comparison
        from datetime import timezone
        now = datetime.now(timezone.utc)
        if latest.tzinfo is None:
            latest = latest.replace(tzinfo=timezone.utc)
        
        time_since_latest = now - latest
        stale_threshold = timedelta(hours=stale_hours)
        
        return time_since_latest > stale_threshold
    
    def fetch_news(self, hours_back: int = None, force: bool = False) -> List[Dict]:
        """
        Fetch news from all configured RSS sources.
        
        Args:
            hours_back: How many hours of news to fetch (default: from config)
            force: If True, bypass cooldown check
            
        Returns:
            List of news article dictionaries
        """
        # Check cooldown unless forced
        if not force and not self.should_fetch():
            logger.info("Skipping news fetch - in cooldown period")
            return []
        
        # Use hours_back from config if not provided
        if hours_back is None:
            hours_back = self.settings.get("hours_back", 24)
        
        all_articles = []
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        # Load feeds from config
        for category, feed_list in self.feeds_config.items():
            # Sort by priority (lower number = higher priority)
            sorted_feeds = sorted(
                [f for f in feed_list if f.get("enabled", True)],
                key=lambda x: x.get("priority", 1)
            )
            
            for feed_config in sorted_feeds:
                feed_url = feed_config.get("url")
                if not feed_url:
                    continue
                
                try:
                    logger.info(f"Fetching from {feed_url}")
                    feed = feedparser.parse(feed_url)
                    
                    for entry in feed.entries:
                        # Parse published date
                        published = None
                        if hasattr(entry, 'published_parsed') and entry.published_parsed:
                            try:
                                published = datetime(*entry.published_parsed[:6])
                            except:
                                pass
                        
                        # Skip if too old
                        if published and published < cutoff_time:
                            continue
                        
                        article = {
                            "title": entry.get("title", ""),
                            "description": entry.get("description", ""),
                            "link": entry.get("link", ""),
                            "source": feed_url,
                            "category": category,
                            "published_date": published.isoformat() if published else None,
                        }
                        
                        all_articles.append(article)
                        
                        # Limit total articles
                        if len(all_articles) >= self.max_headlines_per_fetch:
                            logger.info(f"Reached max headlines limit ({self.max_headlines_per_fetch})")
                            break
                    
                    if len(all_articles) >= self.max_headlines_per_fetch:
                        break
                        
                except Exception as e:
                    logger.error(f"Error fetching {feed_url}: {e}")
                    continue
        
        # Update last fetch time
        self.last_fetch_time = datetime.now()
        
        logger.info(f"Fetched {len(all_articles)} news articles")
        return all_articles
    
    def is_crypto_related(self, title: str, description: str = "") -> bool:
        """
        Check if a news article is crypto/SOL related.
        
        Args:
            title: Article title
            description: Article description
            
        Returns:
            True if article is crypto-related
        """
        text = (title + " " + description).lower()
        return any(keyword in text for keyword in CRYPTO_KEYWORDS)
    
    def _keyword_based_sentiment(self, text: str) -> Tuple[float, str]:
        """
        Simple keyword-based sentiment analysis as fallback when classifier isn't trained.
        
        Args:
            text: Article text (title + description)
            
        Returns:
            Tuple of (sentiment_score, sentiment_label)
        """
        if not text:
            return 0.0, "neutral"
        
        text_lower = text.lower()
        
        # Positive keywords
        positive_keywords = [
            "surge", "rally", "gain", "up", "rise", "soar", "jump", "boost", "growth",
            "profit", "success", "breakthrough", "approval", "adoption", "partnership",
            "launch", "announcement", "milestone", "record", "high", "bullish", "optimistic",
            "increase", "expand", "positive", "strong", "win", "victory", "breakthrough"
        ]
        
        # Negative keywords
        negative_keywords = [
            "crash", "drop", "fall", "down", "decline", "plunge", "loss", "risk",
            "concern", "warning", "ban", "regulation", "lawsuit", "hack", "breach",
            "scam", "fraud", "investigation", "crisis", "bearish", "pessimistic", "fear",
            "decrease", "collapse", "failure", "negative", "weak", "defeat", "problem"
        ]
        
        positive_count = sum(1 for keyword in positive_keywords if keyword in text_lower)
        negative_count = sum(1 for keyword in negative_keywords if keyword in text_lower)
        
        # Calculate sentiment score
        if positive_count > negative_count:
            score = min(0.3 + (positive_count - negative_count) * 0.15, 1.0)
            if score > 0.6:
                label = "very_good"
            else:
                label = "good"
        elif negative_count > positive_count:
            score = max(-0.3 - (negative_count - positive_count) * 0.15, -1.0)
            if score < -0.6:
                label = "very_bad"
            else:
                label = "bad"
        else:
            score = 0.0
            label = "neutral"
        
        return float(score), label
    
    def generate_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Generate embedding for text using local sentence-transformers model.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector or None if model not available
        """
        if not self.embedding_model:
            return None
        
        try:
            embedding = self.embedding_model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
    
    def predict_sentiment(self, embedding: np.ndarray, text: str = "") -> Tuple[float, str]:
        """
        Predict sentiment (good/bad for trading) from embedding.
        
        Args:
            embedding: News article embedding
            text: Optional text for keyword-based fallback
            
        Returns:
            Tuple of (sentiment_score, sentiment_label)
            sentiment_score: -1 (very bad) to +1 (very good)
            sentiment_label: "very_bad", "bad", "neutral", "good", "very_good"
        """
        if not self.sentiment_classifier or not SKLEARN_AVAILABLE:
            # Fallback: use simple keyword-based sentiment
            return self._keyword_based_sentiment(text)
        
        try:
            # Check if model is fitted
            if not hasattr(self.sentiment_classifier, 'classes_'):
                logger.debug("Sentiment classifier not trained yet, using keyword-based fallback")
                return self._keyword_based_sentiment(text)
            
            # Predict probability of positive class
            prob = self.sentiment_classifier.predict_proba(embedding.reshape(1, -1))
            
            # Convert to sentiment score (-1 to +1)
            # Assuming binary classification: [negative_prob, positive_prob]
            if prob.shape[1] == 2:
                sentiment_score = prob[0][1] * 2 - 1  # Scale from [0,1] to [-1,1]
            else:
                # Multi-class: use the highest probability class
                sentiment_score = (np.argmax(prob[0]) / (prob.shape[1] - 1)) * 2 - 1
            
            # Convert to label
            if sentiment_score < -0.6:
                label = "very_bad"
            elif sentiment_score < -0.2:
                label = "bad"
            elif sentiment_score < 0.2:
                label = "neutral"
            elif sentiment_score < 0.6:
                label = "good"
            else:
                label = "very_good"
            
            return float(sentiment_score), label
            
        except Exception as e:
            logger.debug(f"Sentiment classifier not trained or error: {e}. Using keyword-based fallback.")
            return self._keyword_based_sentiment(text)
    
    def predict_topic(self, embedding: np.ndarray) -> str:
        """
        Predict topic category from embedding.
        
        Args:
            embedding: News article embedding
            
        Returns:
            Topic label (e.g., "crypto", "finance", "tech", "regulation", etc.)
        """
        if not self.topic_classifier or not SKLEARN_AVAILABLE:
            return "unknown"
        
        try:
            # Check if model is fitted
            if not hasattr(self.topic_classifier, 'classes_'):
                logger.debug("Topic classifier not trained yet, returning unknown")
                return "unknown"
            
            topic_idx = self.topic_classifier.predict(embedding.reshape(1, -1))[0]
            topic_labels = ["crypto", "finance", "tech", "regulation", "market", "other"]
            if topic_idx < len(topic_labels):
                return topic_labels[topic_idx]
            return "unknown"
        except Exception as e:
            logger.debug(f"Topic classifier not trained or error: {e}. Returning unknown.")
            return "unknown"
    
    def process_and_store_news(self, articles: List[Dict], current_price: float) -> int:
        """
        Process news articles: generate embeddings, predict sentiment, store in DB.
        
        Args:
            articles: List of news article dictionaries
            current_price: Current SOL price (for tracking impact)
            
        Returns:
            Number of articles processed
        """
        if not self.embedding_model:
            logger.error("Embedding model not available. Cannot process news.")
            return 0
        
        conn = sqlite3.connect(NEWS_DB)
        cursor = conn.cursor()
        
        processed = 0
        for article in articles:
            try:
                # Check if already exists
                cursor.execute("SELECT id FROM news_articles WHERE link = ?", (article["link"],))
                if cursor.fetchone():
                    continue
                
                # Check if crypto-related
                is_crypto = self.is_crypto_related(article["title"], article.get("description", ""))
                
                # Generate embedding
                text = article["title"] + " " + article.get("description", "")
                embedding = self.generate_embedding(text)
                
                if embedding is None:
                    continue
                
                # Predict sentiment (pass text for keyword fallback)
                sentiment_score, sentiment_label = self.predict_sentiment(embedding, text)
                
                # Predict topic
                topic_label = self.predict_topic(embedding)
                
                # Store in database
                cursor.execute("""
                    INSERT INTO news_articles (
                        title, description, link, source, category,
                        published_date, is_crypto_related, embedding,
                        sentiment_score, sentiment_label, topic_label,
                        price_at_news
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    article["title"],
                    article.get("description", ""),
                    article["link"],
                    article["source"],
                    article["category"],
                    article.get("published_date"),
                    is_crypto,
                    embedding.tobytes(),
                    sentiment_score,
                    sentiment_label,
                    topic_label,
                    current_price
                ))
                
                processed += 1
                
            except Exception as e:
                logger.error(f"Error processing article '{article.get('title', 'unknown')}': {e}")
                continue
        
        conn.commit()
        conn.close()
        logger.info(f"Processed and stored {processed} news articles")
        return processed
    
    def get_recent_news_for_rl_agent(
        self,
        hours: int = 24,
        max_headlines: int = 20,
        crypto_only: bool = False,
    ) -> List[Dict]:
        """
        Get recent news articles with embeddings and sentiment for RL agent.
        
        This provides per-headline data (not aggregated) for flexible model input.
        
        Args:
            hours: How many hours of news to retrieve
            max_headlines: Maximum number of headlines to return
            crypto_only: Only include crypto-related news
            
        Returns:
            List of dicts with keys:
                - 'embedding': np.ndarray (embedding_dim,)
                - 'sentiment_score': float (-1 to 1)
                - 'headline': str
                - 'cluster_id': int (optional)
                - 'title': str
                - 'link': str
                - 'published_date': str
        """
        conn = sqlite3.connect(NEWS_DB)
        cursor = conn.cursor()
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        query = """
            SELECT id, title, link, published_date, embedding, sentiment_score, 
                   sentiment_label, topic_label, is_crypto_related
            FROM news_articles
            WHERE published_date >= ?
        """
        params = [cutoff_time.isoformat()]
        
        if crypto_only:
            query += " AND is_crypto_related = 1"
        
        query += " ORDER BY published_date DESC LIMIT ?"
        params.append(max_headlines)
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()
        
        news_items = []
        for row in results:
            article_id, title, link, pub_date, embedding_bytes, sentiment_score, \
                sentiment_label, topic_label, is_crypto = row
            
            # Decode embedding from bytes
            embedding = None
            if embedding_bytes:
                try:
                    embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                except Exception as e:
                    logger.warning(f"Failed to decode embedding for article {article_id}: {e}")
                    continue
            
            if embedding is None:
                continue
            
            news_items.append({
                "embedding": embedding,
                "sentiment_score": float(sentiment_score) if sentiment_score else 0.0,
                "headline": title,
                "title": title,
                "link": link,
                "published_date": pub_date,
                "cluster_id": None,  # Will be populated by clustering if available
                "article_id": article_id,
            })
        
        logger.info(f"Retrieved {len(news_items)} news items for RL agent")
        return news_items
    
    def cluster_news_topics(
        self,
        hours: int = 168,  # 1 week
        min_cluster_size: int = 3,
        n_clusters: int = 10,
    ) -> Dict[int, Dict]:
        """
        Cluster news articles by topic using embeddings.
        
        Args:
            hours: How many hours of news to cluster
            min_cluster_size: Minimum articles per cluster (for HDBSCAN)
            n_clusters: Number of clusters (for KMeans)
            
        Returns:
            Dict mapping cluster_id to cluster info
        """
        try:
            from sklearn.cluster import KMeans
            CLUSTERING_AVAILABLE = True
        except ImportError:
            CLUSTERING_AVAILABLE = False
            logger.warning("scikit-learn not available for clustering")
            return {}
        
        if not CLUSTERING_AVAILABLE:
            return {}
        
        conn = sqlite3.connect(NEWS_DB)
        cursor = conn.cursor()
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        cursor.execute("""
            SELECT id, title, embedding, sentiment_score, published_date
            FROM news_articles
            WHERE published_date >= ? AND embedding IS NOT NULL
            ORDER BY published_date DESC
        """, (cutoff_time.isoformat(),))
        
        results = cursor.fetchall()
        conn.close()
        
        if len(results) < min_cluster_size:
            logger.info(f"Not enough articles for clustering: {len(results)} < {min_cluster_size}")
            return {}
        
        # Extract embeddings
        embeddings = []
        article_ids = []
        titles = []
        
        for row in results:
            article_id, title, embedding_bytes, sentiment_score, pub_date = row
            try:
                embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                embeddings.append(embedding)
                article_ids.append(article_id)
                titles.append(title)
            except Exception as e:
                logger.warning(f"Failed to decode embedding for article {article_id}: {e}")
                continue
        
        if len(embeddings) < min_cluster_size:
            return {}
        
        embeddings_array = np.array(embeddings)
        
        # Use KMeans for clustering
        n_clusters_actual = min(n_clusters, len(embeddings) // min_cluster_size)
        if n_clusters_actual < 2:
            n_clusters_actual = 2
        
        kmeans = KMeans(n_clusters=n_clusters_actual, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings_array)
        
        # Group articles by cluster
        clusters = {}
        for i, (article_id, title, cluster_id) in enumerate(zip(article_ids, titles, cluster_labels)):
            if cluster_id not in clusters:
                clusters[cluster_id] = {
                    "cluster_id": int(cluster_id),
                    "article_ids": [],
                    "headlines": [],
                    "centroid": kmeans.cluster_centers_[cluster_id].tolist(),
                }
            clusters[cluster_id]["article_ids"].append(article_id)
            clusters[cluster_id]["headlines"].append(title)
        
        # Generate cluster labels from representative headlines
        for cluster_id, cluster_info in clusters.items():
            # Use top 3 headlines as label
            headlines = cluster_info["headlines"][:3]
            cluster_label = self._generate_cluster_label(headlines)
            cluster_info["label"] = cluster_label
            cluster_info["representative_headlines"] = headlines
            cluster_info["size"] = len(cluster_info["article_ids"])
        
        # Store clusters in database
        self._store_clusters(clusters)
        
        logger.info(f"Clustered {len(embeddings)} articles into {len(clusters)} topics")
        return clusters
    
    def _generate_cluster_label(self, headlines: List[str]) -> str:
        """
        Generate a human-readable label for a cluster from headlines.
        
        Args:
            headlines: List of representative headlines
            
        Returns:
            Cluster label string
        """
        if not headlines:
            return "unknown"
        
        # Simple keyword extraction
        text = " ".join(headlines).lower()
        
        # Common topic keywords
        topic_keywords = {
            "regulation": ["regulation", "regulatory", "sec", "cfdc", "law", "legal", "ban", "approval"],
            "politics": ["trump", "biden", "president", "congress", "senate", "election", "policy"],
            "earnings": ["earnings", "revenue", "profit", "quarterly", "financial", "results"],
            "technology": ["technology", "tech", "innovation", "development", "launch", "update"],
            "market": ["market", "trading", "price", "volatility", "bull", "bear", "rally", "crash"],
            "crypto": ["bitcoin", "ethereum", "crypto", "blockchain", "defi", "nft", "solana"],
        }
        
        scores = {}
        for topic, keywords in topic_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                scores[topic] = score
        
        if scores:
            top_topic = max(scores.items(), key=lambda x: x[1])[0]
            return top_topic
        else:
            # Use first few words of first headline
            first_words = headlines[0].split()[:3]
            return "_".join(first_words).lower()[:30]
    
    def _store_clusters(self, clusters: Dict[int, Dict]):
        """Store cluster information in database."""
        conn = sqlite3.connect(NEWS_DB)
        cursor = conn.cursor()
        
        # Ensure table exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS news_clusters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cluster_id INTEGER NOT NULL,
                cluster_label TEXT,
                representative_headlines TEXT,
                embedding_centroid BLOB,
                article_count INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        for cluster_info in clusters.values():
            import json
            cursor.execute("""
                INSERT INTO news_clusters (
                    cluster_id, cluster_label, representative_headlines,
                    embedding_centroid, article_count
                ) VALUES (?, ?, ?, ?, ?)
            """, (
                cluster_info["cluster_id"],
                cluster_info["label"],
                json.dumps(cluster_info["representative_headlines"]),
                np.array(cluster_info["centroid"]).tobytes(),
                cluster_info["size"],
            ))
        
        conn.commit()
        conn.close()
    
    def get_cluster_for_article(self, article_id: int) -> Optional[int]:
        """
        Get cluster ID for a specific article.
        
        Args:
            article_id: Article ID
            
        Returns:
            Cluster ID or None
        """
        # This would require maintaining a mapping table
        # For now, return None - would need to implement article-cluster mapping
        return None
    
    def get_recent_news_features(self, hours: int = 24, crypto_only: bool = True) -> Dict:
        """
        Get aggregated news sentiment features for the trading system.
        
        Args:
            hours: How many hours of news to aggregate
            crypto_only: Only include crypto-related news
            
        Returns:
            Dictionary of news features for bandit system, including latest_fetch_timestamp
        """
        conn = sqlite3.connect(NEWS_DB)
        cursor = conn.cursor()
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        query = """
            SELECT sentiment_score, sentiment_label, topic_label, is_crypto_related
            FROM news_articles
            WHERE published_date >= ?
        """
        params = [cutoff_time.isoformat()]
        
        if crypto_only:
            query += " AND is_crypto_related = 1"
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        
        # Get latest fetch timestamp
        cursor.execute("SELECT MAX(fetched_date) FROM news_articles")
        latest_fetch_result = cursor.fetchone()
        latest_fetch_timestamp = None
        if latest_fetch_result and latest_fetch_result[0]:
            try:
                ts_str = str(latest_fetch_result[0])
                if 'T' in ts_str:
                    if ts_str.endswith('Z'):
                        latest_fetch_timestamp = datetime.fromisoformat(ts_str[:-1])
                    elif '+' in ts_str or ts_str.count('-') > 2:
                        latest_fetch_timestamp = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                    else:
                        latest_fetch_timestamp = datetime.fromisoformat(ts_str)
                else:
                    latest_fetch_timestamp = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S')
            except (ValueError, AttributeError):
                pass
        
        conn.close()
        
        if not results:
            # Return neutral features if no news
            return {
                "news_sentiment_score": 0.0,
                "news_sentiment_label": "neutral",
                "news_count": 0,
                "news_positive_count": 0,
                "news_negative_count": 0,
                "news_crypto_count": 0,
                "latest_fetch_timestamp": latest_fetch_timestamp,
            }
        
        sentiment_scores = [r[0] for r in results if r[0] is not None]
        sentiment_labels = [r[1] for r in results if r[1]]
        
        # Calculate aggregated features - ensure native Python types
        if sentiment_scores:
            avg_sentiment = float(np.mean(sentiment_scores))  # Convert numpy scalar to Python float
        else:
            avg_sentiment = 0.0
        
        positive_count = int(sum(1 for label in sentiment_labels if label in ["good", "very_good"]))
        negative_count = int(sum(1 for label in sentiment_labels if label in ["bad", "very_bad"]))
        
        # Determine overall label
        if avg_sentiment > 0.3:
            overall_label = "positive"
        elif avg_sentiment < -0.3:
            overall_label = "negative"
        else:
            overall_label = "neutral"
        
        # Get latest news timestamp for display (use the one we already fetched)
        latest_timestamp_str = None
        if latest_fetch_timestamp:
            latest_timestamp_str = latest_fetch_timestamp.strftime("%Y-%m-%d %H:%M:%S")
        
        return {
            "news_sentiment_score": float(avg_sentiment),  # Explicit float conversion
            "news_sentiment_label": str(overall_label),  # Explicit string conversion
            "news_count": int(len(results)),  # Explicit int conversion
            "news_positive_count": int(positive_count),  # Explicit int conversion
            "news_negative_count": int(negative_count),  # Explicit int conversion
            "news_crypto_count": int(sum(1 for r in results if r[3])),  # Explicit int conversion
            "latest_fetch_timestamp": latest_fetch_timestamp,  # Latest news fetch timestamp (datetime object)
            "latest_fetch_timestamp_str": latest_timestamp_str,  # Latest news fetch timestamp (formatted string)
        }
    
    def train_sentiment_classifier(self, min_labeled_samples: int = 50):
        """
        Train the sentiment classifier on labeled news data.
        
        This should be called after manually labeling news articles based on
        their actual impact on price movements.
        
        Args:
            min_labeled_samples: Minimum number of labeled samples required
        """
        if not SKLEARN_AVAILABLE:
            logger.error("scikit-learn not available. Cannot train classifier.")
            return
        
        conn = sqlite3.connect(NEWS_DB)
        cursor = conn.cursor()
        
        # Get labeled articles with embeddings
        cursor.execute("""
            SELECT embedding, actual_impact
            FROM news_articles
            WHERE is_labeled = 1 AND embedding IS NOT NULL AND actual_impact IS NOT NULL
        """)
        results = cursor.fetchall()
        conn.close()
        
        if len(results) < min_labeled_samples:
            logger.warning(
                f"Only {len(results)} labeled samples available. "
                f"Need at least {min_labeled_samples} to train. "
                "Label more articles by updating actual_impact in the database."
            )
            return
        
        # Prepare data
        X = []
        y = []
        
        for embedding_bytes, actual_impact in results:
            embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
            X.append(embedding)
            # Convert actual_impact to binary label: positive impact = 1, negative = 0
            y.append(1 if actual_impact > 0 else 0)
        
        X = np.array(X)
        y = np.array(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train classifier
        self.sentiment_classifier.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.sentiment_classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Sentiment classifier trained. Accuracy: {accuracy:.2%}")
        logger.info(f"\n{classification_report(y_test, y_pred)}")
        
        # Save model
        model_path = MODEL_DIR / "sentiment_classifier.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(self.sentiment_classifier, f)
        
        logger.info(f"Saved trained model to {model_path}")
    
    def update_news_impact(self, hours_later: int = 24):
        """
        Update news articles with actual price impact.
        
        This should be called periodically to:
        1. Fetch prices at time of news + X hours later
        2. Calculate actual_impact
        3. Mark articles as labeled for training
        
        Args:
            hours_later: How many hours after news to check price
        """
        # This would integrate with your price database
        # For now, it's a placeholder that you can implement
        logger.info("update_news_impact() - Implement price tracking integration")


def main():
    """Test the news sentiment analyzer."""
    print("Initializing News Sentiment Analyzer...")
    analyzer = NewsSentimentAnalyzer()
    
    print("\nFetching news...")
    articles = analyzer.fetch_news(hours_back=24)
    print(f"Fetched {len(articles)} articles")
    
    # Get current SOL price (placeholder - integrate with your price fetcher)
    current_price = 150.0  # Replace with actual price
    
    print("\nProcessing and storing news...")
    processed = analyzer.process_and_store_news(articles, current_price)
    print(f"Processed {processed} articles")
    
    print("\nGetting recent news features...")
    features = analyzer.get_recent_news_features(hours=24, crypto_only=True)
    print(f"News features: {features}")
    
    print("\n✅ News sentiment analysis complete!")


if __name__ == "__main__":
    main()

