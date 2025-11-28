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

# News sources - diversified to avoid crypto-only bias
NEWS_SOURCES = {
    "crypto": [
        # "https://coindesk.com/feed/",  # Currently rate-limited (HTTP 429)
        "https://cointelegraph.com/rss",  # Verified working
        "https://decrypt.co/feed",  # Verified working
    ],
    "finance": [
        "https://feeds.bloomberg.com/markets/news.rss",
        # Note: Financial Times and Reuters feeds are not currently accessible
        # FT redirects to HTML, Reuters has connection issues
    ],
    "tech": [
        "https://techcrunch.com/feed/",
        "https://www.theverge.com/rss/index.xml",
        "https://arstechnica.com/feed/",
    ],
    "general": [
        "https://feeds.bbci.co.uk/news/rss.xml",  # BBC News - General (verified working)
        "http://feeds.abcnews.com/abcnews/topstories",  # ABC News - Top Stories (verified working)
        "http://www.npr.org/rss/rss.php?id=1001",  # NPR - National News (verified working)
        "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",  # New York Times - Home Page (verified working)
        "http://feeds.nbcnews.com/feeds/topstories",  # NBC News - Top Stories (verified working)
        "http://feeds.foxnews.com/foxnews/latest",  # Fox News - Latest Headlines (verified working)
        # Note: Reuters feeds have connection issues, removed for now
    ]
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
    
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the news sentiment analyzer.
        
        Args:
            embedding_model_name: Name of sentence-transformers model to use
                                  (default: all-MiniLM-L6-v2 - small, fast, good quality)
        """
        self.embedding_model_name = embedding_model_name
        self.embedding_model = None
        self.sentiment_classifier = None
        self.topic_classifier = None
        
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
    
    def fetch_news(self, hours_back: int = 24) -> List[Dict]:
        """
        Fetch news from all configured RSS sources.
        
        Args:
            hours_back: How many hours of news to fetch
            
        Returns:
            List of news article dictionaries
        """
        all_articles = []
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        for category, feeds in NEWS_SOURCES.items():
            for feed_url in feeds:
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
                        
                except Exception as e:
                    logger.error(f"Error fetching {feed_url}: {e}")
                    continue
        
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
    
    def get_recent_news_features(self, hours: int = 24, crypto_only: bool = True) -> Dict:
        """
        Get aggregated news sentiment features for the trading system.
        
        Args:
            hours: How many hours of news to aggregate
            crypto_only: Only include crypto-related news
            
        Returns:
            Dictionary of news features for bandit system
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
        
        return {
            "news_sentiment_score": float(avg_sentiment),  # Explicit float conversion
            "news_sentiment_label": str(overall_label),  # Explicit string conversion
            "news_count": int(len(results)),  # Explicit int conversion
            "news_positive_count": int(positive_count),  # Explicit int conversion
            "news_negative_count": int(negative_count),  # Explicit int conversion
            "news_crypto_count": int(sum(1 for r in results if r[3])),  # Explicit int conversion
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

