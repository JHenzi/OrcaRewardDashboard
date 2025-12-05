"""
Fill News Data Gaps

This script identifies gaps in news data and fills them using embedding similarity
and contextual interpolation. This ensures training data has consistent news coverage.

Best Practices Implemented:
1. Embedding Similarity Matching - Find similar news before/after gaps
2. Contextual Interpolation - Use weighted average of similar embeddings
3. Temporal Smoothing - Respect time-dependent patterns
4. Sentiment Preservation - Maintain sentiment consistency
5. Validation - Check filled data quality
"""

import os
# Set tokenizers parallelism before importing sentence-transformers
# This prevents warnings when subprocesses are spawned
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sqlite3
import numpy as np
import pickle
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import json

# For embeddings similarity
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database paths
PRICE_DB = "sol_prices.db"
NEWS_DB = "news_sentiment.db"

# Configuration
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 default
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MIN_SIMILARITY_THRESHOLD = 0.5  # Minimum cosine similarity to consider news "similar"
MAX_GAP_HOURS = 24  # Maximum gap size to fill (don't fill huge historical gaps)


class NewsGapFiller:
    """
    Fills gaps in news data using embedding similarity and interpolation.
    """
    
    def __init__(
        self,
        news_db_path: str = NEWS_DB,
        price_db_path: str = PRICE_DB,
        max_gap_hours: int = MAX_GAP_HOURS,
    ):
        self.news_db_path = news_db_path
        self.price_db_path = price_db_path
        self.max_gap_hours = max_gap_hours
        self.embedding_model = None
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
                logger.info(f"Loaded embedding model: {EMBEDDING_MODEL}")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                self.embedding_model = None
    
    def identify_gaps(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Tuple[datetime, datetime]]:
        """
        Identify gaps in news data relative to price data timestamps.
        
        Args:
            start_date: Start date for analysis (default: earliest price data)
            end_date: End date for analysis (default: latest price data)
            
        Returns:
            List of (gap_start, gap_end) tuples
        """
        logger.info("Identifying gaps in news data...")
        
        # Get price data hourly buckets
        price_conn = sqlite3.connect(self.price_db_path)
        price_cursor = price_conn.cursor()
        
        if start_date is None or end_date is None:
            price_cursor.execute('SELECT MIN(timestamp), MAX(timestamp) FROM sol_prices')
            min_ts, max_ts = price_cursor.fetchone()
            if start_date is None:
                start_date = datetime.fromisoformat(min_ts.replace(' ', 'T'))
            if end_date is None:
                end_date = datetime.fromisoformat(max_ts.replace(' ', 'T'))
        
        # Get hourly buckets from price data
        price_cursor.execute('''
            SELECT DISTINCT strftime('%Y-%m-%d %H:00:00', timestamp) as hour
            FROM sol_prices
            WHERE timestamp >= ? AND timestamp <= ?
            ORDER BY hour
        ''', (start_date.isoformat(), end_date.isoformat()))
        price_hours = {row[0] for row in price_cursor.fetchall()}
        price_conn.close()
        
        # Get news data hourly buckets
        news_conn = sqlite3.connect(self.news_db_path)
        news_cursor = news_conn.cursor()
        news_cursor.execute('''
            SELECT DISTINCT strftime('%Y-%m-%d %H:00:00', published_date) as hour
            FROM news_articles
            WHERE published_date >= ? AND published_date <= ?
            ORDER BY hour
        ''', (start_date.isoformat(), end_date.isoformat()))
        news_hours = {row[0] for row in news_cursor.fetchall()}
        news_conn.close()
        
        # Find missing hours
        missing_hours = sorted(price_hours - news_hours)
        
        # Group consecutive missing hours into gaps
        gaps = []
        if missing_hours:
            gap_start = datetime.fromisoformat(missing_hours[0].replace(' ', 'T'))
            gap_end = gap_start
            
            for hour_str in missing_hours[1:]:
                hour = datetime.fromisoformat(hour_str.replace(' ', 'T'))
                if (hour - gap_end) <= timedelta(hours=1):
                    gap_end = hour
                else:
                    # End of current gap, start new one
                    if (gap_end - gap_start).total_seconds() / 3600 <= self.max_gap_hours:
                        gaps.append((gap_start, gap_end))
                    gap_start = hour
                    gap_end = hour
            
            # Add final gap
            if (gap_end - gap_start).total_seconds() / 3600 <= self.max_gap_hours:
                gaps.append((gap_start, gap_end))
        
        logger.info(f"Found {len(missing_hours)} missing hours grouped into {len(gaps)} gaps")
        return gaps
    
    def get_similar_news(
        self,
        target_embedding: np.ndarray,
        timestamp: datetime,
        hours_window: int = 48,
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Find similar news articles based on embedding similarity.
        
        Args:
            target_embedding: Target embedding to match
            timestamp: Timestamp for the gap
            hours_window: Hours before/after to search
            top_k: Number of similar articles to return
            
        Returns:
            List of similar news articles with similarity scores
        """
        conn = sqlite3.connect(self.news_db_path)
        cursor = conn.cursor()
        
        # Get news within window
        start_time = timestamp - timedelta(hours=hours_window)
        end_time = timestamp + timedelta(hours=hours_window)
        
        cursor.execute('''
            SELECT id, title, embedding, sentiment_score, published_date
            FROM news_articles
            WHERE published_date >= ? AND published_date <= ?
            AND embedding IS NOT NULL
            ORDER BY published_date DESC
        ''', (start_time.isoformat(), end_time.isoformat()))
        
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return []
        
        # Calculate similarities
        similarities = []
        for row in rows:
            article_id, title, embedding_blob, sentiment_score, pub_date = row
            
            try:
                # Decode embedding - handle both pickle and raw numpy bytes
                if isinstance(embedding_blob, bytes):
                    # Try pickle first
                    try:
                        embedding = pickle.loads(embedding_blob)
                    except (pickle.UnpicklingError, ValueError, TypeError):
                        # If pickle fails, try raw numpy bytes
                        try:
                            embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                            # Reshape if needed (should be 384 dim)
                            if len(embedding) == EMBEDDING_DIM:
                                embedding = embedding.reshape(EMBEDDING_DIM)
                            elif len(embedding) > EMBEDDING_DIM:
                                # Truncate if too long
                                embedding = embedding[:EMBEDDING_DIM]
                            else:
                                # Too short, skip
                                logger.debug(f"Article {article_id}: embedding too short ({len(embedding)} bytes)")
                                continue
                        except Exception as e2:
                            logger.debug(f"Article {article_id}: failed to decode as numpy bytes: {e2}")
                            continue
                else:
                    embedding = embedding_blob
                
                if not isinstance(embedding, np.ndarray):
                    embedding = np.array(embedding, dtype=np.float32)
                
                # Validate embedding shape
                if len(embedding) != EMBEDDING_DIM:
                    logger.debug(f"Article {article_id}: wrong embedding dimension ({len(embedding)} != {EMBEDDING_DIM})")
                    continue
                
                # Check for NaN/inf
                if not np.isfinite(embedding).all():
                    logger.debug(f"Article {article_id}: embedding contains NaN/inf")
                    continue
                
                # Calculate cosine similarity
                similarity = np.dot(target_embedding, embedding) / (
                    np.linalg.norm(target_embedding) * np.linalg.norm(embedding)
                )
                
                if similarity >= MIN_SIMILARITY_THRESHOLD:
                    similarities.append({
                        'id': article_id,
                        'title': title,
                        'embedding': embedding,
                        'sentiment_score': float(sentiment_score) if sentiment_score else 0.0,
                        'published_date': pub_date,
                        'similarity': float(similarity),
                    })
            except Exception as e:
                logger.warning(f"Error processing embedding for article {article_id}: {e}")
                continue
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]
    
    def interpolate_news(
        self,
        before_news: List[Dict],
        after_news: List[Dict],
        gap_timestamp: datetime,
    ) -> Dict:
        """
        Interpolate news for a gap timestamp using before/after news.
        
        Uses weighted average of embeddings based on temporal distance.
        
        Args:
            before_news: News articles before the gap
            after_news: News articles after the gap
            gap_timestamp: Timestamp to fill
            
        Returns:
            Interpolated news article dict
        """
        # Find most similar pair between before and after
        best_pair = None
        best_similarity = 0.0
        
        for before in before_news:
            for after in after_news:
                similarity = np.dot(
                    before['embedding'],
                    after['embedding']
                ) / (
                    np.linalg.norm(before['embedding']) *
                    np.linalg.norm(after['embedding'])
                )
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_pair = (before, after)
        
        if best_pair is None or best_similarity < MIN_SIMILARITY_THRESHOLD:
            # No good match, use forward fill (most recent before)
            if before_news:
                source = before_news[0]
            elif after_news:
                source = after_news[0]
            else:
                # No news available, create neutral placeholder
                return self._create_neutral_news(gap_timestamp)
            
            return {
                'title': f"[Filled] {source['title']}",
                'embedding': source['embedding'].copy(),
                'sentiment_score': source['sentiment_score'],
                'is_filled': True,
                'source_article_id': source['id'],
                'fill_method': 'forward_fill',
            }
        
        # Interpolate between best pair
        before, after = best_pair
        
        # Calculate temporal weights
        before_time = datetime.fromisoformat(before['published_date'].replace(' ', 'T'))
        after_time = datetime.fromisoformat(after['published_date'].replace(' ', 'T'))
        
        total_diff = (after_time - before_time).total_seconds()
        if total_diff == 0:
            weight_before = 0.5
        else:
            gap_to_before = (gap_timestamp - before_time).total_seconds()
            weight_before = 1.0 - (gap_to_before / total_diff)
            weight_before = max(0.0, min(1.0, weight_before))
        
        weight_after = 1.0 - weight_before
        
        # Weighted average of embeddings
        interpolated_embedding = (
            weight_before * before['embedding'] +
            weight_after * after['embedding']
        )
        # Normalize
        interpolated_embedding = interpolated_embedding / np.linalg.norm(interpolated_embedding)
        
        # Weighted average of sentiment
        interpolated_sentiment = (
            weight_before * before['sentiment_score'] +
            weight_after * after['sentiment_score']
        )
        
        return {
            'title': f"[Filled] Similar to: {before['title'][:50]}...",
            'embedding': interpolated_embedding.astype(np.float32),
            'sentiment_score': float(interpolated_sentiment),
            'is_filled': True,
            'source_article_ids': [before['id'], after['id']],
            'fill_method': 'interpolation',
            'similarity': float(best_similarity),
        }
    
    def _create_neutral_news(self, timestamp: datetime) -> Dict:
        """Create a neutral placeholder news article."""
        # Create a zero embedding (will be masked out in training)
        neutral_embedding = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        
        return {
            'title': "[Filled] No news available",
            'embedding': neutral_embedding,
            'sentiment_score': 0.0,
            'is_filled': True,
            'fill_method': 'neutral_placeholder',
        }
    
    def fill_gap(
        self,
        gap_start: datetime,
        gap_end: datetime,
    ) -> int:
        """
        Fill a single gap in news data.
        
        Args:
            gap_start: Start of gap
            gap_end: End of gap
            
        Returns:
            Number of articles created
        """
        logger.info(f"Filling gap from {gap_start} to {gap_end}")
        
        # Get news before gap
        before_news = self._get_news_near_timestamp(
            gap_start - timedelta(hours=1),
            hours_window=24,
            top_k=10,
        )
        
        # Get news after gap
        after_news = self._get_news_near_timestamp(
            gap_end + timedelta(hours=1),
            hours_window=24,
            top_k=10,
        )
        
        if not before_news and not after_news:
            logger.warning(f"No news available near gap {gap_start} to {gap_end}, skipping")
            return 0
        
        # Fill each missing hour in the gap
        filled_count = 0
        current = gap_start
        
        while current <= gap_end:
            # Interpolate news for this hour
            filled_news = self.interpolate_news(before_news, after_news, current)
            
            # Insert into database
            if self._insert_filled_news(current, filled_news):
                filled_count += 1
            
            current += timedelta(hours=1)
        
        logger.info(f"Filled {filled_count} hours in gap")
        return filled_count
    
    def _get_news_near_timestamp(
        self,
        timestamp: datetime,
        hours_window: int = 24,
        top_k: int = 10,
    ) -> List[Dict]:
        """Get news articles near a timestamp."""
        conn = sqlite3.connect(self.news_db_path)
        cursor = conn.cursor()
        
        start_time = timestamp - timedelta(hours=hours_window)
        end_time = timestamp + timedelta(hours=hours_window)
        
        # Get more articles than needed to account for corrupted ones
        cursor.execute('''
            SELECT id, title, embedding, sentiment_score, published_date
            FROM news_articles
            WHERE published_date >= ? AND published_date <= ?
            AND embedding IS NOT NULL
            AND title NOT LIKE '[Filled]%'
            ORDER BY ABS(julianday(published_date) - julianday(?))
            LIMIT ?
        ''', (start_time.isoformat(), end_time.isoformat(), timestamp.isoformat(), top_k * 3))
        
        rows = cursor.fetchall()
        conn.close()
        
        news_items = []
        for row in rows:
            article_id, title, embedding_blob, sentiment_score, pub_date = row
            try:
                # Decode embedding - handle both pickle and raw numpy bytes
                if isinstance(embedding_blob, bytes):
                    # Try pickle first
                    try:
                        embedding = pickle.loads(embedding_blob)
                    except (pickle.UnpicklingError, ValueError, TypeError, UnicodeDecodeError):
                        # If pickle fails, try raw numpy bytes
                        try:
                            embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                            # Reshape if needed (should be 384 dim)
                            if len(embedding) == EMBEDDING_DIM:
                                embedding = embedding.reshape(EMBEDDING_DIM)
                            elif len(embedding) > EMBEDDING_DIM:
                                # Truncate if too long
                                embedding = embedding[:EMBEDDING_DIM]
                            else:
                                # Too short, skip
                                logger.debug(f"Article {article_id}: embedding too short ({len(embedding)} bytes)")
                                continue
                        except Exception as e2:
                            logger.debug(f"Article {article_id}: failed to decode as numpy bytes: {e2}")
                            continue
                else:
                    embedding = embedding_blob
                
                if not isinstance(embedding, np.ndarray):
                    embedding = np.array(embedding, dtype=np.float32)
                
                # Validate embedding shape
                if len(embedding) != EMBEDDING_DIM:
                    logger.debug(f"Article {article_id}: wrong embedding dimension ({len(embedding)} != {EMBEDDING_DIM})")
                    continue
                
                # Check for NaN/inf
                if not np.isfinite(embedding).all():
                    logger.debug(f"Article {article_id}: embedding contains NaN/inf")
                    continue
                
                news_items.append({
                    'id': article_id,
                    'title': title,
                    'embedding': embedding,
                    'sentiment_score': float(sentiment_score) if sentiment_score else 0.0,
                    'published_date': pub_date,
                })
                
                # Stop once we have enough valid articles
                if len(news_items) >= top_k:
                    break
            except Exception:
                continue
        
        return news_items
    
    def _insert_filled_news(
        self,
        timestamp: datetime,
        news_data: Dict,
    ) -> bool:
        """Insert filled news article into database."""
        conn = sqlite3.connect(self.news_db_path)
        cursor = conn.cursor()
        
        try:
            # Check if already exists
            cursor.execute('''
                SELECT id FROM news_articles
                WHERE published_date = ?
                AND title LIKE '[Filled]%'
            ''', (timestamp.isoformat(),))
            
            if cursor.fetchone():
                logger.debug(f"Filled news already exists for {timestamp}")
                conn.close()
                return False
            
            # Serialize embedding
            embedding_blob = pickle.dumps(news_data['embedding'])
            
            # Determine sentiment label
            sentiment_score = news_data['sentiment_score']
            if sentiment_score > 0.1:
                sentiment_label = 'positive'
            elif sentiment_score < -0.1:
                sentiment_label = 'negative'
            else:
                sentiment_label = 'neutral'
            
            # Insert
            cursor.execute('''
                INSERT INTO news_articles (
                    title, description, link, source, category,
                    published_date, fetched_date,
                    is_crypto_related, embedding, sentiment_score,
                    sentiment_label, is_labeled
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                news_data['title'],
                json.dumps({
                    'is_filled': True,
                    'fill_method': news_data.get('fill_method', 'unknown'),
                    'source_article_ids': news_data.get('source_article_ids', news_data.get('source_article_id')),
                    'similarity': news_data.get('similarity'),
                }),
                f"filled://{timestamp.isoformat()}",  # Unique link for filled articles
                'filled',
                'filled',
                timestamp.isoformat(),
                datetime.now().isoformat(),
                0,  # Not crypto-related by default
                embedding_blob,
                sentiment_score,
                sentiment_label,
                0,  # Not labeled
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Error inserting filled news: {e}")
            conn.rollback()
            conn.close()
            return False
    
    def fill_all_gaps(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        dry_run: bool = False,
    ) -> Dict:
        """
        Fill all identified gaps in news data.
        
        Args:
            start_date: Start date for analysis
            end_date: End date for analysis
            dry_run: If True, only report what would be filled
            
        Returns:
            Summary dict with statistics
        """
        gaps = self.identify_gaps(start_date, end_date)
        
        if not gaps:
            logger.info("No gaps found in news data")
            return {
                'gaps_found': 0,
                'articles_created': 0,
                'hours_filled': 0,
            }
        
        logger.info(f"Found {len(gaps)} gaps to fill")
        
        total_filled = 0
        total_hours = 0
        
        for gap_start, gap_end in gaps:
            hours_in_gap = int((gap_end - gap_start).total_seconds() / 3600) + 1
            total_hours += hours_in_gap
            
            if not dry_run:
                filled = self.fill_gap(gap_start, gap_end)
                total_filled += filled
            else:
                logger.info(f"Would fill {hours_in_gap} hours from {gap_start} to {gap_end}")
        
        summary = {
            'gaps_found': len(gaps),
            'hours_to_fill': total_hours,
            'articles_created': total_filled if not dry_run else 0,
            'hours_filled': total_filled if not dry_run else 0,
        }
        
        logger.info(f"Summary: {summary}")
        return summary


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fill gaps in news data")
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Only report what would be filled, do not modify database'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date (ISO format, e.g., 2025-11-27T00:00:00)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date (ISO format)'
    )
    parser.add_argument(
        '--max-gap-hours',
        type=int,
        default=MAX_GAP_HOURS,
        help=f'Maximum gap size to fill (default: {MAX_GAP_HOURS})'
    )
    
    args = parser.parse_args()
    
    # Update max gap hours if provided
    if args.max_gap_hours != MAX_GAP_HOURS:
        # Pass as parameter to filler instead of global
        logger.info(f"Using max gap hours: {args.max_gap_hours}")
    
    filler = NewsGapFiller(max_gap_hours=args.max_gap_hours)
    
    start_date = None
    end_date = None
    
    if args.start_date:
        start_date = datetime.fromisoformat(args.start_date)
    if args.end_date:
        end_date = datetime.fromisoformat(args.end_date)
    
    summary = filler.fill_all_gaps(
        start_date=start_date,
        end_date=end_date,
        dry_run=args.dry_run,
    )
    
    if args.dry_run:
        print("\n✅ Dry run complete. Use without --dry-run to apply changes.")
    else:
        print(f"\n✅ Filled {summary['articles_created']} news articles across {summary['hours_filled']} hours")


if __name__ == "__main__":
    main()

