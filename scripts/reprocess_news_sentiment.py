#!/usr/bin/env python3
"""
Re-process news articles with keyword-based sentiment analysis.

This script updates existing articles in the database that have neutral sentiment
(score 0.0) with keyword-based sentiment analysis as a fallback until the
classifier is trained.
"""

import sqlite3
from datetime import datetime, timedelta
import sys
import os

# Add project root to path (parent of scripts directory)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from news_sentiment import NewsSentimentAnalyzer

def reprocess_articles(hours_back=24*7):  # Default: last 7 days
    """Re-process articles with keyword-based sentiment."""
    analyzer = NewsSentimentAnalyzer()
    
    # Database path relative to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    db_path = os.path.join(project_root, 'news_sentiment.db')
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cutoff_time = datetime.now() - timedelta(hours=hours_back)
    
    # Get articles with neutral sentiment
    cursor.execute('''
        SELECT id, title, description, sentiment_score, sentiment_label
        FROM news_articles 
        WHERE published_date >= ? AND (sentiment_score = 0.0 OR sentiment_label = 'neutral')
    ''', (cutoff_time.isoformat(),))
    
    articles = cursor.fetchall()
    print(f"Found {len(articles)} articles to re-process...")
    
    updated = 0
    positive_count = 0
    negative_count = 0
    neutral_count = 0
    
    for article_id, title, desc, old_score, old_label in articles:
        text = (title + ' ' + (desc or '')).strip()
        if text:
            new_score, new_label = analyzer._keyword_based_sentiment(text)
            
            # Only update if sentiment changed
            if new_label != old_label or abs(new_score - old_score) > 0.01:
                cursor.execute('''
                    UPDATE news_articles 
                    SET sentiment_score = ?, sentiment_label = ?
                    WHERE id = ?
                ''', (new_score, new_label, article_id))
                updated += 1
                
                if new_label in ['good', 'very_good']:
                    positive_count += 1
                elif new_label in ['bad', 'very_bad']:
                    negative_count += 1
                else:
                    neutral_count += 1
                
                if updated <= 10:  # Show first 10 examples
                    print(f"  Updated: {title[:60]}... -> {new_label} ({new_score:.2f})")
    
    conn.commit()
    conn.close()
    
    print(f"\n✅ Re-processed {updated} articles")
    print(f"   Positive: {positive_count}")
    print(f"   Negative: {negative_count}")
    print(f"   Neutral: {neutral_count}")
    
    return updated

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Re-process news articles with keyword-based sentiment')
    parser.add_argument('--hours', type=int, default=24*7, help='Hours back to process (default: 168 = 7 days)')
    args = parser.parse_args()
    
    reprocess_articles(hours_back=args.hours)

