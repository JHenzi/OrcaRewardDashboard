#!/usr/bin/env python3
"""Fix corrupted news embeddings in the database."""

import sqlite3
import pickle
import numpy as np
import os
from pathlib import Path

# Find the database (check current dir and project root)
db_path = 'news_sentiment.db'
if not Path(db_path).exists():
    # Try project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    db_path = os.path.join(project_root, 'news_sentiment.db')

if not Path(db_path).exists():
    print(f'Database not found at {db_path}')
    print('Checking current directory...')
    db_path = 'news_sentiment.db'
    if not Path(db_path).exists():
        print('ERROR: news_sentiment.db not found')
        exit(1)

print(f'Using database: {db_path}')

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Check if table exists
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='news_articles'")
if not cursor.fetchone():
    print('ERROR: news_articles table does not exist')
    conn.close()
    exit(1)

# Find all articles with embeddings
cursor.execute('SELECT id, embedding FROM news_articles WHERE embedding IS NOT NULL')
rows = cursor.fetchall()

print(f'Checking {len(rows)} articles with embeddings...')

bad_ids = []
for article_id, blob in rows:
    if blob:
        try:
            # Try to deserialize
            emb = pickle.loads(blob)
            arr = np.array(emb, dtype=np.float32)
            # Validate shape and values
            if arr.shape != (384,) or not np.isfinite(arr).all():
                print(f'Invalid shape/values: article {article_id}')
                bad_ids.append(article_id)
        except Exception as e:
            error_msg = str(e)
            if 'BYTEARRAY8' in error_msg or 'exceeds' in error_msg:
                print(f'Corrupted: article {article_id}: {error_msg[:60]}')
            bad_ids.append(article_id)

print(f'\nFound {len(bad_ids)} corrupted embeddings')

# Clear corrupted embeddings
if bad_ids:
    cursor.executemany('UPDATE news_articles SET embedding = NULL WHERE id = ?', [(i,) for i in bad_ids])
    conn.commit()
    print(f'Cleared {len(bad_ids)} corrupted embeddings')

conn.close()
print('Done!')
