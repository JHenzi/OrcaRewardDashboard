#!/usr/bin/env python3
"""Regenerate embeddings for articles that have NULL embeddings."""

import sqlite3
import pickle
import numpy as np
import os
from pathlib import Path
import sys

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = script_dir
sys.path.insert(0, project_root)

from news_sentiment import NewsSentimentAnalyzer

# Find the database
db_path = 'news_sentiment.db'
if not Path(db_path).exists():
    db_path = os.path.join(project_root, 'news_sentiment.db')

if not Path(db_path).exists():
    print(f'ERROR: news_sentiment.db not found')
    exit(1)

print(f'Using database: {db_path}')

# Initialize news analyzer (loads embedding model)
print('Loading embedding model...')
analyzer = NewsSentimentAnalyzer()
if not analyzer.embedding_model:
    print('ERROR: Could not load embedding model')
    exit(1)

print('Embedding model loaded successfully')

# Connect to database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Find articles with NULL embeddings but have title/description
cursor.execute("""
    SELECT id, title, description 
    FROM news_articles 
    WHERE embedding IS NULL 
    AND (title IS NOT NULL OR description IS NOT NULL)
""")
rows = cursor.fetchall()

print(f'Found {len(rows)} articles needing embeddings...')

regenerated = 0
failed = 0

for article_id, title, description in rows:
    try:
        # Combine title and description
        text = (title or "") + " " + (description or "")
        text = text.strip()
        
        if not text:
            continue
        
        # Generate embedding
        embedding = analyzer.generate_embedding(text)
        
        if embedding is None:
            print(f'Failed to generate embedding for article {article_id}')
            failed += 1
            continue
        
        # Validate embedding
        if embedding.shape != (384,):
            print(f'Invalid embedding shape for article {article_id}: {embedding.shape}')
            failed += 1
            continue
        
        if not np.isfinite(embedding).all():
            print(f'Non-finite values in embedding for article {article_id}')
            failed += 1
            continue
        
        # Serialize and store
        embedding_blob = pickle.dumps(embedding.astype(np.float32))
        
        cursor.execute(
            'UPDATE news_articles SET embedding = ? WHERE id = ?',
            (embedding_blob, article_id)
        )
        
        regenerated += 1
        
        if regenerated % 100 == 0:
            conn.commit()
            print(f'Regenerated {regenerated} embeddings...')
            
    except Exception as e:
        print(f'Error processing article {article_id}: {e}')
        failed += 1
        continue

# Final commit
conn.commit()
conn.close()

print(f'\nDone!')
print(f'Regenerated: {regenerated}')
print(f'Failed: {failed}')
