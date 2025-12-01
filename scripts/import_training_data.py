"""
Import Training Data

Imports exported training data to bootstrap databases.
Handles existing data gracefully (appends, doesn't overwrite).
"""

import sqlite3
import json
import logging
from pathlib import Path
import argparse
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingDataImporter:
    """Import training data from exports."""
    
    def __init__(
        self,
        price_db_path: str = "sol_prices.db",
        news_db_path: str = "news_sentiment.db",
    ):
        self.price_db_path = price_db_path
        self.news_db_path = news_db_path
    
    def import_price_data(self, sql_file: Path) -> int:
        """Import price data from SQL dump."""
        logger.info(f"Importing price data from {sql_file}...")
        
        if not sql_file.exists():
            logger.error(f"File not found: {sql_file}")
            return 0
        
        conn = sqlite3.connect(self.price_db_path)
        cursor = conn.cursor()
        
        # Read and execute SQL
        with open(sql_file, 'r') as f:
            sql = f.read()
        
        # Execute SQL (skip CREATE TABLE if exists)
        # Split by semicolons and execute statements
        statements = [s.strip() for s in sql.split(';') if s.strip() and not s.strip().startswith('--')]
        
        imported = 0
        for statement in statements:
            if statement.upper().startswith('CREATE TABLE'):
                # Skip CREATE TABLE (database should already exist)
                continue
            elif statement.upper().startswith('BEGIN TRANSACTION'):
                continue
            elif statement.upper().startswith('COMMIT'):
                continue
            elif statement.upper().startswith('INSERT'):
                try:
                    cursor.execute(statement)
                    imported += 1
                except sqlite3.IntegrityError as e:
                    # Skip duplicates
                    logger.debug(f"Skipping duplicate: {e}")
                    continue
                except Exception as e:
                    logger.warning(f"Error executing statement: {e}")
                    continue
        
        conn.commit()
        conn.close()
        
        logger.info(f"Imported {imported} price records")
        return imported
    
    def import_news_data(self, sql_file: Path) -> int:
        """Import news data from SQL dump."""
        logger.info(f"Importing news data from {sql_file}...")
        
        if not sql_file.exists():
            logger.error(f"File not found: {sql_file}")
            return 0
        
        conn = sqlite3.connect(self.news_db_path)
        cursor = conn.cursor()
        
        # Read SQL (may be binary if contains BLOB)
        try:
            with open(sql_file, 'r', encoding='utf-8') as f:
                sql = f.read()
        except UnicodeDecodeError:
            # Try binary mode
            with open(sql_file, 'rb') as f:
                sql = f.read().decode('utf-8', errors='ignore')
        
        # Execute SQL
        statements = [s.strip() for s in sql.split(';') if s.strip() and not s.strip().startswith('--')]
        
        imported = 0
        for statement in statements:
            if statement.upper().startswith('CREATE TABLE'):
                # Execute CREATE TABLE to ensure table exists
                try:
                    cursor.execute(statement)
                except sqlite3.OperationalError:
                    # Table already exists, skip
                    pass
                continue
            elif statement.upper().startswith('BEGIN TRANSACTION'):
                continue
            elif statement.upper().startswith('COMMIT'):
                continue
            elif statement.upper().startswith('INSERT'):
                try:
                    cursor.execute(statement)
                    imported += 1
                except sqlite3.IntegrityError as e:
                    # Skip duplicates (based on UNIQUE constraint)
                    logger.debug(f"Skipping duplicate: {e}")
                    continue
                except Exception as e:
                    logger.warning(f"Error executing statement: {e}")
                    continue
        
        conn.commit()
        conn.close()
        
        logger.info(f"Imported {imported} news records")
        return imported
    
    def import_training_episodes(self, pkl_file: Path, output_path: str = "training_data/episodes.pkl") -> bool:
        """Import training episodes."""
        if not pkl_file.exists():
            logger.warning(f"Training episodes file not found: {pkl_file}")
            return False
        
        logger.info(f"Copying training episodes from {pkl_file}...")
        
        import shutil
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pkl_file, output_path)
        
        logger.info(f"Copied training episodes to {output_path}")
        return True
    
    def import_all(self, export_dir: Path) -> Dict:
        """Import all data from export directory."""
        logger.info(f"Importing training data from {export_dir}...")
        
        if not export_dir.exists():
            logger.error(f"Export directory not found: {export_dir}")
            return {}
        
        # Check for manifest
        manifest_file = export_dir / "manifest.json"
        if manifest_file.exists():
            with open(manifest_file, 'r') as f:
                manifest = json.load(f)
            logger.info(f"Importing data exported on {manifest.get('export_date')}")
            logger.info(f"Cutoff date: {manifest.get('cutoff_date')}")
        
        results = {}
        
        # Import price data
        price_file = export_dir / "sol_prices_export.sql"
        if price_file.exists():
            results['price_records'] = self.import_price_data(price_file)
        else:
            logger.warning(f"Price data file not found: {price_file}")
        
        # Import news data
        news_file = export_dir / "news_sentiment_export.sql"
        if news_file.exists():
            results['news_records'] = self.import_news_data(news_file)
        else:
            logger.warning(f"News data file not found: {news_file}")
        
        # Import training episodes
        episodes_file = export_dir / "episodes.pkl"
        if episodes_file.exists():
            results['episodes_imported'] = self.import_training_episodes(episodes_file)
        else:
            logger.info("Training episodes not included (will need to regenerate)")
        
        logger.info("✅ Import complete!")
        return results


def main():
    parser = argparse.ArgumentParser(description="Import training data from export")
    parser.add_argument(
        'export_dir',
        type=str,
        help='Directory containing exported data (should contain manifest.json)'
    )
    parser.add_argument(
        '--price-db',
        type=str,
        default='sol_prices.db',
        help='Path to price database (default: sol_prices.db)'
    )
    parser.add_argument(
        '--news-db',
        type=str,
        default='news_sentiment.db',
        help='Path to news database (default: news_sentiment.db)'
    )
    
    args = parser.parse_args()
    
    importer = TrainingDataImporter(
        price_db_path=args.price_db,
        news_db_path=args.news_db,
    )
    
    results = importer.import_all(Path(args.export_dir))
    
    print("\n✅ Import Summary:")
    print(f"  Price records: {results.get('price_records', 0):,}")
    print(f"  News records: {results.get('news_records', 0):,}")
    if results.get('episodes_imported'):
        print(f"  Training episodes: Imported")
    else:
        print(f"  Training episodes: Not included (regenerate with: python3 -m rl_agent.training_data_prep)")


if __name__ == "__main__":
    main()

