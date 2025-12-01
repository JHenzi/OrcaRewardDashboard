"""
Export Training Data for Sharing

Exports recent training data (after fixes) for users to bootstrap their databases.
Excludes old data (June/July) that predates recent fixes.

Exports:
- Price data (recent only, after fixes)
- News data (recent only, after fixes)
- Training episodes (if available)
- Metadata about what's included
"""

import sqlite3
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
import argparse
import pickle
from typing import Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cutoff date: Only export data after this date (when fixes were in place)
# Adjust this date based on when your fixes were implemented
DATA_CUTOFF_DATE = datetime(2025, 11, 27)  # Only export data from Nov 27 onwards

# Export paths
EXPORT_DIR = Path("training_data_exports")  # For archives (ignored)
LATEST_EXPORT_DIR = Path("training_data/export")  # For latest (committed)
LATEST_EXPORT_DIR.mkdir(parents=True, exist_ok=True)


class TrainingDataExporter:
    """Export training data for sharing."""
    
    def __init__(
        self,
        price_db_path: str = "sol_prices.db",
        news_db_path: str = "news_sentiment.db",
        cutoff_date: Optional[datetime] = None,
    ):
        self.price_db_path = price_db_path
        self.news_db_path = news_db_path
        self.cutoff_date = cutoff_date or DATA_CUTOFF_DATE
        
    def export_price_data(self, output_path: Path) -> Dict:
        """Export price data after cutoff date."""
        logger.info(f"Exporting price data from {self.cutoff_date} onwards...")
        
        conn = sqlite3.connect(self.price_db_path)
        cursor = conn.cursor()
        
        # Get data after cutoff
        cursor.execute('''
            SELECT timestamp, rate
            FROM sol_prices
            WHERE timestamp >= ?
            ORDER BY timestamp
        ''', (self.cutoff_date.isoformat(),))
        
        rows = cursor.fetchall()
        conn.close()
        
        # Write to SQL dump
        with open(output_path, 'w') as f:
            f.write("-- Price Data Export\n")
            f.write(f"-- Exported: {datetime.now().isoformat()}\n")
            f.write(f"-- Cutoff Date: {self.cutoff_date.isoformat()}\n")
            f.write(f"-- Records: {len(rows)}\n\n")
            f.write("BEGIN TRANSACTION;\n\n")
            f.write("CREATE TABLE IF NOT EXISTS sol_prices (\n")
            f.write("    id INTEGER PRIMARY KEY AUTOINCREMENT,\n")
            f.write("    timestamp TEXT NOT NULL,\n")
            f.write("    rate REAL NOT NULL\n")
            f.write(");\n\n")
            
            for timestamp, rate in rows:
                # Escape single quotes
                timestamp_escaped = timestamp.replace("'", "''")
                f.write(f"INSERT INTO sol_prices (timestamp, rate) VALUES ('{timestamp_escaped}', {rate});\n")
            
            f.write("\nCOMMIT;\n")
        
        logger.info(f"Exported {len(rows)} price records to {output_path}")
        return {
            'records': len(rows),
            'start_date': rows[0][0] if rows else None,
            'end_date': rows[-1][0] if rows else None,
        }
    
    def export_news_data(self, output_path: Path) -> Dict:
        """Export news data after cutoff date."""
        logger.info(f"Exporting news data from {self.cutoff_date} onwards...")
        
        conn = sqlite3.connect(self.news_db_path)
        cursor = conn.cursor()
        
        # Get data after cutoff
        cursor.execute('''
            SELECT 
                title, description, link, source, category,
                published_date, fetched_date,
                is_crypto_related, embedding, sentiment_score,
                sentiment_label, topic_label, price_at_news,
                price_1h_later, price_24h_later, actual_impact, is_labeled
            FROM news_articles
            WHERE published_date >= ?
            ORDER BY published_date
        ''', (self.cutoff_date.isoformat(),))
        
        rows = cursor.fetchall()
        conn.close()
        
        # Write to SQL dump
        with open(output_path, 'wb') as f:
            # Write header as text
            header = f"""-- News Data Export
-- Exported: {datetime.now().isoformat()}
-- Cutoff Date: {self.cutoff_date.isoformat()}
-- Records: {len(rows)}

BEGIN TRANSACTION;

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
);

"""
            f.write(header.encode('utf-8'))
            
            # Write data (need to handle BLOB properly)
            for row in rows:
                values = []
                for i, val in enumerate(row):
                    if val is None:
                        values.append('NULL')
                    elif i == 8:  # embedding (BLOB)
                        # Store as hex string for SQL
                        if isinstance(val, bytes):
                            values.append(f"X'{val.hex()}'")
                        else:
                            values.append('NULL')
                    elif isinstance(val, str):
                        # Escape single quotes
                        val_escaped = val.replace("'", "''")
                        values.append(f"'{val_escaped}'")
                    elif isinstance(val, bool):
                        values.append('1' if val else '0')
                    else:
                        values.append(str(val))
                
                f.write(f"INSERT INTO news_articles ({', '.join(['title', 'description', 'link', 'source', 'category', 'published_date', 'fetched_date', 'is_crypto_related', 'embedding', 'sentiment_score', 'sentiment_label', 'topic_label', 'price_at_news', 'price_1h_later', 'price_24h_later', 'actual_impact', 'is_labeled'])}) VALUES ({', '.join(values)});\n".encode('utf-8'))
            
            f.write(b"\nCOMMIT;\n")
        
        logger.info(f"Exported {len(rows)} news records to {output_path}")
        return {
            'records': len(rows),
            'start_date': rows[0][5] if rows else None,  # published_date
            'end_date': rows[-1][5] if rows else None,
        }
    
    def export_training_episodes(self, episodes_path: str, output_path: Path) -> Optional[Dict]:
        """Export training episodes if available."""
        episodes_file = Path(episodes_path)
        if not episodes_file.exists():
            logger.warning(f"Training episodes not found at {episodes_path}, skipping")
            return None
        
        logger.info(f"Exporting training episodes from {episodes_path}...")
        
        # Copy episodes file
        import shutil
        shutil.copy2(episodes_file, output_path)
        
        # Get metadata
        with open(episodes_file, 'rb') as f:
            episodes = pickle.load(f)
        
        logger.info(f"Exported {len(episodes)} training episodes to {output_path}")
        return {
            'episodes': len(episodes),
            'file_size_mb': output_path.stat().st_size / (1024 * 1024),
        }
    
    def create_manifest(self, export_info: Dict, output_path: Path):
        """Create manifest file with export metadata."""
        manifest = {
            'export_date': datetime.now().isoformat(),
            'cutoff_date': self.cutoff_date.isoformat(),
            'note': 'Only data after cutoff date is included (excludes old pre-fix data)',
            'price_data': export_info.get('price_data', {}),
            'news_data': export_info.get('news_data', {}),
            'training_episodes': export_info.get('training_episodes'),
            'import_instructions': {
                'step1': 'Run: python3 scripts/import_training_data.py <export_directory>',
                'step2': 'This will create/update sol_prices.db and news_sentiment.db',
                'step3': 'Then run: python3 -m rl_agent.training_data_prep to create episodes',
            },
        }
        
        with open(output_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Created manifest at {output_path}")
    
    def export_all(
        self,
        output_dir: Optional[Path] = None,
        episodes_path: str = "training_data/episodes.pkl",
        use_latest_location: bool = True,
    ) -> Path:
        """Export all training data.
        
        Args:
            output_dir: Custom output directory (if None, uses latest location or archive)
            episodes_path: Path to training episodes file
            use_latest_location: If True and output_dir is None, export to committed location
        """
        if output_dir is None:
            if use_latest_location:
                # Export to committed location (latest)
                output_dir = LATEST_EXPORT_DIR
            else:
                # Export to archive location (timestamped)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = EXPORT_DIR / f"training_data_export_{timestamp}"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Exporting training data to {output_dir}")
        logger.info(f"Cutoff date: {self.cutoff_date.isoformat()} (only data after this date)")
        
        export_info = {}
        
        # Export price data
        price_file = output_dir / "sol_prices_export.sql"
        export_info['price_data'] = self.export_price_data(price_file)
        
        # Export news data
        news_file = output_dir / "news_sentiment_export.sql"
        export_info['news_data'] = self.export_news_data(news_file)
        
        # Export training episodes (if available)
        episodes_file = output_dir / "episodes.pkl"
        episodes_info = self.export_training_episodes(episodes_path, episodes_file)
        if episodes_info:
            export_info['training_episodes'] = episodes_info
        
        # Create manifest
        manifest_file = output_dir / "manifest.json"
        self.create_manifest(export_info, manifest_file)
        
        # Create README
        readme_file = output_dir / "README.md"
        self._create_readme(readme_file, export_info)
        
        logger.info(f"✅ Export complete! Files in {output_dir}")
        return output_dir
    
    def _create_readme(self, output_path: Path, export_info: Dict):
        """Create README for export."""
        readme = f"""# Training Data Export

**Export Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Cutoff Date:** {self.cutoff_date.isoformat()}

## What's Included

This export contains training data **after** the cutoff date (recent data with fixes applied).
Old data (June/July) that predates fixes is **excluded**.

### Price Data
- **Records:** {export_info.get('price_data', {}).get('records', 0):,}
- **Date Range:** {export_info.get('price_data', {}).get('start_date', 'N/A')} to {export_info.get('price_data', {}).get('end_date', 'N/A')}
- **File:** `sol_prices_export.sql`

### News Data
- **Records:** {export_info.get('news_data', {}).get('records', 0):,}
- **Date Range:** {export_info.get('news_data', {}).get('start_date', 'N/A')} to {export_info.get('news_data', {}).get('end_date', 'N/A')}
- **File:** `news_sentiment_export.sql`

### Training Episodes
"""
        if export_info.get('training_episodes'):
            readme += f"- **Episodes:** {export_info['training_episodes']['episodes']:,}\n"
            readme += f"- **File Size:** {export_info['training_episodes']['file_size_mb']:.2f} MB\n"
            readme += f"- **File:** `episodes.pkl`\n"
        else:
            readme += "- Not included (regenerate after import)\n"
        
        readme += """
## How to Import

1. **Import databases:**
   ```bash
   python3 scripts/import_training_data.py .
   ```

2. **Regenerate training episodes (if not included):**
   ```bash
   python3 -m rl_agent.training_data_prep
   ```

3. **Train the model:**
   ```bash
   python3 scripts/train_rl_agent.py --epochs 10
   ```

## Notes

- This export only includes data after the cutoff date to avoid sharing outdated data
- Databases will be created/updated in the project root
- Existing data will be preserved (new data is appended)
"""
        
        with open(output_path, 'w') as f:
            f.write(readme)


def main():
    parser = argparse.ArgumentParser(description="Export training data for sharing")
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory (default: training_data_exports/training_data_export_TIMESTAMP)'
    )
    parser.add_argument(
        '--cutoff-date',
        type=str,
        help=f'Cutoff date (ISO format, default: {DATA_CUTOFF_DATE.isoformat()})'
    )
    parser.add_argument(
        '--episodes-path',
        type=str,
        default='training_data/episodes.pkl',
        help='Path to training episodes file'
    )
    parser.add_argument(
        '--archive',
        action='store_true',
        help='Export to archive location (timestamped) instead of latest location (committed)'
    )
    
    args = parser.parse_args()
    
    cutoff_date = None
    if args.cutoff_date:
        cutoff_date = datetime.fromisoformat(args.cutoff_date)
    
    exporter = TrainingDataExporter(cutoff_date=cutoff_date)
    output_dir = exporter.export_all(
        output_dir=Path(args.output_dir) if args.output_dir else None,
        episodes_path=args.episodes_path,
        use_latest_location=not args.archive,
    )
    
    print(f"\n✅ Export complete!")
    print(f"📁 Files exported to: {output_dir}")
    print(f"\nTo share, zip the directory:")
    print(f"  zip -r {output_dir.name}.zip {output_dir.name}/")


if __name__ == "__main__":
    main()

