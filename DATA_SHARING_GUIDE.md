# Training Data Sharing Guide

## Overview

We don't commit databases to the repository (best practice), but we need to share training data with users. This guide explains how to export and import training data.

## Philosophy

- ✅ **Export recent data only** - Excludes old data (June/July) that predates fixes
- ✅ **User bootstrap** - Users import data to bootstrap their databases
- ✅ **No old data** - Only share data after fixes were implemented
- ✅ **Regenerable** - Users can regenerate training episodes from imported data

## Export Process

### When to Export

Export training data when:
- Sharing the project with new users
- Publishing a release
- After significant data collection periods
- Before major changes that might affect data format

### How to Export

```bash
# Export all training data (default cutoff: 2025-11-27)
python3 scripts/export_training_data.py

# Export with custom cutoff date
python3 scripts/export_training_data.py --cutoff-date "2025-11-27T00:00:00"

# Export to specific directory
python3 scripts/export_training_data.py --output-dir my_export
```

### What Gets Exported

1. **Price Data** (`sol_prices_export.sql`)
   - All price records after cutoff date
   - SQL dump format for easy import

2. **News Data** (`news_sentiment_export.sql`)
   - All news articles after cutoff date
   - Includes embeddings, sentiment scores
   - SQL dump format

3. **Training Episodes** (`episodes.pkl`, optional)
   - Pre-generated training episodes
   - Can be regenerated from databases if not included

4. **Manifest** (`manifest.json`)
   - Metadata about the export
   - Date ranges, record counts
   - Import instructions

5. **README** (`README.md`)
   - Human-readable export information
   - Import instructions

### Export Directory Structure

```
training_data_exports/
└── training_data_export_20251201_120000/
    ├── manifest.json
    ├── README.md
    ├── sol_prices_export.sql
    ├── news_sentiment_export.sql
    └── episodes.pkl (optional)
```

## Import Process

### For New Users

1. **Get the export:**
   - Download the export zip file
   - Extract to a directory

2. **Import the data:**
   ```bash
   python3 scripts/import_training_data.py <export_directory>
   ```

3. **Regenerate training episodes** (if not included):
   ```bash
   python3 -m rl_agent.training_data_prep
   ```

4. **Train the model:**
   ```bash
   python3 scripts/train_rl_agent.py --epochs 10
   ```

### Import Behavior

- **Appends data** - Doesn't overwrite existing data
- **Skips duplicates** - Handles UNIQUE constraints gracefully
- **Creates databases** - Creates databases if they don't exist
- **Preserves existing** - Existing data is preserved

## Cutoff Date

**Default Cutoff:** `2025-11-27`

This date represents when recent fixes were implemented. Data before this date is excluded because:
- It may have incorrect embeddings (corrupted data)
- It predates recent fixes and improvements
- It's not needed for training (recent data is sufficient)

**To change cutoff date:**
- Update `DATA_CUTOFF_DATE` in `scripts/export_training_data.py`
- Or use `--cutoff-date` argument

## Sharing Options

### Option 1: Include in Release

1. Export data before release
2. Zip the export directory
3. Attach to GitHub release
4. Users download and import

**Pros:**
- Easy for users
- Versioned with releases
- Can update with each release

**Cons:**
- Larger release files
- Need to update with each release

### Option 2: Separate Repository

1. Create `orca-training-data` repository
2. Export and commit exports
3. Users clone and import

**Pros:**
- Separate from code
- Can version exports
- Easy to update

**Cons:**
- Another repository to maintain
- Still need to update

### Option 3: Cloud Storage

1. Export data
2. Upload to cloud storage (S3, Google Drive, etc.)
3. Share link in README
4. Users download and import

**Pros:**
- No repository bloat
- Easy to update
- Can version with dates

**Cons:**
- External dependency
- Need to maintain links

### Option 4: Regeneration Instructions

1. Don't share data
2. Provide instructions for data collection
3. Users collect their own data

**Pros:**
- No data to share
- Users get fresh data
- No maintenance

**Cons:**
- Users need to wait for data collection
- May not have historical data

## Recommended Approach

**Hybrid: Option 1 + Option 4**

1. **For releases:** Include export in GitHub release
2. **For development:** Provide regeneration instructions
3. **For new users:** Can use either approach

This gives users flexibility while providing a quick start option.

## File Sizes

Typical export sizes:
- Price data: ~5-10 MB (SQL dump)
- News data: ~10-20 MB (SQL dump with embeddings)
- Training episodes: ~50-200 MB (pickle file)
- **Total:** ~65-230 MB

**Recommendation:**
- Include price and news data (small, essential)
- Optionally include episodes (large, but saves regeneration time)
- Or provide instructions to regenerate episodes

## Versioning

Exports should be versioned:
- Include export date in directory name
- Include cutoff date in manifest
- Document what fixes/improvements are included

Example:
```
training_data_export_20251201_120000/  # Export date
  manifest.json  # Contains cutoff date and metadata
```

## Maintenance

### Regular Exports

- Export after significant data collection (weekly/monthly)
- Update cutoff date as fixes are implemented
- Archive old exports (don't delete, but don't actively share)

### Updating Cutoff Date

When new fixes are implemented:
1. Update `DATA_CUTOFF_DATE` in export script
2. Export new data with new cutoff
3. Document what fixes are included

## Troubleshooting

### Import Fails

- Check SQL file format
- Ensure databases exist or can be created
- Check for encoding issues (especially with BLOB data)

### Missing Data

- Verify cutoff date is correct
- Check export manifest for date ranges
- Ensure source databases have data after cutoff

### Large File Sizes

- Consider excluding training episodes (regenerate instead)
- Compress exports (zip)
- Split exports by date range if needed

## Best Practices

1. ✅ **Always export after cutoff date** - Don't share old data
2. ✅ **Include manifest** - Users know what they're getting
3. ✅ **Test imports** - Verify exports can be imported
4. ✅ **Document cutoff date** - Users understand what's included
5. ✅ **Version exports** - Track what's in each export
6. ✅ **Provide regeneration option** - Users can collect their own data

## Example Workflow

### For Project Maintainer

```bash
# 1. Export training data
python3 scripts/export_training_data.py

# 2. Zip the export
cd training_data_exports
zip -r training_data_20251201.zip training_data_export_*/

# 3. Attach to GitHub release
# (manual step)

# 4. Update README with download link
```

### For New User

```bash
# 1. Download export from release
# (manual step)

# 2. Extract
unzip training_data_20251201.zip

# 3. Import
python3 scripts/import_training_data.py training_data_export_*/

# 4. Regenerate episodes (if not included)
python3 -m rl_agent.training_data_prep

# 5. Train
python3 scripts/train_rl_agent.py --epochs 10
```

---

**Last Updated:** 2025-12-01
**Current Cutoff Date:** 2025-11-27

