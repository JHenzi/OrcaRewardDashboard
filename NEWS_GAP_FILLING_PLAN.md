# News Data Gap Filling Plan

## Problem Statement

The RL agent training requires consistent news data coverage. However, there are gaps in the news database where price data exists but news articles are missing. These gaps can cause issues during training when the model expects news features but finds none.

## Current Situation

**Analysis Results:**
- Price data spans: 2025-07-01 to 2025-12-01 (206,359 data points)
- News data spans: 2025-11-27 to 2025-12-01 (1,026 articles)
- **Gap identified**: 6 missing hours in recent data (since 2025-11-27)
- **Large historical gap**: No news data from July to November (this is expected - news collection started recently)

## Solution: Embedding-Based Gap Filling

### Approach

We use **embedding similarity** to intelligently fill gaps:

1. **Identify Gaps**: Find hours where price data exists but news is missing
2. **Find Similar News**: Use cosine similarity on embeddings to find related news before/after the gap
3. **Interpolate**: If surrounding news has similar embeddings (e.g., both about "trump"), interpolate between them
4. **Forward/Backward Fill**: If no similar news found, use most recent/upcoming news
5. **Neutral Placeholder**: If no news available at all, create neutral placeholder (zero embedding, masked in training)

### Best Practices Implemented

Based on research on time series imputation:

1. **Contextual Interpolation** ✅
   - Weighted average of embeddings based on temporal distance
   - Preserves semantic meaning (if old and new news are similar, gap news is similar)

2. **Temporal Smoothing** ✅
   - Respects time-dependent patterns
   - Uses news from nearby time periods (24-48 hour window)

3. **Embedding Similarity Matching** ✅
   - Cosine similarity threshold (0.5 minimum)
   - Finds most similar news pairs before/after gap

4. **Sentiment Preservation** ✅
   - Interpolates sentiment scores along with embeddings
   - Maintains sentiment consistency

5. **Validation** ✅
   - Marks filled articles with `[Filled]` prefix
   - Stores metadata about fill method and source articles
   - Can be filtered out if needed

## Implementation

### Script: `scripts/fill_news_gaps.py`

**Features:**
- Identifies gaps relative to price data timestamps
- Uses embedding similarity to find related news
- Interpolates or forward/backward fills based on context
- Inserts filled articles into database with metadata
- Dry-run mode for testing

**Usage:**

```bash
# Dry run (see what would be filled)
python3 scripts/fill_news_gaps.py --dry-run

# Fill gaps in recent data (default: all data)
python3 scripts/fill_news_gaps.py

# Fill gaps in specific date range
python3 scripts/fill_news_gaps.py \
    --start-date "2025-11-27T00:00:00" \
    --end-date "2025-12-01T00:00:00"

# Adjust maximum gap size to fill (default: 24 hours)
python3 scripts/fill_news_gaps.py --max-gap-hours 48
```

### Algorithm Details

**For each gap:**

1. Get news articles within 24 hours before gap start
2. Get news articles within 24 hours after gap end
3. For each missing hour in gap:
   - Find most similar pair between before/after news (cosine similarity)
   - If similarity > threshold (0.5):
     - Interpolate embeddings using temporal weights
     - Interpolate sentiment scores
   - Else:
     - Use forward fill (most recent before) or backward fill (next after)
   - If no news available:
     - Create neutral placeholder (zero embedding, will be masked)

**Interpolation Formula:**

```
weight_before = 1.0 - (gap_time - before_time) / (after_time - before_time)
weight_after = 1.0 - weight_before

interpolated_embedding = weight_before * before_embedding + weight_after * after_embedding
interpolated_sentiment = weight_before * before_sentiment + weight_after * after_sentiment
```

## Database Schema

Filled articles are inserted with:
- `title`: `[Filled] ...` (prefixed to identify)
- `link`: `filled://{timestamp}` (unique identifier)
- `source`: `"filled"`
- `description`: JSON metadata with:
  - `is_filled`: true
  - `fill_method`: "interpolation" | "forward_fill" | "neutral_placeholder"
  - `source_article_ids`: IDs of source articles used
  - `similarity`: cosine similarity score (if interpolation)

## Integration with Training

The filled articles are automatically used during training:

1. `TrainingDataPrep.get_news_at_time()` will include filled articles
2. Filled articles have valid embeddings and sentiment scores
3. Neutral placeholders (zero embeddings) are automatically masked by `StateEncoder`
4. Training proceeds normally with no special handling needed

## Validation

After filling gaps, validate:

```python
# Check filled articles
SELECT COUNT(*) FROM news_articles WHERE title LIKE '[Filled]%';

# Check gap coverage
# (Run gap identification again - should show 0 gaps)
```

## Limitations

1. **Large Historical Gaps**: Script only fills gaps up to `MAX_GAP_HOURS` (default: 24 hours)
   - Rationale: Very large gaps (weeks/months) shouldn't be filled with interpolated data
   - Solution: Only fill recent gaps where we have surrounding context

2. **No News Context**: If there's no news before/after a gap, creates neutral placeholder
   - This is acceptable - zero embeddings are masked in training

3. **Embedding Quality**: Depends on quality of embeddings from sentence-transformers
   - Using `all-MiniLM-L6-v2` (384-dim) which is good for semantic similarity

## Next Steps

1. **Run Gap Analysis**:
   ```bash
   python3 scripts/fill_news_gaps.py --dry-run
   ```

2. **Fill Gaps** (if gaps found):
   ```bash
   python3 scripts/fill_news_gaps.py
   ```

3. **Validate**:
   - Check database for filled articles
   - Re-run gap identification to confirm gaps are filled
   - Test training data preparation to ensure news data is available

4. **Before Training**:
   - Always run gap filling before training
   - Consider adding to training pipeline/script

## Research References

Best practices for time series imputation:
- Linear interpolation for smooth data
- Forward/backward filling for sparse data
- Contextual interpolation using related features (embeddings in our case)
- Validation to ensure imputed data maintains statistical properties

Our approach combines:
- **Embedding similarity** (contextual matching)
- **Temporal interpolation** (respects time patterns)
- **Fallback strategies** (forward fill, neutral placeholder)

This ensures filled data is semantically consistent and temporally appropriate.

