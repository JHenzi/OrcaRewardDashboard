# MLOps Strategy for RL Agent

This document outlines the MLOps (Machine Learning Operations) strategy for the RL agent, including automated retraining, model versioning, and deployment practices.

## Overview

The RL agent uses a production-ready MLOps pipeline that ensures:
- ✅ **Automated weekly retraining** - Keeps model current with new data
- ✅ **Model validation** - Only deploy models that pass validation
- ✅ **Graceful rollback** - Keep old model if new one fails
- ✅ **Model versioning** - Track all model versions with timestamps
- ✅ **Automatic cleanup** - Remove old models after 30 days (configurable)

## Architecture

### Components

1. **ModelManager** (`rl_agent/model_manager.py`)
   - Handles model versioning and deployment
   - Validates models before deployment
   - Manages model archives
   - Automatic cleanup of old models

2. **RetrainingScheduler** (`rl_agent/retraining_scheduler.py`)
   - Schedules weekly retraining
   - Runs asynchronously (non-blocking)
   - Integrates with ModelManager for safe deployment
   - Handles errors gracefully

3. **Integration in app.py**
   - Initializes ModelManager and scheduler on startup
   - Loads current model automatically
   - Provides API endpoints for status and manual retraining

## Model Versioning

### Version Format

Models are versioned with timestamps:
```
checkpoint_20251128_153000.pt
```

This format ensures:
- Unique versions (timestamp-based)
- Chronological ordering
- Easy identification of training time

### Model Storage

```
models/rl_agent/
├── checkpoint_20251128_153000.pt  # Current active model
├── metadata.json                  # Model metadata and version tracking
└── archive/
    ├── checkpoint_20251121_140000.pt  # Archived models
    ├── checkpoint_20251114_130000.pt
    └── ...
```

### Metadata Structure

```json
{
  "models": [
    {
      "version": "20251128_153000",
      "deployed_at": "2025-11-28T15:30:00",
      "source_path": "models/rl_agent/checkpoint_20251128_153000.pt",
      "active_path": "models/rl_agent/checkpoint_20251128_153000.pt",
      "training_epochs": 5,
      "training_loss": 0.0234
    }
  ],
  "current_version": "20251128_153000",
  "last_retrain_time": "2025-11-28T15:30:00",
  "next_retrain_time": "2025-12-05T15:30:00"
}
```

## Deployment Process

### Automatic Deployment Flow

1. **Scheduler triggers retraining** (weekly)
2. **Data preparation** - Prepares training episodes
3. **Model training** - Trains new model (incremental, 5 epochs)
4. **Model validation** - Validates new model:
   - ✅ File exists and is readable
   - ✅ Can load model state
   - ✅ Model structure matches expected architecture
   - ✅ Forward pass works correctly
5. **Archive current model** - Moves old model to archive
6. **Deploy new model** - Copies to active location
7. **Update metadata** - Records deployment
8. **Cleanup old models** - Removes models older than 30 days

### Safety Features

- **Validation before deployment** - New model must pass all checks
- **Keep old model** - Current model stays active if new one fails
- **Rollback capability** - Can manually revert to previous version
- **Error handling** - All errors logged, no silent failures

## Retraining Schedule

### Weekly Retraining

- **Interval**: 7 days
- **Mode**: Incremental (last 30 days of data)
- **Epochs**: 5 (fewer than initial training)
- **Execution**: Asynchronous (non-blocking)

### Manual Retraining

You can manually trigger retraining via API:

```bash
curl -X POST http://localhost:5030/api/rl-agent/retrain
```

Or with async control:

```bash
curl -X POST http://localhost:5030/api/rl-agent/retrain \
  -H "Content-Type: application/json" \
  -d '{"async": true}'
```

## Model Retention Policy

### Default: 30 Days

- **Active model**: Always kept (current version)
- **Archived models**: Kept for 30 days
- **Minimum retention**: At least last 3 models (safety)
- **Automatic cleanup**: Runs after each deployment

### Customization

You can adjust retention in `app.py`:

```python
rl_model_manager = ModelManager(
    retention_days=30,  # Change this
)
```

## API Endpoints

### Get Status

```bash
GET /api/rl-agent/status
```

Returns:
```json
{
  "success": true,
  "status": {
    "model_loaded": true,
    "model_info": {
      "current_version": "20251128_153000",
      "current_path": "models/rl_agent/checkpoint_20251128_153000.pt",
      "total_models": 4,
      "retention_days": 30
    },
    "scheduler": {
      "enabled": true,
      "is_training": false,
      "last_retrain_time": "2025-11-28T15:30:00",
      "next_retrain_time": "2025-12-05T15:30:00",
      "interval_days": 7
    }
  }
}
```

### Trigger Retraining

```bash
POST /api/rl-agent/retrain
```

## Monitoring

### Logs

All retraining activities are logged:

```
2025-11-28 15:30:00 - INFO - 🔄 Starting automated retraining...
2025-11-28 15:30:05 - INFO - Step 1: Preparing training data...
2025-11-28 15:35:10 - INFO - ✅ Training data prepared
2025-11-28 15:35:15 - INFO - Step 2: Training model...
2025-11-28 16:20:30 - INFO - ✅ Model trained: models/rl_agent/checkpoint_20251128_153000.pt
2025-11-28 16:20:35 - INFO - Step 3: Deploying new model...
2025-11-28 16:20:40 - INFO - ✅ Model checkpoint_20251128_153000.pt validated successfully
2025-11-28 16:20:45 - INFO - ✅ Deployed model version 20251128_153000
2025-11-28 16:20:50 - INFO - ✅ Retraining complete! New model 20251128_153000 deployed
```

### Error Handling

If retraining fails:
```
2025-11-28 15:30:00 - ERROR - ❌ Training failed: ...
2025-11-28 15:30:05 - INFO - ⚠️ Keeping current model active
```

## Best Practices

### 1. Always Validate Before Deploy

The ModelManager automatically validates models, but you can also validate manually:

```python
from rl_agent.model_manager import ModelManager
from rl_agent.model import TradingActorCritic

manager = ModelManager()
is_valid, error = manager.validate_model(
    model_path=Path("models/rl_agent/checkpoint_new.pt"),
    model_class=TradingActorCritic,
    model_kwargs={...}
)
```

### 2. Monitor Retraining

Check status regularly:
```bash
curl http://localhost:5030/api/rl-agent/status
```

### 3. Manual Rollback

If needed, you can manually revert to a previous version:

```python
# Load a specific archived model
model = manager.load_current_model(...)
# Then deploy it using deploy_model()
```

### 4. Backup Important Models

Before major changes, backup the current model:
```bash
cp models/rl_agent/checkpoint_*.pt backup/
```

## Configuration

### Environment Variables

No additional environment variables needed - uses existing configuration.

### File Paths

All paths are configurable in `app.py`:

```python
rl_model_manager = ModelManager(
    model_dir="models/rl_agent",      # Active models
    archive_dir="models/rl_agent/archive",  # Archived models
    retention_days=30,                # Retention policy
)

rl_retraining_scheduler = RetrainingScheduler(
    model_manager=rl_model_manager,
    interval_days=7,                  # Weekly retraining
    enabled=True,                      # Enable/disable scheduler
)
```

## Troubleshooting

### Model Not Loading

1. Check if model file exists:
   ```bash
   ls -la models/rl_agent/checkpoint_*.pt
   ```

2. Check metadata:
   ```bash
   cat models/rl_agent/metadata.json
   ```

3. Validate model manually (see Best Practices)

### Retraining Not Running

1. Check scheduler status:
   ```bash
   curl http://localhost:5030/api/rl-agent/status
   ```

2. Check logs for errors

3. Verify scheduler is enabled:
   ```python
   rl_retraining_scheduler.enabled = True
   ```

### Old Models Not Being Cleaned

1. Check retention_days setting
2. Verify cleanup runs after deployment (check logs)
3. Manually trigger cleanup:
   ```python
   rl_model_manager.cleanup_old_models()
   ```

## Future Enhancements

Potential improvements:
- **A/B testing** - Compare model versions
- **Performance metrics** - Track model performance over time
- **Rollback automation** - Auto-rollback on performance degradation
- **Distributed training** - Support for multi-GPU training
- **Model compression** - Quantization for faster inference
- **Monitoring dashboard** - Visual model version tracking

