# External Markets (BTC & S&P 500) Implementation

## Status: Infrastructure Ready, NOT Used in Training Yet

### ⚠️ Important Limitation

**We do NOT have historical BTC and S&P 500 data aligned with our SOL price history.**

- Current fetcher only gets **current prices** (not historical)
- Training requires **historical data** aligned with SOL timestamps
- **Training currently uses zeros** for external market features (model learns to ignore them)

### ✅ Completed (Infrastructure Only)

1. **Documentation**: Created `TRAINING_DATA_SPECIFICATION.md` with complete feature documentation
2. **External Markets Fetcher**: Created `external_markets.py` to fetch **current** BTC and S&P 500 prices
3. **State Encoder**: Updated to encode BTC and S&P 500 returns (will pad with zeros if unavailable)
4. **Training Script**: Updated to pass external market prices (currently empty lists → zeros)

### ❌ NOT Used in Training

- **Training Data Prep**: Does NOT fetch external markets (no historical data available)
- **Training Episodes**: External market features are **zeros** during training
- **Model**: Will learn to ignore external market features (they're all zeros)

### ⚠️ Still Needed (When Historical Data Available)

1. **Historical Data Source**: Need historical BTC and S&P 500 prices aligned with SOL timestamps
2. **Backfill Script**: Script to fetch/import historical external market data
3. **Model Architecture**: The `TradingActorCritic` model needs to accept `btc_returns` and `sp500_returns` inputs
4. **Model Forward Pass**: Update model's `forward()` method to process external market data
5. **Integration**: Update `rl_agent/integration.py` to fetch external market prices during inference

## How to Complete

### Step 1: Update Model Architecture

In `rl_agent/model.py`, update `TradingActorCritic.__init__()`:

```python
def __init__(
    self,
    # ... existing params ...
    include_external_markets: bool = True,  # NEW
):
    # ... existing code ...
    
    if include_external_markets:
        # External market branch (BTC + S&P 500)
        self.external_market_branch = nn.Sequential(
            nn.Linear(60, 32),  # 30 BTC + 30 S&P 500 returns
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        
        # Update market_latent_dim to include external markets
        market_latent_dim = 256 + 32  # existing + external markets
        self.market_latent = nn.Sequential(
            nn.Linear(market_latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
```

### Step 2: Update Forward Pass

In `TradingActorCritic.forward()`, add:

```python
def forward(
    self,
    # ... existing inputs ...
    btc_returns: Optional[torch.Tensor] = None,  # NEW: (batch_size, 30)
    sp500_returns: Optional[torch.Tensor] = None,  # NEW: (batch_size, 30)
):
    # ... existing code ...
    
    # External market branch (NEW)
    if btc_returns is not None and sp500_returns is not None:
        external_returns = torch.cat([btc_returns, sp500_returns], dim=1)  # (batch, 60)
        external_latent = self.external_market_branch(external_returns)  # (batch, 32)
    else:
        external_latent = torch.zeros(batch_size, 32, device=price_features.device)
    
    # Combine with market latent
    market_latent_input = torch.cat([market_latent_output, external_latent], dim=1)
    market_latent = self.market_latent(market_latent_input)
    
    # ... rest of forward pass ...
```

### Step 3: Update Integration

In `rl_agent/integration.py`, update `make_decision()` to fetch external market prices:

```python
def make_decision(self):
    # ... existing code ...
    
    # Fetch external market prices (NEW)
    from external_markets import get_recent_prices, EXTERNAL_MARKETS_DB
    btc_data = get_recent_prices(EXTERNAL_MARKETS_DB, "btc_prices", hours=2.5)
    sp500_data = get_recent_prices(EXTERNAL_MARKETS_DB, "sp500_prices", hours=2.5)
    
    btc_prices = [p for _, p in btc_data]
    sp500_prices = [p for _, p in sp500_data]
    
    # Encode state with external markets
    state_dict = self.state_encoder.encode_full_state(
        # ... existing params ...
        btc_prices=btc_prices,
        sp500_prices=sp500_prices,
    )
    
    # ... rest of decision making ...
```

## API Keys Needed

### Alpha Vantage (S&P 500)
- **Free tier**: 5 API calls per minute, 500 calls per day
- **Get key**: https://www.alphavantage.co/support/#api-key
- **Set in `.env`**: `ALPHA_VANTAGE_API_KEY=your_key_here`

### CoinGecko (BTC)
- **No API key needed** for basic price data
- **Rate limit**: 10-50 calls per minute (depends on plan)

## Data Fetching

Run periodically to fetch external market prices:

```bash
# Fetch current prices (can be added to cron)
python external_markets.py
```

Or integrate into your price fetcher loop.

## Benefits

1. **BTC Correlation**: SOL often follows BTC (correlation ~0.7-0.9)
2. **Risk-On/Risk-Off**: S&P 500 indicates market sentiment
3. **Leading Indicators**: External markets often move before SOL
4. **Macro Context**: Better understanding of broader market conditions

## Feature Dimensions

- **BTC Returns**: 30 features (2.5 hours of 5-min returns)
- **S&P 500 Returns**: 30 features (2.5 hours of 5-min returns)
- **Total New Features**: 60

## Notes

- External market data is optional - model will work with zeros if unavailable
- Historical data can be backfilled using the fetcher
- Consider adding correlation features (SOL/BTC ratio, etc.) in future
