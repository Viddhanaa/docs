# Prometheus AI Engine

AI-powered prediction and portfolio optimization engine for VIDDHANA.

## Overview

Prometheus is the core AI engine that powers VIDDHANA's intelligent wealth management. It provides:

- **LSTM + Transformer Models**: Price prediction using deep learning
- **Q-Learning Portfolio Optimizer**: Reinforcement learning for portfolio rebalancing
- **Technical Analysis**: RSI, MACD, Bollinger Bands, and more
- **Feature Engineering**: Comprehensive data transformation pipeline
- **FastAPI Server**: Production-ready inference API

## Architecture

```
prometheus-ai/
├── src/prometheus/
│   ├── models/
│   │   ├── lstm_predictor.py      # LSTM price prediction
│   │   ├── transformer_model.py   # Transformer forecaster
│   │   └── q_learning.py          # DQN portfolio optimizer
│   ├── features/
│   │   ├── technical_indicators.py # RSI, MACD, Bollinger
│   │   └── pipeline.py            # Feature engineering
│   ├── inference/
│   │   └── server.py              # FastAPI server
│   └── training/
│       └── trainer.py             # Training pipelines
├── tests/
│   └── test_models.py             # Pytest tests
├── Dockerfile
├── pyproject.toml
└── requirements.txt
```

## Installation

### From PyPI (when published)

```bash
pip install prometheus-ai
```

### From Source

```bash
git clone https://github.com/viddhana/prometheus-ai.git
cd prometheus-ai
pip install -e ".[dev]"
```

### Using Docker

```bash
# Build image
docker build -t prometheus-ai .

# Run container
docker run -p 8000:8000 prometheus-ai
```

## Quick Start

### Price Prediction

```python
from prometheus.models.lstm_predictor import LSTMPredictor
import numpy as np

# Initialize model
model = LSTMPredictor(
    input_dim=64,
    hidden_dim=256,
    output_horizon=7
)

# Generate predictions
historical_data = np.random.randn(30, 64).astype(np.float32)
predictions, confidence = model.predict(historical_data)

print(f"7-day predictions: {predictions}")
print(f"Confidence scores: {confidence}")
```

### Portfolio Optimization

```python
from prometheus.models.q_learning import (
    PortfolioOptimizer,
    PortfolioState,
    RiskProfile
)

# Initialize optimizer
optimizer = PortfolioOptimizer(state_dim=32)

# Define portfolio and risk profile
portfolio = PortfolioState(
    total_value=100000,
    asset_allocation={"BTC": 0.5, "ETH": 0.3, "USDC": 0.2},
    unrealized_pnl=0,
    volatility_30d=0.2,
    sharpe_ratio=1.5,
    market_regime="sideways"
)

risk_profile = RiskProfile(
    risk_tolerance=0.5,
    time_to_goal=24,
    investment_amount=100000,
    monthly_contribution=1000
)

# Get rebalancing recommendation
market_data = {"btc_return_7d": 0.02, "fear_greed_index": 55}
result = optimizer.get_rebalance_recommendation(
    portfolio, risk_profile, market_data
)

print(f"Action: {result['action']}")
print(f"Recommendations: {result['recommendations']}")
```

### Technical Indicators

```python
import pandas as pd
from prometheus.features.technical_indicators import TechnicalIndicators

# Load price data
data = pd.read_csv("prices.csv")

# Calculate indicators
indicators = TechnicalIndicators()
rsi = indicators.calculate_rsi(data["close"], period=14)
macd, signal, hist = indicators.calculate_macd(data["close"])
upper, middle, lower = indicators.calculate_bollinger_bands(data["close"])

# Calculate all at once
all_indicators = indicators.calculate_all(data)
```

## API Endpoints

Start the server:

```bash
prometheus-server
# or
python -m prometheus.inference.server
```

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/predict` | POST | Price prediction |
| `/optimize` | POST | Portfolio optimization |
| `/risk` | POST | Risk assessment |
| `/ws/predictions/{asset}` | WebSocket | Real-time predictions |

### Example Requests

**Price Prediction:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"asset": "BTC", "horizon": 7}'
```

**Portfolio Optimization:**

```bash
curl -X POST http://localhost:8000/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "portfolio": {"BTC": 50000, "ETH": 30000, "USDC": 20000},
    "risk_tolerance": 0.5,
    "time_to_goal": 24
  }'
```

**Risk Assessment:**

```bash
curl -X POST http://localhost:8000/risk \
  -H "Content-Type: application/json" \
  -d '{"portfolio": {"BTC": 50000, "ETH": 30000}}'
```

## Training

### Train Price Predictor

```python
from prometheus.training.trainer import ModelTrainer, TrainingConfig
from prometheus.models.lstm_predictor import LSTMPredictor
from torch.utils.data import DataLoader

# Configure training
config = TrainingConfig(
    epochs=100,
    batch_size=64,
    learning_rate=0.001,
    early_stopping=True,
    patience=10
)

# Initialize model and trainer
model = LSTMPredictor(input_dim=64, hidden_dim=256)
trainer = ModelTrainer(model, config)

# Train
results = trainer.train(train_loader, val_loader)
print(f"Best validation loss: {results['best_val_loss']}")
```

### Train RL Agent

```python
from prometheus.training.trainer import RLTrainer
from prometheus.models.q_learning import PortfolioOptimizer

optimizer = PortfolioOptimizer(state_dim=32)
rl_trainer = RLTrainer(optimizer)

results = rl_trainer.train(episodes=1000)
print(f"Average reward: {results['avg_reward']}")
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=prometheus --cov-report=html

# Run specific test
pytest tests/test_models.py::TestLSTMPredictor -v
```

## Configuration

Environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_PATH` | Path to model weights | `/app/models` |
| `PROMETHEUS_ENV` | Environment mode | `development` |
| `CUDA_VISIBLE_DEVICES` | GPU devices to use | `0` |

## Performance

### Model Performance (Target)

| Metric | Target |
|--------|--------|
| LSTM MAE | < 5% |
| MAPE | < 10% |
| RL Win Rate | > 80% |

### API Performance

| Metric | Target |
|--------|--------|
| Inference Latency (p99) | < 100ms |
| Throughput | > 1000 req/s |
| Uptime | 99.9% |

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## Related

- [VIDDHANA Documentation](https://docs.viddhana.io)
- [Atlas Chain](https://github.com/viddhana/atlas-chain)
- [Vault Manager](https://github.com/viddhana/vault-manager)
