# Prometheus AI Engine Implementation Guide

> Detailed implementation guide for the VIDDHANA AI-powered prediction and optimization system

---

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Forecasting Models](#forecasting-models)
4. [Portfolio Optimization (RL)](#portfolio-optimization-rl)
5. [Data Pipeline](#data-pipeline)
6. [Inference API](#inference-api)
7. [Smart Contract Integration](#smart-contract-integration)
8. [Model Training & Deployment](#model-training--deployment)
9. [Testing & Validation](#testing--validation)

---

## Overview

Prometheus is the AI engine that powers VIDDHANA's intelligent wealth management. It combines:
- **LSTM + Transformer** models for price prediction
- **Reinforcement Learning (Q-Learning)** for portfolio optimization
- **Risk Assessment** models for user protection

### Core Formulas

**Time-Series Forecasting:**
$$y_{t+h} = f(y_{t-w:t}, X_{t-w:t}, C_t)$$

Where:
- $w$: Window length (30 days)
- $X_{t-w:t}$: On-chain data vector (volume, whale movements)
- $C_t$: Contextual factors (Inflation rate, Fed interest rate)

**Q-Learning Update:**
$$Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$$

Where:
- $s$ (State): Current financial profile
- $a$ (Action): Rebalancing decision (Buy/Sell)
- $r$ (Reward): Realized profit/loss
- $\gamma$: Discount factor (0.9)

---

## Architecture

### System Architecture

```
+------------------------------------------------------------------+
|                     PROMETHEUS AI ENGINE                          |
+------------------------------------------------------------------+
|                                                                   |
|  +-------------------+     +-------------------+                  |
|  |   Data Ingestion  |     |   Feature Store   |                  |
|  |   (Kafka/Redis)   |---->|   (PostgreSQL)    |                  |
|  +-------------------+     +-------------------+                  |
|           |                         |                             |
|           v                         v                             |
|  +-------------------+     +-------------------+                  |
|  |   Feature         |     |   Model Registry  |                  |
|  |   Engineering     |     |   (MLflow)        |                  |
|  +-------------------+     +-------------------+                  |
|           |                         |                             |
|           v                         v                             |
|  +-------------------+     +-------------------+                  |
|  |   Training        |     |   Inference       |                  |
|  |   Pipeline        |     |   Server          |                  |
|  |   (PyTorch)       |     |   (FastAPI)       |                  |
|  +-------------------+     +-------------------+                  |
|                                     |                             |
|                                     v                             |
|                        +-------------------+                      |
|                        |   Smart Contract  |                      |
|                        |   Interface       |                      |
|                        +-------------------+                      |
|                                                                   |
+------------------------------------------------------------------+
```

### Component Overview

| Component | Technology | Purpose |
|-----------|------------|---------|
| Data Ingestion | Apache Kafka | Real-time data streaming |
| Feature Store | PostgreSQL + Redis | Feature storage & caching |
| Feature Engineering | Pandas + NumPy | Data transformation |
| Model Training | PyTorch | LSTM/Transformer training |
| Model Registry | MLflow | Version control & deployment |
| Inference Server | FastAPI | Real-time predictions |
| Contract Interface | Web3.py | Blockchain interaction |

---

## Forecasting Models

### LSTM + Transformer Architecture

```python
# src/models/lstm_transformer.py
import torch
import torch.nn as nn
from typing import Tuple

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class LSTMTransformerForecaster(nn.Module):
    """
    Hybrid LSTM + Transformer model for price forecasting.
    
    Architecture:
    1. LSTM extracts sequential patterns
    2. Transformer captures long-range dependencies
    3. Regression head outputs price prediction
    """
    
    def __init__(
        self,
        input_dim: int = 64,        # Number of input features
        hidden_dim: int = 256,       # LSTM hidden dimension
        num_layers: int = 2,         # LSTM layers
        nhead: int = 8,              # Transformer attention heads
        transformer_layers: int = 4, # Transformer encoder layers
        dropout: float = 0.1,
        output_horizon: int = 7      # Days to predict ahead
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_horizon = output_horizon
        
        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        # Project LSTM output to transformer dimension
        self.lstm_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, transformer_layers)
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_horizon)
        )
        
        # Confidence estimation head
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, output_horizon),
            nn.Sigmoid()
        )
    
    def forward(
        self, 
        x: torch.Tensor,
        context: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            context: Optional context features (batch, context_dim)
        
        Returns:
            predictions: Price predictions (batch, output_horizon)
            confidence: Confidence scores (batch, output_horizon)
        """
        batch_size = x.size(0)
        
        # LSTM encoding
        lstm_out, (h_n, c_n) = self.lstm(x)
        lstm_out = self.lstm_proj(lstm_out)
        
        # Add positional encoding
        lstm_out = self.pos_encoder(lstm_out)
        
        # Transformer encoding
        transformer_out = self.transformer(lstm_out)
        
        # Use last token for prediction
        final_repr = transformer_out[:, -1, :]
        
        # Generate predictions and confidence
        predictions = self.output_head(final_repr)
        confidence = self.confidence_head(final_repr)
        
        return predictions, confidence


class PricePredictor:
    """High-level interface for price prediction."""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        self.model.eval()
        self.scaler = self._load_scaler(model_path)
    
    def _load_model(self, path: str) -> LSTMTransformerForecaster:
        model = LSTMTransformerForecaster()
        model.load_state_dict(torch.load(f"{path}/model.pt"))
        return model.to(self.device)
    
    def predict(
        self, 
        asset: str, 
        historical_data: np.ndarray,
        horizon: int = 7
    ) -> dict:
        """
        Generate price prediction.
        
        Args:
            asset: Asset symbol (e.g., "BTC")
            historical_data: Array of shape (window_size, features)
            horizon: Days ahead to predict
        
        Returns:
            Dictionary with predictions and metadata
        """
        # Preprocess
        scaled_data = self.scaler.transform(historical_data)
        x = torch.tensor(scaled_data, dtype=torch.float32).unsqueeze(0)
        x = x.to(self.device)
        
        # Predict
        with torch.no_grad():
            predictions, confidence = self.model(x)
        
        # Post-process
        predictions = predictions.cpu().numpy()[0][:horizon]
        confidence = confidence.cpu().numpy()[0][:horizon]
        
        # Inverse scale predictions
        predictions = self.scaler.inverse_transform_target(predictions)
        
        return {
            "asset": asset,
            "horizon_days": horizon,
            "predictions": predictions.tolist(),
            "confidence": confidence.tolist(),
            "timestamp": int(time.time())
        }
```

### Feature Engineering

```python
# src/pipelines/feature_engineering.py
import pandas as pd
import numpy as np
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    window_size: int = 30
    price_features: bool = True
    volume_features: bool = True
    onchain_features: bool = True
    macro_features: bool = True


class FeatureEngineer:
    """
    Feature engineering pipeline for Prometheus AI.
    
    Generates features from:
    1. Price data (OHLCV)
    2. On-chain metrics (volume, whale movements)
    3. Macro factors (inflation, interest rates)
    """
    
    def __init__(self, config: FeatureConfig = None):
        self.config = config or FeatureConfig()
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate all features from raw data."""
        features = pd.DataFrame(index=data.index)
        
        if self.config.price_features:
            features = pd.concat([
                features, 
                self._price_features(data)
            ], axis=1)
        
        if self.config.volume_features:
            features = pd.concat([
                features,
                self._volume_features(data)
            ], axis=1)
        
        if self.config.onchain_features:
            features = pd.concat([
                features,
                self._onchain_features(data)
            ], axis=1)
        
        if self.config.macro_features:
            features = pd.concat([
                features,
                self._macro_features(data)
            ], axis=1)
        
        return features.dropna()
    
    def _price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate price-based features."""
        df = pd.DataFrame(index=data.index)
        
        # Returns
        df['return_1d'] = data['close'].pct_change(1)
        df['return_7d'] = data['close'].pct_change(7)
        df['return_30d'] = data['close'].pct_change(30)
        
        # Moving averages
        df['sma_7'] = data['close'].rolling(7).mean() / data['close']
        df['sma_30'] = data['close'].rolling(30).mean() / data['close']
        df['ema_7'] = data['close'].ewm(span=7).mean() / data['close']
        
        # Volatility
        df['volatility_7d'] = data['close'].pct_change().rolling(7).std()
        df['volatility_30d'] = data['close'].pct_change().rolling(30).std()
        
        # RSI
        df['rsi_14'] = self._calculate_rsi(data['close'], 14)
        
        # MACD
        ema_12 = data['close'].ewm(span=12).mean()
        ema_26 = data['close'].ewm(span=26).mean()
        df['macd'] = (ema_12 - ema_26) / data['close']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        
        # Bollinger Bands
        sma_20 = data['close'].rolling(20).mean()
        std_20 = data['close'].rolling(20).std()
        df['bb_upper'] = (sma_20 + 2 * std_20) / data['close']
        df['bb_lower'] = (sma_20 - 2 * std_20) / data['close']
        df['bb_width'] = (df['bb_upper'] - df['bb_lower'])
        
        return df
    
    def _volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate volume-based features."""
        df = pd.DataFrame(index=data.index)
        
        df['volume_sma_7'] = data['volume'].rolling(7).mean() / data['volume']
        df['volume_sma_30'] = data['volume'].rolling(30).mean() / data['volume']
        df['volume_change'] = data['volume'].pct_change()
        
        # Volume-price correlation
        df['vp_corr_7d'] = data['close'].rolling(7).corr(data['volume'])
        
        return df
    
    def _onchain_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate on-chain features."""
        df = pd.DataFrame(index=data.index)
        
        if 'whale_transactions' in data.columns:
            df['whale_tx_sma'] = data['whale_transactions'].rolling(7).mean()
            df['whale_tx_change'] = data['whale_transactions'].pct_change()
        
        if 'active_addresses' in data.columns:
            df['active_addr_sma'] = data['active_addresses'].rolling(7).mean()
            df['active_addr_change'] = data['active_addresses'].pct_change()
        
        if 'exchange_inflow' in data.columns:
            df['exchange_netflow'] = (
                data['exchange_inflow'] - data.get('exchange_outflow', 0)
            )
        
        return df
    
    def _macro_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate macro-economic features."""
        df = pd.DataFrame(index=data.index)
        
        if 'inflation_rate' in data.columns:
            df['inflation'] = data['inflation_rate']
        
        if 'fed_rate' in data.columns:
            df['fed_rate'] = data['fed_rate']
            df['fed_rate_change'] = data['fed_rate'].diff()
        
        if 'dxy_index' in data.columns:
            df['dxy'] = data['dxy_index'].pct_change()
        
        return df
    
    @staticmethod
    def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
```

---

## Portfolio Optimization (RL)

### Q-Learning Agent

```python
# src/models/rl_optimizer.py
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum

class Action(Enum):
    """Portfolio actions."""
    HOLD = 0
    BUY_CONSERVATIVE = 1   # Buy stable assets
    BUY_MODERATE = 2       # Buy balanced mix
    BUY_AGGRESSIVE = 3     # Buy volatile assets
    SELL_PARTIAL = 4       # Sell 25%
    SELL_MAJOR = 5         # Sell 50%
    REBALANCE = 6          # Rebalance to target


@dataclass
class RiskProfile:
    """User risk profile."""
    risk_tolerance: float    # 0-1 scale
    time_to_goal: int        # months
    investment_amount: float
    monthly_contribution: float


@dataclass
class PortfolioState:
    """Current portfolio state."""
    total_value: float
    asset_allocation: Dict[str, float]
    unrealized_pnl: float
    volatility_30d: float
    sharpe_ratio: float
    market_regime: str  # "bull", "bear", "sideways"


class DQNNetwork(nn.Module):
    """Deep Q-Network for portfolio optimization."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class PortfolioOptimizer:
    """
    Reinforcement Learning agent for portfolio optimization.
    
    Uses Q-Learning update rule:
    Q(s,a) <- Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
    """
    
    def __init__(
        self,
        state_dim: int = 32,
        learning_rate: float = 0.001,
        gamma: float = 0.9,        # Discount factor
        epsilon: float = 0.1,       # Exploration rate
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.01,
        buffer_size: int = 100000,
        batch_size: int = 64
    ):
        self.state_dim = state_dim
        self.action_dim = len(Action)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size
        
        # Networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQNNetwork(state_dim, self.action_dim).to(self.device)
        self.target_net = DQNNetwork(state_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(), 
            lr=learning_rate
        )
        
        # Experience replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
    
    def encode_state(
        self, 
        portfolio: PortfolioState, 
        risk_profile: RiskProfile,
        market_data: Dict
    ) -> np.ndarray:
        """Encode portfolio and market state into feature vector."""
        
        state = []
        
        # Portfolio features
        state.extend([
            portfolio.total_value / 1_000_000,  # Normalize
            portfolio.unrealized_pnl / portfolio.total_value,
            portfolio.volatility_30d,
            portfolio.sharpe_ratio,
        ])
        
        # Asset allocation (top 10 assets)
        allocations = list(portfolio.asset_allocation.values())[:10]
        allocations += [0] * (10 - len(allocations))
        state.extend(allocations)
        
        # Risk profile features
        state.extend([
            risk_profile.risk_tolerance,
            risk_profile.time_to_goal / 120,  # Normalize to 10 years
            risk_profile.monthly_contribution / 10000,
        ])
        
        # Market features
        state.extend([
            market_data.get('btc_return_7d', 0),
            market_data.get('eth_return_7d', 0),
            market_data.get('market_volatility', 0),
            market_data.get('fear_greed_index', 50) / 100,
            1 if portfolio.market_regime == 'bull' else 0,
            1 if portfolio.market_regime == 'bear' else 0,
        ])
        
        # Pad to state_dim
        state += [0] * (self.state_dim - len(state))
        
        return np.array(state[:self.state_dim], dtype=np.float32)
    
    def select_action(self, state: np.ndarray, training: bool = False) -> Action:
        """Select action using epsilon-greedy policy."""
        
        if training and np.random.random() < self.epsilon:
            return Action(np.random.randint(self.action_dim))
        
        with torch.no_grad():
            state_tensor = torch.tensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            action_idx = q_values.argmax().item()
        
        return Action(action_idx)
    
    def calculate_reward(
        self,
        old_portfolio: PortfolioState,
        new_portfolio: PortfolioState,
        risk_profile: RiskProfile
    ) -> float:
        """Calculate reward based on risk-adjusted returns."""
        
        # Base reward: portfolio return
        portfolio_return = (
            (new_portfolio.total_value - old_portfolio.total_value) 
            / old_portfolio.total_value
        )
        
        # Risk penalty: penalize high volatility for conservative profiles
        volatility_penalty = (
            new_portfolio.volatility_30d * (1 - risk_profile.risk_tolerance)
        )
        
        # Goal proximity bonus: reward stability near goal date
        if risk_profile.time_to_goal <= 12:
            stability_bonus = 0.1 if new_portfolio.volatility_30d < 0.1 else -0.1
        else:
            stability_bonus = 0
        
        # Sharpe ratio bonus
        sharpe_bonus = new_portfolio.sharpe_ratio * 0.01
        
        reward = (
            portfolio_return * 100  # Scale returns
            - volatility_penalty * 10
            + stability_bonus
            + sharpe_bonus
        )
        
        return reward
    
    def train_step(self) -> float:
        """Perform one training step."""
        
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )
        
        states = torch.tensor(states).to(self.device)
        actions = torch.tensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.tensor(next_states).to(self.device)
        dones = torch.tensor(dones).unsqueeze(1).to(self.device)
        
        # Current Q values
        current_q = self.policy_net(states).gather(1, actions)
        
        # Target Q values
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # Loss and update
        loss = nn.MSELoss()(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def update_target_network(self):
        """Soft update target network."""
        tau = 0.005
        for target_param, policy_param in zip(
            self.target_net.parameters(),
            self.policy_net.parameters()
        ):
            target_param.data.copy_(
                tau * policy_param.data + (1 - tau) * target_param.data
            )
    
    def get_rebalance_recommendation(
        self,
        portfolio: PortfolioState,
        risk_profile: RiskProfile,
        market_data: Dict
    ) -> Dict:
        """Generate rebalancing recommendation."""
        
        state = self.encode_state(portfolio, risk_profile, market_data)
        action = self.select_action(state, training=False)
        
        # Map action to specific recommendations
        recommendations = self._action_to_recommendation(
            action, portfolio, risk_profile
        )
        
        return {
            "action": action.name,
            "recommendations": recommendations,
            "confidence": self._get_action_confidence(state),
            "risk_assessment": self._assess_risk(portfolio, risk_profile)
        }
    
    def _action_to_recommendation(
        self,
        action: Action,
        portfolio: PortfolioState,
        risk_profile: RiskProfile
    ) -> List[Dict]:
        """Convert action to specific trade recommendations."""
        
        recommendations = []
        
        if action == Action.BUY_CONSERVATIVE:
            recommendations.append({
                "asset": "USDC",
                "action": "BUY",
                "percentage": 30
            })
            recommendations.append({
                "asset": "ETH",
                "action": "BUY",
                "percentage": 20
            })
        
        elif action == Action.SELL_PARTIAL:
            # Sell most volatile assets
            volatile_assets = self._get_volatile_assets(portfolio)
            for asset in volatile_assets[:2]:
                recommendations.append({
                    "asset": asset,
                    "action": "SELL",
                    "percentage": 25
                })
        
        # ... additional action mappings
        
        return recommendations


class ReplayBuffer:
    """Experience replay buffer for DQN training."""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int):
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(
            *[self.buffer[i] for i in batch]
        )
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)
```

---

## Data Pipeline

### Data Ingestion Service

```python
# src/pipelines/data_ingestion.py
import asyncio
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from typing import Dict, Callable
import json
import logging

logger = logging.getLogger(__name__)


class DataIngestionService:
    """
    Real-time data ingestion from multiple sources.
    
    Sources:
    - Exchange APIs (Binance, Coinbase)
    - On-chain data (Ethereum, Arbitrum)
    - Macro data feeds
    """
    
    def __init__(self, kafka_brokers: str):
        self.kafka_brokers = kafka_brokers
        self.consumers: Dict[str, AIOKafkaConsumer] = {}
        self.producer: AIOKafkaProducer = None
        self.handlers: Dict[str, Callable] = {}
    
    async def start(self):
        """Initialize Kafka connections."""
        self.producer = AIOKafkaProducer(
            bootstrap_servers=self.kafka_brokers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        await self.producer.start()
        
        # Start consumers for each data source
        await self._start_consumer('price_data', self._handle_price_data)
        await self._start_consumer('onchain_data', self._handle_onchain_data)
        await self._start_consumer('macro_data', self._handle_macro_data)
    
    async def _start_consumer(self, topic: str, handler: Callable):
        """Start a Kafka consumer for a topic."""
        consumer = AIOKafkaConsumer(
            topic,
            bootstrap_servers=self.kafka_brokers,
            value_deserializer=lambda v: json.loads(v.decode('utf-8')),
            group_id='prometheus-ingestion'
        )
        await consumer.start()
        self.consumers[topic] = consumer
        self.handlers[topic] = handler
        
        # Start consuming in background
        asyncio.create_task(self._consume(topic))
    
    async def _consume(self, topic: str):
        """Consume messages from a topic."""
        consumer = self.consumers[topic]
        handler = self.handlers[topic]
        
        async for message in consumer:
            try:
                await handler(message.value)
            except Exception as e:
                logger.error(f"Error processing {topic} message: {e}")
    
    async def _handle_price_data(self, data: Dict):
        """Process incoming price data."""
        # Validate data
        required_fields = ['symbol', 'price', 'volume', 'timestamp']
        if not all(f in data for f in required_fields):
            return
        
        # Enrich with calculated fields
        enriched = {
            **data,
            'price_usd': data['price'],
            'processed_at': time.time()
        }
        
        # Store in feature store
        await self._store_features('price', enriched)
        
        # Publish to processed topic
        await self.producer.send('processed_price_data', enriched)
    
    async def _handle_onchain_data(self, data: Dict):
        """Process incoming on-chain data."""
        enriched = {
            **data,
            'chain': data.get('chain', 'ethereum'),
            'processed_at': time.time()
        }
        
        await self._store_features('onchain', enriched)
        await self.producer.send('processed_onchain_data', enriched)
    
    async def _handle_macro_data(self, data: Dict):
        """Process incoming macro-economic data."""
        enriched = {
            **data,
            'source': data.get('source', 'unknown'),
            'processed_at': time.time()
        }
        
        await self._store_features('macro', enriched)
        await self.producer.send('processed_macro_data', enriched)
    
    async def _store_features(self, feature_type: str, data: Dict):
        """Store features in feature store (Redis + PostgreSQL)."""
        # Implementation depends on feature store choice
        pass
    
    async def stop(self):
        """Gracefully shutdown."""
        for consumer in self.consumers.values():
            await consumer.stop()
        if self.producer:
            await self.producer.stop()
```

---

## Inference API

### FastAPI Server

```python
# src/api/inference_server.py
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import logging

app = FastAPI(
    title="Prometheus AI API",
    description="AI-powered predictions for VIDDHANA",
    version="1.0.0"
)

logger = logging.getLogger(__name__)


# Request/Response Models
class PricePredictionRequest(BaseModel):
    asset: str = Field(..., description="Asset symbol (e.g., BTC, ETH)")
    horizon: int = Field(7, ge=1, le=30, description="Days to predict ahead")


class PricePredictionResponse(BaseModel):
    asset: str
    horizon_days: int
    predictions: List[float]
    confidence: List[float]
    timestamp: int


class PortfolioOptimizationRequest(BaseModel):
    user_id: str
    portfolio: Dict[str, float]
    risk_tolerance: float = Field(..., ge=0, le=1)
    time_to_goal: int = Field(..., ge=1, description="Months to goal")


class RebalanceRecommendation(BaseModel):
    asset: str
    action: str
    percentage: float


class PortfolioOptimizationResponse(BaseModel):
    action: str
    recommendations: List[RebalanceRecommendation]
    confidence: float
    risk_assessment: Dict


class RiskAssessmentRequest(BaseModel):
    portfolio: Dict[str, float]


class RiskAssessmentResponse(BaseModel):
    risk_score: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    recommendations: List[str]


# Dependency injection for models
def get_price_predictor():
    from src.models.lstm_transformer import PricePredictor
    return PricePredictor(model_path="/models/price_predictor")


def get_portfolio_optimizer():
    from src.models.rl_optimizer import PortfolioOptimizer
    return PortfolioOptimizer()


# Endpoints
@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}


@app.post("/v1/predict/price", response_model=PricePredictionResponse)
async def predict_price(
    request: PricePredictionRequest,
    predictor: PricePredictor = Depends(get_price_predictor)
):
    """
    Generate price prediction for an asset.
    
    Uses LSTM + Transformer model trained on historical data.
    """
    try:
        # Fetch historical data
        historical_data = await fetch_historical_data(
            request.asset, 
            window_size=30
        )
        
        # Generate prediction
        result = predictor.predict(
            asset=request.asset,
            historical_data=historical_data,
            horizon=request.horizon
        )
        
        return PricePredictionResponse(**result)
    
    except Exception as e:
        logger.error(f"Price prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/optimize/portfolio", response_model=PortfolioOptimizationResponse)
async def optimize_portfolio(
    request: PortfolioOptimizationRequest,
    optimizer: PortfolioOptimizer = Depends(get_portfolio_optimizer)
):
    """
    Get portfolio rebalancing recommendations.
    
    Uses Q-Learning agent trained on historical portfolio performance.
    """
    try:
        # Build portfolio state
        portfolio_state = await build_portfolio_state(
            request.user_id,
            request.portfolio
        )
        
        # Build risk profile
        risk_profile = RiskProfile(
            risk_tolerance=request.risk_tolerance,
            time_to_goal=request.time_to_goal,
            investment_amount=sum(request.portfolio.values()),
            monthly_contribution=0  # Fetch from user profile
        )
        
        # Fetch market data
        market_data = await fetch_market_data()
        
        # Get recommendation
        result = optimizer.get_rebalance_recommendation(
            portfolio_state,
            risk_profile,
            market_data
        )
        
        return PortfolioOptimizationResponse(**result)
    
    except Exception as e:
        logger.error(f"Portfolio optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/assess/risk", response_model=RiskAssessmentResponse)
async def assess_risk(request: RiskAssessmentRequest):
    """
    Assess portfolio risk metrics.
    """
    try:
        # Calculate risk metrics
        risk_metrics = calculate_risk_metrics(request.portfolio)
        
        return RiskAssessmentResponse(
            risk_score=risk_metrics['risk_score'],
            volatility=risk_metrics['volatility'],
            sharpe_ratio=risk_metrics['sharpe_ratio'],
            max_drawdown=risk_metrics['max_drawdown'],
            recommendations=risk_metrics['recommendations']
        )
    
    except Exception as e:
        logger.error(f"Risk assessment failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# WebSocket for real-time updates
from fastapi import WebSocket

@app.websocket("/ws/predictions/{asset}")
async def websocket_predictions(websocket: WebSocket, asset: str):
    """Stream real-time predictions for an asset."""
    await websocket.accept()
    
    try:
        predictor = get_price_predictor()
        
        while True:
            # Fetch latest data
            historical_data = await fetch_historical_data(asset, 30)
            
            # Generate prediction
            result = predictor.predict(asset, historical_data, horizon=1)
            
            # Send to client
            await websocket.send_json(result)
            
            # Wait before next update
            await asyncio.sleep(60)  # Update every minute
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()
```

---

## Smart Contract Integration

### AI-Contract Interface

```python
# src/integration/contract_interface.py
from web3 import Web3
from typing import Dict, Optional
import json
import logging

logger = logging.getLogger(__name__)


class ContractInterface:
    """
    Interface between Prometheus AI and Atlas Chain smart contracts.
    
    Responsibilities:
    - Submit AI recommendations to PolicyEngine
    - Read portfolio state from VaultManager
    - Trigger auto-rebalancing when conditions are met
    """
    
    def __init__(
        self,
        rpc_url: str,
        private_key: str,
        contract_addresses: Dict[str, str]
    ):
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        self.account = self.w3.eth.account.from_key(private_key)
        self.contracts = self._load_contracts(contract_addresses)
    
    def _load_contracts(self, addresses: Dict[str, str]) -> Dict:
        """Load contract ABIs and create contract instances."""
        contracts = {}
        
        for name, address in addresses.items():
            abi = self._load_abi(name)
            contracts[name] = self.w3.eth.contract(
                address=Web3.to_checksum_address(address),
                abi=abi
            )
        
        return contracts
    
    def _load_abi(self, contract_name: str) -> list:
        """Load contract ABI from file."""
        with open(f"abis/{contract_name}.json") as f:
            return json.load(f)
    
    async def get_user_portfolio(self, user_address: str) -> Dict:
        """Fetch user's portfolio from VaultManager."""
        vault = self.contracts['VaultManager']
        
        # Call view function
        portfolio_data = vault.functions.getUserPortfolio(
            Web3.to_checksum_address(user_address)
        ).call()
        
        return {
            'total_value': portfolio_data[0],
            'assets': portfolio_data[1],
            'allocations': portfolio_data[2]
        }
    
    async def submit_rebalance_recommendation(
        self,
        user_address: str,
        recommendations: list,
        confidence: float
    ) -> str:
        """Submit AI recommendation to PolicyEngine."""
        policy_engine = self.contracts['PolicyEngine']
        
        # Encode recommendations
        encoded_recommendations = self._encode_recommendations(recommendations)
        
        # Build transaction
        tx = policy_engine.functions.submitAIRecommendation(
            Web3.to_checksum_address(user_address),
            encoded_recommendations,
            int(confidence * 10000)  # Scale to basis points
        ).build_transaction({
            'from': self.account.address,
            'nonce': self.w3.eth.get_transaction_count(self.account.address),
            'gas': 500000,
            'gasPrice': self.w3.eth.gas_price
        })
        
        # Sign and send
        signed_tx = self.w3.eth.account.sign_transaction(tx, self.account.key)
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        
        logger.info(f"Submitted rebalance recommendation: {tx_hash.hex()}")
        
        return tx_hash.hex()
    
    async def trigger_auto_rebalance(
        self,
        user_address: str,
        inflation_rate: float
    ) -> Optional[str]:
        """Trigger auto-rebalancing for a user."""
        policy_engine = self.contracts['PolicyEngine']
        
        # Check if rebalancing is allowed
        can_rebalance = policy_engine.functions.canRebalance(
            Web3.to_checksum_address(user_address)
        ).call()
        
        if not can_rebalance:
            logger.info(f"Rebalancing not allowed for {user_address}")
            return None
        
        # Execute rebalancing
        tx = policy_engine.functions.autoRebalance(
            Web3.to_checksum_address(user_address),
            int(inflation_rate * 100)  # Scale to basis points
        ).build_transaction({
            'from': self.account.address,
            'nonce': self.w3.eth.get_transaction_count(self.account.address),
            'gas': 1000000,
            'gasPrice': self.w3.eth.gas_price
        })
        
        signed_tx = self.w3.eth.account.sign_transaction(tx, self.account.key)
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        
        logger.info(f"Triggered auto-rebalance: {tx_hash.hex()}")
        
        return tx_hash.hex()
    
    def _encode_recommendations(self, recommendations: list) -> bytes:
        """Encode recommendations for smart contract."""
        # ABI encode the recommendations
        return self.w3.codec.encode(
            ['tuple(address,uint8,uint256)[]'],
            [[(r['asset'], r['action'], r['amount']) for r in recommendations]]
        )
```

---

## Model Training & Deployment

### Training Pipeline

```python
# src/training/train_forecaster.py
import torch
from torch.utils.data import DataLoader
import mlflow
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class ForecastingTrainer:
    """Training pipeline for LSTM-Transformer forecasting model."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config['learning_rate'],
            epochs=config['epochs'],
            steps_per_epoch=len(train_loader)
        )
        
        self.criterion = nn.MSELoss()
    
    def train(self) -> Dict:
        """Run full training loop."""
        
        mlflow.start_run()
        mlflow.log_params(self.config)
        
        best_val_loss = float('inf')
        
        for epoch in range(self.config['epochs']):
            # Training phase
            train_loss = self._train_epoch()
            
            # Validation phase
            val_loss, val_metrics = self._validate()
            
            # Log metrics
            mlflow.log_metrics({
                'train_loss': train_loss,
                'val_loss': val_loss,
                **val_metrics
            }, step=epoch)
            
            logger.info(
                f"Epoch {epoch}: train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, mae={val_metrics['mae']:.4f}"
            )
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint('best_model.pt')
                mlflow.pytorch.log_model(self.model, "model")
        
        mlflow.end_run()
        
        return {
            'best_val_loss': best_val_loss,
            'final_metrics': val_metrics
        }
    
    def _train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for batch in self.train_loader:
            x, y = batch
            x, y = x.to(self.device), y.to(self.device)
            
            self.optimizer.zero_grad()
            
            predictions, _ = self.model(x)
            loss = self.criterion(predictions, y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def _validate(self) -> tuple:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)
                
                predictions, _ = self.model(x)
                loss = self.criterion(predictions, y)
                
                total_loss += loss.item()
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(y.cpu().numpy())
        
        # Calculate metrics
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        
        mae = np.mean(np.abs(predictions - targets))
        mape = np.mean(np.abs((targets - predictions) / targets)) * 100
        
        metrics = {
            'mae': mae,
            'mape': mape,
            'rmse': np.sqrt(np.mean((predictions - targets) ** 2))
        }
        
        return total_loss / len(self.val_loader), metrics
    
    def _save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, f"checkpoints/{filename}")
```

### Model Deployment

```yaml
# kubernetes/prometheus-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prometheus-ai
  labels:
    app: prometheus-ai
spec:
  replicas: 3
  selector:
    matchLabels:
      app: prometheus-ai
  template:
    metadata:
      labels:
        app: prometheus-ai
    spec:
      containers:
      - name: prometheus-api
        image: viddhana/prometheus-ai:latest
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_PATH
          value: "/models"
        - name: MLFLOW_TRACKING_URI
          valueFrom:
            secretKeyRef:
              name: prometheus-secrets
              key: mlflow-uri
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
            nvidia.com/gpu: "1"
          limits:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: "1"
        volumeMounts:
        - name: models
          mountPath: /models
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: prometheus-models-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: prometheus-ai-service
spec:
  selector:
    app: prometheus-ai
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP
```

---

## Testing & Validation

### Unit Tests

```python
# tests/test_forecaster.py
import pytest
import torch
import numpy as np
from src.models.lstm_transformer import LSTMTransformerForecaster, PricePredictor


class TestLSTMTransformer:
    @pytest.fixture
    def model(self):
        return LSTMTransformerForecaster(
            input_dim=64,
            hidden_dim=128,
            output_horizon=7
        )
    
    def test_forward_shape(self, model):
        batch_size = 32
        seq_len = 30
        input_dim = 64
        
        x = torch.randn(batch_size, seq_len, input_dim)
        predictions, confidence = model(x)
        
        assert predictions.shape == (batch_size, 7)
        assert confidence.shape == (batch_size, 7)
    
    def test_confidence_range(self, model):
        x = torch.randn(16, 30, 64)
        _, confidence = model(x)
        
        assert (confidence >= 0).all()
        assert (confidence <= 1).all()
    
    def test_gradient_flow(self, model):
        x = torch.randn(8, 30, 64, requires_grad=True)
        predictions, _ = model(x)
        
        loss = predictions.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestPortfolioOptimizer:
    @pytest.fixture
    def optimizer(self):
        from src.models.rl_optimizer import PortfolioOptimizer
        return PortfolioOptimizer(state_dim=32)
    
    def test_action_selection(self, optimizer):
        state = np.random.randn(32).astype(np.float32)
        action = optimizer.select_action(state, training=False)
        
        assert action.value in range(7)  # 7 possible actions
    
    def test_reward_calculation(self, optimizer):
        from src.models.rl_optimizer import PortfolioState, RiskProfile
        
        old_portfolio = PortfolioState(
            total_value=100000,
            asset_allocation={'BTC': 0.5, 'ETH': 0.3, 'USDC': 0.2},
            unrealized_pnl=0,
            volatility_30d=0.2,
            sharpe_ratio=1.5,
            market_regime='bull'
        )
        
        new_portfolio = PortfolioState(
            total_value=105000,
            asset_allocation={'BTC': 0.4, 'ETH': 0.3, 'USDC': 0.3},
            unrealized_pnl=5000,
            volatility_30d=0.15,
            sharpe_ratio=1.8,
            market_regime='bull'
        )
        
        risk_profile = RiskProfile(
            risk_tolerance=0.5,
            time_to_goal=24,
            investment_amount=100000,
            monthly_contribution=1000
        )
        
        reward = optimizer.calculate_reward(
            old_portfolio, new_portfolio, risk_profile
        )
        
        # Positive return should give positive reward
        assert reward > 0
```

### Integration Tests

```python
# tests/test_integration.py
import pytest
import asyncio
from httpx import AsyncClient
from src.api.inference_server import app


@pytest.fixture
def client():
    return AsyncClient(app=app, base_url="http://test")


@pytest.mark.asyncio
async def test_price_prediction_endpoint(client):
    response = await client.post(
        "/v1/predict/price",
        json={"asset": "BTC", "horizon": 7}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert data['asset'] == 'BTC'
    assert len(data['predictions']) == 7
    assert len(data['confidence']) == 7
    assert all(0 <= c <= 1 for c in data['confidence'])


@pytest.mark.asyncio
async def test_portfolio_optimization_endpoint(client):
    response = await client.post(
        "/v1/optimize/portfolio",
        json={
            "user_id": "test_user",
            "portfolio": {"BTC": 50000, "ETH": 30000, "USDC": 20000},
            "risk_tolerance": 0.5,
            "time_to_goal": 24
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert 'action' in data
    assert 'recommendations' in data
    assert 'confidence' in data


@pytest.mark.asyncio
async def test_risk_assessment_endpoint(client):
    response = await client.post(
        "/v1/assess/risk",
        json={
            "portfolio": {"BTC": 50000, "ETH": 30000, "USDC": 20000}
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert 'risk_score' in data
    assert 'volatility' in data
    assert 'sharpe_ratio' in data
```

### Acceptance Criteria

```markdown
## Prometheus AI Acceptance Criteria

### Model Performance
- [ ] LSTM model MAE < 5% on validation set
- [ ] MAPE < 10% for 7-day predictions
- [ ] RL agent achieves positive Sharpe ratio in backtests
- [ ] Backtesting shows 80%+ win rate for recommendations

### API Performance
- [ ] Inference latency < 100ms (p99)
- [ ] Throughput > 1000 predictions/second
- [ ] 99.9% API uptime over 30 days

### Integration
- [ ] Successfully reads portfolio data from VaultManager
- [ ] Successfully submits recommendations to PolicyEngine
- [ ] Auto-rebalancing triggers correctly based on conditions

### Security
- [ ] API authentication implemented
- [ ] Rate limiting in place
- [ ] Model inputs validated and sanitized
- [ ] No sensitive data in logs
```

---

## Next Steps

After completing Prometheus AI setup:
1. Proceed to `04_SMART_CONTRACTS.md` for contract integration
2. Configure model training pipelines
3. Update `TRACKER.md` with completion status

---

*Document Version: 1.0.0*
