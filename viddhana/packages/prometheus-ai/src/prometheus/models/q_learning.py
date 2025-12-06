"""
Q-Learning based portfolio optimizer for reinforcement learning.

This module implements a Deep Q-Network (DQN) agent for portfolio
optimization using the Q-Learning update rule:
Q(s,a) <- Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Action(Enum):
    """Portfolio actions for the RL agent."""
    
    HOLD = 0
    BUY_CONSERVATIVE = 1    # Buy stable assets (stablecoins, blue chips)
    BUY_MODERATE = 2        # Buy balanced mix
    BUY_AGGRESSIVE = 3      # Buy volatile/growth assets
    SELL_PARTIAL = 4        # Sell 25% of holdings
    SELL_MAJOR = 5          # Sell 50% of holdings
    REBALANCE = 6           # Rebalance to target allocation


@dataclass
class RiskProfile:
    """User's risk profile configuration."""
    
    risk_tolerance: float       # 0.0 (conservative) to 1.0 (aggressive)
    time_to_goal: int          # Months until investment goal
    investment_amount: float    # Total investment amount in USD
    monthly_contribution: float # Monthly contribution amount
    
    def __post_init__(self) -> None:
        """Validate risk profile parameters."""
        if not 0 <= self.risk_tolerance <= 1:
            raise ValueError("risk_tolerance must be between 0 and 1")
        if self.time_to_goal < 1:
            raise ValueError("time_to_goal must be at least 1 month")


@dataclass
class PortfolioState:
    """Current state of a portfolio."""
    
    total_value: float                      # Total portfolio value in USD
    asset_allocation: Dict[str, float]      # Asset -> allocation percentage
    unrealized_pnl: float                   # Unrealized profit/loss
    volatility_30d: float                   # 30-day volatility
    sharpe_ratio: float                     # Risk-adjusted return metric
    market_regime: str                      # "bull", "bear", or "sideways"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "total_value": self.total_value,
            "asset_allocation": self.asset_allocation,
            "unrealized_pnl": self.unrealized_pnl,
            "volatility_30d": self.volatility_30d,
            "sharpe_ratio": self.sharpe_ratio,
            "market_regime": self.market_regime,
        }


class DQNNetwork(nn.Module):
    """
    Deep Q-Network for portfolio optimization.
    
    Architecture uses dueling DQN with value and advantage streams
    for more stable learning.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
    ) -> None:
        """
        Initialize DQN network.
        
        Args:
            state_dim: Dimension of state space.
            action_dim: Number of possible actions.
            hidden_dim: Hidden layer dimension.
        """
        super().__init__()
        
        # Shared feature extractor
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        
        # Value stream (state value)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # Advantage stream (action advantage)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computing Q-values for all actions.
        
        Uses dueling architecture: Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
        
        Args:
            state: State tensor of shape (batch_size, state_dim).
            
        Returns:
            Q-values of shape (batch_size, action_dim).
        """
        features = self.feature_layer(state)
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Dueling DQN: Q = V + A - mean(A)
        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        
        return q_values


class ReplayBuffer:
    """
    Experience replay buffer for DQN training.
    
    Stores transitions and allows random sampling for breaking
    temporal correlations in training data.
    """
    
    def __init__(self, capacity: int) -> None:
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store.
        """
        self.capacity = capacity
        self.buffer: List[Optional[Tuple]] = []
        self.position = 0
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Add a transition to the buffer.
        
        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Next state.
            done: Whether episode terminated.
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(
        self,
        batch_size: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample.
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones).
        """
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)


class PortfolioOptimizer:
    """
    Reinforcement Learning agent for portfolio optimization.
    
    Implements Deep Q-Learning with:
    - Dueling DQN architecture
    - Experience replay
    - Target network for stability
    - Epsilon-greedy exploration
    
    Q-Learning update:
    Q(s,a) <- Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
    """
    
    def __init__(
        self,
        state_dim: int = 32,
        learning_rate: float = 0.001,
        gamma: float = 0.9,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.01,
        buffer_size: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 100,
    ) -> None:
        """
        Initialize portfolio optimizer.
        
        Args:
            state_dim: Dimension of state space.
            learning_rate: Learning rate for optimizer.
            gamma: Discount factor for future rewards.
            epsilon: Initial exploration rate.
            epsilon_decay: Epsilon decay per episode.
            min_epsilon: Minimum exploration rate.
            buffer_size: Replay buffer capacity.
            batch_size: Training batch size.
            target_update_freq: Steps between target network updates.
        """
        self.state_dim = state_dim
        self.action_dim = len(Action)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.step_count = 0
        
        # Device selection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.policy_net = DQNNetwork(state_dim, self.action_dim).to(self.device)
        self.target_net = DQNNetwork(state_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=learning_rate,
        )
        
        # Experience replay
        self.replay_buffer = ReplayBuffer(buffer_size)
    
    def encode_state(
        self,
        portfolio: PortfolioState,
        risk_profile: RiskProfile,
        market_data: Dict[str, float],
    ) -> np.ndarray:
        """
        Encode portfolio and market state into feature vector.
        
        Args:
            portfolio: Current portfolio state.
            risk_profile: User's risk profile.
            market_data: Current market indicators.
            
        Returns:
            State vector of shape (state_dim,).
        """
        state = []
        
        # Portfolio features (normalized)
        state.extend([
            portfolio.total_value / 1_000_000,
            portfolio.unrealized_pnl / (portfolio.total_value + 1e-8),
            portfolio.volatility_30d,
            portfolio.sharpe_ratio / 3.0,  # Normalize to typical range
        ])
        
        # Asset allocation (top 10 assets, padded)
        allocations = list(portfolio.asset_allocation.values())[:10]
        allocations += [0.0] * (10 - len(allocations))
        state.extend(allocations)
        
        # Risk profile features
        state.extend([
            risk_profile.risk_tolerance,
            risk_profile.time_to_goal / 120,  # Normalize to 10 years
            risk_profile.monthly_contribution / 10000,
        ])
        
        # Market features
        state.extend([
            market_data.get("btc_return_7d", 0.0),
            market_data.get("eth_return_7d", 0.0),
            market_data.get("market_volatility", 0.0),
            market_data.get("fear_greed_index", 50) / 100,
            1.0 if portfolio.market_regime == "bull" else 0.0,
            1.0 if portfolio.market_regime == "bear" else 0.0,
            1.0 if portfolio.market_regime == "sideways" else 0.0,
        ])
        
        # Pad to state_dim
        state += [0.0] * (self.state_dim - len(state))
        
        return np.array(state[:self.state_dim], dtype=np.float32)
    
    def select_action(
        self,
        state: np.ndarray,
        training: bool = False,
    ) -> Action:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state vector.
            training: Whether in training mode (enables exploration).
            
        Returns:
            Selected action.
        """
        # Exploration
        if training and np.random.random() < self.epsilon:
            return Action(np.random.randint(self.action_dim))
        
        # Exploitation
        with torch.no_grad():
            state_tensor = torch.tensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            action_idx = q_values.argmax().item()
        
        return Action(action_idx)
    
    def calculate_reward(
        self,
        old_portfolio: PortfolioState,
        new_portfolio: PortfolioState,
        risk_profile: RiskProfile,
    ) -> float:
        """
        Calculate reward based on risk-adjusted returns.
        
        Reward function considers:
        - Portfolio return (primary reward)
        - Volatility penalty (scaled by risk tolerance)
        - Sharpe ratio bonus
        - Goal proximity stability bonus
        
        Args:
            old_portfolio: Previous portfolio state.
            new_portfolio: Current portfolio state.
            risk_profile: User's risk profile.
            
        Returns:
            Calculated reward value.
        """
        # Base reward: portfolio return
        portfolio_return = (
            (new_portfolio.total_value - old_portfolio.total_value)
            / (old_portfolio.total_value + 1e-8)
        )
        
        # Volatility penalty: penalize high volatility for conservative profiles
        volatility_penalty = (
            new_portfolio.volatility_30d * (1 - risk_profile.risk_tolerance)
        )
        
        # Goal proximity bonus: reward stability when near goal
        stability_bonus = 0.0
        if risk_profile.time_to_goal <= 12:
            stability_bonus = 0.1 if new_portfolio.volatility_30d < 0.1 else -0.1
        
        # Sharpe ratio bonus
        sharpe_bonus = new_portfolio.sharpe_ratio * 0.01
        
        # Drawdown penalty
        drawdown_penalty = 0.0
        if new_portfolio.unrealized_pnl < 0:
            drawdown_penalty = abs(new_portfolio.unrealized_pnl / new_portfolio.total_value) * 0.5
        
        reward = (
            portfolio_return * 100  # Scale returns
            - volatility_penalty * 10
            + stability_bonus
            + sharpe_bonus
            - drawdown_penalty
        )
        
        return float(reward)
    
    def train_step(self) -> float:
        """
        Perform one training step with experience replay.
        
        Returns:
            Training loss value.
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )
        
        # Convert to tensors
        states = torch.tensor(states).to(self.device)
        actions = torch.tensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.tensor(next_states).to(self.device)
        dones = torch.tensor(dones).unsqueeze(1).to(self.device)
        
        # Current Q values
        current_q = self.policy_net(states).gather(1, actions)
        
        # Target Q values using target network
        with torch.no_grad():
            # Double DQN: use policy net to select action, target net for value
            next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # Huber loss for stability
        loss = F.smooth_l1_loss(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Update step count and target network
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.update_target_network()
        
        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def update_target_network(self, tau: float = 0.005) -> None:
        """
        Soft update target network parameters.
        
        θ_target = τ * θ_policy + (1 - τ) * θ_target
        
        Args:
            tau: Soft update coefficient.
        """
        for target_param, policy_param in zip(
            self.target_net.parameters(),
            self.policy_net.parameters(),
        ):
            target_param.data.copy_(
                tau * policy_param.data + (1 - tau) * target_param.data
            )
    
    def get_rebalance_recommendation(
        self,
        portfolio: PortfolioState,
        risk_profile: RiskProfile,
        market_data: Dict[str, float],
    ) -> Dict:
        """
        Generate portfolio rebalancing recommendation.
        
        Args:
            portfolio: Current portfolio state.
            risk_profile: User's risk profile.
            market_data: Current market indicators.
            
        Returns:
            Dictionary with action, recommendations, confidence, and risk assessment.
        """
        state = self.encode_state(portfolio, risk_profile, market_data)
        action = self.select_action(state, training=False)
        
        # Get Q-values for confidence estimation
        with torch.no_grad():
            state_tensor = torch.tensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            q_values_np = q_values.cpu().numpy()[0]
        
        # Calculate confidence from Q-value distribution
        q_softmax = np.exp(q_values_np) / np.sum(np.exp(q_values_np))
        confidence = float(q_softmax[action.value])
        
        # Generate specific recommendations
        recommendations = self._action_to_recommendation(
            action, portfolio, risk_profile
        )
        
        # Assess current risk
        risk_assessment = self._assess_risk(portfolio, risk_profile)
        
        return {
            "action": action.name,
            "recommendations": recommendations,
            "confidence": confidence,
            "risk_assessment": risk_assessment,
            "q_values": {a.name: float(q_values_np[a.value]) for a in Action},
        }
    
    def _action_to_recommendation(
        self,
        action: Action,
        portfolio: PortfolioState,
        risk_profile: RiskProfile,
    ) -> List[Dict]:
        """
        Convert action to specific trade recommendations.
        
        Args:
            action: Selected action.
            portfolio: Current portfolio state.
            risk_profile: User's risk profile.
            
        Returns:
            List of trade recommendations.
        """
        recommendations = []
        
        if action == Action.HOLD:
            pass  # No action needed
        
        elif action == Action.BUY_CONSERVATIVE:
            recommendations.extend([
                {"asset": "USDC", "action": "BUY", "percentage": 40},
                {"asset": "ETH", "action": "BUY", "percentage": 30},
                {"asset": "BTC", "action": "BUY", "percentage": 30},
            ])
        
        elif action == Action.BUY_MODERATE:
            recommendations.extend([
                {"asset": "BTC", "action": "BUY", "percentage": 35},
                {"asset": "ETH", "action": "BUY", "percentage": 35},
                {"asset": "USDC", "action": "BUY", "percentage": 20},
                {"asset": "LINK", "action": "BUY", "percentage": 10},
            ])
        
        elif action == Action.BUY_AGGRESSIVE:
            recommendations.extend([
                {"asset": "ETH", "action": "BUY", "percentage": 40},
                {"asset": "BTC", "action": "BUY", "percentage": 30},
                {"asset": "SOL", "action": "BUY", "percentage": 20},
                {"asset": "AVAX", "action": "BUY", "percentage": 10},
            ])
        
        elif action == Action.SELL_PARTIAL:
            # Sell 25% of volatile assets
            volatile_assets = self._get_volatile_assets(portfolio)
            for asset in volatile_assets[:2]:
                recommendations.append({
                    "asset": asset,
                    "action": "SELL",
                    "percentage": 25,
                })
        
        elif action == Action.SELL_MAJOR:
            # Sell 50% of volatile assets
            volatile_assets = self._get_volatile_assets(portfolio)
            for asset in volatile_assets[:3]:
                recommendations.append({
                    "asset": asset,
                    "action": "SELL",
                    "percentage": 50,
                })
        
        elif action == Action.REBALANCE:
            # Rebalance to target allocation based on risk profile
            target = self._get_target_allocation(risk_profile)
            for asset, target_pct in target.items():
                current_pct = portfolio.asset_allocation.get(asset, 0) * 100
                diff = target_pct - current_pct
                if abs(diff) > 5:  # Only rebalance if difference > 5%
                    action_type = "BUY" if diff > 0 else "SELL"
                    recommendations.append({
                        "asset": asset,
                        "action": action_type,
                        "percentage": abs(diff),
                    })
        
        return recommendations
    
    def _get_volatile_assets(self, portfolio: PortfolioState) -> List[str]:
        """Get list of assets sorted by volatility (most volatile first)."""
        # In production, this would use actual volatility data
        volatile_order = ["SOL", "AVAX", "LINK", "ETH", "BTC", "USDC"]
        return [a for a in volatile_order if a in portfolio.asset_allocation]
    
    def _get_target_allocation(self, risk_profile: RiskProfile) -> Dict[str, float]:
        """Get target allocation based on risk profile."""
        rt = risk_profile.risk_tolerance
        
        if rt < 0.3:  # Conservative
            return {"USDC": 40, "BTC": 30, "ETH": 20, "LINK": 10}
        elif rt < 0.6:  # Moderate
            return {"BTC": 35, "ETH": 35, "USDC": 20, "LINK": 10}
        else:  # Aggressive
            return {"ETH": 40, "BTC": 30, "SOL": 15, "AVAX": 10, "USDC": 5}
    
    def _assess_risk(
        self,
        portfolio: PortfolioState,
        risk_profile: RiskProfile,
    ) -> Dict:
        """
        Assess current portfolio risk level.
        
        Args:
            portfolio: Current portfolio state.
            risk_profile: User's risk profile.
            
        Returns:
            Risk assessment dictionary.
        """
        # Calculate risk score (0-100)
        volatility_risk = min(portfolio.volatility_30d * 200, 50)
        concentration_risk = self._calculate_concentration_risk(portfolio) * 30
        market_risk = 20 if portfolio.market_regime == "bear" else 10
        
        risk_score = volatility_risk + concentration_risk + market_risk
        
        # Determine if risk matches profile
        expected_risk = risk_profile.risk_tolerance * 60 + 20
        risk_mismatch = abs(risk_score - expected_risk) > 20
        
        return {
            "risk_score": min(risk_score, 100),
            "volatility_risk": volatility_risk,
            "concentration_risk": concentration_risk,
            "market_risk": market_risk,
            "matches_profile": not risk_mismatch,
            "recommendation": self._get_risk_recommendation(risk_score, risk_profile),
        }
    
    def _calculate_concentration_risk(self, portfolio: PortfolioState) -> float:
        """Calculate concentration risk based on asset distribution."""
        allocations = list(portfolio.asset_allocation.values())
        if not allocations:
            return 0.0
        
        # Herfindahl-Hirschman Index (normalized)
        hhi = sum(a ** 2 for a in allocations)
        return float(hhi)
    
    def _get_risk_recommendation(
        self,
        risk_score: float,
        risk_profile: RiskProfile,
    ) -> str:
        """Get risk recommendation based on score and profile."""
        expected = risk_profile.risk_tolerance * 60 + 20
        
        if risk_score > expected + 20:
            return "Portfolio risk is higher than your profile. Consider reducing volatile assets."
        elif risk_score < expected - 20:
            return "Portfolio risk is lower than optimal. Consider adding growth assets."
        else:
            return "Portfolio risk aligns with your risk profile."
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "step_count": self.step_count,
        }, path)
    
    def load(self, path: str) -> None:
        """Load model from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.step_count = checkpoint["step_count"]
