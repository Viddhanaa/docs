"""Type definitions for the VIDDHANA SDK."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum


class Network(str, Enum):
    """Available VIDDHANA networks."""
    MAINNET = "mainnet"
    TESTNET = "testnet"


@dataclass
class NetworkConfig:
    """Network configuration."""
    chain_id: int
    rpc_url: str
    ws_url: str
    api_url: str


@dataclass
class Asset:
    """Asset in a portfolio."""
    symbol: str
    address: str
    balance: str
    value: str
    allocation: float


@dataclass
class PortfolioMetrics:
    """Portfolio performance metrics."""
    volatility_30d: float
    sharpe_ratio: float
    unrealized_pnl: float
    unrealized_pnl_percent: float


@dataclass
class Portfolio:
    """User portfolio data."""
    address: str
    total_value: str
    currency: str
    assets: List[Asset]
    metrics: PortfolioMetrics
    last_updated: int


@dataclass
class RebalanceAction:
    """Single rebalancing action."""
    asset: str
    action: str  # "BUY" or "SELL"
    percentage: float
    amount: str
    value_usd: str


@dataclass
class RebalanceEvent:
    """Rebalancing event record."""
    tx_hash: str
    block_number: int
    timestamp: int
    reason: str
    actions: List[RebalanceAction]
    ai_confidence: float
    gas_used: int
    gas_cost: str


@dataclass
class RebalanceHistory:
    """Rebalancing history for an address."""
    address: str
    history: List[RebalanceEvent]
    total_rebalances: int


@dataclass
class PricePredictionPoint:
    """Single price prediction point."""
    day: int
    price: float
    confidence: float


@dataclass
class PricePrediction:
    """AI price prediction result."""
    asset: str
    current_price: float
    horizon: int
    predictions: List[PricePredictionPoint]
    trend: str
    volatility_forecast: float
    model_version: str
    generated_at: int


@dataclass
class PortfolioOptimization:
    """Portfolio optimization recommendation."""
    action: str
    recommendations: List[Dict[str, Any]]
    confidence: float
    risk_assessment: Dict[str, Any]


@dataclass
class RiskAssessment:
    """Risk assessment result."""
    risk_score: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    recommendations: List[str]


@dataclass
class VaultInfo:
    """Vault information."""
    vault_id: str
    owner: str
    assets: List[Asset]
    total_value: str
    strategy: str
    created_at: int


@dataclass
class ChainInfo:
    """Blockchain information."""
    chain_id: int
    block_number: int
    gas_price: int
    network_name: str


@dataclass
class BlockInfo:
    """Block information."""
    number: int
    hash: str
    parent_hash: str
    timestamp: int
    transactions: List[str]
    gas_used: int
    gas_limit: int


@dataclass
class Proposal:
    """Governance proposal."""
    proposal_id: str
    proposer: str
    title: str
    description: str
    start_block: int
    end_block: int
    for_votes: str
    against_votes: str
    status: str


@dataclass
class Vote:
    """Vote record."""
    voter: str
    proposal_id: str
    support: bool
    votes: str
    reason: Optional[str] = None


@dataclass
class Delegation:
    """Delegation record."""
    delegator: str
    delegatee: str
    votes: str
    timestamp: int


@dataclass
class TransactionReceipt:
    """Transaction receipt."""
    tx_hash: str
    block_number: int
    status: bool
    gas_used: int
    effective_gas_price: int
    logs: List[Dict[str, Any]] = field(default_factory=list)
