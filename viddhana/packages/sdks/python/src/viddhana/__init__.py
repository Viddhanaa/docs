"""
VIDDHANA Python SDK

A Python SDK for interacting with the VIDDHANA blockchain network,
including Atlas Chain, Prometheus AI, Vault management, and Governance.

Example:
    >>> from viddhana import ViddhanaClient
    >>> 
    >>> client = ViddhanaClient(
    ...     rpc_url="https://rpc.testnet.viddhana.network",
    ...     api_key="your-api-key",
    ... )
    >>> client.connect()
    >>> 
    >>> # Get chain info
    >>> info = client.atlas.get_chain_info()
    >>> print(f"Connected to chain {info.chain_id}")
    >>> 
    >>> # Get AI prediction
    >>> prediction = client.ai.predict_price("BTC", horizon=7)
    >>> print(f"BTC trend: {prediction.trend}")
    >>> 
    >>> client.disconnect()
"""

from .client import ViddhanaClient
from .atlas import AtlasModule
from .vault import VaultModule
from .ai import AIModule
from .governance import GovernanceModule
from .types import (
    # Enums
    Network,
    # Config
    NetworkConfig,
    # Portfolio
    Asset,
    Portfolio,
    PortfolioMetrics,
    VaultInfo,
    # Rebalancing
    RebalanceAction,
    RebalanceEvent,
    RebalanceHistory,
    # AI
    PricePrediction,
    PricePredictionPoint,
    PortfolioOptimization,
    RiskAssessment,
    # Chain
    ChainInfo,
    BlockInfo,
    # Governance
    Proposal,
    Vote,
    Delegation,
    # Transactions
    TransactionReceipt,
)

__version__ = "1.0.0"
__author__ = "VIDDHANA Team"
__email__ = "sdk@viddhana.network"

__all__ = [
    # Main client
    "ViddhanaClient",
    # Modules
    "AtlasModule",
    "VaultModule",
    "AIModule",
    "GovernanceModule",
    # Enums
    "Network",
    # Config
    "NetworkConfig",
    # Portfolio types
    "Asset",
    "Portfolio",
    "PortfolioMetrics",
    "VaultInfo",
    # Rebalancing types
    "RebalanceAction",
    "RebalanceEvent",
    "RebalanceHistory",
    # AI types
    "PricePrediction",
    "PricePredictionPoint",
    "PortfolioOptimization",
    "RiskAssessment",
    # Chain types
    "ChainInfo",
    "BlockInfo",
    # Governance types
    "Proposal",
    "Vote",
    "Delegation",
    # Transaction types
    "TransactionReceipt",
]
