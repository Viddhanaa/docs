# API & SDK Reference Implementation Guide

> Detailed implementation guide for VIDDHANA JSON-RPC API and multi-language SDKs

---

## Table of Contents
1. [Overview](#overview)
2. [JSON-RPC API](#json-rpc-api)
3. [JavaScript SDK](#javascript-sdk)
4. [Python SDK](#python-sdk)
5. [Go SDK](#go-sdk)
6. [WebSocket API](#websocket-api)
7. [Authentication](#authentication)
8. [Rate Limiting](#rate-limiting)
9. [Testing](#testing)

---

## Overview

VIDDHANA provides comprehensive APIs and SDKs for interacting with:
- **Atlas Chain**: EVM-compatible JSON-RPC + custom methods
- **Prometheus AI**: Prediction and optimization endpoints
- **DePIN**: Sensor data and rewards
- **Tokenomics**: Staking, vesting, governance

### API Endpoints

| Environment | Base URL |
|-------------|----------|
| Mainnet RPC | `https://rpc.viddhana.network` |
| Testnet RPC | `https://rpc.testnet.viddhana.network` |
| API Gateway | `https://api.viddhana.network` |
| WebSocket | `wss://ws.viddhana.network` |

---

## JSON-RPC API

### Standard Ethereum Methods

All standard `eth_*` methods are supported for EVM compatibility.

### Custom VIDDHANA Methods

#### vdh_getPortfolio

Returns user's portfolio details.

**Request:**
```json
{
  "jsonrpc": "2.0",
  "method": "vdh_getPortfolio",
  "params": ["0x1234..."],
  "id": 1
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "address": "0x1234...",
    "totalValue": "125000.00",
    "currency": "USD",
    "assets": [
      {
        "symbol": "BTC",
        "address": "0xabc...",
        "balance": "1.5",
        "value": "75000.00",
        "allocation": 60.0
      },
      {
        "symbol": "ETH",
        "address": "0xdef...",
        "balance": "15.0",
        "value": "30000.00",
        "allocation": 24.0
      },
      {
        "symbol": "USDC",
        "address": "0x789...",
        "balance": "20000.0",
        "value": "20000.00",
        "allocation": 16.0
      }
    ],
    "metrics": {
      "volatility30d": 0.18,
      "sharpeRatio": 1.45,
      "unrealizedPnL": 15000.00,
      "unrealizedPnLPercent": 13.6
    },
    "lastUpdated": 1704067200
  }
}
```

#### vdh_getRebalanceHistory

Returns rebalancing history for a user.

**Request:**
```json
{
  "jsonrpc": "2.0",
  "method": "vdh_getRebalanceHistory",
  "params": [
    "0x1234...",
    {
      "fromBlock": 1000000,
      "toBlock": "latest",
      "limit": 50
    }
  ],
  "id": 1
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "address": "0x1234...",
    "history": [
      {
        "txHash": "0xabc...",
        "blockNumber": 1050000,
        "timestamp": 1704067200,
        "reason": "Goal Proximity Mode",
        "actions": [
          {
            "asset": "BTC",
            "action": "SELL",
            "percentage": 25,
            "amount": "0.5",
            "valueUSD": "25000.00"
          },
          {
            "asset": "USDC",
            "action": "BUY",
            "percentage": 25,
            "amount": "25000.0",
            "valueUSD": "25000.00"
          }
        ],
        "aiConfidence": 0.85,
        "gasUsed": 150000,
        "gasCost": "0.00015"
      }
    ],
    "totalRebalances": 12
  }
}
```

#### vdh_getAIPrediction

Get AI price prediction for an asset.

**Request:**
```json
{
  "jsonrpc": "2.0",
  "method": "vdh_getAIPrediction",
  "params": [
    {
      "asset": "BTC",
      "horizon": 7
    }
  ],
  "id": 1
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "asset": "BTC",
    "currentPrice": 50000.00,
    "horizon": 7,
    "predictions": [
      {"day": 1, "price": 50500.00, "confidence": 0.82},
      {"day": 2, "price": 51200.00, "confidence": 0.78},
      {"day": 3, "price": 51800.00, "confidence": 0.74},
      {"day": 4, "price": 52100.00, "confidence": 0.70},
      {"day": 5, "price": 52500.00, "confidence": 0.66},
      {"day": 6, "price": 52800.00, "confidence": 0.62},
      {"day": 7, "price": 53000.00, "confidence": 0.58}
    ],
    "trend": "bullish",
    "volatilityForecast": 0.15,
    "modelVersion": "v2.3.1",
    "generatedAt": 1704067200
  }
}
```

#### vdh_getSensorRewards

Get DePIN sensor rewards.

**Request:**
```json
{
  "jsonrpc": "2.0",
  "method": "vdh_getSensorRewards",
  "params": ["sensor-solar-001"],
  "id": 1
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "sensorId": "sensor-solar-001",
    "owner": "0x1234...",
    "deviceType": "solar_panel",
    "status": "active",
    "rewards": {
      "pending": "15.50",
      "claimed": "450.00",
      "lifetime": "465.50",
      "currency": "VDH"
    },
    "performance": {
      "uptimePercent": 99.5,
      "dataQualityScore": 0.98,
      "dailyRate": "0.60",
      "errorCount": 2
    },
    "lastDataPoint": 1704067200
  }
}
```

#### vdh_getTokenomicsStats

Get tokenomics statistics.

**Request:**
```json
{
  "jsonrpc": "2.0",
  "method": "vdh_getTokenomicsStats",
  "params": [],
  "id": 1
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "totalSupply": "1000000000",
    "circulatingSupply": "223000000",
    "burnedSupply": "5000000",
    "stakedSupply": "75000000",
    "price": {
      "usd": 0.45,
      "change24h": 5.2,
      "change7d": 12.8
    },
    "marketCap": "100350000",
    "fdv": "450000000",
    "stakingAPY": 12.5,
    "nextBuybackDate": 1706745600,
    "lastBuyback": {
      "date": 1704067200,
      "amount": "500000",
      "burned": "250000"
    }
  }
}
```

---

## JavaScript SDK

### Installation

```bash
npm install @viddhana/sdk
# or
yarn add @viddhana/sdk
```

### SDK Structure

```typescript
// packages/sdks/javascript/src/index.ts
export { ViddhanaClient } from './client';
export { AtlasChain } from './modules/chain';
export { PrometheusAI } from './modules/ai';
export { Portfolio } from './modules/portfolio';
export { DePIN } from './modules/depin';
export { Staking } from './modules/staking';

// Types
export * from './types';
```

### Core Client

```typescript
// packages/sdks/javascript/src/client.ts
import { ethers } from 'ethers';
import { AtlasChain } from './modules/chain';
import { PrometheusAI } from './modules/ai';
import { Portfolio } from './modules/portfolio';
import { DePIN } from './modules/depin';
import { Staking } from './modules/staking';
import { ViddhanaConfig, NetworkConfig } from './types';

const NETWORKS: Record<string, NetworkConfig> = {
  mainnet: {
    chainId: 13370,
    rpcUrl: 'https://rpc.viddhana.network',
    wsUrl: 'wss://ws.viddhana.network',
    apiUrl: 'https://api.viddhana.network',
  },
  testnet: {
    chainId: 13370,
    rpcUrl: 'https://rpc.testnet.viddhana.network',
    wsUrl: 'wss://ws.testnet.viddhana.network',
    apiUrl: 'https://api.testnet.viddhana.network',
  },
};

export class ViddhanaClient {
  public readonly chain: AtlasChain;
  public readonly ai: PrometheusAI;
  public readonly portfolio: Portfolio;
  public readonly depin: DePIN;
  public readonly staking: Staking;

  private readonly provider: ethers.JsonRpcProvider;
  private readonly signer?: ethers.Signer;
  private readonly config: ViddhanaConfig;

  constructor(config: ViddhanaConfig) {
    this.config = config;
    
    const network = NETWORKS[config.network || 'mainnet'];
    this.provider = new ethers.JsonRpcProvider(config.rpcUrl || network.rpcUrl);
    
    if (config.privateKey) {
      this.signer = new ethers.Wallet(config.privateKey, this.provider);
    }
    
    // Initialize modules
    this.chain = new AtlasChain(this.provider, this.signer);
    this.ai = new PrometheusAI(network.apiUrl, config.apiKey);
    this.portfolio = new Portfolio(this.provider, this.signer, network.apiUrl);
    this.depin = new DePIN(this.provider, network.apiUrl);
    this.staking = new Staking(this.provider, this.signer);
  }

  async connect(provider: ethers.BrowserProvider): Promise<ethers.Signer> {
    const signer = await provider.getSigner();
    return signer;
  }

  getProvider(): ethers.JsonRpcProvider {
    return this.provider;
  }

  getSigner(): ethers.Signer | undefined {
    return this.signer;
  }
}
```

### Portfolio Module

```typescript
// packages/sdks/javascript/src/modules/portfolio.ts
import { ethers } from 'ethers';
import { PortfolioData, Asset, RebalanceHistory, RebalanceAction } from '../types';

export class Portfolio {
  private readonly provider: ethers.JsonRpcProvider;
  private readonly signer?: ethers.Signer;
  private readonly apiUrl: string;

  constructor(
    provider: ethers.JsonRpcProvider,
    signer: ethers.Signer | undefined,
    apiUrl: string
  ) {
    this.provider = provider;
    this.signer = signer;
    this.apiUrl = apiUrl;
  }

  /**
   * Get portfolio for an address
   */
  async getPortfolio(address: string): Promise<PortfolioData> {
    const result = await this.provider.send('vdh_getPortfolio', [address]);
    return result as PortfolioData;
  }

  /**
   * Get rebalancing history
   */
  async getRebalanceHistory(
    address: string,
    options?: {
      fromBlock?: number;
      toBlock?: number | 'latest';
      limit?: number;
    }
  ): Promise<RebalanceHistory> {
    const result = await this.provider.send('vdh_getRebalanceHistory', [
      address,
      options || {},
    ]);
    return result as RebalanceHistory;
  }

  /**
   * Get AI optimization recommendation
   */
  async getOptimizationRecommendation(
    address: string
  ): Promise<{
    actions: RebalanceAction[];
    confidence: number;
    reason: string;
  }> {
    if (!this.signer) {
      throw new Error('Signer required for optimization');
    }

    const response = await fetch(`${this.apiUrl}/v1/optimize/portfolio`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        user_id: address,
        portfolio: await this.getPortfolio(address),
      }),
    });

    return response.json();
  }

  /**
   * Execute rebalancing
   */
  async executeRebalance(
    actions: RebalanceAction[]
  ): Promise<ethers.TransactionResponse> {
    if (!this.signer) {
      throw new Error('Signer required for rebalancing');
    }

    const policyEngine = new ethers.Contract(
      POLICY_ENGINE_ADDRESS,
      POLICY_ENGINE_ABI,
      this.signer
    );

    const tx = await policyEngine.executeManualRebalance(actions);
    return tx;
  }

  /**
   * Set user profile for auto-rebalancing
   */
  async setProfile(params: {
    riskTolerance: number;  // 0-10000 basis points
    timeToGoal: number;     // months
    autoRebalance: boolean;
  }): Promise<ethers.TransactionResponse> {
    if (!this.signer) {
      throw new Error('Signer required');
    }

    const policyEngine = new ethers.Contract(
      POLICY_ENGINE_ADDRESS,
      POLICY_ENGINE_ABI,
      this.signer
    );

    const tx = await policyEngine.setUserProfile(
      params.riskTolerance,
      params.timeToGoal,
      params.autoRebalance
    );
    return tx;
  }

  /**
   * Deposit assets to vault
   */
  async deposit(
    asset: string,
    amount: bigint
  ): Promise<ethers.TransactionResponse> {
    if (!this.signer) {
      throw new Error('Signer required');
    }

    // Approve token transfer
    const token = new ethers.Contract(asset, ERC20_ABI, this.signer);
    const approveTx = await token.approve(VAULT_MANAGER_ADDRESS, amount);
    await approveTx.wait();

    // Execute deposit
    const vaultManager = new ethers.Contract(
      VAULT_MANAGER_ADDRESS,
      VAULT_MANAGER_ABI,
      this.signer
    );

    const tx = await vaultManager.deposit(asset, amount);
    return tx;
  }

  /**
   * Withdraw assets from vault
   */
  async withdraw(
    asset: string,
    amount: bigint
  ): Promise<ethers.TransactionResponse> {
    if (!this.signer) {
      throw new Error('Signer required');
    }

    const vaultManager = new ethers.Contract(
      VAULT_MANAGER_ADDRESS,
      VAULT_MANAGER_ABI,
      this.signer
    );

    const tx = await vaultManager.withdraw(asset, amount);
    return tx;
  }
}
```

### Prometheus AI Module

```typescript
// packages/sdks/javascript/src/modules/ai.ts
import { PricePrediction, PortfolioOptimization, RiskAssessment } from '../types';

export class PrometheusAI {
  private readonly apiUrl: string;
  private readonly apiKey?: string;

  constructor(apiUrl: string, apiKey?: string) {
    this.apiUrl = apiUrl;
    this.apiKey = apiKey;
  }

  private async request<T>(endpoint: string, body: object): Promise<T> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };

    if (this.apiKey) {
      headers['Authorization'] = `Bearer ${this.apiKey}`;
    }

    const response = await fetch(`${this.apiUrl}${endpoint}`, {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    return response.json();
  }

  /**
   * Get price prediction for an asset
   */
  async predictPrice(
    asset: string,
    horizon: number = 7
  ): Promise<PricePrediction> {
    return this.request<PricePrediction>('/v1/predict/price', {
      asset,
      horizon,
    });
  }

  /**
   * Get portfolio optimization recommendation
   */
  async optimizePortfolio(params: {
    userId: string;
    portfolio: Record<string, number>;
    riskTolerance: number;
    timeToGoal: number;
  }): Promise<PortfolioOptimization> {
    return this.request<PortfolioOptimization>('/v1/optimize/portfolio', {
      user_id: params.userId,
      portfolio: params.portfolio,
      risk_tolerance: params.riskTolerance,
      time_to_goal: params.timeToGoal,
    });
  }

  /**
   * Assess portfolio risk
   */
  async assessRisk(
    portfolio: Record<string, number>
  ): Promise<RiskAssessment> {
    return this.request<RiskAssessment>('/v1/assess/risk', {
      portfolio,
    });
  }

  /**
   * Stream real-time predictions via WebSocket
   */
  streamPredictions(
    asset: string,
    onUpdate: (prediction: PricePrediction) => void,
    onError?: (error: Error) => void
  ): () => void {
    const wsUrl = this.apiUrl.replace('https://', 'wss://').replace('http://', 'ws://');
    const ws = new WebSocket(`${wsUrl}/ws/predictions/${asset}`);

    ws.onmessage = (event) => {
      const prediction = JSON.parse(event.data);
      onUpdate(prediction);
    };

    ws.onerror = (event) => {
      if (onError) {
        onError(new Error('WebSocket error'));
      }
    };

    // Return cleanup function
    return () => ws.close();
  }
}
```

### Usage Examples

```typescript
// examples/basic-usage.ts
import { ViddhanaClient } from '@viddhana/sdk';

async function main() {
  // Initialize client
  const client = new ViddhanaClient({
    network: 'testnet',
    privateKey: process.env.PRIVATE_KEY,
    apiKey: process.env.API_KEY,
  });

  // Get portfolio
  const address = '0x1234...';
  const portfolio = await client.portfolio.getPortfolio(address);
  console.log('Portfolio:', portfolio);

  // Get AI prediction
  const prediction = await client.ai.predictPrice('BTC', 7);
  console.log('BTC Prediction:', prediction);

  // Get optimization recommendation
  const optimization = await client.ai.optimizePortfolio({
    userId: address,
    portfolio: { BTC: 50000, ETH: 30000, USDC: 20000 },
    riskTolerance: 0.5,
    timeToGoal: 24,
  });
  console.log('Recommendation:', optimization);

  // Deposit assets
  const tx = await client.portfolio.deposit(
    '0xUSDC...',
    ethers.parseUnits('1000', 6)
  );
  console.log('Deposit TX:', tx.hash);

  // Stake VDH tokens
  const stakeTx = await client.staking.stake(
    ethers.parseEther('10000'),
    365 * 24 * 60 * 60  // 1 year lock
  );
  console.log('Stake TX:', stakeTx.hash);
}

main().catch(console.error);
```

---

## Python SDK

### Installation

```bash
pip install viddhana-sdk
```

### SDK Structure

```python
# packages/sdks/python/viddhana/__init__.py
from .client import ViddhanaClient
from .modules.chain import AtlasChain
from .modules.ai import PrometheusAI
from .modules.portfolio import Portfolio
from .modules.depin import DePIN
from .modules.staking import Staking

__all__ = [
    'ViddhanaClient',
    'AtlasChain',
    'PrometheusAI',
    'Portfolio',
    'DePIN',
    'Staking',
]

__version__ = '1.0.0'
```

### Core Client

```python
# packages/sdks/python/viddhana/client.py
from dataclasses import dataclass
from typing import Optional
from web3 import Web3
from eth_account import Account

from .modules.chain import AtlasChain
from .modules.ai import PrometheusAI
from .modules.portfolio import Portfolio
from .modules.depin import DePIN
from .modules.staking import Staking


@dataclass
class NetworkConfig:
    chain_id: int
    rpc_url: str
    ws_url: str
    api_url: str


NETWORKS = {
    'mainnet': NetworkConfig(
        chain_id=13370,
        rpc_url='https://rpc.viddhana.network',
        ws_url='wss://ws.viddhana.network',
        api_url='https://api.viddhana.network',
    ),
    'testnet': NetworkConfig(
        chain_id=13370,
        rpc_url='https://rpc.testnet.viddhana.network',
        ws_url='wss://ws.testnet.viddhana.network',
        api_url='https://api.testnet.viddhana.network',
    ),
}


class ViddhanaClient:
    """Main client for interacting with VIDDHANA network."""
    
    def __init__(
        self,
        network: str = 'mainnet',
        private_key: Optional[str] = None,
        rpc_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.network_config = NETWORKS[network]
        self.w3 = Web3(Web3.HTTPProvider(rpc_url or self.network_config.rpc_url))
        
        self.account = None
        if private_key:
            self.account = Account.from_key(private_key)
        
        # Initialize modules
        self.chain = AtlasChain(self.w3, self.account)
        self.ai = PrometheusAI(self.network_config.api_url, api_key)
        self.portfolio = Portfolio(self.w3, self.account, self.network_config.api_url)
        self.depin = DePIN(self.w3, self.network_config.api_url)
        self.staking = Staking(self.w3, self.account)
    
    @property
    def address(self) -> Optional[str]:
        """Get the connected account address."""
        return self.account.address if self.account else None
    
    def is_connected(self) -> bool:
        """Check if connected to the network."""
        return self.w3.is_connected()
```

### Prometheus AI Module

```python
# packages/sdks/python/viddhana/modules/ai.py
import httpx
from dataclasses import dataclass
from typing import List, Dict, Optional
import asyncio


@dataclass
class PricePrediction:
    asset: str
    horizon_days: int
    predictions: List[float]
    confidence: List[float]
    timestamp: int


@dataclass
class PortfolioOptimization:
    action: str
    recommendations: List[Dict]
    confidence: float
    risk_assessment: Dict


@dataclass
class RiskAssessment:
    risk_score: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    recommendations: List[str]


class PrometheusAI:
    """Client for Prometheus AI prediction and optimization API."""
    
    def __init__(self, api_url: str, api_key: Optional[str] = None):
        self.api_url = api_url
        self.api_key = api_key
        self._client = httpx.Client(
            base_url=api_url,
            headers=self._get_headers(),
            timeout=30.0,
        )
        self._async_client = httpx.AsyncClient(
            base_url=api_url,
            headers=self._get_headers(),
            timeout=30.0,
        )
    
    def _get_headers(self) -> Dict[str, str]:
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        return headers
    
    def predict_price(self, asset: str, horizon: int = 7) -> PricePrediction:
        """
        Get price prediction for an asset.
        
        Args:
            asset: Asset symbol (e.g., 'BTC', 'ETH')
            horizon: Number of days to predict ahead (1-30)
        
        Returns:
            PricePrediction with daily predictions and confidence scores
        """
        response = self._client.post(
            '/v1/predict/price',
            json={'asset': asset, 'horizon': horizon},
        )
        response.raise_for_status()
        data = response.json()
        
        return PricePrediction(
            asset=data['asset'],
            horizon_days=data['horizon_days'],
            predictions=data['predictions'],
            confidence=data['confidence'],
            timestamp=data['timestamp'],
        )
    
    async def predict_price_async(self, asset: str, horizon: int = 7) -> PricePrediction:
        """Async version of predict_price."""
        response = await self._async_client.post(
            '/v1/predict/price',
            json={'asset': asset, 'horizon': horizon},
        )
        response.raise_for_status()
        data = response.json()
        
        return PricePrediction(
            asset=data['asset'],
            horizon_days=data['horizon_days'],
            predictions=data['predictions'],
            confidence=data['confidence'],
            timestamp=data['timestamp'],
        )
    
    def optimize_portfolio(
        self,
        user_id: str,
        portfolio: Dict[str, float],
        risk_tolerance: float,
        time_to_goal: int,
    ) -> PortfolioOptimization:
        """
        Get portfolio optimization recommendation.
        
        Args:
            user_id: User identifier
            portfolio: Current portfolio as {asset: value} dict
            risk_tolerance: Risk tolerance 0-1
            time_to_goal: Months until investment goal
        
        Returns:
            PortfolioOptimization with recommended actions
        """
        response = self._client.post(
            '/v1/optimize/portfolio',
            json={
                'user_id': user_id,
                'portfolio': portfolio,
                'risk_tolerance': risk_tolerance,
                'time_to_goal': time_to_goal,
            },
        )
        response.raise_for_status()
        data = response.json()
        
        return PortfolioOptimization(
            action=data['action'],
            recommendations=data['recommendations'],
            confidence=data['confidence'],
            risk_assessment=data['risk_assessment'],
        )
    
    def assess_risk(self, portfolio: Dict[str, float]) -> RiskAssessment:
        """
        Assess portfolio risk metrics.
        
        Args:
            portfolio: Portfolio as {asset: value} dict
        
        Returns:
            RiskAssessment with risk metrics and recommendations
        """
        response = self._client.post(
            '/v1/assess/risk',
            json={'portfolio': portfolio},
        )
        response.raise_for_status()
        data = response.json()
        
        return RiskAssessment(
            risk_score=data['risk_score'],
            volatility=data['volatility'],
            sharpe_ratio=data['sharpe_ratio'],
            max_drawdown=data['max_drawdown'],
            recommendations=data['recommendations'],
        )
    
    def close(self):
        """Close HTTP clients."""
        self._client.close()
    
    async def aclose(self):
        """Close async HTTP client."""
        await self._async_client.aclose()
```

### Usage Examples

```python
# examples/basic_usage.py
import os
from viddhana import ViddhanaClient

def main():
    # Initialize client
    client = ViddhanaClient(
        network='testnet',
        private_key=os.environ.get('PRIVATE_KEY'),
        api_key=os.environ.get('API_KEY'),
    )
    
    # Check connection
    print(f"Connected: {client.is_connected()}")
    print(f"Address: {client.address}")
    
    # Get price prediction
    prediction = client.ai.predict_price('BTC', horizon=7)
    print(f"\nBTC 7-day prediction:")
    for i, (price, conf) in enumerate(zip(prediction.predictions, prediction.confidence)):
        print(f"  Day {i+1}: ${price:,.2f} (confidence: {conf:.1%})")
    
    # Get portfolio optimization
    portfolio = {'BTC': 50000, 'ETH': 30000, 'USDC': 20000}
    optimization = client.ai.optimize_portfolio(
        user_id=client.address,
        portfolio=portfolio,
        risk_tolerance=0.5,
        time_to_goal=24,
    )
    
    print(f"\nOptimization recommendation: {optimization.action}")
    print(f"Confidence: {optimization.confidence:.1%}")
    for rec in optimization.recommendations:
        print(f"  - {rec['action']} {rec['percentage']}% of {rec['asset']}")
    
    # Assess portfolio risk
    risk = client.ai.assess_risk(portfolio)
    print(f"\nRisk Assessment:")
    print(f"  Risk Score: {risk.risk_score:.2f}")
    print(f"  Volatility: {risk.volatility:.1%}")
    print(f"  Sharpe Ratio: {risk.sharpe_ratio:.2f}")
    print(f"  Max Drawdown: {risk.max_drawdown:.1%}")


if __name__ == '__main__':
    main()
```

---

## Go SDK

### Installation

```bash
go get github.com/viddhana/sdk-go
```

### SDK Structure

```go
// packages/sdks/go/viddhana/client.go
package viddhana

import (
    "context"
    "crypto/ecdsa"
    "math/big"

    "github.com/ethereum/go-ethereum/common"
    "github.com/ethereum/go-ethereum/ethclient"
)

type NetworkConfig struct {
    ChainID int64
    RPCURL  string
    WSURL   string
    APIURL  string
}

var Networks = map[string]NetworkConfig{
    "mainnet": {
        ChainID: 13370,
        RPCURL:  "https://rpc.viddhana.network",
        WSURL:   "wss://ws.viddhana.network",
        APIURL:  "https://api.viddhana.network",
    },
    "testnet": {
        ChainID: 13370,
        RPCURL:  "https://rpc.testnet.viddhana.network",
        WSURL:   "wss://ws.testnet.viddhana.network",
        APIURL:  "https://api.testnet.viddhana.network",
    },
}

type ClientConfig struct {
    Network    string
    RPCURL     string
    PrivateKey *ecdsa.PrivateKey
    APIKey     string
}

type Client struct {
    config    ClientConfig
    ethClient *ethclient.Client
    
    Chain     *AtlasChain
    AI        *PrometheusAI
    Portfolio *Portfolio
    DePIN     *DePIN
    Staking   *Staking
}

func NewClient(cfg ClientConfig) (*Client, error) {
    network := Networks[cfg.Network]
    if cfg.RPCURL == "" {
        cfg.RPCURL = network.RPCURL
    }
    
    ethClient, err := ethclient.Dial(cfg.RPCURL)
    if err != nil {
        return nil, err
    }
    
    client := &Client{
        config:    cfg,
        ethClient: ethClient,
    }
    
    // Initialize modules
    client.Chain = NewAtlasChain(ethClient, cfg.PrivateKey)
    client.AI = NewPrometheusAI(network.APIURL, cfg.APIKey)
    client.Portfolio = NewPortfolio(ethClient, cfg.PrivateKey, network.APIURL)
    client.DePIN = NewDePIN(ethClient, network.APIURL)
    client.Staking = NewStaking(ethClient, cfg.PrivateKey)
    
    return client, nil
}

func (c *Client) Close() {
    c.ethClient.Close()
}

func (c *Client) Address() common.Address {
    if c.config.PrivateKey == nil {
        return common.Address{}
    }
    return crypto.PubkeyToAddress(c.config.PrivateKey.PublicKey)
}
```

### Prometheus AI Module

```go
// packages/sdks/go/viddhana/ai.go
package viddhana

import (
    "bytes"
    "context"
    "encoding/json"
    "fmt"
    "net/http"
    "time"
)

type PricePrediction struct {
    Asset       string    `json:"asset"`
    HorizonDays int       `json:"horizon_days"`
    Predictions []float64 `json:"predictions"`
    Confidence  []float64 `json:"confidence"`
    Timestamp   int64     `json:"timestamp"`
}

type PortfolioOptimization struct {
    Action          string                   `json:"action"`
    Recommendations []map[string]interface{} `json:"recommendations"`
    Confidence      float64                  `json:"confidence"`
    RiskAssessment  map[string]interface{}   `json:"risk_assessment"`
}

type RiskAssessment struct {
    RiskScore       float64  `json:"risk_score"`
    Volatility      float64  `json:"volatility"`
    SharpeRatio     float64  `json:"sharpe_ratio"`
    MaxDrawdown     float64  `json:"max_drawdown"`
    Recommendations []string `json:"recommendations"`
}

type PrometheusAI struct {
    apiURL     string
    apiKey     string
    httpClient *http.Client
}

func NewPrometheusAI(apiURL, apiKey string) *PrometheusAI {
    return &PrometheusAI{
        apiURL: apiURL,
        apiKey: apiKey,
        httpClient: &http.Client{
            Timeout: 30 * time.Second,
        },
    }
}

func (p *PrometheusAI) request(ctx context.Context, endpoint string, body interface{}, result interface{}) error {
    jsonBody, err := json.Marshal(body)
    if err != nil {
        return err
    }
    
    req, err := http.NewRequestWithContext(ctx, "POST", p.apiURL+endpoint, bytes.NewBuffer(jsonBody))
    if err != nil {
        return err
    }
    
    req.Header.Set("Content-Type", "application/json")
    if p.apiKey != "" {
        req.Header.Set("Authorization", "Bearer "+p.apiKey)
    }
    
    resp, err := p.httpClient.Do(req)
    if err != nil {
        return err
    }
    defer resp.Body.Close()
    
    if resp.StatusCode != http.StatusOK {
        return fmt.Errorf("API error: %d", resp.StatusCode)
    }
    
    return json.NewDecoder(resp.Body).Decode(result)
}

// PredictPrice gets price prediction for an asset
func (p *PrometheusAI) PredictPrice(ctx context.Context, asset string, horizon int) (*PricePrediction, error) {
    var result PricePrediction
    
    err := p.request(ctx, "/v1/predict/price", map[string]interface{}{
        "asset":   asset,
        "horizon": horizon,
    }, &result)
    
    if err != nil {
        return nil, err
    }
    
    return &result, nil
}

// OptimizePortfolio gets portfolio optimization recommendation
func (p *PrometheusAI) OptimizePortfolio(
    ctx context.Context,
    userID string,
    portfolio map[string]float64,
    riskTolerance float64,
    timeToGoal int,
) (*PortfolioOptimization, error) {
    var result PortfolioOptimization
    
    err := p.request(ctx, "/v1/optimize/portfolio", map[string]interface{}{
        "user_id":        userID,
        "portfolio":      portfolio,
        "risk_tolerance": riskTolerance,
        "time_to_goal":   timeToGoal,
    }, &result)
    
    if err != nil {
        return nil, err
    }
    
    return &result, nil
}

// AssessRisk assesses portfolio risk
func (p *PrometheusAI) AssessRisk(ctx context.Context, portfolio map[string]float64) (*RiskAssessment, error) {
    var result RiskAssessment
    
    err := p.request(ctx, "/v1/assess/risk", map[string]interface{}{
        "portfolio": portfolio,
    }, &result)
    
    if err != nil {
        return nil, err
    }
    
    return &result, nil
}
```

### Usage Examples

```go
// examples/basic_usage.go
package main

import (
    "context"
    "fmt"
    "log"
    "os"

    "github.com/ethereum/go-ethereum/crypto"
    "github.com/viddhana/sdk-go/viddhana"
)

func main() {
    // Load private key
    privateKey, err := crypto.HexToECDSA(os.Getenv("PRIVATE_KEY"))
    if err != nil {
        log.Fatal(err)
    }

    // Initialize client
    client, err := viddhana.NewClient(viddhana.ClientConfig{
        Network:    "testnet",
        PrivateKey: privateKey,
        APIKey:     os.Getenv("API_KEY"),
    })
    if err != nil {
        log.Fatal(err)
    }
    defer client.Close()

    ctx := context.Background()

    // Get price prediction
    prediction, err := client.AI.PredictPrice(ctx, "BTC", 7)
    if err != nil {
        log.Fatal(err)
    }

    fmt.Println("BTC 7-day prediction:")
    for i, price := range prediction.Predictions {
        fmt.Printf("  Day %d: $%.2f (confidence: %.1f%%)\n",
            i+1, price, prediction.Confidence[i]*100)
    }

    // Get portfolio optimization
    portfolio := map[string]float64{
        "BTC":  50000,
        "ETH":  30000,
        "USDC": 20000,
    }

    optimization, err := client.AI.OptimizePortfolio(
        ctx,
        client.Address().Hex(),
        portfolio,
        0.5,
        24,
    )
    if err != nil {
        log.Fatal(err)
    }

    fmt.Printf("\nOptimization: %s (confidence: %.1f%%)\n",
        optimization.Action, optimization.Confidence*100)

    // Assess risk
    risk, err := client.AI.AssessRisk(ctx, portfolio)
    if err != nil {
        log.Fatal(err)
    }

    fmt.Printf("\nRisk Assessment:\n")
    fmt.Printf("  Risk Score: %.2f\n", risk.RiskScore)
    fmt.Printf("  Volatility: %.1f%%\n", risk.Volatility*100)
    fmt.Printf("  Sharpe Ratio: %.2f\n", risk.SharpeRatio)
}
```

---

## WebSocket API

### Connection

```javascript
// WebSocket connection for real-time updates
const ws = new WebSocket('wss://ws.viddhana.network');

ws.onopen = () => {
  // Subscribe to portfolio updates
  ws.send(JSON.stringify({
    type: 'subscribe',
    channel: 'portfolio',
    address: '0x1234...'
  }));
  
  // Subscribe to price predictions
  ws.send(JSON.stringify({
    type: 'subscribe',
    channel: 'predictions',
    asset: 'BTC'
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  switch (data.channel) {
    case 'portfolio':
      console.log('Portfolio update:', data.payload);
      break;
    case 'predictions':
      console.log('New prediction:', data.payload);
      break;
    case 'rebalance':
      console.log('Rebalance executed:', data.payload);
      break;
  }
};
```

### Available Channels

| Channel | Description | Payload |
|---------|-------------|---------|
| `portfolio` | Portfolio value updates | Portfolio data |
| `predictions` | AI prediction updates | Price predictions |
| `rebalance` | Rebalancing events | Transaction details |
| `rewards` | DePIN reward updates | Reward data |
| `blocks` | New block notifications | Block header |

---

## Authentication

### API Key Authentication

```bash
# Include API key in headers
curl -X POST https://api.viddhana.network/v1/predict/price \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"asset": "BTC", "horizon": 7}'
```

### Signed Requests (for sensitive operations)

```typescript
import { ethers } from 'ethers';

async function signedRequest(endpoint: string, body: object, signer: ethers.Signer) {
  const timestamp = Date.now();
  const message = JSON.stringify({ ...body, timestamp });
  const signature = await signer.signMessage(message);
  
  return fetch(`https://api.viddhana.network${endpoint}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-Signature': signature,
      'X-Timestamp': timestamp.toString(),
      'X-Address': await signer.getAddress(),
    },
    body: message,
  });
}
```

---

## Rate Limiting

### Limits by Tier

| Tier | RPC Requests | API Requests | WebSocket Connections |
|------|--------------|--------------|----------------------|
| Free | 100/min | 20/min | 1 |
| Basic | 1,000/min | 100/min | 5 |
| Pro | 10,000/min | 1,000/min | 20 |
| Enterprise | Unlimited | Unlimited | Unlimited |

### Rate Limit Headers

```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 950
X-RateLimit-Reset: 1704067260
```

---

## Testing

### SDK Tests

```typescript
// packages/sdks/javascript/tests/ai.test.ts
import { describe, it, expect, beforeAll } from 'vitest';
import { ViddhanaClient } from '../src';

describe('PrometheusAI', () => {
  let client: ViddhanaClient;

  beforeAll(() => {
    client = new ViddhanaClient({
      network: 'testnet',
      apiKey: process.env.TEST_API_KEY,
    });
  });

  it('should get price prediction', async () => {
    const prediction = await client.ai.predictPrice('BTC', 7);
    
    expect(prediction.asset).toBe('BTC');
    expect(prediction.predictions).toHaveLength(7);
    expect(prediction.confidence).toHaveLength(7);
    expect(prediction.confidence.every(c => c >= 0 && c <= 1)).toBe(true);
  });

  it('should get portfolio optimization', async () => {
    const optimization = await client.ai.optimizePortfolio({
      userId: '0x1234',
      portfolio: { BTC: 50000, ETH: 30000 },
      riskTolerance: 0.5,
      timeToGoal: 24,
    });
    
    expect(optimization.action).toBeDefined();
    expect(optimization.confidence).toBeGreaterThan(0);
    expect(optimization.recommendations).toBeInstanceOf(Array);
  });

  it('should assess portfolio risk', async () => {
    const risk = await client.ai.assessRisk({
      BTC: 50000,
      ETH: 30000,
      USDC: 20000,
    });
    
    expect(risk.riskScore).toBeGreaterThanOrEqual(0);
    expect(risk.volatility).toBeGreaterThanOrEqual(0);
    expect(risk.sharpeRatio).toBeDefined();
  });
});
```

---

## Acceptance Criteria

```markdown
## API & SDK Acceptance Criteria

### JSON-RPC API
- [ ] All standard eth_* methods working
- [ ] Custom vdh_* methods implemented
- [ ] Error responses properly formatted
- [ ] Rate limiting functional

### JavaScript SDK
- [ ] npm package published
- [ ] TypeScript types exported
- [ ] All modules functional
- [ ] WebSocket support working

### Python SDK
- [ ] PyPI package published
- [ ] Async support implemented
- [ ] Type hints complete
- [ ] All modules functional

### Go SDK
- [ ] go mod compatible
- [ ] Context support for cancellation
- [ ] All modules functional
- [ ] Error handling complete

### Documentation
- [ ] API reference complete
- [ ] SDK quick start guides
- [ ] Code examples for all languages
- [ ] Error codes documented
```

---

*Document Version: 1.0.0*
