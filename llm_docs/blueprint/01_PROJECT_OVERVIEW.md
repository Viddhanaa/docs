# VIDDHANA Project Overview

> Implementation guide for setting up the VIDDHANA Developer Documentation Hub foundation

---

## Table of Contents
1. [Project Summary](#project-summary)
2. [Architecture Overview](#architecture-overview)
3. [Technology Stack](#technology-stack)
4. [Repository Structure](#repository-structure)
5. [Development Environment Setup](#development-environment-setup)
6. [Coding Standards](#coding-standards)

---

## Project Summary

VIDDHANA is a comprehensive wealth management platform built on blockchain technology, featuring:

- **Atlas Chain**: Layer 3 AppChain with 100,000+ TPS
- **Prometheus AI Engine**: LSTM + Transformer models for portfolio optimization
- **DeFi Integration**: Yield calculation and dynamic rebalancing
- **DePIN Infrastructure**: IoT sensor network for Real World Assets
- **SocialFi Layer**: Reputation-based community features

### Core Value Proposition
"The Operating System for Wealth" - A fully integrated platform combining AI-driven investment strategies with blockchain transparency.

---

## Architecture Overview

### Quad-Core Architecture Diagram

```
+------------------------------------------------------------------+
|                        VIDDHANA ECOSYSTEM                         |
+------------------------------------------------------------------+
|                                                                   |
|  +----------------+    +-------------------+    +---------------+ |
|  |   Frontend     |    |   Prometheus AI   |    |   Atlas L3    | |
|  |   (Docusaurus) |<-->|   Engine (Python) |<-->|   Chain       | |
|  +----------------+    +-------------------+    +---------------+ |
|         |                      |                       |          |
|         v                      v                       v          |
|  +----------------+    +-------------------+    +---------------+ |
|  |   API Gateway  |    |   ML Pipeline     |    |   Validators  | |
|  |   (REST/WS)    |    |   (Inference)     |    |   (21 nodes)  | |
|  +----------------+    +-------------------+    +---------------+ |
|         |                      |                       |          |
|         +----------------------+-----------------------+          |
|                                |                                  |
|  +----------------+    +-------------------+    +---------------+ |
|  |   Smart        |    |   Oracle          |    |   DePIN       | |
|  |   Contracts    |<-->|   Network         |<-->|   Sensors     | |
|  +----------------+    +-------------------+    +---------------+ |
|                                                                   |
+------------------------------------------------------------------+
                                 |
                                 v
                    +------------------------+
                    |   Ethereum L1          |
                    |   (Settlement Layer)   |
                    +------------------------+
```

### Data Flow

```
User Request --> API Gateway --> Route Decision
                                    |
                    +---------------+---------------+
                    |               |               |
                    v               v               v
              Read Data      AI Prediction    Execute TX
                    |               |               |
                    v               v               v
              PostgreSQL    Prometheus AI    Atlas Chain
                    |               |               |
                    +---------------+---------------+
                                    |
                                    v
                              Response to User
```

---

## Technology Stack

### Backend Infrastructure

| Layer | Technology | Version | Purpose |
|-------|------------|---------|---------|
| Blockchain | Arbitrum Orbit / Cosmos SDK | Latest | L3 AppChain |
| Smart Contracts | Solidity | 0.8.x | Business logic |
| AI Engine | Python | 3.11+ | ML models |
| API Server | Node.js / FastAPI | 20.x / 0.100+ | REST/WebSocket |
| Database | PostgreSQL | 15+ | Persistent storage |
| Cache | Redis | 7+ | Session/caching |
| Message Queue | Apache Kafka | 3.x | Event streaming |
| IoT Protocol | MQTT | 5.0 | DePIN sensors |

### Frontend Documentation

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| Framework | Docusaurus | 3.x | Documentation site |
| Styling | Tailwind CSS | 3.x | UI styling |
| Math Rendering | KaTeX | Latest | LaTeX formulas |
| Syntax Highlight | Prism | Latest | Code blocks |
| Search | Algolia DocSearch | Latest | Full-text search |

### DevOps & Tooling

| Tool | Purpose |
|------|---------|
| Docker | Containerization |
| Kubernetes | Orchestration |
| Terraform | Infrastructure as Code |
| GitHub Actions | CI/CD |
| Grafana + Prometheus | Monitoring |
| Hardhat | Smart contract development |
| MLflow | ML model versioning |

---

## Repository Structure

### Recommended Monorepo Layout

```
viddhana/
├── .github/
│   ├── workflows/
│   │   ├── ci.yml
│   │   ├── deploy-docs.yml
│   │   ├── smart-contracts.yml
│   │   └── ai-pipeline.yml
│   └── CODEOWNERS
│
├── packages/
│   ├── atlas-chain/                 # L3 blockchain configuration
│   │   ├── config/
│   │   │   ├── genesis.json
│   │   │   ├── chain-config.json
│   │   │   └── validators.json
│   │   ├── consensus/
│   │   ├── nodes/
│   │   └── scripts/
│   │
│   ├── contracts/                   # Smart contracts
│   │   ├── src/
│   │   │   ├── core/
│   │   │   │   ├── PolicyEngine.sol
│   │   │   │   ├── VaultManager.sol
│   │   │   │   └── RiskController.sol
│   │   │   ├── token/
│   │   │   │   ├── VDHToken.sol
│   │   │   │   └── Staking.sol
│   │   │   ├── governance/
│   │   │   │   └── Governance.sol
│   │   │   └── oracles/
│   │   │       └── OracleVerifier.sol
│   │   ├── test/
│   │   ├── scripts/
│   │   └── hardhat.config.ts
│   │
│   ├── prometheus-ai/              # AI Engine
│   │   ├── src/
│   │   │   ├── models/
│   │   │   │   ├── lstm_forecaster.py
│   │   │   │   ├── rl_optimizer.py
│   │   │   │   └── risk_assessor.py
│   │   │   ├── pipelines/
│   │   │   │   ├── data_ingestion.py
│   │   │   │   └── feature_engineering.py
│   │   │   ├── api/
│   │   │   │   └── inference_server.py
│   │   │   └── utils/
│   │   ├── tests/
│   │   ├── notebooks/
│   │   └── requirements.txt
│   │
│   ├── api-server/                  # REST/WebSocket API
│   │   ├── src/
│   │   │   ├── routes/
│   │   │   ├── middleware/
│   │   │   ├── services/
│   │   │   └── utils/
│   │   ├── tests/
│   │   └── package.json
│   │
│   ├── depin-oracle/               # DePIN validator network
│   │   ├── cmd/
│   │   ├── internal/
│   │   ├── pkg/
│   │   └── go.mod
│   │
│   └── sdks/                       # Client SDKs
│       ├── javascript/
│       │   ├── src/
│       │   └── package.json
│       ├── python/
│       │   ├── viddhana/
│       │   └── pyproject.toml
│       └── go/
│           ├── viddhana/
│           └── go.mod
│
├── docs/                           # Docusaurus documentation
│   ├── docusaurus.config.js
│   ├── sidebars.js
│   ├── src/
│   │   ├── components/
│   │   ├── css/
│   │   └── pages/
│   ├── docs/
│   │   ├── getting-started/
│   │   ├── atlas-chain/
│   │   ├── prometheus-ai/
│   │   ├── core-mechanics/
│   │   ├── smart-contracts/
│   │   ├── rwa-depin/
│   │   ├── tokenomics/
│   │   └── references/
│   └── static/
│
├── infrastructure/                 # IaC configurations
│   ├── terraform/
│   ├── kubernetes/
│   └── docker/
│
├── scripts/                        # Utility scripts
│   ├── setup.sh
│   ├── deploy.sh
│   └── test-all.sh
│
├── turbo.json                      # Turborepo config
├── package.json                    # Root package.json
├── pnpm-workspace.yaml
└── README.md
```

---

## Development Environment Setup

### Prerequisites

```bash
# Required software versions
node >= 20.0.0
pnpm >= 8.0.0
python >= 3.11
go >= 1.21
docker >= 24.0
docker-compose >= 2.20
```

### Step 1: Clone and Install

```bash
# Clone repository
git clone https://github.com/viddhana/viddhana.git
cd viddhana

# Install dependencies (using pnpm workspaces)
pnpm install

# Install Python dependencies
cd packages/prometheus-ai
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

### Step 2: Environment Configuration

```bash
# Copy environment templates
cp .env.example .env
cp packages/contracts/.env.example packages/contracts/.env
cp packages/prometheus-ai/.env.example packages/prometheus-ai/.env

# Required environment variables
# .env
DATABASE_URL=postgresql://user:pass@localhost:5432/viddhana
REDIS_URL=redis://localhost:6379
KAFKA_BROKERS=localhost:9092
MQTT_BROKER=mqtt://localhost:1883

# packages/contracts/.env
PRIVATE_KEY=<deployer_private_key>
ATLAS_RPC_URL=https://rpc.testnet.viddhana.network
ETHERSCAN_API_KEY=<api_key>

# packages/prometheus-ai/.env
MODEL_PATH=/models
MLFLOW_TRACKING_URI=http://localhost:5000
```

### Step 3: Local Infrastructure

```bash
# Start local services
docker-compose up -d

# This starts:
# - PostgreSQL (port 5432)
# - Redis (port 6379)
# - Kafka (port 9092)
# - MQTT Broker (port 1883)
# - Local Hardhat node (port 8545)
```

### Step 4: Verify Setup

```bash
# Run all tests
pnpm test

# Run specific package tests
pnpm --filter contracts test
pnpm --filter prometheus-ai test
pnpm --filter api-server test

# Start development servers
pnpm dev
```

---

## Coding Standards

### Solidity Standards

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title Contract Title
 * @author VIDDHANA Team
 * @notice User-facing description
 * @dev Technical implementation notes
 */
contract ExampleContract {
    // State variables: private by default, explicit visibility
    uint256 private _value;
    
    // Events: PascalCase, past tense
    event ValueUpdated(uint256 indexed oldValue, uint256 indexed newValue);
    
    // Errors: custom errors for gas efficiency
    error InvalidValue(uint256 provided, uint256 minimum);
    
    // Modifiers: camelCase
    modifier onlyPositive(uint256 value) {
        if (value == 0) revert InvalidValue(value, 1);
        _;
    }
    
    // Functions: explicit visibility, NatSpec comments
    /**
     * @notice Sets a new value
     * @param newValue The value to set
     * @return success Whether the operation succeeded
     */
    function setValue(uint256 newValue) 
        external 
        onlyPositive(newValue) 
        returns (bool success) 
    {
        uint256 oldValue = _value;
        _value = newValue;
        emit ValueUpdated(oldValue, newValue);
        return true;
    }
}
```

### TypeScript Standards

```typescript
// Use strict mode
'use strict';

// Explicit types, no `any`
interface UserProfile {
  readonly id: string;
  riskTolerance: number;
  timeToGoal: number; // months
}

// Async/await, proper error handling
async function getUserProfile(userId: string): Promise<UserProfile> {
  try {
    const response = await fetch(`/api/users/${userId}/profile`);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    return response.json() as Promise<UserProfile>;
  } catch (error) {
    logger.error('Failed to fetch user profile', { userId, error });
    throw error;
  }
}

// Export named exports, not default
export { getUserProfile, type UserProfile };
```

### Python Standards

```python
"""Module docstring describing the purpose."""

from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PricePrediction:
    """Immutable prediction result from Prometheus AI."""
    
    asset: str
    horizon_days: int
    predicted_price: float
    confidence: float
    timestamp: int


class PrometheusPredictor:
    """LSTM-based price prediction model.
    
    Attributes:
        model_path: Path to the trained model weights.
        window_size: Number of days for historical context.
    """
    
    def __init__(self, model_path: str, window_size: int = 30) -> None:
        self.model_path = model_path
        self.window_size = window_size
        self._model: Optional[torch.nn.Module] = None
    
    def predict(self, asset: str, horizon: int) -> PricePrediction:
        """Generate price prediction for an asset.
        
        Args:
            asset: Asset symbol (e.g., "BTC", "ETH").
            horizon: Number of days ahead to predict.
        
        Returns:
            PricePrediction object with results.
        
        Raises:
            ModelNotLoadedError: If model hasn't been initialized.
            InvalidAssetError: If asset is not supported.
        """
        logger.info("Generating prediction", extra={"asset": asset, "horizon": horizon})
        # Implementation...
```

### Git Commit Standards

```
type(scope): subject

body

footer
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Code restructure
- `test`: Tests
- `chore`: Maintenance

**Example:**
```
feat(contracts): implement PolicyEngine auto-rebalancing

- Add risk tolerance threshold checks
- Implement inflation protection logic
- Add time-to-goal based volatility reduction

Closes #123
Tracker: Updates PolicyEngine status to COMPLETED
```

---

## Next Steps

After completing this setup:

1. Proceed to `02_ATLAS_CHAIN_IMPLEMENTATION.md` for blockchain setup
2. Proceed to `03_PROMETHEUS_AI_ENGINE.md` for AI model development
3. Update `TRACKER.md` with progress

---

*Document Version: 1.0.0*
