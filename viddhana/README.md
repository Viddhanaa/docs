# VIDDHANA

> The Operating System for Wealth - A comprehensive wealth management platform built on blockchain technology.

## Overview

VIDDHANA is a fully integrated platform combining AI-driven investment strategies with blockchain transparency, featuring:

- **Atlas Chain**: Layer 3 AppChain with 100,000+ TPS
- **Prometheus AI Engine**: LSTM + Transformer models for portfolio optimization
- **DeFi Integration**: Yield calculation and dynamic rebalancing
- **DePIN Infrastructure**: IoT sensor network for Real World Assets
- **SocialFi Layer**: Reputation-based community features

## Repository Structure

```
viddhana/
├── packages/
│   ├── contracts/          # Solidity smart contracts (Hardhat)
│   ├── api-server/         # REST/WebSocket API (Node.js)
│   ├── prometheus-ai/      # AI/ML engine (Python)
│   ├── depin-oracle/       # DePIN validator network (Go)
│   └── sdks/
│       ├── javascript/     # JavaScript/TypeScript SDK
│       ├── python/         # Python SDK
│       └── go/             # Go SDK
├── docs/                   # Docusaurus documentation
└── infrastructure/         # Terraform, Kubernetes, Docker configs
```

## Prerequisites

- Node.js >= 18.0.0
- pnpm >= 8.0.0
- Python >= 3.11
- Go >= 1.21
- Docker >= 24.0
- Docker Compose >= 2.20

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/viddhana/viddhana.git
cd viddhana

# Install dependencies
pnpm install
```

### 2. Environment Setup

```bash
cp .env.example .env
# Edit .env with your configuration
```

### 3. Start Local Infrastructure

```bash
# Start PostgreSQL, Redis, Hardhat node, Kafka, MQTT, MLflow
pnpm docker:up
```

### 4. Development

```bash
# Start all packages in development mode
pnpm dev

# Build all packages
pnpm build

# Run all tests
pnpm test

# Lint all packages
pnpm lint
```

## Available Scripts

| Script | Description |
|--------|-------------|
| `pnpm dev` | Start development servers |
| `pnpm build` | Build all packages |
| `pnpm test` | Run all tests |
| `pnpm lint` | Lint all packages |
| `pnpm clean` | Clean build artifacts |
| `pnpm docker:up` | Start Docker services |
| `pnpm docker:down` | Stop Docker services |

## Technology Stack

| Layer | Technology |
|-------|------------|
| Blockchain | Arbitrum Orbit / Cosmos SDK |
| Smart Contracts | Solidity 0.8.x |
| AI Engine | Python 3.11+ |
| API Server | Node.js / FastAPI |
| Database | PostgreSQL 15+ |
| Cache | Redis 7+ |
| Message Queue | Apache Kafka |
| IoT Protocol | MQTT 5.0 |
| Documentation | Docusaurus 3.x |

## Documentation

See the `/docs` directory for full documentation, or visit the documentation site.

## License

Proprietary - All rights reserved.
