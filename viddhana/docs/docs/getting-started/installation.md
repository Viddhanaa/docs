---
sidebar_position: 1
title: Installation
---

# Installation

Set up your development environment to build on VIDDHANA.

## Prerequisites

Ensure you have the following installed:

| Tool | Version | Purpose |
|------|---------|---------|
| Node.js | >= 18.0 | JavaScript runtime |
| pnpm/npm | Latest | Package manager |
| Git | >= 2.0 | Version control |
| Docker | >= 24.0 | Container runtime (optional) |

### Install Node.js

```bash
# Using nvm (recommended)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
nvm install 18
nvm use 18

# Verify installation
node --version  # v18.x.x
```

## Install the SDK

### JavaScript / TypeScript

```bash
# Using npm
npm install @viddhana/sdk

# Using pnpm
pnpm add @viddhana/sdk

# Using yarn
yarn add @viddhana/sdk
```

### Python

```bash
pip install viddhana
```

## Network Configuration

Add VIDDHANA networks to your configuration:

### Mainnet

| Property | Value |
|----------|-------|
| Network Name | VIDDHANA Atlas |
| RPC URL | `https://rpc.viddhana.com` |
| Chain ID | `13370` |
| Currency Symbol | VDH |
| Block Explorer | `https://scan.viddhana.com` |

### Testnet (Local Development)

| Property | Value |
|----------|-------|
| Network Name | VIDDHANA Testnet |
| RPC URL | `http://localhost:8545` |
| Chain ID | `1337` |
| Currency Symbol | VDH |

## Deployed Contract Addresses

### Testnet (Chain ID: 1337)

| Contract | Address |
|----------|---------|
| VDH Token | `0x384B37ab47B51f13D32fc2C19ea97147eC89fCD4` |
| VaultManager | `0xdC503c4E0F865C2cF198528354A8BCD19ffAF3F5` |
| PolicyEngine | `0xCD375A9355f765990b3f030B71C316e52a5353d2` |
| VDHGovernance | `0xAF53F4F1feAbea3aA9030b38Cac6dB68691BfD03` |

## Wallet Setup

### MetaMask Configuration

1. Open MetaMask and click on the network dropdown
2. Select "Add Network" → "Add a network manually"
3. Enter the network details from the table above
4. Click "Save"

### Programmatic Configuration

```typescript
import { ViddhanaClient } from '@viddhana/sdk';

const client = new ViddhanaClient({
  rpcUrl: 'https://rpc.viddhana.com',
  chainId: 13370,
});
```

## API Endpoints

| Service | URL | Description |
|---------|-----|-------------|
| RPC | https://rpc.viddhana.com | Blockchain JSON-RPC |
| API | https://api.viddhana.com | REST & JSON-RPC API |
| Docs | https://docs.viddhana.com | Documentation |
| Explorer | https://scan.viddhana.com | Block Explorer |

## Verify Installation

Test your setup with a simple script:

```typescript
import { ethers } from 'ethers';

async function main() {
  const provider = new ethers.JsonRpcProvider('https://rpc.viddhana.com');
  
  const blockNumber = await provider.getBlockNumber();
  console.log('Current block:', blockNumber);
  
  const network = await provider.getNetwork();
  console.log('Chain ID:', network.chainId);
}

main().catch(console.error);
```

### Using curl

```bash
# Check API health
curl https://api.viddhana.com/health

# Get chain info
curl -X POST https://api.viddhana.com/rpc \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"atlas_getChainInfo","params":[]}'
```

## Docker Setup (Optional)

Run a complete VIDDHANA development stack:

```bash
# Clone repository
git clone https://github.com/viddhana/viddhana.git
cd viddhana

# Start all services
docker-compose up -d

# Check status
docker-compose ps
```

### Services Started

| Service | Port | Description |
|---------|------|-------------|
| Blockchain Node | 8545 | Local RPC endpoint |
| API Server | 4000 | REST & JSON-RPC API |
| Block Explorer | 15000 | Transaction explorer |
| PostgreSQL | 5432 | Database |
| Redis | 6379 | Cache |

## Hardhat Configuration

Example `hardhat.config.ts` for VIDDHANA:

```typescript
import { HardhatUserConfig } from "hardhat/config";
import "@nomicfoundation/hardhat-toolbox";
import * as dotenv from "dotenv";

dotenv.config();

const config: HardhatUserConfig = {
  solidity: "0.8.20",
  networks: {
    viddhana: {
      url: "https://rpc.viddhana.com",
      chainId: 13370,
      accounts: process.env.PRIVATE_KEY ? [process.env.PRIVATE_KEY] : [],
    },
    localhost: {
      url: "http://localhost:8545",
      chainId: 1337,
    },
  },
};

export default config;
```

## Troubleshooting

### Common Issues

**Node.js version mismatch**
```bash
nvm use 18
```

**Connection refused**
- Ensure the RPC endpoint is accessible
- Check firewall settings
- Verify RPC URL is correct

**Transaction failed**
- Check account has sufficient VDH for gas
- Verify contract address is correct
- Ensure correct network is selected

**Invalid chain ID**
- Mainnet: 13370
- Testnet: 1337

---

Next: [Quickstart Guide](/docs/getting-started/quickstart)
