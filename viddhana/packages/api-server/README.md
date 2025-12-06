# VIDDHANA API Server

Production-ready TypeScript API server providing JSON-RPC, REST, and WebSocket interfaces for the VIDDHANA Network.

## Features

- **JSON-RPC 2.0** - Full Ethereum-compatible RPC + custom VIDDHANA methods
- **REST API** - RESTful endpoints for all services
- **WebSocket** - Real-time subscriptions via Socket.IO
- **Authentication** - API key and JWT-based auth
- **Rate Limiting** - Tiered rate limits (Free, Basic, Pro, Enterprise)
- **Database** - PostgreSQL with Prisma ORM
- **Docker** - Production-ready containerization

## Quick Start

```bash
# Install dependencies
npm install

# Generate Prisma client
npm run prisma:generate

# Run database migrations
npm run prisma:migrate

# Start development server
npm run dev

# Build for production
npm run build

# Start production server
npm start
```

## Environment Variables

Create a `.env` file:

```env
# Server
PORT=3000
HOST=0.0.0.0
NODE_ENV=development

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/viddhana

# Blockchain
RPC_URL=http://localhost:8545
CHAIN_ID=13370

# Authentication
JWT_SECRET=your-super-secret-jwt-key

# Redis (optional, for distributed rate limiting)
REDIS_URL=redis://localhost:6379

# Logging
LOG_LEVEL=info

# CORS
CORS_ORIGINS=http://localhost:3000,http://localhost:8080
```

## API Endpoints

### JSON-RPC (POST /rpc)

#### Atlas Chain Methods

| Method | Description |
|--------|-------------|
| `atlas_getChainInfo` | Get chain information |
| `atlas_getBalance` | Get account balance |
| `atlas_getBlock` | Get block by number or hash |

#### Vault Methods

| Method | Description |
|--------|-------------|
| `vault_create` | Create a new vault |
| `vault_deposit` | Deposit assets to vault |
| `vault_withdraw` | Withdraw assets from vault |
| `vault_getVault` | Get vault information |

#### AI Methods

| Method | Description |
|--------|-------------|
| `ai_getPrediction` | Get AI price prediction |
| `ai_optimizePortfolio` | Get portfolio optimization |

#### VDH Custom Methods

| Method | Description |
|--------|-------------|
| `vdh_getPortfolio` | Get user portfolio |
| `vdh_getRebalanceHistory` | Get rebalance history |
| `vdh_getAIPrediction` | Get AI prediction |
| `vdh_getSensorRewards` | Get DePIN sensor rewards |
| `vdh_getTokenomicsStats` | Get tokenomics statistics |

#### Standard Ethereum Methods

| Method | Description |
|--------|-------------|
| `eth_chainId` | Get chain ID |
| `eth_blockNumber` | Get current block number |
| `eth_getBalance` | Get account balance |
| `eth_gasPrice` | Get current gas price |
| `eth_getTransactionCount` | Get transaction count |
| `eth_sendRawTransaction` | Send signed transaction |
| `eth_getTransactionReceipt` | Get transaction receipt |
| `eth_call` | Call contract method |
| `eth_estimateGas` | Estimate gas for transaction |

### REST API (GET/POST /api/v1/*)

#### Chain Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/status` | GET | API status |
| `/api/v1/chain/info` | GET | Chain information |
| `/api/v1/chain/blocks/:blockNumber` | GET | Get block |
| `/api/v1/chain/transactions/:txHash` | GET | Get transaction |

#### Account Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/accounts/:address/balance` | GET | Get balance |
| `/api/v1/accounts/:address/transactions` | GET | Get transactions |

#### Portfolio Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/portfolio/:address` | GET | Get portfolio |
| `/api/v1/portfolio/:address/history` | GET | Get history |

#### Vault Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/vaults` | GET | List vaults |
| `/api/v1/vaults` | POST | Create vault |
| `/api/v1/vaults/:vaultId` | GET | Get vault |
| `/api/v1/vaults/:vaultId/deposit` | POST | Deposit |
| `/api/v1/vaults/:vaultId/withdraw` | POST | Withdraw |

#### AI Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/predict/price` | POST | Price prediction |
| `/api/v1/optimize/portfolio` | POST | Portfolio optimization |
| `/api/v1/assess/risk` | POST | Risk assessment |

#### DePIN Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/sensors/:sensorId` | GET | Get sensor info |
| `/api/v1/sensors/:sensorId/rewards` | GET | Get rewards |

#### Tokenomics Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/tokenomics/stats` | GET | Tokenomics stats |
| `/api/v1/staking/stats` | GET | Staking stats |

### WebSocket (wss://...)

Connect to the WebSocket server and subscribe to channels:

```javascript
const socket = io('wss://api.viddhana.network', {
  auth: { apiKey: 'your-api-key' }
});

// Subscribe to portfolio updates
socket.emit('subscribe', {
  type: 'subscribe',
  channel: 'portfolio',
  params: { address: '0x...' }
});

// Listen for messages
socket.on('message', (data) => {
  console.log(data.channel, data.payload);
});
```

#### Available Channels

| Channel | Description |
|---------|-------------|
| `portfolio` | Portfolio value updates |
| `predictions` | AI prediction updates |
| `rebalance` | Rebalancing events |
| `rewards` | DePIN reward updates |
| `blocks` | New block notifications |
| `prices` | Price updates |

## Rate Limits

| Tier | RPC Requests | API Requests | WebSocket Connections |
|------|--------------|--------------|----------------------|
| Free | 100/min | 20/min | 1 |
| Basic | 1,000/min | 100/min | 5 |
| Pro | 10,000/min | 1,000/min | 20 |
| Enterprise | Unlimited | Unlimited | Unlimited |

## Authentication

### API Key

Include in requests:

```bash
# Header
Authorization: Bearer YOUR_API_KEY
# or
X-API-Key: YOUR_API_KEY
```

### Signed Requests

For sensitive operations:

```bash
curl -X POST https://api.viddhana.network/api/v1/vaults \
  -H "Content-Type: application/json" \
  -H "X-Signature: 0x..." \
  -H "X-Timestamp: 1704067200000" \
  -H "X-Address: 0x1234..." \
  -d '{"name": "My Vault", "riskTolerance": 5000}'
```

## Docker

```bash
# Build image
docker build -t viddhana-api-server .

# Run container
docker run -d \
  -p 3000:3000 \
  -e DATABASE_URL="postgresql://..." \
  -e JWT_SECRET="..." \
  -e RPC_URL="http://..." \
  viddhana-api-server
```

## Project Structure

```
api-server/
├── src/
│   ├── index.ts              # Entry point
│   ├── server.ts             # Express + Socket.IO setup
│   ├── routes/
│   │   ├── rpc.ts            # JSON-RPC handler
│   │   └── rest.ts           # REST API routes
│   ├── middleware/
│   │   ├── auth.ts           # Authentication
│   │   └── rateLimit.ts      # Rate limiting
│   ├── services/
│   │   ├── blockchain.ts     # Ethers.js service
│   │   └── logger.ts         # Winston logger
│   ├── websocket/
│   │   └── manager.ts        # WebSocket subscriptions
│   └── types/
│       └── index.ts          # TypeScript types
├── prisma/
│   └── schema.prisma         # Database schema
├── Dockerfile
├── package.json
├── tsconfig.json
└── README.md
```

## License

MIT License - VIDDHANA Team
