---
sidebar_position: 1
title: Overview
---

# API Reference Overview

VIDDHANA provides a comprehensive JSON-RPC API compatible with Ethereum tooling.

## Endpoints

| Network | Endpoint |
|---------|----------|
| Mainnet | `https://rpc.viddhana.com` |
| Testnet | `http://localhost:8545` |
| API Server | `https://api.viddhana.com` |
| WebSocket (Mainnet) | `wss://ws.viddhana.com` |

## Authentication

Public endpoints are available without authentication. For higher rate limits, include an API key:

```bash
curl -X POST https://rpc.viddhana.com \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}'
```

## Rate Limits

| Tier | Requests/Second | Daily Limit |
|------|-----------------|-------------|
| Free | 10 | 100,000 |
| Developer | 50 | 1,000,000 |
| Enterprise | Unlimited | Unlimited |

## Supported Methods

### Standard Ethereum Methods

VIDDHANA supports all standard Ethereum JSON-RPC methods:

- `eth_chainId`
- `eth_blockNumber`
- `eth_getBalance`
- `eth_getTransactionCount`
- `eth_getBlockByNumber`
- `eth_getBlockByHash`
- `eth_getTransactionByHash`
- `eth_getTransactionReceipt`
- `eth_sendRawTransaction`
- `eth_call`
- `eth_estimateGas`
- `eth_getLogs`
- And more...

### VIDDHANA-Specific Methods

| Method | Description |
|--------|-------------|
| `vdh_getValidators` | Get current validator set |
| `vdh_getStakingInfo` | Get staking statistics |
| `vdh_getAIPrediction` | Query Prometheus AI predictions |
| `vdh_getDePINNodes` | List active DePIN nodes |

## Response Format

All responses follow the JSON-RPC 2.0 specification:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": "0x..."
}
```

Error responses:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "error": {
    "code": -32600,
    "message": "Invalid Request"
  }
}
```

## Error Codes

| Code | Message | Description |
|------|---------|-------------|
| -32700 | Parse error | Invalid JSON |
| -32600 | Invalid Request | Invalid JSON-RPC request |
| -32601 | Method not found | Unknown method |
| -32602 | Invalid params | Invalid method parameters |
| -32603 | Internal error | Internal JSON-RPC error |
| -32000 | Server error | Generic server error |
| -32001 | Resource not found | Block, tx, etc. not found |
| -32002 | Resource unavailable | Resource temporarily unavailable |
| -32003 | Transaction rejected | Transaction rejected by node |

## Quick Examples

### Get Current Block Number

```bash
curl -X POST https://rpc.viddhana.com \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "eth_blockNumber",
    "params": [],
    "id": 1
  }'
```

### Get Account Balance

```bash
curl -X POST https://rpc.viddhana.com \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "eth_getBalance",
    "params": ["0xYourAddress", "latest"],
    "id": 1
  }'
```

### Send Transaction

```bash
curl -X POST https://rpc.viddhana.com \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "eth_sendRawTransaction",
    "params": ["0xSignedTransactionData"],
    "id": 1
  }'
```

## WebSocket Subscriptions

Connect to WebSocket endpoint for real-time updates:

```javascript
const ws = new WebSocket('wss://ws.viddhana.com');

// Subscribe to new blocks
ws.send(JSON.stringify({
  jsonrpc: '2.0',
  method: 'eth_subscribe',
  params: ['newHeads'],
  id: 1
}));

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('New block:', data.params.result);
};
```

---

Next: [JSON-RPC Methods](/docs/api-reference/json-rpc)
