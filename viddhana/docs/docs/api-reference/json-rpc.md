---
sidebar_position: 2
title: JSON-RPC Methods
---

# JSON-RPC Methods

Complete reference for all JSON-RPC methods supported by VIDDHANA.

## Chain Methods

### eth_chainId

Returns the current chain ID.

**Parameters:** None

**Returns:** `string` - Hex-encoded chain ID

```bash
# Request
curl -X POST https://rpc.viddhana.com \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"eth_chainId","params":[],"id":1}'

# Response
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": "0x1e61"  // 7777 in decimal (mainnet)
}
```

### eth_blockNumber

Returns the current block number.

**Parameters:** None

**Returns:** `string` - Hex-encoded block number

```bash
# Request
{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}

# Response
{"jsonrpc":"2.0","id":1,"result":"0x1a2b3c"}
```

## Account Methods

### eth_getBalance

Returns the balance of an account.

**Parameters:**
1. `address` - Account address
2. `block` - Block number or tag (`latest`, `pending`, `earliest`)

**Returns:** `string` - Balance in wei (hex)

```bash
# Request
{
  "jsonrpc": "2.0",
  "method": "eth_getBalance",
  "params": ["0x742d35Cc6634C0532925a3b844Bc9e7595f0Ab1F", "latest"],
  "id": 1
}

# Response
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": "0x56bc75e2d63100000"  // 100 VDH in wei
}
```

### eth_getTransactionCount

Returns the number of transactions sent from an address (nonce).

**Parameters:**
1. `address` - Account address
2. `block` - Block number or tag

**Returns:** `string` - Nonce (hex)

```bash
# Request
{
  "jsonrpc": "2.0",
  "method": "eth_getTransactionCount",
  "params": ["0x742d35Cc6634C0532925a3b844Bc9e7595f0Ab1F", "latest"],
  "id": 1
}

# Response
{"jsonrpc":"2.0","id":1,"result":"0x5"}
```

## Block Methods

### eth_getBlockByNumber

Returns block information by block number.

**Parameters:**
1. `blockNumber` - Block number (hex) or tag
2. `fullTransactions` - If `true`, returns full transaction objects

**Returns:** Block object or `null`

```bash
# Request
{
  "jsonrpc": "2.0",
  "method": "eth_getBlockByNumber",
  "params": ["latest", false],
  "id": 1
}

# Response
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "number": "0x1a2b3c",
    "hash": "0x...",
    "parentHash": "0x...",
    "timestamp": "0x...",
    "transactions": ["0x...", "0x..."],
    "gasUsed": "0x...",
    "gasLimit": "0x..."
  }
}
```

### eth_getBlockByHash

Returns block information by block hash.

**Parameters:**
1. `blockHash` - 32-byte block hash
2. `fullTransactions` - If `true`, returns full transaction objects

**Returns:** Block object or `null`

## Transaction Methods

### eth_sendRawTransaction

Submits a signed transaction to the network.

**Parameters:**
1. `signedTxData` - Signed transaction data (hex)

**Returns:** `string` - Transaction hash

```bash
# Request
{
  "jsonrpc": "2.0",
  "method": "eth_sendRawTransaction",
  "params": ["0xf86c..."],
  "id": 1
}

# Response
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": "0x88df016429689c079f3b2f6ad39fa052532c56795b733da78a91ebe6a713944b"
}
```

### eth_getTransactionByHash

Returns transaction information by hash.

**Parameters:**
1. `txHash` - Transaction hash

**Returns:** Transaction object or `null`

```bash
# Request
{
  "jsonrpc": "2.0",
  "method": "eth_getTransactionByHash",
  "params": ["0x88df016429689c079f3b2f6ad39fa052532c56795b733da78a91ebe6a713944b"],
  "id": 1
}

# Response
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "hash": "0x88df...",
    "from": "0x...",
    "to": "0x...",
    "value": "0x...",
    "gas": "0x...",
    "gasPrice": "0x...",
    "input": "0x...",
    "nonce": "0x...",
    "blockHash": "0x...",
    "blockNumber": "0x...",
    "transactionIndex": "0x0"
  }
}
```

### eth_getTransactionReceipt

Returns the receipt of a transaction.

**Parameters:**
1. `txHash` - Transaction hash

**Returns:** Receipt object or `null`

## Contract Methods

### eth_call

Executes a read-only contract call.

**Parameters:**
1. `callObject` - Transaction call object
2. `block` - Block number or tag

**Returns:** `string` - Return data (hex)

```bash
# Request
{
  "jsonrpc": "2.0",
  "method": "eth_call",
  "params": [
    {
      "to": "0xContractAddress",
      "data": "0x70a08231000000000000000000000000YourAddress"
    },
    "latest"
  ],
  "id": 1
}

# Response
{"jsonrpc":"2.0","id":1,"result":"0x0000000000000000000000000000000000000000000000056bc75e2d63100000"}
```

### eth_estimateGas

Estimates gas required for a transaction.

**Parameters:**
1. `callObject` - Transaction call object

**Returns:** `string` - Estimated gas (hex)

### eth_getLogs

Returns logs matching filter criteria.

**Parameters:**
1. `filterObject` - Filter parameters

**Returns:** Array of log objects

```bash
# Request
{
  "jsonrpc": "2.0",
  "method": "eth_getLogs",
  "params": [{
    "fromBlock": "0x1",
    "toBlock": "latest",
    "address": "0xContractAddress",
    "topics": ["0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"]
  }],
  "id": 1
}
```

## VIDDHANA-Specific Methods

### vdh_getValidators

Returns the current validator set.

**Parameters:** None

**Returns:** Array of validator objects

```bash
# Request
{"jsonrpc":"2.0","method":"vdh_getValidators","params":[],"id":1}

# Response
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": [
    {
      "address": "0x...",
      "stake": "1000000000000000000000",
      "commission": "0.05",
      "uptime": "99.9"
    }
  ]
}
```

### vdh_getStakingInfo

Returns staking statistics.

**Parameters:**
1. `address` (optional) - Staker address

**Returns:** Staking info object

### vdh_getAIPrediction

Query Prometheus AI for predictions.

**Parameters:**
1. `asset` - Asset symbol (e.g., "BTC", "ETH")
2. `horizon` - Prediction horizon in days

**Returns:** Prediction object

```bash
# Request
{
  "jsonrpc": "2.0",
  "method": "vdh_getAIPrediction",
  "params": ["BTC", 7],
  "id": 1
}

# Response
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "asset": "BTC",
    "currentPrice": "67500.00",
    "predictedPrice": "69200.00",
    "confidence": "0.78",
    "horizon": 7,
    "timestamp": "2024-01-15T12:00:00Z"
  }
}
```

---

See also: [API Overview](/docs/api-reference/overview) | [JavaScript SDK](/docs/sdks/javascript)
