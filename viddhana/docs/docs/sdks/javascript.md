---
sidebar_position: 1
title: JavaScript SDK
---

# JavaScript SDK

The official JavaScript/TypeScript SDK for building on VIDDHANA.

## Installation

```bash
npm install @viddhana/sdk
```

Or with pnpm:

```bash
pnpm add @viddhana/sdk
```

Or with yarn:

```bash
yarn add @viddhana/sdk
```

## Quick Start

```typescript
import { ViddhanaSDK } from '@viddhana/sdk';

// Initialize the SDK
const sdk = new ViddhanaSDK({
  network: 'testnet', // or 'mainnet'
  privateKey: process.env.PRIVATE_KEY, // optional for read-only operations
});

// Get current block number
const blockNumber = await sdk.getBlockNumber();
console.log('Current block:', blockNumber);
```

## Configuration

### Constructor Options

| Option | Type | Required | Description |
|--------|------|----------|-------------|
| `network` | `'mainnet' \| 'testnet'` | Yes | Network to connect to |
| `privateKey` | `string` | No | Private key for signing transactions |
| `rpcUrl` | `string` | No | Custom RPC URL |
| `apiKey` | `string` | No | API key for higher rate limits |

### Example Configuration

```typescript
const sdk = new ViddhanaSDK({
  network: 'mainnet',
  privateKey: process.env.PRIVATE_KEY,
  apiKey: process.env.VIDDHANA_API_KEY,
});
```

## Core Methods

### Reading Chain Data

```typescript
// Get block number
const blockNumber = await sdk.getBlockNumber();

// Get chain ID
const chainId = await sdk.getChainId();

// Get balance
const balance = await sdk.getBalance('0xAddress');

// Get transaction
const tx = await sdk.getTransaction('0xTxHash');

// Get block
const block = await sdk.getBlock('latest');
```

### Account Operations

```typescript
// Get nonce
const nonce = await sdk.getNonce('0xAddress');

// Get transaction count
const txCount = await sdk.getTransactionCount('0xAddress');
```

### Sending Transactions

```typescript
// Transfer VDH
const tx = await sdk.transfer({
  to: '0xRecipientAddress',
  amount: '10', // in VDH
});

await tx.wait();
console.log('Transaction hash:', tx.hash);
```

### Contract Interaction

```typescript
// Read contract data
const result = await sdk.readContract({
  address: '0xContractAddress',
  abi: contractABI,
  functionName: 'balanceOf',
  args: ['0xAddress'],
});

// Write to contract
const tx = await sdk.writeContract({
  address: '0xContractAddress',
  abi: contractABI,
  functionName: 'transfer',
  args: ['0xRecipient', '1000000000000000000'],
});

await tx.wait();
```

## VDH Token Methods

```typescript
// Get VDH balance
const balance = await sdk.getVDHBalance('0xAddress');

// Transfer VDH
const tx = await sdk.transferVDH('0xRecipient', '100');

// Approve spending
const tx = await sdk.approveVDH('0xSpender', '1000');

// Sign permit (gasless approval)
const permit = await sdk.signPermit({
  spender: '0xSpender',
  value: '1000',
  deadline: Math.floor(Date.now() / 1000) + 3600,
});
```

## Governance Methods

```typescript
// Delegate voting power
await sdk.delegate('0xDelegateAddress');

// Get voting power
const votes = await sdk.getVotes('0xAddress');

// Get past votes at block
const pastVotes = await sdk.getPastVotes('0xAddress', blockNumber);
```

## WebSocket Subscriptions

```typescript
// Subscribe to new blocks
const unsubscribe = sdk.subscribeToBlocks((block) => {
  console.log('New block:', block.number);
});

// Subscribe to pending transactions
const unsubscribe = sdk.subscribeToPendingTransactions((txHash) => {
  console.log('Pending tx:', txHash);
});

// Subscribe to logs
const unsubscribe = sdk.subscribeToLogs(
  { address: '0xContract', topics: [] },
  (log) => {
    console.log('New log:', log);
  }
);

// Unsubscribe
unsubscribe();
```

## Error Handling

```typescript
import { ViddhanaError, ErrorCodes } from '@viddhana/sdk';

try {
  await sdk.transfer({ to: '0x...', amount: '100' });
} catch (error) {
  if (error instanceof ViddhanaError) {
    switch (error.code) {
      case ErrorCodes.INSUFFICIENT_FUNDS:
        console.log('Not enough VDH');
        break;
      case ErrorCodes.NONCE_TOO_LOW:
        console.log('Nonce too low, retry');
        break;
      default:
        console.log('Error:', error.message);
    }
  }
}
```

## TypeScript Support

The SDK is written in TypeScript and includes full type definitions:

```typescript
import type { 
  Block, 
  Transaction, 
  TransactionReceipt,
  Log,
  ViddhanaConfig 
} from '@viddhana/sdk';

const config: ViddhanaConfig = {
  network: 'testnet',
};

const sdk = new ViddhanaSDK(config);
```

## Best Practices

1. **Environment Variables** - Never hardcode private keys
2. **Error Handling** - Always wrap transactions in try/catch
3. **Gas Estimation** - Use `estimateGas()` before sending transactions
4. **Nonce Management** - Track nonces for multiple transactions
5. **Connection Pooling** - Reuse SDK instances when possible

## Examples

### Full Transfer Example

```typescript
import { ViddhanaSDK } from '@viddhana/sdk';

async function main() {
  const sdk = new ViddhanaSDK({
    network: 'testnet',
    privateKey: process.env.PRIVATE_KEY,
  });

  // Check balance
  const balance = await sdk.getBalance(sdk.address);
  console.log('Balance:', balance, 'VDH');

  // Transfer
  const tx = await sdk.transfer({
    to: '0xRecipientAddress',
    amount: '1',
  });

  console.log('Transaction sent:', tx.hash);
  
  const receipt = await tx.wait();
  console.log('Confirmed in block:', receipt.blockNumber);
}

main().catch(console.error);
```

### Deploy Contract Example

```typescript
import { ViddhanaSDK } from '@viddhana/sdk';
import { ContractFactory } from 'ethers';

async function deployContract() {
  const sdk = new ViddhanaSDK({
    network: 'testnet',
    privateKey: process.env.PRIVATE_KEY,
  });

  const factory = new ContractFactory(abi, bytecode, sdk.signer);
  const contract = await factory.deploy('Constructor Arg');
  
  await contract.waitForDeployment();
  console.log('Deployed to:', await contract.getAddress());
}
```

---

See also: [Python SDK](/docs/sdks/python) | [API Reference](/docs/api-reference/overview)
