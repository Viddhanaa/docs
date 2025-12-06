# @viddhana/sdk

Official JavaScript/TypeScript SDK for the VIDDHANA blockchain network.

## Installation

```bash
npm install @viddhana/sdk
# or
yarn add @viddhana/sdk
# or
pnpm add @viddhana/sdk
```

## Quick Start

```typescript
import { ViddhanaClient } from '@viddhana/sdk';

// Initialize client
const client = new ViddhanaClient({
  network: 'testnet', // or 'mainnet'
  privateKey: process.env.PRIVATE_KEY, // optional, for signing transactions
  apiKey: process.env.API_KEY, // optional, for AI endpoints
});

// Connect to the network
await client.connect();

// Get portfolio
const portfolio = await client.vault.getPortfolio('0x1234...');
console.log('Portfolio value:', portfolio.totalValue);
```

## Configuration

```typescript
interface ViddhanaConfig {
  network?: 'mainnet' | 'testnet'; // Default: 'mainnet'
  rpcUrl?: string;                 // Custom RPC URL (overrides network)
  apiKey?: string;                 // API key for authenticated endpoints
  privateKey?: string;             // Private key for signing transactions
}
```

## Modules

### Atlas Chain (`client.atlas`)

Blockchain interaction methods.

```typescript
// Get chain info
const info = await client.atlas.getChainInfo();
console.log(`Chain ID: ${info.chainId}, Block: ${info.blockNumber}`);

// Get balance
const balance = await client.atlas.getBalance('0x1234...');
console.log(`Balance: ${ethers.formatEther(balance)} VDH`);

// Get block
const block = await client.atlas.getBlock(12345);
console.log(`Block hash: ${block.hash}`);

// Get tokenomics stats
const stats = await client.atlas.getTokenomicsStats();
console.log(`Total supply: ${stats.totalSupply}`);

// Subscribe to new blocks
const unsubscribe = client.atlas.onBlock((blockNumber) => {
  console.log(`New block: ${blockNumber}`);
});
// Later: unsubscribe()
```

### Vault (`client.vault`)

Portfolio and vault operations.

```typescript
// Get portfolio
const portfolio = await client.vault.getPortfolio('0x1234...');
console.log(`Total Value: $${portfolio.totalValue}`);
console.log(`Assets: ${portfolio.assets.length}`);

for (const asset of portfolio.assets) {
  console.log(`  ${asset.symbol}: ${asset.allocation}%`);
}

// Create a vault
const tx = await client.vault.createVault(
  'My Vault',
  'MV',
  ['0xBTC...', '0xETH...', '0xUSDC...'],
  [5000, 3000, 2000] // 50% BTC, 30% ETH, 20% USDC
);
await tx.wait();

// Deposit assets
const depositTx = await client.vault.deposit(
  '0xUSDC...',
  ethers.parseUnits('1000', 6) // 1000 USDC
);
await depositTx.wait();

// Withdraw assets
const withdrawTx = await client.vault.withdraw(
  '0xUSDC...',
  ethers.parseUnits('500', 6) // 500 USDC
);
await withdrawTx.wait();

// Get rebalance history
const history = await client.vault.getRebalanceHistory('0x1234...', {
  limit: 10,
});

// Set user profile for auto-rebalancing
await client.vault.setProfile({
  riskTolerance: 5000, // 50% (basis points)
  timeToGoal: 24,      // 24 months
  autoRebalance: true,
});
```

### Prometheus AI (`client.ai`)

AI predictions and portfolio optimization.

```typescript
// Get price prediction
const prediction = await client.ai.predictPrice('BTC', 7);
console.log(`Trend: ${prediction.trend}`);
console.log(`Predictions:`);
for (const p of prediction.predictions) {
  console.log(`  Day ${p.day}: $${p.price} (${(p.confidence * 100).toFixed(1)}% confidence)`);
}

// Optimize portfolio
const optimization = await client.ai.optimizePortfolio({
  userId: '0x1234...',
  portfolio: { BTC: 50000, ETH: 30000, USDC: 20000 },
  riskTolerance: 0.5, // 0-1
  timeToGoal: 24,     // months
});
console.log(`Action: ${optimization.action}`);
console.log(`Confidence: ${(optimization.confidence * 100).toFixed(1)}%`);
for (const rec of optimization.recommendations) {
  console.log(`  ${rec.action} ${rec.percentage}% of ${rec.asset}`);
}

// Assess portfolio risk
const risk = await client.ai.assessRisk({
  BTC: 50000,
  ETH: 30000,
  USDC: 20000,
});
console.log(`Risk Level: ${risk.riskLevel}`);
console.log(`Volatility: ${(risk.metrics.volatility * 100).toFixed(1)}%`);
console.log(`Sharpe Ratio: ${risk.metrics.sharpeRatio.toFixed(2)}`);

// Stream real-time predictions
const unsubscribe = client.ai.streamPredictions(
  'BTC',
  (prediction) => {
    console.log('New prediction:', prediction);
  },
  (error) => {
    console.error('Stream error:', error);
  }
);
// Later: unsubscribe()
```

### Governance (`client.governance`)

Proposals, voting, and delegation.

```typescript
// Create a proposal
const proposalTx = await client.governance.createProposal(
  ['0xTargetContract...'],                    // targets
  [0n],                                       // values (ETH to send)
  ['0x...encoded function call...'],          // calldatas
  'Proposal to upgrade the protocol to v2.0'  // description
);
const receipt = await proposalTx.wait();

// Cast a vote (0 = Against, 1 = For, 2 = Abstain)
const voteTx = await client.governance.castVote(proposalId, 1);
await voteTx.wait();

// Cast vote with reason
const voteWithReasonTx = await client.governance.castVoteWithReason(
  proposalId,
  1,
  'This proposal improves security and efficiency'
);
await voteWithReasonTx.wait();

// Delegate voting power
const delegateTx = await client.governance.delegate('0xDelegate...');
await delegateTx.wait();

// Get proposal details
const proposal = await client.governance.getProposal(proposalId);
console.log(`State: ${proposal.state}`);
console.log(`For: ${proposal.forVotes}`);
console.log(`Against: ${proposal.againstVotes}`);

// Get voting power
const votingPower = await client.governance.getVotingPower('0x1234...');
console.log(`Voting Power: ${ethers.formatEther(votingPower)} VDH`);

// Get delegation info
const delegation = await client.governance.getDelegation('0x1234...');
console.log(`Delegated to: ${delegation.delegatee}`);
```

## Browser Usage

```typescript
import { ViddhanaClient } from '@viddhana/sdk';
import { ethers } from 'ethers';

// Initialize client
const client = new ViddhanaClient({
  network: 'mainnet',
  apiKey: 'your-api-key',
});

// Connect with MetaMask
const browserProvider = new ethers.BrowserProvider(window.ethereum);
const signer = await client.connectWallet(browserProvider);
console.log('Connected:', await signer.getAddress());
```

## TypeScript Support

The SDK is written in TypeScript and includes full type definitions:

```typescript
import {
  ViddhanaClient,
  ViddhanaConfig,
  PortfolioData,
  PricePrediction,
  Proposal,
  ProposalState,
} from '@viddhana/sdk';

// Full type support
const config: ViddhanaConfig = {
  network: 'testnet',
};

const client = new ViddhanaClient(config);

// Return types are fully typed
const portfolio: PortfolioData = await client.vault.getPortfolio('0x...');
const prediction: PricePrediction = await client.ai.predictPrice('BTC');
```

## Error Handling

```typescript
try {
  await client.connect();
  const portfolio = await client.vault.getPortfolio('0x...');
} catch (error) {
  if (error instanceof Error) {
    console.error('Error:', error.message);
  }
}
```

## API Reference

### ViddhanaClient

| Method | Description |
|--------|-------------|
| `connect()` | Connect to the network |
| `disconnect()` | Disconnect from the network |
| `isConnected()` | Check if connected |
| `getProvider()` | Get the ethers provider |
| `getSigner()` | Get the wallet signer |
| `getAddress()` | Get the connected wallet address |
| `getNetworkConfig()` | Get network configuration |
| `connectWallet(provider)` | Connect with browser wallet |
| `switchNetwork(network)` | Switch to a different network |

### AtlasChain

| Method | Description |
|--------|-------------|
| `getChainInfo()` | Get chain information |
| `getBalance(address)` | Get native token balance |
| `getFormattedBalance(address)` | Get formatted balance in VDH |
| `getBlock(blockHashOrNumber)` | Get block by number or hash |
| `getBlockNumber()` | Get latest block number |
| `getTransactionReceipt(txHash)` | Get transaction receipt |
| `getGasPrice()` | Get current gas price |
| `estimateGas(tx)` | Estimate gas for transaction |
| `getTransactionCount(address)` | Get nonce for address |
| `sendTransaction(signedTx)` | Send signed transaction |
| `waitForTransaction(txHash)` | Wait for confirmation |
| `getTokenomicsStats()` | Get tokenomics statistics |
| `onBlock(callback)` | Subscribe to new blocks |

### Vault

| Method | Description |
|--------|-------------|
| `getPortfolio(address)` | Get portfolio data |
| `getRebalanceHistory(address, options)` | Get rebalance history |
| `createVault(name, symbol, assets, allocations)` | Create a new vault |
| `deposit(asset, amount)` | Deposit to vault |
| `withdraw(asset, amount)` | Withdraw from vault |
| `getVaultInfo(vaultAddress)` | Get vault information |
| `getUserVaults(userAddress)` | Get user's vaults |
| `executeRebalance(actions)` | Execute manual rebalance |
| `setProfile(params)` | Set user profile |
| `getProfile(address)` | Get user profile |
| `getOptimizationRecommendation(address)` | Get AI recommendation |
| `getAllowance(token, owner, spender)` | Check token allowance |
| `getTokenBalance(token, owner)` | Get token balance |

### PrometheusAI

| Method | Description |
|--------|-------------|
| `predictPrice(asset, horizon)` | Get price prediction |
| `optimizePortfolio(params)` | Get optimization recommendation |
| `assessRisk(portfolio)` | Assess portfolio risk |
| `getRiskMetrics(portfolio)` | Get risk metrics only |
| `getSentiment(asset)` | Get market sentiment |
| `getCorrelation(assets)` | Get asset correlations |
| `getVolatilityForecast(asset, days)` | Get volatility forecast |
| `streamPredictions(asset, onUpdate, onError)` | Stream predictions |
| `streamPortfolio(address, onUpdate, onError)` | Stream portfolio updates |
| `batchPredictPrice(assets, horizon)` | Batch predictions |
| `getModelInfo()` | Get AI model info |

### Governance

| Method | Description |
|--------|-------------|
| `createProposal(targets, values, calldatas, description)` | Create proposal |
| `castVote(proposalId, support)` | Cast a vote |
| `castVoteWithReason(proposalId, support, reason)` | Vote with reason |
| `delegate(delegatee)` | Delegate voting power |
| `getProposal(proposalId)` | Get proposal details |
| `getProposalState(proposalId)` | Get proposal state |
| `getVoteReceipt(proposalId, account)` | Get vote receipt |
| `getDelegation(account)` | Get delegation info |
| `getVotingPower(account)` | Get voting power |
| `getPastVotingPower(account, blockNumber)` | Get past voting power |
| `queueProposal(proposalId)` | Queue proposal |
| `executeProposal(proposalId)` | Execute proposal |
| `cancelProposal(proposalId)` | Cancel proposal |
| `getVotingPeriod(proposalId)` | Get voting period |
| `hasVoted(proposalId, account)` | Check if voted |

## Testing

```bash
npm test
```

## Building

```bash
npm run build
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Links

- [Documentation](https://docs.viddhana.network/sdk/javascript)
- [API Reference](https://api.viddhana.network/docs)
- [GitHub](https://github.com/viddhana/sdk-js)
- [Discord](https://discord.gg/viddhana)
