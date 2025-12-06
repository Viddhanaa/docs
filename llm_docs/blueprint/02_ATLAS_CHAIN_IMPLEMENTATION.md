# Atlas Chain Implementation Guide

> Detailed implementation guide for the VIDDHANA Layer 3 AppChain

---

## Table of Contents
1. [Overview](#overview)
2. [Chain Specifications](#chain-specifications)
3. [Architecture Design](#architecture-design)
4. [Consensus Implementation](#consensus-implementation)
5. [Node Setup](#node-setup)
6. [Gas & Fee Logic](#gas--fee-logic)
7. [Bridge Implementation](#bridge-implementation)
8. [Testing & Validation](#testing--validation)

---

## Overview

Atlas Chain is a specialized Layer 3 AppChain designed for high-frequency asset management operations. It settles on Ethereum L1 via Arbitrum L2 for data availability and finality.

### Key Performance Targets

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| Throughput | 100,000+ TPS | Sustained load test |
| Block Time | ~2 seconds | Block timestamp delta |
| Finality | 6 seconds | 6 block confirmations |
| Tx Cost | < $0.001 | Gas price * gas used |

---

## Chain Specifications

### Network Configuration

```json
{
  "chainId": 13370,
  "chainName": "Atlas Chain",
  "nativeCurrency": {
    "name": "Viddhana Token",
    "symbol": "VDH",
    "decimals": 18
  },
  "rpcUrls": {
    "testnet": "https://rpc.testnet.viddhana.network",
    "mainnet": "https://rpc.viddhana.network"
  },
  "blockExplorerUrls": {
    "testnet": "https://explorer.testnet.viddhana.network",
    "mainnet": "https://explorer.viddhana.network"
  }
}
```

### Genesis Configuration

```json
{
  "config": {
    "chainId": 13370,
    "homesteadBlock": 0,
    "eip150Block": 0,
    "eip155Block": 0,
    "eip158Block": 0,
    "byzantiumBlock": 0,
    "constantinopleBlock": 0,
    "petersburgBlock": 0,
    "istanbulBlock": 0,
    "berlinBlock": 0,
    "londonBlock": 0,
    "clique": {
      "period": 2,
      "epoch": 30000
    }
  },
  "difficulty": "1",
  "gasLimit": "30000000",
  "extradata": "0x0000000000000000000000000000000000000000000000000000000000000000<SIGNER_ADDRESSES>0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
  "alloc": {
    "0x...": { "balance": "1000000000000000000000000000" }
  }
}
```

---

## Architecture Design

### Layer Stack

```
+------------------------------------------------------------------+
|                         USER APPLICATIONS                         |
+------------------------------------------------------------------+
                                 |
                                 v
+------------------------------------------------------------------+
|                      ATLAS CHAIN (L3)                            |
|  +----------------+  +----------------+  +--------------------+  |
|  |   EVM Engine   |  |   Consensus    |  |   State Manager    |  |
|  |   (geth/reth)  |  |   (PoA + PoS)  |  |   (LevelDB/MDBX)   |  |
|  +----------------+  +----------------+  +--------------------+  |
|                                                                   |
|  +----------------+  +----------------+  +--------------------+  |
|  |   TX Pool      |  |   Block        |  |   RPC Server       |  |
|  |   (Priority)   |  |   Producer     |  |   (JSON-RPC 2.0)   |  |
|  +----------------+  +----------------+  +--------------------+  |
+------------------------------------------------------------------+
                                 |
                                 v (Batch Submission)
+------------------------------------------------------------------+
|                      ARBITRUM ONE (L2)                           |
|  +--------------------------------------------------------------+|
|  |   Data Availability Layer (Calldata/Blobs)                   ||
|  +--------------------------------------------------------------+|
+------------------------------------------------------------------+
                                 |
                                 v (Settlement)
+------------------------------------------------------------------+
|                      ETHEREUM MAINNET (L1)                       |
|  +--------------------------------------------------------------+|
|  |   Security & Finality                                        ||
|  +--------------------------------------------------------------+|
+------------------------------------------------------------------+
```

### Component Responsibilities

| Component | Responsibility | Technology |
|-----------|----------------|------------|
| EVM Engine | Smart contract execution | go-ethereum/reth |
| Consensus Module | Block validation & production | Custom PoA+PoS |
| State Manager | State storage & retrieval | LevelDB/MDBX |
| TX Pool | Transaction ordering | Priority-based |
| Block Producer | Block creation & signing | Validator software |
| RPC Server | External API | JSON-RPC 2.0 |
| Sequencer | Batch submission to L2 | Custom bridge |

---

## Consensus Implementation

### Hybrid PoA + PoS Design

Atlas Chain uses a hybrid consensus mechanism:
- **PoA (Proof of Authority)**: For block production (fast)
- **PoS (Proof of Stake)**: For economic security (stake-based)

### Validator Set

```typescript
interface ValidatorConfig {
  totalValidators: 21;
  categories: {
    infrastructure: 11;  // Cloud providers, data centers
    ventureCapital: 7;   // Strategic investors
    community: 3;        // Elected by token holders
  };
  minimumStake: "1000000000000000000000000"; // 1M VDH
  slashingConditions: {
    doubleSign: "100%";   // Full stake slashed
    downtime: "5%";       // Per day of downtime
    maliciousData: "50%"; // For oracle manipulation
  };
}
```

### Block Production Algorithm

```python
# Simplified block production logic
class BlockProducer:
    def __init__(self, validators: List[Validator]):
        self.validators = validators
        self.current_proposer_index = 0
        self.block_time = 2  # seconds
    
    def get_next_proposer(self, block_number: int) -> Validator:
        """Round-robin proposer selection with weighted probability."""
        # Weight by stake for tie-breaking
        weights = [v.stake for v in self.validators]
        total_weight = sum(weights)
        
        # Deterministic selection based on block number
        index = block_number % len(self.validators)
        return self.validators[index]
    
    def validate_block(self, block: Block) -> bool:
        """Validate block meets consensus requirements."""
        checks = [
            self._check_proposer(block),
            self._check_timestamp(block),
            self._check_transactions(block),
            self._check_signatures(block),
        ]
        return all(checks)
    
    def _check_proposer(self, block: Block) -> bool:
        expected = self.get_next_proposer(block.number)
        return block.proposer == expected.address
    
    def _check_timestamp(self, block: Block) -> bool:
        return block.timestamp >= self.last_block.timestamp + self.block_time
```

### Finality Rules

```
Block Confirmation Timeline:
+--------+--------+--------+--------+--------+--------+
| Block  | Block  | Block  | Block  | Block  | Block  |  FINAL
|   N    |  N+1   |  N+2   |  N+3   |  N+4   |  N+5   |
+--------+--------+--------+--------+--------+--------+
|  0s    |  2s    |  4s    |  6s    |  8s    |  10s   |  12s
                                                        ^
                                                     Finality
                                                    (6 blocks)
```

---

## Node Setup

### Hardware Requirements

| Node Type | CPU | RAM | Storage | Network |
|-----------|-----|-----|---------|---------|
| Validator | 32 cores | 64 GB | 2 TB NVMe | 1 Gbps |
| Full Node | 16 cores | 32 GB | 1 TB NVMe | 500 Mbps |
| Archive | 32 cores | 128 GB | 8 TB NVMe | 1 Gbps |
| RPC Node | 16 cores | 32 GB | 1 TB NVMe | 1 Gbps |

### Validator Node Setup

```bash
#!/bin/bash
# validator-setup.sh

# 1. Install dependencies
apt update && apt install -y build-essential git golang-go

# 2. Clone Atlas Chain repository
git clone https://github.com/viddhana/atlas-chain.git
cd atlas-chain

# 3. Build the node binary
make build

# 4. Initialize the node
./atlas init --datadir /data/atlas --network testnet

# 5. Import validator key
./atlas account import --datadir /data/atlas --keyfile validator.key

# 6. Configure the node
cat > /data/atlas/config.toml << EOF
[Node]
DataDir = "/data/atlas"
NetworkId = 13370
SyncMode = "snap"

[Node.P2P]
MaxPeers = 100
ListenAddr = ":30303"
BootstrapNodes = [
  "enode://...@bootnode1.viddhana.network:30303",
  "enode://...@bootnode2.viddhana.network:30303"
]

[Node.HTTPServer]
Enabled = true
Addr = "0.0.0.0"
Port = 8545
API = ["eth", "net", "web3", "vdh"]
Cors = ["*"]

[Validator]
Enabled = true
Coinbase = "0x..."  # Your validator address
GasLimit = 30000000
EOF

# 7. Start the validator
./atlas --config /data/atlas/config.toml \
  --validator \
  --mine \
  --unlock 0x... \
  --password /data/atlas/password.txt
```

### Docker Deployment

```yaml
# docker-compose.validator.yml
version: '3.8'

services:
  atlas-validator:
    image: viddhana/atlas-chain:latest
    container_name: atlas-validator
    restart: unless-stopped
    ports:
      - "30303:30303"      # P2P
      - "30303:30303/udp"  # P2P UDP
      - "8545:8545"        # HTTP RPC
      - "8546:8546"        # WebSocket
    volumes:
      - atlas-data:/data/atlas
      - ./config.toml:/config/config.toml:ro
      - ./validator.key:/keys/validator.key:ro
    command: >
      --config /config/config.toml
      --validator
      --mine
    environment:
      - VALIDATOR_PASSWORD=${VALIDATOR_PASSWORD}
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "5"

volumes:
  atlas-data:
```

### Monitoring Setup

```yaml
# prometheus-config.yml
scrape_configs:
  - job_name: 'atlas-validator'
    static_configs:
      - targets: ['localhost:6060']
    metrics_path: /debug/metrics/prometheus

# Key metrics to monitor
# atlas_chain_head_block_number
# atlas_chain_block_processing_time
# atlas_txpool_pending_count
# atlas_p2p_peers_count
# atlas_validator_proposed_blocks_total
# atlas_validator_missed_blocks_total
```

---

## Gas & Fee Logic

### Gas Configuration

```solidity
// Gas constants for Atlas Chain
uint256 constant BASE_GAS_PRICE = 0.001 gwei;      // ~$0.00001 per gas unit
uint256 constant PRIORITY_FEE = 0.0001 gwei;       // Tip for validators
uint256 constant BLOCK_GAS_LIMIT = 30_000_000;     // 30M gas per block

// Example transaction costs
// Simple transfer: 21,000 gas = $0.00021
// ERC20 transfer: 65,000 gas = $0.00065
// Complex DeFi: 500,000 gas = $0.005
```

### Paymaster Implementation (Account Abstraction)

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@account-abstraction/contracts/core/BasePaymaster.sol";

/**
 * @title VDHPaymaster
 * @notice Sponsors gas fees for VIDDHANA users
 * @dev Implements ERC-4337 Paymaster for gas abstraction
 */
contract VDHPaymaster is BasePaymaster {
    
    mapping(address => uint256) public sponsoredGasLimit;
    uint256 public defaultSponsorLimit = 1 ether;  // In gas value
    
    constructor(IEntryPoint _entryPoint) BasePaymaster(_entryPoint) {}
    
    /**
     * @notice Validates if user operation should be sponsored
     */
    function _validatePaymasterUserOp(
        UserOperation calldata userOp,
        bytes32 userOpHash,
        uint256 maxCost
    ) internal override returns (bytes memory context, uint256 validationData) {
        // Check if user is eligible for sponsorship
        address user = userOp.sender;
        
        // Tier 1: VDH stakers get free gas
        if (isVDHStaker(user)) {
            return (abi.encode(user, maxCost), 0);
        }
        
        // Tier 2: New users get limited free gas
        if (isNewUser(user) && maxCost <= defaultSponsorLimit) {
            return (abi.encode(user, maxCost), 0);
        }
        
        // Tier 3: Pay with VDH tokens
        if (hasVDHBalance(user, maxCost)) {
            return (abi.encode(user, maxCost), 0);
        }
        
        revert("Not eligible for sponsorship");
    }
    
    function isVDHStaker(address user) internal view returns (bool) {
        // Check staking contract
        return IStaking(stakingContract).stakedAmount(user) > 0;
    }
}
```

### Dynamic Fee Adjustment

```python
# Fee adjustment algorithm
class DynamicFeeManager:
    def __init__(self):
        self.target_gas_utilization = 0.5  # 50% block utilization target
        self.max_change_rate = 0.125       # 12.5% max change per block
        self.min_base_fee = 1              # 1 wei minimum
    
    def calculate_next_base_fee(
        self, 
        parent_gas_used: int,
        parent_gas_limit: int,
        parent_base_fee: int
    ) -> int:
        """EIP-1559 style base fee calculation."""
        
        parent_gas_target = parent_gas_limit // 2
        
        if parent_gas_used == parent_gas_target:
            return parent_base_fee
        
        if parent_gas_used > parent_gas_target:
            # Increase fee
            gas_used_delta = parent_gas_used - parent_gas_target
            base_fee_delta = max(
                parent_base_fee * gas_used_delta // parent_gas_target // 8,
                1
            )
            return parent_base_fee + base_fee_delta
        else:
            # Decrease fee
            gas_used_delta = parent_gas_target - parent_gas_used
            base_fee_delta = parent_base_fee * gas_used_delta // parent_gas_target // 8
            return max(parent_base_fee - base_fee_delta, self.min_base_fee)
```

---

## Bridge Implementation

### L2 <-> L3 Bridge Architecture

```
+------------------+                    +------------------+
|   Arbitrum L2    |                    |   Atlas L3       |
+------------------+                    +------------------+
|                  |                    |                  |
|  +-----------+   |    Message         |  +-----------+   |
|  | L2Bridge  |<--|----Passing-------->|  | L3Bridge  |   |
|  +-----------+   |                    |  +-----------+   |
|       |          |                    |       |          |
|       v          |                    |       v          |
|  +-----------+   |                    |  +-----------+   |
|  | Outbox    |   |                    |  | Inbox     |   |
|  +-----------+   |                    |  +-----------+   |
|                  |                    |                  |
+------------------+                    +------------------+
```

### Bridge Contract (L2 Side)

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title L2Bridge
 * @notice Handles asset bridging from Arbitrum L2 to Atlas L3
 */
contract L2Bridge {
    
    address public l3BridgeAddress;
    uint256 public nonce;
    
    event DepositInitiated(
        address indexed sender,
        address indexed recipient,
        address token,
        uint256 amount,
        uint256 nonce
    );
    
    event WithdrawalFinalized(
        address indexed recipient,
        address token,
        uint256 amount,
        uint256 nonce
    );
    
    /**
     * @notice Deposit tokens to bridge to L3
     * @param token Token address (address(0) for ETH)
     * @param amount Amount to bridge
     * @param recipient Recipient address on L3
     */
    function deposit(
        address token,
        uint256 amount,
        address recipient
    ) external payable {
        if (token == address(0)) {
            require(msg.value == amount, "ETH amount mismatch");
        } else {
            IERC20(token).transferFrom(msg.sender, address(this), amount);
        }
        
        uint256 currentNonce = nonce++;
        
        // Create message for L3
        bytes memory message = abi.encodeWithSelector(
            IL3Bridge.finalizeDeposit.selector,
            msg.sender,
            recipient,
            token,
            amount,
            currentNonce
        );
        
        // Send to L3 via messaging layer
        _sendMessageToL3(message);
        
        emit DepositInitiated(msg.sender, recipient, token, amount, currentNonce);
    }
    
    /**
     * @notice Finalize withdrawal from L3
     * @dev Called by the sequencer after L3 withdrawal is confirmed
     */
    function finalizeWithdrawal(
        address recipient,
        address token,
        uint256 amount,
        uint256 withdrawalNonce,
        bytes calldata proof
    ) external onlySequencer {
        // Verify proof from L3
        require(_verifyWithdrawalProof(proof), "Invalid proof");
        
        if (token == address(0)) {
            payable(recipient).transfer(amount);
        } else {
            IERC20(token).transfer(recipient, amount);
        }
        
        emit WithdrawalFinalized(recipient, token, amount, withdrawalNonce);
    }
}
```

### Bridge Contract (L3 Side)

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title L3Bridge
 * @notice Handles asset bridging on Atlas L3
 */
contract L3Bridge {
    
    mapping(uint256 => bool) public processedDeposits;
    uint256 public withdrawalNonce;
    
    event DepositFinalized(
        address indexed sender,
        address indexed recipient,
        address token,
        uint256 amount,
        uint256 nonce
    );
    
    event WithdrawalInitiated(
        address indexed sender,
        address indexed recipient,
        address token,
        uint256 amount,
        uint256 nonce
    );
    
    /**
     * @notice Finalize deposit from L2
     * @dev Called by the sequencer after L2 deposit is confirmed
     */
    function finalizeDeposit(
        address sender,
        address recipient,
        address token,
        uint256 amount,
        uint256 depositNonce
    ) external onlySequencer {
        require(!processedDeposits[depositNonce], "Already processed");
        processedDeposits[depositNonce] = true;
        
        // Mint wrapped tokens on L3
        if (token == address(0)) {
            IWrappedETH(wrappedETH).mint(recipient, amount);
        } else {
            IWrappedToken(getWrappedToken(token)).mint(recipient, amount);
        }
        
        emit DepositFinalized(sender, recipient, token, amount, depositNonce);
    }
    
    /**
     * @notice Initiate withdrawal to L2
     */
    function withdraw(
        address token,
        uint256 amount,
        address recipient
    ) external {
        // Burn wrapped tokens
        if (token == address(0)) {
            IWrappedETH(wrappedETH).burn(msg.sender, amount);
        } else {
            IWrappedToken(getWrappedToken(token)).burn(msg.sender, amount);
        }
        
        uint256 currentNonce = withdrawalNonce++;
        
        // Queue withdrawal for L2 finalization
        _queueWithdrawal(recipient, token, amount, currentNonce);
        
        emit WithdrawalInitiated(msg.sender, recipient, token, amount, currentNonce);
    }
}
```

---

## Testing & Validation

### Unit Test Suite

```typescript
// test/consensus.test.ts
import { expect } from "chai";
import { ethers } from "hardhat";

describe("Atlas Chain Consensus", () => {
  let validators: SignerWithAddress[];
  let consensus: AtlasConsensus;

  beforeEach(async () => {
    validators = await ethers.getSigners();
    const ConsensusFactory = await ethers.getContractFactory("AtlasConsensus");
    consensus = await ConsensusFactory.deploy(
      validators.slice(0, 21).map(v => v.address)
    );
  });

  describe("Block Production", () => {
    it("should produce blocks every 2 seconds", async () => {
      const block1 = await ethers.provider.getBlock("latest");
      
      // Wait for next block
      await network.provider.send("evm_mine");
      
      const block2 = await ethers.provider.getBlock("latest");
      
      expect(block2.timestamp - block1.timestamp).to.be.lte(2);
    });

    it("should rotate proposers correctly", async () => {
      const proposer1 = await consensus.getProposer(100);
      const proposer2 = await consensus.getProposer(101);
      
      expect(proposer1).to.not.equal(proposer2);
    });

    it("should require valid validator signature", async () => {
      const invalidValidator = ethers.Wallet.createRandom();
      
      await expect(
        consensus.connect(invalidValidator).proposeBlock({})
      ).to.be.revertedWith("Not a validator");
    });
  });

  describe("Finality", () => {
    it("should finalize after 6 blocks", async () => {
      const tx = await validators[0].sendTransaction({
        to: validators[1].address,
        value: ethers.parseEther("1")
      });
      
      const receipt = await tx.wait();
      const txBlock = receipt.blockNumber;
      
      // Mine 6 more blocks
      for (let i = 0; i < 6; i++) {
        await network.provider.send("evm_mine");
      }
      
      const isFinalized = await consensus.isBlockFinalized(txBlock);
      expect(isFinalized).to.be.true;
    });
  });
});
```

### Load Testing Script

```python
# scripts/load_test.py
import asyncio
import aiohttp
import time
from dataclasses import dataclass
from typing import List

@dataclass
class LoadTestResult:
    total_transactions: int
    successful: int
    failed: int
    duration_seconds: float
    tps: float
    avg_latency_ms: float
    p99_latency_ms: float

async def send_transaction(session: aiohttp.ClientSession, rpc_url: str) -> dict:
    """Send a single transaction and measure latency."""
    start = time.perf_counter()
    
    payload = {
        "jsonrpc": "2.0",
        "method": "eth_sendRawTransaction",
        "params": ["0x..."],  # Pre-signed transaction
        "id": 1
    }
    
    async with session.post(rpc_url, json=payload) as response:
        result = await response.json()
        latency = (time.perf_counter() - start) * 1000  # ms
        
        return {
            "success": "result" in result,
            "latency_ms": latency
        }

async def run_load_test(
    rpc_url: str,
    target_tps: int,
    duration_seconds: int
) -> LoadTestResult:
    """Run load test against Atlas Chain."""
    
    total_txs = target_tps * duration_seconds
    interval = 1.0 / target_tps
    
    results: List[dict] = []
    
    async with aiohttp.ClientSession() as session:
        start_time = time.perf_counter()
        
        tasks = []
        for i in range(total_txs):
            task = asyncio.create_task(send_transaction(session, rpc_url))
            tasks.append(task)
            await asyncio.sleep(interval)
        
        results = await asyncio.gather(*tasks)
        end_time = time.perf_counter()
    
    successful = sum(1 for r in results if r["success"])
    latencies = sorted([r["latency_ms"] for r in results])
    
    return LoadTestResult(
        total_transactions=total_txs,
        successful=successful,
        failed=total_txs - successful,
        duration_seconds=end_time - start_time,
        tps=successful / (end_time - start_time),
        avg_latency_ms=sum(latencies) / len(latencies),
        p99_latency_ms=latencies[int(len(latencies) * 0.99)]
    )

if __name__ == "__main__":
    result = asyncio.run(run_load_test(
        rpc_url="https://rpc.testnet.viddhana.network",
        target_tps=10000,
        duration_seconds=60
    ))
    
    print(f"Load Test Results:")
    print(f"  Total TXs: {result.total_transactions}")
    print(f"  Successful: {result.successful}")
    print(f"  Failed: {result.failed}")
    print(f"  Achieved TPS: {result.tps:.2f}")
    print(f"  Avg Latency: {result.avg_latency_ms:.2f}ms")
    print(f"  P99 Latency: {result.p99_latency_ms:.2f}ms")
```

### Acceptance Criteria Checklist

```markdown
## Atlas Chain Launch Readiness

### Performance
- [ ] Block time consistently ~2 seconds (measured over 10,000 blocks)
- [ ] 6-second finality achieved (verified with reorganization tests)
- [ ] 100,000+ TPS sustained for 1 hour under load test
- [ ] Gas costs < $0.001 per simple transfer

### Security
- [ ] All 21 validators operational and synced
- [ ] Slashing mechanism tested and verified
- [ ] No single point of failure in infrastructure
- [ ] DDoS protection in place for RPC endpoints

### Compatibility
- [ ] EVM compatibility verified (OpenZeppelin contracts deploy)
- [ ] Web3.js/ethers.js integration tested
- [ ] MetaMask connection working
- [ ] Block explorer fully functional

### Bridge
- [ ] L2 -> L3 deposits complete in < 5 minutes
- [ ] L3 -> L2 withdrawals complete in < 30 minutes
- [ ] Bridge contracts audited
- [ ] Emergency pause mechanism tested
```

---

## Troubleshooting Guide

### Common Issues

| Issue | Symptoms | Solution |
|-------|----------|----------|
| Slow sync | Node stuck at old block | Check peer connections, increase max peers |
| High latency | RPC responses > 1s | Scale RPC nodes, check database performance |
| Missed blocks | Validator gaps in production | Check validator uptime, network connectivity |
| Fork detected | Conflicting block at same height | Investigate validator key compromise |

### Debug Commands

```bash
# Check node sync status
atlas attach --exec "eth.syncing"

# View connected peers
atlas attach --exec "admin.peers.length"

# Check validator status
atlas attach --exec "clique.getSigners()"

# View pending transactions
atlas attach --exec "txpool.status"

# Export block for debugging
atlas attach --exec "debug.traceBlock(12345)"
```

---

## Next Steps

After completing Atlas Chain setup:
1. Proceed to `03_PROMETHEUS_AI_ENGINE.md` for AI integration
2. Proceed to `04_SMART_CONTRACTS.md` for core contract deployment
3. Update `TRACKER.md` with completion status

---

*Document Version: 1.0.0*
