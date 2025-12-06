---
sidebar_position: 1
title: Overview
---

# Smart Contract Overview

VIDDHANA's smart contract architecture is designed for security, upgradability, and composability.

## Architecture

The VIDDHANA smart contract suite consists of four main contracts:

```
┌─────────────────────────────────────────────────────────────┐
│                     VIDDHANA Contracts                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │  VDH Token  │───▶│ VaultManager│───▶│PolicyEngine │     │
│  │   (ERC20)   │    │  (Vaults)   │    │    (AI)     │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│         │                  │                  │             │
│         └──────────────────┼──────────────────┘             │
│                            ▼                                │
│                   ┌─────────────────┐                       │
│                   │  VDHGovernance  │                       │
│                   │     (DAO)       │                       │
│                   └─────────────────┘                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Core Contracts

| Contract | Description | Address (Testnet) |
|----------|-------------|-------------------|
| VDH Token | Native ERC-20 governance token | `0x384B37ab47B51f13D32fc2C19ea97147eC89fCD4` |
| VaultManager | Multi-asset vault management | `0xdC503c4E0F865C2cF198528354A8BCD19ffAF3F5` |
| PolicyEngine | AI-driven rebalancing policies | `0xCD375A9355f765990b3f030B71C316e52a5353d2` |
| VDHGovernance | On-chain DAO governance | `0xAF53F4F1feAbea3aA9030b38Cac6dB68691BfD03` |

## Contract Details

### 1. VDH Token

The native ERC-20 token with governance capabilities:

- **Standard**: ERC-20 with ERC-20Votes extension
- **Total Supply**: 1,000,000,000 VDH
- **Features**: Burnable, Pausable, Permit (gasless approvals)

```solidity
interface IVDHToken {
    function mint(address to, uint256 amount) external;
    function burn(uint256 amount) external;
    function delegate(address delegatee) external;
    function getVotes(address account) external view returns (uint256);
}
```

### 2. VaultManager

Manages user vaults for multi-asset portfolio management:

- **Upgradeable**: UUPS proxy pattern
- **Multi-asset**: Support for any ERC-20 token
- **Risk controls**: Per-vault risk parameters

```solidity
interface IVaultManager {
    function createVault(
        string memory name,
        uint256 riskTolerance,
        uint256 timeHorizon
    ) external returns (uint256 vaultId);
    
    function deposit(
        uint256 vaultId,
        address asset,
        uint256 amount
    ) external;
    
    function withdraw(
        uint256 vaultId,
        address asset,
        uint256 amount
    ) external;
    
    function getVaultInfo(uint256 vaultId) external view returns (VaultInfo memory);
}
```

### 3. PolicyEngine

AI-driven policy enforcement for portfolio rebalancing:

- **AI Integration**: Receives signals from Prometheus AI
- **Automated Rebalancing**: Executes trades based on policy rules
- **Risk Management**: Enforces risk limits and stop-losses

```solidity
interface IPolicyEngine {
    function createPolicy(
        uint256 vaultId,
        PolicyType policyType,
        bytes memory params
    ) external returns (uint256 policyId);
    
    function executeRebalance(
        uint256 vaultId,
        RebalanceAction[] memory actions
    ) external;
    
    function setRiskParameters(
        uint256 vaultId,
        RiskParams memory params
    ) external;
}
```

### 4. VDHGovernance

On-chain DAO governance for protocol decisions:

- **Proposal System**: Create and vote on proposals
- **Timelock**: 48-hour delay for execution
- **Quorum**: 4% of total supply required

```solidity
interface IVDHGovernance {
    function propose(
        address[] memory targets,
        uint256[] memory values,
        bytes[] memory calldatas,
        string memory description
    ) external returns (uint256 proposalId);
    
    function castVote(uint256 proposalId, uint8 support) external;
    function execute(uint256 proposalId) external;
}
```

## Key Features

### Upgradability (UUPS)

All core contracts use the UUPS (Universal Upgradeable Proxy Standard) pattern:

```solidity
import "@openzeppelin/contracts-upgradeable/proxy/utils/UUPSUpgradeable.sol";

contract VaultManager is UUPSUpgradeable {
    function _authorizeUpgrade(address newImplementation) 
        internal 
        override 
        onlyRole(UPGRADER_ROLE) 
    {}
}
```

### Access Control

Role-based permissions using OpenZeppelin AccessControl:

```solidity
bytes32 public constant ADMIN_ROLE = keccak256("ADMIN_ROLE");
bytes32 public constant OPERATOR_ROLE = keccak256("OPERATOR_ROLE");
bytes32 public constant AI_ROLE = keccak256("AI_ROLE");
bytes32 public constant UPGRADER_ROLE = keccak256("UPGRADER_ROLE");
```

| Role | Permissions |
|------|-------------|
| ADMIN_ROLE | Full administrative access |
| OPERATOR_ROLE | Execute operations, manage vaults |
| AI_ROLE | Submit AI signals, trigger rebalances |
| UPGRADER_ROLE | Upgrade contract implementations |

### Pausability

Emergency pause functionality for all critical contracts:

```solidity
function pause() external onlyRole(ADMIN_ROLE) {
    _pause();
}

function unpause() external onlyRole(ADMIN_ROLE) {
    _unpause();
}
```

### Reentrancy Protection

All state-changing functions are protected:

```solidity
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";

function withdraw(uint256 vaultId, address asset, uint256 amount) 
    external 
    nonReentrant 
{
    // Safe withdrawal logic
}
```

## Security Measures

| Measure | Description |
|---------|-------------|
| **Audited** | Contracts audited by leading security firms |
| **Timelocked** | Governance actions have 48-hour delay |
| **Multi-sig** | Admin functions require multi-signature |
| **Rate Limited** | Protection against flash loan attacks |
| **Pausable** | Emergency stop functionality |

## Development Standards

### Solidity Version

All contracts use Solidity 0.8.20+ with:

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;
```

### OpenZeppelin Dependencies

```json
{
  "@openzeppelin/contracts": "^5.0.0",
  "@openzeppelin/contracts-upgradeable": "^5.0.0"
}
```

### Testing Requirements

- 100% line coverage
- Fuzz testing for critical functions
- Formal verification for token contracts
- Integration tests with mainnet forks

### Gas Optimization

- Efficient storage patterns (packed structs)
- Batch operations where possible
- View functions for reads
- Events for off-chain indexing

## Contract Addresses

### Testnet (Chain ID: 1337)

| Contract | Address |
|----------|---------|
| VDH Token | `0x384B37ab47B51f13D32fc2C19ea97147eC89fCD4` |
| VaultManager | `0xdC503c4E0F865C2cF198528354A8BCD19ffAF3F5` |
| PolicyEngine | `0xCD375A9355f765990b3f030B71C316e52a5353d2` |
| VDHGovernance | `0xAF53F4F1feAbea3aA9030b38Cac6dB68691BfD03` |

### Mainnet (Chain ID: 13370)

| Contract | Address |
|----------|---------|
| VDH Token | Coming soon |
| VaultManager | Coming soon |
| PolicyEngine | Coming soon |
| VDHGovernance | Coming soon |

## Interacting with Contracts

### Using ethers.js

```typescript
import { ethers } from 'ethers';

const provider = new ethers.JsonRpcProvider('https://rpc.viddhana.com');
const wallet = new ethers.Wallet(privateKey, provider);

// VaultManager ABI (partial)
const vaultManagerABI = [
  "function createVault(string name, uint256 riskTolerance, uint256 timeHorizon) returns (uint256)",
  "function deposit(uint256 vaultId, address asset, uint256 amount)",
  "function getVaultInfo(uint256 vaultId) view returns (tuple(uint256 id, address owner, string name, uint256 totalValue))"
];

const vaultManager = new ethers.Contract(
  "0xdC503c4E0F865C2cF198528354A8BCD19ffAF3F5",
  vaultManagerABI,
  wallet
);

// Create a vault
const tx = await vaultManager.createVault("My Vault", 5000, 365);
await tx.wait();
```

### Using Hardhat

```typescript
import { ethers } from "hardhat";

async function main() {
  const VaultManager = await ethers.getContractAt(
    "VaultManager",
    "0xdC503c4E0F865C2cF198528354A8BCD19ffAF3F5"
  );
  
  const vaultInfo = await VaultManager.getVaultInfo(1);
  console.log("Vault:", vaultInfo);
}
```

---

Next: [VDH Token Contract](/docs/smart-contracts/vdh-token)
