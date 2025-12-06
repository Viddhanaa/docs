---
sidebar_position: 2
title: VDH Token
---

# VDH Token Contract

The VDH token is the native utility token of the VIDDHANA ecosystem.

## Token Specifications

| Property | Value |
|----------|-------|
| Name | VIDDHANA Token |
| Symbol | VDH |
| Decimals | 18 |
| Total Supply | 1,000,000,000 VDH |
| Standard | ERC-20 |

## Contract Features

The VDH token implements:

- **ERC-20** - Standard token interface
- **ERC-20 Permit** - Gasless approvals (EIP-2612)
- **ERC-20 Votes** - On-chain governance support
- **Burnable** - Token burning capability
- **Pausable** - Emergency pause functionality

## Contract Interface

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Burnable.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Pausable.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Permit.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Votes.sol";
import "@openzeppelin/contracts/access/AccessControl.sol";

contract VDHToken is 
    ERC20, 
    ERC20Burnable, 
    ERC20Pausable, 
    ERC20Permit, 
    ERC20Votes, 
    AccessControl 
{
    bytes32 public constant PAUSER_ROLE = keccak256("PAUSER_ROLE");
    bytes32 public constant MINTER_ROLE = keccak256("MINTER_ROLE");

    constructor(address defaultAdmin)
        ERC20("VIDDHANA Token", "VDH")
        ERC20Permit("VIDDHANA Token")
    {
        _grantRole(DEFAULT_ADMIN_ROLE, defaultAdmin);
        _grantRole(PAUSER_ROLE, defaultAdmin);
        _grantRole(MINTER_ROLE, defaultAdmin);
    }

    function mint(address to, uint256 amount) public onlyRole(MINTER_ROLE) {
        _mint(to, amount);
    }

    function pause() public onlyRole(PAUSER_ROLE) {
        _pause();
    }

    function unpause() public onlyRole(PAUSER_ROLE) {
        _unpause();
    }
}
```

## Usage Examples

### Transfer Tokens

```typescript
import { ViddhanaSDK } from '@viddhana/sdk';

const sdk = new ViddhanaSDK({
  network: 'mainnet',
  privateKey: process.env.PRIVATE_KEY,
});

// Transfer 100 VDH
const tx = await sdk.transfer({
  to: '0xRecipientAddress',
  amount: '100', // VDH
});

console.log('Transaction hash:', tx.hash);
```

### Check Balance

```typescript
const balance = await sdk.getBalance('0xYourAddress');
console.log('Balance:', balance, 'VDH');
```

### Gasless Approval (Permit)

```typescript
// Sign permit off-chain
const permit = await sdk.signPermit({
  spender: '0xSpenderAddress',
  value: '1000',
  deadline: Math.floor(Date.now() / 1000) + 3600, // 1 hour
});

// Submit permit (can be done by anyone)
await sdk.submitPermit(permit);
```

### Delegate Voting Power

```typescript
// Delegate to yourself
await sdk.delegate('0xYourAddress');

// Delegate to another address
await sdk.delegate('0xDelegateAddress');

// Check voting power
const votes = await sdk.getVotes('0xYourAddress');
console.log('Voting power:', votes);
```

## Token Distribution

| Allocation | Percentage | Amount | Vesting |
|------------|------------|--------|---------|
| Community Rewards | 30% | 300M VDH | 4 years linear |
| Development | 20% | 200M VDH | 2 years cliff, 2 years linear |
| Team | 15% | 150M VDH | 1 year cliff, 3 years linear |
| Treasury | 15% | 150M VDH | DAO controlled |
| Liquidity | 10% | 100M VDH | Immediate |
| Investors | 10% | 100M VDH | 6 months cliff, 2 years linear |

## Events

```solidity
event Transfer(address indexed from, address indexed to, uint256 value);
event Approval(address indexed owner, address indexed spender, uint256 value);
event DelegateChanged(address indexed delegator, address indexed fromDelegate, address indexed toDelegate);
event DelegateVotesChanged(address indexed delegate, uint256 previousBalance, uint256 newBalance);
```

## Security Considerations

1. **Minting is restricted** - Only addresses with `MINTER_ROLE` can mint
2. **Pausing is restricted** - Only addresses with `PAUSER_ROLE` can pause
3. **Admin is timelocked** - Admin actions go through 48-hour timelock
4. **No infinite approvals** - Users should use permit for safer approvals

---

See also: [Smart Contract Overview](/docs/smart-contracts/overview)
