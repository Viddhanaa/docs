# Tokenomics Implementation Guide

> Detailed implementation guide for $VDH token economics and distribution

---

## Table of Contents
1. [Overview](#overview)
2. [Token Distribution](#token-distribution)
3. [Vesting Contracts](#vesting-contracts)
4. [Staking & Governance](#staking--governance)
5. [Burn Mechanism](#burn-mechanism)
6. [Price Model](#price-model)
7. [Smart Contract Implementation](#smart-contract-implementation)
8. [Testing & Validation](#testing--validation)

---

## Overview

The $VDH token is the native currency of the VIDDHANA ecosystem with:
- **Fixed Supply**: 1,000,000,000 VDH
- **Utility**: Gas, staking, governance, rewards
- **Deflationary**: Buyback & burn mechanism

### Price Valuation Model

$$Price_{VDH} = \frac{Annual\_Revenue \times P/E\_Ratio}{Circulating\_Supply}$$

---

## Token Distribution

### Allocation Table

| Category | Allocation | Tokens | Vesting |
|----------|------------|--------|---------|
| Community | 40% | 400,000,000 | 10% TGE, 36 months linear |
| Developers | 20% | 200,000,000 | 3-year cliff, 2-year linear |
| Ecosystem Fund | 15% | 150,000,000 | Unlocks Year 1-3 |
| Founders | 15% | 150,000,000 | 4-year lock |
| Seed Investors | 10% | 100,000,000 | 2-year lock, 1-year linear |

### Distribution Timeline

```
Year 0 (TGE):
  - Community: 40M (10% of allocation)
  - Total Circulating: 40M

Year 1:
  - Community: +133M (monthly linear)
  - Ecosystem: +50M
  - Total Circulating: ~223M

Year 2:
  - Community: +133M
  - Ecosystem: +50M
  - Seed Investors: +100M (starts unlocking)
  - Total Circulating: ~506M

Year 3:
  - Community: +94M (final)
  - Ecosystem: +50M
  - Developers: +200M (cliff ends, starts vesting)
  - Total Circulating: ~850M

Year 4-5:
  - Developers: Continue vesting
  - Founders: Unlock after Year 4
  - Total Circulating: 1B (fully diluted)
```

---

## Vesting Contracts

### TokenVesting.sol

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";

/**
 * @title TokenVesting
 * @notice Manages token vesting schedules for different allocation categories
 */
contract TokenVesting is Ownable, ReentrancyGuard {
    using SafeERC20 for IERC20;
    
    IERC20 public immutable token;
    
    struct VestingSchedule {
        address beneficiary;
        uint256 totalAmount;
        uint256 startTime;
        uint256 cliffDuration;      // Cliff period in seconds
        uint256 vestingDuration;    // Total vesting duration in seconds
        uint256 tgePercentage;      // Percentage released at TGE (basis points)
        uint256 released;
        bool revocable;
        bool revoked;
    }
    
    // Schedule ID => VestingSchedule
    mapping(bytes32 => VestingSchedule) public vestingSchedules;
    
    // Beneficiary => Schedule IDs
    mapping(address => bytes32[]) public beneficiarySchedules;
    
    // Total tokens allocated to vesting
    uint256 public totalAllocated;
    
    event VestingScheduleCreated(
        bytes32 indexed scheduleId,
        address indexed beneficiary,
        uint256 amount,
        uint256 startTime,
        uint256 cliffDuration,
        uint256 vestingDuration
    );
    event TokensReleased(bytes32 indexed scheduleId, address indexed beneficiary, uint256 amount);
    event VestingRevoked(bytes32 indexed scheduleId);
    
    constructor(address _token) Ownable(msg.sender) {
        token = IERC20(_token);
    }
    
    /**
     * @notice Create a new vesting schedule
     * @param beneficiary Address receiving the tokens
     * @param amount Total amount of tokens to vest
     * @param startTime Start timestamp of vesting
     * @param cliffDuration Duration of cliff in seconds
     * @param vestingDuration Total vesting duration in seconds
     * @param tgePercentage Percentage unlocked at TGE (basis points)
     * @param revocable Whether the schedule can be revoked
     */
    function createVestingSchedule(
        address beneficiary,
        uint256 amount,
        uint256 startTime,
        uint256 cliffDuration,
        uint256 vestingDuration,
        uint256 tgePercentage,
        bool revocable
    ) external onlyOwner returns (bytes32 scheduleId) {
        require(beneficiary != address(0), "Invalid beneficiary");
        require(amount > 0, "Amount must be > 0");
        require(vestingDuration > 0, "Duration must be > 0");
        require(tgePercentage <= 10000, "TGE percentage too high");
        
        // Generate unique schedule ID
        scheduleId = keccak256(
            abi.encodePacked(beneficiary, amount, startTime, block.timestamp)
        );
        
        require(vestingSchedules[scheduleId].beneficiary == address(0), "Schedule exists");
        
        vestingSchedules[scheduleId] = VestingSchedule({
            beneficiary: beneficiary,
            totalAmount: amount,
            startTime: startTime,
            cliffDuration: cliffDuration,
            vestingDuration: vestingDuration,
            tgePercentage: tgePercentage,
            released: 0,
            revocable: revocable,
            revoked: false
        });
        
        beneficiarySchedules[beneficiary].push(scheduleId);
        totalAllocated += amount;
        
        emit VestingScheduleCreated(
            scheduleId,
            beneficiary,
            amount,
            startTime,
            cliffDuration,
            vestingDuration
        );
    }
    
    /**
     * @notice Calculate vested amount for a schedule
     */
    function vestedAmount(bytes32 scheduleId) public view returns (uint256) {
        VestingSchedule storage schedule = vestingSchedules[scheduleId];
        
        if (schedule.revoked) {
            return schedule.released;
        }
        
        return _calculateVestedAmount(schedule);
    }
    
    /**
     * @notice Calculate releasable amount for a schedule
     */
    function releasableAmount(bytes32 scheduleId) public view returns (uint256) {
        VestingSchedule storage schedule = vestingSchedules[scheduleId];
        return vestedAmount(scheduleId) - schedule.released;
    }
    
    /**
     * @notice Release vested tokens
     */
    function release(bytes32 scheduleId) external nonReentrant {
        VestingSchedule storage schedule = vestingSchedules[scheduleId];
        
        require(schedule.beneficiary != address(0), "Schedule not found");
        require(!schedule.revoked, "Schedule revoked");
        
        uint256 releasable = releasableAmount(scheduleId);
        require(releasable > 0, "Nothing to release");
        
        schedule.released += releasable;
        token.safeTransfer(schedule.beneficiary, releasable);
        
        emit TokensReleased(scheduleId, schedule.beneficiary, releasable);
    }
    
    /**
     * @notice Revoke a vesting schedule
     */
    function revoke(bytes32 scheduleId) external onlyOwner {
        VestingSchedule storage schedule = vestingSchedules[scheduleId];
        
        require(schedule.revocable, "Not revocable");
        require(!schedule.revoked, "Already revoked");
        
        uint256 vested = vestedAmount(scheduleId);
        uint256 unreleased = vested - schedule.released;
        uint256 refund = schedule.totalAmount - vested;
        
        schedule.revoked = true;
        totalAllocated -= refund;
        
        // Release vested amount to beneficiary
        if (unreleased > 0) {
            schedule.released += unreleased;
            token.safeTransfer(schedule.beneficiary, unreleased);
        }
        
        // Return unvested to owner
        if (refund > 0) {
            token.safeTransfer(owner(), refund);
        }
        
        emit VestingRevoked(scheduleId);
    }
    
    /**
     * @notice Internal vesting calculation
     */
    function _calculateVestedAmount(VestingSchedule storage schedule) internal view returns (uint256) {
        uint256 currentTime = block.timestamp;
        
        // TGE amount
        uint256 tgeAmount = (schedule.totalAmount * schedule.tgePercentage) / 10000;
        
        // Before start time, only TGE is vested
        if (currentTime < schedule.startTime) {
            return tgeAmount;
        }
        
        // During cliff, only TGE is vested
        uint256 cliffEnd = schedule.startTime + schedule.cliffDuration;
        if (currentTime < cliffEnd) {
            return tgeAmount;
        }
        
        // After cliff, linear vesting
        uint256 vestingEnd = schedule.startTime + schedule.vestingDuration;
        uint256 vestingAmount = schedule.totalAmount - tgeAmount;
        
        if (currentTime >= vestingEnd) {
            return schedule.totalAmount;
        }
        
        uint256 timeVested = currentTime - cliffEnd;
        uint256 vestingPeriod = vestingEnd - cliffEnd;
        uint256 linearVested = (vestingAmount * timeVested) / vestingPeriod;
        
        return tgeAmount + linearVested;
    }
    
    /**
     * @notice Get all schedules for a beneficiary
     */
    function getBeneficiarySchedules(address beneficiary) 
        external 
        view 
        returns (bytes32[] memory) 
    {
        return beneficiarySchedules[beneficiary];
    }
    
    /**
     * @notice Get schedule details
     */
    function getScheduleDetails(bytes32 scheduleId) 
        external 
        view 
        returns (
            address beneficiary,
            uint256 totalAmount,
            uint256 released,
            uint256 releasable,
            uint256 vested,
            bool revoked
        ) 
    {
        VestingSchedule storage schedule = vestingSchedules[scheduleId];
        
        return (
            schedule.beneficiary,
            schedule.totalAmount,
            schedule.released,
            releasableAmount(scheduleId),
            vestedAmount(scheduleId),
            schedule.revoked
        );
    }
}
```

### Allocation Distribution Script

```typescript
// scripts/distribute-allocations.ts
import { ethers } from "hardhat";

const YEAR = 365 * 24 * 60 * 60;
const MONTH = 30 * 24 * 60 * 60;

interface AllocationConfig {
  category: string;
  beneficiaries: {
    address: string;
    amount: bigint;
  }[];
  tgePercentage: number;
  cliffDuration: number;
  vestingDuration: number;
  revocable: boolean;
}

const allocations: AllocationConfig[] = [
  {
    category: "Community",
    beneficiaries: [
      { address: "0x...", amount: ethers.parseEther("400000000") }
    ],
    tgePercentage: 1000, // 10%
    cliffDuration: 0,
    vestingDuration: 36 * MONTH,
    revocable: false
  },
  {
    category: "Developers",
    beneficiaries: [
      { address: "0x...", amount: ethers.parseEther("200000000") }
    ],
    tgePercentage: 0,
    cliffDuration: 3 * YEAR,
    vestingDuration: 5 * YEAR, // 3 year cliff + 2 year vesting
    revocable: true
  },
  {
    category: "Ecosystem",
    beneficiaries: [
      { address: "0x...", amount: ethers.parseEther("150000000") }
    ],
    tgePercentage: 0,
    cliffDuration: 0,
    vestingDuration: 3 * YEAR,
    revocable: false
  },
  {
    category: "Founders",
    beneficiaries: [
      { address: "0x...", amount: ethers.parseEther("50000000") },
      { address: "0x...", amount: ethers.parseEther("50000000") },
      { address: "0x...", amount: ethers.parseEther("50000000") }
    ],
    tgePercentage: 0,
    cliffDuration: 4 * YEAR,
    vestingDuration: 4 * YEAR, // Full lock, then immediate
    revocable: false
  },
  {
    category: "Seed Investors",
    beneficiaries: [
      { address: "0x...", amount: ethers.parseEther("100000000") }
    ],
    tgePercentage: 0,
    cliffDuration: 2 * YEAR,
    vestingDuration: 3 * YEAR, // 2 year lock + 1 year vesting
    revocable: false
  }
];

async function main() {
  const vestingContract = await ethers.getContractAt(
    "TokenVesting",
    process.env.VESTING_CONTRACT_ADDRESS!
  );
  
  const startTime = Math.floor(Date.now() / 1000); // TGE timestamp
  
  for (const allocation of allocations) {
    console.log(`\nProcessing ${allocation.category} allocation...`);
    
    for (const beneficiary of allocation.beneficiaries) {
      console.log(`  Creating schedule for ${beneficiary.address}`);
      
      const tx = await vestingContract.createVestingSchedule(
        beneficiary.address,
        beneficiary.amount,
        startTime,
        allocation.cliffDuration,
        allocation.vestingDuration,
        allocation.tgePercentage,
        allocation.revocable
      );
      
      const receipt = await tx.wait();
      const event = receipt.logs.find(
        (log: any) => log.fragment?.name === "VestingScheduleCreated"
      );
      
      console.log(`    Schedule ID: ${event?.args?.scheduleId}`);
    }
  }
  
  console.log("\nAll allocations distributed!");
}

main().catch(console.error);
```

---

## Staking & Governance

### Enhanced Staking with Governance Power

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/governance/utils/IVotes.sol";

/**
 * @title StakingWithGovernance
 * @notice Staking contract that provides governance voting power
 */
contract StakingWithGovernance is Staking, IVotes {
    
    // Checkpoints for voting power
    struct Checkpoint {
        uint32 fromBlock;
        uint224 votes;
    }
    
    mapping(address => Checkpoint[]) private _checkpoints;
    Checkpoint[] private _totalSupplyCheckpoints;
    
    mapping(address => address) private _delegates;
    
    event DelegateChanged(
        address indexed delegator,
        address indexed fromDelegate,
        address indexed toDelegate
    );
    event DelegateVotesChanged(
        address indexed delegate,
        uint256 previousBalance,
        uint256 newBalance
    );
    
    /**
     * @notice Get voting power for an account
     */
    function getVotes(address account) public view override returns (uint256) {
        uint256 pos = _checkpoints[account].length;
        return pos == 0 ? 0 : _checkpoints[account][pos - 1].votes;
    }
    
    /**
     * @notice Get past voting power at a specific block
     */
    function getPastVotes(address account, uint256 blockNumber) 
        public 
        view 
        override 
        returns (uint256) 
    {
        require(blockNumber < block.number, "Block not yet mined");
        return _checkpointsLookup(_checkpoints[account], blockNumber);
    }
    
    /**
     * @notice Get past total voting power
     */
    function getPastTotalSupply(uint256 blockNumber) 
        public 
        view 
        override 
        returns (uint256) 
    {
        require(blockNumber < block.number, "Block not yet mined");
        return _checkpointsLookup(_totalSupplyCheckpoints, blockNumber);
    }
    
    /**
     * @notice Get current delegate
     */
    function delegates(address account) public view override returns (address) {
        return _delegates[account] == address(0) ? account : _delegates[account];
    }
    
    /**
     * @notice Delegate voting power
     */
    function delegate(address delegatee) public override {
        _delegate(msg.sender, delegatee);
    }
    
    /**
     * @notice Delegate by signature
     */
    function delegateBySig(
        address delegatee,
        uint256 nonce,
        uint256 expiry,
        uint8 v,
        bytes32 r,
        bytes32 s
    ) public override {
        require(block.timestamp <= expiry, "Signature expired");
        
        bytes32 structHash = keccak256(
            abi.encode(
                keccak256("Delegation(address delegatee,uint256 nonce,uint256 expiry)"),
                delegatee,
                nonce,
                expiry
            )
        );
        
        bytes32 digest = _hashTypedDataV4(structHash);
        address signer = ECDSA.recover(digest, v, r, s);
        
        require(nonce == _useNonce(signer), "Invalid nonce");
        _delegate(signer, delegatee);
    }
    
    /**
     * @notice Calculate voting power based on stake and lock period
     * @dev Voting power = staked amount * lock multiplier
     */
    function _calculateVotingPower(address account) internal view returns (uint256) {
        StakeInfo memory stakeInfo = stakes[account];
        if (stakeInfo.amount == 0) return 0;
        
        uint256 multiplier = lockMultipliers[stakeInfo.lockPeriod];
        return (stakeInfo.amount * multiplier) / 10000;
    }
    
    /**
     * @notice Internal delegation logic
     */
    function _delegate(address delegator, address delegatee) internal {
        address currentDelegate = delegates(delegator);
        uint256 delegatorBalance = _calculateVotingPower(delegator);
        
        _delegates[delegator] = delegatee;
        
        emit DelegateChanged(delegator, currentDelegate, delegatee);
        
        _moveVotingPower(currentDelegate, delegatee, delegatorBalance);
    }
    
    /**
     * @notice Move voting power between accounts
     */
    function _moveVotingPower(
        address from,
        address to,
        uint256 amount
    ) internal {
        if (from != to && amount > 0) {
            if (from != address(0)) {
                (uint256 oldWeight, uint256 newWeight) = _writeCheckpoint(
                    _checkpoints[from],
                    _subtract,
                    amount
                );
                emit DelegateVotesChanged(from, oldWeight, newWeight);
            }
            
            if (to != address(0)) {
                (uint256 oldWeight, uint256 newWeight) = _writeCheckpoint(
                    _checkpoints[to],
                    _add,
                    amount
                );
                emit DelegateVotesChanged(to, oldWeight, newWeight);
            }
        }
    }
    
    /**
     * @notice Override stake to update voting power
     */
    function stake(uint256 amount, uint256 lockPeriod) public override {
        address account = msg.sender;
        uint256 oldPower = _calculateVotingPower(account);
        
        super.stake(amount, lockPeriod);
        
        uint256 newPower = _calculateVotingPower(account);
        _moveVotingPower(address(0), delegates(account), newPower - oldPower);
        _writeCheckpoint(_totalSupplyCheckpoints, _add, newPower - oldPower);
    }
    
    /**
     * @notice Override unstake to update voting power
     */
    function unstake(uint256 amount) public override {
        address account = msg.sender;
        uint256 oldPower = _calculateVotingPower(account);
        
        super.unstake(amount);
        
        uint256 newPower = _calculateVotingPower(account);
        _moveVotingPower(delegates(account), address(0), oldPower - newPower);
        _writeCheckpoint(_totalSupplyCheckpoints, _subtract, oldPower - newPower);
    }
    
    // Checkpoint helper functions
    function _checkpointsLookup(Checkpoint[] storage ckpts, uint256 blockNumber)
        private
        view
        returns (uint256)
    {
        uint256 length = ckpts.length;
        
        uint256 low = 0;
        uint256 high = length;
        
        while (low < high) {
            uint256 mid = (low + high) / 2;
            if (ckpts[mid].fromBlock > blockNumber) {
                high = mid;
            } else {
                low = mid + 1;
            }
        }
        
        return high == 0 ? 0 : ckpts[high - 1].votes;
    }
    
    function _writeCheckpoint(
        Checkpoint[] storage ckpts,
        function(uint256, uint256) view returns (uint256) op,
        uint256 delta
    ) private returns (uint256 oldWeight, uint256 newWeight) {
        uint256 pos = ckpts.length;
        oldWeight = pos == 0 ? 0 : ckpts[pos - 1].votes;
        newWeight = op(oldWeight, delta);
        
        if (pos > 0 && ckpts[pos - 1].fromBlock == block.number) {
            ckpts[pos - 1].votes = uint224(newWeight);
        } else {
            ckpts.push(Checkpoint({
                fromBlock: uint32(block.number),
                votes: uint224(newWeight)
            }));
        }
    }
    
    function _add(uint256 a, uint256 b) private pure returns (uint256) {
        return a + b;
    }
    
    function _subtract(uint256 a, uint256 b) private pure returns (uint256) {
        return a - b;
    }
}
```

---

## Burn Mechanism

### Buyback & Burn Contract

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";

/**
 * @title BuybackBurn
 * @notice Handles quarterly buyback and burn of VDH tokens
 * @dev Uses 30% of platform revenue: 50% burned, 50% to treasury
 */
contract BuybackBurn is AccessControl, ReentrancyGuard {
    using SafeERC20 for IERC20;
    
    bytes32 public constant EXECUTOR_ROLE = keccak256("EXECUTOR_ROLE");
    
    IERC20 public immutable vdhToken;
    IERC20 public immutable revenueToken;  // USDC
    address public treasury;
    address public dexRouter;
    
    uint256 public constant BURN_PERCENTAGE = 5000;     // 50%
    uint256 public constant TREASURY_PERCENTAGE = 5000; // 50%
    uint256 public constant BASIS_POINTS = 10000;
    
    // Quarterly stats
    struct QuarterlyStats {
        uint256 quarter;
        uint256 revenueCollected;
        uint256 vdhBought;
        uint256 vdhBurned;
        uint256 vdhToTreasury;
        uint256 executedAt;
    }
    
    mapping(uint256 => QuarterlyStats) public quarterlyStats;
    uint256 public currentQuarter;
    uint256 public totalBurned;
    uint256 public totalToTreasury;
    
    // Revenue accumulator
    uint256 public accumulatedRevenue;
    
    event RevenueDeposited(address indexed from, uint256 amount);
    event BuybackExecuted(
        uint256 indexed quarter,
        uint256 revenueUsed,
        uint256 vdhBought,
        uint256 vdhBurned,
        uint256 vdhToTreasury
    );
    
    constructor(
        address _vdhToken,
        address _revenueToken,
        address _treasury,
        address _dexRouter
    ) {
        vdhToken = IERC20(_vdhToken);
        revenueToken = IERC20(_revenueToken);
        treasury = _treasury;
        dexRouter = _dexRouter;
        
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(EXECUTOR_ROLE, msg.sender);
    }
    
    /**
     * @notice Deposit platform revenue
     * @param amount Amount of revenue tokens to deposit
     */
    function depositRevenue(uint256 amount) external {
        revenueToken.safeTransferFrom(msg.sender, address(this), amount);
        
        // 30% goes to buyback program
        uint256 buybackAmount = (amount * 3000) / BASIS_POINTS;
        accumulatedRevenue += buybackAmount;
        
        // Remaining 70% goes to treasury directly
        uint256 treasuryAmount = amount - buybackAmount;
        revenueToken.safeTransfer(treasury, treasuryAmount);
        
        emit RevenueDeposited(msg.sender, amount);
    }
    
    /**
     * @notice Execute quarterly buyback and burn
     * @param minVdhOut Minimum VDH expected from swap (slippage protection)
     */
    function executeBuyback(uint256 minVdhOut) 
        external 
        onlyRole(EXECUTOR_ROLE) 
        nonReentrant 
    {
        require(accumulatedRevenue > 0, "No revenue to buyback");
        
        uint256 revenueToUse = accumulatedRevenue;
        accumulatedRevenue = 0;
        
        // Approve DEX router
        revenueToken.safeApprove(dexRouter, revenueToUse);
        
        // Execute swap on DEX
        uint256 vdhBought = _swapForVDH(revenueToUse, minVdhOut);
        
        // Calculate split
        uint256 toBurn = (vdhBought * BURN_PERCENTAGE) / BASIS_POINTS;
        uint256 toTreasury = vdhBought - toBurn;
        
        // Burn tokens
        IERC20Burnable(address(vdhToken)).burn(toBurn);
        totalBurned += toBurn;
        
        // Send to treasury
        vdhToken.safeTransfer(treasury, toTreasury);
        totalToTreasury += toTreasury;
        
        // Record stats
        currentQuarter++;
        quarterlyStats[currentQuarter] = QuarterlyStats({
            quarter: currentQuarter,
            revenueCollected: revenueToUse,
            vdhBought: vdhBought,
            vdhBurned: toBurn,
            vdhToTreasury: toTreasury,
            executedAt: block.timestamp
        });
        
        emit BuybackExecuted(
            currentQuarter,
            revenueToUse,
            vdhBought,
            toBurn,
            toTreasury
        );
    }
    
    /**
     * @notice Internal swap function
     */
    function _swapForVDH(uint256 amountIn, uint256 minAmountOut) 
        internal 
        returns (uint256) 
    {
        address[] memory path = new address[](2);
        path[0] = address(revenueToken);
        path[1] = address(vdhToken);
        
        uint256[] memory amounts = IUniswapV2Router(dexRouter)
            .swapExactTokensForTokens(
                amountIn,
                minAmountOut,
                path,
                address(this),
                block.timestamp
            );
        
        return amounts[1];
    }
    
    /**
     * @notice Get buyback statistics
     */
    function getStats() external view returns (
        uint256 pendingRevenue,
        uint256 totalTokensBurned,
        uint256 totalToTreasuryAmount,
        uint256 quartersExecuted
    ) {
        return (
            accumulatedRevenue,
            totalBurned,
            totalToTreasury,
            currentQuarter
        );
    }
    
    /**
     * @notice Update treasury address
     */
    function setTreasury(address _treasury) external onlyRole(DEFAULT_ADMIN_ROLE) {
        treasury = _treasury;
    }
    
    /**
     * @notice Update DEX router
     */
    function setDexRouter(address _dexRouter) external onlyRole(DEFAULT_ADMIN_ROLE) {
        dexRouter = _dexRouter;
    }
}
```

---

## Price Model

### Price Valuation Calculator

```python
# src/tokenomics/price_model.py
from dataclasses import dataclass
from typing import Optional
import math

@dataclass
class TokenomicsParams:
    total_supply: int = 1_000_000_000
    annual_revenue: float = 0
    pe_ratio: float = 15  # Default P/E for growth companies
    circulating_supply: int = 0
    burned_supply: int = 0
    staked_supply: int = 0


class PriceModel:
    """
    Price valuation model for VDH token.
    
    Formula: Price = (Annual Revenue * P/E Ratio) / Circulating Supply
    """
    
    def __init__(self, params: TokenomicsParams):
        self.params = params
    
    def calculate_price(
        self,
        annual_revenue: Optional[float] = None,
        pe_ratio: Optional[float] = None
    ) -> float:
        """
        Calculate theoretical token price.
        
        Args:
            annual_revenue: Override annual revenue
            pe_ratio: Override P/E ratio
        
        Returns:
            Theoretical price per token in USD
        """
        revenue = annual_revenue or self.params.annual_revenue
        pe = pe_ratio or self.params.pe_ratio
        circulating = self.effective_circulating_supply()
        
        if circulating == 0:
            return 0
        
        market_cap = revenue * pe
        price = market_cap / circulating
        
        return price
    
    def effective_circulating_supply(self) -> int:
        """
        Calculate effective circulating supply.
        Considers burned and heavily staked tokens.
        """
        # Burned tokens are permanently removed
        available = self.params.circulating_supply - self.params.burned_supply
        
        # Long-term staked tokens reduce effective supply
        # Apply 50% reduction for staked tokens (they're still circulating but locked)
        effective_staked_reduction = self.params.staked_supply * 0.5
        
        return int(available - effective_staked_reduction)
    
    def calculate_fully_diluted_valuation(self, price: float) -> float:
        """Calculate FDV based on price."""
        return price * self.params.total_supply
    
    def calculate_market_cap(self, price: float) -> float:
        """Calculate market cap based on circulating supply."""
        return price * self.params.circulating_supply
    
    def project_price(
        self,
        years: int,
        revenue_growth_rate: float,
        burn_rate: float = 0.02  # 2% annual burn
    ) -> list:
        """
        Project token price over multiple years.
        
        Args:
            years: Number of years to project
            revenue_growth_rate: Annual revenue growth rate
            burn_rate: Annual token burn rate
        
        Returns:
            List of projected prices per year
        """
        projections = []
        
        current_revenue = self.params.annual_revenue
        current_supply = self.params.circulating_supply
        current_burned = self.params.burned_supply
        
        for year in range(1, years + 1):
            # Grow revenue
            current_revenue *= (1 + revenue_growth_rate)
            
            # Burn tokens
            burn_amount = int(current_supply * burn_rate)
            current_burned += burn_amount
            
            # Calculate price
            temp_params = TokenomicsParams(
                total_supply=self.params.total_supply,
                annual_revenue=current_revenue,
                pe_ratio=self.params.pe_ratio,
                circulating_supply=current_supply,
                burned_supply=current_burned,
                staked_supply=self.params.staked_supply
            )
            
            temp_model = PriceModel(temp_params)
            price = temp_model.calculate_price()
            
            projections.append({
                'year': year,
                'revenue': current_revenue,
                'circulating': current_supply,
                'burned': current_burned,
                'price': price,
                'market_cap': price * current_supply
            })
        
        return projections


# Example usage
if __name__ == "__main__":
    # Year 1 parameters
    params = TokenomicsParams(
        total_supply=1_000_000_000,
        annual_revenue=10_000_000,  # $10M revenue
        pe_ratio=20,
        circulating_supply=223_000_000,  # Year 1 circulating
        burned_supply=0,
        staked_supply=50_000_000  # 50M staked
    )
    
    model = PriceModel(params)
    
    price = model.calculate_price()
    print(f"Theoretical Price: ${price:.4f}")
    print(f"Market Cap: ${model.calculate_market_cap(price):,.0f}")
    print(f"FDV: ${model.calculate_fully_diluted_valuation(price):,.0f}")
    
    # 5-year projection
    projections = model.project_price(
        years=5,
        revenue_growth_rate=0.5,  # 50% annual growth
        burn_rate=0.02
    )
    
    print("\n5-Year Price Projection:")
    for p in projections:
        print(f"  Year {p['year']}: ${p['price']:.4f} (MC: ${p['market_cap']:,.0f})")
```

---

## Testing & Validation

### Token Economics Tests

```typescript
// test/tokenomics.test.ts
import { expect } from "chai";
import { ethers } from "hardhat";
import { time } from "@nomicfoundation/hardhat-network-helpers";

describe("Tokenomics", () => {
  let vdhToken: VDHToken;
  let vesting: TokenVesting;
  let staking: Staking;
  let buyback: BuybackBurn;
  
  const YEAR = 365 * 24 * 60 * 60;
  const MONTH = 30 * 24 * 60 * 60;
  
  beforeEach(async () => {
    // Deploy contracts
    const VDHToken = await ethers.getContractFactory("VDHToken");
    vdhToken = await VDHToken.deploy();
    
    const TokenVesting = await ethers.getContractFactory("TokenVesting");
    vesting = await TokenVesting.deploy(await vdhToken.getAddress());
    
    // ... deploy other contracts
  });
  
  describe("Token Supply", () => {
    it("should have correct max supply", async () => {
      const maxSupply = await vdhToken.MAX_SUPPLY();
      expect(maxSupply).to.equal(ethers.parseEther("1000000000"));
    });
    
    it("should not allow minting beyond max supply", async () => {
      await expect(
        vdhToken.mint(owner.address, ethers.parseEther("1000000001"))
      ).to.be.revertedWith("Max supply exceeded");
    });
  });
  
  describe("Vesting", () => {
    it("should release TGE amount immediately", async () => {
      const amount = ethers.parseEther("1000000");
      const tgePercentage = 1000; // 10%
      
      await vdhToken.mint(await vesting.getAddress(), amount);
      
      const tx = await vesting.createVestingSchedule(
        beneficiary.address,
        amount,
        await time.latest(),
        0,
        36 * MONTH,
        tgePercentage,
        false
      );
      
      const receipt = await tx.wait();
      const scheduleId = receipt.logs[0].args.scheduleId;
      
      const releasable = await vesting.releasableAmount(scheduleId);
      expect(releasable).to.equal(ethers.parseEther("100000")); // 10%
    });
    
    it("should respect cliff period", async () => {
      const amount = ethers.parseEther("1000000");
      
      await vdhToken.mint(await vesting.getAddress(), amount);
      
      const startTime = await time.latest();
      const tx = await vesting.createVestingSchedule(
        beneficiary.address,
        amount,
        startTime,
        3 * YEAR, // 3 year cliff
        5 * YEAR, // 5 year total
        0,
        false
      );
      
      const receipt = await tx.wait();
      const scheduleId = receipt.logs[0].args.scheduleId;
      
      // Advance 2 years (still in cliff)
      await time.increase(2 * YEAR);
      
      const releasable = await vesting.releasableAmount(scheduleId);
      expect(releasable).to.equal(0);
      
      // Advance past cliff
      await time.increase(1 * YEAR + 1);
      
      const releasableAfterCliff = await vesting.releasableAmount(scheduleId);
      expect(releasableAfterCliff).to.be.gt(0);
    });
    
    it("should vest linearly after cliff", async () => {
      const amount = ethers.parseEther("1000000");
      
      await vdhToken.mint(await vesting.getAddress(), amount);
      
      const startTime = await time.latest();
      await vesting.createVestingSchedule(
        beneficiary.address,
        amount,
        startTime,
        YEAR,      // 1 year cliff
        2 * YEAR,  // 2 year total (1 year vesting after cliff)
        0,
        false
      );
      
      // Advance to halfway through vesting
      await time.increase(1.5 * YEAR);
      
      // Should have ~50% vested
      const vested = await vesting.vestedAmount(scheduleId);
      const expected = ethers.parseEther("500000");
      
      // Allow 1% tolerance for timing
      expect(vested).to.be.closeTo(expected, ethers.parseEther("10000"));
    });
  });
  
  describe("Buyback & Burn", () => {
    it("should correctly split buyback between burn and treasury", async () => {
      // Setup: deposit revenue
      const revenue = ethers.parseEther("1000000"); // $1M
      await usdc.approve(await buyback.getAddress(), revenue);
      await buyback.depositRevenue(revenue);
      
      // 30% goes to buyback = $300K
      const accumulated = await buyback.accumulatedRevenue();
      expect(accumulated).to.equal(ethers.parseEther("300000"));
      
      // Execute buyback
      await buyback.executeBuyback(0); // min 0 for testing
      
      const stats = await buyback.quarterlyStats(1);
      
      // 50% burned, 50% to treasury
      expect(stats.vdhBurned).to.equal(stats.vdhToTreasury);
    });
    
    it("should reduce circulating supply after burn", async () => {
      const initialSupply = await vdhToken.totalSupply();
      
      // Execute buyback
      await buyback.executeBuyback(0);
      
      const finalSupply = await vdhToken.totalSupply();
      const totalBurned = await buyback.totalBurned();
      
      expect(initialSupply - finalSupply).to.equal(totalBurned);
    });
  });
  
  describe("Staking Governance Power", () => {
    it("should calculate voting power based on lock period", async () => {
      const stakeAmount = ethers.parseEther("10000");
      
      // Stake for 30 days (1x multiplier)
      await staking.stake(stakeAmount, 30 * 24 * 60 * 60);
      
      let power = await staking.votingPower(user1.address);
      expect(power).to.equal(stakeAmount); // 1x
      
      // Stake for 365 days (2x multiplier)
      await staking.connect(user2).stake(stakeAmount, 365 * 24 * 60 * 60);
      
      power = await staking.votingPower(user2.address);
      expect(power).to.equal(stakeAmount * 2n); // 2x
    });
  });
});
```

### Acceptance Criteria

```markdown
## Tokenomics Acceptance Criteria

### Token Distribution
- [ ] Total supply fixed at 1 billion
- [ ] All allocation percentages correct
- [ ] Vesting schedules properly configured
- [ ] TGE releases working correctly

### Vesting
- [ ] Cliff periods enforced
- [ ] Linear vesting after cliff
- [ ] Revocation working for revocable schedules
- [ ] Multiple beneficiaries supported

### Staking
- [ ] Lock multipliers applied correctly
- [ ] Early withdrawal penalties working
- [ ] Governance voting power calculated
- [ ] Delegation functional

### Buyback & Burn
- [ ] 30% of revenue allocated
- [ ] 50/50 split between burn and treasury
- [ ] Quarterly execution working
- [ ] Stats tracking accurate

### Price Model
- [ ] Formula implementation correct
- [ ] Circulating supply tracking accurate
- [ ] Projections reasonable
```

---

## Next Steps

After completing tokenomics:
1. Proceed to `07_FRONTEND_DOCUMENTATION.md`
2. Deploy vesting contracts
3. Update `TRACKER.md` with completion status

---

*Document Version: 1.0.0*
