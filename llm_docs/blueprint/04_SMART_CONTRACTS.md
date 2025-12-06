# Smart Contracts Implementation Guide

> Detailed implementation guide for VIDDHANA core smart contracts on Atlas Chain

---

## Table of Contents
1. [Overview](#overview)
2. [Contract Architecture](#contract-architecture)
3. [Core Contracts](#core-contracts)
4. [Token Contracts](#token-contracts)
5. [Oracle Integration](#oracle-integration)
6. [Deployment Guide](#deployment-guide)
7. [Security Considerations](#security-considerations)
8. [Testing Suite](#testing-suite)

---

## Overview

VIDDHANA's smart contract layer consists of modular, upgradeable contracts that handle:
- **Policy Engine**: AI-driven auto-rebalancing rules
- **Vault Manager**: User fund custody and management
- **Risk Controller**: Risk threshold enforcement
- **VDH Token**: Native ERC-20 token
- **Staking**: Staking mechanics and rewards
- **Governance**: DAO voting system

---

## Contract Architecture

### System Diagram

```
+------------------------------------------------------------------+
|                    VIDDHANA CONTRACT ARCHITECTURE                 |
+------------------------------------------------------------------+
|                                                                   |
|  +-------------------+        +-------------------+               |
|  |   ProxyAdmin      |        |   AccessControl   |               |
|  |   (Upgrades)      |        |   (Permissions)   |               |
|  +-------------------+        +-------------------+               |
|           |                            |                          |
|           v                            v                          |
|  +--------------------------------------------------------+      |
|  |                     CORE CONTRACTS                      |      |
|  |  +----------------+  +----------------+  +------------+ |      |
|  |  | PolicyEngine   |  | VaultManager   |  | RiskCtrl   | |      |
|  |  +----------------+  +----------------+  +------------+ |      |
|  +--------------------------------------------------------+      |
|           |                     |                |                |
|           v                     v                v                |
|  +--------------------------------------------------------+      |
|  |                    TOKEN CONTRACTS                      |      |
|  |  +----------------+  +----------------+  +------------+ |      |
|  |  | VDHToken       |  | Staking        |  | Governance | |      |
|  |  +----------------+  +----------------+  +------------+ |      |
|  +--------------------------------------------------------+      |
|           |                                                       |
|           v                                                       |
|  +-------------------+        +-------------------+               |
|  |   OracleVerifier  |        |   PriceOracle     |               |
|  |   (DePIN Data)    |        |   (Asset Prices)  |               |
|  +-------------------+        +-------------------+               |
|                                                                   |
+------------------------------------------------------------------+
```

### Access Control Roles

```solidity
// Role definitions
bytes32 public constant AI_ROLE = keccak256("AI_ROLE");
bytes32 public constant ADMIN_ROLE = keccak256("ADMIN_ROLE");
bytes32 public constant OPERATOR_ROLE = keccak256("OPERATOR_ROLE");
bytes32 public constant ORACLE_ROLE = keccak256("ORACLE_ROLE");
bytes32 public constant PAUSER_ROLE = keccak256("PAUSER_ROLE");
```

---

## Core Contracts

### PolicyEngine.sol

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts-upgradeable/access/AccessControlUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/security/PausableUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/security/ReentrancyGuardUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/proxy/utils/Initializable.sol";
import "@openzeppelin/contracts-upgradeable/proxy/utils/UUPSUpgradeable.sol";

/**
 * @title PolicyEngine
 * @author VIDDHANA Team
 * @notice Manages AI-driven auto-rebalancing policies for user portfolios
 * @dev Implements the policy logic described in the VIDDHANA Whitepaper
 */
contract PolicyEngine is 
    Initializable, 
    AccessControlUpgradeable, 
    PausableUpgradeable,
    ReentrancyGuardUpgradeable,
    UUPSUpgradeable 
{
    // ============ Constants ============
    bytes32 public constant AI_ROLE = keccak256("AI_ROLE");
    bytes32 public constant UPGRADER_ROLE = keccak256("UPGRADER_ROLE");
    
    uint256 public constant BASIS_POINTS = 10000;
    uint256 public constant MIN_REBALANCE_INTERVAL = 1 hours;
    uint256 public constant MAX_VOLATILE_PERCENTAGE = 2000; // 20%
    
    // ============ Structs ============
    struct UserProfile {
        uint256 riskTolerance;      // 0-10000 (basis points)
        uint256 timeToGoal;         // in months
        uint256 maxVolatileShare;   // Maximum % in volatile assets
        uint256 lastRebalance;      // Timestamp
        bool autoRebalanceEnabled;
    }
    
    struct RebalanceAction {
        address asset;
        bool isBuy;                 // true = buy, false = sell
        uint256 percentage;         // Percentage of position
        string reason;
    }
    
    struct AIRecommendation {
        address user;
        RebalanceAction[] actions;
        uint256 confidence;         // 0-10000 (basis points)
        uint256 timestamp;
        bool executed;
    }
    
    // ============ State Variables ============
    IVaultManager public vaultManager;
    IRiskController public riskController;
    IPriceOracle public priceOracle;
    
    mapping(address => UserProfile) public userProfiles;
    mapping(address => AIRecommendation[]) public userRecommendations;
    mapping(address => uint256) public lastRebalanceTime;
    
    uint256 public inflationThreshold;  // Trigger inflation protection above this
    uint256 public minConfidenceThreshold;  // Minimum AI confidence to execute
    
    // ============ Events ============
    event ProfileUpdated(address indexed user, uint256 riskTolerance, uint256 timeToGoal);
    event RebalanceExecuted(address indexed user, string reason, uint256 timestamp);
    event AIRecommendationReceived(address indexed user, uint256 confidence);
    event InflationProtectionTriggered(address indexed user, uint256 inflationRate);
    event EmergencyPaused(address indexed pauser);
    
    // ============ Errors ============
    error RebalanceTooSoon(uint256 nextAllowedTime);
    error InsufficientConfidence(uint256 provided, uint256 required);
    error InvalidRiskTolerance(uint256 provided);
    error UserNotRegistered(address user);
    error AutoRebalanceDisabled(address user);
    
    // ============ Initializer ============
    function initialize(
        address _vaultManager,
        address _riskController,
        address _priceOracle
    ) public initializer {
        __AccessControl_init();
        __Pausable_init();
        __ReentrancyGuard_init();
        __UUPSUpgradeable_init();
        
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(UPGRADER_ROLE, msg.sender);
        
        vaultManager = IVaultManager(_vaultManager);
        riskController = IRiskController(_riskController);
        priceOracle = IPriceOracle(_priceOracle);
        
        inflationThreshold = 600;  // 6%
        minConfidenceThreshold = 7000;  // 70%
    }
    
    // ============ User Functions ============
    
    /**
     * @notice Register or update user profile
     * @param riskTolerance Risk tolerance (0-10000 basis points)
     * @param timeToGoal Time to investment goal in months
     * @param autoRebalance Enable automatic rebalancing
     */
    function setUserProfile(
        uint256 riskTolerance,
        uint256 timeToGoal,
        bool autoRebalance
    ) external {
        if (riskTolerance > BASIS_POINTS) {
            revert InvalidRiskTolerance(riskTolerance);
        }
        
        // Calculate max volatile share based on risk tolerance and time to goal
        uint256 maxVolatile = _calculateMaxVolatileShare(riskTolerance, timeToGoal);
        
        userProfiles[msg.sender] = UserProfile({
            riskTolerance: riskTolerance,
            timeToGoal: timeToGoal,
            maxVolatileShare: maxVolatile,
            lastRebalance: block.timestamp,
            autoRebalanceEnabled: autoRebalance
        });
        
        emit ProfileUpdated(msg.sender, riskTolerance, timeToGoal);
    }
    
    /**
     * @notice Check if user can be rebalanced
     */
    function canRebalance(address user) external view returns (bool) {
        UserProfile memory profile = userProfiles[user];
        
        if (!profile.autoRebalanceEnabled) return false;
        if (block.timestamp < profile.lastRebalance + MIN_REBALANCE_INTERVAL) return false;
        
        return true;
    }
    
    // ============ AI Functions ============
    
    /**
     * @notice Submit AI recommendation for a user
     * @dev Only callable by addresses with AI_ROLE
     * @param user User address
     * @param actions Array of rebalancing actions
     * @param confidence Confidence level (0-10000)
     */
    function submitAIRecommendation(
        address user,
        RebalanceAction[] calldata actions,
        uint256 confidence
    ) external onlyRole(AI_ROLE) whenNotPaused {
        if (userProfiles[user].riskTolerance == 0) {
            revert UserNotRegistered(user);
        }
        
        // Store recommendation
        AIRecommendation storage recommendation = userRecommendations[user].push();
        recommendation.user = user;
        recommendation.confidence = confidence;
        recommendation.timestamp = block.timestamp;
        recommendation.executed = false;
        
        for (uint256 i = 0; i < actions.length; i++) {
            recommendation.actions.push(actions[i]);
        }
        
        emit AIRecommendationReceived(user, confidence);
        
        // Auto-execute if confidence is high enough
        if (confidence >= minConfidenceThreshold && userProfiles[user].autoRebalanceEnabled) {
            _executeRebalance(user, actions, "AI Auto-Rebalance");
        }
    }
    
    /**
     * @notice Trigger auto-rebalancing based on market conditions
     * @dev Called by Prometheus AI when conditions are met
     * @param user User address
     * @param inflationRate Current inflation rate (basis points)
     */
    function autoRebalance(
        address user,
        uint256 inflationRate
    ) external onlyRole(AI_ROLE) whenNotPaused nonReentrant {
        UserProfile storage profile = userProfiles[user];
        
        if (!profile.autoRebalanceEnabled) {
            revert AutoRebalanceDisabled(user);
        }
        
        if (block.timestamp < profile.lastRebalance + MIN_REBALANCE_INTERVAL) {
            revert RebalanceTooSoon(profile.lastRebalance + MIN_REBALANCE_INTERVAL);
        }
        
        // Check risk controller approval
        require(
            riskController.approveRebalance(user),
            "Risk controller rejected"
        );
        
        // Rule 1: If goal is near (<= 12 months), reduce volatility
        if (profile.timeToGoal <= 12) {
            _executeGoalProximityRebalance(user, profile);
        }
        
        // Rule 2: Inflation protection
        if (inflationRate > inflationThreshold) {
            _executeInflationProtection(user, inflationRate);
            emit InflationProtectionTriggered(user, inflationRate);
        }
        
        // Update last rebalance time
        profile.lastRebalance = block.timestamp;
    }
    
    // ============ Internal Functions ============
    
    function _calculateMaxVolatileShare(
        uint256 riskTolerance,
        uint256 timeToGoal
    ) internal pure returns (uint256) {
        // Base allocation from risk tolerance
        uint256 baseAllocation = (riskTolerance * 50) / BASIS_POINTS;  // Max 50%
        
        // Reduce based on time to goal
        if (timeToGoal <= 12) {
            return baseAllocation / 2;  // Halve if goal is near
        } else if (timeToGoal <= 24) {
            return (baseAllocation * 75) / 100;  // 75% if 1-2 years
        }
        
        return baseAllocation;
    }
    
    function _executeGoalProximityRebalance(
        address user,
        UserProfile memory profile
    ) internal {
        // Get current volatile asset percentage
        uint256 volatileShare = vaultManager.getVolatileAssetPercentage(user);
        
        // If volatile assets > max allowed, sell excess
        if (volatileShare > profile.maxVolatileShare) {
            uint256 excessPercentage = volatileShare - profile.maxVolatileShare;
            
            RebalanceAction[] memory actions = new RebalanceAction[](2);
            
            // Sell volatile assets
            actions[0] = RebalanceAction({
                asset: address(0),  // Placeholder - actual assets determined by VaultManager
                isBuy: false,
                percentage: excessPercentage,
                reason: "Goal proximity risk reduction"
            });
            
            // Buy stable assets
            actions[1] = RebalanceAction({
                asset: address(0),
                isBuy: true,
                percentage: excessPercentage,
                reason: "Goal proximity stability increase"
            });
            
            _executeRebalance(user, actions, "Goal Proximity Mode");
        }
    }
    
    function _executeInflationProtection(
        address user,
        uint256 inflationRate
    ) internal {
        // Shift cash/stables to inflation hedges (DeFi yield, RWA)
        uint256 cashPercentage = vaultManager.getCashPercentage(user);
        
        if (cashPercentage > 1000) {  // > 10%
            uint256 shiftAmount = cashPercentage / 2;  // Shift 50% of cash
            
            RebalanceAction[] memory actions = new RebalanceAction[](2);
            
            actions[0] = RebalanceAction({
                asset: address(0),
                isBuy: false,
                percentage: shiftAmount,
                reason: "Inflation protection - reduce cash"
            });
            
            actions[1] = RebalanceAction({
                asset: address(0),
                isBuy: true,
                percentage: shiftAmount,
                reason: "Inflation protection - add yield"
            });
            
            _executeRebalance(user, actions, "Inflation Protection Mode");
        }
    }
    
    function _executeRebalance(
        address user,
        RebalanceAction[] memory actions,
        string memory reason
    ) internal {
        // Execute through VaultManager
        for (uint256 i = 0; i < actions.length; i++) {
            if (actions[i].isBuy) {
                vaultManager.executeBuy(
                    user,
                    actions[i].asset,
                    actions[i].percentage
                );
            } else {
                vaultManager.executeSell(
                    user,
                    actions[i].asset,
                    actions[i].percentage
                );
            }
        }
        
        emit RebalanceExecuted(user, reason, block.timestamp);
    }
    
    // ============ Admin Functions ============
    
    function setInflationThreshold(uint256 _threshold) external onlyRole(DEFAULT_ADMIN_ROLE) {
        inflationThreshold = _threshold;
    }
    
    function setMinConfidenceThreshold(uint256 _threshold) external onlyRole(DEFAULT_ADMIN_ROLE) {
        require(_threshold <= BASIS_POINTS, "Invalid threshold");
        minConfidenceThreshold = _threshold;
    }
    
    function pause() external onlyRole(DEFAULT_ADMIN_ROLE) {
        _pause();
        emit EmergencyPaused(msg.sender);
    }
    
    function unpause() external onlyRole(DEFAULT_ADMIN_ROLE) {
        _unpause();
    }
    
    // ============ Upgrade Authorization ============
    
    function _authorizeUpgrade(address newImplementation) internal override onlyRole(UPGRADER_ROLE) {}
}
```

### VaultManager.sol

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts-upgradeable/access/AccessControlUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/security/ReentrancyGuardUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/token/ERC20/utils/SafeERC20Upgradeable.sol";
import "@openzeppelin/contracts-upgradeable/proxy/utils/UUPSUpgradeable.sol";

/**
 * @title VaultManager
 * @notice Manages user fund custody and portfolio operations
 */
contract VaultManager is 
    Initializable,
    AccessControlUpgradeable,
    ReentrancyGuardUpgradeable,
    UUPSUpgradeable
{
    using SafeERC20Upgradeable for IERC20Upgradeable;
    
    // ============ Constants ============
    bytes32 public constant POLICY_ENGINE_ROLE = keccak256("POLICY_ENGINE_ROLE");
    uint256 public constant BASIS_POINTS = 10000;
    
    // ============ Structs ============
    struct Portfolio {
        uint256 totalValue;
        mapping(address => uint256) balances;
        address[] assets;
        uint256 lastUpdated;
    }
    
    struct AssetInfo {
        address token;
        string symbol;
        bool isVolatile;
        bool isYieldBearing;
        bool isEnabled;
    }
    
    // ============ State Variables ============
    mapping(address => Portfolio) private portfolios;
    mapping(address => AssetInfo) public assetRegistry;
    address[] public supportedAssets;
    
    IPriceOracle public priceOracle;
    address public treasury;
    
    uint256 public depositFee;      // Basis points
    uint256 public withdrawalFee;   // Basis points
    uint256 public managementFee;   // Annual, basis points
    
    // ============ Events ============
    event Deposited(address indexed user, address indexed asset, uint256 amount);
    event Withdrawn(address indexed user, address indexed asset, uint256 amount);
    event SwapExecuted(address indexed user, address fromAsset, address toAsset, uint256 amount);
    event PortfolioUpdated(address indexed user, uint256 totalValue);
    
    // ============ Errors ============
    error AssetNotSupported(address asset);
    error InsufficientBalance(uint256 available, uint256 required);
    error InvalidAmount();
    
    // ============ Initializer ============
    function initialize(
        address _priceOracle,
        address _treasury
    ) public initializer {
        __AccessControl_init();
        __ReentrancyGuard_init();
        __UUPSUpgradeable_init();
        
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        
        priceOracle = IPriceOracle(_priceOracle);
        treasury = _treasury;
        
        depositFee = 10;      // 0.1%
        withdrawalFee = 10;   // 0.1%
        managementFee = 100;  // 1% annual
    }
    
    // ============ User Functions ============
    
    /**
     * @notice Deposit assets into the vault
     * @param asset Token address
     * @param amount Amount to deposit
     */
    function deposit(
        address asset,
        uint256 amount
    ) external nonReentrant {
        if (!assetRegistry[asset].isEnabled) {
            revert AssetNotSupported(asset);
        }
        if (amount == 0) {
            revert InvalidAmount();
        }
        
        // Calculate fee
        uint256 fee = (amount * depositFee) / BASIS_POINTS;
        uint256 netAmount = amount - fee;
        
        // Transfer tokens
        IERC20Upgradeable(asset).safeTransferFrom(msg.sender, address(this), amount);
        
        // Send fee to treasury
        if (fee > 0) {
            IERC20Upgradeable(asset).safeTransfer(treasury, fee);
        }
        
        // Update portfolio
        _updatePortfolioBalance(msg.sender, asset, netAmount, true);
        
        emit Deposited(msg.sender, asset, netAmount);
    }
    
    /**
     * @notice Withdraw assets from the vault
     * @param asset Token address
     * @param amount Amount to withdraw
     */
    function withdraw(
        address asset,
        uint256 amount
    ) external nonReentrant {
        Portfolio storage portfolio = portfolios[msg.sender];
        
        if (portfolio.balances[asset] < amount) {
            revert InsufficientBalance(portfolio.balances[asset], amount);
        }
        
        // Calculate fee
        uint256 fee = (amount * withdrawalFee) / BASIS_POINTS;
        uint256 netAmount = amount - fee;
        
        // Update portfolio
        _updatePortfolioBalance(msg.sender, asset, amount, false);
        
        // Transfer tokens
        IERC20Upgradeable(asset).safeTransfer(msg.sender, netAmount);
        
        // Send fee to treasury
        if (fee > 0) {
            IERC20Upgradeable(asset).safeTransfer(treasury, fee);
        }
        
        emit Withdrawn(msg.sender, asset, netAmount);
    }
    
    /**
     * @notice Get user's portfolio value and allocation
     */
    function getUserPortfolio(address user) external view returns (
        uint256 totalValue,
        address[] memory assets,
        uint256[] memory balances,
        uint256[] memory allocations
    ) {
        Portfolio storage portfolio = portfolios[user];
        
        assets = portfolio.assets;
        balances = new uint256[](assets.length);
        allocations = new uint256[](assets.length);
        totalValue = 0;
        
        for (uint256 i = 0; i < assets.length; i++) {
            balances[i] = portfolio.balances[assets[i]];
            uint256 assetValue = _getAssetValue(assets[i], balances[i]);
            totalValue += assetValue;
        }
        
        // Calculate allocations
        for (uint256 i = 0; i < assets.length; i++) {
            if (totalValue > 0) {
                uint256 assetValue = _getAssetValue(assets[i], balances[i]);
                allocations[i] = (assetValue * BASIS_POINTS) / totalValue;
            }
        }
        
        return (totalValue, assets, balances, allocations);
    }
    
    /**
     * @notice Get percentage of volatile assets in portfolio
     */
    function getVolatileAssetPercentage(address user) external view returns (uint256) {
        Portfolio storage portfolio = portfolios[user];
        
        uint256 totalValue = 0;
        uint256 volatileValue = 0;
        
        for (uint256 i = 0; i < portfolio.assets.length; i++) {
            address asset = portfolio.assets[i];
            uint256 balance = portfolio.balances[asset];
            uint256 value = _getAssetValue(asset, balance);
            
            totalValue += value;
            
            if (assetRegistry[asset].isVolatile) {
                volatileValue += value;
            }
        }
        
        if (totalValue == 0) return 0;
        
        return (volatileValue * BASIS_POINTS) / totalValue;
    }
    
    /**
     * @notice Get percentage of cash/stables in portfolio
     */
    function getCashPercentage(address user) external view returns (uint256) {
        Portfolio storage portfolio = portfolios[user];
        
        uint256 totalValue = 0;
        uint256 cashValue = 0;
        
        for (uint256 i = 0; i < portfolio.assets.length; i++) {
            address asset = portfolio.assets[i];
            uint256 balance = portfolio.balances[asset];
            uint256 value = _getAssetValue(asset, balance);
            
            totalValue += value;
            
            if (!assetRegistry[asset].isVolatile && !assetRegistry[asset].isYieldBearing) {
                cashValue += value;
            }
        }
        
        if (totalValue == 0) return 0;
        
        return (cashValue * BASIS_POINTS) / totalValue;
    }
    
    // ============ Policy Engine Functions ============
    
    /**
     * @notice Execute a buy order for a user
     * @dev Only callable by PolicyEngine
     */
    function executeBuy(
        address user,
        address asset,
        uint256 percentage
    ) external onlyRole(POLICY_ENGINE_ROLE) nonReentrant {
        // Implementation: Swap from stable to target asset
        _executeSwap(user, _getStableAsset(), asset, percentage);
    }
    
    /**
     * @notice Execute a sell order for a user
     * @dev Only callable by PolicyEngine
     */
    function executeSell(
        address user,
        address asset,
        uint256 percentage
    ) external onlyRole(POLICY_ENGINE_ROLE) nonReentrant {
        // Implementation: Swap from target asset to stable
        _executeSwap(user, asset, _getStableAsset(), percentage);
    }
    
    // ============ Internal Functions ============
    
    function _updatePortfolioBalance(
        address user,
        address asset,
        uint256 amount,
        bool isDeposit
    ) internal {
        Portfolio storage portfolio = portfolios[user];
        
        if (isDeposit) {
            if (portfolio.balances[asset] == 0) {
                portfolio.assets.push(asset);
            }
            portfolio.balances[asset] += amount;
        } else {
            portfolio.balances[asset] -= amount;
        }
        
        portfolio.lastUpdated = block.timestamp;
        
        // Recalculate total value
        uint256 totalValue = 0;
        for (uint256 i = 0; i < portfolio.assets.length; i++) {
            totalValue += _getAssetValue(
                portfolio.assets[i],
                portfolio.balances[portfolio.assets[i]]
            );
        }
        portfolio.totalValue = totalValue;
        
        emit PortfolioUpdated(user, totalValue);
    }
    
    function _getAssetValue(address asset, uint256 amount) internal view returns (uint256) {
        uint256 price = priceOracle.getPrice(asset);
        return (amount * price) / 1e18;
    }
    
    function _executeSwap(
        address user,
        address fromAsset,
        address toAsset,
        uint256 percentage
    ) internal {
        Portfolio storage portfolio = portfolios[user];
        
        uint256 fromBalance = portfolio.balances[fromAsset];
        uint256 swapAmount = (fromBalance * percentage) / BASIS_POINTS;
        
        if (swapAmount == 0) return;
        
        // Get prices
        uint256 fromPrice = priceOracle.getPrice(fromAsset);
        uint256 toPrice = priceOracle.getPrice(toAsset);
        
        // Calculate output amount (simplified, production would use DEX)
        uint256 toAmount = (swapAmount * fromPrice) / toPrice;
        
        // Update balances
        portfolio.balances[fromAsset] -= swapAmount;
        
        if (portfolio.balances[toAsset] == 0) {
            portfolio.assets.push(toAsset);
        }
        portfolio.balances[toAsset] += toAmount;
        
        emit SwapExecuted(user, fromAsset, toAsset, swapAmount);
    }
    
    function _getStableAsset() internal view returns (address) {
        // Return default stable asset (e.g., USDC)
        return supportedAssets[0];  // Assumes first asset is stable
    }
    
    // ============ Admin Functions ============
    
    function addSupportedAsset(
        address token,
        string calldata symbol,
        bool isVolatile,
        bool isYieldBearing
    ) external onlyRole(DEFAULT_ADMIN_ROLE) {
        assetRegistry[token] = AssetInfo({
            token: token,
            symbol: symbol,
            isVolatile: isVolatile,
            isYieldBearing: isYieldBearing,
            isEnabled: true
        });
        supportedAssets.push(token);
    }
    
    function _authorizeUpgrade(address newImplementation) internal override onlyRole(DEFAULT_ADMIN_ROLE) {}
}
```

### RiskController.sol

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts-upgradeable/access/AccessControlUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/proxy/utils/UUPSUpgradeable.sol";

/**
 * @title RiskController
 * @notice Enforces risk limits and safety thresholds for portfolio operations
 */
contract RiskController is Initializable, AccessControlUpgradeable, UUPSUpgradeable {
    
    // ============ Structs ============
    struct RiskLimits {
        uint256 maxSingleAssetAllocation;   // Max % in single asset
        uint256 maxVolatileAllocation;       // Max % in volatile assets
        uint256 maxDailyTurnover;            // Max % portfolio turnover per day
        uint256 maxDrawdown;                 // Max allowed drawdown before pause
    }
    
    // ============ State Variables ============
    IVaultManager public vaultManager;
    IPriceOracle public priceOracle;
    
    RiskLimits public defaultLimits;
    mapping(address => RiskLimits) public customLimits;
    mapping(address => uint256) public dailyTurnover;
    mapping(address => uint256) public lastTurnoverReset;
    mapping(address => uint256) public portfolioHighWaterMark;
    
    // ============ Events ============
    event RiskLimitBreached(address indexed user, string limitType, uint256 value, uint256 limit);
    event RebalanceRejected(address indexed user, string reason);
    
    // ============ Initializer ============
    function initialize(
        address _vaultManager,
        address _priceOracle
    ) public initializer {
        __AccessControl_init();
        __UUPSUpgradeable_init();
        
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        
        vaultManager = IVaultManager(_vaultManager);
        priceOracle = IPriceOracle(_priceOracle);
        
        // Set default limits
        defaultLimits = RiskLimits({
            maxSingleAssetAllocation: 4000,  // 40%
            maxVolatileAllocation: 6000,      // 60%
            maxDailyTurnover: 2500,           // 25%
            maxDrawdown: 2000                 // 20%
        });
    }
    
    // ============ Risk Check Functions ============
    
    /**
     * @notice Check if a rebalance operation should be approved
     * @param user User address
     * @return approved Whether the rebalance is approved
     */
    function approveRebalance(address user) external returns (bool approved) {
        // Reset daily turnover if needed
        if (block.timestamp > lastTurnoverReset[user] + 1 days) {
            dailyTurnover[user] = 0;
            lastTurnoverReset[user] = block.timestamp;
        }
        
        RiskLimits memory limits = _getLimits(user);
        
        // Check daily turnover limit
        if (dailyTurnover[user] >= limits.maxDailyTurnover) {
            emit RebalanceRejected(user, "Daily turnover limit exceeded");
            return false;
        }
        
        // Check drawdown
        (uint256 currentValue,,,) = vaultManager.getUserPortfolio(user);
        uint256 highWaterMark = portfolioHighWaterMark[user];
        
        if (highWaterMark > 0) {
            uint256 drawdown = ((highWaterMark - currentValue) * 10000) / highWaterMark;
            if (drawdown >= limits.maxDrawdown) {
                emit RiskLimitBreached(user, "drawdown", drawdown, limits.maxDrawdown);
                emit RebalanceRejected(user, "Max drawdown reached");
                return false;
            }
        }
        
        // Update high water mark
        if (currentValue > highWaterMark) {
            portfolioHighWaterMark[user] = currentValue;
        }
        
        return true;
    }
    
    /**
     * @notice Validate a proposed allocation
     * @param user User address
     * @param asset Asset address
     * @param newAllocation Proposed allocation (basis points)
     */
    function validateAllocation(
        address user,
        address asset,
        uint256 newAllocation
    ) external view returns (bool valid, string memory reason) {
        RiskLimits memory limits = _getLimits(user);
        
        // Check single asset limit
        if (newAllocation > limits.maxSingleAssetAllocation) {
            return (false, "Single asset limit exceeded");
        }
        
        // Check if adding to volatile assets exceeds limit
        // (This would need actual allocation data from VaultManager)
        
        return (true, "");
    }
    
    /**
     * @notice Record turnover for rate limiting
     */
    function recordTurnover(address user, uint256 turnoverAmount) external {
        require(
            hasRole(DEFAULT_ADMIN_ROLE, msg.sender) || msg.sender == address(vaultManager),
            "Unauthorized"
        );
        
        dailyTurnover[user] += turnoverAmount;
    }
    
    // ============ Internal Functions ============
    
    function _getLimits(address user) internal view returns (RiskLimits memory) {
        RiskLimits memory custom = customLimits[user];
        
        // Return custom limits if set, otherwise default
        if (custom.maxSingleAssetAllocation > 0) {
            return custom;
        }
        return defaultLimits;
    }
    
    // ============ Admin Functions ============
    
    function setDefaultLimits(RiskLimits calldata limits) external onlyRole(DEFAULT_ADMIN_ROLE) {
        defaultLimits = limits;
    }
    
    function setCustomLimits(address user, RiskLimits calldata limits) external onlyRole(DEFAULT_ADMIN_ROLE) {
        customLimits[user] = limits;
    }
    
    function _authorizeUpgrade(address newImplementation) internal override onlyRole(DEFAULT_ADMIN_ROLE) {}
}
```

---

## Token Contracts

### VDHToken.sol

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Burnable.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Permit.sol";
import "@openzeppelin/contracts/access/AccessControl.sol";

/**
 * @title VDHToken
 * @notice Native token for the VIDDHANA ecosystem
 * @dev Fixed supply of 1 billion tokens
 */
contract VDHToken is ERC20, ERC20Burnable, ERC20Permit, AccessControl {
    
    bytes32 public constant MINTER_ROLE = keccak256("MINTER_ROLE");
    
    uint256 public constant MAX_SUPPLY = 1_000_000_000 * 10**18;  // 1 billion
    
    constructor() ERC20("Viddhana Token", "VDH") ERC20Permit("Viddhana Token") {
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(MINTER_ROLE, msg.sender);
    }
    
    /**
     * @notice Mint new tokens
     * @dev Only callable by addresses with MINTER_ROLE
     */
    function mint(address to, uint256 amount) external onlyRole(MINTER_ROLE) {
        require(totalSupply() + amount <= MAX_SUPPLY, "Max supply exceeded");
        _mint(to, amount);
    }
}
```

### Staking.sol

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts-upgradeable/token/ERC20/utils/SafeERC20Upgradeable.sol";
import "@openzeppelin/contracts-upgradeable/security/ReentrancyGuardUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/access/AccessControlUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/proxy/utils/UUPSUpgradeable.sol";

/**
 * @title Staking
 * @notice Staking contract for VDH tokens with rewards and governance power
 */
contract Staking is 
    Initializable,
    ReentrancyGuardUpgradeable,
    AccessControlUpgradeable,
    UUPSUpgradeable
{
    using SafeERC20Upgradeable for IERC20Upgradeable;
    
    // ============ Structs ============
    struct StakeInfo {
        uint256 amount;
        uint256 startTime;
        uint256 lockPeriod;     // in seconds
        uint256 rewardDebt;
        uint256 pendingRewards;
    }
    
    struct Pool {
        uint256 totalStaked;
        uint256 accRewardPerShare;
        uint256 lastRewardTime;
        uint256 rewardRate;     // Rewards per second
    }
    
    // ============ State Variables ============
    IERC20Upgradeable public vdhToken;
    Pool public pool;
    
    mapping(address => StakeInfo) public stakes;
    
    uint256 public constant PRECISION = 1e18;
    uint256 public minStakeAmount;
    uint256 public earlyWithdrawPenalty;  // Basis points
    
    // Lock period multipliers (basis points, 10000 = 1x)
    mapping(uint256 => uint256) public lockMultipliers;
    
    // ============ Events ============
    event Staked(address indexed user, uint256 amount, uint256 lockPeriod);
    event Unstaked(address indexed user, uint256 amount, uint256 penalty);
    event RewardsClaimed(address indexed user, uint256 amount);
    event RewardRateUpdated(uint256 newRate);
    
    // ============ Initializer ============
    function initialize(address _vdhToken) public initializer {
        __ReentrancyGuard_init();
        __AccessControl_init();
        __UUPSUpgradeable_init();
        
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        
        vdhToken = IERC20Upgradeable(_vdhToken);
        
        minStakeAmount = 100 * 10**18;  // 100 VDH minimum
        earlyWithdrawPenalty = 1000;    // 10%
        
        // Set lock multipliers
        lockMultipliers[30 days] = 10000;    // 1x
        lockMultipliers[90 days] = 12500;    // 1.25x
        lockMultipliers[180 days] = 15000;   // 1.5x
        lockMultipliers[365 days] = 20000;   // 2x
        
        pool = Pool({
            totalStaked: 0,
            accRewardPerShare: 0,
            lastRewardTime: block.timestamp,
            rewardRate: 0
        });
    }
    
    // ============ User Functions ============
    
    /**
     * @notice Stake VDH tokens
     * @param amount Amount to stake
     * @param lockPeriod Lock period in seconds (30, 90, 180, or 365 days)
     */
    function stake(uint256 amount, uint256 lockPeriod) external nonReentrant {
        require(amount >= minStakeAmount, "Below minimum stake");
        require(lockMultipliers[lockPeriod] > 0, "Invalid lock period");
        
        _updatePool();
        
        StakeInfo storage stakeInfo = stakes[msg.sender];
        
        // If user has existing stake, claim pending rewards
        if (stakeInfo.amount > 0) {
            uint256 pending = _pendingRewards(msg.sender);
            if (pending > 0) {
                stakeInfo.pendingRewards += pending;
            }
        }
        
        // Transfer tokens
        vdhToken.safeTransferFrom(msg.sender, address(this), amount);
        
        // Update stake info
        stakeInfo.amount += amount;
        stakeInfo.startTime = block.timestamp;
        stakeInfo.lockPeriod = lockPeriod;
        stakeInfo.rewardDebt = (stakeInfo.amount * pool.accRewardPerShare) / PRECISION;
        
        pool.totalStaked += amount;
        
        emit Staked(msg.sender, amount, lockPeriod);
    }
    
    /**
     * @notice Unstake VDH tokens
     * @param amount Amount to unstake
     */
    function unstake(uint256 amount) external nonReentrant {
        StakeInfo storage stakeInfo = stakes[msg.sender];
        require(stakeInfo.amount >= amount, "Insufficient stake");
        
        _updatePool();
        
        // Calculate and store pending rewards
        uint256 pending = _pendingRewards(msg.sender);
        stakeInfo.pendingRewards += pending;
        
        // Check if early withdrawal
        uint256 penalty = 0;
        if (block.timestamp < stakeInfo.startTime + stakeInfo.lockPeriod) {
            penalty = (amount * earlyWithdrawPenalty) / 10000;
        }
        
        // Update stake
        stakeInfo.amount -= amount;
        stakeInfo.rewardDebt = (stakeInfo.amount * pool.accRewardPerShare) / PRECISION;
        pool.totalStaked -= amount;
        
        // Transfer tokens (minus penalty)
        uint256 transferAmount = amount - penalty;
        vdhToken.safeTransfer(msg.sender, transferAmount);
        
        // Penalty goes to reward pool
        if (penalty > 0) {
            // Could redistribute to stakers or burn
        }
        
        emit Unstaked(msg.sender, amount, penalty);
    }
    
    /**
     * @notice Claim pending rewards
     */
    function claimRewards() external nonReentrant {
        _updatePool();
        
        StakeInfo storage stakeInfo = stakes[msg.sender];
        uint256 pending = _pendingRewards(msg.sender) + stakeInfo.pendingRewards;
        
        require(pending > 0, "No rewards to claim");
        
        stakeInfo.pendingRewards = 0;
        stakeInfo.rewardDebt = (stakeInfo.amount * pool.accRewardPerShare) / PRECISION;
        
        // Apply lock multiplier
        uint256 multiplier = lockMultipliers[stakeInfo.lockPeriod];
        uint256 boostedReward = (pending * multiplier) / 10000;
        
        vdhToken.safeTransfer(msg.sender, boostedReward);
        
        emit RewardsClaimed(msg.sender, boostedReward);
    }
    
    /**
     * @notice Get staked amount for a user
     */
    function stakedAmount(address user) external view returns (uint256) {
        return stakes[user].amount;
    }
    
    /**
     * @notice Get governance voting power for a user
     * @dev Voting power = staked amount * lock multiplier
     */
    function votingPower(address user) external view returns (uint256) {
        StakeInfo memory stakeInfo = stakes[user];
        if (stakeInfo.amount == 0) return 0;
        
        uint256 multiplier = lockMultipliers[stakeInfo.lockPeriod];
        return (stakeInfo.amount * multiplier) / 10000;
    }
    
    // ============ Internal Functions ============
    
    function _updatePool() internal {
        if (block.timestamp <= pool.lastRewardTime) return;
        if (pool.totalStaked == 0) {
            pool.lastRewardTime = block.timestamp;
            return;
        }
        
        uint256 timeElapsed = block.timestamp - pool.lastRewardTime;
        uint256 reward = timeElapsed * pool.rewardRate;
        
        pool.accRewardPerShare += (reward * PRECISION) / pool.totalStaked;
        pool.lastRewardTime = block.timestamp;
    }
    
    function _pendingRewards(address user) internal view returns (uint256) {
        StakeInfo memory stakeInfo = stakes[user];
        if (stakeInfo.amount == 0) return 0;
        
        uint256 accReward = pool.accRewardPerShare;
        
        if (block.timestamp > pool.lastRewardTime && pool.totalStaked > 0) {
            uint256 timeElapsed = block.timestamp - pool.lastRewardTime;
            uint256 reward = timeElapsed * pool.rewardRate;
            accReward += (reward * PRECISION) / pool.totalStaked;
        }
        
        return (stakeInfo.amount * accReward) / PRECISION - stakeInfo.rewardDebt;
    }
    
    // ============ Admin Functions ============
    
    function setRewardRate(uint256 _rewardRate) external onlyRole(DEFAULT_ADMIN_ROLE) {
        _updatePool();
        pool.rewardRate = _rewardRate;
        emit RewardRateUpdated(_rewardRate);
    }
    
    function _authorizeUpgrade(address newImplementation) internal override onlyRole(DEFAULT_ADMIN_ROLE) {}
}
```

---

## Deployment Guide

### Deployment Script

```typescript
// scripts/deploy.ts
import { ethers, upgrades } from "hardhat";

async function main() {
  const [deployer] = await ethers.getSigners();
  console.log("Deploying contracts with:", deployer.address);
  
  // 1. Deploy VDH Token
  console.log("\n1. Deploying VDH Token...");
  const VDHToken = await ethers.getContractFactory("VDHToken");
  const vdhToken = await VDHToken.deploy();
  await vdhToken.waitForDeployment();
  console.log("VDH Token deployed to:", await vdhToken.getAddress());
  
  // 2. Deploy Price Oracle (mock for testnet)
  console.log("\n2. Deploying Price Oracle...");
  const PriceOracle = await ethers.getContractFactory("MockPriceOracle");
  const priceOracle = await PriceOracle.deploy();
  await priceOracle.waitForDeployment();
  console.log("Price Oracle deployed to:", await priceOracle.getAddress());
  
  // 3. Deploy VaultManager (upgradeable)
  console.log("\n3. Deploying VaultManager...");
  const VaultManager = await ethers.getContractFactory("VaultManager");
  const vaultManager = await upgrades.deployProxy(
    VaultManager,
    [await priceOracle.getAddress(), deployer.address],
    { kind: "uups" }
  );
  await vaultManager.waitForDeployment();
  console.log("VaultManager deployed to:", await vaultManager.getAddress());
  
  // 4. Deploy RiskController (upgradeable)
  console.log("\n4. Deploying RiskController...");
  const RiskController = await ethers.getContractFactory("RiskController");
  const riskController = await upgrades.deployProxy(
    RiskController,
    [await vaultManager.getAddress(), await priceOracle.getAddress()],
    { kind: "uups" }
  );
  await riskController.waitForDeployment();
  console.log("RiskController deployed to:", await riskController.getAddress());
  
  // 5. Deploy PolicyEngine (upgradeable)
  console.log("\n5. Deploying PolicyEngine...");
  const PolicyEngine = await ethers.getContractFactory("PolicyEngine");
  const policyEngine = await upgrades.deployProxy(
    PolicyEngine,
    [
      await vaultManager.getAddress(),
      await riskController.getAddress(),
      await priceOracle.getAddress()
    ],
    { kind: "uups" }
  );
  await policyEngine.waitForDeployment();
  console.log("PolicyEngine deployed to:", await policyEngine.getAddress());
  
  // 6. Deploy Staking
  console.log("\n6. Deploying Staking...");
  const Staking = await ethers.getContractFactory("Staking");
  const staking = await upgrades.deployProxy(
    Staking,
    [await vdhToken.getAddress()],
    { kind: "uups" }
  );
  await staking.waitForDeployment();
  console.log("Staking deployed to:", await staking.getAddress());
  
  // 7. Configure roles
  console.log("\n7. Configuring roles...");
  
  // Grant AI_ROLE to Prometheus AI address (placeholder)
  const AI_ROLE = ethers.keccak256(ethers.toUtf8Bytes("AI_ROLE"));
  const AI_ADDRESS = process.env.PROMETHEUS_AI_ADDRESS || deployer.address;
  await policyEngine.grantRole(AI_ROLE, AI_ADDRESS);
  
  // Grant POLICY_ENGINE_ROLE to PolicyEngine
  const POLICY_ENGINE_ROLE = ethers.keccak256(ethers.toUtf8Bytes("POLICY_ENGINE_ROLE"));
  await vaultManager.grantRole(POLICY_ENGINE_ROLE, await policyEngine.getAddress());
  
  console.log("Roles configured successfully");
  
  // 8. Output deployment summary
  console.log("\n========== DEPLOYMENT SUMMARY ==========");
  console.log("Network:", network.name);
  console.log("VDH Token:", await vdhToken.getAddress());
  console.log("Price Oracle:", await priceOracle.getAddress());
  console.log("VaultManager:", await vaultManager.getAddress());
  console.log("RiskController:", await riskController.getAddress());
  console.log("PolicyEngine:", await policyEngine.getAddress());
  console.log("Staking:", await staking.getAddress());
  console.log("==========================================");
}

main().catch(console.error);
```

### Hardhat Configuration

```typescript
// hardhat.config.ts
import { HardhatUserConfig } from "hardhat/config";
import "@nomicfoundation/hardhat-toolbox";
import "@openzeppelin/hardhat-upgrades";
import "hardhat-gas-reporter";
import "solidity-coverage";

const config: HardhatUserConfig = {
  solidity: {
    version: "0.8.20",
    settings: {
      optimizer: {
        enabled: true,
        runs: 200
      }
    }
  },
  networks: {
    atlasTestnet: {
      url: "https://rpc.testnet.viddhana.network",
      chainId: 13370,
      accounts: process.env.PRIVATE_KEY ? [process.env.PRIVATE_KEY] : []
    },
    atlasMainnet: {
      url: "https://rpc.viddhana.network",
      chainId: 13370,
      accounts: process.env.PRIVATE_KEY ? [process.env.PRIVATE_KEY] : []
    }
  },
  gasReporter: {
    enabled: true,
    currency: "USD"
  },
  etherscan: {
    apiKey: process.env.ETHERSCAN_API_KEY
  }
};

export default config;
```

---

## Security Considerations

### Audit Checklist

```markdown
## Security Audit Checklist

### Access Control
- [ ] All admin functions protected with appropriate roles
- [ ] Role hierarchy properly configured
- [ ] No single point of failure for critical operations

### Reentrancy
- [ ] All external calls use ReentrancyGuard
- [ ] Check-Effects-Interactions pattern followed
- [ ] No callbacks to untrusted contracts

### Integer Overflow
- [ ] Solidity 0.8+ automatic overflow checks
- [ ] Explicit checks for edge cases
- [ ] Safe math operations for percentages

### Upgradability
- [ ] UUPS pattern correctly implemented
- [ ] Storage gaps for future upgrades
- [ ] Initializer functions protected

### Oracle Security
- [ ] Price manipulation protections
- [ ] Staleness checks for oracle data
- [ ] Fallback oracle mechanisms

### Economic Security
- [ ] Flash loan attack prevention
- [ ] Sandwich attack mitigation
- [ ] Slippage protection
```

---

## Next Steps

After completing smart contracts:
1. Proceed to `05_DEPIN_RWA_INTEGRATION.md` for oracle integration
2. Run security audit
3. Update `TRACKER.md` with completion status

---

*Document Version: 1.0.0*
