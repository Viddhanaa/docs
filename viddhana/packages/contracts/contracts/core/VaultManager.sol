// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts-upgradeable/proxy/utils/Initializable.sol";
import "@openzeppelin/contracts-upgradeable/access/AccessControlUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/utils/ReentrancyGuardUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/utils/PausableUpgradeable.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";

/**
 * @title VaultManager
 * @author VIDDHANA Team
 * @notice Manages user fund custody and portfolio operations
 * @dev Uses SafeERC20 from non-upgradeable contracts (OZ 5.x pattern)
 */
contract VaultManager is 
    Initializable,
    AccessControlUpgradeable,
    ReentrancyGuardUpgradeable,
    PausableUpgradeable
{
    using SafeERC20 for IERC20;
    
    // ============ Constants ============
    bytes32 public constant POLICY_ENGINE_ROLE = keccak256("POLICY_ENGINE_ROLE");
    bytes32 public constant OPERATOR_ROLE = keccak256("OPERATOR_ROLE");
    uint256 public constant BASIS_POINTS = 10000;
    
    // ============ Structs ============
    struct AssetInfo {
        address token;
        string symbol;
        bool isVolatile;
        bool isYieldBearing;
        bool isEnabled;
    }
    
    struct UserBalance {
        uint256 balance;
        uint256 lastUpdated;
    }
    
    // ============ State Variables ============
    mapping(address => mapping(address => UserBalance)) public userBalances; // user => asset => balance
    mapping(address => address[]) public userAssets; // user => list of assets
    mapping(address => AssetInfo) public assetRegistry;
    address[] public supportedAssets;
    
    address public treasury;
    
    uint256 public depositFee;      // Basis points
    uint256 public withdrawalFee;   // Basis points
    
    // ============ Events ============
    event Deposited(address indexed user, address indexed asset, uint256 amount);
    event Withdrawn(address indexed user, address indexed asset, uint256 amount);
    event AssetAdded(address indexed asset, string symbol, bool isVolatile);
    event FeesUpdated(uint256 depositFee, uint256 withdrawalFee);
    
    // ============ Errors ============
    error AssetNotSupported(address asset);
    error InsufficientBalance(uint256 available, uint256 required);
    error InvalidAmount();
    error ZeroAddress();
    
    /// @custom:oz-upgrades-unsafe-allow constructor
    constructor() {
        _disableInitializers();
    }
    
    // ============ Initializer ============
    function initialize(address _treasury) public initializer {
        __AccessControl_init();
        __ReentrancyGuard_init();
        __Pausable_init();
        
        if (_treasury == address(0)) revert ZeroAddress();
        
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(OPERATOR_ROLE, msg.sender);
        
        treasury = _treasury;
        
        depositFee = 10;      // 0.1%
        withdrawalFee = 10;   // 0.1%
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
    ) external nonReentrant whenNotPaused {
        if (!assetRegistry[asset].isEnabled) {
            revert AssetNotSupported(asset);
        }
        if (amount == 0) {
            revert InvalidAmount();
        }
        
        // Calculate fee
        uint256 fee = (amount * depositFee) / BASIS_POINTS;
        uint256 netAmount = amount - fee;
        
        // Transfer tokens from user
        IERC20(asset).safeTransferFrom(msg.sender, address(this), amount);
        
        // Send fee to treasury
        if (fee > 0) {
            IERC20(asset).safeTransfer(treasury, fee);
        }
        
        // Update user balance
        _addToUserBalance(msg.sender, asset, netAmount);
        
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
    ) external nonReentrant whenNotPaused {
        UserBalance storage userBalance = userBalances[msg.sender][asset];
        
        if (userBalance.balance < amount) {
            revert InsufficientBalance(userBalance.balance, amount);
        }
        
        // Calculate fee
        uint256 fee = (amount * withdrawalFee) / BASIS_POINTS;
        uint256 netAmount = amount - fee;
        
        // Update user balance
        userBalance.balance -= amount;
        userBalance.lastUpdated = block.timestamp;
        
        // Transfer tokens to user
        IERC20(asset).safeTransfer(msg.sender, netAmount);
        
        // Send fee to treasury
        if (fee > 0) {
            IERC20(asset).safeTransfer(treasury, fee);
        }
        
        emit Withdrawn(msg.sender, asset, netAmount);
    }
    
    /**
     * @notice Get user balance for a specific asset
     * @param user User address
     * @param asset Asset address
     */
    function getUserBalance(address user, address asset) external view returns (uint256) {
        return userBalances[user][asset].balance;
    }
    
    /**
     * @notice Get all user balances
     * @param user User address
     */
    function getUserPortfolio(address user) external view returns (
        address[] memory assets,
        uint256[] memory balances
    ) {
        assets = userAssets[user];
        balances = new uint256[](assets.length);
        
        for (uint256 i = 0; i < assets.length; i++) {
            balances[i] = userBalances[user][assets[i]].balance;
        }
        
        return (assets, balances);
    }
    
    /**
     * @notice Get percentage of volatile assets in portfolio
     * @param user User address
     */
    function getVolatileAssetPercentage(address user) external view returns (uint256) {
        address[] memory assets = userAssets[user];
        
        uint256 totalValue = 0;
        uint256 volatileValue = 0;
        
        for (uint256 i = 0; i < assets.length; i++) {
            address asset = assets[i];
            uint256 balance = userBalances[user][asset].balance;
            
            totalValue += balance;
            
            if (assetRegistry[asset].isVolatile) {
                volatileValue += balance;
            }
        }
        
        if (totalValue == 0) return 0;
        
        return (volatileValue * BASIS_POINTS) / totalValue;
    }
    
    /**
     * @notice Get percentage of cash/stables in portfolio
     * @param user User address
     */
    function getCashPercentage(address user) external view returns (uint256) {
        address[] memory assets = userAssets[user];
        
        uint256 totalValue = 0;
        uint256 cashValue = 0;
        
        for (uint256 i = 0; i < assets.length; i++) {
            address asset = assets[i];
            uint256 balance = userBalances[user][asset].balance;
            
            totalValue += balance;
            
            if (!assetRegistry[asset].isVolatile && !assetRegistry[asset].isYieldBearing) {
                cashValue += balance;
            }
        }
        
        if (totalValue == 0) return 0;
        
        return (cashValue * BASIS_POINTS) / totalValue;
    }
    
    // ============ Policy Engine Functions ============
    
    /**
     * @notice Execute a buy order for a user (swap from stable to target asset)
     * @dev Only callable by PolicyEngine
     */
    function executeBuy(
        address user,
        address fromAsset,
        address toAsset,
        uint256 percentage
    ) external onlyRole(POLICY_ENGINE_ROLE) nonReentrant {
        _executeSwap(user, fromAsset, toAsset, percentage);
    }
    
    /**
     * @notice Execute a sell order for a user (swap from target asset to stable)
     * @dev Only callable by PolicyEngine
     */
    function executeSell(
        address user,
        address fromAsset,
        address toAsset,
        uint256 percentage
    ) external onlyRole(POLICY_ENGINE_ROLE) nonReentrant {
        _executeSwap(user, fromAsset, toAsset, percentage);
    }
    
    // ============ Internal Functions ============
    
    function _addToUserBalance(
        address user,
        address asset,
        uint256 amount
    ) internal {
        UserBalance storage userBalance = userBalances[user][asset];
        
        // If first deposit of this asset, track it
        if (userBalance.balance == 0) {
            userAssets[user].push(asset);
        }
        
        userBalance.balance += amount;
        userBalance.lastUpdated = block.timestamp;
    }
    
    function _executeSwap(
        address user,
        address fromAsset,
        address toAsset,
        uint256 percentage
    ) internal {
        UserBalance storage fromBalance = userBalances[user][fromAsset];
        
        uint256 swapAmount = (fromBalance.balance * percentage) / BASIS_POINTS;
        if (swapAmount == 0) return;
        
        // For simplicity, 1:1 swap (in production, use DEX/oracle)
        fromBalance.balance -= swapAmount;
        
        _addToUserBalance(user, toAsset, swapAmount);
    }
    
    // ============ Admin Functions ============
    
    /**
     * @notice Add a supported asset
     */
    function addSupportedAsset(
        address token,
        string calldata symbol,
        bool isVolatile,
        bool isYieldBearing
    ) external onlyRole(DEFAULT_ADMIN_ROLE) {
        if (token == address(0)) revert ZeroAddress();
        
        assetRegistry[token] = AssetInfo({
            token: token,
            symbol: symbol,
            isVolatile: isVolatile,
            isYieldBearing: isYieldBearing,
            isEnabled: true
        });
        supportedAssets.push(token);
        
        emit AssetAdded(token, symbol, isVolatile);
    }
    
    /**
     * @notice Update fees
     */
    function setFees(uint256 _depositFee, uint256 _withdrawalFee) external onlyRole(DEFAULT_ADMIN_ROLE) {
        require(_depositFee <= 1000 && _withdrawalFee <= 1000, "Fee too high"); // Max 10%
        depositFee = _depositFee;
        withdrawalFee = _withdrawalFee;
        
        emit FeesUpdated(_depositFee, _withdrawalFee);
    }
    
    /**
     * @notice Update treasury address
     */
    function setTreasury(address _treasury) external onlyRole(DEFAULT_ADMIN_ROLE) {
        if (_treasury == address(0)) revert ZeroAddress();
        treasury = _treasury;
    }
    
    /**
     * @notice Pause the contract
     */
    function pause() external onlyRole(DEFAULT_ADMIN_ROLE) {
        _pause();
    }
    
    /**
     * @notice Unpause the contract
     */
    function unpause() external onlyRole(DEFAULT_ADMIN_ROLE) {
        _unpause();
    }
    
    /**
     * @notice Get list of supported assets
     */
    function getSupportedAssets() external view returns (address[] memory) {
        return supportedAssets;
    }
}
