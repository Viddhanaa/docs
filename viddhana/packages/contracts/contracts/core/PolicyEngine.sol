// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts-upgradeable/proxy/utils/Initializable.sol";
import "@openzeppelin/contracts-upgradeable/access/AccessControlUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/utils/ReentrancyGuardUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/utils/PausableUpgradeable.sol";

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
    ReentrancyGuardUpgradeable
{
    // ============ Constants ============
    bytes32 public constant AI_ROLE = keccak256("AI_ROLE");
    bytes32 public constant OPERATOR_ROLE = keccak256("OPERATOR_ROLE");
    
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
        bool isActive;
    }
    
    struct RebalanceAction {
        address fromAsset;
        address toAsset;
        uint256 percentage;         // Percentage of position
        string reason;
    }
    
    // ============ State Variables ============
    address public vaultManager;
    
    mapping(address => UserProfile) public userProfiles;
    mapping(address => uint256) public lastRebalanceTime;
    
    uint256 public inflationThreshold;      // Trigger inflation protection above this
    uint256 public minConfidenceThreshold;  // Minimum AI confidence to execute
    uint256 public totalUsers;
    
    // ============ Events ============
    event ProfileCreated(address indexed user, uint256 riskTolerance, uint256 timeToGoal);
    event ProfileUpdated(address indexed user, uint256 riskTolerance, uint256 timeToGoal);
    event RebalanceExecuted(address indexed user, string reason, uint256 timestamp);
    event AIRecommendationReceived(address indexed user, uint256 confidence);
    event InflationProtectionTriggered(address indexed user, uint256 inflationRate);
    event VaultManagerUpdated(address indexed newVaultManager);
    
    // ============ Errors ============
    error RebalanceTooSoon(uint256 nextAllowedTime);
    error InsufficientConfidence(uint256 provided, uint256 required);
    error InvalidRiskTolerance(uint256 provided);
    error UserNotRegistered(address user);
    error AutoRebalanceDisabled(address user);
    error ZeroAddress();
    
    /// @custom:oz-upgrades-unsafe-allow constructor
    constructor() {
        _disableInitializers();
    }
    
    // ============ Initializer ============
    function initialize(address _vaultManager) public initializer {
        __AccessControl_init();
        __Pausable_init();
        __ReentrancyGuard_init();
        
        if (_vaultManager == address(0)) revert ZeroAddress();
        
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(OPERATOR_ROLE, msg.sender);
        
        vaultManager = _vaultManager;
        
        inflationThreshold = 600;       // 6%
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
        
        bool isNew = !userProfiles[msg.sender].isActive;
        
        // Calculate max volatile share based on risk tolerance and time to goal
        uint256 maxVolatile = _calculateMaxVolatileShare(riskTolerance, timeToGoal);
        
        userProfiles[msg.sender] = UserProfile({
            riskTolerance: riskTolerance,
            timeToGoal: timeToGoal,
            maxVolatileShare: maxVolatile,
            lastRebalance: block.timestamp,
            autoRebalanceEnabled: autoRebalance,
            isActive: true
        });
        
        if (isNew) {
            totalUsers++;
            emit ProfileCreated(msg.sender, riskTolerance, timeToGoal);
        } else {
            emit ProfileUpdated(msg.sender, riskTolerance, timeToGoal);
        }
    }
    
    /**
     * @notice Check if user can be rebalanced
     * @param user User address
     */
    function canRebalance(address user) external view returns (bool) {
        UserProfile memory profile = userProfiles[user];
        
        if (!profile.isActive) return false;
        if (!profile.autoRebalanceEnabled) return false;
        if (block.timestamp < profile.lastRebalance + MIN_REBALANCE_INTERVAL) return false;
        
        return true;
    }
    
    /**
     * @notice Get user profile
     * @param user User address
     */
    function getUserProfile(address user) external view returns (
        uint256 riskTolerance,
        uint256 timeToGoal,
        uint256 maxVolatileShare,
        uint256 lastRebalance,
        bool autoRebalanceEnabled,
        bool isActive
    ) {
        UserProfile memory profile = userProfiles[user];
        return (
            profile.riskTolerance,
            profile.timeToGoal,
            profile.maxVolatileShare,
            profile.lastRebalance,
            profile.autoRebalanceEnabled,
            profile.isActive
        );
    }
    
    // ============ AI Functions ============
    
    /**
     * @notice Submit AI recommendation and execute rebalance for a user
     * @dev Only callable by addresses with AI_ROLE
     * @param user User address
     * @param actions Array of rebalancing actions
     * @param confidence Confidence level (0-10000)
     */
    function executeAIRecommendation(
        address user,
        RebalanceAction[] calldata actions,
        uint256 confidence
    ) external onlyRole(AI_ROLE) whenNotPaused nonReentrant {
        UserProfile storage profile = userProfiles[user];
        
        if (!profile.isActive) {
            revert UserNotRegistered(user);
        }
        
        if (!profile.autoRebalanceEnabled) {
            revert AutoRebalanceDisabled(user);
        }
        
        if (block.timestamp < profile.lastRebalance + MIN_REBALANCE_INTERVAL) {
            revert RebalanceTooSoon(profile.lastRebalance + MIN_REBALANCE_INTERVAL);
        }
        
        if (confidence < minConfidenceThreshold) {
            revert InsufficientConfidence(confidence, minConfidenceThreshold);
        }
        
        emit AIRecommendationReceived(user, confidence);
        
        // Execute through VaultManager
        _executeRebalance(user, actions, "AI Auto-Rebalance");
        
        // Update last rebalance time
        profile.lastRebalance = block.timestamp;
    }
    
    /**
     * @notice Trigger inflation protection rebalance
     * @dev Called by AI when inflation threshold is exceeded
     * @param user User address
     * @param inflationRate Current inflation rate (basis points)
     * @param actions Rebalancing actions to execute
     */
    function triggerInflationProtection(
        address user,
        uint256 inflationRate,
        RebalanceAction[] calldata actions
    ) external onlyRole(AI_ROLE) whenNotPaused nonReentrant {
        require(inflationRate > inflationThreshold, "Below inflation threshold");
        
        UserProfile storage profile = userProfiles[user];
        
        if (!profile.isActive) {
            revert UserNotRegistered(user);
        }
        
        emit InflationProtectionTriggered(user, inflationRate);
        
        _executeRebalance(user, actions, "Inflation Protection");
        
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
    
    function _executeRebalance(
        address user,
        RebalanceAction[] calldata actions,
        string memory reason
    ) internal {
        IVaultManager vault = IVaultManager(vaultManager);
        
        for (uint256 i = 0; i < actions.length; i++) {
            vault.executeBuy(
                user,
                actions[i].fromAsset,
                actions[i].toAsset,
                actions[i].percentage
            );
        }
        
        emit RebalanceExecuted(user, reason, block.timestamp);
    }
    
    // ============ Admin Functions ============
    
    /**
     * @notice Set inflation threshold
     */
    function setInflationThreshold(uint256 _threshold) external onlyRole(DEFAULT_ADMIN_ROLE) {
        inflationThreshold = _threshold;
    }
    
    /**
     * @notice Set minimum confidence threshold for AI recommendations
     */
    function setMinConfidenceThreshold(uint256 _threshold) external onlyRole(DEFAULT_ADMIN_ROLE) {
        require(_threshold <= BASIS_POINTS, "Invalid threshold");
        minConfidenceThreshold = _threshold;
    }
    
    /**
     * @notice Update vault manager address
     */
    function setVaultManager(address _vaultManager) external onlyRole(DEFAULT_ADMIN_ROLE) {
        if (_vaultManager == address(0)) revert ZeroAddress();
        vaultManager = _vaultManager;
        emit VaultManagerUpdated(_vaultManager);
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
}

// ============ Interfaces ============
interface IVaultManager {
    function executeBuy(
        address user,
        address fromAsset,
        address toAsset,
        uint256 percentage
    ) external;
    
    function executeSell(
        address user,
        address fromAsset,
        address toAsset,
        uint256 percentage
    ) external;
    
    function getVolatileAssetPercentage(address user) external view returns (uint256);
    function getCashPercentage(address user) external view returns (uint256);
}
