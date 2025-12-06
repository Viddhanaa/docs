// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts-upgradeable/proxy/utils/Initializable.sol";
import "@openzeppelin/contracts-upgradeable/access/AccessControlUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/utils/ReentrancyGuardUpgradeable.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";

/**
 * @title VDHGovernance
 * @author VIDDHANA Team
 * @notice Governance contract for VDH token holders
 * @dev Allows proposal creation, voting, and execution
 */
contract VDHGovernance is 
    Initializable,
    AccessControlUpgradeable,
    ReentrancyGuardUpgradeable
{
    // ============ Constants ============
    bytes32 public constant EXECUTOR_ROLE = keccak256("EXECUTOR_ROLE");
    
    uint256 public constant BASIS_POINTS = 10000;
    uint256 public constant MIN_VOTING_PERIOD = 1 days;
    uint256 public constant MAX_VOTING_PERIOD = 14 days;
    
    // ============ Enums ============
    enum ProposalState {
        Pending,
        Active,
        Canceled,
        Defeated,
        Succeeded,
        Queued,
        Executed,
        Expired
    }
    
    // ============ Structs ============
    struct Proposal {
        uint256 id;
        address proposer;
        string title;
        string description;
        uint256 forVotes;
        uint256 againstVotes;
        uint256 abstainVotes;
        uint256 startTime;
        uint256 endTime;
        bool executed;
        bool canceled;
        mapping(address => bool) hasVoted;
        mapping(address => uint256) voterVotes;
    }
    
    struct ProposalInfo {
        uint256 id;
        address proposer;
        string title;
        string description;
        uint256 forVotes;
        uint256 againstVotes;
        uint256 abstainVotes;
        uint256 startTime;
        uint256 endTime;
        bool executed;
        bool canceled;
    }
    
    // ============ State Variables ============
    IERC20 public vdhToken;
    
    mapping(uint256 => Proposal) public proposals;
    uint256 public proposalCount;
    
    uint256 public proposalThreshold;    // Min tokens to create proposal
    uint256 public quorumThreshold;      // Min votes for quorum (basis points of total supply)
    uint256 public votingPeriod;         // Duration of voting
    uint256 public votingDelay;          // Delay before voting starts
    
    // ============ Events ============
    event ProposalCreated(
        uint256 indexed proposalId,
        address indexed proposer,
        string title,
        uint256 startTime,
        uint256 endTime
    );
    event VoteCast(
        address indexed voter,
        uint256 indexed proposalId,
        uint8 support,
        uint256 weight
    );
    event ProposalExecuted(uint256 indexed proposalId);
    event ProposalCanceled(uint256 indexed proposalId);
    event QuorumUpdated(uint256 oldQuorum, uint256 newQuorum);
    event VotingPeriodUpdated(uint256 oldPeriod, uint256 newPeriod);
    
    // ============ Errors ============
    error InsufficientBalance(uint256 required, uint256 available);
    error InvalidProposal(uint256 proposalId);
    error ProposalNotActive(uint256 proposalId);
    error AlreadyVoted(address voter, uint256 proposalId);
    error ProposalNotSucceeded(uint256 proposalId);
    error ProposalAlreadyExecuted(uint256 proposalId);
    error InvalidVotingPeriod();
    error ZeroAddress();
    
    /// @custom:oz-upgrades-unsafe-allow constructor
    constructor() {
        _disableInitializers();
    }
    
    // ============ Initializer ============
    function initialize(address _vdhToken) public initializer {
        __AccessControl_init();
        __ReentrancyGuard_init();
        
        if (_vdhToken == address(0)) revert ZeroAddress();
        
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(EXECUTOR_ROLE, msg.sender);
        
        vdhToken = IERC20(_vdhToken);
        
        proposalThreshold = 100_000 * 10**18;  // 100k VDH to create proposal
        quorumThreshold = 400;                  // 4% of total supply
        votingPeriod = 7 days;
        votingDelay = 1 days;
    }
    
    // ============ Proposal Functions ============
    
    /**
     * @notice Create a new proposal
     * @param title Proposal title
     * @param description Proposal description
     */
    function propose(
        string calldata title,
        string calldata description
    ) external returns (uint256) {
        uint256 proposerBalance = vdhToken.balanceOf(msg.sender);
        
        if (proposerBalance < proposalThreshold) {
            revert InsufficientBalance(proposalThreshold, proposerBalance);
        }
        
        proposalCount++;
        uint256 proposalId = proposalCount;
        
        Proposal storage proposal = proposals[proposalId];
        proposal.id = proposalId;
        proposal.proposer = msg.sender;
        proposal.title = title;
        proposal.description = description;
        proposal.startTime = block.timestamp + votingDelay;
        proposal.endTime = proposal.startTime + votingPeriod;
        
        emit ProposalCreated(
            proposalId,
            msg.sender,
            title,
            proposal.startTime,
            proposal.endTime
        );
        
        return proposalId;
    }
    
    /**
     * @notice Cast a vote on a proposal
     * @param proposalId Proposal ID
     * @param support 0 = Against, 1 = For, 2 = Abstain
     */
    function castVote(uint256 proposalId, uint8 support) external nonReentrant {
        Proposal storage proposal = proposals[proposalId];
        
        if (proposal.id == 0) {
            revert InvalidProposal(proposalId);
        }
        
        if (getProposalState(proposalId) != ProposalState.Active) {
            revert ProposalNotActive(proposalId);
        }
        
        if (proposal.hasVoted[msg.sender]) {
            revert AlreadyVoted(msg.sender, proposalId);
        }
        
        uint256 weight = vdhToken.balanceOf(msg.sender);
        
        proposal.hasVoted[msg.sender] = true;
        proposal.voterVotes[msg.sender] = weight;
        
        if (support == 0) {
            proposal.againstVotes += weight;
        } else if (support == 1) {
            proposal.forVotes += weight;
        } else {
            proposal.abstainVotes += weight;
        }
        
        emit VoteCast(msg.sender, proposalId, support, weight);
    }
    
    /**
     * @notice Execute a successful proposal
     * @param proposalId Proposal ID
     */
    function execute(uint256 proposalId) external onlyRole(EXECUTOR_ROLE) nonReentrant {
        if (getProposalState(proposalId) != ProposalState.Succeeded) {
            revert ProposalNotSucceeded(proposalId);
        }
        
        Proposal storage proposal = proposals[proposalId];
        
        if (proposal.executed) {
            revert ProposalAlreadyExecuted(proposalId);
        }
        
        proposal.executed = true;
        
        emit ProposalExecuted(proposalId);
    }
    
    /**
     * @notice Cancel a proposal (only proposer or admin)
     * @param proposalId Proposal ID
     */
    function cancel(uint256 proposalId) external {
        Proposal storage proposal = proposals[proposalId];
        
        require(
            msg.sender == proposal.proposer || hasRole(DEFAULT_ADMIN_ROLE, msg.sender),
            "Not authorized"
        );
        
        proposal.canceled = true;
        
        emit ProposalCanceled(proposalId);
    }
    
    // ============ View Functions ============
    
    /**
     * @notice Get proposal state
     * @param proposalId Proposal ID
     */
    function getProposalState(uint256 proposalId) public view returns (ProposalState) {
        Proposal storage proposal = proposals[proposalId];
        
        if (proposal.id == 0) {
            return ProposalState.Pending;
        }
        
        if (proposal.canceled) {
            return ProposalState.Canceled;
        }
        
        if (proposal.executed) {
            return ProposalState.Executed;
        }
        
        if (block.timestamp < proposal.startTime) {
            return ProposalState.Pending;
        }
        
        if (block.timestamp <= proposal.endTime) {
            return ProposalState.Active;
        }
        
        // Check if quorum is met
        uint256 totalVotes = proposal.forVotes + proposal.againstVotes + proposal.abstainVotes;
        uint256 quorumRequired = (vdhToken.totalSupply() * quorumThreshold) / BASIS_POINTS;
        
        if (totalVotes < quorumRequired) {
            return ProposalState.Defeated;
        }
        
        // Check if proposal passed
        if (proposal.forVotes > proposal.againstVotes) {
            return ProposalState.Succeeded;
        }
        
        return ProposalState.Defeated;
    }
    
    /**
     * @notice Get proposal info
     * @param proposalId Proposal ID
     */
    function getProposal(uint256 proposalId) external view returns (ProposalInfo memory) {
        Proposal storage proposal = proposals[proposalId];
        
        return ProposalInfo({
            id: proposal.id,
            proposer: proposal.proposer,
            title: proposal.title,
            description: proposal.description,
            forVotes: proposal.forVotes,
            againstVotes: proposal.againstVotes,
            abstainVotes: proposal.abstainVotes,
            startTime: proposal.startTime,
            endTime: proposal.endTime,
            executed: proposal.executed,
            canceled: proposal.canceled
        });
    }
    
    /**
     * @notice Check if an address has voted on a proposal
     * @param proposalId Proposal ID
     * @param voter Voter address
     */
    function hasVoted(uint256 proposalId, address voter) external view returns (bool) {
        return proposals[proposalId].hasVoted[voter];
    }
    
    /**
     * @notice Get vote count for a voter on a proposal
     * @param proposalId Proposal ID
     * @param voter Voter address
     */
    function getVotes(uint256 proposalId, address voter) external view returns (uint256) {
        return proposals[proposalId].voterVotes[voter];
    }
    
    // ============ Admin Functions ============
    
    /**
     * @notice Set proposal threshold
     */
    function setProposalThreshold(uint256 _threshold) external onlyRole(DEFAULT_ADMIN_ROLE) {
        proposalThreshold = _threshold;
    }
    
    /**
     * @notice Set quorum threshold
     */
    function setQuorumThreshold(uint256 _threshold) external onlyRole(DEFAULT_ADMIN_ROLE) {
        require(_threshold <= BASIS_POINTS, "Invalid threshold");
        uint256 oldQuorum = quorumThreshold;
        quorumThreshold = _threshold;
        emit QuorumUpdated(oldQuorum, _threshold);
    }
    
    /**
     * @notice Set voting period
     */
    function setVotingPeriod(uint256 _period) external onlyRole(DEFAULT_ADMIN_ROLE) {
        if (_period < MIN_VOTING_PERIOD || _period > MAX_VOTING_PERIOD) {
            revert InvalidVotingPeriod();
        }
        uint256 oldPeriod = votingPeriod;
        votingPeriod = _period;
        emit VotingPeriodUpdated(oldPeriod, _period);
    }
    
    /**
     * @notice Set voting delay
     */
    function setVotingDelay(uint256 _delay) external onlyRole(DEFAULT_ADMIN_ROLE) {
        votingDelay = _delay;
    }
}
