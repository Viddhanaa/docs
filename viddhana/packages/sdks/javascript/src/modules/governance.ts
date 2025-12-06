/**
 * Governance Module - Proposals, voting, and delegation
 */

import { ethers } from 'ethers';
import type {
  Proposal,
  ProposalState,
  VoteReceipt,
  DelegationInfo,
} from '../types';

// Contract ABIs
const GOVERNOR_ABI = [
  'function propose(address[] targets, uint256[] values, bytes[] calldatas, string description) returns (uint256)',
  'function castVote(uint256 proposalId, uint8 support) returns (uint256)',
  'function castVoteWithReason(uint256 proposalId, uint8 support, string reason) returns (uint256)',
  'function queue(uint256 proposalId) returns (uint256)',
  'function execute(uint256 proposalId) returns (uint256)',
  'function cancel(uint256 proposalId) returns (uint256)',
  'function state(uint256 proposalId) view returns (uint8)',
  'function proposalVotes(uint256 proposalId) view returns (uint256 againstVotes, uint256 forVotes, uint256 abstainVotes)',
  'function proposalSnapshot(uint256 proposalId) view returns (uint256)',
  'function proposalDeadline(uint256 proposalId) view returns (uint256)',
  'function getVotes(address account, uint256 blockNumber) view returns (uint256)',
  'function hasVoted(uint256 proposalId, address account) view returns (bool)',
  'function proposals(uint256 proposalId) view returns (tuple(address proposer, uint256 eta, uint256 startBlock, uint256 endBlock, uint256 forVotes, uint256 againstVotes, uint256 abstainVotes, bool canceled, bool executed))',
];

const VDH_TOKEN_ABI = [
  'function delegate(address delegatee) returns ()',
  'function delegates(address account) view returns (address)',
  'function getVotes(address account) view returns (uint256)',
  'function getPastVotes(address account, uint256 blockNumber) view returns (uint256)',
  'function getPastTotalSupply(uint256 blockNumber) view returns (uint256)',
];

// Contract addresses
const CONTRACT_ADDRESSES = {
  mainnet: {
    governor: '0x0000000000000000000000000000000000000003',
    vdhToken: '0x0000000000000000000000000000000000000004',
  },
  testnet: {
    governor: '0x0000000000000000000000000000000000000003',
    vdhToken: '0x0000000000000000000000000000000000000004',
  },
};

const PROPOSAL_STATES: ProposalState[] = [
  'Pending',
  'Active',
  'Canceled',
  'Defeated',
  'Succeeded',
  'Queued',
  'Expired',
  'Executed',
];

/**
 * Governance module for proposals, voting, and delegation
 *
 * @example
 * ```typescript
 * // Create a proposal
 * const tx = await client.governance.createProposal(
 *   ['0xTargetContract...'],
 *   [0n],
 *   ['0x...'],
 *   'Proposal to upgrade the protocol'
 * );
 *
 * // Cast a vote
 * await client.governance.castVote(proposalId, 1); // 1 = For
 *
 * // Delegate voting power
 * await client.governance.delegate('0xDelegate...');
 * ```
 */
export class Governance {
  private readonly provider: ethers.JsonRpcProvider;
  private readonly signer: ethers.Wallet | null;

  constructor(
    provider: ethers.JsonRpcProvider,
    signer: ethers.Wallet | null
  ) {
    this.provider = provider;
    this.signer = signer;
  }

  /**
   * Creates a new governance proposal
   *
   * @param targets - Array of target contract addresses
   * @param values - Array of ETH values to send
   * @param calldatas - Array of encoded function calls
   * @param description - Proposal description
   * @returns Transaction response
   */
  async createProposal(
    targets: string[],
    values: bigint[],
    calldatas: string[],
    description: string
  ): Promise<ethers.TransactionResponse> {
    this.requireSigner();

    const governor = new ethers.Contract(
      CONTRACT_ADDRESSES.mainnet.governor,
      GOVERNOR_ABI,
      this.signer!
    );

    const tx = await governor.propose(targets, values, calldatas, description);
    return tx as ethers.TransactionResponse;
  }

  /**
   * Casts a vote on a proposal
   *
   * @param proposalId - The proposal ID
   * @param support - Vote type: 0 = Against, 1 = For, 2 = Abstain
   * @returns Transaction response
   */
  async castVote(
    proposalId: string | bigint,
    support: 0 | 1 | 2
  ): Promise<ethers.TransactionResponse> {
    this.requireSigner();

    const governor = new ethers.Contract(
      CONTRACT_ADDRESSES.mainnet.governor,
      GOVERNOR_ABI,
      this.signer!
    );

    const tx = await governor.castVote(proposalId, support);
    return tx as ethers.TransactionResponse;
  }

  /**
   * Casts a vote with a reason
   *
   * @param proposalId - The proposal ID
   * @param support - Vote type: 0 = Against, 1 = For, 2 = Abstain
   * @param reason - Reason for the vote
   * @returns Transaction response
   */
  async castVoteWithReason(
    proposalId: string | bigint,
    support: 0 | 1 | 2,
    reason: string
  ): Promise<ethers.TransactionResponse> {
    this.requireSigner();

    const governor = new ethers.Contract(
      CONTRACT_ADDRESSES.mainnet.governor,
      GOVERNOR_ABI,
      this.signer!
    );

    const tx = await governor.castVoteWithReason(proposalId, support, reason);
    return tx as ethers.TransactionResponse;
  }

  /**
   * Delegates voting power to another address
   *
   * @param delegatee - Address to delegate to
   * @returns Transaction response
   */
  async delegate(delegatee: string): Promise<ethers.TransactionResponse> {
    this.requireSigner();

    const vdhToken = new ethers.Contract(
      CONTRACT_ADDRESSES.mainnet.vdhToken,
      VDH_TOKEN_ABI,
      this.signer!
    );

    const tx = await vdhToken.delegate(delegatee);
    return tx as ethers.TransactionResponse;
  }

  /**
   * Gets proposal details
   *
   * @param proposalId - The proposal ID
   * @returns Proposal information
   */
  async getProposal(proposalId: string | bigint): Promise<Proposal> {
    const governor = new ethers.Contract(
      CONTRACT_ADDRESSES.mainnet.governor,
      GOVERNOR_ABI,
      this.provider
    );

    const [proposalData, state, votes] = await Promise.all([
      governor.proposals(proposalId),
      governor.state(proposalId),
      governor.proposalVotes(proposalId),
    ]);

    return {
      id: proposalId.toString(),
      proposer: proposalData.proposer as string,
      targets: [], // Would need separate storage/events to retrieve
      values: [],
      calldatas: [],
      description: '', // Would need to retrieve from events
      state: PROPOSAL_STATES[state as number] ?? 'Pending',
      startBlock: Number(proposalData.startBlock),
      endBlock: Number(proposalData.endBlock),
      forVotes: votes.forVotes as bigint,
      againstVotes: votes.againstVotes as bigint,
      abstainVotes: votes.abstainVotes as bigint,
    };
  }

  /**
   * Gets the state of a proposal
   *
   * @param proposalId - The proposal ID
   * @returns Proposal state
   */
  async getProposalState(proposalId: string | bigint): Promise<ProposalState> {
    const governor = new ethers.Contract(
      CONTRACT_ADDRESSES.mainnet.governor,
      GOVERNOR_ABI,
      this.provider
    );

    const state = await governor.state(proposalId);
    return PROPOSAL_STATES[state as number] ?? 'Pending';
  }

  /**
   * Gets vote receipt for an account on a proposal
   *
   * @param proposalId - The proposal ID
   * @param account - The voter's address
   * @returns Vote receipt
   */
  async getVoteReceipt(
    proposalId: string | bigint,
    account: string
  ): Promise<VoteReceipt> {
    const governor = new ethers.Contract(
      CONTRACT_ADDRESSES.mainnet.governor,
      GOVERNOR_ABI,
      this.provider
    );

    const hasVoted = await governor.hasVoted(proposalId, account);
    const snapshot = await governor.proposalSnapshot(proposalId);
    const votes = await governor.getVotes(account, snapshot);

    return {
      hasVoted: hasVoted as boolean,
      support: 0, // Would need to retrieve from events
      votes: votes as bigint,
    };
  }

  /**
   * Gets delegation info for an account
   *
   * @param account - The account address
   * @returns Delegation information
   */
  async getDelegation(account: string): Promise<DelegationInfo> {
    const vdhToken = new ethers.Contract(
      CONTRACT_ADDRESSES.mainnet.vdhToken,
      VDH_TOKEN_ABI,
      this.provider
    );

    const [delegatee, votingPower] = await Promise.all([
      vdhToken.delegates(account),
      vdhToken.getVotes(account),
    ]);

    return {
      delegator: account,
      delegatee: delegatee as string,
      votingPower: votingPower as bigint,
      delegatedAt: 0, // Would need to retrieve from events
    };
  }

  /**
   * Gets current voting power for an account
   *
   * @param account - The account address
   * @returns Current voting power
   */
  async getVotingPower(account: string): Promise<bigint> {
    const vdhToken = new ethers.Contract(
      CONTRACT_ADDRESSES.mainnet.vdhToken,
      VDH_TOKEN_ABI,
      this.provider
    );

    return vdhToken.getVotes(account) as Promise<bigint>;
  }

  /**
   * Gets past voting power at a specific block
   *
   * @param account - The account address
   * @param blockNumber - Block number to check
   * @returns Voting power at that block
   */
  async getPastVotingPower(
    account: string,
    blockNumber: number
  ): Promise<bigint> {
    const vdhToken = new ethers.Contract(
      CONTRACT_ADDRESSES.mainnet.vdhToken,
      VDH_TOKEN_ABI,
      this.provider
    );

    return vdhToken.getPastVotes(account, blockNumber) as Promise<bigint>;
  }

  /**
   * Queues a successful proposal for execution
   *
   * @param proposalId - The proposal ID
   * @returns Transaction response
   */
  async queueProposal(
    proposalId: string | bigint
  ): Promise<ethers.TransactionResponse> {
    this.requireSigner();

    const governor = new ethers.Contract(
      CONTRACT_ADDRESSES.mainnet.governor,
      GOVERNOR_ABI,
      this.signer!
    );

    const tx = await governor.queue(proposalId);
    return tx as ethers.TransactionResponse;
  }

  /**
   * Executes a queued proposal
   *
   * @param proposalId - The proposal ID
   * @returns Transaction response
   */
  async executeProposal(
    proposalId: string | bigint
  ): Promise<ethers.TransactionResponse> {
    this.requireSigner();

    const governor = new ethers.Contract(
      CONTRACT_ADDRESSES.mainnet.governor,
      GOVERNOR_ABI,
      this.signer!
    );

    const tx = await governor.execute(proposalId);
    return tx as ethers.TransactionResponse;
  }

  /**
   * Cancels a proposal
   *
   * @param proposalId - The proposal ID
   * @returns Transaction response
   */
  async cancelProposal(
    proposalId: string | bigint
  ): Promise<ethers.TransactionResponse> {
    this.requireSigner();

    const governor = new ethers.Contract(
      CONTRACT_ADDRESSES.mainnet.governor,
      GOVERNOR_ABI,
      this.signer!
    );

    const tx = await governor.cancel(proposalId);
    return tx as ethers.TransactionResponse;
  }

  /**
   * Gets proposal voting period
   *
   * @param proposalId - The proposal ID
   * @returns Start and end blocks
   */
  async getVotingPeriod(
    proposalId: string | bigint
  ): Promise<{ startBlock: number; endBlock: number }> {
    const governor = new ethers.Contract(
      CONTRACT_ADDRESSES.mainnet.governor,
      GOVERNOR_ABI,
      this.provider
    );

    const [startBlock, endBlock] = await Promise.all([
      governor.proposalSnapshot(proposalId),
      governor.proposalDeadline(proposalId),
    ]);

    return {
      startBlock: Number(startBlock),
      endBlock: Number(endBlock),
    };
  }

  /**
   * Checks if an account has voted on a proposal
   *
   * @param proposalId - The proposal ID
   * @param account - The account address
   * @returns True if the account has voted
   */
  async hasVoted(proposalId: string | bigint, account: string): Promise<boolean> {
    const governor = new ethers.Contract(
      CONTRACT_ADDRESSES.mainnet.governor,
      GOVERNOR_ABI,
      this.provider
    );

    return governor.hasVoted(proposalId, account) as Promise<boolean>;
  }

  /**
   * Requires signer to be present
   * @private
   */
  private requireSigner(): void {
    if (!this.signer) {
      throw new Error('Signer required for this operation. Provide privateKey in config.');
    }
  }
}
