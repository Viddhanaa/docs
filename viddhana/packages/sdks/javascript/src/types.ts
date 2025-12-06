/**
 * VIDDHANA SDK Type Definitions
 */

// ============================================================================
// Configuration Types
// ============================================================================

/**
 * Network configuration for VIDDHANA networks
 */
export interface NetworkConfig {
  chainId: number;
  rpcUrl: string;
  wsUrl: string;
  apiUrl: string;
}

/**
 * Client configuration options
 */
export interface ViddhanaConfig {
  /** Network to connect to: 'mainnet' or 'testnet' */
  network?: 'mainnet' | 'testnet';
  /** Custom RPC URL (overrides network default) */
  rpcUrl?: string;
  /** API key for authenticated requests */
  apiKey?: string;
  /** Private key for signing transactions */
  privateKey?: string;
}

// ============================================================================
// Chain Types
// ============================================================================

/**
 * Chain information
 */
export interface ChainInfo {
  chainId: number;
  name: string;
  symbol: string;
  blockNumber: number;
  gasPrice: string;
}

/**
 * Block information
 */
export interface Block {
  number: number;
  hash: string;
  parentHash: string;
  timestamp: number;
  nonce: string;
  difficulty: bigint;
  gasLimit: bigint;
  gasUsed: bigint;
  miner: string;
  transactions: string[];
}

/**
 * Transaction receipt
 */
export interface TransactionReceipt {
  transactionHash: string;
  blockNumber: number;
  blockHash: string;
  from: string;
  to: string;
  gasUsed: bigint;
  status: number;
  logs: Log[];
}

/**
 * Event log
 */
export interface Log {
  address: string;
  topics: string[];
  data: string;
  blockNumber: number;
  transactionHash: string;
  logIndex: number;
}

// ============================================================================
// Portfolio Types
// ============================================================================

/**
 * Asset information in a portfolio
 */
export interface Asset {
  symbol: string;
  address: string;
  balance: string;
  value: string;
  allocation: number;
}

/**
 * Portfolio metrics
 */
export interface PortfolioMetrics {
  volatility30d: number;
  sharpeRatio: number;
  unrealizedPnL: number;
  unrealizedPnLPercent: number;
}

/**
 * Complete portfolio data
 */
export interface PortfolioData {
  address: string;
  totalValue: string;
  currency: string;
  assets: Asset[];
  metrics: PortfolioMetrics;
  lastUpdated: number;
}

/**
 * Vault information
 */
export interface VaultInfo {
  address: string;
  name: string;
  symbol: string;
  totalAssets: string;
  totalShares: string;
  apy: number;
  strategy: string;
}

/**
 * Rebalancing action
 */
export interface RebalanceAction {
  asset: string;
  action: 'BUY' | 'SELL';
  percentage: number;
  amount: string;
  valueUSD: string;
}

/**
 * Single rebalance event
 */
export interface RebalanceEvent {
  txHash: string;
  blockNumber: number;
  timestamp: number;
  reason: string;
  actions: RebalanceAction[];
  aiConfidence: number;
  gasUsed: number;
  gasCost: string;
}

/**
 * Rebalancing history response
 */
export interface RebalanceHistory {
  address: string;
  history: RebalanceEvent[];
  totalRebalances: number;
}

// ============================================================================
// AI Types
// ============================================================================

/**
 * Single day prediction
 */
export interface DayPrediction {
  day: number;
  price: number;
  confidence: number;
}

/**
 * Price prediction response
 */
export interface PricePrediction {
  asset: string;
  currentPrice: number;
  horizon: number;
  predictions: DayPrediction[];
  trend: 'bullish' | 'bearish' | 'neutral';
  volatilityForecast: number;
  modelVersion: string;
  generatedAt: number;
}

/**
 * Portfolio optimization recommendation
 */
export interface OptimizationRecommendation {
  asset: string;
  action: 'BUY' | 'SELL' | 'HOLD';
  percentage: number;
  reason: string;
}

/**
 * Portfolio optimization response
 */
export interface PortfolioOptimization {
  action: string;
  recommendations: OptimizationRecommendation[];
  confidence: number;
  riskAssessment: RiskMetrics;
  expectedReturn: number;
  timestamp: number;
}

/**
 * Risk metrics
 */
export interface RiskMetrics {
  riskScore: number;
  volatility: number;
  sharpeRatio: number;
  maxDrawdown: number;
  beta: number;
  valueAtRisk: number;
}

/**
 * Risk assessment response
 */
export interface RiskAssessment {
  portfolio: Record<string, number>;
  metrics: RiskMetrics;
  recommendations: string[];
  riskLevel: 'low' | 'medium' | 'high' | 'very_high';
  timestamp: number;
}

// ============================================================================
// Governance Types
// ============================================================================

/**
 * Proposal state
 */
export type ProposalState =
  | 'Pending'
  | 'Active'
  | 'Canceled'
  | 'Defeated'
  | 'Succeeded'
  | 'Queued'
  | 'Expired'
  | 'Executed';

/**
 * Proposal information
 */
export interface Proposal {
  id: string;
  proposer: string;
  targets: string[];
  values: bigint[];
  calldatas: string[];
  description: string;
  state: ProposalState;
  startBlock: number;
  endBlock: number;
  forVotes: bigint;
  againstVotes: bigint;
  abstainVotes: bigint;
}

/**
 * Vote receipt
 */
export interface VoteReceipt {
  hasVoted: boolean;
  support: number; // 0 = Against, 1 = For, 2 = Abstain
  votes: bigint;
}

/**
 * Delegation info
 */
export interface DelegationInfo {
  delegator: string;
  delegatee: string;
  votingPower: bigint;
  delegatedAt: number;
}

// ============================================================================
// DePIN Types
// ============================================================================

/**
 * Sensor information
 */
export interface SensorInfo {
  sensorId: string;
  owner: string;
  deviceType: string;
  status: 'active' | 'inactive' | 'maintenance';
  location?: {
    latitude: number;
    longitude: number;
  };
}

/**
 * Sensor rewards
 */
export interface SensorRewards {
  sensorId: string;
  owner: string;
  deviceType: string;
  status: string;
  rewards: {
    pending: string;
    claimed: string;
    lifetime: string;
    currency: string;
  };
  performance: {
    uptimePercent: number;
    dataQualityScore: number;
    dailyRate: string;
    errorCount: number;
  };
  lastDataPoint: number;
}

// ============================================================================
// Staking Types
// ============================================================================

/**
 * Staking position
 */
export interface StakingPosition {
  staker: string;
  amount: bigint;
  lockedUntil: number;
  rewards: bigint;
  tier: string;
}

/**
 * Staking statistics
 */
export interface StakingStats {
  totalStaked: string;
  stakersCount: number;
  currentAPY: number;
  rewardsDistributed: string;
}

// ============================================================================
// Tokenomics Types
// ============================================================================

/**
 * Token price information
 */
export interface TokenPrice {
  usd: number;
  change24h: number;
  change7d: number;
}

/**
 * Buyback information
 */
export interface BuybackInfo {
  date: number;
  amount: string;
  burned: string;
}

/**
 * Tokenomics statistics
 */
export interface TokenomicsStats {
  totalSupply: string;
  circulatingSupply: string;
  burnedSupply: string;
  stakedSupply: string;
  price: TokenPrice;
  marketCap: string;
  fdv: string;
  stakingAPY: number;
  nextBuybackDate: number;
  lastBuyback: BuybackInfo;
}

// ============================================================================
// Event Types
// ============================================================================

/**
 * WebSocket subscription options
 */
export interface SubscriptionOptions {
  channel: 'portfolio' | 'predictions' | 'rebalance' | 'rewards' | 'blocks';
  address?: string;
  asset?: string;
}

/**
 * WebSocket message
 */
export interface WebSocketMessage<T = unknown> {
  channel: string;
  payload: T;
  timestamp: number;
}
