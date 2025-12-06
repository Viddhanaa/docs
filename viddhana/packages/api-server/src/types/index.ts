/**
 * VIDDHANA API Server Type Definitions
 */

import { Request } from 'express';

// ============================================================================
// JSON-RPC Types
// ============================================================================

export interface JsonRpcRequest {
  jsonrpc: '2.0';
  method: string;
  params?: unknown[];
  id: number | string;
}

export interface JsonRpcResponse<T = unknown> {
  jsonrpc: '2.0';
  id: number | string;
  result?: T;
  error?: JsonRpcError;
}

export interface JsonRpcError {
  code: number;
  message: string;
  data?: unknown;
}

// Standard JSON-RPC error codes
export enum JsonRpcErrorCode {
  PARSE_ERROR = -32700,
  INVALID_REQUEST = -32600,
  METHOD_NOT_FOUND = -32601,
  INVALID_PARAMS = -32602,
  INTERNAL_ERROR = -32603,
  // Custom error codes
  UNAUTHORIZED = -32000,
  RATE_LIMITED = -32001,
  RESOURCE_NOT_FOUND = -32002,
  INVALID_ADDRESS = -32003,
  BLOCKCHAIN_ERROR = -32004,
}

// ============================================================================
// Chain Types
// ============================================================================

export interface ChainInfo {
  chainId: number;
  name: string;
  symbol: string;
  blockNumber: number;
  gasPrice: string;
  blockTime: number;
  version: string;
}

export interface BlockInfo {
  number: number;
  hash: string | null;
  parentHash: string;
  timestamp: number;
  gasLimit: string;
  gasUsed: string;
  transactions: string[];
  transactionCount: number;
}

export interface BalanceInfo {
  address: string;
  balance: string;
  balanceFormatted: string;
  symbol: string;
  decimals: number;
}

// ============================================================================
// Vault Types
// ============================================================================

export interface VaultInfo {
  id: string;
  owner: string;
  name: string;
  totalValue: string;
  currency: string;
  assets: VaultAsset[];
  createdAt: number;
  lastUpdated: number;
}

export interface VaultAsset {
  symbol: string;
  address: string;
  balance: string;
  value: string;
  allocation: number;
}

export interface VaultCreateParams {
  name: string;
  riskTolerance: number;
  timeToGoal: number;
  autoRebalance: boolean;
}

export interface VaultDepositParams {
  vaultId: string;
  asset: string;
  amount: string;
}

export interface VaultWithdrawParams {
  vaultId: string;
  asset: string;
  amount: string;
}

export interface VaultTransaction {
  hash: string;
  type: 'deposit' | 'withdraw' | 'rebalance';
  status: 'pending' | 'confirmed' | 'failed';
  timestamp: number;
}

// ============================================================================
// Portfolio Types
// ============================================================================

export interface PortfolioData {
  address: string;
  totalValue: string;
  currency: string;
  assets: PortfolioAsset[];
  metrics: PortfolioMetrics;
  lastUpdated: number;
}

export interface PortfolioAsset {
  symbol: string;
  address: string;
  balance: string;
  value: string;
  allocation: number;
}

export interface PortfolioMetrics {
  volatility30d: number;
  sharpeRatio: number;
  unrealizedPnL: number;
  unrealizedPnLPercent: number;
}

// ============================================================================
// AI Prediction Types
// ============================================================================

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

export interface DayPrediction {
  day: number;
  price: number;
  confidence: number;
}

export interface RiskAssessment {
  riskScore: number;
  volatility: number;
  sharpeRatio: number;
  maxDrawdown: number;
  recommendations: string[];
}

// ============================================================================
// DePIN Types
// ============================================================================

export interface SensorRewards {
  sensorId: string;
  owner: string;
  deviceType: string;
  status: 'active' | 'inactive' | 'maintenance';
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
// Tokenomics Types
// ============================================================================

export interface TokenomicsStats {
  totalSupply: string;
  circulatingSupply: string;
  burnedSupply: string;
  stakedSupply: string;
  price: {
    usd: number;
    change24h: number;
    change7d: number;
  };
  marketCap: string;
  fdv: string;
  stakingAPY: number;
  nextBuybackDate: number;
  lastBuyback: {
    date: number;
    amount: string;
    burned: string;
  };
}

// ============================================================================
// WebSocket Types
// ============================================================================

export type WebSocketChannel = 
  | 'portfolio'
  | 'predictions'
  | 'rebalance'
  | 'rewards'
  | 'blocks'
  | 'prices';

export interface WebSocketSubscription {
  type: 'subscribe' | 'unsubscribe';
  channel: WebSocketChannel;
  params?: Record<string, unknown>;
}

export interface WebSocketMessage<T = unknown> {
  channel: WebSocketChannel;
  payload: T;
  timestamp: number;
}

// ============================================================================
// Auth Types
// ============================================================================

export interface ApiKeyInfo {
  id: string;
  key: string;
  tier: ApiTier;
  userId: string;
  permissions: string[];
  createdAt: Date;
  expiresAt?: Date;
  lastUsedAt?: Date;
}

export type ApiTier = 'free' | 'basic' | 'pro' | 'enterprise';

export interface AuthenticatedRequest extends Request {
  apiKey?: ApiKeyInfo;
  userId?: string;
}

// ============================================================================
// Rate Limit Types
// ============================================================================

export interface RateLimitConfig {
  windowMs: number;
  maxRequests: number;
  message: string;
}

export const RATE_LIMITS: Record<ApiTier, RateLimitConfig> = {
  free: {
    windowMs: 60 * 1000,
    maxRequests: 100,
    message: 'Free tier rate limit exceeded. Upgrade for higher limits.',
  },
  basic: {
    windowMs: 60 * 1000,
    maxRequests: 1000,
    message: 'Basic tier rate limit exceeded.',
  },
  pro: {
    windowMs: 60 * 1000,
    maxRequests: 10000,
    message: 'Pro tier rate limit exceeded.',
  },
  enterprise: {
    windowMs: 60 * 1000,
    maxRequests: 1000000, // Effectively unlimited
    message: 'Enterprise rate limit exceeded.',
  },
};

// ============================================================================
// Config Types
// ============================================================================

export interface ServerConfig {
  port: number;
  host: string;
  env: 'development' | 'production' | 'test';
  corsOrigins: string[];
  rpcUrl: string;
  chainId: number;
  redisUrl?: string;
  jwtSecret: string;
  logLevel: string;
}
