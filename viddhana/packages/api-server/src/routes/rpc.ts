/**
 * JSON-RPC Route Handler
 * 
 * Handles all JSON-RPC 2.0 requests for:
 * - Atlas Chain methods (atlas_*)
 * - Vault methods (vault_*)
 * - AI Prediction methods (ai_*)
 * - Standard Ethereum methods (eth_*)
 */

import { Router, Request, Response } from 'express';
import { z } from 'zod';
import { BlockchainService } from '../services/blockchain';
import { logger } from '../services/logger';
import {
  JsonRpcRequest,
  JsonRpcResponse,
  JsonRpcError,
  JsonRpcErrorCode,
  ChainInfo,
  BalanceInfo,
  BlockInfo,
  VaultInfo,
  VaultTransaction,
  PricePrediction,
  PortfolioData,
  TokenomicsStats,
  SensorRewards,
} from '../types';

const router = Router();
const blockchainService = new BlockchainService();

// ============================================================================
// Validation Schemas
// ============================================================================

const JsonRpcRequestSchema = z.object({
  jsonrpc: z.literal('2.0'),
  method: z.string().min(1),
  params: z.array(z.unknown()).optional().default([]),
  id: z.union([z.string(), z.number()]),
});

const AddressSchema = z.string().regex(/^0x[a-fA-F0-9]{40}$/, 'Invalid address format');
const BlockNumberSchema = z.union([z.number(), z.literal('latest'), z.literal('earliest'), z.literal('pending')]);

// ============================================================================
// RPC Method Handlers
// ============================================================================

type RpcHandler = (params: unknown[]) => Promise<unknown>;

const rpcHandlers: Record<string, RpcHandler> = {
  // ==========================================================================
  // Atlas Chain Methods
  // ==========================================================================

  /**
   * Get chain information
   */
  atlas_getChainInfo: async (): Promise<ChainInfo> => {
    const [blockNumber, gasPrice] = await Promise.all([
      blockchainService.getBlockNumber(),
      blockchainService.getGasPrice(),
    ]);

    return {
      chainId: blockchainService.getChainId(),
      name: 'VIDDHANA Atlas',
      symbol: 'VDH',
      blockNumber,
      gasPrice: gasPrice.toString(),
      blockTime: 2,
      version: '1.0.0',
    };
  },

  /**
   * Get account balance
   */
  atlas_getBalance: async (params: unknown[]): Promise<BalanceInfo> => {
    const address = AddressSchema.parse(params[0]);
    const blockTag = params[1] ? BlockNumberSchema.parse(params[1]) : 'latest';

    const balance = await blockchainService.getBalance(address, blockTag);

    return {
      address,
      balance: balance.toString(),
      balanceFormatted: blockchainService.formatEther(balance),
      symbol: 'VDH',
      decimals: 18,
    };
  },

  /**
   * Get block by number or hash
   */
  atlas_getBlock: async (params: unknown[]): Promise<BlockInfo | null> => {
    const blockId = params[0];
    const includeTransactions = params[1] === true;

    let block;
    if (typeof blockId === 'string' && blockId.startsWith('0x') && blockId.length === 66) {
      // Block hash
      block = await blockchainService.getBlockByHash(blockId, includeTransactions);
    } else {
      // Block number or tag
      const blockNumber = BlockNumberSchema.parse(blockId);
      block = await blockchainService.getBlockByNumber(blockNumber, includeTransactions);
    }

    if (!block) return null;

    // Handle transactions - when includeTransactions is true, block.transactions are TransactionResponse objects
    // When false, they are just hash strings
    const txHashes: string[] = includeTransactions
      ? block.transactions.map((tx) => typeof tx === 'string' ? tx : tx.hash)
      : block.transactions as string[];

    return {
      number: block.number,
      hash: block.hash,
      parentHash: block.parentHash,
      timestamp: block.timestamp,
      gasLimit: block.gasLimit.toString(),
      gasUsed: block.gasUsed.toString(),
      transactions: txHashes,
      transactionCount: block.transactions.length,
    };
  },

  // ==========================================================================
  // Vault Methods
  // ==========================================================================

  /**
   * Create a new vault
   */
  vault_create: async (params: unknown[]): Promise<VaultInfo> => {
    const createParams = z.object({
      owner: AddressSchema,
      name: z.string().min(1).max(64),
      riskTolerance: z.number().min(0).max(10000),
      timeToGoal: z.number().min(1).max(1200),
      autoRebalance: z.boolean(),
    }).parse(params[0]);

    // In production, this would interact with the smart contract
    const vaultId = `vault_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;

    return {
      id: vaultId,
      owner: createParams.owner,
      name: createParams.name,
      totalValue: '0',
      currency: 'USD',
      assets: [],
      createdAt: Math.floor(Date.now() / 1000),
      lastUpdated: Math.floor(Date.now() / 1000),
    };
  },

  /**
   * Deposit assets to vault
   */
  vault_deposit: async (params: unknown[]): Promise<VaultTransaction> => {
    const depositParams = z.object({
      vaultId: z.string(),
      asset: AddressSchema,
      amount: z.string().regex(/^\d+$/),
      from: AddressSchema,
    }).parse(params[0]);

    // In production, this would create and submit a transaction
    const txHash = `0x${Array(64).fill(0).map(() => 
      Math.floor(Math.random() * 16).toString(16)).join('')}`;

    return {
      hash: txHash,
      type: 'deposit',
      status: 'pending',
      timestamp: Math.floor(Date.now() / 1000),
    };
  },

  /**
   * Withdraw assets from vault
   */
  vault_withdraw: async (params: unknown[]): Promise<VaultTransaction> => {
    const withdrawParams = z.object({
      vaultId: z.string(),
      asset: AddressSchema,
      amount: z.string().regex(/^\d+$/),
      to: AddressSchema,
    }).parse(params[0]);

    // In production, this would create and submit a transaction
    const txHash = `0x${Array(64).fill(0).map(() => 
      Math.floor(Math.random() * 16).toString(16)).join('')}`;

    return {
      hash: txHash,
      type: 'withdraw',
      status: 'pending',
      timestamp: Math.floor(Date.now() / 1000),
    };
  },

  /**
   * Get vault info by ID
   */
  vault_getVault: async (params: unknown[]): Promise<VaultInfo | null> => {
    const vaultId = z.string().parse(params[0]);

    // Mock data - in production, fetch from blockchain/database
    return {
      id: vaultId,
      owner: '0x1234567890123456789012345678901234567890',
      name: 'My Investment Vault',
      totalValue: '125000.00',
      currency: 'USD',
      assets: [
        { symbol: 'BTC', address: '0xabc...', balance: '1.5', value: '75000.00', allocation: 60 },
        { symbol: 'ETH', address: '0xdef...', balance: '15.0', value: '30000.00', allocation: 24 },
        { symbol: 'USDC', address: '0x789...', balance: '20000.0', value: '20000.00', allocation: 16 },
      ],
      createdAt: 1704067200,
      lastUpdated: Math.floor(Date.now() / 1000),
    };
  },

  // ==========================================================================
  // AI Prediction Methods
  // ==========================================================================

  /**
   * Get AI price prediction
   */
  ai_getPrediction: async (params: unknown[]): Promise<PricePrediction> => {
    const predictionParams = z.object({
      asset: z.string().min(1),
      horizon: z.number().min(1).max(30).default(7),
    }).parse(params[0] || {});

    const { asset, horizon } = predictionParams;

    // Mock prediction data - in production, call Prometheus AI service
    const basePrice = asset === 'BTC' ? 50000 : asset === 'ETH' ? 3000 : 1;
    const predictions = [];

    for (let day = 1; day <= horizon; day++) {
      const priceDelta = (Math.random() - 0.4) * 0.02; // Slight upward bias
      const price = basePrice * (1 + priceDelta * day);
      const confidence = Math.max(0.5, 0.9 - (day * 0.04)); // Decreasing confidence

      predictions.push({
        day,
        price: Math.round(price * 100) / 100,
        confidence: Math.round(confidence * 100) / 100,
      });
    }

    const trend = predictions[predictions.length - 1].price > basePrice ? 'bullish' : 
                  predictions[predictions.length - 1].price < basePrice ? 'bearish' : 'neutral';

    return {
      asset,
      currentPrice: basePrice,
      horizon,
      predictions,
      trend,
      volatilityForecast: 0.15,
      modelVersion: 'v2.3.1',
      generatedAt: Math.floor(Date.now() / 1000),
    };
  },

  /**
   * Get portfolio optimization
   */
  ai_optimizePortfolio: async (params: unknown[]): Promise<object> => {
    const optimizeParams = z.object({
      userId: z.string(),
      portfolio: z.record(z.number()),
      riskTolerance: z.number().min(0).max(1),
      timeToGoal: z.number().min(1),
    }).parse(params[0]);

    // Mock optimization - in production, call Prometheus AI
    return {
      action: 'REBALANCE',
      recommendations: [
        { asset: 'BTC', action: 'HOLD', percentage: 0, reason: 'Position within target allocation' },
        { asset: 'ETH', action: 'BUY', percentage: 5, reason: 'Underweight relative to risk profile' },
        { asset: 'USDC', action: 'SELL', percentage: 5, reason: 'Reduce cash position for growth' },
      ],
      confidence: 0.85,
      riskAssessment: {
        currentRisk: 0.45,
        targetRisk: optimizeParams.riskTolerance,
        adjustmentNeeded: false,
      },
      estimatedReturn: 0.12,
      timestamp: Math.floor(Date.now() / 1000),
    };
  },

  // ==========================================================================
  // VDH Custom Methods (from blueprint)
  // ==========================================================================

  /**
   * Get user portfolio
   */
  vdh_getPortfolio: async (params: unknown[]): Promise<PortfolioData> => {
    const address = AddressSchema.parse(params[0]);

    return {
      address,
      totalValue: '125000.00',
      currency: 'USD',
      assets: [
        { symbol: 'BTC', address: '0xabc...', balance: '1.5', value: '75000.00', allocation: 60 },
        { symbol: 'ETH', address: '0xdef...', balance: '15.0', value: '30000.00', allocation: 24 },
        { symbol: 'USDC', address: '0x789...', balance: '20000.0', value: '20000.00', allocation: 16 },
      ],
      metrics: {
        volatility30d: 0.18,
        sharpeRatio: 1.45,
        unrealizedPnL: 15000,
        unrealizedPnLPercent: 13.6,
      },
      lastUpdated: Math.floor(Date.now() / 1000),
    };
  },

  /**
   * Get rebalance history
   */
  vdh_getRebalanceHistory: async (params: unknown[]): Promise<object> => {
    const address = AddressSchema.parse(params[0]);
    const options = z.object({
      fromBlock: z.number().optional(),
      toBlock: z.union([z.number(), z.literal('latest')]).optional(),
      limit: z.number().max(100).optional().default(50),
    }).parse(params[1] || {});

    return {
      address,
      history: [
        {
          txHash: '0xabc...',
          blockNumber: 1050000,
          timestamp: 1704067200,
          reason: 'Goal Proximity Mode',
          actions: [
            { asset: 'BTC', action: 'SELL', percentage: 25, amount: '0.5', valueUSD: '25000.00' },
            { asset: 'USDC', action: 'BUY', percentage: 25, amount: '25000.0', valueUSD: '25000.00' },
          ],
          aiConfidence: 0.85,
          gasUsed: 150000,
          gasCost: '0.00015',
        },
      ],
      totalRebalances: 12,
    };
  },

  /**
   * Get AI prediction (VDH format)
   */
  vdh_getAIPrediction: async (params: unknown[]): Promise<PricePrediction> => {
    return rpcHandlers.ai_getPrediction(params);
  },

  /**
   * Get sensor rewards
   */
  vdh_getSensorRewards: async (params: unknown[]): Promise<SensorRewards> => {
    const sensorId = z.string().parse(params[0]);

    return {
      sensorId,
      owner: '0x1234567890123456789012345678901234567890',
      deviceType: 'solar_panel',
      status: 'active',
      rewards: {
        pending: '15.50',
        claimed: '450.00',
        lifetime: '465.50',
        currency: 'VDH',
      },
      performance: {
        uptimePercent: 99.5,
        dataQualityScore: 0.98,
        dailyRate: '0.60',
        errorCount: 2,
      },
      lastDataPoint: Math.floor(Date.now() / 1000),
    };
  },

  /**
   * Get tokenomics stats
   */
  vdh_getTokenomicsStats: async (): Promise<TokenomicsStats> => {
    return {
      totalSupply: '1000000000',
      circulatingSupply: '223000000',
      burnedSupply: '5000000',
      stakedSupply: '75000000',
      price: {
        usd: 0.45,
        change24h: 5.2,
        change7d: 12.8,
      },
      marketCap: '100350000',
      fdv: '450000000',
      stakingAPY: 12.5,
      nextBuybackDate: 1706745600,
      lastBuyback: {
        date: 1704067200,
        amount: '500000',
        burned: '250000',
      },
    };
  },

  // ==========================================================================
  // Standard Ethereum Methods (Proxy)
  // ==========================================================================

  eth_chainId: async (): Promise<string> => {
    return `0x${blockchainService.getChainId().toString(16)}`;
  },

  eth_blockNumber: async (): Promise<string> => {
    const blockNumber = await blockchainService.getBlockNumber();
    return `0x${blockNumber.toString(16)}`;
  },

  eth_getBalance: async (params: unknown[]): Promise<string> => {
    const address = AddressSchema.parse(params[0]);
    const balance = await blockchainService.getBalance(address);
    return `0x${balance.toString(16)}`;
  },

  eth_gasPrice: async (): Promise<string> => {
    const gasPrice = await blockchainService.getGasPrice();
    return `0x${gasPrice.toString(16)}`;
  },

  eth_getTransactionCount: async (params: unknown[]): Promise<string> => {
    const address = AddressSchema.parse(params[0]);
    const count = await blockchainService.getTransactionCount(address);
    return `0x${count.toString(16)}`;
  },

  eth_sendRawTransaction: async (params: unknown[]): Promise<string> => {
    const signedTx = z.string().parse(params[0]);
    const txHash = await blockchainService.sendRawTransaction(signedTx);
    return txHash;
  },

  eth_getTransactionReceipt: async (params: unknown[]): Promise<object | null> => {
    const txHash = z.string().parse(params[0]);
    return blockchainService.getTransactionReceipt(txHash);
  },

  eth_call: async (params: unknown[]): Promise<string> => {
    const callParams = z.object({
      to: AddressSchema,
      data: z.string(),
      from: AddressSchema.optional(),
      gas: z.string().optional(),
      gasPrice: z.string().optional(),
      value: z.string().optional(),
    }).parse(params[0]);
    
    const blockTag = params[1] ? BlockNumberSchema.parse(params[1]) : 'latest';
    return blockchainService.call(callParams, blockTag);
  },

  eth_estimateGas: async (params: unknown[]): Promise<string> => {
    const txParams = z.object({
      to: AddressSchema.optional(),
      data: z.string().optional(),
      from: AddressSchema.optional(),
      value: z.string().optional(),
    }).parse(params[0]);

    const gas = await blockchainService.estimateGas(txParams);
    return `0x${gas.toString(16)}`;
  },

  net_version: async (): Promise<string> => {
    return blockchainService.getChainId().toString();
  },

  web3_clientVersion: async (): Promise<string> => {
    return 'VIDDHANA/v1.0.0';
  },
};

// ============================================================================
// Request Handler
// ============================================================================

async function handleRpcRequest(request: JsonRpcRequest): Promise<JsonRpcResponse> {
  const { method, params, id } = request;

  try {
    const handler = rpcHandlers[method];

    if (!handler) {
      return {
        jsonrpc: '2.0',
        id,
        error: {
          code: JsonRpcErrorCode.METHOD_NOT_FOUND,
          message: `Method '${method}' not found`,
        },
      };
    }

    const result = await handler(params || []);

    return {
      jsonrpc: '2.0',
      id,
      result,
    };

  } catch (error) {
    logger.error(`RPC error for method ${method}:`, error);

    if (error instanceof z.ZodError) {
      return {
        jsonrpc: '2.0',
        id,
        error: {
          code: JsonRpcErrorCode.INVALID_PARAMS,
          message: 'Invalid parameters',
          data: error.errors,
        },
      };
    }

    return {
      jsonrpc: '2.0',
      id,
      error: {
        code: JsonRpcErrorCode.INTERNAL_ERROR,
        message: error instanceof Error ? error.message : 'Internal error',
      },
    };
  }
}

// ============================================================================
// Route Handler
// ============================================================================

router.post('/', async (req: Request, res: Response) => {
  const body = req.body;

  try {
    // Handle batch requests
    if (Array.isArray(body)) {
      const requests = body.map((r) => JsonRpcRequestSchema.safeParse(r));
      const invalidIndex = requests.findIndex((r) => !r.success);

      if (invalidIndex !== -1) {
        return res.json({
          jsonrpc: '2.0',
          id: null,
          error: {
            code: JsonRpcErrorCode.INVALID_REQUEST,
            message: `Invalid request at index ${invalidIndex}`,
          },
        });
      }

      const validRequests = requests.map((r) => r.data!);
      const responses = await Promise.all(validRequests.map(handleRpcRequest));
      
      return res.json(responses);
    }

    // Handle single request
    const parseResult = JsonRpcRequestSchema.safeParse(body);

    if (!parseResult.success) {
      return res.json({
        jsonrpc: '2.0',
        id: body?.id ?? null,
        error: {
          code: JsonRpcErrorCode.INVALID_REQUEST,
          message: 'Invalid JSON-RPC request',
          data: parseResult.error.errors,
        },
      });
    }

    const response = await handleRpcRequest(parseResult.data);
    return res.json(response);

  } catch (error) {
    logger.error('RPC route error:', error);

    return res.json({
      jsonrpc: '2.0',
      id: body?.id ?? null,
      error: {
        code: JsonRpcErrorCode.PARSE_ERROR,
        message: 'Failed to parse request',
      },
    });
  }
});

// List available methods
router.get('/methods', (_req: Request, res: Response) => {
  res.json({
    methods: Object.keys(rpcHandlers).sort(),
    version: '1.0.0',
  });
});

export { router as rpcRouter };
