/**
 * REST API Route Handler
 * 
 * RESTful endpoints for VIDDHANA services
 */

import { Router, Request, Response, NextFunction } from 'express';
import { z } from 'zod';
import { PrismaClient } from '@prisma/client';
import { BlockchainService } from '../services/blockchain';
import { logger } from '../services/logger';
import { AuthenticatedRequest } from '../types';

const router = Router();
const prisma = new PrismaClient();
const blockchainService = new BlockchainService();

// ============================================================================
// Validation Schemas
// ============================================================================

const AddressSchema = z.string().regex(/^0x[a-fA-F0-9]{40}$/, 'Invalid address');
const PaginationSchema = z.object({
  page: z.coerce.number().min(1).default(1),
  limit: z.coerce.number().min(1).max(100).default(20),
});

// ============================================================================
// Health & Status
// ============================================================================

router.get('/status', async (_req: Request, res: Response): Promise<void> => {
  const [blockNumber, gasPrice] = await Promise.all([
    blockchainService.getBlockNumber().catch(() => 0),
    blockchainService.getGasPrice().catch(() => BigInt(0)),
  ]);

  res.json({
    status: 'operational',
    chain: {
      connected: blockNumber > 0,
      blockNumber,
      gasPrice: gasPrice.toString(),
    },
    timestamp: new Date().toISOString(),
  });
});

// ============================================================================
// Chain Endpoints
// ============================================================================

/**
 * GET /api/v1/chain/info
 * Get chain information
 */
router.get('/chain/info', async (_req: Request, res: Response): Promise<void> => {
  try {
    const [blockNumber, gasPrice] = await Promise.all([
      blockchainService.getBlockNumber(),
      blockchainService.getGasPrice(),
    ]);

    res.json({
      chainId: blockchainService.getChainId(),
      name: 'VIDDHANA Atlas',
      symbol: 'VDH',
      blockNumber,
      gasPrice: gasPrice.toString(),
      blockTime: 2,
      version: '1.0.0',
    });
  } catch (error) {
    logger.error('Failed to get chain info:', error);
    res.status(500).json({ error: 'Failed to get chain info' });
  }
});

/**
 * GET /api/v1/chain/blocks/:blockNumber
 * Get block by number
 */
router.get('/chain/blocks/:blockNumber', async (req: Request, res: Response): Promise<void> => {
  try {
    const blockNumber = req.params.blockNumber === 'latest' 
      ? 'latest' 
      : parseInt(req.params.blockNumber, 10);

    const block = await blockchainService.getBlockByNumber(blockNumber, true);

    if (!block) {
      res.status(404).json({ error: 'Block not found' });
      return;
    }

    res.json(block);
  } catch (error) {
    logger.error('Failed to get block:', error);
    res.status(500).json({ error: 'Failed to get block' });
  }
});

/**
 * GET /api/v1/chain/transactions/:txHash
 * Get transaction by hash
 */
router.get('/chain/transactions/:txHash', async (req: Request, res: Response): Promise<void> => {
  try {
    const receipt = await blockchainService.getTransactionReceipt(req.params.txHash);

    if (!receipt) {
      res.status(404).json({ error: 'Transaction not found' });
      return;
    }

    res.json(receipt);
  } catch (error) {
    logger.error('Failed to get transaction:', error);
    res.status(500).json({ error: 'Failed to get transaction' });
  }
});

// ============================================================================
// Account Endpoints
// ============================================================================

/**
 * GET /api/v1/accounts/:address/balance
 * Get account balance
 */
router.get('/accounts/:address/balance', async (req: Request, res: Response): Promise<void> => {
  try {
    const address = AddressSchema.parse(req.params.address);
    const balance = await blockchainService.getBalance(address);

    res.json({
      address,
      balance: balance.toString(),
      balanceFormatted: blockchainService.formatEther(balance),
      symbol: 'VDH',
      decimals: 18,
    });
  } catch (error) {
    if (error instanceof z.ZodError) {
      res.status(400).json({ error: 'Invalid address format' });
      return;
    }
    logger.error('Failed to get balance:', error);
    res.status(500).json({ error: 'Failed to get balance' });
  }
});

/**
 * GET /api/v1/accounts/:address/transactions
 * Get account transactions
 */
router.get('/accounts/:address/transactions', async (req: Request, res: Response): Promise<void> => {
  try {
    const address = AddressSchema.parse(req.params.address);
    const { page, limit } = PaginationSchema.parse(req.query);

    // In production, query from indexed database
    const transactions = await prisma.transaction.findMany({
      where: {
        OR: [
          { from: address.toLowerCase() },
          { to: address.toLowerCase() },
        ],
      },
      orderBy: { blockNumber: 'desc' },
      skip: (page - 1) * limit,
      take: limit,
    });

    const total = await prisma.transaction.count({
      where: {
        OR: [
          { from: address.toLowerCase() },
          { to: address.toLowerCase() },
        ],
      },
    });

    res.json({
      transactions,
      pagination: {
        page,
        limit,
        total,
        totalPages: Math.ceil(total / limit),
      },
    });
  } catch (error) {
    if (error instanceof z.ZodError) {
      res.status(400).json({ error: 'Invalid parameters' });
      return;
    }
    logger.error('Failed to get transactions:', error);
    res.status(500).json({ error: 'Failed to get transactions' });
  }
});

// ============================================================================
// Portfolio Endpoints
// ============================================================================

/**
 * GET /api/v1/portfolio/:address
 * Get user portfolio
 */
router.get('/portfolio/:address', async (req: Request, res: Response): Promise<void> => {
  try {
    const address = AddressSchema.parse(req.params.address);

    // Mock data - in production, aggregate from blockchain and price feeds
    res.json({
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
    });
  } catch (error) {
    if (error instanceof z.ZodError) {
      res.status(400).json({ error: 'Invalid address format' });
      return;
    }
    logger.error('Failed to get portfolio:', error);
    res.status(500).json({ error: 'Failed to get portfolio' });
  }
});

/**
 * GET /api/v1/portfolio/:address/history
 * Get portfolio history
 */
router.get('/portfolio/:address/history', async (req: Request, res: Response): Promise<void> => {
  try {
    const address = AddressSchema.parse(req.params.address);
    const period = z.enum(['1d', '7d', '30d', '90d', '1y', 'all']).default('30d').parse(req.query.period);

    // Mock history data
    const dataPoints = [];
    const now = Date.now();
    const periodDays = period === '1d' ? 1 : period === '7d' ? 7 : period === '30d' ? 30 : 
                       period === '90d' ? 90 : period === '1y' ? 365 : 730;
    
    for (let i = periodDays; i >= 0; i--) {
      dataPoints.push({
        timestamp: Math.floor((now - i * 24 * 60 * 60 * 1000) / 1000),
        totalValue: (100000 + Math.random() * 50000).toFixed(2),
      });
    }

    res.json({
      address,
      period,
      dataPoints,
    });
  } catch (error) {
    logger.error('Failed to get portfolio history:', error);
    res.status(500).json({ error: 'Failed to get portfolio history' });
  }
});

// ============================================================================
// Vault Endpoints
// ============================================================================

/**
 * GET /api/v1/vaults
 * List user's vaults
 */
router.get('/vaults', async (req: AuthenticatedRequest, res: Response): Promise<void> => {
  try {
    const { page, limit } = PaginationSchema.parse(req.query);

    const vaults = await prisma.vault.findMany({
      where: { userId: req.userId },
      orderBy: { createdAt: 'desc' },
      skip: (page - 1) * limit,
      take: limit,
    });

    res.json({
      vaults,
      pagination: { page, limit },
    });
  } catch (error) {
    logger.error('Failed to list vaults:', error);
    res.status(500).json({ error: 'Failed to list vaults' });
  }
});

/**
 * POST /api/v1/vaults
 * Create a new vault
 */
router.post('/vaults', async (req: AuthenticatedRequest, res: Response): Promise<void> => {
  try {
    const createParams = z.object({
      name: z.string().min(1).max(64),
      riskTolerance: z.number().min(0).max(10000),
      timeToGoal: z.number().min(1).max(1200),
      autoRebalance: z.boolean().default(true),
    }).parse(req.body);

    const vault = await prisma.vault.create({
      data: {
        userId: req.userId!,
        name: createParams.name,
        riskTolerance: createParams.riskTolerance,
        timeToGoal: createParams.timeToGoal,
        autoRebalance: createParams.autoRebalance,
        totalValue: '0',
      },
    });

    res.status(201).json(vault);
  } catch (error) {
    if (error instanceof z.ZodError) {
      res.status(400).json({ error: 'Invalid parameters', details: error.errors });
      return;
    }
    logger.error('Failed to create vault:', error);
    res.status(500).json({ error: 'Failed to create vault' });
  }
});

/**
 * GET /api/v1/vaults/:vaultId
 * Get vault details
 */
router.get('/vaults/:vaultId', async (req: Request, res: Response): Promise<void> => {
  try {
    const vault = await prisma.vault.findUnique({
      where: { id: req.params.vaultId },
      include: { assets: true },
    });

    if (!vault) {
      res.status(404).json({ error: 'Vault not found' });
      return;
    }

    res.json(vault);
  } catch (error) {
    logger.error('Failed to get vault:', error);
    res.status(500).json({ error: 'Failed to get vault' });
  }
});

/**
 * POST /api/v1/vaults/:vaultId/deposit
 * Deposit assets to vault
 */
router.post('/vaults/:vaultId/deposit', async (req: AuthenticatedRequest, res: Response): Promise<void> => {
  try {
    const depositParams = z.object({
      asset: AddressSchema,
      amount: z.string().regex(/^\d+$/),
    }).parse(req.body);

    // In production, this would create a blockchain transaction
    const deposit = await prisma.vaultTransaction.create({
      data: {
        vaultId: req.params.vaultId,
        type: 'DEPOSIT',
        asset: depositParams.asset,
        amount: depositParams.amount,
        status: 'PENDING',
      },
    });

    res.status(201).json(deposit);
  } catch (error) {
    if (error instanceof z.ZodError) {
      res.status(400).json({ error: 'Invalid parameters', details: error.errors });
      return;
    }
    logger.error('Failed to deposit:', error);
    res.status(500).json({ error: 'Failed to process deposit' });
  }
});

/**
 * POST /api/v1/vaults/:vaultId/withdraw
 * Withdraw assets from vault
 */
router.post('/vaults/:vaultId/withdraw', async (req: AuthenticatedRequest, res: Response): Promise<void> => {
  try {
    const withdrawParams = z.object({
      asset: AddressSchema,
      amount: z.string().regex(/^\d+$/),
    }).parse(req.body);

    // In production, this would create a blockchain transaction
    const withdrawal = await prisma.vaultTransaction.create({
      data: {
        vaultId: req.params.vaultId,
        type: 'WITHDRAW',
        asset: withdrawParams.asset,
        amount: withdrawParams.amount,
        status: 'PENDING',
      },
    });

    res.status(201).json(withdrawal);
  } catch (error) {
    if (error instanceof z.ZodError) {
      res.status(400).json({ error: 'Invalid parameters', details: error.errors });
      return;
    }
    logger.error('Failed to withdraw:', error);
    res.status(500).json({ error: 'Failed to process withdrawal' });
  }
});

// ============================================================================
// AI Prediction Endpoints
// ============================================================================

/**
 * POST /api/v1/predict/price
 * Get price prediction
 */
router.post('/predict/price', async (req: Request, res: Response): Promise<void> => {
  try {
    const predictionParams = z.object({
      asset: z.string().min(1),
      horizon: z.number().min(1).max(30).default(7),
    }).parse(req.body);

    const { asset, horizon } = predictionParams;

    // Mock prediction - in production, call Prometheus AI
    const basePrice = asset === 'BTC' ? 50000 : asset === 'ETH' ? 3000 : 1;
    const predictions = [];

    for (let day = 1; day <= horizon; day++) {
      const priceDelta = (Math.random() - 0.4) * 0.02;
      const price = basePrice * (1 + priceDelta * day);
      const confidence = Math.max(0.5, 0.9 - (day * 0.04));

      predictions.push({
        day,
        price: Math.round(price * 100) / 100,
        confidence: Math.round(confidence * 100) / 100,
      });
    }

    res.json({
      asset,
      currentPrice: basePrice,
      horizon,
      predictions,
      trend: predictions[predictions.length - 1].price > basePrice ? 'bullish' : 'bearish',
      volatilityForecast: 0.15,
      modelVersion: 'v2.3.1',
      generatedAt: Math.floor(Date.now() / 1000),
    });
  } catch (error) {
    if (error instanceof z.ZodError) {
      res.status(400).json({ error: 'Invalid parameters', details: error.errors });
      return;
    }
    logger.error('Failed to get prediction:', error);
    res.status(500).json({ error: 'Failed to get prediction' });
  }
});

/**
 * POST /api/v1/optimize/portfolio
 * Get portfolio optimization
 */
router.post('/optimize/portfolio', async (req: Request, res: Response): Promise<void> => {
  try {
    const optimizeParams = z.object({
      user_id: z.string(),
      portfolio: z.record(z.number()),
      risk_tolerance: z.number().min(0).max(1),
      time_to_goal: z.number().min(1),
    }).parse(req.body);

    res.json({
      action: 'REBALANCE',
      recommendations: [
        { asset: 'BTC', action: 'HOLD', percentage: 0, reason: 'Position within target' },
        { asset: 'ETH', action: 'BUY', percentage: 5, reason: 'Underweight' },
        { asset: 'USDC', action: 'SELL', percentage: 5, reason: 'Reduce cash' },
      ],
      confidence: 0.85,
      risk_assessment: {
        current_risk: 0.45,
        target_risk: optimizeParams.risk_tolerance,
        adjustment_needed: false,
      },
      estimated_return: 0.12,
      timestamp: Math.floor(Date.now() / 1000),
    });
  } catch (error) {
    if (error instanceof z.ZodError) {
      res.status(400).json({ error: 'Invalid parameters', details: error.errors });
      return;
    }
    logger.error('Failed to optimize portfolio:', error);
    res.status(500).json({ error: 'Failed to optimize portfolio' });
  }
});

/**
 * POST /api/v1/assess/risk
 * Assess portfolio risk
 */
router.post('/assess/risk', async (req: Request, res: Response): Promise<void> => {
  try {
    // Validate input (portfolio is required for real implementation)
    z.object({
      portfolio: z.record(z.number()),
    }).parse(req.body);

    res.json({
      risk_score: 0.45,
      volatility: 0.18,
      sharpe_ratio: 1.45,
      max_drawdown: 0.25,
      recommendations: [
        'Consider adding stablecoins to reduce volatility',
        'Portfolio is well-diversified across asset classes',
      ],
    });
  } catch (error) {
    if (error instanceof z.ZodError) {
      res.status(400).json({ error: 'Invalid parameters', details: error.errors });
      return;
    }
    logger.error('Failed to assess risk:', error);
    res.status(500).json({ error: 'Failed to assess risk' });
  }
});

// ============================================================================
// DePIN Endpoints
// ============================================================================

/**
 * GET /api/v1/sensors/:sensorId
 * Get sensor info
 */
router.get('/sensors/:sensorId', async (req: Request, res: Response): Promise<void> => {
  try {
    res.json({
      sensorId: req.params.sensorId,
      owner: '0x1234567890123456789012345678901234567890',
      deviceType: 'solar_panel',
      status: 'active',
      location: { lat: 37.7749, lng: -122.4194 },
      registeredAt: Math.floor(Date.now() / 1000) - 86400 * 30,
    });
  } catch (error) {
    logger.error('Failed to get sensor:', error);
    res.status(500).json({ error: 'Failed to get sensor info' });
  }
});

/**
 * GET /api/v1/sensors/:sensorId/rewards
 * Get sensor rewards
 */
router.get('/sensors/:sensorId/rewards', async (req: Request, res: Response): Promise<void> => {
  try {
    res.json({
      sensorId: req.params.sensorId,
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
    });
  } catch (error) {
    logger.error('Failed to get sensor rewards:', error);
    res.status(500).json({ error: 'Failed to get sensor rewards' });
  }
});

// ============================================================================
// Tokenomics Endpoints
// ============================================================================

/**
 * GET /api/v1/tokenomics/stats
 * Get tokenomics statistics
 */
router.get('/tokenomics/stats', async (_req: Request, res: Response): Promise<void> => {
  try {
    res.json({
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
      nextBuybackDate: Math.floor(Date.now() / 1000) + 86400 * 30,
      lastBuyback: {
        date: Math.floor(Date.now() / 1000) - 86400 * 30,
        amount: '500000',
        burned: '250000',
      },
    });
  } catch (error) {
    logger.error('Failed to get tokenomics stats:', error);
    res.status(500).json({ error: 'Failed to get tokenomics stats' });
  }
});

/**
 * GET /api/v1/staking/stats
 * Get staking statistics
 */
router.get('/staking/stats', async (_req: Request, res: Response): Promise<void> => {
  try {
    res.json({
      totalStaked: '75000000',
      totalStakers: 12500,
      averageStakeDuration: 180,
      currentAPY: 12.5,
      rewardsDistributed: '5000000',
    });
  } catch (error) {
    logger.error('Failed to get staking stats:', error);
    res.status(500).json({ error: 'Failed to get staking stats' });
  }
});

export { router as restRouter };
