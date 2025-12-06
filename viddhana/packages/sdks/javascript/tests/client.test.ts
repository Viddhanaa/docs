/**
 * VIDDHANA SDK Tests
 */

import { ViddhanaClient } from '../src/client';
import { AtlasChain } from '../src/modules/atlas';
import { Vault } from '../src/modules/vault';
import { PrometheusAI } from '../src/modules/ai';
import { Governance } from '../src/modules/governance';

// Mock ethers
jest.mock('ethers', () => {
  const mockProvider = {
    getNetwork: jest.fn().mockResolvedValue({ chainId: 13371n }),
    getBlockNumber: jest.fn().mockResolvedValue(12345),
    getBalance: jest.fn().mockResolvedValue(1000000000000000000n),
    getBlock: jest.fn().mockResolvedValue({
      number: 12345,
      hash: '0x123...',
      parentHash: '0x122...',
      timestamp: 1704067200,
      nonce: '0x0',
      difficulty: 0n,
      gasLimit: 30000000n,
      gasUsed: 15000000n,
      miner: '0xminer...',
      transactions: ['0xtx1', '0xtx2'],
    }),
    getFeeData: jest.fn().mockResolvedValue({ gasPrice: 1000000000n }),
    getTransactionReceipt: jest.fn().mockResolvedValue(null),
    send: jest.fn().mockImplementation((method: string) => {
      if (method === 'vdh_getPortfolio') {
        return {
          address: '0x1234...',
          totalValue: '125000.00',
          currency: 'USD',
          assets: [
            { symbol: 'BTC', address: '0xabc...', balance: '1.5', value: '75000.00', allocation: 60 },
          ],
          metrics: { volatility30d: 0.18, sharpeRatio: 1.45, unrealizedPnL: 15000, unrealizedPnLPercent: 13.6 },
          lastUpdated: 1704067200,
        };
      }
      if (method === 'vdh_getRebalanceHistory') {
        return { address: '0x1234...', history: [], totalRebalances: 0 };
      }
      if (method === 'vdh_getTokenomicsStats') {
        return { totalSupply: '1000000000', circulatingSupply: '223000000' };
      }
      return {};
    }),
    destroy: jest.fn(),
    on: jest.fn(),
    off: jest.fn(),
  };

  return {
    ethers: {
      JsonRpcProvider: jest.fn(() => mockProvider),
      Wallet: jest.fn(() => ({
        address: '0xTestWallet...',
        provider: mockProvider,
      })),
      Contract: jest.fn(() => ({
        getVotes: jest.fn().mockResolvedValue(1000000000000000000000n),
        delegates: jest.fn().mockResolvedValue('0xDelegate...'),
      })),
      formatEther: jest.fn((value: bigint) => (Number(value) / 1e18).toString()),
      parseEther: jest.fn((value: string) => BigInt(Math.floor(parseFloat(value) * 1e18))),
      parseUnits: jest.fn((value: string, decimals: number) =>
        BigInt(Math.floor(parseFloat(value) * Math.pow(10, decimals)))
      ),
      BrowserProvider: jest.fn(),
    },
  };
});

// Mock fetch for AI module tests
const mockFetch = jest.fn().mockImplementation((url: string) => {
  if (url.includes('/v1/predict/price')) {
    return Promise.resolve({
      ok: true,
      json: () =>
        Promise.resolve({
          asset: 'BTC',
          currentPrice: 50000,
          horizon: 7,
          predictions: [
            { day: 1, price: 50500, confidence: 0.82 },
            { day: 2, price: 51200, confidence: 0.78 },
          ],
          trend: 'bullish',
          volatilityForecast: 0.15,
          modelVersion: 'v2.3.1',
          generatedAt: 1704067200,
        }),
    });
  }
  if (url.includes('/v1/optimize/portfolio')) {
    return Promise.resolve({
      ok: true,
      json: () =>
        Promise.resolve({
          action: 'REBALANCE',
          recommendations: [
            { asset: 'BTC', action: 'SELL', percentage: 10, reason: 'Reduce volatility' },
          ],
          confidence: 0.85,
          riskAssessment: { riskScore: 0.5 },
          expectedReturn: 0.12,
          timestamp: 1704067200,
        }),
    });
  }
  if (url.includes('/v1/assess/risk')) {
    return Promise.resolve({
      ok: true,
      json: () =>
        Promise.resolve({
          portfolio: { BTC: 50000, ETH: 30000 },
          metrics: {
            riskScore: 0.5,
            volatility: 0.18,
            sharpeRatio: 1.45,
            maxDrawdown: 0.25,
            beta: 1.1,
            valueAtRisk: 5000,
          },
          recommendations: ['Diversify into stablecoins'],
          riskLevel: 'medium',
          timestamp: 1704067200,
        }),
    });
  }
  return Promise.resolve({ ok: false, status: 404 });
});

// Assign mock to global fetch
global.fetch = mockFetch as unknown as typeof fetch;

describe('ViddhanaClient', () => {
  let client: ViddhanaClient;

  beforeEach(() => {
    client = new ViddhanaClient({
      network: 'testnet',
      apiKey: 'test-api-key',
    });
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('constructor', () => {
    it('should create a client with default options', () => {
      const defaultClient = new ViddhanaClient({});
      expect(defaultClient).toBeDefined();
      expect(defaultClient.atlas).toBeInstanceOf(AtlasChain);
      expect(defaultClient.vault).toBeInstanceOf(Vault);
      expect(defaultClient.ai).toBeInstanceOf(PrometheusAI);
      expect(defaultClient.governance).toBeInstanceOf(Governance);
    });

    it('should create a client with testnet configuration', () => {
      expect(client).toBeDefined();
      expect(client.getNetworkConfig().chainId).toBe(13371);
    });

    it('should throw error for unknown network', () => {
      expect(() => {
        new ViddhanaClient({ network: 'unknown' as 'mainnet' | 'testnet' });
      }).toThrow('Unknown network');
    });
  });

  describe('connect', () => {
    it('should connect successfully', async () => {
      await expect(client.connect()).resolves.toBeUndefined();
      expect(client.isConnected()).toBe(true);
    });
  });

  describe('disconnect', () => {
    it('should disconnect successfully', async () => {
      await client.connect();
      client.disconnect();
      expect(client.isConnected()).toBe(false);
    });
  });

  describe('getAddress', () => {
    it('should return null when no signer', () => {
      expect(client.getAddress()).toBeNull();
    });

    it('should return address when signer is configured', () => {
      const clientWithSigner = new ViddhanaClient({
        network: 'testnet',
        privateKey: '0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef',
      });
      expect(clientWithSigner.getAddress()).toBe('0xTestWallet...');
    });
  });

  describe('switchNetwork', () => {
    it('should return a new client for the specified network', () => {
      const newClient = client.switchNetwork('mainnet');
      expect(newClient).toBeInstanceOf(ViddhanaClient);
      expect(newClient.getNetworkConfig().chainId).toBe(13370);
    });
  });
});

describe('AtlasChain', () => {
  let client: ViddhanaClient;

  beforeEach(() => {
    client = new ViddhanaClient({ network: 'testnet' });
  });

  describe('getChainInfo', () => {
    it('should return chain information', async () => {
      const info = await client.atlas.getChainInfo();
      expect(info.chainId).toBe(13371);
      expect(info.name).toBe('VIDDHANA Atlas');
      expect(info.symbol).toBe('VDH');
      expect(info.blockNumber).toBe(12345);
    });
  });

  describe('getBalance', () => {
    it('should return balance for an address', async () => {
      const balance = await client.atlas.getBalance('0x1234...');
      expect(balance).toBe(1000000000000000000n);
    });
  });

  describe('getBlock', () => {
    it('should return block information', async () => {
      const block = await client.atlas.getBlock(12345);
      expect(block).toBeDefined();
      expect(block?.number).toBe(12345);
    });
  });

  describe('getTokenomicsStats', () => {
    it('should return tokenomics statistics', async () => {
      const stats = await client.atlas.getTokenomicsStats();
      expect(stats.totalSupply).toBe('1000000000');
    });
  });
});

describe('Vault', () => {
  let client: ViddhanaClient;

  beforeEach(() => {
    client = new ViddhanaClient({ network: 'testnet' });
  });

  describe('getPortfolio', () => {
    it('should return portfolio data', async () => {
      const portfolio = await client.vault.getPortfolio('0x1234...');
      expect(portfolio.address).toBe('0x1234...');
      expect(portfolio.totalValue).toBe('125000.00');
      expect(portfolio.assets).toHaveLength(1);
      expect(portfolio.assets[0]?.symbol).toBe('BTC');
    });
  });

  describe('getRebalanceHistory', () => {
    it('should return rebalance history', async () => {
      const history = await client.vault.getRebalanceHistory('0x1234...', { limit: 10 });
      expect(history.address).toBe('0x1234...');
      expect(history.history).toEqual([]);
      expect(history.totalRebalances).toBe(0);
    });
  });
});

describe('PrometheusAI', () => {
  let client: ViddhanaClient;

  beforeEach(() => {
    client = new ViddhanaClient({
      network: 'testnet',
      apiKey: 'test-api-key',
    });
  });

  describe('predictPrice', () => {
    it('should return price prediction', async () => {
      const prediction = await client.ai.predictPrice('BTC', 7);
      expect(prediction.asset).toBe('BTC');
      expect(prediction.trend).toBe('bullish');
      expect(prediction.predictions).toHaveLength(2);
      expect(prediction.predictions[0]?.confidence).toBe(0.82);
    });
  });

  describe('optimizePortfolio', () => {
    it('should return optimization recommendation', async () => {
      const optimization = await client.ai.optimizePortfolio({
        userId: '0x1234...',
        portfolio: { BTC: 50000, ETH: 30000 },
        riskTolerance: 0.5,
        timeToGoal: 24,
      });
      expect(optimization.action).toBe('REBALANCE');
      expect(optimization.confidence).toBe(0.85);
      expect(optimization.recommendations).toHaveLength(1);
    });
  });

  describe('assessRisk', () => {
    it('should return risk assessment', async () => {
      const assessment = await client.ai.assessRisk({ BTC: 50000, ETH: 30000 });
      expect(assessment.riskLevel).toBe('medium');
      expect(assessment.metrics.riskScore).toBe(0.5);
      expect(assessment.metrics.sharpeRatio).toBe(1.45);
      expect(assessment.recommendations).toContain('Diversify into stablecoins');
    });
  });
});

describe('Governance', () => {
  let client: ViddhanaClient;

  beforeEach(() => {
    client = new ViddhanaClient({
      network: 'testnet',
      privateKey: '0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef',
    });
  });

  describe('getVotingPower', () => {
    it('should return voting power for an address', async () => {
      const power = await client.governance.getVotingPower('0x1234...');
      expect(power).toBe(1000000000000000000000n);
    });
  });

  describe('getDelegation', () => {
    it('should return delegation info', async () => {
      const delegation = await client.governance.getDelegation('0x1234...');
      expect(delegation.delegator).toBe('0x1234...');
      expect(delegation.delegatee).toBe('0xDelegate...');
    });
  });

  describe('createProposal', () => {
    it('should throw error without signer', async () => {
      const noSignerClient = new ViddhanaClient({ network: 'testnet' });
      await expect(
        noSignerClient.governance.createProposal([], [], [], 'Test Proposal')
      ).rejects.toThrow('Signer required');
    });
  });

  describe('castVote', () => {
    it('should throw error without signer', async () => {
      const noSignerClient = new ViddhanaClient({ network: 'testnet' });
      await expect(noSignerClient.governance.castVote('123', 1)).rejects.toThrow(
        'Signer required'
      );
    });
  });

  describe('delegate', () => {
    it('should throw error without signer', async () => {
      const noSignerClient = new ViddhanaClient({ network: 'testnet' });
      await expect(noSignerClient.governance.delegate('0xDelegate...')).rejects.toThrow(
        'Signer required'
      );
    });
  });
});
