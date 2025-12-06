/**
 * Vault Module - Portfolio and vault operations
 */

import { ethers } from 'ethers';
import type {
  PortfolioData,
  VaultInfo,
  RebalanceHistory,
  RebalanceAction,
} from '../types';

// Contract ABIs (minimal interfaces)
const ERC20_ABI = [
  'function approve(address spender, uint256 amount) returns (bool)',
  'function balanceOf(address owner) view returns (uint256)',
  'function allowance(address owner, address spender) view returns (uint256)',
];

const VAULT_MANAGER_ABI = [
  'function deposit(address asset, uint256 amount) returns (uint256)',
  'function withdraw(address asset, uint256 amount) returns (uint256)',
  'function createVault(string name, string symbol, address[] assets, uint256[] allocations) returns (address)',
  'function getVaultInfo(address vault) view returns (tuple(address, string, string, uint256, uint256, uint256, string))',
  'function getUserVaults(address user) view returns (address[])',
];

const POLICY_ENGINE_ABI = [
  'function executeManualRebalance(tuple(address asset, uint8 action, uint256 percentage, uint256 amount, uint256 valueUSD)[] actions) returns (bool)',
  'function setUserProfile(uint256 riskTolerance, uint256 timeToGoal, bool autoRebalance) returns (bool)',
  'function getUserProfile(address user) view returns (tuple(uint256, uint256, bool))',
];

// Contract addresses (to be configured per network)
const CONTRACT_ADDRESSES = {
  mainnet: {
    vaultManager: '0x0000000000000000000000000000000000000001',
    policyEngine: '0x0000000000000000000000000000000000000002',
  },
  testnet: {
    vaultManager: '0x0000000000000000000000000000000000000001',
    policyEngine: '0x0000000000000000000000000000000000000002',
  },
};

/**
 * Vault module for portfolio management and vault operations
 *
 * @example
 * ```typescript
 * // Get portfolio
 * const portfolio = await client.vault.getPortfolio('0x...');
 * console.log(`Total Value: $${portfolio.totalValue}`);
 *
 * // Deposit to vault
 * const tx = await client.vault.deposit('0xUSDC...', ethers.parseUnits('1000', 6));
 * await tx.wait();
 * ```
 */
export class Vault {
  private readonly provider: ethers.JsonRpcProvider;
  private readonly signer: ethers.Wallet | null;
  private readonly apiUrl: string;

  constructor(
    provider: ethers.JsonRpcProvider,
    signer: ethers.Wallet | null,
    apiUrl: string
  ) {
    this.provider = provider;
    this.signer = signer;
    this.apiUrl = apiUrl;
  }

  /**
   * Gets portfolio data for an address
   *
   * @param address - The address to get portfolio for
   * @returns Portfolio data including assets, values, and metrics
   */
  async getPortfolio(address: string): Promise<PortfolioData> {
    const result = await this.provider.send('vdh_getPortfolio', [address]);
    return result as PortfolioData;
  }

  /**
   * Gets rebalancing history for an address
   *
   * @param address - The address to get history for
   * @param options - Optional filter options
   * @returns Rebalance history with transaction details
   */
  async getRebalanceHistory(
    address: string,
    options?: {
      fromBlock?: number;
      toBlock?: number | 'latest';
      limit?: number;
    }
  ): Promise<RebalanceHistory> {
    const result = await this.provider.send('vdh_getRebalanceHistory', [
      address,
      options ?? {},
    ]);
    return result as RebalanceHistory;
  }

  /**
   * Creates a new vault
   *
   * @param name - Vault name
   * @param symbol - Vault token symbol
   * @param assets - Array of asset addresses
   * @param allocations - Array of allocation percentages (basis points, sum = 10000)
   * @returns Transaction response
   */
  async createVault(
    name: string,
    symbol: string,
    assets: string[],
    allocations: number[]
  ): Promise<ethers.TransactionResponse> {
    this.requireSigner();

    const vaultManager = new ethers.Contract(
      CONTRACT_ADDRESSES.mainnet.vaultManager,
      VAULT_MANAGER_ABI,
      this.signer!
    );

    const tx = await vaultManager.createVault(name, symbol, assets, allocations);
    return tx as ethers.TransactionResponse;
  }

  /**
   * Deposits assets into a vault
   *
   * @param asset - Asset token address
   * @param amount - Amount to deposit (in wei/smallest unit)
   * @returns Transaction response
   */
  async deposit(
    asset: string,
    amount: bigint
  ): Promise<ethers.TransactionResponse> {
    this.requireSigner();

    // First approve the token transfer
    const token = new ethers.Contract(asset, ERC20_ABI, this.signer!);
    const approveTx = await token.approve(
      CONTRACT_ADDRESSES.mainnet.vaultManager,
      amount
    );
    await approveTx.wait();

    // Then execute deposit
    const vaultManager = new ethers.Contract(
      CONTRACT_ADDRESSES.mainnet.vaultManager,
      VAULT_MANAGER_ABI,
      this.signer!
    );

    const tx = await vaultManager.deposit(asset, amount);
    return tx as ethers.TransactionResponse;
  }

  /**
   * Withdraws assets from a vault
   *
   * @param asset - Asset token address
   * @param amount - Amount to withdraw (in wei/smallest unit)
   * @returns Transaction response
   */
  async withdraw(
    asset: string,
    amount: bigint
  ): Promise<ethers.TransactionResponse> {
    this.requireSigner();

    const vaultManager = new ethers.Contract(
      CONTRACT_ADDRESSES.mainnet.vaultManager,
      VAULT_MANAGER_ABI,
      this.signer!
    );

    const tx = await vaultManager.withdraw(asset, amount);
    return tx as ethers.TransactionResponse;
  }

  /**
   * Gets vault information
   *
   * @param vaultAddress - Vault contract address
   * @returns Vault information
   */
  async getVaultInfo(vaultAddress: string): Promise<VaultInfo> {
    const vaultManager = new ethers.Contract(
      CONTRACT_ADDRESSES.mainnet.vaultManager,
      VAULT_MANAGER_ABI,
      this.provider
    );

    const info = await vaultManager.getVaultInfo(vaultAddress);
    return {
      address: info[0] as string,
      name: info[1] as string,
      symbol: info[2] as string,
      totalAssets: info[3].toString(),
      totalShares: info[4].toString(),
      apy: Number(info[5]) / 100,
      strategy: info[6] as string,
    };
  }

  /**
   * Gets all vaults for a user
   *
   * @param userAddress - User's address
   * @returns Array of vault addresses
   */
  async getUserVaults(userAddress: string): Promise<string[]> {
    const vaultManager = new ethers.Contract(
      CONTRACT_ADDRESSES.mainnet.vaultManager,
      VAULT_MANAGER_ABI,
      this.provider
    );

    return vaultManager.getUserVaults(userAddress) as Promise<string[]>;
  }

  /**
   * Executes a manual rebalance
   *
   * @param actions - Array of rebalance actions
   * @returns Transaction response
   */
  async executeRebalance(
    actions: RebalanceAction[]
  ): Promise<ethers.TransactionResponse> {
    this.requireSigner();

    const policyEngine = new ethers.Contract(
      CONTRACT_ADDRESSES.mainnet.policyEngine,
      POLICY_ENGINE_ABI,
      this.signer!
    );

    const formattedActions = actions.map((action) => ({
      asset: action.asset,
      action: action.action === 'BUY' ? 0 : 1,
      percentage: action.percentage,
      amount: ethers.parseEther(action.amount),
      valueUSD: ethers.parseUnits(action.valueUSD, 6),
    }));

    const tx = await policyEngine.executeManualRebalance(formattedActions);
    return tx as ethers.TransactionResponse;
  }

  /**
   * Sets user profile for auto-rebalancing
   *
   * @param params - Profile parameters
   * @returns Transaction response
   */
  async setProfile(params: {
    riskTolerance: number; // 0-10000 basis points
    timeToGoal: number; // months
    autoRebalance: boolean;
  }): Promise<ethers.TransactionResponse> {
    this.requireSigner();

    const policyEngine = new ethers.Contract(
      CONTRACT_ADDRESSES.mainnet.policyEngine,
      POLICY_ENGINE_ABI,
      this.signer!
    );

    const tx = await policyEngine.setUserProfile(
      params.riskTolerance,
      params.timeToGoal,
      params.autoRebalance
    );
    return tx as ethers.TransactionResponse;
  }

  /**
   * Gets user profile settings
   *
   * @param address - User address
   * @returns User profile
   */
  async getProfile(address: string): Promise<{
    riskTolerance: number;
    timeToGoal: number;
    autoRebalance: boolean;
  }> {
    const policyEngine = new ethers.Contract(
      CONTRACT_ADDRESSES.mainnet.policyEngine,
      POLICY_ENGINE_ABI,
      this.provider
    );

    const profile = await policyEngine.getUserProfile(address);
    return {
      riskTolerance: Number(profile[0]),
      timeToGoal: Number(profile[1]),
      autoRebalance: profile[2] as boolean,
    };
  }

  /**
   * Gets AI optimization recommendation
   *
   * @param address - User address
   * @returns Optimization recommendation
   */
  async getOptimizationRecommendation(address: string): Promise<{
    actions: RebalanceAction[];
    confidence: number;
    reason: string;
  }> {
    const portfolio = await this.getPortfolio(address);

    const response = await fetch(`${this.apiUrl}/v1/optimize/portfolio`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        user_id: address,
        portfolio,
      }),
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    return response.json() as Promise<{
      actions: RebalanceAction[];
      confidence: number;
      reason: string;
    }>;
  }

  /**
   * Checks token allowance
   *
   * @param tokenAddress - Token contract address
   * @param ownerAddress - Token owner address
   * @param spenderAddress - Spender address (defaults to vault manager)
   * @returns Allowance amount
   */
  async getAllowance(
    tokenAddress: string,
    ownerAddress: string,
    spenderAddress?: string
  ): Promise<bigint> {
    const token = new ethers.Contract(tokenAddress, ERC20_ABI, this.provider);
    return token.allowance(
      ownerAddress,
      spenderAddress ?? CONTRACT_ADDRESSES.mainnet.vaultManager
    ) as Promise<bigint>;
  }

  /**
   * Gets token balance
   *
   * @param tokenAddress - Token contract address
   * @param ownerAddress - Token owner address
   * @returns Token balance
   */
  async getTokenBalance(
    tokenAddress: string,
    ownerAddress: string
  ): Promise<bigint> {
    const token = new ethers.Contract(tokenAddress, ERC20_ABI, this.provider);
    return token.balanceOf(ownerAddress) as Promise<bigint>;
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
