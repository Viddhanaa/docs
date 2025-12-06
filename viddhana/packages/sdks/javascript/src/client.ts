/**
 * VIDDHANA Client - Main entry point for the SDK
 */

import { ethers } from 'ethers';
import { AtlasChain } from './modules/atlas';
import { Vault } from './modules/vault';
import { PrometheusAI } from './modules/ai';
import { Governance } from './modules/governance';
import type { ViddhanaConfig, NetworkConfig } from './types';

/**
 * Network configurations for VIDDHANA
 */
const NETWORKS: Record<string, NetworkConfig> = {
  mainnet: {
    chainId: 13370,
    rpcUrl: 'https://rpc.viddhana.network',
    wsUrl: 'wss://ws.viddhana.network',
    apiUrl: 'https://api.viddhana.network',
  },
  testnet: {
    chainId: 13371,
    rpcUrl: 'https://rpc.testnet.viddhana.network',
    wsUrl: 'wss://ws.testnet.viddhana.network',
    apiUrl: 'https://api.testnet.viddhana.network',
  },
};

/**
 * Main VIDDHANA client for interacting with the network
 *
 * @example
 * ```typescript
 * const client = new ViddhanaClient({
 *   network: 'testnet',
 *   privateKey: process.env.PRIVATE_KEY,
 *   apiKey: process.env.API_KEY,
 * });
 *
 * await client.connect();
 * const portfolio = await client.vault.getPortfolio('0x...');
 * ```
 */
export class ViddhanaClient {
  /** Atlas chain module for chain operations */
  public readonly atlas: AtlasChain;

  /** Vault module for portfolio and vault operations */
  public readonly vault: Vault;

  /** Prometheus AI module for predictions and optimization */
  public readonly ai: PrometheusAI;

  /** Governance module for proposals and voting */
  public readonly governance: Governance;

  private readonly provider: ethers.JsonRpcProvider;
  private signer: ethers.Wallet | null = null;
  private readonly config: ViddhanaConfig;
  private readonly networkConfig: NetworkConfig;
  private connected = false;

  /**
   * Creates a new ViddhanaClient instance
   *
   * @param config - Client configuration options
   */
  constructor(config: ViddhanaConfig) {
    this.config = config;

    // Get network configuration
    const networkName = config.network ?? 'mainnet';
    const network = NETWORKS[networkName];
    if (!network) {
      throw new Error(`Unknown network: ${networkName}`);
    }
    this.networkConfig = network;

    // Initialize provider
    const rpcUrl = config.rpcUrl ?? network.rpcUrl;
    this.provider = new ethers.JsonRpcProvider(rpcUrl);

    // Initialize signer if private key provided
    if (config.privateKey) {
      this.signer = new ethers.Wallet(config.privateKey, this.provider);
    }

    // Initialize modules
    this.atlas = new AtlasChain(this.provider, this.signer);
    this.vault = new Vault(this.provider, this.signer, network.apiUrl);
    this.ai = new PrometheusAI(network.apiUrl, config.apiKey);
    this.governance = new Governance(this.provider, this.signer);
  }

  /**
   * Connects to the VIDDHANA network
   *
   * @returns Promise that resolves when connected
   * @throws Error if connection fails
   */
  async connect(): Promise<void> {
    try {
      // Verify connection by getting network info
      const network = await this.provider.getNetwork();
      const expectedChainId = BigInt(this.networkConfig.chainId);

      if (network.chainId !== expectedChainId) {
        throw new Error(
          `Chain ID mismatch. Expected ${expectedChainId}, got ${network.chainId}`
        );
      }

      this.connected = true;
    } catch (error) {
      this.connected = false;
      throw new Error(
        `Failed to connect: ${error instanceof Error ? error.message : 'Unknown error'}`
      );
    }
  }

  /**
   * Disconnects from the VIDDHANA network
   */
  disconnect(): void {
    this.connected = false;
    this.provider.destroy();
  }

  /**
   * Checks if the client is connected
   *
   * @returns True if connected
   */
  isConnected(): boolean {
    return this.connected;
  }

  /**
   * Gets the JSON-RPC provider
   *
   * @returns The ethers provider instance
   */
  getProvider(): ethers.JsonRpcProvider {
    return this.provider;
  }

  /**
   * Gets the signer for signing transactions
   *
   * @returns The wallet signer or null if not configured
   */
  getSigner(): ethers.Wallet | null {
    return this.signer;
  }

  /**
   * Gets the connected wallet address
   *
   * @returns The wallet address or null if no signer
   */
  getAddress(): string | null {
    return this.signer?.address ?? null;
  }

  /**
   * Gets the current network configuration
   *
   * @returns Network configuration object
   */
  getNetworkConfig(): NetworkConfig {
    return this.networkConfig;
  }

  /**
   * Connects with a browser wallet (MetaMask, etc.)
   *
   * @param browserProvider - ethers BrowserProvider from wallet
   * @returns The connected signer
   */
  async connectWallet(
    browserProvider: ethers.BrowserProvider
  ): Promise<ethers.Signer> {
    const signer = await browserProvider.getSigner();
    const network = await browserProvider.getNetwork();

    if (network.chainId !== BigInt(this.networkConfig.chainId)) {
      throw new Error(
        `Please switch to VIDDHANA network (Chain ID: ${this.networkConfig.chainId})`
      );
    }

    this.connected = true;
    return signer;
  }

  /**
   * Switches the network
   *
   * @param network - Network name ('mainnet' or 'testnet')
   * @returns New ViddhanaClient instance for the network
   */
  switchNetwork(network: 'mainnet' | 'testnet'): ViddhanaClient {
    return new ViddhanaClient({
      ...this.config,
      network,
    });
  }
}
