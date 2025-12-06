/**
 * Blockchain Service
 * 
 * Ethers.js wrapper for interacting with the VIDDHANA Atlas chain
 */

import { ethers, JsonRpcProvider, TransactionRequest, Block, TransactionReceipt } from 'ethers';
import { logger } from './logger';

export class BlockchainService {
  private provider: JsonRpcProvider;
  private chainId: number;

  constructor() {
    const rpcUrl = process.env.RPC_URL || 'http://localhost:8545';
    this.chainId = parseInt(process.env.CHAIN_ID || '1337', 10);
    
    // Don't specify static network - let ethers.js auto-detect from the node
    // This prevents "network changed" errors when chain ID doesn't match
    this.provider = new JsonRpcProvider(rpcUrl);

    logger.info(`Blockchain service initialized with RPC: ${rpcUrl}`);
  }

  /**
   * Get the chain ID
   */
  getChainId(): number {
    return this.chainId;
  }

  /**
   * Get current block number
   */
  async getBlockNumber(): Promise<number> {
    return this.provider.getBlockNumber();
  }

  /**
   * Get current gas price
   */
  async getGasPrice(): Promise<bigint> {
    const feeData = await this.provider.getFeeData();
    return feeData.gasPrice || BigInt(0);
  }

  /**
   * Get account balance
   */
  async getBalance(
    address: string,
    blockTag: number | string = 'latest'
  ): Promise<bigint> {
    return this.provider.getBalance(address, blockTag);
  }

  /**
   * Get transaction count (nonce)
   */
  async getTransactionCount(
    address: string,
    blockTag: number | string = 'latest'
  ): Promise<number> {
    return this.provider.getTransactionCount(address, blockTag);
  }

  /**
   * Get block by number
   */
  async getBlockByNumber(
    blockNumber: number | string,
    includeTransactions: boolean = false
  ): Promise<Block | null> {
    return this.provider.getBlock(blockNumber, includeTransactions);
  }

  /**
   * Get block by hash
   */
  async getBlockByHash(
    blockHash: string,
    includeTransactions: boolean = false
  ): Promise<Block | null> {
    return this.provider.getBlock(blockHash, includeTransactions);
  }

  /**
   * Get transaction receipt
   */
  async getTransactionReceipt(txHash: string): Promise<TransactionReceipt | null> {
    return this.provider.getTransactionReceipt(txHash);
  }

  /**
   * Send raw transaction
   */
  async sendRawTransaction(signedTx: string): Promise<string> {
    const response = await this.provider.broadcastTransaction(signedTx);
    return response.hash;
  }

  /**
   * Call a contract method (read-only)
   */
  async call(
    transaction: TransactionRequest,
    blockTag: number | string = 'latest'
  ): Promise<string> {
    return this.provider.call({ ...transaction }, blockTag);
  }

  /**
   * Estimate gas for a transaction
   */
  async estimateGas(transaction: TransactionRequest): Promise<bigint> {
    return this.provider.estimateGas(transaction);
  }

  /**
   * Get logs
   */
  async getLogs(filter: ethers.Filter): Promise<ethers.Log[]> {
    return this.provider.getLogs(filter);
  }

  /**
   * Format wei to ether
   */
  formatEther(wei: bigint): string {
    return ethers.formatEther(wei);
  }

  /**
   * Parse ether to wei
   */
  parseEther(ether: string): bigint {
    return ethers.parseEther(ether);
  }

  /**
   * Format units
   */
  formatUnits(value: bigint, decimals: number): string {
    return ethers.formatUnits(value, decimals);
  }

  /**
   * Parse units
   */
  parseUnits(value: string, decimals: number): bigint {
    return ethers.parseUnits(value, decimals);
  }

  /**
   * Check if address is valid
   */
  isValidAddress(address: string): boolean {
    return ethers.isAddress(address);
  }

  /**
   * Get checksum address
   */
  getChecksumAddress(address: string): string {
    return ethers.getAddress(address);
  }

  /**
   * Create contract instance
   */
  getContract(
    address: string,
    abi: ethers.InterfaceAbi,
    signerOrProvider?: ethers.Signer | ethers.Provider
  ): ethers.Contract {
    return new ethers.Contract(
      address,
      abi,
      signerOrProvider || this.provider
    );
  }

  /**
   * Get provider
   */
  getProvider(): JsonRpcProvider {
    return this.provider;
  }

  /**
   * Create wallet from private key
   */
  createWallet(privateKey: string): ethers.Wallet {
    return new ethers.Wallet(privateKey, this.provider);
  }

  /**
   * Subscribe to new blocks
   */
  onBlock(callback: (blockNumber: number) => void): void {
    this.provider.on('block', callback);
  }

  /**
   * Unsubscribe from new blocks
   */
  offBlock(callback: (blockNumber: number) => void): void {
    this.provider.off('block', callback);
  }

  /**
   * Check connection
   */
  async isConnected(): Promise<boolean> {
    try {
      await this.provider.getNetwork();
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Get network info
   */
  async getNetwork(): Promise<ethers.Network> {
    return this.provider.getNetwork();
  }

  /**
   * Wait for transaction confirmation
   */
  async waitForTransaction(
    txHash: string,
    confirmations: number = 1
  ): Promise<TransactionReceipt | null> {
    return this.provider.waitForTransaction(txHash, confirmations);
  }
}

// Singleton instance
let blockchainServiceInstance: BlockchainService | null = null;

export function getBlockchainService(): BlockchainService {
  if (!blockchainServiceInstance) {
    blockchainServiceInstance = new BlockchainService();
  }
  return blockchainServiceInstance;
}
