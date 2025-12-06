/**
 * Atlas Chain Module - Chain interaction methods
 */

import { ethers } from 'ethers';
import type { ChainInfo, Block, TransactionReceipt } from '../types';

/**
 * Atlas chain module for interacting with the VIDDHANA blockchain
 *
 * @example
 * ```typescript
 * const chainInfo = await client.atlas.getChainInfo();
 * console.log(`Block: ${chainInfo.blockNumber}`);
 *
 * const balance = await client.atlas.getBalance('0x...');
 * console.log(`Balance: ${ethers.formatEther(balance)} VDH`);
 * ```
 */
export class AtlasChain {
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
   * Gets chain information
   *
   * @returns Chain info including chainId, name, block number, and gas price
   */
  async getChainInfo(): Promise<ChainInfo> {
    const [network, blockNumber, feeData] = await Promise.all([
      this.provider.getNetwork(),
      this.provider.getBlockNumber(),
      this.provider.getFeeData(),
    ]);

    return {
      chainId: Number(network.chainId),
      name: 'VIDDHANA Atlas',
      symbol: 'VDH',
      blockNumber,
      gasPrice: feeData.gasPrice?.toString() ?? '0',
    };
  }

  /**
   * Gets the balance of an address
   *
   * @param address - The address to check
   * @returns Balance in wei as bigint
   */
  async getBalance(address: string): Promise<bigint> {
    return this.provider.getBalance(address);
  }

  /**
   * Gets the formatted balance in VDH
   *
   * @param address - The address to check
   * @returns Balance formatted as string with decimals
   */
  async getFormattedBalance(address: string): Promise<string> {
    const balance = await this.getBalance(address);
    return ethers.formatEther(balance);
  }

  /**
   * Gets a block by number or hash
   *
   * @param blockHashOrNumber - Block number or hash
   * @returns Block information
   */
  async getBlock(
    blockHashOrNumber: number | string = 'latest'
  ): Promise<Block | null> {
    const block = await this.provider.getBlock(blockHashOrNumber);

    if (!block) {
      return null;
    }

    return {
      number: block.number,
      hash: block.hash ?? '',
      parentHash: block.parentHash,
      timestamp: block.timestamp,
      nonce: block.nonce,
      difficulty: BigInt(0), // PoS chains don't have difficulty
      gasLimit: block.gasLimit,
      gasUsed: block.gasUsed,
      miner: block.miner,
      transactions: block.transactions.map((tx) =>
        typeof tx === 'string' ? tx : tx.hash
      ),
    };
  }

  /**
   * Gets the latest block number
   *
   * @returns Current block number
   */
  async getBlockNumber(): Promise<number> {
    return this.provider.getBlockNumber();
  }

  /**
   * Gets a transaction receipt
   *
   * @param txHash - Transaction hash
   * @returns Transaction receipt or null if not found
   */
  async getTransactionReceipt(
    txHash: string
  ): Promise<TransactionReceipt | null> {
    const receipt = await this.provider.getTransactionReceipt(txHash);

    if (!receipt) {
      return null;
    }

    return {
      transactionHash: receipt.hash,
      blockNumber: receipt.blockNumber,
      blockHash: receipt.blockHash,
      from: receipt.from,
      to: receipt.to ?? '',
      gasUsed: receipt.gasUsed,
      status: receipt.status ?? 0,
      logs: receipt.logs.map((log) => ({
        address: log.address,
        topics: log.topics as string[],
        data: log.data,
        blockNumber: log.blockNumber,
        transactionHash: log.transactionHash,
        logIndex: log.index,
      })),
    };
  }

  /**
   * Gets the current gas price
   *
   * @returns Gas price in wei
   */
  async getGasPrice(): Promise<bigint> {
    const feeData = await this.provider.getFeeData();
    return feeData.gasPrice ?? 0n;
  }

  /**
   * Estimates gas for a transaction
   *
   * @param tx - Transaction request
   * @returns Estimated gas limit
   */
  async estimateGas(tx: ethers.TransactionRequest): Promise<bigint> {
    return this.provider.estimateGas(tx);
  }

  /**
   * Gets the transaction count (nonce) for an address
   *
   * @param address - The address to check
   * @returns Transaction count
   */
  async getTransactionCount(address: string): Promise<number> {
    return this.provider.getTransactionCount(address);
  }

  /**
   * Sends a signed transaction
   *
   * @param signedTx - Signed transaction data
   * @returns Transaction response
   */
  async sendTransaction(
    signedTx: string
  ): Promise<ethers.TransactionResponse> {
    return this.provider.broadcastTransaction(signedTx);
  }

  /**
   * Waits for a transaction to be confirmed
   *
   * @param txHash - Transaction hash
   * @param confirmations - Number of confirmations to wait for
   * @returns Transaction receipt
   */
  async waitForTransaction(
    txHash: string,
    confirmations = 1
  ): Promise<TransactionReceipt | null> {
    const receipt = await this.provider.waitForTransaction(
      txHash,
      confirmations
    );

    if (!receipt) {
      return null;
    }

    return {
      transactionHash: receipt.hash,
      blockNumber: receipt.blockNumber,
      blockHash: receipt.blockHash,
      from: receipt.from,
      to: receipt.to ?? '',
      gasUsed: receipt.gasUsed,
      status: receipt.status ?? 0,
      logs: receipt.logs.map((log) => ({
        address: log.address,
        topics: log.topics as string[],
        data: log.data,
        blockNumber: log.blockNumber,
        transactionHash: log.transactionHash,
        logIndex: log.index,
      })),
    };
  }

  /**
   * Gets tokenomics statistics from the chain
   *
   * @returns Tokenomics statistics
   */
  async getTokenomicsStats(): Promise<Record<string, unknown>> {
    const result = await this.provider.send('vdh_getTokenomicsStats', []);
    return result as Record<string, unknown>;
  }

  /**
   * Subscribes to new blocks
   *
   * @param callback - Function to call on each new block
   * @returns Unsubscribe function
   */
  onBlock(callback: (blockNumber: number) => void): () => void {
    this.provider.on('block', callback);
    return () => this.provider.off('block', callback);
  }
}
