/**
 * WebSocket Manager
 * 
 * Manages Socket.IO connections and subscription channels
 */

import { Server as SocketServer, Socket } from 'socket.io';
import { z } from 'zod';
import { logger } from '../services/logger';
import { WebSocketRateLimiter } from '../middleware/rateLimit';
import { WebSocketChannel, WebSocketSubscription, WebSocketMessage, ApiTier } from '../types';
import { BlockchainService } from '../services/blockchain';

const SubscriptionSchema = z.object({
  type: z.enum(['subscribe', 'unsubscribe']),
  channel: z.enum(['portfolio', 'predictions', 'rebalance', 'rewards', 'blocks', 'prices']),
  params: z.record(z.unknown()).optional(),
});

interface ClientInfo {
  id: string;
  tier: ApiTier;
  subscriptions: Set<string>;
  address?: string;
  connectedAt: Date;
}

export class WebSocketManager {
  private io: SocketServer;
  private clients: Map<string, ClientInfo> = new Map();
  private channelSubscribers: Map<string, Set<string>> = new Map();
  private rateLimiter: WebSocketRateLimiter;
  private blockchainService: BlockchainService;

  constructor(io: SocketServer) {
    this.io = io;
    this.rateLimiter = new WebSocketRateLimiter(1000, 50);
    this.blockchainService = new BlockchainService();

    this.setupEventHandlers();
    this.startBlockSubscription();

    logger.info('WebSocket manager initialized');
  }

  /**
   * Setup Socket.IO event handlers
   */
  private setupEventHandlers(): void {
    this.io.on('connection', (socket: Socket) => {
      this.handleConnection(socket);
    });
  }

  /**
   * Handle new client connection
   */
  private handleConnection(socket: Socket): void {
    const clientId = socket.id;
    
    // Extract API key from handshake
    const apiKey = socket.handshake.auth.apiKey || socket.handshake.query.api_key;
    const tier = this.getTierFromApiKey(apiKey as string);

    // Store client info
    this.clients.set(clientId, {
      id: clientId,
      tier,
      subscriptions: new Set(),
      connectedAt: new Date(),
    });

    logger.info(`WebSocket client connected: ${clientId}, tier: ${tier}`);

    // Send welcome message
    socket.emit('connected', {
      id: clientId,
      tier,
      timestamp: Date.now(),
    });

    // Handle subscription requests
    socket.on('subscribe', (data: unknown) => {
      this.handleSubscribe(socket, data);
    });

    socket.on('unsubscribe', (data: unknown) => {
      this.handleUnsubscribe(socket, data);
    });

    // Handle ping/pong
    socket.on('ping', () => {
      socket.emit('pong', { timestamp: Date.now() });
    });

    // Handle disconnection
    socket.on('disconnect', (reason) => {
      this.handleDisconnect(clientId, reason);
    });

    // Handle errors
    socket.on('error', (error) => {
      logger.error(`WebSocket error for ${clientId}:`, error);
    });
  }

  /**
   * Handle subscription request
   */
  private handleSubscribe(socket: Socket, data: unknown): void {
    const clientId = socket.id;
    const client = this.clients.get(clientId);

    if (!client) {
      socket.emit('error', { message: 'Client not found' });
      return;
    }

    // Rate limiting
    if (this.rateLimiter.isRateLimited(clientId, client.tier)) {
      socket.emit('error', { 
        code: 'RATE_LIMITED',
        message: 'Too many requests. Please slow down.',
      });
      return;
    }

    try {
      const subscription = SubscriptionSchema.parse(data);
      const { channel, params } = subscription;

      // Create subscription key
      const subscriptionKey = this.createSubscriptionKey(channel, params);

      // Check subscription limits by tier
      const maxSubscriptions = this.getMaxSubscriptions(client.tier);
      if (client.subscriptions.size >= maxSubscriptions) {
        socket.emit('error', {
          code: 'MAX_SUBSCRIPTIONS',
          message: `Maximum ${maxSubscriptions} subscriptions for ${client.tier} tier`,
        });
        return;
      }

      // Add to client subscriptions
      client.subscriptions.add(subscriptionKey);

      // Add to channel subscribers
      if (!this.channelSubscribers.has(subscriptionKey)) {
        this.channelSubscribers.set(subscriptionKey, new Set());
      }
      this.channelSubscribers.get(subscriptionKey)!.add(clientId);

      // Join Socket.IO room
      socket.join(subscriptionKey);

      // Store address if provided
      if (params?.address && typeof params.address === 'string') {
        client.address = params.address;
      }

      logger.info(`Client ${clientId} subscribed to ${subscriptionKey}`);

      socket.emit('subscribed', {
        channel,
        params,
        subscriptionKey,
        timestamp: Date.now(),
      });

    } catch (error) {
      if (error instanceof z.ZodError) {
        socket.emit('error', {
          code: 'INVALID_SUBSCRIPTION',
          message: 'Invalid subscription format',
          details: error.errors,
        });
      } else {
        logger.error(`Subscription error for ${clientId}:`, error);
        socket.emit('error', { message: 'Failed to subscribe' });
      }
    }
  }

  /**
   * Handle unsubscription request
   */
  private handleUnsubscribe(socket: Socket, data: unknown): void {
    const clientId = socket.id;
    const client = this.clients.get(clientId);

    if (!client) return;

    try {
      const subscription = SubscriptionSchema.parse(data);
      const { channel, params } = subscription;
      const subscriptionKey = this.createSubscriptionKey(channel, params);

      // Remove from client subscriptions
      client.subscriptions.delete(subscriptionKey);

      // Remove from channel subscribers
      const subscribers = this.channelSubscribers.get(subscriptionKey);
      if (subscribers) {
        subscribers.delete(clientId);
        if (subscribers.size === 0) {
          this.channelSubscribers.delete(subscriptionKey);
        }
      }

      // Leave Socket.IO room
      socket.leave(subscriptionKey);

      logger.info(`Client ${clientId} unsubscribed from ${subscriptionKey}`);

      socket.emit('unsubscribed', {
        channel,
        params,
        subscriptionKey,
        timestamp: Date.now(),
      });

    } catch (error) {
      logger.error(`Unsubscription error for ${clientId}:`, error);
    }
  }

  /**
   * Handle client disconnection
   */
  private handleDisconnect(clientId: string, reason: string): void {
    const client = this.clients.get(clientId);

    if (client) {
      // Remove from all channel subscribers
      for (const subscriptionKey of client.subscriptions) {
        const subscribers = this.channelSubscribers.get(subscriptionKey);
        if (subscribers) {
          subscribers.delete(clientId);
          if (subscribers.size === 0) {
            this.channelSubscribers.delete(subscriptionKey);
          }
        }
      }

      this.clients.delete(clientId);
      this.rateLimiter.remove(clientId);
    }

    logger.info(`WebSocket client disconnected: ${clientId}, reason: ${reason}`);
  }

  /**
   * Broadcast message to a channel
   */
  broadcast<T>(channel: WebSocketChannel, payload: T, filter?: Record<string, unknown>): void {
    const subscriptionKey = this.createSubscriptionKey(channel, filter);

    const message: WebSocketMessage<T> = {
      channel,
      payload,
      timestamp: Date.now(),
    };

    this.io.to(subscriptionKey).emit('message', message);
  }

  /**
   * Send message to specific client
   */
  sendToClient<T>(clientId: string, channel: WebSocketChannel, payload: T): void {
    const socket = this.io.sockets.sockets.get(clientId);

    if (socket) {
      const message: WebSocketMessage<T> = {
        channel,
        payload,
        timestamp: Date.now(),
      };
      socket.emit('message', message);
    }
  }

  /**
   * Start block subscription for broadcasting
   */
  private startBlockSubscription(): void {
    this.blockchainService.onBlock(async (blockNumber) => {
      try {
        const block = await this.blockchainService.getBlockByNumber(blockNumber, false);
        
        if (block) {
          this.broadcast('blocks', {
            number: block.number,
            hash: block.hash,
            timestamp: block.timestamp,
            transactionCount: block.transactions.length,
          });
        }
      } catch (error) {
        logger.error('Error broadcasting block:', error);
      }
    });
  }

  /**
   * Create subscription key from channel and params
   */
  private createSubscriptionKey(channel: WebSocketChannel, params?: Record<string, unknown>): string {
    if (!params || Object.keys(params).length === 0) {
      return channel;
    }

    // Sort params for consistent key generation
    const sortedParams = Object.entries(params)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([k, v]) => `${k}=${v}`)
      .join(':');

    return `${channel}:${sortedParams}`;
  }

  /**
   * Get API tier from API key (simplified)
   */
  private getTierFromApiKey(apiKey?: string): ApiTier {
    if (!apiKey) return 'free';
    
    if (apiKey.includes('enterprise')) return 'enterprise';
    if (apiKey.includes('pro')) return 'pro';
    if (apiKey.includes('basic')) return 'basic';
    
    return 'free';
  }

  /**
   * Get max subscriptions for tier
   */
  private getMaxSubscriptions(tier: ApiTier): number {
    const limits: Record<ApiTier, number> = {
      free: 5,
      basic: 20,
      pro: 100,
      enterprise: 1000,
    };
    return limits[tier];
  }

  /**
   * Get connected clients count
   */
  getClientCount(): number {
    return this.clients.size;
  }

  /**
   * Get subscription stats
   */
  getStats(): {
    clients: number;
    subscriptions: number;
    channels: Record<string, number>;
  } {
    const channels: Record<string, number> = {};
    
    for (const [key, subscribers] of this.channelSubscribers.entries()) {
      channels[key] = subscribers.size;
    }

    return {
      clients: this.clients.size,
      subscriptions: Array.from(this.clients.values())
        .reduce((sum, client) => sum + client.subscriptions.size, 0),
      channels,
    };
  }

  /**
   * Emit price updates (called from external source)
   */
  emitPriceUpdate(asset: string, price: number, change24h: number): void {
    this.broadcast('prices', { asset, price, change24h }, { asset });
  }

  /**
   * Emit portfolio update
   */
  emitPortfolioUpdate(address: string, portfolio: object): void {
    this.broadcast('portfolio', portfolio, { address: address.toLowerCase() });
  }

  /**
   * Emit prediction update
   */
  emitPredictionUpdate(asset: string, prediction: object): void {
    this.broadcast('predictions', prediction, { asset });
  }

  /**
   * Emit rebalance event
   */
  emitRebalanceEvent(address: string, rebalance: object): void {
    this.broadcast('rebalance', rebalance, { address: address.toLowerCase() });
  }

  /**
   * Emit rewards update
   */
  emitRewardsUpdate(sensorId: string, rewards: object): void {
    this.broadcast('rewards', rewards, { sensorId });
  }
}
