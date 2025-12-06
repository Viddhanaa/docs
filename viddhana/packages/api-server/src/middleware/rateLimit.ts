/**
 * Rate Limiting Middleware
 * 
 * Implements tiered rate limiting based on API key tier
 */

import { Request, Response, NextFunction } from 'express';
import rateLimit, { RateLimitRequestHandler } from 'express-rate-limit';
import { AuthenticatedRequest, ApiTier, RATE_LIMITS } from '../types';
import { logger } from '../services/logger';

// In-memory store for development; use Redis in production
interface RateLimitStore {
  [key: string]: {
    count: number;
    resetTime: number;
  };
}

const memoryStore: RateLimitStore = {};

/**
 * Get rate limit config for a tier
 */
function getRateLimitForTier(tier: ApiTier): { windowMs: number; max: number; message: string } {
  const config = RATE_LIMITS[tier];
  return {
    windowMs: config.windowMs,
    max: config.maxRequests,
    message: config.message,
  };
}

/**
 * Custom key generator based on API key or IP
 */
function keyGenerator(req: Request): string {
  const authReq = req as AuthenticatedRequest;
  
  if (authReq.apiKey?.id && authReq.apiKey.id !== 'anonymous') {
    return `api:${authReq.apiKey.id}`;
  }
  
  // Fall back to IP address
  const ip = req.ip || req.socket.remoteAddress || 'unknown';
  return `ip:${ip}`;
}

/**
 * Custom skip function for enterprise tier
 */
function skip(req: Request): boolean {
  const authReq = req as AuthenticatedRequest;
  return authReq.apiKey?.tier === 'enterprise';
}

/**
 * Create the main rate limiter
 */
export function createRateLimiter(): RateLimitRequestHandler {
  return rateLimit({
    windowMs: 60 * 1000, // 1 minute window
    max: (req: Request): number => {
      const authReq = req as AuthenticatedRequest;
      const tier = authReq.apiKey?.tier || 'free';
      return RATE_LIMITS[tier].maxRequests;
    },
    keyGenerator,
    skip,
    standardHeaders: true,
    legacyHeaders: false,
    message: (req: Request): object => {
      const authReq = req as AuthenticatedRequest;
      const tier = authReq.apiKey?.tier || 'free';
      return {
        error: {
          code: -32001,
          message: RATE_LIMITS[tier].message,
        },
      };
    },
    handler: (req: Request, res: Response, _next: NextFunction, options: any) => {
      logger.warn(`Rate limit exceeded for ${keyGenerator(req)}`);
      res.status(429).json(options.message);
    },
  });
}

/**
 * Create a stricter rate limiter for specific endpoints
 */
export function createStrictRateLimiter(
  windowMs: number = 60 * 1000,
  maxRequests: number = 10
): RateLimitRequestHandler {
  return rateLimit({
    windowMs,
    max: maxRequests,
    keyGenerator,
    skip,
    standardHeaders: true,
    legacyHeaders: false,
    message: {
      error: {
        code: -32001,
        message: 'Too many requests to this endpoint. Please try again later.',
      },
    },
  });
}

/**
 * Dynamic rate limiter that adjusts based on tier
 */
export function dynamicRateLimiter() {
  return (req: Request, res: Response, next: NextFunction): void => {
    const authReq = req as AuthenticatedRequest;
    const tier = authReq.apiKey?.tier || 'free';
    const key = keyGenerator(req);
    const now = Date.now();

    // Skip for enterprise
    if (tier === 'enterprise') {
      return next();
    }

    const config = getRateLimitForTier(tier);
    const stored = memoryStore[key];

    // Initialize or reset if window expired
    if (!stored || now > stored.resetTime) {
      memoryStore[key] = {
        count: 1,
        resetTime: now + config.windowMs,
      };
      
      res.setHeader('X-RateLimit-Limit', config.max);
      res.setHeader('X-RateLimit-Remaining', config.max - 1);
      res.setHeader('X-RateLimit-Reset', Math.ceil((now + config.windowMs) / 1000));
      
      return next();
    }

    // Increment counter
    stored.count++;

    // Set rate limit headers
    res.setHeader('X-RateLimit-Limit', config.max);
    res.setHeader('X-RateLimit-Remaining', Math.max(0, config.max - stored.count));
    res.setHeader('X-RateLimit-Reset', Math.ceil(stored.resetTime / 1000));

    // Check if over limit
    if (stored.count > config.max) {
      logger.warn(`Rate limit exceeded for ${key}, tier: ${tier}`);
      
      res.status(429).json({
        error: {
          code: -32001,
          message: config.message,
        },
      });
      return;
    }

    next();
  };
}

/**
 * Endpoint-specific rate limiter factory
 */
export function endpointRateLimiter(
  endpoint: string,
  limits: Record<ApiTier, number>
): (req: Request, res: Response, next: NextFunction) => void {
  const endpointStore: RateLimitStore = {};

  return (req: Request, res: Response, next: NextFunction): void => {
    const authReq = req as AuthenticatedRequest;
    const tier = authReq.apiKey?.tier || 'free';
    const baseKey = keyGenerator(req);
    const key = `${baseKey}:${endpoint}`;
    const now = Date.now();
    const windowMs = 60 * 1000;
    const maxRequests = limits[tier];

    // Skip for enterprise if they have unlimited
    if (tier === 'enterprise' && limits.enterprise >= 1000000) {
      return next();
    }

    const stored = endpointStore[key];

    if (!stored || now > stored.resetTime) {
      endpointStore[key] = {
        count: 1,
        resetTime: now + windowMs,
      };
      return next();
    }

    stored.count++;

    if (stored.count > maxRequests) {
      res.status(429).json({
        error: {
          code: -32001,
          message: `Rate limit exceeded for ${endpoint}. Limit: ${maxRequests}/min for ${tier} tier.`,
        },
      });
      return;
    }

    next();
  };
}

/**
 * WebSocket rate limiter
 */
export class WebSocketRateLimiter {
  private store: Map<string, { count: number; resetTime: number }> = new Map();
  private windowMs: number;
  private maxMessages: number;

  constructor(windowMs: number = 1000, maxMessages: number = 100) {
    this.windowMs = windowMs;
    this.maxMessages = maxMessages;

    // Cleanup old entries periodically
    setInterval(() => this.cleanup(), windowMs * 10);
  }

  /**
   * Check if a connection is rate limited
   */
  isRateLimited(connectionId: string, tier: ApiTier = 'free'): boolean {
    const now = Date.now();
    const maxForTier = this.getMaxForTier(tier);
    const stored = this.store.get(connectionId);

    if (!stored || now > stored.resetTime) {
      this.store.set(connectionId, {
        count: 1,
        resetTime: now + this.windowMs,
      });
      return false;
    }

    stored.count++;

    if (stored.count > maxForTier) {
      logger.warn(`WebSocket rate limit exceeded for ${connectionId}`);
      return true;
    }

    return false;
  }

  /**
   * Get max messages per window for a tier
   */
  private getMaxForTier(tier: ApiTier): number {
    const multipliers: Record<ApiTier, number> = {
      free: 1,
      basic: 5,
      pro: 20,
      enterprise: 100,
    };
    return this.maxMessages * multipliers[tier];
  }

  /**
   * Cleanup expired entries
   */
  private cleanup(): void {
    const now = Date.now();
    for (const [key, value] of this.store.entries()) {
      if (now > value.resetTime) {
        this.store.delete(key);
      }
    }
  }

  /**
   * Remove a connection from the store
   */
  remove(connectionId: string): void {
    this.store.delete(connectionId);
  }
}
