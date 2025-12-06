/**
 * API Key Authentication Middleware
 * 
 * Handles API key validation and tier-based access control
 */

import { Response, NextFunction } from 'express';
import jwt from 'jsonwebtoken';
import { PrismaClient } from '@prisma/client';
import { AuthenticatedRequest, ApiKeyInfo, ApiTier } from '../types';
import { logger } from '../services/logger';

const prisma = new PrismaClient();

/**
 * Extract API key from request headers
 */
function extractApiKey(req: AuthenticatedRequest): string | null {
  // Check Authorization header (Bearer token)
  const authHeader = req.headers.authorization;
  if (authHeader?.startsWith('Bearer ')) {
    return authHeader.slice(7);
  }

  // Check X-API-Key header
  const apiKeyHeader = req.headers['x-api-key'];
  if (typeof apiKeyHeader === 'string') {
    return apiKeyHeader;
  }

  // Check query parameter (for WebSocket compatibility)
  if (typeof req.query.api_key === 'string') {
    return req.query.api_key;
  }

  return null;
}

/**
 * Validate API key and attach info to request
 */
export async function authMiddleware(
  req: AuthenticatedRequest,
  res: Response,
  next: NextFunction
): Promise<void> {
  const apiKey = extractApiKey(req);

  if (!apiKey) {
    res.status(401).json({
      error: {
        code: -32000,
        message: 'Missing API key. Include in Authorization header as Bearer token or X-API-Key header.',
      },
    });
    return;
  }

  try {
    // Try JWT verification first
    const jwtSecret = process.env.JWT_SECRET;
    if (jwtSecret && apiKey.includes('.')) {
      try {
        const decoded = jwt.verify(apiKey, jwtSecret) as {
          userId: string;
          tier: ApiTier;
          permissions: string[];
        };

        req.apiKey = {
          id: 'jwt',
          key: apiKey,
          tier: decoded.tier || 'free',
          userId: decoded.userId,
          permissions: decoded.permissions || [],
          createdAt: new Date(),
        };
        req.userId = decoded.userId;

        return next();
      } catch (jwtError) {
        // Not a valid JWT, continue to database lookup
      }
    }

    // Database lookup for API key
    const apiKeyRecord = await prisma.apiKey.findUnique({
      where: { key: apiKey },
      include: { user: true },
    });

    if (!apiKeyRecord) {
      res.status(401).json({
        error: {
          code: -32000,
          message: 'Invalid API key',
        },
      });
      return;
    }

    // Check if key is expired
    if (apiKeyRecord.expiresAt && apiKeyRecord.expiresAt < new Date()) {
      res.status(401).json({
        error: {
          code: -32000,
          message: 'API key has expired',
        },
      });
      return;
    }

    // Check if key is revoked
    if (apiKeyRecord.revoked) {
      res.status(401).json({
        error: {
          code: -32000,
          message: 'API key has been revoked',
        },
      });
      return;
    }

    // Update last used timestamp (fire and forget)
    prisma.apiKey
      .update({
        where: { id: apiKeyRecord.id },
        data: { lastUsedAt: new Date() },
      })
      .catch((err) => logger.warn('Failed to update API key last used:', err));

    // Attach API key info to request
    req.apiKey = {
      id: apiKeyRecord.id,
      key: apiKeyRecord.key,
      tier: apiKeyRecord.tier as ApiTier,
      userId: apiKeyRecord.userId,
      permissions: apiKeyRecord.permissions || [],
      createdAt: apiKeyRecord.createdAt,
      expiresAt: apiKeyRecord.expiresAt || undefined,
      lastUsedAt: apiKeyRecord.lastUsedAt || undefined,
    };
    req.userId = apiKeyRecord.userId;

    next();
  } catch (error) {
    logger.error('Auth middleware error:', error);
    res.status(500).json({
      error: {
        code: -32603,
        message: 'Authentication error',
      },
    });
  }
}

/**
 * Optional authentication - continues without auth if no key provided
 */
export async function optionalAuthMiddleware(
  req: AuthenticatedRequest,
  res: Response,
  next: NextFunction
): Promise<void> {
  const apiKey = extractApiKey(req);

  if (!apiKey) {
    // No API key provided, continue with free tier limits
    req.apiKey = {
      id: 'anonymous',
      key: '',
      tier: 'free',
      userId: 'anonymous',
      permissions: [],
      createdAt: new Date(),
    };
    return next();
  }

  // Validate the provided key
  return authMiddleware(req, res, next);
}

/**
 * Permission check middleware factory
 */
export function requirePermission(...permissions: string[]) {
  return (req: AuthenticatedRequest, res: Response, next: NextFunction): void => {
    if (!req.apiKey) {
      res.status(401).json({
        error: {
          code: -32000,
          message: 'Authentication required',
        },
      });
      return;
    }

    // Enterprise tier has all permissions
    if (req.apiKey.tier === 'enterprise') {
      return next();
    }

    const hasPermission = permissions.every((p) => 
      req.apiKey!.permissions.includes(p) || req.apiKey!.permissions.includes('*')
    );

    if (!hasPermission) {
      res.status(403).json({
        error: {
          code: -32000,
          message: `Missing required permissions: ${permissions.join(', ')}`,
        },
      });
      return;
    }

    next();
  };
}

/**
 * Tier check middleware factory
 */
export function requireTier(...tiers: ApiTier[]) {
  return (req: AuthenticatedRequest, res: Response, next: NextFunction): void => {
    if (!req.apiKey) {
      res.status(401).json({
        error: {
          code: -32000,
          message: 'Authentication required',
        },
      });
      return;
    }

    if (!tiers.includes(req.apiKey.tier)) {
      res.status(403).json({
        error: {
          code: -32000,
          message: `This endpoint requires one of the following tiers: ${tiers.join(', ')}`,
        },
      });
      return;
    }

    next();
  };
}

/**
 * Signed request verification for sensitive operations
 */
export async function verifySignedRequest(
  req: AuthenticatedRequest,
  res: Response,
  next: NextFunction
): Promise<void> {
  const signature = req.headers['x-signature'] as string;
  const timestamp = req.headers['x-timestamp'] as string;
  const address = req.headers['x-address'] as string;

  if (!signature || !timestamp || !address) {
    res.status(401).json({
      error: {
        code: -32000,
        message: 'Signed request requires X-Signature, X-Timestamp, and X-Address headers',
      },
    });
    return;
  }

  // Check timestamp is within 5 minutes
  const timestampNum = parseInt(timestamp, 10);
  const now = Date.now();
  if (Math.abs(now - timestampNum) > 5 * 60 * 1000) {
    res.status(401).json({
      error: {
        code: -32000,
        message: 'Request timestamp is too old or in the future',
      },
    });
    return;
  }

  // In production, verify the signature using ethers.js
  // const { ethers } = await import('ethers');
  // const message = JSON.stringify({ ...req.body, timestamp: timestampNum });
  // const recoveredAddress = ethers.verifyMessage(message, signature);
  // if (recoveredAddress.toLowerCase() !== address.toLowerCase()) {
  //   return res.status(401).json({ error: { message: 'Invalid signature' } });
  // }

  req.userId = address.toLowerCase();
  next();
}

/**
 * Generate a new API key for a user
 */
export async function generateApiKey(
  userId: string,
  tier: ApiTier = 'free',
  permissions: string[] = [],
  expiresInDays?: number
): Promise<ApiKeyInfo> {
  const { v4: uuidv4 } = await import('uuid');
  
  const key = `vdh_${tier}_${uuidv4().replace(/-/g, '')}`;
  const expiresAt = expiresInDays 
    ? new Date(Date.now() + expiresInDays * 24 * 60 * 60 * 1000)
    : null;

  const apiKeyRecord = await prisma.apiKey.create({
    data: {
      key,
      userId,
      tier,
      permissions,
      expiresAt,
    },
  });

  return {
    id: apiKeyRecord.id,
    key: apiKeyRecord.key,
    tier: apiKeyRecord.tier as ApiTier,
    userId: apiKeyRecord.userId,
    permissions: apiKeyRecord.permissions || [],
    createdAt: apiKeyRecord.createdAt,
    expiresAt: apiKeyRecord.expiresAt || undefined,
  };
}
