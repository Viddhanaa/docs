/**
 * VIDDHANA Express + WebSocket Server
 * 
 * Configures Express application with middleware and routes,
 * plus Socket.IO for real-time subscriptions
 */

import express, { Application, Request, Response, NextFunction } from 'express';
import { createServer as createHttpServer, Server as HttpServer } from 'http';
import { Server as SocketServer } from 'socket.io';
import cors from 'cors';
import helmet from 'helmet';
import compression from 'compression';
import morgan from 'morgan';

import { rpcRouter } from './routes/rpc';
import { restRouter } from './routes/rest';
import { authMiddleware } from './middleware/auth';
import { createRateLimiter } from './middleware/rateLimit';
import { WebSocketManager } from './websocket/manager';
import { logger } from './services/logger';
import { JsonRpcErrorCode } from './types';

interface ServerInstance {
  app: Application;
  httpServer: HttpServer;
  io: SocketServer;
  wsManager: WebSocketManager;
}

export function createServer(): ServerInstance {
  const app = express();
  const httpServer = createHttpServer(app);

  // Socket.IO server
  const io = new SocketServer(httpServer, {
    cors: {
      origin: process.env.CORS_ORIGINS?.split(',') || '*',
      methods: ['GET', 'POST'],
      credentials: true,
    },
    pingTimeout: 60000,
    pingInterval: 25000,
  });

  // Initialize WebSocket manager
  const wsManager = new WebSocketManager(io);

  // ============================================================================
  // Middleware
  // ============================================================================

  // Security headers
  app.use(helmet({
    contentSecurityPolicy: false, // Disable for API server
  }));

  // Compression
  app.use(compression());

  // CORS
  app.use(cors({
    origin: process.env.CORS_ORIGINS?.split(',') || '*',
    methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization', 'X-API-Key', 'X-Signature', 'X-Timestamp', 'X-Address'],
    credentials: true,
  }));

  // Request logging
  app.use(morgan('combined', {
    stream: {
      write: (message: string) => logger.http(message.trim()),
    },
  }));

  // Body parsing
  app.use(express.json({ limit: '1mb' }));
  app.use(express.urlencoded({ extended: true, limit: '1mb' }));

  // Rate limiting (applied globally, refined per-route)
  app.use(createRateLimiter());

  // ============================================================================
  // Health Check (no auth required)
  // ============================================================================

  app.get('/health', (_req: Request, res: Response) => {
    res.json({
      status: 'healthy',
      timestamp: new Date().toISOString(),
      version: process.env.npm_package_version || '1.0.0',
      uptime: process.uptime(),
    });
  });

  app.get('/ready', async (_req: Request, res: Response) => {
    try {
      // Add any readiness checks here (database, blockchain connection, etc.)
      res.json({
        status: 'ready',
        timestamp: new Date().toISOString(),
      });
    } catch (error) {
      res.status(503).json({
        status: 'not ready',
        error: error instanceof Error ? error.message : 'Unknown error',
      });
    }
  });

  // ============================================================================
  // API Routes
  // ============================================================================

  // JSON-RPC endpoint (with optional auth for rate limiting tiers)
  app.use('/rpc', rpcRouter);

  // REST API (with auth)
  app.use('/api/v1', authMiddleware, restRouter);

  // ============================================================================
  // Error Handling
  // ============================================================================

  // 404 handler
  app.use((_req: Request, res: Response) => {
    res.status(404).json({
      jsonrpc: '2.0',
      error: {
        code: JsonRpcErrorCode.METHOD_NOT_FOUND,
        message: 'Endpoint not found',
      },
      id: null,
    });
  });

  // Global error handler
  app.use((err: Error, _req: Request, res: Response, _next: NextFunction) => {
    logger.error('Unhandled error:', err);

    res.status(500).json({
      jsonrpc: '2.0',
      error: {
        code: JsonRpcErrorCode.INTERNAL_ERROR,
        message: process.env.NODE_ENV === 'production' 
          ? 'Internal server error' 
          : err.message,
      },
      id: null,
    });
  });

  return { app, httpServer, io, wsManager };
}
