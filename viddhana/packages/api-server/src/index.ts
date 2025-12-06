/**
 * VIDDHANA API Server Entry Point
 * 
 * Initializes and starts the Express + WebSocket server
 */

import 'dotenv/config';
import { createServer } from './server';
import { logger } from './services/logger';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

async function main(): Promise<void> {
  // Validate required environment variables
  const requiredEnvVars = ['JWT_SECRET', 'DATABASE_URL'];
  const missing = requiredEnvVars.filter((v) => !process.env[v]);
  
  if (missing.length > 0) {
    logger.error(`Missing required environment variables: ${missing.join(', ')}`);
    process.exit(1);
  }

  try {
    // Test database connection
    await prisma.$connect();
    logger.info('Database connected successfully');

    // Create and start server
    const { httpServer, app } = createServer();

    const port = parseInt(process.env.PORT || '3000', 10);
    const host = process.env.HOST || '0.0.0.0';

    httpServer.listen(port, host, () => {
      logger.info(`VIDDHANA API Server running on http://${host}:${port}`);
      logger.info(`Environment: ${process.env.NODE_ENV || 'development'}`);
      logger.info('Endpoints:');
      logger.info(`  - JSON-RPC: POST /rpc`);
      logger.info(`  - REST API: /api/v1/*`);
      logger.info(`  - WebSocket: ws://${host}:${port}`);
      logger.info(`  - Health: GET /health`);
    });

    // Graceful shutdown
    const shutdown = async (signal: string): Promise<void> => {
      logger.info(`Received ${signal}, shutting down gracefully...`);
      
      httpServer.close(async () => {
        logger.info('HTTP server closed');
        
        await prisma.$disconnect();
        logger.info('Database disconnected');
        
        process.exit(0);
      });

      // Force exit after 30 seconds
      setTimeout(() => {
        logger.error('Forced shutdown after timeout');
        process.exit(1);
      }, 30000);
    };

    process.on('SIGTERM', () => shutdown('SIGTERM'));
    process.on('SIGINT', () => shutdown('SIGINT'));

  } catch (error) {
    logger.error('Failed to start server:', error);
    await prisma.$disconnect();
    process.exit(1);
  }
}

main();
