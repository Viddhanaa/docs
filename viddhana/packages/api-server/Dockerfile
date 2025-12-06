# VIDDHANA API Server Docker Configuration
# Multi-stage build for optimized production image

# ==============================================================================
# Stage 1: Dependencies
# ==============================================================================
FROM node:20-alpine AS deps

WORKDIR /app

# Install build dependencies for native modules
RUN apk add --no-cache libc6-compat python3 make g++

# Copy package files
COPY package.json package-lock.json* yarn.lock* pnpm-lock.yaml* ./

# Install dependencies based on lockfile
RUN \
  if [ -f yarn.lock ]; then yarn --frozen-lockfile; \
  elif [ -f package-lock.json ]; then npm ci; \
  elif [ -f pnpm-lock.yaml ]; then corepack enable pnpm && pnpm i --frozen-lockfile; \
  else echo "No lockfile found." && npm install; \
  fi

# ==============================================================================
# Stage 2: Builder
# ==============================================================================
FROM node:20-alpine AS builder

WORKDIR /app

# Copy dependencies from deps stage
COPY --from=deps /app/node_modules ./node_modules
COPY . .

# Generate Prisma client
RUN npx prisma generate

# Build TypeScript
RUN npm run build

# Remove dev dependencies
RUN npm prune --production

# ==============================================================================
# Stage 3: Production
# ==============================================================================
FROM node:20-alpine AS runner

WORKDIR /app

# Set production environment
ENV NODE_ENV=production
ENV PORT=3000

# Create non-root user for security
RUN addgroup --system --gid 1001 nodejs && \
    adduser --system --uid 1001 viddhana

# Copy built assets
COPY --from=builder --chown=viddhana:nodejs /app/dist ./dist
COPY --from=builder --chown=viddhana:nodejs /app/node_modules ./node_modules
COPY --from=builder --chown=viddhana:nodejs /app/package.json ./package.json
COPY --from=builder --chown=viddhana:nodejs /app/prisma ./prisma

# Create logs directory
RUN mkdir -p logs && chown viddhana:nodejs logs

# Switch to non-root user
USER viddhana

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:3000/health || exit 1

# Start server
CMD ["node", "dist/index.js"]

# ==============================================================================
# Labels
# ==============================================================================
LABEL org.opencontainers.image.title="VIDDHANA API Server"
LABEL org.opencontainers.image.description="Production-ready JSON-RPC and REST API server for VIDDHANA Network"
LABEL org.opencontainers.image.vendor="VIDDHANA"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.source="https://github.com/viddhana/api-server"
