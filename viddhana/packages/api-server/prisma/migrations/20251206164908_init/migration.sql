-- CreateTable
CREATE TABLE "users" (
    "id" TEXT NOT NULL,
    "address" TEXT NOT NULL,
    "email" TEXT,
    "username" TEXT,
    "tier" TEXT NOT NULL DEFAULT 'free',
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "users_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "api_keys" (
    "id" TEXT NOT NULL,
    "key" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "tier" TEXT NOT NULL DEFAULT 'free',
    "permissions" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "revoked" BOOLEAN NOT NULL DEFAULT false,
    "expiresAt" TIMESTAMP(3),
    "lastUsedAt" TIMESTAMP(3),
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "api_keys_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "vaults" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "contractAddress" TEXT,
    "totalValue" TEXT NOT NULL DEFAULT '0',
    "currency" TEXT NOT NULL DEFAULT 'USD',
    "riskTolerance" INTEGER NOT NULL DEFAULT 5000,
    "timeToGoal" INTEGER NOT NULL DEFAULT 24,
    "autoRebalance" BOOLEAN NOT NULL DEFAULT true,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "vaults_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "vault_assets" (
    "id" TEXT NOT NULL,
    "vaultId" TEXT NOT NULL,
    "symbol" TEXT NOT NULL,
    "address" TEXT NOT NULL,
    "balance" TEXT NOT NULL,
    "value" TEXT NOT NULL,
    "allocation" DOUBLE PRECISION NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "vault_assets_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "vault_transactions" (
    "id" TEXT NOT NULL,
    "vaultId" TEXT NOT NULL,
    "type" TEXT NOT NULL,
    "asset" TEXT NOT NULL,
    "amount" TEXT NOT NULL,
    "txHash" TEXT,
    "status" TEXT NOT NULL DEFAULT 'PENDING',
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "confirmedAt" TIMESTAMP(3),

    CONSTRAINT "vault_transactions_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "rebalance_history" (
    "id" TEXT NOT NULL,
    "vaultId" TEXT NOT NULL,
    "txHash" TEXT,
    "blockNumber" INTEGER,
    "reason" TEXT NOT NULL,
    "actions" JSONB NOT NULL,
    "aiConfidence" DOUBLE PRECISION NOT NULL,
    "gasUsed" INTEGER,
    "gasCost" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "rebalance_history_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "transactions" (
    "id" TEXT NOT NULL,
    "hash" TEXT NOT NULL,
    "blockNumber" INTEGER NOT NULL,
    "blockHash" TEXT NOT NULL,
    "from" TEXT NOT NULL,
    "to" TEXT,
    "value" TEXT NOT NULL,
    "gasPrice" TEXT NOT NULL,
    "gasUsed" INTEGER,
    "input" TEXT,
    "status" INTEGER NOT NULL DEFAULT 1,
    "timestamp" TIMESTAMP(3) NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "transactions_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "blocks" (
    "id" TEXT NOT NULL,
    "number" INTEGER NOT NULL,
    "hash" TEXT NOT NULL,
    "parentHash" TEXT NOT NULL,
    "timestamp" TIMESTAMP(3) NOT NULL,
    "gasLimit" TEXT NOT NULL,
    "gasUsed" TEXT NOT NULL,
    "transactionCount" INTEGER NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "blocks_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "sensors" (
    "id" TEXT NOT NULL,
    "sensorId" TEXT NOT NULL,
    "owner" TEXT NOT NULL,
    "deviceType" TEXT NOT NULL,
    "status" TEXT NOT NULL DEFAULT 'active',
    "location" JSONB,
    "metadata" JSONB,
    "uptimePercent" DOUBLE PRECISION NOT NULL DEFAULT 0,
    "dataQualityScore" DOUBLE PRECISION NOT NULL DEFAULT 0,
    "dailyRate" TEXT NOT NULL DEFAULT '0',
    "errorCount" INTEGER NOT NULL DEFAULT 0,
    "pendingRewards" TEXT NOT NULL DEFAULT '0',
    "claimedRewards" TEXT NOT NULL DEFAULT '0',
    "lifetimeRewards" TEXT NOT NULL DEFAULT '0',
    "lastDataPoint" TIMESTAMP(3),
    "registeredAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "sensors_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "sensor_data" (
    "id" TEXT NOT NULL,
    "sensorId" TEXT NOT NULL,
    "dataType" TEXT NOT NULL,
    "value" DOUBLE PRECISION NOT NULL,
    "unit" TEXT NOT NULL,
    "timestamp" TIMESTAMP(3) NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "sensor_data_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "price_data" (
    "id" TEXT NOT NULL,
    "asset" TEXT NOT NULL,
    "price" DOUBLE PRECISION NOT NULL,
    "volume24h" DOUBLE PRECISION,
    "marketCap" DOUBLE PRECISION,
    "change24h" DOUBLE PRECISION,
    "timestamp" TIMESTAMP(3) NOT NULL,
    "source" TEXT NOT NULL DEFAULT 'aggregated',
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "price_data_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "predictions" (
    "id" TEXT NOT NULL,
    "asset" TEXT NOT NULL,
    "horizon" INTEGER NOT NULL,
    "predictions" JSONB NOT NULL,
    "trend" TEXT NOT NULL,
    "volatilityForecast" DOUBLE PRECISION NOT NULL,
    "modelVersion" TEXT NOT NULL,
    "generatedAt" TIMESTAMP(3) NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "predictions_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "api_logs" (
    "id" TEXT NOT NULL,
    "method" TEXT NOT NULL,
    "requestId" TEXT,
    "apiKeyId" TEXT,
    "userId" TEXT,
    "requestBody" JSONB,
    "responseCode" INTEGER NOT NULL,
    "responseTime" INTEGER NOT NULL,
    "ip" TEXT,
    "userAgent" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "api_logs_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "rate_limit_logs" (
    "id" TEXT NOT NULL,
    "key" TEXT NOT NULL,
    "tier" TEXT NOT NULL,
    "endpoint" TEXT,
    "requestCount" INTEGER NOT NULL,
    "limitExceeded" BOOLEAN NOT NULL DEFAULT false,
    "windowStart" TIMESTAMP(3) NOT NULL,
    "windowEnd" TIMESTAMP(3) NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "rate_limit_logs_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "users_address_key" ON "users"("address");

-- CreateIndex
CREATE UNIQUE INDEX "users_email_key" ON "users"("email");

-- CreateIndex
CREATE UNIQUE INDEX "users_username_key" ON "users"("username");

-- CreateIndex
CREATE INDEX "users_address_idx" ON "users"("address");

-- CreateIndex
CREATE UNIQUE INDEX "api_keys_key_key" ON "api_keys"("key");

-- CreateIndex
CREATE INDEX "api_keys_key_idx" ON "api_keys"("key");

-- CreateIndex
CREATE INDEX "api_keys_userId_idx" ON "api_keys"("userId");

-- CreateIndex
CREATE INDEX "vaults_userId_idx" ON "vaults"("userId");

-- CreateIndex
CREATE INDEX "vaults_contractAddress_idx" ON "vaults"("contractAddress");

-- CreateIndex
CREATE INDEX "vault_assets_vaultId_idx" ON "vault_assets"("vaultId");

-- CreateIndex
CREATE UNIQUE INDEX "vault_assets_vaultId_address_key" ON "vault_assets"("vaultId", "address");

-- CreateIndex
CREATE INDEX "vault_transactions_vaultId_idx" ON "vault_transactions"("vaultId");

-- CreateIndex
CREATE INDEX "vault_transactions_txHash_idx" ON "vault_transactions"("txHash");

-- CreateIndex
CREATE INDEX "vault_transactions_status_idx" ON "vault_transactions"("status");

-- CreateIndex
CREATE INDEX "rebalance_history_vaultId_idx" ON "rebalance_history"("vaultId");

-- CreateIndex
CREATE INDEX "rebalance_history_txHash_idx" ON "rebalance_history"("txHash");

-- CreateIndex
CREATE UNIQUE INDEX "transactions_hash_key" ON "transactions"("hash");

-- CreateIndex
CREATE INDEX "transactions_blockNumber_idx" ON "transactions"("blockNumber");

-- CreateIndex
CREATE INDEX "transactions_from_idx" ON "transactions"("from");

-- CreateIndex
CREATE INDEX "transactions_to_idx" ON "transactions"("to");

-- CreateIndex
CREATE INDEX "transactions_timestamp_idx" ON "transactions"("timestamp");

-- CreateIndex
CREATE UNIQUE INDEX "blocks_number_key" ON "blocks"("number");

-- CreateIndex
CREATE UNIQUE INDEX "blocks_hash_key" ON "blocks"("hash");

-- CreateIndex
CREATE INDEX "blocks_number_idx" ON "blocks"("number");

-- CreateIndex
CREATE INDEX "blocks_timestamp_idx" ON "blocks"("timestamp");

-- CreateIndex
CREATE UNIQUE INDEX "sensors_sensorId_key" ON "sensors"("sensorId");

-- CreateIndex
CREATE INDEX "sensors_sensorId_idx" ON "sensors"("sensorId");

-- CreateIndex
CREATE INDEX "sensors_owner_idx" ON "sensors"("owner");

-- CreateIndex
CREATE INDEX "sensors_deviceType_idx" ON "sensors"("deviceType");

-- CreateIndex
CREATE INDEX "sensors_status_idx" ON "sensors"("status");

-- CreateIndex
CREATE INDEX "sensor_data_sensorId_idx" ON "sensor_data"("sensorId");

-- CreateIndex
CREATE INDEX "sensor_data_timestamp_idx" ON "sensor_data"("timestamp");

-- CreateIndex
CREATE INDEX "sensor_data_dataType_idx" ON "sensor_data"("dataType");

-- CreateIndex
CREATE INDEX "price_data_asset_idx" ON "price_data"("asset");

-- CreateIndex
CREATE INDEX "price_data_timestamp_idx" ON "price_data"("timestamp");

-- CreateIndex
CREATE UNIQUE INDEX "price_data_asset_timestamp_key" ON "price_data"("asset", "timestamp");

-- CreateIndex
CREATE INDEX "predictions_asset_idx" ON "predictions"("asset");

-- CreateIndex
CREATE INDEX "predictions_generatedAt_idx" ON "predictions"("generatedAt");

-- CreateIndex
CREATE INDEX "api_logs_method_idx" ON "api_logs"("method");

-- CreateIndex
CREATE INDEX "api_logs_apiKeyId_idx" ON "api_logs"("apiKeyId");

-- CreateIndex
CREATE INDEX "api_logs_userId_idx" ON "api_logs"("userId");

-- CreateIndex
CREATE INDEX "api_logs_createdAt_idx" ON "api_logs"("createdAt");

-- CreateIndex
CREATE INDEX "rate_limit_logs_key_idx" ON "rate_limit_logs"("key");

-- CreateIndex
CREATE INDEX "rate_limit_logs_windowStart_idx" ON "rate_limit_logs"("windowStart");

-- AddForeignKey
ALTER TABLE "api_keys" ADD CONSTRAINT "api_keys_userId_fkey" FOREIGN KEY ("userId") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "vaults" ADD CONSTRAINT "vaults_userId_fkey" FOREIGN KEY ("userId") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "vault_assets" ADD CONSTRAINT "vault_assets_vaultId_fkey" FOREIGN KEY ("vaultId") REFERENCES "vaults"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "vault_transactions" ADD CONSTRAINT "vault_transactions_vaultId_fkey" FOREIGN KEY ("vaultId") REFERENCES "vaults"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "rebalance_history" ADD CONSTRAINT "rebalance_history_vaultId_fkey" FOREIGN KEY ("vaultId") REFERENCES "vaults"("id") ON DELETE CASCADE ON UPDATE CASCADE;
