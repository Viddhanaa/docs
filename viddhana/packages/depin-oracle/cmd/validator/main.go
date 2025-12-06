// Package main provides the validator node entry point for the DePIN Oracle network.
// Validators are responsible for verifying IoT sensor data and participating in
// BFT consensus to achieve agreement on data validity.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/viddhana/depin-oracle/internal/oracle"
	"github.com/viddhana/depin-oracle/internal/p2p"
	"github.com/viddhana/depin-oracle/internal/storage"
	"github.com/viddhana/depin-oracle/pkg/types"
	"go.uber.org/zap"
)

var (
	configPath     = flag.String("config", "configs/config.yaml", "Path to configuration file")
	validatorIndex = flag.Int("index", 0, "Validator index (0-12)")
	privateKeyPath = flag.String("key", "", "Path to validator private key")
)

func main() {
	flag.Parse()

	// Initialize logger
	logger, err := zap.NewProduction()
	if err != nil {
		panic("failed to initialize logger: " + err.Error())
	}
	defer logger.Sync()

	sugar := logger.Sugar()
	sugar.Info("Starting VIDDHANA DePIN Oracle Validator Node")

	// Load configuration
	cfg, err := types.LoadConfig(*configPath)
	if err != nil {
		sugar.Fatalf("Failed to load configuration: %v", err)
	}

	// Override with command line flags
	if *validatorIndex >= 0 {
		cfg.ValidatorIndex = *validatorIndex
	}
	if *privateKeyPath != "" {
		cfg.PrivateKeyPath = *privateKeyPath
	}

	// Initialize database
	db, err := storage.NewDatabase(cfg.Database)
	if err != nil {
		sugar.Fatalf("Failed to connect to database: %v", err)
	}
	sugar.Info("Connected to database")

	// Initialize P2P network
	p2pNetwork, err := p2p.NewNetwork(p2p.Config{
		ListenAddr:     cfg.P2P.ListenAddr,
		BootstrapPeers: cfg.P2P.BootstrapPeers,
		PrivateKeyPath: cfg.PrivateKeyPath,
	})
	if err != nil {
		sugar.Fatalf("Failed to initialize P2P network: %v", err)
	}
	sugar.Info("P2P network initialized")

	// Initialize Oracle Validator
	validator, err := oracle.NewValidator(oracle.ValidatorConfig{
		PrivateKeyPath:   cfg.PrivateKeyPath,
		ChainRPC:         cfg.ChainRPC,
		OracleContract:   cfg.OracleContract,
		ValidatorIndex:   cfg.ValidatorIndex,
		TotalValidators:  cfg.TotalValidators,
		RequiredSigs:     cfg.RequiredSignatures,
		ConsensusTimeout: time.Duration(cfg.ConsensusTimeoutSec) * time.Second,
		Database:         db,
		P2PNetwork:       p2pNetwork,
		Logger:           sugar,
	})
	if err != nil {
		sugar.Fatalf("Failed to initialize validator: %v", err)
	}
	sugar.Infof("Validator initialized with index %d", cfg.ValidatorIndex)

	// Create context with cancellation
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Start P2P network
	if err := p2pNetwork.Start(ctx); err != nil {
		sugar.Fatalf("Failed to start P2P network: %v", err)
	}
	sugar.Info("P2P network started")

	// Subscribe to sensor data topics
	if err := p2pNetwork.Subscribe("sensors/data", validator.HandleSensorData); err != nil {
		sugar.Fatalf("Failed to subscribe to sensor data: %v", err)
	}

	// Subscribe to consensus messages
	if err := p2pNetwork.Subscribe("consensus/votes", validator.HandleConsensusVote); err != nil {
		sugar.Fatalf("Failed to subscribe to consensus votes: %v", err)
	}

	// Start consensus engine
	go validator.RunConsensus(ctx)

	// Start health check and metrics server
	go startHealthServer(cfg.HealthPort, validator, sugar)

	sugar.Infof("Validator node started successfully on port %d", cfg.HealthPort)

	// Wait for shutdown signal
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	<-sigCh

	sugar.Info("Shutting down validator node...")
	cancel()

	// Graceful shutdown
	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer shutdownCancel()

	if err := validator.Shutdown(shutdownCtx); err != nil {
		sugar.Errorf("Error during shutdown: %v", err)
	}

	if err := p2pNetwork.Stop(); err != nil {
		sugar.Errorf("Error stopping P2P network: %v", err)
	}

	sugar.Info("Validator node stopped")
}

// startHealthServer starts the HTTP server for health checks and metrics
func startHealthServer(port int, validator *oracle.Validator, logger *zap.SugaredLogger) {
	gin.SetMode(gin.ReleaseMode)
	r := gin.New()
	r.Use(gin.Recovery())

	// Health check endpoint
	r.GET("/health", func(c *gin.Context) {
		status := validator.GetStatus()
		if status.IsHealthy {
			c.JSON(200, gin.H{
				"status":          "healthy",
				"validator_index": status.ValidatorIndex,
				"peers_connected": status.PeersConnected,
				"last_consensus":  status.LastConsensusTime,
				"pending_rounds":  status.PendingRounds,
			})
		} else {
			c.JSON(503, gin.H{
				"status": "unhealthy",
				"error":  status.Error,
			})
		}
	})

	// Readiness check
	r.GET("/ready", func(c *gin.Context) {
		if validator.IsReady() {
			c.JSON(200, gin.H{"ready": true})
		} else {
			c.JSON(503, gin.H{"ready": false})
		}
	})

	// Metrics endpoint
	r.GET("/metrics", func(c *gin.Context) {
		metrics := validator.GetMetrics()
		c.JSON(200, metrics)
	})

	// Validator info
	r.GET("/info", func(c *gin.Context) {
		c.JSON(200, gin.H{
			"validator_index":  validator.GetIndex(),
			"validator_addr":   validator.GetAddress(),
			"total_validators": validator.GetTotalValidators(),
			"required_sigs":    validator.GetRequiredSignatures(),
		})
	})

	addr := ":8080"
	if port > 0 {
		addr = fmt.Sprintf(":%d", port)
	}

	logger.Infof("Starting health server on %s", addr)
	if err := r.Run(addr); err != nil {
		logger.Errorf("Health server error: %v", err)
	}
}
