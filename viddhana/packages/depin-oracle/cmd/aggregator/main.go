// Package main provides the price aggregator entry point for the DePIN Oracle network.
// The aggregator collects prices from multiple sources and calculates median values
// for submission to the blockchain.
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
	"github.com/viddhana/depin-oracle/internal/storage"
	"github.com/viddhana/depin-oracle/pkg/types"
	"go.uber.org/zap"
)

var (
	configPath = flag.String("config", "configs/config.yaml", "Path to configuration file")
)

// PriceSource represents a price data source
type PriceSource struct {
	Name     string
	URL      string
	Weight   float64
	Timeout  time.Duration
	Fallback bool
}

func main() {
	flag.Parse()

	// Initialize logger
	logger, err := zap.NewProduction()
	if err != nil {
		panic("failed to initialize logger: " + err.Error())
	}
	defer logger.Sync()

	sugar := logger.Sugar()
	sugar.Info("Starting VIDDHANA DePIN Price Aggregator")

	// Load configuration
	cfg, err := types.LoadConfig(*configPath)
	if err != nil {
		sugar.Fatalf("Failed to load configuration: %v", err)
	}

	// Initialize database
	db, err := storage.NewDatabase(cfg.Database)
	if err != nil {
		sugar.Fatalf("Failed to connect to database: %v", err)
	}
	sugar.Info("Connected to database")

	// Configure price sources
	priceSources := []PriceSource{
		{Name: "Binance", URL: cfg.PriceSources.Binance, Weight: 1.0, Timeout: 5 * time.Second},
		{Name: "Coinbase", URL: cfg.PriceSources.Coinbase, Weight: 1.0, Timeout: 5 * time.Second},
		{Name: "Kraken", URL: cfg.PriceSources.Kraken, Weight: 1.0, Timeout: 5 * time.Second},
		{Name: "CoinGecko", URL: cfg.PriceSources.CoinGecko, Weight: 0.8, Timeout: 10 * time.Second},
		{Name: "Chainlink", URL: cfg.PriceSources.Chainlink, Weight: 1.2, Timeout: 5 * time.Second, Fallback: true},
	}

	// Initialize aggregator
	aggregator := oracle.NewAggregator(oracle.AggregatorConfig{
		Sources:          convertSources(priceSources),
		UpdateInterval:   time.Duration(cfg.AggregatorInterval) * time.Second,
		StaleThreshold:   time.Duration(cfg.StaleThreshold) * time.Second,
		DeviationLimit:   cfg.DeviationLimit,
		MinSources:       cfg.MinPriceSources,
		Database:         db,
		Logger:           sugar,
		ChainRPC:         cfg.ChainRPC,
		OracleContract:   cfg.OracleContract,
		PrivateKeyPath:   cfg.PrivateKeyPath,
	})

	// Create context with cancellation
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Start aggregator
	go aggregator.Run(ctx)
	sugar.Info("Price aggregator started")

	// Start HTTP server for API
	go startAPIServer(cfg.AggregatorPort, aggregator, sugar)

	sugar.Infof("Aggregator running on port %d", cfg.AggregatorPort)

	// Wait for shutdown signal
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	<-sigCh

	sugar.Info("Shutting down aggregator...")
	cancel()

	// Graceful shutdown
	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer shutdownCancel()

	if err := aggregator.Shutdown(shutdownCtx); err != nil {
		sugar.Errorf("Error during shutdown: %v", err)
	}

	sugar.Info("Aggregator stopped")
}

// convertSources converts PriceSource to oracle.Source
func convertSources(sources []PriceSource) []oracle.Source {
	result := make([]oracle.Source, len(sources))
	for i, s := range sources {
		result[i] = oracle.Source{
			Name:     s.Name,
			URL:      s.URL,
			Weight:   s.Weight,
			Timeout:  s.Timeout,
			Fallback: s.Fallback,
		}
	}
	return result
}

// startAPIServer starts the HTTP server for price queries
func startAPIServer(port int, aggregator *oracle.Aggregator, logger *zap.SugaredLogger) {
	gin.SetMode(gin.ReleaseMode)
	r := gin.New()
	r.Use(gin.Recovery())

	// Health check
	r.GET("/health", func(c *gin.Context) {
		c.JSON(200, gin.H{"status": "healthy"})
	})

	// Get current prices
	r.GET("/prices", func(c *gin.Context) {
		prices := aggregator.GetCurrentPrices()
		c.JSON(200, prices)
	})

	// Get specific asset price
	r.GET("/prices/:asset", func(c *gin.Context) {
		asset := c.Param("asset")
		price, err := aggregator.GetAssetPrice(asset)
		if err != nil {
			c.JSON(404, gin.H{"error": err.Error()})
			return
		}
		c.JSON(200, price)
	})

	// Get price history
	r.GET("/prices/:asset/history", func(c *gin.Context) {
		asset := c.Param("asset")
		history, err := aggregator.GetPriceHistory(asset, 100)
		if err != nil {
			c.JSON(500, gin.H{"error": err.Error()})
			return
		}
		c.JSON(200, history)
	})

	// Get aggregator stats
	r.GET("/stats", func(c *gin.Context) {
		stats := aggregator.GetStats()
		c.JSON(200, stats)
	})

	// Get source status
	r.GET("/sources", func(c *gin.Context) {
		sources := aggregator.GetSourceStatus()
		c.JSON(200, sources)
	})

	addr := ":8081"
	if port > 0 {
		addr = fmt.Sprintf(":%d", port)
	}

	logger.Infof("Starting API server on %s", addr)
	if err := r.Run(addr); err != nil {
		logger.Errorf("API server error: %v", err)
	}
}
