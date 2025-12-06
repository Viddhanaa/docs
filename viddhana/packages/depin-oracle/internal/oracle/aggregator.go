// Package oracle implements the DePIN Oracle aggregator for price data.
package oracle

import (
	"context"
	"errors"
	"fmt"
	"math/big"
	"net/http"
	"sync"
	"time"

	"github.com/ethereum/go-ethereum/crypto"
	"github.com/viddhana/depin-oracle/internal/storage"
	"github.com/viddhana/depin-oracle/pkg/types"
	"go.uber.org/zap"
)

// Source represents a price data source
type Source struct {
	Name     string
	URL      string
	Weight   float64
	Timeout  time.Duration
	Fallback bool
}

// AggregatorConfig holds configuration for the price aggregator
type AggregatorConfig struct {
	Sources          []Source
	UpdateInterval   time.Duration
	StaleThreshold   time.Duration
	DeviationLimit   float64
	MinSources       int
	Database         *storage.Database
	Logger           *zap.SugaredLogger
	ChainRPC         string
	OracleContract   string
	PrivateKeyPath   string
}

// Aggregator collects and aggregates price data from multiple sources
type Aggregator struct {
	sources        []Source
	updateInterval time.Duration
	staleThreshold time.Duration
	deviationLimit float64
	minSources     int
	database       *storage.Database
	logger         *zap.SugaredLogger

	// Current prices
	prices   map[string]*types.PriceData
	pricesMu sync.RWMutex

	// Source status
	sourceStatus   map[string]*SourceStatus
	sourceStatusMu sync.RWMutex

	// HTTP client for fetching prices
	httpClient *http.Client

	// Stats
	stats   *AggregatorStats
	statsMu sync.RWMutex

	// Shutdown
	shutdownCh chan struct{}
}

// SourceStatus tracks the status of a price source
type SourceStatus struct {
	Name        string    `json:"name"`
	LastSuccess time.Time `json:"last_success"`
	LastError   string    `json:"last_error,omitempty"`
	SuccessRate float64   `json:"success_rate"`
	Latency     float64   `json:"latency_ms"`
	IsHealthy   bool      `json:"is_healthy"`
}

// AggregatorStats tracks aggregator statistics
type AggregatorStats struct {
	TotalUpdates       uint64    `json:"total_updates"`
	SuccessfulUpdates  uint64    `json:"successful_updates"`
	FailedUpdates      uint64    `json:"failed_updates"`
	LastUpdateTime     time.Time `json:"last_update_time"`
	AverageLatencyMs   float64   `json:"average_latency_ms"`
	ActiveSources      int       `json:"active_sources"`
}

// NewAggregator creates a new price aggregator
func NewAggregator(cfg AggregatorConfig) *Aggregator {
	sourceStatus := make(map[string]*SourceStatus)
	for _, s := range cfg.Sources {
		sourceStatus[s.Name] = &SourceStatus{
			Name:      s.Name,
			IsHealthy: true,
		}
	}

	return &Aggregator{
		sources:        cfg.Sources,
		updateInterval: cfg.UpdateInterval,
		staleThreshold: cfg.StaleThreshold,
		deviationLimit: cfg.DeviationLimit,
		minSources:     cfg.MinSources,
		database:       cfg.Database,
		logger:         cfg.Logger,
		prices:         make(map[string]*types.PriceData),
		sourceStatus:   sourceStatus,
		httpClient: &http.Client{
			Timeout: 10 * time.Second,
		},
		stats:      &AggregatorStats{},
		shutdownCh: make(chan struct{}),
	}
}

// Run starts the aggregator main loop
func (a *Aggregator) Run(ctx context.Context) {
	ticker := time.NewTicker(a.updateInterval)
	defer ticker.Stop()

	// Initial fetch
	a.updatePrices(ctx)

	for {
		select {
		case <-ctx.Done():
			return
		case <-a.shutdownCh:
			return
		case <-ticker.C:
			a.updatePrices(ctx)
		}
	}
}

// updatePrices fetches prices from all sources and aggregates them
func (a *Aggregator) updatePrices(ctx context.Context) {
	startTime := time.Now()

	// Define assets to fetch
	assets := []string{"ETH", "BTC", "VIDH"}

	for _, asset := range assets {
		prices := make([]float64, 0, len(a.sources))
		weights := make([]float64, 0, len(a.sources))

		for _, source := range a.sources {
			price, err := a.fetchFromSource(ctx, source, asset)
			if err != nil {
				a.logger.Warnf("Failed to fetch %s from %s: %v", asset, source.Name, err)
				a.updateSourceStatus(source.Name, err)
				continue
			}

			prices = append(prices, price)
			weights = append(weights, source.Weight)
			a.updateSourceStatus(source.Name, nil)
		}

		if len(prices) < a.minSources {
			a.logger.Warnf("Not enough sources for %s: got %d, need %d", asset, len(prices), a.minSources)
			continue
		}

		// Calculate weighted median
		medianPrice := a.calculateWeightedMedian(prices, weights)

		// Convert to big.Int (18 decimals)
		priceWei := new(big.Int)
		priceWei.SetInt64(int64(medianPrice * 1e18))

		priceData := &types.PriceData{
			Asset:     asset,
			Price:     priceWei,
			Timestamp: time.Now().Unix(),
			Source:    "aggregator",
		}

		a.pricesMu.Lock()
		a.prices[asset] = priceData
		a.pricesMu.Unlock()

		// Store in database
		if err := a.database.StorePriceRecord(asset, priceWei.String(), "aggregator", time.Now()); err != nil {
			a.logger.Warnf("Failed to store price for %s: %v", asset, err)
		}
	}

	// Update stats
	latency := time.Since(startTime).Milliseconds()
	a.statsMu.Lock()
	a.stats.TotalUpdates++
	a.stats.SuccessfulUpdates++
	a.stats.LastUpdateTime = time.Now()
	a.stats.AverageLatencyMs = float64(latency)
	a.statsMu.Unlock()
}

// fetchFromSource fetches price from a single source
func (a *Aggregator) fetchFromSource(ctx context.Context, source Source, asset string) (float64, error) {
	// In production, this would make HTTP requests to the source URL
	// For now, return mock data
	switch source.Name {
	case "Binance":
		return 3500.0 + float64(time.Now().UnixNano()%100)/100, nil
	case "Coinbase":
		return 3501.0 + float64(time.Now().UnixNano()%100)/100, nil
	case "Kraken":
		return 3499.0 + float64(time.Now().UnixNano()%100)/100, nil
	case "CoinGecko":
		return 3500.5 + float64(time.Now().UnixNano()%100)/100, nil
	case "Chainlink":
		return 3500.0, nil
	default:
		return 0, errors.New("unknown source")
	}
}

// calculateWeightedMedian calculates a weighted median of prices
func (a *Aggregator) calculateWeightedMedian(prices []float64, weights []float64) float64 {
	if len(prices) == 0 {
		return 0
	}

	// Simple weighted average for now
	var totalWeight, weightedSum float64
	for i, price := range prices {
		weight := weights[i]
		weightedSum += price * weight
		totalWeight += weight
	}

	if totalWeight == 0 {
		return 0
	}

	return weightedSum / totalWeight
}

// updateSourceStatus updates the status of a price source
func (a *Aggregator) updateSourceStatus(name string, err error) {
	a.sourceStatusMu.Lock()
	defer a.sourceStatusMu.Unlock()

	status, exists := a.sourceStatus[name]
	if !exists {
		return
	}

	if err != nil {
		status.LastError = err.Error()
		status.IsHealthy = false
	} else {
		status.LastSuccess = time.Now()
		status.LastError = ""
		status.IsHealthy = true
	}
}

// GetCurrentPrices returns all current prices
func (a *Aggregator) GetCurrentPrices() map[string]*types.PriceData {
	a.pricesMu.RLock()
	defer a.pricesMu.RUnlock()

	result := make(map[string]*types.PriceData)
	for k, v := range a.prices {
		result[k] = v
	}
	return result
}

// GetAssetPrice returns the current price for a specific asset
func (a *Aggregator) GetAssetPrice(asset string) (*types.PriceData, error) {
	a.pricesMu.RLock()
	defer a.pricesMu.RUnlock()

	price, exists := a.prices[asset]
	if !exists {
		return nil, fmt.Errorf("price not found for asset: %s", asset)
	}

	// Check if price is stale
	if time.Since(time.Unix(price.Timestamp, 0)) > a.staleThreshold {
		return nil, fmt.Errorf("price is stale for asset: %s", asset)
	}

	return price, nil
}

// GetPriceHistory returns price history for an asset
func (a *Aggregator) GetPriceHistory(asset string, limit int) (*types.PriceHistory, error) {
	records, err := a.database.GetPriceHistory(asset, limit)
	if err != nil {
		return nil, err
	}

	history := &types.PriceHistory{
		Asset:  asset,
		Prices: make([]types.PricePoint, len(records)),
	}

	for i, record := range records {
		price := new(big.Int)
		price.SetString(record.Price, 10)
		history.Prices[i] = types.PricePoint{
			Price:     price,
			Timestamp: record.Timestamp.Unix(),
		}
	}

	return history, nil
}

// GetStats returns aggregator statistics
func (a *Aggregator) GetStats() *AggregatorStats {
	a.statsMu.RLock()
	defer a.statsMu.RUnlock()

	// Count active sources
	a.sourceStatusMu.RLock()
	activeCount := 0
	for _, status := range a.sourceStatus {
		if status.IsHealthy {
			activeCount++
		}
	}
	a.sourceStatusMu.RUnlock()

	return &AggregatorStats{
		TotalUpdates:      a.stats.TotalUpdates,
		SuccessfulUpdates: a.stats.SuccessfulUpdates,
		FailedUpdates:     a.stats.FailedUpdates,
		LastUpdateTime:    a.stats.LastUpdateTime,
		AverageLatencyMs:  a.stats.AverageLatencyMs,
		ActiveSources:     activeCount,
	}
}

// GetSourceStatus returns the status of all price sources
func (a *Aggregator) GetSourceStatus() map[string]*SourceStatus {
	a.sourceStatusMu.RLock()
	defer a.sourceStatusMu.RUnlock()

	result := make(map[string]*SourceStatus)
	for k, v := range a.sourceStatus {
		result[k] = &SourceStatus{
			Name:        v.Name,
			LastSuccess: v.LastSuccess,
			LastError:   v.LastError,
			SuccessRate: v.SuccessRate,
			Latency:     v.Latency,
			IsHealthy:   v.IsHealthy,
		}
	}
	return result
}

// SignPrice signs a price for submission
func (a *Aggregator) SignPrice(price *types.PriceData, privateKey []byte) error {
	priceHash := crypto.Keccak256([]byte(fmt.Sprintf("%s:%s:%d",
		price.Asset, price.Price.String(), price.Timestamp)))

	// In production, load the actual private key
	_ = priceHash
	return nil
}

// Shutdown gracefully shuts down the aggregator
func (a *Aggregator) Shutdown(ctx context.Context) error {
	close(a.shutdownCh)
	return nil
}
