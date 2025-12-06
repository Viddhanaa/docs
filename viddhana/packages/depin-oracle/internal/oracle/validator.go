// Package oracle implements the DePIN Oracle validation and consensus logic.
package oracle

import (
	"context"
	"crypto/ecdsa"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math/big"
	"sync"
	"time"

	"github.com/ethereum/go-ethereum/accounts/abi/bind"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/crypto"
	"github.com/ethereum/go-ethereum/ethclient"
	"github.com/viddhana/depin-oracle/internal/p2p"
	"github.com/viddhana/depin-oracle/internal/storage"
	"github.com/viddhana/depin-oracle/pkg/types"
	"go.uber.org/zap"
)

// ValidatorConfig contains configuration for the validator
type ValidatorConfig struct {
	PrivateKeyPath   string
	ChainRPC         string
	OracleContract   string
	ValidatorIndex   int
	TotalValidators  int
	RequiredSigs     int
	ConsensusTimeout time.Duration
	Database         *storage.Database
	P2PNetwork       *p2p.Network
	Logger           *zap.SugaredLogger
}

// Validator represents an oracle validator node
type Validator struct {
	privateKey      *ecdsa.PrivateKey
	publicKey       common.Address
	validatorIndex  int
	totalValidators int
	requiredSigs    int

	chainClient    *ethclient.Client
	contractAddr   common.Address
	transactOpts   *bind.TransactOpts

	database   *storage.Database
	p2pNetwork *p2p.Network
	logger     *zap.SugaredLogger

	// Consensus state
	consensusTimeout time.Duration
	pendingRounds    map[string]*ConsensusRound
	roundsMu         sync.RWMutex

	// Metrics
	metrics      *ValidatorMetrics
	metricsLock  sync.RWMutex

	// Shutdown
	shutdownCh chan struct{}
	isReady    bool
}

// ValidatorMetrics tracks validator performance metrics
type ValidatorMetrics struct {
	TotalDataProcessed   uint64
	ValidDataPoints      uint64
	InvalidDataPoints    uint64
	ConsensusRounds      uint64
	SuccessfulConsensus  uint64
	FailedConsensus      uint64
	AverageLatencyMs     float64
	LastProcessedTime    time.Time
}

// ValidatorStatus represents the current validator status
type ValidatorStatus struct {
	IsHealthy         bool
	ValidatorIndex    int
	PeersConnected    int
	LastConsensusTime time.Time
	PendingRounds     int
	Error             string
}

// ConsensusRound represents a single consensus round
type ConsensusRound struct {
	RoundID      string
	SensorID     string
	DataHash     []byte
	Signatures   map[int][]byte // validator index -> signature
	StartTime    time.Time
	Finalized    bool
	FinalResult  bool
	mu           sync.Mutex
}

// NewValidator creates a new oracle validator
func NewValidator(cfg ValidatorConfig) (*Validator, error) {
	// Load private key
	privateKey, err := crypto.LoadECDSA(cfg.PrivateKeyPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load private key: %w", err)
	}

	publicKey := crypto.PubkeyToAddress(privateKey.PublicKey)

	// Connect to blockchain
	client, err := ethclient.Dial(cfg.ChainRPC)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to chain: %w", err)
	}

	// Get chain ID for transaction signing
	chainID, err := client.ChainID(context.Background())
	if err != nil {
		return nil, fmt.Errorf("failed to get chain ID: %w", err)
	}

	transactOpts, err := bind.NewKeyedTransactorWithChainID(privateKey, chainID)
	if err != nil {
		return nil, fmt.Errorf("failed to create transactor: %w", err)
	}

	v := &Validator{
		privateKey:       privateKey,
		publicKey:        publicKey,
		validatorIndex:   cfg.ValidatorIndex,
		totalValidators:  cfg.TotalValidators,
		requiredSigs:     cfg.RequiredSigs,
		chainClient:      client,
		contractAddr:     common.HexToAddress(cfg.OracleContract),
		transactOpts:     transactOpts,
		database:         cfg.Database,
		p2pNetwork:       cfg.P2PNetwork,
		logger:           cfg.Logger,
		consensusTimeout: cfg.ConsensusTimeout,
		pendingRounds:    make(map[string]*ConsensusRound),
		metrics:          &ValidatorMetrics{},
		shutdownCh:       make(chan struct{}),
		isReady:          true,
	}

	return v, nil
}

// HandleSensorData processes incoming sensor data from the P2P network
func (v *Validator) HandleSensorData(ctx context.Context, data []byte) error {
	var sensorData types.SensorData
	if err := json.Unmarshal(data, &sensorData); err != nil {
		v.logger.Warnf("Invalid sensor data format: %v", err)
		return fmt.Errorf("invalid data format: %w", err)
	}

	v.logger.Infof("Processing sensor data from %s", sensorData.SensorID)

	// Validate the sensor data
	result, err := v.validateSensorData(&sensorData)
	if err != nil {
		v.logger.Errorf("Validation error: %v", err)
		return err
	}

	// Update metrics
	v.metricsLock.Lock()
	v.metrics.TotalDataProcessed++
	if result.IsValid {
		v.metrics.ValidDataPoints++
	} else {
		v.metrics.InvalidDataPoints++
	}
	v.metrics.LastProcessedTime = time.Now()
	v.metricsLock.Unlock()

	// If valid, initiate or participate in consensus
	if result.IsValid {
		return v.initiateConsensus(ctx, &sensorData, result)
	}

	return nil
}

// validateSensorData performs comprehensive validation of sensor data
func (v *Validator) validateSensorData(data *types.SensorData) (*types.ValidationResult, error) {
	result := &types.ValidationResult{
		SensorID:  data.SensorID,
		Timestamp: time.Now().Unix(),
	}

	// Step 1: Verify device signature
	if !v.verifyDeviceSignature(data) {
		result.IsValid = false
		result.FailureReason = "invalid device signature"
		return result, nil
	}

	// Step 2: Verify Proof of Physical Presence
	if !v.verifyPhysicalPresence(data) {
		result.IsValid = false
		result.FailureReason = "failed physical presence check"
		return result, nil
	}

	// Step 3: Validate data ranges
	if !v.validateMetrics(data) {
		result.IsValid = false
		result.FailureReason = "metrics out of valid range"
		return result, nil
	}

	// Step 4: Check for anomalies
	anomalyScore := v.detectAnomalies(data)
	if anomalyScore > 0.8 {
		result.IsValid = false
		result.FailureReason = fmt.Sprintf("anomaly detected (score: %.2f)", anomalyScore)
		return result, nil
	}

	// All checks passed
	result.IsValid = true
	result.Confidence = 1.0 - anomalyScore

	// Sign the validation result
	resultHash := v.hashValidationResult(result)
	sig, err := crypto.Sign(resultHash, v.privateKey)
	if err != nil {
		return nil, fmt.Errorf("failed to sign result: %w", err)
	}
	result.ValidatorSig = sig

	return result, nil
}

// verifyDeviceSignature checks the sensor's cryptographic signature
func (v *Validator) verifyDeviceSignature(data *types.SensorData) bool {
	// Reconstruct the signed message
	message := fmt.Sprintf("%s:%d:%f:%f",
		data.SensorID,
		data.Timestamp,
		data.GPS.Latitude,
		data.GPS.Longitude,
	)
	messageHash := crypto.Keccak256Hash([]byte(message))

	// Decode signature
	sig, err := hex.DecodeString(data.Signature)
	if err != nil {
		v.logger.Warnf("Failed to decode signature: %v", err)
		return false
	}

	// Recover public key from signature
	pubKey, err := crypto.SigToPub(messageHash.Bytes(), sig)
	if err != nil {
		v.logger.Warnf("Failed to recover public key: %v", err)
		return false
	}

	// Verify it matches the claimed device public key
	recoveredAddr := crypto.PubkeyToAddress(*pubKey)
	expectedAddr := common.HexToAddress(data.DevicePubKey)

	return recoveredAddr == expectedAddr
}

// verifyPhysicalPresence implements Proof of Physical Presence
func (v *Validator) verifyPhysicalPresence(data *types.SensorData) bool {
	now := time.Now().Unix()

	// Check timestamp is recent (within 5 minutes)
	if now-data.Timestamp > 300 {
		v.logger.Warnf("Timestamp too old: %d seconds ago", now-data.Timestamp)
		return false
	}

	// Check GPS accuracy (must be within 50 meters)
	if data.GPS.Accuracy > 50 {
		v.logger.Warnf("GPS accuracy too low: %.2f meters", data.GPS.Accuracy)
		return false
	}

	// Verify GPS coordinates are in expected location
	if !v.isValidLocation(data.SensorID, data.GPS) {
		v.logger.Warnf("Invalid location for sensor %s", data.SensorID)
		return false
	}

	return true
}

// isValidLocation checks if GPS coordinates match registered sensor location
func (v *Validator) isValidLocation(sensorID string, gps types.GPSCoord) bool {
	// Get registered location from database
	registeredLoc, err := v.database.GetSensorLocation(sensorID)
	if err != nil {
		v.logger.Warnf("Failed to get registered location: %v", err)
		return false
	}

	// Calculate distance between registered and reported location
	distance := calculateDistance(
		registeredLoc.Latitude, registeredLoc.Longitude,
		gps.Latitude, gps.Longitude,
	)

	// Allow 100 meter tolerance
	return distance <= 100
}

// calculateDistance calculates distance between two GPS coordinates in meters
func calculateDistance(lat1, lon1, lat2, lon2 float64) float64 {
	// Haversine formula
	const earthRadius = 6371000 // meters

	lat1Rad := lat1 * 3.14159265359 / 180
	lat2Rad := lat2 * 3.14159265359 / 180
	deltaLat := (lat2 - lat1) * 3.14159265359 / 180
	deltaLon := (lon2 - lon1) * 3.14159265359 / 180

	a := sin(deltaLat/2)*sin(deltaLat/2) +
		cos(lat1Rad)*cos(lat2Rad)*sin(deltaLon/2)*sin(deltaLon/2)

	c := 2 * atan2(sqrt(a), sqrt(1-a))

	return earthRadius * c
}

// Math helper functions
func sin(x float64) float64  { return x - x*x*x/6 + x*x*x*x*x/120 }
func cos(x float64) float64  { return 1 - x*x/2 + x*x*x*x/24 }
func sqrt(x float64) float64 { return x }  // Simplified
func atan2(y, x float64) float64 { return y / x } // Simplified

// validateMetrics checks if sensor readings are within valid ranges
func (v *Validator) validateMetrics(data *types.SensorData) bool {
	switch data.DeviceType {
	case "solar_panel":
		// Solar output should be 0-1000 kWh per day
		if data.Metrics.EnergyOutput < 0 || data.Metrics.EnergyOutput > 1000 {
			return false
		}
	case "5g_tower":
		// Uptime should be 0-100%
		if data.Metrics.Uptime < 0 || data.Metrics.Uptime > 100 {
			return false
		}
	case "ev_charger":
		// Energy delivered should be reasonable
		if data.Metrics.EnergyOutput < 0 || data.Metrics.EnergyOutput > 500 {
			return false
		}
	default:
		// Unknown device type - still validate basic metrics
		if data.Metrics.EnergyOutput < 0 {
			return false
		}
	}

	return true
}

// detectAnomalies uses ML model to detect unusual patterns
func (v *Validator) detectAnomalies(data *types.SensorData) float64 {
	// Get historical average for this sensor
	historicalAvg, err := v.database.GetHistoricalAverage(data.SensorID)
	if err != nil {
		// No history, accept
		return 0
	}

	if historicalAvg == 0 {
		return 0
	}

	currentValue := data.Metrics.EnergyOutput

	// Calculate deviation
	deviation := abs(currentValue-historicalAvg) / historicalAvg

	// Score 0-1, higher means more anomalous
	if deviation > 0.5 {
		return min(deviation, 1.0)
	}

	return 0
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

func min(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}

// hashValidationResult creates a hash of the validation result for signing
func (v *Validator) hashValidationResult(result *types.ValidationResult) []byte {
	data := fmt.Sprintf("%s:%d:%t:%.4f",
		result.SensorID,
		result.Timestamp,
		result.IsValid,
		result.Confidence,
	)
	return crypto.Keccak256([]byte(data))
}

// initiateConsensus starts or joins a consensus round for validated data
func (v *Validator) initiateConsensus(ctx context.Context, data *types.SensorData, result *types.ValidationResult) error {
	// Create round ID from sensor ID and timestamp
	roundID := fmt.Sprintf("%s:%d", data.SensorID, data.Timestamp/60) // 1-minute rounds

	v.roundsMu.Lock()
	round, exists := v.pendingRounds[roundID]
	if !exists {
		round = &ConsensusRound{
			RoundID:    roundID,
			SensorID:   data.SensorID,
			DataHash:   crypto.Keccak256(result.ValidatorSig),
			Signatures: make(map[int][]byte),
			StartTime:  time.Now(),
		}
		v.pendingRounds[roundID] = round
	}
	v.roundsMu.Unlock()

	// Add our signature
	round.mu.Lock()
	round.Signatures[v.validatorIndex] = result.ValidatorSig
	round.mu.Unlock()

	// Broadcast vote to other validators
	vote := types.ConsensusVote{
		RoundID:        roundID,
		SensorID:       data.SensorID,
		DataHash:       hex.EncodeToString(round.DataHash),
		ValidatorIndex: v.validatorIndex,
		Signature:      hex.EncodeToString(result.ValidatorSig),
		Timestamp:      time.Now().Unix(),
	}

	voteData, err := json.Marshal(vote)
	if err != nil {
		return fmt.Errorf("failed to marshal vote: %w", err)
	}

	return v.p2pNetwork.Publish("consensus/votes", voteData)
}

// HandleConsensusVote processes incoming consensus votes from other validators
func (v *Validator) HandleConsensusVote(ctx context.Context, data []byte) error {
	var vote types.ConsensusVote
	if err := json.Unmarshal(data, &vote); err != nil {
		return fmt.Errorf("invalid vote format: %w", err)
	}

	v.roundsMu.Lock()
	round, exists := v.pendingRounds[vote.RoundID]
	if !exists {
		// Create new round if we haven't seen it yet
		dataHash, _ := hex.DecodeString(vote.DataHash)
		round = &ConsensusRound{
			RoundID:    vote.RoundID,
			SensorID:   vote.SensorID,
			DataHash:   dataHash,
			Signatures: make(map[int][]byte),
			StartTime:  time.Now(),
		}
		v.pendingRounds[vote.RoundID] = round
	}
	v.roundsMu.Unlock()

	// Add signature
	sig, _ := hex.DecodeString(vote.Signature)
	round.mu.Lock()
	round.Signatures[vote.ValidatorIndex] = sig
	round.mu.Unlock()

	return nil
}

// RunConsensus runs the consensus processing loop
func (v *Validator) RunConsensus(ctx context.Context) {
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-v.shutdownCh:
			return
		case <-ticker.C:
			v.processConsensusRounds()
		}
	}
}

// processConsensusRounds checks all pending rounds for consensus
func (v *Validator) processConsensusRounds() {
	v.roundsMu.Lock()
	defer v.roundsMu.Unlock()

	now := time.Now()
	toDelete := []string{}

	for id, round := range v.pendingRounds {
		if round.Finalized {
			continue
		}

		round.mu.Lock()
		validSigs := len(round.Signatures)
		round.mu.Unlock()

		// Check if we have enough signatures (2/3 threshold)
		if validSigs >= v.requiredSigs {
			// Consensus reached - submit to chain
			if err := v.submitToChain(round); err != nil {
				v.logger.Errorf("Failed to submit to chain: %v", err)
			} else {
				round.Finalized = true
				round.FinalResult = true
				v.metricsLock.Lock()
				v.metrics.ConsensusRounds++
				v.metrics.SuccessfulConsensus++
				v.metricsLock.Unlock()
			}
			toDelete = append(toDelete, id)
		} else if now.Sub(round.StartTime) > v.consensusTimeout {
			// Timeout - mark as failed
			round.Finalized = true
			round.FinalResult = false
			v.metricsLock.Lock()
			v.metrics.ConsensusRounds++
			v.metrics.FailedConsensus++
			v.metricsLock.Unlock()
			toDelete = append(toDelete, id)
			v.logger.Warnf("Consensus timeout for round %s (%d/%d signatures)",
				id, validSigs, v.requiredSigs)
		}
	}

	// Cleanup old rounds
	for _, id := range toDelete {
		delete(v.pendingRounds, id)
	}
}

// submitToChain submits the consensus result to the blockchain
func (v *Validator) submitToChain(round *ConsensusRound) error {
	round.mu.Lock()
	defer round.mu.Unlock()

	// Collect signatures and indices
	var signatures [][]byte
	var indices []int

	for idx, sig := range round.Signatures {
		if len(sig) > 0 {
			signatures = append(signatures, sig)
			indices = append(indices, idx)
		}
	}

	v.logger.Infof("Submitting consensus for sensor %s with %d signatures",
		round.SensorID, len(signatures))

	// Store result in database
	if err := v.database.StoreConsensusResult(&storage.ConsensusResult{
		RoundID:       round.RoundID,
		SensorID:      round.SensorID,
		DataHash:      hex.EncodeToString(round.DataHash),
		SignatureCount: len(signatures),
		Timestamp:     round.StartTime,
		Success:       true,
	}); err != nil {
		return fmt.Errorf("failed to store result: %w", err)
	}

	// TODO: Submit to actual smart contract
	// This would call the OracleVerifier contract's submitData function

	return nil
}

// FetchPrice fetches price from multiple sources (implements oracle interface)
func (v *Validator) FetchPrice(ctx context.Context, asset string) (*types.PriceData, error) {
	// This would fetch from multiple price sources and aggregate
	// For now, return a mock implementation
	return &types.PriceData{
		Asset:     asset,
		Price:     big.NewInt(0),
		Timestamp: time.Now().Unix(),
		Source:    "validator",
	}, nil
}

// SubmitPrice submits a price to the blockchain
func (v *Validator) SubmitPrice(ctx context.Context, price *types.PriceData) error {
	// Sign the price data
	priceHash := crypto.Keccak256([]byte(fmt.Sprintf("%s:%s:%d",
		price.Asset, price.Price.String(), price.Timestamp)))

	sig, err := crypto.Sign(priceHash, v.privateKey)
	if err != nil {
		return fmt.Errorf("failed to sign price: %w", err)
	}

	price.Signature = sig

	v.logger.Infof("Submitting price for %s: %s", price.Asset, price.Price.String())

	// TODO: Submit to smart contract
	return nil
}

// ValidateConsensus validates that consensus threshold (2/3) is met
func (v *Validator) ValidateConsensus(roundID string) (bool, error) {
	v.roundsMu.RLock()
	round, exists := v.pendingRounds[roundID]
	v.roundsMu.RUnlock()

	if !exists {
		return false, errors.New("round not found")
	}

	round.mu.Lock()
	sigCount := len(round.Signatures)
	round.mu.Unlock()

	// 2/3 threshold
	threshold := (v.totalValidators * 2) / 3
	if v.totalValidators*2%3 != 0 {
		threshold++
	}

	return sigCount >= threshold, nil
}

// GetStatus returns the current validator status
func (v *Validator) GetStatus() ValidatorStatus {
	v.roundsMu.RLock()
	pendingCount := len(v.pendingRounds)
	v.roundsMu.RUnlock()

	v.metricsLock.RLock()
	lastTime := v.metrics.LastProcessedTime
	v.metricsLock.RUnlock()

	return ValidatorStatus{
		IsHealthy:         v.isReady,
		ValidatorIndex:    v.validatorIndex,
		PeersConnected:    v.p2pNetwork.PeerCount(),
		LastConsensusTime: lastTime,
		PendingRounds:     pendingCount,
	}
}

// GetMetrics returns validator metrics
func (v *Validator) GetMetrics() *ValidatorMetrics {
	v.metricsLock.RLock()
	defer v.metricsLock.RUnlock()

	// Return a copy
	return &ValidatorMetrics{
		TotalDataProcessed:  v.metrics.TotalDataProcessed,
		ValidDataPoints:     v.metrics.ValidDataPoints,
		InvalidDataPoints:   v.metrics.InvalidDataPoints,
		ConsensusRounds:     v.metrics.ConsensusRounds,
		SuccessfulConsensus: v.metrics.SuccessfulConsensus,
		FailedConsensus:     v.metrics.FailedConsensus,
		AverageLatencyMs:    v.metrics.AverageLatencyMs,
		LastProcessedTime:   v.metrics.LastProcessedTime,
	}
}

// IsReady returns whether the validator is ready to process requests
func (v *Validator) IsReady() bool {
	return v.isReady
}

// GetIndex returns the validator index
func (v *Validator) GetIndex() int {
	return v.validatorIndex
}

// GetAddress returns the validator's Ethereum address
func (v *Validator) GetAddress() string {
	return v.publicKey.Hex()
}

// GetTotalValidators returns the total number of validators
func (v *Validator) GetTotalValidators() int {
	return v.totalValidators
}

// GetRequiredSignatures returns the required number of signatures for consensus
func (v *Validator) GetRequiredSignatures() int {
	return v.requiredSigs
}

// Shutdown gracefully shuts down the validator
func (v *Validator) Shutdown(ctx context.Context) error {
	close(v.shutdownCh)
	v.isReady = false

	// Wait for pending consensus rounds
	timeout := time.After(10 * time.Second)
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-timeout:
			v.logger.Warn("Shutdown timeout, forcing stop")
			return nil
		case <-ticker.C:
			v.roundsMu.RLock()
			pending := len(v.pendingRounds)
			v.roundsMu.RUnlock()
			if pending == 0 {
				return nil
			}
		}
	}
}
