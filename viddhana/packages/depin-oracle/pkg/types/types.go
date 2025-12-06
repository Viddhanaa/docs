// Package types defines core data types for the DePIN Oracle network.
package types

import (
	"math/big"

	"github.com/spf13/viper"
)

// Config holds the application configuration
type Config struct {
	// Validator settings
	ValidatorIndex     int    `mapstructure:"validator_index"`
	TotalValidators    int    `mapstructure:"total_validators"`
	RequiredSignatures int    `mapstructure:"required_signatures"`
	PrivateKeyPath     string `mapstructure:"private_key_path"`

	// Chain settings
	ChainRPC       string `mapstructure:"chain_rpc"`
	OracleContract string `mapstructure:"oracle_contract"`

	// P2P settings
	P2P P2PConfig `mapstructure:"p2p"`

	// Database settings
	Database DatabaseConfig `mapstructure:"database"`

	// Timing settings
	ConsensusTimeoutSec int `mapstructure:"consensus_timeout_sec"`

	// Health server
	HealthPort int `mapstructure:"health_port"`

	// Aggregator settings
	AggregatorPort     int            `mapstructure:"aggregator_port"`
	AggregatorInterval int            `mapstructure:"aggregator_interval"`
	StaleThreshold     int            `mapstructure:"stale_threshold"`
	DeviationLimit     float64        `mapstructure:"deviation_limit"`
	MinPriceSources    int            `mapstructure:"min_price_sources"`
	PriceSources       PriceSourceCfg `mapstructure:"price_sources"`
}

// P2PConfig holds P2P network configuration
type P2PConfig struct {
	ListenAddr     string   `mapstructure:"listen_addr"`
	BootstrapPeers []string `mapstructure:"bootstrap_peers"`
}

// DatabaseConfig holds database configuration
type DatabaseConfig struct {
	Host     string `mapstructure:"host"`
	Port     int    `mapstructure:"port"`
	User     string `mapstructure:"user"`
	Password string `mapstructure:"password"`
	DBName   string `mapstructure:"dbname"`
	SSLMode  string `mapstructure:"sslmode"`
}

// PriceSourceCfg holds price source URLs
type PriceSourceCfg struct {
	Binance   string `mapstructure:"binance"`
	Coinbase  string `mapstructure:"coinbase"`
	Kraken    string `mapstructure:"kraken"`
	CoinGecko string `mapstructure:"coingecko"`
	Chainlink string `mapstructure:"chainlink"`
}

// LoadConfig loads configuration from a file
func LoadConfig(path string) (*Config, error) {
	viper.SetConfigFile(path)
	viper.SetConfigType("yaml")

	// Set defaults
	viper.SetDefault("validator_index", 0)
	viper.SetDefault("total_validators", 13)
	viper.SetDefault("required_signatures", 9)
	viper.SetDefault("consensus_timeout_sec", 30)
	viper.SetDefault("health_port", 8080)
	viper.SetDefault("aggregator_port", 8081)
	viper.SetDefault("aggregator_interval", 60)
	viper.SetDefault("stale_threshold", 300)
	viper.SetDefault("deviation_limit", 0.05)
	viper.SetDefault("min_price_sources", 3)

	if err := viper.ReadInConfig(); err != nil {
		return nil, err
	}

	var cfg Config
	if err := viper.Unmarshal(&cfg); err != nil {
		return nil, err
	}

	return &cfg, nil
}

// SensorData represents data from an IoT sensor
type SensorData struct {
	SensorID     string       `json:"sensor_id"`
	DeviceType   string       `json:"device_type"`
	DevicePubKey string       `json:"device_pubkey"`
	Timestamp    int64        `json:"timestamp"`
	GPS          GPSCoord     `json:"gps"`
	Metrics      SensorMetric `json:"metrics"`
	Signature    string       `json:"signature"`
}

// GPSCoord represents GPS coordinates
type GPSCoord struct {
	Latitude  float64 `json:"latitude"`
	Longitude float64 `json:"longitude"`
	Accuracy  float64 `json:"accuracy"`
}

// SensorMetric represents sensor metrics
type SensorMetric struct {
	EnergyOutput float64 `json:"energy_output"`
	Uptime       float64 `json:"uptime"`
	Temperature  float64 `json:"temperature"`
	Humidity     float64 `json:"humidity"`
}

// ValidationResult represents the result of sensor data validation
type ValidationResult struct {
	SensorID      string `json:"sensor_id"`
	Timestamp     int64  `json:"timestamp"`
	IsValid       bool   `json:"is_valid"`
	Confidence    float64 `json:"confidence"`
	FailureReason string `json:"failure_reason,omitempty"`
	ValidatorSig  []byte `json:"validator_sig,omitempty"`
}

// ConsensusVote represents a validator's vote in a consensus round
type ConsensusVote struct {
	RoundID        string `json:"round_id"`
	SensorID       string `json:"sensor_id"`
	DataHash       string `json:"data_hash"`
	ValidatorIndex int    `json:"validator_index"`
	Signature      string `json:"signature"`
	Timestamp      int64  `json:"timestamp"`
}

// PriceData represents aggregated price data
type PriceData struct {
	Asset     string   `json:"asset"`
	Price     *big.Int `json:"price"`
	Timestamp int64    `json:"timestamp"`
	Source    string   `json:"source"`
	Signature []byte   `json:"signature,omitempty"`
}

// PriceHistory represents historical price data
type PriceHistory struct {
	Asset   string       `json:"asset"`
	Prices  []PricePoint `json:"prices"`
}

// PricePoint represents a single price point
type PricePoint struct {
	Price     *big.Int `json:"price"`
	Timestamp int64    `json:"timestamp"`
}
