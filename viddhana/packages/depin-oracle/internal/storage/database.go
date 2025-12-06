// Package storage implements database operations for the DePIN Oracle.
package storage

import (
	"fmt"
	"time"

	"github.com/viddhana/depin-oracle/pkg/types"
	"gorm.io/driver/postgres"
	"gorm.io/gorm"
)

// Database wraps the GORM database connection
type Database struct {
	db *gorm.DB
}

// SensorLocation represents a registered sensor location
type SensorLocation struct {
	ID        uint    `gorm:"primaryKey"`
	SensorID  string  `gorm:"uniqueIndex"`
	Latitude  float64
	Longitude float64
	CreatedAt time.Time
	UpdatedAt time.Time
}

// SensorHistory represents historical sensor data
type SensorHistory struct {
	ID           uint `gorm:"primaryKey"`
	SensorID     string
	EnergyOutput float64
	Timestamp    time.Time
	CreatedAt    time.Time
}

// ConsensusResult represents a stored consensus result
type ConsensusResult struct {
	ID             uint   `gorm:"primaryKey"`
	RoundID        string `gorm:"uniqueIndex"`
	SensorID       string
	DataHash       string
	SignatureCount int
	Timestamp      time.Time
	Success        bool
	CreatedAt      time.Time
}

// PriceRecord represents a stored price record
type PriceRecord struct {
	ID        uint   `gorm:"primaryKey"`
	Asset     string `gorm:"index"`
	Price     string
	Source    string
	Timestamp time.Time
	CreatedAt time.Time
}

// NewDatabase creates a new database connection
func NewDatabase(cfg types.DatabaseConfig) (*Database, error) {
	dsn := fmt.Sprintf(
		"host=%s port=%d user=%s password=%s dbname=%s sslmode=%s",
		cfg.Host, cfg.Port, cfg.User, cfg.Password, cfg.DBName, cfg.SSLMode,
	)

	db, err := gorm.Open(postgres.Open(dsn), &gorm.Config{})
	if err != nil {
		return nil, fmt.Errorf("failed to connect to database: %w", err)
	}

	// Auto-migrate tables
	if err := db.AutoMigrate(
		&SensorLocation{},
		&SensorHistory{},
		&ConsensusResult{},
		&PriceRecord{},
	); err != nil {
		return nil, fmt.Errorf("failed to migrate database: %w", err)
	}

	return &Database{db: db}, nil
}

// GetSensorLocation retrieves the registered location for a sensor
func (d *Database) GetSensorLocation(sensorID string) (*SensorLocation, error) {
	var loc SensorLocation
	if err := d.db.Where("sensor_id = ?", sensorID).First(&loc).Error; err != nil {
		return nil, err
	}
	return &loc, nil
}

// GetHistoricalAverage returns the average energy output for a sensor
func (d *Database) GetHistoricalAverage(sensorID string) (float64, error) {
	var avg float64
	result := d.db.Model(&SensorHistory{}).
		Where("sensor_id = ?", sensorID).
		Select("COALESCE(AVG(energy_output), 0)").
		Scan(&avg)
	if result.Error != nil {
		return 0, result.Error
	}
	return avg, nil
}

// StoreConsensusResult stores a consensus result
func (d *Database) StoreConsensusResult(result *ConsensusResult) error {
	return d.db.Create(result).Error
}

// StoreSensorData stores sensor data history
func (d *Database) StoreSensorData(sensorID string, energyOutput float64, timestamp time.Time) error {
	return d.db.Create(&SensorHistory{
		SensorID:     sensorID,
		EnergyOutput: energyOutput,
		Timestamp:    timestamp,
	}).Error
}

// StorePriceRecord stores a price record
func (d *Database) StorePriceRecord(asset, price, source string, timestamp time.Time) error {
	return d.db.Create(&PriceRecord{
		Asset:     asset,
		Price:     price,
		Source:    source,
		Timestamp: timestamp,
	}).Error
}

// GetLatestPrice retrieves the latest price for an asset
func (d *Database) GetLatestPrice(asset string) (*PriceRecord, error) {
	var record PriceRecord
	if err := d.db.Where("asset = ?", asset).Order("timestamp DESC").First(&record).Error; err != nil {
		return nil, err
	}
	return &record, nil
}

// GetPriceHistory retrieves price history for an asset
func (d *Database) GetPriceHistory(asset string, limit int) ([]PriceRecord, error) {
	var records []PriceRecord
	if err := d.db.Where("asset = ?", asset).Order("timestamp DESC").Limit(limit).Find(&records).Error; err != nil {
		return nil, err
	}
	return records, nil
}

// Close closes the database connection
func (d *Database) Close() error {
	sqlDB, err := d.db.DB()
	if err != nil {
		return err
	}
	return sqlDB.Close()
}
