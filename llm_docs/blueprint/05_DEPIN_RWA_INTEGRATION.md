# DePIN & RWA Integration Guide

> Detailed implementation guide for Decentralized Physical Infrastructure and Real World Asset integration

---

## Table of Contents
1. [Overview](#overview)
2. [DePIN Architecture](#depin-architecture)
3. [Oracle Validator Network](#oracle-validator-network)
4. [IoT Data Pipeline](#iot-data-pipeline)
5. [RWA Tokenization](#rwa-tokenization)
6. [Reward Calculation](#reward-calculation)
7. [Smart Contract Implementation](#smart-contract-implementation)
8. [Testing & Validation](#testing--validation)

---

## Overview

VIDDHANA integrates physical infrastructure and real-world assets through:
- **DePIN Network**: IoT sensors for solar panels, 5G towers, etc.
- **Oracle Validators**: 13-node network for data verification
- **RWA Tokenization**: NFT standards for real estate and infrastructure
- **Proof of Physical Presence**: GPS + Timestamp + Device Signature validation

### Core Formula - DePIN Reward Calculation

$$Reward_{sensor} = Base\_Fee + Data\_Quality\_Bonus - Penalty$$

Where:
- $Base\_Fee$: Fixed rate ($0.50/day)
- $Data\_Quality\_Bonus$: +$0.10 if uptime > 99%
- $Penalty$: -$0.05 for erroneous data or downtime

---

## DePIN Architecture

### System Diagram

```
+------------------------------------------------------------------+
|                     DePIN INFRASTRUCTURE                          |
+------------------------------------------------------------------+
|                                                                   |
|  +-------------------+     +-------------------+                  |
|  |  Physical Assets  |     |   IoT Sensors     |                  |
|  |  - Solar Panels   |---->|   - Energy Meter  |                  |
|  |  - 5G Towers      |     |   - GPS Module    |                  |
|  |  - EV Chargers    |     |   - Signature IC  |                  |
|  +-------------------+     +-------------------+                  |
|                                    |                              |
|                                    v (MQTT/TLS)                   |
|                        +-------------------+                      |
|                        | Data Aggregation  |                      |
|                        | Layer (Kafka)     |                      |
|                        +-------------------+                      |
|                                    |                              |
|                                    v                              |
|  +-------------------+     +-------------------+                  |
|  | Validator Node 1  |     | Validator Node 2  |  ... (13 nodes) |
|  | - Data Validation |     | - Data Validation |                  |
|  | - Consensus Vote  |     | - Consensus Vote  |                  |
|  +-------------------+     +-------------------+                  |
|           |                        |                              |
|           +------------------------+                              |
|                        |                                          |
|                        v (9/13 Signatures Required)               |
|              +-------------------+                                |
|              | OracleVerifier    |                                |
|              | Smart Contract    |                                |
|              +-------------------+                                |
|                        |                                          |
|                        v                                          |
|              +-------------------+                                |
|              | Reward            |                                |
|              | Distribution      |                                |
|              +-------------------+                                |
|                                                                   |
+------------------------------------------------------------------+
```

### Data Flow

```
1. Physical Asset generates data (energy output, uptime, etc.)
2. IoT Sensor captures + signs data with device key
3. Data sent via MQTT to aggregation layer
4. Validators independently verify data
5. 9/13 validators must agree for consensus
6. Verified data written to Atlas Chain
7. Rewards distributed to sensor operators
```

---

## Oracle Validator Network

### Validator Node Implementation

```go
// cmd/validator/main.go
package main

import (
    "context"
    "crypto/ecdsa"
    "log"
    "time"

    "github.com/viddhana/depin-oracle/internal/consensus"
    "github.com/viddhana/depin-oracle/internal/validator"
    "github.com/viddhana/depin-oracle/pkg/mqtt"
    "github.com/viddhana/depin-oracle/pkg/chain"
)

func main() {
    cfg := loadConfig()
    
    // Initialize validator
    v, err := validator.New(validator.Config{
        PrivateKey:     cfg.PrivateKey,
        ChainRPC:       cfg.AtlasRPCURL,
        ContractAddr:   cfg.OracleContractAddr,
        ValidatorIndex: cfg.ValidatorIndex,
    })
    if err != nil {
        log.Fatal(err)
    }
    
    // Connect to MQTT broker
    mqttClient := mqtt.NewClient(cfg.MQTTBroker)
    if err := mqttClient.Connect(); err != nil {
        log.Fatal(err)
    }
    
    // Subscribe to sensor data topics
    mqttClient.Subscribe("sensors/+/data", v.HandleSensorData)
    
    // Start consensus participation
    consensusEngine := consensus.New(v, cfg.Validators)
    go consensusEngine.Run(context.Background())
    
    // Start health check server
    go startHealthServer(cfg.HealthPort)
    
    log.Println("Validator node started")
    select {}
}
```

### Validator Core Logic

```go
// internal/validator/validator.go
package validator

import (
    "crypto/ecdsa"
    "encoding/json"
    "fmt"
    "math/big"
    "time"

    "github.com/ethereum/go-ethereum/common"
    "github.com/ethereum/go-ethereum/crypto"
)

type Validator struct {
    privateKey     *ecdsa.PrivateKey
    publicKey      common.Address
    chainClient    *chain.Client
    validatorIndex int
}

type SensorData struct {
    SensorID    string    `json:"sensor_id"`
    DeviceType  string    `json:"device_type"`
    Timestamp   int64     `json:"timestamp"`
    GPS         GPSCoord  `json:"gps"`
    Metrics     Metrics   `json:"metrics"`
    Signature   []byte    `json:"signature"`
    DevicePubKey string   `json:"device_pubkey"`
}

type GPSCoord struct {
    Latitude  float64 `json:"lat"`
    Longitude float64 `json:"lng"`
    Accuracy  float64 `json:"accuracy"`
}

type Metrics struct {
    EnergyOutput  float64 `json:"energy_output_kwh,omitempty"`
    Uptime        float64 `json:"uptime_percent,omitempty"`
    DataQuality   float64 `json:"data_quality,omitempty"`
    Temperature   float64 `json:"temperature_c,omitempty"`
}

type ValidationResult struct {
    SensorID      string
    IsValid       bool
    Confidence    float64
    ValidatorSig  []byte
    Timestamp     int64
    FailureReason string
}

// HandleSensorData processes incoming sensor data
func (v *Validator) HandleSensorData(data []byte) (*ValidationResult, error) {
    var sensorData SensorData
    if err := json.Unmarshal(data, &sensorData); err != nil {
        return nil, fmt.Errorf("invalid data format: %w", err)
    }
    
    result := &ValidationResult{
        SensorID:  sensorData.SensorID,
        Timestamp: time.Now().Unix(),
    }
    
    // Step 1: Verify device signature
    if !v.verifyDeviceSignature(&sensorData) {
        result.IsValid = false
        result.FailureReason = "invalid device signature"
        return result, nil
    }
    
    // Step 2: Verify Proof of Physical Presence
    if !v.verifyPhysicalPresence(&sensorData) {
        result.IsValid = false
        result.FailureReason = "failed physical presence check"
        return result, nil
    }
    
    // Step 3: Validate data ranges
    if !v.validateMetrics(&sensorData) {
        result.IsValid = false
        result.FailureReason = "metrics out of valid range"
        return result, nil
    }
    
    // Step 4: Check for anomalies
    anomalyScore := v.detectAnomalies(&sensorData)
    if anomalyScore > 0.8 {
        result.IsValid = false
        result.FailureReason = "anomaly detected"
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
func (v *Validator) verifyDeviceSignature(data *SensorData) bool {
    // Reconstruct the signed message
    message := fmt.Sprintf("%s:%d:%f:%f",
        data.SensorID,
        data.Timestamp,
        data.GPS.Latitude,
        data.GPS.Longitude,
    )
    messageHash := crypto.Keccak256Hash([]byte(message))
    
    // Recover public key from signature
    pubKey, err := crypto.SigToPub(messageHash.Bytes(), data.Signature)
    if err != nil {
        return false
    }
    
    // Verify it matches the claimed device public key
    recoveredAddr := crypto.PubkeyToAddress(*pubKey)
    expectedAddr := common.HexToAddress(data.DevicePubKey)
    
    return recoveredAddr == expectedAddr
}

// verifyPhysicalPresence implements Proof of Physical Presence
func (v *Validator) verifyPhysicalPresence(data *SensorData) bool {
    // Check timestamp is recent (within 5 minutes)
    now := time.Now().Unix()
    if now - data.Timestamp > 300 {
        return false
    }
    
    // Check GPS accuracy
    if data.GPS.Accuracy > 50 { // meters
        return false
    }
    
    // Verify GPS coordinates are in expected location
    // (Would check against registered sensor locations)
    if !v.isValidLocation(data.SensorID, data.GPS) {
        return false
    }
    
    return true
}

// validateMetrics checks if sensor readings are within valid ranges
func (v *Validator) validateMetrics(data *SensorData) bool {
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
    }
    
    return true
}

// detectAnomalies uses ML model to detect unusual patterns
func (v *Validator) detectAnomalies(data *SensorData) float64 {
    // Simplified anomaly detection
    // Production would use trained ML model
    
    historicalAvg := v.getHistoricalAverage(data.SensorID)
    currentValue := data.Metrics.EnergyOutput
    
    // Calculate deviation
    if historicalAvg == 0 {
        return 0 // No history, accept
    }
    
    deviation := abs(currentValue - historicalAvg) / historicalAvg
    
    // Score 0-1, higher means more anomalous
    if deviation > 0.5 {
        return min(deviation, 1.0)
    }
    
    return 0
}
```

### Consensus Engine

```go
// internal/consensus/consensus.go
package consensus

import (
    "context"
    "sync"
    "time"
)

const (
    RequiredSignatures = 9
    TotalValidators    = 13
    ConsensusTimeout   = 30 * time.Second
)

type ConsensusEngine struct {
    validator   *validator.Validator
    validators  []ValidatorInfo
    pendingData map[string]*ConsensusRound
    mu          sync.RWMutex
}

type ConsensusRound struct {
    SensorID      string
    DataHash      []byte
    Signatures    map[int][]byte  // validator index -> signature
    StartTime     time.Time
    Finalized     bool
    FinalResult   bool
}

type ValidatorInfo struct {
    Index     int
    Address   common.Address
    Endpoint  string
}

func (c *ConsensusEngine) Run(ctx context.Context) {
    ticker := time.NewTicker(1 * time.Second)
    defer ticker.Stop()
    
    for {
        select {
        case <-ctx.Done():
            return
        case <-ticker.C:
            c.processRounds()
        }
    }
}

func (c *ConsensusEngine) processRounds() {
    c.mu.Lock()
    defer c.mu.Unlock()
    
    now := time.Now()
    
    for id, round := range c.pendingData {
        if round.Finalized {
            continue
        }
        
        // Check if we have enough signatures
        validSigs := 0
        for _, sig := range round.Signatures {
            if len(sig) > 0 {
                validSigs++
            }
        }
        
        if validSigs >= RequiredSignatures {
            // Consensus reached - submit to chain
            c.submitToChain(round)
            round.Finalized = true
            round.FinalResult = true
        } else if now.Sub(round.StartTime) > ConsensusTimeout {
            // Timeout - mark as failed
            round.Finalized = true
            round.FinalResult = false
            delete(c.pendingData, id)
        }
    }
}

func (c *ConsensusEngine) submitToChain(round *ConsensusRound) error {
    // Collect signatures
    var signatures [][]byte
    var indices []int
    
    for idx, sig := range round.Signatures {
        if len(sig) > 0 {
            signatures = append(signatures, sig)
            indices = append(indices, idx)
        }
    }
    
    // Submit to OracleVerifier contract
    return c.validator.chainClient.SubmitOracleUpdate(
        round.SensorID,
        round.DataHash,
        signatures,
        indices,
    )
}
```

---

## IoT Data Pipeline

### MQTT Data Ingestion

```python
# src/depin/mqtt_ingestion.py
import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Callable, Dict, Optional
import paho.mqtt.client as mqtt
from aiokafka import AIOKafkaProducer

logger = logging.getLogger(__name__)

@dataclass
class SensorReading:
    sensor_id: str
    device_type: str
    timestamp: int
    latitude: float
    longitude: float
    gps_accuracy: float
    metrics: Dict
    signature: bytes
    device_pubkey: str


class MQTTIngestionService:
    """
    Ingests IoT sensor data via MQTT and forwards to Kafka.
    """
    
    def __init__(
        self,
        mqtt_broker: str,
        mqtt_port: int,
        kafka_brokers: str,
        kafka_topic: str
    ):
        self.mqtt_broker = mqtt_broker
        self.mqtt_port = mqtt_port
        self.kafka_brokers = kafka_brokers
        self.kafka_topic = kafka_topic
        
        self.mqtt_client: Optional[mqtt.Client] = None
        self.kafka_producer: Optional[AIOKafkaProducer] = None
        self.handlers: Dict[str, Callable] = {}
    
    async def start(self):
        """Initialize connections."""
        # Start Kafka producer
        self.kafka_producer = AIOKafkaProducer(
            bootstrap_servers=self.kafka_brokers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        await self.kafka_producer.start()
        
        # Setup MQTT client
        self.mqtt_client = mqtt.Client(client_id="depin-ingestion")
        self.mqtt_client.on_connect = self._on_connect
        self.mqtt_client.on_message = self._on_message
        
        # Enable TLS
        self.mqtt_client.tls_set()
        
        # Connect
        self.mqtt_client.connect(self.mqtt_broker, self.mqtt_port)
        self.mqtt_client.loop_start()
        
        logger.info("MQTT ingestion service started")
    
    def _on_connect(self, client, userdata, flags, rc):
        """Handle MQTT connection."""
        if rc == 0:
            logger.info("Connected to MQTT broker")
            # Subscribe to all sensor topics
            client.subscribe("sensors/+/data", qos=1)
            client.subscribe("sensors/+/status", qos=1)
        else:
            logger.error(f"MQTT connection failed with code {rc}")
    
    def _on_message(self, client, userdata, msg):
        """Handle incoming MQTT message."""
        try:
            # Parse topic to get sensor ID
            parts = msg.topic.split('/')
            sensor_id = parts[1]
            message_type = parts[2]
            
            # Parse payload
            payload = json.loads(msg.payload.decode('utf-8'))
            
            # Validate basic structure
            if not self._validate_payload(payload):
                logger.warning(f"Invalid payload from {sensor_id}")
                return
            
            # Forward to Kafka for validator processing
            asyncio.create_task(self._forward_to_kafka(sensor_id, message_type, payload))
            
        except Exception as e:
            logger.error(f"Error processing MQTT message: {e}")
    
    def _validate_payload(self, payload: dict) -> bool:
        """Basic payload validation."""
        required_fields = ['sensor_id', 'timestamp', 'gps', 'signature']
        return all(field in payload for field in required_fields)
    
    async def _forward_to_kafka(self, sensor_id: str, msg_type: str, payload: dict):
        """Forward validated data to Kafka."""
        kafka_message = {
            'sensor_id': sensor_id,
            'message_type': msg_type,
            'payload': payload,
            'ingested_at': int(time.time())
        }
        
        await self.kafka_producer.send(
            self.kafka_topic,
            key=sensor_id.encode(),
            value=kafka_message
        )
    
    async def stop(self):
        """Gracefully shutdown."""
        if self.mqtt_client:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
        if self.kafka_producer:
            await self.kafka_producer.stop()
```

### Sensor Device SDK

```python
# sdk/python/viddhana_sensor/sensor.py
import hashlib
import json
import time
from dataclasses import dataclass
from typing import Dict, Optional
import paho.mqtt.client as mqtt
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec

@dataclass
class GPSLocation:
    latitude: float
    longitude: float
    accuracy: float


class ViddhanaSensor:
    """
    SDK for DePIN sensor devices to report data to VIDDHANA network.
    """
    
    def __init__(
        self,
        sensor_id: str,
        device_type: str,
        private_key_path: str,
        mqtt_broker: str,
        mqtt_port: int = 8883
    ):
        self.sensor_id = sensor_id
        self.device_type = device_type
        self.private_key = self._load_private_key(private_key_path)
        self.public_key = self.private_key.public_key()
        
        # MQTT setup
        self.mqtt_client = mqtt.Client(client_id=f"sensor-{sensor_id}")
        self.mqtt_client.tls_set()
        self.mqtt_client.connect(mqtt_broker, mqtt_port)
        self.mqtt_client.loop_start()
    
    def _load_private_key(self, path: str):
        """Load device private key."""
        with open(path, 'rb') as f:
            return serialization.load_pem_private_key(f.read(), password=None)
    
    def report_data(
        self,
        metrics: Dict,
        gps: GPSLocation
    ) -> bool:
        """
        Report sensor data to the network.
        
        Args:
            metrics: Dictionary of sensor metrics (e.g., energy_output, uptime)
            gps: Current GPS location
        
        Returns:
            True if data was sent successfully
        """
        timestamp = int(time.time())
        
        # Build payload
        payload = {
            'sensor_id': self.sensor_id,
            'device_type': self.device_type,
            'timestamp': timestamp,
            'gps': {
                'lat': gps.latitude,
                'lng': gps.longitude,
                'accuracy': gps.accuracy
            },
            'metrics': metrics,
            'device_pubkey': self._get_public_key_hex()
        }
        
        # Sign the payload
        signature = self._sign_payload(payload)
        payload['signature'] = signature.hex()
        
        # Publish to MQTT
        topic = f"sensors/{self.sensor_id}/data"
        result = self.mqtt_client.publish(
            topic,
            json.dumps(payload),
            qos=1
        )
        
        return result.rc == mqtt.MQTT_ERR_SUCCESS
    
    def _sign_payload(self, payload: dict) -> bytes:
        """Sign payload with device private key."""
        # Create message to sign
        message = f"{payload['sensor_id']}:{payload['timestamp']}:{payload['gps']['lat']}:{payload['gps']['lng']}"
        message_bytes = message.encode('utf-8')
        
        # Sign with ECDSA
        signature = self.private_key.sign(
            message_bytes,
            ec.ECDSA(hashes.SHA256())
        )
        
        return signature
    
    def _get_public_key_hex(self) -> str:
        """Get public key as hex string."""
        pub_bytes = self.public_key.public_bytes(
            encoding=serialization.Encoding.X962,
            format=serialization.PublicFormat.CompressedPoint
        )
        return pub_bytes.hex()
    
    def report_status(self, status: str, uptime_percent: float):
        """Report device status."""
        payload = {
            'sensor_id': self.sensor_id,
            'status': status,
            'uptime_percent': uptime_percent,
            'timestamp': int(time.time())
        }
        
        topic = f"sensors/{self.sensor_id}/status"
        self.mqtt_client.publish(topic, json.dumps(payload), qos=1)
    
    def close(self):
        """Cleanup connections."""
        self.mqtt_client.loop_stop()
        self.mqtt_client.disconnect()


# Example usage
if __name__ == "__main__":
    sensor = ViddhanaSensor(
        sensor_id="solar-panel-001",
        device_type="solar_panel",
        private_key_path="/etc/viddhana/device.key",
        mqtt_broker="mqtt.viddhana.network"
    )
    
    # Report hourly energy production
    sensor.report_data(
        metrics={
            'energy_output_kwh': 45.2,
            'temperature_c': 28.5,
            'efficiency_percent': 18.7
        },
        gps=GPSLocation(
            latitude=37.7749,
            longitude=-122.4194,
            accuracy=5.0
        )
    )
```

---

## RWA Tokenization

### RWA NFT Standard

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import "@openzeppelin/contracts/token/ERC721/extensions/ERC721Enumerable.sol";
import "@openzeppelin/contracts/token/ERC721/extensions/ERC721URIStorage.sol";
import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";

/**
 * @title RWAToken
 * @notice NFT representing fractional ownership of Real World Assets
 */
contract RWAToken is 
    ERC721,
    ERC721Enumerable,
    ERC721URIStorage,
    AccessControl,
    ReentrancyGuard
{
    bytes32 public constant MINTER_ROLE = keccak256("MINTER_ROLE");
    bytes32 public constant ORACLE_ROLE = keccak256("ORACLE_ROLE");
    
    struct AssetDetails {
        string assetType;           // "real_estate", "solar_farm", "infrastructure"
        string legalEntity;         // Legal entity owning the asset
        string jurisdiction;        // Legal jurisdiction
        uint256 totalValue;         // Total asset value in USD (18 decimals)
        uint256 totalShares;        // Total number of fractional shares
        uint256 rentalYield;        // Annual yield in basis points
        uint256 lastValuation;      // Timestamp of last valuation
        bool isVerified;            // KYC/legal verification status
    }
    
    struct ShareInfo {
        uint256 assetId;
        uint256 shareCount;
        uint256 purchasePrice;
        uint256 purchaseTime;
    }
    
    // Asset ID => Asset Details
    mapping(uint256 => AssetDetails) public assets;
    
    // Token ID => Share Info
    mapping(uint256 => ShareInfo) public shares;
    
    // Asset ID => Accumulated rental income
    mapping(uint256 => uint256) public accumulatedRentalIncome;
    
    // Token ID => Claimed rental income
    mapping(uint256 => uint256) public claimedIncome;
    
    uint256 private _assetIdCounter;
    uint256 private _tokenIdCounter;
    
    IERC20 public paymentToken;  // USDC or VDH
    
    event AssetRegistered(uint256 indexed assetId, string assetType, uint256 totalValue);
    event SharesMinted(uint256 indexed assetId, uint256 indexed tokenId, address owner, uint256 shares);
    event RentalIncomeDistributed(uint256 indexed assetId, uint256 amount);
    event IncomeClaimed(uint256 indexed tokenId, address owner, uint256 amount);
    event ValuationUpdated(uint256 indexed assetId, uint256 newValue);
    
    constructor(address _paymentToken) ERC721("VIDDHANA RWA", "vRWA") {
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(MINTER_ROLE, msg.sender);
        paymentToken = IERC20(_paymentToken);
    }
    
    /**
     * @notice Register a new Real World Asset
     */
    function registerAsset(
        string calldata assetType,
        string calldata legalEntity,
        string calldata jurisdiction,
        uint256 totalValue,
        uint256 totalShares,
        uint256 rentalYield
    ) external onlyRole(MINTER_ROLE) returns (uint256) {
        uint256 assetId = _assetIdCounter++;
        
        assets[assetId] = AssetDetails({
            assetType: assetType,
            legalEntity: legalEntity,
            jurisdiction: jurisdiction,
            totalValue: totalValue,
            totalShares: totalShares,
            rentalYield: rentalYield,
            lastValuation: block.timestamp,
            isVerified: false
        });
        
        emit AssetRegistered(assetId, assetType, totalValue);
        
        return assetId;
    }
    
    /**
     * @notice Mint fractional shares of an asset
     */
    function mintShares(
        uint256 assetId,
        address to,
        uint256 shareCount,
        string calldata tokenURI_
    ) external onlyRole(MINTER_ROLE) returns (uint256) {
        AssetDetails storage asset = assets[assetId];
        require(asset.totalValue > 0, "Asset not registered");
        require(asset.isVerified, "Asset not verified");
        
        uint256 tokenId = _tokenIdCounter++;
        
        // Calculate price
        uint256 pricePerShare = asset.totalValue / asset.totalShares;
        uint256 totalPrice = pricePerShare * shareCount;
        
        // Transfer payment
        paymentToken.transferFrom(to, address(this), totalPrice);
        
        // Mint NFT
        _safeMint(to, tokenId);
        _setTokenURI(tokenId, tokenURI_);
        
        shares[tokenId] = ShareInfo({
            assetId: assetId,
            shareCount: shareCount,
            purchasePrice: totalPrice,
            purchaseTime: block.timestamp
        });
        
        emit SharesMinted(assetId, tokenId, to, shareCount);
        
        return tokenId;
    }
    
    /**
     * @notice Distribute rental income for an asset
     */
    function distributeRentalIncome(
        uint256 assetId,
        uint256 amount
    ) external onlyRole(ORACLE_ROLE) {
        require(assets[assetId].totalValue > 0, "Asset not registered");
        
        // Transfer rental income to contract
        paymentToken.transferFrom(msg.sender, address(this), amount);
        
        // Add to accumulated income
        accumulatedRentalIncome[assetId] += amount;
        
        emit RentalIncomeDistributed(assetId, amount);
    }
    
    /**
     * @notice Claim accumulated rental income for a token
     */
    function claimIncome(uint256 tokenId) external nonReentrant {
        require(ownerOf(tokenId) == msg.sender, "Not token owner");
        
        ShareInfo storage share = shares[tokenId];
        AssetDetails storage asset = assets[share.assetId];
        
        // Calculate claimable amount
        uint256 totalIncome = accumulatedRentalIncome[share.assetId];
        uint256 shareRatio = (share.shareCount * 1e18) / asset.totalShares;
        uint256 entitled = (totalIncome * shareRatio) / 1e18;
        uint256 claimable = entitled - claimedIncome[tokenId];
        
        require(claimable > 0, "Nothing to claim");
        
        claimedIncome[tokenId] += claimable;
        paymentToken.transfer(msg.sender, claimable);
        
        emit IncomeClaimed(tokenId, msg.sender, claimable);
    }
    
    /**
     * @notice Calculate total return for a token
     * @dev Total Return = (Rental Income + Price Appreciation) / Initial Investment
     */
    function calculateTotalReturn(uint256 tokenId) external view returns (uint256) {
        ShareInfo storage share = shares[tokenId];
        AssetDetails storage asset = assets[share.assetId];
        
        // Current value of shares
        uint256 currentPricePerShare = asset.totalValue / asset.totalShares;
        uint256 currentValue = currentPricePerShare * share.shareCount;
        
        // Rental income earned
        uint256 totalIncome = accumulatedRentalIncome[share.assetId];
        uint256 shareRatio = (share.shareCount * 1e18) / asset.totalShares;
        uint256 rentalIncome = (totalIncome * shareRatio) / 1e18;
        
        // Price appreciation
        int256 appreciation = int256(currentValue) - int256(share.purchasePrice);
        
        // Total return as basis points
        uint256 totalGain = rentalIncome + (appreciation > 0 ? uint256(appreciation) : 0);
        uint256 totalReturn = (totalGain * 10000) / share.purchasePrice;
        
        return totalReturn;
    }
    
    /**
     * @notice Update asset valuation (oracle function)
     */
    function updateValuation(
        uint256 assetId,
        uint256 newValue
    ) external onlyRole(ORACLE_ROLE) {
        require(assets[assetId].totalValue > 0, "Asset not registered");
        
        assets[assetId].totalValue = newValue;
        assets[assetId].lastValuation = block.timestamp;
        
        emit ValuationUpdated(assetId, newValue);
    }
    
    /**
     * @notice Verify an asset (admin function after legal verification)
     */
    function verifyAsset(uint256 assetId) external onlyRole(DEFAULT_ADMIN_ROLE) {
        assets[assetId].isVerified = true;
    }
    
    // Required overrides
    function _update(address to, uint256 tokenId, address auth)
        internal
        override(ERC721, ERC721Enumerable)
        returns (address)
    {
        return super._update(to, tokenId, auth);
    }

    function _increaseBalance(address account, uint128 value)
        internal
        override(ERC721, ERC721Enumerable)
    {
        super._increaseBalance(account, value);
    }

    function tokenURI(uint256 tokenId)
        public
        view
        override(ERC721, ERC721URIStorage)
        returns (string memory)
    {
        return super.tokenURI(tokenId);
    }

    function supportsInterface(bytes4 interfaceId)
        public
        view
        override(ERC721, ERC721Enumerable, ERC721URIStorage, AccessControl)
        returns (bool)
    {
        return super.supportsInterface(interfaceId);
    }
}
```

---

## Reward Calculation

### DePIN Reward Engine

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts-upgradeable/access/AccessControlUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/proxy/utils/UUPSUpgradeable.sol";

/**
 * @title DePINRewards
 * @notice Calculates and distributes rewards to DePIN sensor operators
 */
contract DePINRewards is Initializable, AccessControlUpgradeable, UUPSUpgradeable {
    
    bytes32 public constant ORACLE_ROLE = keccak256("ORACLE_ROLE");
    
    // Reward parameters (in wei, assuming 18 decimals)
    uint256 public baseFeePerDay;           // $0.50 = 500000000000000000
    uint256 public qualityBonusPerDay;      // $0.10 = 100000000000000000
    uint256 public penaltyPerIncident;      // $0.05 = 50000000000000000
    uint256 public uptimeThreshold;         // 99% = 9900 (basis points)
    
    struct SensorStats {
        uint256 totalDataPoints;
        uint256 validDataPoints;
        uint256 uptimeSeconds;
        uint256 totalSeconds;
        uint256 errorCount;
        uint256 lastUpdateTime;
        uint256 pendingRewards;
        uint256 claimedRewards;
    }
    
    mapping(bytes32 => SensorStats) public sensorStats;
    mapping(bytes32 => address) public sensorOwners;
    
    IERC20 public rewardToken;  // VDH token
    
    event RewardsCalculated(bytes32 indexed sensorId, uint256 reward);
    event RewardsClaimed(bytes32 indexed sensorId, address owner, uint256 amount);
    event SensorRegistered(bytes32 indexed sensorId, address owner);
    
    function initialize(address _rewardToken) public initializer {
        __AccessControl_init();
        __UUPSUpgradeable_init();
        
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        
        rewardToken = IERC20(_rewardToken);
        
        // Set default parameters
        baseFeePerDay = 500000000000000000;      // $0.50
        qualityBonusPerDay = 100000000000000000; // $0.10
        penaltyPerIncident = 50000000000000000;  // $0.05
        uptimeThreshold = 9900;                   // 99%
    }
    
    /**
     * @notice Register a new sensor
     */
    function registerSensor(bytes32 sensorId, address owner) external onlyRole(DEFAULT_ADMIN_ROLE) {
        require(sensorOwners[sensorId] == address(0), "Already registered");
        
        sensorOwners[sensorId] = owner;
        sensorStats[sensorId] = SensorStats({
            totalDataPoints: 0,
            validDataPoints: 0,
            uptimeSeconds: 0,
            totalSeconds: 0,
            errorCount: 0,
            lastUpdateTime: block.timestamp,
            pendingRewards: 0,
            claimedRewards: 0
        });
        
        emit SensorRegistered(sensorId, owner);
    }
    
    /**
     * @notice Update sensor statistics (called by oracle)
     */
    function updateSensorStats(
        bytes32 sensorId,
        uint256 dataPoints,
        uint256 validPoints,
        uint256 uptimeSeconds,
        uint256 periodSeconds,
        uint256 errors
    ) external onlyRole(ORACLE_ROLE) {
        SensorStats storage stats = sensorStats[sensorId];
        
        stats.totalDataPoints += dataPoints;
        stats.validDataPoints += validPoints;
        stats.uptimeSeconds += uptimeSeconds;
        stats.totalSeconds += periodSeconds;
        stats.errorCount += errors;
        stats.lastUpdateTime = block.timestamp;
    }
    
    /**
     * @notice Calculate daily rewards for a sensor
     * @dev Reward = Base_Fee + Data_Quality_Bonus - Penalty
     */
    function calculateDailyReward(bytes32 sensorId) public view returns (uint256) {
        SensorStats storage stats = sensorStats[sensorId];
        
        if (stats.totalSeconds == 0) return 0;
        
        // Calculate uptime percentage (basis points)
        uint256 uptimePercent = (stats.uptimeSeconds * 10000) / stats.totalSeconds;
        
        // Start with base fee
        uint256 reward = baseFeePerDay;
        
        // Add quality bonus if uptime > 99%
        if (uptimePercent >= uptimeThreshold) {
            reward += qualityBonusPerDay;
        }
        
        // Subtract penalties for errors
        uint256 totalPenalty = stats.errorCount * penaltyPerIncident;
        if (totalPenalty >= reward) {
            return 0;
        }
        reward -= totalPenalty;
        
        return reward;
    }
    
    /**
     * @notice Process daily rewards for a sensor
     */
    function processDailyRewards(bytes32 sensorId) external onlyRole(ORACLE_ROLE) {
        uint256 reward = calculateDailyReward(sensorId);
        
        sensorStats[sensorId].pendingRewards += reward;
        
        // Reset daily counters
        sensorStats[sensorId].errorCount = 0;
        
        emit RewardsCalculated(sensorId, reward);
    }
    
    /**
     * @notice Claim pending rewards
     */
    function claimRewards(bytes32 sensorId) external {
        require(sensorOwners[sensorId] == msg.sender, "Not sensor owner");
        
        SensorStats storage stats = sensorStats[sensorId];
        uint256 pending = stats.pendingRewards;
        
        require(pending > 0, "No rewards to claim");
        
        stats.pendingRewards = 0;
        stats.claimedRewards += pending;
        
        rewardToken.transfer(msg.sender, pending);
        
        emit RewardsClaimed(sensorId, msg.sender, pending);
    }
    
    /**
     * @notice Get sensor reward summary
     */
    function getSensorRewardSummary(bytes32 sensorId) external view returns (
        uint256 pending,
        uint256 claimed,
        uint256 dailyRate,
        uint256 uptimePercent
    ) {
        SensorStats storage stats = sensorStats[sensorId];
        
        pending = stats.pendingRewards;
        claimed = stats.claimedRewards;
        dailyRate = calculateDailyReward(sensorId);
        
        if (stats.totalSeconds > 0) {
            uptimePercent = (stats.uptimeSeconds * 10000) / stats.totalSeconds;
        }
    }
    
    // Admin functions
    function setRewardParameters(
        uint256 _baseFee,
        uint256 _qualityBonus,
        uint256 _penalty,
        uint256 _uptimeThreshold
    ) external onlyRole(DEFAULT_ADMIN_ROLE) {
        baseFeePerDay = _baseFee;
        qualityBonusPerDay = _qualityBonus;
        penaltyPerIncident = _penalty;
        uptimeThreshold = _uptimeThreshold;
    }
    
    function _authorizeUpgrade(address) internal override onlyRole(DEFAULT_ADMIN_ROLE) {}
}
```

---

## Testing & Validation

### Integration Test Suite

```typescript
// test/depin-integration.test.ts
import { expect } from "chai";
import { ethers } from "hardhat";

describe("DePIN Integration", () => {
  let oracleVerifier: OracleVerifier;
  let depinRewards: DePINRewards;
  let validators: SignerWithAddress[];
  
  const SENSOR_ID = ethers.keccak256(ethers.toUtf8Bytes("solar-panel-001"));
  
  beforeEach(async () => {
    validators = (await ethers.getSigners()).slice(0, 13);
    
    // Deploy contracts
    const OracleVerifier = await ethers.getContractFactory("OracleVerifier");
    oracleVerifier = await OracleVerifier.deploy(
      validators.map(v => v.address),
      9  // Required signatures
    );
    
    const DePINRewards = await ethers.getContractFactory("DePINRewards");
    depinRewards = await upgrades.deployProxy(DePINRewards, [vdhToken.address]);
  });
  
  describe("Oracle Consensus", () => {
    it("should accept data with 9/13 validator signatures", async () => {
      const dataHash = ethers.keccak256(ethers.toUtf8Bytes("test-data"));
      const signatures = [];
      
      // Get 9 signatures
      for (let i = 0; i < 9; i++) {
        const sig = await validators[i].signMessage(ethers.getBytes(dataHash));
        signatures.push(sig);
      }
      
      await expect(
        oracleVerifier.submitData(
          SENSOR_ID,
          dataHash,
          signatures,
          [0, 1, 2, 3, 4, 5, 6, 7, 8]
        )
      ).to.emit(oracleVerifier, "DataAccepted");
    });
    
    it("should reject data with insufficient signatures", async () => {
      const dataHash = ethers.keccak256(ethers.toUtf8Bytes("test-data"));
      const signatures = [];
      
      // Get only 8 signatures
      for (let i = 0; i < 8; i++) {
        const sig = await validators[i].signMessage(ethers.getBytes(dataHash));
        signatures.push(sig);
      }
      
      await expect(
        oracleVerifier.submitData(
          SENSOR_ID,
          dataHash,
          signatures,
          [0, 1, 2, 3, 4, 5, 6, 7]
        )
      ).to.be.revertedWith("Insufficient signatures");
    });
  });
  
  describe("Reward Calculation", () => {
    it("should calculate correct rewards for high uptime", async () => {
      // Register sensor
      await depinRewards.registerSensor(SENSOR_ID, owner.address);
      
      // Update stats: 99.5% uptime, no errors
      await depinRewards.updateSensorStats(
        SENSOR_ID,
        100,   // data points
        100,   // valid points
        85680, // uptime seconds (99.5% of 86400)
        86400, // period seconds (1 day)
        0      // errors
      );
      
      const reward = await depinRewards.calculateDailyReward(SENSOR_ID);
      
      // Should be base fee + quality bonus = $0.60
      expect(reward).to.equal(ethers.parseEther("0.6"));
    });
    
    it("should apply penalties for errors", async () => {
      await depinRewards.registerSensor(SENSOR_ID, owner.address);
      
      // Update stats with errors
      await depinRewards.updateSensorStats(
        SENSOR_ID,
        100,
        95,    // 5 invalid
        86400,
        86400,
        3      // 3 errors
      );
      
      const reward = await depinRewards.calculateDailyReward(SENSOR_ID);
      
      // Base + bonus - (3 * penalty) = 0.50 + 0.10 - 0.15 = $0.45
      expect(reward).to.equal(ethers.parseEther("0.45"));
    });
  });
});
```

### Acceptance Criteria

```markdown
## DePIN & RWA Acceptance Criteria

### Oracle Network
- [ ] 13 validator nodes operational
- [ ] 9/13 consensus threshold working
- [ ] Slashing for malicious validators implemented
- [ ] < 30 second data finality

### IoT Pipeline
- [ ] MQTT ingestion handling 10,000+ messages/second
- [ ] Data validation pipeline working
- [ ] Proof of Physical Presence verification functional
- [ ] GPS + Timestamp + Signature validation working

### RWA Tokenization
- [ ] NFT minting for real estate shares working
- [ ] Rental income distribution functional
- [ ] Total return calculation accurate
- [ ] Secondary market transfers working

### Rewards
- [ ] Daily reward calculation correct
- [ ] Quality bonus applied correctly
- [ ] Penalties deducted properly
- [ ] Reward claims functional
```

---

## Next Steps

After completing DePIN/RWA integration:
1. Proceed to `06_TOKENOMICS_IMPLEMENTATION.md`
2. Deploy validator network to testnet
3. Update `TRACKER.md` with completion status

---

*Document Version: 1.0.0*
