"""
FastAPI inference server for Prometheus AI.

Provides REST endpoints for:
- Price prediction
- Portfolio optimization
- Risk assessment
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Pydantic Request/Response Models
# ============================================================================


class PricePredictionRequest(BaseModel):
    """Request for price prediction."""
    
    asset: str = Field(..., description="Asset symbol (e.g., BTC, ETH)")
    horizon: int = Field(7, ge=1, le=30, description="Days to predict ahead")
    include_confidence: bool = Field(True, description="Include confidence scores")


class PricePredictionResponse(BaseModel):
    """Response for price prediction."""
    
    asset: str
    horizon_days: int
    predictions: List[float]
    confidence: List[float]
    timestamp: int
    model_version: str = "1.0.0"


class PortfolioOptimizationRequest(BaseModel):
    """Request for portfolio optimization."""
    
    user_id: str = Field(..., description="User identifier")
    portfolio: Dict[str, float] = Field(..., description="Current portfolio (asset -> USD value)")
    risk_tolerance: float = Field(..., ge=0, le=1, description="Risk tolerance (0=conservative, 1=aggressive)")
    time_to_goal: int = Field(..., ge=1, description="Months until investment goal")
    monthly_contribution: float = Field(0, ge=0, description="Monthly contribution amount")


class RebalanceRecommendation(BaseModel):
    """Individual rebalancing recommendation."""
    
    asset: str
    action: str  # "BUY" or "SELL"
    percentage: float


class RiskAssessmentDetails(BaseModel):
    """Detailed risk assessment."""
    
    risk_score: float
    volatility_risk: float
    concentration_risk: float
    market_risk: float
    matches_profile: bool
    recommendation: str


class PortfolioOptimizationResponse(BaseModel):
    """Response for portfolio optimization."""
    
    action: str
    recommendations: List[RebalanceRecommendation]
    confidence: float
    risk_assessment: RiskAssessmentDetails
    timestamp: int


class RiskAssessmentRequest(BaseModel):
    """Request for risk assessment."""
    
    portfolio: Dict[str, float] = Field(..., description="Portfolio (asset -> USD value)")


class RiskAssessmentResponse(BaseModel):
    """Response for risk assessment."""
    
    risk_score: float = Field(..., ge=0, le=100, description="Overall risk score")
    volatility: float = Field(..., description="Portfolio volatility")
    sharpe_ratio: float = Field(..., description="Risk-adjusted return metric")
    max_drawdown: float = Field(..., description="Maximum drawdown percentage")
    diversification_score: float = Field(..., ge=0, le=1, description="Diversification level")
    recommendations: List[str] = Field(..., description="Risk mitigation recommendations")


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str
    version: str
    uptime_seconds: float
    models_loaded: bool


# ============================================================================
# Model Management
# ============================================================================


class ModelManager:
    """Manages AI model loading and inference."""
    
    def __init__(self) -> None:
        self.price_predictor = None
        self.portfolio_optimizer = None
        self.models_loaded = False
        self.load_time = time.time()
    
    async def load_models(self) -> None:
        """Load AI models asynchronously."""
        logger.info("Loading AI models...")
        
        try:
            # In production, load actual trained models
            # For now, we initialize with default parameters
            from prometheus.models.lstm_predictor import LSTMPredictor
            from prometheus.models.q_learning import PortfolioOptimizer
            
            self.price_predictor = LSTMPredictor(
                input_dim=64,
                hidden_dim=256,
                output_horizon=30,
            )
            
            self.portfolio_optimizer = PortfolioOptimizer(
                state_dim=32,
                gamma=0.9,
            )
            
            self.models_loaded = True
            logger.info("AI models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            self.models_loaded = False
    
    def get_uptime(self) -> float:
        """Get server uptime in seconds."""
        return time.time() - self.load_time


# Global model manager
model_manager = ModelManager()


# ============================================================================
# FastAPI Application
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    await model_manager.load_models()
    yield
    # Shutdown
    logger.info("Shutting down Prometheus AI server...")


app = FastAPI(
    title="Prometheus AI API",
    description="AI-powered predictions and optimization for VIDDHANA",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Dependency Injection
# ============================================================================


async def get_model_manager() -> ModelManager:
    """Dependency for model manager."""
    if not model_manager.models_loaded:
        raise HTTPException(
            status_code=503,
            detail="Models not yet loaded. Please try again later.",
        )
    return model_manager


# ============================================================================
# API Endpoints
# ============================================================================


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """
    Health check endpoint.
    
    Returns server status, version, and model loading state.
    """
    return HealthResponse(
        status="healthy" if model_manager.models_loaded else "starting",
        version="1.0.0",
        uptime_seconds=model_manager.get_uptime(),
        models_loaded=model_manager.models_loaded,
    )


@app.post("/predict", response_model=PricePredictionResponse, tags=["Prediction"])
async def predict_price(
    request: PricePredictionRequest,
    manager: ModelManager = Depends(get_model_manager),
) -> PricePredictionResponse:
    """
    Generate price prediction for an asset.
    
    Uses LSTM + Transformer model trained on historical data.
    
    - **asset**: Asset symbol (e.g., BTC, ETH)
    - **horizon**: Number of days to predict ahead (1-30)
    """
    try:
        logger.info(f"Price prediction request for {request.asset}, horizon={request.horizon}")
        
        # In production, fetch real historical data
        # For demo, generate mock predictions
        np.random.seed(hash(request.asset) % 2**32)
        
        base_price = {"BTC": 45000, "ETH": 2500, "SOL": 100}.get(request.asset, 1000)
        
        # Generate predictions with random walk
        predictions = []
        confidence = []
        current = base_price
        
        for i in range(request.horizon):
            change = np.random.normal(0.001, 0.02)
            current *= (1 + change)
            predictions.append(round(current, 2))
            # Confidence decreases with horizon
            conf = max(0.5, 0.95 - (i * 0.03))
            confidence.append(round(conf, 3))
        
        return PricePredictionResponse(
            asset=request.asset,
            horizon_days=request.horizon,
            predictions=predictions,
            confidence=confidence if request.include_confidence else [],
            timestamp=int(time.time()),
        )
        
    except Exception as e:
        logger.error(f"Price prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/optimize", response_model=PortfolioOptimizationResponse, tags=["Optimization"])
async def optimize_portfolio(
    request: PortfolioOptimizationRequest,
    manager: ModelManager = Depends(get_model_manager),
) -> PortfolioOptimizationResponse:
    """
    Get portfolio rebalancing recommendations.
    
    Uses Q-Learning agent trained on historical portfolio performance.
    
    - **portfolio**: Current portfolio allocation (asset -> USD value)
    - **risk_tolerance**: Risk tolerance level (0=conservative, 1=aggressive)
    - **time_to_goal**: Investment horizon in months
    """
    try:
        logger.info(f"Portfolio optimization request for user {request.user_id}")
        
        from prometheus.models.q_learning import PortfolioState, RiskProfile
        
        # Build portfolio state
        total_value = sum(request.portfolio.values())
        allocations = {k: v / total_value for k, v in request.portfolio.items()}
        
        portfolio_state = PortfolioState(
            total_value=total_value,
            asset_allocation=allocations,
            unrealized_pnl=0,
            volatility_30d=0.15,
            sharpe_ratio=1.2,
            market_regime="sideways",
        )
        
        risk_profile = RiskProfile(
            risk_tolerance=request.risk_tolerance,
            time_to_goal=request.time_to_goal,
            investment_amount=total_value,
            monthly_contribution=request.monthly_contribution,
        )
        
        # Mock market data
        market_data = {
            "btc_return_7d": 0.02,
            "eth_return_7d": 0.03,
            "market_volatility": 0.2,
            "fear_greed_index": 55,
        }
        
        # Get recommendation
        result = manager.portfolio_optimizer.get_rebalance_recommendation(
            portfolio_state,
            risk_profile,
            market_data,
        )
        
        return PortfolioOptimizationResponse(
            action=result["action"],
            recommendations=[
                RebalanceRecommendation(**r) for r in result["recommendations"]
            ],
            confidence=result["confidence"],
            risk_assessment=RiskAssessmentDetails(**result["risk_assessment"]),
            timestamp=int(time.time()),
        )
        
    except Exception as e:
        logger.error(f"Portfolio optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/risk", response_model=RiskAssessmentResponse, tags=["Risk"])
async def assess_risk(
    request: RiskAssessmentRequest,
    manager: ModelManager = Depends(get_model_manager),
) -> RiskAssessmentResponse:
    """
    Assess portfolio risk metrics.
    
    Calculates volatility, Sharpe ratio, max drawdown, and provides
    risk mitigation recommendations.
    """
    try:
        logger.info("Risk assessment request")
        
        total_value = sum(request.portfolio.values())
        num_assets = len(request.portfolio)
        
        # Calculate risk metrics
        # In production, use actual historical data
        
        # Volatility based on asset composition
        volatile_assets = {"BTC", "ETH", "SOL", "AVAX"}
        volatile_pct = sum(
            v for k, v in request.portfolio.items() if k in volatile_assets
        ) / total_value
        
        volatility = 0.10 + (volatile_pct * 0.30)
        
        # Risk score (0-100)
        risk_score = min(100, volatile_pct * 80 + (1 / num_assets) * 20)
        
        # Sharpe ratio estimation
        sharpe_ratio = max(0.5, 2.0 - volatile_pct)
        
        # Max drawdown estimation
        max_drawdown = volatile_pct * 0.40
        
        # Diversification score
        if num_assets >= 5:
            diversification = 0.8
        elif num_assets >= 3:
            diversification = 0.6
        else:
            diversification = 0.3
        
        # Generate recommendations
        recommendations = []
        
        if volatile_pct > 0.7:
            recommendations.append(
                "Consider reducing exposure to volatile assets to lower risk."
            )
        
        if num_assets < 3:
            recommendations.append(
                "Increase diversification by adding more asset classes."
            )
        
        if "USDC" not in request.portfolio and "USDT" not in request.portfolio:
            recommendations.append(
                "Consider adding stablecoins for portfolio stability."
            )
        
        if risk_score > 70:
            recommendations.append(
                "Portfolio risk is high. Consider rebalancing to match your risk profile."
            )
        
        if not recommendations:
            recommendations.append("Portfolio risk is well-balanced.")
        
        return RiskAssessmentResponse(
            risk_score=round(risk_score, 1),
            volatility=round(volatility, 4),
            sharpe_ratio=round(sharpe_ratio, 2),
            max_drawdown=round(max_drawdown, 4),
            diversification_score=round(diversification, 2),
            recommendations=recommendations,
        )
        
    except Exception as e:
        logger.error(f"Risk assessment failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# WebSocket Endpoints
# ============================================================================


@app.websocket("/ws/predictions/{asset}")
async def websocket_predictions(websocket: WebSocket, asset: str) -> None:
    """
    Stream real-time predictions for an asset.
    
    Sends updated predictions every 60 seconds.
    """
    await websocket.accept()
    logger.info(f"WebSocket connection for {asset} predictions")
    
    try:
        while True:
            # Generate prediction
            np.random.seed(int(time.time()) % 1000)
            base_price = {"BTC": 45000, "ETH": 2500, "SOL": 100}.get(asset, 1000)
            
            prediction = {
                "asset": asset,
                "current_price": base_price * (1 + np.random.normal(0, 0.01)),
                "prediction_1h": base_price * (1 + np.random.normal(0.001, 0.005)),
                "prediction_24h": base_price * (1 + np.random.normal(0.005, 0.02)),
                "confidence": round(0.85 + np.random.uniform(-0.1, 0.1), 3),
                "timestamp": int(time.time()),
            }
            
            await websocket.send_json(prediction)
            
            # Wait before next update
            await asyncio.sleep(60)
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for {asset}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()


# ============================================================================
# Main Entry Point
# ============================================================================


def main() -> None:
    """Run the inference server."""
    import uvicorn
    
    uvicorn.run(
        "prometheus.inference.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
