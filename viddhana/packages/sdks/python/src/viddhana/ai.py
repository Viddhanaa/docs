"""AI module for Prometheus AI predictions and optimizations."""

from typing import Dict, List, Optional
import httpx

from .types import (
    PricePrediction,
    PricePredictionPoint,
    PortfolioOptimization,
    RiskAssessment,
)


class AIModule:
    """Module for interacting with Prometheus AI services."""

    def __init__(self, api_url: str, api_key: Optional[str] = None):
        """
        Initialize AIModule.

        Args:
            api_url: Base URL for the AI API.
            api_key: Optional API key for authentication.
        """
        self._api_url = api_url.rstrip("/")
        self._api_key = api_key
        self._headers = self._build_headers()
        self._client = httpx.Client(
            base_url=self._api_url,
            headers=self._headers,
            timeout=30.0,
        )
        self._async_client: Optional[httpx.AsyncClient] = None

    def _build_headers(self) -> Dict[str, str]:
        """Build request headers."""
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _get_async_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(
                base_url=self._api_url,
                headers=self._headers,
                timeout=30.0,
            )
        return self._async_client

    def predict_price(self, asset: str, horizon: int = 7) -> PricePrediction:
        """
        Get price prediction for an asset.

        Args:
            asset: Asset symbol (e.g., 'BTC', 'ETH').
            horizon: Number of days to predict ahead (1-30).

        Returns:
            PricePrediction with daily predictions and confidence scores.

        Raises:
            httpx.HTTPStatusError: If the API request fails.
        """
        response = self._client.post(
            "/v1/predict/price",
            json={"asset": asset, "horizon": horizon},
        )
        response.raise_for_status()
        data = response.json()

        predictions = [
            PricePredictionPoint(
                day=p["day"],
                price=p["price"],
                confidence=p["confidence"],
            )
            for p in data.get("predictions", [])
        ]

        return PricePrediction(
            asset=data["asset"],
            current_price=data["currentPrice"],
            horizon=data["horizon"],
            predictions=predictions,
            trend=data["trend"],
            volatility_forecast=data["volatilityForecast"],
            model_version=data["modelVersion"],
            generated_at=data["generatedAt"],
        )

    async def predict_price_async(self, asset: str, horizon: int = 7) -> PricePrediction:
        """
        Async version of predict_price.

        Args:
            asset: Asset symbol (e.g., 'BTC', 'ETH').
            horizon: Number of days to predict ahead (1-30).

        Returns:
            PricePrediction with daily predictions and confidence scores.
        """
        client = self._get_async_client()
        response = await client.post(
            "/v1/predict/price",
            json={"asset": asset, "horizon": horizon},
        )
        response.raise_for_status()
        data = response.json()

        predictions = [
            PricePredictionPoint(
                day=p["day"],
                price=p["price"],
                confidence=p["confidence"],
            )
            for p in data.get("predictions", [])
        ]

        return PricePrediction(
            asset=data["asset"],
            current_price=data["currentPrice"],
            horizon=data["horizon"],
            predictions=predictions,
            trend=data["trend"],
            volatility_forecast=data["volatilityForecast"],
            model_version=data["modelVersion"],
            generated_at=data["generatedAt"],
        )

    def optimize_portfolio(
        self,
        user_id: str,
        portfolio: Dict[str, float],
        risk_tolerance: float,
        time_to_goal: int,
    ) -> PortfolioOptimization:
        """
        Get portfolio optimization recommendation.

        Args:
            user_id: User identifier (usually wallet address).
            portfolio: Current portfolio as {asset: value} dict.
            risk_tolerance: Risk tolerance 0-1 (0=conservative, 1=aggressive).
            time_to_goal: Months until investment goal.

        Returns:
            PortfolioOptimization with recommended actions.

        Raises:
            httpx.HTTPStatusError: If the API request fails.
        """
        response = self._client.post(
            "/v1/optimize/portfolio",
            json={
                "user_id": user_id,
                "portfolio": portfolio,
                "risk_tolerance": risk_tolerance,
                "time_to_goal": time_to_goal,
            },
        )
        response.raise_for_status()
        data = response.json()

        return PortfolioOptimization(
            action=data["action"],
            recommendations=data["recommendations"],
            confidence=data["confidence"],
            risk_assessment=data["risk_assessment"],
        )

    async def optimize_portfolio_async(
        self,
        user_id: str,
        portfolio: Dict[str, float],
        risk_tolerance: float,
        time_to_goal: int,
    ) -> PortfolioOptimization:
        """
        Async version of optimize_portfolio.

        Args:
            user_id: User identifier (usually wallet address).
            portfolio: Current portfolio as {asset: value} dict.
            risk_tolerance: Risk tolerance 0-1.
            time_to_goal: Months until investment goal.

        Returns:
            PortfolioOptimization with recommended actions.
        """
        client = self._get_async_client()
        response = await client.post(
            "/v1/optimize/portfolio",
            json={
                "user_id": user_id,
                "portfolio": portfolio,
                "risk_tolerance": risk_tolerance,
                "time_to_goal": time_to_goal,
            },
        )
        response.raise_for_status()
        data = response.json()

        return PortfolioOptimization(
            action=data["action"],
            recommendations=data["recommendations"],
            confidence=data["confidence"],
            risk_assessment=data["risk_assessment"],
        )

    def assess_risk(self, portfolio: Dict[str, float]) -> RiskAssessment:
        """
        Assess portfolio risk metrics.

        Args:
            portfolio: Portfolio as {asset: value} dict.

        Returns:
            RiskAssessment with risk metrics and recommendations.

        Raises:
            httpx.HTTPStatusError: If the API request fails.
        """
        response = self._client.post(
            "/v1/assess/risk",
            json={"portfolio": portfolio},
        )
        response.raise_for_status()
        data = response.json()

        return RiskAssessment(
            risk_score=data["risk_score"],
            volatility=data["volatility"],
            sharpe_ratio=data["sharpe_ratio"],
            max_drawdown=data["max_drawdown"],
            recommendations=data["recommendations"],
        )

    async def assess_risk_async(self, portfolio: Dict[str, float]) -> RiskAssessment:
        """
        Async version of assess_risk.

        Args:
            portfolio: Portfolio as {asset: value} dict.

        Returns:
            RiskAssessment with risk metrics and recommendations.
        """
        client = self._get_async_client()
        response = await client.post(
            "/v1/assess/risk",
            json={"portfolio": portfolio},
        )
        response.raise_for_status()
        data = response.json()

        return RiskAssessment(
            risk_score=data["risk_score"],
            volatility=data["volatility"],
            sharpe_ratio=data["sharpe_ratio"],
            max_drawdown=data["max_drawdown"],
            recommendations=data["recommendations"],
        )

    def close(self):
        """Close HTTP clients."""
        self._client.close()

    async def aclose(self):
        """Close async HTTP client."""
        if self._async_client:
            await self._async_client.aclose()
            self._async_client = None
