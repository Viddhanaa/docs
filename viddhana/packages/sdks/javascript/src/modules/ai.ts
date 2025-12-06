/**
 * Prometheus AI Module - AI predictions and portfolio optimization
 */

import type {
  PricePrediction,
  PortfolioOptimization,
  RiskAssessment,
  RiskMetrics,
} from '../types';

/**
 * Prometheus AI module for predictions, optimization, and risk assessment
 *
 * @example
 * ```typescript
 * // Get price prediction
 * const prediction = await client.ai.predictPrice('BTC', 7);
 * console.log(`7-day trend: ${prediction.trend}`);
 *
 * // Optimize portfolio
 * const optimization = await client.ai.optimizePortfolio({
 *   userId: '0x...',
 *   portfolio: { BTC: 50000, ETH: 30000 },
 *   riskTolerance: 0.5,
 *   timeToGoal: 24,
 * });
 * ```
 */
export class PrometheusAI {
  private readonly apiUrl: string;
  private readonly apiKey?: string;

  constructor(apiUrl: string, apiKey?: string) {
    this.apiUrl = apiUrl;
    this.apiKey = apiKey;
  }

  /**
   * Makes an authenticated API request
   * @private
   */
  private async request<T>(endpoint: string, body: object): Promise<T> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };

    if (this.apiKey) {
      headers['Authorization'] = `Bearer ${this.apiKey}`;
    }

    const response = await fetch(`${this.apiUrl}${endpoint}`, {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`API error ${response.status}: ${errorText}`);
    }

    return response.json() as Promise<T>;
  }

  /**
   * Gets price prediction for an asset
   *
   * @param asset - Asset symbol (e.g., 'BTC', 'ETH')
   * @param horizon - Number of days to predict (1-30)
   * @returns Price prediction with daily forecasts and confidence
   */
  async predictPrice(
    asset: string,
    horizon = 7
  ): Promise<PricePrediction> {
    return this.request<PricePrediction>('/v1/predict/price', {
      asset,
      horizon,
    });
  }

  /**
   * Gets portfolio optimization recommendation
   *
   * @param params - Optimization parameters
   * @returns Portfolio optimization with recommended actions
   */
  async optimizePortfolio(params: {
    userId: string;
    portfolio: Record<string, number>;
    riskTolerance: number; // 0-1
    timeToGoal: number; // months
  }): Promise<PortfolioOptimization> {
    return this.request<PortfolioOptimization>('/v1/optimize/portfolio', {
      user_id: params.userId,
      portfolio: params.portfolio,
      risk_tolerance: params.riskTolerance,
      time_to_goal: params.timeToGoal,
    });
  }

  /**
   * Assesses portfolio risk
   *
   * @param portfolio - Portfolio as { asset: value } mapping
   * @returns Risk assessment with metrics and recommendations
   */
  async assessRisk(
    portfolio: Record<string, number>
  ): Promise<RiskAssessment> {
    return this.request<RiskAssessment>('/v1/assess/risk', {
      portfolio,
    });
  }

  /**
   * Gets risk metrics for a portfolio
   *
   * @param portfolio - Portfolio as { asset: value } mapping
   * @returns Risk metrics including VaR, Sharpe ratio, etc.
   */
  async getRiskMetrics(
    portfolio: Record<string, number>
  ): Promise<RiskMetrics> {
    const assessment = await this.assessRisk(portfolio);
    return assessment.metrics;
  }

  /**
   * Gets market sentiment for an asset
   *
   * @param asset - Asset symbol
   * @returns Sentiment analysis
   */
  async getSentiment(asset: string): Promise<{
    asset: string;
    sentiment: 'bullish' | 'bearish' | 'neutral';
    score: number;
    sources: { name: string; sentiment: string; weight: number }[];
    timestamp: number;
  }> {
    return this.request('/v1/sentiment', { asset });
  }

  /**
   * Gets correlation analysis between assets
   *
   * @param assets - Array of asset symbols
   * @returns Correlation matrix
   */
  async getCorrelation(assets: string[]): Promise<{
    assets: string[];
    matrix: number[][];
    period: string;
    timestamp: number;
  }> {
    return this.request('/v1/correlation', { assets });
  }

  /**
   * Gets volatility forecast for an asset
   *
   * @param asset - Asset symbol
   * @param days - Number of days to forecast
   * @returns Volatility forecast
   */
  async getVolatilityForecast(
    asset: string,
    days = 7
  ): Promise<{
    asset: string;
    currentVolatility: number;
    forecast: { day: number; volatility: number; confidence: number }[];
    trend: 'increasing' | 'decreasing' | 'stable';
    timestamp: number;
  }> {
    return this.request('/v1/volatility/forecast', { asset, days });
  }

  /**
   * Streams real-time predictions via WebSocket
   *
   * @param asset - Asset symbol to stream predictions for
   * @param onUpdate - Callback for prediction updates
   * @param onError - Optional error callback
   * @returns Cleanup function to close the connection
   */
  streamPredictions(
    asset: string,
    onUpdate: (prediction: PricePrediction) => void,
    onError?: (error: Error) => void
  ): () => void {
    const wsUrl = this.apiUrl
      .replace('https://', 'wss://')
      .replace('http://', 'ws://');
    
    const ws = new WebSocket(`${wsUrl}/ws/predictions/${asset}`);

    ws.onopen = () => {
      // Send authentication if API key is present
      if (this.apiKey) {
        ws.send(JSON.stringify({ type: 'auth', apiKey: this.apiKey }));
      }
    };

    ws.onmessage = (event: MessageEvent) => {
      try {
        const prediction = JSON.parse(event.data as string) as PricePrediction;
        onUpdate(prediction);
      } catch (error) {
        if (onError) {
          onError(
            error instanceof Error ? error : new Error('Failed to parse prediction')
          );
        }
      }
    };

    ws.onerror = () => {
      if (onError) {
        onError(new Error('WebSocket connection error'));
      }
    };

    ws.onclose = () => {
      // Connection closed
    };

    // Return cleanup function
    return () => {
      ws.close();
    };
  }

  /**
   * Streams portfolio updates via WebSocket
   *
   * @param address - User address to monitor
   * @param onUpdate - Callback for portfolio updates
   * @param onError - Optional error callback
   * @returns Cleanup function to close the connection
   */
  streamPortfolio(
    address: string,
    onUpdate: (update: { type: string; data: unknown }) => void,
    onError?: (error: Error) => void
  ): () => void {
    const wsUrl = this.apiUrl
      .replace('https://', 'wss://')
      .replace('http://', 'ws://');
    
    const ws = new WebSocket(`${wsUrl}/ws/portfolio/${address}`);

    ws.onopen = () => {
      if (this.apiKey) {
        ws.send(JSON.stringify({ type: 'auth', apiKey: this.apiKey }));
      }
    };

    ws.onmessage = (event: MessageEvent) => {
      try {
        const update = JSON.parse(event.data as string) as {
          type: string;
          data: unknown;
        };
        onUpdate(update);
      } catch (error) {
        if (onError) {
          onError(
            error instanceof Error ? error : new Error('Failed to parse update')
          );
        }
      }
    };

    ws.onerror = () => {
      if (onError) {
        onError(new Error('WebSocket connection error'));
      }
    };

    return () => {
      ws.close();
    };
  }

  /**
   * Batch prediction for multiple assets
   *
   * @param assets - Array of asset symbols
   * @param horizon - Number of days to predict
   * @returns Array of predictions
   */
  async batchPredictPrice(
    assets: string[],
    horizon = 7
  ): Promise<PricePrediction[]> {
    const response = await this.request<{ predictions: PricePrediction[] }>(
      '/v1/predict/batch',
      { assets, horizon }
    );
    return response.predictions;
  }

  /**
   * Gets model information
   *
   * @returns AI model metadata
   */
  async getModelInfo(): Promise<{
    version: string;
    lastUpdated: number;
    accuracy: { price: number; volatility: number };
    supportedAssets: string[];
  }> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };

    if (this.apiKey) {
      headers['Authorization'] = `Bearer ${this.apiKey}`;
    }

    const response = await fetch(`${this.apiUrl}/v1/model/info`, {
      method: 'GET',
      headers,
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    return response.json() as Promise<{
      version: string;
      lastUpdated: number;
      accuracy: { price: number; volatility: number };
      supportedAssets: string[];
    }>;
  }
}
