"""
LSTM-based price prediction model for time-series forecasting.

This module implements an LSTM neural network with positional encoding
for predicting cryptocurrency and asset prices.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """
    Positional encoding for sequence models.
    
    Adds positional information to input embeddings using sine and cosine
    functions of different frequencies.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        """
        Initialize positional encoding.
        
        Args:
            d_model: Dimension of the model embeddings.
            max_len: Maximum sequence length.
            dropout: Dropout probability.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register buffer (not a parameter, but should be saved)
        self.register_buffer("pe", pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input tensor.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
            
        Returns:
            Tensor with positional encoding added.
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class LSTMPredictor(nn.Module):
    """
    LSTM-based price prediction model.
    
    Architecture:
    1. Input projection layer
    2. Bidirectional LSTM layers
    3. Positional encoding
    4. Attention mechanism
    5. Output head for predictions
    6. Confidence estimation head
    
    Implements the time-series forecasting formula:
    y_{t+h} = f(y_{t-w:t}, X_{t-w:t}, C_t)
    """
    
    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 2,
        output_horizon: int = 7,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ) -> None:
        """
        Initialize LSTM predictor.
        
        Args:
            input_dim: Number of input features.
            hidden_dim: LSTM hidden dimension.
            num_layers: Number of LSTM layers.
            output_horizon: Number of days to predict ahead.
            dropout: Dropout probability.
            bidirectional: Whether to use bidirectional LSTM.
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_horizon = output_horizon
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        
        # Project LSTM output
        lstm_output_dim = hidden_dim * self.num_directions
        self.lstm_proj = nn.Linear(lstm_output_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout=dropout)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)
        
        # Output head for price predictions
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, output_horizon),
        )
        
        # Confidence estimation head
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, output_horizon),
            nn.Sigmoid(),
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize model weights using Xavier initialization."""
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
    
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for price prediction.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim).
            context: Optional context features of shape (batch_size, context_dim).
            
        Returns:
            Tuple of:
                - predictions: Price predictions of shape (batch_size, output_horizon).
                - confidence: Confidence scores of shape (batch_size, output_horizon).
        """
        batch_size = x.size(0)
        
        # Input projection
        x = self.input_proj(x)
        
        # LSTM encoding
        lstm_out, (h_n, c_n) = self.lstm(x)
        lstm_out = self.lstm_proj(lstm_out)
        
        # Add positional encoding
        lstm_out = self.pos_encoder(lstm_out)
        
        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        lstm_out = self.attn_norm(lstm_out + attn_out)
        
        # Use last token representation for prediction
        final_repr = lstm_out[:, -1, :]
        
        # Optionally incorporate context
        if context is not None:
            context_proj = nn.Linear(context.size(-1), self.hidden_dim).to(x.device)
            context_repr = context_proj(context)
            final_repr = final_repr + context_repr
        
        # Generate predictions and confidence
        predictions = self.output_head(final_repr)
        confidence = self.confidence_head(final_repr)
        
        return predictions, confidence
    
    def predict(
        self,
        historical_data: np.ndarray,
        device: str = "cpu",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions from numpy array input.
        
        Args:
            historical_data: Input data of shape (seq_len, input_dim) or
                           (batch_size, seq_len, input_dim).
            device: Device to run inference on.
            
        Returns:
            Tuple of predictions and confidence arrays.
        """
        self.eval()
        
        # Handle 2D input
        if historical_data.ndim == 2:
            historical_data = historical_data[np.newaxis, ...]
        
        # Convert to tensor
        x = torch.tensor(historical_data, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            predictions, confidence = self.forward(x)
        
        return predictions.cpu().numpy(), confidence.cpu().numpy()


class LSTMPredictorWithScaling(LSTMPredictor):
    """
    LSTM Predictor with built-in data scaling.
    
    Extends LSTMPredictor to handle automatic scaling and inverse scaling
    of input data and predictions.
    """
    
    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 2,
        output_horizon: int = 7,
        dropout: float = 0.1,
        bidirectional: bool = True,
        scale_min: float = 0.0,
        scale_max: float = 1.0,
    ) -> None:
        """
        Initialize LSTM predictor with scaling parameters.
        
        Args:
            input_dim: Number of input features.
            hidden_dim: LSTM hidden dimension.
            num_layers: Number of LSTM layers.
            output_horizon: Number of days to predict ahead.
            dropout: Dropout probability.
            bidirectional: Whether to use bidirectional LSTM.
            scale_min: Minimum value for scaling.
            scale_max: Maximum value for scaling.
        """
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_horizon=output_horizon,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        
        # Scaling parameters (registered as buffers)
        self.register_buffer("scale_min", torch.tensor(scale_min))
        self.register_buffer("scale_max", torch.tensor(scale_max))
        self.register_buffer("data_min", torch.zeros(input_dim))
        self.register_buffer("data_max", torch.ones(input_dim))
        self.register_buffer("target_min", torch.tensor(0.0))
        self.register_buffer("target_max", torch.tensor(1.0))
    
    def fit_scaler(
        self,
        data: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Fit the scaler to training data.
        
        Args:
            data: Training data of shape (n_samples, seq_len, input_dim).
            target: Optional target data for inverse scaling predictions.
        """
        # Compute min/max across all samples and timesteps
        self.data_min = data.min(dim=0).values.min(dim=0).values
        self.data_max = data.max(dim=0).values.max(dim=0).values
        
        if target is not None:
            self.target_min = target.min()
            self.target_max = target.max()
    
    def scale_input(self, x: torch.Tensor) -> torch.Tensor:
        """Scale input data to [scale_min, scale_max] range."""
        x_scaled = (x - self.data_min) / (self.data_max - self.data_min + 1e-8)
        return x_scaled * (self.scale_max - self.scale_min) + self.scale_min
    
    def inverse_scale_target(self, y: torch.Tensor) -> torch.Tensor:
        """Inverse scale predictions back to original range."""
        y_unscaled = (y - self.scale_min) / (self.scale_max - self.scale_min + 1e-8)
        return y_unscaled * (self.target_max - self.target_min) + self.target_min
