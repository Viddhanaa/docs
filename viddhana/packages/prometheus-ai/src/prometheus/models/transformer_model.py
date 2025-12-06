"""
Transformer-based forecasting model for price prediction.

This module implements a Transformer architecture optimized for
time-series forecasting with attention mechanisms.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for Transformer models.
    """
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1,
    ) -> None:
        """
        Initialize positional encoding.
        
        Args:
            d_model: Model dimension.
            max_len: Maximum sequence length.
            dropout: Dropout probability.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer("pe", pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerForecaster(nn.Module):
    """
    Transformer-based forecasting model for price prediction.
    
    Uses multi-head self-attention to capture long-range dependencies
    in time-series data. Combines local patterns with global context
    for accurate forecasting.
    """
    
    def __init__(
        self,
        input_dim: int = 64,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        output_horizon: int = 7,
        max_seq_len: int = 512,
    ) -> None:
        """
        Initialize Transformer forecaster.
        
        Args:
            input_dim: Number of input features.
            d_model: Dimension of the model.
            nhead: Number of attention heads.
            num_encoder_layers: Number of encoder layers.
            num_decoder_layers: Number of decoder layers.
            dim_feedforward: Dimension of feedforward network.
            dropout: Dropout probability.
            output_horizon: Number of steps to predict ahead.
            max_seq_len: Maximum sequence length.
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.output_horizon = output_horizon
        
        # Input embedding
        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Positional encoding
        self.pos_encoder = TransformerPositionalEncoding(
            d_model=d_model,
            max_len=max_seq_len,
            dropout=dropout,
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
        )
        
        # Optional decoder for autoregressive prediction
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers,
        )
        
        # Learnable query tokens for prediction
        self.query_tokens = nn.Parameter(
            torch.randn(1, output_horizon, d_model) * 0.02
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        
        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )
        
        # Volatility prediction head
        self.volatility_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Softplus(),
        )
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize weights using Xavier initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def _generate_square_subsequent_mask(
        self,
        size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Generate causal mask for autoregressive decoding."""
        mask = torch.triu(
            torch.ones(size, size, device=device) * float("-inf"),
            diagonal=1,
        )
        return mask
    
    def forward(
        self,
        x: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim).
            src_mask: Optional source mask.
            
        Returns:
            Tuple of:
                - predictions: Shape (batch_size, output_horizon).
                - confidence: Shape (batch_size, output_horizon).
                - volatility: Shape (batch_size, output_horizon).
        """
        batch_size = x.size(0)
        
        # Embed input
        x = self.input_embedding(x)
        x = self.pos_encoder(x)
        
        # Encode
        memory = self.encoder(x, mask=src_mask)
        
        # Expand query tokens for batch
        query = self.query_tokens.expand(batch_size, -1, -1)
        
        # Decode with cross-attention to encoded memory
        tgt_mask = self._generate_square_subsequent_mask(
            self.output_horizon,
            x.device,
        )
        decoder_output = self.decoder(
            query,
            memory,
            tgt_mask=tgt_mask,
        )
        
        # Generate outputs
        predictions = self.output_proj(decoder_output).squeeze(-1)
        confidence = self.confidence_head(decoder_output).squeeze(-1)
        volatility = self.volatility_head(decoder_output).squeeze(-1)
        
        return predictions, confidence, volatility
    
    def predict(
        self,
        historical_data: np.ndarray,
        device: str = "cpu",
    ) -> Dict[str, np.ndarray]:
        """
        Generate predictions from numpy array input.
        
        Args:
            historical_data: Input data of shape (seq_len, input_dim) or
                           (batch_size, seq_len, input_dim).
            device: Device to run inference on.
            
        Returns:
            Dictionary with predictions, confidence, and volatility.
        """
        self.eval()
        
        if historical_data.ndim == 2:
            historical_data = historical_data[np.newaxis, ...]
        
        x = torch.tensor(historical_data, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            predictions, confidence, volatility = self.forward(x)
        
        return {
            "predictions": predictions.cpu().numpy(),
            "confidence": confidence.cpu().numpy(),
            "volatility": volatility.cpu().numpy(),
        }


class HybridLSTMTransformer(nn.Module):
    """
    Hybrid LSTM + Transformer model for price forecasting.
    
    Combines LSTM for local sequential patterns with Transformer
    for capturing long-range dependencies.
    """
    
    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 256,
        lstm_layers: int = 2,
        transformer_layers: int = 4,
        nhead: int = 8,
        dropout: float = 0.1,
        output_horizon: int = 7,
    ) -> None:
        """
        Initialize hybrid model.
        
        Args:
            input_dim: Number of input features.
            hidden_dim: Hidden dimension.
            lstm_layers: Number of LSTM layers.
            transformer_layers: Number of transformer encoder layers.
            nhead: Number of attention heads.
            dropout: Dropout probability.
            output_horizon: Number of steps to predict ahead.
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_horizon = output_horizon
        
        # LSTM encoder for local patterns
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True,
        )
        
        # Project LSTM output to transformer dimension
        self.lstm_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Positional encoding
        self.pos_encoder = TransformerPositionalEncoding(
            d_model=hidden_dim,
            dropout=dropout,
        )
        
        # Transformer encoder for global patterns
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers,
        )
        
        # Output heads
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_horizon),
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, output_horizon),
            nn.Sigmoid(),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim).
            context: Optional context features.
            
        Returns:
            Tuple of predictions and confidence scores.
        """
        # LSTM encoding
        lstm_out, _ = self.lstm(x)
        lstm_out = self.lstm_proj(lstm_out)
        
        # Add positional encoding
        lstm_out = self.pos_encoder(lstm_out)
        
        # Transformer encoding
        transformer_out = self.transformer(lstm_out)
        
        # Use last token for prediction
        final_repr = transformer_out[:, -1, :]
        
        # Generate outputs
        predictions = self.output_head(final_repr)
        confidence = self.confidence_head(final_repr)
        
        return predictions, confidence
