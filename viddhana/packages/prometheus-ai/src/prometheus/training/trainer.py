"""
Model training pipeline for Prometheus AI.

Provides training utilities for:
- LSTM/Transformer price prediction models
- Q-Learning portfolio optimization agent
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    
    # Training parameters
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    
    # Scheduler parameters
    use_scheduler: bool = True
    scheduler_type: str = "onecycle"  # "onecycle", "cosine", "step"
    warmup_steps: int = 100
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 10
    min_delta: float = 0.0001
    
    # Gradient clipping
    gradient_clip: float = 1.0
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_best_only: bool = True
    
    # Logging
    log_interval: int = 10
    eval_interval: int = 1
    
    # Device
    device: str = "auto"  # "auto", "cuda", "cpu"
    
    def get_device(self) -> torch.device:
        """Get the appropriate device."""
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)


@dataclass
class TrainingMetrics:
    """Metrics collected during training."""
    
    epoch: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    mae: float = 0.0
    mape: float = 0.0
    rmse: float = 0.0
    learning_rate: float = 0.0
    epoch_time: float = 0.0


class EarlyStopping:
    """Early stopping handler to prevent overfitting."""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
    ) -> None:
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs with no improvement.
            min_delta: Minimum change to qualify as improvement.
            mode: "min" for loss, "max" for accuracy.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation metric.
            
        Returns:
            True if training should stop.
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False


class ModelTrainer:
    """
    Training pipeline for PyTorch models.
    
    Supports:
    - Learning rate scheduling (OneCycleLR, CosineAnnealing, StepLR)
    - Early stopping
    - Gradient clipping
    - Model checkpointing
    - Training metrics logging
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[TrainingConfig] = None,
    ) -> None:
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model to train.
            config: Training configuration.
        """
        self.config = config or TrainingConfig()
        self.device = self.config.get_device()
        
        self.model = model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Scheduler (initialized in train())
        self.scheduler: Optional[Any] = None
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config.patience,
            min_delta=self.config.min_delta,
        ) if self.config.early_stopping else None
        
        # Metrics history
        self.history: List[TrainingMetrics] = []
        self.best_val_loss = float("inf")
        
        # Create checkpoint directory
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    def create_scheduler(self, steps_per_epoch: int) -> None:
        """Create learning rate scheduler."""
        if not self.config.use_scheduler:
            return
        
        total_steps = self.config.epochs * steps_per_epoch
        
        if self.config.scheduler_type == "onecycle":
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.config.learning_rate,
                epochs=self.config.epochs,
                steps_per_epoch=steps_per_epoch,
            )
        elif self.config.scheduler_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
            )
        elif self.config.scheduler_type == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=steps_per_epoch * 10,
                gamma=0.1,
            )
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        callbacks: Optional[List[Callable]] = None,
    ) -> Dict[str, Any]:
        """
        Run the training loop.
        
        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            callbacks: Optional callback functions.
            
        Returns:
            Training results dictionary.
        """
        logger.info(f"Starting training on {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Create scheduler
        self.create_scheduler(len(train_loader))
        
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            epoch_start = time.time()
            
            # Training phase
            train_loss = self._train_epoch(train_loader)
            
            # Validation phase
            val_metrics = {}
            if val_loader is not None:
                val_loss, val_metrics = self._validate(val_loader)
            else:
                val_loss = train_loss
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]
            
            # Create metrics
            metrics = TrainingMetrics(
                epoch=epoch + 1,
                train_loss=train_loss,
                val_loss=val_loss,
                mae=val_metrics.get("mae", 0),
                mape=val_metrics.get("mape", 0),
                rmse=val_metrics.get("rmse", 0),
                learning_rate=current_lr,
                epoch_time=time.time() - epoch_start,
            )
            self.history.append(metrics)
            
            # Logging
            if (epoch + 1) % self.config.log_interval == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{self.config.epochs} - "
                    f"train_loss: {train_loss:.4f}, "
                    f"val_loss: {val_loss:.4f}, "
                    f"mae: {val_metrics.get('mae', 0):.4f}, "
                    f"lr: {current_lr:.6f}"
                )
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                if self.config.save_best_only:
                    self.save_checkpoint("best_model.pt")
            
            # Periodic checkpoint
            if not self.config.save_best_only and (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")
            
            # Early stopping
            if self.early_stopping is not None:
                if self.early_stopping(val_loss):
                    logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break
            
            # Run callbacks
            if callbacks:
                for callback in callbacks:
                    callback(epoch, metrics)
        
        total_time = time.time() - start_time
        
        return {
            "best_val_loss": self.best_val_loss,
            "final_val_loss": self.history[-1].val_loss if self.history else 0,
            "epochs_trained": len(self.history),
            "total_time": total_time,
            "history": self.history,
        }
    
    def _train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            # Handle different batch formats
            if len(batch) == 2:
                x, y = batch
            else:
                x, y = batch[0], batch[1]
            
            x = x.to(self.device)
            y = y.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            output = self.model(x)
            
            # Handle tuple output (predictions, confidence)
            if isinstance(output, tuple):
                predictions = output[0]
            else:
                predictions = output
            
            loss = self.criterion(predictions, y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip,
                )
            
            self.optimizer.step()
            
            # Step scheduler if per-batch
            if self.scheduler is not None and self.config.scheduler_type == "onecycle":
                self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        # Step scheduler if per-epoch
        if self.scheduler is not None and self.config.scheduler_type != "onecycle":
            self.scheduler.step()
        
        return total_loss / num_batches
    
    def _validate(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 2:
                    x, y = batch
                else:
                    x, y = batch[0], batch[1]
                
                x = x.to(self.device)
                y = y.to(self.device)
                
                output = self.model(x)
                
                if isinstance(output, tuple):
                    predictions = output[0]
                else:
                    predictions = output
                
                loss = self.criterion(predictions, y)
                total_loss += loss.item()
                
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(y.cpu().numpy())
        
        # Calculate metrics
        predictions = np.concatenate(all_predictions)
        targets = np.concatenate(all_targets)
        
        mae = np.mean(np.abs(predictions - targets))
        mape = np.mean(np.abs((targets - predictions) / (targets + 1e-8))) * 100
        rmse = np.sqrt(np.mean((predictions - targets) ** 2))
        
        metrics = {
            "mae": float(mae),
            "mape": float(mape),
            "rmse": float(rmse),
        }
        
        return total_loss / len(val_loader), metrics
    
    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        path = os.path.join(self.config.checkpoint_dir, filename)
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "best_val_loss": self.best_val_loss,
            "history": self.history,
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, filename: str) -> None:
        """Load model checkpoint."""
        path = os.path.join(self.config.checkpoint_dir, filename)
        
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.history = checkpoint.get("history", [])
        
        if "scheduler_state_dict" in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        logger.info(f"Loaded checkpoint from {path}")


class RLTrainer:
    """
    Training pipeline for reinforcement learning agent.
    
    Trains the Q-Learning portfolio optimizer through simulated
    market environments.
    """
    
    def __init__(
        self,
        optimizer,  # PortfolioOptimizer
        config: Optional[TrainingConfig] = None,
    ) -> None:
        """
        Initialize RL trainer.
        
        Args:
            optimizer: Portfolio optimizer agent.
            config: Training configuration.
        """
        self.optimizer = optimizer
        self.config = config or TrainingConfig()
        self.history: List[Dict[str, float]] = []
    
    def train(
        self,
        episodes: int = 1000,
        max_steps: int = 252,  # Trading days per year
        callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Train the RL agent through simulated episodes.
        
        Args:
            episodes: Number of training episodes.
            max_steps: Maximum steps per episode.
            callback: Optional callback function.
            
        Returns:
            Training results dictionary.
        """
        logger.info(f"Starting RL training for {episodes} episodes")
        
        total_rewards = []
        losses = []
        
        for episode in range(episodes):
            episode_reward = 0
            episode_loss = 0
            num_steps = 0
            
            # Initialize episode
            state = self._get_initial_state()
            
            for step in range(max_steps):
                # Select action
                action = self.optimizer.select_action(state, training=True)
                
                # Execute action and get reward
                next_state, reward, done = self._step(state, action)
                
                # Store transition
                self.optimizer.replay_buffer.push(
                    state,
                    action.value,
                    reward,
                    next_state,
                    done,
                )
                
                # Train step
                loss = self.optimizer.train_step()
                
                episode_reward += reward
                episode_loss += loss
                num_steps += 1
                
                state = next_state
                
                if done:
                    break
            
            # Update target network
            if (episode + 1) % 10 == 0:
                self.optimizer.update_target_network()
            
            # Record metrics
            avg_loss = episode_loss / num_steps if num_steps > 0 else 0
            total_rewards.append(episode_reward)
            losses.append(avg_loss)
            
            self.history.append({
                "episode": episode + 1,
                "reward": episode_reward,
                "loss": avg_loss,
                "epsilon": self.optimizer.epsilon,
            })
            
            # Logging
            if (episode + 1) % self.config.log_interval == 0:
                avg_reward = np.mean(total_rewards[-100:])
                logger.info(
                    f"Episode {episode + 1}/{episodes} - "
                    f"reward: {episode_reward:.2f}, "
                    f"avg_reward: {avg_reward:.2f}, "
                    f"loss: {avg_loss:.4f}, "
                    f"epsilon: {self.optimizer.epsilon:.3f}"
                )
            
            if callback:
                callback(episode, self.history[-1])
        
        return {
            "total_episodes": episodes,
            "final_reward": total_rewards[-1] if total_rewards else 0,
            "avg_reward": np.mean(total_rewards[-100:]) if total_rewards else 0,
            "history": self.history,
        }
    
    def _get_initial_state(self) -> np.ndarray:
        """Get initial state for an episode."""
        # Simulated initial state
        return np.random.randn(self.optimizer.state_dim).astype(np.float32)
    
    def _step(
        self,
        state: np.ndarray,
        action,  # Action enum
    ) -> Tuple[np.ndarray, float, bool]:
        """
        Execute action and return next state, reward, done.
        
        This is a simulated environment for training.
        In production, this would interact with real market data.
        """
        # Simulate state transition
        next_state = state + np.random.randn(len(state)) * 0.1
        next_state = next_state.astype(np.float32)
        
        # Simulate reward based on action
        reward = np.random.randn() * 0.5  # Random reward for simulation
        
        # Episode terminates with small probability
        done = np.random.random() < 0.01
        
        return next_state, reward, done
