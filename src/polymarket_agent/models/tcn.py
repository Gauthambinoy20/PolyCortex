"""Temporal Convolutional Network with Beta distribution head and ensemble."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class PolymarketTCN(nn.Module):
    """TCN backbone outputting Beta distribution parameters."""

    def __init__(
        self,
        n_features: int = 10,
        n_channels: int = 32,
        n_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Conv1d(n_features, n_channels, 1)
        self.layers = nn.ModuleList(
            [self._make_block(n_channels, kernel_size, dilation=2**i, dropout=dropout) for i in range(n_layers)]
        )
        self.attention = nn.MultiheadAttention(
            n_channels,
            num_heads=4,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(n_channels)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.alpha_head = nn.Sequential(
            nn.Linear(n_channels, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Softplus(),
        )
        self.beta_head = nn.Sequential(
            nn.Linear(n_channels, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Softplus(),
        )

    def _make_block(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> nn.Sequential:
        padding = (kernel_size - 1) * dilation
        return nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=padding),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (batch, timesteps, features)
        x = x.transpose(1, 2)  # (batch, features, timesteps)
        x = self.input_proj(x)  # (batch, channels, timesteps)

        for layer in self.layers:
            residual = x
            out = layer(x)
            out = out[:, :, : x.shape[2]]  # causal trim
            x = out + residual

        # Self-attention
        x = x.transpose(1, 2)  # (batch, timesteps, channels)
        attn_out, attn_weights = self.attention(x, x, x)
        x = self.norm(x + attn_out)

        # Global average pooling
        x = x.transpose(1, 2)  # (batch, channels, timesteps)
        h = self.gap(x).squeeze(-1)  # (batch, channels)

        alpha: torch.Tensor = self.alpha_head(h).squeeze(-1) + 1.0
        beta: torch.Tensor = self.beta_head(h).squeeze(-1) + 1.0
        predicted_prob = alpha / (alpha + beta)
        uncertainty = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))

        return (predicted_prob, uncertainty, attn_weights)


class PolymarketEnsemble(nn.Module):
    """Ensemble of 3 TCN variants with learnable meta-weights."""

    def __init__(self, n_features: int = 10) -> None:
        super().__init__()
        self.models = nn.ModuleList(
            [
                PolymarketTCN(n_features, n_channels=32, n_layers=4, kernel_size=3),
                PolymarketTCN(n_features, n_channels=24, n_layers=6, kernel_size=5),
                PolymarketTCN(n_features, n_channels=48, n_layers=3, kernel_size=7),
            ]
        )
        self.meta_weights = nn.Parameter(torch.ones(3) / 3)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        probs: list[torch.Tensor] = []
        uncertainties: list[torch.Tensor] = []

        for model in self.models:
            prob, unc, _ = model(x)
            probs.append(prob)
            uncertainties.append(unc)

        prob_stack = torch.stack(probs, dim=-1)  # (batch, 3)
        unc_stack = torch.stack(uncertainties, dim=-1)  # (batch, 3)
        weights = F.softmax(self.meta_weights, dim=0)  # (3,)

        # Inverse-uncertainty weighting multiplied by meta weights
        inv_unc = 1.0 / (unc_stack + 1e-8)
        combined_weights = inv_unc * weights.unsqueeze(0)  # (batch, 3)
        combined_weights = combined_weights / combined_weights.sum(dim=-1, keepdim=True)

        ensemble_prob = (prob_stack * combined_weights).sum(dim=-1)
        ensemble_uncertainty = (unc_stack * combined_weights).sum(dim=-1)

        return (ensemble_prob, ensemble_uncertainty)


def train_tcn(
    model: PolymarketEnsemble,
    train_loader: DataLoader[Any],
    val_loader: DataLoader[Any],
    *,
    epochs: int = 50,
    lr: float = 1e-3,
    patience: int = 10,
    save_path: str = "data/models/tcn_latest.pt",
) -> dict[str, Any]:
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
    )
    max_grad_norm = 1.0

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        # --- Training ---
        model.train()
        epoch_loss = 0.0
        n_train = 0
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            predicted_prob, _ = model(features)
            loss = ((predicted_prob - targets) ** 2).mean()  # Brier score
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            epoch_loss += loss.item() * features.size(0)
            n_train += features.size(0)

        avg_train = epoch_loss / max(n_train, 1)
        train_losses.append(avg_train)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(device)
                targets = targets.to(device)
                predicted_prob, _ = model(features)
                loss = ((predicted_prob - targets) ** 2).mean()
                val_loss += loss.item() * features.size(0)
                n_val += features.size(0)

        avg_val = val_loss / max(n_val, 1)
        val_losses.append(avg_val)
        scheduler.step(avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
        else:
            epochs_no_improve += 1

        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                "Epoch %d/%d  train_loss=%.6f  val_loss=%.6f  best=%.6f@%d",
                epoch,
                epochs,
                avg_train,
                avg_val,
                best_val_loss,
                best_epoch,
            )

        if epochs_no_improve >= patience:
            logger.info("Early stopping at epoch %d (patience=%d)", epoch, patience)
            break

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
    }
