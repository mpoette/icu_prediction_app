"""
LSTM pour classification binaire de séries temporelles
======================================================

Module complet et autonome intégrant :
- Architecture LSTM empilée avec options bidirectionnelles
- Entraînement avec early stopping, class weighting, calibration température
- Fine-tuning superficiel (dégel partiel des couches)
- Évaluation robuste avec métriques calibrées

Usage:
    from lstm_time import train_lstm_model, fine_tune_lstm_model, evaluate_lstm_on_test

    # Entraînement from scratch
    model, T, history, splits = train_lstm_model(X_train, y_train)
    
    # Fine-tuning
    model_ft, T_ft, hist_ft, splits_ft = fine_tune_lstm_model(
        "best_lstm_model.pt", X_new, y_new, last_k_layers=1
    )
    
    # Évaluation
    auc, brier, T = evaluate_lstm_on_test(X_test, y_test, "best_lstm_model.pt")

Auteur: Adaptation inspirée de inceptionTimeModified.py
Licence: MIT
"""

from __future__ import annotations
import math
import os
from typing import Dict, Iterable, Optional, Tuple, Union
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    warnings.warn("tqdm non disponible - pas de barre de progression")

try:
    from sklearn.metrics import roc_auc_score, brier_score_loss
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("sklearn non disponible - métriques d'évaluation indisponibles")


def _compute_binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Calcule l'AUC binaire avec fallback sans scikit-learn.
    """
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_score = np.asarray(y_score, dtype=np.float64).reshape(-1)
    uniq = np.unique(y_true)
    if uniq.size < 2:
        raise ValueError("AUC non définie : y_true doit contenir au moins deux classes.")
    # Tri croissant pour obtenir les rangs (Mann-Whitney)
    order = np.argsort(y_score, kind="mergesort")
    y_sorted = y_true[order]
    scores_sorted = y_score[order]
    n_pos = float((y_sorted == 1.0).sum())
    n_neg = float((y_sorted == 0.0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC non définie : besoin d'exemples positifs et négatifs.")
    ranks = np.arange(1, scores_sorted.size + 1, dtype=np.float64)
    # Gestion des ex aequo : moyenne des rangs pour chaque groupe
    unique_scores, counts = np.unique(scores_sorted, return_counts=True)
    start = 0
    avg_ranks = np.empty_like(ranks)
    for count in counts:
        end = start + count
        avg_rank = (start + end + 1) / 2.0
        avg_ranks[start:end] = avg_rank
        start = end
    pos_ranks = avg_ranks[y_sorted == 1.0]
    auc = (pos_ranks.sum() - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
    return float(auc)


# ============================================================================
# ARCHITECTURE LSTM
# ============================================================================

class LSTMClassifier(nn.Module):
    """Modèle LSTM empilé pour classification binaire."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.2,
        fc_units: Optional[Iterable[int]] = None,
        layernorm: bool = True,
        num_pred_classes: int = 1,
    ) -> None:
        super().__init__()
        
        assert input_size > 0, "input_size doit être > 0"
        assert num_layers >= 1, "num_layers doit être >= 1"
        assert hidden_size > 0, "hidden_size doit être > 0"
        assert num_pred_classes >= 1, "num_pred_classes doit être >= 1"
        if fc_units is not None:
            fc_units = list(fc_units)
        
        self.input_args = {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "bidirectional": bidirectional,
            "dropout": dropout,
            "fc_units": fc_units,
            "layernorm": layernorm,
            "num_pred_classes": num_pred_classes,
        }
        
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        
        feat_dim = hidden_size * self.num_directions
        layers = []
        
        if layernorm:
            layers.append(nn.LayerNorm(feat_dim))
        
        layers.append(nn.Dropout(dropout))
        
        if fc_units:
            for units in fc_units:
                layers.append(nn.Linear(feat_dim, units))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                feat_dim = units
        
        self.head = nn.Sequential(*layers) if layers else nn.Identity()
        self.linear = nn.Linear(feat_dim, num_pred_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Entrée attendue (N,T,F), reçu shape {x.shape}")
        
        out, (h_n, _) = self.lstm(x)
        
        # Dernière couche LSTM (concat directions si bidirectionnel)
        if self.bidirectional:
            h_last = torch.cat(
                [h_n[-2], h_n[-1]],
                dim=-1
            )
        else:
            h_last = h_n[-1]
        
        features = self.head(h_last)
        return self.linear(features)


# ============================================================================
# CALIBRATION TEMPÉRATURE
# ============================================================================

class TemperatureCalibrator(nn.Module):
    """Calibration température scalaire"""
    
    def __init__(self, init_T: float = 1.0):
        super().__init__()
        assert init_T > 0, "Température initiale doit être > 0"
        self.log_T = nn.Parameter(
            torch.tensor([math.log(init_T)], dtype=torch.float32)
        )
    
    @property
    def T(self) -> torch.Tensor:
        return self.log_T.exp()
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.T
    
    def fit(
        self,
        logits_val: torch.Tensor,
        y_val: torch.Tensor,
        max_iter: int = 200
    ) -> float:
        self.train()
        opt = torch.optim.LBFGS(
            [self.log_T],
            lr=1.0,
            max_iter=max_iter,
            tolerance_grad=1e-7,
            tolerance_change=1e-9
        )
        
        def closure():
            opt.zero_grad(set_to_none=True)
            z = self.forward(logits_val).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(z, y_val.float())
            loss.backward()
            return loss
        
        loss = opt.step(closure)
        self.eval()
        return float(loss.detach().cpu())


# ============================================================================
# DATASET ET UTILS
# ============================================================================

class TimeSeriesDataset(Dataset):
    """Dataset PyTorch (N, T, F) pour LSTM."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        assert X.ndim == 3, "X doit être (N, T, F)"
        if y.ndim == 2:
            y = y.reshape(-1)
        
        self.X = torch.from_numpy(X).contiguous().float()
        self.y = torch.from_numpy(y).float()
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def stratified_train_val_indices(
    y: np.ndarray,
    val_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Split stratifié manuel pour classification binaire."""
    y = y.reshape(-1)
    rng = np.random.default_rng(seed)
    
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    
    rng.shuffle(idx0)
    rng.shuffle(idx1)
    
    n0_val = int(round(val_ratio * len(idx0)))
    n1_val = int(round(val_ratio * len(idx1)))
    
    val_idx = np.concatenate([idx0[:n0_val], idx1[:n1_val]])
    train_idx = np.concatenate([idx0[n0_val:], idx1[n1_val:]])
    
    rng.shuffle(val_idx)
    rng.shuffle(train_idx)
    
    return train_idx, val_idx


def compute_pos_weight_from_indices(
    y: np.ndarray,
    train_idx: np.ndarray
) -> Optional[torch.Tensor]:
    """Calcule pos_weight pour BCEWithLogitsLoss."""
    ytr = y.reshape(-1)[train_idx]
    
    if set(np.unique(ytr)).issubset({0, 1}):
        pos = (ytr == 1).sum()
        neg = (ytr == 0).sum()
        if pos > 0:
            return torch.tensor([neg / pos], dtype=torch.float32)
    
    return None


def _gather_logits(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device
) -> torch.Tensor:
    """Collecte les logits sur un DataLoader."""
    was_training = model.training
    model.eval()
    outs = []
    
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device, non_blocking=True)
            z = model(xb)
            if z.ndim == 1:
                z = z.unsqueeze(-1)
            outs.append(z)
    
    logits = torch.cat(outs, dim=0)
    
    if was_training:
        model.train()
    
    return logits


# ============================================================================
# ENTRAÎNEMENT FROM SCRATCH
# ============================================================================

def train_lstm_model(
    X: np.ndarray,
    y: np.ndarray,
    *,
    val_ratio: float = 0.2,
    hidden_size: int = 128,
    num_layers: int = 2,
    bidirectional: bool = True,
    dropout: float = 0.2,
    fc_units: Optional[Iterable[int]] = None,
    layernorm: bool = True,
    batch_size: int = 64,
    epochs: int = 100,
    patience: int = 10,
    min_delta: float = 0.0,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    clip_grad: Optional[float] = 1.0,
    use_scheduler: bool = True,
    calibrate: bool = True,
    save_best_path: Optional[str] = "best_lstm_model.pt",
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    progress: bool = True,
) -> Tuple[LSTMClassifier, float, Dict[str, list], Dict[str, np.ndarray]]:
    """
    Entraîne LSTMClassifier from scratch avec gestion complète.
    
    Returns:
        (model, T, history, splits)
    """
    assert X.ndim == 3, f"X doit être (N,T,F), reçu {X.shape}"
    assert X.shape[0] == y.reshape(-1).shape[0], "Mismatch N entre X et y"
    assert X.shape[2] > 0, "X doit avoir au moins 1 feature"
    
    device = torch.device(device)
    
    train_idx, val_idx = stratified_train_val_indices(y, val_ratio=val_ratio, seed=42)
    
    ds = TimeSeriesDataset(X, y)
    train_loader = DataLoader(
        Subset(ds, train_idx),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        Subset(ds, val_idx),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=(device.type == "cuda")
    )
    
    model = LSTMClassifier(
        input_size=X.shape[2],
        hidden_size=hidden_size,
        num_layers=num_layers,
        bidirectional=bidirectional,
        dropout=dropout,
        fc_units=fc_units,
        layernorm=layernorm,
        num_pred_classes=1
    ).to(device)
    
    pos_w = compute_pos_weight_from_indices(y, train_idx)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_w.to(device) if pos_w is not None else None
    )
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    
    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs
        )
    
    best_val_auc = float("-inf")
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    history = {"train_loss": [], "val_loss": [], "val_auc": []}
    patience_count = 0
    
    if progress and TQDM_AVAILABLE:
        pbar = tqdm(range(1, epochs + 1), desc="Training LSTM", leave=True)
    else:
        pbar = range(1, epochs + 1)
    
    for epoch in pbar:
        model.train()
        run_loss = 0.0
        n_obs = 0
        
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device)
            
            logits = model(xb).squeeze(-1)
            loss = criterion(logits, yb)
            
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Loss non finie à epoch {epoch}")
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=clip_grad
                )
            
            optimizer.step()
            
            bs = xb.size(0)
            run_loss += float(loss.detach().cpu()) * bs
            n_obs += bs
        
        train_loss = run_loss / max(1, n_obs)
        
        model.eval()
        val_loss = 0.0
        m_obs = 0
        
        val_probs = []
        val_targets = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device)
                
                logits = model(xb).squeeze(-1)
                loss = criterion(logits, yb)
                
                bs = xb.size(0)
                val_loss += float(loss.detach().cpu()) * bs
                m_obs += bs
                val_probs.append(torch.sigmoid(logits).detach().cpu())
                val_targets.append(yb.detach().cpu())
        
        val_loss /= max(1, m_obs)
        
        val_auc = float("nan")
        if val_probs:
            y_np = torch.cat(val_targets).float().cpu().numpy().reshape(-1)
            p_np = torch.cat(val_probs).float().cpu().numpy().reshape(-1)
            try:
                if SKLEARN_AVAILABLE:
                    val_auc = float(roc_auc_score(y_np, p_np))
                else:
                    val_auc = _compute_binary_auc(y_np, p_np)
            except ValueError as err:
                warnings.warn(f"AUC validation indisponible à l'époque {epoch}: {err}")
        
        if scheduler is not None:
            scheduler.step()
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)
        
        if progress and TQDM_AVAILABLE:
            current_lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix(
                train=f"{train_loss:.4f}",
                val=f"{val_loss:.4f}",
                auc="nan" if not np.isfinite(val_auc) else f"{val_auc:.4f}",
                lr=f"{current_lr:.2e}"
            )
        
        improved = np.isfinite(val_auc) and (val_auc > best_val_auc + min_delta)
        
        if improved:
            best_val_auc = val_auc
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_count = 0
            
            if save_best_path is not None:
                torch.save({
                    "state_dict": best_state,
                    "init_args": model.input_args,
                    "epoch": epoch,
                    "history": history,
                    "train_idx": train_idx,
                    "val_idx": val_idx,
                    "val_loss": val_loss,
                    "val_auc": val_auc
                }, save_best_path)
        else:
            patience_count += 1
            if patience_count >= patience:
                if progress and TQDM_AVAILABLE:
                    pbar.set_postfix_str(f"Early stop @ epoch {epoch}")
                break
    
    if best_state is not None:
        model.load_state_dict(best_state)
        info_auc = "nan" if not np.isfinite(best_val_auc) else f"{best_val_auc:.6f}"
        print(
            f"\n✓ Meilleur modèle chargé (epoch {best_epoch}, "
            f"val_loss={best_val_loss:.6f}, val_auc={info_auc})"
        )
    
    T_value = 1.0
    if calibrate:
        print("Calibration température sur validation...")
        logits_val = _gather_logits(model, val_loader, device)
        y_val = torch.from_numpy(y.reshape(-1)[val_idx]).to(device)
        
        if logits_val.ndim == 1:
            logits_val = logits_val.unsqueeze(-1)
        
        calibrator = TemperatureCalibrator(init_T=1.0).to(device)
        _ = calibrator.fit(logits_val, y_val, max_iter=200)
        T_value = float(calibrator.T.detach().cpu())
        
        print(f"✓ Température calibrée : T = {T_value:.4f}")
        
        if save_best_path is not None and os.path.exists(save_best_path):
            ckpt = torch.load(save_best_path, map_location="cpu", weights_only=False)
            ckpt["temperature"] = T_value
            torch.save(ckpt, save_best_path)
    
    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    splits = {"train_idx": train_idx, "val_idx": val_idx}
    return model, T_value, history, splits


# ============================================================================
# FINE-TUNING SUPERFICIEL
# ============================================================================

def set_trainable_last_k_layers(
    model: LSTMClassifier,
    last_k_layers: int = 1,
    train_linear: bool = True
) -> LSTMClassifier:
    """
    Gèle toutes les couches sauf les K dernières couches LSTM.
    """
    assert isinstance(model, LSTMClassifier), "model doit être LSTMClassifier"
    
    for p in model.parameters():
        p.requires_grad = False
    
    num_layers = model.num_layers
    k = max(0, min(last_k_layers, num_layers))
    target_layers = list(range(num_layers - k, num_layers))
    
    for name, param in model.lstm.named_parameters():
        for layer_idx in target_layers:
            prefixes = [
                f"weight_ih_l{layer_idx}",
                f"weight_hh_l{layer_idx}",
                f"bias_ih_l{layer_idx}",
                f"bias_hh_l{layer_idx}",
            ]
            if model.bidirectional:
                prefixes.extend([
                    f"weight_ih_l{layer_idx}_reverse",
                    f"weight_hh_l{layer_idx}_reverse",
                    f"bias_ih_l{layer_idx}_reverse",
                    f"bias_hh_l{layer_idx}_reverse",
                ])
            if any(name.startswith(prefix) for prefix in prefixes):
                param.requires_grad = True
                break
    
    if train_linear:
        for p in model.head.parameters():
            p.requires_grad = True
        for p in model.linear.parameters():
            p.requires_grad = True
        model.head.train()
        model.linear.train()
    else:
        model.head.eval()
        model.linear.eval()
    
    return model


def load_model_from_checkpoint(
    ckpt_path: str,
    device: Union[str, torch.device] = None
) -> Tuple[LSTMClassifier, dict, Optional[float]]:
    """Charge un modèle depuis un checkpoint."""
    device = torch.device(device) if device is not None else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint non trouvé : {ckpt_path}")
    
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    init_args = ckpt["init_args"]
    state_dict = ckpt.get("state_dict") or ckpt.get("model_state_dict")
    T = ckpt.get("temperature", None)
    
    model = LSTMClassifier(**init_args).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    
    return model, init_args, T


def fine_tune_lstm_model(
    model_or_ckpt: Union[str, LSTMClassifier],
    X_ft: np.ndarray,
    y_ft: np.ndarray,
    *,
    last_k_layers: int = 1,
    train_linear: bool = True,
    reinit_linear: bool = False,
    val_ratio: float = 0.2,
    batch_size: int = 64,
    epochs: int = 50,
    patience: int = 8,
    min_delta: float = 0.0,
    lr: float = 2e-4,
    weight_decay: float = 0.0,
    clip_grad: Optional[float] = 0.5,
    use_scheduler: bool = True,
    calibrate: bool = True,
    save_best_path: Optional[str] = None,
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    progress: bool = True,
) -> Tuple[LSTMClassifier, float, Dict[str, list], Dict[str, np.ndarray]]:
    """Fine-tuning superficiel du modèle LSTM."""
    device = torch.device(device)
    
    if isinstance(model_or_ckpt, str):
        model, init_args, _T = load_model_from_checkpoint(model_or_ckpt, device)
        print(f"Modèle chargé depuis {model_or_ckpt}")
    else:
        model = model_or_ckpt.to(device)
        init_args = getattr(model, "input_args", None) or {}
    
    assert X_ft.ndim == 3, f"X_ft doit être (N,T,F), reçu {X_ft.shape}"
    F_in = X_ft.shape[2]
    if init_args and "input_size" in init_args:
        F_expected = init_args["input_size"]
        assert F_in == F_expected, f"Mismatch features : modèle attend {F_expected}, X_ft a {F_in}"
    
    if reinit_linear:

        if hasattr(model.linear, "reset_parameters"):
            model.linear.reset_parameters()
        else:
            nn.init.kaiming_uniform_(model.linear.weight, a=math.sqrt(5))
            if model.linear.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(model.linear.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(model.linear.bias, -bound, bound)
    
    set_trainable_last_k_layers(
        model,
        last_k_layers=last_k_layers,
        train_linear=train_linear
    )
    
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Fine-tuning : {n_trainable:,} / {n_total:,} paramètres entraînables ({100 * n_trainable / n_total:.1f}%)")
    
    ds = TimeSeriesDataset(X_ft, y_ft)
    train_idx, val_idx = stratified_train_val_indices(y_ft, val_ratio=val_ratio, seed=42)
    
    train_loader = DataLoader(
        Subset(ds, train_idx),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        Subset(ds, val_idx),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=(device.type == "cuda")
    )
    
    pos_w = compute_pos_weight_from_indices(y_ft, train_idx)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_w.to(device) if pos_w is not None else None
    )
    
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )
    
    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs
        )
    
    best_val_auc = float("-inf")
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    history = {"train_loss": [], "val_loss": [], "val_auc": []}
    patience_count = 0
    
    if progress and TQDM_AVAILABLE:
        pbar = tqdm(range(1, epochs + 1), desc="Fine-tuning LSTM", leave=True)
    else:
        pbar = range(1, epochs + 1)
    
    for epoch in pbar:
        model.train()
        run_loss = 0.0
        n_obs = 0
        
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device)
            
            logits = model(xb).squeeze(-1)
            loss = criterion(logits, yb)
            
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Loss non finie à epoch {epoch}")
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    max_norm=clip_grad
                )
            
            optimizer.step()
            
            bs = xb.size(0)
            run_loss += float(loss.detach().cpu()) * bs
            n_obs += bs
        
        train_loss = run_loss / max(1, n_obs)
        
        model.eval()
        val_loss = 0.0
        m_obs = 0
        
        val_probs = []
        val_targets = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device)
                
                logits = model(xb).squeeze(-1)
                loss = criterion(logits, yb)
                
                bs = xb.size(0)
                val_loss += float(loss.detach().cpu()) * bs
                m_obs += bs
                val_probs.append(torch.sigmoid(logits).detach().cpu())
                val_targets.append(yb.detach().cpu())
        
        val_loss /= max(1, m_obs)
        
        val_auc = float("nan")
        if val_probs:
            y_np = torch.cat(val_targets).float().cpu().numpy().reshape(-1)
            p_np = torch.cat(val_probs).float().cpu().numpy().reshape(-1)
            try:
                if SKLEARN_AVAILABLE:
                    val_auc = float(roc_auc_score(y_np, p_np))
                else:
                    val_auc = _compute_binary_auc(y_np, p_np)
            except ValueError as err:
                warnings.warn(f"AUC validation indisponible à l'époque {epoch}: {err}")
        
        if scheduler is not None:
            scheduler.step()
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)
        
        if progress and TQDM_AVAILABLE:
            current_lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix(
                train=f"{train_loss:.4f}",
                val=f"{val_loss:.4f}",
                auc="nan" if not np.isfinite(val_auc) else f"{val_auc:.4f}",
                lr=f"{current_lr:.2e}"
            )
        
        improved = np.isfinite(val_auc) and (val_auc > best_val_auc + min_delta)
        
        if improved:
            best_val_auc = val_auc
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_count = 0
            
            if save_best_path is not None:
                torch.save({
                    "state_dict": best_state,
                    "init_args": getattr(model, "input_args", init_args),
                    "epoch": epoch,
                    "history": history,
                    "train_idx": train_idx,
                    "val_idx": val_idx,
                    "val_loss": val_loss,
                    "val_auc": val_auc
                }, save_best_path)
        else:
            patience_count += 1
            if patience_count >= patience:
                if progress and TQDM_AVAILABLE:
                    pbar.set_postfix_str(f"Early stop @ epoch {epoch}")
                break
    
    if best_state is not None:
        model.load_state_dict(best_state)
        info_auc = "nan" if not np.isfinite(best_val_auc) else f"{best_val_auc:.6f}"
        print(
            f"\n Meilleur modèle chargé (epoch {best_epoch}, "
            f"val_loss={best_val_loss:.6f}, val_auc={info_auc})"
        )
    
    T_value = 1.0
    if calibrate:
        print("Calibration température (validation fine-tuning)")
        logits_val = _gather_logits(model, val_loader, device)
        y_val = torch.from_numpy(y_ft.reshape(-1)[val_idx]).to(device)
        
        if logits_val.ndim == 1:
            logits_val = logits_val.unsqueeze(-1)
        
        calibrator = TemperatureCalibrator(init_T=1.0).to(device)
        _ = calibrator.fit(logits_val, y_val, max_iter=200)
        T_value = float(calibrator.T.detach().cpu())
        
        print(f"Température calibrée : T = {T_value:.4f}")
        
        if save_best_path is not None and os.path.exists(save_best_path):
            ckpt = torch.load(save_best_path, map_location="cpu", weights_only=False)
            ckpt["temperature"] = T_value
            torch.save(ckpt, save_best_path)
    
    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    splits = {"train_idx": train_idx, "val_idx": val_idx}
    return model, T_value, history, splits


# ============================================================================
# ÉVALUATION ROBUSTE
# ============================================================================

@torch.no_grad()
def predict_proba(
    model: nn.Module,
    X: np.ndarray,
    T: float = 1.0,
    device: Union[str, torch.device] = None,
    batch_size: int = 256,
    return_logits: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Prédiction de probabilités calibrées."""
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)
    
    model = model.to(device)
    model.eval()
    
    if not np.isfinite(T) or T <= 0:
        warnings.warn(f"Température invalide ({T}) → fallback T=1.0")
        T = 1.0
    
    if X.ndim != 3:
        raise ValueError(f"X doit être (N,T,F), reçu shape {X.shape}")
    if not np.isfinite(X).all():
        raise ValueError("X contient des NaN/Inf")
    
    y_dummy = np.zeros(len(X))
    ds = TimeSeriesDataset(X, y_dummy)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=(device.type == "cuda")
    )
    
    all_probs = []
    all_logits = [] if return_logits else None
    
    for xb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb).squeeze(-1)
        
        if not torch.isfinite(logits).all():
            warnings.warn("Logits non finis détectés dans un batch")
        
        probs = torch.sigmoid(logits / T)
        
        all_probs.append(probs.cpu().numpy())
        if return_logits:
            all_logits.append(logits.cpu().numpy())
    
    probs = np.concatenate(all_probs, axis=0)
    
    if return_logits:
        logits = np.concatenate(all_logits, axis=0)
        return probs, logits
    
    return probs


def evaluate_lstm_on_test(
    X_test: np.ndarray,
    y_test: np.ndarray,
    checkpoint_path: str,
    *,
    batch_size: int = 256,
    device: Union[str, torch.device] = None,
    return_details: bool = False
) -> Union[Tuple[float, float, float], Tuple[float, float, float, dict]]:
    """Évalue un checkpoint LSTM sur un jeu de test."""
    model, init_args, T = load_model_from_checkpoint(checkpoint_path, device)
    
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)
        model = model.to(device)
    
    if T is None or not np.isfinite(T) or T <= 0:
        warnings.warn(f"Température invalide ou absente ({T}) → T=1.0 (non calibré)")
        T = 1.0
    
    print(f"✓ Modèle chargé : {checkpoint_path}")
    print(f"  Device        : {device}")
    print(f"  Température T : {T:.4f}")
    
    if X_test.ndim != 3:
        raise ValueError(f"X_test doit être (N,T,F), reçu shape {X_test.shape}")
    
    F_required = int(init_args.get("input_size", -1))
    if F_required > 0 and X_test.shape[2] != F_required:
        raise ValueError(
            f"Mismatch features : modèle attend F={F_required}, X_test a F={X_test.shape[2]}"
        )
    
    if y_test.ndim == 2:
        y_test = y_test.reshape(-1)
    
    y_flat = y_test.astype(float)
    uniq = np.unique(y_flat)
    
    if not set(uniq).issubset({0.0, 1.0}):
        raise ValueError(f"y_test doit être binaire {{0,1}}, reçu valeurs uniques : {uniq}")
    if len(uniq) < 2:
        raise ValueError("y_test ne contient qu'une seule classe → AUC non définie")
    if not (np.isfinite(X_test).all() and np.isfinite(y_flat).all()):
        raise ValueError("X_test ou y_test contient des NaN/Inf")
    
    print(f"Prédiction sur {len(X_test)} exemples...")
    try:
        p_test, logits = predict_proba(
            model, X_test, T=T, device=device,
            batch_size=batch_size, return_logits=True
        )
    except Exception as e:
        raise RuntimeError(f"Erreur lors de la prédiction : {e}") from e
    
    print("  Stats prédictions :")
    print(f"    - min  : {np.min(p_test):.6f}")
    print(f"    - max  : {np.max(p_test):.6f}")
    print(f"    - mean : {np.mean(p_test):.6f}")
    print(f"    - NaN  : {np.isnan(p_test).sum()}")
    print(f"    - Inf  : {np.isinf(p_test).sum()}")
    
    if not np.isfinite(p_test).any():
        raise RuntimeError(
            "TOUTES les prédictions sont NaN/Inf ! "
            "Vérifiez que le modèle et les données sont sur le même device."
        )
    
    mask = np.isfinite(p_test) & np.isfinite(y_flat)
    n_invalid = (~mask).sum()
    n_valid = mask.sum()
    
    print(f"  Exemples valides : {n_valid} / {len(y_flat)}")
    
    if n_invalid > 0:
        warnings.warn(f"{n_invalid} exemples avec prédictions non finies (exclus des métriques)")
    if n_valid == 0:
        raise ValueError("Aucune prédiction valide après filtrage")
    
    auc = roc_auc_score(y_flat[mask], p_test[mask])
    brier = brier_score_loss(y_flat[mask], p_test[mask])
    
    print(f"\n{'='*60}")
    print("RÉSULTATS D'ÉVALUATION")
    print(f"{'='*60}")
    print(f"  Checkpoint    : {os.path.basename(checkpoint_path)}")
    print(f"  Device        : {device}")
    print(f"  Température T : {T:.6f}")
    print(f"  AUC-ROC       : {auc:.6f}")
    print(f"  Brier Score   : {brier:.6f}")
    print(f"  N test        : {mask.sum()} / {len(y_flat)}")
    if n_invalid > 0:
        print(f"  N invalides   : {n_invalid}")
    print(f"{'='*60}\n")
    
    if return_details:
        details = {
            "p_test": p_test,
            "logits": logits,
            "y_test": y_flat,
            "mask": mask,
            "init_args": init_args,
            "checkpoint_path": checkpoint_path,
            "device": str(device),
        }
        return auc, brier, T, details
    
    return auc, brier, T


def recompute_temperature(
    model: nn.Module,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: Union[str, torch.device] = None,
    batch_size: int = 256
) -> float:
    """Recalcule la température sur un nouveau jeu de validation."""
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)
    
    model = model.to(device)
    model.eval()
    
    ds = TimeSeriesDataset(X_val, y_val)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    
    logits_val = _gather_logits(model, loader, device)
    y = torch.from_numpy(y_val.reshape(-1)).float().to(device)
    
    if logits_val.ndim == 1:
        logits_val = logits_val.unsqueeze(-1)
    
    calibrator = TemperatureCalibrator(init_T=1.0).to(device)
    _ = calibrator.fit(logits_val, y, max_iter=200)
    T = float(calibrator.T.detach().cpu())
    
    if not np.isfinite(T) or T <= 0:
        warnings.warn(f"Recalibration a produit T={T} invalide → fallback T=1.0")
        T = 1.0
    
    print(f"✓ Nouvelle température calibrée : T = {T:.4f}")
    return T


# ============================================================================
# API SIMPLIFIÉE
# ============================================================================

__all__ = [
    "LSTMClassifier",
    "TemperatureCalibrator",
    "TimeSeriesDataset",
    "train_lstm_model",
    "fine_tune_lstm_model",
    "evaluate_lstm_on_test",
    "predict_proba",
    "load_model_from_checkpoint",
    "recompute_temperature",
    "stratified_train_val_indices",
    "compute_pos_weight_from_indices",
    "set_trainable_last_k_layers",
]


# ============================================================================
# EXEMPLE D'UTILISATION
# ============================================================================

if __name__ == "__main__":
    print("LSTM TimeSeries - Module complet")
    print("=" * 60)
    print("\nExemple d'utilisation :\n")
    
    example_code = '''
# 1. Entraînement from scratch
from lstm_time import train_lstm_model

model, T, history, splits = train_lstm_model(
    X_train, y_train,
    epochs=100,
    patience=10,
    save_best_path="models/lstm_mimic.pt"
)

# 2. Fine-tuning sur nouveau dataset
from lstm_time import fine_tune_lstm_model

model_ft, T_ft, hist_ft, splits_ft = fine_tune_lstm_model(
    "models/lstm_mimic.pt",
    X_ecmo, y_ecmo,
    last_k_layers=1,
    epochs=50,
    save_best_path="models/lstm_ecmo_ft.pt"
)

# 3. Évaluation
from lstm_time import evaluate_lstm_on_test

auc, brier, T = evaluate_lstm_on_test(
    X_test, y_test,
    "models/lstm_ecmo_ft.pt"
)

# 4. Prédiction simple
from lstm_time import predict_proba, load_model_from_checkpoint

model, _, T = load_model_from_checkpoint("models/lstm_ecmo_ft.pt")
probas = predict_proba(model, X_new, T=T)
    '''
    
    print(example_code)
    print("\n" + "=" * 60)
    print("Module prêt à être importé !")
