from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class ModelSaver:
    """Checkpoint saver and loader for model, optimizer, scheduler, and training state."""
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        """Initialize checkpoint directory.
        
        Args:
            checkpoint_dir: Path to checkpoint directory. Relative paths are resolved from cwd.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_score = float('-inf')
        self.last_train_level = None  # Track last train_level to detect level changes
    
    def save(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LambdaLR,
        level_epoch: int,
        best_score: float,
        current_train_level: int = 0,
        grad_scaler: Optional[Any] = None,
        is_best: bool = False
    ) -> Path:
        """Save checkpoint.
        
        Args:
            model: Neural network model.
            optimizer: Optimizer instance.
            scheduler: Learning rate scheduler.
            level_epoch: Current epoch number within this training level.
            best_score: Best validation metric score so far.
            current_train_level: Current training level (default 0).
            grad_scaler: GradScaler instance (optional).
            is_best: Whether this is the best checkpoint.
        
        Returns:
            Path to saved checkpoint.
        """
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_level': current_train_level,
            'level_epoch': level_epoch,
            'best_score': best_score,
        }
        
        if grad_scaler is not None:
            checkpoint['grad_scaler_state_dict'] = grad_scaler.state_dict()
        
        # Save as last checkpoint
        last_path = self.checkpoint_dir / "last.pt"
        torch.save(checkpoint, last_path)
        
        # Save as best checkpoint if applicable
        if is_best:
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(checkpoint, best_path)
            self.best_score = best_score
        
        self.last_train_level = current_train_level
        return last_path
    
    def load(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LambdaLR,
        current_train_level: int = 0,
        grad_scaler: Optional[Any] = None,
        checkpoint_name: str = "last"
    ) -> Tuple[int, float, bool]:
        """Load checkpoint if exists, handling multi-level training transitions.
        
        Args:
            model: Neural network model to restore.
            optimizer: Optimizer to restore.
            scheduler: Scheduler to restore.
            current_train_level: Current training level (default 0).
            grad_scaler: GradScaler to restore (optional).
            checkpoint_name: Either 'best' or 'last' (default: 'last').
        
        Returns:
            Tuple of (level_epoch, best_score, loaded_successfully).
            If not loaded: (0, float('-inf'), False)
        """
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.pt"
        
        if not checkpoint_path.exists():
            return 0, float('-inf'), False
        
        try:
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            
            # At minimum we require model weights to proceed; other keys may be
            # intentionally absent if a previous level transition cleared them.
            if 'model_state_dict' not in checkpoint:
                print(f"Checkpoint corrupted: Missing model_state_dict. Found keys: {list(checkpoint.keys())}")
                return 0, float('-inf'), False

            # Inspect saved train level and epoch metadata
            saved_train_level = checkpoint.get('train_level', 0)
            level_epoch = checkpoint.get('level_epoch', 0)
            best_score = checkpoint.get('best_score', float('-inf'))

            assert saved_train_level <= current_train_level, "Checkpoint train_level cannot be greater than current_train_level"

            # If training level has changed (upgrade), archive only the epoch
            # count and keep model weights — clear optimizer/scheduler/grad_scaler
            # so the new level starts with a clean optimization state.
            if saved_train_level < current_train_level:
                print(f"[Level transition] Previous level: {saved_train_level}, Current level: {current_train_level}")
                old_epoch_key = f"L{saved_train_level}_epoch"
                checkpoint[old_epoch_key] = level_epoch
                print(f"  Archived previous epoch count: {old_epoch_key}={level_epoch}")

                # Reset level epoch and best_score for new level
                level_epoch = 0
                checkpoint['level_epoch'] = level_epoch
                checkpoint['best_score'] = float('-inf')

                # Update train_level to reflect the new active level in the
                # checkpoint and remove optimizer/scheduler/grad_scaler state
                checkpoint['train_level'] = current_train_level
                for k in ('optimizer_state_dict', 'scheduler_state_dict', 'grad_scaler_state_dict'):
                    if k in checkpoint:
                        del checkpoint[k]

                # Persist the transitioned checkpoint so files reflect the
                # archived epoch and cleared states.
                checkpoint_path = self.checkpoint_dir / "last.pt"
                try:
                    torch.save(checkpoint, checkpoint_path)
                    print(f"  Processed immediate save; archived and cleared optimizer/scheduler state")
                except Exception as e:
                    print(f"  ⚠️  Failed to save transition checkpoint: {e}")

                # Always restore model weights for the caller; do NOT restore
                # optimizer/scheduler/grad_scaler — caller should use the fresh
                # optimizer/scheduler instances it created.
                model.load_state_dict(checkpoint['model_state_dict'])

                self.last_train_level = current_train_level
                self.best_score = checkpoint['best_score']
                return level_epoch, self.best_score, True

            # saved_train_level == current_train_level: restore full state when present
            required_keys = {'optimizer_state_dict', 'scheduler_state_dict'}
            if not required_keys.issubset(checkpoint.keys()):
                print(f"Warning: checkpoint missing optimizer/scheduler state; will restore model weights only and start fresh optimizer/scheduler")
                model.load_state_dict(checkpoint['model_state_dict'])
                if grad_scaler is not None and 'grad_scaler_state_dict' in checkpoint:
                    grad_scaler.load_state_dict(checkpoint['grad_scaler_state_dict'])

                self.last_train_level = current_train_level
                self.best_score = best_score
                return level_epoch, best_score, True

            # Full restore path
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            if grad_scaler is not None and 'grad_scaler_state_dict' in checkpoint:
                grad_scaler.load_state_dict(checkpoint['grad_scaler_state_dict'])

            self.last_train_level = current_train_level
            self.best_score = best_score
            return level_epoch, best_score, True
        except Exception as e:
            print(f"Failed to load checkpoint {checkpoint_path}: {e}")
            return 0, float('-inf'), False