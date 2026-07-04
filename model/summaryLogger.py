from pathlib import Path
from typing import Iterable, Mapping, Union

try:
    from torch.utils.tensorboard.writer import SummaryWriter
except Exception:  # tensorboard is optional for smoke tests / minimal runtime environments
    SummaryWriter = None


class _NullWriter:
    def add_scalar(self, *args, **kwargs):
        return None

    def flush(self):
        return None

    def close(self):
        return None


class SummaryLogger:
    """TensorBoard scalar logger with a no-op fallback when tensorboard is absent."""

    def __init__(self, log_dir: str = "log"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if SummaryWriter is None:
            print("TensorBoard is not installed; SummaryLogger will run in no-op mode.")
            self.writer = _NullWriter()
        else:
            self.writer = SummaryWriter(str(self.log_dir))

    def log_train_step(self, global_step: int, loss: float, top1: float, top5: float):
        self.writer.add_scalar("train/loss", loss, global_step)
        self.writer.add_scalar("train/accuracy_top1", top1, global_step)
        self.writer.add_scalar("train/accuracy_top5", top5, global_step)

    def log_train_epoch(self, epoch: int, loss: float, top1: float, top5: float):
        self.writer.add_scalar("epoch/train_loss", loss, epoch)
        self.writer.add_scalar("epoch/train_top1", top1, epoch)
        self.writer.add_scalar("epoch/train_top5", top5, epoch)

    def log_validation(self, epoch: int, loss: float, top1: float, top5: float, is_best: bool = False):
        self.writer.add_scalar("epoch/val_loss", loss, epoch)
        self.writer.add_scalar("epoch/val_top1", top1, epoch)
        self.writer.add_scalar("epoch/val_top5", top5, epoch)
        if is_best:
            self.writer.add_scalar("epoch/val_best_top5", top5, epoch)

    def log_learning_rate(
        self,
        epoch: int,
        lr: Union[float, int, Iterable[Union[float, int]], Mapping[str, Union[float, int]]],
    ):
        if isinstance(lr, (float, int)):
            self.writer.add_scalar("lr/learning_rate", float(lr), epoch)
            return
        if isinstance(lr, Mapping):
            for name, value in lr.items():
                try:
                    self.writer.add_scalar(f"lr/{name}", float(value), epoch)
                except Exception:
                    continue
            return
        if isinstance(lr, Iterable):
            for i, value in enumerate(lr):
                try:
                    self.writer.add_scalar(f"lr/group_{i}", float(value), epoch)
                except Exception:
                    continue

    def close(self):
        if self.writer:
            self.writer.flush()
            self.writer.close()
