from pathlib import Path
from typing import Iterable, Mapping, Union
from torch.utils.tensorboard.writer import SummaryWriter


class SummaryLogger:
    """Tensorboard 日志管理器，最小化对训练循环的侵入

    `log_learning_rate` 现在支持传入单个 lr（float/int）、一个可迭代的 lr 列表/元组，
    或者一个以名称为键的字典（Mapping[str, float]）来记录分层学习率。
    """

    def __init__(self, log_dir: str = "log"):
        """
        初始化 SummaryBoard

        Args:
            log_dir: 日志目录路径，相对或绝对路径
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.log_dir))

    def log_train_step(self, global_step: int, loss: float, top1: float, top5: float):
        """记录训练步骤的指标"""
        self.writer.add_scalar("train/loss", loss, global_step)
        self.writer.add_scalar("train/accuracy_top1", top1, global_step)
        self.writer.add_scalar("train/accuracy_top5", top5, global_step)

    def log_train_epoch(self, epoch: int, loss: float, top1: float, top5: float):
        """记录每个 epoch 的训练统计"""
        self.writer.add_scalar("epoch/train_loss", loss, epoch)
        self.writer.add_scalar("epoch/train_top1", top1, epoch)
        self.writer.add_scalar("epoch/train_top5", top5, epoch)

    def log_validation(self, epoch: int, loss: float, top1: float, top5: float, is_best: bool = False):
        """记录验证指标"""
        self.writer.add_scalar("epoch/val_loss", loss, epoch)
        self.writer.add_scalar("epoch/val_top1", top1, epoch)
        self.writer.add_scalar("epoch/val_top5", top5, epoch)
        if is_best:
            self.writer.add_scalar("epoch/val_best_top5", top5, epoch)

    def log_learning_rate(self, epoch: int, lr: Union[float, int, Iterable[Union[float, int]], Mapping[str, Union[float, int]]]):
        """记录学习率，支持单值、列表/元组或字典（名称->lr）。

        Examples:
            `log_learning_rate(epoch, 1e-3)` -> writes `lr/learning_rate`
            `log_learning_rate(epoch, [1e-3, 5e-4])` -> writes `lr/group_0`, `lr/group_1`
            `log_learning_rate(epoch, {"backbone":1e-4, "head":1e-3})` -> writes `lr/backbone`, `lr/head`
        """
        # 单值
        if isinstance(lr, (float, int)):
            self.writer.add_scalar("lr/learning_rate", float(lr), epoch)
            return

        # 映射（命名分组）
        if isinstance(lr, Mapping):
            for name, value in lr.items():
                try:
                    self.writer.add_scalar(f"lr/{name}", float(value), epoch)
                except Exception:
                    continue
            return

        # 可迭代（按索引记录）
        if isinstance(lr, Iterable):
            for i, value in enumerate(lr):
                try:
                    self.writer.add_scalar(f"lr/group_{i}", float(value), epoch)
                except Exception:
                    continue
            return

    def close(self):
        """关闭 writer"""
        if self.writer:
            self.writer.flush()
            self.writer.close()