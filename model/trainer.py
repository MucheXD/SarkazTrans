from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp.autocast_mode import autocast
from torch.utils.data import DataLoader

from sarkazBert import SarkazBert
from modelSaver import ModelSaver
from summaryLogger import SummaryLogger
from dataset import SarkazCharmap


class SarkazBertTrainer:
    """Trainer for one-hot training or cached-teacher distillation."""

    def __init__(
        self,
        model: SarkazBert,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LambdaLR,
        model_saver: ModelSaver,
        criterion: nn.Module,
        device_str: str = "cpu",
        enable_amp: bool = True,
        enable_gradient_checkpointing: bool = True,
        early_stop_patience: int = 10,
        accumulation_steps: int = 4,
        validate_interval: int = 100,
        log_interval: int = 10,
        current_train_level: int = 0,
        freeze_bert_epochs: int = 0,
        hard_mask: float = -1.0e4,
        sarkaz_mapping: SarkazCharmap = SarkazCharmap(),
        use_teacher_knowledge: bool = False,
        teacher_temperature: float = 1.0,
        summary_logger: Optional[SummaryLogger] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device_str = device_str
        self.model_saver = model_saver
        self.enable_amp = bool(enable_amp and str(device_str).startswith("cuda"))
        self.amp_device_type = "cuda" if str(device_str).startswith("cuda") else "cpu"
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.early_stop_patience = early_stop_patience
        self.accumulation_steps = max(1, int(accumulation_steps))
        self.validate_interval = validate_interval
        self.log_interval = log_interval
        self.current_train_level = current_train_level
        self.freeze_bert_epochs = max(0, freeze_bert_epochs)
        self.hard_mask = hard_mask
        self.sarkaz_mapping = sarkaz_mapping
        self.use_teacher_knowledge = bool(use_teacher_knowledge)
        self.teacher_temperature = float(teacher_temperature)
        if self.use_teacher_knowledge and self.teacher_temperature <= 0:
            raise ValueError("teacher_temperature must be positive when teacher knowledge is enabled")
        self.summary_logger = summary_logger

        self.current_epoch = 0
        self.best_score = float("-inf")
        self.history_stats = {"train_loss": [], "val_loss": [], "val_metric": []}
        self.early_stop_counter = 0
        self.global_step = 0
        self._should_early_stop = False
        self._bert_is_trainable: Optional[bool] = None
        self.grad_scaler = torch.amp.GradScaler(enabled=self.enable_amp)

    def _set_bert_trainable(self, trainable: bool) -> None:
        if hasattr(self.model, "set_bert_trainable"):
            self.model.set_bert_trainable(trainable)
        else:
            for param in self.model.bert_model.parameters():
                param.requires_grad = trainable

    def _apply_bert_freeze_for_epoch(self, epoch: int) -> None:
        trainable = epoch >= self.freeze_bert_epochs
        if self._bert_is_trainable is None or self._bert_is_trainable != trainable:
            if self.freeze_bert_epochs > 0:
                if trainable:
                    print(f"[O] BERT backbone and MLM head unfrozen at epoch {epoch}")
                else:
                    print(f"[O] BERT backbone and MLM head frozen for first {self.freeze_bert_epochs} epochs")
            self._bert_is_trainable = trainable
        self._set_bert_trainable(trainable)

    @staticmethod
    def _pad_vocab_dim(tensor: torch.Tensor, target_vocab_size: int, fill_value: float | bool) -> torch.Tensor:
        current_vocab_size = tensor.size(-1)
        if current_vocab_size == target_vocab_size:
            return tensor
        if current_vocab_size > target_vocab_size:
            return tensor[..., :target_vocab_size]
        pad_shape = (*tensor.shape[:-1], target_vocab_size - current_vocab_size)
        pad_tensor = tensor.new_full(pad_shape, fill_value)
        return torch.cat([tensor, pad_tensor], dim=-1)

    def _apply_student_hard_mask(
        self,
        core_ids: torch.Tensor,
        output_logits: torch.Tensor,
        token_mask: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Apply character constraints to student logits and validate target reachability."""
        batch_size, seq_len, vocab_size = output_logits.size()
        if token_mask.shape != (batch_size, seq_len):
            raise ValueError("token_mask must match output logits batch/seq dimensions")
        if target_ids.shape != (batch_size, seq_len):
            raise ValueError("target_ids must match output logits batch/seq dimensions")

        raw_char_mask = self.sarkaz_mapping.map_core_ids(core_ids.to(output_logits.device))
        char_mask = self._pad_vocab_dim(raw_char_mask, vocab_size, False)
        token_mask_bool = token_mask.to(output_logits.device).bool()
        targets = target_ids.to(output_logits.device).long()

        safe_targets = targets.clamp(min=0, max=vocab_size - 1)
        target_in_vocab = targets < vocab_size
        target_allowed = char_mask.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1) & target_in_vocab
        target_allowed = target_allowed | ~token_mask_bool
        if not target_allowed.all():
            bad_positions = (~target_allowed).nonzero(as_tuple=False)
            first_bad = bad_positions[0].tolist()
            bad_target_id = int(targets[first_bad[0], first_bad[1]].item())
            raise RuntimeError(
                "Target token is masked out by char_mask at position "
                f"(batch={first_bad[0]}, seq={first_bad[1]}), target_id={bad_target_id}"
            )

        return output_logits.masked_fill(~char_mask, self.hard_mask)

    def _calc_loss(
        self,
        core_ids: torch.Tensor,
        output_logits: torch.Tensor,
        token_mask: torch.Tensor,
        target_ids: torch.Tensor,
        teacher_topk_ids: Optional[torch.Tensor] = None,
        teacher_topk_probs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute either one-hot KL loss or sparse cached-teacher KL loss."""
        batch_size, seq_len, vocab_size = output_logits.size()
        if token_mask.size() != (batch_size, seq_len):
            raise ValueError("Output logits and token_mask must have matching batch and sequence dimensions")
        if target_ids.size() != (batch_size, seq_len):
            raise ValueError("Output logits and target_ids must have matching batch and sequence dimensions")

        masked_student_logits = self._apply_student_hard_mask(core_ids, output_logits, token_mask, target_ids)
        logits_flat, targets_flat, mask_flat = self._flatten_core_logits(
            masked_student_logits,
            target_ids,
            token_mask,
            vocab_size,
            ensure_device_str=self.device_str,
        )
        if mask_flat.sum().item() == 0:
            raise RuntimeError("No valid positions to compute loss; token_mask is all zero")

        supervised_logits = logits_flat[mask_flat]

        if self.use_teacher_knowledge:
            if teacher_topk_ids is None or teacher_topk_probs is None:
                raise RuntimeError("teacher knowledge is enabled, but batch does not contain teacher_topk_ids/probs")
            if teacher_topk_ids.shape[:2] != (batch_size, seq_len) or teacher_topk_probs.shape[:2] != (batch_size, seq_len):
                raise ValueError(
                    "teacher_topk_ids/probs must have shape (batch, seq_len, topk) aligned with student output"
                )
            if teacher_topk_ids.shape != teacher_topk_probs.shape:
                raise ValueError("teacher_topk_ids and teacher_topk_probs must have identical shapes")

            topk = int(teacher_topk_ids.size(-1))
            teacher_ids_flat = teacher_topk_ids.to(self.device_str).view(-1, topk)[mask_flat].long()
            teacher_probs_flat = teacher_topk_probs.to(self.device_str).view(-1, topk)[mask_flat].to(supervised_logits.dtype)
            if teacher_ids_flat.numel() == 0:
                raise RuntimeError("teacher knowledge has no supervised entries after token_mask selection")
            if teacher_ids_flat.min().item() < 0 or teacher_ids_flat.max().item() >= vocab_size:
                raise RuntimeError(
                    f"teacher top-k ids must be in [0, {vocab_size}); found min={teacher_ids_flat.min().item()}, max={teacher_ids_flat.max().item()}"
                )

            prob_sums = teacher_probs_flat.sum(dim=-1, keepdim=True)
            if (prob_sums <= 0).any():
                bad = int((prob_sums.squeeze(-1) <= 0).nonzero(as_tuple=False)[0].item())
                raise RuntimeError(f"teacher top-k probabilities sum to zero at supervised row {bad}")
            teacher_probs_flat = teacher_probs_flat / prob_sums.clamp_min(1e-8)

            temperature = self.teacher_temperature
            student_log_probs = F.log_softmax(supervised_logits / temperature, dim=-1)
            selected_student_log_probs = student_log_probs.gather(1, teacher_ids_flat)
            return F.kl_div(
                selected_student_log_probs,
                teacher_probs_flat,
                reduction="batchmean",
                log_target=False,
            ) * (temperature * temperature)

        supervised_targets = targets_flat[mask_flat]
        target_class_logits = supervised_logits.gather(1, supervised_targets.unsqueeze(1)).squeeze(1)
        bad_positions = (target_class_logits == self.hard_mask).nonzero(as_tuple=False).flatten()
        if bad_positions.numel() > 0:
            first_bad = int(bad_positions[0].item())
            flat_indices = mask_flat.nonzero(as_tuple=False).squeeze(1)
            flat_idx = int(flat_indices[first_bad].item())
            batch_idx = flat_idx // token_mask.size(1)
            seq_idx = flat_idx % token_mask.size(1)
            bad_target_id = int(supervised_targets[first_bad].item())
            raise RuntimeError(
                "Target-class logit is hard-masked before criterion at position "
                f"(batch={batch_idx}, seq={seq_idx}), target_id={bad_target_id}, hard_mask={self.hard_mask}"
            )

        log_probs = F.log_softmax(supervised_logits, dim=-1)
        target_probs = F.one_hot(supervised_targets, num_classes=vocab_size).to(dtype=log_probs.dtype)
        return self.criterion(log_probs, target_probs)

    def token_accuracy(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, top_k: int = 1) -> float:
        if mask.sum().item() == 0:
            return 0.0
        k = min(top_k, logits.size(-1))
        _, top_k_predictions = torch.topk(logits, k=k, dim=-1)
        correct = (top_k_predictions == targets.unsqueeze(-1)).any(dim=-1)
        correct_count = correct[mask].sum().item()
        total = mask.sum().item()
        return correct_count / total if total > 0 else 0.0

    def _flatten_core_logits(
        self,
        output_logits: torch.Tensor,
        target_ids: torch.Tensor,
        token_mask: torch.Tensor,
        vocab_size: int,
        ensure_device_str: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits_flat = output_logits.to(ensure_device_str).contiguous().view(-1, vocab_size)
        targets_flat = target_ids.to(ensure_device_str).view(-1).long()
        mask_flat = token_mask.to(ensure_device_str).view(-1).bool()
        return logits_flat, targets_flat, mask_flat

    def _batch_accuracy_metrics(
        self,
        core_ids: torch.Tensor,
        output_logits: torch.Tensor,
        target_ids: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> Tuple[float, float]:
        with torch.inference_mode():
            metric_logits = self._apply_student_hard_mask(core_ids, output_logits, token_mask, target_ids)
            logits_flat, targets_flat, mask_flat = self._flatten_core_logits(
                metric_logits,
                target_ids,
                token_mask,
                metric_logits.size(-1),
                ensure_device_str=self.device_str,
            )
            return (
                self.token_accuracy(logits_flat, targets_flat, mask_flat, top_k=1),
                self.token_accuracy(logits_flat, targets_flat, mask_flat, top_k=5),
            )

    def _update_validation_state(self, val_loss: float, val_acc: float, checkpoint_epoch: int) -> bool:
        self.history_stats["val_loss"].append(val_loss)
        self.history_stats["val_metric"].append(val_acc)

        is_best = val_acc > self.best_score
        if is_best:
            self.best_score = val_acc
            self.early_stop_counter = 0
            print(f"  Best score updated: {self.best_score:.4f}")
        else:
            self.early_stop_counter += 1
            print(f"  Early stop counter: {self.early_stop_counter}/{self.early_stop_patience}")

        self.model_saver.save(
            self.model,
            self.optimizer,
            self.scheduler,
            checkpoint_epoch,
            self.best_score,
            current_train_level=self.current_train_level,
            grad_scaler=self.grad_scaler if self.enable_amp else None,
            is_best=is_best,
        )

        self._should_early_stop = self.early_stop_counter >= self.early_stop_patience
        return self._should_early_stop

    def _move_batch(self, batch: dict) -> dict:
        moved = {
            "head_ids": batch["head_ids"].to(self.device_str, non_blocking=True),
            "core_ids": batch["core_ids"].to(self.device_str, non_blocking=True),
            "attention_mask": batch["attention_mask"].to(self.device_str, non_blocking=True),
            "token_type_ids": batch["token_type_ids"].to(self.device_str, non_blocking=True),
            "token_mask": batch["token_mask"].to(self.device_str, non_blocking=True),
            "target_ids": batch["target_ids"].to(self.device_str, non_blocking=True),
        }
        if "teacher_topk_ids" in batch:
            moved["teacher_topk_ids"] = batch["teacher_topk_ids"].to(self.device_str, non_blocking=True)
            moved["teacher_topk_probs"] = batch["teacher_topk_probs"].to(self.device_str, non_blocking=True)
        return moved

    def train_epoch(self) -> Tuple[float, float, float]:
        self.model.train()
        total_loss = 0.0
        total_accuracy_top1 = 0.0
        total_accuracy_top5 = 0.0
        num_batches = 0

        self.optimizer.zero_grad(set_to_none=True)
        for batch_idx, raw_batch in enumerate(self.train_loader):
            batch = self._move_batch(raw_batch)

            with autocast(device_type=self.amp_device_type, enabled=self.enable_amp):
                output_logits = self.model(
                    batch["head_ids"],
                    batch["core_ids"],
                    batch["attention_mask"],
                    batch["token_type_ids"],
                )
                loss = self._calc_loss(
                    batch["core_ids"],
                    output_logits,
                    batch["token_mask"],
                    batch["target_ids"],
                    teacher_topk_ids=batch.get("teacher_topk_ids"),
                    teacher_topk_probs=batch.get("teacher_topk_probs"),
                )

            scaled_loss = loss / self.accumulation_steps
            if self.enable_amp:
                self.grad_scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            should_step = ((batch_idx + 1) % self.accumulation_steps == 0) or (batch_idx + 1 == len(self.train_loader))
            if should_step:
                if self.enable_amp:
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()
                else:
                    self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1

            total_loss += float(loss.item())
            num_batches += 1
            batch_top1, batch_top5 = self._batch_accuracy_metrics(
                batch["core_ids"], output_logits.detach(), batch["target_ids"], batch["token_mask"]
            )
            total_accuracy_top1 += batch_top1
            total_accuracy_top5 += batch_top5

            if self.log_interval > 0 and (batch_idx + 1) % self.log_interval == 0:
                print(f"  [Batch {batch_idx + 1}/{len(self.train_loader)}]")
                print(f"    Current Loss={loss.item():.4f}, Top1={batch_top1:.4f}, Top5={batch_top5:.4f}")
                if self.summary_logger:
                    self.summary_logger.log_train_step(self.global_step, loss.item(), batch_top1, batch_top5)
                    lrs = [pg.get("lr", None) for pg in self.optimizer.param_groups]
                    self.summary_logger.log_learning_rate(self.global_step, lrs)

            if should_step and self.validate_interval > 0 and self.global_step % self.validate_interval == 0:
                val_loss, val_top1, val_top5 = self.validate()
                print(
                    f"  [Step Validation] GlobalStep {self.global_step} in Epoch {self.current_epoch}: "
                    f"Val Loss={val_loss:.4f}, Val Top1={val_top1:.4f}, Val Top5={val_top5:.4f}"
                )
                if self._update_validation_state(val_loss, val_top5, self.current_epoch):
                    print("\nEarly stopping triggered")
                    self.model.train()
                    break
                self.model.train()

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_top1 = total_accuracy_top1 / num_batches if num_batches > 0 else 0.0
        avg_top5 = total_accuracy_top5 / num_batches if num_batches > 0 else 0.0
        return avg_loss, avg_top1, avg_top5

    def validate(self) -> Tuple[float, float, float]:
        self.model.eval()
        total_loss = 0.0
        total_accuracy_top1 = 0.0
        total_accuracy_top5 = 0.0
        num_batches = 0

        with torch.inference_mode():
            for raw_batch in self.val_loader:
                batch = self._move_batch(raw_batch)
                with autocast(device_type=self.amp_device_type, enabled=self.enable_amp):
                    output_logits = self.model(
                        batch["head_ids"],
                        batch["core_ids"],
                        batch["attention_mask"],
                        batch["token_type_ids"],
                    )
                    loss = self._calc_loss(
                        batch["core_ids"],
                        output_logits,
                        batch["token_mask"],
                        batch["target_ids"],
                        teacher_topk_ids=batch.get("teacher_topk_ids"),
                        teacher_topk_probs=batch.get("teacher_topk_probs"),
                    )

                batch_top1, batch_top5 = self._batch_accuracy_metrics(
                    batch["core_ids"], output_logits, batch["target_ids"], batch["token_mask"]
                )
                total_accuracy_top1 += batch_top1
                total_accuracy_top5 += batch_top5
                total_loss += float(loss.item())
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_top1 = total_accuracy_top1 / num_batches if num_batches > 0 else 0.0
        avg_top5 = total_accuracy_top5 / num_batches if num_batches > 0 else 0.0
        return avg_loss, avg_top1, avg_top5

    def train(self, target_epochs: int) -> None:
        start_epoch, prev_best_score, loaded = self.model_saver.load(
            self.model,
            self.optimizer,
            self.scheduler,
            current_train_level=self.current_train_level,
            grad_scaler=self.grad_scaler if self.enable_amp else None,
            checkpoint_name="last",
        )

        self.model = self.model.to(self.device_str)
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(self.device_str)

        if self.enable_gradient_checkpointing and str(self.device_str).startswith("cuda"):
            try:
                self.model.bert_model.gradient_checkpointing_enable()
                print("[O] Gradient checkpointing enabled")
            except Exception as exc:
                print(f"[!] Warning: Could not enable gradient checkpointing: {exc}")

        if loaded:
            self.current_epoch = start_epoch
            self.best_score = prev_best_score
            print(f"[O] Loaded checkpoint from level_epoch {start_epoch}, best_score={prev_best_score:.4f}")
            print(f"    Current training level: {self.current_train_level}")
        else:
            self.current_epoch = 0
            self.best_score = float("-inf")
            print("[X] No checkpoint found, starting from scratch")

        print(f"\nStarting training from epoch {self.current_epoch} (target {target_epochs} epochs)")

        for epoch in range(self.current_epoch, target_epochs):
            self._apply_bert_freeze_for_epoch(epoch)
            self.current_epoch = epoch
            print(f"\n[Epoch {epoch}/{target_epochs}]")

            train_loss, train_top1, train_top5 = self.train_epoch()
            self.history_stats["train_loss"].append(train_loss)
            if self._should_early_stop:
                break

            val_loss, val_top1, val_top5 = self.validate()
            print(
                f"  Train Loss={train_loss:.4f}, Train Top1={train_top1:.4f}, Train Top5={train_top5:.4f}\n"
                f"  Val Loss={val_loss:.4f}, Val Top1={val_top1:.4f}, Val Top5={val_top5:.4f}"
            )

            if self.summary_logger:
                self.summary_logger.log_train_epoch(epoch, train_loss, train_top1, train_top5)
                self.summary_logger.log_validation(epoch, val_loss, val_top1, val_top5, is_best=val_top5 > self.best_score)

            if self._update_validation_state(val_loss, val_top5, epoch):
                print(f"\nEarly stopping at epoch {epoch}")
                break

        if self.summary_logger:
            self.summary_logger.close()

        print(f"\nTraining finished. Best validation accuracy: {self.best_score:.4f}")
