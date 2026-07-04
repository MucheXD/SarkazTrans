from typing import Tuple, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp.autocast_mode import autocast
from sarkazBert import SarkazBert
from modelSaver import ModelSaver
from summaryLogger import SummaryLogger
from dataset import SarkazCharmap
from torch.utils.data import DataLoader

class SarkazBertTrainer:
    """训练器"""
    def __init__(
        self,
        model: SarkazBert,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LambdaLR,
        model_saver: ModelSaver,
        criterion: nn.Module,
        device_str: str = 'cpu',
        enable_amp: bool = True,
        enable_gradient_checkpointing: bool = True,
        early_stop_patience: int = 10,
        accumulation_steps: int = 4,
        validate_interval: int = 100,
        log_interval: int = 10,
        current_train_level: int = 0,
        freeze_bert_epochs: int = 0,
        hard_mask: float = -1e4,
        sarkaz_mapping: SarkazCharmap = SarkazCharmap(),
        teacher_model: Optional[Any] = None,
        teacher_temperature: float = 1.0,
        summary_logger: Optional[SummaryLogger] = None,
    ):
        # 延迟将模型搬到 GPU，在 train() 开始时执行，避免与加载检查点重复占用显存
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device_str = device_str
        self.model_saver = model_saver
        self.enable_amp = enable_amp
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.early_stop_patience = early_stop_patience
        self.accumulation_steps = accumulation_steps
        self.validate_interval = validate_interval
        self.log_interval = log_interval
        self.current_train_level = current_train_level
        self.freeze_bert_epochs = max(0, freeze_bert_epochs)
        self.hard_mask = hard_mask
        self.sarkaz_mapping = sarkaz_mapping
        self.teacher_model = teacher_model
        self.teacher_temperature = teacher_temperature
        self.use_teacher_supervision = self.teacher_model is not None and self.teacher_temperature > 0
        self.summary_logger = summary_logger
        
        # 训练状态跟踪
        self.current_epoch = 0
        self.best_score = float('-inf') # 初始最佳分数
        self.history_stats = {'train_loss': [], 'val_loss': [], 'val_metric': []}
        
        # 早停设置
        self.early_stop_counter = 0
        # 全局步数计数（用于按 step 做验证）
        self.global_step = 0
        # 当在训练中触发早停时 标记以便上层循环结束 在本类中无使用
        self._should_early_stop = False
        # 跟踪 BERT 主模型当前是否处于冻结状态，避免重复打印日志。
        self._bert_is_trainable: Optional[bool] = None
        
        # 使用混合精度训练
        if self.enable_amp:
            self.grad_scaler = torch.amp.grad_scaler.GradScaler()

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
                    print(f"[O] BERT backbone and MLM head unfrozen at epoch {epoch} (freeze first {self.freeze_bert_epochs} epochs)")
                else:
                    print(f"[O] BERT backbone and MLM head frozen for first {self.freeze_bert_epochs} epochs")
            self._bert_is_trainable = trainable
        self._set_bert_trainable(trainable)

    def _calc_loss(
        self,
        core_ids: torch.Tensor,
        output_logics: torch.Tensor,
        token_mask: torch.Tensor,
        target_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算掩码后的 KL 散度损失。

        Args:
            head_ids: Tensor of shape (batch, seq_len) — 头部 token IDs。
            core_ids: Tensor of shape (batch, seq_len) — 核心 token IDs。
            output_logics: Tensor of shape (batch, seq_len, dict_size) — 模型预测的 logits（已由 MLM head 切片到 dict_size）。
            token_mask: Tensor of shape (batch, seq_len) — 0/1 掩码，1 表示该位置参与损失计算。
            target_ids: Tensor of shape (batch, seq_len) — 目标 id，后续会展开成 one-hot 分布。

        Returns:
            标量损失张量（可反向传播）。
        """
        # output_logics: (batch, seq_len, vocab)
        # token_mask has shape (batch, seq_len)
        # 获取张量大小 进行断言以便及早发现问题
        batch_size, seq_len, vocab_size = output_logics.size()
        mask_batch_size, mask_seq_len = token_mask.size()
        assert batch_size == mask_batch_size and seq_len == mask_seq_len, "Output logits and token mask must have matching batch and sequence dimensions"
        tar_batch_size, tar_seq_len = target_ids.size()
        assert batch_size == tar_batch_size and seq_len == tar_seq_len, "Output logits and target ids must have matching batch and sequence dimensions"

        def _pad_vocab_dim(tensor: torch.Tensor, target_vocab_size: int, fill_value: float | bool) -> torch.Tensor:
            current_vocab_size = tensor.size(-1)
            if current_vocab_size == target_vocab_size:
                return tensor
            if current_vocab_size > target_vocab_size:
                return tensor[..., :target_vocab_size]
            pad_shape = (*tensor.shape[:-1], target_vocab_size - current_vocab_size)
            pad_tensor = tensor.new_full(pad_shape, fill_value)
            return torch.cat([tensor, pad_tensor], dim=-1)

        # 获取硬规则掩码
        # 硬规则帮助模型专注学习有效输出的特定子集，忽略无效位置的预测
        raw_char_mask = self.sarkaz_mapping.map_core_ids(core_ids).to(self.device_str)
        student_vocab_size = vocab_size
        student_mask_vocab_size = max(student_vocab_size, raw_char_mask.size(-1))
        student_output_logics = _pad_vocab_dim(output_logics, student_mask_vocab_size, self.hard_mask)
        student_char_mask = _pad_vocab_dim(raw_char_mask, student_mask_vocab_size, False)
        # 目标 token 对应的 logits 位置必须保留，否则说明映射或数据存在错误
        token_mask_bool = token_mask.to(self.device_str).bool()
        target_allowed = student_char_mask.gather(-1, target_ids.long().unsqueeze(-1)).squeeze(-1)
        target_allowed = target_allowed | ~token_mask_bool
        if not target_allowed.all():
            bad_positions = (~target_allowed).nonzero(as_tuple=False)
            first_bad = bad_positions[0].tolist()
            bad_target_id = target_ids[first_bad[0], first_bad[1]].item()
            raise RuntimeError(
                "Target token is masked out by char_mask at position "
                f"(batch={first_bad[0]}, seq={first_bad[1]}), target_id={bad_target_id}"
            )
        # 应用硬规则掩码，将无效位置的 logits 设置为一个很小的值，使其在 softmax 后接近于 0
        student_output_logics = student_output_logics.masked_fill(~student_char_mask, self.hard_mask)

        # 展平后，仅对 token_mask 参与监督的位置构造 KLDiv 所需的输入/目标。
        logits, targets, mask = self._flatten_core_logits(
            student_output_logics[..., :student_vocab_size],
            target_ids,
            token_mask,
            student_vocab_size,
            ensure_device_str=self.device_str,
        )

        # === 教师监督学习模式 ===
        if self.use_teacher_supervision:
            teacher_attention_mask = core_ids.ne(0).long()
            with torch.inference_mode():
                # 教师模型输出形状: (batch_size, seq_len, 21128)
                teacher_logits = self.teacher_model.Answer(core_ids, teacher_attention_mask)
            teacher_logits = teacher_logits.to(self.device_str)
            
            # 1. 将教师模型的词表维度动态对齐到学生模型的词表维度 (例如 21128 -> 8100)
            teacher_logits_rescaled = _pad_vocab_dim(teacher_logits, student_vocab_size, self.hard_mask)
            
            # 2. 对教师模型同样应用硬规则掩码 (截取与当前学生词表一致的前缀掩码)
            student_char_mask_sliced = student_char_mask[..., :student_vocab_size]
            teacher_logits_rescaled = teacher_logits_rescaled.masked_fill(~student_char_mask_sliced, self.hard_mask)
            
            # 3. 将教师模型的 logits 展平为 2D 形状: (batch_size * seq_len, student_vocab_size)
            teacher_logits_flat = teacher_logits_rescaled.contiguous().view(-1, student_vocab_size)
            
            # 4. 核心修复：复用前面第140行已经安全展平的学生 logits 和 1D mask [5632] 进行切片
            supervised_student_logits = logits[mask]
            supervised_teacher_logits = teacher_logits_flat[mask]
            
            # 5. 计算知识蒸馏损失
            temperature = self.teacher_temperature
            student_log_probs = F.log_softmax(supervised_student_logits / temperature, dim=-1)
            teacher_log_probs = F.log_softmax(supervised_teacher_logits / temperature, dim=-1)
            return F.kl_div(student_log_probs, teacher_log_probs, reduction="batchmean", log_target=True) * (temperature * temperature)

        # === 仅使用 KLDivLoss 计算损失 ===
        # 在进入 criterion 前再检查一次：参与监督的位置上，目标类别的 logit 不能已经被 hard_mask 掉
        supervised_logits = logits[mask]
        supervised_targets = targets[mask]
        target_class_logits = supervised_logits.gather(1, supervised_targets.unsqueeze(1)).squeeze(1)
        bad_positions = (target_class_logits == self.hard_mask).nonzero(as_tuple=False).flatten()
        if bad_positions.numel() > 0:
            first_bad = bad_positions[0].item()
            flat_indices = mask.nonzero(as_tuple=False).squeeze(1)
            flat_idx = flat_indices[first_bad].item()
            batch_idx = flat_idx // token_mask.size(1)
            seq_idx = flat_idx % token_mask.size(1)
            bad_target_id = supervised_targets[first_bad].item()
            raise RuntimeError(
                "Target-class logit is hard-masked before criterion at position "
                f"(batch={batch_idx}, seq={seq_idx}), target_id={bad_target_id}, hard_mask={self.hard_mask}"
            )

        assert mask.sum() > 0, "No valid positions to compute loss (token_mask may be all zeros)"

        # KLDivLoss 需要输入 log-probabilities 和与之同形状的目标分布。
        log_probs = F.log_softmax(supervised_logits, dim=-1)
        target_probs = F.one_hot(supervised_targets, num_classes=vocab_size).to(dtype=log_probs.dtype)

        return self.criterion(log_probs, target_probs)

    def token_accuracy(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
        top_k: int = 1
    ) -> float:
        """Compute token-level top-k accuracy.
        
        Args:
            logits: Tensor of shape (batch*seq_len, vocab_size)
            targets: Tensor of shape (batch*seq_len,)
            mask: Tensor of shape (batch*seq_len,) with bool type
            top_k: K for top-k accuracy (default 1)
        
        Returns:
            Accuracy as float in [0, 1].
        """
        if mask.sum() == 0:
            return 0.0
        
        # 获取 top-k 预测的索引
        _, top_k_predictions = torch.topk(logits, k=top_k, dim=-1)
        
        # 检查 targets 是否在 top-k 预测中
        # top_k_predictions shape: (batch*seq_len, top_k)
        # targets shape: (batch*seq_len,)
        correct = (top_k_predictions == targets.unsqueeze(-1)).any(dim=-1)
        
        correct_count = correct[mask].sum().item()
        total = mask.sum().item()
        return correct_count / total if total > 0 else 0.0

    def _flatten_core_logits(self, output_logits: torch.Tensor, target_ids: torch.Tensor, token_mask: torch.Tensor, vocab_size: int, ensure_device_str: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """展平模型输出的 logits 返回 (logits_flat, targets_flat, mask_flat).
        """
        logits_flat = output_logits.to(ensure_device_str).contiguous().view(-1, vocab_size)
        targets_flat = target_ids.to(ensure_device_str).view(-1).long()
        mask_flat = token_mask.to(ensure_device_str).view(-1).bool()
        return logits_flat, targets_flat, mask_flat

    def _update_validation_state(
        self,
        val_loss: float,
        val_acc: float,
        checkpoint_epoch: int,
    ) -> bool:
        """更新验证状态、保存 checkpoint，并返回是否触发早停。"""
        self.history_stats['val_loss'].append(val_loss)
        self.history_stats['val_metric'].append(val_acc)

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
            is_best=is_best
        )

        self._should_early_stop = self.early_stop_counter >= self.early_stop_patience
        return self._should_early_stop

    def train_epoch(self) -> Tuple[float, float, float]:
        """Train for one epoch.
        
        Returns:
            Tuple of (avg_loss, avg_accuracy_top1, avg_accuracy_top5)
        """
        self.model.train() # 设置模型为训练模式
        total_loss = 0.0
        total_accuracy_top1 = 0.0
        total_accuracy_top5 = 0.0
        num_batches = 0  # 直接跟踪处理的实际 batch 数
        
        for batch_idx, batch in enumerate(self.train_loader): # 遍历训练数据加载器中的每个 batch
            # 在首个 batch 时检查 batch_size 是否可以被 accumulation_steps 整除
            if batch_idx == 0:
                batch_size = batch["head_ids"].size(0)
                assert batch_size % self.accumulation_steps == 0, \
                    f"batch_size ({batch_size}) must be divisible by accumulation_steps ({self.accumulation_steps})"
            
            # 仅在累积周期开始时重置梯度
            if batch_idx % self.accumulation_steps == 0:
                self.optimizer.zero_grad() # 重置梯度
            
            # 载入到设备
            head_ids = batch["head_ids"].to(self.device_str)
            core_ids = batch["core_ids"].to(self.device_str)
            attention_mask = batch["attention_mask"].to(self.device_str)
            token_type_ids = batch["token_type_ids"].to(self.device_str)
            token_mask = batch["token_mask"].to(self.device_str)
            target_ids = batch["target_ids"].to(self.device_str)
            
            # 数据形状验证：确保批次正确对齐
            batch_size = head_ids.size(0)
            assert batch_size == core_ids.size(0) == token_mask.size(0) == target_ids.size(0), \
                f"Batch size mismatch: head_ids {head_ids.shape} vs core_ids {core_ids.shape} vs token_mask {token_mask.shape} vs target_ids {target_ids.shape}"
            
            # 前向传播和损失计算
            # 注意: 前向传播的掩码是 attention_mask 因为特殊字符需要被模型注意到
            # 然而损失计算使用 token_mask 因为特殊字符是不体现在输出的
            if self.enable_amp:
                with autocast(device_type=self.device_str):
                    output_logits = self.model(head_ids, core_ids, attention_mask, token_type_ids)
                    loss = self._calc_loss(core_ids, output_logits, token_mask, target_ids, attention_mask)
            else:
                output_logits = self.model(head_ids, core_ids, attention_mask, token_type_ids)
                loss = self._calc_loss(core_ids, output_logits, token_mask, target_ids, attention_mask)

            # 对损失进行缩放以实现梯度累积
            scaled_loss = loss / self.accumulation_steps
            
            # 反向传播和优化
            if self.enable_amp:
                self.grad_scaler.scale(scaled_loss).backward()
                # 从这一步开始 数学上 batch 已经完成 后续使用 batch_idx 都需要增加 1
                # 仅在累积完成时执行 optimizer.step()
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()
            else:
                scaled_loss.backward()
                # 仅在累积完成时执行 optimizer.step()
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
            
            # 累积完成时更新学习率
            # 注意: 调度器 step 应该每个 batch 调用以配合 Batch 语义
            self.scheduler.step()

            # 增加计数（仅在累积完成时增加全局步数），并累加 loss/metric（按原始 batch 口径）
            if (batch_idx + 1) % self.accumulation_steps == 0:
                self.global_step += 1
            total_loss += loss.item()
            num_batches += 1
            
            # 计算 Batch top1 和 top5 准确率
            with torch.inference_mode():
                logits_flat, targets_flat, mask_flat = self._flatten_core_logits(output_logits, target_ids, token_mask, output_logits.size(-1), ensure_device_str=self.device_str)
                batch_top1 = self.token_accuracy(logits_flat, targets_flat, mask_flat, top_k=1)
                batch_top5 = self.token_accuracy(logits_flat, targets_flat, mask_flat, top_k=5)
                total_accuracy_top1 += batch_top1
                total_accuracy_top5 += batch_top5

            # 根据设定打印指标（同时记录到 TensorBoard，包括分层学习率）
            if self.log_interval > 0 and (batch_idx + 1) % self.log_interval == 0:
                print(f"  [Batch {batch_idx + 1}/{len(self.train_loader)}]")
                print(f"    Current Loss={loss.item():.4f}, Top1={batch_top1:.4f}, Top5={batch_top5:.4f}")
                if self.summary_logger: # 记录到 TensorBoard
                    self.summary_logger.log_train_step(self.global_step, loss.item(), batch_top1, batch_top5)
                    # 记录分层学习率：收集所有 param_group 的 lr 并按 group 写入
                    lrs = [pg.get('lr', None) for pg in self.optimizer.param_groups]
                    self.summary_logger.log_learning_rate(self.global_step, lrs)

            # 如果配置了 validate_steps，则每隔指定 step 做一次验证并保存 checkpoint
            if self.validate_interval > 0 and (batch_idx + 1) % self.validate_interval == 0:
                val_loss, val_top1, val_top5 = self.validate()
                print(f"  [Step Validation] Step{batch_idx + 1} in Epoch {self.current_epoch}: Val Loss={val_loss:.4f}, Val Top1={val_top1:.4f}, Val Top5={val_top5:.4f}")
                # 若触发早停，则退出训练循环（使用 top5 作为监控指标）
                if self._update_validation_state(val_loss, val_top5, self.current_epoch):
                    print(f"\nEarly stopping triggered")
                    self.model.train()
                    break
                self.model.train()
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_top1 = total_accuracy_top1 / num_batches if num_batches > 0 else 0.0
        avg_top5 = total_accuracy_top5 / num_batches if num_batches > 0 else 0.0
        return avg_loss, avg_top1, avg_top5

    def validate(self) -> Tuple[float, float, float]:
        """Validate on validation set.
        
        Returns:
            Tuple of (avg_loss, avg_accuracy_top1, avg_accuracy_top5)
        """
        
        self.model.eval() # 设置模型为评估模式
        total_loss = 0.0
        total_accuracy_top1 = 0.0
        total_accuracy_top5 = 0.0
        num_batches = 0
        
        with torch.inference_mode():
            for batch in self.val_loader:
                # Move batch to device
                head_ids = batch["head_ids"].to(self.device_str)
                core_ids = batch["core_ids"].to(self.device_str)
                attention_mask = batch["attention_mask"].to(self.device_str)
                token_type_ids = batch["token_type_ids"].to(self.device_str)
                token_mask = batch["token_mask"].to(self.device_str)
                target_ids = batch["target_ids"].to(self.device_str)
                
                # Forward pass
                if self.enable_amp:
                    with autocast(device_type=self.device_str):
                        output_logits = self.model(head_ids, core_ids, attention_mask, token_type_ids)
                        loss = self._calc_loss(core_ids, output_logits, token_mask, target_ids, attention_mask)
                else:
                    output_logits = self.model(head_ids, core_ids, attention_mask, token_type_ids)
                    loss = self._calc_loss(core_ids, output_logits, token_mask, target_ids, attention_mask)
                
                # Compute metrics (top1 and top5)
                logits_flat, targets_flat, mask_flat = self._flatten_core_logits(output_logits, target_ids, token_mask, output_logits.size(-1), ensure_device_str=self.device_str)
                batch_top1 = self.token_accuracy(logits_flat, targets_flat, mask_flat, top_k=1)
                batch_top5 = self.token_accuracy(logits_flat, targets_flat, mask_flat, top_k=5)
                total_accuracy_top1 += batch_top1
                total_accuracy_top5 += batch_top5
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_top1 = total_accuracy_top1 / num_batches if num_batches > 0 else 0.0
        avg_top5 = total_accuracy_top5 / num_batches if num_batches > 0 else 0.0

        return avg_loss, avg_top1, avg_top5

    def train(self, target_epochs: int) -> None:
        """Train the model for given number of epochs.
        
        Args:
            target_epochs: Number of epochs to train.
        """
        # 从检查点加载，处理多级训练逻辑
        start_epoch, prev_best_score, loaded = self.model_saver.load(
            self.model,
            self.optimizer,
            self.scheduler,
            current_train_level=self.current_train_level,
            grad_scaler=self.grad_scaler if self.enable_amp else None,
            checkpoint_name="last"
        )
        
        # 加载完检查点后再将模型搬到 GPU，避免重复占用显存
        self.model = self.model.to(self.device_str)
        
        # 将 optimizer 的状态（动量缓冲区等）也移动到相同设备
        # 这是必要的，因为 optimizer 在模型还在 CPU 时被初始化/加载
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device_str)
        
        # 根据配置启用梯度检查点以减少激活张量占用（可减少 20-30% 的 GPU 显存）
        # 代价是增加计算时间（重新计算激活而非存储）
        if self.enable_gradient_checkpointing and self.device_str.startswith("cuda"):
            try:
                self.model.bert_model.gradient_checkpointing_enable()
                print("[O] Gradient checkpointing enabled")
            except Exception as e:
                print(f"[!] Warning: Could not enable gradient checkpointing: {e}")
        
        if loaded:
            self.current_epoch = start_epoch
            self.best_score = prev_best_score
            print(f"[O] Loaded checkpoint from level_epoch {start_epoch}, best_score={prev_best_score:.4f}")
            print(f"    Current training level: {self.current_train_level}")
        else:
            self.current_epoch = 0
            self.best_score = float('-inf')
            print("[X] No checkpoint found, starting from scratch")
        
        print(f"\nStarting training from epoch {self.current_epoch} (target {target_epochs} epochs)")
        
        for epoch in range(self.current_epoch, target_epochs):

            self._apply_bert_freeze_for_epoch(epoch)

            self.current_epoch = epoch # 登记全局变量
            print(f"\n[Epoch {epoch}/{target_epochs}]")
            
            # 进行一个 Epoch 的训练（返回 loss, top1, top5）
            train_loss, train_top1, train_top5 = self.train_epoch()
            self.history_stats['train_loss'].append(train_loss)
            
            # 每个 epoch 结束后进行验证（返回 loss, top1, top5）
            val_loss, val_top1, val_top5 = self.validate()
            
            print(
                f"  Train Loss={train_loss:.4f}, Train Top1={train_top1:.4f}, Train Top5={train_top5:.4f}\n"
                f"  Val Loss={val_loss:.4f}, Val Top1={val_top1:.4f}, Val Top5={val_top5:.4f}"
            )
            
            # 记录到 TensorBoard（训练/验证统计） —— 学习率在 train_epoch 的 log_interval 中记录
            if self.summary_logger:
                self.summary_logger.log_train_epoch(epoch, train_loss, train_top1, train_top5)
                self.summary_logger.log_validation(epoch, val_loss, val_top1, val_top5, is_best=val_top5 > self.best_score)
            
            # 更新验证状态并检查是否触发早停（以 Val Top5 为主指标）
            if self._update_validation_state(val_loss, val_top5, epoch):
                print(f"\nEarly stopping at epoch {epoch}")
                break
        
        # 关闭 TensorBoard writer
        if self.summary_logger:
            self.summary_logger.close()
        
        print(f"\nTraining finished. Best validation accuracy: {self.best_score:.4f}")