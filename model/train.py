"""Full training script for SarkazBert.

Supports resuming from checkpoints, configurable hyperparameters,
and an optional `--max-samples` for quick smoke runs.
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
from torch.optim import AdamW

from transformers import BertForMaskedLM, get_linear_schedule_with_warmup

from tokenizer import SarkazTokenizer
from dataset import SarkazDataset
from sarkazBert import SarkazBert
from trainer import SarkazBertTrainer
from modelSaver import ModelSaver
from summaryLogger import SummaryLogger


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-data", type=str, default="model/data/pretrain.jsonl")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--validate-share", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64) # check before run prod=64
    p.add_argument("--embedding-lr", type=float, default=8e-5, help="Learning rate for custom embeddings")
    p.add_argument("--bert-emb-lr", type=float, default=5e-5, help="Learning rate for BERT embeddings")
    p.add_argument("--bert-low-lr", type=float, default=4e-5, help="Learning rate for BERT encoder layers 1-4")
    p.add_argument("--bert-mid-lr", type=float, default=8e-6, help="Learning rate for BERT encoder layers 5-8")
    p.add_argument("--bert-high-lr", type=float, default=2e-5, help="Learning rate for BERT encoder layers 9-12")
    p.add_argument("--mlm-head-lr", type=float, default=2e-5, help="Learning rate for the pretrained MLM head")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--log-interval", type=int, default=100)
    p.add_argument("--validate-interval", type=int, default=2000, help="Validate every N steps (set to 0 to disable step-based validation)")
    p.add_argument("--enable-amp", type=bool, default=True, help="Enable automatic mixed precision for faster training and reduced memory usage")
    p.add_argument("--enable-gradient-checkpointing", type=bool, default=False, help="Enable gradient checkpointing to reduce GPU memory")
    p.add_argument("--freeze-bert-epochs", type=int, default=2, help="Freeze the BERT backbone for the first N epochs")
    p.add_argument("--train-level", type=int, default=0, help="Current training level for multi-level training (default 0)")
    p.add_argument("--teacher-model-path", type=str, default="", help="Optional teacher model directory for supervised distillation")
    p.add_argument("--teacher-temperature", type=float, default=1.5, help="Teacher distillation temperature; enable supervision when > 0")
    p.add_argument("--max-samples", type=int, default=0 , help="Set to >0 for quick tests") # check before run prod=0
    p.add_argument("--loader-workers", type=int, default=4) # check before run prod=4
    p.add_argument("--accumulation-steps", type=int, default=1, help="Number of steps to accumulate gradients for (default 4)")
    return p.parse_args()


def load_teacher_model(teacher_model_path: str, device: str):
    if not teacher_model_path or teacher_model_path.strip() == "":
        return None

    teacher_module_path = Path(__file__).resolve().parent / "teacher-model.py"
    spec = importlib.util.spec_from_file_location("teacher_model_module", teacher_module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load teacher module from {teacher_module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    teacher_model_dir = Path(teacher_model_path)
    if not teacher_model_dir.is_absolute():
        teacher_model_dir = Path(__file__).resolve().parent.parent / teacher_model_dir
    return module.TeacherModel(model_dir=teacher_model_dir, device=device)


def main():

    # 加载训练配置
    args = get_args()
    project_root = Path(__file__).resolve().parent.parent

    # 选择训练设备
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 定位训练数据
    data_path = Path(args.train_data)
    if not data_path.is_absolute():
        data_path = project_root / data_path
    data_path = data_path.resolve()
    if not data_path.exists():
        print(f"Data not found: {data_path}")
        return

    # 定位检查点数据
    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = project_root / checkpoint_dir
    checkpoint_dir = checkpoint_dir.resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 创建 tokenizer 和 dataset 实例
    print("Creating tokenizer and dataset...")
    tokenizer = SarkazTokenizer()
    dataset = SarkazDataset(raw_data=str(data_path), tokenizer=tokenizer, max_length=256)

    # 如果设置了 max_samples，则使用数据集的子集（适用于快速测试）
    if args.max_samples and args.max_samples > 0:
        n = min(len(dataset), args.max_samples)
        dataset = Subset(dataset, range(n))

    # 将数据集划分为训练集和验证集
    total = len(dataset)
    val_len = max(1, int(total * args.validate_share))
    train_len = total - val_len
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

    # 创建数据加载器
    train_loader = DataLoader(train_dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            collate_fn=tokenizer.collate,
                            num_workers=args.loader_workers,
                            pin_memory=(device != "cpu"),
                            persistent_workers=True,
                            drop_last=True)  # drop_last=True 以确保每个 batch 都有 accumulation_steps 的样本数
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            collate_fn=tokenizer.collate,
                            num_workers=args.loader_workers,
                            pin_memory=(device != "cpu"),
                            persistent_workers=True,
                            drop_last=True)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    teacher_model = None
    use_teacher_supervision = bool(args.teacher_model_path and args.teacher_temperature > 0)
    if args.teacher_model_path and args.teacher_temperature <= 0:
        raise ValueError("teacher-temperature must be greater than 0 when teacher-model-path is provided")

    if use_teacher_supervision:
        teacher_model = load_teacher_model(args.teacher_model_path, device)
        print(f"Teacher supervision enabled: path={args.teacher_model_path}, temperature={args.teacher_temperature:g}")

    # 加载模型实例
    print("Creating model...")
    bert_model_dir = Path(__file__).resolve().parent / "bert-base-chinese"
    mlm_model = BertForMaskedLM.from_pretrained(str(bert_model_dir))
    model = SarkazBert(mlm_model)

    # 学习率分组：custom embedding / BERT embeddings / BERT low-mid-high / MLM head
    bert_layers = model.bert_model.encoder.layer

    def layer_params(start: int, end: int):
        params = []
        for layer in bert_layers[start:end]:
            params.extend(layer.parameters())
        return params

    def unique_params(module, seen_param_ids: set[int]):
        params = []
        for param in module.parameters():
            if id(param) in seen_param_ids:
                continue
            seen_param_ids.add(id(param))
            params.append(param)
        return params

    seen_param_ids: set[int] = set()
    embedding_params = unique_params(model.embedding, seen_param_ids)
    bert_embedding_params = unique_params(model.bert_model.embeddings, seen_param_ids)
    bert_low_params = unique_params(torch.nn.ModuleList(bert_layers[:4]), seen_param_ids)
    bert_mid_params = unique_params(torch.nn.ModuleList(bert_layers[4:8]), seen_param_ids)
    bert_high_params = unique_params(torch.nn.ModuleList(bert_layers[8:12]), seen_param_ids)
    mlm_head_params = unique_params(model.mlm_head, seen_param_ids)

    optimizer = AdamW(
        [
            {"params": embedding_params, "lr": args.embedding_lr},
            {"params": bert_embedding_params, "lr": args.bert_emb_lr},
            {"params": bert_low_params, "lr": args.bert_low_lr},
            {"params": bert_mid_params, "lr": args.bert_mid_lr},
            {"params": bert_high_params, "lr": args.bert_high_lr},
            {"params": mlm_head_params, "lr": args.mlm_head_lr},
        ]
    )
    print(
        "Optimizer lr groups: "
        f"embedding_lr={args.embedding_lr:.6g}, "
        f"bert_emb_lr={args.bert_emb_lr:.6g}, "
        f"bert_low_lr={args.bert_low_lr:.6g}, "
        f"bert_mid_lr={args.bert_mid_lr:.6g}, "
        f"bert_high_lr={args.bert_high_lr:.6g}, "
        f"mlm_head_lr={args.mlm_head_lr:.6g}"
    )
    model_saver = ModelSaver(checkpoint_dir=str(checkpoint_dir))

    # 计算总训练步骤数并配置学习率调度器
    target_steps = max(1, len(train_loader) * max(1, args.epochs))
    print(f"Target total training steps: {target_steps}")
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                # 设置预热 避免初期随机初始化的层破坏 bert 的预训练权重
                                                num_warmup_steps = max(500, target_steps // 15),
                                                num_training_steps = target_steps)

    # 尝试加载最后一次训练检查点
    last_ckpt = checkpoint_dir / "last.pt"
    if last_ckpt.exists():
        try:
            ck = torch.load(str(last_ckpt), map_location=device)
            # Only require model weights for metadata inspection; optimizer/
            # scheduler may be intentionally absent after a level transition.
            if 'model_state_dict' not in ck:
                print("Checkpoint corrupted: missing model_state_dict. Need Action")
                return

            saved_train_level = ck.get('train_level', 0)
            level_epoch = ck.get('level_epoch', ck.get('epoch', 0))  # Fallback to 'epoch' for old checkpoints

            print(f"Checkpoint loaded: train_level={saved_train_level}, level_epoch={level_epoch}, current_train_level={args.train_level}")

            if 'optimizer_state_dict' not in ck or 'scheduler_state_dict' not in ck:
                print("  Note: checkpoint missing optimizer/scheduler state — will start fresh optimizer/scheduler for current run")

            # Check if level has changed
            if saved_train_level != args.train_level:
                print(f"[Level change detected] {saved_train_level} -> {args.train_level}")
                print(f"  Previous level epoch {level_epoch} will be archived as L{saved_train_level}_epoch")

            if level_epoch >= args.epochs:
                print(f"Checkpoint indicates {level_epoch} completed epochs, which >= requested total {args.epochs}. Nothing to do.")
                return
        except Exception as e:
            print(f"Warning: failed to read existing checkpoint metadata: {e}")
            return
    # 加载检查点出现任何问题都要返回，避免在不确定状态下继续训练

    # 创建日志目录和 TensorBoard 实例
    log_dir = project_root / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_board = SummaryLogger(str(log_dir))

    # 创建训练器实例
    trainer = SarkazBertTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        model_saver=model_saver,
        criterion = nn.KLDivLoss(reduction="batchmean"),
        device_str=device,
        enable_amp=args.enable_amp,
        enable_gradient_checkpointing=args.enable_gradient_checkpointing,
        validate_interval=args.validate_interval,
        log_interval=args.log_interval,
        current_train_level=args.train_level,
        freeze_bert_epochs=args.freeze_bert_epochs,
        teacher_model=teacher_model,
        teacher_temperature=args.teacher_temperature,
        summary_logger=summary_board,
        accumulation_steps=args.accumulation_steps
    )

    # 启动训练
    trainer.train(target_epochs=args.epochs)

    # Print summary
    if trainer.history_stats['train_loss']:
        print("Training finished.")
        print(f"Final train loss: {trainer.history_stats['train_loss'][-1]:.4f}")
        print(f"Final val loss: {trainer.history_stats['val_loss'][-1]:.4f}")
        print(f"Final val accuracy: {trainer.history_stats['val_metric'][-1]:.4f}")
        print(f"Best accuracy: {trainer.best_score:.4f}")
    else:
        print("Training finished but no history recorded. (Maybe no epochs were run)")


if __name__ == "__main__":
    main()