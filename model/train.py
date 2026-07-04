"""Full training script for SarkazBert.

The teacher workflow is two-stage when --teacher-model-path is provided:
  1. Before student construction, run BERT teacher with per-token [MASK] passes and
     cache top-k hard-masked distributions on CPU/disk.
  2. Unload the teacher, then train the student against the cached sparse knowledge.
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import importlib.util
import math
import sys
from pathlib import Path
from typing import Any

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
from knowledge_cache import (
    TeacherKnowledgeConfig,
    TeacherKnowledgeDataset,
    build_or_load_teacher_knowledge,
    expected_knowledge_metadata,
)


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-data", type=str, default="data/pretrain.jsonl")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--validate-share", type=float, default=0.1)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--embedding-lr", type=float, default=8e-5, help="Learning rate for custom embeddings")
    p.add_argument("--bert-emb-lr", type=float, default=5e-5, help="Learning rate for BERT embeddings")
    p.add_argument("--bert-low-lr", type=float, default=4e-5, help="Learning rate for BERT encoder layers 1-4")
    p.add_argument("--bert-mid-lr", type=float, default=8e-6, help="Learning rate for BERT encoder layers 5-8")
    p.add_argument("--bert-high-lr", type=float, default=2e-5, help="Learning rate for BERT encoder layers 9-12")
    p.add_argument("--mlm-head-lr", type=float, default=2e-5, help="Learning rate for the pretrained MLM head")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--log-interval", type=int, default=100)
    p.add_argument("--validate-interval", type=int, default=2000, help="Validate every N optimizer steps; set 0 to disable")
    p.add_argument("--enable-amp", type=str2bool, default=True, help="Enable CUDA automatic mixed precision")
    p.add_argument("--enable-gradient-checkpointing", type=str2bool, default=False, help="Enable gradient checkpointing")
    p.add_argument("--freeze-bert-epochs", type=int, default=2, help="Freeze BERT backbone for the first N epochs")
    p.add_argument("--train-level", type=int, default=0, help="Current training level")
    p.add_argument("--max-samples", type=int, default=0, help="Set to 0 for full training data")
    p.add_argument("--loader-workers", type=int, default=4)
    p.add_argument("--accumulation-steps", type=int, default=1)

    # Teacher knowledge extraction/cache options.
    p.add_argument("--teacher-model-path", type=str, default="teacher", help="Teacher model directory; enables cached distillation")
    p.add_argument("--teacher-temperature", type=float, default=1.5)
    p.add_argument("--teacher-topk", type=int, default=8)
    p.add_argument("--teacher-student-vocab-size", type=int, default=8100)
    p.add_argument("--teacher-knowledge-cache", type=str, default="", help="Path to .pt cache. Default: checkpoint-dir/teacher_knowledge_*.pt")
    p.add_argument("--rebuild-teacher-knowledge", action="store_true")
    p.add_argument("--teacher-extract-batch-size", type=int, default=4)
    p.add_argument("--teacher-vector-chunk-size", type=int, default=64)
    p.add_argument("--teacher-cache-workers", type=int, default=0)
    return p.parse_args()


def resolve_project_path(path_value: str | Path, project_root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()

    # Prefer the original project-root convention, but also support running the
    # extracted package directly from its own directory.
    project_candidate = (project_root / path).resolve()
    if project_candidate.exists():
        return project_candidate

    script_dir = Path(__file__).resolve().parent
    script_candidate = (script_dir / path).resolve()
    if script_candidate.exists():
        return script_candidate

    # Compatibility with old defaults such as "model/data/pretrain.jsonl" when
    # the zip is extracted without an enclosing "model" directory.
    parts = path.parts
    if len(parts) > 1 and parts[0] == script_dir.name:
        stripped_candidate = (script_dir / Path(*parts[1:])).resolve()
        if stripped_candidate.exists():
            return stripped_candidate
    if len(parts) > 1 and parts[0] == "model":
        stripped_candidate = (script_dir / Path(*parts[1:])).resolve()
        if stripped_candidate.exists():
            return stripped_candidate

    return project_candidate


def load_teacher_model(teacher_model_path: str | Path, device: str):
    teacher_module_path = Path(__file__).resolve().parent / "teacher-model.py"
    spec = importlib.util.spec_from_file_location("teacher_model_module", teacher_module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load teacher module from {teacher_module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.TeacherModel(model_dir=teacher_model_path, device=device)


def default_teacher_cache_path(checkpoint_dir: Path, data_path: Path, teacher_topk: int, temperature: float, max_samples: int) -> Path:
    data_stem = data_path.stem.replace(" ", "_")
    sample_tag = f"max{max_samples}" if max_samples and max_samples > 0 else "full"
    temp_tag = f"t{temperature:g}".replace(".", "p")
    return checkpoint_dir / f"teacher_knowledge_{data_stem}_{sample_tag}_top{teacher_topk}_{temp_tag}.pt"


def build_dataloader(dataset, tokenizer, batch_size: int, shuffle: bool, num_workers: int, device: str, drop_last: bool):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=tokenizer.collate,
        num_workers=num_workers,
        pin_memory=(str(device).startswith("cuda")),
        persistent_workers=num_workers > 0,
        drop_last=drop_last,
    )


def main():
    args = get_args()
    project_root = Path(__file__).resolve().parent.parent

    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_path = resolve_project_path(args.train_data, project_root)
    if not data_path.exists():
        print(f"Data not found: {data_path}")
        return

    checkpoint_dir = resolve_project_path(args.checkpoint_dir, project_root)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("Creating tokenizer and dataset...")
    tokenizer = SarkazTokenizer()
    dataset = SarkazDataset(raw_data=str(data_path), tokenizer=tokenizer, max_length=256)

    if args.max_samples and args.max_samples > 0:
        n = min(len(dataset), args.max_samples)
        dataset = Subset(dataset, range(n))
        print(f"Using max_samples subset: {n}")

    use_teacher_supervision = bool(args.teacher_model_path and args.teacher_temperature > 0)
    if args.teacher_model_path and args.teacher_temperature <= 0:
        raise ValueError("teacher-temperature must be greater than 0 when teacher-model-path is provided")

    if use_teacher_supervision:
        teacher_model_dir = resolve_project_path(args.teacher_model_path, project_root)
        cache_path = (
            resolve_project_path(args.teacher_knowledge_cache, project_root)
            if args.teacher_knowledge_cache
            else default_teacher_cache_path(
                checkpoint_dir,
                data_path,
                args.teacher_topk,
                args.teacher_temperature,
                args.max_samples,
            )
        )
        expected_metadata = expected_knowledge_metadata(
            dataset=dataset,
            train_data_path=data_path,
            teacher_model_path=teacher_model_dir,
            topk=args.teacher_topk,
            student_vocab_size=args.teacher_student_vocab_size,
            temperature=args.teacher_temperature,
        )
        knowledge_config = TeacherKnowledgeConfig(
            cache_path=cache_path,
            topk=args.teacher_topk,
            student_vocab_size=args.teacher_student_vocab_size,
            temperature=args.teacher_temperature,
            extract_batch_size=args.teacher_extract_batch_size,
            vector_chunk_size=args.teacher_vector_chunk_size,
            num_workers=args.teacher_cache_workers,
            pin_memory=False,
        )

        def teacher_factory():
            print(f"Loading teacher model for knowledge extraction: {teacher_model_dir}")
            return load_teacher_model(teacher_model_dir, device)

        knowledge_store = build_or_load_teacher_knowledge(
            dataset=dataset,
            tokenizer=tokenizer,
            config=knowledge_config,
            expected_metadata=expected_metadata,
            teacher_model_factory=teacher_factory,
            force_rebuild=args.rebuild_teacher_knowledge,
        )
        dataset = TeacherKnowledgeDataset(dataset, knowledge_store)
        print("Teacher model is no longer resident; student training will use cached top-k knowledge.")

    total = len(dataset)
    if total < 2:
        raise ValueError("dataset must contain at least two samples for train/validation split")
    val_len = max(1, int(total * args.validate_share))
    val_len = min(val_len, total - 1)
    train_len = total - val_len
    split_generator = torch.Generator().manual_seed(args.split_seed)
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len], generator=split_generator)

    train_loader = build_dataloader(
        train_dataset,
        tokenizer,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.loader_workers,
        device=device,
        drop_last=False,
    )
    val_loader = build_dataloader(
        val_dataset,
        tokenizer,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.loader_workers,
        device=device,
        drop_last=False,
    )
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    print("Creating student model...")
    bert_model_dir = Path(__file__).resolve().parent / "bert-base-chinese"
    mlm_model = BertForMaskedLM.from_pretrained(str(bert_model_dir), local_files_only=True)
    model = SarkazBert(mlm_model, dict_size=args.teacher_student_vocab_size)

    bert_layers = model.bert_model.encoder.layer

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
    optimizer_steps_per_epoch = max(1, math.ceil(len(train_loader) / max(1, args.accumulation_steps)))
    target_steps = max(1, optimizer_steps_per_epoch * max(1, args.epochs))
    print(f"Target optimizer steps: {target_steps}")
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(10, target_steps // 15),
        num_training_steps=target_steps,
    )

    last_ckpt = checkpoint_dir / "last.pt"
    if last_ckpt.exists():
        try:
            try:
                ck = torch.load(str(last_ckpt), map_location="cpu", weights_only=False)
            except TypeError:
                ck = torch.load(str(last_ckpt), map_location="cpu")
            if "model_state_dict" not in ck:
                print("Checkpoint corrupted: missing model_state_dict. Need action.")
                return
            saved_train_level = ck.get("train_level", 0)
            level_epoch = ck.get("level_epoch", ck.get("epoch", 0))
            print(
                f"Checkpoint metadata: train_level={saved_train_level}, "
                f"level_epoch={level_epoch}, current_train_level={args.train_level}"
            )
            if "optimizer_state_dict" not in ck or "scheduler_state_dict" not in ck:
                print("  Note: checkpoint missing optimizer/scheduler state; optimizer will start fresh if loaded")
            if saved_train_level != args.train_level:
                print(f"[Level change detected] {saved_train_level} -> {args.train_level}")
            if level_epoch >= args.epochs:
                print(f"Checkpoint indicates {level_epoch} completed epochs, >= requested total {args.epochs}. Nothing to do.")
                return
        except Exception as exc:
            print(f"Warning: failed to read existing checkpoint metadata: {exc}")
            return

    log_dir = project_root / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_board = SummaryLogger(str(log_dir))

    trainer = SarkazBertTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        model_saver=model_saver,
        criterion=nn.KLDivLoss(reduction="batchmean"),
        device_str=device,
        enable_amp=args.enable_amp,
        enable_gradient_checkpointing=args.enable_gradient_checkpointing,
        early_stop_patience=args.patience,
        validate_interval=args.validate_interval,
        log_interval=args.log_interval,
        current_train_level=args.train_level,
        freeze_bert_epochs=args.freeze_bert_epochs,
        use_teacher_knowledge=use_teacher_supervision,
        teacher_temperature=args.teacher_temperature,
        summary_logger=summary_board,
        accumulation_steps=args.accumulation_steps,
    )

    trainer.train(target_epochs=args.epochs)

    if trainer.history_stats["train_loss"]:
        print("Training finished.")
        print(f"Final train loss: {trainer.history_stats['train_loss'][-1]:.4f}")
        print(f"Final val loss: {trainer.history_stats['val_loss'][-1]:.4f}")
        print(f"Final val accuracy: {trainer.history_stats['val_metric'][-1]:.4f}")
        print(f"Best accuracy: {trainer.best_score:.4f}")
    else:
        print("Training finished but no history recorded. Maybe no epochs were run.")


if __name__ == "__main__":
    main()
