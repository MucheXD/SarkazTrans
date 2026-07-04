from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from dataset import SarkazCharmap


CACHE_VERSION = 1


@dataclass(frozen=True)
class TeacherKnowledgeConfig:
    cache_path: Path
    topk: int = 8
    student_vocab_size: int = 8100
    temperature: float = 1.0
    hard_mask: float = -1.0e9
    extract_batch_size: int = 4
    vector_chunk_size: int = 64
    num_workers: int = 0
    pin_memory: bool = False

    def __post_init__(self):
        if self.topk <= 0:
            raise ValueError("teacher topk must be positive")
        if self.student_vocab_size <= 0:
            raise ValueError("student_vocab_size must be positive")
        if self.temperature <= 0:
            raise ValueError("teacher temperature must be positive")
        if self.extract_batch_size <= 0:
            raise ValueError("teacher extract batch size must be positive")
        if self.vector_chunk_size <= 0:
            raise ValueError("teacher vector chunk size must be positive")


class TeacherKnowledgeStore:
    """Compact ragged store for teacher top-k distributions.

    Data layout:
      - sample_indices: shape (num_samples,)
      - offsets: shape (num_samples + 1,), offsets into concatenated rows
      - topk_ids: shape (total_core_positions, topk), int16
      - topk_probs: shape (total_core_positions, topk), float16
    """

    def __init__(
        self,
        sample_indices: torch.Tensor,
        offsets: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_probs: torch.Tensor,
        metadata: Dict[str, Any],
    ):
        if sample_indices.dim() != 1:
            raise ValueError("sample_indices must be 1D")
        if offsets.dim() != 1 or offsets.numel() != sample_indices.numel() + 1:
            raise ValueError("offsets must be 1D with len(sample_indices)+1 elements")
        if topk_ids.shape != topk_probs.shape or topk_ids.dim() != 2:
            raise ValueError("topk_ids and topk_probs must have the same 2D shape")
        if int(offsets[-1].item()) != topk_ids.size(0):
            raise ValueError("last offset must equal the number of stored rows")

        self.sample_indices = sample_indices.cpu().long()
        self.offsets = offsets.cpu().long()
        self.topk_ids = topk_ids.cpu().to(torch.int16)
        self.topk_probs = topk_probs.cpu().to(torch.float16)
        self.metadata = dict(metadata)
        self._row_by_sample_index = {int(sample_id): row for row, sample_id in enumerate(self.sample_indices.tolist())}

    @property
    def topk(self) -> int:
        return int(self.topk_ids.size(1))

    def __len__(self) -> int:
        return int(self.sample_indices.numel())

    def has_sample(self, sample_index: int) -> bool:
        return int(sample_index) in self._row_by_sample_index

    def get(self, sample_index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self._row_by_sample_index.get(int(sample_index))
        if row is None:
            raise KeyError(f"teacher knowledge missing for sample_index={sample_index}")
        start = int(self.offsets[row].item())
        end = int(self.offsets[row + 1].item())
        return self.topk_ids[start:end], self.topk_probs[start:end]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "version": CACHE_VERSION,
                "sample_indices": self.sample_indices,
                "offsets": self.offsets,
                "topk_ids": self.topk_ids,
                "topk_probs": self.topk_probs,
                "metadata": self.metadata,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, map_location: str | torch.device = "cpu") -> "TeacherKnowledgeStore":
        path = Path(path)
        try:
            payload = torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location=map_location)
        if payload.get("version") != CACHE_VERSION:
            raise ValueError(f"unsupported teacher knowledge cache version: {payload.get('version')}")
        return cls(
            sample_indices=payload["sample_indices"],
            offsets=payload["offsets"],
            topk_ids=payload["topk_ids"],
            topk_probs=payload["topk_probs"],
            metadata=payload.get("metadata", {}),
        )

    def is_compatible(self, expected_metadata: Dict[str, Any]) -> bool:
        keys = [
            "dataset_len",
            "topk",
            "student_vocab_size",
            "temperature",
            "teacher_model_path",
            "train_data_path",
            "train_data_size",
            "train_data_mtime_ns",
        ]
        for key in keys:
            if self.metadata.get(key) != expected_metadata.get(key):
                return False
        return True


class TeacherKnowledgeDataset(Dataset):
    """Dataset wrapper that attaches cached teacher top-k tensors to each sample."""

    def __init__(self, base_dataset: Dataset, knowledge: TeacherKnowledgeStore):
        self.base_dataset = base_dataset
        self.knowledge = knowledge

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = dict(self.base_dataset[index])
        sample_index = int(item.get("sample_index", index))
        topk_ids, topk_probs = self.knowledge.get(sample_index)
        item["teacher_topk_ids"] = topk_ids
        item["teacher_topk_probs"] = topk_probs
        return item


def expected_knowledge_metadata(
    dataset: Dataset,
    train_data_path: str | Path,
    teacher_model_path: str | Path,
    topk: int,
    student_vocab_size: int,
    temperature: float,
) -> Dict[str, Any]:
    train_data_path = Path(train_data_path).resolve()
    teacher_model_path = Path(teacher_model_path).resolve()
    stat = train_data_path.stat() if train_data_path.exists() else None
    return {
        "version": CACHE_VERSION,
        "dataset_len": len(dataset),
        "topk": int(topk),
        "student_vocab_size": int(student_vocab_size),
        "temperature": float(temperature),
        "teacher_model_path": str(teacher_model_path),
        "train_data_path": str(train_data_path),
        "train_data_size": int(stat.st_size) if stat else None,
        "train_data_mtime_ns": int(stat.st_mtime_ns) if stat else None,
    }


def _metadata_json(metadata: Dict[str, Any]) -> str:
    return json.dumps(metadata, ensure_ascii=False, sort_keys=True)


def _normalize_fp16_distribution(top_probs: torch.Tensor) -> torch.Tensor:
    """Move tail mass to top-1 and correct the float16 rounding residue."""
    top_probs = top_probs.to(torch.float32)
    tail = (1.0 - top_probs.sum(dim=-1, keepdim=True)).clamp_min(0.0)
    top_probs[:, :1] = top_probs[:, :1] + tail

    top_probs_fp16 = top_probs.to(torch.float16)
    residue = 1.0 - top_probs_fp16.to(torch.float32).sum(dim=-1)
    corrected_top1 = (top_probs_fp16[:, 0].to(torch.float32) + residue).clamp_min(0.0)
    top_probs_fp16[:, 0] = corrected_top1.to(torch.float16)
    return top_probs_fp16


def _extract_topk_from_teacher_logits(
    teacher_logits: torch.Tensor,
    core_ids: torch.Tensor,
    target_ids: torch.Tensor,
    token_mask: torch.Tensor,
    core_attention_mask: torch.Tensor,
    sarkaz_mapping: SarkazCharmap,
    config: TeacherKnowledgeConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return padded batch tensors (batch, seq_len, topk) on CPU."""
    device = teacher_logits.device
    batch_size, seq_len, teacher_vocab_size = teacher_logits.shape
    if teacher_vocab_size < config.student_vocab_size:
        raise ValueError(
            f"teacher vocab size {teacher_vocab_size} is smaller than student_vocab_size {config.student_vocab_size}"
        )

    core_ids_device = core_ids.to(device=device, dtype=torch.long)
    target_ids_device = target_ids.to(device=device, dtype=torch.long)
    token_mask_device = token_mask.to(device=device).bool()
    core_attention_mask_device = core_attention_mask.to(device=device).bool()
    valid_mask = token_mask_device & core_attention_mask_device

    batch_top_ids = torch.zeros((batch_size, seq_len, config.topk), dtype=torch.int16)
    batch_top_probs = torch.zeros((batch_size, seq_len, config.topk), dtype=torch.float16)

    if valid_mask.sum().item() == 0:
        return batch_top_ids, batch_top_probs

    char_mask = sarkaz_mapping.map_core_ids(core_ids_device)
    if char_mask.size(-1) != config.student_vocab_size:
        if char_mask.size(-1) > config.student_vocab_size:
            char_mask = char_mask[..., : config.student_vocab_size]
        else:
            pad_shape = (*char_mask.shape[:-1], config.student_vocab_size - char_mask.size(-1))
            char_mask = torch.cat([char_mask, torch.zeros(pad_shape, dtype=torch.bool, device=device)], dim=-1)

    selected_char_mask = char_mask[valid_mask]
    allowed_counts = selected_char_mask.sum(dim=-1)
    if (allowed_counts <= 0).any():
        bad = (allowed_counts <= 0).nonzero(as_tuple=False)[0].item()
        flat_indices = valid_mask.view(-1).nonzero(as_tuple=False).view(-1)
        flat_idx = int(flat_indices[bad].item())
        raise RuntimeError(
            f"character hard mask has no allowed tokens at flat valid index {flat_idx}; check map.txt/core_ids"
        )

    safe_targets = target_ids_device.clamp(min=0, max=config.student_vocab_size - 1)
    target_in_vocab = target_ids_device < config.student_vocab_size
    target_allowed = char_mask.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1) & target_in_vocab
    target_allowed = target_allowed | ~valid_mask
    if not target_allowed.all():
        bad_pos = (~target_allowed).nonzero(as_tuple=False)[0].tolist()
        bad_target = int(target_ids_device[bad_pos[0], bad_pos[1]].item())
        raise RuntimeError(
            "target token is excluded by the teacher/student hard mask during knowledge extraction: "
            f"batch={bad_pos[0]}, seq={bad_pos[1]}, target_id={bad_target}"
        )

    logits = teacher_logits[..., : config.student_vocab_size].to(torch.float32)
    logits = logits.masked_fill(~char_mask, config.hard_mask)
    supervised_logits = logits[valid_mask]
    probabilities = torch.softmax(supervised_logits / config.temperature, dim=-1)

    k = min(config.topk, config.student_vocab_size)
    top_probs, top_ids = torch.topk(probabilities, k=k, dim=-1)
    if k < config.topk:
        pad_cols = config.topk - k
        top_ids = torch.cat(
            [top_ids, torch.zeros((top_ids.size(0), pad_cols), dtype=top_ids.dtype, device=device)], dim=-1
        )
        top_probs = torch.cat(
            [top_probs, torch.zeros((top_probs.size(0), pad_cols), dtype=top_probs.dtype, device=device)], dim=-1
        )

    top_probs_fp16 = _normalize_fp16_distribution(top_probs)

    flat_mask_cpu = valid_mask.view(-1).cpu()
    flat_ids = batch_top_ids.view(-1, config.topk)
    flat_probs = batch_top_probs.view(-1, config.topk)
    flat_ids[flat_mask_cpu] = top_ids.detach().cpu().to(torch.int16)
    flat_probs[flat_mask_cpu] = top_probs_fp16.detach().cpu()
    return batch_top_ids, batch_top_probs


def extract_teacher_knowledge(
    dataset: Dataset,
    tokenizer: Any,
    teacher_model: Any,
    config: TeacherKnowledgeConfig,
    metadata: Dict[str, Any],
) -> TeacherKnowledgeStore:
    """Run the teacher once over the dataset and build a compact top-k cache."""
    loader = DataLoader(
        dataset,
        batch_size=config.extract_batch_size,
        shuffle=False,
        collate_fn=tokenizer.collate,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.num_workers > 0,
        drop_last=False,
    )
    sarkaz_mapping = SarkazCharmap()

    sample_index_chunks = []
    offsets = [0]
    ids_chunks = []
    probs_chunks = []

    total_samples = len(dataset)
    processed = 0
    print(
        "Extracting teacher knowledge: "
        f"samples={total_samples}, batch={config.extract_batch_size}, topk={config.topk}, "
        f"student_vocab={config.student_vocab_size}, temperature={config.temperature:g}"
    )

    for batch_idx, batch in enumerate(loader, start=1):
        sample_indices = batch.get("sample_index")
        if sample_indices is None:
            sample_indices = torch.arange(processed, processed + batch["core_ids"].size(0), dtype=torch.long)

        target_ids = batch["target_ids"].to(teacher_model.device, non_blocking=True)
        token_mask = batch["token_mask"].to(teacher_model.device, non_blocking=True)
        core_attention_mask = batch["core_attention_mask"].to(teacher_model.device, non_blocking=True)

        with torch.inference_mode():
            teacher_logits = teacher_model.Answer(
                target_ids=target_ids,
                attention_mask=core_attention_mask,
                token_mask=token_mask,
                vector_chunk_size=config.vector_chunk_size,
            )
            batch_top_ids, batch_top_probs = _extract_topk_from_teacher_logits(
                teacher_logits=teacher_logits,
                core_ids=batch["core_ids"],
                target_ids=batch["target_ids"],
                token_mask=batch["token_mask"],
                core_attention_mask=batch["core_attention_mask"],
                sarkaz_mapping=sarkaz_mapping,
                config=config,
            )

        core_lengths = batch["core_attention_mask"].sum(dim=1).cpu().long()
        for row, sample_index in enumerate(sample_indices.cpu().tolist()):
            length = int(core_lengths[row].item())
            sample_index_chunks.append(int(sample_index))
            ids_chunks.append(batch_top_ids[row, :length].contiguous())
            probs_chunks.append(batch_top_probs[row, :length].contiguous())
            offsets.append(offsets[-1] + length)

        processed += int(batch["core_ids"].size(0))
        if batch_idx == 1 or processed == total_samples or batch_idx % 50 == 0:
            total_positions = offsets[-1]
            print(f"  teacher cache batch {batch_idx}: {processed}/{total_samples} samples, {total_positions} positions")

        del teacher_logits, batch_top_ids, batch_top_probs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if ids_chunks:
        topk_ids = torch.cat(ids_chunks, dim=0).to(torch.int16)
        topk_probs = torch.cat(probs_chunks, dim=0).to(torch.float16)
    else:
        topk_ids = torch.zeros((0, config.topk), dtype=torch.int16)
        topk_probs = torch.zeros((0, config.topk), dtype=torch.float16)

    store = TeacherKnowledgeStore(
        sample_indices=torch.tensor(sample_index_chunks, dtype=torch.long),
        offsets=torch.tensor(offsets, dtype=torch.long),
        topk_ids=topk_ids,
        topk_probs=topk_probs,
        metadata=metadata,
    )
    return store


def build_or_load_teacher_knowledge(
    dataset: Dataset,
    tokenizer: Any,
    config: TeacherKnowledgeConfig,
    expected_metadata: Dict[str, Any],
    teacher_model_factory: Callable[[], Any],
    force_rebuild: bool = False,
) -> TeacherKnowledgeStore:
    cache_path = Path(config.cache_path)
    if cache_path.exists() and not force_rebuild:
        try:
            store = TeacherKnowledgeStore.load(cache_path)
            if store.is_compatible(expected_metadata):
                print(f"Loaded teacher knowledge cache: {cache_path}")
                print(
                    f"  samples={len(store)}, positions={int(store.offsets[-1].item())}, topk={store.topk}"
                )
                return store
            print("Teacher knowledge cache metadata mismatch; rebuilding.")
            print(f"  expected={_metadata_json(expected_metadata)}")
            print(f"  found={_metadata_json(store.metadata)}")
        except Exception as exc:
            print(f"Failed to load teacher knowledge cache {cache_path}: {exc}; rebuilding.")

    teacher_model = teacher_model_factory()
    try:
        store = extract_teacher_knowledge(dataset, tokenizer, teacher_model, config, expected_metadata)
        store.save(cache_path)
        print(f"Saved teacher knowledge cache: {cache_path}")
        print(f"  positions={int(store.offsets[-1].item())}, ids_dtype=int16, probs_dtype=float16")
        return store
    finally:
        try:
            if hasattr(teacher_model, "unload"):
                teacher_model.unload()
            else:
                del teacher_model
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
