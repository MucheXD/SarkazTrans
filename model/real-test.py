"""Interactive inference script for SarkazBert.

Loads the best checkpoint, prints metadata, and lets the user type an
encoded input string to inspect top-1, beam-search top-5, and per-position
top-5 token distributions.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
from torch.amp.autocast_mode import autocast
from transformers import BertModel

from dataset import SarkazCharmap
from sarkazBert import SarkazBert
from tokenizer import SarkazTokenizer


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
DEFAULT_BERT_DIR = BASE_DIR / "bert-base-chinese"


def get_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Interactive SarkazBert test script")
	parser.add_argument("--checkpoint-dir", type=str, default=str(DEFAULT_CHECKPOINT_DIR))
	parser.add_argument("--checkpoint-name", type=str, default="best")
	parser.add_argument("--device", type=str, default=None)
	parser.add_argument("--beam-width", type=int, default=5)
	parser.add_argument("--topk", type=int, default=5)
	parser.add_argument(
		"--default-t",
		type=int,
		default=0,
		choices=(0, 1),
		help="Default t value when the prompt does not include a prefix like 0|text or 1|text",
	)
	return parser.parse_args()


def resolve_checkpoint_dir(raw_path: str) -> Path:
	checkpoint_dir = Path(raw_path)
	if not checkpoint_dir.is_absolute():
		checkpoint_dir = PROJECT_ROOT / checkpoint_dir
	return checkpoint_dir.resolve()


def pick_device(device_arg: str | None) -> torch.device:
	if device_arg:
		return torch.device(device_arg)
	if torch.cuda.is_available():
		return torch.device("cuda")
	return torch.device("cpu")


def load_model(device: torch.device, checkpoint_dir: Path, checkpoint_name: str) -> Tuple[SarkazBert, dict, SarkazTokenizer, SarkazCharmap]:
	tokenizer = SarkazTokenizer()
	charmap = SarkazCharmap()
	bert_model = BertModel.from_pretrained(str(DEFAULT_BERT_DIR))
	model = SarkazBert(bert_model)

	checkpoint_path = checkpoint_dir / f"{checkpoint_name}.pt"
	if not checkpoint_path.exists():
		raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

	checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
	if "model_state_dict" not in checkpoint:
		raise RuntimeError(f"Checkpoint missing model_state_dict: {checkpoint_path}")

	ck = checkpoint["model_state_dict"]
	try:
		model.load_state_dict(ck, strict=True)
	except RuntimeError as e:
		# Try a non-strict load and print helpful diagnostics so the user
		# understands which keys differ between the checkpoint and current model.
		print(f"Warning: strict state_dict load failed: {e}")
		model_keys = set(model.state_dict().keys())
		ck_keys = set(ck.keys())
		missing = sorted(list(model_keys - ck_keys))
		unexpected = sorted(list(ck_keys - model_keys))
		if missing:
			print(f"  Missing keys in checkpoint (will be randomly initialized): {len(missing)} examples, first 10 -> {missing[:10]}")
		if unexpected:
			print(f"  Unexpected keys in checkpoint (ignored): {len(unexpected)} examples, first 10 -> {unexpected[:10]}")
		model.load_state_dict(ck, strict=False)
	model = model.to(device)
	model.eval()
	return model, checkpoint, tokenizer, charmap


def print_metadata(
	device: torch.device,
	checkpoint_path: Path,
	checkpoint: dict,
	tokenizer: SarkazTokenizer,
	charmap: SarkazCharmap,
	model: SarkazBert,
) -> None:
	bert_cfg = model.bert_model.config
	print("=" * 80)
	print("SarkazBert interactive test")
	print(f"Device: {device}")
	print(f"Checkpoint: {checkpoint_path}")
	print(f"Checkpoint train_level: {checkpoint.get('train_level', 'N/A')}")
	print(f"Checkpoint level_epoch: {checkpoint.get('level_epoch', 'N/A')}")
	print(f"Checkpoint best_score: {checkpoint.get('best_score', 'N/A')}")
	print(f"Tokenizer vocab size: {len(tokenizer.id_to_token)}")
	print(f"Supported input chars: {len(tokenizer._input_char_to_id)}")
	print(f"Char map size: {len(charmap.chars)}")
	print(f"Model dict size: {model.mapper.out_features}")
	print(f"BERT hidden size: {bert_cfg.hidden_size}")
	print(f"BERT layers: {bert_cfg.num_hidden_layers}")
	print(f"AMP on CUDA: {device.type == 'cuda'}")
	print("Input format: 0|abcde or 1|abcde; plain text uses default t=0")
	print("Commands: exit / quit / q")
	print("=" * 80)


def parse_user_input(raw_text: str, default_t: int) -> Tuple[int, str]:
	text = raw_text.strip()
	if "|" in text and len(text) >= 3 and text[0] in {"0", "1"}:
		maybe_t, maybe_text = text.split("|", 1)
		if maybe_t in {"0", "1"}:
			return int(maybe_t), maybe_text
	return default_t, text


def build_inputs(tokenizer: SarkazTokenizer, t: int, input_text: str, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	normalized = tokenizer._normalize(input_text)
	if not normalized:
		raise ValueError("输入不能为空")

	unknown_chars = [ch for ch in normalized if ch not in "abcdefghijklmnopqrstuvwxyz"]
	if unknown_chars:
		unique_unknown = "".join(dict.fromkeys(unknown_chars))
		raise ValueError(f"输入包含未支持字符: {unique_unknown}. 这里只支持 a-z")

	core_ids, _ = tokenizer._encode_core(normalized)
	head_token_ids = [tokenizer.cls_id, tokenizer.magic_id_0 if t == 0 else tokenizer.magic_id_1, tokenizer.sep_id]
	head_ids = torch.tensor([head_token_ids], dtype=torch.long, device=device)
	core_ids = torch.tensor([core_ids], dtype=torch.long, device=device)
	attention_mask = torch.ones((1, head_ids.size(1) + core_ids.size(1)), dtype=torch.long, device=device)
	return head_ids, core_ids, attention_mask


def infer_logits(
	model: SarkazBert,
	charmap: SarkazCharmap,
	head_ids: torch.Tensor,
	core_ids: torch.Tensor,
	attention_mask: torch.Tensor,
	device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
	with torch.inference_mode():
		if device.type == "cuda":
			with autocast(device_type="cuda"):
				logits = model(head_ids, core_ids, attention_mask)
		else:
			logits = model(head_ids, core_ids, attention_mask)

		char_mask = charmap.map_core_ids(core_ids)
		masked_logits = logits.masked_fill(~char_mask, -1e4)
		probabilities = torch.softmax(masked_logits, dim=-1)
	return masked_logits, probabilities


def decode_token_ids(tokenizer: SarkazTokenizer, token_ids: Sequence[int]) -> str:
	decoded = tokenizer.decode(list(token_ids))
	if isinstance(decoded, list):
		decoded = "".join(map(str, decoded))
	return str(decoded).replace("*", "?")


def greedy_sentence(tokenizer: SarkazTokenizer, probabilities: torch.Tensor) -> Tuple[str, List[int]]:
	top1_ids = torch.argmax(probabilities, dim=-1).squeeze(0).tolist()
	return decode_token_ids(tokenizer, top1_ids), top1_ids


def beam_search_sentence(
	tokenizer: SarkazTokenizer,
	probabilities: torch.Tensor,
	beam_width: int,
) -> Tuple[str, List[int], float]:
	beams: List[Tuple[List[int], float]] = [([], 0.0)]
	seq_len = probabilities.size(1)

	for position in range(seq_len):
		position_probs = probabilities[0, position]
		top_probs, top_ids = torch.topk(position_probs, k=min(beam_width, position_probs.size(-1)))

		expanded: List[Tuple[List[int], float]] = []
		for prefix_ids, prefix_score in beams:
			for token_id, prob in zip(top_ids.tolist(), top_probs.tolist()):
				score = prefix_score + math.log(max(prob, 1e-12))
				expanded.append((prefix_ids + [token_id], score))

		expanded.sort(key=lambda item: item[1], reverse=True)
		beams = expanded[:beam_width]

	best_ids, best_score = beams[0]
	return decode_token_ids(tokenizer, best_ids), best_ids, best_score


def print_topk_table(tokenizer: SarkazTokenizer, probabilities: torch.Tensor, topk: int) -> None:
	seq_len = probabilities.size(1)
	k = min(topk, probabilities.size(-1))

	for position in range(seq_len):
		top_probs, top_ids = torch.topk(probabilities[0, position], k=k)
		cells = []
		for token_id, prob in zip(top_ids.tolist(), top_probs.tolist()):
			token = tokenizer.id_to_token.get(token_id, "?")
			token_display = token if token != "\n" else "\\n"
			cells.append(f"{token_display} {prob:.3f}")
		print(f"Pos {position + 1:02d}: | " + " | ".join(cells) + " |")


def run_once(
	model: SarkazBert,
	tokenizer: SarkazTokenizer,
	charmap: SarkazCharmap,
	device: torch.device,
	t: int,
	input_text: str,
	beam_width: int,
	topk: int,
) -> None:
	head_ids, core_ids, attention_mask = build_inputs(tokenizer, t, input_text, device)
	masked_logits, probabilities = infer_logits(model, charmap, head_ids, core_ids, attention_mask, device)
	top1_sentence, _ = greedy_sentence(tokenizer, probabilities)
	beam_sentence, _, beam_score = beam_search_sentence(tokenizer, probabilities, beam_width=beam_width)

	print(f"Input: {tokenizer._normalize(input_text)}")
	print(f"Top1: {top1_sentence}")
	print(f"Top5: {beam_sentence} (score={beam_score:.4f})")
	print(f"Length: {masked_logits.size(1)} positions")
	print_topk_table(tokenizer, probabilities, topk=topk)


def main() -> None:
	args = get_args()
	checkpoint_dir = resolve_checkpoint_dir(args.checkpoint_dir)
	device = pick_device(args.device)

	model, checkpoint, tokenizer, charmap = load_model(device, checkpoint_dir, args.checkpoint_name)
	checkpoint_path = checkpoint_dir / f"{args.checkpoint_name}.pt"
	print_metadata(device, checkpoint_path, checkpoint, tokenizer, charmap, model)

	while True:
		try:
			raw = input("> ").strip()
		except (EOFError, KeyboardInterrupt):
			print()
			break

		if not raw:
			continue
		if raw.lower() in {"exit", "quit", "q"}:
			break

		try:
			t, text = parse_user_input(raw, args.default_t)
			run_once(
				model=model,
				tokenizer=tokenizer,
				charmap=charmap,
				device=device,
				t=t,
				input_text=text,
				beam_width=args.beam_width,
				topk=args.topk,
			)
		except Exception as exc:
			print(f"Error: {exc}")


if __name__ == "__main__":
	main()
