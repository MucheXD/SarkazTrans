"""Standard BERT token inspection aligned with Teacher Model.

Reads text from stdin, passes the FULL unmasked text to the model,
and prints the top-k token predictions for each position.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = BASE_DIR.parent / "resources" / "bert-base-chinese"


def get_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="BERT token top-k tester (Aligned with Teacher)")
	parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_DIR))
	parser.add_argument("--device", type=str, default=None)
	parser.add_argument("--topk", type=int, default=5)
	parser.add_argument("--heatmap-topk", type=int, default=20)
	parser.add_argument("--heatmap-dir", type=str, default=str(BASE_DIR.parent / "test" / "bert-test-heatmaps"))
	parser.add_argument("--temperature", type=float, default=1.0)
	return parser.parse_args()


def pick_device(device_arg: str | None) -> torch.device:
	if device_arg:
		return torch.device(device_arg)
	if torch.cuda.is_available():
		return torch.device("cuda")
	return torch.device("cpu")


def load_model(model_path: str, device: torch.device) -> Tuple[object, torch.nn.Module]:
	model_dir = Path(model_path)
	if not model_dir.exists():
		raise FileNotFoundError(f"Model directory not found: {model_dir}")

	load_target = str(model_dir)
	tokenizer = AutoTokenizer.from_pretrained(load_target, local_files_only=True)
	model = AutoModelForMaskedLM.from_pretrained(load_target, local_files_only=True)
	model.to(device)
	model.eval()
	return tokenizer, model


def normalize_display_token(token: str | None) -> str:
	if token is None:
		return "[UNK]"
	return str(token).replace("\n", "\\n").replace("\t", "\\t").replace(" ", "␠")


def predict_topk_for_text(
	tokenizer: object,
	model: torch.nn.Module,
	device: torch.device,
	text: str,
	topk: int,
	temperature: float,
) -> List[Tuple[str, float, List[Tuple[str, float]]]]:
	if temperature <= 0:
		raise ValueError("temperature must be greater than 0")

	encoded = tokenizer(text, return_tensors="pt", add_special_tokens=True)
	input_ids = encoded["input_ids"].to(device)
	attention_mask = encoded.get("attention_mask")
	if attention_mask is not None:
		attention_mask = attention_mask.to(device)
	token_type_ids = encoded.get("token_type_ids")
	if token_type_ids is not None:
		token_type_ids = token_type_ids.to(device)

	token_ids = input_ids[0].tolist()
	input_tokens = tokenizer.convert_ids_to_tokens(token_ids)
	
	# 去掉开头的 CLS 和结尾的 SEP 计算有效位置
	num_positions = len(input_tokens) - 2
	if num_positions <= 0:
		raise ValueError("输入太短，无法进行预测")

	special_ids = set(getattr(tokenizer, "all_special_ids", []))

	# --- 核心改动：统一到目前 Teacher 的无遮掩行为，完全不破坏原句，直接 1 次前向传播 ---
	with torch.inference_mode():
		model_kwargs = {"input_ids": input_ids}
		if attention_mask is not None:
			model_kwargs["attention_mask"] = attention_mask
		if token_type_ids is not None:
			model_kwargs["token_type_ids"] = token_type_ids
		
		# 直接送入完整文本（不加 [MASK]）
		outputs = model(**model_kwargs)
		
		# 提取中间有效位置的 logits，排除两端的 CLS 和 SEP。形状变为: (num_positions, vocab_size)
		logits = outputs.logits[0, 1:-1]

	# 屏蔽特殊字符
	if special_ids:
		special_index = torch.tensor(sorted(special_ids), device=logits.device)
		logits = logits.clone()
		logits[:, special_index] = float("-inf")

	k = min(topk, logits.size(-1))
	# 应用温度和 Softmax
	probs = torch.softmax(logits / temperature, dim=-1)
	
	# 抓取原词在当前上下文中的预测概率
	col_indices = torch.arange(1, len(input_tokens) - 1, device=device)
	original_token_ids = input_ids[0, col_indices]
	row_indices = torch.arange(num_positions, device=device)
	source_probs = probs[row_indices, original_token_ids].tolist()
	
	# 获取 TopK 的概率与词 ID
	top_probs, top_ids = torch.topk(probs, k=k, dim=-1)
	top_probs_list = top_probs.tolist()
	top_ids_list = top_ids.tolist()

	# 整理为标准输出格式（仅在 CPU 上组装数据结构，不含任何模型计算）
	results: List[Tuple[str, float, List[Tuple[str, float]]]] = []
	for i, position in enumerate(range(1, len(input_tokens) - 1)):
		candidates: List[Tuple[str, float]] = []
		for token_id, prob in zip(top_ids_list[i], top_probs_list[i]):
			token = tokenizer.convert_ids_to_tokens([token_id])[0]
			candidates.append((normalize_display_token(token), float(prob)))

		results.append((normalize_display_token(input_tokens[position]), source_probs[i], candidates))

	return results


def print_table(text: str, predictions: List[Tuple[str, float, List[Tuple[str, float]]]], display_topk: int) -> None:
	print(f"输入: {text}")
	for index, (source_token, source_prob, candidates) in enumerate(predictions, start=1):
		cells = [f"{token} {prob:.3f}" for token, prob in candidates[:display_topk]]
		print(f"| {index:02d} | {source_token} | {source_prob:.3f} | " + " | ".join(cells) + " |")


def create_heatmap_image(
	text: str,
	predictions: List[Tuple[str, float, List[Tuple[str, float]]]],
	heatmap_dir: Path,
	temperature: float,
) -> Path:
	if temperature == 0:
		raise ValueError("temperature cannot be 0 for softmax heatmap generation")

	import matplotlib

	matplotlib.use("Agg")
	from matplotlib import font_manager
	import matplotlib.pyplot as plt

	for font_name in ("Microsoft YaHei", "SimHei", "SimSun", "Source Han Sans SC"):
		if font_name in {font.name for font in font_manager.fontManager.ttflist}:
			plt.rcParams["font.family"] = "sans-serif"
			plt.rcParams["font.sans-serif"] = [font_name]
			plt.rcParams["axes.unicode_minus"] = False
			break
	else:
		raise RuntimeError("No supported Chinese font found. Please install Microsoft YaHei, SimHei, SimSun, or Source Han Sans SC.")

	heatmap_dir.mkdir(parents=True, exist_ok=True)
	rows = len(predictions)
	cols = max((len(candidates) for _, _, candidates in predictions), default=0)
	if rows == 0 or cols == 0:
		raise ValueError("No predictions available to build heatmap")

	matrix = [[0.0 for _ in range(cols)] for _ in range(rows)]
	labels = [["" for _ in range(cols)] for _ in range(rows)]
	row_labels = []

	for row_index, (source_token, _, candidates) in enumerate(predictions):
		row_labels.append(source_token)
		for col_index, (token, probability) in enumerate(candidates):
			matrix[row_index][col_index] = probability
			labels[row_index][col_index] = f"{token}\n{probability:.3f}"

	fig_width = max(10.0, cols * 0.75)
	fig_height = max(3.5, rows * 0.55)
	fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=160)
	im = ax.imshow(matrix, aspect="auto", cmap="viridis", interpolation="nearest")

	ax.set_xticks(range(cols))
	ax.set_xticklabels([f"Top{i + 1}" for i in range(cols)], rotation=0)
	ax.set_yticks(range(rows))
	ax.set_yticklabels([f"{index + 1:02d} {token}" for index, token in enumerate(row_labels)])
	ax.set_xlabel("Candidate rank")
	ax.set_ylabel("Output token")
	ax.set_title(f"Top{cols} candidate probability heatmap (temperature={temperature:g})")

	threshold = max(max(row) for row in matrix) * 0.5
	for row_index in range(rows):
		for col_index in range(cols):
			value = matrix[row_index][col_index]
			if value <= 0:
				continue
			text_color = "white" if value <= threshold else "black"
			ax.text(col_index, row_index, labels[row_index][col_index], ha="center", va="center", fontsize=7, color=text_color)

	fig.colorbar(im, ax=ax, label="Probability", fraction=0.046, pad=0.04)
	fig.tight_layout()
	stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
	safe_text = "".join(ch if ch.isalnum() else "_" for ch in text[:12]).strip("_") or "input"
	output_path = heatmap_dir / f"heatmap-{safe_text}-{stamp}.png"
	fig.savefig(output_path, bbox_inches="tight")
	plt.close(fig)
	return output_path


def prompt_temperature(default_temperature: float) -> float:
	while True:
		try:
			raw = input(f"temperature [{default_temperature:g}]> ").strip()
		except (EOFError, KeyboardInterrupt):
			print()
			raise
		if not raw:
			return default_temperature
		try:
			temperature = float(raw)
			if temperature <= 0:
				raise ValueError
			return temperature
		except ValueError:
			print("Error: temperature must be a positive number")


def iter_inputs() -> Iterable[str]:
	if sys.stdin.isatty():
		while True:
			try:
				line = input("> ")
			except (EOFError, KeyboardInterrupt):
				print()
				return
			text = line.strip()
			if not text:
				continue
			if text.lower() in {"exit", "quit", "q"}:
				return
			yield text
		return

	for line in sys.stdin:
		text = line.strip()
		if text:
			yield text


def main() -> None:
	args = get_args()
	device = pick_device(args.device)
	tokenizer, model = load_model(args.model_path, device)
	heatmap_dir = Path(args.heatmap_dir)

	if sys.stdin.isatty():
		while True:
			try:
				temperature = prompt_temperature(args.temperature)
				text = input("text> ").strip()
			except (EOFError, KeyboardInterrupt):
				print()
				break
			if not text:
				continue
			if text.lower() in {"exit", "quit", "q"}:
				break
			try:
				predictions = predict_topk_for_text(tokenizer, model, device, text, args.heatmap_topk, temperature)
				print_table(text, predictions, args.topk)
				if temperature != 0:
					image_path = create_heatmap_image(text, predictions, heatmap_dir, temperature)
					print(f"Heatmap saved to: {image_path}")
			except Exception as exc:
				print(f"Error: {exc}")
	else:
		for text in iter_inputs():
			try:
				predictions = predict_topk_for_text(tokenizer, model, device, text, args.heatmap_topk, args.temperature)
				print_table(text, predictions, args.topk)
				if args.temperature != 0:
					image_path = create_heatmap_image(text, predictions, heatmap_dir, args.temperature)
					print(f"Heatmap saved to: {image_path}")
			except Exception as exc:
				print(f"Error: {exc}")


if __name__ == "__main__":
	main()