import json
import os
from functools import lru_cache


def wikidat_deserialize(input_file_path, output_file_path):
	with open(input_file_path, 'r', encoding='utf-8') as f:
		pool = json.load(f)

	@lru_cache(maxsize=None)
	def decode_index(index):
		value = pool[index]

		if isinstance(value, dict):
			decoded = {}
			for raw_key, raw_value in value.items():
				decoded_key = decode_key(raw_key)
				decoded[decoded_key] = decode_value(raw_value)
			return decoded

		if isinstance(value, list):
			return [decode_value(item) for item in value]

		return value


	def decode_key(raw_key):
		if isinstance(raw_key, str) and raw_key.startswith('_') and raw_key[1:].isdigit():
			decoded_key = decode_value(int(raw_key[1:]))
			if isinstance(decoded_key, (dict, list)):
				return json.dumps(decoded_key, ensure_ascii=False, separators=(',', ':'))
			return decoded_key
		return raw_key


	def decode_value(raw_value):
		if isinstance(raw_value, int) and not isinstance(raw_value, bool):
			if raw_value >= 0:
				return decode_index(raw_value)
			return None
		return raw_value


	def find_root_index():
		for index in range(len(pool)):
			decoded = decode_index(index)
			if isinstance(decoded, dict) and set(decoded.keys()) == {'meta', 'data', 'refs'}:
				return index
		raise ValueError('未找到可作为根节点的对象')


	root_index = find_root_index()
	result = decode_index(root_index)

	with open(output_file_path, 'w', encoding='utf-8') as f:
		json.dump(result, f, ensure_ascii=False, indent=2)
		f.write('\n')

	return output_file_path


if __name__ == '__main__':
	base_dir = os.path.dirname(os.path.abspath(__file__))
	data_path = os.path.join(base_dir, 'data', 'missions.data')
	output_path = os.path.join(base_dir, 'data', 'missions.json')
	print(wikidat_deserialize(data_path, output_path))
