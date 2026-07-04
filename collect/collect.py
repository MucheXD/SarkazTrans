import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import requests

from deserialize import wikidat_deserialize

# 此脚本从 warfarin.wiki 采集数据
BASE_URL = "https://warfarin.wiki/cn"

def _ensure_dirs(base_dir: Path) -> Dict[str, Path]:
    raw_dir = base_dir / "data" / "raw"
    json_dir = base_dir / "data" / "json"
    raw_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    return {"raw": raw_dir, "json": json_dir}


def _safe_filename(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_")


def _save_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fetch_with_retries(session: requests.Session, url: str, retries: int = 3, backoff: float = 0.5) -> Optional[bytes]:
    last_exc: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, timeout=30)
            if resp.status_code == 200:
                return resp.content
            else:
                logging.warning("Non-200 %s on %s (attempt %d)", resp.status_code, url, attempt)
        except Exception as e:
            last_exc = e
            logging.warning("Request error on %s (attempt %d): %s", url, attempt, e)

        sleep_for = backoff * attempt
        time.sleep(sleep_for)

    logging.error("Failed to fetch %s after %d attempts. Last error: %s", url, retries, last_exc)
    return None


def collect_all_data(base_dir: Optional[Union[str, Path]] = None,
                     base_url: str = BASE_URL,
                     min_delay: float = 0.2,
                     max_retries: int = 3,
                     force_recollect: bool = False) -> Dict[str, List[str]]:
    """
    执行全部采集流程：
    1. 获取 /missions.data 并保存到 data/raw/missions.data
    2. 调用 wikidat_deserialize 反序列化到 data/json/missions.json
    3. 遍历反序列化结果中的 data 下的 object，若 object 包含 `id` 字段，则请求 /missions/{id}.data 保存并反序列化

    返回字典包含成功与失败的 id 列表。
    """
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent
    else:
        base_dir = Path(base_dir)
    # mypy: base_dir is now a Path
    assert isinstance(base_dir, Path)

    # logging setup
    log_path = base_dir / "data" / "collect.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s",
                        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()])

    dirs = _ensure_dirs(base_dir)
    raw_dir: Path = dirs["raw"]
    json_dir: Path = dirs["json"]

    session = requests.Session()
    session.headers.update({"User-Agent": "collect-script/1.0"})

    result: Dict[str, List[str]] = {"succeeded": [], "failed": []}

    # 1. 获取 missions.data
    missions_url = f"{base_url}/missions.data"
    logging.info("Fetching %s", missions_url)
    missions_raw = _fetch_with_retries(session, missions_url, retries=max_retries)
    time.sleep(min_delay)

    if missions_raw is None:
        logging.error("Failed to download missions.data, aborting")
        return result

    missions_raw_path = raw_dir / "missions.data"
    _save_bytes(missions_raw_path, missions_raw)
    logging.info("Saved raw missions to %s", missions_raw_path)

    # 2. 反序列化
    missions_json_path = json_dir / "missions.json"
    try:
        wikidat_deserialize(str(missions_raw_path), str(missions_json_path))
        logging.info("Deserialized missions to %s", missions_json_path)
    except Exception as e:
        logging.exception("Deserialization failed for %s: %s", missions_raw_path, e)
        return result

    # 3. 遍历 data 中的 object，寻找带 id 的项
    try:
        decoded = _load_json(missions_json_path)
    except Exception as e:
        logging.exception("Failed to load deserialized JSON %s: %s", missions_json_path, e)
        return result

    data_section = decoded.get("data") if isinstance(decoded, dict) else None
    if data_section is None:
        logging.error("No 'data' section found in %s", missions_json_path)
        return result

    # normalize iterable of objects
    objects = []
    if isinstance(data_section, dict):
        for v in data_section.values():
            objects.append(v)
    elif isinstance(data_section, list):
        objects = data_section
    else:
        logging.warning("Unexpected data section type: %s", type(data_section))

    for obj in objects:
        if not isinstance(obj, dict):
            continue
        obj_id = obj.get("id")
        if not obj_id:
            continue
        safe_id = _safe_filename(str(obj_id))
        raw_path = raw_dir / f"{safe_id}.data"
        json_out = json_dir / f"{safe_id}.json"

        # 如果未强制重采集且 raw 已存在，则跳过网络请求
        if raw_path.exists() and not force_recollect:
            logging.info("Raw exists for id=%s at %s; skipping download", obj_id, raw_path)

            # 如果 json 不存在，则直接从 raw 反序列化
            if not json_out.exists():
                try:
                    wikidat_deserialize(str(raw_path), str(json_out))
                    logging.info("Deserialized existing raw %s -> %s", raw_path, json_out)
                    result["succeeded"].append(str(obj_id))
                except Exception as e:
                    logging.exception("Deserialization failed for existing raw %s: %s", raw_path, e)
                    result["failed"].append(str(obj_id))
            else:
                logging.info("JSON already exists for id=%s at %s; nothing to do", obj_id, json_out)

            continue

        # 否则（raw 不存在或强制重采集）进行网络请求并保存
        obj_url = f"{base_url}/missions/{obj_id}.data"
        logging.info("Fetching %s for id=%s", obj_url, obj_id)

        content = _fetch_with_retries(session, obj_url, retries=max_retries)
        time.sleep(min_delay)

        if content is None:
            logging.error("Failed to fetch data for id=%s", obj_id)
            result["failed"].append(str(obj_id))
            # 如果已有 raw 文件但本次请求失败且 json 不存在，尝试从已有 raw 反序列化
            if raw_path.exists() and not json_out.exists():
                try:
                    wikidat_deserialize(str(raw_path), str(json_out))
                    logging.info("Deserialized existing raw after fetch-failure %s -> %s", raw_path, json_out)
                    result["succeeded"].append(str(obj_id))
                except Exception as e:
                    logging.exception("Deserialization also failed for existing raw %s: %s", raw_path, e)
            continue

        _save_bytes(raw_path, content)
        logging.info("Saved raw %s", raw_path)

        try:
            wikidat_deserialize(str(raw_path), str(json_out))
            logging.info("Deserialized %s -> %s", raw_path, json_out)
            result["succeeded"].append(str(obj_id))
        except Exception as e:
            logging.exception("Deserialization failed for %s: %s", raw_path, e)
            result["failed"].append(str(obj_id))

    logging.info("Collect finished. succeeded=%d failed=%d", len(result["succeeded"]), len(result["failed"]))
    return result


if __name__ == "__main__":
    res = collect_all_data()
    print(res)
