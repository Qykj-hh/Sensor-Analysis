#!/usr/bin/env python3
import os
import json
import glob
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import threading
import pytz
import pandas as pd

CONFIG = {
    "tsv_directory": "/gpfs/group/belle2/group/accelerator/pv/2024/AA_tsv/",
    "abnormal_json": "./aborts_classified.json",
    "normal_json": "./normal_timestamps.json",
    "output_dir": "./dataset",
    "checkpoint_file": "./dataset/.checkpoint.txt",
    "log_file": "./build_dataset.log",
    "max_workers": 16,
    "time_before_sec": 30 * 60 + 5,
    "time_after_sec": 5,
    "match_before_sec": 60,
    "timezone": "Asia/Tokyo",
    "log_every": 200,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.FileHandler(CONFIG["log_file"], encoding="utf-8"), logging.StreamHandler()],
)
log = logging.getLogger(__name__)


def parse_timestamp_ms(ts: str, tz) -> int | None:
    if not ts:
        return None
    ts = ts.strip()
    if not ts:
        return None
    main = ts[:19]
    try:
        dt = datetime.strptime(main, "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None
    micro = 0
    if "." in ts:
        frac = ts.split(".", 1)[1]
        frac = "".join(ch for ch in frac if ch.isdigit())
        if frac:
            micro = int(frac[:6].ljust(6, "0"))
    dt = dt.replace(microsecond=micro)
    try:
        dt = tz.localize(dt)
    except Exception:
        try:
            dt = dt.replace(tzinfo=tz)
        except Exception:
            return None
    return int(dt.timestamp() * 1000)


def load_records(json_path: str, tz, default_label: str | None = None) -> list[dict]:
    if not os.path.exists(json_path):
        return []
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out: list[dict] = []
    for item in data:
        ts = item.get("timestamp", "")
        label = default_label if default_label else item.get("critical_trigger", "Unknown")
        ts_ms = parse_timestamp_ms(ts, tz)
        if ts_ms is None:
            continue
        out.append({"ts_ms": ts_ms, "label": label, "ts_str": ts})
    return out


def sanitize_label(name: str) -> str:
    if not name:
        return "Unknown"
    cleaned = name.strip()
    for ch in ["\\", "/", "?", "*", ":", '"', "<", ">", "|"]:
        cleaned = cleaned.replace(ch, "_")
    cleaned = cleaned.replace(" ", "_")
    return cleaned if cleaned else "Unknown"


def read_last_data_rel_ts(path: str, start_pos: int) -> int | None:
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            if size <= start_pos:
                return None
            read_size = min(8192, size - start_pos)
            f.seek(-read_size, 2)
            chunk = f.read(read_size)
            lines = chunk.splitlines()
            for line in reversed(lines):
                if not line or line.startswith(b"#"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                try:
                    return int(parts[0])
                except Exception:
                    continue
    except Exception:
        return None
    return None


def index_tsv_file(path: str) -> dict | None:
    try:
        t0 = None
        nval = None
        pv_name = None
        start_rel = None
        start_pos = 0
        with open(path, "rb") as f:
            for _ in range(20):
                line = f.readline()
                if not line:
                    break
                try:
                    text = line.decode("utf-8", errors="ignore")
                except Exception:
                    text = ""
                if text.startswith("#"):
                    if "# t0" in text:
                        parts = text.split()
                        if len(parts) >= 3:
                            try:
                                t0 = int(parts[2])
                            except Exception:
                                pass
                    if "# Nval" in text:
                        parts = text.split()
                        if len(parts) >= 3:
                            try:
                                nval = int(parts[2])
                            except Exception:
                                pass
                    if "# PV name" in text:
                        parts = text.split()
                        if parts:
                            pv_name = parts[-1]
                    continue
                start_pos = f.tell() - len(line)
                parts = text.split()
                if len(parts) >= 2:
                    try:
                        start_rel = int(parts[0])
                    except Exception:
                        start_rel = None
                break
        if t0 is None or nval != 1 or start_rel is None:
            return None
        end_rel = read_last_data_rel_ts(path, start_pos)
        if end_rel is None:
            end_rel = start_rel
        return {
            "path": path,
            "t0": t0,
            "start_rel": start_rel,
            "end_rel": end_rel,
            "pv_name": pv_name,
        }
    except Exception:
        return None


def scan_tsv(directory: str, workers: int) -> list[dict]:
    files = glob.glob(os.path.join(directory, "*.tsv"))
    if not files:
        return []
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as exe:
        futures = {exe.submit(index_tsv_file, f): f for f in files}
        for future in as_completed(futures):
            try:
                info = future.result()
            except Exception:
                info = None
            if info:
                results.append(info)
    return results


def load_checkpoint(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            return {line.strip() for line in f if line.strip()}
    except Exception:
        return set()


def save_checkpoint(path: str, done: set[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in sorted(done):
            f.write(item + "\n")


def iter_tsv_data(path: str):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line:
                continue
            if line[0] == "#":
                continue
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                rel_ts = int(parts[0])
                val = float(parts[1])
            except Exception:
                continue
            yield rel_ts, val


def find_matched_events(path: str, t0: int, candidates: list[dict]) -> set[int]:
    if not candidates:
        return set()
    starts = sorted((c["match_start_ms"] - t0, i) for i, c in enumerate(candidates))
    ends = sorted((c["match_end_ms"] - t0, i) for i, c in enumerate(candidates))
    min_start = starts[0][0]
    max_end = ends[-1][0]
    matched: set[int] = set()
    active: set[int] = set()
    start_i = 0
    end_i = 0
    for rel_ts, _ in iter_tsv_data(path):
        if rel_ts < min_start:
            continue
        if rel_ts > max_end:
            break
        while start_i < len(starts) and starts[start_i][0] <= rel_ts:
            active.add(starts[start_i][1])
            start_i += 1
        while end_i < len(ends) and ends[end_i][0] < rel_ts:
            idx = ends[end_i][1]
            if idx in active:
                active.remove(idx)
            end_i += 1
        if active:
            matched.update(active)
            active.clear()
            if len(matched) == len(candidates):
                break
    return matched


def collect_window_points(path: str, t0: int, candidates: list[dict]) -> dict[str, list[tuple[int, float]]]:
    label_points: dict[str, list[tuple[int, float]]] = defaultdict(list)
    if not candidates:
        return label_points
    starts = sorted((c["window_start_ms"] - t0, i) for i, c in enumerate(candidates))
    ends = sorted((c["window_end_ms"] - t0, i) for i, c in enumerate(candidates))
    min_start = starts[0][0]
    max_end = ends[-1][0]
    label_counts: dict[str, int] = defaultdict(int)
    start_i = 0
    end_i = 0
    for rel_ts, val in iter_tsv_data(path):
        if rel_ts < min_start:
            continue
        if rel_ts > max_end:
            break
        while start_i < len(starts) and starts[start_i][0] <= rel_ts:
            idx = starts[start_i][1]
            label_counts[candidates[idx]["label_dir"]] += 1
            start_i += 1
        while end_i < len(ends) and ends[end_i][0] < rel_ts:
            idx = ends[end_i][1]
            label = candidates[idx]["label_dir"]
            if label_counts[label] > 0:
                label_counts[label] -= 1
            end_i += 1
        if label_counts:
            abs_ts = t0 + rel_ts
            for label, count in label_counts.items():
                if count > 0:
                    label_points[label].append((abs_ts, val))
    return label_points


def write_parquet(points: list[tuple[int, float]], pv_name: str, path: str) -> int:
    if not points:
        return 0
    df_new = pd.DataFrame(points, columns=["epoch_ms", "value"])
    df_new["pv_name"] = pv_name
    if os.path.exists(path):
        try:
            df_old = pd.read_parquet(path, columns=["epoch_ms", "value", "pv_name"])
            df = pd.concat([df_old, df_new], ignore_index=True)
        except Exception:
            df = df_new
    else:
        df = df_new
    df = df.drop_duplicates(subset=["epoch_ms"], keep="first")
    df = df.sort_values("epoch_ms")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False, compression="snappy")
    return len(df_new)


def build_events(normal_records: list[dict], abnormal_records: list[dict], cfg: dict) -> list[dict]:
    events: list[dict] = []
    before_ms = int(cfg["time_before_sec"]) * 1000
    after_ms = int(cfg["time_after_sec"]) * 1000
    match_ms = int(cfg["match_before_sec"]) * 1000
    for rec in normal_records:
        ts_ms = rec["ts_ms"]
        events.append(
            {
                "ts_ms": ts_ms,
                "label_dir": "Normal",
                "window_start_ms": ts_ms - before_ms,
                "window_end_ms": ts_ms - after_ms,
                "match_start_ms": ts_ms - match_ms,
                "match_end_ms": ts_ms,
            }
        )
    for rec in abnormal_records:
        ts_ms = rec["ts_ms"]
        trigger = rec["label"]
        label_dir = os.path.join("Abnormal", sanitize_label(trigger))
        events.append(
            {
                "ts_ms": ts_ms,
                "label_dir": label_dir,
                "window_start_ms": ts_ms - before_ms,
                "window_end_ms": ts_ms - after_ms,
                "match_start_ms": ts_ms - match_ms,
                "match_end_ms": ts_ms,
            }
        )
    return events


def process_file(file_info: dict, events: list[dict], cfg: dict) -> tuple[str, bool, int, int, int, str]:
    path = file_info["path"]
    t0 = file_info["t0"]
    file_start_abs = t0 + file_info["start_rel"]
    file_end_abs = t0 + file_info["end_rel"]
    pv_file = os.path.splitext(os.path.basename(path))[0]
    pv_name = file_info.get("pv_name") or pv_file
    candidates = [
        e
        for e in events
        if e["match_start_ms"] <= file_end_abs and e["match_end_ms"] >= file_start_abs
    ]
    if not candidates:
        return path, True, 0, 0, 0, "no_candidates"
    matched_idx = find_matched_events(path, t0, candidates)
    if not matched_idx:
        return path, True, 0, 0, 0, "no_match"
    matched = [candidates[i] for i in sorted(matched_idx)]
    label_points = collect_window_points(path, t0, matched)
    if not label_points:
        return path, True, 0, 0, 0, "no_points"
    total_written = 0
    window_count = 0
    for label_dir, points in label_points.items():
        if not points:
            continue
        out_dir = os.path.join(cfg["output_dir"], label_dir)
        out_path = os.path.join(out_dir, f"{pv_file}.parquet")
        total_written += write_parquet(points, pv_name, out_path)
        window_count += 1
    if total_written == 0:
        return path, True, 0, len(matched), window_count, "empty_write"
    return path, True, total_written, len(matched), window_count, "ok"


def main() -> None:
    cfg = CONFIG
    tz = pytz.timezone(cfg["timezone"])
    log.info("加载时间戳")
    abnormal = load_records(cfg["abnormal_json"], tz)
    normal = load_records(cfg["normal_json"], tz, "Normal")
    events = build_events(normal, abnormal, cfg)
    log.info("异常标签: %d 正常标签: %d", len(abnormal), len(normal))
    log.info("扫描TSV文件")
    tsv_files = scan_tsv(cfg["tsv_directory"], cfg["max_workers"])
    log.info("有效TSV: %d", len(tsv_files))
    if not tsv_files:
        return
    done = load_checkpoint(cfg["checkpoint_file"])
    pending = [f for f in tsv_files if f["path"] not in done]
    log.info("待处理TSV: %d", len(pending))
    if not pending:
        return
    os.makedirs(cfg["output_dir"], exist_ok=True)
    lock = threading.Lock()
    processed = [0]
    skipped = [0]
    skipped_no_candidates = [0]
    skipped_no_match = [0]
    skipped_no_points = [0]
    skipped_empty_write = [0]
    written_total = [0]
    matched_total = [0]
    windows_total = [0]
    with ThreadPoolExecutor(max_workers=cfg["max_workers"]) as exe:
        futures = {exe.submit(process_file, f, events, cfg): f["path"] for f in pending}
        for future in as_completed(futures):
            path = futures[future]
            try:
                res_path, ok, written, matched, windows, status = future.result()
            except Exception:
                res_path, ok, written, matched, windows, status = path, False, 0, 0, 0, "error"
            with lock:
                processed[0] += 1
                if ok:
                    done.add(res_path)
                    save_checkpoint(cfg["checkpoint_file"], done)
                    written_total[0] += written
                    matched_total[0] += matched
                    windows_total[0] += windows
                    if status != "ok":
                        skipped[0] += 1
                        if status == "no_candidates":
                            skipped_no_candidates[0] += 1
                        elif status == "no_match":
                            skipped_no_match[0] += 1
                        elif status == "no_points":
                            skipped_no_points[0] += 1
                        elif status == "empty_write":
                            skipped_empty_write[0] += 1
                    if status == "ok":
                        log.info("处理 %s 窗口 %d 写入 %d", os.path.basename(res_path), windows, written)
                    else:
                        log.info("跳过 %s 原因 %s", os.path.basename(res_path), status)
                if processed[0] % cfg["log_every"] == 0 or processed[0] == len(pending):
                    remaining = len(pending) - processed[0]
                    log.info(
                        "进度 %d/%d 剩余 %d 写入 %d 匹配 %d 窗口 %d 跳过 %d",
                        processed[0],
                        len(pending),
                        remaining,
                        written_total[0],
                        matched_total[0],
                        windows_total[0],
                        skipped[0],
                    )
    log.info(
        "完成 写入 %d 匹配 %d 窗口 %d 跳过 %d (无候选 %d 无匹配 %d 无数据 %d 空写入 %d) 总计 %d",
        written_total[0],
        matched_total[0],
        windows_total[0],
        skipped[0],
        skipped_no_candidates[0],
        skipped_no_match[0],
        skipped_no_points[0],
        skipped_empty_write[0],
        len(tsv_files),
    )


if __name__ == "__main__":
    main()
