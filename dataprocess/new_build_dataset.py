import os
import json
import logging
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

import pytz
import pandas as pd


CONFIG = {
    "input_dataset_dir": "/ghi/fs01/belle2/bdata/users/liuyu/_0204/dataset",
    "events_json": "./events_merged.json",
    "output_dataset_dir": "./new_dataset",
    "checkpoint_file": "./new_dataset/checkpoint.txt",
    "log_file": "./new_dataset/log.txt",
    "timezone": "Asia/Tokyo",
    "max_workers": 32,
    "time_before_sec": 30 * 60 + 5,
    "time_after_sec": 5,
    "log_every": 50,
    "resume": True,
    "dry_run": False,
    "limit_files": 0,
    "limit_events": 0,
}


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


def sanitize_dir_name(name: str) -> str:
    if not name:
        return "Unknown"
    cleaned = name.strip()
    for ch in ["\\", "/", "?", "*", ":", '"', "<", ">", "|"]:
        cleaned = cleaned.replace(ch, "_")
    cleaned = cleaned.replace(" ", "_")
    cleaned = cleaned.replace("\t", "_")
    cleaned = cleaned.replace("\n", "_")
    return cleaned if cleaned else "Unknown"


def load_events(events_path: str, tz, cfg: dict) -> list[dict]:
    if not os.path.exists(events_path):
        raise FileNotFoundError(events_path)
    with open(events_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    before_ms = int(cfg["time_before_sec"]) * 1000
    after_ms = int(cfg["time_after_sec"]) * 1000
    events: list[dict] = []
    used_dir: dict[str, int] = {}
    for item in data:
        ts_str = str(item.get("timestamp", "")).strip()
        ts_ms = parse_timestamp_ms(ts_str, tz)
        if ts_ms is None:
            continue
        split = str(item.get("split", "")).strip().lower()
        if split not in {"train", "test", "val"}:
            split = "train"
        base_dir = sanitize_dir_name(ts_str)
        suffix = used_dir.get(base_dir, 0)
        used_dir[base_dir] = suffix + 1
        event_dir = base_dir if suffix == 0 else f"{base_dir}__{suffix}"
        events.append(
            {
                "timestamp": ts_str,
                "epoch_ms": ts_ms,
                "event_level1": item.get("event_level1", "Unknown"),
                "event_level2": item.get("event_level2", "Unknown"),
                "split": split,
                "event_dir": event_dir,
                "window_start_ms": ts_ms - before_ms,
                "window_end_ms": ts_ms - after_ms,
            }
        )
    if int(cfg.get("limit_events") or 0) > 0:
        events = events[: int(cfg["limit_events"])]
    return events


def find_parquet_files(root_dir: str, limit: int) -> list[str]:
    out: list[str] = []
    for r, _, files in os.walk(root_dir):
        for fn in files:
            if fn.lower().endswith(".parquet"):
                out.append(os.path.join(r, fn))
                if limit and len(out) >= limit:
                    return out
    return out


def load_checkpoint(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            return {line.strip() for line in f if line.strip()}
    except Exception:
        return set()


def append_checkpoint(path: str, item: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(item + "\n")


def normalize_rel_path(path: str, base_dir: str) -> str:
    try:
        rel = os.path.relpath(path, base_dir)
    except Exception:
        rel = path
    return rel.replace("\\", "/")


def slice_df_by_window(df: pd.DataFrame, start_ms: int, end_ms: int):
    if df.empty:
        return df
    ts = df["epoch_ms"].to_numpy()
    left = ts.searchsorted(start_ms, side="left")
    right = ts.searchsorted(end_ms, side="right")
    if right <= left:
        return df.iloc[0:0]
    return df.iloc[left:right]


def process_parquet_file(
    parquet_path: str,
    events: list[dict],
    cfg: dict,
    input_root: str,
    output_root: str,
    log: logging.Logger,
) -> tuple[str, bool, int, int, Counter]:
    rel = normalize_rel_path(parquet_path, input_root)
    try:
        try:
            df = pd.read_parquet(parquet_path, columns=["epoch_ms", "value", "pv_name"])
        except Exception:
            df = pd.read_parquet(parquet_path)
    except Exception as e:
        log.error("读取失败 %s: %s", rel, e)
        return rel, False, 0, 0, Counter({"read_error": 1})
    if df.empty or "epoch_ms" not in df.columns or "value" not in df.columns:
        return rel, True, 0, 0, Counter({"empty_or_missing_cols": 1})
    try:
        df["epoch_ms"] = pd.to_numeric(df["epoch_ms"], errors="coerce")
        df = df.dropna(subset=["epoch_ms"])
        df["epoch_ms"] = df["epoch_ms"].astype("int64", errors="ignore")
    except Exception:
        pass
    if df.empty:
        return rel, True, 0, 0, Counter({"empty_after_coerce": 1})
    try:
        df = df.sort_values("epoch_ms")
    except Exception:
        pass
    pv_file = os.path.splitext(os.path.basename(parquet_path))[0]
    written_files = 0
    written_rows = 0
    stats = Counter()
    for ev in events:
        win = slice_df_by_window(df, int(ev["window_start_ms"]), int(ev["window_end_ms"]))
        if win.empty:
            continue
        out_dir = os.path.join(output_root, ev["split"], ev["event_dir"])
        out_path = os.path.join(out_dir, f"{pv_file}.parquet")
        if os.path.exists(out_path):
            stats["exists_skip"] += 1
            continue
        stats["write"] += 1
        written_files += 1
        written_rows += int(len(win))
        if cfg.get("dry_run"):
            continue
        os.makedirs(out_dir, exist_ok=True)
        try:
            win.to_parquet(out_path, index=False, compression="snappy")
        except Exception as e:
            log.error("写入失败 %s -> %s: %s", rel, normalize_rel_path(out_path, output_root), e)
            stats["write_error"] += 1
            try:
                if os.path.exists(out_path):
                    os.remove(out_path)
            except Exception:
                pass
            return rel, False, written_files, written_rows, stats
    if written_files == 0:
        stats["no_window_hit"] += 1
    return rel, True, written_files, written_rows, stats


def write_events_index(events: list[dict], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = [
        {
            "timestamp": e["timestamp"],
            "epoch_ms": int(e["epoch_ms"]),
            "event_level1": e.get("event_level1", "Unknown"),
            "event_level2": e.get("event_level2", "Unknown"),
            "split": e["split"],
            "event_dir": e["event_dir"],
        }
        for e in events
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    cfg = CONFIG
    output_root = os.path.abspath(cfg["output_dataset_dir"])
    log_file = str(cfg.get("log_file") or "").strip() or os.path.join(output_root, "split_dataset_by_events.log")
    checkpoint_file = str(cfg.get("checkpoint_file") or "").strip() or os.path.join(output_root, ".checkpoint.txt")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()],
    )
    log = logging.getLogger("split_dataset_by_events")

    input_root = os.path.abspath(cfg["input_dataset_dir"])
    tz = pytz.timezone(cfg["timezone"])

    events = load_events(cfg["events_json"], tz, cfg)
    if not events:
        log.error("未加载到任何有效事件: %s", cfg["events_json"])
        return
    split_counts = Counter(e["split"] for e in events)
    log.info("事件数: %d (train=%d val=%d test=%d)", len(events), split_counts["train"], split_counts["val"], split_counts["test"])

    os.makedirs(output_root, exist_ok=True)
    write_events_index(events, os.path.join(output_root, "events_index.json"))

    limit_files = int(cfg.get("limit_files") or 0)
    parquet_files = find_parquet_files(input_root, limit_files)
    log.info("Parquet文件数: %d", len(parquet_files))
    if not parquet_files:
        return

    done: set[str] = set()
    if cfg.get("resume"):
        done = load_checkpoint(checkpoint_file)
        log.info("已完成: %d", len(done))

    pending = []
    for p in parquet_files:
        rel = normalize_rel_path(p, input_root)
        if rel in done:
            continue
        pending.append(p)
    log.info("待处理: %d", len(pending))
    if not pending:
        return

    lock = threading.Lock()
    processed = [0]
    ok_files = [0]
    err_files = [0]
    out_files_total = [0]
    out_rows_total = [0]
    stats_total = Counter()

    with ThreadPoolExecutor(max_workers=int(cfg["max_workers"])) as exe:
        futures = {exe.submit(process_parquet_file, p, events, cfg, input_root, output_root, log): p for p in pending}
        for future in as_completed(futures):
            p = futures[future]
            rel = normalize_rel_path(p, input_root)
            try:
                rel_done, ok, out_files, out_rows, stats = future.result()
            except Exception as e:
                rel_done, ok, out_files, out_rows, stats = rel, False, 0, 0, Counter({"exception": 1})
                log.error("处理异常 %s: %s", rel, e)
            with lock:
                processed[0] += 1
                stats_total.update(stats)
                if ok:
                    ok_files[0] += 1
                    out_files_total[0] += int(out_files)
                    out_rows_total[0] += int(out_rows)
                    if int(out_files) > 0:
                        log.info("完成 %s 输出文件 %d 输出行 %d", rel_done, int(out_files), int(out_rows))
                    if cfg.get("resume"):
                        done.add(rel_done)
                        append_checkpoint(checkpoint_file, rel_done)
                else:
                    err_files[0] += 1
                if processed[0] % int(cfg["log_every"]) == 0 or processed[0] == len(pending):
                    log.info(
                        "进度 %d/%d 输出文件 %d 输出行 %d 成功 %d 失败 %d",
                        processed[0],
                        len(pending),
                        out_files_total[0],
                        out_rows_total[0],
                        ok_files[0],
                        err_files[0],
                    )

    log.info(
        "完成 处理=%d 成功=%d 失败=%d 输出文件=%d 输出行=%d 统计=%s",
        len(pending),
        ok_files[0],
        err_files[0],
        out_files_total[0],
        out_rows_total[0],
        dict(stats_total),
    )


if __name__ == "__main__":
    main()
