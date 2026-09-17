#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
原始语料下载 (fetch_raw.py)

六个科室 CSV 共约 370MB，不入 Git。本脚本按 data/raw/MANIFEST.json 记录的 URL
重新下载并校验 sha256——校验不通过直接报错，避免半截文件悄悄进入数据流水线。

首次下载（MANIFEST 尚不存在时）用内置的 URL 表，并在下载后生成 MANIFEST。

用法：
    python experiments/lora_triage_20260917/fetch_raw.py
    python experiments/lora_triage_20260917/fetch_raw.py --force   # 重下已存在的文件
"""

import argparse
import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW_DIR = HERE / "data" / "raw"
MANIFEST = RAW_DIR / "MANIFEST.json"

BASE = "https://raw.githubusercontent.com/Toyhom/Chinese-medical-dialogue-data/master/"
SOURCE_PATHS = {
    "内科": "Data_数据/IM_内科/内科5000-33000.csv",
    "外科": "Data_数据/Surgical_外科/外科5-14000.csv",
    "妇产科": "Data_数据/OAGD_妇产科/妇产科6-28000.csv",
    "儿科": "Data_数据/Pediatric_儿科/儿科5-14000.csv",
    "肿瘤科": "Data_数据/Oncology_肿瘤科/肿瘤科5-10000.csv",
    "男科": "Data_数据/Andriatria_男科/男科5-13000.csv",
}


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="下载并校验科室分诊原始语料")
    parser.add_argument("--force", action="store_true", help="已存在的文件也重新下载")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    expected = {}
    if MANIFEST.exists():
        expected = {entry["name"]: entry["sha256"]
                    for entry in json.loads(MANIFEST.read_text(encoding="utf-8"))["files"]}

    entries = []
    failures = []
    for label, rel in SOURCE_PATHS.items():
        name = rel.split("/")[-1]
        out = RAW_DIR / name
        url = BASE + urllib.parse.quote(rel)
        if args.force or not out.exists():
            print(f"下载 {label} ...", flush=True)
            urllib.request.urlretrieve(url, out)
        digest = sha256_file(out)
        want = expected.get(name)
        if want and digest != want:
            failures.append(f"{name}: 期望 {want[:16]}…，实际 {digest[:16]}…")
        status = "✅" if (want is None or digest == want) else "❌"
        print(f"{status} {label:4s} {out.stat().st_size / 1e6:7.1f} MB  {digest[:16]}  {name}")
        entries.append({"label": label, "name": name, "bytes": out.stat().st_size,
                        "sha256": digest, "url": url})

    if failures:
        raise SystemExit("sha256 校验失败：\n  " + "\n  ".join(failures))

    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps({
            "description": "中文医疗问答语料的冻结依赖记录。六个 CSV 共约 370MB，均不进 Git。",
            "repository": "https://github.com/Toyhom/Chinese-medical-dialogue-data",
            "encoding": "GB18030",
            "files": entries,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已生成 {MANIFEST}")
    print("RAW_DATA_READY")


if __name__ == "__main__":
    main()
