#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""H2025 Clean Restart —— 公共库。

边界纪律（fail-closed）
    * 本项目一切**写入**必须落在 D:\\H2025_clean_restart 之内。
    * 旧工程（D:\\牛脸识别, D:\\cattle_reid_isolated_20260915, D:\\Cow_project）与 C 盘只读。
    * 读取只允许：原始数据、公开预训练权重、旧说明/审计报告。

精度纪律
    * 正式实验必须 FP32：tf32_matmul / tf32_cudnn 全部显式关闭。
"""
import hashlib
import json
import os
import random
import re
import sys
from datetime import datetime
from pathlib import Path

# --------------------------------------------------------------------------- #
# 路径
# --------------------------------------------------------------------------- #
ROOT = Path(r"D:\H2025_clean_restart")
DATA_ROOT = Path(r"D:\牛脸识别\data\Holstein2025_original")
REF_ROOT = Path(r"D:\复现论文")
LEGACY_ARCHIVE = Path(r"D:\H2025_legacy_archive")

_SUBDIRS = [
    "configs", "data_audit", "manifests", "splits", "models", "models/global",
    "models/local", "codebooks", "features", "scores", "statistics",
    "visualizations", "reports", "logs", "scripts", "environment", "locks",
]


def _norm(p):
    return os.path.normcase(os.path.abspath(os.fspath(p)))


_PROJECT = _norm(ROOT)
_FORBID = [
    _norm("C:/"),
    _norm("D:/牛脸识别"),
    _norm("D:/cattle_reid_isolated_20260915"),
    _norm("D:/Cow_project"),
    _norm("D:/H2025_legacy_archive"),
    _norm("D:/H2025_confirmatory_reid"),
]


def _under(child, parent):
    c, p = _norm(child), _norm(parent)
    if c == p:
        return True
    sep = os.sep
    return c.startswith(p if p.endswith(sep) else p + sep)


class PathViolation(RuntimeError):
    pass


def guard_write(path, purpose=""):
    rp = Path(path)
    if not rp.is_absolute():
        rp = ROOT / rp
    for f in _FORBID:
        if _under(rp, f):
            raise PathViolation("拒绝写入禁写前缀：%s（命中 %s；用途 %s）" % (rp, f, purpose or "-"))
    if not _under(rp, _PROJECT):
        raise PathViolation("拒绝写入项目外路径：%s（允许根 %s）" % (rp, ROOT))
    return rp


def ensure_dir(path, purpose=""):
    p = guard_write(path, purpose)
    p.mkdir(parents=True, exist_ok=True)
    return p


def sub(name):
    """取一个已声明的可写子目录。"""
    if name not in _SUBDIRS:
        raise PathViolation("未声明的子目录：%s" % name)
    return ensure_dir(ROOT / name)


# --------------------------------------------------------------------------- #
# 精度 / 复现
# --------------------------------------------------------------------------- #
def disable_tf32():
    """正式实验统一 FP32。返回实测状态字典。"""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    return {
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "float32_matmul_precision": getattr(torch, "get_float32_matmul_precision", lambda: "n/a")(),
    }


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def env_snapshot():
    import platform
    d = {"python": sys.version.split()[0], "executable": sys.executable,
         "platform": platform.platform(), "generated_at": now_iso()}
    try:
        import torch
        d.update({"torch": torch.__version__, "torch_cuda": torch.version.cuda,
                  "cudnn": torch.backends.cudnn.version(),
                  "cuda_available": torch.cuda.is_available(),
                  "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                  "gpu_mem_bytes": (torch.cuda.get_device_properties(0).total_memory
                                    if torch.cuda.is_available() else None)})
    except Exception as e:
        d["torch_error"] = str(e)[:200]
    try:
        import numpy, cv2, PIL
        d.update({"numpy": numpy.__version__, "opencv": cv2.__version__, "pillow": PIL.__version__})
    except Exception as e:
        d["lib_error"] = str(e)[:200]
    return d


# --------------------------------------------------------------------------- #
# IO / 哈希
# --------------------------------------------------------------------------- #
def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def sha256_file(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def json_dump(obj, rel_or_abs, indent=1):
    p = ensure_dir(Path(rel_or_abs).parent) / Path(rel_or_abs).name
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=indent), encoding="utf-8")
    return p


def json_load(p, default=None):
    p = Path(p)
    if not p.is_absolute():
        p = ROOT / p
    if not p.is_file():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# H2025 文件名解析
# --------------------------------------------------------------------------- #
RE_AUTO = re.compile(r"^(?P<start>20\d{12})_(?P<end>20\d{12})_(?P<frame>\d{4,6})_cls(?P<cls>\d+)$")
RE_AUTO_OBJ = re.compile(
    r"^(?P<start>20\d{12})_(?P<end>20\d{12})_(?P<mid1>\d{8})_(?P<mid2>\d{8})_"
    r"(?P<frame>\d+)_obj(?P<obj>\d+)_cls(?P<cls>\d+)$")
RE_MANUAL = re.compile(
    r"^(?P<tag>[A-Za-z]+\d+)_(?P<capdate>\d{4}\.\d{2}\.\d{2})(?P<hms>\d{2}_\d{2}_\d{2})_manual_"
    r"(?P<procts>\d{8})_(?P<procms>\d{8})_(?P<frame>\d+)_obj(?P<obj>\d+)_cls(?P<cls>\d+)$")
RE_CURATED = re.compile(r"^(?P<frame>\d{4,6})_(?P<role>g|q)(?P<idx>\d+)$")
RE_HAND = re.compile(r"^(?P<m>\d{2})(?P<d>\d{2})-(?P<a>\d+)-(?P<b>\d+)-(?P<c>\d+)\s*\((?P<idx>\d+)\)$")
# 示例：328-1-11-1(1).jpg / 0414-1-5-7-1.jpg / 403-1-9-4.jpg / 417-1-11(1).jpg / 330-1-6-8_(1).jpg
RE_MMDD = re.compile(r"^(?P<prefix>\d{1,4})(?P<rest>(?:[-_]\d+)*(?:[-_ ]?\(\d+\))?)$")

ASSUMED_YEAR = 2025  # 数据集时间戳全部落在 2025 年（见 H2025_B3_METADATA_PROVENANCE.md）


def _mmdd_to_date(prefix):
    """把 MMDD 前缀规范化为 2025-MM-DD。返回 (iso_date, normalized_prefix) 或 (None, None)。

    '328'->'0328'->03-28 ; '414'->'0414'->04-14 ; '0414'->04-14 ;
    '001' 等非法月份一律判为未知（bare sequence number）。
    """
    p = prefix
    if len(p) == 3:
        p = "0" + p
    if len(p) != 4:
        return None, None
    mm, dd = int(p[:2]), int(p[2:])
    if not (1 <= mm <= 12 and 1 <= dd <= 31):
        return None, None
    return "%d-%02d-%02d" % (ASSUMED_YEAR, mm, dd), p


def parse_stem(stem):
    """解析文件名主干 → 结构化的命名/会话信息。

    重要：所有时间/会话/耳号字段一律为 possible_*，来自文件名推断，
    未经上游验证，禁止写成 verified_session / verified_day / verified_recording_window。
    """
    base = {"naming": "unknown", "possible_clip": None, "possible_frame": None,
            "possible_timestamp": None, "possible_session": None,
            "possible_date": None, "possible_date_source": None,
            "animal_tag": None, "object_index": None,
            "capture_date": None, "capture_timestamp": None,
            "processing_date": None, "processing_timestamp": None,
            "year_assumed": ASSUMED_YEAR}

    m = RE_AUTO.match(stem)
    if m:
        s = m.group("start")
        iso_d = "%s-%s-%s" % (s[:4], s[4:6], s[6:8])
        base.update({"naming": "auto_timestamp",
                     "possible_clip": "%s_%s" % (s, m.group("end")),
                     "possible_frame": int(m.group("frame")),
                     "possible_timestamp": s,
                     "possible_session": "%s_%s" % (s, m.group("end")),
                     "possible_date": iso_d,
                     "possible_date_source": "filename_clip_start_timestamp14digit",
                     # 该族的时间戳即录制起始时刻，故 capture == possible
                     "capture_date": iso_d,
                     "capture_timestamp": timestamp_to_iso(s),
                     "cls_suffix": int(m.group("cls"))})
        return base

    m = RE_AUTO_OBJ.match(stem)
    if m:
        s = m.group("start")
        iso_d = "%s-%s-%s" % (s[:4], s[4:6], s[6:8])
        base.update({"naming": "auto_timestamp_with_obj",
                     "possible_clip": "%s_%s" % (s, m.group("end")),
                     "possible_frame": int(m.group("frame")),
                     "possible_timestamp": s,
                     "possible_session": "%s_%s" % (s, m.group("end")),
                     "possible_date": iso_d,
                     "possible_date_source": "filename_clip_start_timestamp14digit",
                     "capture_date": iso_d,
                     "capture_timestamp": timestamp_to_iso(s),
                     "object_index": int(m.group("obj")),
                     "cls_suffix": int(m.group("cls")),
                     "mid_fields": [m.group("mid1"), m.group("mid2")]})
        return base

    m = RE_MANUAL.match(stem)
    if m:
        # 关键修正（H2025 B3）：该族文件名里同时含
        #   (a) 真正的拍摄时刻  capdate + hms   —— 例如 2025.03.2623_12_18 = 2025-03-26 23:12:18
        #   (b) 人工处理时刻    procts + procms —— 例如 20250416_20271347
        # 二者相差 20~21 天。旧版本把 possible_date 取成 (b)，导致 holdout 的“跨天”属性失真。
        # 现在：possible_date / capture_* 一律取 (a)；处理时刻单独保留在 processing_*。
        cap_d = m.group("capdate").replace(".", "-")            # 2025-03-26
        hh, mm_, ss = m.group("hms").split("_")                 # 23_12_18
        hms = "%s:%s:%s" % (hh, mm_, ss)
        cap_ts = "%sT%s" % (cap_d, hms)
        pts, pms = m.group("procts"), m.group("procms")
        proc_d = "%s-%s-%s" % (pts[:4], pts[4:6], pts[6:8])
        proc_ts = "%s-%s-%sT%s:%s:%s" % (pts[:4], pts[4:6], pts[6:8],
                                         pms[0:2], pms[2:4], pms[4:6])
        base.update({"naming": "animal_tag_manual",
                     "animal_tag": m.group("tag"),
                     "possible_clip": None,
                     "possible_frame": int(m.group("frame")),
                     "possible_timestamp": None,
                     "possible_session": cap_d.replace("-", "") + hms.replace(":", ""),
                     "possible_date": cap_d,
                     "possible_date_source": "filename_capture_datetime",
                     "capture_date": cap_d,
                     "capture_timestamp": cap_ts,
                     "processing_date": proc_d,
                     "processing_timestamp": proc_ts,
                     "capture_date_field": cap_d,
                     "object_index": int(m.group("obj")),
                     "cls_suffix": int(m.group("cls"))})
        return base

    m = RE_CURATED.match(stem)
    if m:
        base.update({"naming": "curated_frame",
                     "possible_frame": int(m.group("frame")),
                     "possible_session": m.group("frame"),
                     "possible_date_source": None,
                     "role_tag": m.group("role"),
                     "curated_index": int(m.group("idx"))})
        return base

    m = RE_HAND.match(stem)
    if m:
        iso_h, norm_h = _mmdd_to_date(m.group("m") + m.group("d"))
        base.update({"naming": "hand_named",
                     "possible_session": "%s-%s-%s-%s" % (m.group("m"), m.group("d"),
                                                          m.group("a"), m.group("b")),
                     "possible_date": iso_h,
                     "possible_date_source": "filename_mmdd_prefix" if iso_h else None,
                     "capture_date": iso_h,
                     "hand_index": int(m.group("idx"))})
        return base

    m = RE_MMDD.match(stem)
    if m:
        iso, norm = _mmdd_to_date(m.group("prefix"))
        base.update({"naming": "mmdd_prefix_index" if iso else "bare_sequence",
                     "possible_session": ("%s%s" % (norm, m.group("rest"))) if norm else None,
                     "possible_date": iso,
                     "possible_date_source": "filename_mmdd_prefix" if iso else None,
                     "capture_date": iso,
                     "mmdd_prefix_raw": m.group("prefix"),
                     "index_suffix": m.group("rest") or None})
        return base

    return base


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def iter_images(source):
    """枚举某个 source 下的全部图像 → (abs_path, rel_path, identity_label)"""
    base = DATA_ROOT / source
    for id_dir in sorted([d for d in base.iterdir() if d.is_dir()]):
        for f in sorted(id_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                yield f, "%s/%s/%s" % (source, id_dir.name, f.name), id_dir.name


def timestamp_to_iso(ts14):
    if not ts14 or len(ts14) != 14:
        return None
    return "%s-%s-%sT%s:%s:%s" % (ts14[0:4], ts14[4:6], ts14[6:8],
                                  ts14[8:10], ts14[10:12], ts14[12:14])


def print_env_banner():
    snap = env_snapshot()
    print("[env] python=%s torch=%s cuda=%s gpu=%s"
          % (snap.get("python"), snap.get("torch"), snap.get("torch_cuda"), snap.get("gpu")), flush=True)
    return snap
