"""
=============================================================================
PRODUCT:    Murakami PDF AI-Optimizer v7.20.0 (Hybrid GS + PikePDF Deep-Structural)
AUTHOR:     Hideyoshi_Murakami
DATE:       2024-2026

HOTFIX NOTE (UIは変更しない):
  - 「そもそも縮まない」主因だった PdfImage 周りは維持しつつ、
    追加で “より縮む/より壊さない” 方向に内部ロジックを強化（UIは完全据え置き）。
  - v7.20 系の UI 表示文字列・配置・項目は一切変更しない。

SECURITY/RELEASE NOTE (公開向け):
  - 免責事項(Disclaimer)を自動生成し、フッター内に折りたたみで表示（UI最小変更）
  - 公開時の情報漏えい対策：例外の詳細表示は環境変数で制御（NEO_DEBUG=1 のみ詳細）
  - DoS対策：アップロードサイズ上限（NEO_MAX_UPLOAD_MB, default=25MB）
  - キャッシュのユーザー混線対策：セッションIDを導入し、キャッシュキーに含める
    + TTL & 上限数でメモリ残留を抑制

  - Ghostscript は本ビルドでは **強制無効化** しています（制約環境前提 / GS-Emulation）。
    ※ GS(pdfwrite) を一切呼ばない設計なので、Render等でGSが使えない/入れられない環境でも問題ありません。
    ※ 代わりに pikepdf + Pillow のみで “壊さず縮む” を最大化します。
=============================================================================
"""

import streamlit as st
import io
import math
import random
import numpy as np
from PIL import Image, ImageEnhance
import pikepdf
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

import os
import re
import hashlib
import threading
import zlib
import shutil
import time
import uuid
from collections import OrderedDict

# ==============================
# Runtime profile (FREE / STARTER)
# ==============================
# UIは変えずに、Render等のリソース制約に合わせて内部上限を調整できるようにする。
# - env: PDF_COMP_PROFILE=FREE (default) / STARTER
# - 既に個別envが設定されている場合はそちらが優先される（setdefault）。
try:
    import profiles as _profiles
    _ACTIVE_PROFILE = _profiles.apply_profile_from_env()
except Exception:
    _ACTIVE_PROFILE = str(os.environ.get("PDF_COMP_PROFILE", "FREE")).strip().upper() or "FREE"


# --- スレッド警告対策：Python 3.13 / 最新Streamlit対応 ---
try:
    from streamlit.runtime.scriptrunner import add_script_run_context
except ImportError:
    try:
        from streamlit.runtime.scriptrunner.script_run_context import add_script_run_context
    except ImportError:
        def add_script_run_context(thread=None):
            pass


# ==============================
# Public/Release configs
# ==============================
def _env_flag(name: str, default: str = "0") -> bool:
    return str(os.environ.get(name, default)).strip().lower() in ("1", "true", "yes", "on")

def _env_int(name: str, default: int, min_v: int | None = None, max_v: int | None = None) -> int:
    """Read int env var with optional clamp."""
    try:
        v = int(str(os.environ.get(name, str(default))).strip())
    except Exception:
        v = int(default)

    try:
        if min_v is not None:
            v = max(int(min_v), v)
        if max_v is not None:
            v = min(int(max_v), v)
    except Exception:
        pass
    return int(v)

SHOW_DEBUG = _env_flag("NEO_DEBUG", "0")  # 1=st.exception を出す

# ★重要：GS-Emulation 強制（GSコマンドは一切実行しない）
FORCE_GS_EMULATION = True

# NOTE: 互換のため env 変数は読むが、FORCE_GS_EMULATION=True の場合は最終的に必ず無効化する
_ENABLE_GS_ENV = (_env_int("NEO_ENABLE_GS", 1, 0, 1) == 1)
ENABLE_GS = (False if FORCE_GS_EMULATION else bool(_ENABLE_GS_ENV))

NEO_REPO_URL = str(os.environ.get("NEO_REPO_URL", "")).strip()  # optional: shown in footer/disclaimer
MAX_UPLOAD_MB = _env_int("NEO_MAX_UPLOAD_MB", 25)
MAX_UPLOAD_BYTES = int(MAX_UPLOAD_MB) * 1024 * 1024

# GSタイムアウトは残す（UI/設定互換のため）が、FORCE_GS_EMULATION=True の場合は未使用
GS_TIMEOUT_SEC = _env_int("NEO_GS_TIMEOUT", 60)  # 公開時は短め推奨

CACHE_TTL_SECONDS = _env_int("NEO_CACHE_TTL_SECONDS", 20 * 60)  # 20分
CACHE_MAX_ENTRIES_PER_SESSION = _env_int("NEO_CACHE_MAX_ENTRIES", 24)
MAX_PDF_PAGES = _env_int("NEO_MAX_PDF_PAGES", 120)  # ページ数上限（DoS/メモリ対策）
MAX_ITERATIONS_HARD = _env_int("NEO_MAX_ITERATIONS", 80)  # UI値を内部でクランプ
MAX_THREADS_HARD = _env_int("NEO_MAX_THREADS", 6)  # UI値を内部でクランプ
MAX_RUNTIME_SECONDS = _env_int("NEO_MAX_RUNTIME_SECONDS", 120)  # 全体の時間上限（秒）


# ==============================
# Public hardening: Concurrency + Rate limit
# ==============================
MAX_CONCURRENT_JOBS = _env_int("NEO_MAX_CONCURRENT", 1)  # 同時実行数（無料枠向けは1推奨）
RATE_LIMIT_PER_MIN = _env_int("NEO_RATE_LIMIT_PER_MIN", 10)  # 1分あたり開始回数（クライアント単位）
RATE_WINDOW_SEC = 60

# グローバル制御（プロセス単位）
_JOB_SEMAPHORE = threading.BoundedSemaphore(max(1, int(MAX_CONCURRENT_JOBS)))
_RATE_LOCK = threading.Lock()
_RATE_BUCKET = OrderedDict()  # client_id -> [timestamps...]


def _get_session_id() -> str:
    """
    Streamlitセッション単位のID
    - st.session_state はメインスレッドで確実に触れるため、ここで生成
    - ワーカースレッドには session_id を引数で渡す
    """
    try:
        sid = st.session_state.get("_NEO_SESSION_ID", None)
        if not sid:
            sid = uuid.uuid4().hex
            st.session_state["_NEO_SESSION_ID"] = sid
        return str(sid)
    except Exception:
        return uuid.uuid4().hex


def _get_client_id() -> str:
    """可能な範囲でクライアントIDを推定（X-Forwarded-For 等）。取れない場合はセッションIDで代替。"""
    try:
        try:
            hdrs = getattr(getattr(st, "context", None), "headers", None)
        except Exception:
            hdrs = None

        if hdrs:
            xff = hdrs.get("x-forwarded-for") or hdrs.get("X-Forwarded-For")
            if xff:
                ip = str(xff).split(",")[0].strip()
                if ip:
                    return f"ip:{ip}"
            rip = hdrs.get("x-real-ip") or hdrs.get("X-Real-Ip") or hdrs.get("X-Real-IP")
            if rip:
                ip = str(rip).strip()
                if ip:
                    return f"ip:{ip}"
        return f"sid:{_get_session_id()}"
    except Exception:
        return f"sid:{_get_session_id()}"


def _prune_rate_bucket(now: float):
    try:
        keys = list(_RATE_BUCKET.keys())
        for k in keys:
            ts = _RATE_BUCKET.get(k, [])
            ts = [t for t in ts if now - t <= RATE_WINDOW_SEC]
            if ts:
                _RATE_BUCKET[k] = ts
            else:
                try:
                    del _RATE_BUCKET[k]
                except Exception:
                    pass
        while len(_RATE_BUCKET) > 2048:
            try:
                _RATE_BUCKET.popitem(last=False)
            except Exception:
                break
    except Exception:
        pass


def _rate_limit_check_and_mark(client_id: str) -> tuple[bool, int]:
    """(ok, retry_after_sec)"""
    if RATE_LIMIT_PER_MIN <= 0:
        return (True, 0)
    now = time.time()
    with _RATE_LOCK:
        _prune_rate_bucket(now)
        ts = _RATE_BUCKET.get(client_id, [])
        ts = [t for t in ts if now - t <= RATE_WINDOW_SEC]
        if len(ts) >= int(RATE_LIMIT_PER_MIN):
            oldest = min(ts) if ts else now
            retry = int(max(1, RATE_WINDOW_SEC - (now - oldest)))
            _RATE_BUCKET[client_id] = ts
            return (False, retry)
        ts.append(now)
        _RATE_BUCKET[client_id] = ts
        return (True, 0)


# ==============================
# Disclaimer (Auto-Generated)
# ==============================
def _generate_disclaimer(product: str, version: str, author: str) -> str:
    repo = NEO_REPO_URL if NEO_REPO_URL else "(set NEO_REPO_URL env to show repo link)"
    gs_enabled = "YES" if ENABLE_GS else "NO"

    def _gs_detected_simple() -> bool:
        for _n in ("gs", "gswin64c", "gswin32c"):
            try:
                if shutil.which(_n):
                    return True
            except Exception:
                pass
        return False

    gs_detected = "YES" if _gs_detected_simple() else "NO"

    return f"""
<details>
<summary><b>免責事項（Disclaimer）</b></summary>

### 免責事項（日本語）
- 本ソフトウェア「{product} {version}」（以下「本ソフト」）は **現状有姿 (AS IS)** で提供されます。  
- 作者（{author}）および関係者は、本ソフトの使用または使用不能から生じる **いかなる損害（直接損害・間接損害・逸失利益・データ損失・業務中断等）についても責任を負いません**。  
- 本ソフトはPDFを再圧縮・再構成しますが、**出力PDFの完全性・互換性・可読性・真正性を保証しません**。必ず事前にバックアップを取り、重要書類（履歴書/契約書/公的書類等）は最終確認のうえ自己責任でご利用ください。  
- アップロードされたPDF/画像には機微情報が含まれる場合があります。**機密性が要求される文書のアップロードは避ける**か、自己の管理下（ローカル環境等）で実行してください。  
- 本ソフトは外部ライブラリ・外部実行ファイル（例：Ghostscript）を利用する場合があります。**第三者コンポーネント由来の不具合/脆弱性/挙動**について作者は責任を負いません。  
- 法務・医療・金融等の助言を提供するものではありません。

### OSS / Third-party licenses
- Source: {repo}
- Ghostscript usage: enabled={gs_enabled}, detected={gs_detected}
  Ghostscriptは **AGPL v3** または商用ライセンスのデュアルライセンスです。Ghostscriptを同梱/配布する場合は、そのライセンス条件に従ってください。  
  ※本アプリは「GS-Emulation」ビルドのため、GSコマンドを実行しません（検出のみ行う場合があります）。

### Disclaimer (English)
- This software "{product} {version}" is provided **AS IS**, without warranty of any kind.  
- The author ({author}) shall not be liable for any damages arising from the use or inability to use this software.  
- Output PDF integrity/compatibility is **not guaranteed**. Always keep backups and verify outputs.  
- Avoid uploading sensitive documents to a public instance; run locally if confidentiality is required.

</details>
""".strip()


_PRODUCT_NAME = "NEO PDF AI-Optimizer"
_VERSION = "v7.20.0"
_AUTHOR = "Hideyoshi_Murakami"
DISCLAIMER_HTML = _generate_disclaimer(_PRODUCT_NAME, _VERSION, _AUTHOR)


# ==============================
# Global cache (thread-safe, session-isolated)
# ==============================
# key: (session_id, orig_hash, state_key_tuple)
# val: (timestamp, res_bytes, preview_img, sample_imgs)
_COMPRESS_CACHE = OrderedDict()
_CACHE_LOCK = threading.Lock()

_GLOBAL_ORIG_HASH = None

_ORIG_HAS_COLOR = False
_DOC_PROFILE = "mixed"
_DOC_IMAGE_WEIGHT = 0.5


def _cache_prune_locked(session_id: str):
    now = time.time()

    kill_keys = []
    for k, v in list(_COMPRESS_CACHE.items()):
        try:
            ts = float(v[0])
        except Exception:
            ts = 0.0
        if (now - ts) > float(CACHE_TTL_SECONDS):
            kill_keys.append(k)
    for k in kill_keys:
        try:
            del _COMPRESS_CACHE[k]
        except Exception:
            pass

    keys_for_sid = [k for k in _COMPRESS_CACHE.keys() if len(k) >= 1 and k[0] == session_id]
    excess = len(keys_for_sid) - int(CACHE_MAX_ENTRIES_PER_SESSION)
    if excess > 0:
        for k in keys_for_sid[:excess]:
            try:
                del _COMPRESS_CACHE[k]
            except Exception:
                pass


# ==============================
# GS detect helpers (NO-EXEC)
# ==============================
def _find_gs_exe() -> str | None:
    if FORCE_GS_EMULATION:
        return None
    if not ENABLE_GS:
        return None

    for name in ("gs", "gswin64c", "gswin32c"):
        p = shutil.which(name)
        if p:
            return p
    return None


def _gs_available() -> bool:
    return _find_gs_exe() is not None


def _state_key(state) -> tuple:
    return (
        state.quality,
        state.scale,
        state.obj_mode,
        state.compress_streams,
        state.linearize,
        state.recompress_images,
        state.engine,
        state.jpeg_subsampling,
        state.jpeg_progressive,
        state.img_color_mode,
        state.dedupe_images,
        state.pike_encode_mode,

        state.gs_preset,
        state.gs_downsample,
        state.gs_image_res,
        state.gs_color_strategy,
        state.gs_convert_cmyk_to_rgb,

        state.gs_passthrough_jpeg,
        state.gs_jpegq,
        state.gs_autofilter_color,
        state.gs_force_dctencode,

        state.gs_color_res,
        state.gs_gray_res,
        state.gs_mono_res,
        state.gs_downsample_threshold,
        state.gs_downsample_type,
        state.gs_enable_mono,

        bool(_gs_available()),
        bool(ENABLE_GS),
        bool(FORCE_GS_EMULATION)
    )


# ==============================
# 0. Utility (Image Metrics)
# ==============================
def _convert_to_rgb_safely(pil_img: Image.Image):
    try:
        if pil_img.mode in ("RGBA", "P"):
            bg = Image.new("RGB", pil_img.size, (255, 255, 255))
            if pil_img.mode == "RGBA":
                bg.paste(pil_img, mask=pil_img.split()[3])
            else:
                bg.paste(pil_img.convert("RGB"))
            return bg
        return pil_img.convert("RGB")
    except Exception:
        try:
            return pil_img.convert("RGB")
        except Exception:
            return None


def _mean_saturation(pil_img: Image.Image) -> float:
    try:
        hsv = pil_img.convert("HSV")
        arr = np.array(hsv).astype(np.float32)
        s = arr[..., 1]
        return float(np.mean(s)) / 255.0
    except Exception:
        return 0.0


def _color_penalty(orig_rgb: Image.Image, after_rgb: Image.Image) -> float:
    try:
        s0 = _mean_saturation(orig_rgb)
        s1 = _mean_saturation(after_rgb)
    except Exception:
        return 0.8

    if s0 < 0.08:
        return 1.0

    if s1 < 0.06:
        return 0.05

    drop = s1 / max(1e-6, s0)
    if drop < 0.35:
        return 0.2
    if drop < 0.6:
        return 0.6
    return 1.0


def _sharpness_energy(pil_rgb: Image.Image) -> float:
    try:
        img = pil_rgb.resize((256, 256), Image.BILINEAR)
        arr = np.array(img).astype(np.float32)
        gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
        gx = np.abs(np.diff(gray, axis=1))
        gy = np.abs(np.diff(gray, axis=0))
        e = float(np.mean(gx)) + float(np.mean(gy))
        return max(1e-6, e)
    except Exception:
        return 1e-6


def _zoom_robustness_coeff(orig_rgb: Image.Image, after_rgb: Image.Image) -> float:
    try:
        ow, oh = orig_rgb.size
        aw, ah = after_rgb.size
        if ow <= 0 or oh <= 0 or aw <= 0 or ah <= 0:
            return 0.75

        dim_ratio = min(aw / ow, ah / oh)

        if _DOC_PROFILE == "image":
            dim_base = 0.98
            sharp_base = 0.85
        elif _DOC_PROFILE == "mixed":
            dim_base = 0.94
            sharp_base = 0.78
        else:
            dim_base = 0.55
            sharp_base = 0.50

        dim_k = max(0.0, min(1.0, dim_ratio / dim_base))

        s0 = _sharpness_energy(orig_rgb)
        s1 = _sharpness_energy(after_rgb)
        sharp_ratio = s1 / max(1e-6, s0)
        sharp_k = max(0.0, min(1.0, sharp_ratio / sharp_base))

        base = (220, 220)
        o2 = orig_rgb.resize(base, Image.BILINEAR).resize((base[0] * 2, base[1] * 2), Image.BICUBIC)
        a2 = after_rgb.resize(base, Image.BILINEAR).resize((base[0] * 2, base[1] * 2), Image.BICUBIC)

        ao = np.array(o2).astype(np.float32)
        aa = np.array(a2).astype(np.float32)
        mse = float(np.mean((ao - aa) ** 2))
        mse_k = max(0.0, min(1.0, 1.0 - (mse / 65025.0)))

        coeff = (0.45 * dim_k) + (0.35 * min(sharp_k, mse_k)) + (0.20 * sharp_k)
        return float(max(0.0, min(1.0, coeff)))
    except Exception:
        return 0.75


# ==============================
# 0.5 Eerie Contrast Guard (rare cases)
# ==============================
def _luma_stats_rgb(pil_rgb: Image.Image) -> tuple[float, float, float]:
    """
    return (p05, p95, std) on luma in [0..255]
    """
    try:
        img = pil_rgb.resize((160, 160), Image.BILINEAR)
        arr = np.array(img).astype(np.float32)
        y = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
        p05 = float(np.percentile(y, 5))
        p95 = float(np.percentile(y, 95))
        std = float(np.std(y))
        return (p05, p95, std)
    except Exception:
        return (0.0, 255.0, 64.0)


def _soft_tone_guard(pil_rgb: Image.Image) -> Image.Image:
    """
    「不気味」抑制のためのごく弱いトーン調整（やり過ぎない）
    """
    try:
        img = pil_rgb
        img = ImageEnhance.Contrast(img).enhance(0.965)
        img = ImageEnhance.Color(img).enhance(0.985)
        return img
    except Exception:
        return pil_rgb


def _guard_eerie_contrast(orig_rgb: Image.Image, pil_rgb: Image.Image, q: int, subsampling: int) -> tuple[Image.Image, int, int]:
    """
    軽量プレビューで「暗部潰れ/コントラスト過多」になりそうなケースのみ保守
    - 画像の一部が不気味になるのは、低Q+強サブサンプリングで起きがち
    - ここは “まれに” の抑制なので、常時重くならないよう最小限
    """
    try:
        if q >= 70:
            return (pil_rgb, q, subsampling)

        o_p05, o_p95, o_std = _luma_stats_rgb(orig_rgb)
        a_p05, a_p95, a_std = _luma_stats_rgb(pil_rgb)

        o_rng = max(1e-6, (o_p95 - o_p05))
        a_rng = max(1e-6, (a_p95 - a_p05))

        # 元が「自然」寄り（標準偏差そこそこ、レンジそこそこ）
        # なのに、圧縮後がレンジ/STDだけ急に増えたら違和感の候補
        if (a_rng > o_rng * 1.20) and (a_std > o_std * 1.18) and (q <= 60):
            # まずはトーンをほんの少し戻す
            pil_rgb2 = _soft_tone_guard(pil_rgb)
            # それでも不安なら品質を少し上げる（数値は最小限）
            q2 = min(70, max(q, 62))
            ss2 = min(subsampling, 1)  # 2は荒れやすいので 1 へ
            return (pil_rgb2, q2, ss2)

        # 暗部潰れ（p05が持ち上がる/逆に落ちる）でも違和感になるケースがある
        if (abs(a_p05 - o_p05) > 18.0) and (q <= 55):
            pil_rgb2 = _soft_tone_guard(pil_rgb)
            q2 = min(68, max(q, 60))
            ss2 = min(subsampling, 1)
            return (pil_rgb2, q2, ss2)

        return (pil_rgb, q, subsampling)
    except Exception:
        return (pil_rgb, q, subsampling)


# ==============================
# 1. PDF Sampling (multi-page)
# ==============================
def _is_name(obj, name_str: str) -> bool:
    try:
        return str(obj) == name_str
    except Exception:
        return False


def _iter_xobject_images_from_resources(resources, visited: set):
    if resources is None:
        return
    try:
        xobj = resources.get("/XObject", None)
        if not xobj:
            return

        for name, obj in list(xobj.items()):
            try:
                st0 = obj
                try:
                    st0 = st0.get_object()
                except Exception:
                    pass

                oid = None
                try:
                    oid = int(st0.objgen[0]) * 1000000 + int(st0.objgen[1])
                except Exception:
                    oid = id(st0)

                if oid in visited:
                    continue
                visited.add(oid)

                subtype = st0.get("/Subtype", None)
                if _is_name(subtype, "/Image"):
                    yield (xobj, name, st0)
                    continue

                if _is_name(subtype, "/Form"):
                    res2 = st0.get("/Resources", None)
                    if res2:
                        for it in _iter_xobject_images_from_resources(res2, visited):
                            yield it
            except Exception:
                continue
    except Exception:
        return


def _extract_sample_images_as_rgb(pdf_bytes: bytes, max_pages: int = 8, max_total: int = 6):
    picked = []
    try:
        with pikepdf.open(io.BytesIO(pdf_bytes)) as pdf:
            page_count = min(max_pages, len(pdf.pages))
            for pi in range(page_count):
                page = pdf.pages[pi]
                res = page.get("/Resources", None)
                if not res:
                    continue

                candidates = []
                visited = set()
                for _, __, img_stream in _iter_xobject_images_from_resources(res, visited):
                    try:
                        pimg = pikepdf.PdfImage(img_stream)
                        pil = pimg.as_pil_image()
                        pil = _convert_to_rgb_safely(pil)
                        if pil is None:
                            continue
                        w, h = pil.size
                        area = int(w) * int(h)
                        candidates.append((area, pil))
                    except Exception:
                        continue

                if not candidates:
                    continue
                candidates.sort(key=lambda x: x[0], reverse=True)
                picked.append(candidates[0][1])

                if len(picked) >= max_total:
                    break
    except Exception:
        pass

    return picked


# ==============================
# 2. Heuristics: photo/text-like
# ==============================
def _edge_strength(pil_img: Image.Image) -> float:
    try:
        img = pil_img.convert("RGB").resize((160, 160), Image.BILINEAR)
        arr = np.array(img).astype(np.float32)
        gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
        gx = np.abs(np.diff(gray, axis=1))
        gy = np.abs(np.diff(gray, axis=0))
        return float(np.mean(gx) + np.mean(gy))
    except Exception:
        return 0.0


def _is_photo_like(pil_img: Image.Image) -> bool:
    try:
        img = pil_img
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        img = img.resize((128, 128), Image.BILINEAR)
        arr = np.array(img)

        if img.mode == "RGB":
            gray = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]).astype(np.float32)
        else:
            gray = arr.astype(np.float32)

        gx = np.abs(np.diff(gray, axis=1))
        gy = np.abs(np.diff(gray, axis=0))
        edge = float(np.mean(gx)) + float(np.mean(gy))

        if img.mode == "RGB":
            var_rgb = float(np.mean(np.var(arr.astype(np.float32), axis=(0, 1))))
            r = arr[..., 0].astype(np.float32)
            g = arr[..., 1].astype(np.float32)
            b = arr[..., 2].astype(np.float32)
            chdiff = float(np.mean(np.abs(r - g)) + np.mean(np.abs(g - b)) + np.mean(np.abs(b - r))) / 3.0
        else:
            var_rgb = float(np.var(arr.astype(np.float32)))
            chdiff = 0.0

        if edge > 26.0:
            return False

        return (var_rgb > 420.0) or (chdiff > 6.0)
    except Exception:
        return True


def _is_texty_image(pil_img: Image.Image) -> bool:
    try:
        e = _edge_strength(pil_img)
        return e >= 28.0
    except Exception:
        return False


# ==============================
# 3. Encode Helpers (Pillow + “PDF-valid image stream”)
# ==============================
def _encode_image_to_jpeg(pil_img: Image.Image, quality: int, subsampling: int, progressive: bool) -> bytes | None:
    try:
        img = pil_img
        if img.mode != "RGB":
            img = _convert_to_rgb_safely(img)
            if img is None:
                return None

        buf = io.BytesIO()
        save_kwargs = {
            "format": "JPEG",
            "quality": int(quality),
            "optimize": True,
            "progressive": bool(progressive),
        }
        try:
            save_kwargs["subsampling"] = int(subsampling)
        except Exception:
            pass

        img.save(buf, **save_kwargs)
        return buf.getvalue()
    except Exception:
        return None


def _zlib_deflate_best(data: bytes, level: int = 9) -> bytes:
    try:
        if data is None:
            return b""

        if len(data) >= 4_000_000:
            cobj = zlib.compressobj(
                level=int(level),
                method=zlib.DEFLATED,
                wbits=15,
                memLevel=9,
                strategy=zlib.Z_DEFAULT_STRATEGY,
            )
            return cobj.compress(data) + cobj.flush()

        best = None
        best_len = 10**18

        for strategy in (zlib.Z_DEFAULT_STRATEGY, zlib.Z_FILTERED, zlib.Z_RLE):
            try:
                cobj = zlib.compressobj(
                    level=int(level),
                    method=zlib.DEFLATED,
                    wbits=15,
                    memLevel=9,
                    strategy=strategy,
                )
                comp = cobj.compress(data) + cobj.flush()
                if comp is not None and len(comp) < best_len:
                    best = comp
                    best_len = len(comp)
            except Exception:
                continue

        if best is None:
            return zlib.compress(data, int(level))
        return best
    except Exception:
        try:
            return zlib.compress(data, int(level))
        except Exception:
            return b""


def _pack_bits_1bpp(img_l: Image.Image, threshold: int = 128) -> tuple[bytes, int, int]:
    w, h = img_l.size
    arr = np.array(img_l, dtype=np.uint8)
    bits = (arr >= threshold).astype(np.uint8)

    row_bytes = (w + 7) // 8
    out = bytearray(row_bytes * h)

    for y in range(h):
        for x in range(w):
            byte_i = y * row_bytes + (x // 8)
            bit_i = 7 - (x % 8)
            if bits[y, x]:
                out[byte_i] |= (1 << bit_i)

    return bytes(out), w, h


def _make_pdf_image_stream_flate(pdf, pil_img: Image.Image, force_mono_1bit: bool = False) -> pikepdf.Stream | None:
    try:
        if force_mono_1bit:
            img_l = pil_img.convert("L")
            raw_bits, w, h = _pack_bits_1bpp(img_l, threshold=128)
            comp = _zlib_deflate_best(raw_bits, level=9)

            st0 = pikepdf.Stream(pdf, comp)
            st0["/Type"] = pikepdf.Name("/XObject")
            st0["/Subtype"] = pikepdf.Name("/Image")
            st0["/Width"] = int(w)
            st0["/Height"] = int(h)
            st0["/ColorSpace"] = pikepdf.Name("/DeviceGray")
            st0["/BitsPerComponent"] = 1
            st0["/Filter"] = pikepdf.Name("/FlateDecode")
            return st0

        if pil_img.mode == "L":
            img = pil_img
            w, h = img.size
            raw = img.tobytes()
            comp = _zlib_deflate_best(raw, level=9)

            st0 = pikepdf.Stream(pdf, comp)
            st0["/Type"] = pikepdf.Name("/XObject")
            st0["/Subtype"] = pikepdf.Name("/Image")
            st0["/Width"] = int(w)
            st0["/Height"] = int(h)
            st0["/ColorSpace"] = pikepdf.Name("/DeviceGray")
            st0["/BitsPerComponent"] = 8
            st0["/Filter"] = pikepdf.Name("/FlateDecode")
            return st0

        img = _convert_to_rgb_safely(pil_img)
        if img is None:
            return None
        w, h = img.size
        raw = img.tobytes()
        comp = _zlib_deflate_best(raw, level=9)

        st0 = pikepdf.Stream(pdf, comp)
        st0["/Type"] = pikepdf.Name("/XObject")
        st0["/Subtype"] = pikepdf.Name("/Image")
        st0["/Width"] = int(w)
        st0["/Height"] = int(h)
        st0["/ColorSpace"] = pikepdf.Name("/DeviceRGB")
        st0["/BitsPerComponent"] = 8
        st0["/Filter"] = pikepdf.Name("/FlateDecode")
        return st0

    except Exception:
        return None


# ==============================
# 4. "GS-Emulation" preset logic (NO-EXEC GS)
# ==============================
def _preset_to_quality(preset: str) -> int:
    if preset == "screen":
        return 30
    if preset == "ebook":
        return 45
    if preset == "printer":
        return 65
    if preset == "prepress":
        return 82
    return 45


def _apply_gs_preset_defaults(state) -> None:
    p = str(getattr(state, "gs_preset", "ebook"))

    if p == "screen":
        state.gs_color_res = 72
        state.gs_gray_res = 72
        state.gs_mono_res = 300
        state.gs_downsample_type = "subsample"
    elif p == "ebook":
        state.gs_color_res = 150
        state.gs_gray_res = 150
        state.gs_mono_res = 300
        state.gs_downsample_type = "bicubic"
    elif p == "printer":
        state.gs_color_res = 220
        state.gs_gray_res = 220
        state.gs_mono_res = 600
        state.gs_downsample_type = "average"
    elif p == "prepress":
        state.gs_color_res = 300
        state.gs_gray_res = 300
        state.gs_mono_res = 1200
        state.gs_downsample_type = "bicubic"
    else:
        state.gs_color_res = 150
        state.gs_gray_res = 150
        state.gs_mono_res = 300
        state.gs_downsample_type = "bicubic"

    state.gs_downsample_threshold = 1.35


def _merge_gs_emu_into_pike_params(state) -> None:
    try:
        _apply_gs_preset_defaults(state)

        base_q = _preset_to_quality(getattr(state, "gs_preset", "ebook"))
        try:
            jq = int(getattr(state, "gs_jpegq", base_q))
        except Exception:
            jq = base_q

        # 攻める方向：下限を少し下げる
        q = int(max(18, min(92, jq)))
        state.quality = q

        if getattr(state, "gs_force_dctencode", True):
            state.pike_encode_mode = "jpeg"
        else:
            state.pike_encode_mode = "auto"

        state.recompress_images = True

        if state.gs_preset == "screen":
            state.jpeg_subsampling = 2
        elif state.gs_preset == "ebook":
            state.jpeg_subsampling = 1
        else:
            state.jpeg_subsampling = 0

        state.jpeg_progressive = True

        if getattr(state, "gs_color_strategy", "LeaveColorUnchanged") == "Gray":
            state.img_color_mode = "gray"
        else:
            state.img_color_mode = "keep"

        state.compress_streams = True
        state.obj_mode = "gen"
    except Exception:
        pass


def _apply_safety_guards(state):
    global _DOC_IMAGE_WEIGHT

    # 元がカラーなら強制保持（写真白黒化は絶対NG）
    if _ORIG_HAS_COLOR:
        try:
            state.img_color_mode = "keep"
            state.gs_color_strategy = "LeaveColorUnchanged"
        except Exception:
            pass

    # GS実行は禁止
    if getattr(state, "engine", "pike") == "gs":
        _merge_gs_emu_into_pike_params(state)

    # mixed/image は最低限だけ守る（ただし以前よりは攻める）
    if _DOC_PROFILE in ("mixed", "image"):
        try:
            light_mixed = (_DOC_PROFILE == "mixed" and float(_DOC_IMAGE_WEIGHT) < 0.42)

            if not light_mixed:
                if int(state.jpeg_subsampling) == 2:
                    state.jpeg_subsampling = 1

            q_floor = 44 if light_mixed else 48
            if int(state.quality) < q_floor:
                state.quality = q_floor

            res_floor = 130 if light_mixed else 150
            if int(getattr(state, "gs_color_res", 150)) < res_floor:
                state.gs_color_res = res_floor
            if int(getattr(state, "gs_gray_res", 150)) < res_floor:
                state.gs_gray_res = res_floor

            if float(getattr(state, "gs_downsample_threshold", 1.35)) < 1.05:
                state.gs_downsample_threshold = 1.05
        except Exception:
            pass

    # text は「縮ませる」優先（壊れにくさは保持、画質下限は撤廃寄り）
    if _DOC_PROFILE == "text":
        try:
            # 低Q許容（ただし極端は避ける）
            if int(state.quality) < 18:
                state.quality = 18

            # テキスト主体はサブサンプリングも許容
            # ただし不気味化は後段ガードで抑制
            if int(state.jpeg_subsampling) not in (0, 1, 2):
                state.jpeg_subsampling = 2

            # DPI も下げる余地を残す
            if int(getattr(state, "gs_color_res", 150)) < 72:
                state.gs_color_res = 72
            if int(getattr(state, "gs_gray_res", 150)) < 72:
                state.gs_gray_res = 72

            # しきい値は下げ、よりダウンサンプルされやすく
            if float(getattr(state, "gs_downsample_threshold", 1.35)) < 1.00:
                state.gs_downsample_threshold = 1.00
        except Exception:
            pass

    return state


# ==============================
# 4.5 GS Downsample logic helpers (Emulation)
# ==============================
def _page_inches(page) -> tuple[float, float]:
    try:
        mb = page.get("/MediaBox", None)
        if not mb or len(mb) < 4:
            return (8.27, 11.69)
        x0, y0, x1, y1 = [float(v) for v in mb]
        w_pt = max(1.0, abs(x1 - x0))
        h_pt = max(1.0, abs(y1 - y0))
        return (w_pt / 72.0, h_pt / 72.0)
    except Exception:
        return (8.27, 11.69)


def _downsample_resample_filter(gs_downsample_type: str):
    t = str(gs_downsample_type).lower()
    if t == "subsample":
        return Image.NEAREST
    if t == "average":
        return Image.BOX
    return Image.BICUBIC


def _gs_should_downsample(
    w_px: int, h_px: int,
    page_w_in: float, page_h_in: float,
    target_dpi: int,
    threshold: float
) -> tuple[bool, int, int]:
    if w_px <= 0 or h_px <= 0 or page_w_in <= 0 or page_h_in <= 0 or target_dpi <= 0:
        return (False, w_px, h_px)

    max_w = int(page_w_in * float(target_dpi))
    max_h = int(page_h_in * float(target_dpi))
    max_w = max(1, max_w)
    max_h = max(1, max_h)

    if (w_px > max_w * threshold) or (h_px > max_h * threshold):
        s = min(max_w / float(w_px), max_h / float(h_px))
        s = max(0.01, min(1.0, s))
        new_w = max(1, int(w_px * s))
        new_h = max(1, int(h_px * s))
        return (True, new_w, new_h)

    return (False, w_px, h_px)


def _gs_classify_image(pil_img: Image.Image) -> str:
    try:
        if pil_img.mode == "1":
            return "mono"

        if pil_img.mode == "L":
            arr = np.array(pil_img.resize((128, 128), Image.BILINEAR))
            uniq = len(np.unique(arr))
            if uniq <= 8:
                return "mono"
            return "gray"

        rgb = _convert_to_rgb_safely(pil_img)
        if rgb is None:
            return "color"
        s = _mean_saturation(rgb)
        if s < 0.03:
            return "gray"
        return "color"
    except Exception:
        return "color"


# ==============================
# 5.1 Flate recompress
# ==============================
def _is_simple_flate(stream) -> bool:
    try:
        flt = stream.get("/Filter", None)
        if flt is None:
            return False

        if isinstance(flt, pikepdf.Array):
            if len(flt) != 1:
                return False
            if str(flt[0]) != "/FlateDecode":
                return False
        else:
            if str(flt) != "/FlateDecode":
                return False

        if stream.get("/DecodeParms", None) is not None:
            return False

        return True
    except Exception:
        return False


def _recompress_simple_flate_in_place(pdf, stream, level: int = 9):
    try:
        if not _is_simple_flate(stream):
            return (None, 0)

        try:
            decoded = stream.read_bytes()
        except Exception:
            return (None, 0)

        if decoded is None:
            return (None, 0)

        new_comp = _zlib_deflate_best(decoded, level=level)

        try:
            old_raw = stream.read_raw_bytes()
            if old_raw is not None and len(new_comp) >= len(old_raw) * 0.995:
                return (None, 0)
        except Exception:
            pass

        meta = {}
        try:
            for k in stream.keys():
                if str(k) == "/Length":
                    continue
                if str(k) in ("/Filter", "/DecodeParms"):
                    continue
                meta[k] = stream.get(k)
        except Exception:
            pass

        ns = pikepdf.Stream(pdf, new_comp, **meta)
        ns["/Filter"] = pikepdf.Name("/FlateDecode")
        try:
            if "/DecodeParms" in ns:
                del ns["/DecodeParms"]
        except Exception:
            pass

        saved = 0
        try:
            old_raw2 = stream.read_raw_bytes()
            if old_raw2 is not None:
                saved = max(0, len(old_raw2) - len(new_comp))
        except Exception:
            pass

        return (ns, saved)
    except Exception:
        return (None, 0)


# ==============================
# 6. PDF color detection (stronger)
# ==============================
_COLOR_OP_RE = re.compile(rb"(?<![\w/])(rg|RG|k|K|sc|SC|scn|SCN|cs|CS)(?![\w/])")


def _page_content_bytes(page) -> bytes:
    try:
        c = page.get("/Contents", None)
        if c is None:
            return b""
        if isinstance(c, pikepdf.Array):
            out = b""
            for s in c:
                try:
                    out += s.read_bytes()
                except Exception:
                    try:
                        out += s.read_raw_bytes()
                    except Exception:
                        pass
            return out
        try:
            return c.read_bytes()
        except Exception:
            try:
                return c.read_raw_bytes()
            except Exception:
                return b""
    except Exception:
        return b""


def _resources_has_color_space(page) -> bool:
    try:
        res = page.get("/Resources", None)
        if not res:
            return False
        cs = res.get("/ColorSpace", None)
        if cs:
            s = str(cs)
            if ("DeviceRGB" in s) or ("ICCBased" in s) or ("DeviceN" in s) or ("Separation" in s):
                return True
        xo = res.get("/XObject", None)
        if xo:
            s = str(xo)
            if ("DeviceRGB" in s) or ("ICCBased" in s) or ("DeviceN" in s) or ("Separation" in s):
                return True
        return False
    except Exception:
        return False


def _pdf_has_color(pdf_bytes: bytes, scan_pages: int = 12, max_imgs: int = 18) -> bool:
    try:
        imgs = []
        with pikepdf.open(io.BytesIO(pdf_bytes)) as pdf:
            n = min(scan_pages, len(pdf.pages))
            for pi in range(n):
                page = pdf.pages[pi]

                content = _page_content_bytes(page)
                if content and _COLOR_OP_RE.search(content):
                    return True
                if _resources_has_color_space(page):
                    return True

                res = page.get("/Resources", None)
                if not res:
                    continue
                visited = set()
                for _, __, img_obj in _iter_xobject_images_from_resources(res, visited):
                    try:
                        pil = pikepdf.PdfImage(img_obj).as_pil_image()
                        pil = _convert_to_rgb_safely(pil)
                        if pil is None:
                            continue
                        imgs.append(pil)
                    except Exception:
                        continue
                    if len(imgs) >= max_imgs:
                        break
                if len(imgs) >= max_imgs:
                    break

        if not imgs:
            return False

        for im in imgs:
            s = _mean_saturation(im)
            if s >= 0.045:
                return True

            arr = np.array(im.resize((96, 96), Image.BILINEAR)).astype(np.float32)
            r = arr[..., 0]
            g = arr[..., 1]
            b = arr[..., 2]
            chdiff = float(np.mean(np.abs(r - g)) + np.mean(np.abs(g - b)) + np.mean(np.abs(b - r))) / 3.0
            if chdiff >= 3.2:
                return True

        return False
    except Exception:
        return False


# ==============================
# 7. Profile estimation
# ==============================
def _estimate_doc_profile(pdf_bytes: bytes, scan_pages: int = 16):
    global _DOC_IMAGE_WEIGHT
    try:
        img_pages = 0
        total_pages = 0
        areas = []

        with pikepdf.open(io.BytesIO(pdf_bytes)) as pdf:
            total_pages = min(scan_pages, len(pdf.pages))
            for i in range(total_pages):
                page = pdf.pages[i]
                res = page.get("/Resources", None)
                if not res:
                    continue

                visited = set()
                page_has_img = False
                cnt = 0
                for _, __, img_obj in _iter_xobject_images_from_resources(res, visited):
                    page_has_img = True
                    cnt += 1
                    try:
                        pil = pikepdf.PdfImage(img_obj).as_pil_image()
                        pil = _convert_to_rgb_safely(pil)
                        if pil is None:
                            continue
                        w, h = pil.size
                        areas.append(int(w) * int(h))
                    except Exception:
                        continue
                    if cnt >= 2:
                        break

                if page_has_img:
                    img_pages += 1

        if total_pages <= 0:
            _DOC_IMAGE_WEIGHT = 0.0
            return "text"

        img_page_ratio = img_pages / total_pages
        avg_area = float(np.mean(areas)) if areas else 0.0

        area_k = max(0.0, min(1.0, (avg_area - 2.0e5) / (1.0e6 - 2.0e5))) if avg_area > 0 else 0.0
        _DOC_IMAGE_WEIGHT = float(max(0.0, min(1.0, 0.55 * img_page_ratio + 0.45 * area_k)))

        if img_page_ratio >= 0.55 and avg_area >= 7.0e5:
            return "image"
        if img_page_ratio >= 0.20:
            return "mixed"
        return "text"
    except Exception:
        _DOC_IMAGE_WEIGHT = 0.5
        return "mixed"


def _get_target_reduction_by_profile(profile: str) -> float:
    # 攻める方向へ（探索がより縮む解を選びやすく）
    if profile == "image":
        return 0.65
    if profile == "mixed":
        return 0.72
    return 0.80


# ==============================
# 8. Core: Metadata strip + Structural save
# ==============================
def _pikepdf_strip_metadata_in_memory(pdf_bytes: bytes) -> bytes:
    try:
        with pikepdf.open(io.BytesIO(pdf_bytes)) as pdf:
            try:
                with pdf.open_metadata() as meta:
                    meta.clear()
            except Exception:
                pass

            try:
                if "/Metadata" in pdf.Root:
                    del pdf.Root["/Metadata"]
            except Exception:
                pass

            try:
                if hasattr(pdf, "docinfo") and pdf.docinfo is not None:
                    for k in list(pdf.docinfo.keys()):
                        try:
                            del pdf.docinfo[k]
                        except Exception:
                            pass
            except Exception:
                pass

            out = io.BytesIO()
            pdf.save(out, force_version="1.6", linearize=False)
            return out.getvalue()
    except Exception:
        return pdf_bytes


# ==============================
# 9. Font/ToUnicode/CMap Optimization (Text heavy)
# ==============================
def _iter_page_fonts(pdf):
    """
    page resources -> /Font -> font dictionaries (including descendants if possible)
    """
    for page in pdf.pages:
        try:
            res = page.get("/Resources", None)
            if not res:
                continue
            fonts = res.get("/Font", None)
            if not fonts:
                continue
            if isinstance(fonts, pikepdf.Dictionary) or isinstance(fonts, dict):
                for _, fobj in list(fonts.items()):
                    try:
                        fd = fobj
                        try:
                            fd = fd.get_object()
                        except Exception:
                            pass
                        yield fd
                    except Exception:
                        continue
        except Exception:
            continue


def _optimize_font_related_streams(pdf):
    """
    日本語PDFなどでサイズを食う：
      - /ToUnicode ストリーム
      - FontDescriptor の /FontFile /FontFile2 /FontFile3
    を「単純Flateのみ」安全に再圧縮 + 同一rawのdedupe を行う
    """
    try:
        dedupe_map = {}  # sha1(raw) -> stream object
        for font in _iter_page_fonts(pdf):
            try:
                # ToUnicode
                try:
                    tu = font.get("/ToUnicode", None)
                    if isinstance(tu, pikepdf.Stream):
                        ns, _ = _recompress_simple_flate_in_place(pdf, tu, level=9)
                        if ns is not None:
                            try:
                                raw = ns.read_raw_bytes()
                                if raw is not None:
                                    hh = hashlib.sha1(raw).hexdigest()
                                    if hh in dedupe_map:
                                        font["/ToUnicode"] = dedupe_map[hh]
                                    else:
                                        font["/ToUnicode"] = ns
                                        dedupe_map[hh] = ns
                                else:
                                    font["/ToUnicode"] = ns
                            except Exception:
                                font["/ToUnicode"] = ns
                except Exception:
                    pass

                # DescendantFonts (Type0)
                descendants = []
                try:
                    df = font.get("/DescendantFonts", None)
                    if isinstance(df, pikepdf.Array):
                        descendants = list(df)
                except Exception:
                    descendants = []

                candidates = [font] + descendants

                for f0 in candidates:
                    try:
                        f0o = f0
                        try:
                            f0o = f0o.get_object()
                        except Exception:
                            pass

                        fd = f0o.get("/FontDescriptor", None)
                        if fd is None:
                            continue
                        try:
                            fd = fd.get_object()
                        except Exception:
                            pass
                        if not isinstance(fd, (pikepdf.Dictionary, dict)):
                            continue

                        for key in ("/FontFile", "/FontFile2", "/FontFile3"):
                            try:
                                fs = fd.get(key, None)
                                if not isinstance(fs, pikepdf.Stream):
                                    continue

                                ns, _ = _recompress_simple_flate_in_place(pdf, fs, level=9)
                                if ns is None:
                                    continue

                                try:
                                    raw = ns.read_raw_bytes()
                                    if raw is not None:
                                        hh = hashlib.sha1(raw).hexdigest()
                                        if hh in dedupe_map:
                                            fd[key] = dedupe_map[hh]
                                        else:
                                            fd[key] = ns
                                            dedupe_map[hh] = ns
                                    else:
                                        fd[key] = ns
                                except Exception:
                                    fd[key] = ns
                            except Exception:
                                continue
                    except Exception:
                        continue
            except Exception:
                continue
    except Exception:
        pass


# ==============================
# 10. Compression State + MCTS
# ==============================
class CompressionState:
    def __init__(
        self,
        quality=65,
        scale=0.95,
        obj_mode="gen",
        compress_streams=True,
        linearize=False,
        recompress_images=True,
        engine="gs",
        jpeg_subsampling=2,
        jpeg_progressive=True,
        img_color_mode="keep",
        dedupe_images=True,
        pike_encode_mode="auto",
        gs_preset="ebook",
        gs_downsample=False,
        gs_image_res=150,
        gs_color_strategy="LeaveColorUnchanged",
        gs_convert_cmyk_to_rgb=True,
        gs_passthrough_jpeg=False,
        gs_jpegq=45,
        gs_autofilter_color=False,
        gs_force_dctencode=True,
        gs_color_res=150,
        gs_gray_res=150,
        gs_mono_res=300,
        gs_downsample_threshold=1.35,
        gs_downsample_type="bicubic",
        gs_enable_mono=True,
    ):
        self.quality = int(quality)
        self.scale = float(scale)
        self.obj_mode = str(obj_mode)
        self.compress_streams = bool(compress_streams)
        self.linearize = bool(linearize)
        self.recompress_images = bool(recompress_images)
        self.engine = str(engine)
        self.jpeg_subsampling = int(jpeg_subsampling)
        self.jpeg_progressive = bool(jpeg_progressive)
        self.img_color_mode = str(img_color_mode)
        self.dedupe_images = bool(dedupe_images)
        self.pike_encode_mode = str(pike_encode_mode)

        self.gs_preset = str(gs_preset)
        self.gs_downsample = bool(gs_downsample)
        self.gs_image_res = int(gs_image_res)
        self.gs_color_strategy = str(gs_color_strategy)
        self.gs_convert_cmyk_to_rgb = bool(gs_convert_cmyk_to_rgb)

        self.gs_passthrough_jpeg = bool(gs_passthrough_jpeg)
        self.gs_jpegq = int(gs_jpegq)
        self.gs_autofilter_color = bool(gs_autofilter_color)
        self.gs_force_dctencode = bool(gs_force_dctencode)

        self.gs_color_res = int(gs_color_res)
        self.gs_gray_res = int(gs_gray_res)
        self.gs_mono_res = int(gs_mono_res)
        self.gs_downsample_threshold = float(gs_downsample_threshold)
        self.gs_downsample_type = str(gs_downsample_type)
        self.gs_enable_mono = bool(gs_enable_mono)

    def possible_actions(self):
        actions = []
        profile = _DOC_PROFILE

        actions.append(("engine", "gs" if self.engine == "pike" else "pike"))

        for p in ["screen", "ebook", "printer", "prepress"]:
            if p != self.gs_preset:
                actions.append(("gs_preset", p))

        if profile == "image":
            jq_candidates = [55, 60, 70, 80, 90]
        elif profile == "mixed":
            jq_candidates = [35, 40, 50, 60, 70, 80, 90]
        else:
            jq_candidates = [18, 22, 28, 35, 40, 50, 60, 70, 80]
        for jq in jq_candidates:
            if jq != self.gs_jpegq:
                actions.append(("gs_jpegq", jq))

        if profile in ("mixed", "image"):
            color_res_candidates = [120, 150, 200, 300]
            gray_res_candidates = [120, 150, 200, 300]
        else:
            color_res_candidates = [72, 96, 120, 150, 200, 300]
            gray_res_candidates = [72, 96, 120, 150, 200, 300]

        mono_res_candidates = [300, 600, 1200]

        for r in color_res_candidates:
            if r != self.gs_color_res:
                actions.append(("gs_color_res", r))
        for r in gray_res_candidates:
            if r != self.gs_gray_res:
                actions.append(("gs_gray_res", r))
        for r in mono_res_candidates:
            if r != self.gs_mono_res:
                actions.append(("gs_mono_res", r))

        for t in ["subsample", "average", "bicubic"]:
            if t != self.gs_downsample_type:
                actions.append(("gs_downsample_type", t))

        for th in [1.0, 1.2, 1.35, 1.5, 1.8, 2.0]:
            if abs(th - self.gs_downsample_threshold) > 1e-9:
                actions.append(("gs_downsample_threshold", th))

        if _ORIG_HAS_COLOR:
            if self.gs_color_strategy != "LeaveColorUnchanged":
                actions.append(("gs_color_strategy", "LeaveColorUnchanged"))
        else:
            for cs in ["LeaveColorUnchanged", "Gray"]:
                if cs != self.gs_color_strategy:
                    actions.append(("gs_color_strategy", cs))

        actions.append(("gs_passthrough_jpeg", not self.gs_passthrough_jpeg))
        actions.append(("gs_autofilter_color", not self.gs_autofilter_color))
        actions.append(("gs_force_dctencode", not self.gs_force_dctencode))

        if profile == "image":
            q_candidates = [55, 60, 70, 80, 90]
        elif profile == "mixed":
            q_candidates = [35, 40, 50, 60, 70, 80, 90]
        else:
            q_candidates = [18, 22, 28, 35, 40, 50, 60, 70, 80]
        for q in q_candidates:
            if q != self.quality:
                actions.append(("quality", q))

        actions.append(("obj_mode", "max" if self.obj_mode == "gen" else "gen"))
        actions.append(("compress_streams", not self.compress_streams))
        actions.append(("linearize", not self.linearize))
        actions.append(("recompress_images", not self.recompress_images))

        if profile in ("mixed", "image"):
            ss_candidates = [0, 1]
        else:
            ss_candidates = [0, 1, 2]
        for ss in ss_candidates:
            if ss != self.jpeg_subsampling:
                actions.append(("jpeg_subsampling", ss))

        actions.append(("jpeg_progressive", not self.jpeg_progressive))

        if not _ORIG_HAS_COLOR:
            actions.append(("img_color_mode", "gray" if self.img_color_mode == "keep" else "keep"))
        else:
            if self.img_color_mode != "keep":
                actions.append(("img_color_mode", "keep"))

        actions.append(("dedupe_images", not self.dedupe_images))

        for em in ["auto", "jpeg", "flate"]:
            if em != self.pike_encode_mode:
                actions.append(("pike_encode_mode", em))

        return actions

    def next_state(self, action):
        k, v = action
        return CompressionState(
            quality=v if k == "quality" else self.quality,
            scale=v if k == "scale" else self.scale,
            obj_mode=v if k == "obj_mode" else self.obj_mode,
            compress_streams=v if k == "compress_streams" else self.compress_streams,
            linearize=v if k == "linearize" else self.linearize,
            recompress_images=v if k == "recompress_images" else self.recompress_images,
            engine=v if k == "engine" else self.engine,
            jpeg_subsampling=v if k == "jpeg_subsampling" else self.jpeg_subsampling,
            jpeg_progressive=v if k == "jpeg_progressive" else self.jpeg_progressive,
            img_color_mode=v if k == "img_color_mode" else self.img_color_mode,
            dedupe_images=v if k == "dedupe_images" else self.dedupe_images,
            pike_encode_mode=v if k == "pike_encode_mode" else self.pike_encode_mode,

            gs_preset=v if k == "gs_preset" else self.gs_preset,
            gs_downsample=v if k == "gs_downsample" else self.gs_downsample,
            gs_image_res=v if k == "gs_image_res" else self.gs_image_res,
            gs_color_strategy=v if k == "gs_color_strategy" else self.gs_color_strategy,
            gs_convert_cmyk_to_rgb=v if k == "gs_convert_cmyk_to_rgb" else self.gs_convert_cmyk_to_rgb,

            gs_passthrough_jpeg=v if k == "gs_passthrough_jpeg" else self.gs_passthrough_jpeg,
            gs_jpegq=v if k == "gs_jpegq" else self.gs_jpegq,
            gs_autofilter_color=v if k == "gs_autofilter_color" else self.gs_autofilter_color,
            gs_force_dctencode=v if k == "gs_force_dctencode" else self.gs_force_dctencode,

            gs_color_res=v if k == "gs_color_res" else self.gs_color_res,
            gs_gray_res=v if k == "gs_gray_res" else self.gs_gray_res,
            gs_mono_res=v if k == "gs_mono_res" else self.gs_mono_res,
            gs_downsample_threshold=v if k == "gs_downsample_threshold" else self.gs_downsample_threshold,
            gs_downsample_type=v if k == "gs_downsample_type" else self.gs_downsample_type,
            gs_enable_mono=v if k == "gs_enable_mono" else self.gs_enable_mono,
        )

    def random_rollout(self, depth=3):
        state = self
        for _ in range(depth):
            acts = state.possible_actions()
            state = state.next_state(random.choice(acts))
        return state

    def evaluate_soft_target(
        self,
        orig_size: int,
        new_size: int,
        ssim_score: float,
        color_coeff: float,
        zoom_coeff: float,
        target_reduction: float,
    ) -> float:
        if orig_size <= 0:
            return -1.0

        reduction = (orig_size - new_size) / orig_size
        if reduction <= 0:
            return -1.0

        # 攻める：サイズ優先に寄せる（qualityの指数を緩める）
        q_mix = (0.15 + 0.85 * float(ssim_score)) * float(color_coeff) * float(zoom_coeff)
        q_mix = float(max(0.0, min(1.0, q_mix)))

        t = float(max(1e-6, min(0.95, target_reduction)))
        reach = reduction / t
        reach_k = float(max(0.0, min(1.35, reach)))

        if reach_k < 1.0:
            size_term = (reach_k ** 2.1)
        else:
            extra = min(0.40, reduction - t)
            size_term = 1.0 + 0.50 * (extra / 0.40)

        wq = 1.15 if _DOC_PROFILE in ("mixed", "image") else 0.95
        quality_term = (q_mix ** (1.8 * wq))

        score = 135.0 * size_term * (0.30 + 0.70 * quality_term)

        if _ORIG_HAS_COLOR and color_coeff < 0.25:
            score *= 0.05

        return float(score)


class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.untried = state.possible_actions()
        self.visits = 0
        self.value = 0.0

    def ucb(self, total_visits):
        if self.visits == 0:
            return float("inf")
        tv = max(1, int(total_visits))
        return (self.value / self.visits) + 1.41 * math.sqrt(math.log(tv) / self.visits)


# ==============================
# 11. Pike-only deep compression core
# ==============================
def _iter_all_page_image_xobjects(pdf):
    for page in pdf.pages:
        page_w_in, page_h_in = _page_inches(page)
        res = page.get("/Resources", None)
        visited = set()
        if not res:
            continue
        for xobj_dict, name, img_stream in _iter_xobject_images_from_resources(res, visited):
            yield (page, page_w_in, page_h_in, xobj_dict, name, img_stream)


def _iter_form_streams(resources, visited):
    if resources is None:
        return
    try:
        xobj = resources.get("/XObject", None)
        if not xobj:
            return
        for name, obj in list(xobj.items()):
            try:
                st0 = obj
                try:
                    st0 = st0.get_object()
                except Exception:
                    pass

                oid = None
                try:
                    oid = int(st0.objgen[0]) * 1000000 + int(st0.objgen[1])
                except Exception:
                    oid = id(st0)
                if oid in visited:
                    continue
                visited.add(oid)

                subtype = st0.get("/Subtype", None)
                if _is_name(subtype, "/Form"):
                    yield (xobj, name, st0)
                    res2 = st0.get("/Resources", None)
                    if res2:
                        for it in _iter_form_streams(res2, visited):
                            yield it
                elif _is_name(subtype, "/Image"):
                    continue
            except Exception:
                continue
    except Exception:
        return


def _looks_like_dct_image_stream(img_stream) -> bool:
    try:
        flt = img_stream.get("/Filter", None)
        if flt is None:
            return False
        s = str(flt)
        return "DCTDecode" in s
    except Exception:
        return False


def _choose_image_output_kind(pil_img: Image.Image, state: CompressionState) -> str:
    mode = str(getattr(state, "pike_encode_mode", "auto")).lower()

    photo = _is_photo_like(pil_img)
    texty = _is_texty_image(pil_img)
    kind = _gs_classify_image(pil_img)

    if mode == "jpeg":
        return "jpeg"
    if mode == "flate":
        return "flate"

    if texty:
        if kind in ("mono", "gray"):
            return "flate"
        return "jpeg"

    if photo:
        return "jpeg"

    if kind in ("mono", "gray"):
        return "flate"

    return "jpeg"


def _execute_pike_only(pdf_bytes: bytes, state: CompressionState):
    preview_img = None
    state = _apply_safety_guards(state)

    try:
        stripped = _pikepdf_strip_metadata_in_memory(pdf_bytes)

        with pikepdf.open(io.BytesIO(stripped)) as pdf:
            try:
                with pdf.open_metadata() as meta:
                    meta.clear()
            except Exception:
                pass
            try:
                if "/Metadata" in pdf.Root:
                    del pdf.Root["/Metadata"]
            except Exception:
                pass
            try:
                if hasattr(pdf, "docinfo") and pdf.docinfo is not None:
                    for k in list(pdf.docinfo.keys()):
                        try:
                            del pdf.docinfo[k]
                        except Exception:
                            pass
            except Exception:
                pass

            dedupe_img_map = {}
            dedupe_flate_map = {}

            # ==========================
            # (A) 画像再圧縮
            # ==========================
            if getattr(state, "recompress_images", True):
                for page, page_w_in, page_h_in, xobj_dict, name, img in _iter_all_page_image_xobjects(pdf):
                    try:
                        try:
                            if "/SMask" in img or "/Mask" in img:
                                continue
                        except Exception:
                            pass

                        orig_raw_for_hash = None
                        try:
                            orig_raw_for_hash = img.read_raw_bytes()
                        except Exception:
                            orig_raw_for_hash = None

                        h0 = None
                        if state.dedupe_images and orig_raw_for_hash:
                            h0 = hashlib.sha1(orig_raw_for_hash).hexdigest()
                            if h0 in dedupe_img_map:
                                xobj_dict[name] = dedupe_img_map[h0]
                                continue

                        is_input_jpeg = _looks_like_dct_image_stream(img)

                        try:
                            pimg = pikepdf.PdfImage(img)
                            pil_img = pimg.as_pil_image()
                        except Exception:
                            continue

                        if state.img_color_mode == "gray":
                            try:
                                pil_img = pil_img.convert("L")
                            except Exception:
                                pil_img = _convert_to_rgb_safely(pil_img)
                        else:
                            pass

                        if pil_img is None:
                            continue

                        kind = _gs_classify_image(pil_img)
                        texty = _is_texty_image(pil_img)
                        photo_like = _is_photo_like(pil_img)

                        if kind == "mono":
                            target_dpi = int(getattr(state, "gs_mono_res", 300))
                        elif kind == "gray":
                            target_dpi = int(getattr(state, "gs_gray_res", 150))
                        else:
                            target_dpi = int(getattr(state, "gs_color_res", 150))

                        threshold = float(getattr(state, "gs_downsample_threshold", 1.35))

                        do_down = False
                        new_w, new_h = pil_img.size

                        # textyは基本ダウンサンプルしない（ただし text profile の場合は“多少”許す）
                        allow_texty_down = (_DOC_PROFILE == "text")
                        if (not texty) or allow_texty_down:
                            do_down, new_w, new_h = _gs_should_downsample(
                                w_px=int(pil_img.size[0]),
                                h_px=int(pil_img.size[1]),
                                page_w_in=float(page_w_in),
                                page_h_in=float(page_h_in),
                                target_dpi=int(target_dpi),
                                threshold=float(threshold)
                            )

                        if (
                            getattr(state, "gs_passthrough_jpeg", False)
                            and is_input_jpeg
                            and (not do_down)
                            and (state.img_color_mode == "keep")
                        ):
                            if state.dedupe_images and h0 is not None:
                                dedupe_img_map[h0] = img
                            continue

                        if do_down and (new_w != pil_img.size[0] or new_h != pil_img.size[1]):
                            resample = _downsample_resample_filter(getattr(state, "gs_downsample_type", "bicubic"))
                            try:
                                pil_img = pil_img.resize((int(new_w), int(new_h)), resample=resample)
                            except Exception:
                                pass

                        out_kind = _choose_image_output_kind(pil_img, state)

                        new_stream = None
                        new_cost_bytes = None

                        if out_kind == "jpeg":
                            q = int(state.quality)
                            q = max(18, min(92, q))
                            ss = int(state.jpeg_subsampling)
                            prog = bool(state.jpeg_progressive)

                            # textyは滲みやすいが、圧縮も欲しい：一段だけ妥協
                            if texty:
                                ss = min(ss, 1)
                                q = max(q, 58)

                            pil_for_j = pil_img
                            if pil_for_j.mode != "RGB":
                                pil_for_j = _convert_to_rgb_safely(pil_for_j)
                                if pil_for_j is None:
                                    continue

                            # 不気味コントラスト抑制：orig と比較できるときだけ軽量ガード
                            try:
                                orig_rgb = _convert_to_rgb_safely(pil_for_j)
                                if orig_rgb is None:
                                    orig_rgb = pil_for_j
                                pil_guarded, q2, ss2 = _guard_eerie_contrast(orig_rgb, pil_for_j, q=q, subsampling=ss)
                                q, ss = int(q2), int(ss2)
                                pil_for_j = pil_guarded
                            except Exception:
                                pass

                            jb = _encode_image_to_jpeg(pil_for_j, quality=q, subsampling=ss, progressive=prog)
                            if not jb:
                                continue

                            new_stream = pikepdf.Stream(pdf, jb)
                            new_stream["/Type"] = pikepdf.Name("/XObject")
                            new_stream["/Subtype"] = pikepdf.Name("/Image")
                            new_stream["/Filter"] = pikepdf.Name("/DCTDecode")
                            new_stream["/Width"] = int(pil_for_j.size[0])
                            new_stream["/Height"] = int(pil_for_j.size[1])
                            new_stream["/ColorSpace"] = pikepdf.Name("/DeviceRGB")
                            new_stream["/BitsPerComponent"] = 8

                            new_cost_bytes = jb

                        else:
                            force_mono_1bit = False
                            if kind == "mono" and getattr(state, "gs_enable_mono", True) and (not photo_like):
                                force_mono_1bit = True

                            fl = _make_pdf_image_stream_flate(pdf, pil_img, force_mono_1bit=force_mono_1bit)
                            if fl is None:
                                continue
                            new_stream = fl

                            try:
                                new_cost_bytes = new_stream.read_raw_bytes()
                            except Exception:
                                new_cost_bytes = None

                        replace_ok = True
                        try:
                            old_raw = img.read_raw_bytes()
                            if old_raw is not None and new_cost_bytes is not None:
                                if len(new_cost_bytes) >= len(old_raw) * 0.995:
                                    replace_ok = False
                                if do_down and len(new_cost_bytes) <= len(old_raw) * 1.10:
                                    replace_ok = True
                        except Exception:
                            pass

                        if not replace_ok:
                            if state.dedupe_images and h0 is not None:
                                dedupe_img_map[h0] = img
                            continue

                        if state.dedupe_images:
                            try:
                                raw1 = new_stream.read_raw_bytes()
                                if raw1 is not None:
                                    h1 = hashlib.sha1(raw1).hexdigest()
                                    if h1 in dedupe_img_map:
                                        xobj_dict[name] = dedupe_img_map[h1]
                                    else:
                                        xobj_dict[name] = new_stream
                                        dedupe_img_map[h1] = new_stream
                                else:
                                    xobj_dict[name] = new_stream
                            except Exception:
                                xobj_dict[name] = new_stream
                        else:
                            xobj_dict[name] = new_stream

                        if preview_img is None:
                            try:
                                preview_img = pil_img.convert("RGB") if pil_img.mode != "RGB" else pil_img.copy()
                            except Exception:
                                pass

                    except Exception:
                        continue

            # ==========================
            # (B) Flate再圧縮：Contents / Form
            # ==========================
            try:
                for page in pdf.pages:
                    c = page.get("/Contents", None)
                    if c is None:
                        continue

                    if isinstance(c, pikepdf.Array):
                        for i, s in enumerate(list(c)):
                            try:
                                st0 = s
                                try:
                                    st0 = st0.get_object()
                                except Exception:
                                    pass
                                if not isinstance(st0, pikepdf.Stream):
                                    continue

                                ns, _ = _recompress_simple_flate_in_place(pdf, st0, level=9)
                                if ns is None:
                                    continue

                                try:
                                    raw = ns.read_raw_bytes()
                                    if raw is not None:
                                        hh = hashlib.sha1(raw).hexdigest()
                                        if hh in dedupe_flate_map:
                                            c[i] = dedupe_flate_map[hh]
                                        else:
                                            c[i] = ns
                                            dedupe_flate_map[hh] = ns
                                    else:
                                        c[i] = ns
                                except Exception:
                                    c[i] = ns
                            except Exception:
                                continue
                    else:
                        try:
                            st0 = c
                            try:
                                st0 = st0.get_object()
                            except Exception:
                                pass
                            if isinstance(st0, pikepdf.Stream):
                                ns, _ = _recompress_simple_flate_in_place(pdf, st0, level=9)
                                if ns is not None:
                                    try:
                                        raw = ns.read_raw_bytes()
                                        if raw is not None:
                                            hh = hashlib.sha1(raw).hexdigest()
                                            if hh in dedupe_flate_map:
                                                page["/Contents"] = dedupe_flate_map[hh]
                                            else:
                                                page["/Contents"] = ns
                                                dedupe_flate_map[hh] = ns
                                        else:
                                            page["/Contents"] = ns
                                    except Exception:
                                        page["/Contents"] = ns
                        except Exception:
                            pass

                for page in pdf.pages:
                    res = page.get("/Resources", None)
                    if not res:
                        continue
                    visited = set()
                    for xobj_dict, name, form_st in _iter_form_streams(res, visited):
                        try:
                            ns, _ = _recompress_simple_flate_in_place(pdf, form_st, level=9)
                            if ns is None:
                                continue
                            try:
                                raw = ns.read_raw_bytes()
                                if raw is not None:
                                    hh = hashlib.sha1(raw).hexdigest()
                                    if hh in dedupe_flate_map:
                                        xobj_dict[name] = dedupe_flate_map[hh]
                                    else:
                                        xobj_dict[name] = ns
                                        dedupe_flate_map[hh] = ns
                                else:
                                    xobj_dict[name] = ns
                            except Exception:
                                xobj_dict[name] = ns
                        except Exception:
                            continue

            except Exception:
                pass

            # ==========================
            # (C) Font / ToUnicode / CMap の圧縮強化（日本語PDFの縮み）
            # ==========================
            _optimize_font_related_streams(pdf)

            out_buf = io.BytesIO()
            om = (
                pikepdf.ObjectStreamMode.generate
                if state.obj_mode == "gen"
                else pikepdf.ObjectStreamMode.disable
            )

            pdf.save(
                out_buf,
                compress_streams=bool(state.compress_streams),
                object_stream_mode=om,
                force_version="1.6",
                linearize=bool(getattr(state, "linearize", False)),
            )

            out_bytes = out_buf.getvalue()
            sample_imgs = _extract_sample_images_as_rgb(out_bytes, max_pages=8, max_total=6)

            return out_bytes, preview_img, sample_imgs

    except Exception:
        return None, None, []


# ==============================
# 12. Hybrid executor: (FORCE GS-Emulation)
# ==============================
def _execute_hybrid(pdf_bytes: bytes, state: CompressionState):
    state = _apply_safety_guards(state)
    stripped = _pikepdf_strip_metadata_in_memory(pdf_bytes)
    return _execute_pike_only(stripped, state)


def execute_compression(pdf_bytes: bytes, state: CompressionState, session_id: str):
    global _GLOBAL_ORIG_HASH

    if _GLOBAL_ORIG_HASH is None:
        try:
            _GLOBAL_ORIG_HASH = hashlib.sha1(pdf_bytes).hexdigest()
        except Exception:
            _GLOBAL_ORIG_HASH = "nohash"

    key = (str(session_id), _GLOBAL_ORIG_HASH, _state_key(state))

    with _CACHE_LOCK:
        _cache_prune_locked(str(session_id))
        if key in _COMPRESS_CACHE:
            ts, res_b, prev, samples = _COMPRESS_CACHE[key]
            return res_b, prev, samples

    try:
        res, preview_img, samples = _execute_hybrid(pdf_bytes, state)
        if res is None:
            return None, None, []

        if not samples:
            samples = _extract_sample_images_as_rgb(res, max_pages=8, max_total=6)

        if preview_img is None and samples:
            preview_img = samples[0]

        with _CACHE_LOCK:
            _cache_prune_locked(str(session_id))
            _COMPRESS_CACHE[key] = (time.time(), res, preview_img, samples)

        return res, preview_img, samples
    except Exception:
        return None, None, []


# ==============================
# 13. Streamlit UI Layer (UIは変更しない)
# ==============================
st.set_page_config(page_title="Hideyoshi_Murakami PDF AI-Optimizer v7.20.0", layout="wide")

st.markdown("""
    # 🚀 Hideyoshi_Murakami PDF AI-Optimizer v7.20.0
    **Hybrid GS + PikePDF Deep-Structural Build (GS Auto-Use if Available)**
""")

# X handle (rerunしても消えない固定表示)
st.caption("X（旧Twitter）: @nagisa7654321")

uploaded_file = st.file_uploader("PDFファイルをアップロード (履歴書対応)", type="pdf")

if uploaded_file:
    orig_bytes = uploaded_file.read()

    if orig_bytes is None:
        st.error("ファイル読み込みに失敗しました。")
        st.stop()

    if len(orig_bytes) > MAX_UPLOAD_BYTES:
        st.error(f"ファイルが大きすぎます（上限 {MAX_UPLOAD_MB}MB）。サイズを小さくして再試行してください。")
        st.stop()

    orig_size = len(orig_bytes)

    try:
        with pikepdf.open(io.BytesIO(orig_bytes)) as _pdf_tmp:
            _page_cnt = len(_pdf_tmp.pages)
        if _page_cnt > int(MAX_PDF_PAGES):
            st.error(f"ページ数が多すぎます（{_page_cnt} pages / 上限 {int(MAX_PDF_PAGES)} pages）。ページ数を減らして再試行してください。")
            st.stop()
    except Exception:
        st.error("PDFの解析に失敗しました（破損/特殊形式の可能性があります）。")
        st.stop()

    session_id = _get_session_id()

    _client_id = _get_client_id()
    _ok, _retry = _rate_limit_check_and_mark(_client_id)
    if not _ok:
        st.error(f"短時間のリクエストが多すぎます。{int(_retry)}秒ほど待ってから再試行してください。")
        st.stop()

    try:
        _GLOBAL_ORIG_HASH = hashlib.sha1(orig_bytes).hexdigest()
    except Exception:
        _GLOBAL_ORIG_HASH = "nohash"

    st.info(f"オリジナルサイズ: {orig_size/1024:.1f} KB")

    st.caption(f"Ghostscript detected: {'YES' if shutil.which('gs') or shutil.which('gswin64c') or shutil.which('gswin32c') else 'NO'} | enabled={'YES' if ENABLE_GS else 'NO'} (自動使用)")

    st.markdown("### 🎯 目標削減率（ソフト目標）")
    target_mode = st.radio(
        "目標削減率を選択（※満たせない場合でも最善解を返します）",
        ["Auto (文書判定)", "30%", "50%", "70%", "90%", "Custom"],
        index=0,
        horizontal=True
    )

    custom_target = 0.70
    if target_mode == "Custom":
        custom_target = st.slider("Custom目標削減率", 10, 95, 70) / 100.0

    col1, col2 = st.columns(2)
    with col1:
        iterations = st.slider("AI探索ステップ数 (思考の深さ)", 10, 140, 40)
    with col2:
        threads = st.slider("並列スレッド数 (処理速度)", 1, 10, 5)

    try:
        iterations = int(min(int(iterations), int(MAX_ITERATIONS_HARD)))
    except Exception:
        iterations = 40
    try:
        threads = int(min(int(threads), int(MAX_THREADS_HARD)))
    except Exception:
        threads = 5

    start_btn = st.button("AI最適化開始 (構造レベル解析)", type="primary")

    if start_btn:
        _acquired = False
        try:
            try:
                _acquired = _JOB_SEMAPHORE.acquire(timeout=1.0)
            except Exception:
                _acquired = True
            if not _acquired:
                st.error("現在混雑しています。少し待ってから再試行してください。")
                st.stop()

            _t0 = time.time()
            _time_limited = False
            _ORIG_HAS_COLOR = _pdf_has_color(orig_bytes)
            _DOC_PROFILE = _estimate_doc_profile(orig_bytes)

            if target_mode.startswith("Auto"):
                TARGET_REDUCTION = _get_target_reduction_by_profile(_DOC_PROFILE)
            elif target_mode == "30%":
                TARGET_REDUCTION = 0.30
            elif target_mode == "50%":
                TARGET_REDUCTION = 0.50
            elif target_mode == "70%":
                TARGET_REDUCTION = 0.70
            elif target_mode == "90%":
                TARGET_REDUCTION = 0.90
            else:
                TARGET_REDUCTION = float(custom_target)

            st.caption(
                f"target={TARGET_REDUCTION*100:.0f}% (soft) | profile={_DOC_PROFILE} | color={'Y' if _ORIG_HAS_COLOR else 'N'} | img_weight={_DOC_IMAGE_WEIGHT:.2f}"
            )

            orig_samples = _extract_sample_images_as_rgb(orig_bytes, max_pages=8, max_total=6)
            orig_sample_for_preview = orig_samples[0] if orig_samples else None

            MIN_Q_INIT = 0.24
            MIN_Q_FLOOR = 0.10
            STALL_WINDOW = max(6, int(iterations * 0.15))
            GATE_RELAX_RATE = 0.90
            gate_q = float(MIN_Q_INIT)
            stall_count = 0

            best_score = -1e18
            best_state = None
            best_size = orig_size
            best_preview_img = None

            progress_bar = st.progress(0)
            status_text = st.empty()
            chart_placeholder = st.empty()
            preview_placeholder = st.empty()

            history = []

            # --------------------------
            # Seed（より攻める寄り）
            # --------------------------
            seed_states = []

            if _DOC_PROFILE in ("mixed", "image"):
                seed_states.append(CompressionState(
                    engine="gs", gs_preset="printer", gs_jpegq=60,
                    gs_autofilter_color=True, gs_passthrough_jpeg=True,
                    gs_downsample_threshold=1.35, gs_downsample_type="average",
                    pike_encode_mode="auto"
                ))
                seed_states.append(CompressionState(
                    engine="gs", gs_preset="prepress", gs_jpegq=70,
                    gs_autofilter_color=True, gs_passthrough_jpeg=True,
                    gs_downsample_threshold=1.35, gs_downsample_type="bicubic",
                    pike_encode_mode="auto"
                ))
                seed_states.append(CompressionState(
                    engine="pike", quality=70, scale=1.0, pike_encode_mode="jpeg",
                    jpeg_subsampling=0, img_color_mode="keep", recompress_images=True
                ))
            else:
                seed_states.append(CompressionState(
                    engine="gs", gs_preset="screen", gs_jpegq=28,
                    gs_autofilter_color=True, gs_passthrough_jpeg=False,
                    gs_downsample_threshold=1.20, gs_downsample_type="subsample",
                    pike_encode_mode="auto"
                ))
                seed_states.append(CompressionState(
                    engine="gs", gs_preset="ebook", gs_jpegq=35,
                    gs_autofilter_color=True, gs_passthrough_jpeg=True,
                    gs_downsample_threshold=1.20, gs_downsample_type="bicubic",
                    pike_encode_mode="auto"
                ))
                seed_states.append(CompressionState(
                    engine="pike", quality=28, scale=0.90, pike_encode_mode="auto",
                    jpeg_subsampling=2, img_color_mode="keep", recompress_images=True
                ))

            # --------------------------
            # Seed評価
            # --------------------------
            for ss in seed_states:
                if (time.time() - _t0) > float(MAX_RUNTIME_SECONDS):
                    _time_limited = True
                    break
                ss = _apply_safety_guards(ss)
                res_bytes, p_img, after_samples = execute_compression(orig_bytes, ss, session_id=session_id)
                if not res_bytes:
                    continue

                curr_size = len(res_bytes)

                ssim_v = 1.0
                color_k = 1.0
                zoom_k = 1.0

                if orig_samples and after_samples:
                    try:
                        m = min(len(orig_samples), len(after_samples))
                        ssim_list, color_list, zoom_list = [], [], []
                        for j in range(m):
                            o = orig_samples[j]
                            a = after_samples[j].resize(o.size)

                            a1 = np.array(o).astype(np.float32)
                            a2 = np.array(a).astype(np.float32)
                            mse = float(np.mean((a1 - a2) ** 2))
                            ssim_list.append(max(0.0, 1.0 - (mse / 65025.0)))

                            color_list.append(_color_penalty(o, a))
                            zoom_list.append(_zoom_robustness_coeff(o, a))

                        ssim_v = float(min(ssim_list)) if ssim_list else 1.0
                        color_k = float(min(color_list)) if color_list else 1.0
                        zoom_k = float(min(zoom_list)) if zoom_list else 1.0
                    except Exception:
                        ssim_v, color_k, zoom_k = 0.82, 0.70, 0.70

                q_mix = (0.15 + 0.85 * float(ssim_v)) * float(color_k) * float(zoom_k)

                gate_pen = 1.0
                if q_mix < gate_q:
                    gate_pen = 0.12

                score = ss.evaluate_soft_target(
                    orig_size=orig_size,
                    new_size=curr_size,
                    ssim_score=ssim_v,
                    color_coeff=color_k,
                    zoom_coeff=zoom_k,
                    target_reduction=TARGET_REDUCTION
                ) * gate_pen

                if score > best_score:
                    best_score = score
                    best_state = ss
                    best_size = curr_size
                    if p_img is not None:
                        best_preview_img = p_img

            # --------------------------
            # MCTS本体
            # --------------------------
            root = Node(CompressionState())
            for i in range(iterations):
                if (time.time() - _t0) > float(MAX_RUNTIME_SECONDS):
                    _time_limited = True
                    break
                prev_best_score = float(best_score)

                node = root
                while node.children and not node.untried:
                    node = max(node.children, key=lambda n: n.ucb(i + 1))

                if node.untried:
                    action = node.untried.pop()
                    child = Node(node.state.next_state(action), node)
                    node.children.append(child)
                    node = child

                futures = []
                rollout_states = []
                with ThreadPoolExecutor(max_workers=threads) as executor:
                    for _ in range(threads):
                        rollout_state = node.state.random_rollout(depth=3)
                        rollout_state = _apply_safety_guards(rollout_state)
                        f = executor.submit(execute_compression, orig_bytes, rollout_state, session_id)
                        try:
                            add_script_run_context(f)
                        except Exception:
                            pass
                        futures.append(f)
                        rollout_states.append(rollout_state)

                    scores = []

                    for idx, f in enumerate(futures):
                        r_state = rollout_states[idx]
                        try:
                            res_bytes, p_img, after_samples = f.result()
                        except Exception as e:
                            if SHOW_DEBUG:
                                st.exception(e)
                            else:
                                st.error("内部処理でエラーが発生しました。別のPDFで試すか、探索ステップ/スレッドを下げてください。")
                            continue

                        if not res_bytes:
                            continue

                        curr_size = len(res_bytes)

                        ssim_v = 1.0
                        color_k = 1.0
                        zoom_k = 1.0

                        if orig_samples and after_samples:
                            try:
                                m = min(len(orig_samples), len(after_samples))
                                ssim_list, color_list, zoom_list = [], [], []
                                for j in range(m):
                                    o = orig_samples[j]
                                    a = after_samples[j].resize(o.size)

                                    a1 = np.array(o).astype(np.float32)
                                    a2 = np.array(a).astype(np.float32)
                                    mse = float(np.mean((a1 - a2) ** 2))
                                    ssim_list.append(max(0.0, 1.0 - (mse / 65025.0)))

                                    color_list.append(_color_penalty(o, a))
                                    zoom_list.append(_zoom_robustness_coeff(o, a))

                                ssim_v = float(min(ssim_list)) if ssim_list else 1.0
                                color_k = float(min(color_list)) if color_list else 1.0
                                zoom_k = float(min(zoom_list)) if zoom_list else 1.0
                            except Exception:
                                ssim_v, color_k, zoom_k = 0.82, 0.70, 0.70

                        q_mix = (0.15 + 0.85 * float(ssim_v)) * float(color_k) * float(zoom_k)

                        gate_pen = 1.0
                        if q_mix < gate_q:
                            gate_pen = 0.12

                        score = r_state.evaluate_soft_target(
                            orig_size=orig_size,
                            new_size=curr_size,
                            ssim_score=ssim_v,
                            color_coeff=color_k,
                            zoom_coeff=zoom_k,
                            target_reduction=TARGET_REDUCTION
                        ) * gate_pen

                        scores.append(score)

                        if score > best_score:
                            best_score = score
                            best_state = r_state
                            best_size = curr_size
                            if p_img is not None:
                                best_preview_img = p_img

                    avg_score = float(np.mean(scores)) if scores else 0.0
                    temp_n = node
                    while temp_n:
                        temp_n.visits += 1
                        temp_n.value += avg_score
                        temp_n = temp_n.parent

                improved = (best_score > prev_best_score * 1.0005)
                if improved:
                    stall_count = 0
                else:
                    stall_count += 1

                if stall_count >= STALL_WINDOW:
                    gate_q = max(MIN_Q_FLOOR, gate_q * GATE_RELAX_RATE)
                    stall_count = 0

                history.append(best_size / 1024)
                progress_bar.progress((i + 1) / iterations)
                reduction_now = (orig_size - best_size) / orig_size if orig_size > 0 else 0.0
                status_text.text(
                    f"AI最適化中... {i+1}/{iterations} | best: {best_size/1024:.1f} KB"
                    f" | 削減率: {reduction_now*100:.1f}% | 目標(soft): {TARGET_REDUCTION*100:.0f}%"
                    f" | profile={_DOC_PROFILE} | color={'Y' if _ORIG_HAS_COLOR else 'N'} | gate={gate_q:.2f}"
                    f" | img_weight={_DOC_IMAGE_WEIGHT:.2f}"
                )

                if best_preview_img is not None and orig_sample_for_preview is not None:
                    preview_placeholder.image(
                        [orig_sample_for_preview, best_preview_img],
                        caption=["Before", "After (Best Found)"],
                        width=300
                    )

                fig, ax = plt.subplots(figsize=(8, 2.5))
                ax.plot(history, linewidth=2)
                ax.set_ylabel("Size (KB)")
                chart_placeholder.pyplot(fig)
                plt.close(fig)

            if _time_limited:
                st.warning("時間上限に達したため、探索を途中で終了しました（公開環境の保護のため）。現在のベスト結果を返します。")

            if best_state is None:
                st.error("❌ 最適化に失敗しました（内部処理がすべて失敗）。入力PDFの構造が特殊な可能性があります。")
            else:
                st.success(f"✅ 最適化完了! 削減率: {(1-best_size/orig_size)*100:.1f}%（目標はsoft）")

                best_state = _apply_safety_guards(best_state)

                final_pdf, _, _ = execute_compression(orig_bytes, best_state, session_id=session_id)
                if final_pdf:
                    st.download_button(
                        label="最適化済みPDFをダウンロード",
                        data=final_pdf,
                        file_name=f"NEO_AI_v7.20.0_{uploaded_file.name}",
                        mime="application/pdf"
                    )

        except Exception as e:
            if SHOW_DEBUG:
                st.exception(e)
            else:
                st.error("内部エラーが発生しました。入力PDFを変えるか、時間をおいて再試行してください。")

        finally:
            if _acquired:
                try:
                    _JOB_SEMAPHORE.release()
                except Exception:
                    pass


# Footer (UIは変更しない：既存の枠内に折りたたみを差し込むだけ)
st.markdown(f"""
<div style="text-align:center;color:#666;padding:20px;">
  © 2024-2026 Hideyoshi_Murakami | Neo PDF Engine v7.20.0 | <b>X: @nagisa7654321</b>
  <div style="max-width:900px;margin:10px auto;text-align:left;color:#666;">
    {DISCLAIMER_HTML}
  </div>
</div>
""", unsafe_allow_html=True)
