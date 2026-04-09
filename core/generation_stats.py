"""
모델별 콘텐츠 생성 시간 통계를 기록하고 관리하는 모듈.

이미지 모델(Gemini)과 영상 모델(Kling) 각각에 대해 생성 완료까지 걸린
실제 시간을 누적하여 지수 이동 평균(EMA)으로 추적합니다.
저장 위치: _generation_stats.json (프로젝트 루트)
"""

import json
import os
import threading

_STATS_FILE = "_generation_stats.json"
_EMA_ALPHA = 0.3  # 새 샘플의 가중치 (0 < α ≤ 1, 작을수록 오래된 평균을 더 반영)
_lock = threading.Lock()


# ── 내부 I/O ──────────────────────────────────────────────────────────────────

def _load() -> dict:
    if os.path.exists(_STATS_FILE):
        try:
            with open(_STATS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"image_models": {}, "video_models": {}}


def _save(data: dict) -> None:
    try:
        with open(_STATS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# ── 공개 API ──────────────────────────────────────────────────────────────────

def record_image_generation(model_name: str, elapsed_seconds: float) -> None:
    """단일 이미지 생성 완료 시간을 기록하고 EMA 평균을 갱신합니다."""
    if elapsed_seconds <= 0:
        return
    with _lock:
        data = _load()
        models = data.setdefault("image_models", {})
        entry = models.get(model_name)
        if entry is None or entry.get("count", 0) == 0:
            models[model_name] = {"count": 1, "avg_seconds": elapsed_seconds}
        else:
            old_avg = entry["avg_seconds"]
            new_avg = _EMA_ALPHA * elapsed_seconds + (1 - _EMA_ALPHA) * old_avg
            models[model_name] = {"count": entry["count"] + 1, "avg_seconds": new_avg}
        _save(data)


def record_video_generation(model_name: str, elapsed_seconds: float) -> None:
    """영상 생성 완료 시간을 기록하고 EMA 평균을 갱신합니다."""
    if elapsed_seconds <= 0:
        return
    with _lock:
        data = _load()
        models = data.setdefault("video_models", {})
        entry = models.get(model_name)
        if entry is None or entry.get("count", 0) == 0:
            models[model_name] = {"count": 1, "avg_seconds": elapsed_seconds}
        else:
            old_avg = entry["avg_seconds"]
            new_avg = _EMA_ALPHA * elapsed_seconds + (1 - _EMA_ALPHA) * old_avg
            models[model_name] = {"count": entry["count"] + 1, "avg_seconds": new_avg}
        _save(data)


def get_avg_image_time(model_name: str, default: float = 60.0) -> float:
    """모델별 평균 이미지 생성 시간(초)을 반환합니다. 데이터가 없으면 default 반환."""
    with _lock:
        data = _load()
    entry = data.get("image_models", {}).get(model_name)
    if entry and entry.get("count", 0) > 0 and entry.get("avg_seconds", 0) > 0:
        return entry["avg_seconds"]
    return default


def get_avg_video_time(model_name: str, default: float = 120.0) -> float:
    """모델별 평균 영상 생성 시간(초)을 반환합니다. 데이터가 없으면 default 반환."""
    with _lock:
        data = _load()
    entry = data.get("video_models", {}).get(model_name)
    if entry and entry.get("count", 0) > 0 and entry.get("avg_seconds", 0) > 0:
        return entry["avg_seconds"]
    return default
